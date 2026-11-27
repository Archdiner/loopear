import torch
import torch.nn.functional as F
from torch import nn

from loopear.d.model import Block, MLPHead, _trunc, patchify
from loopear.d.objective import (
    broadcast_time_valid,
    global_ntxent,
    local_mld,
    patch_valid_from_frames,
    sample_freq_mask,
    sample_time_mask,
)
from loopear.d.spec import (
    D_LOCAL,
    D_MODEL,
    FFN_RATIO,
    GRID_F,
    GRID_T,
    N_HEADS,
    N_LATENTS,
    P_CROSS_LAYERS,
    P_SELF_LAYERS,
    PERCEIVER_ENCODER_PARAMS,
    TAU_GLOBAL,
    TAU_LOCAL,
)


def patch_and_latent_time() -> tuple[torch.Tensor, torch.Tensor]:
    latent_time = torch.linspace(0, GRID_T - 1, N_LATENTS)
    patch_time = torch.arange(GRID_T, dtype=torch.float32).repeat_interleave(GRID_F)
    return patch_time, latent_time


def apply_time_bias(scores: torch.Tensor, alpha: float) -> torch.Tensor:
    if float(alpha) == 0.0:
        return scores
    patch_time, latent_time = patch_and_latent_time()
    patch_time = patch_time.to(device=scores.device, dtype=scores.dtype)
    latent_time = latent_time.to(device=scores.device, dtype=scores.dtype)
    delta = patch_time.view(1, 1, 1, -1) - latent_time.view(1, 1, -1, 1)
    return scores - float(alpha) * delta.square()


class CrossBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_ratio: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} is not divisible by heads {n_heads}")
        self.n_heads = n_heads
        self.ln_q = nn.LayerNorm(d_model)
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        hidden = ffn_ratio * d_model
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)

    def forward(
        self,
        latents: torch.Tensor,
        tokens: torch.Tensor,
        key_pad: torch.Tensor,
        alpha: float = 0.0,
    ) -> torch.Tensor:
        h = self.ln_q(latents)
        b, m, d = h.shape
        n = tokens.shape[1]
        hd = d // self.n_heads
        q = self.q(h).view(b, m, self.n_heads, hd).permute(0, 2, 1, 3)
        k = self.k(tokens).view(b, n, self.n_heads, hd).permute(0, 2, 1, 3)
        v = self.v(tokens).view(b, n, self.n_heads, hd).permute(0, 2, 1, 3)
        scores = torch.matmul(q, k.transpose(-2, -1)) * (hd ** -0.5)
        scores = apply_time_bias(scores, alpha)
        scores = scores.masked_fill(key_pad[:, None, None, :], -1e9)
        weights = torch.softmax(scores, dim=-1)
        y = torch.matmul(weights, v)
        y = y.permute(0, 2, 1, 3).contiguous().view(b, m, d)
        latents = latents + self.proj(y)
        latents = latents + self.fc2(F.gelu(self.fc1(self.ln2(latents))))
        return latents


class LoopEarPEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch = nn.Linear(64, D_MODEL)
        self.pos_time = nn.Parameter(torch.empty(GRID_T, D_MODEL))
        self.pos_freq = nn.Parameter(torch.empty(GRID_F, D_MODEL))
        self.latents = nn.Parameter(torch.empty(N_LATENTS, D_MODEL))
        self.cross_blocks = nn.ModuleList(
            [CrossBlock(D_MODEL, N_HEADS, FFN_RATIO) for _ in range(P_CROSS_LAYERS)]
        )
        self.self_blocks = nn.ModuleList(
            [Block(D_MODEL, N_HEADS, FFN_RATIO) for _ in range(P_SELF_LAYERS)]
        )
        self.norm = nn.LayerNorm(D_MODEL)
        self.time_alpha = 0.0
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _trunc(self.patch.weight)
        nn.init.zeros_(self.patch.bias)
        _trunc(self.pos_time)
        _trunc(self.pos_freq)
        _trunc(self.latents)
        for block in list(self.cross_blocks) + list(self.self_blocks):
            for mod in block.modules():
                if isinstance(mod, nn.Linear):
                    _trunc(mod.weight)
                    if mod.bias is not None:
                        nn.init.zeros_(mod.bias)
                elif isinstance(mod, nn.LayerNorm):
                    nn.init.ones_(mod.weight)
                    nn.init.zeros_(mod.bias)
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)

    def forward(
        self, mel: torch.Tensor, visible: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if visible.shape[:3] != (mel.shape[0], GRID_T, GRID_F):
            raise ValueError(
                f"visible {tuple(visible.shape)} does not match mel {tuple(mel.shape)}"
            )
        if not bool(visible.flatten(1).any(dim=1).all()):
            raise ValueError("a crop has no visible patches")
        tokens = self.patch(patchify(mel))
        tokens = tokens + self.pos_time[None, :, None, :] + self.pos_freq[None, None, :, :]
        b, t, f, d = tokens.shape
        flat = tokens.reshape(b, t * f, d)
        key_pad = ~visible.reshape(b, t * f)
        latents = self.latents.view(1, N_LATENTS, d).expand(b, N_LATENTS, d).clone()
        latent_pad = torch.zeros(b, N_LATENTS, dtype=torch.bool, device=latents.device)
        alpha = float(self.time_alpha)
        for cross, self_block in zip(self.cross_blocks, self.self_blocks):
            latents = cross(latents, flat, key_pad, alpha)
            latents = self_block(latents, latent_pad)
        latents = self.norm(latents)
        return latents, latents.mean(dim=1)


class LoopEarP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = LoopEarPEncoder()
        self.global_head = MLPHead(D_MODEL, 128, 64)

    def student_parameters(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        for module in (self.encoder, self.global_head):
            params.extend(p for p in module.parameters() if p.requires_grad)
        return params

    def project_global(self, summary: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.global_head(summary), dim=-1, eps=1e-6)


def init_perceiver(seed: int) -> LoopEarP:
    from loopear.losses import set_seed

    set_seed(seed)
    return LoopEarP()


class LatentBankPredictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pos_time = nn.Parameter(torch.empty(GRID_T, D_MODEL))
        self.pos_freq = nn.Parameter(torch.empty(GRID_F, D_MODEL))
        self.view_embed = nn.Parameter(torch.empty(2, D_MODEL))
        self.ln_q = nn.LayerNorm(D_MODEL)
        self.ln_kv = nn.LayerNorm(D_MODEL)
        self.q = nn.Linear(D_MODEL, D_MODEL)
        self.k = nn.Linear(D_MODEL, D_MODEL)
        self.v = nn.Linear(D_MODEL, D_MODEL)
        self.proj = nn.Linear(D_MODEL, D_MODEL)
        self.n_heads = N_HEADS
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _trunc(self.pos_time)
        _trunc(self.pos_freq)
        _trunc(self.view_embed)
        for layer in (self.q, self.k, self.v, self.proj):
            _trunc(layer.weight)
            nn.init.zeros_(layer.bias)
        for norm in (self.ln_q, self.ln_kv):
            nn.init.ones_(norm.weight)
            nn.init.zeros_(norm.bias)

    def forward(self, latents: torch.Tensor, view_index: int) -> torch.Tensor:
        if latents.ndim != 3 or latents.shape[1:] != (N_LATENTS, D_MODEL):
            raise ValueError("the local predictor queries the latent bank")
        batch = latents.shape[0]
        queries = self.pos_time[:, None, :] + self.pos_freq[None, :, :]
        queries = queries + self.view_embed[int(view_index)].view(1, 1, D_MODEL)
        queries = queries.reshape(1, GRID_T * GRID_F, D_MODEL).expand(batch, -1, -1)
        head_dim = D_MODEL // self.n_heads
        q = self.q(self.ln_q(queries)).view(batch, GRID_T * GRID_F, self.n_heads, head_dim)
        k = self.k(self.ln_kv(latents)).view(batch, N_LATENTS, self.n_heads, head_dim)
        v = self.v(self.ln_kv(latents)).view(batch, N_LATENTS, self.n_heads, head_dim)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        scores = torch.matmul(q, k.transpose(-2, -1)) * (head_dim ** -0.5)
        mixed = torch.matmul(torch.softmax(scores, dim=-1), v)
        mixed = mixed.permute(0, 2, 1, 3).contiguous().view(batch, GRID_T * GRID_F, D_MODEL)
        return self.proj(mixed).view(batch, GRID_T, GRID_F, D_MODEL)


class LoopEarPLocal(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = LoopEarPEncoder()
        self.encoder.time_alpha = 0.0
        self.global_head = MLPHead(D_MODEL, 128, 64)
        self.predictor = LatentBankPredictor()
        self.local_head = MLPHead(D_MODEL, D_MODEL, D_LOCAL)
        self.teacher_encoder = LoopEarPEncoder()
        self.teacher_encoder.time_alpha = 0.0
        self.teacher_predictor = LatentBankPredictor()
        self.teacher_local = MLPHead(D_MODEL, D_MODEL, D_LOCAL)
        self.reset_teacher()

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher_encoder.eval()
        self.teacher_predictor.eval()
        self.teacher_local.eval()
        self.encoder.time_alpha = 0.0
        self.teacher_encoder.time_alpha = 0.0
        return self

    def reset_teacher(self) -> None:
        self.teacher_encoder.load_state_dict(self.encoder.state_dict())
        self.teacher_predictor.load_state_dict(self.predictor.state_dict())
        self.teacher_local.load_state_dict(self.local_head.state_dict())
        self.teacher_encoder.time_alpha = 0.0
        for module in (self.teacher_encoder, self.teacher_predictor, self.teacher_local):
            module.eval()
            for param in module.parameters():
                param.requires_grad_(False)

    def student_parameters(self) -> list[nn.Parameter]:
        chosen = []
        for module in (self.encoder, self.global_head, self.predictor, self.local_head):
            chosen.extend(param for param in module.parameters() if param.requires_grad)
        return chosen

    def project_global(self, summary: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.global_head(summary), dim=-1, eps=1e-6)

    def project_local(self, grid: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.local_head(grid), dim=-1, eps=1e-6)

    @torch.no_grad()
    def project_teacher(self, grid: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.teacher_local(grid), dim=-1, eps=1e-6)

    @torch.no_grad()
    def ema_update(self, momentum: float) -> None:
        pairs = (
            (self.encoder, self.teacher_encoder),
            (self.predictor, self.teacher_predictor),
            (self.local_head, self.teacher_local),
        )
        for source, target in pairs:
            for student, teacher in zip(source.parameters(), target.parameters()):
                teacher.mul_(momentum).add_(student.detach(), alpha=1.0 - momentum)
        self.teacher_encoder.time_alpha = 0.0


def init_perceiver_local(seed: int) -> LoopEarPLocal:
    from loopear.losses import set_seed

    set_seed(seed)
    model = LoopEarPLocal()
    model.encoder.time_alpha = 0.0
    model.teacher_encoder.time_alpha = 0.0
    return model


def perceiver_local_loss(
    model: LoopEarPLocal,
    mel1: torch.Tensor,
    mel2: torch.Tensor,
    frame_valid1: torch.Tensor,
    frame_valid2: torch.Tensor,
    generator: torch.Generator,
    lambda_local: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if float(model.encoder.time_alpha) != 0.0 or float(model.teacher_encoder.time_alpha) != 0.0:
        raise RuntimeError("P-L keeps the ordinary unanchored cross-attention")
    valid1 = broadcast_time_valid(patch_valid_from_frames(frame_valid1))
    valid2 = broadcast_time_valid(patch_valid_from_frames(frame_valid2))
    if bool((valid1.flatten(1).sum(dim=1) == 0).any()) or bool(
        (valid2.flatten(1).sum(dim=1) == 0).any()
    ):
        raise ValueError("a crop has no valid patches")
    mask1 = sample_time_mask(valid1.any(dim=-1), generator)
    mask1 = mask1[:, :, None].expand_as(valid1) & valid1
    mask2 = sample_freq_mask(mel2.shape[0], generator)
    mask2 = mask2[:, None, :].expand_as(valid2) & valid2
    latents1, summary1 = model.encoder(mel1, ~mask1 & valid1)
    latents2, summary2 = model.encoder(mel2, ~mask2 & valid2)
    loss_g = global_ntxent(
        model.project_global(summary1),
        model.project_global(summary2),
        TAU_GLOBAL,
    )
    pred1 = model.project_local(model.predictor(latents1, 0))
    pred2 = model.project_local(model.predictor(latents2, 1))
    with torch.no_grad():
        teacher1, _summary1 = model.teacher_encoder(mel1, valid1)
        teacher2, _summary2 = model.teacher_encoder(mel2, valid2)
        target1 = model.project_teacher(model.teacher_predictor(teacher1, 0))
        target2 = model.project_teacher(model.teacher_predictor(teacher2, 1))
    loss_l = 0.5 * (
        local_mld(pred1, target1, mask1, valid1, TAU_LOCAL, 1)
        + local_mld(pred2, target2, mask2, valid2, TAU_LOCAL, 1)
    )
    return loss_g + float(lambda_local) * loss_l, loss_g, loss_l


def visible_cross_logit_stats(
    encoder: LoopEarPEncoder, mel: torch.Tensor, visible: torch.Tensor
) -> dict[str, float]:
    block = encoder.cross_blocks[0]
    tokens = encoder.patch(patchify(mel))
    tokens = tokens + encoder.pos_time[None, :, None, :] + encoder.pos_freq[None, None, :, :]
    b, t, f, d = tokens.shape
    flat = tokens.reshape(b, t * f, d)
    key_pad = ~visible.reshape(b, t * f)
    latents = encoder.latents.view(1, N_LATENTS, d).expand(b, N_LATENTS, d)
    h = block.ln_q(latents)
    hd = d // block.n_heads
    q = block.q(h).view(b, N_LATENTS, block.n_heads, hd).permute(0, 2, 1, 3)
    k = block.k(flat).view(b, t * f, block.n_heads, hd).permute(0, 2, 1, 3)
    scores = torch.matmul(q, k.transpose(-2, -1)) * (hd ** -0.5)
    patch_time, latent_time = patch_and_latent_time()
    penalty = (patch_time.view(1, -1) - latent_time.view(-1, 1)).square()
    penalty = penalty.to(dtype=scores.dtype).view(1, 1, N_LATENTS, -1).expand_as(scores)
    keep = (~key_pad).view(b, 1, 1, t * f).expand_as(scores)
    kept_scores = scores[keep]
    kept_penalty = penalty[keep]
    return {
        "logit_sq": float((kept_scores * kept_scores).sum()),
        "penalty_sq": float((kept_penalty * kept_penalty).sum()),
        "count": float(kept_scores.numel()),
    }


def content_and_anchor_sums(
    encoder: LoopEarPEncoder,
    mel: torch.Tensor,
    visible: torch.Tensor,
    alpha: float,
) -> list[dict[str, float]]:
    tokens = encoder.patch(patchify(mel))
    tokens = tokens + encoder.pos_time[None, :, None, :] + encoder.pos_freq[None, None, :, :]
    batch, time_n, freq_n, width = tokens.shape
    flat = tokens.reshape(batch, time_n * freq_n, width)
    key_pad = ~visible.reshape(batch, time_n * freq_n)
    latents = encoder.latents.view(1, N_LATENTS, width).expand(batch, N_LATENTS, width).clone()
    latent_pad = torch.zeros(batch, N_LATENTS, dtype=torch.bool, device=latents.device)
    patch_time, latent_time = patch_and_latent_time()
    delta2 = (patch_time.view(1, -1) - latent_time.view(-1, 1)).square()
    rows = []
    for block, self_block in zip(encoder.cross_blocks, encoder.self_blocks):
        hidden = block.ln_q(latents)
        head_dim = width // block.n_heads
        q = block.q(hidden).view(batch, N_LATENTS, block.n_heads, head_dim).permute(0, 2, 1, 3)
        k = block.k(flat).view(batch, flat.shape[1], block.n_heads, head_dim).permute(0, 2, 1, 3)
        scores = torch.matmul(q, k.transpose(-2, -1)) * (head_dim ** -0.5)
        penalty = (float(alpha) * delta2).to(dtype=scores.dtype, device=scores.device)
        penalty = penalty.view(1, 1, N_LATENTS, -1).expand_as(scores)
        keep = (~key_pad).view(batch, 1, 1, flat.shape[1]).expand_as(scores)
        content = scores[keep]
        held = penalty[keep]
        rows.append(
            {
                "logit_sq": float((content * content).sum()),
                "logit_abs": float(content.abs().sum()),
                "penalty_sq": float((held * held).sum()),
                "penalty_abs": float(held.abs().sum()),
                "exceed": float((held.abs() > content.abs()).sum()),
                "count": float(content.numel()),
            }
        )
        latents = block(latents, flat, key_pad, float(alpha))
        latents = self_block(latents, latent_pad)
    return rows


def encoder_parameter_count(model: LoopEarPEncoder | None = None) -> int:
    if model is None:
        return PERCEIVER_ENCODER_PARAMS
    return int(sum(p.numel() for p in model.parameters()))


def analytic_perceiver_flops(batch: int = 1, n_tokens: int = 200) -> dict[str, float]:
    d = D_MODEL
    m = N_LATENTS
    hidden = FFN_RATIO * d
    patch_macs = batch * n_tokens * 64 * d
    cross_q = batch * m * d * d
    cross_k = batch * n_tokens * d * d
    cross_v = batch * n_tokens * d * d
    cross_proj = batch * m * d * d
    cross_ffn = batch * m * (d * hidden + hidden * d)
    cross_attn = batch * (2 * m * n_tokens * d)
    cross_macs = (cross_q + cross_k + cross_v + cross_proj + cross_ffn + cross_attn) * P_CROSS_LAYERS
    self_linear = batch * m * ((4 * d * d) + (d * hidden + hidden * d))
    self_attn = batch * (2 * m * m * d)
    self_macs = (self_linear + self_attn) * P_SELF_LAYERS
    macs = patch_macs + cross_macs + self_macs
    score_bytes = batch * N_HEADS * m * n_tokens * 4
    return {
        "batch": float(batch),
        "latents": float(m),
        "input_tokens": float(n_tokens),
        "cross_layers": float(P_CROSS_LAYERS),
        "self_layers": float(P_SELF_LAYERS),
        "gemm_macs": float(macs),
        "gemm_flops_mul_add": float(2 * macs),
        "gemm_gflops": float(2 * macs) / 1e9,
        "cross_score_bytes_one_layer": float(score_bytes),
        "self_score_bytes_one_layer": float(batch * N_HEADS * m * m * 4),
    }
