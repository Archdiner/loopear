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

    def set_time_alpha(self, alpha: float) -> None:
        value = float(alpha)
        self.encoder.time_alpha = value
        self.teacher_encoder.time_alpha = value

    def train(self, mode: bool = True):
        alpha = float(self.encoder.time_alpha)
        super().train(mode)
        self.teacher_encoder.eval()
        self.teacher_predictor.eval()
        self.teacher_local.eval()
        self.encoder.time_alpha = alpha
        self.teacher_encoder.time_alpha = alpha
        return self

    def reset_teacher(self) -> None:
        alpha = float(self.encoder.time_alpha)
        self.teacher_encoder.load_state_dict(self.encoder.state_dict())
        self.teacher_predictor.load_state_dict(self.predictor.state_dict())
        self.teacher_local.load_state_dict(self.local_head.state_dict())
        self.teacher_encoder.time_alpha = alpha
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
        self.teacher_encoder.time_alpha = float(self.encoder.time_alpha)


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
    if float(model.encoder.time_alpha) != float(model.teacher_encoder.time_alpha):
        raise RuntimeError("student and teacher time anchors differ")
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


def within_recording_time_sums(
    student: torch.Tensor,
    teacher: torch.Tensor,
    valid: torch.Tensor,
    query: torch.Tensor | None = None,
) -> dict[str, float]:
    if student.shape != teacher.shape or student.ndim != 4:
        raise ValueError("student and teacher grids must match")
    if valid.shape != student.shape[:3] or valid.dtype != torch.bool:
        raise ValueError("valid must be a boolean grid")
    if query is None:
        chosen = valid
    else:
        if query.shape != valid.shape or query.dtype != torch.bool:
            raise ValueError("query mask must match valid")
        chosen = query & valid
    batch, time_n, _freq_n, _width = student.shape
    sim = torch.einsum("btfd,bsfd->bfts", student, teacher)
    target_sim = torch.einsum("btfd,bsfd->bfts", teacher, teacher)
    valid_bft = valid.permute(0, 2, 1)
    chosen_bft = chosen.permute(0, 2, 1)
    eye = torch.eye(time_n, dtype=torch.bool, device=student.device).view(1, 1, time_n, time_n)
    both = valid_bft[:, :, :, None] & valid_bft[:, :, None, :]
    others = both & ~eye
    correct = sim.diagonal(dim1=-2, dim2=-1)
    n_other = others.sum(dim=-1)
    ranked = chosen_bft & (n_other > 0)
    other_scores = sim.masked_fill(~others, -1e9)
    best_other = other_scores.amax(dim=-1)
    strictly = ranked & (correct > best_other)
    tied = ranked & ~strictly & (correct >= best_other)
    n_higher = (ranked[..., None] & others & (sim > correct[..., None])).sum(dim=-1)
    upper = torch.triu(
        torch.ones(time_n, time_n, dtype=torch.bool, device=student.device), diagonal=1
    )
    teacher_pairs = others & upper.view(1, 1, time_n, time_n)
    teacher_cos = target_sim[teacher_pairs]
    cross = torch.einsum("btfd,csfd->bctf", student, teacher)
    self_rec = torch.eye(batch, dtype=torch.bool, device=student.device)
    cross_ok = (
        valid[:, None, :, :]
        & valid[None, :, :, :]
        & ~self_rec[:, :, None, None]
        & chosen[:, None, :, :]
    )
    correct_btf = correct.permute(0, 2, 1)
    has_cross = cross_ok.any(dim=1)
    cross_max = cross.masked_fill(~cross_ok, -1e9).amax(dim=1)
    beat_cross = has_cross & (correct_btf > cross_max)
    other_weight = others.to(sim.dtype) * ranked[..., None].to(sim.dtype)
    return {
        "n_queries_with_other_moment": float(ranked.sum()),
        "n_queries_without_other_moment": float((chosen_bft & (n_other == 0)).sum()),
        "n_correct_strictly_best": float(strictly.sum()),
        "n_tied_for_best": float(tied.sum()),
        "n_worse_than_another_moment": float((ranked & ~strictly & ~tied).sum()),
        "rank_sum": float((1.0 + n_higher.to(sim.dtype))[ranked].sum()),
        "correct_cosine_sum": float(correct[ranked].sum()),
        "other_time_cosine_sum": float((sim * other_weight).sum()),
        "other_time_pair_count": float(other_weight.sum()),
        "n_teacher_pairs": float(teacher_pairs.sum()),
        "teacher_cosine_sum": float(teacher_cos.sum()) if int(teacher_pairs.sum()) else 0.0,
        "n_teacher_pairs_below_0_99": float((teacher_cos < 0.99).sum()) if int(teacher_pairs.sum()) else 0.0,
        "n_teacher_pairs_below_0_50": float((teacher_cos < 0.50).sum()) if int(teacher_pairs.sum()) else 0.0,
        "n_other_recording_pairs": float(cross_ok.sum()),
        "other_recording_cosine_sum": float(cross[cross_ok].sum()) if int(cross_ok.sum()) else 0.0,
        "n_queries_with_other_recording": float(has_cross.sum()),
        "n_correct_beats_other_recording_max": float(beat_cross.sum()),
    }


def _unanchored_cross_scores(
    encoder: LoopEarPEncoder, mel: torch.Tensor, visible: torch.Tensor
) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
    tokens = encoder.patch(patchify(mel))
    tokens = tokens + encoder.pos_time[None, :, None, :] + encoder.pos_freq[None, None, :, :]
    batch, time_n, freq_n, width = tokens.shape
    flat = tokens.reshape(batch, time_n * freq_n, width)
    key_pad = ~visible.reshape(batch, time_n * freq_n)
    latents = encoder.latents.view(1, N_LATENTS, width).expand(batch, N_LATENTS, width).clone()
    latent_pad = torch.zeros(batch, N_LATENTS, dtype=torch.bool, device=latents.device)
    patch_time, latent_time = patch_and_latent_time()
    delta2 = (patch_time.view(1, -1) - latent_time.view(-1, 1)).square()
    scores = []
    for block, self_block in zip(encoder.cross_blocks, encoder.self_blocks):
        hidden = block.ln_q(latents)
        head_dim = width // block.n_heads
        q = block.q(hidden).view(batch, N_LATENTS, block.n_heads, head_dim).permute(0, 2, 1, 3)
        k = block.k(flat).view(batch, flat.shape[1], block.n_heads, head_dim).permute(0, 2, 1, 3)
        content = torch.matmul(q, k.transpose(-2, -1)) * (head_dim ** -0.5)
        scores.append(content)
        latents = block(latents, flat, key_pad, 0.0)
        latents = self_block(latents, latent_pad)
    return scores, key_pad, delta2.to(dtype=scores[0].dtype, device=scores[0].device)


def _enough_visible_keys(key_pad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    visible = ~key_pad
    enough = visible.sum(dim=-1) >= 2
    return visible & enough[:, None], enough


def centered_variation_sums(
    encoder: LoopEarPEncoder, mel: torch.Tensor, visible: torch.Tensor
) -> dict[str, float]:
    layers, key_pad, delta2 = _unanchored_cross_scores(encoder, mel, visible)
    content_sq = 0.0
    penalty_sq = 0.0
    count = 0.0
    visible, _enough = _enough_visible_keys(key_pad)
    penalty = delta2.view(1, 1, N_LATENTS, -1)
    for scores in layers:
        vis = visible[:, None, None, :].to(dtype=scores.dtype).expand_as(scores)
        held_penalty = penalty.expand_as(scores)
        denom = vis.sum(dim=-1, keepdim=True).clamp(min=1.0)
        content_mean = (scores * vis).sum(dim=-1, keepdim=True) / denom
        penalty_mean = (held_penalty * vis).sum(dim=-1, keepdim=True) / denom
        content = (scores - content_mean) * vis
        held = (held_penalty - penalty_mean) * vis
        content_sq += float((content * content).sum())
        penalty_sq += float((held * held).sum())
        count += float(vis.sum())
    return {
        "content_centered_sq": content_sq,
        "penalty_centered_sq": penalty_sq,
        "count": count,
    }


def attention_shift_sums(
    encoder: LoopEarPEncoder, mel: torch.Tensor, visible: torch.Tensor, alpha: float
) -> dict[str, float]:
    layers, key_pad, delta2 = _unanchored_cross_scores(encoder, mel, visible)
    visible_keys, enough = _enough_visible_keys(key_pad)
    penalty = delta2.view(1, 1, N_LATENTS, -1)
    tv_sum = 0.0
    argmax_changed = 0.0
    n_queries = 0.0
    offset_tv_sum = 0.0
    removed_mean_tv_sum = 0.0
    for scores in layers:
        vis = visible_keys[:, None, None, :].to(dtype=scores.dtype)
        denom = vis.sum(dim=-1, keepdim=True).clamp(min=1.0)
        penalty_mean = (penalty * vis).sum(dim=-1, keepdim=True) / denom
        base = scores.masked_fill(key_pad[:, None, None, :], -1e9)
        moved = apply_time_bias(scores, float(alpha)).masked_fill(key_pad[:, None, None, :], -1e9)
        centered = (scores - float(alpha) * (penalty - penalty_mean)).masked_fill(
            key_pad[:, None, None, :], -1e9
        )
        shifted = (scores + 4.0).masked_fill(key_pad[:, None, None, :], -1e9)
        weights = torch.softmax(base, dim=-1)
        changed = torch.softmax(moved, dim=-1)
        same_offset = torch.softmax(shifted, dim=-1)
        same_mean = torch.softmax(centered, dim=-1)
        tv = 0.5 * (weights - changed).abs().sum(dim=-1)
        offset_tv = 0.5 * (weights - same_offset).abs().sum(dim=-1)
        mean_tv = 0.5 * (changed - same_mean).abs().sum(dim=-1)
        tv_sum += float(tv[enough].sum())
        offset_tv_sum += float(offset_tv[enough].sum())
        removed_mean_tv_sum += float(mean_tv[enough].sum())
        argmax_changed += float(((weights.argmax(dim=-1) != changed.argmax(dim=-1))[enough]).sum())
        n_queries += float(enough.sum() * tv.shape[1] * tv.shape[2])
    return {
        "tv_sum": tv_sum,
        "argmax_changed": argmax_changed,
        "n_queries": n_queries,
        "constant_offset_tv_sum": offset_tv_sum,
        "removed_mean_tv_sum": removed_mean_tv_sum,
    }


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
