import torch
import torch.nn.functional as F
from torch import nn

from loopear.d.model import Block, MLPHead, _trunc, patchify
from loopear.d.spec import (
    D_MODEL,
    FFN_RATIO,
    GRID_F,
    GRID_T,
    N_HEADS,
    N_LATENTS,
    P_CROSS_LAYERS,
    P_SELF_LAYERS,
    PERCEIVER_ENCODER_PARAMS,
)


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
        self, latents: torch.Tensor, tokens: torch.Tensor, key_pad: torch.Tensor
    ) -> torch.Tensor:
        h = self.ln_q(latents)
        b, m, d = h.shape
        n = tokens.shape[1]
        hd = d // self.n_heads
        q = self.q(h).view(b, m, self.n_heads, hd).permute(0, 2, 1, 3)
        k = self.k(tokens).view(b, n, self.n_heads, hd).permute(0, 2, 1, 3)
        v = self.v(tokens).view(b, n, self.n_heads, hd).permute(0, 2, 1, 3)
        scores = torch.matmul(q, k.transpose(-2, -1)) * (hd ** -0.5)
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
        for cross, self_block in zip(self.cross_blocks, self.self_blocks):
            latents = cross(latents, flat, key_pad)
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
