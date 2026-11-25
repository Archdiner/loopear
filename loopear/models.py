import math

import torch
import torch.nn.functional as F
from torch import nn

from loopear.encoder import D_E, HOP_E, NUM_ENCODER_LAYERS


def lengths_to_padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    positions = torch.arange(max_len, device=lengths.device, dtype=torch.long)
    return positions.unsqueeze(0) >= lengths.unsqueeze(1)


def make_timestamps(
    t_e: int,
    t_z: int,
    stride: int,
    device: torch.device,
    hop_e: float = HOP_E,
) -> tuple[torch.Tensor, torch.Tensor]:
    ts_e = (torch.arange(t_e, device=device, dtype=torch.float32) + 0.5) * hop_e
    ts_z = (
        torch.arange(t_z, device=device, dtype=torch.float32) + 0.5
    ) * (hop_e * stride)
    return ts_e, ts_z


def _pad_to_stride(
    x: torch.Tensor,
    pad: torch.Tensor,
    stride: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    b, t, _ = x.shape
    rem = t % stride
    if rem == 0:
        return x, pad
    extra = stride - rem
    x = F.pad(x, (0, 0, 0, extra), value=0.0)
    pad = F.pad(pad, (0, extra), value=True)
    return x, pad


def masked_avg_pool(
    x: torch.Tensor,
    pad: torch.Tensor,
    stride: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if stride == 1:
        return x.masked_fill(pad.unsqueeze(-1), 0.0), pad
    x_p, pad_p = _pad_to_stride(x, pad, stride)
    b, t_p, d = x_p.shape
    t_z = t_p // stride
    x_bins = x_p.view(b, t_z, stride, d)
    valid = (~pad_p).view(b, t_z, stride)
    counts = valid.sum(dim=2, keepdim=True)
    z_pad = counts.squeeze(-1) == 0
    weighted = x_bins * valid.unsqueeze(-1).to(x.dtype)
    z = weighted.sum(dim=2) / counts.clamp(min=1).to(x.dtype)
    return z.masked_fill(z_pad.unsqueeze(-1), 0.0), z_pad


class TemporalPool(nn.Module):
    def __init__(self, d: int, stride: int = 8, mode: str = "avg"):
        super().__init__()
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        if mode not in ("avg", "strided_conv", "attn"):
            raise ValueError(f"Unsupported pool_mode: {mode}")
        self.d = d
        self.stride = stride
        self.mode = mode
        if mode == "strided_conv" and stride > 1:
            self.conv = nn.Conv1d(d, d, kernel_size=stride, stride=stride)
        else:
            self.conv = None
        if mode == "attn" and stride > 1:
            self.attn_score = nn.Linear(d, 1)
        else:
            self.attn_score = None

    def forward(
        self,
        x: torch.Tensor,
        pad: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.stride == 1 or self.mode == "avg":
            return masked_avg_pool(x, pad, self.stride)
        x_clean = x.masked_fill(pad.unsqueeze(-1), 0.0)
        x_p, pad_p = _pad_to_stride(x_clean, pad, self.stride)
        b, t_p, d = x_p.shape
        t_z = t_p // self.stride
        valid = (~pad_p).view(b, t_z, self.stride)
        z_pad = valid.sum(dim=2) == 0
        if self.mode == "strided_conv":
            assert self.conv is not None
            z = self.conv(x_p.transpose(1, 2)).transpose(1, 2)
            return z.masked_fill(z_pad.unsqueeze(-1), 0.0), z_pad
        assert self.attn_score is not None
        x_bins = x_p.view(b, t_z, self.stride, d)
        logits = self.attn_score(x_bins).squeeze(-1)
        bin_pad = ~valid
        logits = logits.masked_fill(bin_pad, -1e9)
        weights = F.softmax(logits, dim=2)
        weights = weights.masked_fill(bin_pad, 0.0)
        z = (x_bins * weights.unsqueeze(-1)).sum(dim=2)
        return z.masked_fill(z_pad.unsqueeze(-1), 0.0), z_pad


class MultiLayerFusion(nn.Module):
    def __init__(self, n_layers: int = NUM_ENCODER_LAYERS):
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(n_layers))

    def forward(self, layers: torch.Tensor) -> torch.Tensor:
        w = F.softmax(self.layer_logits, dim=0).view(1, -1, 1, 1)
        return (w * layers).sum(dim=1)


class ContinuousTimeRoPE(nn.Module):
    def __init__(
        self,
        d_head: int = 64,
        f_min: float = 0.05,
        f_max: float = 25.0,
    ):
        super().__init__()
        if d_head % 2 != 0:
            raise ValueError(f"d_head must be even, got {d_head}")
        freqs = torch.logspace(math.log10(f_min), math.log10(f_max), d_head // 2)
        self.register_buffer("omega", 2.0 * math.pi * freqs)

    def apply(self, x: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        ang = ts.to(self.omega.dtype)[:, None] * self.omega[None, :]
        cos = ang.cos()[None, None].to(x.dtype)
        sin = ang.sin()[None, None].to(x.dtype)
        xe = x[..., 0::2]
        xo = x[..., 1::2]
        return torch.stack((xe * cos - xo * sin, xe * sin + xo * cos), dim=-1).flatten(
            -2
        )


class SwiGLU(nn.Module):
    def __init__(self, d: int, d_ff: int, drop: float = 0.1):
        super().__init__()
        self.w_gu = nn.Linear(d, 2 * d_ff, bias=False)
        self.w_d = nn.Linear(d_ff, d, bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g, u = self.w_gu(x).chunk(2, dim=-1)
        return self.drop(self.w_d(F.silu(g) * u))


class MultiHeadAttnRoPE(nn.Module):
    def __init__(self, d: int = 384, h: int = 6, drop: float = 0.1):
        super().__init__()
        if d % h != 0:
            raise ValueError(f"d={d} must be divisible by h={h}")
        self.h = h
        self.dh = d // h
        self.drop = drop
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)
        self.rope = ContinuousTimeRoPE(self.dh)

    def project_kv(
        self,
        kv: torch.Tensor,
        t_kv: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, _ = kv.shape
        k = self.k(kv).view(b, t, self.h, self.dh).transpose(1, 2)
        k = self.rope.apply(k, t_kv)
        v = self.v(kv).view(b, t, self.h, self.dh).transpose(1, 2)
        return k, v

    def forward(
        self,
        q_in: torch.Tensor,
        t_q: torch.Tensor,
        kv_in: torch.Tensor | None = None,
        t_kv: torch.Tensor | None = None,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        kv_pad: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, tq, d = q_in.shape
        q = self.q(q_in).view(b, tq, self.h, self.dh).transpose(1, 2)
        q = self.rope.apply(q, t_q)
        if kv_cache is not None:
            k, v = kv_cache
        else:
            if kv_in is None or t_kv is None:
                raise ValueError("kv_in and t_kv required when kv_cache is None")
            k, v = self.project_kv(kv_in, t_kv)
        mask = (~kv_pad)[:, None, None, :] if kv_pad is not None else None
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.drop if self.training else 0.0,
            is_causal=False,
        )
        return self.o(out.transpose(1, 2).contiguous().view(b, tq, d))


class SelfAttnFFNBlock(nn.Module):
    def __init__(
        self,
        d: int = 384,
        h: int = 6,
        d_ff: int = 768,
        drop: float = 0.1,
        ls: float = 0.1,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.sa = MultiHeadAttnRoPE(d, h, drop)
        self.g1 = nn.Parameter(torch.full((d,), ls))
        self.ln2 = nn.LayerNorm(d)
        self.ff = SwiGLU(d, d_ff, drop)
        self.g2 = nn.Parameter(torch.full((d,), ls))

    def forward(
        self,
        z: torch.Tensor,
        t_z: torch.Tensor,
        z_pad: torch.Tensor,
        film: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        h = self.ln1(z)
        if film is not None:
            h = h * (1.0 + film[0]) + film[1]
        z = z + self.g1 * self.sa(h, t_z, kv_in=h, t_kv=t_z, kv_pad=z_pad)
        z = z + self.g2 * self.ff(self.ln2(z))
        return z.masked_fill(z_pad.unsqueeze(-1), 0.0)


class EvidenceRefineBlock(nn.Module):
    def __init__(
        self,
        d: int = 384,
        h: int = 6,
        d_ff: int = 768,
        drop: float = 0.1,
        ls: float = 0.1,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.sa = MultiHeadAttnRoPE(d, h, drop)
        self.g1 = nn.Parameter(torch.full((d,), ls))
        self.ln2 = nn.LayerNorm(d)
        self.ca = MultiHeadAttnRoPE(d, h, drop)
        self.g2 = nn.Parameter(torch.full((d,), ls))
        self.ln3 = nn.LayerNorm(d)
        self.ff = SwiGLU(d, d_ff, drop)
        self.g3 = nn.Parameter(torch.full((d,), ls))

    def precompute_kv(
        self,
        e_norm: torch.Tensor,
        t_e: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.ca.project_kv(e_norm, t_e)

    def forward(
        self,
        z: torch.Tensor,
        t_z: torch.Tensor,
        e_pad: torch.Tensor,
        z_pad: torch.Tensor,
        e_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        e_norm: torch.Tensor | None = None,
        t_e: torch.Tensor | None = None,
        film: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        h = self.ln1(z)
        if film is not None:
            h = h * (1.0 + film[0]) + film[1]
        z = z + self.g1 * self.sa(h, t_z, kv_in=h, t_kv=t_z, kv_pad=z_pad)
        if e_kv is not None:
            ca_out = self.ca(self.ln2(z), t_z, kv_cache=e_kv, kv_pad=e_pad)
        else:
            ca_out = self.ca(
                self.ln2(z), t_z, kv_in=e_norm, t_kv=t_e, kv_pad=e_pad
            )
        z = z + self.g2 * ca_out
        z = z + self.g3 * self.ff(self.ln3(z))
        return z.masked_fill(z_pad.unsqueeze(-1), 0.0)


class BaselineALast(nn.Module):
    def __init__(
        self,
        d_e: int = D_E,
        d: int = 384,
        stride: int = 8,
        pool_mode: str = "avg",
    ):
        super().__init__()
        self.out_dim = d
        self.stride = stride
        self.depths = (1,)
        self.pool = TemporalPool(d_e, stride=stride, mode=pool_mode)
        self.proj = nn.Linear(d_e, d)
        self.out_ln = nn.LayerNorm(d)

    def forward_states(
        self,
        e: torch.Tensor,
        e_len: torch.Tensor,
        depths: tuple[int, ...] | None = None,
        hidden_states: torch.Tensor | None = None,
        layer_pool: torch.Tensor | None = None,
    ) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
        e_pad = lengths_to_padding_mask(e_len, e.shape[1])
        z, z_pad = self.pool(e, e_pad)
        z = self.out_ln(self.proj(z)).masked_fill(z_pad.unsqueeze(-1), 0.0)
        return {1: z}, z_pad


class BaselineAMultiLayer(nn.Module):
    def __init__(
        self,
        n_layers: int = NUM_ENCODER_LAYERS,
        d_e: int = D_E,
        d: int = 384,
        stride: int = 8,
        pool_mode: str = "avg",
    ):
        super().__init__()
        self.out_dim = d
        self.stride = stride
        self.depths = (1,)
        self.fusion = MultiLayerFusion(n_layers=n_layers)
        self.pool = TemporalPool(d_e, stride=stride, mode=pool_mode)
        self.proj = nn.Linear(d_e, d)
        self.out_ln = nn.LayerNorm(d)

    def forward_states(
        self,
        e: torch.Tensor,
        e_len: torch.Tensor,
        depths: tuple[int, ...] | None = None,
        hidden_states: torch.Tensor | None = None,
        layer_pool: torch.Tensor | None = None,
    ) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
        e_pad = lengths_to_padding_mask(e_len, e.shape[1])
        if layer_pool is not None:
            fused = self.fusion(layer_pool)
            _, z_pad = masked_avg_pool(
                e[:, : layer_pool.shape[2] * self.stride], e_pad, self.stride
            )
            z_pad = z_pad[:, : fused.shape[1]]
            z = self.out_ln(self.proj(fused)).masked_fill(z_pad.unsqueeze(-1), 0.0)
            return {1: z}, z_pad
        if hidden_states is not None:
            fused_e = self.fusion(hidden_states)
        else:
            fused_e = e
        z, z_pad = self.pool(fused_e, e_pad)
        z = self.out_ln(self.proj(z)).masked_fill(z_pad.unsqueeze(-1), 0.0)
        return {1: z}, z_pad


class BaselineB(nn.Module):
    def __init__(
        self,
        variant: str = "compute",
        d_e: int = D_E,
        stride: int = 8,
        pool_mode: str = "avg",
        drop: float = 0.1,
        d: int | None = None,
        h: int | None = None,
        d_ff: int | None = None,
    ):
        super().__init__()
        if variant == "compute":
            n_l, def_d, def_h, def_ff = 1, 320, 5, 640
        elif variant == "param":
            n_l, def_d, def_h, def_ff = 2, 320, 5, 640
        else:
            raise ValueError(f"Unsupported BaselineB variant: {variant}")
        d = def_d if d is None else d
        h = def_h if h is None else h
        d_ff = def_ff if d_ff is None else d_ff
        self.variant = variant
        self.out_dim = d
        self.stride = stride
        self.depths = (1,)
        self.in_proj = nn.Linear(d_e, d)
        self.blocks = nn.ModuleList(
            [SelfAttnFFNBlock(d, h, d_ff, drop) for _ in range(n_l)]
        )
        self.pool = TemporalPool(d, stride=stride, mode=pool_mode)
        self.out_ln = nn.LayerNorm(d)

    def forward_states(
        self,
        e: torch.Tensor,
        e_len: torch.Tensor,
        depths: tuple[int, ...] | None = None,
        hidden_states: torch.Tensor | None = None,
        layer_pool: torch.Tensor | None = None,
    ) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
        e_pad = lengths_to_padding_mask(e_len, e.shape[1])
        ts_e = (
            torch.arange(e.shape[1], device=e.device, dtype=torch.float32) + 0.5
        ) * HOP_E
        h = self.in_proj(e).masked_fill(e_pad.unsqueeze(-1), 0.0)
        for blk in self.blocks:
            h = blk(h, ts_e, e_pad)
        z, z_pad = self.pool(h, e_pad)
        z = self.out_ln(z).masked_fill(z_pad.unsqueeze(-1), 0.0)
        return {1: z}, z_pad


class BaselineC(nn.Module):
    def __init__(
        self,
        variant: str = "param",
        d_e: int = D_E,
        stride: int = 8,
        pool_mode: str = "avg",
        drop: float = 0.1,
    ):
        super().__init__()
        cfg_map = {
            "param": (4, 192, 4, 384),
            "compute": (4, 384, 6, 768),
            "wide": (1, 768, 12, 1536),
        }
        if variant not in cfg_map:
            raise ValueError(f"Unsupported BaselineC variant: {variant}")
        n_b, d, h, d_ff = cfg_map[variant]
        self.variant = variant
        self.out_dim = d
        self.stride = stride
        self.depths = (n_b,)
        self.e_proj = nn.Linear(d_e, d)
        self.e_ln = nn.LayerNorm(d)
        self.pool = TemporalPool(d, stride=stride, mode=pool_mode)
        self.z0_ln = nn.LayerNorm(d)
        self.out_ln = nn.LayerNorm(d)
        self.blocks = nn.ModuleList(
            [EvidenceRefineBlock(d, h, d_ff, drop) for _ in range(n_b)]
        )

    def forward_states(
        self,
        e: torch.Tensor,
        e_len: torch.Tensor,
        depths: tuple[int, ...] | None = None,
        hidden_states: torch.Tensor | None = None,
        layer_pool: torch.Tensor | None = None,
    ) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
        target_depths = depths if depths is not None else self.depths
        e_pad = lengths_to_padding_mask(e_len, e.shape[1])
        ep = self.e_proj(e).masked_fill(e_pad.unsqueeze(-1), 0.0)
        e_norm = self.e_ln(ep).masked_fill(e_pad.unsqueeze(-1), 0.0)
        z, z_pad = self.pool(ep, e_pad)
        z = self.z0_ln(z).masked_fill(z_pad.unsqueeze(-1), 0.0)
        ts_e, ts_z = make_timestamps(e.shape[1], z.shape[1], self.stride, e.device)
        out: dict[int, torch.Tensor] = {}
        for k, blk in enumerate(self.blocks, start=1):
            e_kv = blk.precompute_kv(e_norm, ts_e)
            z = blk(z, ts_z, e_pad, z_pad, e_kv=e_kv)
            if k in target_depths:
                out[k] = self.out_ln(z).masked_fill(z_pad.unsqueeze(-1), 0.0)
        return out, z_pad


class BaselineD(nn.Module):
    def __init__(
        self,
        d_e: int = D_E,
        d: int = 384,
        h: int = 6,
        d_ff: int = 1024,
        stride: int = 8,
        max_k: int = 4,
        pool_mode: str = "avg",
        use_film: bool = True,
        drop: float = 0.1,
    ):
        super().__init__()
        self.out_dim = d
        self.stride = stride
        self.max_k = max_k
        self.depths = (1, 2, 4) if max_k >= 4 else tuple(range(1, max_k + 1))
        self.e_proj = nn.Linear(d_e, d)
        self.e_ln = nn.LayerNorm(d)
        self.pool = TemporalPool(d, stride=stride, mode=pool_mode)
        self.z0_ln = nn.LayerNorm(d)
        self.out_ln = nn.LayerNorm(d)
        self.ln_c = nn.LayerNorm(d)
        self.pass1_ca = MultiHeadAttnRoPE(d, h, drop)
        self.g_c = nn.Parameter(torch.full((d,), 0.1))
        self.latent_blk = SelfAttnFFNBlock(d, h, d_ff, drop)
        self.film_emb = nn.Embedding(max_k, 2 * d) if use_film else None
        if self.film_emb is not None:
            nn.init.zeros_(self.film_emb.weight)

    def forward_states(
        self,
        e: torch.Tensor,
        e_len: torch.Tensor,
        depths: tuple[int, ...] | None = None,
        hidden_states: torch.Tensor | None = None,
        layer_pool: torch.Tensor | None = None,
    ) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
        target_depths = depths if depths is not None else self.depths
        e_pad = lengths_to_padding_mask(e_len, e.shape[1])
        ep = self.e_proj(e).masked_fill(e_pad.unsqueeze(-1), 0.0)
        e_norm = self.e_ln(ep).masked_fill(e_pad.unsqueeze(-1), 0.0)
        z, z_pad = self.pool(ep, e_pad)
        z = self.z0_ln(z).masked_fill(z_pad.unsqueeze(-1), 0.0)
        ts_e, ts_z = make_timestamps(e.shape[1], z.shape[1], self.stride, e.device)
        out: dict[int, torch.Tensor] = {}
        for k in range(1, max(target_depths) + 1):
            film = (
                self.film_emb.weight[k - 1].chunk(2)
                if self.film_emb is not None
                else None
            )
            if k == 1:
                z = z + self.g_c * self.pass1_ca(
                    self.ln_c(z), ts_z, kv_in=e_norm, t_kv=ts_e, kv_pad=e_pad
                )
                z = z.masked_fill(z_pad.unsqueeze(-1), 0.0)
            z = self.latent_blk(z, ts_z, z_pad, film=film)
            if k in target_depths:
                out[k] = self.out_ln(z).masked_fill(z_pad.unsqueeze(-1), 0.0)
        return out, z_pad


class LoopEarModel(nn.Module):
    def __init__(
        self,
        d_e: int = D_E,
        d: int = 384,
        h: int = 6,
        d_ff: int = 768,
        stride: int = 8,
        max_k: int = 4,
        pool_mode: str = "avg",
        use_film: bool = True,
        use_multilayer: bool = False,
        n_layers: int = NUM_ENCODER_LAYERS,
        drop: float = 0.1,
    ):
        super().__init__()
        self.out_dim = d
        self.stride = stride
        self.max_k = max_k
        self.pool_mode = pool_mode
        self.use_multilayer = use_multilayer
        self.depths = (1, 2, 4) if max_k >= 4 else tuple(range(1, max_k + 1))
        self.fusion = MultiLayerFusion(n_layers=n_layers) if use_multilayer else None
        self.e_proj = nn.Linear(d_e, d)
        self.e_ln = nn.LayerNorm(d)
        self.pool = TemporalPool(d, stride=stride, mode=pool_mode)
        self.z0_ln = nn.LayerNorm(d)
        self.out_ln = nn.LayerNorm(d)
        self.refine = EvidenceRefineBlock(d, h, d_ff, drop)
        self.film_emb = nn.Embedding(max_k, 2 * d) if use_film else None
        if self.film_emb is not None:
            nn.init.zeros_(self.film_emb.weight)

    def forward_states(
        self,
        e: torch.Tensor,
        e_len: torch.Tensor,
        depths: tuple[int, ...] | None = None,
        hidden_states: torch.Tensor | None = None,
        layer_pool: torch.Tensor | None = None,
        use_kv_cache: bool = True,
    ) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
        target_depths = depths if depths is not None else self.depths
        e_in = (
            self.fusion(hidden_states)
            if (self.fusion is not None and hidden_states is not None)
            else e
        )
        e_pad = lengths_to_padding_mask(e_len, e_in.shape[1])
        ep = self.e_proj(e_in).masked_fill(e_pad.unsqueeze(-1), 0.0)
        e_norm = self.e_ln(ep).masked_fill(e_pad.unsqueeze(-1), 0.0)
        z, z_pad = self.pool(ep, e_pad)
        z = self.z0_ln(z).masked_fill(z_pad.unsqueeze(-1), 0.0)
        ts_e, ts_z = make_timestamps(e_in.shape[1], z.shape[1], self.stride, e_in.device)
        e_kv = self.refine.precompute_kv(e_norm, ts_e) if use_kv_cache else None
        out: dict[int, torch.Tensor] = {}
        for k in range(1, max(target_depths) + 1):
            film = (
                self.film_emb.weight[k - 1].chunk(2)
                if self.film_emb is not None
                else None
            )
            z = self.refine(
                z,
                ts_z,
                e_pad,
                z_pad,
                e_kv=e_kv,
                e_norm=None if use_kv_cache else e_norm,
                t_e=None if use_kv_cache else ts_e,
                film=film,
            )
            if k in target_depths:
                out[k] = self.out_ln(z).masked_fill(z_pad.unsqueeze(-1), 0.0)
        return out, z_pad


def build_arm(arm: str, **kwargs) -> nn.Module:
    if arm == "A_last":
        return BaselineALast(**kwargs)
    if arm == "A_ml":
        return BaselineAMultiLayer(**kwargs)
    if arm == "B_comp":
        return BaselineB(variant="compute", **kwargs)
    if arm == "B_param":
        return BaselineB(variant="param", **kwargs)
    if arm == "C_param":
        return BaselineC(variant="param", **kwargs)
    if arm == "C_comp":
        return BaselineC(variant="compute", **kwargs)
    if arm == "C_wide":
        return BaselineC(variant="wide", **kwargs)
    if arm == "D":
        return BaselineD(**kwargs)
    if arm == "E":
        return LoopEarModel(**kwargs)
    raise ValueError(f"Unknown arm: {arm}")
