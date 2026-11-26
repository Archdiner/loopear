import math

import torch
import torch.nn.functional as F
from torch import nn

from loopear.d.spec import (
    D_GLOBAL,
    D_LOCAL,
    D_MODEL,
    D_PRED,
    FFN_RATIO,
    GLOBAL_HIDDEN,
    GRID_F,
    GRID_T,
    N_HEADS,
    N_LAYERS,
    N_MELS,
    PATCH_BINS,
    PATCH_DIM,
    PATCH_FRAMES,
    PRED_HEADS,
    PRED_LAYERS,
)


def _trunc(tensor: torch.Tensor) -> None:
    nn.init.trunc_normal_(tensor, std=0.02)


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_ratio: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} is not divisible by heads {n_heads}")
        self.n_heads = n_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        hidden = ffn_ratio * d_model
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)

    def forward(self, x: torch.Tensor, key_pad: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        b, n, d = h.shape
        hd = d // self.n_heads
        qkv = self.qkv(h).view(b, n, 3, self.n_heads, hd)
        q, k, v = qkv.unbind(dim=2)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        scale = hd ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        scores = scores.masked_fill(key_pad[:, None, None, :], -1e9)
        weights = torch.softmax(scores, dim=-1)
        y = torch.matmul(weights, v)
        y = y.permute(0, 2, 1, 3).contiguous().view(b, n, d)
        x = x + self.proj(y)
        x = x + self.fc2(F.gelu(self.fc1(self.ln2(x))))
        return x


def patchify(mel: torch.Tensor) -> torch.Tensor:
    if mel.ndim != 3:
        raise ValueError(f"mel must be [B, frames, bins], got {tuple(mel.shape)}")
    b, frames, bins = mel.shape
    if frames != GRID_T * PATCH_FRAMES or bins != N_MELS:
        raise ValueError(f"mel shape {tuple(mel.shape)} is not a full crop")
    x = mel.view(b, GRID_T, PATCH_FRAMES, GRID_F, PATCH_BINS)
    x = x.permute(0, 1, 3, 2, 4).contiguous()
    return x.view(b, GRID_T, GRID_F, PATCH_DIM)


def _sincos(length: int, dim: int) -> torch.Tensor:
    if dim % 2 != 0:
        raise ValueError("sincos dim must be even")
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / float(dim))
    )
    pe = torch.zeros(length, dim)
    pe[:, 0::2] = torch.sin(position * div)
    pe[:, 1::2] = torch.cos(position * div)
    return pe


class LoopEarDEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch = nn.Linear(PATCH_DIM, D_MODEL)
        self.cls = nn.Parameter(torch.empty(D_MODEL))
        self.pos_time = nn.Parameter(torch.empty(GRID_T, D_MODEL))
        self.pos_freq = nn.Parameter(torch.empty(GRID_F, D_MODEL))
        self.blocks = nn.ModuleList(
            [Block(D_MODEL, N_HEADS, FFN_RATIO) for _ in range(N_LAYERS)]
        )
        self.norm = nn.LayerNorm(D_MODEL)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _trunc(self.patch.weight)
        nn.init.zeros_(self.patch.bias)
        _trunc(self.cls)
        _trunc(self.pos_time)
        _trunc(self.pos_freq)
        for block in self.blocks:
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
        tokens = self.patch(patchify(mel))
        tokens = tokens + self.pos_time[None, :, None, :] + self.pos_freq[None, None, :, :]
        return self._attend(tokens, visible)

    def _attend(
        self, tokens: torch.Tensor, visible: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, f, d = tokens.shape
        p = t * f
        flat = tokens.reshape(b, p, d)
        vis = visible.reshape(b, p)
        counts = vis.sum(dim=1)
        max_n = int(counts.max().item()) if b else 0
        pos = torch.cumsum(vis.int(), dim=1) - 1
        pos_clamped = pos.clamp(min=0)
        src = torch.where(vis.unsqueeze(-1), flat, torch.zeros_like(flat))
        if max_n == 0:
            packed = flat.new_zeros(b, 0, d)
        else:
            packed = flat.new_zeros(b, max_n, d)
            packed = packed.scatter_add(
                1, pos_clamped.unsqueeze(-1).expand(b, p, d), src
            )
        cls = self.cls.view(1, 1, d).expand(b, 1, d)
        seq = torch.cat([cls, packed], dim=1)
        key_pad = torch.ones(b, seq.shape[1], dtype=torch.bool, device=seq.device)
        key_pad[:, 0] = False
        if max_n > 0:
            slots = torch.arange(max_n, device=seq.device).view(1, max_n)
            key_pad[:, 1:] = slots >= counts[:, None]
        for block in self.blocks:
            seq = block(seq, key_pad)
        seq = self.norm(seq)
        cls_out = seq[:, 0]
        grid = flat.new_zeros(b, p, d)
        if max_n > 0:
            gathered = torch.gather(
                seq[:, 1:, :], 1, pos_clamped.unsqueeze(-1).expand(b, p, d)
            )
            grid = torch.where(vis.unsqueeze(-1), gathered, torch.zeros_like(flat))
        return grid.view(b, t, f, d), cls_out


class MLPHead(nn.Module):
    def __init__(self, d_in: int, d_hid: int, d_out: int) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(d_in)
        self.fc1 = nn.Linear(d_in, d_hid)
        self.fc2 = nn.Linear(d_hid, d_out)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.ones_(self.ln.weight)
        nn.init.zeros_(self.ln.bias)
        _trunc(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        _trunc(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(self.ln(x))))


class Predictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mask_token = nn.Parameter(torch.empty(D_MODEL))
        self.in_proj = nn.Linear(D_MODEL, D_PRED)
        self.view_embed = nn.Parameter(torch.empty(2, D_PRED))
        self.blocks = nn.ModuleList(
            [Block(D_PRED, PRED_HEADS, FFN_RATIO) for _ in range(PRED_LAYERS)]
        )
        self.final_ln = nn.LayerNorm(D_PRED)
        self.out_proj = nn.Linear(D_PRED, D_MODEL)
        t_pe = _sincos(GRID_T, D_PRED // 2)
        f_pe = _sincos(GRID_F, D_PRED // 2)
        pos = torch.cat(
            [
                t_pe[:, None, :].expand(GRID_T, GRID_F, D_PRED // 2),
                f_pe[None, :, :].expand(GRID_T, GRID_F, D_PRED // 2),
            ],
            dim=-1,
        )
        self.register_buffer("pos", pos, persistent=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _trunc(self.mask_token)
        _trunc(self.in_proj.weight)
        nn.init.zeros_(self.in_proj.bias)
        _trunc(self.view_embed)
        for block in self.blocks:
            for mod in block.modules():
                if isinstance(mod, nn.Linear):
                    _trunc(mod.weight)
                    if mod.bias is not None:
                        nn.init.zeros_(mod.bias)
                elif isinstance(mod, nn.LayerNorm):
                    nn.init.ones_(mod.weight)
                    nn.init.zeros_(mod.bias)
        nn.init.ones_(self.final_ln.weight)
        nn.init.zeros_(self.final_ln.bias)
        _trunc(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def inputs(
        self, grid: torch.Tensor, masked: torch.Tensor, pad: torch.Tensor
    ) -> torch.Tensor:
        pred_in = grid.clone()
        pred_in = torch.where(masked.unsqueeze(-1), self.mask_token.view(1, 1, 1, -1), pred_in)
        pred_in = torch.where(pad.unsqueeze(-1), torch.zeros_like(pred_in), pred_in)
        return pred_in

    def forward(
        self,
        grid: torch.Tensor,
        masked: torch.Tensor,
        pad: torch.Tensor,
        view_index: int,
    ) -> torch.Tensor:
        pred_in = self.inputs(grid, masked, pad)
        h = self.in_proj(pred_in)
        h = h + self.pos.unsqueeze(0) + self.view_embed[view_index].view(1, 1, 1, -1)
        b, t, f, d = h.shape
        flat = h.reshape(b, t * f, d)
        key_pad = pad.reshape(b, t * f)
        for block in self.blocks:
            flat = block(flat, key_pad)
        flat = self.final_ln(flat)
        return self.out_proj(flat).view(b, t, f, D_MODEL)


class LoopEarDSSL(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = LoopEarDEncoder()
        self.predictor = Predictor()
        self.global_head = MLPHead(D_MODEL, GLOBAL_HIDDEN, D_GLOBAL)
        self.local_head = MLPHead(D_MODEL, D_MODEL, D_LOCAL)
        self.teacher_encoder = LoopEarDEncoder()
        self.teacher_local = MLPHead(D_MODEL, D_MODEL, D_LOCAL)
        self.reset_teacher()

    def train(self, mode: bool = True) -> "LoopEarDSSL":
        super().train(mode)
        self.teacher_encoder.eval()
        self.teacher_local.eval()
        return self

    def teacher_modules(self) -> tuple[nn.Module, nn.Module]:
        return self.teacher_encoder, self.teacher_local

    def reset_teacher(self) -> None:
        self.teacher_encoder.load_state_dict(self.encoder.state_dict())
        self.teacher_local.load_state_dict(self.local_head.state_dict())
        for module in self.teacher_modules():
            module.eval()
            for param in module.parameters():
                param.requires_grad_(False)

    def student_parameters(self) -> list[nn.Parameter]:
        modules = (self.encoder, self.predictor, self.global_head, self.local_head)
        params: list[nn.Parameter] = []
        for module in modules:
            params.extend(p for p in module.parameters() if p.requires_grad)
        return params

    @torch.no_grad()
    def ema_update(self, momentum: float) -> None:
        pairs = (
            (self.encoder, self.teacher_encoder),
            (self.local_head, self.teacher_local),
        )
        for src, dst in pairs:
            for ps, pt in zip(src.parameters(), dst.parameters()):
                pt.mul_(momentum).add_(ps.detach(), alpha=1.0 - momentum)

    def encode_student(
        self, mel: torch.Tensor, visible: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(mel, visible)

    def project_global(self, cls: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.global_head(cls), dim=-1, eps=1e-6)

    def project_local_student(self, pred: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.local_head(pred), dim=-1, eps=1e-6)

    @torch.no_grad()
    def encode_teacher(
        self, mel: torch.Tensor, visible: torch.Tensor
    ) -> torch.Tensor:
        grid, _cls = self.teacher_encoder(mel, visible)
        return F.normalize(self.teacher_local(grid), dim=-1, eps=1e-6)
