import math
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from scipy.optimize import linear_sum_assignment
from torch import nn

from loopear.encoder import SR

N_MELS: int = 128
WIN: int = 400
HOP: int = 160
N_FFT: int = 512
WIDTH: int = 96
EXPAND: int = 2
KERNEL: int = 5
N_BLOCKS: int = 12
LATENT: int = 64
MAX_SLOTS: int = 6
SLOT_ITERS: int = 2
CONTEXT: int = 32
PARAM_MIN: int = 500_000
PARAM_MAX: int = 2_000_000
TAU: float = 0.05
MARGIN: float = 0.25
TEMPERATURE: float = 0.05
LAM_SAME: float = 1.0
LAM_STABLE: float = 1.0
LAM_MOVE: float = 1.0
LAM_HOLD: float = 1.0
LAM_OCC: float = 0.05
LAM_DIR: float = 1.0
PAIR_KINDS: frozenset[str] = frozenset({"same", "intervene", "global", "appear"})


def _mel_filterbank(n_fft: int, n_mels: int, sr: int) -> torch.Tensor:
    fb = torchaudio.functional.melscale_fbanks(
        n_freqs=n_fft // 2 + 1,
        f_min=20.0,
        f_max=float(sr) / 2.0,
        n_mels=n_mels,
        sample_rate=sr,
        norm="slaney",
        mel_scale="htk",
    )
    if fb.shape[0] != n_mels:
        fb = fb.transpose(0, 1).contiguous()
    return fb.to(dtype=torch.float32)


class CausalConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int,
        stride: int = 1,
        groups: int = 1,
    ):
        super().__init__()
        self.left = int(kernel - 1)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel,
            stride=stride,
            groups=groups,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.left, 0)))


class FrameNorm(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class DSBlock(nn.Module):
    def __init__(self, channels: int, expand: int, kernel: int):
        super().__init__()
        hidden = channels * expand
        self.norm = FrameNorm(channels)
        self.pw1 = nn.Conv1d(channels, hidden, 1, bias=False)
        self.dw = CausalConv1d(hidden, hidden, kernel, groups=hidden)
        self.pw2 = nn.Conv1d(hidden, channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.hardswish(self.pw1(self.norm(x)))
        y = F.hardswish(self.dw(y))
        return x + self.pw2(y)


def mel_frame_count(n_samples: int) -> int:
    if n_samples < N_FFT:
        return 0
    return 1 + (int(n_samples) - N_FFT) // HOP


def stem_frame_count(n_mel: int) -> int:
    if n_mel <= 0:
        return 0
    return (int(n_mel) + 4 - KERNEL) // 2 + 1


def straight_through_gate(g: torch.Tensor) -> torch.Tensor:
    hard = (g > 0.5).to(dtype=g.dtype)
    return g + (hard - g).detach()


def stem_rate_logmel(mel: torch.Tensor, frames: int) -> torch.Tensor:
    if frames <= 0:
        raise ValueError("frames must be positive")
    usable = (int(frames) - 1) * 2 + 1
    if mel.shape[-1] < usable:
        raise ValueError("log-mel is shorter than the stem grid")
    cropped = mel[..., :usable]
    paired = cropped[..., : usable - 1].reshape(
        mel.shape[0], mel.shape[1], frames - 1, 2
    )
    body = paired.mean(dim=-1)
    last = cropped[..., -1:]
    return torch.cat([body, last], dim=-1)


class ProvisionalListener(nn.Module):
    def __init__(
        self,
        n_mels: int = N_MELS,
        width: int = WIDTH,
        expand: int = EXPAND,
        kernel: int = KERNEL,
        n_blocks: int = N_BLOCKS,
        latent: int = LATENT,
        slots: int = MAX_SLOTS,
        iterations: int = SLOT_ITERS,
        context: int = CONTEXT,
        sr: int = SR,
    ):
        super().__init__()
        if slots < 1:
            raise ValueError("slots must be positive")
        if iterations < 1:
            raise ValueError("iterations must be positive")
        if context < 1:
            raise ValueError("context must be positive")
        self.sr = int(sr)
        self.n_mels = int(n_mels)
        self.latent = int(latent)
        self.slots = int(slots)
        self.iterations = int(iterations)
        self.context = int(context)
        self.register_buffer("window", torch.hann_window(WIN), persistent=False)
        self.register_buffer(
            "mel_fb", _mel_filterbank(N_FFT, n_mels, sr), persistent=False
        )
        self.stem = nn.Sequential(
            CausalConv1d(n_mels, width, kernel, stride=2),
            FrameNorm(width),
            nn.Hardswish(),
        )
        self.blocks = nn.Sequential(
            *[DSBlock(width, expand, kernel) for _ in range(n_blocks)]
        )
        self.out_norm = FrameNorm(width)
        self.frame = nn.Linear(width, latent)
        self.q = nn.Linear(latent, latent, bias=False)
        self.k = nn.Linear(latent, latent, bias=False)
        self.v = nn.Linear(latent, latent, bias=False)
        self.norm_slot = nn.LayerNorm(latent)
        self.norm_upd = nn.LayerNorm(latent)
        self.mlp = nn.Sequential(
            nn.Linear(latent, latent * 2),
            nn.Hardswish(),
            nn.Linear(latent * 2, latent),
        )
        self.occ = nn.Linear(latent, 1)
        self.to_mel = nn.Linear(latent, n_mels, bias=False)
        self.mu = nn.Parameter(torch.zeros(latent))
        self.log_sigma = nn.Parameter(torch.full((latent,), -4.0))
        nn.init.constant_(self.occ.bias, 2.0)

    def logmel(self, wav: torch.Tensor) -> torch.Tensor:
        spec = torch.stft(
            wav,
            n_fft=N_FFT,
            hop_length=HOP,
            win_length=WIN,
            window=self.window,
            center=False,
            return_complex=True,
        )
        power = spec.abs().square()
        mel = torch.matmul(self.mel_fb, power)
        return torch.log(mel.clamp_min(1e-5))

    def features(self, wav: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mel = self.logmel(wav)
        x = self.blocks(self.stem(mel))
        x = self.out_norm(x)
        return mel, self.frame(x.transpose(1, 2))

    def _windows(self, frames: torch.Tensor) -> torch.Tensor:
        padded = F.pad(frames, (0, 0, self.context - 1, 0))
        unfolded = padded.unfold(dimension=1, size=self.context, step=1)
        return unfolded.transpose(-1, -2)

    def _attend(self, slots: torch.Tensor, windows: torch.Tensor) -> torch.Tensor:
        q = self.q(self.norm_slot(slots))
        k = self.k(windows)
        v = self.v(windows)
        scale = math.sqrt(float(self.latent))
        logits = torch.einsum("btkd,btwd->btkw", q, k) / scale
        attn = torch.softmax(logits, dim=2)
        weights = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        updates = torch.einsum("btkw,btwd->btkd", weights, v)
        return slots + self.mlp(self.norm_upd(updates))

    def forward(
        self,
        wav: torch.Tensor,
        generator: torch.Generator | None = None,
        slot_init: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if wav.ndim != 2:
            raise ValueError("wav must have shape (batch, samples)")
        mel, frames = self.features(wav)
        b, t, d = frames.shape
        windows = self._windows(frames)
        if slot_init is None:
            eps = torch.randn(
                b, self.slots, d, device=frames.device, dtype=frames.dtype, generator=generator
            )
            slot0 = self.mu + self.log_sigma.exp().clamp(max=2.0) * eps
        else:
            slot0 = slot_init
            if slot0.shape != (b, self.slots, d):
                raise ValueError("slot_init must have shape (batch, slots, latent)")
        slots = slot0.unsqueeze(1).expand(b, t, self.slots, d).contiguous()
        for _ in range(self.iterations):
            slots = self._attend(slots, windows)
        occupancy = torch.sigmoid(self.occ(slots).squeeze(-1))
        gate = straight_through_gate(occupancy)
        visible = gate.unsqueeze(-1) * slots
        reconstruction = self.to_mel(visible).sum(dim=2)
        target = stem_rate_logmel(mel, t).transpose(1, 2)
        if target.shape[1] != reconstruction.shape[1]:
            raise RuntimeError(
                f"stem grid {reconstruction.shape[1]} != log-mel grid {target.shape[1]}"
            )
        return {
            "visible": visible,
            "gate": gate,
            "occupancy": occupancy,
            "reconstruction": reconstruction,
            "target": target,
            "slots": slots,
        }


def parameter_count(model: nn.Module) -> int:
    return int(sum(param.numel() for param in model.parameters()))


def conv_linear_macs(model: nn.Module, wav: torch.Tensor) -> int:
    total = 0

    def conv_hook(mod: nn.Module, _inp: tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
        nonlocal total
        if not isinstance(mod, nn.Conv1d):
            return
        _b, channels, steps = out.shape
        cin = int(mod.in_channels // mod.groups)
        kernel = int(mod.kernel_size[0])
        total += int(channels * steps * cin * kernel)

    def linear_hook(mod: nn.Module, _inp: tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
        nonlocal total
        if isinstance(mod, nn.Linear):
            total += int(out.numel() * mod.in_features)

    handles = []
    for module in model.modules():
        if isinstance(module, nn.Conv1d):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))
    with torch.inference_mode():
        model(wav)
    for handle in handles:
        handle.remove()
    return int(total)


def _pairwise_cost(z: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    left = z.detach().reshape(z.shape[0], z.shape[1], -1)
    right = z2.detach().reshape(z2.shape[0], z2.shape[1], -1)
    return torch.cdist(left, right, p=2)


def match_slots(z: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    cost = _pairwise_cost(z, z2).cpu().numpy()
    rows = []
    for batch in range(cost.shape[0]):
        _r, c = linear_sum_assignment(cost[batch])
        rows.append(c)
    return torch.tensor(np.stack(rows), device=z.device, dtype=torch.long)


def _gather(values: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    view = index.view(index.shape[0], index.shape[1], *([1] * (values.ndim - 2)))
    return torch.gather(values, 1, view.expand_as(values))


def _slot_distance(z: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    mse = (z - z2).pow(2).mean(dim=(-1, -2))
    return torch.sqrt(mse + 1e-8)


def _cosine(z: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    left = z.reshape(z.shape[0], z.shape[1], -1)
    right = z2.reshape(z2.shape[0], z2.shape[1], -1)
    left = F.normalize(left, dim=-1, eps=1e-6)
    right = F.normalize(right, dim=-1, eps=1e-6)
    return (left * right).sum(dim=-1)


def variation_loss(
    z: torch.Tensor,
    g: torch.Tensor,
    yhat: torch.Tensor,
    y: torch.Tensor,
    z2: torch.Tensor,
    g2: torch.Tensor,
    yhat2: torch.Tensor,
    y2: torch.Tensor,
    kind: str,
    unchanged_present: bool = False,
    tau: float = TAU,
    margin: float = MARGIN,
    temperature: float = TEMPERATURE,
    lam_same: float = LAM_SAME,
    lam_stable: float = LAM_STABLE,
    lam_move: float = LAM_MOVE,
    lam_hold: float = LAM_HOLD,
    lam_occ: float = LAM_OCC,
    lam_dir: float = LAM_DIR,
) -> dict[str, torch.Tensor]:
    if kind not in PAIR_KINDS:
        raise ValueError(f"unknown pair kind: {kind}")
    if z.shape != z2.shape:
        raise ValueError("slot tensors must share a shape")
    index = match_slots(z, z2)
    z2m = _gather(z2, index)
    g2m = _gather(g2, index)
    d = _slot_distance(z, z2m)
    u = g * g2m
    evidence = 0.5 * ((yhat - y).abs().mean() + (yhat2 - y2).abs().mean())
    occ = 0.5 * (g.mean() + g2.mean())
    same = (u * F.relu(d - tau).square()).sum(dim=-1) / u.sum(dim=-1).clamp_min(1e-6)
    selector = torch.softmax(d / temperature, dim=-1).detach()
    stable = ((1.0 - selector) * u * d.square()).sum(dim=-1) / u.sum(dim=-1).clamp_min(1e-6)
    move_weight = selector * u
    moved = (move_weight * d).sum(dim=-1) / move_weight.sum(dim=-1).clamp_min(1e-6)
    move = F.relu(margin - moved).square()
    masked = torch.where(u > 0, -d / temperature, torch.full_like(d, -1e9))
    hold_w = torch.softmax(masked, dim=-1)
    coverage = u.sum(dim=-1)
    hold_dist = (hold_w * d.square()).sum(dim=-1)
    hold = torch.where(coverage > 0, hold_dist, torch.ones_like(hold_dist))
    hold = hold + F.relu(1.0 - coverage).square()
    direction = (u * F.relu(1.0 - _cosine(z, z2m)).square()).sum(dim=-1)
    direction = direction / u.sum(dim=-1).clamp_min(1e-6)
    zero = evidence.new_zeros(())
    same_term = same.mean() if kind in {"same", "appear"} else zero
    stable_term = stable.mean() if kind == "intervene" else zero
    move_term = move.mean() if kind == "intervene" else zero
    hold_term = hold.mean() if kind == "intervene" and unchanged_present else zero
    dir_term = direction.mean() if kind == "global" else zero
    total = (
        evidence
        + lam_same * same_term
        + lam_stable * stable_term
        + lam_move * move_term
        + lam_hold * hold_term
        + lam_occ * occ
        + lam_dir * dir_term
    )
    return {
        "total": total,
        "evidence": evidence,
        "same": same_term,
        "stable": stable_term,
        "move": move_term,
        "hold": hold_term,
        "occupancy": occ,
        "direction": dir_term,
    }


def _pack(
    slots: list[list[float]],
    gates: list[float],
    slots2: list[list[float]],
    gates2: list[float],
    y: list[float] | None = None,
    yhat: list[float] | None = None,
    y2: list[float] | None = None,
    yhat2: list[float] | None = None,
) -> dict[str, torch.Tensor]:
    z = torch.tensor(slots, dtype=torch.float32).view(1, len(slots), 1, -1)
    z2 = torch.tensor(slots2, dtype=torch.float32).view(1, len(slots2), 1, -1)
    g = torch.tensor(gates, dtype=torch.float32).view(1, -1)
    g2 = torch.tensor(gates2, dtype=torch.float32).view(1, -1)
    feat = z.shape[-1]
    if y is None:
        target = torch.zeros(1, 1, feat)
        pred = target.clone()
        target2 = target.clone()
        pred2 = target.clone()
    else:
        target = torch.tensor(y, dtype=torch.float32).view(1, 1, -1)
        pred = torch.tensor(yhat, dtype=torch.float32).view(1, 1, -1)
        target2 = torch.tensor(y2, dtype=torch.float32).view(1, 1, -1)
        pred2 = torch.tensor(yhat2, dtype=torch.float32).view(1, 1, -1)
    return {"z": z, "g": g, "y": target, "yhat": pred, "z2": z2, "g2": g2, "y2": target2, "yhat2": pred2}


def _total(packed: dict[str, torch.Tensor], kind: str, unchanged_present: bool = False) -> float:
    loss = variation_loss(kind=kind, unchanged_present=unchanged_present, **packed)
    return float(loss["total"].detach())


def degenerate_report() -> dict[str, float]:
    selective = _pack(
        [[1.0, 0.0], [0.0, 1.0]],
        [1.0, 1.0],
        [[1.0, 0.0], [0.0, 2.0]],
        [1.0, 1.0],
        [1.0, 1.0],
        [1.0, 1.0],
        [1.0, 2.0],
        [1.0, 2.0],
    )
    single = _pack(
        [[1.0, 1.0], [0.0, 0.0]],
        [1.0, 0.0],
        [[1.0, 2.0], [0.0, 0.0]],
        [1.0, 0.0],
        [1.0, 1.0],
        [1.0, 1.0],
        [1.0, 2.0],
        [1.0, 2.0],
    )
    duplicate = _pack(
        [[1.0, 0.0], [1.0, 0.0]],
        [1.0, 1.0],
        [[0.0, 1.0], [0.0, 1.0]],
        [1.0, 1.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
    )
    constant = _pack(
        [[1.0, 0.0], [0.0, 1.0]],
        [1.0, 1.0],
        [[1.0, 0.0], [0.0, 1.0]],
        [1.0, 1.0],
        [1.0, 0.0],
        [0.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0],
    )
    hidden = _pack(
        [[1.0, 0.0], [0.0, 4.0]],
        [1.0, 0.0],
        [[1.0, 0.0], [4.0, 0.0]],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
    )
    nuisance = _pack(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [1.0, 1.0],
        [[1.0, 0.0, 0.0], [0.0, 2.0, 5.0]],
        [1.0, 1.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 2.0],
        [1.0, 2.0],
    )
    clean = _pack(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [1.0, 1.0],
        [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
        [1.0, 1.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 2.0],
        [1.0, 2.0],
    )
    quiet_keep = _pack(
        [[1.0, 0.0], [0.0, 0.2]],
        [1.0, 1.0],
        [[1.0, 0.0], [0.0, 0.2]],
        [1.0, 1.0],
        [1.0, 0.01],
        [1.0, 0.01],
        [1.0, 0.01],
        [1.0, 0.01],
    )
    quiet_drop = _pack(
        [[1.0, 0.0], [0.0, 0.0]],
        [1.0, 0.0],
        [[1.0, 0.0], [0.0, 0.0]],
        [1.0, 0.0],
        [1.0, 0.01],
        [1.0, 0.0],
        [1.0, 0.01],
        [1.0, 0.0],
    )
    quiet_move_keep = _pack(
        [[1.0, 0.0], [0.0, 0.2]],
        [1.0, 1.0],
        [[1.0, 0.0], [0.0, 0.8]],
        [1.0, 1.0],
        [1.0, 0.01],
        [1.0, 0.01],
        [1.0, 0.04],
        [1.0, 0.04],
    )
    quiet_move_drop = _pack(
        [[1.0, 0.0], [0.0, 0.0]],
        [1.0, 0.0],
        [[1.0, 0.0], [0.0, 0.0]],
        [1.0, 0.0],
        [1.0, 0.01],
        [1.0, 0.0],
        [1.0, 0.04],
        [1.0, 0.0],
    )
    merged = _pack(
        [[1.0, 1.0], [0.0, 0.0]],
        [1.0, 0.0],
        [[2.0, 2.0], [0.0, 0.0]],
        [1.0, 0.0],
        [1.0, 1.0],
        [1.0, 1.0],
        [2.0, 2.0],
        [2.0, 2.0],
    )
    split = _pack(
        [[1.0, 0.0], [0.0, 1.0]],
        [1.0, 1.0],
        [[2.0, 0.0], [0.0, 2.0]],
        [1.0, 1.0],
        [1.0, 1.0],
        [1.0, 1.0],
        [2.0, 2.0],
        [2.0, 2.0],
    )
    gain = _pack(
        [[1.0, 0.0], [0.0, 1.0]],
        [1.0, 1.0],
        [[2.0, 0.0], [0.0, 2.0]],
        [1.0, 1.0],
    )
    within_same = _pack(
        [[1.0, 0.0], [0.0, 1.0]],
        [1.0, 1.0],
        [[1.0, 0.0], [0.0, 2.0]],
        [1.0, 1.0],
    )
    appear = _pack(
        [[1.0, 0.0], [0.0, 0.0]],
        [1.0, 0.0],
        [[1.0, 0.0], [0.0, 1.0]],
        [1.0, 1.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [1.0, 1.0],
    )
    appear_miss = _pack(
        [[1.0, 0.0], [0.0, 0.0]],
        [1.0, 0.0],
        [[1.0, 0.0], [0.0, 0.0]],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [1.0, 0.0],
    )
    two_move = _pack(
        [[1.0, 0.0], [0.0, 1.0], [0.2, 0.2]],
        [1.0, 1.0, 1.0],
        [[1.6, 0.0], [0.0, 1.6], [0.2, 0.2]],
        [1.0, 1.0, 1.0],
    )
    one_move = _pack(
        [[1.0, 0.0], [0.0, 1.0], [0.2, 0.2]],
        [1.0, 1.0, 1.0],
        [[2.2, 0.0], [0.0, 1.0], [0.2, 0.2]],
        [1.0, 1.0, 1.0],
    )
    natural = _pack(
        [[1.0, 0.0], [0.0, 1.0]],
        [1.0, 1.0],
        [[1.02, 0.0], [0.0, 0.99]],
        [1.0, 1.0],
    )
    shifted = _pack(
        [[1.0, 0.0], [0.0, 1.0]],
        [1.0, 1.0],
        [[1.0, 0.0], [0.0, 1.4]],
        [1.0, 1.0],
    )
    values = {
        "selective_intervene": _total(selective, "intervene", True),
        "single_slot_intervene": _total(single, "intervene", True),
        "duplicate_intervene": _total(duplicate, "intervene", True),
        "constant_intervene": _total(constant, "intervene", True),
        "hidden_inactive_intervene": _total(hidden, "intervene", True),
        "nuisance_intervene": _total(nuisance, "intervene", True),
        "clean_intervene": _total(clean, "intervene", True),
        "quiet_keep_same": _total(quiet_keep, "same"),
        "quiet_drop_same": _total(quiet_drop, "same"),
        "quiet_keep_intervene": _total(quiet_move_keep, "intervene", True),
        "quiet_drop_intervene": _total(quiet_move_drop, "intervene", True),
        "correlated_merged": _total(merged, "intervene", False),
        "correlated_split": _total(split, "intervene", False),
        "gain_as_intervene": _total(gain, "intervene", True),
        "gain_as_global": _total(gain, "global"),
        "within_as_same": _total(within_same, "same"),
        "within_as_intervene": _total(within_same, "intervene", True),
        "appear_represented": _total(appear, "appear"),
        "appear_missed": _total(appear_miss, "appear"),
        "two_slots_move": _total(two_move, "intervene", True),
        "one_slot_moves": _total(one_move, "intervene", True),
        "natural_same": _total(natural, "same"),
        "shifted_same": _total(shifted, "same"),
    }
    swapped = variation_loss(
        z=selective["z2"],
        g=selective["g2"],
        yhat=selective["yhat2"],
        y=selective["y2"],
        z2=selective["z"],
        g2=selective["g"],
        yhat2=selective["yhat"],
        y2=selective["y"],
        kind="intervene",
        unchanged_present=True,
    )
    values["selective_swapped"] = float(swapped["total"].detach())
    perm = selective["z"][:, [1, 0]].contiguous()
    permed = variation_loss(
        z=perm,
        g=selective["g"][:, [1, 0]],
        yhat=selective["yhat"],
        y=selective["y"],
        z2=selective["z2"][:, [1, 0]].contiguous(),
        g2=selective["g2"][:, [1, 0]],
        yhat2=selective["yhat2"],
        y2=selective["y2"],
        kind="intervene",
        unchanged_present=True,
    )
    values["selective_permuted"] = float(permed["total"].detach())
    values.update(_time_average_cases())
    return values


def _time_average_cases() -> dict[str, float]:
    def case(frames: int, changed: int, delta: float) -> float:
        z = torch.zeros(1, 2, frames, 2)
        z[0, 0, :, 0] = 1.0
        z[0, 1, :, 1] = 1.0
        z2 = z.clone()
        if changed > 0:
            z2[0, 1, :changed, 1] = z2[0, 1, :changed, 1] + delta
        g = torch.ones(1, 2)
        y = torch.zeros(1, 1, 2)
        loss = variation_loss(
            z=z,
            g=g,
            yhat=y,
            y=y,
            z2=z2,
            g2=g,
            yhat2=y,
            y2=y,
            kind="intervene",
            unchanged_present=True,
        )
        return float(loss["total"].detach())

    smear = case(32, 32, 1.0 / math.sqrt(32.0))
    return {
        "time_no_move_32": case(32, 0, 0.0),
        "time_brief_32": case(32, 1, 1.0),
        "time_sustained_32": case(32, 32, 1.0),
        "time_matched_smear_32": smear,
    }


def _block_changes_frame(
    model: ProvisionalListener,
    n_samples: int,
    frame: int,
    start: int,
    stop: int,
    init: torch.Tensor,
    ref: torch.Tensor,
) -> bool:
    wav = torch.zeros(1, n_samples)
    wav[0, start:stop] = 1.0
    with torch.inference_mode():
        trial = model(wav, slot_init=init)["reconstruction"][0, frame]
    return not torch.allclose(trial, ref, atol=1e-5, rtol=0.0)


def measure_support(model: ProvisionalListener, n_samples: int = 8000) -> dict[str, int]:
    model.eval()
    init = torch.zeros(1, model.slots, model.latent)
    with torch.inference_mode():
        base = model(torch.zeros(1, n_samples), slot_init=init)["reconstruction"]
    frame = min(4, int(base.shape[1]) - 1)
    last = int(base.shape[1]) - 1
    step = HOP * 2
    latest = -1
    earliest = n_samples
    for start in range(0, n_samples, step):
        stop = min(n_samples, start + step)
        if _block_changes_frame(model, n_samples, frame, start, stop, init, base[0, frame]):
            latest = stop - 1
        if _block_changes_frame(model, n_samples, last, start, stop, init, base[0, last]):
            earliest = min(earliest, start)
    nominal = frame * step
    return {
        "probe_frame": int(frame),
        "last_frame": int(last),
        "last_affecting_sample": int(latest),
        "lookahead_samples": int(max(0, latest - nominal)),
        "first_affecting_sample_of_last_frame": int(earliest),
        "past_samples_of_last_frame": int(max(0, n_samples - 1 - earliest)),
        "frames": int(base.shape[1]),
        "block_samples": int(step),
    }


def measure_runtime(model: ProvisionalListener, seconds: float = 10.0, repeats: int = 3) -> dict[str, float]:
    n_samples = int(seconds * model.sr)
    generator = torch.Generator().manual_seed(0)
    wav = torch.randn(1, n_samples, generator=generator)
    init = torch.zeros(1, model.slots, model.latent)
    previous = torch.get_num_threads()
    timings: dict[int, float] = {}
    model.eval()
    try:
        for threads in (1, 4):
            torch.set_num_threads(threads)
            with torch.inference_mode():
                model(wav, slot_init=init)
                samples = []
                for _ in range(repeats):
                    start = time.perf_counter()
                    model(wav, slot_init=init)
                    samples.append(time.perf_counter() - start)
            timings[threads] = float(sorted(samples)[len(samples) // 2])
    finally:
        torch.set_num_threads(previous)
    macs = conv_linear_macs(model, wav[:1, : model.sr])
    return {
        "seconds": float(seconds),
        "median_s_1thread": timings[1],
        "median_s_4thread": timings[4],
        "rtf_1thread": timings[1] / float(seconds),
        "rtf_4thread": timings[4] / float(seconds),
        "conv_linear_macs_per_s": float(macs),
    }


def measure_step(model: ProvisionalListener) -> float:
    generator = torch.Generator().manual_seed(1)
    wav = torch.randn(2, model.sr, generator=generator)
    wav2 = torch.randn(2, model.sr, generator=generator)
    model.train()
    start = time.perf_counter()
    left = model(wav)
    right = model(wav2)
    loss = variation_loss(
        z=left["visible"].permute(0, 2, 1, 3),
        g=left["gate"].mean(dim=1),
        yhat=left["reconstruction"],
        y=left["target"],
        z2=right["visible"].permute(0, 2, 1, 3),
        g2=right["gate"].mean(dim=1),
        yhat2=right["reconstruction"],
        y2=right["target"],
        kind="intervene",
        unchanged_present=True,
    )["total"]
    loss.backward()
    elapsed = time.perf_counter() - start
    model.zero_grad(set_to_none=True)
    return float(elapsed)


def full_report(seconds: float = 10.0) -> dict[str, Any]:
    model = ProvisionalListener()
    runtime = measure_runtime(model, seconds=seconds)
    support = measure_support(model, n_samples=48_000)
    step_s = measure_step(ProvisionalListener())
    return {
        "gpu_usd": 0.0,
        "params": parameter_count(model),
        "max_slots": model.slots,
        "latent": model.latent,
        "context_frames": model.context,
        "slot_iterations": model.iterations,
        "blocks": N_BLOCKS,
        "width": WIDTH,
        **runtime,
        **{f"support_{key}": value for key, value in support.items()},
        "one_step_s_batch2_1s": step_s,
        "degenerates": degenerate_report(),
    }
