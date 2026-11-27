from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch import nn

from loopear.d.model import Block, _trunc, patchify
from loopear.d.objective import broadcast_time_valid, patch_valid_from_frames
from loopear.d.spec import (
    CROP_SAMPLES,
    D_MODEL,
    ENCODER_PARAMS,
    FFN_RATIO,
    GRID_F,
    GRID_T,
    HOP_SAMPLES,
    N_HEADS,
    N_LAYERS,
    PATCH_DIM,
    SR,
)

HORIZONS: tuple[int, ...] = (1, 4, 12)
TAU: float = 0.1
VAR_WEIGHT: float = 0.1
NEIGHBOR: int = 1
VAR_TARGET: float = 1.0
MIX_SNR_LOW: float = -5.0
MIX_SNR_HIGH: float = 5.0
LAMBDA_MIX_MAX: float = 0.5
LAMBDA_MIX_RAMP: int = 1000
CAUSAL_ENCODER_PARAMS: int = ENCODER_PARAMS - D_MODEL


def lambda_mix_at(step: int, ramp_steps: int = LAMBDA_MIX_RAMP, maximum: float = LAMBDA_MIX_MAX) -> float:
    if ramp_steps <= 0:
        return float(maximum)
    progress = min(max(int(step), 0), int(ramp_steps)) / float(ramp_steps)
    return float(maximum) * progress


def _future_disallow(device: torch.device) -> torch.Tensor:
    time_id = torch.arange(GRID_T, device=device).repeat_interleave(GRID_F)
    return time_id.view(1, -1) > time_id.view(-1, 1)


class CausalLoopEarDEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch = nn.Linear(PATCH_DIM, D_MODEL)
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

    def forward(self, mel: torch.Tensor, visible: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if visible.shape[:3] != (mel.shape[0], GRID_T, GRID_F):
            raise ValueError(
                f"visible {tuple(visible.shape)} does not match mel {tuple(mel.shape)}"
            )
        if visible.dtype != torch.bool:
            raise ValueError("visible must be boolean")
        if not bool(visible.flatten(1).any(dim=1).all()):
            raise ValueError("a crop has no visible patches")
        tokens = self.patch(patchify(mel))
        tokens = tokens + self.pos_time[None, :, None, :] + self.pos_freq[None, None, :, :]
        b, t, f, d = tokens.shape
        flat = tokens.reshape(b, t * f, d)
        vis = visible.reshape(b, t * f)
        key_pad = ~vis
        disallow = _future_disallow(flat.device)
        for block in self.blocks:
            flat = block(flat, key_pad, disallow=disallow)
        flat = self.norm(flat)
        flat = flat.masked_fill(~vis.unsqueeze(-1), 0.0)
        grid = flat.view(b, t, f, d)
        return grid, clip_summary(grid, visible)


def clip_summary(grid: torch.Tensor, visible: torch.Tensor) -> torch.Tensor:
    weights = visible.to(grid.dtype).unsqueeze(-1)
    denom = weights.sum(dim=(1, 2)).clamp(min=1.0)
    return (grid * weights).sum(dim=(1, 2)) / denom


def frequency_pool(grid: torch.Tensor) -> torch.Tensor:
    return grid.mean(dim=2)


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1, eps=1e-6)


def teacher_target(grid: torch.Tensor) -> torch.Tensor:
    return l2_normalize(frequency_pool(grid))


class HorizonMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(D_MODEL)
        self.fc1 = nn.Linear(D_MODEL, D_MODEL)
        self.fc2 = nn.Linear(D_MODEL, D_MODEL)
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


class CausalLoopEarD(nn.Module):
    def __init__(self, separation: bool = False) -> None:
        super().__init__()
        self.separation = bool(separation)
        self.encoder = CausalLoopEarDEncoder()
        self.teacher_encoder = CausalLoopEarDEncoder()
        self.horizons = nn.ModuleList([HorizonMLP() for _ in HORIZONS])
        self.sep_heads = nn.ModuleList([HorizonMLP(), HorizonMLP()]) if separation else None
        self.reset_teacher()

    def train(self, mode: bool = True) -> "CausalLoopEarD":
        super().train(mode)
        self.teacher_encoder.eval()
        return self

    def reset_teacher(self) -> None:
        self.teacher_encoder.load_state_dict(self.encoder.state_dict())
        self.teacher_encoder.eval()
        for param in self.teacher_encoder.parameters():
            param.requires_grad_(False)

    def student_parameters(self) -> list[nn.Parameter]:
        modules: list[nn.Module] = [self.encoder, self.horizons]
        if self.sep_heads is not None:
            modules.append(self.sep_heads)
        params: list[nn.Parameter] = []
        for module in modules:
            params.extend(p for p in module.parameters() if p.requires_grad)
        return params

    @torch.no_grad()
    def ema_update(self, momentum: float) -> None:
        for student, teacher in zip(self.encoder.parameters(), self.teacher_encoder.parameters()):
            teacher.mul_(momentum).add_(student.detach(), alpha=1.0 - momentum)


def time_valid_from_frames(frame_valid: torch.Tensor) -> torch.Tensor:
    return patch_valid_from_frames(frame_valid)


def visible_from_frames(frame_valid: torch.Tensor) -> torch.Tensor:
    return broadcast_time_valid(time_valid_from_frames(frame_valid))


def temporal_infonce(
    query: torch.Tensor,
    target: torch.Tensor,
    valid_time: torch.Tensor,
    horizon: int,
    tau: float = TAU,
    neighbor: int = NEIGHBOR,
) -> tuple[torch.Tensor, int]:
    if query.shape != target.shape or query.ndim != 3:
        raise ValueError("query and target must be [B, T, D]")
    if valid_time.shape != query.shape[:2] or valid_time.dtype != torch.bool:
        raise ValueError("valid_time must be a boolean [B, T] mask")
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    if tau <= 0.0:
        raise ValueError("temperature must be positive")
    if neighbor < 1:
        raise ValueError("neighbor distance must be at least 1")
    b, t, _d = query.shape
    if horizon >= t or not bool(valid_time.any()):
        return query.sum() * 0.0, 0
    sim = torch.einsum("btd,bsd->bts", query, target) / float(tau)
    idx = torch.arange(t, device=query.device)
    pos = idx + int(horizon)
    in_range = pos < t
    dist = (idx.view(1, -1) - pos.view(-1, 1)).abs()
    allow = dist != int(neighbor)
    pos_valid = torch.zeros_like(valid_time)
    pos_valid[:, : t - int(horizon)] = valid_time[:, int(horizon) :]
    anchor = valid_time & pos_valid & in_range.view(1, t)
    if not bool(anchor.any()):
        return query.sum() * 0.0, 0
    logits = sim.masked_fill(~allow.view(1, t, t), -1e9)
    logits = logits.masked_fill(~valid_time[:, None, :], -1e9)
    logz = torch.logsumexp(logits, dim=-1)
    pos_index = pos.clamp(max=t - 1).view(1, t, 1).expand(b, t, 1)
    pos_logit = sim.gather(2, pos_index).squeeze(-1)
    loss = -(pos_logit - logz)
    return loss[anchor].mean(), int(anchor.sum())


def batch_variance_hinge(grid: torch.Tensor, visible: torch.Tensor, target: float = VAR_TARGET) -> torch.Tensor:
    if grid.ndim != 4 or visible.shape != grid.shape[:3]:
        raise ValueError("grid and visible shapes do not match")
    flat = grid[visible]
    if flat.shape[0] < 2:
        return grid.sum() * 0.0
    std = torch.sqrt(flat.var(dim=0, unbiased=False) + 1e-4)
    return torch.relu(float(target) - std).mean()


def predictive_loss(
    student_grid: torch.Tensor,
    teacher_grid: torch.Tensor,
    valid_time: torch.Tensor,
    heads: nn.ModuleList,
    horizons: tuple[int, ...] = HORIZONS,
    tau: float = TAU,
) -> tuple[torch.Tensor, dict[str, float]]:
    pooled = frequency_pool(student_grid)
    target = teacher_target(teacher_grid)
    terms = []
    counts: dict[str, float] = {}
    for horizon, head in zip(horizons, heads):
        pred = l2_normalize(head(pooled))
        loss_h, n_h = temporal_infonce(pred, target.detach(), valid_time, int(horizon), tau)
        counts[f"n_h{int(horizon)}"] = float(n_h)
        counts[f"loss_h{int(horizon)}"] = float(loss_h.detach()) if n_h else 0.0
        if n_h:
            terms.append(loss_h)
    if not terms:
        total = student_grid.sum() * 0.0
    else:
        total = torch.stack(terms).mean()
    counts["n_horizons"] = float(len(terms))
    return total, counts


def pit_cosine_loss(
    pred1: torch.Tensor,
    pred2: torch.Tensor,
    tgt1: torch.Tensor,
    tgt2: torch.Tensor,
    valid_time: torch.Tensor,
) -> torch.Tensor:
    if pred1.shape != pred2.shape or pred1.shape != tgt1.shape or pred1.shape != tgt2.shape:
        raise ValueError("PIT streams must share shape [B, T, D]")
    if valid_time.shape != pred1.shape[:2]:
        raise ValueError("PIT validity must be [B, T]")

    def _mean_dist(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        cos = (pred * tgt).sum(dim=-1)
        mask = valid_time.to(pred.dtype)
        return ((1.0 - cos) * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1.0)

    keep = valid_time.any(dim=-1)
    if not bool(keep.any()):
        return pred1.sum() * 0.0
    straight = _mean_dist(pred1, tgt1) + _mean_dist(pred2, tgt2)
    swapped = _mean_dist(pred1, tgt2) + _mean_dist(pred2, tgt1)
    return torch.minimum(straight, swapped)[keep].mean()


def mix_at_snr(clean: torch.Tensor, other: torch.Tensor, snr_db: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if clean.shape != other.shape or clean.ndim != 2:
        raise ValueError("waveforms must be [B, samples]")
    if snr_db.shape != (clean.shape[0],):
        raise ValueError("snr_db must be [B]")
    power_clean = clean.pow(2).mean(dim=-1).clamp_min(1e-8)
    power_other = other.pow(2).mean(dim=-1).clamp_min(1e-8)
    ratio = torch.pow(clean.new_tensor(10.0), snr_db.to(clean.dtype) / 10.0)
    scale = torch.sqrt(power_clean / (power_other * ratio))
    return clean + scale[:, None] * other, scale


def sample_snr(batch: int, generator: torch.Generator, low: float = MIX_SNR_LOW, high: float = MIX_SNR_HIGH) -> torch.Tensor:
    draw = torch.rand(int(batch), generator=generator)
    return float(low) + (float(high) - float(low)) * draw


def crop_waveform(wav: torch.Tensor, start_frame: int) -> torch.Tensor:
    flat = wav.detach().float().reshape(-1)
    sample = int(start_frame) * HOP_SAMPLES
    out = flat.new_zeros(CROP_SAMPLES)
    if sample < 0 or sample >= int(flat.shape[0]):
        return out
    piece = flat[sample : sample + CROP_SAMPLES]
    out[: int(piece.shape[0])] = piece
    return out


def read_crop_waveform(path: Path, start_frame: int) -> torch.Tensor:
    info = sf.info(str(path))
    if int(info.samplerate) != SR:
        raise AssertionError(f"unexpected sample rate {info.samplerate} at {path}")
    sample = int(start_frame) * HOP_SAMPLES
    if sample >= int(info.frames) or sample < 0:
        return torch.zeros(CROP_SAMPLES)
    wav, sr = sf.read(
        str(path),
        start=sample,
        frames=CROP_SAMPLES,
        dtype="float32",
        always_2d=True,
    )
    if int(sr) != SR:
        raise AssertionError(f"unexpected sample rate {sr} at {path}")
    mono = wav.mean(axis=1) if wav.shape[1] != 1 else wav[:, 0]
    out = np.zeros(CROP_SAMPLES, dtype=np.float32)
    out[: int(mono.shape[0])] = np.ascontiguousarray(mono, dtype=np.float32)
    return torch.from_numpy(out)


def causal_step_loss(
    model: CausalLoopEarD,
    mel: torch.Tensor,
    frame_valid: torch.Tensor,
    mix_mel: torch.Tensor | None = None,
    mix_frame_valid: torch.Tensor | None = None,
    step: int = 0,
) -> dict[str, torch.Tensor]:
    visible = visible_from_frames(frame_valid)
    valid_time = time_valid_from_frames(frame_valid)
    student_grid, _summary = model.encoder(mel, visible)
    with torch.no_grad():
        teacher_grid, _teacher_summary = model.teacher_encoder(mel, visible)
    loss_pred, counts = predictive_loss(
        student_grid, teacher_grid, valid_time, model.horizons
    )
    loss_var = batch_variance_hinge(student_grid, visible)
    loss = loss_pred + float(VAR_WEIGHT) * loss_var
    loss_pit = loss.new_zeros(())
    lam = 0.0
    if mix_mel is not None:
        if model.sep_heads is None or mix_frame_valid is None:
            raise ValueError("mixture loss needs separation heads and a validity mask")
        lam = lambda_mix_at(step)
        mix_visible = visible_from_frames(mix_frame_valid)
        mix_time = time_valid_from_frames(mix_frame_valid)
        mix_grid, _mix_summary = model.encoder(mix_mel, mix_visible)
        mix_pool = frequency_pool(mix_grid)
        pred1 = l2_normalize(model.sep_heads[0](mix_pool))
        pred2 = l2_normalize(model.sep_heads[1](mix_pool))
        target = teacher_target(teacher_grid).detach()
        other = torch.roll(target, shifts=1, dims=0)
        other_valid = torch.roll(valid_time, shifts=1, dims=0)
        pit_valid = mix_time & valid_time & other_valid
        loss_pit = pit_cosine_loss(pred1, pred2, target, other, pit_valid)
        loss = loss + float(lam) * loss_pit
    out = {
        "loss": loss,
        "loss_pred": loss_pred,
        "loss_var": loss_var,
        "loss_pit": loss_pit,
        "lambda_mix": loss.new_tensor(float(lam)),
    }
    for key, value in counts.items():
        out[key] = loss.new_tensor(float(value))
    return out
