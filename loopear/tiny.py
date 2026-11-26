from typing import Any

import torch
import torch.nn.functional as F
import torchaudio
from torch import nn

from loopear.encoder import D_E, SR

N_MELS: int = 128
WIN: int = 400
HOP: int = 160
N_FFT: int = 512
WIDTH: int = 128
EXPAND: int = 2
KERNEL: int = 5
N_BLOCKS: int = 14
OUT_DIM: int = 128
PARAM_MIN: int = 900_000
PARAM_MAX: int = 1_100_000


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


class TinyEncoder(nn.Module):
    def __init__(
        self,
        n_mels: int = N_MELS,
        width: int = WIDTH,
        expand: int = EXPAND,
        kernel: int = KERNEL,
        n_blocks: int = N_BLOCKS,
        out_dim: int = OUT_DIM,
        teacher_dim: int = D_E,
        sr: int = SR,
    ):
        super().__init__()
        self.sr = int(sr)
        self.n_mels = int(n_mels)
        self.out_dim = int(out_dim)
        self.teacher_dim = int(teacher_dim)
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
        self.frame = nn.Linear(width, out_dim)
        self.teacher_proj = nn.Linear(out_dim, teacher_dim)

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

    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        mel = self.logmel(wav)
        x = self.blocks(self.stem(mel))
        x = self.out_norm(x)
        return self.frame(x.transpose(1, 2))

    def project(self, frames: torch.Tensor) -> torch.Tensor:
        return self.teacher_proj(frames)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        return self.encode(wav)


def inference_parameter_count(model: nn.Module) -> int:
    total = 0
    for name, param in model.named_parameters():
        if name.startswith("teacher_proj."):
            continue
        total += int(param.numel())
    return total


def mel_frame_count(n_samples: int) -> int:
    if n_samples < N_FFT:
        return 0
    return 1 + (int(n_samples) - N_FFT) // HOP


def stem_frame_count(n_mel: int) -> int:
    if n_mel <= 0:
        return 0
    return (int(n_mel) + 4 - KERNEL) // 2 + 1


def wav_to_frame_lengths(lengths: torch.Tensor) -> torch.Tensor:
    out = []
    for n in lengths.detach().cpu().tolist():
        out.append(stem_frame_count(mel_frame_count(int(n))))
    return torch.tensor(out, device=lengths.device, dtype=torch.long)


def _valid_mask(lengths: torch.Tensor, steps: int) -> torch.Tensor:
    positions = torch.arange(steps, device=lengths.device)
    capped = lengths.clamp(min=0, max=steps)
    return positions.unsqueeze(0) < capped.unsqueeze(1)


def frame_distill_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    steps = min(int(pred.shape[1]), int(target.shape[1]))
    pred = pred[:, :steps]
    target = target[:, :steps].detach()
    mask = _valid_mask(lengths, steps)
    denom = mask.sum().clamp_min(1).to(pred.dtype)
    l1 = (pred - target).abs().mean(dim=-1)
    l1 = (l1 * mask).sum() / denom
    pred_n = F.normalize(pred, dim=-1, eps=1e-6)
    tgt_n = F.normalize(target, dim=-1, eps=1e-6)
    cosine = (pred_n * tgt_n).sum(dim=-1).clamp(-1.0 + 1e-4, 1.0 - 1e-4)
    cos_term = -torch.log(torch.sigmoid(cosine))
    cos_term = (cos_term * mask).sum() / denom
    return l1 + cos_term


def hard_residual(mix: torch.Tensor, counterpart: torch.Tensor) -> torch.Tensor:
    denom = counterpart.square().sum(dim=-1, keepdim=True).clamp_min(1e-6)
    coef = (mix * counterpart).sum(dim=-1, keepdim=True) / denom
    return mix - coef * counterpart


def distill_pair_loss(
    model: TinyEncoder,
    frames_a: torch.Tensor,
    frames_b: torch.Tensor,
    teacher_a: torch.Tensor,
    teacher_b: torch.Tensor,
    lengths_a: torch.Tensor,
    lengths_b: torch.Tensor,
) -> torch.Tensor:
    loss_a = frame_distill_loss(model.project(frames_a), teacher_a, lengths_a)
    loss_b = frame_distill_loss(model.project(frames_b), teacher_b, lengths_b)
    return 0.5 * (loss_a + loss_b)


def mixture_augment_loss(
    model: TinyEncoder,
    frames_mix: torch.Tensor,
    teacher_mix: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    return frame_distill_loss(model.project(frames_mix), teacher_mix, lengths)


def averaged_constituent_loss(
    model: TinyEncoder,
    frames_mix: torch.Tensor,
    teacher_a: torch.Tensor,
    teacher_b: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    steps = min(int(teacher_a.shape[1]), int(teacher_b.shape[1]))
    target = 0.5 * (teacher_a[:, :steps] + teacher_b[:, :steps])
    return frame_distill_loss(model.project(frames_mix), target, lengths)


def constituent_align_loss(
    model: TinyEncoder,
    frames_mix: torch.Tensor,
    frames_a: torch.Tensor,
    frames_b: torch.Tensor,
    teacher_a: torch.Tensor,
    teacher_b: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    projected_mix = model.project(frames_mix)
    projected_a = model.project(frames_a)
    projected_b = model.project(frames_b)
    toward_a = hard_residual(projected_mix, projected_b)
    toward_b = hard_residual(projected_mix, projected_a)
    loss_a = frame_distill_loss(toward_a, teacher_a, lengths)
    loss_b = frame_distill_loss(toward_b, teacher_b, lengths)
    return 0.5 * (loss_a + loss_b)


def mix_waveforms(
    wav_a: torch.Tensor,
    wav_b: torch.Tensor,
    gain_db: torch.Tensor,
) -> torch.Tensor:
    gain = torch.pow(10.0, gain_db.to(wav_a.dtype) / 20.0).view(-1, 1)
    mixed = wav_a + gain * wav_b
    peak = mixed.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = torch.where(peak > 0.99, 0.99 / peak, torch.ones_like(peak))
    return mixed * scale


def latest_sample_for_frame(frame_index: int) -> int:
    mel_index = int(frame_index) * 2
    return mel_index * HOP + N_FFT


def causal_mismatch(
    model: TinyEncoder,
    wav: torch.Tensor,
    atol: float = 1e-5,
) -> dict[str, Any]:
    model_was_training = model.training
    model.eval()
    with torch.no_grad():
        base = model.encode(wav)
        n_frames = int(base.shape[1])
        worst = 0.0
        checked = 0
        for frame_index in range(0, n_frames, max(1, n_frames // 8)):
            cut = latest_sample_for_frame(frame_index)
            if cut >= int(wav.shape[-1]):
                continue
            altered = wav.clone()
            altered[:, cut:] = torch.randn_like(altered[:, cut:])
            other = model.encode(altered)
            delta = (base[:, frame_index] - other[:, frame_index]).abs().max().item()
            worst = max(worst, float(delta))
            checked += 1
    if model_was_training:
        model.train()
    return {"checked_frames": checked, "max_abs_diff": worst, "ok": worst <= atol and checked > 0}


def teacher_pair_report(
    teacher_a: torch.Tensor,
    teacher_b: torch.Tensor,
    silence: torch.Tensor,
    tone: torch.Tensor,
    student_steps: int,
    teacher_steps: int,
) -> dict[str, Any]:
    def _cos(x: torch.Tensor, y: torch.Tensor) -> float:
        steps = min(int(x.shape[1]), int(y.shape[1]))
        xn = F.normalize(x[:, :steps].reshape(-1, x.shape[-1]), dim=-1)
        yn = F.normalize(y[:, :steps].reshape(-1, y.shape[-1]), dim=-1)
        return float((xn * yn).sum(dim=-1).mean().item())

    finite = bool(
        torch.isfinite(teacher_a).all()
        and torch.isfinite(teacher_b).all()
        and torch.isfinite(silence).all()
        and torch.isfinite(tone).all()
    )
    sil_tone = _cos(silence, tone)
    pair_cos = abs(_cos(teacher_a, teacher_b))
    return {
        "finite": finite,
        "silence_tone_cosine": sil_tone,
        "constituent_abs_cosine": pair_cos,
        "student_frames": int(student_steps),
        "teacher_frames": int(teacher_steps),
        "frame_gap": abs(int(student_steps) - int(teacher_steps)),
        "ok": bool(
            finite
            and sil_tone < 0.98
            and pair_cos < 0.98
            and abs(int(student_steps) - int(teacher_steps)) <= 2
        ),
    }


def receptive_field_seconds() -> float:
    stem_s = (KERNEL - 1) * (HOP / float(SR))
    block_s = N_BLOCKS * (KERNEL - 1) * 0.02
    return stem_s + block_s + (WIN / float(SR))


def assert_param_gate(model: TinyEncoder) -> dict[str, Any]:
    count = inference_parameter_count(model)
    proj = int(sum(p.numel() for p in model.teacher_proj.parameters()))
    return {
        "inference_params": count,
        "teacher_proj_params": proj,
        "receptive_field_s": round(receptive_field_seconds(), 4),
        "ok": PARAM_MIN <= count <= PARAM_MAX,
    }

