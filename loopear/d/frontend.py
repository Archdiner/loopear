import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio

from loopear.d.spec import (
    CROP_SAMPLES,
    HOP_SAMPLES,
    LOGMEL_EPS,
    N_FFT,
    N_FRAMES,
    N_MELS,
    SR,
    WIN_SAMPLES,
)


def build_mel() -> torchaudio.transforms.MelSpectrogram:
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=SR,
        n_fft=N_FFT,
        win_length=WIN_SAMPLES,
        hop_length=HOP_SAMPLES,
        f_min=0.0,
        f_max=float(SR // 2),
        n_mels=N_MELS,
        power=2.0,
        center=False,
        mel_scale="htk",
        norm=None,
    )


def waveform_logmel(wav: torch.Tensor, mel: torchaudio.transforms.MelSpectrogram) -> torch.Tensor:
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    spec = mel(wav)
    return torch.log(spec.clamp_min(LOGMEL_EPS)).transpose(1, 2).squeeze(0)


def frame_count(n_samples: int) -> int:
    if n_samples < WIN_SAMPLES:
        return 1 if n_samples > 0 else 0
    return 1 + (int(n_samples) - N_FFT) // HOP_SAMPLES


def _crop_start(n_frames: int, generator: torch.Generator | None) -> int:
    if n_frames >= N_FRAMES:
        span = n_frames - N_FRAMES + 1
        if generator is None:
            return max((n_frames - N_FRAMES) // 2, 0)
        return int(torch.randint(0, span, (1,), generator=generator).item())
    return 0


def _slice_crop(logmel: torch.Tensor, start: int) -> tuple[torch.Tensor, torch.Tensor]:
    n = int(logmel.shape[0])
    if n >= N_FRAMES:
        if start < 0 or start + N_FRAMES > n:
            raise ValueError("crop start is outside the recording")
        crop = logmel[start : start + N_FRAMES]
        valid = torch.ones(N_FRAMES, dtype=torch.bool)
        return crop, valid
    if start != 0:
        raise ValueError("a short recording has only one offset")
    crop = torch.zeros(N_FRAMES, N_MELS, dtype=logmel.dtype)
    if n > 0:
        crop[:n] = logmel
    valid = torch.zeros(N_FRAMES, dtype=torch.bool)
    valid[:n] = True
    return crop, valid


def crop_logmel(
    logmel: torch.Tensor,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if logmel.ndim != 2 or logmel.shape[1] != N_MELS:
        raise ValueError(f"logmel must be [frames, {N_MELS}]")
    start = _crop_start(int(logmel.shape[0]), generator)
    return _slice_crop(logmel, start)


def crop_logmel_with_start(
    logmel: torch.Tensor,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if logmel.ndim != 2 or logmel.shape[1] != N_MELS:
        raise ValueError(f"logmel must be [frames, {N_MELS}]")
    start = _crop_start(int(logmel.shape[0]), generator)
    crop, valid = _slice_crop(logmel, start)
    return crop, valid, int(start)


def normalize_logmel(logmel: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (logmel - mean.view(1, -1)) / std.view(1, -1).clamp_min(1e-6)


class RunningNorm:
    def __init__(self) -> None:
        self.count = 0
        self.sum = np.zeros(N_MELS, dtype=np.float64)
        self.sumsq = np.zeros(N_MELS, dtype=np.float64)

    def update(self, frames: np.ndarray) -> None:
        if frames.size == 0:
            return
        x = frames.astype(np.float64, copy=False)
        self.count += int(x.shape[0])
        self.sum += x.sum(axis=0)
        self.sumsq += np.square(x).sum(axis=0)

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        if self.count < 2:
            raise RuntimeError("not enough frames to fit log-mel norm")
        mean = self.sum / float(self.count)
        var = self.sumsq / float(self.count) - np.square(mean)
        std = np.sqrt(np.maximum(var, 1e-8))
        return mean.astype(np.float32), std.astype(np.float32)


def frontend_config(mean: list[float], std: list[float]) -> dict[str, Any]:
    return {
        "sr": SR,
        "n_mels": N_MELS,
        "n_fft": N_FFT,
        "win_samples": WIN_SAMPLES,
        "hop_samples": HOP_SAMPLES,
        "center": False,
        "mel_scale": "htk",
        "norm": None,
        "power": 2.0,
        "log": "natural",
        "log_floor": LOGMEL_EPS,
        "n_frames": N_FRAMES,
        "crop_samples": CROP_SAMPLES,
        "mean": mean,
        "std": std,
    }


def canonical_sha256(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def write_frontend_config(path: Path, mean: np.ndarray, std: np.ndarray) -> dict[str, Any]:
    payload = frontend_config([float(v) for v in mean.tolist()], [float(v) for v in std.tolist()])
    payload["config_sha256"] = canonical_sha256(
        {k: v for k, v in payload.items() if k != "config_sha256"}
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def load_frontend_config(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    body = {k: v for k, v in payload.items() if k != "config_sha256"}
    digest = canonical_sha256(body)
    if payload.get("config_sha256") != digest:
        raise AssertionError("frontend config hash does not match its body")
    return payload
