from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import torch
import torch.nn.functional as F
import torchaudio
from torch import nn

from loopear.encoder import SR

EFFICIENTAT_SR: int = 32000
WIDTHS: dict[str, float] = {
    "mn01": 0.1,
    "mn04": 0.4,
    "mn05": 0.5,
}
WEIGHTS: dict[str, str] = {
    "mn01": "https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/mn01_as_mAP_298.pt",
    "mn04": "https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/mn04_as_mAP_432.pt",
    "mn05": "https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/mn05_as_mAP_443.pt",
}


def _divisible(value: float, divisor: int = 8) -> int:
    new_v = max(divisor, int(value + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * value:
        new_v += divisor
    return int(new_v)


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int,
        stride: int = 1,
        groups: int = 1,
        dilation: int = 1,
        activation: type[nn.Module] | None = nn.Hardswish,
    ):
        pad = (kernel - 1) // 2 * dilation
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel,
                stride=stride,
                padding=pad,
                dilation=dilation,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch, eps=0.001, momentum=0.01),
        ]
        if activation is not None:
            layers.append(activation(inplace=True))
        super().__init__(*layers)


class ChannelSE(nn.Module):
    def __init__(self, channels: int, se_r: int = 4):
        super().__init__()
        hidden = _divisible(channels / se_r)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)
        self.act = nn.ReLU()
        self.scale_act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = x.mean(dim=(2, 3))
        scale = self.scale_act(self.fc2(self.act(self.fc1(scale))))
        return x * scale[:, :, None, None]


class ConcurrentSE(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conc_se_layers = nn.ModuleList([ChannelSE(channels)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [layer(x) for layer in self.conc_se_layers]
        return torch.stack(outs, dim=0).max(dim=0).values


class InvertedResidual(nn.Module):
    def __init__(
        self,
        in_ch: int,
        exp_ch: int,
        out_ch: int,
        kernel: int,
        stride: int,
        use_se: bool,
        use_hs: bool,
    ):
        super().__init__()
        act = nn.Hardswish if use_hs else nn.ReLU
        layers: list[nn.Module] = []
        if exp_ch != in_ch:
            layers.append(ConvNormAct(in_ch, exp_ch, 1, activation=act))
        layers.append(
            ConvNormAct(exp_ch, exp_ch, kernel, stride=stride, groups=exp_ch, activation=act)
        )
        if use_se:
            layers.append(ConcurrentSE(exp_ch))
        layers.append(ConvNormAct(exp_ch, out_ch, 1, activation=None))
        self.block = nn.Sequential(*layers)
        self.use_res = stride == 1 and in_ch == out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        if self.use_res:
            y = y + x
        return y


def _configs(width: float) -> list[tuple[int, int, int, int, int, bool, bool]]:
    raw = [
        (16, 3, 16, 16, 1, False, False),
        (16, 3, 64, 24, 2, False, False),
        (24, 3, 72, 24, 1, False, False),
        (24, 5, 72, 40, 2, True, False),
        (40, 5, 120, 40, 1, True, False),
        (40, 5, 120, 40, 1, True, False),
        (40, 3, 240, 80, 2, False, True),
        (80, 3, 200, 80, 1, False, True),
        (80, 3, 184, 80, 1, False, True),
        (80, 3, 184, 80, 1, False, True),
        (80, 3, 480, 112, 1, True, True),
        (112, 3, 672, 112, 1, True, True),
        (112, 5, 672, 160, 2, True, True),
        (160, 5, 960, 160, 1, True, True),
        (160, 5, 960, 160, 1, True, True),
    ]
    out = []
    for cin, k, exp, cout, stride, se, hs in raw:
        out.append(
            (
                _divisible(cin * width),
                k,
                _divisible(exp * width),
                _divisible(cout * width),
                stride,
                se,
                hs,
            )
        )
    return out


class EfficientATEncoder(nn.Module):
    def __init__(self, width: float, n_classes: int = 527):
        super().__init__()
        cfgs = _configs(width)
        first_out = cfgs[0][0]
        layers: list[nn.Module] = [
            ConvNormAct(1, first_out, 3, stride=2, activation=nn.Hardswish)
        ]
        for cin, k, exp, cout, stride, se, hs in cfgs:
            layers.append(InvertedResidual(cin, exp, cout, k, stride, se, hs))
        last_in = cfgs[-1][3]
        last_out = 6 * last_in
        layers.append(ConvNormAct(last_in, last_out, 1, activation=nn.Hardswish))
        self.features = nn.Sequential(*layers)
        last_channel = _divisible(1280 * width)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(last_out, last_channel),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(last_channel, n_classes),
        )
        self.register_buffer("window", torch.hann_window(800, periodic=False), persistent=False)
        self.register_buffer(
            "preemph", torch.tensor([[[-0.97, 1.0]]], dtype=torch.float32), persistent=False
        )
        basis, _ = torchaudio.compliance.kaldi.get_mel_banks(
            128, 1024, EFFICIENTAT_SR, 0.0, 15000.0, 100.0, -500.0, 1.0
        )
        basis = F.pad(basis, (0, 1))
        self.register_buffer("mel_basis", basis.to(dtype=torch.float32), persistent=False)

    def logmel(self, wav_32k: torch.Tensor) -> torch.Tensor:
        x = F.conv1d(wav_32k.unsqueeze(1), self.preemph).squeeze(1)
        spec = torch.stft(
            x,
            n_fft=1024,
            hop_length=320,
            win_length=800,
            center=True,
            window=self.window,
            return_complex=True,
        )
        power = spec.abs().square()
        mel = torch.matmul(self.mel_basis, power)
        return (mel + 1e-5).log().add(4.5).div(5.0)

    def frame_map(self, mel: torch.Tensor) -> torch.Tensor:
        x = mel.unsqueeze(1)
        for layer in self.features:
            x = layer(x)
        return x.mean(dim=2)

    def forward(self, wav_16k: torch.Tensor) -> torch.Tensor:
        wav_32k = torchaudio.functional.resample(wav_16k, SR, EFFICIENTAT_SR)
        frames = self.frame_map(self.logmel(wav_32k))
        return frames.transpose(1, 2)


def download_weights(name: str, root: Path | str) -> Path:
    dest_root = Path(root)
    dest_root.mkdir(parents=True, exist_ok=True)
    dest = dest_root / f"{name}.pt"
    if not dest.exists():
        urlretrieve(WEIGHTS[name], dest)
    return dest


def load_efficientat(name: str, root: Path | str, device: str = "cpu") -> EfficientATEncoder:
    model = EfficientATEncoder(WIDTHS[name])
    state = torch.load(download_weights(name, root), map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model.to(device)


def native_hop_s() -> float:
    return 320.0 / float(EFFICIENTAT_SR) * 32.0


def parameter_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


class SsondoStudent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = EfficientATEncoder(1.0)
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=EFFICIENTAT_SR,
            n_fft=1024,
            win_length=1024,
            hop_length=512,
            f_min=50.0,
            f_max=16000.0,
            n_mels=128,
            power=2.0,
            center=False,
            norm="slaney",
            mel_scale="slaney",
        )

    def forward(self, wav_16k: torch.Tensor) -> torch.Tensor:
        wav_32k = torchaudio.functional.resample(wav_16k, SR, EFFICIENTAT_SR)
        mel = (self.mel(wav_32k) + 0.001).log()
        frames = self.backbone.frame_map(mel)
        return frames.transpose(1, 2)


def load_ssondo_student(device: str = "cpu") -> SsondoStudent:
    from huggingface_hub import hf_hub_download

    model = SsondoStudent()
    ckpt_path = hf_hub_download("mohammedali2501/ssondo", "matpac_mobilenetv3.ckpt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    prefix = "student_model.model."
    feat = {
        key[len(prefix) :]: value
        for key, value in ckpt["state_dict"].items()
        if key.startswith(prefix + "features.")
    }
    missing, unexpected = model.backbone.load_state_dict(feat, strict=False)
    bad_missing = [key for key in missing if not key.startswith("classifier.")]
    if bad_missing or unexpected:
        raise RuntimeError(f"S-SONDO load mismatch missing={bad_missing[:4]} unexpected={unexpected[:4]}")
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model.to(device)


def ssondo_inference_params(model: SsondoStudent) -> int:
    return int(
        sum(
            param.numel()
            for name, param in model.backbone.named_parameters()
            if name.startswith("features.")
        )
    )


def profile_summary(name: str, seconds: float, median_s: float, params: int) -> dict[str, Any]:
    return {
        "name": name,
        "params": params,
        "native_hop_s": native_hop_s(),
        "median_s": median_s,
        "rtf": median_s / seconds,
        "audio_s": seconds,
    }

