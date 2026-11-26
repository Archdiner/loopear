import argparse
import contextlib
import hashlib
import warnings
import io
import json
import os
import subprocess
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio

from loopear.data import STARSS23_CLASSES, rasterize_starss_frames_50hz
from loopear.d.diagnostics import cpu_seconds, hours, peak_rss_gb
from loopear.d.frontend import build_mel, frame_count, waveform_logmel
from loopear.d.fsd import audio_path, items_for, load_manifest
from loopear.d.probe import fit_linear_probe, fixed_project
from loopear.d.spec import CROP_SAMPLES, CROP_SECONDS, D_MODEL, GRID_T, HOP_SAMPLES, N_FRAMES, SR
from loopear.encoder import load_mono_audio, load_usad2_extractor
from loopear.harness import write_result_json

EFFICIENTAT_COMMIT = "a425fdce92572e602a1d5634799bd9f1f2efa806"
EFFICIENTAT_URL = "https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/mn04_as_mAP_432.pt"
EFFICIENTAT_SHA256 = "899a8c6217063f6941793102a2780b3dd788ab11fce468d9cf5f0577c453321c"
USAD_SNAPSHOT = "513277f85c2a5dcb07d0cb3ce0981e1d622c97d8"
PROBE_STEPS = 250
PROBE_LR = 1e-2
FRAME_STEPS = 50


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _ensure_efficientat() -> tuple[Path, Path]:
    root = Path.home() / ".cache" / "efficientat"
    code = root / "EfficientAT"
    ckpt = root / "mn04_as_mAP_432.pt"
    if not (code / "models" / "mn" / "model.py").exists():
        root.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "https://github.com/fschmid56/EfficientAT.git", str(code)],
            check=True,
        )
        subprocess.run(["git", "checkout", EFFICIENTAT_COMMIT], cwd=code, check=True)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=code, text=True).strip()
    if head != EFFICIENTAT_COMMIT:
        raise SystemExit(f"EfficientAT checkout {head} is not {EFFICIENTAT_COMMIT}")
    if not ckpt.exists():
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(EFFICIENTAT_URL, ckpt)
    digest = _sha256(ckpt)
    if digest != EFFICIENTAT_SHA256:
        raise SystemExit(f"EfficientAT checkpoint hash {digest}")
    return code, ckpt


def _load_efficientat(code: Path, ckpt: Path) -> tuple[torch.nn.Module, torch.nn.Module, int]:
    cwd = os.getcwd()
    sys.path.insert(0, str(code))
    os.chdir(code)
    try:
        from helpers.utils import NAME_TO_WIDTH
        from models.mn.model import get_model
        from models.preprocess import AugmentMelSTFT
    finally:
        os.chdir(cwd)
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        model = get_model(
            width_mult=NAME_TO_WIDTH("mn04_as"),
            pretrained_name=None,
            num_classes=527,
            head_type="mlp",
        )
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    with contextlib.redirect_stdout(sink):
        mel = AugmentMelSTFT(n_mels=128, sr=32000, win_length=800, hopsize=320)
    mel.eval()
    params = int(sum(p.numel() for p in model.parameters()))
    return model, mel, params


def _read_fsd(path: Path) -> np.ndarray:
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if int(sr) != SR:
        raise AssertionError(f"sr {sr} at {path}")
    if wav.ndim != 1:
        wav = np.mean(wav, axis=-1)
    return np.ascontiguousarray(wav, dtype=np.float32)


def _spans(n_samples: int, max_windows: int) -> list[tuple[int, int]]:
    n_frames = frame_count(n_samples)
    if n_frames <= N_FRAMES:
        starts = [0]
    else:
        starts = list(range(0, n_frames - N_FRAMES + 1, N_FRAMES))[: max(int(max_windows), 1)]
        if not starts:
            starts = [0]
    spans = []
    for start in starts:
        left = start * HOP_SAMPLES
        right = min(n_samples, left + CROP_SAMPLES)
        if right > left:
            spans.append((left, right))
    return spans


def _windows(wav: np.ndarray, max_windows: int) -> list[torch.Tensor]:
    pieces = []
    for left, right in _spans(int(wav.shape[0]), max_windows):
        pieces.append(torch.from_numpy(np.ascontiguousarray(wav[left:right])))
    return pieces


def _to_steps(seq: torch.Tensor, steps: int) -> torch.Tensor:
    if seq.shape[0] == steps:
        return seq
    if seq.shape[0] < 2:
        return seq[:1].expand(steps, seq.shape[1]).contiguous()
    column = seq.transpose(0, 1).unsqueeze(0)
    aligned = torch.nn.functional.interpolate(column, size=steps, mode="linear", align_corners=False)
    return aligned.squeeze(0).transpose(0, 1).contiguous()


def _usad_forward(model: torch.nn.Module, waves: list[torch.Tensor], with_frames: bool) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    lengths = torch.tensor([int(wave.shape[0]) for wave in waves], dtype=torch.long)
    width = int(lengths.max())
    batch = torch.zeros(len(waves), width, dtype=torch.float32)
    for index, wave in enumerate(waves):
        batch[index, : wave.shape[0]] = wave
    with torch.inference_mode():
        out = model(wavs=batch, wav_lengths=lengths)
    hidden = out["x"].float().cpu()
    valid = [min(max(int(n), 0), int(hidden.shape[1])) for n in out["x_lengths"].tolist()]
    means = []
    frames = []
    for index, count in enumerate(valid):
        if count <= 0:
            means.append(torch.zeros(hidden.shape[-1]))
            frames.append(torch.zeros(FRAME_STEPS, hidden.shape[-1]))
            continue
        piece = hidden[index, :count]
        means.append(piece.mean(dim=0))
        if with_frames:
            frames.append(_to_steps(piece, FRAME_STEPS))
    return means, frames


def _eat_forward(
    model: torch.nn.Module,
    mel: torch.nn.Module,
    waves: list[torch.Tensor],
    with_frames: bool,
) -> tuple[list[torch.Tensor], list[torch.Tensor], dict[str, int]]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for index, wave in enumerate(waves):
        grouped[int(wave.shape[0])].append(index)
    means: list[torch.Tensor | None] = [None] * len(waves)
    frames: list[torch.Tensor | None] = [None] * len(waves)
    observed = {"mel_frames": 0, "fmap_time": 0, "fmap_freq": 0, "full_crop_windows": 0}
    for _length, indexes in grouped.items():
        chosen = [waves[index] for index in indexes]
        stacked = torch.stack(chosen, dim=0)
        with torch.inference_mode():
            waveform = torchaudio.functional.resample(stacked, SR, 32000)
            spec = mel(waveform).unsqueeze(1)
            _logits, feats = model(spec)
            fmap = None
            if with_frames:
                _logits, fmaps = model._forward_impl(spec, return_fmaps=True)
                fmap = fmaps[-1].float().cpu()
        feats = feats.float().cpu()
        if feats.ndim == 1:
            feats = feats.unsqueeze(0)
        if int(chosen[0].shape[0]) == CROP_SAMPLES:
            observed["full_crop_windows"] += len(indexes)
            observed["mel_frames"] = int(spec.shape[-1])
            if fmap is not None:
                observed["fmap_time"] = int(fmap.shape[-1])
                observed["fmap_freq"] = int(fmap.shape[-2])
        for local, index in enumerate(indexes):
            means[index] = feats[local]
            if fmap is not None:
                sequence = fmap[local].mean(dim=1).transpose(0, 1).contiguous()
                frames[index] = _to_steps(sequence, FRAME_STEPS)
    if any(item is None for item in means) or (with_frames and any(item is None for item in frames)):
        raise RuntimeError("EfficientAT missed a window")
    return means, frames, observed


def _embed_rows(
    rows: list[dict[str, Any]],
    audio_root: Path,
    usad: torch.nn.Module,
    eat: torch.nn.Module,
    mel: torch.nn.Module,
    max_windows: int,
    batch: int,
    check_mel: torch.nn.Module | None,
) -> dict[str, Any]:
    usad_rows = []
    eat_rows = []
    n_windows = 0
    n_short = 0
    native = {"mel_frames": 0, "fmap_time": 0, "fmap_freq": 0, "usad_frames": 0}
    pending: list[torch.Tensor] = []
    owners: list[int] = []
    bucket_u: dict[int, list[torch.Tensor]] = defaultdict(list)
    bucket_e: dict[int, list[torch.Tensor]] = defaultdict(list)

    def _flush() -> None:
        if not pending:
            return
        u_means, _u_frames = _usad_forward(usad, pending, with_frames=False)
        e_means, _e_frames, observed = _eat_forward(eat, mel, pending, with_frames=False)
        if observed["mel_frames"]:
            native["mel_frames"] = observed["mel_frames"]
        for local, owner in enumerate(owners):
            bucket_u[owner].append(u_means[local])
            bucket_e[owner].append(e_means[local])
        pending.clear()
        owners.clear()

    for row_index, row in enumerate(rows):
        wav = _read_fsd(audio_path(audio_root, row))
        if check_mel is not None and row_index == 0:
            spec = waveform_logmel(torch.from_numpy(wav), check_mel)
            if int(spec.shape[0]) != frame_count(int(wav.shape[0])):
                raise AssertionError("window grid does not match the LoopEar mel")
        pieces = _windows(wav, max_windows)
        if not pieces:
            pieces = [torch.zeros(CROP_SAMPLES)]
        for piece in pieces:
            n_windows += 1
            if int(piece.shape[0]) < CROP_SAMPLES:
                n_short += 1
            pending.append(piece)
            owners.append(row_index)
            if len(pending) >= batch:
                _flush()
        if row_index % 64 == 63:
            print(f"clips {row_index + 1}/{len(rows)}", flush=True)
    _flush()
    if native["fmap_time"] == 0 or native["mel_frames"] == 0:
        _means, _frames, observed = _eat_forward(eat, mel, [torch.zeros(CROP_SAMPLES)], with_frames=True)
        native["mel_frames"] = observed["mel_frames"]
        native["fmap_time"] = observed["fmap_time"]
        native["fmap_freq"] = observed["fmap_freq"]
        del _means, _frames
    with torch.inference_mode():
        probe = usad(wavs=torch.zeros(1, CROP_SAMPLES), wav_lengths=torch.tensor([CROP_SAMPLES]))
    native["usad_frames"] = int(probe["x_lengths"][0])
    for row_index in range(len(rows)):
        usad_rows.append(torch.stack(bucket_u[row_index]).mean(dim=0))
        eat_rows.append(torch.stack(bucket_e[row_index]).mean(dim=0))
    return {
        "usad": torch.stack(usad_rows),
        "efficientat": torch.stack(eat_rows),
        "n_windows": n_windows,
        "n_short_windows": n_short,
        "native": native,
    }


def _classes(train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]]) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    counts: Counter[str] = Counter()
    for row in train_rows + val_rows:
        counts.update(row["labels"])
    names = sorted(label for label, count in counts.items() if count >= 3)
    index = {name: pos for pos, name in enumerate(names)}
    y_train = torch.zeros(len(train_rows), len(names))
    y_val = torch.zeros(len(val_rows), len(names))
    for row_index, row in enumerate(train_rows):
        for label in row["labels"]:
            if label in index:
                y_train[row_index, index[label]] = 1.0
    for row_index, row in enumerate(val_rows):
        for label in row["labels"]:
            if label in index:
                y_val[row_index, index[label]] = 1.0
    keep = (y_train.sum(dim=0) >= 2) & (y_val.sum(dim=0) >= 2)
    kept = [name for name, flag in zip(names, keep.tolist()) if flag]
    return kept, y_train[:, keep], y_val[:, keep]


def _probe(x_train: torch.Tensor, x_val: torch.Tensor, y_train: torch.Tensor, y_val: torch.Tensor, seed: int) -> dict[str, Any]:
    projected_train = fixed_project(x_train, D_MODEL, seed)
    projected_val = fixed_project(x_val, D_MODEL, seed)
    scored = fit_linear_probe(projected_train, y_train, projected_val, y_val, PROBE_STEPS, PROBE_LR, seed)
    perm = torch.Generator().manual_seed(seed + 7)
    shuffled = y_train[torch.randperm(y_train.shape[0], generator=perm)]
    control = fit_linear_probe(projected_train, shuffled, projected_val, y_val, PROBE_STEPS, PROBE_LR, seed)
    expected = D_MODEL * int(y_train.shape[1]) + int(y_train.shape[1])
    if int(scored["probe_params"]) != expected:
        raise AssertionError(f"probe params {scored['probe_params']} != {expected}")
    return {
        "input_dim": int(x_train.shape[1]),
        "projected_dim": D_MODEL,
        "probe": scored,
        "shuffled_label_probe": control,
    }


def _few_shot_status(train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]]) -> dict[str, Any]:
    train_primary = {row["labels"][0] if row["labels"] else "" for row in train_rows}
    by_class: dict[str, int] = defaultdict(int)
    for row in val_rows:
        label = row["labels"][0] if row["labels"] else ""
        if label and label not in train_primary:
            by_class[label] += 1
    usable = {name: count for name, count in by_class.items() if count >= 6}
    if not usable:
        return {
            "status": "null",
            "reason": "no primary val label is absent from the 5h subset with at least 6 clips",
            "n_unfamiliar_primaries": len(by_class),
            "scored": False,
        }
    return {
        "status": "measurable",
        "n_unfamiliar_primaries": len(by_class),
        "n_with_six_clips": len(usable),
        "scored": False,
    }


def _parse_starss_csv(text: str) -> list[tuple[int, int]]:
    parsed = []
    for line in text.splitlines():
        cols = [part.strip() for part in line.split(",")]
        if len(cols) >= 2 and cols[0].isdigit():
            parsed.append((int(cols[0]), int(cols[1])))
    return parsed


def _onset_lift(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    inactive = scores[labels.sum(axis=1) == 0]
    background = float(inactive.mean()) if inactive.size else float(scores.mean())
    peaks = []
    above = 0
    for t in range(labels.shape[0]):
        for cls in range(labels.shape[1]):
            if labels[t, cls] < 0.5:
                continue
            if t > 0 and labels[t - 1, cls] >= 0.5:
                continue
            lo = max(0, t - 2)
            hi = min(labels.shape[0], t + 3)
            peak = float(scores[lo:hi, cls].max())
            peaks.append(peak - background)
            if peak > background:
                above += 1
    if not peaks:
        return {"n_onsets": 0, "mean_peak_minus_background": None, "fraction_above_background": None}
    return {
        "n_onsets": len(peaks),
        "mean_peak_minus_background": float(np.mean(peaks)),
        "fraction_above_background": float(above / len(peaks)),
        "background_mean": background,
        "tolerance_frames": 2,
        "tolerance_s": 0.08,
    }


def _val_scores(x_train: torch.Tensor, y_train: torch.Tensor, x_val: torch.Tensor, seed: int) -> np.ndarray:
    torch.manual_seed(seed)
    head = torch.nn.Linear(D_MODEL, int(y_train.shape[1]))
    opt = torch.optim.AdamW(head.parameters(), lr=PROBE_LR, weight_decay=0.0)
    for _ in range(PROBE_STEPS):
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(head(x_train), y_train)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return torch.sigmoid(head(x_val)).numpy()


def _starss_windows(wav: np.ndarray) -> list[tuple[int, torch.Tensor]]:
    n_frames = frame_count(int(wav.shape[0]))
    starts = list(range(0, max(n_frames - N_FRAMES, 0) + 1, N_FRAMES))[:6]
    windows = []
    for start in starts:
        left = start * HOP_SAMPLES
        right = min(int(wav.shape[0]), left + CROP_SAMPLES)
        if right <= left:
            continue
        windows.append((start, torch.from_numpy(np.ascontiguousarray(wav[left:right]))))
    return windows


def _starss(
    usad: torch.nn.Module,
    eat: torch.nn.Module,
    mel: torch.nn.Module,
    seed: int,
) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download

    manifest = json.loads(Path("data/manifests/starss23.json").read_text(encoding="utf-8"))
    chosen = [row for row in manifest["items"] if row["split"] == "val"][:4]
    if len(chosen) < 4:
        return {"status": "null", "reason": "need 4 val STARSS23 rows in the existing manifest"}
    clips = []
    fmap_time = 0
    for row in chosen:
        wav_src = hf_hub_download("labhamlet/STARSS23", f"original_audios/{row['id']}.wav", repo_type="dataset")
        csv_src = None
        last_error: Exception | None = None
        for prefix in ("labels/test/", "labels/valid/", "labels/"):
            try:
                csv_src = hf_hub_download("labhamlet/STARSS23", f"{prefix}{row['id']}.csv", repo_type="dataset")
                break
            except Exception as exc:
                last_error = exc
        if csv_src is None:
            raise last_error if last_error is not None else FileNotFoundError(row["id"])
        wav = load_mono_audio(wav_src, target_sr=SR, channel=0)
        events = _parse_starss_csv(Path(csv_src).read_text(encoding="utf-8"))
        windows = _starss_windows(wav)
        waves = [piece for _start, piece in windows]
        u_means, u_frames = _usad_forward(usad, waves, with_frames=True)
        e_means, e_frames, observed = _eat_forward(eat, mel, waves, with_frames=True)
        if observed["fmap_time"]:
            fmap_time = observed["fmap_time"]
        packed = []
        for index, (start, piece) in enumerate(windows):
            labels = rasterize_starss_frames_50hz(
                events,
                float(start) * 0.01,
                GRID_T,
                len(STARSS23_CLASSES),
                hop_e=0.04,
                label_hop_s=0.1,
            )
            packed.append(
                {
                    "usad": u_frames[index],
                    "efficientat": e_frames[index],
                    "usad_clip": u_means[index],
                    "efficientat_clip": e_means[index],
                    "y": labels,
                    "samples": int(piece.shape[0]),
                }
            )
        clips.append({"id": row["id"], "group": row["group_id"], "windows": packed})
    with torch.inference_mode():
        full = usad(wavs=torch.zeros(1, CROP_SAMPLES), wav_lengths=torch.tensor([CROP_SAMPLES]))
    native_usad = int(full["x_lengths"][0])
    train_clips = clips[:2]
    val_clips = clips[2:]

    def _stack(group: list[dict[str, Any]], key: str) -> tuple[torch.Tensor, np.ndarray]:
        xs = [win[key] for clip in group for win in clip["windows"]]
        ys = [win["y"] for clip in group for win in clip["windows"]]
        return torch.cat(xs, dim=0), np.concatenate(ys, axis=0)

    y_train = torch.tensor(_stack(train_clips, "usad")[1])
    y_val = torch.tensor(_stack(val_clips, "usad")[1])
    report_models = {}
    onset = {}
    for name in ("usad", "efficientat"):
        x_train, y_from_train = _stack(train_clips, name)
        x_val, y_from_val = _stack(val_clips, name)
        if not np.array_equal(y_from_train, y_train.numpy()) or not np.array_equal(y_from_val, y_val.numpy()):
            raise AssertionError("STARSS label stack drifted")
        probed = _probe(x_train, x_val, y_train, y_val, seed)
        projected_train = fixed_project(x_train, D_MODEL, seed)
        projected_val = fixed_project(x_val, D_MODEL, seed)
        onset[name] = _onset_lift(y_val.numpy(), _val_scores(projected_train, y_train, projected_val, seed))
        report_models[name] = probed
    cosines = {}
    for name, key in (("usad", "usad_clip"), ("efficientat", "efficientat_clip")):
        same = []
        diff = []
        for i in range(len(clips)):
            for j in range(i + 1, len(clips)):
                left = torch.nn.functional.normalize(clips[i]["windows"][0][key], dim=0)
                right = torch.nn.functional.normalize(clips[j]["windows"][0][key], dim=0)
                value = float(torch.dot(left, right))
                if clips[i]["group"] == clips[j]["group"]:
                    same.append(value)
                else:
                    diff.append(value)
        cosines[name] = {
            "same_room_cosine_mean": float(np.mean(same)) if same else None,
            "different_room_cosine_mean": float(np.mean(diff)) if diff else None,
            "n_same_pairs": len(same),
            "n_diff_pairs": len(diff),
        }
    extra = {}
    try:
        extra_wav = hf_hub_download(
            "labhamlet/STARSS23", "original_audios/fold4_room10_mix003.wav", repo_type="dataset"
        )
        extra_audio = load_mono_audio(extra_wav, target_sr=SR, channel=0)
        extra_windows = _starss_windows(extra_audio)[:1]
        u_means, _u_frames = _usad_forward(usad, [extra_windows[0][1]], with_frames=False)
        e_means, _e_frames, _observed = _eat_forward(eat, mel, [extra_windows[0][1]], with_frames=False)
        anchor_u = torch.nn.functional.normalize(clips[0]["windows"][0]["usad_clip"], dim=0)
        anchor_e = torch.nn.functional.normalize(clips[0]["windows"][0]["efficientat_clip"], dim=0)
        extra = {
            "ids": [clips[0]["id"], "fold4_room10_mix003"],
            "same_group": clips[0]["group"] == "room10",
            "usad_cosine": float(torch.dot(anchor_u, torch.nn.functional.normalize(u_means[0], dim=0))),
            "efficientat_cosine": float(torch.dot(anchor_e, torch.nn.functional.normalize(e_means[0], dim=0))),
        }
    except Exception as exc:
        extra = {"status": "fail", "reason": f"{type(exc).__name__}: {exc}"}
    return {
        "status": "measured",
        "files": [clip["id"] for clip in clips],
        "rooms": [clip["group"] for clip in clips],
        "train_files": [clip["id"] for clip in train_clips],
        "val_files": [clip["id"] for clip in val_clips],
        "n_train_frames": int(y_train.shape[0]),
        "n_val_frames": int(y_val.shape[0]),
        "n_classes": int(y_train.shape[1]),
        "frame_macro_ap": report_models,
        "onset": onset,
        "cosine": cosines,
        "same_room_extra": extra,
        "native_usad_frames_on_full_crop": native_usad,
        "native_efficientat_fmap_time": fmap_time,
        "alignment": "linear interpolation of native frames onto 50 steps of 40 ms",
        "manifest_rewritten": False,
        "limit": (
            "four fold4 files, at most six non-overlapping 2.015 s windows each; "
            "labels are the same 100 ms STARSS frames rasterized onto the 40 ms grid; "
            "USAD frames are 20 ms encoder steps and EfficientAT frames are the last feature map, "
            "about 320 ms native stride, both interpolated to 50 steps; "
            "the 527-way EfficientAT head is not the probe"
        ),
    }


def _macro(block: dict[str, Any]) -> float | None:
    return block["probe"]["val"]["macro_ap"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifests/fsd50k_loopear_d.json")
    parser.add_argument("--audio-root", default="data/raw/fsd50k")
    parser.add_argument("--data-report", default="results/loopear_d_data.json")
    parser.add_argument("--global-report", default="results/loopear_d_eval_global.json")
    parser.add_argument("--train-clips", type=int, default=256)
    parser.add_argument("--max-windows", type=int, default=4)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", default="results/loopear_d_baselines.json")
    args = parser.parse_args()
    if torch.cuda.is_available():
        raise SystemExit("CUDA is visible; this comparison stays on CPU")
    warnings.filterwarnings("ignore", message="Don't use ConvNormActivation")
    warnings.filterwarnings("ignore", message="stft with return_complex=False is deprecated")
    warnings.filterwarnings("ignore", message="`torch.cuda.amp.autocast")
    torch.set_num_threads(4)
    cpu0 = cpu_seconds()
    wall0 = time.perf_counter()
    manifest = load_manifest(Path(args.manifest))
    data_report = json.loads(Path(args.data_report).read_text(encoding="utf-8"))
    global_report = json.loads(Path(args.global_report).read_text(encoding="utf-8"))
    val_ids = {int(i) for i in data_report["val_ids"]}
    train_rows = items_for(manifest, "h5")[: args.train_clips]
    val_rows = [row for row in manifest["items"] if int(row["freesound_id"]) in val_ids]
    for row in val_rows:
        if row["split"] != "val":
            raise PermissionError("val id resolved to a non-val row")
    for row in train_rows:
        if row["split"] != "train":
            raise PermissionError("probe train row is not train")
    if len(val_rows) != 160 or len(train_rows) != 256:
        raise SystemExit(f"split is {len(train_rows)}/{len(val_rows)}, expected 256/160")
    classes, y_train, y_val = _classes(train_rows, val_rows)
    if len(classes) != 28:
        raise SystemExit(f"expected 28 classes, got {len(classes)}")
    prevalence = float(y_val.mean())
    if abs(prevalence - float(global_report["val_label_prevalence"])) > 1e-6:
        raise SystemExit(f"val prevalence {prevalence} does not match the 5-hour probe")
    code, ckpt = _ensure_efficientat()
    eat, eat_mel, eat_params = _load_efficientat(code, ckpt)
    usad, _dtype = load_usad2_extractor("cpu")
    usad_params = int(sum(p.numel() for p in usad.parameters()))
    audio_root = Path(args.audio_root)
    checker = build_mel()
    print("embedding train", flush=True)
    train_feat = _embed_rows(train_rows, audio_root, usad, eat, eat_mel, args.max_windows, args.batch, checker)
    print("embedding val", flush=True)
    val_feat = _embed_rows(val_rows, audio_root, usad, eat, eat_mel, args.max_windows, args.batch, None)
    probes = {}
    for name in ("usad", "efficientat"):
        block = _probe(train_feat[name], val_feat[name], y_train, y_val, args.seed)
        if int(block["probe"]["probe_params"]) != 2716:
            raise SystemExit(f"{name} probe params {block['probe']['probe_params']}")
        probes[name] = block
    for name in ("usad", "efficientat"):
        if not torch.isfinite(train_feat[name]).all() or not torch.isfinite(val_feat[name]).all():
            raise SystemExit(f"non-finite {name} embeddings")
    few = _few_shot_status(train_rows, val_rows)
    print("starss", flush=True)
    starss = _starss(usad, eat, eat_mel, args.seed)
    if starss.get("files") != global_report["starss"]["files"]:
        raise SystemExit(f"STARSS files {starss.get('files')}")
    if int(starss.get("n_val_frames", 0)) != int(global_report["starss"]["n_val_frames"]):
        raise SystemExit("STARSS frame count does not match the 5-hour probe")
    global_patch = float(global_report["linear_probe"]["patch_mean"]["probe"]["val"]["macro_ap"])
    global_cls = float(global_report["linear_probe"]["cls"]["probe"]["val"]["macro_ap"])
    global_starss = float(global_report["starss"]["frame_macro_ap"]["encoder"]["macro_ap"])
    dsp_starss = float(global_report["starss"]["frame_macro_ap"]["dsp"]["macro_ap"])
    clip_vs_global = {
        name: {
            "macro_ap": _macro(probes[name]),
            "beats_global_patch_mean": bool(_macro(probes[name]) > global_patch),
            "delta_vs_global_patch_mean": float(_macro(probes[name]) - global_patch),
        }
        for name in probes
    }
    frame_vs_global = {}
    for name in ("usad", "efficientat"):
        block = starss["frame_macro_ap"][name]
        value = _macro(block)
        shuffled = float(block["shuffled_label_probe"]["val"]["macro_ap"])
        label_sensitive = abs(float(value) - shuffled) > 1e-4
        frame_vs_global[name] = {
            "macro_ap": value,
            "shuffled_macro_ap": shuffled,
            "label_sensitive": label_sensitive,
            "beats_global_frame": bool(value > global_starss and label_sensitive),
            "raw_above_global_frame": bool(value > global_starss),
            "delta_vs_global_frame": float(value - global_starss),
            "beats_dsp_frame": bool(value > dsp_starss and label_sensitive),
        }
    mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    report = {
        "task": "loopear_d_frozen_baselines",
        "gpu_usd": 0.0,
        "cuda_available": False,
        "mps_available": mps,
        "audioset_audio_downloaded": False,
        "fsd50k_test_audio_read": False,
        "sealed_realdesed_test_read": False,
        "training_steps": 0,
        "n_train_clips": len(train_rows),
        "n_val_clips": len(val_rows),
        "n_classes": len(classes),
        "classes": classes,
        "val_label_prevalence": float(y_val.mean()),
        "crop_s": CROP_SECONDS,
        "max_windows": args.max_windows,
        "window_rule": "same non-overlapping 200-frame spans as encode_logmel_windows, mean pooled",
        "readout": "fixed Gaussian projection to 96, seed 42, then Linear, AdamW lr 1e-2, 250 steps, seed 42",
        "usad": {
            "name": "MIT-SLS/USAD2-Small",
            "snapshot": USAD_SNAPSHOT,
            "params": usad_params,
            "feature_dim": int(train_feat["usad"].shape[1]),
            "native_frames_on_full_crop": train_feat["native"]["usad_frames"],
            "n_train_windows": train_feat["n_windows"],
            "n_val_windows": val_feat["n_windows"],
            "n_short_windows": int(train_feat["n_short_windows"] + val_feat["n_short_windows"]),
            "probe": probes["usad"],
        },
        "efficientat": {
            "name": "mn04_as",
            "published_audioset_map": 0.432,
            "commit": EFFICIENTAT_COMMIT,
            "checkpoint_sha256": EFFICIENTAT_SHA256,
            "checkpoint_bytes": ckpt.stat().st_size,
            "params": eat_params,
            "feature_dim": int(train_feat["efficientat"].shape[1]),
            "frontend": "32 kHz, preemphasis 0.97, n_fft 1024, win 800, hop 320, 128 Kaldi mels, fmax 15000, eval mode",
            "precision": "checkpoint trained in float16; this run is float32 CPU",
            "head_used": False,
            "mel_frames_on_full_crop": train_feat["native"]["mel_frames"],
            "fmap_time_on_full_crop": train_feat["native"]["fmap_time"],
            "fmap_freq_on_full_crop": train_feat["native"]["fmap_freq"],
            "n_train_windows": train_feat["n_windows"],
            "n_val_windows": val_feat["n_windows"],
            "n_short_windows": int(train_feat["n_short_windows"] + val_feat["n_short_windows"]),
            "probe": probes["efficientat"],
        },
        "global_only_reference": {
            "source": args.global_report,
            "patch_mean_macro_ap": global_patch,
            "cls_macro_ap": global_cls,
            "starss_frame_macro_ap": global_starss,
            "starss_dsp_macro_ap": dsp_starss,
        },
        "beats_global_only": {"clip": clip_vs_global, "starss_frame": frame_vs_global},
        "few_shot_unfamiliar": few,
        "starss": starss,
        "wall_hours": hours(time.perf_counter() - wall0),
        "cpu_hours": hours(cpu_seconds() - cpu0),
        "peak_rss_gb": peak_rss_gb(),
    }
    write_result_json(args.report, report)
    print(
        json.dumps(
            {
                "usad_clip": clip_vs_global["usad"],
                "efficientat_clip": clip_vs_global["efficientat"],
                "starss": frame_vs_global,
                "few_shot": few["status"],
                "gpu_usd": 0.0,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
