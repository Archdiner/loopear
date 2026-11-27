import argparse
import importlib.util
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from loopear.data import STARSS23_CLASSES, rasterize_starss_frames_50hz
from loopear.d.diagnostics import cpu_seconds, hours, peak_rss_gb
from loopear.d.frontend import build_mel, load_frontend_config, normalize_logmel, waveform_logmel
from loopear.d.fsd import audio_path, items_for, load_manifest, rank_eligible
from loopear.d.loop import init_model
from loopear.d.objective import broadcast_time_valid, patch_valid_from_frames
from loopear.d.perceiver import LoopEarPEncoder, init_perceiver
from loopear.d.probe import fit_linear_probe, fixed_project, logmel_stats, macro_average_precision
from loopear.d.protocol import (
    clip_window_starts,
    coordinate_pool,
    crop_energy_profile,
    event_window_starts,
    few_example_classes,
    fourier_coordinates,
    mean_pearson,
    starss_keep,
    tag_class_names,
)
from loopear.d.spec import CROP_SAMPLES, D_MODEL, GRID_T, HOP_SAMPLES, N_FRAMES, N_MELS, SR
from loopear.encoder import load_mono_audio, load_usad2_extractor
from loopear.harness import write_result_json
from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parent.parent
VAL_CLIPS = 640
TAG_WINDOWS = 4
PROBE_STEPS = 250
PROBE_LR = 1e-2
SEED = 42
PROFILE_BINS = 10


def _baselines():
    path = Path(__file__).with_name("eval_loopear_d_baselines.py")
    spec = importlib.util.spec_from_file_location("eval_loopear_d_baselines", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_d(path: Path):
    model = init_model(0)
    model.encoder.load_state_dict(load_file(str(path)))
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def _load_p(path: Path, time_alpha: float = 0.0) -> LoopEarPEncoder:
    model = LoopEarPEncoder()
    model.load_state_dict(load_file(str(path)))
    model.time_alpha = float(time_alpha)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def _official_val(manifest: dict[str, Any], n: int) -> list[dict[str, Any]]:
    by_id = {}
    copies = []
    for row in manifest["items"]:
        if row["split"] != "val":
            continue
        by_id[int(row["freesound_id"])] = row
        copied = dict(row)
        copied["split"] = "train"
        copied["duplicate_of"] = None
        copies.append(copied)
    ranked = rank_eligible(copies)
    chosen = []
    for row in ranked[:n]:
        source = by_id[int(row["freesound_id"])]
        if source["split"] != "val":
            raise PermissionError("validation probe selected a non-val row")
        chosen.append(source)
    return chosen


def _read_flac(path: Path) -> np.ndarray:
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if int(sr) != SR:
        raise AssertionError(f"sr {sr} at {path}")
    if wav.ndim != 1:
        wav = np.mean(wav, axis=-1)
    return np.ascontiguousarray(wav, dtype=np.float32)


def _window(logmel: torch.Tensor, start: int, mean: torch.Tensor, std: torch.Tensor):
    chunk = logmel[start : start + N_FRAMES]
    valid = torch.zeros(N_FRAMES, dtype=torch.bool)
    if int(chunk.shape[0]) < N_FRAMES:
        padded = torch.zeros(N_FRAMES, N_MELS, dtype=logmel.dtype)
        if int(chunk.shape[0]) > 0:
            padded[: chunk.shape[0]] = chunk
            valid[: chunk.shape[0]] = True
        chunk = padded
    else:
        valid[:] = True
    norm = normalize_logmel(chunk, mean, std) * valid.unsqueeze(-1)
    return norm, valid


def _multi_hot(rows: list[dict[str, Any]], names: list[str]) -> torch.Tensor:
    index = {name: pos for pos, name in enumerate(names)}
    out = torch.zeros(len(rows), len(names))
    for row_index, row in enumerate(rows):
        for label in set(row["labels"]):
            if label in index:
                out[row_index, index[label]] = 1.0
    return out


def _probe_pair(x_train: torch.Tensor, x_val: torch.Tensor, y_train: torch.Tensor, y_val: torch.Tensor) -> dict[str, Any]:
    projected_train = fixed_project(x_train, D_MODEL, SEED)
    projected_val = fixed_project(x_val, D_MODEL, SEED)
    scored = fit_linear_probe(projected_train, y_train, projected_val, y_val, PROBE_STEPS, PROBE_LR, SEED)
    perm = torch.Generator().manual_seed(SEED + 7)
    shuffled = y_train[torch.randperm(y_train.shape[0], generator=perm)]
    control = fit_linear_probe(projected_train, shuffled, projected_val, y_val, PROBE_STEPS, PROBE_LR, SEED)
    return {
        "input_dim": int(x_train.shape[1]),
        "projected_dim": D_MODEL,
        "probe": scored,
        "shuffled_label_probe": control,
    }


def _fit_map(x_train: torch.Tensor, y_train: torch.Tensor, x_val: torch.Tensor, seed: int) -> np.ndarray:
    torch.manual_seed(seed)
    layer = torch.nn.Linear(int(x_train.shape[1]), int(y_train.shape[1]))
    opt = torch.optim.AdamW(layer.parameters(), lr=PROBE_LR, weight_decay=0.0)
    for _ in range(PROBE_STEPS):
        opt.zero_grad(set_to_none=True)
        loss = F.mse_loss(layer(x_train), y_train)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return layer(x_val).numpy()


def _fit_shared(tokens_train: torch.Tensor, y_train: torch.Tensor, tokens_val: torch.Tensor, seed: int) -> np.ndarray:
    torch.manual_seed(seed)
    layer = torch.nn.Linear(int(tokens_train.shape[-1]), 1)
    opt = torch.optim.AdamW(layer.parameters(), lr=PROBE_LR, weight_decay=0.0)
    for _ in range(PROBE_STEPS):
        opt.zero_grad(set_to_none=True)
        pred = layer(tokens_train).squeeze(-1)
        loss = F.mse_loss(pred, y_train)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return layer(tokens_val).squeeze(-1).numpy()


def _project_time(tokens: torch.Tensor) -> torch.Tensor:
    n, steps, dim = tokens.shape
    flat = fixed_project(tokens.reshape(n * steps, dim), D_MODEL, SEED)
    return flat.reshape(n, steps, D_MODEL)


def _time_bins(grid: torch.Tensor, visible: torch.Tensor) -> torch.Tensor:
    time_tokens = grid.mean(dim=2)
    weights = visible.any(dim=-1).to(time_tokens.dtype)
    width = GRID_T // PROFILE_BINS
    bins = []
    for index in range(PROFILE_BINS):
        sl = time_tokens[:, index * width : (index + 1) * width]
        w = weights[:, index * width : (index + 1) * width]
        bins.append((sl * w.unsqueeze(-1)).sum(dim=1) / w.sum(dim=1).clamp_min(1.0).unsqueeze(-1))
    return torch.stack(bins, dim=1)


def _split_bins(seq: torch.Tensor, n_bins: int) -> torch.Tensor:
    if int(seq.shape[0]) == 0:
        return torch.zeros(n_bins, int(seq.shape[-1]) if seq.ndim == 2 else D_MODEL)
    if int(seq.shape[0]) < n_bins:
        seq = seq[:1].expand(n_bins, seq.shape[-1]).contiguous()
    edges = np.linspace(0, int(seq.shape[0]), n_bins + 1).astype(int)
    parts = []
    for left, right in zip(edges[:-1], edges[1:]):
        if right <= left:
            parts.append(torch.zeros(seq.shape[-1], dtype=seq.dtype))
        else:
            parts.append(seq[left:right].mean(dim=0))
    return torch.stack(parts, dim=0)


def _scores(x_train: torch.Tensor, y_train: torch.Tensor, x_val: torch.Tensor, seed: int) -> np.ndarray:
    torch.manual_seed(seed)
    layer = torch.nn.Linear(int(x_train.shape[1]), int(y_train.shape[1]))
    opt = torch.optim.AdamW(layer.parameters(), lr=PROBE_LR, weight_decay=0.0)
    for _ in range(PROBE_STEPS):
        opt.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(layer(x_train), y_train)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return torch.sigmoid(layer(x_val)).numpy()


def _onset_within_windows(labels: list[np.ndarray], scores: list[np.ndarray]) -> dict[str, Any]:
    inactive = [score[label.sum(axis=1) == 0] for label, score in zip(labels, scores)]
    pooled = np.concatenate(inactive, axis=0) if any(part.size for part in inactive) else np.zeros((0, 1))
    background = float(pooled.mean()) if pooled.size else 0.0
    peaks = []
    above = 0
    for label, score in zip(labels, scores):
        for t in range(label.shape[0]):
            for cls in range(label.shape[1]):
                if label[t, cls] < 0.5:
                    continue
                if t > 0 and label[t - 1, cls] >= 0.5:
                    continue
                lo = max(0, t - 2)
                hi = min(label.shape[0], t + 3)
                peak = float(score[lo:hi, cls].max())
                peaks.append(peak - background)
                if peak > background:
                    above += 1
    if not peaks:
        return {"n_onsets": 0, "mean_lift": None, "n_above_background": 0}
    return {"n_onsets": len(peaks), "mean_lift": float(np.mean(peaks)), "n_above_background": above}


def _unseen_primary(train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]]) -> dict[str, int]:
    train_primary = {row["labels"][0] for row in train_rows if row["labels"]}
    counts: dict[str, int] = defaultdict(int)
    for row in val_rows:
        if not row["labels"]:
            continue
        label = row["labels"][0]
        if label not in train_primary:
            counts[label] += 1
    return {
        "n_unfamiliar_primaries": len(counts),
        "n_with_at_least_6_clips": sum(1 for count in counts.values() if count >= 6),
    }


def _encode_fsd(rows, models, helpers, mean, std, mel_bank) -> dict[str, Any]:
    n = len(rows)
    has_k1 = models.get("d_k1") is not None
    names = {
        "d_global_cls": D_MODEL,
        "d_global_patch": D_MODEL,
        "p_summary": D_MODEL,
        "p_latents": 32 * D_MODEL,
        "usad": None,
        "efficientat": None,
        "dsp": N_MELS * 2,
    }
    if has_k1:
        names["d_k1_cls"] = D_MODEL
        names["d_k1_patch"] = D_MODEL
    sums = {key: None for key in names}
    counts = np.zeros(n, dtype=np.float64)
    center = {
        "profile": np.zeros((n, PROFILE_BINS), dtype=np.float32),
        "d_global_local": np.zeros((n, PROFILE_BINS, D_MODEL), dtype=np.float32),
        "d_k1_local": np.zeros((n, PROFILE_BINS, D_MODEL), dtype=np.float32),
        "d_global_cls": np.zeros((n, D_MODEL), dtype=np.float32),
        "d_k1_cls": np.zeros((n, D_MODEL), dtype=np.float32),
        "p_summary": np.zeros((n, D_MODEL), dtype=np.float32),
        "p_latents": np.zeros((n, 32 * D_MODEL), dtype=np.float32),
        "p_random": np.zeros((n, 32 * D_MODEL), dtype=np.float32),
        "usad_mean": None,
        "usad_local": None,
        "eat_mean": None,
        "eat_local": None,
        "dsp": np.zeros((n, N_MELS * 2), dtype=np.float32),
        "full": np.zeros(n, dtype=bool),
    }
    if models.get("p_reference") is not None:
        center["p_reference"] = np.zeros((n, 32 * D_MODEL), dtype=np.float32)
    pending: list[dict[str, Any]] = []
    native = {"mel_frames": 0, "fmap_time": 0, "fmap_freq": 0, "usad_frames": 0}
    n_windows = 0
    n_short = 0

    def _ensure(key: str, dim: int) -> None:
        if sums[key] is None:
            sums[key] = np.zeros((n, dim), dtype=np.float64)

    def _flush() -> None:
        if not pending:
            return
        mel = torch.stack([item["norm"] for item in pending])
        valid = torch.stack([item["valid"] for item in pending])
        visible = broadcast_time_valid(patch_valid_from_frames(valid))
        ok = visible.flatten(1).any(dim=1)
        if not bool(ok.all()):
            keep = [index for index, flag in enumerate(ok.tolist()) if flag]
            if not keep:
                pending.clear()
                return
            chosen = [pending[index] for index in keep]
            pending.clear()
            pending.extend(chosen)
            _flush()
            return
        with torch.inference_mode():
            g_grid, g_cls = models["d_global"].encoder(mel, visible)
            k_grid = k_cls = None
            if has_k1:
                k_grid, k_cls = models["d_k1"].encoder(mel, visible)
            latents, summary = models["p"](mel, visible)
            random_latents = None
            if models.get("p_random") is not None:
                random_latents, _random_summary = models["p_random"](mel, visible)
            reference_latents = None
            if models.get("p_reference") is not None:
                reference_latents, _reference_summary = models["p_reference"](mel, visible)
        g_patch = []
        k_patch = []
        for index in range(mel.shape[0]):
            g_tok = g_grid[index][visible[index]]
            g_patch.append(g_tok.mean(dim=0) if g_tok.shape[0] else torch.zeros(D_MODEL))
            if has_k1:
                k_tok = k_grid[index][visible[index]]
                k_patch.append(k_tok.mean(dim=0) if k_tok.shape[0] else torch.zeros(D_MODEL))
        g_patch_t = torch.stack(g_patch)
        k_patch_t = torch.stack(k_patch) if has_k1 else None
        g_local = _time_bins(g_grid, visible)
        k_local = _time_bins(k_grid, visible) if has_k1 else None
        dsp = logmel_stats(mel, valid)
        waves = [item["wave"] for item in pending]
        u_means, u_frames = helpers._usad_forward(models["usad"], waves, with_frames=True)
        e_means, e_frames, observed = helpers._eat_forward(models["eat"], models["eat_mel"], waves, with_frames=True)
        if observed["mel_frames"]:
            native["mel_frames"] = int(observed["mel_frames"])
            native["fmap_time"] = int(observed["fmap_time"])
            native["fmap_freq"] = int(observed["fmap_freq"])
        usad_dim = int(u_means[0].shape[0])
        eat_dim = int(e_means[0].shape[0])
        _ensure("usad", usad_dim)
        _ensure("efficientat", eat_dim)
        for key, dim in names.items():
            if dim is not None:
                _ensure(key, dim)
        packed = {
            "d_global_cls": g_cls,
            "d_global_patch": g_patch_t,
            "p_summary": summary,
            "p_latents": latents.reshape(latents.shape[0], -1),
            "dsp": dsp,
        }
        if has_k1:
            packed["d_k1_cls"] = k_cls
            packed["d_k1_patch"] = k_patch_t
        for local, item in enumerate(pending):
            owner = int(item["owner"])
            counts[owner] += 1.0
            for key, value in packed.items():
                sums[key][owner] += value[local].detach().float().cpu().numpy()
            sums["usad"][owner] += u_means[local].detach().float().cpu().numpy()
            sums["efficientat"][owner] += e_means[local].detach().float().cpu().numpy()
            if item["center"] and bool(item["valid"].all()) and int(item["wave"].shape[0]) == CROP_SAMPLES:
                center["full"][owner] = True
                center["profile"][owner] = crop_energy_profile(item["norm"]).numpy()
                center["d_global_local"][owner] = g_local[local].detach().float().cpu().numpy()
                center["d_global_cls"][owner] = g_cls[local].detach().float().cpu().numpy()
                if has_k1:
                    center["d_k1_local"][owner] = k_local[local].detach().float().cpu().numpy()
                    center["d_k1_cls"][owner] = k_cls[local].detach().float().cpu().numpy()
                center["p_summary"][owner] = summary[local].detach().float().cpu().numpy()
                center["p_latents"][owner] = latents[local].reshape(-1).detach().float().cpu().numpy()
                if random_latents is not None:
                    center["p_random"][owner] = random_latents[local].reshape(-1).detach().float().cpu().numpy()
                if reference_latents is not None:
                    center["p_reference"][owner] = reference_latents[local].reshape(-1).detach().float().cpu().numpy()
                center["dsp"][owner] = dsp[local].detach().float().cpu().numpy()
                if center["usad_mean"] is None:
                    center["usad_mean"] = np.zeros((n, usad_dim), dtype=np.float32)
                    center["usad_local"] = np.zeros((n, PROFILE_BINS, usad_dim), dtype=np.float32)
                    center["eat_mean"] = np.zeros((n, eat_dim), dtype=np.float32)
                    center["eat_local"] = np.zeros((n, PROFILE_BINS, eat_dim), dtype=np.float32)
                center["usad_mean"][owner] = u_means[local].detach().float().cpu().numpy()
                center["eat_mean"][owner] = e_means[local].detach().float().cpu().numpy()
                center["usad_local"][owner] = _split_bins(u_frames[local], PROFILE_BINS).numpy()
                center["eat_local"][owner] = _split_bins(e_frames[local], PROFILE_BINS).numpy()
        pending.clear()

    for row_index, row in enumerate(rows):
        wav = _read_flac(audio_path(Path("data/raw/fsd50k"), row))
        spec = waveform_logmel(torch.from_numpy(wav), mel_bank)
        starts, center_i = clip_window_starts(int(spec.shape[0]), N_FRAMES, TAG_WINDOWS)
        for window_index, start in enumerate(starts):
            norm, valid = _window(spec, start, mean, std)
            if not bool(valid.any()):
                continue
            left = int(start) * HOP_SAMPLES
            right = min(int(wav.shape[0]), left + CROP_SAMPLES)
            if right <= left:
                continue
            wave = torch.from_numpy(np.ascontiguousarray(wav[left:right]))
            n_windows += 1
            if int(wave.shape[0]) < CROP_SAMPLES:
                n_short += 1
            pending.append(
                {
                    "owner": row_index,
                    "norm": norm,
                    "valid": valid,
                    "wave": wave,
                    "center": window_index == center_i,
                }
            )
            if len(pending) >= 8:
                _flush()
        if row_index % 128 == 127:
            print(f"fsd clips {row_index + 1}/{n}", flush=True)
    _flush()
    with torch.inference_mode():
        probe = models["usad"](wavs=torch.zeros(1, CROP_SAMPLES), wav_lengths=torch.tensor([CROP_SAMPLES]))
    native["usad_frames"] = int(probe["x_lengths"][0])
    denom = np.clip(counts, 1.0, None)
    means = {key: (value / denom[:, None]).astype(np.float32) for key, value in sums.items() if value is not None}
    empty = int((counts == 0).sum())
    return {
        "means": means,
        "center": center,
        "counts": counts,
        "n_windows": n_windows,
        "n_short_windows": n_short,
        "n_empty_clips": empty,
        "native": native,
        "n_full_fidelity": int(center["full"].sum()),
    }


def _fidelity_block(encoded: dict[str, Any], n_train: int) -> dict[str, Any]:
    full = encoded["center"]["full"]
    train_idx = np.where(full[:n_train])[0]
    val_idx = np.where(full[n_train:])[0] + n_train
    target_train = torch.from_numpy(encoded["center"]["profile"][train_idx])
    target_val = torch.from_numpy(encoded["center"]["profile"][val_idx])
    out = {
        "n_train_full_crops": int(train_idx.shape[0]),
        "n_val_full_crops": int(val_idx.shape[0]),
        "target": "10-bin mean log-mel energy of one full 2.015 s crop, centered inside the crop",
        "fit": "250 steps, AdamW lr 1e-2, seed 42, mean per-window Pearson r on validation crops",
    }
    global_features = {
        "loopear_d_global_cls": encoded["center"]["d_global_cls"],
        "loopear_p_summary": encoded["center"]["p_summary"],
        "loopear_p_latents": encoded["center"]["p_latents"],
        "usad2_small": encoded["center"]["usad_mean"],
        "efficientat_mn04_as": encoded["center"]["eat_mean"],
        "dsp": encoded["center"]["dsp"],
    }
    local_features = {
        "loopear_d_global_time": encoded["center"]["d_global_local"],
        "usad2_small_time": encoded["center"]["usad_local"],
        "efficientat_mn04_as_time": encoded["center"]["eat_local"],
    }
    if "d_k1_cls" in encoded["means"]:
        global_features["loopear_d_k1_cls"] = encoded["center"]["d_k1_cls"]
        local_features["loopear_d_k1_time"] = encoded["center"]["d_k1_local"]
    for name, matrix in global_features.items():
        x_train = fixed_project(torch.from_numpy(matrix[train_idx]), D_MODEL, SEED)
        x_val = fixed_project(torch.from_numpy(matrix[val_idx]), D_MODEL, SEED)
        pred = _fit_map(x_train, target_train, x_val, SEED)
        out[name] = {"readout": "linear_96_to_10", **mean_pearson(pred, target_val.numpy())}
    ceiling = _fit_map(target_train, target_train, target_val, SEED)
    out["profile_identity"] = {"readout": "linear_10_to_10", **mean_pearson(ceiling, target_val.numpy())}
    for name, matrix in local_features.items():
        train_tokens = _project_time(torch.from_numpy(matrix[train_idx]))
        val_tokens = _project_time(torch.from_numpy(matrix[val_idx]))
        pred = _fit_shared(train_tokens, target_train, val_tokens, SEED)
        out[name] = {"readout": "shared_linear_on_10_time_bins", **mean_pearson(pred, target_val.numpy())}
    return out


def _few_block(rows_train, rows_val, features: dict[str, np.ndarray]) -> dict[str, Any]:
    specs = few_example_classes(rows_train, rows_val, k=4)
    payload = {
        "k_positive": 4,
        "k_negative": 4,
        "support": "development clips in the 5.001 h subset",
        "query": "official FSD50K validation clips in this probe",
        "n_classes": len(specs),
        "rule": "a class needs 4 development positives, 4 development negatives, and 4 validation positives",
        "models": {},
    }
    if not specs:
        payload["status"] = "null"
        return payload
    payload["status"] = "measured"
    payload["labels"] = [spec["label"] for spec in specs]
    for name, matrix in features.items():
        aps = []
        controls = []
        prevalences = []
        x_all_train = torch.from_numpy(matrix[: len(rows_train)])
        x_all_val = torch.from_numpy(matrix[len(rows_train) :])
        for spec in specs:
            pos = spec["train_positive"]
            neg = spec["train_negative"]
            x_train = fixed_project(torch.cat([x_all_train[pos], x_all_train[neg]], dim=0), D_MODEL, SEED)
            y_train = torch.tensor([1.0] * 4 + [0.0] * 4)
            y_val_np = np.asarray(
                [1.0 if spec["label"] in set(row["labels"]) else 0.0 for row in rows_val],
                dtype=np.float32,
            )
            x_val = fixed_project(x_all_val, D_MODEL, SEED)
            score = _scores(x_train, y_train.view(-1, 1), x_val, SEED)[:, 0]
            measured = macro_average_precision(y_val_np.reshape(-1, 1), score.reshape(-1, 1))["macro_ap"]
            if measured is None:
                continue
            aps.append(float(measured))
            perm = torch.Generator().manual_seed(SEED + 7)
            shuffled = y_train[torch.randperm(8, generator=perm)].view(-1, 1)
            control = _scores(x_train, shuffled, x_val, SEED)[:, 0]
            control_ap = macro_average_precision(y_val_np.reshape(-1, 1), control.reshape(-1, 1))["macro_ap"]
            controls.append(float(control_ap) if control_ap is not None else float("nan"))
            prevalences.append(float(y_val_np.mean()))
        if not aps:
            payload["models"][name] = {"status": "null"}
            continue
        payload["models"][name] = {
            "macro_ap": float(np.mean(aps)),
            "shuffled_support_macro_ap": float(np.nanmean(controls)),
            "mean_val_prevalence": float(np.mean(prevalences)),
            "n_classes_scored": len(aps),
        }
    return payload


def _parse_events(text: str) -> list[tuple[int, int]]:
    parsed = []
    for line in text.splitlines():
        cols = [part.strip() for part in line.split(",")]
        if len(cols) >= 2 and cols[0].isdigit():
            parsed.append((int(cols[0]), int(cols[1])))
    return parsed


def _label_csv(name: str) -> str:
    last_error: Exception | None = None
    for prefix in ("labels/train/", "labels/test/", "labels/valid/", "labels/"):
        try:
            return hf_hub_download("labhamlet/STARSS23", prefix + name + ".csv", repo_type="dataset")
        except Exception as exc:
            last_error = exc
    raise FileNotFoundError(name) if last_error is None else last_error


def _starss(models, helpers, mean, std, mel_bank) -> dict[str, Any]:
    manifest = load_manifest(Path("data/manifests/starss23.json"))
    clips = []
    for row in manifest["items"]:
        if row["split"] not in ("train", "val"):
            raise PermissionError(f"STARSS manifest row {row['id']} is {row['split']}")
        wav_src = hf_hub_download(
            "labhamlet/STARSS23",
            "original_audios/" + row["id"] + ".wav",
            repo_type="dataset",
        )
        csv_src = _label_csv(row["id"])
        wav = load_mono_audio(wav_src, target_sr=SR, channel=0)
        spec = waveform_logmel(torch.from_numpy(wav), mel_bank)
        events = _parse_events(Path(csv_src).read_text(encoding="utf-8"))
        duration = float(wav.shape[0]) / float(SR)
        starts = event_window_starts(duration, [(idx * 0.1, cls) for idx, cls in events], max_windows=6)
        windows = []
        for start_s in starts:
            sample = int(round(start_s * SR))
            sample = min(sample, max(int(wav.shape[0]) - 1, 0))
            if sample + CROP_SAMPLES > int(wav.shape[0]):
                sample = max(int(wav.shape[0]) - CROP_SAMPLES, 0)
            frame = sample // HOP_SAMPLES
            norm, valid = _window(spec, frame, mean, std)
            if not bool(valid.any()):
                continue
            visible = broadcast_time_valid(patch_valid_from_frames(valid.unsqueeze(0)))
            wave = torch.from_numpy(np.ascontiguousarray(wav[sample : sample + CROP_SAMPLES]))
            with torch.inference_mode():
                grid, cls = models["d_global"].encoder(norm.unsqueeze(0), visible)
                k_grid = None
                if models.get("d_k1") is not None:
                    k_grid, _k_cls = models["d_k1"].encoder(norm.unsqueeze(0), visible)
                _latents, summary = models["p"](norm.unsqueeze(0), visible)
            _u_means, u_frames = helpers._usad_forward(models["usad"], [wave], with_frames=True)
            _e_means, e_frames, _observed = helpers._eat_forward(
                models["eat"], models["eat_mel"], [wave], with_frames=True
            )
            if int(u_frames[0].shape[0]) != GRID_T or int(e_frames[0].shape[0]) != GRID_T:
                raise AssertionError("STARSS frame grid is not 50 steps")
            grouped = norm.view(GRID_T, 4, N_MELS)
            dsp = torch.cat([grouped.mean(dim=1), grouped.std(dim=1, unbiased=False)], dim=-1)
            labels = rasterize_starss_frames_50hz(
                events,
                float(sample) / float(SR),
                GRID_T,
                len(STARSS23_CLASSES),
                hop_e=0.04,
                label_hop_s=0.1,
            )
            windows.append(
                {
                    "d_global_time": grid[0].mean(dim=1).detach().float().cpu(),
                    "d_global_cls": cls[0].detach().float().cpu().unsqueeze(0).expand(GRID_T, D_MODEL).contiguous(),
                    "p_summary": summary[0].detach().float().cpu().unsqueeze(0).expand(GRID_T, D_MODEL).contiguous(),
                    "usad": u_frames[0].detach().float().cpu(),
                    "efficientat": e_frames[0].detach().float().cpu(),
                    "dsp": dsp.detach().float().cpu(),
                    "y": labels,
                    "cls_anchor": cls[0].detach().float().cpu(),
                }
            )
            if k_grid is not None:
                windows[-1]["d_k1_time"] = k_grid[0].mean(dim=1).detach().float().cpu()
        clips.append({"id": row["id"], "split": row["split"], "group": row["group_id"], "windows": windows})
        print(f"starss {row['id']} windows {len(windows)}", flush=True)
    train_clips = [clip for clip in clips if clip["split"] == "train"]
    val_clips = [clip for clip in clips if clip["split"] == "val"]

    def _collect(group, key):
        xs = []
        ys = []
        files = []
        y_windows = []
        for file_index, clip in enumerate(group):
            for window in clip["windows"]:
                xs.append(window[key])
                ys.append(window["y"])
                files.append(np.full(window["y"].shape[0], file_index, dtype=np.int64))
                y_windows.append(window["y"])
        if not xs:
            raise RuntimeError("STARSS split produced no windows")
        return torch.cat(xs, dim=0), np.concatenate(ys, axis=0), np.concatenate(files, axis=0), y_windows

    y_probe_train = None
    y_val = None
    per_model = {}
    keys = ["d_global_time", "d_global_cls", "p_summary", "usad", "efficientat", "dsp"]
    if models.get("d_k1") is not None:
        keys.insert(2, "d_k1_time")
    window_labels = None
    for key in keys:
        x_train, y_train, files_tr, _train_windows = _collect(train_clips, key)
        x_val, y_here, files_va, val_windows = _collect(val_clips, key)
        if y_probe_train is None:
            y_probe_train = y_train
            y_val = y_here
            window_labels = val_windows
        keep, detail = starss_keep(y_train, y_here, files_tr, files_va, min_files=2, min_frames=40)
        if key == "dsp":
            class_detail = detail
            class_keep = keep
        x_tr = fixed_project(x_train, D_MODEL, SEED)
        x_va = fixed_project(x_val, D_MODEL, SEED)
        y_tr = torch.from_numpy(y_train[:, keep])
        y_va = torch.from_numpy(y_here[:, keep])
        if int(keep.sum()) == 0:
            per_model[key] = {"status": "null", "reason": "no class met the support rule"}
            continue
        real = _scores(x_tr, y_tr, x_va, SEED)
        perm = torch.Generator().manual_seed(SEED + 7)
        order = torch.randperm(y_tr.shape[0], generator=perm)
        shuffled = _scores(x_tr, y_tr[order], x_va, SEED)
        real_ap = macro_average_precision(y_va.numpy(), real)
        shuffled_ap = macro_average_precision(y_va.numpy(), shuffled)
        score_windows = []
        cursor = 0
        for window in val_windows:
            n_frames = window.shape[0]
            score_windows.append(real[cursor : cursor + n_frames])
            cursor += n_frames
        label_windows = [window[:, keep] for window in window_labels]
        per_model[key] = {
            "status": "measured",
            "val": real_ap,
            "shuffled_label_probe": shuffled_ap,
            "label_sensitive": abs(float(real_ap["macro_ap"]) - float(shuffled_ap["macro_ap"])) > 1e-6,
            "onset_within_windows": _onset_within_windows(label_windows, score_windows),
            "probe_params": int(D_MODEL * int(keep.sum()) + int(keep.sum())),
        }
    names = [STARSS23_CLASSES[row["class_index"]] for row in class_detail]
    for row, name in zip(class_detail, names):
        row["class"] = name
    return {
        "manifest_rewritten": False,
        "train_files": [clip["id"] for clip in train_clips],
        "val_files": [clip["id"] for clip in val_clips],
        "n_train_windows": sum(len(clip["windows"]) for clip in train_clips),
        "n_val_windows": sum(len(clip["windows"]) for clip in val_clips),
        "n_train_frames": int(y_probe_train.shape[0]),
        "n_val_frames": int(y_val.shape[0]),
        "support_rule": "at least 2 recordings and 40 positive 40 ms frames on both the fold3 probe-train side and the fold4 eval side",
        "classes": class_detail,
        "n_classes_scored": int(class_keep.sum()),
        "models": per_model,
        "frame_axis": "50 patch steps of 40 ms inside each 2.015 s window; LoopEar-P has no time axis, so its summary is repeated on every step",
    }


def _fit_per_bin(tokens_train: torch.Tensor, y_train: torch.Tensor, tokens_val: torch.Tensor, seed: int) -> tuple[np.ndarray, int]:
    torch.manual_seed(seed)
    weight = torch.nn.Parameter(torch.empty(int(tokens_train.shape[1]), int(tokens_train.shape[-1])))
    torch.nn.init.xavier_uniform_(weight)
    bias = torch.nn.Parameter(torch.zeros(int(tokens_train.shape[1])))
    opt = torch.optim.AdamW([weight, bias], lr=PROBE_LR, weight_decay=0.0)
    for _ in range(PROBE_STEPS):
        opt.zero_grad(set_to_none=True)
        pred = (tokens_train * weight).sum(dim=-1) + bias
        loss = F.mse_loss(pred, y_train)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = ((tokens_val * weight).sum(dim=-1) + bias).numpy()
    return pred, int(weight.numel() + bias.numel())


def _coordinate_block(encoded: dict[str, Any], n_train: int, evaluated_name: str = "trained_perceiver") -> dict[str, Any]:
    full = encoded["center"]["full"]
    train_idx = np.where(full[:n_train])[0]
    val_idx = np.where(full[n_train:])[0] + n_train
    target_train = torch.from_numpy(encoded["center"]["profile"][train_idx])
    target_val = torch.from_numpy(encoded["center"]["profile"][val_idx])
    coords = fourier_coordinates(PROFILE_BINS, D_MODEL)

    def _from_latents(flat: np.ndarray) -> torch.Tensor:
        latents = torch.from_numpy(flat).view(-1, 32, D_MODEL)
        return coordinate_pool(latents, coords)

    sources = {
        evaluated_name: _from_latents(encoded["center"]["p_latents"]),
        "random_init_perceiver": _from_latents(encoded["center"]["p_random"]),
        "loopear_d_time_bins": torch.from_numpy(encoded["center"]["d_global_local"]),
    }
    if "p_reference" in encoded["center"]:
        reference_name = "trained_perceiver" if evaluated_name != "trained_perceiver" else "perceiver_reference"
        sources[reference_name] = _from_latents(encoded["center"]["p_reference"])
    out = {
        "equal_compute_extension": False,
        "trained_steps": None,
        "question": "Can a frozen coordinate-conditioned readout recover the 10-bin log-mel profile from Perceiver latents?",
        "coordinate": "fixed Fourier features of the bin index, then a parameter-free attention pool over the 32 latents",
        "shared_linear_params": D_MODEL + 1,
        "per_bin_linear_params": PROFILE_BINS * D_MODEL + PROFILE_BINS,
        "random_init": "LoopEar-P encoder at seed 42 before any optimizer step",
        "time_explicit_control": "the same two heads on LoopEar-D time bins, which already index time",
        "models": {},
    }
    for name, tokens in sources.items():
        train_tokens = tokens[train_idx]
        val_tokens = tokens[val_idx]
        shared = _fit_shared(train_tokens, target_train, val_tokens, SEED)
        per_bin, per_params = _fit_per_bin(train_tokens, target_train, val_tokens, SEED)
        out["models"][name] = {
            "shared_linear": {"params": D_MODEL + 1, **mean_pearson(shared, target_val.numpy())},
            "per_bin_linear": {"params": per_params, **mean_pearson(per_bin, target_val.numpy())},
        }
    def _r(name: str, head: str) -> float | None:
        row = out["models"].get(name)
        if row is None:
            return None
        return row[head]["mean_r"]

    def _sub(left: str, right: str, head: str) -> float | None:
        a = _r(left, head)
        b = _r(right, head)
        if a is None or b is None:
            return None
        return float(a - b)

    out["reading"] = {
        "shared_budget_matches_the_time_bin_head": True,
        "per_bin_budget": PROFILE_BINS * D_MODEL + PROFILE_BINS,
        "evaluated_name": evaluated_name,
        "evaluated_minus_random_per_bin": _sub(evaluated_name, "random_init_perceiver", "per_bin_linear"),
        "evaluated_minus_random_shared": _sub(evaluated_name, "random_init_perceiver", "shared_linear"),
        "explicit_minus_evaluated_per_bin": _sub("loopear_d_time_bins", evaluated_name, "per_bin_linear"),
        "explicit_minus_evaluated_shared": _sub("loopear_d_time_bins", evaluated_name, "shared_linear"),
        "evaluated_minus_trained_perceiver_per_bin": _sub(evaluated_name, "trained_perceiver", "per_bin_linear"),
        "evaluated_minus_trained_perceiver_shared": _sub(evaluated_name, "trained_perceiver", "shared_linear"),
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d-global", default="results/loopear_d_cmp_global_encoder.safetensors")
    parser.add_argument("--d-k1", default="results/loopear_d_k1_encoder.safetensors")
    parser.add_argument("--p", default="results/loopear_p_encoder.safetensors")
    parser.add_argument("--skip-k1", action="store_true")
    parser.add_argument("--coordinate", action="store_true")
    parser.add_argument("--coordinate-name", default="trained_perceiver")
    parser.add_argument("--time-alpha", type=float, default=0.0)
    parser.add_argument("--p-reference", default="")
    parser.add_argument("--report", default="results/loopear_protocol.json")
    args = parser.parse_args()
    if torch.cuda.is_available():
        raise SystemExit("cuda is visible; refusing to start rather than risk GPU spend")
    torch.set_num_threads(4)
    os.chdir(ROOT)
    t0 = cpu_seconds()
    manifest = load_manifest(Path("data/manifests/fsd50k_loopear_d.json"))
    train_rows = items_for(manifest, "h5")
    val_rows = _official_val(manifest, VAL_CLIPS)
    for row in train_rows + val_rows:
        if row["split"] == "test":
            raise PermissionError("protocol tried to read the sealed FSD50K test split")
    missing = [
        int(row["freesound_id"])
        for row in val_rows
        if not audio_path(Path("data/raw/fsd50k"), row).exists()
    ]
    if missing:
        raise SystemExit(f"{len(missing)} validation clips are not on disk")
    config = load_frontend_config(Path("data/manifests/loopear_d_frontend.json"))
    mean = torch.tensor(config["mean"], dtype=torch.float32)
    std = torch.tensor(config["std"], dtype=torch.float32)
    helpers = _baselines()
    code, ckpt = helpers._ensure_efficientat()
    eat, eat_mel, eat_params = helpers._load_efficientat(code, ckpt)
    usad, _dtype = load_usad2_extractor("cpu")
    models = {
        "d_global": _load_d(Path(args.d_global)),
        "p": _load_p(Path(args.p), args.time_alpha),
        "usad": usad,
        "eat": eat,
        "eat_mel": eat_mel,
    }
    if not args.skip_k1:
        models["d_k1"] = _load_d(Path(args.d_k1))
    if args.coordinate:
        random_model = init_perceiver(SEED)
        random_model.eval()
        for param in random_model.parameters():
            param.requires_grad = False
        models["p_random"] = random_model.encoder
    if args.p_reference:
        models["p_reference"] = _load_p(Path(args.p_reference), 0.0)
    mel_bank = build_mel()
    rows = train_rows + val_rows
    print(f"encoding {len(train_rows)} development clips and {len(val_rows)} val clips", flush=True)
    encoded = _encode_fsd(rows, models, helpers, mean, std, mel_bank)
    names = tag_class_names(train_rows, val_rows)
    y_train = _multi_hot(train_rows, names)
    y_val = _multi_hot(val_rows, names)
    tag_features = {
        "loopear_d_global_cls": encoded["means"]["d_global_cls"],
        "loopear_d_global_patch_mean": encoded["means"]["d_global_patch"],
        "loopear_p_summary": encoded["means"]["p_summary"],
        "loopear_p_latents": encoded["means"]["p_latents"],
        "efficientat_mn04_as": encoded["means"]["efficientat"],
        "usad2_small": encoded["means"]["usad"],
        "dsp": encoded["means"]["dsp"],
    }
    if "d_k1_cls" in encoded["means"]:
        tag_features["loopear_d_k1_cls"] = encoded["means"]["d_k1_cls"]
        tag_features["loopear_d_k1_patch_mean"] = encoded["means"]["d_k1_patch"]
    tag = {}
    n_train = len(train_rows)
    for name, matrix in tag_features.items():
        tag[name] = _probe_pair(
            torch.from_numpy(matrix[:n_train]),
            torch.from_numpy(matrix[n_train:]),
            y_train,
            y_val,
        )
        print(
            name,
            tag[name]["probe"]["val"]["macro_ap"],
            "shuffled",
            tag[name]["shuffled_label_probe"]["val"]["macro_ap"],
            flush=True,
        )
    fidelity = _fidelity_block(encoded, n_train)
    print("fidelity", {key: value.get("mean_r") for key, value in fidelity.items() if isinstance(value, dict)}, flush=True)
    few = _few_block(train_rows, val_rows, tag_features)
    print("few-example", few.get("status"), few.get("n_classes"), flush=True)
    coordinate = _coordinate_block(encoded, n_train, args.coordinate_name) if args.coordinate else None
    if coordinate is not None:
        print("coordinate", {name: row["per_bin_linear"]["mean_r"] for name, row in coordinate["models"].items()}, flush=True)
    full_val = [row for row in manifest["items"] if row["split"] == "val"]
    starss = _starss(models, helpers, mean, std, mel_bank)
    fmap_time = int(encoded["native"]["fmap_time"] or 0)
    mel_frames = int(encoded["native"]["mel_frames"] or 0)
    stride_s = None
    if fmap_time and mel_frames:
        stride_s = (mel_frames * 320 / 32000.0) / fmap_time
    payload = {
        "task": "loopear_repaired_protocol",
        "gpu_usd": 0.0,
        "manifest_rewritten": False,
        "fsd50k_test_audio_read": False,
        "sealed_realdesed_test_read": False,
        "h20_plan_file_rewritten": False,
        "h20_5000_step_runs_finished": Path(args.d_global).name == "loopear_d_h20_global_5000_encoder.safetensors",
        "equal_compute_extension": False,
        "encoders": {
            "d_global": args.d_global,
            "d_k1": None if args.skip_k1 else args.d_k1,
            "p": args.p,
            "p_time_alpha": float(args.time_alpha),
            "p_reference": args.p_reference or None,
            "coordinate_random_init": bool(args.coordinate),
            "coordinate_name": args.coordinate_name if args.coordinate else None,
        },
        "development": {
            "clips": len(train_rows),
            "hours": 5.0007959164265126,
            "role": "probe training and few-example support",
        },
        "validation": {
            "clips": len(val_rows),
            "selection": "same rank_eligible order as the 160-clip probe, extended to 640 official val clips",
            "first_160_are_the_previous_probe": True,
        },
        "tag_probe": {
            "classes": names,
            "n_classes": len(names),
            "n_train": len(train_rows),
            "n_val": len(val_rows),
            "windows_per_clip": TAG_WINDOWS,
            "n_windows": encoded["n_windows"],
            "n_short_windows": encoded["n_short_windows"],
            "n_empty_clips": encoded["n_empty_clips"],
            "val_prevalence_mean": float(y_val.mean()) if y_val.numel() else None,
            "readout": "fixed Gaussian project to 96, Linear, AdamW lr 1e-2, 250 steps, seed 42",
            "models": tag,
        },
        "acoustic_fidelity": fidelity,
        "coordinate_readout": coordinate,
        "few_example": few,
        "unseen_primary_six_clip_rule": {
            "scored": False,
            "on_this_validation_probe": _unseen_primary(train_rows, val_rows),
            "on_all_official_val_metadata": _unseen_primary(train_rows, full_val),
            "reason": "This protocol does not require 6 clips of a primary that is absent from the 5-hour subset.",
        },
        "starss23": starss,
        "representation_notes": {
            "efficientat_params": eat_params,
            "efficientat_native_stride_s": stride_s,
            "efficientat_mel_frames_full_crop": mel_frames,
            "efficientat_fmap_time_full_crop": fmap_time,
            "usad_frames_full_crop": encoded["native"]["usad_frames"],
            "old_160_clip_probe_is_not_usad_parity": True,
            "old_single_seed_probe_is_not_a_local_learning_verdict": True,
        },
        "cpu_hours": hours(cpu_seconds() - t0),
        "peak_rss_gb": peak_rss_gb(),
    }
    report_path = Path(args.report)
    if report_path.exists():
        raise SystemExit(f"refusing to overwrite {report_path}")
    path = write_result_json(report_path, payload)
    print(json.dumps({"wrote": str(path), "gpu_usd": 0.0, "cpu_hours": payload["cpu_hours"]}))


if __name__ == "__main__":
    main()
