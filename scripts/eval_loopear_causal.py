import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from loopear.data import STARSS23_CLASSES, rasterize_starss_frames_50hz
from loopear.d.causal import CausalLoopEarDEncoder, frequency_pool
from loopear.d.causal_probe import score_event_order
from loopear.d.diagnostics import cpu_seconds, hours, peak_rss_gb
from loopear.d.frontend import build_mel, load_frontend_config, waveform_logmel
from loopear.d.fsd import audio_path, extract_audio, items_for, load_manifest
from loopear.d.objective import broadcast_time_valid, patch_valid_from_frames
from loopear.d.probe import fixed_project, macro_average_precision
from loopear.d.protocol import clip_window_starts, event_window_starts, starss_keep, tag_class_names
from loopear.d.spec import CROP_SAMPLES, D_MODEL, GRID_T, HOP_SAMPLES, N_FRAMES, SR
from loopear.encoder import load_mono_audio
from loopear.harness import SpendLedger, write_result_json


ROOT = Path(__file__).resolve().parent.parent
LOCKED_TIME_ALPHA = 0.0027896127554686775
VAL_CLIPS = 640
TAG_WINDOWS = 4
ENCODE_BATCH = 16
SEED = 42
EXPECTED_WINDOWS = 6676
EXPECTED_SHORT = 329
EXPECTED_STARSS_TRAIN = 7200
EXPECTED_STARSS_VAL = 4200


def _protocol():
    path = Path(__file__).with_name("eval_loopear_protocol.py")
    spec = importlib.util.spec_from_file_location("eval_loopear_protocol_causal", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_causal(path: Path) -> CausalLoopEarDEncoder:
    model = CausalLoopEarDEncoder()
    model.load_state_dict(load_file(str(path)))
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def _visible(valid: torch.Tensor) -> torch.Tensor:
    return broadcast_time_valid(patch_valid_from_frames(valid))


def _patch_mean(grid: torch.Tensor, visible: torch.Tensor) -> torch.Tensor:
    means = []
    for index in range(int(grid.shape[0])):
        kept = grid[index][visible[index]]
        if int(kept.shape[0]) == 0:
            means.append(grid.new_zeros(D_MODEL))
        else:
            means.append(kept.mean(dim=0))
    return torch.stack(means, dim=0)


def _ensure_audio(rows: list[dict[str, Any]]) -> None:
    missing = [row for row in rows if not audio_path(Path("data/raw/fsd50k"), row).exists()]
    if not missing:
        return
    for row in missing:
        if row["split"] == "test":
            raise PermissionError("refusing to materialize FSD50K test audio")
    print(json.dumps({"extracting_clips": len(missing), "split": missing[0]["split"]}), flush=True)
    extract_audio(missing, Path("data/raw/fsd50k"))


def _encode_fsd(rows, models, mean, std, mel_bank, proto) -> dict[str, Any]:
    n = len(rows)
    names = {
        "dc_summary": D_MODEL,
        "dcp_summary": D_MODEL,
        "d_cls": D_MODEL,
        "d_patch": D_MODEL,
        "p_summary": D_MODEL,
        "p_latents": 32 * D_MODEL,
    }
    sums = {key: np.zeros((n, dim), dtype=np.float64) for key, dim in names.items()}
    counts = np.zeros(n, dtype=np.float64)
    pending: list[dict[str, Any]] = []
    n_windows = 0
    n_short = 0

    def _flush() -> None:
        nonlocal pending
        if not pending:
            return
        mel = torch.stack([item["norm"] for item in pending])
        valid = torch.stack([item["valid"] for item in pending])
        visible = _visible(valid)
        ok = visible.flatten(1).any(dim=1)
        if not bool(ok.all()):
            kept = [pending[index] for index, flag in enumerate(ok.tolist()) if flag]
            pending = []
            if not kept:
                return
            pending.extend(kept)
            _flush()
            return
        with torch.inference_mode():
            _dc_grid, dc_summary = models["dc"](mel, visible)
            _dcp_grid, dcp_summary = models["dcp"](mel, visible)
            d_grid, d_cls = models["dense"].encoder(mel, visible)
            latents, p_summary = models["plt"](mel, visible)
            packed = {
                "dc_summary": dc_summary,
                "dcp_summary": dcp_summary,
                "d_cls": d_cls,
                "d_patch": _patch_mean(d_grid, visible),
                "p_summary": p_summary,
                "p_latents": latents.reshape(latents.shape[0], -1),
            }
        for local, item in enumerate(pending):
            owner = int(item["owner"])
            counts[owner] += 1.0
            for key, value in packed.items():
                sums[key][owner] += value[local].detach().float().cpu().numpy()
        pending = []

    for row_index, row in enumerate(rows):
        if row["split"] == "test":
            raise PermissionError("protocol tried to read the sealed FSD50K test split")
        wav = proto._read_flac(audio_path(Path("data/raw/fsd50k"), row))
        spec = waveform_logmel(torch.from_numpy(wav), mel_bank)
        starts, _center = clip_window_starts(int(spec.shape[0]), N_FRAMES, TAG_WINDOWS)
        for start in starts:
            norm, valid = proto._window(spec, start, mean, std)
            if not bool(valid.any()):
                continue
            left = int(start) * HOP_SAMPLES
            right = min(int(wav.shape[0]), left + CROP_SAMPLES)
            if right <= left:
                continue
            n_windows += 1
            if right - left < CROP_SAMPLES:
                n_short += 1
            pending.append({"owner": row_index, "norm": norm, "valid": valid})
            if len(pending) >= ENCODE_BATCH:
                _flush()
        if row_index % 128 == 127:
            print(f"fsd clips {row_index + 1}/{n}", flush=True)
    _flush()
    denom = np.clip(counts, 1.0, None)
    means = {key: (value / denom[:, None]).astype(np.float32) for key, value in sums.items()}
    return {
        "means": means,
        "n_windows": n_windows,
        "n_short_windows": n_short,
        "n_empty_clips": int((counts == 0).sum()),
    }


def _repeat(summary: torch.Tensor) -> torch.Tensor:
    return summary[0].detach().float().cpu().unsqueeze(0).expand(GRID_T, D_MODEL).contiguous()


def _starss(models, mean, std, mel_bank, proto) -> dict[str, Any]:
    manifest = load_manifest(Path("data/manifests/starss23.json"))
    clips = []
    for row in manifest["items"]:
        if row["split"] not in ("train", "val"):
            raise PermissionError(f"STARSS manifest row {row['id']} is {row['split']}")
        if "realdesed" in str(row["id"]).lower() or "fsd50k" in str(row["id"]).lower():
            raise PermissionError(row["id"])
        wav_src = hf_hub_download(
            "labhamlet/STARSS23", "original_audios/" + row["id"] + ".wav", repo_type="dataset"
        )
        if "realdesed" in wav_src.lower() or "fsd50k" in wav_src.lower():
            raise PermissionError(wav_src)
        csv_src = proto._label_csv(row["id"])
        wav = load_mono_audio(wav_src, target_sr=SR, channel=0)
        spec = waveform_logmel(torch.from_numpy(wav), mel_bank)
        events = proto._parse_events(Path(csv_src).read_text(encoding="utf-8"))
        duration = float(wav.shape[0]) / float(SR)
        starts = event_window_starts(duration, [(idx * 0.1, cls) for idx, cls in events], max_windows=6)
        windows = []
        for start_s in starts:
            sample = int(round(start_s * SR))
            sample = min(sample, max(int(wav.shape[0]) - 1, 0))
            if sample + CROP_SAMPLES > int(wav.shape[0]):
                sample = max(int(wav.shape[0]) - CROP_SAMPLES, 0)
            frame = sample // HOP_SAMPLES
            norm, valid = proto._window(spec, frame, mean, std)
            if not bool(valid.any()):
                continue
            visible = _visible(valid.unsqueeze(0))
            if not bool(visible.flatten(1).any()):
                continue
            with torch.inference_mode():
                dc_grid, dc_summary = models["dc"](norm.unsqueeze(0), visible)
                dcp_grid, dcp_summary = models["dcp"](norm.unsqueeze(0), visible)
                d_grid, d_cls = models["dense"].encoder(norm.unsqueeze(0), visible)
                _latents, p_summary = models["plt"](norm.unsqueeze(0), visible)
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
                    "dc_time": frequency_pool(dc_grid)[0].detach().float().cpu(),
                    "dc_summary": _repeat(dc_summary),
                    "dcp_time": frequency_pool(dcp_grid)[0].detach().float().cpu(),
                    "dcp_summary": _repeat(dcp_summary),
                    "d_time": d_grid[0].mean(dim=1).detach().float().cpu(),
                    "d_cls": _repeat(d_cls),
                    "p_summary": _repeat(p_summary),
                    "y": labels,
                }
            )
        clips.append({"id": row["id"], "split": row["split"], "windows": windows})
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

    keys = ["dc_time", "dc_summary", "dcp_time", "dcp_summary", "d_time", "d_cls", "p_summary"]
    y_train = None
    y_val = None
    files_train = None
    files_val = None
    window_labels = None
    features = {}
    for key in keys:
        x_train, y_here_train, files_tr, _train_windows = _collect(train_clips, key)
        x_val, y_here, files_va, val_windows = _collect(val_clips, key)
        if y_train is None:
            y_train = y_here_train
            y_val = y_here
            files_train = files_tr
            files_val = files_va
            window_labels = val_windows
        features[key] = (x_train, x_val)
    keep, detail = starss_keep(y_train, y_val, files_train, files_val, min_files=2, min_frames=40)
    names = [STARSS23_CLASSES[row["class_index"]] for row in detail]
    for row, name in zip(detail, names):
        row["class"] = name
    y_tr = torch.from_numpy(y_train[:, keep])
    y_va = torch.from_numpy(y_val[:, keep])
    label_windows = [window[:, keep] for window in window_labels]
    per_model = {}
    for key, (x_train, x_val) in features.items():
        if int(keep.sum()) == 0:
            per_model[key] = {"status": "null", "reason": "no class met the support rule"}
            continue
        x_tr = fixed_project(x_train, D_MODEL, SEED)
        x_va = fixed_project(x_val, D_MODEL, SEED)
        real = proto._scores(x_tr, y_tr, x_va, SEED)
        perm = torch.Generator().manual_seed(SEED + 7)
        order = torch.randperm(y_tr.shape[0], generator=perm)
        shuffled = proto._scores(x_tr, y_tr[order], x_va, SEED)
        real_ap = macro_average_precision(y_va.numpy(), real)
        shuffled_ap = macro_average_precision(y_va.numpy(), shuffled)
        score_windows = []
        cursor = 0
        for window in window_labels:
            n_frames = window.shape[0]
            score_windows.append(real[cursor : cursor + n_frames])
            cursor += n_frames
        per_model[key] = {
            "status": "measured",
            "val": real_ap,
            "shuffled_label_probe": shuffled_ap,
            "onset_within_windows": proto._onset_within_windows(label_windows, score_windows),
            "probe_params": int(D_MODEL * int(keep.sum()) + int(keep.sum())),
        }
    return {
        "train_files": [clip["id"] for clip in train_clips],
        "val_files": [clip["id"] for clip in val_clips],
        "n_train_frames": int(y_train.shape[0]),
        "n_val_frames": int(y_val.shape[0]),
        "n_classes_scored": int(keep.sum()),
        "classes": detail,
        "models": per_model,
        "frame_axis": "50 patch steps of 40 ms; a clip summary is repeated on every step",
    }


def _tag_block(encoded, train_rows, val_rows, proto) -> tuple[dict[str, Any], dict[str, Any]]:
    names = tag_class_names(train_rows, val_rows)
    y_train = proto._multi_hot(train_rows, names)
    y_val = proto._multi_hot(val_rows, names)
    n_train = len(train_rows)
    tag = {}
    matrices = {}
    for key, matrix in encoded["means"].items():
        matrices[key] = matrix
        tag[key] = proto._probe_pair(
            torch.from_numpy(matrix[:n_train]),
            torch.from_numpy(matrix[n_train:]),
            y_train,
            y_val,
        )
        print(key, tag[key]["probe"]["val"]["macro_ap"], flush=True)
    few = proto._few_block(train_rows, val_rows, matrices)
    print("few-example", few.get("status"), few.get("n_classes"), flush=True)
    return tag, few


def _saved_numbers() -> dict[str, Any]:
    protocol = json.loads(Path("results/loopear_plt_h20_protocol.json").read_text(encoding="utf-8"))
    event = json.loads(Path("results/loopear_pltc_event_order.json").read_text(encoding="utf-8"))
    tags = protocol["tag_probe"]["models"]
    few = protocol["few_example"]["models"]
    star = protocol["starss23"]["models"]

    def _pack(tag_name: str | None, few_name: str | None, star_name: str | None) -> dict[str, Any]:
        onset = None if star_name is None else star[star_name]["onset_within_windows"]
        return {
            "tag_ap": None if tag_name is None else tags[tag_name]["probe"]["val"]["macro_ap"],
            "four_example": None if few_name is None else few[few_name]["macro_ap"],
            "starss_frame_ap": None if star_name is None else star[star_name]["val"]["macro_ap"],
            "onset_lift": None if onset is None else onset["mean_lift"],
            "onset_above_background": None if onset is None else onset["n_above_background"],
            "n_onsets": None if onset is None else onset["n_onsets"],
        }

    plt = event["models"]["plt"]
    return {
        "protocol": "results/loopear_plt_h20_protocol.json",
        "event_order": "results/loopear_pltc_event_order.json",
        "dense_class_token": _pack("loopear_d_global_cls", "loopear_d_global_cls", "d_global_cls"),
        "dense_patch_mean": _pack("loopear_d_global_patch_mean", "loopear_d_global_patch_mean", None),
        "dense_time": _pack(None, None, "d_global_time"),
        "plt_summary": _pack("loopear_p_summary", "loopear_p_summary", "p_summary"),
        "plt_latents": _pack("loopear_p_latents", "loopear_p_latents", None),
        "plt_event_order": {
            "n_pairs": plt["n_pairs"],
            "n_correct": plt["n_correct"],
            "accuracy": plt["accuracy"],
            "reversed_accuracy": plt["reversed_audio"]["accuracy"],
            "reversed_correct": plt["reversed_audio"]["n_correct"],
            "chance": 0.5,
            "recomputed": False,
            "reason": "the saved EMA checkpoint is not in this checkout; the student safetensors do not include it",
        },
    }


def _measured_pack(tag, few, star, tag_key, few_key, star_key) -> dict[str, Any]:
    onset = None if star_key is None else star["models"][star_key]["onset_within_windows"]
    return {
        "tag_ap": None if tag_key is None else tag[tag_key]["probe"]["val"]["macro_ap"],
        "four_example": None if few_key is None else few["models"][few_key]["macro_ap"],
        "starss_frame_ap": None if star_key is None else star["models"][star_key]["val"]["macro_ap"],
        "onset_lift": None if onset is None else onset["mean_lift"],
        "onset_above_background": None if onset is None else onset["n_above_background"],
        "n_onsets": None if onset is None else onset["n_onsets"],
        "tag_shuffled": None if tag_key is None else tag[tag_key]["shuffled_label_probe"]["val"]["macro_ap"],
        "four_example_shuffled": None if few_key is None else few["models"][few_key]["shuffled_support_macro_ap"],
        "starss_shuffled": None if star_key is None else star["models"][star_key]["shuffled_label_probe"]["macro_ap"],
    }


def _dipco() -> dict[str, Any]:
    manifest = load_manifest(Path("data/manifests/dipco.json"))
    present = [row["id"] for row in manifest["items"] if Path(row["audio_path"]).exists()]
    return {
        "manifest": "data/manifests/dipco.json",
        "manifest_committed": True,
        "n_items": len(manifest["items"]),
        "n_audio_present": len(present),
        "scored": False,
        "reason": "The split manifest is committed. The waveforms it names are not in this checkout, so overlap was not scored.",
    }


def _close(left: float | None, right: float | None) -> bool | None:
    if left is None or right is None:
        return None
    return abs(float(left) - float(right)) <= 1e-4


def _encode_summary(rows, encoder, mean, std, mel_bank, proto) -> dict[str, Any]:
    n = len(rows)
    sums = np.zeros((n, D_MODEL), dtype=np.float64)
    counts = np.zeros(n, dtype=np.float64)
    pending: list[dict[str, Any]] = []
    n_windows = 0
    n_short = 0

    def _flush() -> None:
        nonlocal pending
        if not pending:
            return
        mel = torch.stack([item["norm"] for item in pending])
        valid = torch.stack([item["valid"] for item in pending])
        visible = _visible(valid)
        ok = visible.flatten(1).any(dim=1)
        if not bool(ok.all()):
            kept = [pending[index] for index, flag in enumerate(ok.tolist()) if flag]
            pending = []
            if not kept:
                return
            pending.extend(kept)
            _flush()
            return
        with torch.inference_mode():
            _grid, summary = encoder(mel, visible)
        for local, item in enumerate(pending):
            owner = int(item["owner"])
            counts[owner] += 1.0
            sums[owner] += summary[local].detach().float().cpu().numpy()
        pending = []

    for row_index, row in enumerate(rows):
        if row["split"] == "test":
            raise PermissionError("protocol tried to read the sealed FSD50K test split")
        wav = proto._read_flac(audio_path(Path("data/raw/fsd50k"), row))
        spec = waveform_logmel(torch.from_numpy(wav), mel_bank)
        starts, _center = clip_window_starts(int(spec.shape[0]), N_FRAMES, TAG_WINDOWS)
        for start in starts:
            norm, valid = proto._window(spec, start, mean, std)
            if not bool(valid.any()):
                continue
            left = int(start) * HOP_SAMPLES
            right = min(int(wav.shape[0]), left + CROP_SAMPLES)
            if right <= left:
                continue
            n_windows += 1
            if right - left < CROP_SAMPLES:
                n_short += 1
            pending.append({"owner": row_index, "norm": norm, "valid": valid})
            if len(pending) >= ENCODE_BATCH:
                _flush()
        if row_index % 128 == 127:
            print(f"fsd clips {row_index + 1}/{n}", flush=True)
    _flush()
    denom = np.clip(counts, 1.0, None)
    means = {"summary": (sums / denom[:, None]).astype(np.float32)}
    return {
        "means": means,
        "n_windows": n_windows,
        "n_short_windows": n_short,
        "n_empty_clips": int((counts == 0).sum()),
    }


def _starss_time(encoder, mean, std, mel_bank, proto) -> dict[str, Any]:
    manifest = load_manifest(Path("data/manifests/starss23.json"))
    clips = []
    for row in manifest["items"]:
        if row["split"] not in ("train", "val"):
            raise PermissionError(f"STARSS manifest row {row['id']} is {row['split']}")
        if "realdesed" in str(row["id"]).lower() or "fsd50k" in str(row["id"]).lower():
            raise PermissionError(row["id"])
        wav_src = hf_hub_download(
            "labhamlet/STARSS23", "original_audios/" + row["id"] + ".wav", repo_type="dataset"
        )
        if "realdesed" in wav_src.lower() or "fsd50k" in wav_src.lower():
            raise PermissionError(wav_src)
        csv_src = proto._label_csv(row["id"])
        wav = load_mono_audio(wav_src, target_sr=SR, channel=0)
        spec = waveform_logmel(torch.from_numpy(wav), mel_bank)
        events = proto._parse_events(Path(csv_src).read_text(encoding="utf-8"))
        duration = float(wav.shape[0]) / float(SR)
        starts = event_window_starts(duration, [(idx * 0.1, cls) for idx, cls in events], max_windows=6)
        windows = []
        for start_s in starts:
            sample = int(round(start_s * SR))
            sample = min(sample, max(int(wav.shape[0]) - 1, 0))
            if sample + CROP_SAMPLES > int(wav.shape[0]):
                sample = max(int(wav.shape[0]) - CROP_SAMPLES, 0)
            frame = sample // HOP_SAMPLES
            norm, valid = proto._window(spec, frame, mean, std)
            if not bool(valid.any()):
                continue
            visible = _visible(valid.unsqueeze(0))
            if not bool(visible.flatten(1).any()):
                continue
            with torch.inference_mode():
                grid, _summary = encoder(norm.unsqueeze(0), visible)
            labels = rasterize_starss_frames_50hz(
                events,
                float(sample) / float(SR),
                GRID_T,
                len(STARSS23_CLASSES),
                hop_e=0.04,
                label_hop_s=0.1,
            )
            windows.append({"time": frequency_pool(grid)[0].detach().float().cpu(), "y": labels})
        clips.append({"id": row["id"], "split": row["split"], "windows": windows})
        print(f"starss {row['id']} windows {len(windows)}", flush=True)
    train_clips = [clip for clip in clips if clip["split"] == "train"]
    val_clips = [clip for clip in clips if clip["split"] == "val"]

    def _collect(group):
        xs = []
        ys = []
        files = []
        y_windows = []
        for file_index, clip in enumerate(group):
            for window in clip["windows"]:
                xs.append(window["time"])
                ys.append(window["y"])
                files.append(np.full(window["y"].shape[0], file_index, dtype=np.int64))
                y_windows.append(window["y"])
        if not xs:
            raise RuntimeError("STARSS split produced no windows")
        return torch.cat(xs, dim=0), np.concatenate(ys, axis=0), np.concatenate(files, axis=0), y_windows

    x_train, y_train, files_train, _train_windows = _collect(train_clips)
    x_val, y_val, files_val, window_labels = _collect(val_clips)
    keep, detail = starss_keep(y_train, y_val, files_train, files_val, min_files=2, min_frames=40)
    names = [STARSS23_CLASSES[row["class_index"]] for row in detail]
    for row, name in zip(detail, names):
        row["class"] = name
    if int(keep.sum()) == 0:
        scored = {"status": "null", "reason": "no class met the support rule"}
    else:
        y_tr = torch.from_numpy(y_train[:, keep])
        y_va = torch.from_numpy(y_val[:, keep])
        label_windows = [window[:, keep] for window in window_labels]
        x_tr = fixed_project(x_train, D_MODEL, SEED)
        x_va = fixed_project(x_val, D_MODEL, SEED)
        real = proto._scores(x_tr, y_tr, x_va, SEED)
        perm = torch.Generator().manual_seed(SEED + 7)
        order = torch.randperm(y_tr.shape[0], generator=perm)
        shuffled = proto._scores(x_tr, y_tr[order], x_va, SEED)
        real_ap = macro_average_precision(y_va.numpy(), real)
        shuffled_ap = macro_average_precision(y_va.numpy(), shuffled)
        score_windows = []
        cursor = 0
        for window in window_labels:
            n_frames = window.shape[0]
            score_windows.append(real[cursor : cursor + n_frames])
            cursor += n_frames
        scored = {
            "status": "measured",
            "val": real_ap,
            "shuffled_label_probe": shuffled_ap,
            "onset_within_windows": proto._onset_within_windows(label_windows, score_windows),
            "probe_params": int(D_MODEL * int(keep.sum()) + int(keep.sum())),
        }
    return {
        "train_files": [clip["id"] for clip in train_clips],
        "val_files": [clip["id"] for clip in val_clips],
        "n_train_frames": int(y_train.shape[0]),
        "n_val_frames": int(y_val.shape[0]),
        "n_classes_scored": int(keep.sum()),
        "classes": detail,
        "models": {"time": scored},
        "frame_axis": "50 patch steps of 40 ms from the frequency-pooled causal grid",
    }


def score_causal_control(encoder_path: Path, report_path: Path) -> dict[str, Any]:
    if torch.cuda.is_available():
        raise SystemExit("cuda is visible; refusing to start rather than risk GPU spend")
    if report_path.exists():
        raise SystemExit(f"refusing to overwrite {report_path}")
    if not encoder_path.exists():
        raise SystemExit(f"missing weights {encoder_path}")
    torch.set_num_threads(4)
    os.chdir(ROOT)
    proto = _protocol()
    manifest = load_manifest(Path("data/manifests/fsd50k_loopear_d.json"))
    train_rows = items_for(manifest, "h5")
    val_rows = proto._official_val(manifest, VAL_CLIPS)
    for row in train_rows + val_rows:
        if row["split"] == "test":
            raise PermissionError("protocol tried to read the sealed FSD50K test split")
    _ensure_audio(train_rows + val_rows)
    config = load_frontend_config(Path("data/manifests/loopear_d_frontend.json"))
    mean = torch.tensor(config["mean"], dtype=torch.float32)
    std = torch.tensor(config["std"], dtype=torch.float32)
    mel_bank = build_mel()
    encoder = _load_causal(encoder_path)
    rows = train_rows + val_rows
    print(f"encoding {len(train_rows)} development clips and {len(val_rows)} val clips", flush=True)
    encoded = _encode_summary(rows, encoder, mean, std, mel_bank, proto)
    if encoded["n_windows"] != EXPECTED_WINDOWS or encoded["n_short_windows"] != EXPECTED_SHORT or encoded["n_empty_clips"] != 0:
        raise SystemExit(
            f"FSD windowing does not match the saved protocol: {encoded['n_windows']} windows, "
            f"{encoded['n_short_windows']} short, {encoded['n_empty_clips']} empty"
        )
    tag, few = _tag_block(encoded, train_rows, val_rows, proto)
    star = _starss_time(encoder, mean, std, mel_bank, proto)
    if star["n_train_frames"] != EXPECTED_STARSS_TRAIN or star["n_val_frames"] != EXPECTED_STARSS_VAL:
        raise SystemExit(
            f"STARSS frames {star['n_train_frames']}/{star['n_val_frames']} do not match 7200/4200"
        )
    onset = star["models"]["time"]["onset_within_windows"]
    payload = {
        "encoder": str(encoder_path),
        "representation": "causal encoder clip summary for tags; frequency-pooled time grid for STARSS",
        "tag_ap": tag["summary"]["probe"]["val"]["macro_ap"],
        "tag_shuffled": tag["summary"]["shuffled_label_probe"]["val"]["macro_ap"],
        "four_example_ap": few["models"]["summary"]["macro_ap"],
        "four_example_shuffled": few["models"]["summary"]["shuffled_support_macro_ap"],
        "four_example_classes": few.get("n_classes"),
        "starss_frame_ap": star["models"]["time"]["val"]["macro_ap"],
        "starss_shuffled": star["models"]["time"]["shuffled_label_probe"]["macro_ap"],
        "onset_lift": onset["mean_lift"],
        "onset_above_background": onset["n_above_background"],
        "n_onsets": onset["n_onsets"],
        "n_windows": encoded["n_windows"],
        "n_short_windows": encoded["n_short_windows"],
        "starss_train_frames": star["n_train_frames"],
        "starss_val_frames": star["n_val_frames"],
        "starss_classes": star["n_classes_scored"],
        "fsd50k_test_audio_read": False,
        "sealed_realdesed_test_read": False,
        "dipco_scored": False,
        "gpu_usd": 0.0,
    }
    write_result_json(report_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-encoder", default="")
    parser.add_argument("--control-report", default="")
    parser.add_argument("--dc", default="results/loopear_dc_h20_5000_encoder.safetensors")
    parser.add_argument("--dcp", default="results/loopear_dcp_h20_5000_encoder.safetensors")
    parser.add_argument("--dc-teacher", default="results/loopear_dc_h20_5000_teacher.safetensors")
    parser.add_argument("--dcp-teacher", default="results/loopear_dcp_h20_5000_teacher.safetensors")
    parser.add_argument("--dense", default="results/loopear_d_h20_global_5000_encoder.safetensors")
    parser.add_argument("--plt", default="results/loopear_plt_h20_global_5000_encoder.safetensors")
    parser.add_argument("--report", default="results/loopear_causal_h20_protocol.json")
    args = parser.parse_args()
    if args.control_encoder:
        if not args.control_report:
            raise SystemExit("control scoring needs --control-report")
        payload = score_causal_control(Path(args.control_encoder), Path(args.control_report))
        print(json.dumps({
            "tag_ap": payload["tag_ap"],
            "four_example_ap": payload["four_example_ap"],
            "starss_frame_ap": payload["starss_frame_ap"],
            "onset_lift": payload["onset_lift"],
        }), flush=True)
        return
    if torch.cuda.is_available():
        raise SystemExit("cuda is visible; refusing to start rather than risk GPU spend")
    report = Path(args.report)
    if report.exists():
        raise SystemExit(f"refusing to overwrite {report}")
    for path in (args.dc, args.dcp, args.dc_teacher, args.dcp_teacher, args.dense, args.plt):
        if not Path(path).exists():
            raise SystemExit(f"missing weights {path}")
    torch.set_num_threads(4)
    os.chdir(ROOT)
    ledger = SpendLedger(Path("ledger/loopear_causal_spend.jsonl"), cap_usd=0.0)
    ledger.record("loopear_causal_eval", "cpu", 0.0, "no paid GPU")
    cpu0 = cpu_seconds()
    proto = _protocol()
    manifest = load_manifest(Path("data/manifests/fsd50k_loopear_d.json"))
    train_rows = items_for(manifest, "h5")
    val_rows = proto._official_val(manifest, VAL_CLIPS)
    _ensure_audio(train_rows + val_rows)
    config = load_frontend_config(Path("data/manifests/loopear_d_frontend.json"))
    mean = torch.tensor(config["mean"], dtype=torch.float32)
    std = torch.tensor(config["std"], dtype=torch.float32)
    mel_bank = build_mel()
    models = {
        "dc": _load_causal(Path(args.dc)),
        "dcp": _load_causal(Path(args.dcp)),
        "dense": proto._load_d(Path(args.dense)),
        "plt": proto._load_p(Path(args.plt), LOCKED_TIME_ALPHA),
    }
    if float(models["plt"].time_alpha) != LOCKED_TIME_ALPHA:
        raise AssertionError("P-LT time alpha was not applied")
    rows = train_rows + val_rows
    print(f"encoding {len(train_rows)} development clips and {len(val_rows)} val clips", flush=True)
    encoded = _encode_fsd(rows, models, mean, std, mel_bank, proto)
    print(json.dumps({k: encoded[k] for k in ("n_windows", "n_short_windows", "n_empty_clips")}), flush=True)
    if encoded["n_windows"] != EXPECTED_WINDOWS or encoded["n_short_windows"] != EXPECTED_SHORT or encoded["n_empty_clips"] != 0:
        raise SystemExit(
            f"FSD windowing does not match the saved protocol: {encoded['n_windows']} windows, "
            f"{encoded['n_short_windows']} short, {encoded['n_empty_clips']} empty"
        )
    tag, few = _tag_block(encoded, train_rows, val_rows, proto)
    star = _starss(models, mean, std, mel_bank, proto)
    if star["n_train_frames"] != EXPECTED_STARSS_TRAIN or star["n_val_frames"] != EXPECTED_STARSS_VAL:
        raise SystemExit(
            f"STARSS frames {star['n_train_frames']}/{star['n_val_frames']} do not match 7200/4200"
        )
    print("event order D-C", flush=True)
    dc_order = score_event_order(_load_causal(Path(args.dc_teacher)))
    print("event order D-CP", flush=True)
    dcp_order = score_event_order(_load_causal(Path(args.dcp_teacher)))
    saved = _saved_numbers()
    measured = {
        "D-C": _measured_pack(tag, few, star, "dc_summary", "dc_summary", "dc_time"),
        "D-C-summary-repeated": _measured_pack(tag, few, star, None, None, "dc_summary"),
        "D-CP": _measured_pack(tag, few, star, "dcp_summary", "dcp_summary", "dcp_time"),
        "D-CP-summary-repeated": _measured_pack(tag, few, star, None, None, "dcp_summary"),
        "dense_class_token": _measured_pack(tag, few, star, "d_cls", "d_cls", "d_cls"),
        "dense_patch_mean": _measured_pack(tag, few, star, "d_patch", "d_patch", None),
        "dense_time": _measured_pack(tag, few, star, None, None, "d_time"),
        "plt_summary": _measured_pack(tag, few, star, "p_summary", "p_summary", "p_summary"),
        "plt_latents": _measured_pack(tag, few, star, "p_latents", "p_latents", None),
    }
    match = {
        "dense_class_token_tag_ap": _close(measured["dense_class_token"]["tag_ap"], saved["dense_class_token"]["tag_ap"]),
        "dense_time_frame_ap": _close(measured["dense_time"]["starss_frame_ap"], saved["dense_time"]["starss_frame_ap"]),
        "dense_onset_lift": _close(measured["dense_time"]["onset_lift"], saved["dense_time"]["onset_lift"]),
        "plt_summary_tag_ap": _close(measured["plt_summary"]["tag_ap"], saved["plt_summary"]["tag_ap"]),
        "plt_summary_frame_ap": _close(measured["plt_summary"]["starss_frame_ap"], saved["plt_summary"]["starss_frame_ap"]),
        "plt_four_example": _close(measured["plt_summary"]["four_example"], saved["plt_summary"]["four_example"]),
    }
    dipco = _dipco()
    payload = {
        "task": "loopear_causal_protocol",
        "gpu_usd": 0.0,
        "fsd50k_test_audio_read": False,
        "sealed_realdesed_test_read": False,
        "crema_d_read": False,
        "vocalset_read": False,
        "maestro_read": False,
        "development_clips": len(train_rows),
        "validation_clips": len(val_rows),
        "windows_per_clip": TAG_WINDOWS,
        "fsd_windows": encoded["n_windows"],
        "fsd_short_windows": encoded["n_short_windows"],
        "tag_classes": tag_class_names(train_rows, val_rows),
        "readout": "fixed Gaussian project to 96, Linear, AdamW lr 1e-2, 250 steps, seed 42",
        "causal_clip_embedding": "mean of the block-causal grid",
        "causal_starss_primary": "frequency-pooled time grid",
        "weights": {
            "dc": args.dc,
            "dcp": args.dcp,
            "dc_teacher": args.dc_teacher,
            "dcp_teacher": args.dcp_teacher,
            "dense": args.dense,
            "plt": args.plt,
            "plt_time_alpha": LOCKED_TIME_ALPHA,
        },
        "measured": measured,
        "event_order": {
            "chance": 0.5,
            "D-C": dc_order,
            "D-CP": dcp_order,
        },
        "saved": saved,
        "recompute_matches_saved": match,
        "dipco": dipco,
        "tag_probe": tag,
        "few_example": few,
        "starss23": star,
        "cpu_s": cpu_seconds() - cpu0,
        "cpu_hours": hours(cpu_seconds() - cpu0),
        "peak_rss_gb": peak_rss_gb(),
    }
    path = write_result_json(report, payload)
    print(json.dumps({
        "wrote": str(path),
        "dc_tag_ap": measured["D-C"]["tag_ap"],
        "dcp_tag_ap": measured["D-CP"]["tag_ap"],
        "dc_frame_ap": measured["D-C"]["starss_frame_ap"],
        "dcp_frame_ap": measured["D-CP"]["starss_frame_ap"],
        "dc_order": dc_order["accuracy"],
        "dcp_order": dcp_order["accuracy"],
        "match": match,
        "gpu_usd": 0.0,
    }), flush=True)


if __name__ == "__main__":
    main()
