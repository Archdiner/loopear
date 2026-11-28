import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from loopear.data import STARSS23_CLASSES, rasterize_starss_frames_50hz
from loopear.d.causal import CausalLoopEarD, causal_step_loss, chance_nce_combined, frequency_pool
from loopear.d.causal_probe import teacher_temporal_stats
from loopear.d.diagnostics import cpu_seconds, hours, peak_rss_gb
from loopear.d.frontend import build_mel, load_frontend_config, waveform_logmel
from loopear.d.fsd import MelCache, audio_path, extract_audio, items_for, load_manifest
from loopear.d.objective import broadcast_time_valid, patch_valid_from_frames
from loopear.d.probe import fixed_project, macro_average_precision
from loopear.d.protocol import clip_window_starts, event_window_starts, starss_keep, tag_class_names
from loopear.d.spec import CROP_SAMPLES, D_MODEL, GRID_T, HOP_SAMPLES, N_FRAMES, SEED, SR
from loopear.encoder import load_mono_audio
from loopear.harness import SpendLedger, write_result_json


ROOT = Path(__file__).resolve().parent.parent
VAL_CLIPS = 640
TAG_WINDOWS = 4
ENCODE_BATCH = 16
EXPECTED_WINDOWS = 6676
EXPECTED_SHORT = 329
EXPECTED_STARSS_TRAIN = 7200
EXPECTED_STARSS_VAL = 4200


def _protocol():
    path = Path(__file__).with_name("eval_loopear_protocol.py")
    spec = importlib.util.spec_from_file_location("eval_loopear_protocol_gated", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _visible(valid: torch.Tensor) -> torch.Tensor:
    return broadcast_time_valid(patch_valid_from_frames(valid))


def _load(path: Path) -> CausalLoopEarD:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model = CausalLoopEarD(separation=False)
    model.encoder.load_state_dict(blob["student_encoder"])
    model.teacher_encoder.load_state_dict(blob["teacher_encoder"])
    model.predictor.heads.load_state_dict(blob["horizons"])
    if "order_head" in blob:
        model.order_head.load_state_dict(blob["order_head"])
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def _ensure_audio(rows: list[dict[str, Any]]) -> None:
    missing = [row for row in rows if not audio_path(Path("data/raw/fsd50k"), row).exists()]
    if not missing:
        return
    for row in missing:
        if row["split"] == "test":
            raise PermissionError("refusing to materialize FSD50K test audio")
    extract_audio(missing, Path("data/raw/fsd50k"))


def _encode_fsd(rows, encoder, mean, std, mel_bank, proto) -> dict[str, Any]:
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
    return {
        "means": {"gated_summary": (sums / denom[:, None]).astype(np.float32)},
        "n_windows": n_windows,
        "n_short_windows": n_short,
        "n_empty_clips": int((counts == 0).sum()),
    }


def _repeat(summary: torch.Tensor) -> torch.Tensor:
    return summary[0].detach().float().cpu().unsqueeze(0).expand(GRID_T, D_MODEL).contiguous()


def _starss(encoder, mean, std, mel_bank, proto) -> dict[str, Any]:
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
        wav = load_mono_audio(wav_src, target_sr=SR, channel=0)
        spec = waveform_logmel(torch.from_numpy(wav), mel_bank)
        events = proto._parse_events(Path(proto._label_csv(row["id"])).read_text(encoding="utf-8"))
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
                grid, summary = encoder(norm.unsqueeze(0), visible)
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
                    "time": frequency_pool(grid)[0].detach().float().cpu(),
                    "summary": _repeat(summary),
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
        for file_index, clip in enumerate(group):
            for window in clip["windows"]:
                xs.append(window[key])
                ys.append(window["y"])
                files.append(np.full(window["y"].shape[0], file_index, dtype=np.int64))
        if not xs:
            raise RuntimeError("STARSS split produced no windows")
        return torch.cat(xs, dim=0), np.concatenate(ys, axis=0), np.concatenate(files, axis=0)

    x_train, y_train, files_train = _collect(train_clips, "time")
    x_val, y_val, files_val = _collect(val_clips, "time")
    summary_train_x, _summary_train_y, _summary_train_files = _collect(train_clips, "summary")
    summary_val_x, _summary_val_y, _summary_val_files = _collect(val_clips, "summary")
    keep, detail = starss_keep(y_train, y_val, files_train, files_val, min_files=2, min_frames=40)
    for row, name in zip(detail, [STARSS23_CLASSES[item["class_index"]] for item in detail]):
        row["class"] = name
    y_tr = torch.from_numpy(y_train[:, keep])
    y_va = torch.from_numpy(y_val[:, keep])
    label_windows = []
    cursor = 0
    for clip in val_clips:
        for window in clip["windows"]:
            n_frames = window["y"].shape[0]
            label_windows.append(window["y"][:, keep])
            cursor += n_frames
    if cursor != int(y_val.shape[0]):
        raise RuntimeError("STARSS validation windows do not cover the frame table")

    def _score(train_x, val_x):
        x_tr = fixed_project(train_x, D_MODEL, SEED)
        x_va = fixed_project(val_x, D_MODEL, SEED)
        real = proto._scores(x_tr, y_tr, x_va, SEED)
        perm = torch.Generator().manual_seed(SEED + 7)
        order = torch.randperm(y_tr.shape[0], generator=perm)
        shuffled = proto._scores(x_tr, y_tr[order], x_va, SEED)
        real_ap = macro_average_precision(y_va.numpy(), real)
        shuffled_ap = macro_average_precision(y_va.numpy(), shuffled)
        score_windows = []
        start = 0
        for window in label_windows:
            n_frames = window.shape[0]
            score_windows.append(real[start : start + n_frames])
            start += n_frames
        return {
            "status": "measured",
            "val": real_ap,
            "shuffled_label_probe": shuffled_ap,
            "onset_within_windows": proto._onset_within_windows(label_windows, score_windows),
        }

    return {
        "train_files": [clip["id"] for clip in train_clips],
        "val_files": [clip["id"] for clip in val_clips],
        "n_train_frames": int(y_train.shape[0]),
        "n_val_frames": int(y_val.shape[0]),
        "n_classes_scored": int(keep.sum()),
        "classes": detail,
        "models": {
            "gated_time": _score(x_train, x_val),
            "gated_summary": _score(summary_train_x, summary_val_x),
        },
        "frame_axis": "50 patch steps of 40 ms; a clip summary is repeated on every step",
    }


def _probe(model: CausalLoopEarD, cache: MelCache) -> dict[str, float]:
    order_gen = torch.Generator().manual_seed(SEED)
    crop_gen = torch.Generator().manual_seed(SEED + 17)
    order: list[int] = []
    cursor = 0
    real_pred = []
    real_perm = []
    silent = []
    batch = 64
    model.eval()
    with torch.no_grad():
        for _ in range(8):
            if cursor + batch > len(order):
                order = torch.randperm(len(cache), generator=order_gen).tolist()
                cursor = 0
            chosen = order[cursor : cursor + batch]
            cursor += batch
            crops = cache.crop_once(chosen, crop_gen)
            real = causal_step_loss(model, crops["mel"], crops["frame_valid"], score_gap=True)
            zeros = causal_step_loss(model, torch.zeros_like(crops["mel"]), crops["frame_valid"])
            real_pred.append(float(real["loss_pred"]))
            real_perm.append(float(real["loss_perm"]))
            silent.append(float(zeros["loss_pred"]))
    real_mean = float(sum(real_pred) / len(real_pred))
    perm_mean = float(sum(real_perm) / len(real_perm))
    silent_mean = float(sum(silent) / len(silent))
    chance = chance_nce_combined(batch)
    return {
        "real_loss": real_mean,
        "perm_loss": perm_mean,
        "gap_nats": perm_mean - real_mean,
        "silent_loss": silent_mean,
        "chance": chance,
        "silent_minus_chance": silent_mean - chance,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    if torch.cuda.is_available():
        raise SystemExit("cuda is visible; refusing to start rather than risk GPU spend")
    report = Path(args.report)
    if report.exists():
        raise SystemExit(f"refusing to overwrite {report}")
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise SystemExit(f"missing checkpoint {checkpoint}")
    torch.set_num_threads(4)
    os.chdir(ROOT)
    ledger = SpendLedger(Path("ledger/loopear_causal_spend.jsonl"), cap_usd=0.0)
    ledger.record("loopear_dgated_eval", "cpu", 0.0, "no paid GPU")
    cpu0 = cpu_seconds()
    model = _load(checkpoint)
    proto = _protocol()
    manifest = load_manifest(Path("data/manifests/fsd50k_loopear_d.json"))
    train_rows = items_for(manifest, "h20")
    for row in train_rows:
        if row["split"] == "test":
            raise PermissionError("training cache rows included FSD50K test")
    config = load_frontend_config(Path("data/manifests/loopear_d_frontend.json"))
    mean = torch.tensor(config["mean"], dtype=torch.float32)
    std = torch.tensor(config["std"], dtype=torch.float32)
    cache = MelCache(train_rows, Path("cache/loopear_d_mel"), mean, std)
    probe = _probe(model, cache)
    temporal = teacher_temporal_stats(model.teacher_encoder, cache, 8, 64, SEED)
    tag_rows = items_for(manifest, "h5")
    val_rows = proto._official_val(manifest, VAL_CLIPS)
    _ensure_audio(tag_rows + val_rows)
    mel_bank = build_mel()
    rows = tag_rows + val_rows
    print(f"encoding {len(tag_rows)} development clips and {len(val_rows)} val clips", flush=True)
    encoded = _encode_fsd(rows, model.encoder, mean, std, mel_bank, proto)
    if encoded["n_windows"] != EXPECTED_WINDOWS or encoded["n_short_windows"] != EXPECTED_SHORT or encoded["n_empty_clips"] != 0:
        raise SystemExit(
            f"FSD windowing does not match the saved protocol: {encoded['n_windows']} windows, "
            f"{encoded['n_short_windows']} short, {encoded['n_empty_clips']} empty"
        )
    names = tag_class_names(tag_rows, val_rows)
    y_train = proto._multi_hot(tag_rows, names)
    y_val = proto._multi_hot(val_rows, names)
    matrix = encoded["means"]["gated_summary"]
    tag = proto._probe_pair(
        torch.from_numpy(matrix[: len(tag_rows)]),
        torch.from_numpy(matrix[len(tag_rows) :]),
        y_train,
        y_val,
    )
    few = proto._few_block(tag_rows, val_rows, {"gated_summary": matrix})
    print("tag", tag["probe"]["val"]["macro_ap"], "few", few["models"]["gated_summary"]["macro_ap"], flush=True)
    star = _starss(model.encoder, mean, std, mel_bank, proto)
    if star["n_train_frames"] != EXPECTED_STARSS_TRAIN or star["n_val_frames"] != EXPECTED_STARSS_VAL:
        raise SystemExit(
            f"STARSS frames {star['n_train_frames']}/{star['n_val_frames']} do not match 7200/4200"
        )
    onset = star["models"]["gated_time"]["onset_within_windows"]
    payload = {
        "task": "loopear_dgated_protocol",
        "gpu_usd": 0.0,
        "checkpoint": str(checkpoint),
        "fsd50k_test_audio_read": False,
        "sealed_realdesed_test_read": False,
        "crema_d_read": False,
        "vocalset_read": False,
        "maestro_read": False,
        "development_clips": len(tag_rows),
        "validation_clips": len(val_rows),
        "fsd_windows": encoded["n_windows"],
        "fsd_short_windows": encoded["n_short_windows"],
        "tag_ap": tag["probe"]["val"]["macro_ap"],
        "tag_shuffled": tag["shuffled_label_probe"]["val"]["macro_ap"],
        "four_example_ap": few["models"]["gated_summary"]["macro_ap"],
        "four_example_shuffled": few["models"]["gated_summary"]["shuffled_support_macro_ap"],
        "starss_frame_ap": star["models"]["gated_time"]["val"]["macro_ap"],
        "starss_summary_frame_ap": star["models"]["gated_summary"]["val"]["macro_ap"],
        "onset_lift": onset["mean_lift"],
        "onset_above_background": onset["n_above_background"],
        "n_onsets": onset["n_onsets"],
        "silent_loss": probe["silent_loss"],
        "scramble_gap": probe["gap_nats"],
        "real_loss": probe["real_loss"],
        "perm_loss": probe["perm_loss"],
        "chance": probe["chance"],
        "teacher_cosine": temporal["teacher_offdiag_temporal_cosine"],
        "teacher_rank": temporal["mean_within_crop_effective_rank"],
        "teacher_rank_min": temporal["min_within_crop_effective_rank"],
        "probe": probe,
        "temporal": temporal,
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
        "tag_ap": payload["tag_ap"],
        "four_example_ap": payload["four_example_ap"],
        "starss_frame_ap": payload["starss_frame_ap"],
        "onset_lift": payload["onset_lift"],
        "silent_loss": payload["silent_loss"],
        "scramble_gap": payload["scramble_gap"],
        "teacher_cosine": payload["teacher_cosine"],
        "teacher_rank": payload["teacher_rank"],
        "gpu_usd": 0.0,
    }), flush=True)


if __name__ == "__main__":
    main()
