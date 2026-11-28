import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from loopear.d.causal import CausalLoopEarD, frequency_pool
from loopear.d.causal_probe import teacher_temporal_stats
from loopear.d.diagnostics import cpu_seconds, hours, peak_rss_gb
from loopear.d.frontend import build_mel, load_frontend_config, waveform_logmel
from loopear.d.fsd import audio_path, items_for, load_manifest
from loopear.d.objective import broadcast_time_valid, patch_valid_from_frames
from loopear.d.probe import fixed_project, macro_average_precision
from loopear.d.protocol import clip_window_starts, event_window_starts, starss_keep, tag_class_names
from loopear.d.spec import CROP_SAMPLES, D_MODEL, GRID_T, HOP_SAMPLES, N_FRAMES, SR
from loopear.data import STARSS23_CLASSES, rasterize_starss_frames_50hz
from loopear.encoder import load_mono_audio
from loopear.harness import SpendLedger, write_result_json
from loopear.losses import set_seed


ROOT = Path(__file__).resolve().parent.parent
VAL_CLIPS = 640
TAG_WINDOWS = 4
ENCODE_BATCH = 16
EXPECTED_WINDOWS = 6676
EXPECTED_SHORT = 329
EXPECTED_STARSS_TRAIN = 7200
EXPECTED_STARSS_VAL = 4200


def _module(name: str):
    path = Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location(path.stem + "_pit", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _visible(valid: torch.Tensor) -> torch.Tensor:
    return broadcast_time_valid(patch_valid_from_frames(valid))


def _load_model(path: Path) -> CausalLoopEarD:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model = CausalLoopEarD()
    model.encoder.load_state_dict(blob["student_encoder"])
    model.teacher_encoder.load_state_dict(blob["teacher_encoder"])
    model.predictor.heads.load_state_dict(blob["horizons"])
    if list(model.predictor.horizons) != [4, 10, 20]:
        raise AssertionError(f"horizons {model.predictor.horizons}")
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


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
    empty = int((counts == 0).sum())
    if empty:
        raise RuntimeError(f"{empty} clips produced no windows")
    means = (sums / counts[:, None]).astype(np.float32)
    return {
        "means": {"summary": means},
        "n_windows": n_windows,
        "n_short_windows": n_short,
        "n_empty_clips": empty,
    }


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
        csv_src = proto._label_csv(row["id"])
        wav = load_mono_audio(wav_src, target_sr=SR, channel=0)
        spec = waveform_logmel(torch.from_numpy(wav), mel_bank)
        events = proto._parse_events(Path(csv_src).read_text(encoding="utf-8"))
        duration = float(wav.shape[0]) / float(SR)
        starts = event_window_starts(duration, [(idx * 0.1, cls) for idx, cls in events], max_windows=6)
        windows = []
        pending = []

        def _flush_windows() -> None:
            nonlocal pending
            if not pending:
                return
            mel = torch.stack([item["norm"] for item in pending])
            visible = torch.cat([item["visible"] for item in pending], dim=0)
            with torch.inference_mode():
                grid, _summary = encoder(mel, visible)
                pooled = frequency_pool(grid).detach().float().cpu()
            for index, item in enumerate(pending):
                windows.append({"time": pooled[index], "y": item["y"]})
            pending = []

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
            labels = rasterize_starss_frames_50hz(
                events,
                float(sample) / float(SR),
                GRID_T,
                len(STARSS23_CLASSES),
                hop_e=0.04,
                label_hop_s=0.1,
            )
            pending.append({"norm": norm, "visible": visible, "y": labels})
            if len(pending) >= ENCODE_BATCH:
                _flush_windows()
        _flush_windows()
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
    y_tr = torch.from_numpy(y_train[:, keep])
    y_va = torch.from_numpy(y_val[:, keep])
    label_windows = [window[:, keep] for window in window_labels]
    if int(keep.sum()) == 0:
        measured = {"status": "null", "reason": "no class met the support rule"}
    else:
        x_tr = fixed_project(x_train, D_MODEL, 42)
        x_va = fixed_project(x_val, D_MODEL, 42)
        real = proto._scores(x_tr, y_tr, x_va, 42)
        perm = torch.Generator().manual_seed(42 + 7)
        order = torch.randperm(y_tr.shape[0], generator=perm)
        shuffled = proto._scores(x_tr, y_tr[order], x_va, 42)
        real_ap = macro_average_precision(y_va.numpy(), real)
        shuffled_ap = macro_average_precision(y_va.numpy(), shuffled)
        score_windows = []
        cursor = 0
        for window in window_labels:
            n_frames = window.shape[0]
            score_windows.append(real[cursor : cursor + n_frames])
            cursor += n_frames
        measured = {
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
        "time": measured,
        "frame_axis": "50 patch steps of 40 ms; frequency-pooled causal grid",
    }


def _macro(block: dict[str, Any] | None, path: tuple[str, ...]) -> float | None:
    cursor: Any = block
    for key in path:
        if not isinstance(cursor, dict) or key not in cursor:
            return None
        cursor = cursor[key]
    if cursor is None:
        return None
    return float(cursor)


def score_model(
    model: CausalLoopEarD,
    cache,
    dest: Path,
    batch: int,
    seed: int,
    step: int,
    checkpoint: str,
) -> dict[str, Any]:
    if dest.exists():
        raise FileExistsError(dest)
    training = model.training
    cpu0 = cpu_seconds()
    eval_mod = _module("eval_loopear_causal.py")
    train_mod = _module("train_loopear_causal.py")
    proto = eval_mod._protocol()
    model.eval()
    manifest = load_manifest(Path("data/manifests/fsd50k_loopear_d.json"))
    train_rows = items_for(manifest, "h5")
    val_rows = proto._official_val(manifest, VAL_CLIPS)
    for row in train_rows + val_rows:
        if row["split"] == "test":
            raise PermissionError("protocol tried to read the sealed FSD50K test split")
    eval_mod._ensure_audio(train_rows + val_rows)
    config = load_frontend_config(Path("data/manifests/loopear_d_frontend.json"))
    mean = torch.tensor(config["mean"], dtype=torch.float32)
    std = torch.tensor(config["std"], dtype=torch.float32)
    mel_bank = build_mel()
    rows = train_rows + val_rows
    print(f"encoding {len(train_rows)} development clips and {len(val_rows)} val clips", flush=True)
    encoded = _encode_summary(rows, model.encoder, mean, std, mel_bank, proto)
    if encoded["n_windows"] != EXPECTED_WINDOWS or encoded["n_short_windows"] != EXPECTED_SHORT:
        raise SystemExit(
            f"FSD windowing does not match the saved protocol: {encoded['n_windows']} windows, "
            f"{encoded['n_short_windows']} short"
        )
    names = tag_class_names(train_rows, val_rows)
    y_train = proto._multi_hot(train_rows, names)
    y_val = proto._multi_hot(val_rows, names)
    summary = torch.from_numpy(encoded["means"]["summary"])
    tag = proto._probe_pair(summary[: len(train_rows)], summary[len(train_rows) :], y_train, y_val)
    few = proto._few_block(train_rows, val_rows, encoded["means"])
    print("tag", tag["probe"]["val"]["macro_ap"], flush=True)
    print("four", few["models"]["summary"]["macro_ap"], flush=True)
    star = _starss(model.encoder, mean, std, mel_bank, proto)
    if star["n_train_frames"] != EXPECTED_STARSS_TRAIN or star["n_val_frames"] != EXPECTED_STARSS_VAL:
        raise SystemExit(
            f"STARSS frames {star['n_train_frames']}/{star['n_val_frames']} do not match 7200/4200"
        )
    corpus = {
        "clips": len(cache.rows),
        "hours": sum(float(row["duration_s"]) for row in cache.rows) / 3600.0,
    }
    set_seed(int(seed))
    probe = train_mod._score_probe(model, train_mod._fixed_probe(cache, int(batch), 8, int(seed)), int(batch))
    temporal = teacher_temporal_stats(model.teacher_encoder, cache, 8, int(batch), int(seed))
    onset = star["time"].get("onset_within_windows") if isinstance(star["time"], dict) else None
    payload = {
        "task": "loopear_latent_pit_checkpoint",
        "step": int(step),
        "checkpoint": str(checkpoint),
        "gpu_usd": 0.0,
        "from_scratch": True,
        "fsd50k_test_audio_read": False,
        "sealed_realdesed_test_read": False,
        "crema_d_read": False,
        "vocalset_read": False,
        "maestro_read": False,
        "dipco_scored": False,
        "development_clips": len(train_rows),
        "validation_clips": len(val_rows),
        "corpus": corpus,
        "fsd_windows": encoded["n_windows"],
        "fsd_short_windows": encoded["n_short_windows"],
        "causal_clip_embedding": "mean of the block-causal grid",
        "causal_starss_primary": "frequency-pooled time grid",
        "tag_ap": _macro(tag, ("probe", "val", "macro_ap")),
        "four_example_ap": _macro(few, ("models", "summary", "macro_ap")),
        "starss_frame_ap": _macro(star["time"], ("val", "macro_ap")),
        "onset_lift": None if not isinstance(onset, dict) else onset.get("mean_lift"),
        "silent_clip_loss": probe["silent_loss"],
        "scramble_gap": probe["gap_nats"],
        "real_nce": probe["real_loss"],
        "perm_nce": probe["perm_loss"],
        "teacher_offdiag_temporal_cosine": temporal["teacher_offdiag_temporal_cosine"],
        "teacher_within_crop_rank": temporal["mean_within_crop_effective_rank"],
        "teacher_min_rank": temporal["min_within_crop_effective_rank"],
        "probe": probe,
        "temporal": temporal,
        "tag_probe": tag,
        "few_example": few,
        "starss23": star,
        "cpu_s": cpu_seconds() - cpu0,
        "cpu_hours": hours(cpu_seconds() - cpu0),
        "peak_rss_gb": peak_rss_gb(),
    }
    if training:
        model.train()
    path = write_result_json(dest, payload)
    print(json.dumps({
        "wrote": str(path),
        "step": int(step),
        "tag_ap": payload["tag_ap"],
        "four_example_ap": payload["four_example_ap"],
        "starss_frame_ap": payload["starss_frame_ap"],
        "onset_lift": payload["onset_lift"],
        "silent_clip_loss": payload["silent_clip_loss"],
        "scramble_gap": payload["scramble_gap"],
        "teacher_offdiag_temporal_cosine": payload["teacher_offdiag_temporal_cosine"],
        "teacher_within_crop_rank": payload["teacher_within_crop_rank"],
        "gpu_usd": 0.0,
    }), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--step", type=int, required=True)
    args = parser.parse_args()
    if torch.cuda.is_available():
        raise SystemExit("cuda is visible; refusing to start rather than risk GPU spend")
    dest = Path(args.out)
    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise SystemExit(f"missing checkpoint {ckpt}")
    torch.set_num_threads(4)
    os.chdir(ROOT)
    ledger = SpendLedger(Path("ledger/loopear_causal_spend.jsonl"), cap_usd=0.0)
    ledger.record("loopear_latent_pit_eval", "cpu", 0.0, "no paid GPU")
    train_mod = _module("train_loopear_causal.py")
    cache, _corpus = train_mod._prepare(
        Path("data/manifests/fsd50k_loopear_d.json"),
        Path("cache/loopear_d_mel"),
        Path("data/manifests/loopear_d_frontend.json"),
    )
    model = _load_model(ckpt)
    score_model(model, cache, dest, int(args.batch), int(args.seed), int(args.step), str(ckpt))


if __name__ == "__main__":
    main()
