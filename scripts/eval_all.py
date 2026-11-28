import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from loopear.data import STARSS23_CLASSES, rasterize_starss_frames_50hz
from loopear.d.causal import CausalLoopEarDEncoder, frequency_pool
from loopear.d.causal_probe import score_event_order, score_half_concat_event_order
from loopear.d.esc50_stress import load_esc50_train, score_esc50_ood
from loopear.d.frontend import build_mel, load_frontend_config, waveform_logmel
from loopear.d.fsd import MelCache, audio_path, extract_audio, items_for, load_manifest
from loopear.d.objective import broadcast_time_valid, patch_valid_from_frames
from loopear.d.order_probe import score_train_splices
from loopear.d.probe import fixed_project, macro_average_precision
from loopear.d.protocol import event_window_starts, starss_keep
from loopear.d.spec import CROP_SAMPLES, D_MODEL, GRID_T, HOP_SAMPLES, N_FRAMES, SR
from loopear.encoder import load_mono_audio
from loopear.harness import write_result_json


ROOT = Path(__file__).resolve().parent.parent
ENCODE_BATCH = 16
EXPECTED_WINDOWS = 6676
EXPECTED_SHORT = 329
EXPECTED_STARSS_TRAIN = 7200
EXPECTED_STARSS_VAL = 4200


def _protocol():
    path = Path(__file__).with_name("eval_loopear_protocol.py")
    spec = importlib.util.spec_from_file_location("eval_loopear_protocol_all", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _visible(valid: torch.Tensor) -> torch.Tensor:
    return broadcast_time_valid(patch_valid_from_frames(valid))


def load_encoder(path: Path) -> CausalLoopEarDEncoder:
    encoder = CausalLoopEarDEncoder()
    if path.suffix == ".safetensors":
        state = load_file(str(path))
    else:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if "teacher" in path.name and isinstance(blob.get("teacher_encoder"), dict):
            state = blob["teacher_encoder"]
        elif isinstance(blob.get("student_encoder"), dict):
            state = blob["student_encoder"]
        elif isinstance(blob.get("teacher_encoder"), dict):
            state = blob["teacher_encoder"]
        else:
            raise SystemExit(f"{path} has no causal encoder weights")
    encoder.load_state_dict(state)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad_(False)
    return encoder


def _ensure_audio(rows: list[dict[str, Any]]) -> None:
    missing = [row for row in rows if not audio_path(Path("data/raw/fsd50k"), row).exists()]
    if not missing:
        return
    for row in missing:
        if row["split"] == "test":
            raise PermissionError("refusing to materialize FSD50K test audio")
        if row["split"] not in ("train", "val"):
            raise PermissionError(f"refusing split {row['split']}")
    print(json.dumps({"extracting_clips": len(missing), "split": missing[0]["split"]}), flush=True)
    extract_audio(missing, Path("data/raw/fsd50k"))


def _encode_summaries(rows, encoder, mean, std, mel_bank, proto) -> dict[str, Any]:
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
        starts, _center = proto.clip_window_starts(int(spec.shape[0]), N_FRAMES, proto.TAG_WINDOWS)
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
        "means": {"summary": (sums / denom[:, None]).astype(np.float32)},
        "n_windows": n_windows,
        "n_short_windows": n_short,
        "n_empty_clips": int((counts == 0).sum()),
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
            windows.append(
                {
                    "time": frequency_pool(grid)[0].detach().float().cpu(),
                    "y": labels,
                }
            )
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
    if int(y_train.shape[0]) != EXPECTED_STARSS_TRAIN or int(y_val.shape[0]) != EXPECTED_STARSS_VAL:
        raise SystemExit(
            f"STARSS frames {int(y_train.shape[0])}/{int(y_val.shape[0])} do not match 7200/4200"
        )
    keep, _detail = starss_keep(y_train, y_val, files_train, files_val, min_files=2, min_frames=40)
    y_tr = torch.from_numpy(y_train[:, keep])
    y_va = torch.from_numpy(y_val[:, keep])
    x_tr = fixed_project(x_train, D_MODEL, proto.SEED)
    x_va = fixed_project(x_val, D_MODEL, proto.SEED)
    real = proto._scores(x_tr, y_tr, x_va, proto.SEED)
    if int(real.shape[1]) != int(keep.sum()):
        raise RuntimeError("frame-probe scores do not match the kept classes")
    score_windows = []
    cursor = 0
    for window in window_labels:
        n_frames = window.shape[0]
        score_windows.append(real[cursor : cursor + n_frames])
        cursor += n_frames
    if cursor != int(real.shape[0]):
        raise RuntimeError("frame-probe scores do not cover the val windows")
    label_windows = [window[:, keep] for window in window_labels]
    ap = macro_average_precision(y_va.numpy(), real)
    onset = proto._onset_within_windows(label_windows, score_windows)
    return {
        "frame_ap": ap["macro_ap"],
        "onset_lift": onset["mean_lift"],
        "n_onsets": onset["n_onsets"],
        "n_classes_scored": int(keep.sum()),
        "n_train_frames": int(y_train.shape[0]),
        "n_val_frames": int(y_val.shape[0]),
    }


def _cell(value: Any) -> str:
    if value is None:
        return "absent"
    if isinstance(value, str):
        return value
    return f"{float(value):.4f}"


def _ratio(correct: int | None, total: int | None) -> str:
    if correct is None or total is None:
        return "absent"
    return f"{int(correct)}/{int(total)}"


def _markdown(rows: list[dict[str, str]]) -> str:
    headers = [
        "probe",
        "FSD50K tag AP",
        "four-example AP",
        "STARSS23 frame AP",
        "onset lift",
        "event order",
        "half-concat",
        "500-pair",
        "ESC-50 OOD",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row[header] for header in headers) + " |")
    return "\n".join(lines)


def score_checkpoint(path: Path, proto, mean, std, mel_bank, cache, esc50) -> dict[str, Any]:
    encoder = load_encoder(path)
    print("patch event order", flush=True)
    order = score_event_order(encoder)
    print(
        json.dumps({"event_order": f"{order['n_correct']}/{order['n_pairs']}", "accuracy": order["accuracy"]}),
        flush=True,
    )
    print("half-concat event order", flush=True)
    half = score_half_concat_event_order(encoder)
    print(
        json.dumps({"half_concat": f"{half['n_correct']}/{half['n_pairs']}", "accuracy": half["accuracy"]}),
        flush=True,
    )
    print("500-pair train splice", flush=True)
    splice = score_train_splices(encoder, cache, n_pairs=500, seed=42, batch=ENCODE_BATCH)
    print(
        json.dumps(
            {
                "splice": f"{splice['n_correct']}/{splice['n_directed']}",
                "accuracy": splice["accuracy"],
                "n_skipped": splice["n_skipped"],
            }
        ),
        flush=True,
    )
    manifest = load_manifest(Path("data/manifests/fsd50k_loopear_d.json"))
    train_rows = items_for(manifest, "h5")
    val_rows = proto._official_val(manifest, proto.VAL_CLIPS)
    _ensure_audio(train_rows + val_rows)
    rows = train_rows + val_rows
    print(f"encoding {len(train_rows)} development clips and {len(val_rows)} val clips", flush=True)
    encoded = _encode_summaries(rows, encoder, mean, std, mel_bank, proto)
    if encoded["n_windows"] != EXPECTED_WINDOWS or encoded["n_short_windows"] != EXPECTED_SHORT or encoded["n_empty_clips"] != 0:
        raise SystemExit(
            f"FSD windowing does not match the saved protocol: {encoded['n_windows']} windows, "
            f"{encoded['n_short_windows']} short, {encoded['n_empty_clips']} empty"
        )
    tag, few = _tag_and_few(encoded, train_rows, val_rows, proto)
    print("STARSS frame probe", flush=True)
    star = _starss(encoder, mean, std, mel_bank, proto)
    print("ESC-50 polyphonic stress", flush=True)
    ood = score_esc50_ood(encoder, esc50, mean, std, mel_bank)
    print(
        json.dumps({"esc50_ood": f"{ood['n_correct']}/{ood['n_clips']}", "accuracy": ood["accuracy"], "by_snr": ood["by_snr"]}),
        flush=True,
    )
    return {
        "checkpoint": str(path),
        "tag_ap": tag["probe"]["val"]["macro_ap"],
        "four_example_ap": None if few.get("status") != "measured" else few["models"]["summary"]["macro_ap"],
        "frame_ap": star["frame_ap"],
        "onset_lift": star["onset_lift"],
        "event_order": order,
        "half_concat": half,
        "splice": splice,
        "starss": star,
        "esc50_ood": ood,
        "fsd_windows": encoded["n_windows"],
    }


def _tag_and_few(encoded, train_rows, val_rows, proto):
    from loopear.d.protocol import tag_class_names

    names = tag_class_names(train_rows, val_rows)
    y_train = proto._multi_hot(train_rows, names)
    y_val = proto._multi_hot(val_rows, names)
    matrix = encoded["means"]["summary"]
    n_train = len(train_rows)
    tag = proto._probe_pair(
        torch.from_numpy(matrix[:n_train]),
        torch.from_numpy(matrix[n_train:]),
        y_train,
        y_val,
    )
    few = proto._few_block(train_rows, val_rows, {"summary": matrix})
    return tag, few


def _row_from_score(name: str, score: dict[str, Any]) -> dict[str, str]:
    order = score["event_order"]
    half = score["half_concat"]
    splice = score["splice"]
    return {
        "probe": name,
        "FSD50K tag AP": _cell(score["tag_ap"]),
        "four-example AP": _cell(score["four_example_ap"]),
        "STARSS23 frame AP": _cell(score["frame_ap"]),
        "onset lift": _cell(score["onset_lift"]),
        "event order": _ratio(order["n_correct"], order["n_pairs"]),
        "half-concat": _ratio(half["n_correct"], half["n_pairs"]),
        "500-pair": _ratio(splice["n_correct"], splice["n_directed"]),
        "ESC-50 OOD": _ratio(score["esc50_ood"]["n_correct"], score["esc50_ood"]["n_clips"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--out", default="results/probe_audit.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    proto = _protocol()
    config = load_frontend_config(Path("data/manifests/loopear_d_frontend.json"))
    mean = torch.tensor(config["mean"], dtype=torch.float32)
    std = torch.tensor(config["std"], dtype=torch.float32)
    mel_bank = build_mel()
    manifest = load_manifest(Path("data/manifests/fsd50k_loopear_d.json"))
    train_rows = items_for(manifest, "h20")
    cache = MelCache(train_rows, Path("cache/loopear_d_mel"), mean, std)
    print("loading ashraq/esc50 train", flush=True)
    esc50 = load_esc50_train(ROOT / "data" / "esc50_cache")
    table_rows = []
    payload = []
    for raw in args.checkpoints:
        path = Path(raw)
        print(f"scoring {path}", flush=True)
        score = score_checkpoint(path, proto, mean, std, mel_bank, cache, esc50)
        payload.append(score)
        table_rows.append(_row_from_score(path.name, score))
    table = _markdown(table_rows)
    print(table, flush=True)
    saved = []
    for score in payload:
        order = dict(score["event_order"])
        order["n_prototype_onsets"] = {str(key): value for key, value in order["n_prototype_onsets"].items()}
        saved.append({**score, "event_order": order})
    write_result_json(
        Path(args.out),
        {
            "task": "probe_audit",
            "gpu_usd": 0.0,
            "fsd50k_test_audio_read": False,
            "sealed_realdesed_test_read": False,
            "rows": saved,
            "table": table,
        },
    )


if __name__ == "__main__":
    main()
