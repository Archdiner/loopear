import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from loopear.data import STARSS23_CLASSES
from loopear.d.diagnostics import cpu_seconds, hours, peak_rss_gb
from loopear.d.frontend import build_mel, load_frontend_config, normalize_logmel, waveform_logmel
from loopear.d.fsd import load_manifest
from loopear.d.objective import broadcast_time_valid, patch_valid_from_frames
from loopear.d.perceiver import load_local_with_teacher
from loopear.d.spec import CROP_SAMPLES, HOP_SAMPLES, N_FRAMES, PATCH_FRAMES, SR
from loopear.encoder import load_mono_audio
from loopear.harness import write_result_json


LOCKED_TIME_ALPHA = 0.0027896127554686775
MIN_GAP_S = 0.2
MAX_GAP_S = 1.5
LEAD_S = 0.45
MAX_PAIRS = 48
MAX_ONSETS_PER_CLASS = 32


def event_runs(events: list[tuple[int, int]]) -> list[tuple[int, int, int]]:
    grouped: dict[int, list[int]] = {}
    for frame, cls in events:
        grouped.setdefault(int(cls), []).append(int(frame))
    runs: list[tuple[int, int, int]] = []
    for cls, frames in grouped.items():
        ordered = sorted(set(frames))
        start = ordered[0]
        prev = ordered[0]
        for frame in ordered[1:]:
            if frame > prev + 1:
                runs.append((start, prev, cls))
                start = frame
            prev = frame
        runs.append((start, prev, cls))
    runs.sort()
    return runs


def candidate_pairs(
    runs: list[tuple[int, int, int]],
    min_gap_s: float = MIN_GAP_S,
    max_gap_s: float = MAX_GAP_S,
    label_hop_s: float = 0.1,
) -> list[tuple[int, int, int, int, float]]:
    pairs = []
    for index, (onset_a, end_a, class_a) in enumerate(runs):
        for onset_b, _end_b, class_b in runs[index + 1 :]:
            if class_a == class_b or onset_b <= end_a:
                continue
            gap = (onset_b - onset_a) * label_hop_s
            if gap < min_gap_s or gap > max_gap_s:
                continue
            pairs.append((onset_a, onset_b, class_a, class_b, gap))
    return pairs


def patch_index(onset_s: float, sample_start: int) -> int | None:
    rel = int(round(onset_s * SR)) - int(sample_start)
    if rel < 0 or rel >= CROP_SAMPLES:
        return None
    frame = rel // HOP_SAMPLES
    if frame < 0 or frame >= N_FRAMES:
        return None
    return int(frame // PATCH_FRAMES)


def place_crop(earlier_s: float, later_s: float, n_samples: int) -> int | None:
    sample = int(round(max(0.0, earlier_s - LEAD_S) * SR))
    if sample + CROP_SAMPLES > int(n_samples):
        sample = max(int(n_samples) - CROP_SAMPLES, 0)
    early = patch_index(earlier_s, sample)
    late = patch_index(later_s, sample)
    if early is None or late is None or early == late:
        return None
    return sample


def reversed_patch(onset_s: float, sample_start: int) -> int | None:
    rel = int(round(onset_s * SR)) - int(sample_start)
    flipped = CROP_SAMPLES - 1 - rel
    if flipped < 0 or flipped >= CROP_SAMPLES:
        return None
    frame = flipped // HOP_SAMPLES
    if frame < 0 or frame >= N_FRAMES:
        return None
    return int(frame // PATCH_FRAMES)


def _parse_events(text: str) -> list[tuple[int, int]]:
    parsed = []
    for line in text.splitlines():
        cols = [part.strip() for part in line.split(",")]
        if len(cols) >= 2 and cols[0].isdigit():
            parsed.append((int(cols[0]), int(cols[1])))
    return parsed


def _refuse(path: str) -> None:
    lowered = path.lower()
    if "realdesed" in lowered or "fsd50k" in lowered:
        raise PermissionError(f"refusing a sealed or non-STARSS path: {path}")


def _label_text(row: dict[str, Any]) -> str:
    local = Path(str(row["label_path"]))
    _refuse(str(local))
    if local.exists():
        return local.read_text(encoding="utf-8")
    last_error: Exception | None = None
    for prefix in ("labels/train/", "labels/test/", "labels/valid/", "labels/"):
        try:
            downloaded = hf_hub_download(
                "labhamlet/STARSS23", prefix + row["id"] + ".csv", repo_type="dataset"
            )
            _refuse(downloaded)
            return Path(downloaded).read_text(encoding="utf-8")
        except Exception as exc:
            last_error = exc
    raise FileNotFoundError(row["id"]) if last_error is None else last_error


def _wave(row: dict[str, Any]) -> np.ndarray:
    local = Path(str(row["audio_path"]))
    _refuse(str(local))
    if local.exists():
        return load_mono_audio(local, target_sr=SR, channel=0)
    downloaded = hf_hub_download(
        "labhamlet/STARSS23", "original_audios/" + row["id"] + ".wav", repo_type="dataset"
    )
    _refuse(downloaded)
    return load_mono_audio(downloaded, target_sr=SR, channel=0)


def _crop_mel(wav: np.ndarray, sample: int, mel_bank, mean: torch.Tensor, std: torch.Tensor, reverse: bool):
    chunk = np.ascontiguousarray(wav[sample : sample + CROP_SAMPLES], dtype=np.float32)
    if chunk.shape[0] < CROP_SAMPLES:
        chunk = np.pad(chunk, (0, CROP_SAMPLES - int(chunk.shape[0])))
    if reverse:
        chunk = chunk[::-1].copy()
    spec = waveform_logmel(torch.from_numpy(chunk), mel_bank)
    if int(spec.shape[0]) != N_FRAMES:
        raise RuntimeError(f"crop log-mel has {spec.shape[0]} frames")
    valid = torch.ones(N_FRAMES, dtype=torch.bool)
    heard = int(wav.shape[0]) - int(sample)
    if heard < CROP_SAMPLES:
        last = max((heard - 1) // HOP_SAMPLES, 0)
        if reverse:
            valid[: N_FRAMES - 1 - last] = False
        else:
            valid[last + 1 :] = False
    norm = normalize_logmel(spec, mean, std) * valid.unsqueeze(-1)
    return norm, valid


def _moment(model, mel: torch.Tensor, valid: torch.Tensor, patch: int) -> torch.Tensor:
    visible = broadcast_time_valid(patch_valid_from_frames(valid.unsqueeze(0)))
    latents, _summary = model.teacher_encoder(mel.unsqueeze(0), visible)
    local = model.project_teacher(model.teacher_predictor(latents, 0))
    return F.normalize(local[0, int(patch)].mean(dim=0), dim=0)


def _prefers_first(early: torch.Tensor, late: torch.Tensor, proto_first: torch.Tensor, proto_second: torch.Tensor):
    early_score = torch.dot(early, proto_first) - torch.dot(early, proto_second)
    late_score = torch.dot(late, proto_first) - torch.dot(late, proto_second)
    if float(early_score) == float(late_score):
        return None
    return bool(float(early_score) > float(late_score))


def _collect(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    onsets = []
    pairs = []
    for row in rows:
        if row["split"] not in ("train", "val"):
            raise PermissionError(f"STARSS row {row['id']} is {row['split']}")
        if "realdesed" in str(row["id"]).lower() or "fsd50k" in str(row["id"]).lower():
            raise PermissionError(row["id"])
        events = _parse_events(_label_text(row))
        runs = event_runs(events)
        for onset, _end, cls in runs:
            if 0 <= int(cls) < len(STARSS23_CLASSES):
                onsets.append({"id": row["id"], "split": row["split"], "onset_s": onset * 0.1, "class": int(cls)})
        if row["split"] != "val":
            continue
        for onset_a, onset_b, class_a, class_b, gap in candidate_pairs(runs):
            pairs.append(
                {
                    "id": row["id"],
                    "earlier_s": onset_a * 0.1,
                    "later_s": onset_b * 0.1,
                    "class_a": int(class_a),
                    "class_b": int(class_b),
                    "gap_s": float(gap),
                }
            )
    pairs.sort(key=lambda item: (item["id"], item["earlier_s"], item["later_s"], item["class_a"], item["class_b"]))
    return onsets, pairs[:MAX_PAIRS]


def _score_model(name: str, checkpoint: Path, time_alpha: float, waves: dict[str, np.ndarray], onsets, pairs, mel_bank, mean, std):
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    required = ("teacher_encoder", "teacher_predictor", "teacher_local")
    if any(key not in blob or not isinstance(blob[key], dict) for key in required):
        raise SystemExit(f"{name} checkpoint is missing EMA tensors")
    del blob
    model = load_local_with_teacher(str(checkpoint), time_alpha)
    model.eval()
    buckets: dict[int, list[torch.Tensor]] = {index: [] for index in range(len(STARSS23_CLASSES))}
    used = {index: 0 for index in range(len(STARSS23_CLASSES))}
    with torch.no_grad():
        for onset in onsets:
            if onset["split"] != "train" or used[onset["class"]] >= MAX_ONSETS_PER_CLASS:
                continue
            sample = place_crop(onset["onset_s"], onset["onset_s"] + 0.4, int(waves[onset["id"]].shape[0]))
            if sample is None:
                sample = int(round(max(0.0, onset["onset_s"] - LEAD_S) * SR))
                sample = min(sample, max(int(waves[onset["id"]].shape[0]) - CROP_SAMPLES, 0))
            patch = patch_index(onset["onset_s"], sample)
            if patch is None:
                continue
            mel, valid = _crop_mel(waves[onset["id"]], sample, mel_bank, mean, std, False)
            if not bool(valid[patch * PATCH_FRAMES : (patch + 1) * PATCH_FRAMES].any()):
                continue
            buckets[onset["class"]].append(_moment(model, mel, valid, patch))
            used[onset["class"]] += 1
        prototypes = {}
        for cls, vectors in buckets.items():
            if vectors:
                prototypes[cls] = F.normalize(torch.stack(vectors, dim=0).mean(dim=0), dim=0)
        correct = 0
        ties = 0
        scored = 0
        reversed_correct = 0
        reversed_ties = 0
        reversed_scored = 0
        for pair in pairs:
            if pair["class_a"] not in prototypes or pair["class_b"] not in prototypes:
                continue
            sample = place_crop(pair["earlier_s"], pair["later_s"], int(waves[pair["id"]].shape[0]))
            if sample is None:
                continue
            early_patch = patch_index(pair["earlier_s"], sample)
            late_patch = patch_index(pair["later_s"], sample)
            rev_early = reversed_patch(pair["later_s"], sample)
            rev_late = reversed_patch(pair["earlier_s"], sample)
            if None in (early_patch, late_patch, rev_early, rev_late) or early_patch == late_patch:
                continue
            mel, valid = _crop_mel(waves[pair["id"]], sample, mel_bank, mean, std, False)
            early = _moment(model, mel, valid, int(early_patch))
            late = _moment(model, mel, valid, int(late_patch))
            decision = _prefers_first(early, late, prototypes[pair["class_a"]], prototypes[pair["class_b"]])
            scored += 1
            if decision is None:
                ties += 1
            elif decision:
                correct += 1
            rev_mel, rev_valid = _crop_mel(waves[pair["id"]], sample, mel_bank, mean, std, True)
            rev_decision = _prefers_first(
                _moment(model, rev_mel, rev_valid, int(rev_early)),
                _moment(model, rev_mel, rev_valid, int(rev_late)),
                prototypes[pair["class_b"]],
                prototypes[pair["class_a"]],
            )
            reversed_scored += 1
            if rev_decision is None:
                reversed_ties += 1
            elif rev_decision:
                reversed_correct += 1
    return {
        "checkpoint": str(checkpoint),
        "ema_tensors_in_checkpoint": True,
        "time_alpha": float(time_alpha),
        "representation": "saved EMA local vectors, mean over frequency at the onset patch",
        "prototypes": "mean of at most 32 train-split onset vectors per class",
        "n_prototype_onsets": used,
        "n_pairs": scored,
        "n_correct": correct,
        "n_ties": ties,
        "accuracy": None if scored == 0 else correct / scored,
        "chance": 0.5,
        "reversed_audio": {
            "n_pairs": reversed_scored,
            "n_correct": reversed_correct,
            "n_ties": reversed_ties,
            "accuracy": None if reversed_scored == 0 else reversed_correct / reversed_scored,
            "scored_against": "the reversed event order",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pltc", type=Path, required=True)
    parser.add_argument("--plt", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--time-alpha", type=float, default=LOCKED_TIME_ALPHA)
    args = parser.parse_args()
    if args.report.exists():
        raise SystemExit(f"refusing to overwrite {args.report}")
    if torch.cuda.is_available():
        raise SystemExit("cuda is visible; refusing to start rather than risk GPU spend")
    torch.set_num_threads(4)
    started = cpu_seconds()
    wall0 = time.perf_counter()
    manifest = load_manifest(Path("data/manifests/starss23.json"))
    rows = list(manifest["items"])
    onsets, pairs = _collect(rows)
    needed = {item["id"] for item in onsets} | {item["id"] for item in pairs}
    waves = {}
    for row in rows:
        if row["id"] in needed:
            waves[row["id"]] = _wave(row)
    config = load_frontend_config(Path("data/manifests/loopear_d_frontend.json"))
    mean = torch.tensor(config["mean"], dtype=torch.float32)
    std = torch.tensor(config["std"], dtype=torch.float32)
    mel_bank = build_mel()
    payload = {
        "task": "loopear_event_order",
        "corpus": "starss23",
        "probe_split": "val",
        "prototype_split": "train",
        "min_gap_s": MIN_GAP_S,
        "max_gap_s": MAX_GAP_S,
        "lead_s": LEAD_S,
        "max_pairs": MAX_PAIRS,
        "n_pairs_selected": len(pairs),
        "chance": 0.5,
        "trained_against_this_probe": False,
        "fsd50k_test_audio_read": False,
        "sealed_realdesed_test_read": False,
        "crema_d_read": False,
        "vocalset_read": False,
        "maestro_read": False,
        "models": {
            "pltc": _score_model("pltc", args.pltc, args.time_alpha, waves, onsets, pairs, mel_bank, mean, std),
            "plt": _score_model("plt", args.plt, args.time_alpha, waves, onsets, pairs, mel_bank, mean, std),
        },
        "gpu_usd": 0.0,
    }
    payload["cpu_s"] = cpu_seconds() - started
    payload["wall_s"] = time.perf_counter() - wall0
    payload["cpu_hours"] = hours(payload["cpu_s"])
    payload["peak_rss_gb"] = peak_rss_gb()
    write_result_json(args.report, payload)
    print(
        json.dumps(
            {
                "wrote": str(args.report),
                "pltc_accuracy": payload["models"]["pltc"]["accuracy"],
                "plt_accuracy": payload["models"]["plt"]["accuracy"],
                "n_pairs": payload["models"]["pltc"]["n_pairs"],
                "gpu_usd": 0.0,
            }
        )
    )


if __name__ == "__main__":
    main()
