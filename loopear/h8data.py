import json
from pathlib import Path
from typing import Any

import numpy as np

from loopear.data import (
    DIPCO_OVERLAP_CLASSES,
    REALDESED_CLASSES,
    STARSS23_CLASSES,
    rasterize_intervals_50hz,
    rasterize_overlap_50hz,
    rasterize_starss_frames_50hz,
)
from loopear.encoder import SR, load_mono_audio
from loopear.harness import SealedSplitGuard

DEV_CORPORA: tuple[str, ...] = ("starss23", "realdesed", "dipco", "ami_sdm")
TRANSFER_CORPORA: tuple[str, ...] = ("crema_d", "vocalset", "maestro_real")
BLOCKED_PATH_MARKERS: tuple[str, ...] = (
    "realdesed/test/",
    "realdesed_test",
    "test.zip",
)


def assert_not_sealed(item: dict[str, Any]) -> None:
    split = str(item.get("split", ""))
    path = str(item.get("audio_path", "")).replace("\\", "/")
    if split in {"test", "realdesed_test"}:
        raise PermissionError(f"sealed split blocked: {split}")
    for marker in BLOCKED_PATH_MARKERS:
        if marker in path:
            raise PermissionError(f"sealed path blocked: {path}")


def load_manifest(root: Path | str, corpus: str) -> dict[str, Any]:
    path = Path(root) / "data" / "manifests" / f"{corpus}.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if corpus == "realdesed":
        SealedSplitGuard(("realdesed_test",)).check_access("realdesed_train")
        SealedSplitGuard(("realdesed_test",)).check_access("realdesed_val")
    for item in manifest["items"]:
        assert_not_sealed(item)
    return manifest


def iter_items(manifest: dict[str, Any], split: str) -> list[dict[str, Any]]:
    rows = []
    for item in manifest["items"]:
        if str(item["split"]) != split:
            continue
        assert_not_sealed(item)
        rows.append(item)
    return rows


def audio_ready(root: Path | str, corpora: tuple[str, ...] = DEV_CORPORA) -> bool:
    base = Path(root)
    for corpus in corpora:
        manifest = load_manifest(base, corpus)
        for item in manifest["items"]:
            if not (base / item["audio_path"]).exists():
                return False
    return True


def train_hours(root: Path | str, corpora: tuple[str, ...] = DEV_CORPORA) -> float:
    total = 0.0
    for corpus in corpora:
        manifest = load_manifest(root, corpus)
        for item in iter_items(manifest, "train"):
            total += float(item["duration_s"])
    return total / 3600.0


def load_wav(root: Path | str, item: dict[str, Any], start_s: float, end_s: float) -> np.ndarray:
    assert_not_sealed(item)
    path = Path(root) / str(item["audio_path"])
    return load_mono_audio(
        path,
        target_sr=SR,
        channel=item.get("channel"),
        start_s=start_s,
        end_s=end_s,
    )


def window_spans(
    duration_s: float,
    win_s: float,
    hop_s: float,
    max_windows: int,
) -> list[tuple[float, float]]:
    if duration_s <= 0.1:
        return []
    if duration_s <= win_s:
        return [(0.0, float(duration_s))]
    spans: list[tuple[float, float]] = []
    start = 0.0
    while start + 0.1 < duration_s and len(spans) < max_windows:
        spans.append((start, min(duration_s, start + win_s)))
        start += hop_s
    return spans


def frame_targets(
    corpus: str,
    item: dict[str, Any],
    start_s: float,
    n_frames: int,
    root: Path | str,
) -> np.ndarray | None:
    if corpus == "starss23":
        events: list[tuple[int, int]] = []
        label_path = Path(root) / str(item["label_path"])
        for line in label_path.read_text(encoding="utf-8").splitlines():
            cols = [part.strip() for part in line.split(",")]
            if len(cols) >= 2 and cols[0].isdigit():
                events.append((int(cols[0]), int(cols[1])))
        return rasterize_starss_frames_50hz(
            events, start_s, n_frames, n_classes=len(STARSS23_CLASSES)
        )
    if corpus == "realdesed":
        intervals = [
            (float(row[0]), float(row[1]), int(row[2])) for row in item["intervals"]
        ]
        return rasterize_intervals_50hz(
            intervals, start_s, n_frames, n_classes=len(REALDESED_CLASSES)
        )
    if corpus == "dipco":
        intervals = [
            (float(row[0]), float(row[1]), str(row[2]))
            for row in item["speaker_intervals"]
        ]
        overlap = rasterize_overlap_50hz(intervals, start_s, n_frames)
        out = np.zeros((n_frames, len(DIPCO_OVERLAP_CLASSES)), dtype=np.float32)
        if n_frames > 0:
            out[np.arange(n_frames), overlap.astype(np.int64)] = 1.0
        return out
    return None


def cross_group_pair_count(items: list[dict[str, Any]]) -> int:
    groups = []
    for item in items:
        groups.append((str(item.get("corpus", "")), str(item["group_id"])))
    unique = sorted(set(groups))
    if len(unique) < 2:
        return 0
    return len(unique) * (len(unique) - 1) // 2
