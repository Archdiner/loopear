import math
from collections import defaultdict
from typing import Any

import numpy as np
import torch


def fourier_coordinates(n_bins: int, dim: int) -> torch.Tensor:
    if int(dim) % 2 != 0:
        raise ValueError("coordinate dim must be even")
    half = int(dim) // 2
    position = torch.arange(int(n_bins), dtype=torch.float32).unsqueeze(1)
    rate = torch.exp(-torch.arange(half, dtype=torch.float32) * (math.log(10000.0) / max(half, 1)))
    angle = position * rate.unsqueeze(0)
    return torch.cat([torch.sin(angle), torch.cos(angle)], dim=1)


def coordinate_pool(latents: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    if latents.ndim != 3 or coords.ndim != 2 or latents.shape[-1] != coords.shape[-1]:
        raise ValueError("latents must be [batch, latents, dim] and coords [bins, dim]")
    scale = float(latents.shape[-1]) ** -0.5
    scores = torch.matmul(latents, coords.transpose(0, 1)) * scale
    weights = torch.softmax(scores, dim=1)
    return torch.einsum("bmk,bmd->bkd", weights, latents)


def clip_window_starts(n_frames: int, crop: int, max_windows: int) -> tuple[list[int], int]:
    if int(n_frames) <= int(crop):
        return [0], 0
    starts = list(range(0, int(n_frames) - int(crop) + 1, int(crop)))[: max(int(max_windows), 1)]
    if not starts:
        starts = [0]
    mid = (int(n_frames) - int(crop)) / 2.0
    center = min(range(len(starts)), key=lambda i: (abs(starts[i] - mid), i))
    return starts, center


def crop_energy_profile(mel: torch.Tensor, n_bins: int = 10) -> torch.Tensor:
    if mel.ndim != 2:
        raise ValueError("mel must be [frames, bins]")
    frames = int(mel.shape[0])
    if frames % int(n_bins) != 0:
        raise ValueError("frame count must divide into equal bins")
    energy = mel.mean(dim=1).view(int(n_bins), frames // int(n_bins)).mean(dim=1)
    return energy - energy.mean()


def mean_pearson(pred: np.ndarray, target: np.ndarray, min_std: float = 1e-4) -> dict[str, Any]:
    predicted = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if predicted.shape != truth.shape or predicted.ndim != 2:
        raise ValueError("pred and target must share shape [n, bins]")
    keep = (predicted.std(axis=1) >= float(min_std)) & (truth.std(axis=1) >= float(min_std))
    n_keep = int(keep.sum())
    if n_keep == 0:
        return {"mean_r": None, "n": 0, "n_skipped_flat": int(predicted.shape[0])}
    left = predicted[keep]
    right = truth[keep]
    left = left - left.mean(axis=1, keepdims=True)
    right = right - right.mean(axis=1, keepdims=True)
    numer = (left * right).sum(axis=1)
    denom = np.sqrt((left * left).sum(axis=1) * (right * right).sum(axis=1))
    corr = numer / np.clip(denom, 1e-12, None)
    return {
        "mean_r": float(corr.mean()),
        "n": n_keep,
        "n_skipped_flat": int(predicted.shape[0] - n_keep),
    }


def tag_class_names(
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    min_total: int = 3,
    min_each: int = 2,
) -> list[str]:
    counts: dict[str, int] = defaultdict(int)
    train_pos: dict[str, int] = defaultdict(int)
    val_pos: dict[str, int] = defaultdict(int)
    for row in train_rows:
        for label in set(row["labels"]):
            counts[label] += 1
            train_pos[label] += 1
    for row in val_rows:
        for label in set(row["labels"]):
            counts[label] += 1
            val_pos[label] += 1
    names = [
        label
        for label in sorted(counts)
        if counts[label] >= int(min_total)
        and train_pos[label] >= int(min_each)
        and val_pos[label] >= int(min_each)
    ]
    return names


def few_example_classes(
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    k: int = 4,
) -> list[dict[str, Any]]:
    train_pos: dict[str, list[int]] = defaultdict(list)
    val_pos: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(train_rows):
        for label in set(row["labels"]):
            train_pos[label].append(index)
    for index, row in enumerate(val_rows):
        for label in set(row["labels"]):
            val_pos[label].append(index)
    train_neg = {
        label: [index for index in range(len(train_rows)) if index not in set(positives)]
        for label, positives in train_pos.items()
    }
    chosen = []
    for label in sorted(train_pos):
        positives = train_pos[label]
        negatives = train_neg[label]
        queries = val_pos.get(label, [])
        if len(positives) < int(k) or len(negatives) < int(k) or len(queries) < int(k):
            continue
        chosen.append(
            {
                "label": label,
                "train_positive": positives[: int(k)],
                "train_negative": negatives[: int(k)],
                "n_train_positive": len(positives),
                "n_val_positive": len(queries),
            }
        )
    return chosen


def event_window_starts(
    duration_s: float,
    events: list[tuple[float, int]],
    max_windows: int = 6,
    window_s: float = 2.015,
    min_gap_s: float = 0.5,
) -> list[float]:
    if float(duration_s) <= float(window_s):
        return [0.0]
    last = max(0.0, float(duration_s) - float(window_s))
    by_class: dict[int, list[float]] = defaultdict(list)
    for time_s, class_id in events:
        if 0.0 <= float(time_s) <= float(duration_s):
            by_class[int(class_id)].append(float(time_s))
    raw = []
    for class_id in sorted(by_class):
        times = sorted(by_class[class_id])
        middle = times[len(times) // 2]
        start = min(max(0.0, middle - 0.4), last)
        raw.append(round(start, 2))
    kept: list[float] = []
    for start in sorted(raw):
        if all(abs(start - earlier) >= float(min_gap_s) - 1e-9 for earlier in kept):
            kept.append(start)
    slots = max(int(max_windows), 1)
    if len(kept) < slots:
        for index in range(slots):
            start = round(last * index / max(slots - 1, 1), 2)
            if all(abs(start - earlier) >= float(min_gap_s) - 1e-9 for earlier in kept):
                kept.append(start)
            if len(kept) >= slots:
                break
    return sorted(kept)[:slots]


def starss_keep(
    y_train: np.ndarray,
    y_val: np.ndarray,
    file_train: np.ndarray,
    file_val: np.ndarray,
    min_files: int = 2,
    min_frames: int = 40,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    train = np.asarray(y_train)
    val = np.asarray(y_val)
    train_files = np.asarray(file_train)
    val_files = np.asarray(file_val)
    if train.ndim != 2 or val.ndim != 2 or train.shape[1] != val.shape[1]:
        raise ValueError("STARSS labels must be [frames, classes] on both sides")
    keep = np.zeros(train.shape[1], dtype=bool)
    detail = []
    for cls in range(train.shape[1]):
        train_on = train[:, cls] >= 0.5
        val_on = val[:, cls] >= 0.5
        n_train_frames = int(train_on.sum())
        n_val_frames = int(val_on.sum())
        n_train_files = len(set(train_files[train_on].tolist())) if n_train_frames else 0
        n_val_files = len(set(val_files[val_on].tolist())) if n_val_frames else 0
        ok = (
            n_train_files >= int(min_files)
            and n_val_files >= int(min_files)
            and n_train_frames >= int(min_frames)
            and n_val_frames >= int(min_frames)
        )
        keep[cls] = ok
        detail.append(
            {
                "class_index": cls,
                "train_positive_frames": n_train_frames,
                "val_positive_frames": n_val_frames,
                "train_positive_files": int(n_train_files),
                "val_positive_files": int(n_val_files),
                "scored": bool(ok),
            }
        )
    return keep, detail
