from typing import Any

import torch

from loopear.d.causal import (
    CausalLoopEarDEncoder,
    l2_normalize,
    teacher_target,
    visible_from_frames,
)
from loopear.d.causal_probe import _event_module
from loopear.d.spec import N_FRAMES


def half_means(pooled: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if pooled.ndim == 2:
        pooled = pooled.unsqueeze(0)
        drop = True
    elif pooled.ndim == 3:
        drop = False
    else:
        raise ValueError("pooled must be [T, D] or [B, T, D]")
    if int(pooled.shape[1]) < 2:
        raise ValueError("the time axis needs two halves")
    mid = int(pooled.shape[1]) // 2
    first = pooled[:, :mid].mean(dim=1)
    second = pooled[:, mid:].mean(dim=1)
    if drop:
        return first[0], second[0]
    return first, second


def half_concat(pooled: torch.Tensor) -> torch.Tensor:
    first, second = half_means(pooled)
    return torch.cat([l2_normalize(first), l2_normalize(second)], dim=-1)


def preference_from_concat(
    clip: torch.Tensor,
    proto_first: torch.Tensor,
    proto_second: torch.Tensor,
):
    prefers = _event_module()._prefers_first
    single = clip.ndim == 1
    if single:
        clip = clip.unsqueeze(0)
        proto_first = proto_first.unsqueeze(0)
        proto_second = proto_second.unsqueeze(0)
    width = int(proto_first.shape[-1])
    if int(clip.shape[-1]) != width * 2:
        raise ValueError("half-concat width must be twice the prototype width")
    decisions = []
    for index in range(int(clip.shape[0])):
        decisions.append(
            prefers(
                clip[index, :width],
                clip[index, width:],
                proto_first[index],
                proto_second[index],
            )
        )
    if single:
        return decisions[0]
    return decisions


def build_splice_pairs(rows: list[dict[str, Any]], n_pairs: int = 500, seed: int = 42) -> list[dict[str, int]]:
    if int(n_pairs) < 1:
        raise ValueError("n_pairs must be positive")
    for row in rows:
        split = str(row["split"])
        if split == "test":
            raise PermissionError("FSD50K test recordings are sealed")
        if split != "train":
            raise PermissionError(f"order pairs refuse split {split}")
        label = str(row.get("id", ""))
        if "realdesed" in label.lower() or "fsd50k_test" in label.lower():
            raise PermissionError(label)
    count = len(rows)
    if count < 2:
        raise ValueError("order pairs need two train recordings")
    if int(n_pairs) > count * (count - 1) // 2:
        raise ValueError("not enough distinct train recordings")
    generator = torch.Generator().manual_seed(int(seed))
    chosen: list[dict[str, int]] = []
    seen: set[tuple[int, int]] = set()
    attempts = 0
    limit = int(n_pairs) * 40
    while len(chosen) < int(n_pairs):
        attempts += 1
        if attempts > limit:
            raise RuntimeError("could not sample distinct train pairs")
        pick = torch.randint(0, count, (2,), generator=generator)
        left = int(pick[0])
        right = int(pick[1])
        if left == right:
            continue
        key = (left, right) if left < right else (right, left)
        if key in seen:
            continue
        seen.add(key)
        chosen.append(
            {
                "index_a": left,
                "index_b": right,
                "freesound_a": int(rows[left]["freesound_id"]),
                "freesound_b": int(rows[right]["freesound_id"]),
            }
        )
    return chosen


def splice_halves(
    mel_a: torch.Tensor,
    valid_a: torch.Tensor,
    mel_b: torch.Tensor,
    valid_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if mel_a.shape != mel_b.shape or int(mel_a.shape[0]) < 2:
        raise ValueError("spliced crops must share a time axis of at least two frames")
    if valid_a.shape != valid_b.shape or valid_a.shape[0] != mel_a.shape[0]:
        raise ValueError("validity masks must match the crops")
    mid = int(mel_a.shape[0]) // 2
    ab_mel = torch.cat([mel_a[:mid], mel_b[:mid]], dim=0)
    ab_valid = torch.cat([valid_a[:mid], valid_b[:mid]], dim=0)
    ba_mel = torch.cat([mel_b[:mid], mel_a[:mid]], dim=0)
    ba_valid = torch.cat([valid_b[:mid], valid_a[:mid]], dim=0)
    return ab_mel, ab_valid, ba_mel, ba_valid


def _visible_crop(valid: torch.Tensor) -> bool:
    return bool(visible_from_frames(valid.unsqueeze(0)).flatten().any())


def _encode(encoder: CausalLoopEarDEncoder, mel: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    grid, _summary = encoder(mel, visible_from_frames(valid))
    return teacher_target(grid)


@torch.no_grad()
def score_splice_batch(
    encoder: CausalLoopEarDEncoder,
    mel_a: torch.Tensor,
    valid_a: torch.Tensor,
    mel_b: torch.Tensor,
    valid_b: torch.Tensor,
) -> dict[str, Any]:
    if mel_a.ndim != 3 or mel_a.shape != mel_b.shape:
        raise ValueError("splice batch must be [B, T, M] for both clips")
    if int(mel_a.shape[1]) != N_FRAMES:
        raise ValueError("splice crops must be one full crop")
    encoder.eval()
    kept_a_mel = []
    kept_a_valid = []
    kept_b_mel = []
    kept_b_valid = []
    ab_mels = []
    ab_valids = []
    ba_mels = []
    ba_valids = []
    n_skipped = 0
    for index in range(int(mel_a.shape[0])):
        ab_mel, ab_valid, ba_mel, ba_valid = splice_halves(
            mel_a[index], valid_a[index], mel_b[index], valid_b[index]
        )
        if not all(_visible_crop(mask) for mask in (valid_a[index], valid_b[index], ab_valid, ba_valid)):
            n_skipped += 1
            continue
        kept_a_mel.append(mel_a[index])
        kept_a_valid.append(valid_a[index])
        kept_b_mel.append(mel_b[index])
        kept_b_valid.append(valid_b[index])
        ab_mels.append(ab_mel)
        ab_valids.append(ab_valid)
        ba_mels.append(ba_mel)
        ba_valids.append(ba_valid)
    if not kept_a_mel:
        tally = _tally([], [])
        tally["n_skipped"] = n_skipped
        return tally
    pooled_a = _encode(encoder, torch.stack(kept_a_mel, dim=0), torch.stack(kept_a_valid, dim=0))
    pooled_b = _encode(encoder, torch.stack(kept_b_mel, dim=0), torch.stack(kept_b_valid, dim=0))
    pooled_ab = _encode(encoder, torch.stack(ab_mels, dim=0), torch.stack(ab_valids, dim=0))
    pooled_ba = _encode(encoder, torch.stack(ba_mels, dim=0), torch.stack(ba_valids, dim=0))
    width = int(pooled_a.shape[-1])
    proto_a = half_concat(pooled_a)[:, :width]
    proto_b = half_concat(pooled_b)[:, :width]
    decisions_ab = preference_from_concat(half_concat(pooled_ab), proto_a, proto_b)
    decisions_ba = preference_from_concat(half_concat(pooled_ba), proto_b, proto_a)
    tally = _tally(decisions_ab, decisions_ba)
    tally["n_skipped"] = n_skipped
    return tally


def score_train_splices(
    encoder: CausalLoopEarDEncoder,
    cache: Any,
    n_pairs: int = 500,
    seed: int = 42,
    batch: int = 16,
) -> dict[str, Any]:
    pairs = build_splice_pairs(list(cache.rows), n_pairs=n_pairs, seed=seed)
    generator = torch.Generator().manual_seed(int(seed))
    mel_a = []
    valid_a = []
    mel_b = []
    valid_b = []
    for pair in pairs:
        crop_a = cache.crop_once([int(pair["index_a"])], generator)
        crop_b = cache.crop_once([int(pair["index_b"])], generator)
        mel_a.append(crop_a["mel"][0])
        valid_a.append(crop_a["frame_valid"][0])
        mel_b.append(crop_b["mel"][0])
        valid_b.append(crop_b["frame_valid"][0])
    parts = []
    step = max(int(batch), 1)
    for start in range(0, len(pairs), step):
        stop = start + step
        parts.append(
            score_splice_batch(
                encoder,
                torch.stack(mel_a[start:stop], dim=0),
                torch.stack(valid_a[start:stop], dim=0),
                torch.stack(mel_b[start:stop], dim=0),
                torch.stack(valid_b[start:stop], dim=0),
            )
        )
    merged = _merge_tallies(parts, pairs)
    merged["seed"] = int(seed)
    return merged


def _merge_tallies(parts: list[dict[str, Any]], pairs: list[dict[str, int]]) -> dict[str, Any]:
    n_pairs = sum(int(part["n_pairs"]) for part in parts)
    n_directed = sum(int(part["n_directed"]) for part in parts)
    n_correct = sum(int(part["n_correct"]) for part in parts)
    n_ties = sum(int(part["n_ties"]) for part in parts)
    n_pairs_both = sum(int(part["n_pairs_both"]) for part in parts)
    n_skipped = sum(int(part["n_skipped"]) for part in parts)
    return {
        "label": "500-pair FSD50K train splice, [A | B] against [B | A], half-concat preference",
        "split": "train",
        "n_pairs_built": len(pairs),
        "n_pairs": n_pairs,
        "n_directed": n_directed,
        "n_correct": n_correct,
        "n_ties": n_ties,
        "n_pairs_both": n_pairs_both,
        "n_skipped": n_skipped,
        "accuracy": None if n_directed == 0 else n_correct / n_directed,
        "pair_accuracy": None if n_pairs == 0 else n_pairs_both / n_pairs,
        "chance": 0.5,
    }


def _tally(decisions_ab: list[bool | None], decisions_ba: list[bool | None]) -> dict[str, Any]:
    directed = decisions_ab + decisions_ba
    scored = len(directed)
    correct = sum(1 for item in directed if item is True)
    ties = sum(1 for item in directed if item is None)
    pair_correct = sum(
        1 for left, right in zip(decisions_ab, decisions_ba) if left is True and right is True
    )
    return {
        "label": "500-pair FSD50K train splice, [A | B] against [B | A], half-concat preference",
        "n_pairs": len(decisions_ab),
        "n_directed": scored,
        "n_correct": correct,
        "n_ties": ties,
        "n_pairs_both": pair_correct,
        "accuracy": None if scored == 0 else correct / scored,
        "pair_accuracy": None if not decisions_ab else pair_correct / len(decisions_ab),
        "chance": 0.5,
    }
