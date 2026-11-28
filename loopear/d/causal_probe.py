import importlib.util
from pathlib import Path
from typing import Any

import torch

from loopear.d.causal import (
    CausalLoopEarDEncoder,
    l2_normalize,
    teacher_target,
    time_valid_from_frames,
    visible_from_frames,
)
from loopear.d.diagnostics import embedding_diagnostics
from loopear.d.fsd import MelCache
from loopear.d.spec import GRID_T


def _event_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "eval_loopear_event_order.py"
    spec = importlib.util.spec_from_file_location("loopear_event_order_eval", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _upper_cosine(vectors: torch.Tensor, valid: torch.Tensor) -> tuple[float, int]:
    sim = torch.einsum("btd,bsd->bts", vectors, vectors)
    both = valid[:, :, None] & valid[:, None, :]
    upper = torch.triu(torch.ones(GRID_T, GRID_T, dtype=torch.bool, device=vectors.device), diagonal=1)
    mask = both & upper.view(1, GRID_T, GRID_T)
    count = int(mask.sum())
    if count == 0:
        return float("nan"), 0
    return float(sim[mask].sum()), count


@torch.no_grad()
def teacher_temporal_stats(
    encoder: CausalLoopEarDEncoder,
    cache: MelCache,
    n_batches: int,
    batch: int,
    seed: int,
) -> dict[str, Any]:
    if n_batches < 1 or batch < 2:
        raise ValueError("temporal stats need at least one batch of two crops")
    encoder.eval()
    order_gen = torch.Generator().manual_seed(int(seed) + 10007)
    crop_gen = torch.Generator().manual_seed(int(seed) + 10007 + 17)
    order: list[int] = []
    cursor = 0
    pooled_sum = 0.0
    pooled_n = 0
    freq_sum = 0.0
    freq_n = 0
    ranks: list[float] = []
    n_crops = 0
    for _ in range(int(n_batches)):
        if cursor + int(batch) > len(order):
            order = torch.randperm(len(cache), generator=order_gen).tolist()
            cursor = 0
        chosen = order[cursor : cursor + int(batch)]
        cursor += int(batch)
        crops = cache.crop_once(chosen, crop_gen)
        visible = visible_from_frames(crops["frame_valid"])
        grid, _summary = encoder(crops["mel"], visible)
        valid = time_valid_from_frames(crops["frame_valid"])
        pooled = teacher_target(grid)
        part, count = _upper_cosine(pooled, valid)
        if count:
            pooled_sum += part
            pooled_n += count
        tokens = l2_normalize(grid)
        freq_sim = torch.einsum("btfd,bsfd->bfts", tokens, tokens)
        upper = torch.triu(torch.ones(GRID_T, GRID_T, dtype=torch.bool, device=grid.device), diagonal=1)
        pair = (valid[:, :, None] & valid[:, None, :]) & upper.view(1, GRID_T, GRID_T)
        freq_mask = pair[:, None, :, :].expand_as(freq_sim)
        freq_count = int(freq_mask.sum())
        if freq_count:
            freq_sum += float(freq_sim[freq_mask].sum())
            freq_n += freq_count
        for row in range(int(grid.shape[0])):
            kept = grid[row][visible[row]]
            if int(kept.shape[0]) >= 2:
                ranks.append(float(embedding_diagnostics(kept)["effective_rank"]))
        n_crops += int(grid.shape[0])
    return {
        "n_crops": n_crops,
        "n_batches": int(n_batches),
        "batch": int(batch),
        "seed": int(seed) + 10007,
        "teacher_offdiag_temporal_cosine": None if pooled_n == 0 else pooled_sum / float(pooled_n),
        "teacher_per_frequency_offdiag_cosine": None if freq_n == 0 else freq_sum / float(freq_n),
        "n_pooled_pairs": pooled_n,
        "n_per_frequency_pairs": freq_n,
        "mean_within_crop_effective_rank": None if not ranks else float(sum(ranks) / len(ranks)),
        "min_within_crop_effective_rank": None if not ranks else float(min(ranks)),
        "n_ranked_crops": len(ranks),
        "representation": "EMA teacher grid after the encoder LayerNorm; pooled cosine is the L2 frequency mean",
    }


def _moment(encoder: CausalLoopEarDEncoder, mel: torch.Tensor, valid: torch.Tensor, patch: int) -> torch.Tensor:
    visible = visible_from_frames(valid.unsqueeze(0))
    grid, _summary = encoder(mel.unsqueeze(0), visible)
    pooled = teacher_target(grid)
    return pooled[0, int(patch)]


@torch.no_grad()
def score_event_order(encoder: CausalLoopEarDEncoder) -> dict[str, Any]:
    module = _event_module()
    encoder.eval()
    manifest = module.load_manifest(Path("data/manifests/starss23.json"))
    rows = list(manifest["items"])
    onsets, pairs = module._collect(rows)
    needed = {item["id"] for item in onsets} | {item["id"] for item in pairs}
    waves = {}
    for row in rows:
        if row["id"] in needed:
            waves[row["id"]] = module._wave(row)
    config = module.load_frontend_config(Path("data/manifests/loopear_d_frontend.json"))
    mean = torch.tensor(config["mean"], dtype=torch.float32)
    std = torch.tensor(config["std"], dtype=torch.float32)
    mel_bank = module.build_mel()
    buckets: dict[int, list[torch.Tensor]] = {index: [] for index in range(len(module.STARSS23_CLASSES))}
    used = {index: 0 for index in range(len(module.STARSS23_CLASSES))}
    for onset in onsets:
        if onset["split"] != "train" or used[onset["class"]] >= module.MAX_ONSETS_PER_CLASS:
            continue
        wave = waves[onset["id"]]
        sample = module.place_crop(onset["onset_s"], onset["onset_s"] + 0.4, int(wave.shape[0]))
        if sample is None:
            sample = int(round(max(0.0, onset["onset_s"] - module.LEAD_S) * module.SR))
            sample = min(sample, max(int(wave.shape[0]) - module.CROP_SAMPLES, 0))
        patch = module.patch_index(onset["onset_s"], sample)
        if patch is None:
            continue
        mel, valid = module._crop_mel(wave, sample, mel_bank, mean, std, False)
        if not bool(valid[patch * module.PATCH_FRAMES : (patch + 1) * module.PATCH_FRAMES].any()):
            continue
        buckets[onset["class"]].append(_moment(encoder, mel, valid, patch))
        used[onset["class"]] += 1
    prototypes = {}
    for cls, vectors in buckets.items():
        if vectors:
            prototypes[cls] = torch.nn.functional.normalize(torch.stack(vectors, dim=0).mean(dim=0), dim=0)
    correct = 0
    ties = 0
    scored = 0
    reversed_correct = 0
    reversed_ties = 0
    reversed_scored = 0
    for pair in pairs:
        if pair["class_a"] not in prototypes or pair["class_b"] not in prototypes:
            continue
        wave = waves[pair["id"]]
        sample = module.place_crop(pair["earlier_s"], pair["later_s"], int(wave.shape[0]))
        if sample is None:
            continue
        early_patch = module.patch_index(pair["earlier_s"], sample)
        late_patch = module.patch_index(pair["later_s"], sample)
        rev_early = module.reversed_patch(pair["later_s"], sample)
        rev_late = module.reversed_patch(pair["earlier_s"], sample)
        if None in (early_patch, late_patch, rev_early, rev_late) or early_patch == late_patch:
            continue
        mel, valid = module._crop_mel(wave, sample, mel_bank, mean, std, False)
        decision = module._prefers_first(
            _moment(encoder, mel, valid, int(early_patch)),
            _moment(encoder, mel, valid, int(late_patch)),
            prototypes[pair["class_a"]],
            prototypes[pair["class_b"]],
        )
        scored += 1
        if decision is None:
            ties += 1
        elif decision:
            correct += 1
        rev_mel, rev_valid = module._crop_mel(wave, sample, mel_bank, mean, std, True)
        rev_decision = module._prefers_first(
            _moment(encoder, rev_mel, rev_valid, int(rev_early)),
            _moment(encoder, rev_mel, rev_valid, int(rev_late)),
            prototypes[pair["class_b"]],
            prototypes[pair["class_a"]],
        )
        reversed_scored += 1
        if rev_decision is None:
            reversed_ties += 1
        elif rev_decision:
            reversed_correct += 1
    return {
        "representation": "EMA teacher frequency-pooled encoder output after LayerNorm and L2-norm, at the onset patch",
        "prototypes": "mean of at most 32 train-split onset vectors per class",
        "n_prototype_onsets": used,
        "n_pairs_selected": len(pairs),
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
