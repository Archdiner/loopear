import math
from typing import Any

import numpy as np
import torch

from loopear.encoder import SR
from loopear.h8data import DEV_CORPORA, iter_items, load_manifest, load_wav
from loopear.tiny import (
    TinyEncoder,
    averaged_constituent_loss,
    constituent_align_loss,
    distill_pair_loss,
    mix_waveforms,
    mixture_augment_loss,
    wav_to_frame_lengths,
)

GAINS: tuple[float, ...] = (-12.0, -6.0, 0.0, 6.0, 12.0)
CROP_S: float = 4.0
N_PAIRS: int = 512
BATCH: int = 4
STEPS: int = 400
LR: float = 1e-3
WEIGHT_DECAY: float = 0.01
SEED: int = 42
ARMS: tuple[str, ...] = ("distill", "mixture", "averaged", "align")


def train_items(root: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for corpus in DEV_CORPORA:
        manifest = load_manifest(root, corpus)
        for item in iter_items(manifest, "train"):
            row = dict(item)
            row["corpus"] = corpus
            rows.append(row)
    return rows


def _crop(root: str, item: dict[str, Any], start_s: float, crop_s: float) -> torch.Tensor:
    wav = load_wav(root, item, start_s, start_s + crop_s)
    n = int(round(crop_s * SR))
    if wav.shape[0] < n:
        wav = np.pad(wav, (0, n - wav.shape[0]))
    return torch.from_numpy(np.ascontiguousarray(wav[:n], dtype=np.float32))


def build_pair_bank(
    root: str,
    n_pairs: int = N_PAIRS,
    seed: int = SEED,
    crop_s: float = CROP_S,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    pool = [it for it in train_items(root) if float(it["duration_s"]) + 1e-6 >= crop_s]
    if len({(it["corpus"], str(it["group_id"])) for it in pool}) < 2:
        raise RuntimeError("Need at least two training recordings")
    pairs: list[dict[str, Any]] = []
    guard = 0
    while len(pairs) < n_pairs and guard < n_pairs * 200:
        guard += 1
        i, j = (int(v) for v in rng.integers(0, len(pool), size=2))
        left, right = pool[i], pool[j]
        if (left["corpus"], str(left["group_id"])) == (right["corpus"], str(right["group_id"])):
            continue
        gain = float(rng.choice(np.array(GAINS)))
        start_a = float(rng.random()) * (float(left["duration_s"]) - crop_s)
        start_b = float(rng.random()) * (float(right["duration_s"]) - crop_s)
        wav_a = _crop(root, left, start_a, crop_s)
        wav_b = _crop(root, right, start_b, crop_s)
        mixed = mix_waveforms(wav_a.unsqueeze(0), wav_b.unsqueeze(0), torch.tensor([gain]))[0]
        pairs.append(
            {
                "wav_a": wav_a,
                "wav_b": wav_b,
                "wav_m": mixed,
                "gain_db": gain,
                "corpus_a": left["corpus"],
                "corpus_b": right["corpus"],
                "group_a": str(left["group_id"]),
                "group_b": str(right["group_id"]),
            }
        )
    if len(pairs) < n_pairs:
        raise RuntimeError(f"Built {len(pairs)} pairs, needed {n_pairs}")
    return pairs


def _stack_rows(rows: list[torch.Tensor]) -> torch.Tensor:
    steps = min(int(row.shape[0]) for row in rows)
    return torch.stack([row[:steps] for row in rows], dim=0)


def cache_teachers(
    teacher: torch.nn.Module,
    pairs: list[dict[str, Any]],
    batch: int = 4,
) -> dict[str, list[torch.Tensor]]:
    cached: dict[str, list[torch.Tensor]] = {"a": [], "b": [], "m": []}
    keys = {"a": "wav_a", "b": "wav_b", "m": "wav_m"}
    teacher.eval()
    with torch.no_grad():
        for name, wave_key in keys.items():
            for start in range(0, len(pairs), batch):
                wav = torch.stack(
                    [pairs[i][wave_key] for i in range(start, min(len(pairs), start + batch))]
                )
                out = teacher(wavs=wav)["x"].float().cpu()
                for row in out:
                    cached[name].append(row)
                print(f"teacher {name} {min(len(pairs), start + batch)}/{len(pairs)}", flush=True)
    return cached


def _batch(
    pairs: list[dict[str, Any]],
    teachers: dict[str, list[torch.Tensor]],
    step: int,
    batch: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n = len(pairs)
    idx = [(step * batch + k) % n for k in range(batch)]
    wav_a = torch.stack([pairs[i]["wav_a"] for i in idx])
    wav_b = torch.stack([pairs[i]["wav_b"] for i in idx])
    wav_m = torch.stack([pairs[i]["wav_m"] for i in idx])
    teacher_a = _stack_rows([teachers["a"][i] for i in idx])
    teacher_b = _stack_rows([teachers["b"][i] for i in idx])
    teacher_m = _stack_rows([teachers["m"][i] for i in idx])
    student_len = wav_to_frame_lengths(torch.full((batch,), wav_a.shape[-1], dtype=torch.long))
    teacher_len = min(int(teacher_a.shape[1]), int(teacher_b.shape[1]), int(teacher_m.shape[1]))
    lengths = torch.minimum(student_len, torch.full_like(student_len, teacher_len))
    return wav_a, wav_b, wav_m, teacher_a, teacher_b, teacher_m, lengths


def arm_loss(
    model: TinyEncoder,
    arm: str,
    wav_a: torch.Tensor,
    wav_b: torch.Tensor,
    wav_m: torch.Tensor,
    teacher_a: torch.Tensor,
    teacher_b: torch.Tensor,
    teacher_m: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    if arm == "distill":
        return distill_pair_loss(
            model, model.encode(wav_a), model.encode(wav_b), teacher_a, teacher_b, lengths, lengths
        )
    if arm == "mixture":
        return mixture_augment_loss(model, model.encode(wav_m), teacher_m, lengths)
    if arm == "averaged":
        return averaged_constituent_loss(
            model, model.encode(wav_m), teacher_a, teacher_b, lengths
        )
    if arm == "align":
        return constituent_align_loss(
            model,
            model.encode(wav_m),
            model.encode(wav_a),
            model.encode(wav_b),
            teacher_a,
            teacher_b,
            lengths,
        )
    raise KeyError(arm)


def train_arm(
    pairs: list[dict[str, Any]],
    teachers: dict[str, list[torch.Tensor]],
    arm: str,
    steps: int = STEPS,
    batch: int = BATCH,
    lr: float = LR,
    weight_decay: float = WEIGHT_DECAY,
    seed: int = SEED,
    repeat_first: bool = False,
) -> tuple[TinyEncoder, dict[str, Any]]:
    torch.manual_seed(seed)
    model = TinyEncoder()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    before = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if not name.startswith("teacher_proj.")
    }
    losses: list[float] = []
    for step in range(steps):
        packed = _batch(pairs, teachers, 0 if repeat_first else step, batch)
        opt.zero_grad(set_to_none=True)
        loss = arm_loss(model, arm, *packed)
        if not torch.isfinite(loss):
            losses.append(float("nan"))
            break
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().item()))
        if step == 0 or step + 1 == steps or (step + 1) % 50 == 0:
            print(f"{arm} step {step + 1}/{steps} loss {losses[-1]:.4f}", flush=True)
    moved = 0.0
    for name, param in model.named_parameters():
        if name.startswith("teacher_proj."):
            continue
        moved = max(moved, float((param.detach() - before[name]).abs().max().item()))
    finite = bool(losses) and all(math.isfinite(v) for v in losses)
    return model, {
        "arm": arm,
        "steps": len(losses),
        "first_loss": None if not losses else losses[0],
        "last_loss": None if not losses else losses[-1],
        "finite": finite,
        "max_abs_param_move": moved,
        "losses": losses,
    }
