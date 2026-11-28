from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from loopear.d.frontend import crop_logmel, normalize_logmel
from loopear.d.spec import D_MODEL, N_FRAMES, N_MELS
from loopear.d.fsd import MelCache
from loopear.d.model import LoopEarDSSL
from loopear.d.objective import broadcast_time_valid, patch_valid_from_frames


def fixed_project(x: torch.Tensor, width: int, seed: int) -> torch.Tensor:
    if x.shape[1] == width:
        return x
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    weight = torch.randn(x.shape[1], width, generator=gen)
    weight = weight / weight.norm(dim=0, keepdim=True).clamp_min(1e-6)
    return x @ weight


def macro_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score

    scores = []
    skipped = 0
    for cls in range(y_true.shape[1]):
        column = y_true[:, cls]
        if int(column.max()) == int(column.min()):
            skipped += 1
            continue
        scores.append(float(average_precision_score(column, y_score[:, cls])))
    if not scores:
        return {"macro_ap": None, "classes_scored": 0, "classes_skipped": skipped}
    return {
        "macro_ap": float(np.mean(scores)),
        "classes_scored": len(scores),
        "classes_skipped": skipped,
    }


def fit_linear_probe(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    steps: int,
    lr: float,
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    layer = nn.Linear(int(x_train.shape[1]), int(y_train.shape[1]))
    opt = torch.optim.AdamW(layer.parameters(), lr=lr, weight_decay=0.0)
    for _ in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(layer(x_train), y_train)
        loss.backward()
        opt.step()
    with torch.no_grad():
        train_score = torch.sigmoid(layer(x_train)).numpy()
        val_score = torch.sigmoid(layer(x_val)).numpy()
    return {
        "train": macro_average_precision(y_train.numpy(), train_score),
        "val": macro_average_precision(y_val.numpy(), val_score),
        "probe_params": int(sum(p.numel() for p in layer.parameters())),
        "steps": int(steps),
        "lr": float(lr),
        "seed": int(seed),
    }


def logmel_stats(logmel: torch.Tensor, frame_valid: torch.Tensor) -> torch.Tensor:
    valid = frame_valid.unsqueeze(-1).to(logmel.dtype)
    denom = valid.sum(dim=1).clamp(min=1.0)
    mean = (logmel * valid).sum(dim=1) / denom
    var = ((logmel - mean.unsqueeze(1)).pow(2) * valid).sum(dim=1) / denom
    return torch.cat([mean, var.sqrt()], dim=-1)


@torch.no_grad()
def encode_rows(
    model: LoopEarDSSL,
    cache: MelCache,
    indices: list[int],
) -> dict[str, torch.Tensor]:
    model.eval()
    cls_rows = []
    patch_rows = []
    stats_rows = []
    for index in indices:
        raw = cache._load(index)
        crop, valid = crop_logmel(raw, None)
        norm = normalize_logmel(crop, cache.mean, cache.std) * valid.unsqueeze(-1)
        mel = norm.unsqueeze(0)
        frame_valid = valid.unsqueeze(0)
        visible = broadcast_time_valid(patch_valid_from_frames(frame_valid))
        grid, cls = model.encoder(mel, visible)
        cls_rows.append(cls[0].detach().float().cpu())
        tokens = grid[0][visible[0]].detach().float().cpu()
        if tokens.shape[0] == 0:
            patch_rows.append(torch.zeros(D_MODEL))
        else:
            patch_rows.append(tokens.mean(dim=0))
        stats_rows.append(logmel_stats(norm.unsqueeze(0), frame_valid)[0].cpu())
    return {
        "cls": torch.stack(cls_rows, dim=0),
        "patch_mean": torch.stack(patch_rows, dim=0),
        "dsp": torch.stack(stats_rows, dim=0),
    }


def binary_accuracy(scores: np.ndarray, labels: np.ndarray) -> float:
    pred = (scores >= 0.5).astype(np.int32)
    return float((pred == labels.astype(np.int32)).mean())


def fit_binary(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    steps: int,
    lr: float,
    seed: int,
) -> dict[str, float]:
    torch.manual_seed(seed)
    layer = nn.Linear(int(x_train.shape[1]), 1)
    opt = torch.optim.AdamW(layer.parameters(), lr=lr, weight_decay=0.0)
    target = y_train.float().view(-1, 1)
    for _ in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(layer(x_train), target)
        loss.backward()
        opt.step()
    with torch.no_grad():
        score = torch.sigmoid(layer(x_val)).view(-1).numpy()
    labels = y_val.numpy().astype(np.int32)
    return {
        "accuracy": binary_accuracy(score, labels),
        "n_val": int(labels.shape[0]),
        "probe_params": float(sum(p.numel() for p in layer.parameters())),
        "chance": 0.5,
    }


def dsp_width() -> int:
    return N_MELS * 2


@torch.no_grad()
def encode_logmel_windows(
    model: LoopEarDSSL,
    logmel: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    max_windows: int,
) -> dict[str, torch.Tensor]:
    n_frames = int(logmel.shape[0])
    if n_frames <= N_FRAMES:
        starts = [0]
    else:
        starts = list(range(0, n_frames - N_FRAMES + 1, N_FRAMES))[: max(int(max_windows), 1)]
        if not starts:
            starts = [0]
    cls_rows = []
    patch_rows = []
    dsp_rows = []
    for start in starts:
        chunk = logmel[start : start + N_FRAMES]
        valid = torch.zeros(N_FRAMES, dtype=torch.bool)
        if chunk.shape[0] < N_FRAMES:
            padded = torch.zeros(N_FRAMES, N_MELS, dtype=logmel.dtype)
            if chunk.shape[0] > 0:
                padded[: chunk.shape[0]] = chunk
                valid[: chunk.shape[0]] = True
            chunk = padded
        else:
            valid[:] = True
        norm = normalize_logmel(chunk, mean, std) * valid.unsqueeze(-1)
        visible = broadcast_time_valid(patch_valid_from_frames(valid.unsqueeze(0)))
        grid, cls = model.encoder(norm.unsqueeze(0), visible)
        cls_rows.append(cls[0].detach().float().cpu())
        tokens = grid[0][visible[0]].detach().float().cpu()
        if tokens.shape[0] == 0:
            patch_rows.append(torch.zeros(D_MODEL))
        else:
            patch_rows.append(tokens.mean(dim=0))
        dsp_rows.append(logmel_stats(norm.unsqueeze(0), valid.unsqueeze(0))[0].cpu())
    return {
        "cls": torch.stack(cls_rows).mean(dim=0),
        "patch_mean": torch.stack(patch_rows).mean(dim=0),
        "dsp": torch.stack(dsp_rows).mean(dim=0),
    }
