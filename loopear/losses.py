import math
import random

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve
from torch import nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def masked_mean_pool(z: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
    valid = (~pad).unsqueeze(-1).to(z.dtype)
    return (z * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)


def masked_max_pool(z: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
    return z.masked_fill(pad.unsqueeze(-1), -1e4).amax(dim=1)


class MultiTaskHeads(nn.Module):
    def __init__(
        self,
        d: int,
        n_ev: int = 15,
        d_spk: int = 192,
        n_spk: int = 120,
    ):
        super().__init__()
        self.d = d
        self.n_ev = n_ev
        self.d_spk = d_spk
        self.n_spk = n_spk
        self.ev_head = nn.Linear(d, n_ev)
        self.spk_attn = nn.Sequential(
            nn.Linear(d, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
        self.spk_proj = nn.Linear(2 * d, d_spk, bias=False)
        self.spk_ln = nn.LayerNorm(d_spk)
        self.spk_w = nn.Parameter(torch.empty(n_spk, d_spk))
        nn.init.xavier_uniform_(self.spk_w)

    def event_logits(self, z: torch.Tensor) -> torch.Tensor:
        return self.ev_head(z)

    def speaker_embed(self, z: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        scores = self.spk_attn(z).squeeze(-1).masked_fill(pad, -1e9)
        w = F.softmax(scores, dim=-1).masked_fill(pad, 0.0).unsqueeze(-1)
        mu = (w * z).sum(dim=1)
        var = (w * (z - mu.unsqueeze(1)).pow(2)).sum(dim=1)
        std = var.clamp(min=1e-6).sqrt()
        return F.normalize(self.spk_ln(self.spk_proj(torch.cat((mu, std), dim=-1))), dim=-1)


def aam_softmax_loss(
    emb: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    s: float = 30.0,
    m: float = 0.2,
) -> torch.Tensor:
    emb_norm = F.normalize(emb, dim=-1)
    w_norm = F.normalize(weight, dim=-1)
    cos_theta = F.linear(emb_norm, w_norm).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    sin_theta = torch.sqrt((1.0 - cos_theta.pow(2)).clamp(min=1e-6))
    cos_m = math.cos(m)
    sin_m = math.sin(m)
    th = math.cos(math.pi - m)
    mm = math.sin(math.pi - m) * m
    phi = torch.where(cos_theta > th, cos_theta * cos_m - sin_theta * sin_m, cos_theta - mm)
    one_hot = F.one_hot(labels.long(), num_classes=weight.shape[0]).to(cos_theta.dtype)
    logits = s * (one_hot * phi + (1.0 - one_hot) * cos_theta)
    return F.cross_entropy(logits, labels.long())


def multidepth_event_loss(
    states: dict[int, torch.Tensor],
    pad: torch.Tensor,
    heads: MultiTaskHeads,
    y_frame: torch.Tensor,
    y_clip: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
    clip_weight: float = 0.5,
) -> torch.Tensor:
    valid = (~pad).unsqueeze(-1).float()
    denom = valid.sum().clamp(min=1.0)
    wk = 1.0 / float(len(states))
    total = torch.zeros((), device=pad.device, dtype=torch.float32)
    for z in states.values():
        logits = heads.event_logits(z)
        frame_bce = (
            F.binary_cross_entropy_with_logits(
                logits, y_frame, pos_weight=pos_weight, reduction="none"
            )
            * valid
        ).sum() / denom
        clip_logits = masked_max_pool(logits, pad)
        clip_bce = F.binary_cross_entropy_with_logits(clip_logits, y_clip)
        total = total + wk * (frame_bce + clip_weight * clip_bce)
    return total


def multidepth_speaker_loss(
    states: dict[int, torch.Tensor],
    pad: torch.Tensor,
    heads: MultiTaskHeads,
    spk_ids: torch.Tensor,
    s: float = 30.0,
    m: float = 0.2,
) -> torch.Tensor:
    wk = 1.0 / float(len(states))
    total = torch.zeros((), device=pad.device, dtype=torch.float32)
    for z in states.values():
        emb = heads.speaker_embed(z, pad)
        total = total + wk * aam_softmax_loss(emb, heads.spk_w, spk_ids, s=s, m=m)
    return total


def multidepth_loss(
    model: nn.Module,
    heads: MultiTaskHeads,
    ev_batch: dict[str, torch.Tensor],
    spk_batch: dict[str, torch.Tensor],
    pos_weight: torch.Tensor | None = None,
    lam_spk: float = 0.5,
) -> dict[str, torch.Tensor]:
    sev, mev = model.forward_states(
        ev_batch["E"],
        ev_batch["lengths"],
        depths=model.depths,
        hidden_states=ev_batch.get("hidden_states"),
        layer_pool=ev_batch.get("layer_pool"),
    )
    l_ev = multidepth_event_loss(
        sev,
        mev,
        heads,
        ev_batch["Y_f"],
        ev_batch["Y_c"],
        pos_weight=pos_weight,
    )
    sspk, mspk = model.forward_states(
        spk_batch["E"],
        spk_batch["lengths"],
        depths=model.depths,
        hidden_states=spk_batch.get("hidden_states"),
        layer_pool=spk_batch.get("layer_pool"),
    )
    l_spk = multidepth_speaker_loss(sspk, mspk, heads, spk_batch["spk_id"])
    return {"total": l_ev + lam_spk * l_spk, "event": l_ev, "speaker": l_spk}


def compute_cross_chapter_eer(
    embeddings: np.ndarray,
    speaker_ids: np.ndarray,
    chapter_ids: np.ndarray,
) -> dict[str, float]:
    emb = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.clip(norms, 1e-8, None)
    sim = emb @ emb.T
    n = emb.shape[0]
    scores: list[float] = []
    labels: list[int] = []
    for i in range(n):
        for j in range(i + 1, n):
            same_spk = speaker_ids[i] == speaker_ids[j]
            same_chap = chapter_ids[i] == chapter_ids[j]
            if same_spk and same_chap:
                continue
            scores.append(float(sim[i, j]))
            labels.append(1 if same_spk else 0)
    if not scores or len(set(labels)) < 2:
        raise ValueError("Need both cross-chapter positive and negative speaker pairs")
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float(0.5 * (fpr[idx] + fnr[idx]))
    return {
        "eer": eer,
        "n_pairs": float(len(labels)),
        "n_pos": float(sum(labels)),
    }
