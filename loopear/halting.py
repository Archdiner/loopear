from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from loopear.losses import MultiTaskHeads
from loopear.oracle import audiotetris_scores, evaluate_h6_gate

HALT_FEAT_DIM = 16


def _binary_entropy(probs: torch.Tensor) -> torch.Tensor:
    p = probs.clamp(1e-5, 1.0 - 1e-5)
    return -(p * p.log() + (1.0 - p) * (1.0 - p).log())


def _categorical_entropy(probs: torch.Tensor) -> torch.Tensor:
    p = probs.clamp(1e-5, 1.0)
    return -(p * p.log()).sum(dim=-1)


def _masked_mean_vec(x: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
    valid = (~pad).float()
    return (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)


def _masked_max_vec(x: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
    return x.masked_fill(pad, -1e9).amax(dim=1).clamp(min=0.0)


@torch.no_grad()
def extract_halting_features(
    e_seq: torch.Tensor,
    e_len: torch.Tensor,
    z_prev: torch.Tensor,
    z_curr: torch.Tensor,
    z_pad: torch.Tensor,
    heads: MultiTaskHeads,
    step_k: int,
    prev_probs: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    bsz, t_e, _ = e_seq.shape
    device = z_curr.device
    e_f = e_seq.float()
    e_diff_mean = torch.zeros(bsz, device=device)
    e_diff_std = torch.zeros(bsz, device=device)
    e_diff_max = torch.zeros(bsz, device=device)
    e_var_mean = torch.zeros(bsz, device=device)
    for b in range(bsz):
        n = min(int(e_len[b].item()), t_e)
        if n > 1:
            seq = e_f[b, :n]
            diffs = (seq[1:] - seq[:-1]).pow(2).mean(dim=-1)
            e_diff_mean[b] = diffs.mean()
            e_diff_std[b] = diffs.std(unbiased=False)
            e_diff_max[b] = diffs.max()
            e_var_mean[b] = seq.var(dim=0, unbiased=False).mean()
    zf_prev = z_prev.float()
    zf_curr = z_curr.float()
    dz_l2 = (zf_curr - zf_prev).norm(dim=-1)
    dz_norm = _masked_mean_vec(dz_l2, z_pad)
    cos_t = F.cosine_similarity(zf_curr, zf_prev, dim=-1)
    dz_cos = _masked_mean_vec(cos_t, z_pad)
    if zf_curr.shape[1] > 1:
        z_step = (zf_curr[:, 1:] - zf_curr[:, :-1]).norm(dim=-1)
        z_step_pad = z_pad[:, 1:] | z_pad[:, :-1]
        z_diff_mean = _masked_mean_vec(z_step, z_step_pad)
    else:
        z_diff_mean = torch.zeros(bsz, device=device)
    valid_u = (~z_pad).unsqueeze(-1).float()
    valid_cnt = valid_u.sum(dim=1).clamp(min=1.0)
    z_mu = (zf_curr * valid_u).sum(dim=1, keepdim=True) / valid_cnt.unsqueeze(1)
    z_var = (((zf_curr - z_mu).pow(2) * valid_u).sum(dim=1) / valid_cnt).mean(dim=-1)
    p_ev = torch.sigmoid(heads.event_logits(zf_curr))
    ev_ent = _masked_mean_vec(_binary_entropy(p_ev).mean(dim=-1), z_pad)
    ev_max = _masked_max_vec(p_ev.amax(dim=-1), z_pad)
    curr_probs: dict[str, torch.Tensor] = {"ev": p_ev}
    shifts: list[torch.Tensor] = []
    if prev_probs is not None and "ev" in prev_probs:
        shifts.append(_masked_mean_vec((p_ev - prev_probs["ev"]).abs().mean(dim=-1), z_pad))
    if heads.starss_head is not None:
        p_st = torch.sigmoid(heads.starss_logits(zf_curr))
        st_ent = _masked_mean_vec(_binary_entropy(p_st).mean(dim=-1), z_pad)
        curr_probs["st"] = p_st
        if prev_probs is not None and "st" in prev_probs:
            shifts.append(
                _masked_mean_vec((p_st - prev_probs["st"]).abs().mean(dim=-1), z_pad)
            )
    else:
        st_ent = torch.zeros(bsz, device=device)
    if heads.ovl_head is not None:
        p_ov = F.softmax(heads.overlap_logits(zf_curr), dim=-1)
        ov_ent = _masked_mean_vec(_categorical_entropy(p_ov), z_pad)
        p2 = p_ov[..., -1]
        ov_p2_mean = _masked_mean_vec(p2, z_pad)
        ov_p2_max = _masked_max_vec(p2, z_pad)
        curr_probs["ov"] = p_ov
        if prev_probs is not None and "ov" in prev_probs:
            shifts.append(
                _masked_mean_vec((p_ov - prev_probs["ov"]).abs().mean(dim=-1), z_pad)
            )
    else:
        ov_ent = torch.zeros(bsz, device=device)
        ov_p2_mean = torch.zeros(bsz, device=device)
        ov_p2_max = torch.zeros(bsz, device=device)
    if shifts:
        pred_shift = torch.stack(shifts, dim=0).mean(dim=0)
    else:
        pred_shift = torch.zeros(bsz, device=device)
    step_norm = torch.full((bsz,), float(step_k) / 4.0, device=device)
    feats = torch.stack(
        [
            torch.log1p(e_diff_mean.clamp(min=0.0)),
            torch.log1p(e_diff_std.clamp(min=0.0)),
            torch.log1p(e_diff_max.clamp(min=0.0)),
            torch.log1p(e_var_mean.clamp(min=0.0)),
            torch.log1p(dz_norm.clamp(min=0.0)),
            dz_cos,
            torch.log1p(z_diff_mean.clamp(min=0.0)),
            torch.log1p(z_var.clamp(min=0.0)),
            ev_ent,
            ev_max,
            st_ent,
            ov_ent,
            ov_p2_mean,
            ov_p2_max,
            pred_shift,
            step_norm,
        ],
        dim=-1,
    )
    return torch.nan_to_num(feats, nan=0.0, posinf=10.0, neginf=-10.0), curr_probs


class HaltingGate(nn.Module):
    def __init__(self, in_dim: int = HALT_FEAT_DIM, hidden_dim: int = 32, drop: float = 0.1):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.register_buffer("feat_mean", torch.zeros(in_dim))
        self.register_buffer("feat_std", torch.ones(in_dim))
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, 1),
        )

    def set_norm_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.feat_mean.copy_(mean.float())
        self.feat_std.copy_(std.float().clamp(min=1e-4))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        x = (feats.float() - self.feat_mean) / self.feat_std
        return self.net(x).squeeze(-1)

    @torch.no_grad()
    def continue_prob(self, feats: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(feats))


def fit_halting_gate(
    feats_train: np.ndarray,
    gains_train: np.ndarray,
    hidden_dim: int = 32,
    epochs: int = 240,
    lr: float = 3e-3,
    weight_decay: float = 2e-2,
    seed: int = 42,
    min_gain: float = 1e-4,
) -> HaltingGate:
    g = torch.Generator().manual_seed(seed)
    x_np = np.asarray(feats_train, dtype=np.float32)
    y_gain = np.asarray(gains_train, dtype=np.float32)
    if x_np.ndim != 2 or x_np.shape[0] == 0:
        raise ValueError("feats_train must be a non-empty 2D array")
    x_t = torch.from_numpy(x_np)
    y_bin = (y_gain > min_gain).astype(np.float32)
    y_t = torch.from_numpy(y_bin)
    w_t = torch.from_numpy(1.0 + 15.0 * np.abs(y_gain))
    w_t = w_t / w_t.mean().clamp(min=1e-6)
    torch.manual_seed(seed)
    gate = HaltingGate(in_dim=x_np.shape[1], hidden_dim=hidden_dim, drop=0.1)
    gate.set_norm_stats(x_t.mean(dim=0), x_t.std(dim=0, unbiased=False))
    opt = torch.optim.AdamW(gate.parameters(), lr=lr, weight_decay=weight_decay)
    n = x_t.shape[0]
    bsz = min(64, n)
    gate.train()
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, bsz):
            idx = perm[i : i + bsz]
            logits = gate(x_t[idx])
            loss = (
                F.binary_cross_entropy_with_logits(logits, y_t[idx], reduction="none")
                * w_t[idx]
            ).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    gate.eval()
    return gate


def _simulate_policy_at_threshold(
    p_cont_k1: np.ndarray,
    p_cont_k2: np.ndarray,
    scores_by_k: dict[int, np.ndarray],
    flops_by_k: dict[int, float],
    tau1: float,
    tau2: float,
    gate_gflops: float = 1e-5,
) -> dict[str, Any]:
    s1 = np.asarray(scores_by_k[1], dtype=np.float64)
    s2 = np.asarray(scores_by_k[2], dtype=np.float64)
    s4 = np.asarray(scores_by_k[4], dtype=np.float64)
    n = s1.shape[0]
    go2 = p_cont_k1 >= tau1
    go4 = go2 & (p_cont_k2 >= tau2)
    chosen_k = np.where(go4, 4, np.where(go2, 2, 1))
    chosen_score = np.where(chosen_k == 4, s4, np.where(chosen_k == 2, s2, s1))
    chosen_flops = np.where(
        chosen_k == 4,
        flops_by_k[4] + 2.0 * gate_gflops,
        np.where(
            chosen_k == 2,
            flops_by_k[2] + 2.0 * gate_gflops,
            flops_by_k[1] + gate_gflops,
        ),
    )
    overthinking = float(np.mean(chosen_score < (s1 - 1e-6)))
    improved = float(np.mean(chosen_score > (s1 + 1e-6)))
    return {
        "tau1": round(float(tau1), 4),
        "tau2": round(float(tau2), 4),
        "avg_k": round(float(np.mean(chosen_k)), 4),
        "avg_gflops": round(float(np.mean(chosen_flops)), 6),
        "score": round(float(np.mean(chosen_score)), 5),
        "overthinking_rate": round(overthinking, 4),
        "improved_rate": round(improved, 4),
        "depth_dist": {
            "1": round(float(np.mean(chosen_k == 1)), 4),
            "2": round(float(np.mean(chosen_k == 2)), 4),
            "4": round(float(np.mean(chosen_k == 4)), 4),
        },
        "n": int(n),
    }


def _interp_uniform_score(avg_k: float, u1: float, u2: float, u4: float) -> float:
    if avg_k <= 1.0:
        return u1
    if avg_k <= 2.0:
        a = (avg_k - 1.0) / 1.0
        return u1 + a * (u2 - u1)
    if avg_k <= 4.0:
        a = (avg_k - 2.0) / 2.0
        return u2 + a * (u4 - u2)
    return u4


def evaluate_halting_pareto(
    gate: HaltingGate,
    feats_k1: np.ndarray,
    feats_k2: np.ndarray,
    scores_by_k: dict[int, np.ndarray],
    flops_by_k: dict[int, float],
    e_seq: np.ndarray,
    gate_gflops: float = 1e-5,
    n_grid: int = 21,
) -> dict[str, Any]:
    if isinstance(e_seq, np.ndarray):
        e_rows = [np.asarray(e_seq[i], dtype=np.float32) for i in range(e_seq.shape[0])]
    else:
        e_rows = [np.asarray(row, dtype=np.float32) for row in e_seq]
    oracle_raw = evaluate_h6_gate(scores_by_k, flops_by_k, e_rows)
    oracle = {
        "uniform_score_lo": round(float(oracle_raw["uniform_score_lo"]), 5),
        "uniform_score_hi": round(float(oracle_raw["uniform_score_hi"]), 5),
        "oracle_savings": round(float(oracle_raw["oracle_savings"]), 4),
        "overthinking_rate_margin02": round(float(oracle_raw["overthinking_rate"]), 4),
        "improved_rate_margin02": round(float(oracle_raw["improved_rate"]), 4),
        "oracle_max_score": round(
            float(max(p["score"] for p in oracle_raw["oracle_curve"])), 5
        ),
        "h6_go": bool(oracle_raw["h6_go"]),
    }
    gate.eval()
    with torch.no_grad():
        p1 = gate.continue_prob(torch.from_numpy(np.asarray(feats_k1, dtype=np.float32))).cpu().numpy()
        p2 = gate.continue_prob(torch.from_numpy(np.asarray(feats_k2, dtype=np.float32))).cpu().numpy()
    at_scores = audiotetris_scores(e_rows)
    s1 = np.asarray(scores_by_k[1], dtype=np.float64)
    s2 = np.asarray(scores_by_k[2], dtype=np.float64)
    s4 = np.asarray(scores_by_k[4], dtype=np.float64)
    u1, u2, u4 = float(np.mean(s1)), float(np.mean(s2)), float(np.mean(s4))
    u4_overthinking = float(np.mean(s4 < (s1 - 1e-6)))
    u2_overthinking = float(np.mean(s2 < (s1 - 1e-6)))
    quantiles = np.linspace(0.0, 1.0, n_grid)
    learned_curve: list[dict[str, Any]] = []
    at_curve: list[dict[str, Any]] = []
    all_p = np.concatenate([p1, p2])
    taus = sorted(set([0.0, 1.01] + [float(np.quantile(all_p, q)) for q in quantiles]))
    for tau in taus:
        pt = _simulate_policy_at_threshold(
            p1, p2, scores_by_k, flops_by_k, tau, tau, gate_gflops=gate_gflops
        )
        u_match = _interp_uniform_score(pt["avg_k"], u1, u2, u4)
        pt["uniform_matched_score"] = round(u_match, 5)
        pt["delta_vs_uniform"] = round(pt["score"] - u_match, 5)
        learned_curve.append(pt)
    at_pairs: list[tuple[float, float]] = [
        (float(np.min(at_scores)) - 1e-6, float(np.min(at_scores)) - 1e-6),
        (float(np.max(at_scores)) + 1e-6, float(np.max(at_scores)) + 1e-6),
    ]
    for q in quantiles:
        t1 = float(np.quantile(at_scores, q))
        t2 = float(np.quantile(at_scores, min(1.0, float(q) + 0.25)))
        at_pairs.append((t1, t1))
        at_pairs.append((t1, t2))
    seen_at: set[tuple[float, float]] = set()
    for t1, t2 in sorted(at_pairs):
        key = (round(t1, 6), round(t2, 6))
        if key in seen_at:
            continue
        seen_at.add(key)
        pt = _simulate_policy_at_threshold(
            at_scores,
            at_scores,
            scores_by_k,
            flops_by_k,
            t1,
            t2,
            gate_gflops=0.0,
        )
        u_match = _interp_uniform_score(pt["avg_k"], u1, u2, u4)
        pt["uniform_matched_score"] = round(u_match, 5)
        pt["delta_vs_uniform"] = round(pt["score"] - u_match, 5)
        at_curve.append(pt)

    def _pick_near_k(curve: list[dict[str, Any]], target_k: float) -> dict[str, Any]:
        return min(curve, key=lambda r: abs(r["avg_k"] - target_k))

    best_learned = max(learned_curve, key=lambda r: r["score"])
    best_gain_vs_u = max(learned_curve, key=lambda r: r["delta_vs_uniform"])
    at_matched_best = _pick_near_k(at_curve, best_learned["avg_k"])
    mid_learned = _pick_near_k(learned_curve, 2.0)
    mid_at = _pick_near_k(at_curve, 2.0)
    learned_k_grid = [pt["avg_k"] for pt in learned_curve]
    learned_s_grid = [pt["score"] for pt in learned_curve]
    at_k_grid = [pt["avg_k"] for pt in at_curve]
    at_s_grid = [pt["score"] for pt in at_curve]
    l_order = np.argsort(learned_k_grid)
    a_order = np.argsort(at_k_grid)
    auc_learned = float(
        np.trapezoid(
            np.asarray(learned_s_grid)[l_order], np.asarray(learned_k_grid)[l_order]
        )
        / 3.0
    )
    auc_at = float(
        np.trapezoid(np.asarray(at_s_grid)[a_order], np.asarray(at_k_grid)[a_order])
        / 3.0
    )
    auc_uniform = float(((u1 + u2) * 0.5 * 1.0 + (u2 + u4) * 0.5 * 2.0) / 3.0)
    return {
        "uniform": {
            "k1": round(u1, 5),
            "k2": round(u2, 5),
            "k4": round(u4, 5),
            "k2_overthinking_rate": round(u2_overthinking, 4),
            "k4_overthinking_rate": round(u4_overthinking, 4),
        },
        "oracle": oracle,
        "best_learned": best_learned,
        "best_delta_vs_uniform": best_gain_vs_u,
        "audiotetris_matched_to_best": at_matched_best,
        "at_budget_k2": {
            "learned": mid_learned,
            "audiotetris": mid_at,
            "uniform_k2": round(u2, 5),
            "learned_minus_uniform_k2": round(mid_learned["score"] - u2, 5),
            "learned_minus_audiotetris": round(
                mid_learned["score"] - mid_at["score"], 5
            ),
        },
        "pareto_auc_avg_score": {
            "learned": round(auc_learned, 5),
            "audiotetris": round(auc_at, 5),
            "uniform": round(auc_uniform, 5),
            "learned_minus_uniform": round(auc_learned - auc_uniform, 5),
            "learned_minus_audiotetris": round(auc_learned - auc_at, 5),
        },
        "learned_beats_uniform_auc": bool(auc_learned > auc_uniform + 1e-5),
        "learned_beats_audiotetris_auc": bool(auc_learned > auc_at + 1e-5),
        "overthinking_reduction_vs_k4": round(
            u4_overthinking - best_learned["overthinking_rate"], 4
        ),
        "learned_curve": learned_curve,
        "audiotetris_curve": at_curve,
    }
