from typing import Any

import numpy as np


def audiotetris_scores(e_seq: list[np.ndarray]) -> np.ndarray:
    vals: list[float] = []
    for e in e_seq:
        arr = np.asarray(e, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] < 2:
            vals.append(0.0)
        else:
            diffs = np.diff(arr, axis=0)
            vals.append(float(np.mean(np.sum(diffs * diffs, axis=-1))))
    return np.asarray(vals, dtype=np.float64)


def knapsack_oracle(
    scores_by_k: dict[int, np.ndarray],
    flops_by_k: dict[int, float],
    budget: float,
) -> dict[str, Any]:
    ks = sorted(scores_by_k.keys(), key=lambda k: flops_by_k[k])
    base_k = ks[0]
    n = int(len(scores_by_k[base_k]))
    chosen = np.full(n, base_k, dtype=np.int64)
    total_flops = float(n) * float(flops_by_k[base_k])
    max_total_flops = float(n) * float(budget)
    upgrades: list[tuple[float, int, int, int, float, float]] = []
    for i in range(n):
        hull_k = [base_k]
        hull_c = [float(flops_by_k[base_k])]
        hull_s = [float(scores_by_k[base_k][i])]
        for k in ks[1:]:
            c = float(flops_by_k[k])
            s = float(scores_by_k[k][i])
            if s <= hull_s[-1]:
                continue
            while len(hull_k) >= 2:
                s_prev = (hull_s[-1] - hull_s[-2]) / max(hull_c[-1] - hull_c[-2], 1e-12)
                s_curr = (s - hull_s[-1]) / max(c - hull_c[-1], 1e-12)
                if s_curr >= s_prev:
                    hull_k.pop()
                    hull_c.pop()
                    hull_s.pop()
                else:
                    break
            hull_k.append(k)
            hull_c.append(c)
            hull_s.append(s)
        for j in range(len(hull_k) - 1):
            dc = hull_c[j + 1] - hull_c[j]
            ds = hull_s[j + 1] - hull_s[j]
            if dc > 0.0 and ds > 0.0:
                upgrades.append((ds / dc, i, hull_k[j], hull_k[j + 1], dc, ds))
    upgrades.sort(key=lambda item: item[0], reverse=True)
    for _, idx, k_from, k_to, dc, _ in upgrades:
        if chosen[idx] == k_from and total_flops + dc <= max_total_flops + 1e-9:
            chosen[idx] = k_to
            total_flops += dc
    selected_scores = np.array(
        [float(scores_by_k[int(chosen[i])][i]) for i in range(n)], dtype=np.float64
    )
    return {
        "budget": float(budget),
        "avg_flops": float(total_flops / max(n, 1)),
        "score": float(selected_scores.mean()) if n > 0 else 0.0,
        "depth_fracs": {int(k): float((chosen == k).mean()) for k in ks},
    }


def threshold_policy_score(
    signal: np.ndarray,
    scores_by_k: dict[int, np.ndarray],
    flops_by_k: dict[int, float],
    budget: float,
) -> dict[str, Any]:
    ks = sorted(scores_by_k.keys(), key=lambda k: flops_by_k[k])
    base_k = ks[0]
    sig = np.asarray(signal, dtype=np.float64)
    n = int(len(sig))
    order = np.argsort(-sig, kind="mergesort")
    chosen = np.full(n, base_k, dtype=np.int64)
    total_flops = float(n) * float(flops_by_k[base_k])
    max_total_flops = float(n) * float(budget)
    for k_prev, k_next in zip(ks[:-1], ks[1:]):
        dc = float(flops_by_k[k_next]) - float(flops_by_k[k_prev])
        for idx in order:
            if chosen[idx] == k_prev and total_flops + dc <= max_total_flops + 1e-9:
                chosen[idx] = k_next
                total_flops += dc
    selected_scores = np.array(
        [float(scores_by_k[int(chosen[i])][i]) for i in range(n)], dtype=np.float64
    )
    return {
        "budget": float(budget),
        "avg_flops": float(total_flops / max(n, 1)),
        "score": float(selected_scores.mean()) if n > 0 else 0.0,
        "depth_fracs": {int(k): float((chosen == k).mean()) for k in ks},
    }


def overthinking_rate(
    s1: np.ndarray,
    s4: np.ndarray,
    margin: float = 0.02,
) -> float:
    a = np.asarray(s1, dtype=np.float64)
    b = np.asarray(s4, dtype=np.float64)
    return float((b < a - margin).mean())


def query_independence_check(
    z_by_k: dict[int, np.ndarray],
    k_lo: int = 1,
    k_hi: int = 4,
    collapse_cos_threshold: float = 0.95,
) -> dict[str, Any]:
    z_lo = np.asarray(z_by_k[k_lo], dtype=np.float64)
    z_hi = np.asarray(z_by_k[k_hi], dtype=np.float64)
    if z_lo.ndim == 3:
        z_lo = z_lo.mean(axis=1)
        z_hi = z_hi.mean(axis=1)
    delta = z_hi - z_lo
    norms = np.linalg.norm(delta, axis=1)
    mean_norm = float(norms.mean())
    n = delta.shape[0]
    if n < 2 or mean_norm < 1e-8:
        return {
            "mean_delta_norm": mean_norm,
            "delta_pairwise_cos": 1.0 if mean_norm < 1e-8 else 0.0,
            "delta_var_ratio": 0.0,
            "collapsed": True,
        }
    unit = delta / np.clip(norms[:, None], 1e-12, None)
    sim = unit @ unit.T
    tri = sim[np.triu_indices(n, k=1)]
    mean_cos = float(tri.mean())
    centered = delta - delta.mean(axis=0, keepdims=True)
    var_ratio = float(
        np.mean(np.sum(centered * centered, axis=1))
        / max(float(np.mean(norms * norms)), 1e-12)
    )
    return {
        "mean_delta_norm": mean_norm,
        "delta_pairwise_cos": mean_cos,
        "delta_var_ratio": var_ratio,
        "collapsed": bool(mean_cos > collapse_cos_threshold),
    }


def evaluate_h6_gate(
    scores_by_k: dict[int, np.ndarray],
    flops_by_k: dict[int, float],
    e_seq: list[np.ndarray],
    z_by_k: dict[int, np.ndarray] | None = None,
    n_grid: int = 25,
) -> dict[str, Any]:
    ks = sorted(scores_by_k.keys())
    k_min, k_max = ks[0], ks[-1]
    s_lo = np.asarray(scores_by_k[k_min], dtype=np.float64)
    s_hi = np.asarray(scores_by_k[k_max], dtype=np.float64)
    delta_e = audiotetris_scores(e_seq)
    budgets = np.linspace(flops_by_k[k_min], flops_by_k[k_max], n_grid)
    oracle_curve = [knapsack_oracle(scores_by_k, flops_by_k, float(b)) for b in budgets]
    tetris_curve = [
        threshold_policy_score(delta_e, scores_by_k, flops_by_k, float(b))
        for b in budgets
    ]
    uniform_gain = float(s_hi.mean() - s_lo.mean())
    target = float(s_lo.mean() + 0.99 * max(uniform_gain, 0.0))
    b99 = min(
        (p["avg_flops"] for p in oracle_curve if p["score"] >= target - 1e-12),
        default=float(flops_by_k[k_max]),
    )
    savings = 1.0 - float(b99) / float(flops_by_k[k_max])
    ot_rate = overthinking_rate(s_lo, s_hi, margin=0.02)
    improved_rate = float((s_hi > s_lo + 0.02).mean())
    result: dict[str, Any] = {
        "uniform_score_lo": float(s_lo.mean()),
        "uniform_score_hi": float(s_hi.mean()),
        "uniform_gain": uniform_gain,
        "b99_flops": float(b99),
        "oracle_savings": float(savings),
        "overthinking_rate": ot_rate,
        "improved_rate": improved_rate,
        "oracle_curve": oracle_curve,
        "tetris_curve": tetris_curve,
        "h6_go": bool(savings >= 0.40 and improved_rate >= 0.25 and uniform_gain > 0.0),
    }
    if z_by_k is not None:
        result["collapse_check"] = query_independence_check(
            z_by_k, k_lo=k_min, k_hi=k_max
        )
    return result
