import argparse
import copy
import gc
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loopear.data import (
    PILOT_CORPUS_SOURCES,
    REALDESED_CLASSES,
    STARSS23_CLASSES,
    downsample_binary_targets,
    downsample_overlap_targets,
    rasterize_intervals_50hz,
    rasterize_overlap_50hz,
    rasterize_starss_frames_50hz,
    upsample_frame_preds,
)
from loopear.encoder import HOP_E, T_E_MAX, ShardedFeatureCache, load_mono_audio
from loopear.harness import (
    assert_zero_pretraining_overlap,
    write_result_json,
    write_run_manifest,
)
from loopear.losses import (
    MultiTaskHeads,
    compute_cross_chapter_eer,
    multidepth_event_loss,
    multidepth_overlap_loss,
    multidepth_speaker_loss,
    set_seed,
)
from loopear.models import build_arm
from loopear.oracle import evaluate_h6_gate
from loopear.stream import LoopEarStream
from scripts.eval_transfer import evaluate_frozen_transfer

ARM_SPECS: dict[str, tuple[str, dict[str, Any]]] = {
    "A_last": ("A_last", {"stride": 8}),
    "A_ml": ("A_ml", {"stride": 8}),
    "B_comp": ("B_comp", {"stride": 8}),
    "C_param": ("C_param", {"stride": 8}),
    "D": ("D", {"stride": 8, "max_k": 4}),
    "E_s8": ("E", {"stride": 8, "max_k": 4, "pool_mode": "avg"}),
    "E_s4": ("E", {"stride": 4, "max_k": 4, "pool_mode": "avg"}),
    "E_s8_conv": ("E", {"stride": 8, "max_k": 4, "pool_mode": "strided_conv"}),
}


def _load_split_arrays(
    cache_root: Path,
    cache_dir: str,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray | None]:
    cache = ShardedFeatureCache(cache_root)
    idx = cache.read_index(cache_dir)
    items = idx["items"]
    shards: dict[str, dict[str, np.ndarray]] = {}
    order: list[str] = []
    for it in items:
        sh = str(it["shard"])
        if sh not in shards:
            shards[sh] = cache.load_shard(cache_dir, sh)
            order.append(sh)
    e_all = np.concatenate([shards[s]["E"] for s in order], axis=0)
    l_all = np.concatenate([shards[s]["lengths"] for s in order], axis=0)
    lp_all = (
        np.concatenate([shards[s]["layer_pool"] for s in order], axis=0)
        if all("layer_pool" in shards[s] for s in order)
        else None
    )
    return items, e_all, l_all, lp_all


def _load_starss_events(label_path: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for line in Path(label_path).read_text(encoding="utf-8").splitlines():
        cols = [c.strip() for c in line.split(",")]
        if len(cols) >= 2 and cols[0].isdigit():
            out.append((int(cols[0]), int(cols[1])))
    return out


def _build_task_bundle(
    manifest_dir: Path,
    cache_root: Path,
    cache_index: dict[str, Any],
) -> dict[str, Any]:
    caches = cache_index["caches"]
    bundle: dict[str, Any] = {}

    for corpus in ("starss23", "realdesed", "dipco", "ami_sdm"):
        man = json.loads((manifest_dir / f"{corpus}.json").read_text(encoding="utf-8"))
        by_id = {it["id"]: it for it in man["items"]}
        bundle[corpus] = {"manifest": man}
        for split in ("train", "val"):
            items, e_arr, l_arr, lp_arr = _load_split_arrays(
                cache_root, caches[corpus][split]["cache_dir"]
            )
            n = len(items)
            entry: dict[str, Any] = {
                "items": items,
                "E": e_arr,
                "lengths": l_arr,
                "layer_pool": lp_arr,
            }
            if corpus == "starss23":
                n_cls = len(STARSS23_CLASSES)
                ev_cache: dict[str, list[tuple[int, int]]] = {}
                y_50 = np.zeros((n, T_E_MAX, n_cls), dtype=np.float32)
                y_clip = np.zeros((n, n_cls), dtype=np.float32)
                for i, it in enumerate(items):
                    sid = str(it["source_id"])
                    if sid not in ev_cache:
                        ev_cache[sid] = _load_starss_events(by_id[sid]["label_path"])
                    nf = int(l_arr[i])
                    mat = rasterize_starss_frames_50hz(
                        ev_cache[sid], float(it["start_s"]), nf, n_cls, hop_e=HOP_E
                    )
                    y_50[i, :nf] = mat
                    if nf > 0:
                        y_clip[i] = mat.max(axis=0)
                entry["y_50hz"] = y_50
                entry["y_clip"] = y_clip
            elif corpus == "realdesed":
                n_cls = len(REALDESED_CLASSES)
                y_50 = np.zeros((n, T_E_MAX, n_cls), dtype=np.float32)
                y_clip = np.zeros((n, n_cls), dtype=np.float32)
                for i, it in enumerate(items):
                    rec = by_id[str(it["source_id"])]
                    intervals = [
                        (float(a), float(b), int(c)) for a, b, c in rec["intervals"]
                    ]
                    nf = int(l_arr[i])
                    mat = rasterize_intervals_50hz(
                        intervals, float(it["start_s"]), nf, n_cls, hop_e=HOP_E
                    )
                    y_50[i, :nf] = mat
                    if nf > 0:
                        y_clip[i] = mat.max(axis=0)
                entry["y_50hz"] = y_50
                entry["y_clip"] = y_clip
            elif corpus == "dipco":
                y_50 = np.zeros((n, T_E_MAX), dtype=np.int64)
                for i, it in enumerate(items):
                    rec = by_id[str(it["source_id"])]
                    spk_ivs = [
                        (float(a), float(b), str(s))
                        for a, b, s in rec["speaker_intervals"]
                    ]
                    nf = int(l_arr[i])
                    y_50[i, :nf] = rasterize_overlap_50hz(
                        spk_ivs, float(it["start_s"]), nf, hop_e=HOP_E
                    )
                entry["y_50hz"] = y_50
            elif corpus == "ami_sdm":
                entry["speaker_idx"] = np.array(
                    [int(by_id[str(it["source_id"])]["speaker_idx"]) for it in items],
                    dtype=np.int64,
                )
                entry["speaker_id"] = np.array(
                    [str(by_id[str(it["source_id"])]["speaker_id"]) for it in items]
                )
                entry["meeting_id"] = np.array(
                    [str(by_id[str(it["source_id"])]["meeting_id"]) for it in items]
                )
            bundle[corpus][split] = entry

    for corpus in ("starss23", "realdesed"):
        tr = bundle[corpus]["train"]
        pos = np.zeros((tr["y_50hz"].shape[2],), dtype=np.float64)
        tot = 0.0
        for i in range(tr["E"].shape[0]):
            nf = int(tr["lengths"][i])
            pos += tr["y_50hz"][i, :nf].sum(axis=0)
            tot += float(nf)
        neg = np.maximum(tot - pos, 1.0)
        pw = np.clip(neg / np.maximum(pos, 1.0), 1.0, 10.0).astype(np.float32)
        bundle[corpus]["pos_weight"] = torch.from_numpy(pw)

    tr_dip = bundle["dipco"]["train"]
    counts = np.zeros((3,), dtype=np.float64)
    for i in range(tr_dip["E"].shape[0]):
        nf = int(tr_dip["lengths"][i])
        counts += np.bincount(tr_dip["y_50hz"][i, :nf], minlength=3)
    inv_sqrt = np.sqrt(counts.sum() / np.maximum(counts, 1.0))
    inv_sqrt = (inv_sqrt / inv_sqrt.mean()).astype(np.float32)
    bundle["dipco"]["class_weight"] = torch.from_numpy(inv_sqrt)
    return bundle


def _prepare_stride_targets(
    bundle: dict[str, Any],
    stride: int,
) -> dict[str, np.ndarray]:
    t_z_max = math.ceil(T_E_MAX / stride)
    out: dict[str, np.ndarray] = {}
    for corpus in ("starss23", "realdesed"):
        tr = bundle[corpus]["train"]
        n = int(tr["E"].shape[0])
        n_cls = int(tr["y_50hz"].shape[2])
        arr = np.zeros((n, t_z_max, n_cls), dtype=np.float32)
        for i in range(n):
            arr[i] = downsample_binary_targets(
                tr["y_50hz"][i], int(tr["lengths"][i]), stride, t_z_max
            )
        out[corpus] = arr
    tr_d = bundle["dipco"]["train"]
    n_d = int(tr_d["E"].shape[0])
    arr_d = np.zeros((n_d, t_z_max), dtype=np.int64)
    for i in range(n_d):
        arr_d[i] = downsample_overlap_targets(
            tr_d["y_50hz"][i], int(tr_d["lengths"][i]), stride, t_z_max
        )
    out["dipco"] = arr_d
    return out


def _sample_batch(
    split_data: dict[str, Any],
    idx: np.ndarray,
    use_lp: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    e_b = torch.from_numpy(split_data["E"][idx].astype(np.float32))
    l_b = torch.from_numpy(split_data["lengths"][idx].astype(np.int64))
    lp_b = (
        torch.from_numpy(split_data["layer_pool"][idx].astype(np.float32))
        if (use_lp and split_data["layer_pool"] is not None)
        else None
    )
    return e_b, l_b, lp_b


@torch.no_grad()
def _evaluate_arm_val(
    arm_name: str,
    model: torch.nn.Module,
    heads: MultiTaskHeads,
    bundle: dict[str, Any],
    batch_size: int = 32,
    collect_oracle_states: bool = False,
) -> dict[str, Any]:
    model.eval()
    heads.eval()
    use_lp = arm_name == "A_ml"
    depths = tuple(getattr(model, "depths", (1,)))
    stride = int(getattr(model, "stride", 8))

    def _run_dense(corpus: str, head_mode: str):
        val = bundle[corpus]["val"]
        n = int(val["E"].shape[0])
        probs_by_k: dict[int, list[np.ndarray]] = {k: [] for k in depths}
        z_by_k: dict[int, list[np.ndarray]] = {k: [] for k in depths}
        for b0 in range(0, n, batch_size):
            b1 = min(n, b0 + batch_size)
            idx = np.arange(b0, b1)
            e_b, l_b, lp_b = _sample_batch(val, idx, use_lp)
            states, z_pad = model.forward_states(
                e_b, l_b, depths=depths, layer_pool=lp_b
            )
            for k in depths:
                z = states[k]
                if head_mode == "starss":
                    p = torch.sigmoid(heads.starss_logits(z)).cpu().numpy()
                elif head_mode == "realdesed":
                    p = torch.sigmoid(heads.event_logits(z)).cpu().numpy()
                else:
                    p = torch.softmax(heads.overlap_logits(z), dim=-1).cpu().numpy()
                for i in range(b1 - b0):
                    nf = int(val["lengths"][b0 + i])
                    probs_by_k[k].append(upsample_frame_preds(p[i], stride, nf))
                    if collect_oracle_states:
                        valid_z = z[i, ~z_pad[i]].cpu().numpy()
                        z_by_k[k].append(
                            valid_z.mean(axis=0)
                            if valid_z.shape[0] > 0
                            else np.zeros((z.shape[2],), dtype=np.float32)
                        )
        return probs_by_k, z_by_k

    st_probs, st_z = _run_dense("starss23", "starss")
    rd_probs, rd_z = _run_dense("realdesed", "realdesed")
    dp_probs, dp_z = _run_dense("dipco", "dipco")

    ami_val = bundle["ami_sdm"]["val"]
    n_ami = int(ami_val["E"].shape[0])
    ami_embs: dict[int, list[np.ndarray]] = {k: [] for k in depths}
    for b0 in range(0, n_ami, batch_size):
        b1 = min(n_ami, b0 + batch_size)
        idx = np.arange(b0, b1)
        e_b, l_b, lp_b = _sample_batch(ami_val, idx, use_lp)
        states, z_pad = model.forward_states(e_b, l_b, depths=depths, layer_pool=lp_b)
        for k in depths:
            ami_embs[k].append(heads.speaker_embed(states[k], z_pad).cpu().numpy())

    metrics_by_k: dict[int, dict[str, Any]] = {}
    win_scores_by_k: dict[int, dict[str, np.ndarray]] = {}
    for k in depths:
        st_y_all = np.concatenate(
            [
                bundle["starss23"]["val"]["y_50hz"][i, : int(bundle["starss23"]["val"]["lengths"][i])]
                for i in range(len(st_probs[k]))
            ],
            axis=0,
        )
        st_p_all = np.concatenate(st_probs[k], axis=0)
        st_active = [c for c in range(st_y_all.shape[1]) if st_y_all[:, c].sum() > 0]
        st_macro_map = float(
            np.mean(
                [average_precision_score(st_y_all[:, c], st_p_all[:, c]) for c in st_active]
            )
        )
        st_micro_map = float(
            average_precision_score(st_y_all.ravel(), st_p_all.ravel())
        )
        st_micro_f1 = float(
            f1_score(
                st_y_all.ravel(), (st_p_all.ravel() >= 0.5).astype(np.int32), zero_division=0
            )
        )
        st_win_scores = np.array(
            [
                float(
                    f1_score(
                        bundle["starss23"]["val"]["y_50hz"][
                            i, : int(bundle["starss23"]["val"]["lengths"][i])
                        ].ravel(),
                        (st_probs[k][i].ravel() >= 0.5).astype(np.int32),
                        zero_division=0,
                    )
                )
                for i in range(len(st_probs[k]))
            ],
            dtype=np.float64,
        )

        rd_y_all = np.concatenate(
            [
                bundle["realdesed"]["val"]["y_50hz"][i, : int(bundle["realdesed"]["val"]["lengths"][i])]
                for i in range(len(rd_probs[k]))
            ],
            axis=0,
        )
        rd_p_all = np.concatenate(rd_probs[k], axis=0)
        rd_active = [c for c in range(rd_y_all.shape[1]) if rd_y_all[:, c].sum() > 0]
        rd_macro_map = float(
            np.mean(
                [average_precision_score(rd_y_all[:, c], rd_p_all[:, c]) for c in rd_active]
            )
        )
        rd_micro_map = float(
            average_precision_score(rd_y_all.ravel(), rd_p_all.ravel())
        )
        rd_micro_f1 = float(
            f1_score(
                rd_y_all.ravel(), (rd_p_all.ravel() >= 0.5).astype(np.int32), zero_division=0
            )
        )
        rd_win_scores = np.array(
            [
                float(
                    f1_score(
                        bundle["realdesed"]["val"]["y_50hz"][
                            i, : int(bundle["realdesed"]["val"]["lengths"][i])
                        ].ravel(),
                        (rd_probs[k][i].ravel() >= 0.5).astype(np.int32),
                        zero_division=0,
                    )
                )
                for i in range(len(rd_probs[k]))
            ],
            dtype=np.float64,
        )

        dp_y_all = np.concatenate(
            [
                bundle["dipco"]["val"]["y_50hz"][i, : int(bundle["dipco"]["val"]["lengths"][i])]
                for i in range(len(dp_probs[k]))
            ],
            axis=0,
        )
        dp_p_all = np.concatenate(dp_probs[k], axis=0)
        dp_pred_all = np.argmax(dp_p_all, axis=1)
        dp_macro_f1 = float(
            f1_score(dp_y_all, dp_pred_all, average="macro", zero_division=0)
        )
        dp_ovl_f1 = float(
            f1_score(
                (dp_y_all == 2).astype(np.int32),
                (dp_pred_all == 2).astype(np.int32),
                zero_division=0,
            )
        )
        dp_acc = float((dp_y_all == dp_pred_all).mean())
        dp_win_scores = np.array(
            [
                float(
                    f1_score(
                        bundle["dipco"]["val"]["y_50hz"][
                            i, : int(bundle["dipco"]["val"]["lengths"][i])
                        ],
                        np.argmax(dp_probs[k][i], axis=1),
                        average="macro",
                        zero_division=0,
                    )
                )
                for i in range(len(dp_probs[k]))
            ],
            dtype=np.float64,
        )

        emb_k = np.concatenate(ami_embs[k], axis=0)
        eer_info = compute_cross_chapter_eer(
            emb_k, ami_val["speaker_id"], ami_val["meeting_id"]
        )
        ami_eer = float(eer_info["eer"])
        ami_score = 1.0 - ami_eer

        composite = 0.25 * st_macro_map + 0.25 * rd_macro_map + 0.25 * dp_macro_f1 + 0.25 * ami_score
        metrics_by_k[k] = {
            "starss_frame_macro_map": round(st_macro_map, 4),
            "starss_frame_micro_map": round(st_micro_map, 4),
            "starss_frame_micro_f1": round(st_micro_f1, 4),
            "realdesed_frame_macro_map": round(rd_macro_map, 4),
            "realdesed_frame_micro_map": round(rd_micro_map, 4),
            "realdesed_frame_micro_f1": round(rd_micro_f1, 4),
            "dipco_macro_f1": round(dp_macro_f1, 4),
            "dipco_overlap2_f1": round(dp_ovl_f1, 4),
            "dipco_acc": round(dp_acc, 4),
            "ami_eer": round(ami_eer, 4),
            "ami_1_minus_eer": round(ami_score, 4),
            "composite": round(composite, 4),
        }
        win_scores_by_k[k] = {
            "starss23": st_win_scores,
            "realdesed": rd_win_scores,
            "dipco": dp_win_scores,
            "combined_dense": np.concatenate(
                [st_win_scores, rd_win_scores, dp_win_scores], axis=0
            ),
            "ami_emb": emb_k,
        }

    out: dict[str, Any] = {
        "metrics_by_k": metrics_by_k,
        "win_scores_by_k": win_scores_by_k,
    }
    if collect_oracle_states:
        out["oracle_z_by_k"] = {
            k: np.stack(st_z[k] + rd_z[k] + dp_z[k], axis=0) for k in depths
        }
    return out


def _paired_bootstrap_ci(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    n = int(diff.shape[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        "delta": round(float(diff.mean()), 4),
        "ci95_lo": round(float(lo), 4),
        "ci95_hi": round(float(hi), 4),
        "ci_gt_zero": bool(lo > 0.0),
    }


def _paired_bootstrap_eer_ci(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    spk_ids: np.ndarray,
    mtg_ids: np.ndarray,
    n_boot: int = 400,
    seed: int = 42,
) -> dict[str, float]:
    base_a = 1.0 - float(compute_cross_chapter_eer(emb_a, spk_ids, mtg_ids)["eer"])
    base_b = 1.0 - float(compute_cross_chapter_eer(emb_b, spk_ids, mtg_ids)["eer"])
    n = int(emb_a.shape[0])
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            sa = 1.0 - float(
                compute_cross_chapter_eer(emb_a[idx], spk_ids[idx], mtg_ids[idx])["eer"]
            )
            sb = 1.0 - float(
                compute_cross_chapter_eer(emb_b[idx], spk_ids[idx], mtg_ids[idx])["eer"]
            )
            deltas.append(sa - sb)
        except ValueError:
            continue
    arr = np.asarray(deltas, dtype=np.float64)
    lo, hi = np.percentile(arr, [2.5, 97.5]) if arr.size > 0 else (0.0, 0.0)
    return {
        "delta": round(float(base_a - base_b), 4),
        "ci95_lo": round(float(lo), 4),
        "ci95_hi": round(float(hi), 4),
        "ci_gt_zero": bool(lo > 0.0),
    }


def _train_single_arm(
    arm_name: str,
    arm_type: str,
    ref_kwargs: dict[str, Any],
    bundle: dict[str, Any],
    epochs: int,
    steps_per_epoch: int,
    batch_size: int,
    lr: float,
    seed: int,
    ckpt_dir: Path,
) -> dict[str, Any]:
    set_seed(seed)
    model = build_arm(arm_type, **ref_kwargs)
    n_spk = int(bundle["ami_sdm"]["manifest"]["n_train_speakers"])
    head_kwargs = {
        "d": int(model.out_dim),
        "n_ev": len(REALDESED_CLASSES),
        "d_spk": 192,
        "n_spk": n_spk,
        "n_starss": len(STARSS23_CLASSES),
        "n_ovl": 3,
    }
    heads = MultiTaskHeads(**head_kwargs)
    init_weights = {
        name: p.detach().clone()
        for name, p in list(model.named_parameters()) + list(heads.named_parameters())
    }

    run_id = f"stage1_pilot_{arm_name}_s{seed}"
    write_run_manifest(
        run_id,
        {
            "arm_name": arm_name,
            "arm_type": arm_type,
            "refiner_kwargs": ref_kwargs,
            "head_kwargs": head_kwargs,
            "epochs": epochs,
            "steps_per_epoch": steps_per_epoch,
            "batch_size": batch_size,
            "lr": lr,
            "seed": seed,
            "datasets": ["starss23", "realdesed", "dipco", "ami_sdm"],
        },
        ledger_dir="ledger",
    )

    stride = int(getattr(model, "stride", 8))
    use_lp = arm_name == "A_ml"
    stride_targets = _prepare_stride_targets(bundle, stride)
    params = list(model.parameters()) + list(heads.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
    total_steps = epochs * steps_per_epoch
    warmup = max(1, int(0.1 * total_steps))

    def _lr_lambda(step: int) -> float:
        if step < warmup:
            return float(step + 1) / float(warmup)
        prog = float(step - warmup) / max(float(total_steps - warmup), 1.0)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
    rng = np.random.default_rng(seed)

    st_tr = bundle["starss23"]["train"]
    rd_tr = bundle["realdesed"]["train"]
    dp_tr = bundle["dipco"]["train"]
    am_tr = bundle["ami_sdm"]["train"]
    st_pw = bundle["starss23"]["pos_weight"]
    rd_pw = bundle["realdesed"]["pos_weight"]
    dp_cw = bundle["dipco"]["class_weight"]

    t0 = time.perf_counter()
    best_sc = -1.0
    best_state: dict[str, Any] | None = None
    best_epoch = 0
    history: list[dict[str, float]] = []

    for ep in range(1, epochs + 1):
        model.train()
        heads.train()
        ep_loss = 0.0
        for _ in range(steps_per_epoch):
            opt.zero_grad(set_to_none=True)

            idx_st = rng.integers(0, st_tr["E"].shape[0], size=batch_size)
            e_st, l_st, lp_st = _sample_batch(st_tr, idx_st, use_lp)
            s_st, m_st = model.forward_states(
                e_st, l_st, depths=model.depths, layer_pool=lp_st
            )
            tz_st = m_st.shape[1]
            y_st_f = torch.from_numpy(stride_targets["starss23"][idx_st, :tz_st])
            y_st_c = torch.from_numpy(st_tr["y_clip"][idx_st])
            loss_st = multidepth_event_loss(
                s_st, m_st, heads, y_st_f, y_st_c, pos_weight=st_pw, head_name="starss"
            )

            idx_rd = rng.integers(0, rd_tr["E"].shape[0], size=batch_size)
            e_rd, l_rd, lp_rd = _sample_batch(rd_tr, idx_rd, use_lp)
            s_rd, m_rd = model.forward_states(
                e_rd, l_rd, depths=model.depths, layer_pool=lp_rd
            )
            tz_rd = m_rd.shape[1]
            y_rd_f = torch.from_numpy(stride_targets["realdesed"][idx_rd, :tz_rd])
            y_rd_c = torch.from_numpy(rd_tr["y_clip"][idx_rd])
            loss_rd = multidepth_event_loss(
                s_rd, m_rd, heads, y_rd_f, y_rd_c, pos_weight=rd_pw, head_name="ev"
            )

            idx_dp = rng.integers(0, dp_tr["E"].shape[0], size=batch_size)
            e_dp, l_dp, lp_dp = _sample_batch(dp_tr, idx_dp, use_lp)
            s_dp, m_dp = model.forward_states(
                e_dp, l_dp, depths=model.depths, layer_pool=lp_dp
            )
            tz_dp = m_dp.shape[1]
            y_dp = torch.from_numpy(stride_targets["dipco"][idx_dp, :tz_dp])
            loss_dp = multidepth_overlap_loss(
                s_dp, m_dp, heads, y_dp, class_weight=dp_cw
            )

            idx_am = rng.integers(0, am_tr["E"].shape[0], size=batch_size)
            e_am, l_am, lp_am = _sample_batch(am_tr, idx_am, use_lp)
            s_am, m_am = model.forward_states(
                e_am, l_am, depths=model.depths, layer_pool=lp_am
            )
            y_am = torch.from_numpy(am_tr["speaker_idx"][idx_am])
            loss_am = multidepth_speaker_loss(s_am, m_am, heads, y_am)

            total = loss_st + loss_rd + loss_dp + 0.5 * loss_am
            if not torch.isfinite(total):
                raise RuntimeError(f"Non-finite loss in arm {arm_name} epoch {ep}")
            total.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            ep_loss += float(total.item())

        val_res = _evaluate_arm_val(arm_name, model, heads, bundle)
        k_max = max(model.depths)
        sc = float(val_res["metrics_by_k"][k_max]["composite"])
        history.append(
            {
                "epoch": ep,
                "train_loss": round(ep_loss / steps_per_epoch, 4),
                "val_composite": round(sc, 4),
            }
        )
        if sc > best_sc:
            best_sc = sc
            best_epoch = ep
            best_state = {
                "refiner": copy.deepcopy(model.state_dict()),
                "heads": copy.deepcopy(heads.state_dict()),
            }

    train_wall_s = time.perf_counter() - t0
    assert best_state is not None
    model.load_state_dict(best_state["refiner"], strict=True)
    heads.load_state_dict(best_state["heads"], strict=True)

    max_delta = max(
        float((p.detach() - init_weights[name]).abs().max().item())
        for name, p in list(model.named_parameters()) + list(heads.named_parameters())
    )
    if max_delta <= 1e-4:
        raise AssertionError(f"Weight movement check failed for {arm_name}: {max_delta}")

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"stage1_arm_{arm_name}.pt"
    payload = {
        "arm_name": arm_name,
        "arm_type": arm_type,
        "refiner_kwargs": ref_kwargs,
        "head_kwargs": head_kwargs,
        "labels": list(REALDESED_CLASSES),
        "refiner": model.state_dict(),
        "heads": heads.state_dict(),
        "best_epoch": best_epoch,
        "best_composite": best_sc,
    }
    torch.save(payload, ckpt_path)

    reloaded_blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_re = build_arm(arm_type, **reloaded_blob["refiner_kwargs"])
    model_re.load_state_dict(reloaded_blob["refiner"], strict=True)
    model_re.eval()
    model.eval()
    sample_idx = np.arange(min(4, st_tr["E"].shape[0]))
    e_s, l_s, lp_s = _sample_batch(st_tr, sample_idx, use_lp)
    with torch.no_grad():
        st_a, _ = model.forward_states(e_s, l_s, depths=(max(model.depths),), layer_pool=lp_s)
        st_b, _ = model_re.forward_states(e_s, l_s, depths=(max(model.depths),), layer_pool=lp_s)
    reload_diff = float((st_a[max(model.depths)] - st_b[max(model.depths)]).abs().max().item())
    if reload_diff >= 1e-5:
        raise AssertionError(
            f"Reload parity check failed for {arm_name}: max_abs_diff={reload_diff}"
        )

    collect_oracle = arm_name in ("E_s8", "E_s4")
    final_eval = _evaluate_arm_val(
        arm_name, model, heads, bundle, collect_oracle_states=collect_oracle
    )
    print(
        f"[train] {arm_name:10s} best_ep={best_epoch:2d} wall={train_wall_s:5.1f}s "
        f"delta={max_delta:.4f} reload_diff={reload_diff:.1e} "
        + " | ".join(
            f"k={k}: comp={m['composite']:.4f} (ST={m['starss_frame_macro_map']:.3f}, "
            f"RD={m['realdesed_frame_macro_map']:.3f}, DP={m['dipco_macro_f1']:.3f}, "
            f"AMI_1-EER={m['ami_1_minus_eer']:.3f})"
            for k, m in final_eval["metrics_by_k"].items()
        ),
        flush=True,
    )
    return {
        "arm_name": arm_name,
        "arm_type": arm_type,
        "model": model,
        "heads": heads,
        "ckpt_path": str(ckpt_path),
        "best_epoch": best_epoch,
        "train_wall_s": round(train_wall_s, 2),
        "max_weight_delta": round(max_delta, 6),
        "reload_max_abs_diff": reload_diff,
        "params_ref": sum(p.numel() for p in model.parameters()),
        "params_heads": sum(p.numel() for p in heads.parameters()),
        "history": history,
        "eval": final_eval,
    }


def _verify_live_stream(
    ckpt_path: Path,
    manifest_dir: Path,
) -> dict[str, Any]:
    stream = LoopEarStream.from_checkpoint(ckpt_path, depth=2, win_s=4.0, step_s=1.0, device="cpu")
    st_man = json.loads((manifest_dir / "starss23.json").read_text(encoding="utf-8"))
    rd_man = json.loads((manifest_dir / "realdesed.json").read_text(encoding="utf-8"))
    am_man = json.loads((manifest_dir / "ami_sdm.json").read_text(encoding="utf-8"))

    st_val = [it for it in st_man["items"] if it["split"] == "val"][0]
    rd_val = [it for it in rd_man["items"] if it["split"] == "val"][0]
    am_vals = [it for it in am_man["items"] if it["split"] == "val"]

    by_spk: dict[str, list[dict[str, Any]]] = {}
    for it in am_vals:
        by_spk.setdefault(it["speaker_id"], []).append(it)
    spks = sorted(by_spk.keys())
    same_a = by_spk[spks[0]][0]
    same_b = next(
        (x for x in by_spk[spks[0]][1:] if x["meeting_id"] != same_a["meeting_id"]),
        by_spk[spks[0]][1],
    )
    diff_b = by_spk[spks[1]][0]

    wav_st = load_mono_audio(st_val["audio_path"], start_s=0.0, end_s=12.0)
    wav_rd = load_mono_audio(rd_val["audio_path"], start_s=0.0, end_s=6.0)
    wav_sa = load_mono_audio(same_a["audio_path"])
    wav_sb = load_mono_audio(same_b["audio_path"])
    wav_db = load_mono_audio(diff_b["audio_path"])

    encode_checks: dict[str, Any] = {}
    for k in (1, 2, 4):
        out = stream.encode(wav_rd[: 4 * 16000], k=k)
        encode_checks[str(k)] = {
            "z_shape": list(out["z"].shape),
            "ev_mean": round(float(out["ev"].mean().item()), 4),
            "vec_norm": round(float(out["vec"].norm().item()), 4),
            "spk_norm": round(float(out["spk"].norm().item()), 4),
            "all_finite": bool(
                torch.isfinite(out["z"]).all()
                and torch.isfinite(out["ev"]).all()
                and torch.isfinite(out["spk"]).all()
            ),
        }

    enr_rd = stream.enroll("realdesed_val_clip", wav_rd[: 4 * 16000])
    enr_st = stream.enroll("starss_val_clip", wav_st[: 4 * 16000])
    cmp_same = stream.compare(wav_sa, wav_sb, k=4)
    cmp_diff = stream.compare(wav_sa, wav_db, k=4)

    stream.set_depth(4)
    chunk_n = 16000
    state: dict[str, Any] = {}
    for i in range(0, wav_st.shape[0], chunk_n):
        state = stream.feed(wav_st[i : i + chunk_n])
    return {
        "checkpoint": str(ckpt_path),
        "val_files_used": {
            "starss23": st_val["id"],
            "realdesed": rd_val["id"],
            "ami_same_a": same_a["id"],
            "ami_same_b": same_b["id"],
            "ami_diff_b": diff_b["id"],
        },
        "encode_by_depth": encode_checks,
        "enroll": {"realdesed": enr_rd, "starss": enr_st},
        "compare_same_speaker_cross_meeting": cmp_same,
        "compare_diff_speaker": cmp_diff,
        "stream_state": {
            "depth": state["depth"],
            "windows_processed": state["windows_processed"],
            "enrolled_hits": state["enrolled_hits"],
            "n_events": len(state["events"]),
            "n_recurring_clusters": len(state["recurring"]),
            "recurring_top3": state["recurring"][:3],
            "mean_ms": state["mean_ms"],
            "rtf": state["rtf"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", default="data/manifests")
    parser.add_argument("--cache-root", default="cache/stage1_pilot")
    parser.add_argument("--ckpt-dir", default="checkpoints")
    parser.add_argument("--stage0-profile", default="results/stage0_profile.json")
    parser.add_argument("--out", default="results/stage1_pilot.json")
    parser.add_argument(
        "--store-copy",
        default="/cursor/stores/bc-01a213ed-7a8a-7981-b509-7012340964c4/internal/stage1_pilot.json",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps-per-epoch", type=int, default=28)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--skip-diagnostic", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(int(args.threads))
    assert_zero_pretraining_overlap(PILOT_CORPUS_SOURCES)

    manifest_dir = Path(args.manifest_dir)
    cache_root = Path(args.cache_root)
    ckpt_dir = Path(args.ckpt_dir)
    cache_index = json.loads(
        (manifest_dir / "pilot_cache_index.json").read_text(encoding="utf-8")
    )
    stage0 = json.loads(Path(args.stage0_profile).read_text(encoding="utf-8"))

    print("[pilot] Loading cached Stage 1 features and 50 Hz targets ...", flush=True)
    bundle = _build_task_bundle(manifest_dir, cache_root, cache_index)

    arm_order = ["A_last", "A_ml", "B_comp", "C_param", "D", "E_s8", "E_s4"]
    if not args.skip_diagnostic:
        arm_order.append("E_s8_conv")

    trained: dict[str, dict[str, Any]] = {}
    for arm_name in arm_order:
        arm_type, ref_kwargs = ARM_SPECS[arm_name]
        trained[arm_name] = _train_single_arm(
            arm_name=arm_name,
            arm_type=arm_type,
            ref_kwargs=ref_kwargs,
            bundle=bundle,
            epochs=int(args.epochs),
            steps_per_epoch=int(args.steps_per_epoch),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            seed=int(args.seed),
            ckpt_dir=ckpt_dir,
        )
        if arm_name == "A_ml":
            for corpus in ("starss23", "realdesed", "dipco", "ami_sdm"):
                for split in ("train", "val"):
                    bundle[corpus][split]["layer_pool"] = None
            gc.collect()

    contrasts_spec = [
        ("E_s8@4 - E_s8@1", ("E_s8", 4), ("E_s8", 1)),
        ("E_s4@4 - E_s4@1", ("E_s4", 4), ("E_s4", 1)),
        ("E_s8@4 - D@4", ("E_s8", 4), ("D", 4)),
        ("E_s4@4 - D@4", ("E_s4", 4), ("D", 4)),
        ("E_s8@4 - A_ml@1", ("E_s8", 4), ("A_ml", 1)),
        ("E_s4@4 - A_ml@1", ("E_s4", 4), ("A_ml", 1)),
        ("E_s8@4 - B_comp@1", ("E_s8", 4), ("B_comp", 1)),
        ("E_s4@4 - B_comp@1", ("E_s4", 4), ("B_comp", 1)),
        ("E_s4@4 - E_s8@4", ("E_s4", 4), ("E_s8", 4)),
    ]
    if "E_s8_conv" in trained:
        contrasts_spec.append(("E_s8_conv@4 - E_s8@4", ("E_s8_conv", 4), ("E_s8", 4)))

    ami_val = bundle["ami_sdm"]["val"]
    paired_cis: dict[str, Any] = {}
    for label, (a_arm, a_k), (b_arm, b_k) in contrasts_spec:
        wa = trained[a_arm]["eval"]["win_scores_by_k"][a_k]
        wb = trained[b_arm]["eval"]["win_scores_by_k"][b_k]
        paired_cis[label] = {
            "combined_dense": _paired_bootstrap_ci(
                wa["combined_dense"], wb["combined_dense"], seed=int(args.seed)
            ),
            "starss23": _paired_bootstrap_ci(
                wa["starss23"], wb["starss23"], seed=int(args.seed) + 1
            ),
            "realdesed": _paired_bootstrap_ci(
                wa["realdesed"], wb["realdesed"], seed=int(args.seed) + 2
            ),
            "dipco": _paired_bootstrap_ci(
                wa["dipco"], wb["dipco"], seed=int(args.seed) + 3
            ),
            "ami_1_minus_eer": _paired_bootstrap_eer_ci(
                wa["ami_emb"],
                wb["ami_emb"],
                ami_val["speaker_id"],
                ami_val["meeting_id"],
                seed=int(args.seed) + 4,
            ),
            "composite_point_delta": round(
                float(trained[a_arm]["eval"]["metrics_by_k"][a_k]["composite"])
                - float(trained[b_arm]["eval"]["metrics_by_k"][b_k]["composite"]),
                4,
            ),
        }

    e_seq_all: list[np.ndarray] = []
    for corpus in ("starss23", "realdesed", "dipco"):
        val = bundle[corpus]["val"]
        for i in range(int(val["E"].shape[0])):
            nf = int(val["lengths"][i])
            e_seq_all.append(val["E"][i, :nf].astype(np.float32))

    h6_oracle: dict[str, Any] = {}
    for arm_name in ("E_s8", "E_s4"):
        flops_raw = stage0["arms"][arm_name]["gflops_10s"]["refine"]
        flops_by_k = {int(k): float(v) for k, v in flops_raw.items()}
        scores_by_k = {
            k: trained[arm_name]["eval"]["win_scores_by_k"][k]["combined_dense"]
            for k in (1, 2, 4)
        }
        z_by_k = trained[arm_name]["eval"]["oracle_z_by_k"]
        gate = evaluate_h6_gate(scores_by_k, flops_by_k, e_seq_all, z_by_k=z_by_k)
        per_task_gate: dict[str, Any] = {}
        offset = 0
        for corpus in ("starss23", "realdesed", "dipco"):
            n_c = int(bundle[corpus]["val"]["E"].shape[0])
            task_scores = {
                k: trained[arm_name]["eval"]["win_scores_by_k"][k][corpus]
                for k in (1, 2, 4)
            }
            tg = evaluate_h6_gate(
                task_scores, flops_by_k, e_seq_all[offset : offset + n_c]
            )
            per_task_gate[corpus] = {
                "uniform_score_lo": tg["uniform_score_lo"],
                "uniform_score_hi": tg["uniform_score_hi"],
                "uniform_gain": tg["uniform_gain"],
                "oracle_savings": tg["oracle_savings"],
                "improved_rate": tg["improved_rate"],
                "overthinking_rate": tg["overthinking_rate"],
                "h6_go": tg["h6_go"],
            }
            offset += n_c
        h6_oracle[arm_name] = {
            "flops_by_k_gflops": flops_by_k,
            "combined_dense": gate,
            "per_task": per_task_gate,
        }
        print(
            f"[oracle] {arm_name}: gain={gate['uniform_gain']:+.4f}, "
            f"savings={gate['oracle_savings']:.3f}, improved={gate['improved_rate']:.3f}, "
            f"overthinking={gate['overthinking_rate']:.3f}, h6_go={gate['h6_go']}",
            flush=True,
        )

    del e_seq_all, bundle
    gc.collect()

    print("[transfer] Running frozen transfer evaluations + shuffled controls ...", flush=True)
    transfer_results: dict[str, Any] = {}
    for arm_name in arm_order:
        res_t = evaluate_frozen_transfer(
            arm_name=arm_name,
            model=trained[arm_name]["model"],
            heads=trained[arm_name]["heads"],
            manifest_dir=manifest_dir,
            cache_root=cache_root,
            cache_index=cache_index,
            seed=int(args.seed),
        )
        transfer_results[arm_name] = res_t
        k_max = str(max(getattr(trained[arm_name]["model"], "depths", (1,))))
        m = res_t["by_depth"][k_max]
        print(
            f"[transfer] {arm_name:10s}@{k_max}: emo={m['cremad_emotion']['acc']:.3f} "
            f"(shuf={m['cremad_emotion']['shuffled_acc']:.3f}) | "
            f"EER={m['cremad_cross_sentence_eer']['backbone_eer']:.3f} "
            f"(spk={m['cremad_cross_sentence_eer']['speaker_head_eer']:.3f}, "
            f"shuf={m['cremad_cross_sentence_eer']['shuffled_eer']:.3f}) | "
            f"VS={m['vocalset_technique']['acc']:.3f} (shuf={m['vocalset_technique']['shuffled_acc']:.3f}) | "
            f"MR={m['maestro_real_soft_events']['macro_map']:.3f} "
            f"(shuf={m['maestro_real_soft_events']['shuffled_macro_map']:.3f})",
            flush=True,
        )

    best_e_arm = (
        "E_s4"
        if trained["E_s4"]["eval"]["metrics_by_k"][4]["composite"]
        >= trained["E_s8"]["eval"]["metrics_by_k"][4]["composite"]
        else "E_s8"
    )
    best_ckpt_dest = ckpt_dir / "stage1_pilot_best.pt"
    shutil.copyfile(trained[best_e_arm]["ckpt_path"], best_ckpt_dest)
    print(f"[stream] Verifying LoopEarStream with {best_ckpt_dest} ({best_e_arm}) ...", flush=True)
    stream_verify = _verify_live_stream(best_ckpt_dest, manifest_dir)
    print(f"[stream] Verified: {json.dumps(stream_verify['stream_state'])}", flush=True)

    arms_summary: dict[str, Any] = {}
    for arm_name in arm_order:
        tr_item = trained[arm_name]
        arms_summary[arm_name] = {
            "arm_type": tr_item["arm_type"],
            "params_ref": tr_item["params_ref"],
            "params_heads": tr_item["params_heads"],
            "best_epoch": tr_item["best_epoch"],
            "train_wall_s": tr_item["train_wall_s"],
            "max_weight_delta": tr_item["max_weight_delta"],
            "reload_max_abs_diff": tr_item["reload_max_abs_diff"],
            "history": tr_item["history"],
            "val_metrics_by_k": {
                str(k): v for k, v in tr_item["eval"]["metrics_by_k"].items()
            },
        }

    payload = {
        "stage": "stage1_pilot",
        "cloud_gpu_spend_usd": 0.0,
        "seed": int(args.seed),
        "hyperparams": {
            "epochs": int(args.epochs),
            "steps_per_epoch": int(args.steps_per_epoch),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
        },
        "data_summary": {
            "extract_total_audio_hours": cache_index["extract_total_audio_hours"],
            "extract_total_wall_s": cache_index["extract_total_wall_s"],
            "extract_throughput_s_per_h": cache_index["extract_throughput_s_per_h"],
            "datasets": cache_index["datasets"],
            "caches": cache_index["caches"],
        },
        "arms": arms_summary,
        "paired_bootstrap_cis": paired_cis,
        "h6_oracle": h6_oracle,
        "frozen_transfer": transfer_results,
        "best_e_arm": best_e_arm,
        "stream_verification": stream_verify,
    }
    out_path = write_result_json(args.out, payload)
    if args.store_copy:
        dest_copy = Path(args.store_copy)
        dest_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(out_path, dest_copy)
    print(f"[pilot] Saved results to {out_path} and {args.store_copy}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
