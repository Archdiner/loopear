import argparse
import copy
import gc
import json
import math
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loopear.data import (
    CREMAD_EMOTIONS,
    MAESTRO_REAL_CLASSES,
    PILOT_CORPUS_SOURCES,
    REALDESED_CLASSES,
    STARSS23_CLASSES,
    VOCALSET_TECHNIQUES,
    compute_onset_and_boundary_metrics,
    downsample_binary_targets,
    downsample_overlap_targets,
    rasterize_intervals_50hz,
    rasterize_overlap_50hz,
    rasterize_soft_labels_50hz,
    rasterize_starss_frames_50hz,
    upsample_frame_preds,
)
from loopear.encoder import (
    HOP_E,
    T_E_MAX,
    TAP_LAYERS,
    ShardedFeatureCache,
    load_mono_audio,
)
from loopear.halting import (
    HaltingGate,
    evaluate_halting_pareto,
    extract_halting_features,
    fit_halting_gate,
)
from loopear.harness import (
    assert_zero_pretraining_overlap,
    write_result_json,
    write_run_manifest,
)
from loopear.losses import (
    MultiTaskHeads,
    apply_feature_augmentation,
    compute_cross_chapter_eer,
    masked_mean_pool,
    multidepth_event_loss,
    multidepth_overlap_loss,
    multidepth_speaker_loss,
    set_seed,
)
from loopear.models import LoopEarModel, build_arm
from loopear.oracle import evaluate_h6_gate
from loopear.stream import LoopEarStream

STAGE2_ARMS: dict[str, tuple[str, dict[str, Any]]] = {
    "A_ml": ("A_ml", {"stride": 8}),
    "B_comp_ml": (
        "B_comp",
        {
            "stride": 8,
            "use_multilayer": True,
            "n_layers": len(TAP_LAYERS),
            "use_local_stem": True,
            "ls": 0.3,
        },
    ),
    "D_ml": (
        "D",
        {
            "stride": 8,
            "max_k": 4,
            "use_multilayer": True,
            "n_layers": len(TAP_LAYERS),
            "use_local_stem": True,
            "ls": 0.3,
        },
    ),
    "E_ml_res_s8": (
        "E",
        {
            "stride": 8,
            "max_k": 4,
            "pool_mode": "avg",
            "use_multilayer": True,
            "n_layers": len(TAP_LAYERS),
            "use_local_stem": True,
            "ls": 0.3,
        },
    ),
    "E_ml_res_s4": (
        "E",
        {
            "stride": 4,
            "max_k": 4,
            "pool_mode": "avg",
            "use_multilayer": True,
            "n_layers": len(TAP_LAYERS),
            "use_local_stem": True,
            "ls": 0.3,
        },
    ),
}


def _load_tap_split(
    tap_root: Path,
    tap_dir: str,
    s1_root: Path | None = None,
    s1_dir: str | None = None,
    load_layer_pool: bool = True,
) -> dict[str, Any]:
    cache = ShardedFeatureCache(tap_root)
    idx = cache.read_index(tap_dir)
    items = idx["items"]
    shards: dict[str, dict[str, np.ndarray]] = {}
    order: list[str] = []
    for it in items:
        sh = str(it["shard"])
        if sh not in shards:
            shards[sh] = cache.load_shard(tap_dir, sh)
            order.append(sh)
    e_taps = np.concatenate([shards[s]["E_taps"] for s in order], axis=0)
    layer_clip = np.concatenate([shards[s]["layer_clip"] for s in order], axis=0)
    lengths = np.concatenate([shards[s]["lengths"] for s in order], axis=0)
    lp_arr: np.ndarray | None = None
    if load_layer_pool and s1_root is not None and s1_dir is not None:
        s1_cache = ShardedFeatureCache(s1_root)
        s1_idx = s1_cache.read_index(s1_dir)
        s1_shards: dict[str, dict[str, np.ndarray]] = {}
        s1_order: list[str] = []
        for it in s1_idx["items"]:
            sh = str(it["shard"])
            if sh not in s1_shards:
                s1_shards[sh] = s1_cache.load_shard(s1_dir, sh)
                s1_order.append(sh)
        if all("layer_pool" in s1_shards[s] for s in s1_order):
            lp_arr = np.concatenate(
                [s1_shards[s]["layer_pool"] for s in s1_order], axis=0
            )
    return {
        "items": items,
        "E_taps": e_taps,
        "layer_clip": layer_clip,
        "lengths": lengths,
        "layer_pool": lp_arr,
    }


def _load_starss_events(label_path: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for line in Path(label_path).read_text(encoding="utf-8").splitlines():
        cols = [c.strip() for c in line.split(",")]
        if len(cols) >= 2 and cols[0].isdigit():
            out.append((int(cols[0]), int(cols[1])))
    return out


def _build_stage2_bundle(
    manifest_dir: Path,
    tap_root: Path,
    tap_index: dict[str, Any],
    s1_root: Path,
    s1_index: dict[str, Any],
) -> dict[str, Any]:
    bundle: dict[str, Any] = {}
    for corpus in ("starss23", "realdesed", "dipco", "ami_sdm"):
        man = json.loads((manifest_dir / f"{corpus}.json").read_text(encoding="utf-8"))
        by_id = {it["id"]: it for it in man["items"]}
        bundle[corpus] = {"manifest": man}
        for split in ("train", "val"):
            entry = _load_tap_split(
                tap_root,
                tap_index["caches"][corpus][split]["cache_dir"],
                s1_root=s1_root,
                s1_dir=s1_index["caches"][corpus][split]["cache_dir"],
                load_layer_pool=True,
            )
            items = entry["items"]
            l_arr = entry["lengths"]
            n = len(items)
            entry["group_id"] = np.array(
                [str(by_id[str(it["source_id"])]["group_id"]) for it in items]
            )
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
        for i in range(tr["lengths"].shape[0]):
            nf = int(tr["lengths"][i])
            pos += tr["y_50hz"][i, :nf].sum(axis=0)
            tot += float(nf)
        neg = np.maximum(tot - pos, 1.0)
        pw = np.clip(neg / np.maximum(pos, 1.0), 1.0, 10.0).astype(np.float32)
        bundle[corpus]["pos_weight"] = torch.from_numpy(pw)

    tr_dip = bundle["dipco"]["train"]
    counts = np.zeros((3,), dtype=np.float64)
    for i in range(tr_dip["lengths"].shape[0]):
        nf = int(tr_dip["lengths"][i])
        counts += np.bincount(tr_dip["y_50hz"][i, :nf], minlength=3)
    inv_sqrt = np.sqrt(counts.sum() / np.maximum(counts, 1.0))
    inv_sqrt = (inv_sqrt / inv_sqrt.mean()).astype(np.float32)
    bundle["dipco"]["class_weight"] = torch.from_numpy(inv_sqrt)
    return bundle


def _prepare_stride_targets(bundle: dict[str, Any], stride: int) -> dict[str, np.ndarray]:
    t_z_max = math.ceil(T_E_MAX / stride)
    out: dict[str, np.ndarray] = {}
    for corpus in ("starss23", "realdesed"):
        tr = bundle[corpus]["train"]
        n = int(tr["lengths"].shape[0])
        n_cls = int(tr["y_50hz"].shape[2])
        arr = np.zeros((n, t_z_max, n_cls), dtype=np.float32)
        for i in range(n):
            arr[i] = downsample_binary_targets(
                tr["y_50hz"][i], int(tr["lengths"][i]), stride, t_z_max
            )
        out[corpus] = arr
    tr_d = bundle["dipco"]["train"]
    n_d = int(tr_d["lengths"].shape[0])
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
    augment: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor,
]:
    l_b = torch.from_numpy(split_data["lengths"][idx].astype(np.int64))
    t_cap = min(T_E_MAX, int(math.ceil(max(int(l_b.max().item()), 8) / 8.0) * 8))
    lc_b = torch.from_numpy(split_data["layer_clip"][idx].astype(np.float32))
    if use_lp:
        e_last = torch.from_numpy(
            split_data["E_taps"][idx, -1, :t_cap].astype(np.float32)
        )
        hs_b = None
        tz_cap = math.ceil(t_cap / 8)
        lp_b = (
            torch.from_numpy(
                split_data["layer_pool"][idx, :, :tz_cap].astype(np.float32)
            )
            if split_data["layer_pool"] is not None
            else None
        )
        if augment:
            lc_b = lc_b + torch.randn_like(lc_b) * 0.015
            if lp_b is not None:
                lp_len = torch.div(l_b + 7, 8, rounding_mode="floor").clamp(min=1)
                lp_b = apply_feature_augmentation(
                    lp_b,
                    lp_len,
                    time_mask_prob=0.5,
                    n_masks=2,
                    max_mask_frames=3,
                    noise_std=0.0,
                )
        return e_last, l_b, hs_b, lp_b, lc_b

    taps = torch.from_numpy(split_data["E_taps"][idx, :, :t_cap].astype(np.float32))
    if augment:
        taps = apply_feature_augmentation(
            taps, l_b, time_mask_prob=0.5, n_masks=2, max_mask_frames=16, noise_std=0.0
        )
        lc_b = lc_b + torch.randn_like(lc_b) * 0.015
    e_last = taps[:, -1, :, :]
    return e_last, l_b, taps, None, lc_b


def _compute_window_dense_score(
    corpus: str,
    y_true: np.ndarray,
    p_pred: np.ndarray,
) -> float:
    if corpus in ("starss23", "realdesed"):
        y_bin = (y_true >= 0.5).astype(np.int32).ravel()
        p_bin = (p_pred >= 0.5).astype(np.int32).ravel()
        if y_bin.sum() == 0 and p_bin.sum() == 0:
            return 1.0
        return float(f1_score(y_bin, p_bin, zero_division=0))
    return float(np.mean(np.argmax(p_pred, axis=-1) == y_true))


@torch.no_grad()
def _evaluate_arm_val(
    arm_name: str,
    model: torch.nn.Module,
    heads: MultiTaskHeads,
    bundle: dict[str, Any],
    batch_size: int = 32,
    compute_onset: bool = True,
    collect_oracle_states: bool = False,
) -> dict[str, Any]:
    model.eval()
    heads.eval()
    use_lp = arm_name == "A_ml"
    depths = tuple(getattr(model, "depths", (1,)))
    stride = int(getattr(model, "stride", 8))

    def _run_dense(corpus: str, head_mode: str):
        val = bundle[corpus]["val"]
        n = int(val["lengths"].shape[0])
        probs_by_k: dict[int, list[np.ndarray]] = {k: [] for k in depths}
        z_by_k: dict[int, list[np.ndarray]] = {k: [] for k in depths}
        for b0 in range(0, n, batch_size):
            b1 = min(n, b0 + batch_size)
            idx = np.arange(b0, b1)
            e_b, l_b, hs_b, lp_b, _ = _sample_batch(val, idx, use_lp, augment=False)
            states, z_pad = model.forward_states(
                e_b, l_b, depths=depths, hidden_states=hs_b, layer_pool=lp_b
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
    n_ami = int(ami_val["lengths"].shape[0])
    ami_embs: dict[int, list[np.ndarray]] = {k: [] for k in depths}
    for b0 in range(0, n_ami, batch_size):
        b1 = min(n_ami, b0 + batch_size)
        idx = np.arange(b0, b1)
        e_b, l_b, hs_b, lp_b, lc_b = _sample_batch(ami_val, idx, use_lp, augment=False)
        states, z_pad = model.forward_states(
            e_b, l_b, depths=depths, hidden_states=hs_b, layer_pool=lp_b
        )
        for k in depths:
            ami_embs[k].append(
                heads.speaker_embed(states[k], z_pad, layer_clip=lc_b).cpu().numpy()
            )

    metrics_by_k: dict[int, dict[str, Any]] = {}
    win_scores_by_k: dict[int, dict[str, np.ndarray]] = {}
    st_y_list = [
        bundle["starss23"]["val"]["y_50hz"][i, : int(bundle["starss23"]["val"]["lengths"][i])]
        for i in range(int(bundle["starss23"]["val"]["lengths"].shape[0]))
    ]
    rd_y_list = [
        bundle["realdesed"]["val"]["y_50hz"][i, : int(bundle["realdesed"]["val"]["lengths"][i])]
        for i in range(int(bundle["realdesed"]["val"]["lengths"].shape[0]))
    ]
    dp_y_list = [
        bundle["dipco"]["val"]["y_50hz"][i, : int(bundle["dipco"]["val"]["lengths"][i])]
        for i in range(int(bundle["dipco"]["val"]["lengths"].shape[0]))
    ]

    for k in depths:
        st_y_all = np.concatenate(st_y_list, axis=0)
        st_p_all = np.concatenate(st_probs[k], axis=0)
        st_active = [c for c in range(st_y_all.shape[1]) if st_y_all[:, c].sum() > 0]
        st_macro_map = float(
            np.mean(
                [average_precision_score(st_y_all[:, c], st_p_all[:, c]) for c in st_active]
            )
        )

        rd_y_all = np.concatenate(rd_y_list, axis=0)
        rd_p_all = np.concatenate(rd_probs[k], axis=0)
        rd_active = [c for c in range(rd_y_all.shape[1]) if rd_y_all[:, c].sum() > 0]
        rd_macro_map = float(
            np.mean(
                [average_precision_score(rd_y_all[:, c], rd_p_all[:, c]) for c in rd_active]
            )
        )

        dp_y_all = np.concatenate(dp_y_list, axis=0)
        dp_p_all = np.concatenate(dp_probs[k], axis=0)
        dp_pred = np.argmax(dp_p_all, axis=-1)
        dp_mf1 = float(f1_score(dp_y_all, dp_pred, average="macro", zero_division=0))
        dp_acc = float(np.mean(dp_pred == dp_y_all))

        embs = np.concatenate(ami_embs[k], axis=0)
        ami_eer = compute_cross_chapter_eer(
            embs, ami_val["speaker_id"], ami_val["meeting_id"]
        )
        ami_inv_eer = 1.0 - float(ami_eer["eer"])
        comp = float(np.mean([st_macro_map, rd_macro_map, dp_mf1, ami_inv_eer]))

        entry: dict[str, Any] = {
            "starss23_macro_map": round(st_macro_map, 4),
            "realdesed_macro_map": round(rd_macro_map, 4),
            "dipco_macro_f1": round(dp_mf1, 4),
            "dipco_acc": round(dp_acc, 4),
            "ami_eer": round(float(ami_eer["eer"]), 4),
            "ami_1_minus_eer": round(ami_inv_eer, 4),
            "composite": round(comp, 4),
        }
        if compute_onset:
            entry["starss23_micro_map"] = round(
                float(average_precision_score(st_y_all.ravel(), st_p_all.ravel())), 4
            )
            entry["starss23_micro_f1"] = round(
                float(
                    f1_score(
                        st_y_all.ravel(),
                        (st_p_all.ravel() >= 0.5).astype(np.int32),
                        zero_division=0,
                    )
                ),
                4,
            )
            entry["realdesed_micro_map"] = round(
                float(average_precision_score(rd_y_all.ravel(), rd_p_all.ravel())), 4
            )
            entry["realdesed_micro_f1"] = round(
                float(
                    f1_score(
                        rd_y_all.ravel(),
                        (rd_p_all.ravel() >= 0.5).astype(np.int32),
                        zero_division=0,
                    )
                ),
                4,
            )
            entry["dipco_overlap2_ap"] = round(
                float(
                    average_precision_score(
                        (dp_y_all == 2).astype(np.int32), dp_p_all[:, 2]
                    )
                ),
                4,
            )
            st_on = compute_onset_and_boundary_metrics(st_y_list, st_probs[k], hop_e=HOP_E)
            rd_on = compute_onset_and_boundary_metrics(rd_y_list, rd_probs[k], hop_e=HOP_E)
            entry["timing_starss23"] = st_on
            entry["timing_realdesed"] = rd_on
            entry["timing_mean"] = {
                key: round(0.5 * (st_on[key] + rd_on[key]), 4) for key in st_on
            }
            win_scores_by_k[k] = {
                "starss23": np.array(
                    [
                        _compute_window_dense_score("starss23", st_y_list[i], st_probs[k][i])
                        for i in range(len(st_probs[k]))
                    ],
                    dtype=np.float64,
                ),
                "realdesed": np.array(
                    [
                        _compute_window_dense_score("realdesed", rd_y_list[i], rd_probs[k][i])
                        for i in range(len(rd_probs[k]))
                    ],
                    dtype=np.float64,
                ),
                "dipco": np.array(
                    [
                        _compute_window_dense_score("dipco", dp_y_list[i], dp_probs[k][i])
                        for i in range(len(dp_probs[k]))
                    ],
                    dtype=np.float64,
                ),
            }
        metrics_by_k[k] = entry

    out: dict[str, Any] = {
        "metrics_by_k": metrics_by_k,
        "window_scores_by_k": win_scores_by_k,
    }
    if collect_oracle_states:
        out["oracle_states"] = {
            "starss23": {k: np.stack(st_z[k], axis=0) for k in depths},
            "realdesed": {k: np.stack(rd_z[k], axis=0) for k in depths},
            "dipco": {k: np.stack(dp_z[k], axis=0) for k in depths},
        }
    return out


def _inspect_learned_weights(
    model: torch.nn.Module,
    heads: MultiTaskHeads,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if hasattr(model, "fusion") and model.fusion is not None:
        w = F.softmax(model.fusion.layer_logits.detach().float(), dim=0)
        out["acoustic_layer_weights"] = [round(float(v), 4) for v in w.cpu().tolist()]
    if hasattr(model, "local_stem") and model.local_stem is not None:
        a = model.local_stem.alpha.detach().float()
        out["local_stem_alpha_l2"] = round(float(a.norm().item()), 4)
        out["local_stem_alpha_abs_mean"] = round(float(a.abs().mean().item()), 4)
    spk_w = heads.speaker_layer_weights()
    if spk_w is not None:
        out["speaker_layer_weights"] = spk_w
        if heads.spk_clip_gate is not None:
            out["speaker_clip_gate"] = round(float(heads.spk_clip_gate.detach().item()), 4)
    return out


def _train_one_arm(
    arm_name: str,
    bundle: dict[str, Any],
    epochs: int,
    steps_per_epoch: int,
    batch_size: int,
    lr: float,
    seed: int,
    eval_every: int = 2,
) -> dict[str, Any]:
    set_seed(seed)
    rng = np.random.default_rng(seed)
    base_arm, arm_kwargs = STAGE2_ARMS[arm_name]
    model = build_arm(base_arm, **arm_kwargs)
    n_spk = int(bundle["ami_sdm"]["manifest"]["n_train_speakers"])
    head_kwargs = {
        "d": model.out_dim,
        "n_ev": len(REALDESED_CLASSES),
        "d_spk": 192,
        "n_spk": n_spk,
        "n_starss": len(STARSS23_CLASSES),
        "n_ovl": 3,
        "use_layer_clip_spk": True,
        "n_spk_layers": 12,
        "d_e": 384,
    }
    heads = MultiTaskHeads(**head_kwargs)
    stride = int(getattr(model, "stride", 8))
    use_lp = arm_name == "A_ml"
    depths = tuple(getattr(model, "depths", (1,)))
    stride_targets = _prepare_stride_targets(bundle, stride)

    params = list(model.parameters()) + list(heads.parameters())
    n_params = int(sum(p.numel() for p in params if p.requires_grad))
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
    total_steps = epochs * steps_per_epoch
    warmup_steps = max(1, min(2 * steps_per_epoch, total_steps // 6))

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        prog = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)

    n_st = int(bundle["starss23"]["train"]["lengths"].shape[0])
    n_rd = int(bundle["realdesed"]["train"]["lengths"].shape[0])
    n_dp = int(bundle["dipco"]["train"]["lengths"].shape[0])
    n_am = int(bundle["ami_sdm"]["train"]["lengths"].shape[0])
    pw_st = bundle["starss23"]["pos_weight"]
    pw_rd = bundle["realdesed"]["pos_weight"]
    cw_dp = bundle["dipco"]["class_weight"]

    t0 = time.perf_counter()
    best_comp = -1.0
    best_epoch = 0
    best_state: tuple[dict[str, Any], dict[str, Any]] | None = None
    history: list[dict[str, Any]] = []

    for ep in range(1, epochs + 1):
        model.train()
        heads.train()
        loss_acc = 0.0
        for _ in range(steps_per_epoch):
            opt.zero_grad(set_to_none=True)

            i_st = rng.integers(0, n_st, size=batch_size)
            e_st, l_st, hs_st, lp_st, _ = _sample_batch(
                bundle["starss23"]["train"], i_st, use_lp, augment=True
            )
            st_states, st_pad = model.forward_states(
                e_st, l_st, depths=depths, hidden_states=hs_st, layer_pool=lp_st
            )
            t_z = st_pad.shape[1]
            y_st_f = torch.from_numpy(stride_targets["starss23"][i_st, :t_z])
            y_st_c = torch.from_numpy(bundle["starss23"]["train"]["y_clip"][i_st])
            loss_st = multidepth_event_loss(
                st_states, st_pad, heads, y_st_f, y_st_c, pos_weight=pw_st, head_name="starss"
            )

            i_rd = rng.integers(0, n_rd, size=batch_size)
            e_rd, l_rd, hs_rd, lp_rd, _ = _sample_batch(
                bundle["realdesed"]["train"], i_rd, use_lp, augment=True
            )
            rd_states, rd_pad = model.forward_states(
                e_rd, l_rd, depths=depths, hidden_states=hs_rd, layer_pool=lp_rd
            )
            t_z_rd = rd_pad.shape[1]
            y_rd_f = torch.from_numpy(stride_targets["realdesed"][i_rd, :t_z_rd])
            y_rd_c = torch.from_numpy(bundle["realdesed"]["train"]["y_clip"][i_rd])
            loss_rd = multidepth_event_loss(
                rd_states, rd_pad, heads, y_rd_f, y_rd_c, pos_weight=pw_rd, head_name="ev"
            )

            i_dp = rng.integers(0, n_dp, size=batch_size)
            e_dp, l_dp, hs_dp, lp_dp, _ = _sample_batch(
                bundle["dipco"]["train"], i_dp, use_lp, augment=True
            )
            dp_states, dp_pad = model.forward_states(
                e_dp, l_dp, depths=depths, hidden_states=hs_dp, layer_pool=lp_dp
            )
            t_z_dp = dp_pad.shape[1]
            y_dp = torch.from_numpy(stride_targets["dipco"][i_dp, :t_z_dp])
            loss_dp = multidepth_overlap_loss(
                dp_states, dp_pad, heads, y_dp, class_weight=cw_dp
            )

            i_am = rng.integers(0, n_am, size=batch_size)
            e_am, l_am, hs_am, lp_am, lc_am = _sample_batch(
                bundle["ami_sdm"]["train"], i_am, use_lp, augment=True
            )
            am_states, am_pad = model.forward_states(
                e_am, l_am, depths=depths, hidden_states=hs_am, layer_pool=lp_am
            )
            y_am = torch.from_numpy(bundle["ami_sdm"]["train"]["speaker_idx"][i_am])
            loss_am = multidepth_speaker_loss(
                am_states, am_pad, heads, y_am, layer_clip=lc_am
            )

            loss = loss_st + loss_rd + 0.7 * loss_dp + 0.35 * loss_am
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 2.0)
            opt.step()
            sched.step()
            loss_acc += float(loss.item())

        if ep == 1 or ep % eval_every == 0 or ep >= epochs - 2:
            ev_ep = _evaluate_arm_val(
                arm_name, model, heads, bundle, compute_onset=False
            )
            k_max = max(depths)
            comp = float(ev_ep["metrics_by_k"][k_max]["composite"])
            history.append(
                {
                    "epoch": ep,
                    "train_loss": round(loss_acc / steps_per_epoch, 4),
                    "val_composite": round(comp, 4),
                    "lr": round(float(opt.param_groups[0]["lr"]), 6),
                }
            )
            if comp > best_comp:
                best_comp = comp
                best_epoch = ep
                best_state = (
                    copy.deepcopy(model.state_dict()),
                    copy.deepcopy(heads.state_dict()),
                )

    train_s = time.perf_counter() - t0
    if best_state is not None:
        model.load_state_dict(best_state[0])
        heads.load_state_dict(best_state[1])
    final_eval = _evaluate_arm_val(
        arm_name,
        model,
        heads,
        bundle,
        compute_onset=True,
        collect_oracle_states=(max(depths) >= 4),
    )
    k_max = max(depths)
    last_delta = (
        round(history[-1]["val_composite"] - history[-2]["val_composite"], 4)
        if len(history) >= 2
        else 0.0
    )
    return {
        "arm": arm_name,
        "base_arm": base_arm,
        "arm_kwargs": arm_kwargs,
        "head_kwargs": head_kwargs,
        "seed": seed,
        "n_params": n_params,
        "train_wall_s": round(train_s, 2),
        "best_epoch": best_epoch,
        "last_eval_delta": last_delta,
        "history": history,
        "learned_weights": _inspect_learned_weights(model, heads),
        "metrics_by_k": final_eval["metrics_by_k"],
        "window_scores_by_k": final_eval["window_scores_by_k"],
        "oracle_states": final_eval.get("oracle_states"),
        "model": model,
        "heads": heads,
        "best_composite": float(final_eval["metrics_by_k"][k_max]["composite"]),
    }


def _paired_bootstrap_ci(
    a: np.ndarray,
    b: np.ndarray,
    n_resamples: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    n = diff.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    boots = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "mean_diff": round(float(diff.mean()), 5),
        "ci95_low": round(float(lo), 5),
        "ci95_high": round(float(hi), 5),
        "n_windows": int(n),
    }


@torch.no_grad()
def _extract_loopear_halting_dataset(
    model: LoopEarModel,
    heads: MultiTaskHeads,
    bundle: dict[str, Any],
    split: str,
    batch_size: int = 24,
    noise_std: float = 0.0,
    seed: int = 42,
) -> dict[str, Any]:
    model.eval()
    heads.eval()
    stride = int(model.stride)
    gen = torch.Generator().manual_seed(seed)
    f1_list: list[np.ndarray] = []
    f2_list: list[np.ndarray] = []
    s1_list: list[np.ndarray] = []
    s2_list: list[np.ndarray] = []
    s4_list: list[np.ndarray] = []
    e_list: list[np.ndarray] = []
    g_list: list[str] = []
    c_list: list[str] = []
    for corpus, head_mode in (
        ("starss23", "starss"),
        ("realdesed", "realdesed"),
        ("dipco", "dipco"),
    ):
        sp = bundle[corpus][split]
        n = int(sp["lengths"].shape[0])
        for b0 in range(0, n, batch_size):
            b1 = min(n, b0 + batch_size)
            idx = np.arange(b0, b1)
            taps = torch.from_numpy(sp["E_taps"][idx].astype(np.float32))
            if noise_std > 0.0:
                taps = taps + torch.randn(taps.shape, generator=gen) * noise_std
            l_b = torch.from_numpy(sp["lengths"][idx].astype(np.int64))
            e_last = taps[:, -1, :, :]
            ctx = model.prepare_evidence(e_last, l_b, hidden_states=taps, use_kv_cache=True)
            pad = ctx["z_pad"]
            z_cur, z1 = model.step_refine(ctx["z0"], 1, ctx, use_kv_cache=True)
            f1, p1 = extract_halting_features(
                ctx["e_in"], l_b, ctx["z0"], z1, pad, heads, step_k=1
            )
            z_cur, z2 = model.step_refine(z_cur, 2, ctx, use_kv_cache=True)
            f2, _ = extract_halting_features(
                ctx["e_in"], l_b, z1, z2, pad, heads, step_k=2, prev_probs=p1
            )
            z_cur, _ = model.step_refine(z_cur, 3, ctx, use_kv_cache=True)
            _, z4 = model.step_refine(z_cur, 4, ctx, use_kv_cache=True)
            f1_list.append(f1.cpu().numpy())
            f2_list.append(f2.cpu().numpy())
            e_list.append(ctx["e_in"].cpu().numpy().astype(np.float16))
            for i in range(b1 - b0):
                g_list.append(f"{corpus}:{sp['group_id'][b0 + i]}")
                c_list.append(corpus)
            for k_z, dst in ((z1, s1_list), (z2, s2_list), (z4, s4_list)):
                if head_mode == "starss":
                    p = torch.sigmoid(heads.starss_logits(k_z)).cpu().numpy()
                elif head_mode == "realdesed":
                    p = torch.sigmoid(heads.event_logits(k_z)).cpu().numpy()
                else:
                    p = torch.softmax(heads.overlap_logits(k_z), dim=-1).cpu().numpy()
                sc = np.zeros((b1 - b0,), dtype=np.float64)
                for i in range(b1 - b0):
                    nf = int(sp["lengths"][b0 + i])
                    up = upsample_frame_preds(p[i], stride, nf)
                    yt = sp["y_50hz"][b0 + i, :nf]
                    sc[i] = _compute_window_dense_score(corpus, yt, up)
                dst.append(sc)
    return {
        "feats_k1": np.concatenate(f1_list, axis=0),
        "feats_k2": np.concatenate(f2_list, axis=0),
        "scores_by_k": {
            1: np.concatenate(s1_list, axis=0),
            2: np.concatenate(s2_list, axis=0),
            4: np.concatenate(s4_list, axis=0),
        },
        "e_seq": np.concatenate(e_list, axis=0),
        "groups": np.array(g_list),
        "corpora": np.array(c_list),
    }


def _train_and_eval_halting_dev(
    model: LoopEarModel,
    heads: MultiTaskHeads,
    bundle: dict[str, Any],
    flops_by_k: dict[int, float],
    seed: int = 42,
) -> tuple[HaltingGate, dict[str, Any]]:
    tr_clean = _extract_loopear_halting_dataset(
        model, heads, bundle, split="train", noise_std=0.0, seed=seed
    )
    tr_noisy = _extract_loopear_halting_dataset(
        model, heads, bundle, split="train", noise_std=0.08, seed=seed + 100
    )
    va = _extract_loopear_halting_dataset(
        model, heads, bundle, split="val", noise_std=0.0, seed=seed
    )
    x_tr = np.concatenate(
        [
            tr_clean["feats_k1"],
            tr_clean["feats_k2"],
            tr_noisy["feats_k1"],
            tr_noisy["feats_k2"],
        ],
        axis=0,
    )
    g1_c = (
        np.maximum(tr_clean["scores_by_k"][2], tr_clean["scores_by_k"][4])
        - tr_clean["scores_by_k"][1]
    )
    g2_c = tr_clean["scores_by_k"][4] - tr_clean["scores_by_k"][2]
    g1_n = (
        np.maximum(tr_noisy["scores_by_k"][2], tr_noisy["scores_by_k"][4])
        - tr_noisy["scores_by_k"][1]
    )
    g2_n = tr_noisy["scores_by_k"][4] - tr_noisy["scores_by_k"][2]
    y_tr = np.concatenate([g1_c, g2_c, g1_n, g2_n], axis=0)
    gate = fit_halting_gate(x_tr, y_tr, hidden_dim=32, epochs=240, lr=3e-3, seed=seed)
    primary = evaluate_halting_pareto(
        gate,
        va["feats_k1"],
        va["feats_k2"],
        va["scores_by_k"],
        flops_by_k,
        va["e_seq"].astype(np.float32),
    )

    uniq_groups = sorted(set(va["groups"].tolist()))
    rng = np.random.default_rng(seed + 500)
    rng.shuffle(uniq_groups)
    grp_to_fold = {g: i % 3 for i, g in enumerate(uniq_groups)}
    folds = np.array([grp_to_fold[g] for g in va["groups"]], dtype=np.int64)
    oof_p1 = np.zeros((va["feats_k1"].shape[0],), dtype=np.float64)
    oof_p2 = np.zeros((va["feats_k2"].shape[0],), dtype=np.float64)
    for f_id in range(3):
        m_tr = folds != f_id
        m_te = folds == f_id
        xf = np.concatenate([va["feats_k1"][m_tr], va["feats_k2"][m_tr]], axis=0)
        gf1 = (
            np.maximum(va["scores_by_k"][2][m_tr], va["scores_by_k"][4][m_tr])
            - va["scores_by_k"][1][m_tr]
        )
        gf2 = va["scores_by_k"][4][m_tr] - va["scores_by_k"][2][m_tr]
        yf = np.concatenate([gf1, gf2], axis=0)
        g_fold = fit_halting_gate(
            xf, yf, hidden_dim=32, epochs=220, lr=3e-3, seed=seed + f_id
        )
        with torch.no_grad():
            oof_p1[m_te] = (
                g_fold.continue_prob(torch.from_numpy(va["feats_k1"][m_te].astype(np.float32)))
                .cpu()
                .numpy()
            )
            oof_p2[m_te] = (
                g_fold.continue_prob(torch.from_numpy(va["feats_k2"][m_te].astype(np.float32)))
                .cpu()
                .numpy()
            )
    s1 = va["scores_by_k"][1]
    s2 = va["scores_by_k"][2]
    s4 = va["scores_by_k"][4]
    u1, u2, u4 = float(np.mean(s1)), float(np.mean(s2)), float(np.mean(s4))
    all_oof = np.concatenate([oof_p1, oof_p2])
    oof_pts: list[dict[str, Any]] = []
    for tau in sorted(
        set([0.0, 1.01] + [float(np.quantile(all_oof, q)) for q in np.linspace(0.0, 1.0, 21)])
    ):
        go2 = oof_p1 >= tau
        go4 = go2 & (oof_p2 >= tau)
        ck = np.where(go4, 4, np.where(go2, 2, 1))
        cs = np.where(ck == 4, s4, np.where(ck == 2, s2, s1))
        oof_pts.append(
            {
                "tau": round(float(tau), 4),
                "avg_k": round(float(np.mean(ck)), 4),
                "score": round(float(np.mean(cs)), 5),
                "overthinking_rate": round(float(np.mean(cs < (s1 - 1e-6))), 4),
            }
        )
    oof_best = max(oof_pts, key=lambda r: r["score"])
    oof_k2 = min(oof_pts, key=lambda r: abs(r["avg_k"] - 2.0))
    order = np.argsort([p["avg_k"] for p in oof_pts])
    oof_auc = float(
        np.trapezoid(
            np.asarray([p["score"] for p in oof_pts])[order],
            np.asarray([p["avg_k"] for p in oof_pts])[order],
        )
        / 3.0
    )
    u_auc = float(((u1 + u2) * 0.5 * 1.0 + (u2 + u4) * 0.5 * 2.0) / 3.0)
    return gate, {
        "train_calibrated_primary": primary,
        "val_group_oof_diagnostic": {
            "n_val_groups": len(uniq_groups),
            "best_oof": oof_best,
            "at_budget_k2": oof_k2,
            "oof_pareto_auc": round(oof_auc, 5),
            "uniform_pareto_auc": round(u_auc, 5),
            "oof_minus_uniform_auc": round(oof_auc - u_auc, 5),
        },
        "recommended_halt_threshold": float(primary["at_budget_k2"]["learned"]["tau1"]),
    }


def _aggregate_clip_embeddings(
    items: list[dict[str, Any]],
    embs: np.ndarray,
    manifest_items: dict[str, dict[str, Any]],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    by_src: dict[str, list[np.ndarray]] = {}
    order: list[str] = []
    for idx_i, it in enumerate(items):
        sid = str(it["source_id"])
        if sid not in by_src:
            by_src[sid] = []
            order.append(sid)
        by_src[sid].append(embs[idx_i])
    vecs: list[np.ndarray] = []
    metas: list[dict[str, Any]] = []
    for sid in order:
        arr = np.mean(np.stack(by_src[sid], axis=0), axis=0)
        norm = float(np.linalg.norm(arr))
        if norm > 1e-8:
            arr = arr / norm
        vecs.append(arr)
        metas.append(manifest_items[sid])
    return np.stack(vecs, axis=0), metas


def _fit_linear_probe(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_ev: np.ndarray,
    y_ev: np.ndarray,
    seed: int = 42,
) -> dict[str, Any]:
    scaler = StandardScaler()
    x_tr_s = scaler.fit_transform(x_tr)
    x_ev_s = scaler.transform(x_ev)
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
    clf.fit(x_tr_s, y_tr)
    preds = clf.predict(x_ev_s)
    correct = (preds == y_ev).astype(np.float64)
    acc = float(accuracy_score(y_ev, preds))
    mf1 = float(f1_score(y_ev, preds, average="macro", zero_division=0))

    rng = np.random.default_rng(seed + 1000)
    y_shuf = rng.permutation(y_tr)
    clf_shuf = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
    clf_shuf.fit(x_tr_s, y_shuf)
    preds_shuf = clf_shuf.predict(x_ev_s)
    shuf_acc = float(accuracy_score(y_ev, preds_shuf))
    shuf_mf1 = float(f1_score(y_ev, preds_shuf, average="macro", zero_division=0))
    return {
        "metrics": {
            "acc": round(acc, 4),
            "macro_f1": round(mf1, 4),
            "shuffled_acc": round(shuf_acc, 4),
            "shuffled_macro_f1": round(shuf_mf1, 4),
            "chance_uniform": round(1.0 / max(len(set(y_ev.tolist())), 1), 4),
        },
        "per_item_score": correct,
    }


@torch.no_grad()
def _extract_transfer_embeddings(
    arm_name: str,
    model: torch.nn.Module,
    heads: MultiTaskHeads,
    split_data: dict[str, Any],
    depths: tuple[int, ...],
    batch_size: int = 32,
    return_frames: bool = False,
    extract_halting: bool = False,
) -> dict[str, Any]:
    model.eval()
    heads.eval()
    use_lp = arm_name == "A_ml"
    stride = int(getattr(model, "stride", 8))
    n = int(split_data["lengths"].shape[0])
    z_mean_by_k: dict[int, list[np.ndarray]] = {k: [] for k in depths}
    spk_by_k: dict[int, list[np.ndarray]] = {k: [] for k in depths}
    frames_by_k: dict[int, list[np.ndarray]] = {k: [] for k in depths}
    f1_list: list[np.ndarray] = []
    f2_list: list[np.ndarray] = []
    e_list: list[np.ndarray] = []

    for b0 in range(0, n, batch_size):
        b1 = min(n, b0 + batch_size)
        idx = np.arange(b0, b1)
        e_b, l_b, hs_b, lp_b, lc_b = _sample_batch(
            split_data, idx, use_lp, augment=False
        )
        if extract_halting and isinstance(model, LoopEarModel) and depths == (1, 2, 4):
            ctx = model.prepare_evidence(e_b, l_b, hidden_states=hs_b, use_kv_cache=True)
            z_pad = ctx["z_pad"]
            z_cur, z1 = model.step_refine(ctx["z0"], 1, ctx, use_kv_cache=True)
            f1, p1 = extract_halting_features(
                ctx["e_in"], l_b, ctx["z0"], z1, z_pad, heads, step_k=1
            )
            z_cur, z2 = model.step_refine(z_cur, 2, ctx, use_kv_cache=True)
            f2, _ = extract_halting_features(
                ctx["e_in"], l_b, z1, z2, z_pad, heads, step_k=2, prev_probs=p1
            )
            z_cur, _ = model.step_refine(z_cur, 3, ctx, use_kv_cache=True)
            _, z4 = model.step_refine(z_cur, 4, ctx, use_kv_cache=True)
            states = {1: z1, 2: z2, 4: z4}
            f1_list.append(f1.cpu().numpy())
            f2_list.append(f2.cpu().numpy())
            e_in_np = ctx["e_in"].detach().cpu().numpy()
            for i in range(b1 - b0):
                n_i = min(int(l_b[i].item()), int(e_in_np.shape[1]))
                e_list.append(np.ascontiguousarray(e_in_np[i, : max(n_i, 1)], dtype=np.float16))
        else:
            states, z_pad = model.forward_states(
                e_b, l_b, depths=depths, hidden_states=hs_b, layer_pool=lp_b
            )
        for k in depths:
            z = states[k]
            z_mean_by_k[k].append(F.normalize(masked_mean_pool(z, z_pad), dim=-1).cpu().numpy())
            spk_by_k[k].append(
                heads.speaker_embed(z, z_pad, layer_clip=lc_b).cpu().numpy()
            )
            if return_frames:
                z_np = z.cpu().numpy()
                for i in range(b1 - b0):
                    nf = int(split_data["lengths"][b0 + i])
                    frames_by_k[k].append(upsample_frame_preds(z_np[i], stride, nf))

    out: dict[str, Any] = {
        "z_mean": {k: np.concatenate(z_mean_by_k[k], axis=0) for k in depths},
        "spk": {k: np.concatenate(spk_by_k[k], axis=0) for k in depths},
    }
    if return_frames:
        out["frames_50hz"] = frames_by_k
    if extract_halting and f1_list:
        out["feats_k1"] = np.concatenate(f1_list, axis=0)
        out["feats_k2"] = np.concatenate(f2_list, axis=0)
        out["e_seq"] = e_list
    return out


def _eval_maestro_linear_probe(
    tr_frames: list[np.ndarray],
    tr_y: list[np.ndarray],
    ev_frames: list[np.ndarray],
    ev_y: list[np.ndarray],
    seed: int = 42,
) -> dict[str, Any]:
    x_tr = np.concatenate([f[::5] for f in tr_frames], axis=0)
    y_tr = np.concatenate([y[::5] for y in tr_y], axis=0)
    x_ev = np.concatenate([f[::5] for f in ev_frames], axis=0)
    y_ev = np.concatenate([y[::5] for y in ev_y], axis=0)
    y_tr_bin = (y_tr >= 0.5).astype(np.int32)
    y_ev_bin = (y_ev >= 0.5).astype(np.int32)

    scaler = StandardScaler()
    x_tr_s = scaler.fit_transform(x_tr)
    x_ev_s = scaler.transform(x_ev)
    n_cls = y_tr.shape[1]
    ev_probs_all = np.zeros_like(y_ev, dtype=np.float32)
    shuf_probs_all = np.zeros_like(y_ev, dtype=np.float32)
    win_probs = [np.zeros((f[::5].shape[0], n_cls), dtype=np.float32) for f in ev_frames]
    rng = np.random.default_rng(seed + 2000)
    valid_cls: list[int] = []
    for c in range(n_cls):
        if y_tr_bin[:, c].sum() < 2 or y_tr_bin[:, c].sum() == y_tr_bin.shape[0]:
            continue
        clf = LogisticRegression(C=0.5, max_iter=500, random_state=seed)
        clf.fit(x_tr_s, y_tr_bin[:, c])
        ev_probs_all[:, c] = clf.predict_proba(x_ev_s)[:, 1]
        for wi, f in enumerate(ev_frames):
            win_probs[wi][:, c] = clf.predict_proba(scaler.transform(f[::5]))[:, 1]
        y_shuf = rng.permutation(y_tr_bin[:, c])
        clf_s = LogisticRegression(C=0.5, max_iter=500, random_state=seed)
        clf_s.fit(x_tr_s, y_shuf)
        shuf_probs_all[:, c] = clf_s.predict_proba(x_ev_s)[:, 1]
        if y_ev_bin[:, c].sum() > 0:
            valid_cls.append(c)

    macro_map = float(
        np.mean(
            [
                average_precision_score(y_ev_bin[:, c], ev_probs_all[:, c])
                for c in valid_cls
            ]
        )
    )
    shuf_map = float(
        np.mean(
            [
                average_precision_score(y_ev_bin[:, c], shuf_probs_all[:, c])
                for c in valid_cls
            ]
        )
    )
    micro_f1 = float(
        f1_score(
            y_ev_bin.ravel(),
            (ev_probs_all.ravel() >= 0.5).astype(np.int32),
            zero_division=0,
        )
    )
    per_win_scores = np.zeros((len(ev_frames),), dtype=np.float64)
    for wi, yw in enumerate(ev_y):
        yb = (yw[::5] >= 0.5).astype(np.int32).ravel()
        pb = (win_probs[wi] >= 0.5).astype(np.int32).ravel()
        per_win_scores[wi] = (
            1.0 if (yb.sum() == 0 and pb.sum() == 0) else float(f1_score(yb, pb, zero_division=0))
        )
    return {
        "metrics": {
            "macro_map": round(macro_map, 4),
            "micro_f1": round(micro_f1, 4),
            "shuffled_macro_map": round(shuf_map, 4),
            "n_eval_classes": len(valid_cls),
        },
        "per_item_score": per_win_scores,
    }


def _evaluate_all_transfer_low_ram(
    ckpt_records: list[dict[str, Any]],
    halting_gates: dict[tuple[str, int], HaltingGate],
    flops_map: dict[str, dict[int, float]],
    manifest_dir: Path,
    tap_root: Path,
    tap_index: dict[str, Any],
    s1_root: Path,
    s1_index: dict[str, Any],
) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[tuple[str, int], dict[str, Any]]]:
    assert_zero_pretraining_overlap(["crema_d", "vocalset", "maestro_real"])
    transfer_out: dict[tuple[str, int], dict[str, Any]] = {
        (r["arm"], r["seed"]): {} for r in ckpt_records
    }
    halting_transfer_items: dict[tuple[str, int], dict[str, Any]] = {}

    def _load_ckpt_model(rec: dict[str, Any]):
        payload = torch.load(rec["ckpt_path"], map_location="cpu", weights_only=False)
        model = build_arm(rec["base_arm"], **rec["arm_kwargs"])
        model.load_state_dict(payload["refiner"], strict=True)
        model.eval()
        heads = MultiTaskHeads(**rec["head_kwargs"])
        heads.load_state_dict(payload["heads"], strict=True)
        heads.eval()
        return model, heads

    for corpus in ("crema_d", "vocalset", "maestro_real"):
        man = json.loads((manifest_dir / f"{corpus}.json").read_text(encoding="utf-8"))
        by_id = {it["id"]: it for it in man["items"]}
        sp_tr = _load_tap_split(
            tap_root,
            tap_index["caches"][corpus]["probe_train"]["cache_dir"],
            s1_root=s1_root,
            s1_dir=s1_index["caches"][corpus]["probe_train"]["cache_dir"],
            load_layer_pool=True,
        )
        sp_ev = _load_tap_split(
            tap_root,
            tap_index["caches"][corpus]["probe_eval"]["cache_dir"],
            s1_root=s1_root,
            s1_dir=s1_index["caches"][corpus]["probe_eval"]["cache_dir"],
            load_layer_pool=True,
        )
        tr_y_m: list[np.ndarray] = []
        ev_y_m: list[np.ndarray] = []
        if corpus == "maestro_real":
            n_cls = len(MAESTRO_REAL_CLASSES)
            for it, nf in zip(sp_tr["items"], sp_tr["lengths"]):
                rec_m = by_id[str(it["source_id"])]
                tr_y_m.append(
                    rasterize_soft_labels_50hz(
                        rec_m["soft_labels"],
                        float(it["start_s"]),
                        int(nf),
                        n_cls,
                        hop_e=HOP_E,
                    )
                )
            for it, nf in zip(sp_ev["items"], sp_ev["lengths"]):
                rec_m = by_id[str(it["source_id"])]
                ev_y_m.append(
                    rasterize_soft_labels_50hz(
                        rec_m["soft_labels"],
                        float(it["start_s"]),
                        int(nf),
                        n_cls,
                        hop_e=HOP_E,
                    )
                )

        for rec in ckpt_records:
            arm_name = rec["arm"]
            seed = int(rec["seed"])
            key = (arm_name, seed)
            model, heads = _load_ckpt_model(rec)
            depths = tuple(getattr(model, "depths", (1,)))
            need_halt = key in halting_gates and depths == (1, 2, 4)
            tr_emb = _extract_transfer_embeddings(
                arm_name,
                model,
                heads,
                sp_tr,
                depths=depths,
                return_frames=(corpus == "maestro_real"),
                extract_halting=False,
            )
            ev_emb = _extract_transfer_embeddings(
                arm_name,
                model,
                heads,
                sp_ev,
                depths=depths,
                return_frames=(corpus == "maestro_real"),
                extract_halting=need_halt,
            )

            by_k_res: dict[int, dict[str, Any]] = {}
            per_item_by_k: dict[int, np.ndarray] = {}
            for k in depths:
                if corpus == "crema_d":
                    x_tr, m_tr = _aggregate_clip_embeddings(sp_tr["items"], tr_emb["z_mean"][k], by_id)
                    x_ev, m_ev = _aggregate_clip_embeddings(sp_ev["items"], ev_emb["z_mean"][k], by_id)
                    y_tr = np.array([CREMAD_EMOTIONS.index(m["emotion"]) for m in m_tr], dtype=np.int64)
                    y_ev = np.array([CREMAD_EMOTIONS.index(m["emotion"]) for m in m_ev], dtype=np.int64)
                    emo = _fit_linear_probe(x_tr, y_tr, x_ev, y_ev, seed=seed)
                    spk_ev, m_spk = _aggregate_clip_embeddings(sp_ev["items"], ev_emb["spk"][k], by_id)
                    eer = compute_cross_chapter_eer(
                        spk_ev,
                        np.array([m["speaker_id"] for m in m_spk]),
                        np.array([m["sentence_id"] for m in m_spk]),
                    )
                    by_k_res[k] = {
                        "emotion": emo["metrics"],
                        "speaker_eer": round(float(eer["eer"]), 4),
                        "speaker_1_minus_eer": round(1.0 - float(eer["eer"]), 4),
                    }
                    per_item_by_k[k] = emo["per_item_score"]
                elif corpus == "vocalset":
                    x_tr, m_tr = _aggregate_clip_embeddings(sp_tr["items"], tr_emb["z_mean"][k], by_id)
                    x_ev, m_ev = _aggregate_clip_embeddings(sp_ev["items"], ev_emb["z_mean"][k], by_id)
                    y_tr = np.array(
                        [VOCALSET_TECHNIQUES.index(m["technique"]) for m in m_tr], dtype=np.int64
                    )
                    y_ev = np.array(
                        [VOCALSET_TECHNIQUES.index(m["technique"]) for m in m_ev], dtype=np.int64
                    )
                    tech = _fit_linear_probe(x_tr, y_tr, x_ev, y_ev, seed=seed)
                    by_k_res[k] = tech["metrics"]
                    per_item_by_k[k] = tech["per_item_score"]
                else:
                    mae = _eval_maestro_linear_probe(
                        tr_emb["frames_50hz"][k],
                        tr_y_m,
                        ev_emb["frames_50hz"][k],
                        ev_y_m,
                        seed=seed,
                    )
                    by_k_res[k] = mae["metrics"]
                    per_item_by_k[k] = mae["per_item_score"]

            transfer_out[key][corpus] = by_k_res

            if need_halt and "feats_k1" in ev_emb:
                if corpus in ("crema_d", "vocalset"):
                    by_src_idx: dict[str, list[int]] = {}
                    order_s: list[str] = []
                    for ii, it in enumerate(sp_ev["items"]):
                        sid = str(it["source_id"])
                        if sid not in by_src_idx:
                            by_src_idx[sid] = []
                            order_s.append(sid)
                        by_src_idx[sid].append(ii)
                    f1_agg = np.stack(
                        [ev_emb["feats_k1"][by_src_idx[s]].mean(axis=0) for s in order_s], axis=0
                    )
                    f2_agg = np.stack(
                        [ev_emb["feats_k2"][by_src_idx[s]].mean(axis=0) for s in order_s], axis=0
                    )
                    e_agg = [ev_emb["e_seq"][by_src_idx[s][0]] for s in order_s]
                else:
                    f1_agg = ev_emb["feats_k1"]
                    f2_agg = ev_emb["feats_k2"]
                    e_agg = ev_emb["e_seq"]
                slot = halting_transfer_items.setdefault(
                    key,
                    {"f1": [], "f2": [], "e": [], "s1": [], "s2": [], "s4": [], "per_corpus": {}},
                )
                slot["f1"].append(f1_agg)
                slot["f2"].append(f2_agg)
                slot["e"].append(e_agg)
                slot["s1"].append(per_item_by_k[1])
                slot["s2"].append(per_item_by_k[2])
                slot["s4"].append(per_item_by_k[4])
                slot["per_corpus"][corpus] = evaluate_halting_pareto(
                    halting_gates[key],
                    f1_agg,
                    f2_agg,
                    {1: per_item_by_k[1], 2: per_item_by_k[2], 4: per_item_by_k[4]},
                    flops_map[arm_name],
                    e_agg,
                )

            del model, heads, tr_emb, ev_emb
        del sp_tr, sp_ev
        gc.collect()

    for key, res in transfer_out.items():
        depths = sorted(res["crema_d"].keys())
        comp_by_k: dict[int, float] = {}
        for k in depths:
            comp_by_k[k] = round(
                float(
                    np.mean(
                        [
                            res["crema_d"][k]["emotion"]["acc"],
                            res["crema_d"][k]["speaker_1_minus_eer"],
                            res["vocalset"][k]["acc"],
                            res["maestro_real"][k]["macro_map"],
                        ]
                    )
                ),
                4,
            )
        res["transfer_composite_by_k"] = comp_by_k

    halting_transfer_out: dict[tuple[str, int], dict[str, Any]] = {}
    for key, slot in halting_transfer_items.items():
        arm_name = key[0]
        pooled = evaluate_halting_pareto(
            halting_gates[key],
            np.concatenate(slot["f1"], axis=0),
            np.concatenate(slot["f2"], axis=0),
            {
                1: np.concatenate(slot["s1"], axis=0),
                2: np.concatenate(slot["s2"], axis=0),
                4: np.concatenate(slot["s4"], axis=0),
            },
            flops_map[arm_name],
            [row for chunk in slot["e"] for row in chunk],
        )
        halting_transfer_out[key] = {
            "pooled_zero_overlap_transfer": pooled,
            "per_corpus": slot["per_corpus"],
        }
    return transfer_out, halting_transfer_out


def _verify_live_adaptive_stream(
    ckpt_path: Path,
    manifest_dir: Path,
) -> dict[str, Any]:
    s1 = LoopEarStream.from_checkpoint(ckpt=ckpt_path, depth=2, adaptive=True, device="cpu")
    s2 = LoopEarStream.from_checkpoint(ckpt=ckpt_path, depth=2, adaptive=True, device="cpu")

    ami_man = json.loads((manifest_dir / "ami_sdm.json").read_text(encoding="utf-8"))
    val_items = [it for it in ami_man["items"] if it["split"] == "val"]
    by_spk: dict[str, list[dict[str, Any]]] = {}
    for it in val_items:
        by_spk.setdefault(str(it["speaker_id"]), []).append(it)
    same_a = same_b = diff_b = None
    for spk, rows in by_spk.items():
        by_m: dict[str, dict[str, Any]] = {}
        for r in rows:
            by_m.setdefault(str(r["meeting_id"]), r)
        if len(by_m) >= 2:
            m_list = list(by_m.values())
            same_a, same_b = m_list[0], m_list[1]
            for other_spk, o_rows in by_spk.items():
                if other_spk != spk and o_rows:
                    diff_b = o_rows[0]
                    break
            break
    if same_a is None or same_b is None or diff_b is None:
        raise RuntimeError("Could not find cross-meeting AMI val verification clips")

    wav_a = load_mono_audio(same_a["audio_path"])
    wav_b = load_mono_audio(same_b["audio_path"])
    wav_c = load_mono_audio(diff_b["audio_path"])
    enc1 = s1.encode(wav_a, k=4)
    enc2 = s2.encode(wav_a, k=4)
    reload_diff = float((enc1["vec"] - enc2["vec"]).abs().max().item())
    reload_spk_diff = float((enc1["spk"] - enc2["spk"]).abs().max().item())
    cmp_same = s1.compare(wav_a, wav_b, k=4)
    cmp_diff = s1.compare(wav_a, wav_c, k=4)

    dip_man = json.loads((manifest_dir / "dipco.json").read_text(encoding="utf-8"))
    dip_val = [it for it in dip_man["items"] if it["split"] == "val"]
    st_man = json.loads((manifest_dir / "starss23.json").read_text(encoding="utf-8"))
    st_val = [it for it in st_man["items"] if it["split"] == "val"]
    rd_man = json.loads((manifest_dir / "realdesed.json").read_text(encoding="utf-8"))
    rd_val = [it for it in rd_man["items"] if it["split"] == "val"]

    quiet_depths: list[int] = []
    dense_depths: list[int] = []
    adapt_ms: list[float] = []
    fixed4_ms: list[float] = []
    fixed1_ms: list[float] = []
    quiet_segs: list[np.ndarray] = []
    dense_segs: list[tuple[float, np.ndarray]] = []

    for rec in dip_val:
        wav = load_mono_audio(rec["audio_path"])
        spk_ivs = [(float(a), float(b), str(s)) for a, b, s in rec["speaker_intervals"]]
        dur_s = len(wav) / 16000.0
        for start_s in np.arange(0.0, max(0.0, dur_s - 4.0), 8.0):
            s0 = int(round(float(start_s) * 16000))
            seg = wav[s0 : s0 + 4 * 16000]
            if seg.shape[0] < 4 * 16000:
                continue
            ov = rasterize_overlap_50hz(spk_ivs, float(start_s), 200, hop_e=HOP_E)
            ov2_frac = float(np.mean(ov == 2))
            sil_frac = float(np.mean(ov == 0))
            if ov2_frac >= 0.15:
                dense_segs.append((ov2_frac, seg))
            elif (sil_frac >= 0.50 or ov2_frac == 0.0) and len(quiet_segs) < 16:
                quiet_segs.append(seg)
    dense_segs.sort(key=lambda row: row[0], reverse=True)
    chosen = [("quiet", seg) for seg in quiet_segs[:12]] + [
        ("dense", seg) for _, seg in dense_segs[:12]
    ]
    for kind, seg in chosen:
        t0 = time.perf_counter()
        res_a = s1.encode(seg, adaptive=True)
        adapt_ms.append((time.perf_counter() - t0) * 1000.0)
        t1 = time.perf_counter()
        s1.encode(seg, k=4)
        fixed4_ms.append((time.perf_counter() - t1) * 1000.0)
        t2 = time.perf_counter()
        s1.encode(seg, k=1)
        fixed1_ms.append((time.perf_counter() - t2) * 1000.0)
        if kind == "dense":
            dense_depths.append(int(res_a["depth_used"]))
        else:
            quiet_depths.append(int(res_a["depth_used"]))

    stream_demo = LoopEarStream.from_checkpoint(
        ckpt=ckpt_path, depth=2, adaptive=True, win_s=4.0, step_s=1.0, device="cpu"
    )
    st_wav = load_mono_audio(st_val[0]["audio_path"])[: 12 * 16000]
    rd_wav = load_mono_audio(rd_val[0]["audio_path"])[: 10 * 16000]
    stream_demo.enroll("starss_clip", st_wav[: 4 * 16000])
    state_st = stream_demo.feed(st_wav)
    state_rd = stream_demo.feed(rd_wav)

    def _dist(ds: list[int]) -> dict[str, Any]:
        if not ds:
            return {"n": 0, "avg_k": 0.0, "pct_k1": 0.0, "pct_k2": 0.0, "pct_k4": 0.0}
        arr = np.array(ds, dtype=np.int64)
        return {
            "n": int(arr.size),
            "avg_k": round(float(arr.mean()), 4),
            "pct_k1": round(float(np.mean(arr == 1)), 4),
            "pct_k2": round(float(np.mean(arr == 2)), 4),
            "pct_k4": round(float(np.mean(arr == 4)), 4),
        }

    return {
        "checkpoint": str(ckpt_path),
        "reload_max_abs_diff": reload_diff,
        "reload_spk_max_abs_diff": reload_spk_diff,
        "ami_same_speaker_cross_meeting_cos": cmp_same["speaker_cos"],
        "ami_diff_speaker_cos": cmp_diff["speaker_cos"],
        "ami_cos_margin": round(cmp_same["speaker_cos"] - cmp_diff["speaker_cos"], 4),
        "dipco_window_benchmark": {
            "n_windows": len(adapt_ms),
            "adaptive_mean_ms": round(float(np.mean(adapt_ms)), 2),
            "fixed_k1_mean_ms": round(float(np.mean(fixed1_ms)), 2),
            "fixed_k4_mean_ms": round(float(np.mean(fixed4_ms)), 2),
            "quiet_simple_windows": _dist(quiet_depths),
            "dense_overlapped_windows": _dist(dense_depths),
        },
        "streaming_starss_realdesed": {
            "windows_processed": state_rd["windows_processed"],
            "adaptive": state_rd["adaptive"],
            "halt_threshold": state_rd["halt_threshold"],
            "avg_depth": state_rd["avg_depth"],
            "depth_counts": state_rd["depth_counts"],
            "mean_ms": state_rd["mean_ms"],
            "rtf": state_rd["rtf"],
            "starss_events_detected": len(state_st["events"]),
            "total_events_detected": len(state_rd["events"]),
        },
    }


def _summarize_seeds(vals: list[float]) -> dict[str, float]:
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "mean": round(float(arr.mean()), 4),
        "std": round(float(arr.std(ddof=0)), 4),
        "min": round(float(arr.min()), 4),
        "max": round(float(arr.max()), 4),
    }


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _parse_stage2_train_log(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    out: dict[tuple[str, int], dict[str, Any]] = {}
    if not path.is_file():
        return out
    arm_re = re.compile(
        r"seed=(\d+) (\S+)@\d+: comp=([0-9.]+) .* best_ep=(\d+)/(\d+) in ([0-9.]+)s"
    )
    halt_re = re.compile(
        r"halting\(dev\) (\S+) s=(\d+): auc_learned=([0-9.]+) vs unif=([0-9.]+) "
        r"vs AT=([0-9.]+) overthink_red=([+-][0-9.]+)"
    )
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = arm_re.search(line)
        if m:
            seed = int(m.group(1))
            arm = m.group(2)
            out[(arm, seed)] = {
                "logged_composite": float(m.group(3)),
                "best_epoch": int(m.group(4)),
                "epochs": int(m.group(5)),
                "train_wall_s": float(m.group(6)),
            }
            continue
        h = halt_re.search(line)
        if h:
            arm = h.group(1)
            seed = int(h.group(2))
            slot = out.setdefault((arm, seed), {})
            slot["logged_halting_dev"] = {
                "auc_learned": float(h.group(3)),
                "auc_uniform": float(h.group(4)),
                "auc_audiotetris": float(h.group(5)),
                "overthinking_reduction_vs_k4": float(h.group(6)),
            }
    return out


def _eval_saved_halting_on_val(
    model: LoopEarModel,
    heads: MultiTaskHeads,
    gate: HaltingGate,
    bundle: dict[str, Any],
    flops_by_k: dict[int, float],
    seed: int,
) -> dict[str, Any]:
    va = _extract_loopear_halting_dataset(
        model, heads, bundle, split="val", noise_std=0.0, seed=seed
    )
    primary = evaluate_halting_pareto(
        gate,
        va["feats_k1"],
        va["feats_k2"],
        va["scores_by_k"],
        flops_by_k,
        va["e_seq"].astype(np.float32),
    )
    uniq_groups = sorted(set(va["groups"].tolist()))
    rng = np.random.default_rng(seed + 500)
    rng.shuffle(uniq_groups)
    grp_to_fold = {g: i % 3 for i, g in enumerate(uniq_groups)}
    folds = np.array([grp_to_fold[g] for g in va["groups"]], dtype=np.int64)
    oof_p1 = np.zeros((va["feats_k1"].shape[0],), dtype=np.float64)
    oof_p2 = np.zeros((va["feats_k2"].shape[0],), dtype=np.float64)
    for f_id in range(3):
        m_tr = folds != f_id
        m_te = folds == f_id
        xf = np.concatenate([va["feats_k1"][m_tr], va["feats_k2"][m_tr]], axis=0)
        gf1 = (
            np.maximum(va["scores_by_k"][2][m_tr], va["scores_by_k"][4][m_tr])
            - va["scores_by_k"][1][m_tr]
        )
        gf2 = va["scores_by_k"][4][m_tr] - va["scores_by_k"][2][m_tr]
        yf = np.concatenate([gf1, gf2], axis=0)
        g_fold = fit_halting_gate(
            xf, yf, hidden_dim=32, epochs=220, lr=3e-3, seed=seed + f_id
        )
        with torch.no_grad():
            oof_p1[m_te] = (
                g_fold.continue_prob(
                    torch.from_numpy(va["feats_k1"][m_te].astype(np.float32))
                )
                .cpu()
                .numpy()
            )
            oof_p2[m_te] = (
                g_fold.continue_prob(
                    torch.from_numpy(va["feats_k2"][m_te].astype(np.float32))
                )
                .cpu()
                .numpy()
            )
    s1 = va["scores_by_k"][1]
    s2 = va["scores_by_k"][2]
    s4 = va["scores_by_k"][4]
    u1, u2, u4 = float(np.mean(s1)), float(np.mean(s2)), float(np.mean(s4))
    all_oof = np.concatenate([oof_p1, oof_p2])
    oof_pts: list[dict[str, Any]] = []
    for tau in sorted(
        set(
            [0.0, 1.01]
            + [float(np.quantile(all_oof, q)) for q in np.linspace(0.0, 1.0, 21)]
        )
    ):
        go2 = oof_p1 >= tau
        go4 = go2 & (oof_p2 >= tau)
        ck = np.where(go4, 4, np.where(go2, 2, 1))
        cs = np.where(ck == 4, s4, np.where(ck == 2, s2, s1))
        oof_pts.append(
            {
                "tau": round(float(tau), 4),
                "avg_k": round(float(np.mean(ck)), 4),
                "score": round(float(np.mean(cs)), 5),
                "overthinking_rate": round(float(np.mean(cs < (s1 - 1e-6))), 4),
            }
        )
    oof_best = max(oof_pts, key=lambda r: r["score"])
    oof_k2 = min(oof_pts, key=lambda r: abs(r["avg_k"] - 2.0))
    order = np.argsort([p["avg_k"] for p in oof_pts])
    oof_auc = float(
        np.trapezoid(
            np.asarray([p["score"] for p in oof_pts])[order],
            np.asarray([p["avg_k"] for p in oof_pts])[order],
        )
        / 3.0
    )
    u_auc = float(((u1 + u2) * 0.5 * 1.0 + (u2 + u4) * 0.5 * 2.0) / 3.0)
    return {
        "train_calibrated_primary": primary,
        "val_group_oof_diagnostic": {
            "n_val_groups": len(uniq_groups),
            "best_oof": oof_best,
            "at_budget_k2": oof_k2,
            "oof_pareto_auc": round(oof_auc, 5),
            "uniform_pareto_auc": round(u_auc, 5),
            "oof_minus_uniform_auc": round(oof_auc - u_auc, 5),
            "recomputed_from_saved_val_features": True,
        },
        "recommended_halt_threshold": float(primary["at_budget_k2"]["learned"]["tau1"]),
        "gate_source": "checkpoint_saved_during_training",
    }


def _resume_from_checkpoints(
    ckpt_dir: Path,
    bundle: dict[str, Any],
    flops_map: dict[str, dict[int, float]],
    seeds: list[int],
    train_log: Path,
) -> tuple[
    list[dict[str, Any]],
    dict[tuple[str, int], dict[str, Any]],
    dict[tuple[str, int], dict[int, np.ndarray]],
    dict[tuple[str, int], HaltingGate],
    dict[tuple[str, int], dict[str, Any]],
    dict[tuple[str, int], dict[str, Any]],
]:
    logged = _parse_stage2_train_log(train_log)
    ckpt_records: list[dict[str, Any]] = []
    runs_by_arm_seed: dict[tuple[str, int], dict[str, Any]] = {}
    win_scores_store: dict[tuple[str, int], dict[int, np.ndarray]] = {}
    halting_gates: dict[tuple[str, int], HaltingGate] = {}
    halting_dev_results: dict[tuple[str, int], dict[str, Any]] = {}
    h6_oracle_results: dict[tuple[str, int], dict[str, Any]] = {}
    for seed in seeds:
        for arm_name in STAGE2_ARMS:
            ckpt_file = ckpt_dir / f"{arm_name}_s{seed}.pt"
            if not ckpt_file.is_file():
                raise FileNotFoundError(f"Missing checkpoint {ckpt_file}")
            payload = torch.load(ckpt_file, map_location="cpu", weights_only=False)
            base_arm = str(payload["base_arm"])
            arm_kwargs = dict(payload["refiner_kwargs"])
            head_kwargs = dict(payload["head_kwargs"])
            model = build_arm(base_arm, **arm_kwargs)
            model.load_state_dict(payload["refiner"], strict=True)
            model.eval()
            heads = MultiTaskHeads(**head_kwargs)
            heads.load_state_dict(payload["heads"], strict=True)
            heads.eval()
            ev = _evaluate_arm_val(
                arm_name,
                model,
                heads,
                bundle,
                compute_onset=True,
                collect_oracle_states=(max(getattr(model, "depths", (1,))) >= 4),
            )
            k_max = max(ev["metrics_by_k"].keys())
            m_top = ev["metrics_by_k"][k_max]
            log_row = logged.get((arm_name, seed), {})
            print(
                f"[stage2-resume] seed={seed} {arm_name}@{k_max}: "
                f"comp={m_top['composite']:.4f} "
                f"(logged {log_row.get('logged_composite', float('nan'))}) "
                f"onset100={m_top['timing_mean']['onset_f1_100ms']:.4f}",
                flush=True,
            )
            win_scores_store[(arm_name, seed)] = {
                k: np.concatenate(
                    [
                        ev["window_scores_by_k"][k]["starss23"],
                        ev["window_scores_by_k"][k]["realdesed"],
                        ev["window_scores_by_k"][k]["dipco"],
                    ],
                    axis=0,
                )
                for k in ev["window_scores_by_k"]
            }
            if payload.get("halting_gate") is not None and isinstance(model, LoopEarModel):
                gate = HaltingGate(**dict(payload.get("halting_gate_kwargs", {})))
                gate.load_state_dict(payload["halting_gate"], strict=True)
                gate.eval()
                halting_gates[(arm_name, seed)] = gate
                h_dev = _eval_saved_halting_on_val(
                    model, heads, gate, bundle, flops_map[arm_name], seed
                )
                halting_dev_results[(arm_name, seed)] = h_dev
                p_dev = h_dev["train_calibrated_primary"]
                print(
                    f"[stage2-resume]   halting(dev) {arm_name} s={seed}: "
                    f"auc_learned={p_dev['pareto_auc_avg_score']['learned']:.5f} "
                    f"vs unif={p_dev['pareto_auc_avg_score']['uniform']:.5f} "
                    f"vs AT={p_dev['pareto_auc_avg_score']['audiotetris']:.5f} "
                    f"overthink_red={p_dev['overthinking_reduction_vs_k4']:+.4f}",
                    flush=True,
                )
                if ev.get("oracle_states") is not None:
                    e_cat = np.concatenate(
                        [
                            bundle["starss23"]["val"]["E_taps"][:, -1],
                            bundle["realdesed"]["val"]["E_taps"][:, -1],
                            bundle["dipco"]["val"]["E_taps"][:, -1],
                        ],
                        axis=0,
                    ).astype(np.float32)
                    z_cat = {
                        k: np.concatenate(
                            [
                                ev["oracle_states"]["starss23"][k],
                                ev["oracle_states"]["realdesed"][k],
                                ev["oracle_states"]["dipco"][k],
                            ],
                            axis=0,
                        )
                        for k in (1, 2, 4)
                    }
                    h6_oracle_results[(arm_name, seed)] = evaluate_h6_gate(
                        win_scores_store[(arm_name, seed)],
                        flops_map[arm_name],
                        [e_cat[i] for i in range(e_cat.shape[0])],
                        z_by_k=z_cat,
                    )
            n_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
            n_params += int(sum(p.numel() for p in heads.parameters() if p.requires_grad))
            runs_by_arm_seed[(arm_name, seed)] = {
                "arm": arm_name,
                "seed": seed,
                "n_params": n_params,
                "train_wall_s": log_row.get("train_wall_s"),
                "best_epoch": log_row.get("best_epoch"),
                "last_eval_delta": None,
                "history": [],
                "history_note": (
                    "Per-epoch curves were not stored in the checkpoint. "
                    "best_epoch and train_wall_s are parsed from the training log. "
                    "metrics_by_k are a fresh eval of the saved best weights."
                ),
                "train_log": log_row,
                "learned_weights": _inspect_learned_weights(model, heads),
                "metrics_by_k": ev["metrics_by_k"],
                "best_composite": float(m_top["composite"]),
            }
            ckpt_records.append(
                {
                    "arm": arm_name,
                    "base_arm": base_arm,
                    "arm_kwargs": arm_kwargs,
                    "head_kwargs": head_kwargs,
                    "seed": seed,
                    "ckpt_path": ckpt_file,
                }
            )
            del model, heads, ev
            gc.collect()
    return (
        ckpt_records,
        runs_by_arm_seed,
        win_scores_store,
        halting_gates,
        halting_dev_results,
        h6_oracle_results,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 multi-layer fusion & learned halting")
    parser.add_argument("--manifest-dir", type=Path, default=Path("data/manifests"))
    parser.add_argument("--tap-root", type=Path, default=Path("cache/stage2_taps"))
    parser.add_argument(
        "--tap-index", type=Path, default=Path("data/manifests/stage2_tap_cache_index.json")
    )
    parser.add_argument("--s1-root", type=Path, default=Path("cache/stage1_pilot"))
    parser.add_argument(
        "--s1-index", type=Path, default=Path("data/manifests/pilot_cache_index.json")
    )
    parser.add_argument(
        "--stage0-profile", type=Path, default=Path("results/stage0_profile.json")
    )
    parser.add_argument("--epochs", type=int, default=26)
    parser.add_argument("--steps-per-epoch", type=int, default=28)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume-ckpt-dir", type=Path, default=None)
    parser.add_argument("--train-log", type=Path, default=Path("/tmp/stage2_train.log"))
    parser.add_argument("--ckpt-out", type=Path, default=Path("checkpoints/stage2_best.pt"))
    parser.add_argument(
        "--out", type=Path, default=Path("results/stage2_multilayer_halting.json")
    )
    parser.add_argument(
        "--store-copy",
        type=Path,
        default=Path(
            "/cursor/stores/bc-01a213ed-7a8a-7981-b509-7012340964c4/internal/stage2_multilayer_halting.json"
        ),
    )
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    t_global = time.perf_counter()

    assert_zero_pretraining_overlap(PILOT_CORPUS_SOURCES)
    tap_index = json.loads(args.tap_index.read_text(encoding="utf-8"))
    s1_index = json.loads(args.s1_index.read_text(encoding="utf-8"))
    stage0 = json.loads(args.stage0_profile.read_text(encoding="utf-8"))
    flops_map: dict[str, dict[int, float]] = {
        "E_ml_res_s8": {
            1: float(stage0["arms"]["E_s8"]["gflops_10s"]["refine"]["1"]),
            2: float(stage0["arms"]["E_s8"]["gflops_10s"]["refine"]["2"]),
            4: float(stage0["arms"]["E_s8"]["gflops_10s"]["refine"]["4"]),
        },
        "E_ml_res_s4": {
            1: float(stage0["arms"]["E_s4"]["gflops_10s"]["refine"]["1"]),
            2: float(stage0["arms"]["E_s4"]["gflops_10s"]["refine"]["2"]),
            4: float(stage0["arms"]["E_s4"]["gflops_10s"]["refine"]["4"]),
        },
    }

    t_b0 = time.perf_counter()
    print("[stage2] Loading multi-layer 50 Hz tap bundle (train/val) ...", flush=True)
    bundle = _build_stage2_bundle(
        args.manifest_dir, args.tap_root, tap_index, args.s1_root, s1_index
    )
    print(f"[stage2] Loaded train/val bundle in {time.perf_counter() - t_b0:.1f}s", flush=True)

    tmp_ckpt_dir = Path("/tmp/stage2_ckpts")
    tmp_ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_records: list[dict[str, Any]] = []
    runs_by_arm_seed: dict[tuple[str, int], dict[str, Any]] = {}
    win_scores_store: dict[tuple[str, int], dict[int, np.ndarray]] = {}
    halting_gates: dict[tuple[str, int], HaltingGate] = {}
    halting_dev_results: dict[tuple[str, int], dict[str, Any]] = {}
    h6_oracle_results: dict[tuple[str, int], dict[str, Any]] = {}

    if args.resume_ckpt_dir is not None:
        (
            ckpt_records,
            runs_by_arm_seed,
            win_scores_store,
            halting_gates,
            halting_dev_results,
            h6_oracle_results,
        ) = _resume_from_checkpoints(
            args.resume_ckpt_dir,
            bundle,
            flops_map,
            list(args.seeds),
            args.train_log,
        )
    for seed in args.seeds:
        if args.resume_ckpt_dir is not None:
            break
        for arm_name in STAGE2_ARMS:
            res = _train_one_arm(
                arm_name=arm_name,
                bundle=bundle,
                epochs=args.epochs,
                steps_per_epoch=args.steps_per_epoch,
                batch_size=args.batch_size,
                lr=args.lr,
                seed=seed,
                eval_every=4,
            )
            k_max = max(res["metrics_by_k"].keys())
            m_top = res["metrics_by_k"][k_max]
            print(
                f"[stage2] seed={seed} {arm_name}@{k_max}: comp={m_top['composite']:.4f} "
                f"(st={m_top['starss23_macro_map']:.4f}, rd={m_top['realdesed_macro_map']:.4f}, "
                f"dp={m_top['dipco_macro_f1']:.4f}, ami={m_top['ami_1_minus_eer']:.4f}) "
                f"onset100={m_top['timing_mean']['onset_f1_100ms']:.4f} "
                f"bnd60={m_top['timing_mean']['boundary_f1_60ms']:.4f} "
                f"best_ep={res['best_epoch']}/{args.epochs} in {res['train_wall_s']:.1f}s",
                flush=True,
            )
            win_scores_store[(arm_name, seed)] = {
                k: np.concatenate(
                    [
                        res["window_scores_by_k"][k]["starss23"],
                        res["window_scores_by_k"][k]["realdesed"],
                        res["window_scores_by_k"][k]["dipco"],
                    ],
                    axis=0,
                )
                for k in res["window_scores_by_k"]
            }

            gate_payload: dict[str, Any] | None = None
            rec_thr = 0.5
            if arm_name in ("E_ml_res_s8", "E_ml_res_s4") and isinstance(
                res["model"], LoopEarModel
            ):
                gate, h_dev = _train_and_eval_halting_dev(
                    res["model"],
                    res["heads"],
                    bundle,
                    flops_map[arm_name],
                    seed=seed,
                )
                halting_gates[(arm_name, seed)] = gate
                halting_dev_results[(arm_name, seed)] = h_dev
                gate_payload = gate.state_dict()
                rec_thr = float(h_dev["recommended_halt_threshold"])
                if res["oracle_states"] is not None:
                    e_cat = np.concatenate(
                        [
                            bundle["starss23"]["val"]["E_taps"][:, -1],
                            bundle["realdesed"]["val"]["E_taps"][:, -1],
                            bundle["dipco"]["val"]["E_taps"][:, -1],
                        ],
                        axis=0,
                    ).astype(np.float32)
                    z_cat = {
                        k: np.concatenate(
                            [
                                res["oracle_states"]["starss23"][k],
                                res["oracle_states"]["realdesed"][k],
                                res["oracle_states"]["dipco"][k],
                            ],
                            axis=0,
                        )
                        for k in (1, 2, 4)
                    }
                    h6_oracle_results[(arm_name, seed)] = evaluate_h6_gate(
                        win_scores_store[(arm_name, seed)],
                        flops_map[arm_name],
                        [e_cat[i] for i in range(e_cat.shape[0])],
                        z_by_k=z_cat,
                    )
                p_dev = h_dev["train_calibrated_primary"]
                print(
                    f"[stage2]   halting(dev) {arm_name} s={seed}: "
                    f"auc_learned={p_dev['pareto_auc_avg_score']['learned']:.5f} "
                    f"vs unif={p_dev['pareto_auc_avg_score']['uniform']:.5f} "
                    f"vs AT={p_dev['pareto_auc_avg_score']['audiotetris']:.5f} "
                    f"overthink_red={p_dev['overthinking_reduction_vs_k4']:+.4f}",
                    flush=True,
                )

            ckpt_file = tmp_ckpt_dir / f"{arm_name}_s{seed}.pt"
            torch.save(
                {
                    "arm": arm_name,
                    "base_arm": res["base_arm"],
                    "seed": seed,
                    "refiner_kwargs": res["arm_kwargs"],
                    "head_kwargs": res["head_kwargs"],
                    "labels": list(REALDESED_CLASSES),
                    "tap_layers": list(TAP_LAYERS),
                    "refiner": res["model"].state_dict(),
                    "heads": res["heads"].state_dict(),
                    "halting_gate": gate_payload,
                    "halting_gate_kwargs": {"in_dim": 16, "hidden_dim": 32},
                    "default_halt_threshold": rec_thr,
                },
                ckpt_file,
            )
            ckpt_records.append(
                {
                    "arm": arm_name,
                    "base_arm": res["base_arm"],
                    "arm_kwargs": res["arm_kwargs"],
                    "head_kwargs": res["head_kwargs"],
                    "seed": seed,
                    "ckpt_path": ckpt_file,
                }
            )
            runs_by_arm_seed[(arm_name, seed)] = {
                "arm": arm_name,
                "seed": seed,
                "n_params": res["n_params"],
                "train_wall_s": res["train_wall_s"],
                "best_epoch": res["best_epoch"],
                "last_eval_delta": res["last_eval_delta"],
                "history": res["history"],
                "learned_weights": res["learned_weights"],
                "metrics_by_k": res["metrics_by_k"],
                "best_composite": res["best_composite"],
            }
            del res
            gc.collect()

    paired_cis: dict[str, dict[str, float]] = {}

    def _mean_win(arm: str, depth: int) -> np.ndarray:
        return np.mean(
            np.stack([win_scores_store[(arm, s)][depth] for s in args.seeds], axis=0),
            axis=0,
        )

    comparisons = [
        ("E_ml_res_s8@4_minus_B_comp_ml", ("E_ml_res_s8", 4), ("B_comp_ml", 1)),
        ("E_ml_res_s4@4_minus_B_comp_ml", ("E_ml_res_s4", 4), ("B_comp_ml", 1)),
        ("E_ml_res_s8@4_minus_D_ml@4", ("E_ml_res_s8", 4), ("D_ml", 4)),
        ("E_ml_res_s4@4_minus_D_ml@4", ("E_ml_res_s4", 4), ("D_ml", 4)),
        ("E_ml_res_s4@4_minus_E_ml_res_s8@4", ("E_ml_res_s4", 4), ("E_ml_res_s8", 4)),
        ("E_ml_res_s8@4_minus_E_ml_res_s8@1", ("E_ml_res_s8", 4), ("E_ml_res_s8", 1)),
        ("E_ml_res_s4@4_minus_E_ml_res_s4@1", ("E_ml_res_s4", 4), ("E_ml_res_s4", 1)),
        ("E_ml_res_s8@4_minus_A_ml", ("E_ml_res_s8", 4), ("A_ml", 1)),
        ("E_ml_res_s4@4_minus_A_ml", ("E_ml_res_s4", 4), ("A_ml", 1)),
    ]
    for cname, (a_arm, a_k), (b_arm, b_k) in comparisons:
        paired_cis[cname] = _paired_bootstrap_ci(
            _mean_win(a_arm, a_k), _mean_win(b_arm, b_k), n_resamples=2000, seed=42
        )

    del bundle
    gc.collect()

    print("[stage2] Running zero-overlap frozen transfer + transfer halting ...", flush=True)
    transfer_by_arm_seed, halting_transfer_by_arm_seed = _evaluate_all_transfer_low_ram(
        ckpt_records,
        halting_gates,
        flops_map,
        args.manifest_dir,
        args.tap_root,
        tap_index,
        args.s1_root,
        s1_index,
    )

    arm_summary: dict[str, Any] = {}
    for arm_name in STAGE2_ARMS:
        sample_rec = runs_by_arm_seed[(arm_name, args.seeds[0])]
        k_max = max(sample_rec["metrics_by_k"].keys())
        dev_keys = (
            "composite",
            "starss23_macro_map",
            "realdesed_macro_map",
            "dipco_macro_f1",
            "ami_1_minus_eer",
            "ami_eer",
        )
        timing_keys = (
            "onset_f1_60ms",
            "onset_f1_100ms",
            "onset_f1_200ms",
            "boundary_f1_60ms",
            "boundary_f1_100ms",
            "interior_f1",
        )
        dev_agg = {
            mk: _summarize_seeds(
                [runs_by_arm_seed[(arm_name, s)]["metrics_by_k"][k_max][mk] for s in args.seeds]
            )
            for mk in dev_keys
        }
        timing_agg = {
            tk: _summarize_seeds(
                [
                    runs_by_arm_seed[(arm_name, s)]["metrics_by_k"][k_max]["timing_mean"][tk]
                    for s in args.seeds
                ]
            )
            for tk in timing_keys
        }
        tr_agg = {
            "transfer_composite": _summarize_seeds(
                [
                    transfer_by_arm_seed[(arm_name, s)]["transfer_composite_by_k"][k_max]
                    for s in args.seeds
                ]
            ),
            "crema_d_emotion_acc": _summarize_seeds(
                [
                    transfer_by_arm_seed[(arm_name, s)]["crema_d"][k_max]["emotion"]["acc"]
                    for s in args.seeds
                ]
            ),
            "crema_d_speaker_1_minus_eer": _summarize_seeds(
                [
                    transfer_by_arm_seed[(arm_name, s)]["crema_d"][k_max]["speaker_1_minus_eer"]
                    for s in args.seeds
                ]
            ),
            "vocalset_acc": _summarize_seeds(
                [
                    transfer_by_arm_seed[(arm_name, s)]["vocalset"][k_max]["acc"]
                    for s in args.seeds
                ]
            ),
            "maestro_macro_map": _summarize_seeds(
                [
                    transfer_by_arm_seed[(arm_name, s)]["maestro_real"][k_max]["macro_map"]
                    for s in args.seeds
                ]
            ),
        }
        depth_gains = None
        if k_max >= 4:
            depth_gains = {
                "dev_k4_minus_k1": _summarize_seeds(
                    [
                        runs_by_arm_seed[(arm_name, s)]["metrics_by_k"][4]["composite"]
                        - runs_by_arm_seed[(arm_name, s)]["metrics_by_k"][1]["composite"]
                        for s in args.seeds
                    ]
                ),
                "transfer_k4_minus_k1": _summarize_seeds(
                    [
                        transfer_by_arm_seed[(arm_name, s)]["transfer_composite_by_k"][4]
                        - transfer_by_arm_seed[(arm_name, s)]["transfer_composite_by_k"][1]
                        for s in args.seeds
                    ]
                ),
            }
        arm_summary[arm_name] = {
            "k_eval": k_max,
            "n_params": sample_rec["n_params"],
            "dev": dev_agg,
            "timing_mean": timing_agg,
            "transfer": tr_agg,
            "depth_gains": depth_gains,
        }
        print(
            f"[stage2-summary] {arm_name}@{k_max}: "
            f"dev={dev_agg['composite']['mean']:.4f}±{dev_agg['composite']['std']:.4f} | "
            f"transfer={tr_agg['transfer_composite']['mean']:.4f}±{tr_agg['transfer_composite']['std']:.4f} | "
            f"onset100={timing_agg['onset_f1_100ms']['mean']:.4f} | "
            f"bnd60={timing_agg['boundary_f1_60ms']['mean']:.4f}",
            flush=True,
        )

    best_e_key = max(
        [(a, s) for a in ("E_ml_res_s8", "E_ml_res_s4") for s in args.seeds],
        key=lambda k: (
            0.5 * runs_by_arm_seed[k]["best_composite"]
            + 0.5 * transfer_by_arm_seed[k]["transfer_composite_by_k"][4]
        ),
    )
    args.ckpt_out.parent.mkdir(parents=True, exist_ok=True)
    best_src = tmp_ckpt_dir / f"{best_e_key[0]}_s{best_e_key[1]}.pt"
    shutil.copy2(best_src, args.ckpt_out)
    print(
        f"[stage2] Saved winning checkpoint {best_e_key[0]} seed={best_e_key[1]} -> {args.ckpt_out}",
        flush=True,
    )

    print("[stage2] Verifying live LoopEarStream(adaptive=True) on real val WAVs ...", flush=True)
    stream_verify = _verify_live_adaptive_stream(args.ckpt_out, args.manifest_dir)

    halting_summary: dict[str, Any] = {}
    for arm_name in ("E_ml_res_s8", "E_ml_res_s4"):
        dev_aucs = [
            halting_dev_results[(arm_name, s)]["train_calibrated_primary"]["pareto_auc_avg_score"]
            for s in args.seeds
        ]
        tr_aucs = [
            halting_transfer_by_arm_seed[(arm_name, s)]["pooled_zero_overlap_transfer"][
                "pareto_auc_avg_score"
            ]
            for s in args.seeds
        ]
        halting_summary[arm_name] = {
            "dev_pareto_auc": {
                "learned": _summarize_seeds([x["learned"] for x in dev_aucs]),
                "uniform": _summarize_seeds([x["uniform"] for x in dev_aucs]),
                "audiotetris": _summarize_seeds([x["audiotetris"] for x in dev_aucs]),
                "learned_minus_uniform": _summarize_seeds(
                    [x["learned_minus_uniform"] for x in dev_aucs]
                ),
                "learned_minus_audiotetris": _summarize_seeds(
                    [x["learned_minus_audiotetris"] for x in dev_aucs]
                ),
            },
            "dev_overthinking_reduction_vs_k4": _summarize_seeds(
                [
                    halting_dev_results[(arm_name, s)]["train_calibrated_primary"][
                        "overthinking_reduction_vs_k4"
                    ]
                    for s in args.seeds
                ]
            ),
            "transfer_pareto_auc": {
                "learned": _summarize_seeds([x["learned"] for x in tr_aucs]),
                "uniform": _summarize_seeds([x["uniform"] for x in tr_aucs]),
                "audiotetris": _summarize_seeds([x["audiotetris"] for x in tr_aucs]),
                "learned_minus_uniform": _summarize_seeds(
                    [x["learned_minus_uniform"] for x in tr_aucs]
                ),
                "learned_minus_audiotetris": _summarize_seeds(
                    [x["learned_minus_audiotetris"] for x in tr_aucs]
                ),
            },
            "transfer_overthinking_reduction_vs_k4": _summarize_seeds(
                [
                    halting_transfer_by_arm_seed[(arm_name, s)]["pooled_zero_overlap_transfer"][
                        "overthinking_reduction_vs_k4"
                    ]
                    for s in args.seeds
                ]
            ),
            "per_seed_dev": {str(s): halting_dev_results[(arm_name, s)] for s in args.seeds},
            "per_seed_transfer": {
                str(s): halting_transfer_by_arm_seed[(arm_name, s)] for s in args.seeds
            },
            "h6_oracle_gate_per_seed": {
                str(s): h6_oracle_results[(arm_name, s)]
                for s in args.seeds
                if (arm_name, s) in h6_oracle_results
            },
        }

    total_wall_s = time.perf_counter() - t_global
    payload = {
        "stage": "stage2_multilayer_halting",
        "config": {
            "tap_layers": list(TAP_LAYERS),
            "epochs": args.epochs,
            "steps_per_epoch": args.steps_per_epoch,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seeds": args.seeds,
            "ls_init": 0.3,
            "cloud_gpu_spend_usd": 0.0,
            "total_wall_s": round(total_wall_s, 2),
            "resumed_from_checkpoints": (
                str(args.resume_ckpt_dir) if args.resume_ckpt_dir is not None else None
            ),
        },
        "arm_summary": arm_summary,
        "paired_bootstrap_ci_dev": paired_cis,
        "halting_summary": halting_summary,
        "best_checkpoint": {
            "arm": best_e_key[0],
            "seed": best_e_key[1],
            "path": str(args.ckpt_out),
            "dev_composite": runs_by_arm_seed[best_e_key]["best_composite"],
            "transfer_composite": transfer_by_arm_seed[best_e_key]["transfer_composite_by_k"][4],
        },
        "live_adaptive_stream_verification": stream_verify,
        "per_seed_runs": {
            f"{arm}_s{seed}": {
                **runs_by_arm_seed[(arm, seed)],
                "transfer": transfer_by_arm_seed[(arm, seed)],
            }
            for seed in args.seeds
            for arm in STAGE2_ARMS
        },
    }
    write_result_json(args.out, _jsonable(payload))
    if args.store_copy is not None:
        args.store_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.out, args.store_copy)
    write_run_manifest(
        "stage2_multilayer_halting",
        {
            "stage": "stage2_multilayer_halting",
            "seeds": args.seeds,
            "epochs": args.epochs,
            "steps_per_epoch": args.steps_per_epoch,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "tap_layers": list(TAP_LAYERS),
            "cloud_gpu_spend_usd": 0.0,
        },
        ledger_dir="ledger",
    )
    print(f"[stage2] DONE in {total_wall_s:.1f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
