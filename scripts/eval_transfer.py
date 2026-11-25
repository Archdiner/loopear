import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score
from sklearn.preprocessing import StandardScaler

from loopear.data import (
    CREMAD_EMOTIONS,
    MAESTRO_REAL_CLASSES,
    VOCALSET_TECHNIQUES,
    rasterize_soft_labels_50hz,
    upsample_frame_preds,
)
from loopear.encoder import HOP_E, ShardedFeatureCache
from loopear.harness import assert_zero_pretraining_overlap, write_result_json
from loopear.losses import MultiTaskHeads, compute_cross_chapter_eer, masked_mean_pool
from loopear.models import build_arm


def _load_cached_split(
    cache_root: Path,
    cache_dir: str,
    load_layer_pool: bool = False,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray | None]:
    cache = ShardedFeatureCache(cache_root)
    idx = cache.read_index(cache_dir)
    items = idx["items"]
    shards: dict[str, dict[str, np.ndarray]] = {}
    e_list: list[np.ndarray] = []
    l_list: list[np.ndarray] = []
    lp_list: list[np.ndarray] = []
    seen_shards: list[str] = []
    for it in items:
        sh = str(it["shard"])
        if sh not in shards:
            shards[sh] = cache.load_shard(cache_dir, sh)
            seen_shards.append(sh)
    for sh in seen_shards:
        data = shards[sh]
        e_list.append(data["E"])
        l_list.append(data["lengths"])
        if load_layer_pool and "layer_pool" in data:
            lp_list.append(data["layer_pool"])
    e_all = np.concatenate(e_list, axis=0)
    l_all = np.concatenate(l_list, axis=0)
    lp_all = np.concatenate(lp_list, axis=0) if lp_list else None
    return items, e_all, l_all, lp_all


@torch.no_grad()
def _extract_arm_window_embeddings(
    model: torch.nn.Module,
    heads: MultiTaskHeads,
    e_arr: np.ndarray,
    l_arr: np.ndarray,
    lp_arr: np.ndarray | None,
    depth: int,
    batch_size: int = 32,
    return_frames: bool = False,
) -> dict[str, Any]:
    model.eval()
    heads.eval()
    n = int(e_arr.shape[0])
    z_mean_list: list[np.ndarray] = []
    spk_list: list[np.ndarray] = []
    frame_list: list[np.ndarray] = []
    stride = int(getattr(model, "stride", 8))
    for b0 in range(0, n, batch_size):
        b1 = min(n, b0 + batch_size)
        e_b = torch.from_numpy(e_arr[b0:b1].astype(np.float32))
        l_b = torch.from_numpy(l_arr[b0:b1].astype(np.int64))
        lp_b = (
            torch.from_numpy(lp_arr[b0:b1].astype(np.float32))
            if lp_arr is not None
            else None
        )
        states, z_pad = model.forward_states(
            e_b, l_b, depths=(depth,), layer_pool=lp_b
        )
        z = states[depth]
        z_mean = F.normalize(masked_mean_pool(z, z_pad), dim=-1)
        spk = heads.speaker_embed(z, z_pad)
        z_mean_list.append(z_mean.cpu().numpy())
        spk_list.append(spk.cpu().numpy())
        if return_frames:
            z_np = z.cpu().numpy()
            for i in range(b1 - b0):
                nf = int(l_arr[b0 + i])
                up = upsample_frame_preds(z_np[i], stride=stride, n_frames=nf)
                frame_list.append(up)
    out: dict[str, Any] = {
        "z_mean": np.concatenate(z_mean_list, axis=0),
        "spk": np.concatenate(spk_list, axis=0),
    }
    if return_frames:
        out["frames_50hz"] = frame_list
    return out


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


def _fit_linear_multiclass(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_ev: np.ndarray,
    y_ev: np.ndarray,
    seed: int = 42,
) -> dict[str, float]:
    scaler = StandardScaler()
    x_tr_s = scaler.fit_transform(x_tr)
    x_ev_s = scaler.transform(x_ev)
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
    clf.fit(x_tr_s, y_tr)
    preds = clf.predict(x_ev_s)
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
        "acc": round(acc, 4),
        "macro_f1": round(mf1, 4),
        "shuffled_acc": round(shuf_acc, 4),
        "shuffled_macro_f1": round(shuf_mf1, 4),
        "chance_uniform": round(1.0 / max(len(set(y_ev.tolist())), 1), 4),
    }


def _eval_maestro_segments(
    tr_items: list[dict[str, Any]],
    tr_frames: list[np.ndarray],
    ev_items: list[dict[str, Any]],
    ev_frames: list[np.ndarray],
    manifest_items: dict[str, dict[str, Any]],
    seed: int = 42,
) -> dict[str, float]:
    frames_per_sec = int(round(1.0 / HOP_E))
    n_cls = len(MAESTRO_REAL_CLASSES)

    def _build_1s(items, frames_list):
        x_rows: list[np.ndarray] = []
        y_rows: list[np.ndarray] = []
        for it, fr in zip(items, frames_list):
            rec = manifest_items[str(it["source_id"])]
            start_s = float(it["start_s"])
            nf = int(fr.shape[0])
            soft = rasterize_soft_labels_50hz(
                rec["soft_labels"], start_s, nf, n_cls, hop_e=HOP_E
            )
            n_sec = int(math.floor(nf / frames_per_sec))
            for s in range(n_sec):
                f0 = s * frames_per_sec
                f1 = f0 + frames_per_sec
                x_rows.append(fr[f0:f1].mean(axis=0))
                y_rows.append((soft[f0:f1].mean(axis=0) >= 0.5).astype(np.int32))
        return np.stack(x_rows, axis=0), np.stack(y_rows, axis=0)

    x_tr, y_tr = _build_1s(tr_items, tr_frames)
    x_ev, y_ev = _build_1s(ev_items, ev_frames)
    scaler = StandardScaler()
    x_tr_s = scaler.fit_transform(x_tr)
    x_ev_s = scaler.transform(x_ev)

    rng = np.random.default_rng(seed + 2000)
    aps: list[float] = []
    shuf_aps: list[float] = []
    f1s: list[float] = []
    shuf_f1s: list[float] = []
    valid_cols: list[int] = []
    probs_ev = np.zeros((y_ev.shape[0], n_cls), dtype=np.float32)
    probs_shuf = np.zeros((y_ev.shape[0], n_cls), dtype=np.float32)

    for c in range(n_cls):
        if y_tr[:, c].sum() < 5 or (y_tr.shape[0] - y_tr[:, c].sum()) < 5:
            continue
        if y_ev[:, c].sum() < 2 or (y_ev.shape[0] - y_ev[:, c].sum()) < 2:
            continue
        valid_cols.append(c)
        clf = LogisticRegression(
            C=0.5, max_iter=1000, class_weight="balanced", random_state=seed
        )
        clf.fit(x_tr_s, y_tr[:, c])
        p = clf.predict_proba(x_ev_s)[:, 1]
        probs_ev[:, c] = p
        aps.append(float(average_precision_score(y_ev[:, c], p)))
        f1s.append(float(f1_score(y_ev[:, c], (p >= 0.5).astype(np.int32), zero_division=0)))

        y_perm = rng.permutation(y_tr[:, c])
        clf_s = LogisticRegression(
            C=0.5, max_iter=1000, class_weight="balanced", random_state=seed
        )
        clf_s.fit(x_tr_s, y_perm)
        ps = clf_s.predict_proba(x_ev_s)[:, 1]
        probs_shuf[:, c] = ps
        shuf_aps.append(float(average_precision_score(y_ev[:, c], ps)))
        shuf_f1s.append(
            float(f1_score(y_ev[:, c], (ps >= 0.5).astype(np.int32), zero_division=0))
        )

    y_ev_sub = y_ev[:, valid_cols]
    micro_ap = float(
        average_precision_score(y_ev_sub.ravel(), probs_ev[:, valid_cols].ravel())
    )
    shuf_micro_ap = float(
        average_precision_score(y_ev_sub.ravel(), probs_shuf[:, valid_cols].ravel())
    )
    return {
        "n_active_classes": len(valid_cols),
        "n_eval_segments": int(y_ev.shape[0]),
        "macro_map": round(float(np.mean(aps)), 4),
        "micro_map": round(micro_ap, 4),
        "macro_f1": round(float(np.mean(f1s)), 4),
        "shuffled_macro_map": round(float(np.mean(shuf_aps)), 4),
        "shuffled_micro_map": round(shuf_micro_ap, 4),
        "shuffled_macro_f1": round(float(np.mean(shuf_f1s)), 4),
    }


def evaluate_frozen_transfer(
    arm_name: str,
    model: torch.nn.Module,
    heads: MultiTaskHeads,
    manifest_dir: Path,
    cache_root: Path,
    cache_index: dict[str, Any],
    depths: tuple[int, ...] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    assert_zero_pretraining_overlap(["crema_d", "vocalset", "maestro_real"])
    use_lp = arm_name == "A_ml"
    target_depths = depths if depths is not None else tuple(getattr(model, "depths", (1,)))
    caches = cache_index["caches"]

    cremad_man = json.loads((manifest_dir / "crema_d.json").read_text(encoding="utf-8"))
    cremad_by_id = {it["id"]: it for it in cremad_man["items"]}
    vs_man = json.loads((manifest_dir / "vocalset.json").read_text(encoding="utf-8"))
    vs_by_id = {it["id"]: it for it in vs_man["items"]}
    mr_man = json.loads((manifest_dir / "maestro_real.json").read_text(encoding="utf-8"))
    mr_by_id = {it["id"]: it for it in mr_man["items"]}

    cd_results: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    cd_tr = _load_cached_split(cache_root, caches["crema_d"]["probe_train"]["cache_dir"], use_lp)
    cd_ev = _load_cached_split(cache_root, caches["crema_d"]["probe_eval"]["cache_dir"], use_lp)
    for k in target_depths:
        cd_tr_out = _extract_arm_window_embeddings(model, heads, cd_tr[1], cd_tr[2], cd_tr[3], k)
        cd_ev_out = _extract_arm_window_embeddings(model, heads, cd_ev[1], cd_ev[2], cd_ev[3], k)
        x_cd_tr, m_cd_tr = _aggregate_clip_embeddings(cd_tr[0], cd_tr_out["z_mean"], cremad_by_id)
        x_cd_ev, m_cd_ev = _aggregate_clip_embeddings(cd_ev[0], cd_ev_out["z_mean"], cremad_by_id)
        spk_cd_ev, _ = _aggregate_clip_embeddings(cd_ev[0], cd_ev_out["spk"], cremad_by_id)
        y_cd_tr = np.array([m["emotion_idx"] for m in m_cd_tr], dtype=np.int64)
        y_cd_ev = np.array([m["emotion_idx"] for m in m_cd_ev], dtype=np.int64)
        cremad_emo = _fit_linear_multiclass(x_cd_tr, y_cd_tr, x_cd_ev, y_cd_ev, seed=seed)

        spk_ids = np.array([m["speaker_id"] for m in m_cd_ev])
        sent_ids = np.array([m["sentence_id"] for m in m_cd_ev])
        eer_z = compute_cross_chapter_eer(x_cd_ev, spk_ids, sent_ids)
        eer_spk = compute_cross_chapter_eer(spk_cd_ev, spk_ids, sent_ids)
        rng = np.random.default_rng(seed + 3000)
        shuf_spk = rng.permutation(spk_ids)
        eer_shuf = compute_cross_chapter_eer(x_cd_ev, shuf_spk, sent_ids)
        cremad_eer = {
            "backbone_eer": round(float(eer_z["eer"]), 4),
            "backbone_1_minus_eer": round(1.0 - float(eer_z["eer"]), 4),
            "speaker_head_eer": round(float(eer_spk["eer"]), 4),
            "speaker_head_1_minus_eer": round(1.0 - float(eer_spk["eer"]), 4),
            "shuffled_eer": round(float(eer_shuf["eer"]), 4),
            "n_pairs": int(eer_z["n_pairs"]),
            "n_eval_actors": len(set(spk_ids.tolist())),
        }
        cd_results[str(k)] = (cremad_emo, cremad_eer)
    del cd_tr, cd_ev

    vs_results: dict[str, dict[str, Any]] = {}
    vs_tr = _load_cached_split(cache_root, caches["vocalset"]["probe_train"]["cache_dir"], use_lp)
    vs_ev = _load_cached_split(cache_root, caches["vocalset"]["probe_eval"]["cache_dir"], use_lp)
    for k in target_depths:
        vs_tr_out = _extract_arm_window_embeddings(model, heads, vs_tr[1], vs_tr[2], vs_tr[3], k)
        vs_ev_out = _extract_arm_window_embeddings(model, heads, vs_ev[1], vs_ev[2], vs_ev[3], k)
        x_vs_tr, m_vs_tr = _aggregate_clip_embeddings(vs_tr[0], vs_tr_out["z_mean"], vs_by_id)
        x_vs_ev, m_vs_ev = _aggregate_clip_embeddings(vs_ev[0], vs_ev_out["z_mean"], vs_by_id)
        y_vs_tr = np.array([m["technique_idx"] for m in m_vs_tr], dtype=np.int64)
        y_vs_ev = np.array([m["technique_idx"] for m in m_vs_ev], dtype=np.int64)
        vs_results[str(k)] = _fit_linear_multiclass(x_vs_tr, y_vs_tr, x_vs_ev, y_vs_ev, seed=seed)
    del vs_tr, vs_ev

    mr_results: dict[str, dict[str, Any]] = {}
    mr_tr = _load_cached_split(cache_root, caches["maestro_real"]["probe_train"]["cache_dir"], use_lp)
    mr_ev = _load_cached_split(cache_root, caches["maestro_real"]["probe_eval"]["cache_dir"], use_lp)
    for k in target_depths:
        mr_tr_out = _extract_arm_window_embeddings(
            model, heads, mr_tr[1], mr_tr[2], mr_tr[3], k, return_frames=True
        )
        mr_ev_out = _extract_arm_window_embeddings(
            model, heads, mr_ev[1], mr_ev[2], mr_ev[3], k, return_frames=True
        )
        mr_results[str(k)] = _eval_maestro_segments(
            mr_tr[0],
            mr_tr_out["frames_50hz"],
            mr_ev[0],
            mr_ev_out["frames_50hz"],
            mr_by_id,
            seed=seed,
        )
    del mr_tr, mr_ev

    by_depth: dict[str, Any] = {}
    for k in target_depths:
        ks = str(k)
        cremad_emo, cremad_eer = cd_results[ks]
        vocalset_res = vs_results[ks]
        maestro_res = mr_results[ks]
        transfer_composite = float(
            0.25 * cremad_emo["acc"]
            + 0.25 * cremad_eer["backbone_1_minus_eer"]
            + 0.25 * vocalset_res["acc"]
            + 0.25 * maestro_res["macro_map"]
        )
        by_depth[ks] = {
            "cremad_emotion": cremad_emo,
            "cremad_cross_sentence_eer": cremad_eer,
            "vocalset_technique": vocalset_res,
            "maestro_real_soft_events": maestro_res,
            "transfer_composite": round(transfer_composite, 4),
        }
    return {
        "arm": arm_name,
        "classes": {
            "crema_d": list(CREMAD_EMOTIONS),
            "vocalset": list(VOCALSET_TECHNIQUES),
            "maestro_real": list(MAESTRO_REAL_CLASSES),
        },
        "by_depth": by_depth,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", default="data/manifests")
    parser.add_argument("--cache-root", default="cache/stage1_pilot")
    parser.add_argument("--ckpt-dir", default="checkpoints")
    parser.add_argument("--out", default="results/stage1_transfer.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest_dir = Path(args.manifest_dir)
    cache_root = Path(args.cache_root)
    ckpt_dir = Path(args.ckpt_dir)
    cache_index = json.loads(
        (manifest_dir / "pilot_cache_index.json").read_text(encoding="utf-8")
    )
    results: dict[str, Any] = {}
    for ckpt_path in sorted(ckpt_dir.glob("stage1_arm_*.pt")):
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        arm_name = str(payload["arm_name"])
        arm_type = str(payload["arm_type"])
        model = build_arm(arm_type, **payload["refiner_kwargs"])
        model.load_state_dict(payload["refiner"], strict=True)
        heads = MultiTaskHeads(**payload["head_kwargs"])
        heads.load_state_dict(payload["heads"], strict=True)
        res = evaluate_frozen_transfer(
            arm_name,
            model,
            heads,
            manifest_dir,
            cache_root,
            cache_index,
            seed=args.seed,
        )
        results[arm_name] = res
        k_max = str(max(getattr(model, "depths", (1,))))
        m = res["by_depth"][k_max]
        print(
            f"[transfer] {arm_name}@{k_max}: CREMA-D emo={m['cremad_emotion']['acc']:.3f} "
            f"(shuf={m['cremad_emotion']['shuffled_acc']:.3f}), "
            f"EER={m['cremad_cross_sentence_eer']['backbone_eer']:.3f} "
            f"(spk_head={m['cremad_cross_sentence_eer']['speaker_head_eer']:.3f}, shuf={m['cremad_cross_sentence_eer']['shuffled_eer']:.3f}), "
            f"VocalSet={m['vocalset_technique']['acc']:.3f} (shuf={m['vocalset_technique']['shuffled_acc']:.3f}), "
            f"MAESTRO={m['maestro_real_soft_events']['macro_map']:.3f} (shuf={m['maestro_real_soft_events']['shuffled_macro_map']:.3f})",
            flush=True,
        )
    write_result_json(args.out, {"transfer": results})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
