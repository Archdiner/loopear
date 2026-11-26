import argparse
import copy
import gc
import hashlib
import json
import math
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
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_stage2 as stage2
from loopear.data import (
    PILOT_CORPUS_SOURCES,
    REALDESED_CLASSES,
    SEALED_SPLITS,
    STARSS23_CLASSES,
    compute_onset_and_boundary_metrics,
    upsample_frame_preds,
)
from loopear.encoder import HOP_E, T_E_MAX, TAP_LAYERS
from loopear.harness import (
    SealedSplitGuard,
    assert_zero_pretraining_overlap,
    write_result_json,
    write_run_manifest,
)
from loopear.losses import (
    MultiTaskHeads,
    fullrate_multilabel_losses_by_depth,
    mean_depth_loss,
    monotonic_penalty,
    overlap_losses_by_depth,
    set_seed,
    speaker_losses_by_depth,
)
from loopear.models import FullRateResidualHead, LoopEarModel

ARM_NAME = "E_depth_probe"
MAX_EPOCHS = 8
ONLY_SEED = 42


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _t_cap(lengths: torch.Tensor) -> int:
    return min(T_E_MAX, int(math.ceil(max(int(lengths.max().item()), 8) / 8.0) * 8))


def _run_states(
    model: LoopEarModel,
    e: torch.Tensor,
    lengths: torch.Tensor,
    hidden: torch.Tensor | None,
) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
    ctx = model.prepare_evidence(e, lengths, hidden_states=hidden, use_kv_cache=True)
    z = ctx["z0"]
    states: dict[int, torch.Tensor] = {}
    for k in range(1, max(model.depths) + 1):
        z, z_out = model.step_refine(z, k, ctx, use_kv_cache=True)
        if k in model.depths:
            states[k] = z_out
    return states, ctx


def _task_plus_penalty(per_depth: dict[int, torch.Tensor]) -> torch.Tensor:
    return mean_depth_loss(per_depth) + monotonic_penalty(per_depth)


def _build_model(n_spk: int) -> tuple[LoopEarModel, torch.nn.ModuleDict, MultiTaskHeads, dict[str, Any]]:
    model = LoopEarModel(
        stride=8,
        max_k=4,
        use_multilayer=True,
        n_layers=len(TAP_LAYERS),
        use_local_stem=True,
        ls=0.3,
    )
    readouts = torch.nn.ModuleDict(
        {
            "starss": FullRateResidualHead(model.out_dim, len(STARSS23_CLASSES)),
            "ev": FullRateResidualHead(model.out_dim, len(REALDESED_CLASSES)),
        }
    )
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
    return model, readouts, heads, head_kwargs


def _arm_kwargs() -> dict[str, Any]:
    return {
        "stride": 8,
        "max_k": 4,
        "use_multilayer": True,
        "n_layers": len(TAP_LAYERS),
        "use_local_stem": True,
        "ls": 0.3,
    }


@torch.no_grad()
def _evaluate_probe(
    model: LoopEarModel,
    readouts: torch.nn.ModuleDict,
    heads: MultiTaskHeads,
    bundle: dict[str, Any],
    batch_size: int = 24,
    compute_onset: bool = True,
) -> dict[str, Any]:
    model.eval()
    readouts.eval()
    heads.eval()
    depths = tuple(model.depths)
    stride = int(model.stride)

    def _event_probs(corpus: str, readout_name: str) -> dict[int, list[np.ndarray]]:
        val = bundle[corpus]["val"]
        n = int(val["lengths"].shape[0])
        probs = {k: [] for k in depths}
        readout = readouts[readout_name]
        for b0 in range(0, n, batch_size):
            b1 = min(n, b0 + batch_size)
            idx = np.arange(b0, b1)
            e_b, l_b, hs_b, _, _ = stage2._sample_batch(val, idx, False, augment=False)
            states, ctx = _run_states(model, e_b, l_b, hs_b)
            logits = readout.logits_by_depth(
                ctx["e_norm"],
                ctx["e_pad"],
                states,
                stride,
                freeze_first=False,
            )
            for k in depths:
                p = torch.sigmoid(logits[k]).cpu().numpy()
                for i in range(b1 - b0):
                    nf = int(val["lengths"][b0 + i])
                    probs[k].append(np.ascontiguousarray(p[i, :nf]))
        return probs

    def _overlap_probs() -> dict[int, list[np.ndarray]]:
        val = bundle["dipco"]["val"]
        n = int(val["lengths"].shape[0])
        probs = {k: [] for k in depths}
        for b0 in range(0, n, batch_size):
            b1 = min(n, b0 + batch_size)
            idx = np.arange(b0, b1)
            e_b, l_b, hs_b, _, _ = stage2._sample_batch(val, idx, False, augment=False)
            states, ctx = _run_states(model, e_b, l_b, hs_b)
            for k in depths:
                p = torch.softmax(heads.overlap_logits(states[k]), dim=-1).cpu().numpy()
                for i in range(b1 - b0):
                    nf = int(val["lengths"][b0 + i])
                    probs[k].append(upsample_frame_preds(p[i], stride, nf))
        return probs

    st_probs = _event_probs("starss23", "starss")
    rd_probs = _event_probs("realdesed", "ev")
    dp_probs = _overlap_probs()

    ami_val = bundle["ami_sdm"]["val"]
    n_ami = int(ami_val["lengths"].shape[0])
    ami_embs: dict[int, list[np.ndarray]] = {k: [] for k in depths}
    for b0 in range(0, n_ami, batch_size):
        b1 = min(n_ami, b0 + batch_size)
        idx = np.arange(b0, b1)
        e_b, l_b, hs_b, _, lc_b = stage2._sample_batch(ami_val, idx, False, augment=False)
        states, ctx = _run_states(model, e_b, l_b, hs_b)
        for k in depths:
            ami_embs[k].append(
                heads.speaker_embed(states[k], ctx["z_pad"], layer_clip=lc_b).cpu().numpy()
            )

    def _trim_list(corpus: str) -> list[np.ndarray]:
        val = bundle[corpus]["val"]
        return [
            val["y_50hz"][i, : int(val["lengths"][i])]
            for i in range(int(val["lengths"].shape[0]))
        ]

    st_y = _trim_list("starss23")
    rd_y = _trim_list("realdesed")
    dp_y = _trim_list("dipco")
    metrics_by_k: dict[int, dict[str, Any]] = {}
    for k in depths:
        st_y_all = np.concatenate(st_y, axis=0)
        st_p_all = np.concatenate(st_probs[k], axis=0)
        st_active = [c for c in range(st_y_all.shape[1]) if st_y_all[:, c].sum() > 0]
        st_map = float(
            np.mean([average_precision_score(st_y_all[:, c], st_p_all[:, c]) for c in st_active])
        )
        rd_y_all = np.concatenate(rd_y, axis=0)
        rd_p_all = np.concatenate(rd_probs[k], axis=0)
        rd_active = [c for c in range(rd_y_all.shape[1]) if rd_y_all[:, c].sum() > 0]
        rd_map = float(
            np.mean([average_precision_score(rd_y_all[:, c], rd_p_all[:, c]) for c in rd_active])
        )
        dp_y_all = np.concatenate(dp_y, axis=0)
        dp_p_all = np.concatenate(dp_probs[k], axis=0)
        dp_pred = np.argmax(dp_p_all, axis=-1)
        dp_mf1 = float(f1_score(dp_y_all, dp_pred, average="macro", zero_division=0))
        embs = np.concatenate(ami_embs[k], axis=0)
        ami = stage2.compute_cross_chapter_eer(embs, ami_val["speaker_id"], ami_val["meeting_id"])
        ami_inv = 1.0 - float(ami["eer"])
        comp = float(np.mean([st_map, rd_map, dp_mf1, ami_inv]))
        entry: dict[str, Any] = {
            "starss23_macro_map": round(st_map, 4),
            "realdesed_macro_map": round(rd_map, 4),
            "dipco_macro_f1": round(dp_mf1, 4),
            "ami_eer": round(float(ami["eer"]), 4),
            "ami_1_minus_eer": round(ami_inv, 4),
            "composite": round(comp, 4),
            "composite_raw": comp,
        }
        if compute_onset:
            st_on = compute_onset_and_boundary_metrics(st_y, st_probs[k], hop_e=HOP_E)
            rd_on = compute_onset_and_boundary_metrics(rd_y, rd_probs[k], hop_e=HOP_E)
            entry["timing_starss23"] = st_on
            entry["timing_realdesed"] = rd_on
            entry["timing_mean"] = {
                key: round(0.5 * (float(st_on[key]) + float(rd_on[key])), 4) for key in st_on
            }
            entry["onset_f1_100ms_raw"] = 0.5 * (
                float(st_on["onset_f1_100ms"]) + float(rd_on["onset_f1_100ms"])
            )
            entry["window_f1_mean"] = round(
                float(
                    np.mean(
                        [
                            np.mean(
                                [
                                    stage2._compute_window_dense_score(
                                        "starss23", st_y[i], st_probs[k][i]
                                    )
                                    for i in range(len(st_y))
                                ]
                            ),
                            np.mean(
                                [
                                    stage2._compute_window_dense_score(
                                        "realdesed", rd_y[i], rd_probs[k][i]
                                    )
                                    for i in range(len(rd_y))
                                ]
                            ),
                            np.mean(
                                [
                                    stage2._compute_window_dense_score(
                                        "dipco", dp_y[i], dp_probs[k][i]
                                    )
                                    for i in range(len(dp_y))
                                ]
                            ),
                        ]
                    )
                ),
                4,
            )
        metrics_by_k[k] = entry
    return {"metrics_by_k": metrics_by_k}


def _train_probe(
    bundle: dict[str, Any],
    epochs: int,
    steps_per_epoch: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> dict[str, Any]:
    set_seed(seed)
    rng = np.random.default_rng(seed)
    n_spk = int(bundle["ami_sdm"]["manifest"]["n_train_speakers"])
    model, readouts, heads, head_kwargs = _build_model(n_spk)
    stride_targets = stage2._prepare_stride_targets(bundle, int(model.stride))
    params = list(model.parameters()) + list(readouts.parameters()) + list(heads.parameters())
    n_params = int(sum(p.numel() for p in params if p.requires_grad))
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
    total_steps = epochs * steps_per_epoch
    warmup_steps = max(1, min(steps_per_epoch, total_steps // 6))

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
    best_comp = -1.0
    best_epoch = 0
    best_state: tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None = None
    history: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    for ep in range(1, epochs + 1):
        model.train()
        readouts.train()
        heads.train()
        loss_acc = 0.0
        pen_acc = 0.0
        for _ in range(steps_per_epoch):
            opt.zero_grad(set_to_none=True)
            i_st = rng.integers(0, n_st, size=batch_size)
            e_st, l_st, hs_st, _, _ = stage2._sample_batch(
                bundle["starss23"]["train"], i_st, False, augment=True
            )
            st_states, st_ctx = _run_states(model, e_st, l_st, hs_st)
            y_st = torch.from_numpy(
                bundle["starss23"]["train"]["y_50hz"][i_st, : _t_cap(l_st)]
            )
            y_st_c = torch.from_numpy(bundle["starss23"]["train"]["y_clip"][i_st])
            st_logits = readouts["starss"].logits_by_depth(
                st_ctx["e_norm"],
                st_ctx["e_pad"],
                st_states,
                int(model.stride),
                freeze_first=True,
            )
            st_losses = fullrate_multilabel_losses_by_depth(
                st_logits, y_st, st_ctx["e_pad"], y_st_c, pos_weight=pw_st
            )

            i_rd = rng.integers(0, n_rd, size=batch_size)
            e_rd, l_rd, hs_rd, _, _ = stage2._sample_batch(
                bundle["realdesed"]["train"], i_rd, False, augment=True
            )
            rd_states, rd_ctx = _run_states(model, e_rd, l_rd, hs_rd)
            y_rd = torch.from_numpy(
                bundle["realdesed"]["train"]["y_50hz"][i_rd, : _t_cap(l_rd)]
            )
            y_rd_c = torch.from_numpy(bundle["realdesed"]["train"]["y_clip"][i_rd])
            rd_logits = readouts["ev"].logits_by_depth(
                rd_ctx["e_norm"],
                rd_ctx["e_pad"],
                rd_states,
                int(model.stride),
                freeze_first=True,
            )
            rd_losses = fullrate_multilabel_losses_by_depth(
                rd_logits, y_rd, rd_ctx["e_pad"], y_rd_c, pos_weight=pw_rd
            )

            i_dp = rng.integers(0, n_dp, size=batch_size)
            e_dp, l_dp, hs_dp, _, _ = stage2._sample_batch(
                bundle["dipco"]["train"], i_dp, False, augment=True
            )
            dp_states, dp_ctx = _run_states(model, e_dp, l_dp, hs_dp)
            t_z = dp_ctx["z_pad"].shape[1]
            y_dp = torch.from_numpy(stride_targets["dipco"][i_dp, :t_z])
            dp_losses = overlap_losses_by_depth(
                dp_states, dp_ctx["z_pad"], heads, y_dp, class_weight=cw_dp
            )

            i_am = rng.integers(0, n_am, size=batch_size)
            e_am, l_am, hs_am, _, lc_am = stage2._sample_batch(
                bundle["ami_sdm"]["train"], i_am, False, augment=True
            )
            am_states, am_ctx = _run_states(model, e_am, l_am, hs_am)
            y_am = torch.from_numpy(bundle["ami_sdm"]["train"]["speaker_idx"][i_am])
            am_losses = speaker_losses_by_depth(
                am_states, am_ctx["z_pad"], heads, y_am, layer_clip=lc_am
            )

            pen = (
                monotonic_penalty(st_losses)
                + monotonic_penalty(rd_losses)
                + 0.7 * monotonic_penalty(dp_losses)
                + 0.35 * monotonic_penalty(am_losses)
            )
            loss = (
                _task_plus_penalty(st_losses)
                + _task_plus_penalty(rd_losses)
                + 0.7 * _task_plus_penalty(dp_losses)
                + 0.35 * _task_plus_penalty(am_losses)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 2.0)
            opt.step()
            sched.step()
            loss_acc += float(loss.detach())
            pen_acc += float(pen.detach())

        do_eval = ep == 1 or ep % 2 == 0 or ep == epochs
        row: dict[str, Any] = {
            "epoch": ep,
            "train_loss": round(loss_acc / steps_per_epoch, 4),
            "monotonic_penalty": round(pen_acc / steps_per_epoch, 4),
            "lr": round(float(opt.param_groups[0]["lr"]), 6),
        }
        if do_eval:
            ev = _evaluate_probe(
                model, readouts, heads, bundle, compute_onset=False
            )
            m1 = ev["metrics_by_k"][1]
            m4 = ev["metrics_by_k"][4]
            row["val_composite_k1"] = m1["composite"]
            row["val_composite_k4"] = m4["composite"]
            row["k4_minus_k1_composite"] = round(m4["composite_raw"] - m1["composite_raw"], 4)
            print(
                f"[stage3] epoch {ep}/{epochs}: loss={row['train_loss']:.4f} "
                f"pen={row['monotonic_penalty']:.4f} "
                f"k1={m1['composite']:.4f} k4={m4['composite']:.4f} "
                f"delta={row['k4_minus_k1_composite']:+.4f}",
                flush=True,
            )
            if m4["composite_raw"] > best_comp:
                best_comp = float(m4["composite_raw"])
                best_epoch = ep
                best_state = (
                    copy.deepcopy(model.state_dict()),
                    copy.deepcopy(readouts.state_dict()),
                    copy.deepcopy(heads.state_dict()),
                )
        else:
            print(
                f"[stage3] epoch {ep}/{epochs}: loss={row['train_loss']:.4f} "
                f"pen={row['monotonic_penalty']:.4f}",
                flush=True,
            )
        history.append(row)
        model.train()
        readouts.train()
        heads.train()

    if best_state is None:
        raise RuntimeError("Stage 3 probe produced no evaluated checkpoint")
    model.load_state_dict(best_state[0])
    readouts.load_state_dict(best_state[1])
    heads.load_state_dict(best_state[2])
    final_eval = _evaluate_probe(model, readouts, heads, bundle, compute_onset=True)
    return {
        "model": model,
        "readouts": readouts,
        "heads": heads,
        "head_kwargs": head_kwargs,
        "n_params": n_params,
        "train_wall_s": round(time.perf_counter() - t0, 2),
        "best_epoch": best_epoch,
        "history": history,
        "metrics_by_k": final_eval["metrics_by_k"],
    }


def _baseline_from_stage2(stage2_payload: dict[str, Any], arm: str, ckpt: Path) -> dict[str, Any]:
    rec = stage2_payload["per_seed_runs"][f"{arm}_s42"]
    metrics = rec["metrics_by_k"]["1"]
    transfer = rec["transfer"]
    return {
        "arm": arm,
        "seed": 42,
        "checkpoint": str(ckpt),
        "checkpoint_sha256": _sha256_file(ckpt),
        "retrained": False,
        "source": "results/stage2_multilayer_halting.json per_seed_runs seed 42",
        "dev": {
            "composite": metrics["composite"],
            "starss23_macro_map": metrics["starss23_macro_map"],
            "realdesed_macro_map": metrics["realdesed_macro_map"],
            "dipco_macro_f1": metrics["dipco_macro_f1"],
            "ami_1_minus_eer": metrics["ami_1_minus_eer"],
            "onset_f1_100ms": metrics["timing_mean"]["onset_f1_100ms"],
            "onset_f1_60ms": metrics["timing_mean"]["onset_f1_60ms"],
            "onset_f1_200ms": metrics["timing_mean"]["onset_f1_200ms"],
        },
        "transfer_composite": transfer["transfer_composite_by_k"]["1"],
        "transfer": {
            "crema_d": transfer["crema_d"]["1"],
            "vocalset": transfer["vocalset"]["1"],
            "maestro_real": transfer["maestro_real"]["1"],
        },
    }


def _shuffled_near_chance(transfer: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "crema_d_emotion": {
            "real": transfer["crema_d"][4]["emotion"]["acc"],
            "shuffled": transfer["crema_d"][4]["emotion"]["shuffled_acc"],
            "chance": transfer["crema_d"][4]["emotion"]["chance_uniform"],
        },
        "vocalset": {
            "real": transfer["vocalset"][4]["acc"],
            "shuffled": transfer["vocalset"][4]["shuffled_acc"],
            "chance": transfer["vocalset"][4]["chance_uniform"],
        },
        "maestro_real": {
            "real": transfer["maestro_real"][4]["macro_map"],
            "shuffled": transfer["maestro_real"][4]["shuffled_macro_map"],
            "chance": transfer["maestro_real"][4].get("chance_uniform"),
        },
    }
    near = True
    for row in checks.values():
        shuffled = float(row["shuffled"])
        real = float(row["real"])
        if shuffled > 0.35 or shuffled > real - 0.15:
            near = False
    return {"near_chance": near, "k4": checks}


def _gate(metrics: dict[int, dict[str, Any]], transfer_comp: dict[int, float], shuffled_ok: bool) -> dict[str, Any]:
    dev_k1 = float(metrics[1]["composite_raw"])
    dev_k4 = float(metrics[4]["composite_raw"])
    on_k1 = float(metrics[1]["onset_f1_100ms_raw"])
    on_k4 = float(metrics[4]["onset_f1_100ms_raw"])
    tr_k1 = float(transfer_comp[1])
    tr_k4 = float(transfer_comp[4])
    dev_ok = dev_k4 > dev_k1
    onset_ok = on_k4 > on_k1
    transfer_ok = tr_k4 >= tr_k1
    return {
        "k4_beats_k1_dev_composite": dev_ok,
        "k4_beats_k1_onset_f1_100ms": onset_ok,
        "k4_does_not_lose_transfer_composite": transfer_ok,
        "shuffled_probes_near_chance": shuffled_ok,
        "passed": bool(dev_ok and onset_ok and transfer_ok and shuffled_ok),
        "dev_composite_k1": round(dev_k1, 4),
        "dev_composite_k4": round(dev_k4, 4),
        "dev_composite_k4_minus_k1": round(dev_k4 - dev_k1, 4),
        "onset_f1_100ms_k1": round(on_k1, 4),
        "onset_f1_100ms_k4": round(on_k4, 4),
        "onset_f1_100ms_k4_minus_k1": round(on_k4 - on_k1, 4),
        "transfer_composite_k1": tr_k1,
        "transfer_composite_k4": tr_k4,
        "transfer_composite_k4_minus_k1": round(tr_k4 - tr_k1, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3 cheap depth probe")
    parser.add_argument("--manifest-dir", type=Path, default=Path("data/manifests"))
    parser.add_argument("--tap-root", type=Path, default=Path("cache/stage2_taps"))
    parser.add_argument(
        "--tap-index", type=Path, default=Path("data/manifests/stage2_tap_cache_index.json")
    )
    parser.add_argument("--s1-root", type=Path, default=Path("cache/stage1_pilot"))
    parser.add_argument(
        "--s1-index", type=Path, default=Path("data/manifests/pilot_cache_index.json")
    )
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--steps-per-epoch", type=int, default=28)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--seed", type=int, default=ONLY_SEED)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--baselines",
        type=Path,
        default=Path("results/stage2_multilayer_halting.json"),
    )
    parser.add_argument(
        "--baseline-ckpt-dir", type=Path, default=Path("/tmp/stage2_ckpts")
    )
    parser.add_argument("--ckpt-out", type=Path, default=Path("checkpoints/stage3_depth_probe.pt"))
    parser.add_argument("--out", type=Path, default=Path("results/stage3_depth_probe.json"))
    parser.add_argument(
        "--store-copy",
        type=Path,
        default=Path(
            "/cursor/stores/bc-01a213ed-7a8a-7981-b509-7012340964c4/internal/stage3_depth_probe.json"
        ),
    )
    args = parser.parse_args()
    if int(args.epochs) > MAX_EPOCHS:
        raise SystemExit(f"Stage 3 probe refuses more than {MAX_EPOCHS} epochs")
    if int(args.seed) != ONLY_SEED:
        raise SystemExit(f"Stage 3 probe is seed {ONLY_SEED} only")
    if int(args.epochs) < 1:
        raise SystemExit("Stage 3 probe needs at least 1 epoch")

    torch.set_num_threads(int(args.threads))
    t_global = time.perf_counter()
    assert_zero_pretraining_overlap(PILOT_CORPUS_SOURCES)
    guard = SealedSplitGuard(set(SEALED_SPLITS))
    guard.check_access("realdesed_train")
    guard.check_access("realdesed_val")
    tap_index = json.loads(args.tap_index.read_text(encoding="utf-8"))
    for corpus, splits in tap_index["caches"].items():
        for split_name in splits:
            if "test" in str(split_name):
                raise RuntimeError(f"Refusing sealed or test split {corpus}/{split_name}")
    s1_index = json.loads(args.s1_index.read_text(encoding="utf-8"))
    stage2_payload = json.loads(args.baselines.read_text(encoding="utf-8"))
    baselines = {
        "A_ml": _baseline_from_stage2(
            stage2_payload, "A_ml", args.baseline_ckpt_dir / "A_ml_s42.pt"
        ),
        "B_comp_ml": _baseline_from_stage2(
            stage2_payload, "B_comp_ml", args.baseline_ckpt_dir / "B_comp_ml_s42.pt"
        ),
    }
    print("[stage3] Loading train/val tap bundle ...", flush=True)
    bundle = stage2._build_stage2_bundle(
        args.manifest_dir, args.tap_root, tap_index, args.s1_root, s1_index
    )
    trained = _train_probe(
        bundle,
        epochs=int(args.epochs),
        steps_per_epoch=int(args.steps_per_epoch),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        seed=int(args.seed),
    )
    ckpt_path = args.ckpt_out
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "arm": ARM_NAME,
            "base_arm": "E",
            "seed": int(args.seed),
            "refiner_kwargs": _arm_kwargs(),
            "head_kwargs": trained["head_kwargs"],
            "refiner": trained["model"].state_dict(),
            "readouts": trained["readouts"].state_dict(),
            "heads": trained["heads"].state_dict(),
            "labels": list(REALDESED_CLASSES),
            "tap_layers": list(TAP_LAYERS),
        },
        ckpt_path,
    )
    ckpt_record = {
        "arm": ARM_NAME,
        "base_arm": "E",
        "arm_kwargs": _arm_kwargs(),
        "head_kwargs": trained["head_kwargs"],
        "seed": int(args.seed),
        "ckpt_path": ckpt_path,
    }
    del bundle
    gc.collect()
    print("[stage3] Zero-overlap frozen transfer ...", flush=True)
    transfer_by, _halt = stage2._evaluate_all_transfer_low_ram(
        [ckpt_record],
        {},
        {},
        args.manifest_dir,
        args.tap_root,
        tap_index,
        args.s1_root,
        s1_index,
    )
    transfer = transfer_by[(ARM_NAME, int(args.seed))]
    shuffled = _shuffled_near_chance(transfer)
    gate = _gate(
        trained["metrics_by_k"],
        {int(k): float(v) for k, v in transfer["transfer_composite_by_k"].items()},
        bool(shuffled["near_chance"]),
    )
    payload = {
        "stage": "stage3_depth_probe",
        "question": (
            "Can a later refinement step beat a frozen first pass when the loss "
            "penalizes any step that is worse than stop-grad k=1, and a full-rate "
            "50 Hz residual path feeds the onset/event head?"
        ),
        "method": {
            "arm": ARM_NAME,
            "seed": int(args.seed),
            "epochs": int(args.epochs),
            "steps_per_epoch": int(args.steps_per_epoch),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "stride": 8,
            "depths": [1, 2, 4],
            "stopgrad_k1_target": True,
            "monotonic_penalty": "relu(loss_k - loss_k1.detach()) on event, starss, overlap, and speaker",
            "fullrate_residual": (
                "event and STARSS logits are a 50 Hz linear on stem features plus a "
                "depthwise 50 Hz residual conditioned by the pooled refined state"
            ),
            "selection": "best k=4 dev composite",
            "cloud_gpu_spend_usd": 0.0,
            "realdesed_test": "sealed",
        },
        "fixed_baselines": baselines,
        "probe": {
            "n_params": trained["n_params"],
            "train_wall_s": trained["train_wall_s"],
            "best_epoch": trained["best_epoch"],
            "history": trained["history"],
            "metrics_by_k": trained["metrics_by_k"],
            "transfer_composite_by_k": transfer["transfer_composite_by_k"],
            "transfer": {
                "crema_d": transfer["crema_d"],
                "vocalset": transfer["vocalset"],
                "maestro_real": transfer["maestro_real"],
            },
            "checkpoint": str(ckpt_path),
        },
        "shuffled_controls": shuffled,
        "gate": gate,
        "total_wall_s": round(time.perf_counter() - t_global, 2),
    }
    write_result_json(args.out, stage2._jsonable(payload))
    if args.store_copy is not None:
        args.store_copy.parent.mkdir(parents=True, exist_ok=True)
        args.store_copy.write_text(args.out.read_text(encoding="utf-8"), encoding="utf-8")
    write_run_manifest(
        "stage3_depth_probe",
        {
            "stage": "stage3_depth_probe",
            "seed": int(args.seed),
            "epochs": int(args.epochs),
            "steps_per_epoch": int(args.steps_per_epoch),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "cloud_gpu_spend_usd": 0.0,
        },
        ledger_dir="ledger",
    )
    print(
        f"[stage3] GATE passed={gate['passed']} "
        f"dev {gate['dev_composite_k4_minus_k1']:+.4f} "
        f"onset100 {gate['onset_f1_100ms_k4_minus_k1']:+.4f} "
        f"transfer {gate['transfer_composite_k4_minus_k1']:+.4f} "
        f"-> {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
