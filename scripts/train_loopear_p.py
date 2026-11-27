import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from loopear.d.diagnostics import (
    analytic_encoder_flops,
    cpu_seconds,
    embedding_diagnostics,
    flat_params,
    hours,
    param_delta_l2,
    peak_rss_gb,
)
from loopear.d.frontend import crop_logmel, normalize_logmel
from loopear.d.fsd import MelCache, assert_train_rows, items_for, load_manifest, sha256_of_ids
from loopear.d.objective import (
    TIME_NEIGHBORHOOD_PATCHES,
    broadcast_time_valid,
    describe_crop_offsets,
    global_ntxent,
    patch_valid_from_frames,
    sample_freq_mask,
    sample_time_mask,
)
from loopear.d.perceiver import (
    LoopEarPEncoder,
    analytic_perceiver_flops,
    attention_shift_sums,
    centered_variation_sums,
    content_and_anchor_sums,
    init_perceiver,
    init_perceiver_local,
    perceiver_local_loss,
    perceiver_ltc_loss,
    visible_cross_logit_stats,
    within_recording_time_sums,
)
from loopear.d.spec import (
    CROP_SECONDS,
    ENCODER_PARAMS,
    GRAD_CLIP,
    LR,
    N_LATENTS,
    PERCEIVER_ENCODER_PARAMS,
    SEED,
    TAU_GLOBAL,
    TAU_LOCAL,
    WEIGHT_DECAY,
    ema_at,
    lr_at,
)
from loopear.harness import SpendLedger, write_result_json, write_run_manifest


class Order:
    def __init__(self, n_items: int, batch: int, seed: int) -> None:
        if n_items < batch:
            raise ValueError(f"dataset {n_items} is smaller than batch {batch}")
        self.n_items = n_items
        self.batch = batch
        self.order_gen = torch.Generator().manual_seed(seed)
        self.crop_gen = torch.Generator().manual_seed(seed + 17)
        self.mask_gen = torch.Generator().manual_seed(seed + 91)
        self.order: list[int] = []
        self.cursor = 0

    def indices(self) -> list[int]:
        if self.cursor + self.batch > len(self.order):
            self.order = torch.randperm(self.n_items, generator=self.order_gen).tolist()
            self.cursor = 0
        chosen = self.order[self.cursor : self.cursor + self.batch]
        self.cursor += self.batch
        return chosen


def _load_d_trainer():
    import importlib.util

    path = Path(__file__).resolve().parent / "train_loopear_d.py"
    spec = importlib.util.spec_from_file_location("train_loopear_d_script", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the LoopEar-D trainer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _visible(frame_valid: torch.Tensor) -> torch.Tensor:
    return broadcast_time_valid(patch_valid_from_frames(frame_valid))


def global_forward(
    model: torch.nn.Module,
    mel1: torch.Tensor,
    mel2: torch.Tensor,
    frame_valid1: torch.Tensor,
    frame_valid2: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    valid1 = _visible(frame_valid1)
    valid2 = _visible(frame_valid2)
    if bool((valid1.flatten(1).sum(dim=1) == 0).any()) or bool(
        (valid2.flatten(1).sum(dim=1) == 0).any()
    ):
        raise ValueError("a crop has no valid patches")
    time_mask = sample_time_mask(valid1.any(dim=-1), generator)
    time_mask = time_mask[:, :, None].expand_as(valid1) & valid1
    freq_mask = sample_freq_mask(mel2.shape[0], generator)
    freq_mask = freq_mask[:, None, :].expand_as(valid2) & valid2
    _lat1, summary1 = model.encoder(mel1, ~time_mask & valid1)
    _lat2, summary2 = model.encoder(mel2, ~freq_mask & valid2)
    return global_ntxent(
        model.project_global(summary1),
        model.project_global(summary2),
        TAU_GLOBAL,
    )


def _set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def _encode_cache(model: torch.nn.Module, cache: MelCache, n: int) -> dict[str, Any]:
    summaries = []
    latent_rank = []
    latent_cosine = []
    model.eval()
    with torch.no_grad():
        for index in range(min(n, len(cache))):
            raw = cache._load(index)
            crop, valid = crop_logmel(raw, None)
            norm = normalize_logmel(crop, cache.mean, cache.std) * valid.unsqueeze(-1)
            latents, summary = model.encoder(norm.unsqueeze(0), _visible(valid.unsqueeze(0)))
            summaries.append(summary[0].detach().float().cpu())
            stats = embedding_diagnostics(latents[0].detach().float().cpu())
            latent_rank.append(float(stats["effective_rank"]))
            latent_cosine.append(float(stats["offdiag_cosine_mean"]))
    summary = torch.stack(summaries, dim=0)
    return {
        "summary": embedding_diagnostics(summary),
        "within_crop_latent_effective_rank_mean": float(sum(latent_rank) / len(latent_rank)),
        "within_crop_latent_offdiag_cosine_mean": float(sum(latent_cosine) / len(latent_cosine)),
        "n_crops": int(summary.shape[0]),
    }


def _collapsed(stats: dict[str, Any]) -> bool:
    summary = stats["summary"]
    return bool(summary["collapsed"] or not summary["finite"])


ALPHA_FACTORS = (0.25, 0.5, 1.0, 2.0)
ALPHA_SWEEP_BATCHES = 4
ALPHA_SWEEP_STEPS = 8
ALPHA_RULE = (
    "On four training batches, measure the RMS of the first cross-attention logits "
    "and the RMS of (t_j - mu_i)^2. Candidates set alpha so that RMS of "
    "alpha*(t_j - mu_i)^2 is 0.25, 0.5, 1, or 2 times the logit RMS. "
    "Each candidate trains eight AdamW steps from seed 42 on the 20-hour cache. "
    "Lock the eligible candidate with the lowest mean global NT-Xent loss. "
    "Alpha 0 is recorded and is not eligible. Validation audio is not used. "
    "The locked value is not changed after the probe."
)


def select_time_alpha(cache: MelCache, seed: int, batch: int) -> dict[str, Any]:
    scale_model = init_perceiver(seed)
    scale_model.eval()
    scale_order = Order(len(cache), batch, seed)
    logit_sq = 0.0
    penalty_sq = 0.0
    count = 0.0
    with torch.no_grad():
        for _ in range(ALPHA_SWEEP_BATCHES):
            crops = cache.crops(scale_order.indices(), scale_order.crop_gen)
            stats = visible_cross_logit_stats(
                scale_model.encoder, crops["mel1"], _visible(crops["frame_valid1"])
            )
            logit_sq += float(stats["logit_sq"])
            penalty_sq += float(stats["penalty_sq"])
            count += float(stats["count"])
    if count <= 0.0 or penalty_sq <= 0.0 or logit_sq <= 0.0:
        raise SystemExit("time-alpha scale measurement was empty")
    logit_rms = (logit_sq / count) ** 0.5
    penalty_rms = (penalty_sq / count) ** 0.5
    unit = logit_rms / penalty_rms
    candidates = []
    for factor in (0.0,) + ALPHA_FACTORS:
        alpha = 0.0 if factor == 0.0 else float(factor) * unit
        model = init_perceiver(seed)
        model.encoder.time_alpha = float(alpha)
        opt = torch.optim.AdamW(model.student_parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        order = Order(len(cache), batch, seed)
        losses = []
        for step in range(ALPHA_SWEEP_STEPS):
            crops = cache.crops(order.indices(), order.crop_gen)
            model.train()
            _set_lr(opt, lr_at(step, ALPHA_SWEEP_STEPS))
            loss = global_forward(
                model,
                crops["mel1"],
                crops["mel2"],
                crops["frame_valid1"],
                crops["frame_valid2"],
                order.mask_gen,
            )
            if not torch.isfinite(loss):
                losses.append(float("nan"))
                break
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.student_parameters(), GRAD_CLIP)
            opt.step()
            losses.append(float(loss.detach()))
        finite = [value for value in losses if value == value]
        mean_loss = None if not finite or len(finite) != ALPHA_SWEEP_STEPS else sum(finite) / len(finite)
        candidates.append(
            {
                "factor": float(factor),
                "alpha": float(alpha),
                "losses": losses,
                "mean_loss": mean_loss,
                "eligible": factor != 0.0 and mean_loss is not None,
            }
        )
    eligible = [row for row in candidates if row["eligible"]]
    if not eligible:
        raise SystemExit("time-alpha sweep produced no finite candidate")
    chosen = min(eligible, key=lambda row: float(row["mean_loss"]))
    return {
        "rule": ALPHA_RULE,
        "retuned_after_eval": False,
        "batches": ALPHA_SWEEP_BATCHES,
        "steps": ALPHA_SWEEP_STEPS,
        "batch_recordings": int(batch),
        "seed": int(seed),
        "logit_rms": logit_rms,
        "penalty_rms": penalty_rms,
        "alpha_unit": unit,
        "time_units": "patch index, 0 on the first time patch and GRID_T-1 on the last",
        "anchors": "linspace across the 50 time patches, one anchor per latent",
        "candidates": candidates,
        "alpha": float(chosen["alpha"]),
        "factor": float(chosen["factor"]),
    }


def write_plan(measured_sec_per_step: float | None, report_path: Path) -> dict[str, Any]:
    data = json.loads(Path("results/loopear_d_data.json").read_text(encoding="utf-8"))
    h20 = data["subsets"]["h20"]
    h5 = data["subsets"]["h5"]
    estimate = None
    if measured_sec_per_step is not None:
        estimate = {
            "sec_per_step_from_h5_batch64": measured_sec_per_step,
            "projected_wall_hours_5000_steps": hours(measured_sec_per_step * 5000.0),
            "note": "The 20-hour mel cache is not built. This projection ignores cache creation.",
        }
    plan = {
        "task": "loopear_matched_h20_plan",
        "launched": False,
        "gpu_usd": 0.0,
        "dataset_hours": h20["hours"],
        "dataset_clips": h20["clips"],
        "dataset_id_sha256": data["subset_id_sha256"]["h20"],
        "steps": 5000,
        "batch_recordings": 64,
        "views_per_recording": 2,
        "negatives_per_anchor": 126,
        "seed": SEED,
        "objective": "global",
        "lambda_local": 0.0,
        "local_objective_in_code": True,
        "hours_are_not_steps": (
            f"The corpus is {h20['hours']:.3f} hours ({h20['clips']} clips). "
            "5000 steps at batch 64 are 320000 crop presentations, not 20 hours of optimization."
        ),
        "reference_h5": {
            "hours": h5["hours"],
            "clips": h5["clips"],
            "id_sha256": data["subset_id_sha256"]["h5"],
        },
        "latency_projection": estimate,
        "arms": [
            {
                "model": "loopear-d",
                "objective": "global",
                "lambda_local": 0.0,
                "command": (
                    ".venv/bin/python scripts/train_loopear_p.py --launch-prepared loopear-d"
                ),
                "encoder_out": "results/loopear_d_h20_global_5000_encoder.safetensors",
                "overwrites_existing_checkpoints": False,
            },
            {
                "model": "loopear-p",
                "objective": "global",
                "lambda_local": 0.0,
                "command": (
                    ".venv/bin/python scripts/train_loopear_p.py --launch-prepared loopear-p"
                ),
                "encoder_out": "results/loopear_p_h20_global_5000_encoder.safetensors",
                "overwrites_existing_checkpoints": False,
            },
        ],
    }
    write_result_json(report_path, plan)
    return plan


def _snapshot_encoder(model, ckpt_path: Path, step: int) -> str:
    path = Path("results") / f"{ckpt_path.stem}_step{int(step)}_encoder.safetensors"
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(model.encoder.state_dict(), str(path))
    return str(path)


def run_perceiver(
    cache: MelCache,
    steps: int,
    batch: int,
    seed: int,
    log_path: Path,
    ckpt_path: Path,
    weight_path: Path,
    run_id: str,
    config: dict[str, Any],
    snapshot_every: int | None = None,
    time_alpha: float = 0.0,
) -> dict[str, Any]:
    if torch.cuda.is_available():
        raise SystemExit("cuda is visible; refusing to start rather than risk GPU spend")
    probe_model = init_perceiver(seed)
    probe_model.encoder.time_alpha = float(time_alpha)
    counted = sum(p.numel() for p in probe_model.encoder.parameters())
    if counted != PERCEIVER_ENCODER_PARAMS:
        raise SystemExit(f"encoder params {counted} != {PERCEIVER_ENCODER_PARAMS}")
    probe_opt = torch.optim.AdamW(probe_model.student_parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    probe_order = Order(len(cache), batch, seed + 1)
    write_run_manifest(run_id, config)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    probe = cache.crops(probe_order.indices(), probe_order.crop_gen)
    probe_model.train()
    t_probe = time.perf_counter()
    loss_probe = global_forward(
        probe_model,
        probe["mel1"],
        probe["mel2"],
        probe["frame_valid1"],
        probe["frame_valid2"],
        probe_order.mask_gen,
    )
    probe_opt.zero_grad(set_to_none=True)
    loss_probe.backward()
    torch.nn.utils.clip_grad_norm_(probe_model.student_parameters(), GRAD_CLIP)
    probe_opt.step()
    profile = {
        "sec_per_step": time.perf_counter() - t_probe,
        "peak_rss_gb": peak_rss_gb(),
        "loss": float(loss_probe.detach()),
        "batch": batch,
    }
    print(json.dumps({"profile": profile}), flush=True)
    del probe_model, probe_opt, probe
    model = init_perceiver(seed)
    model.encoder.time_alpha = float(time_alpha)
    opt = torch.optim.AdamW(model.student_parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    order = Order(len(cache), batch, seed)
    init_vec = flat_params(model.encoder).clone()
    t0 = time.perf_counter()
    cpu0 = cpu_seconds()
    rows = []
    snapshots: list[str] = []
    stopped_collapse = False
    with log_path.open("w", encoding="utf-8") as handle:
        for step in range(int(steps)):
            batch_tensors = cache.crops(order.indices(), order.crop_gen)
            model.train()
            lr = lr_at(step, steps)
            _set_lr(opt, lr)
            loss = global_forward(
                model,
                batch_tensors["mel1"],
                batch_tensors["mel2"],
                batch_tensors["frame_valid1"],
                batch_tensors["frame_valid2"],
                order.mask_gen,
            )
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite loss")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.student_parameters(), GRAD_CLIP)
            opt.step()
            stats = {
                "step": step,
                "loss": float(loss.detach()),
                "loss_g": float(loss.detach()),
                "loss_l": 0.0,
                "lambda_local": 0.0,
                "lr": float(lr),
                "grad_norm": float(grad_norm),
            }
            if (step + 1) % 100 == 0:
                snap = _encode_cache(model, cache, min(32, len(cache)))
                stats["diagnostics"] = snap
                if _collapsed(snap):
                    stopped_collapse = True
            rows.append(stats)
            handle.write(json.dumps(stats, sort_keys=True) + "\n")
            finished = step + 1
            if snapshot_every and finished % int(snapshot_every) == 0 and finished < int(steps):
                snapshots.append(_snapshot_encoder(model, ckpt_path, finished))
            if stopped_collapse:
                print(f"perceiver collapsed at step {step}", flush=True)
                break
            if step % 10 == 0 or step + 1 == steps:
                handle.flush()
                print(f"perceiver step {step} loss {stats['loss']:.4f}", flush=True)
    wall = time.perf_counter() - t0
    cpu = cpu_seconds() - cpu0
    diag = _encode_cache(model, cache, min(32, len(cache)))
    moved = param_delta_l2(flat_params(model.encoder), init_vec)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": len(rows),
            "objective": "global",
            "lambda_local": 0.0,
            "encoder": model.encoder.state_dict(),
            "global_head": model.global_head.state_dict(),
            "config": config,
        },
        ckpt_path,
    )
    if weight_path.exists():
        raise FileExistsError(weight_path)
    weight_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(model.encoder.state_dict(), str(weight_path))
    losses = [row["loss"] for row in rows]
    return {
        "objective": "global",
        "lambda_local": 0.0,
        "time_alpha": float(time_alpha),
        "steps": len(rows),
        "steps_requested": int(steps),
        "stopped_collapse": stopped_collapse,
        "batch_recordings": int(batch),
        "views_per_recording": 2,
        "negatives_per_anchor": int(2 * (batch - 1)),
        "denominator_terms_per_anchor": int(1 + 2 * (batch - 1)),
        "crop_presentations": int(len(rows) * batch),
        "presented_crop_hours": hours(len(rows) * batch * CROP_SECONDS),
        "loss_start": losses[0],
        "loss_end": losses[-1],
        "loss_min": min(losses),
        "weights_l2_change": moved,
        "weights_moved": moved > 0.0,
        "diagnostics": diag,
        "collapsed": _collapsed(diag) or stopped_collapse,
        "wall_s": wall,
        "cpu_s": cpu,
        "sec_per_step": wall / max(len(rows), 1),
        "cpu_hours": hours(cpu),
        "wall_hours": hours(wall),
        "profile_step": profile,
        "peak_rss_gb": peak_rss_gb(),
        "encoder_path": str(weight_path),
        "snapshots": snapshots,
        "flops": analytic_perceiver_flops(1, 200),
        "flops_batch": analytic_perceiver_flops(batch, 200),
        "loopear_d_flops": analytic_encoder_flops(1, 201),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _preserved_weight_paths() -> list[Path]:
    names = (
        "loopear_d_h20_global_5000*.safetensors",
        "loopear_p_h20_global_5000*.safetensors",
        "loopear_pt_h20_global_5000*.safetensors",
        "loopear_pl_h20_global_5000*.safetensors",
        "loopear_plt_h20_global_5000*.safetensors",
        "loopear_d_cmp_global_encoder.safetensors",
        "loopear_d_k1_encoder.safetensors",
        "loopear_p_encoder.safetensors",
    )
    found: list[Path] = []
    for name in names:
        found.extend(sorted(Path("results").glob(name)))
    if len(found) < 8:
        raise SystemExit("preserved D, P, and P-T encoder files are missing")
    return found


def _rms_row(total: dict[str, float]) -> dict[str, float]:
    count = float(total["count"])
    if count <= 0.0:
        raise SystemExit("anchor measurement saw no visible keys")
    logit_rms = (total["logit_sq"] / count) ** 0.5
    penalty_rms = (total["penalty_sq"] / count) ** 0.5
    if logit_rms <= 0.0:
        raise SystemExit("content logits were empty")
    return {
        "count": count,
        "content_logit_rms": logit_rms,
        "content_logit_mean_abs": total["logit_abs"] / count,
        "anchor_penalty_rms": penalty_rms,
        "anchor_penalty_mean_abs": total["penalty_abs"] / count,
        "penalty_rms_over_logit_rms": penalty_rms / logit_rms,
        "fraction_abs_penalty_exceeds_abs_logit": total["exceed"] / count,
    }


def measure_locked_penalty(cache: MelCache, seed: int, batches: int = 8) -> dict[str, Any]:
    trained = json.loads(Path("results/loopear_pt_h20_train.json").read_text(encoding="utf-8"))
    alpha = float(trained["time_alpha_sweep"]["alpha"])
    weight_path = Path("results/loopear_pt_h20_global_5000_encoder.safetensors")
    encoder = LoopEarPEncoder()
    encoder.load_state_dict(load_file(str(weight_path)))
    encoder.time_alpha = alpha
    encoder.eval()
    counted = sum(p.numel() for p in encoder.parameters())
    if counted != PERCEIVER_ENCODER_PARAMS:
        raise SystemExit(f"P-T encoder params {counted} != {PERCEIVER_ENCODER_PARAMS}")
    order = Order(len(cache), 64, seed)
    keys = ("logit_sq", "logit_abs", "penalty_sq", "penalty_abs", "exceed", "count")
    blank = lambda: [{key: 0.0 for key in keys} for _ in range(3)]
    totals = {"time_masked": blank(), "freq_masked": blank()}
    with torch.no_grad():
        for _ in range(int(batches)):
            crops = cache.crops(order.indices(), order.crop_gen)
            valid1 = _visible(crops["frame_valid1"])
            valid2 = _visible(crops["frame_valid2"])
            time_mask = sample_time_mask(valid1.any(dim=-1), order.mask_gen)
            time_mask = time_mask[:, :, None].expand_as(valid1) & valid1
            freq_mask = sample_freq_mask(crops["mel2"].shape[0], order.mask_gen)
            freq_mask = freq_mask[:, None, :].expand_as(valid2) & valid2
            paired = (
                ("time_masked", crops["mel1"], ~time_mask & valid1),
                ("freq_masked", crops["mel2"], ~freq_mask & valid2),
            )
            for name, mel, visible in paired:
                rows = content_and_anchor_sums(encoder, mel, visible, alpha)
                if len(rows) != 3:
                    raise SystemExit("expected three cross-attention blocks")
                for index, row in enumerate(rows):
                    for key in keys:
                        totals[name][index][key] += float(row[key])
    views = {}
    for name, layers in totals.items():
        pooled = {key: 0.0 for key in keys}
        for layer in layers:
            for key in keys:
                pooled[key] += layer[key]
        views[name] = {"layers": [_rms_row(layer) for layer in layers], "pooled": _rms_row(pooled)}
    both_layers = []
    for index in range(3):
        merged = {key: 0.0 for key in keys}
        for name in ("time_masked", "freq_masked"):
            for key in keys:
                merged[key] += totals[name][index][key]
        both_layers.append(merged)
    both = {key: 0.0 for key in keys}
    for layer in both_layers:
        for key in keys:
            both[key] += layer[key]
    return {
        "task": "loopear_pt_penalty_versus_logits",
        "alpha": alpha,
        "alpha_changed": False,
        "retrained": False,
        "weights": str(weight_path),
        "batches": int(batches),
        "batch_recordings": 64,
        "seed": int(seed),
        "views_per_batch": 2,
        "cross_blocks": 3,
        "comparison": (
            "content logits are qk/sqrt(d_head) before the additive penalty, "
            "on the visible keys of each real training view, at the latents that reach that block"
        ),
        "views": views,
        "layers": [_rms_row(layer) for layer in both_layers],
        "pooled": _rms_row(both),
        "gpu_usd": 0.0,
    }


def choose_local_lambda(cache: MelCache, seed: int, batch: int) -> dict[str, Any]:
    attempts = []
    for lam in (1.0, 0.1, 0.01, 0.001):
        model = init_perceiver_local(seed)
        model.train()
        order = Order(len(cache), batch, seed)
        crops = cache.crops(order.indices(), order.crop_gen)
        loss, loss_g, loss_l = perceiver_local_loss(
            model,
            crops["mel1"],
            crops["mel2"],
            crops["frame_valid1"],
            crops["frame_valid2"],
            order.mask_gen,
            lam,
        )
        loss_finite = bool(torch.isfinite(loss).all())
        grad_finite = False
        grad_norm = None
        if loss_finite:
            loss.backward()
            square = 0.0
            saw = False
            grad_finite = True
            for param in model.student_parameters():
                if param.grad is None:
                    continue
                saw = True
                if not bool(torch.isfinite(param.grad).all()):
                    grad_finite = False
                    break
                square += float(param.grad.detach().float().square().sum())
            grad_finite = bool(grad_finite and saw)
            if grad_finite:
                grad_norm = square ** 0.5
        attempts.append(
            {
                "lambda_local": float(lam),
                "loss": None if not loss_finite else float(loss.detach()),
                "loss_g": None if not bool(torch.isfinite(loss_g).all()) else float(loss_g.detach()),
                "loss_l": None if not bool(torch.isfinite(loss_l).all()) else float(loss_l.detach()),
                "loss_finite": loss_finite,
                "grad_finite": grad_finite,
                "grad_norm": grad_norm,
            }
        )
        del model
        if loss_finite and grad_finite:
            return {
                "requested_lambda": 1.0,
                "chosen_lambda": float(lam),
                "scaled": float(lam) != 1.0,
                "k": 1,
                "objective": "global_ntxent + lambda * local_mld",
                "rule": (
                    "Use lambda 1 unless this one training batch has a non-finite loss or gradient. "
                    "Then keep the largest of 0.1, 0.01, and 0.001 that is finite. "
                    "Validation audio is not used."
                ),
                "attempts": attempts,
            }
    raise SystemExit("no finite local lambda on the one-batch check")


_RANK_SUM_KEYS = (
    "n_queries_with_other_moment",
    "n_queries_without_other_moment",
    "n_correct_strictly_best",
    "n_tied_for_best",
    "n_worse_than_another_moment",
    "rank_sum",
    "correct_cosine_sum",
    "other_time_cosine_sum",
    "other_time_pair_count",
    "n_teacher_pairs",
    "teacher_cosine_sum",
    "n_teacher_pairs_below_0_99",
    "n_teacher_pairs_below_0_50",
    "n_other_recording_pairs",
    "other_recording_cosine_sum",
    "n_queries_with_other_recording",
    "n_correct_beats_other_recording_max",
)


def _blank_rank() -> dict[str, float]:
    return {key: 0.0 for key in _RANK_SUM_KEYS}


def _add_rank(total: dict[str, float], row: dict[str, float]) -> None:
    for key in _RANK_SUM_KEYS:
        total[key] += float(row[key])


def _rank_means(total: dict[str, float]) -> dict[str, Any]:
    def divide(numerator: float, denominator: float) -> float | None:
        if denominator <= 0.0:
            return None
        return numerator / denominator

    queries = total["n_queries_with_other_moment"]
    return {
        **total,
        "fraction_correct_strictly_best": divide(total["n_correct_strictly_best"], queries),
        "fraction_tied_for_best": divide(total["n_tied_for_best"], queries),
        "mean_rank": divide(total["rank_sum"], queries),
        "mean_correct_cosine": divide(total["correct_cosine_sum"], queries),
        "mean_other_time_cosine": divide(total["other_time_cosine_sum"], total["other_time_pair_count"]),
        "mean_teacher_offdiag_cosine": divide(total["teacher_cosine_sum"], total["n_teacher_pairs"]),
        "fraction_teacher_pairs_below_0_99": divide(
            total["n_teacher_pairs_below_0_99"], total["n_teacher_pairs"]
        ),
        "fraction_teacher_pairs_below_0_50": divide(
            total["n_teacher_pairs_below_0_50"], total["n_teacher_pairs"]
        ),
        "mean_other_recording_cosine": divide(
            total["other_recording_cosine_sum"], total["n_other_recording_pairs"]
        ),
        "fraction_correct_beats_other_recording_max": divide(
            total["n_correct_beats_other_recording_max"], total["n_queries_with_other_recording"]
        ),
    }


def _view_pack(crops: dict[str, torch.Tensor], generator: torch.Generator) -> tuple[dict[str, Any], ...]:
    valid1 = _visible(crops["frame_valid1"])
    valid2 = _visible(crops["frame_valid2"])
    time_mask = sample_time_mask(valid1.any(dim=-1), generator)
    time_mask = time_mask[:, :, None].expand_as(valid1) & valid1
    freq_mask = sample_freq_mask(crops["mel2"].shape[0], generator)
    freq_mask = freq_mask[:, None, :].expand_as(valid2) & valid2
    return (
        {"name": "time_masked", "mel": crops["mel1"], "valid": valid1, "mask": time_mask, "view": 0},
        {"name": "freq_masked", "mel": crops["mel2"], "valid": valid2, "mask": freq_mask, "view": 1},
    )


def measure_pl_time_rank(cache: MelCache, seed: int, batches: int = 16) -> dict[str, Any]:
    weight_path = Path("checkpoints/loopear_pl_h20_global_5000.pt")
    blob = torch.load(weight_path, map_location="cpu", weights_only=False)
    if any(key.startswith("teacher_") for key in blob):
        raise SystemExit("P-L checkpoint contains teacher tensors this measurement did not expect")
    model = init_perceiver_local(0)
    model.encoder.load_state_dict(blob["encoder"])
    model.global_head.load_state_dict(blob["global_head"])
    model.predictor.load_state_dict(blob["predictor"])
    model.local_head.load_state_dict(blob["local_head"])
    model.set_time_alpha(0.0)
    model.reset_teacher()
    model.eval()
    counted = sum(p.numel() for p in model.encoder.parameters())
    if counted != PERCEIVER_ENCODER_PARAMS:
        raise SystemExit(f"P-L encoder params {counted} != {PERCEIVER_ENCODER_PARAMS}")
    order = Order(len(cache), 64, seed)
    pooled = {
        "all_valid": _blank_rank(),
        "masked_queries": _blank_rank(),
    }
    views: dict[str, dict[str, Any]] = {}
    with torch.no_grad():
        for _ in range(int(batches)):
            crops = cache.crops(order.indices(), order.crop_gen)
            for view in _view_pack(crops, order.mask_gen):
                latents, _summary = model.encoder(view["mel"], ~view["mask"] & view["valid"])
                student = model.project_local(model.predictor(latents, int(view["view"])))
                teacher_latents, _teacher_summary = model.teacher_encoder(view["mel"], view["valid"])
                target = model.project_teacher(
                    model.teacher_predictor(teacher_latents, int(view["view"]))
                )
                slot = views.setdefault(
                    str(view["name"]),
                    {"all_valid": _blank_rank(), "masked_queries": _blank_rank()},
                )
                full = within_recording_time_sums(student, target, view["valid"])
                masked = within_recording_time_sums(student, target, view["valid"], view["mask"])
                _add_rank(slot["all_valid"], full)
                _add_rank(slot["masked_queries"], masked)
                _add_rank(pooled["all_valid"], full)
                _add_rank(pooled["masked_queries"], masked)
    reported_views = {}
    for name, slot in views.items():
        reported_views[name] = {
            "all_valid": _rank_means(slot["all_valid"]),
            "masked_queries": _rank_means(slot["masked_queries"]),
        }
    return {
        "task": "loopear_pl_within_recording_time_rank",
        "teacher_weights": (
            "stop-grad copy of the saved student encoder, predictor, and local head; "
            "the EMA teacher tensors were not in the P-L checkpoint"
        ),
        "ema_tensors_in_checkpoint": False,
        "student_checkpoint": str(weight_path),
        "batches": int(batches),
        "batch_recordings": 64,
        "seed": int(seed),
        "same_recording_comparison": "same frequency, other time indices, cosine of L2-normalized local vectors",
        "other_recording_role": "comparison only, not the claim",
        "views": reported_views,
        "pooled": {
            "all_valid": _rank_means(pooled["all_valid"]),
            "masked_queries": _rank_means(pooled["masked_queries"]),
        },
        "gpu_usd": 0.0,
    }


SOFTMAX_ALPHA_RULE = (
    "Alpha is the centered content-logit RMS of the trained P-L encoder divided by "
    "the centered (t_j - mu_i)^2 RMS. Both are measured on eight training batches, "
    "both masked views, and all three cross-attention blocks. Per-query means are "
    "removed first because softmax ignores a constant on a query. The seed-42 "
    "initialization's own centered content RMS is recorded and is not the scale: "
    "those logits sit far below one, so matching them leaves the softmax nearly "
    "unchanged. The locked value is then applied to that initialization, and the "
    "run starts only if the attention distribution moves. There is no eight-step "
    "loss sweep, and the value is not changed after the check."
)


def _centered_rms(encoder: LoopEarPEncoder, cache: MelCache, seed: int, batches: int) -> dict[str, float]:
    order = Order(len(cache), 64, seed)
    content_sq = 0.0
    penalty_sq = 0.0
    count = 0.0
    with torch.no_grad():
        for _ in range(int(batches)):
            crops = cache.crops(order.indices(), order.crop_gen)
            for view in _view_pack(crops, order.mask_gen):
                stats = centered_variation_sums(
                    encoder, view["mel"], ~view["mask"] & view["valid"]
                )
                content_sq += float(stats["content_centered_sq"])
                penalty_sq += float(stats["penalty_centered_sq"])
                count += float(stats["count"])
    if count <= 0.0 or penalty_sq <= 0.0 or content_sq <= 0.0:
        raise SystemExit("centered logit variation was empty")
    return {
        "content_centered_rms": (content_sq / count) ** 0.5,
        "penalty_centered_rms": (penalty_sq / count) ** 0.5,
        "count": count,
    }


def _attention_totals(
    encoder: LoopEarPEncoder,
    cache: MelCache,
    seed: int,
    batches: int,
    alpha: float,
) -> dict[str, float]:
    order = Order(len(cache), 64, seed)
    totals = {
        "tv_sum": 0.0,
        "argmax_changed": 0.0,
        "n_queries": 0.0,
        "constant_offset_tv_sum": 0.0,
        "removed_mean_tv_sum": 0.0,
    }
    with torch.no_grad():
        for _ in range(int(batches)):
            crops = cache.crops(order.indices(), order.crop_gen)
            for view in _view_pack(crops, order.mask_gen):
                shift = attention_shift_sums(
                    encoder, view["mel"], ~view["mask"] & view["valid"], alpha
                )
                for key in totals:
                    totals[key] += float(shift[key])
    if totals["n_queries"] <= 0.0:
        raise SystemExit("attention check saw no queries")
    return totals


def lock_softmax_alpha(cache: MelCache, seed: int, batches: int = 8) -> dict[str, Any]:
    trained = LoopEarPEncoder()
    trained.load_state_dict(load_file("results/loopear_pl_h20_global_5000_encoder.safetensors"))
    trained.time_alpha = 0.0
    trained.eval()
    trained_scale = _centered_rms(trained, cache, seed, batches)
    alpha = trained_scale["content_centered_rms"] / trained_scale["penalty_centered_rms"]
    trained_shift = _attention_totals(trained, cache, seed, batches, alpha)
    trained_queries = trained_shift["n_queries"]
    trained_tv = trained_shift["tv_sum"] / trained_queries
    trained_offset = trained_shift["constant_offset_tv_sum"] / trained_queries
    trained_removed = trained_shift["removed_mean_tv_sum"] / trained_queries
    if trained_offset > 1e-4:
        raise SystemExit("a constant logit offset changed the attention")
    if trained_removed > 1e-4:
        raise SystemExit("removing the per-query penalty mean changed the attention")
    if trained_tv < 0.01:
        raise SystemExit(f"alpha {alpha} does not move trained P-L attention; mean total variation {trained_tv}")
    model = init_perceiver_local(seed)
    model.set_time_alpha(0.0)
    model.eval()
    init_scale = _centered_rms(model.encoder, cache, seed, batches)
    init_matched = init_scale["content_centered_rms"] / init_scale["penalty_centered_rms"]
    init_matched_shift = _attention_totals(model.encoder, cache, seed, batches, init_matched)
    init_shift = _attention_totals(model.encoder, cache, seed, batches, alpha)
    init_queries = init_shift["n_queries"]
    init_tv = init_shift["tv_sum"] / init_queries
    if init_tv < 0.01:
        raise SystemExit(f"alpha {alpha} does not move initialization attention; mean total variation {init_tv}")
    order = Order(len(cache), 64, seed)
    crops = cache.crops(order.indices(), order.crop_gen)
    view = _view_pack(crops, order.mask_gen)[0]
    visible = ~view["mask"] & view["valid"]
    with torch.no_grad():
        model.encoder.time_alpha = 0.0
        plain, _summary = model.encoder(view["mel"], visible)
        model.encoder.time_alpha = float(alpha)
        shifted, _summary = model.encoder(view["mel"], visible)
        moved = not torch.allclose(plain, shifted)
    if not moved:
        raise SystemExit("the encoder forward did not change when the locked alpha was applied")
    return {
        "task": "loopear_plt_softmax_alpha",
        "rule": SOFTMAX_ALPHA_RULE,
        "loss_sweep": False,
        "retuned_after_check": False,
        "alpha_source": "trained_pl_centered_content_rms",
        "batches": int(batches),
        "batch_recordings": 64,
        "seed": int(seed),
        "content_centered_rms": trained_scale["content_centered_rms"],
        "penalty_centered_rms": trained_scale["penalty_centered_rms"],
        "alpha": float(alpha),
        "mean_total_variation": trained_tv,
        "fraction_argmax_changed": trained_shift["argmax_changed"] / trained_queries,
        "n_attention_queries": trained_queries,
        "constant_offset_mean_total_variation": trained_offset,
        "removed_penalty_mean_total_variation": trained_removed,
        "init_content_centered_rms": init_scale["content_centered_rms"],
        "init_penalty_centered_rms": init_scale["penalty_centered_rms"],
        "init_matched_alpha": float(init_matched),
        "init_matched_mean_total_variation": init_matched_shift["tv_sum"] / init_matched_shift["n_queries"],
        "init_mean_total_variation": init_tv,
        "init_fraction_argmax_changed": init_shift["argmax_changed"] / init_queries,
        "init_n_attention_queries": init_queries,
        "encoder_output_changed": True,
        "gpu_usd": 0.0,
    }


def confirm_local_lambda(cache: MelCache, seed: int, batch: int, time_alpha: float) -> dict[str, Any]:
    model = init_perceiver_local(seed)
    model.set_time_alpha(time_alpha)
    model.train()
    order = Order(len(cache), batch, seed)
    crops = cache.crops(order.indices(), order.crop_gen)
    loss, loss_g, loss_l = perceiver_local_loss(
        model,
        crops["mel1"],
        crops["mel2"],
        crops["frame_valid1"],
        crops["frame_valid2"],
        order.mask_gen,
        1.0,
    )
    finite = bool(torch.isfinite(loss).all())
    grad_norm = None
    grad_finite = False
    if finite:
        loss.backward()
        square = 0.0
        saw = False
        grad_finite = True
        for param in model.student_parameters():
            if param.grad is None:
                continue
            saw = True
            if not bool(torch.isfinite(param.grad).all()):
                grad_finite = False
                break
            square += float(param.grad.detach().float().square().sum())
        grad_finite = bool(grad_finite and saw)
        if grad_finite:
            grad_norm = square ** 0.5
    del model
    if not finite or not grad_finite:
        raise SystemExit("lambda 1 is not finite at the locked alpha; it is not rescaled")
    return {
        "requested_lambda": 1.0,
        "chosen_lambda": 1.0,
        "scaled": False,
        "loss": float(loss.detach()),
        "loss_g": float(loss_g.detach()),
        "loss_l": float(loss_l.detach()),
        "grad_norm": grad_norm,
        "time_alpha": float(time_alpha),
    }


def _time_tables(model: Any) -> tuple[torch.Tensor, torch.Tensor]:
    return model.encoder.pos_time.detach(), model.predictor.pos_time.detach()


def measure_offset_crops(cache: MelCache, seed: int, batches: int = 4) -> dict[str, Any]:
    model = init_perceiver_local(seed)
    model.set_time_alpha(0.0)
    model.eval()
    order = Order(len(cache), 64, seed)
    totals = {
        "n_recordings": 0.0,
        "n_same_offset": 0.0,
        "n_unaligned": 0.0,
        "n_aligned_recordings": 0.0,
        "n_index_pairs": 0.0,
        "n_shared_embedding_pairs": 0.0,
        "n_distinct_embedding_pairs": 0.0,
    }
    with torch.no_grad():
        for _ in range(int(batches)):
            crops = cache.crops_with_starts(order.indices(), order.crop_gen)
            row = describe_crop_offsets(crops["start1"], crops["start2"], _time_tables(model))
            for key in totals:
                totals[key] += float(row[key])
    if totals["n_shared_embedding_pairs"] > 0.0:
        raise SystemExit("an aligned moment shares a time positional embedding")
    if totals["n_distinct_embedding_pairs"] <= 0.0:
        raise SystemExit("no crop pair aligns at different time embeddings")
    return {
        "task": "loopear_pltc_offset_crops",
        "neighborhood_patches": TIME_NEIGHBORHOOD_PATCHES,
        "neighborhood_seconds": TIME_NEIGHBORHOOD_PATCHES * 4 * 160 / 16000,
        "patch_frames": 4,
        "alignment": "absolute frame start, exact multiples of 4 frames",
        "batches": int(batches),
        "batch_recordings": 64,
        "seed": int(seed),
        **totals,
        "gpu_usd": 0.0,
    }


def lock_lambda_time(cache: MelCache, seed: int, batch: int, time_alpha: float) -> dict[str, Any]:
    model = init_perceiver_local(seed)
    model.set_time_alpha(time_alpha)
    counted = sum(param.numel() for param in model.encoder.parameters())
    student = sum(param.numel() for param in model.student_parameters())
    reference = init_perceiver_local(seed + 1)
    reference_student = sum(param.numel() for param in reference.student_parameters())
    if counted != PERCEIVER_ENCODER_PARAMS or student != reference_student:
        raise SystemExit(f"parameter count changed: encoder {counted} student {student}")
    if isinstance(model.encoder.time_alpha, torch.nn.Parameter):
        raise SystemExit("time alpha became a parameter")
    model.train()
    order = Order(len(cache), batch, seed)
    chosen = None
    chosen_index = None
    with torch.no_grad():
        for batch_index in range(8):
            crops = cache.crops_with_starts(order.indices(), order.crop_gen)
            _total, _loss_g, loss_l, loss_t, stats = perceiver_ltc_loss(
                model,
                crops["mel1"],
                crops["mel2"],
                crops["frame_valid1"],
                crops["frame_valid2"],
                crops["start1"],
                crops["start2"],
                order.mask_gen,
                1.0,
                1.0,
            )
            if float(stats["n_queries"]) > 0.0 and torch.isfinite(loss_l) and torch.isfinite(loss_t):
                chosen = (crops, float(loss_l.detach()), float(loss_t.detach()), float(stats["n_queries"]))
                chosen_index = batch_index
                break
    if chosen is None or chosen_index is None:
        raise SystemExit("no initialization batch contained an aligned time pair")
    crops, loss_l_value, loss_t_value, n_queries = chosen
    if abs(loss_t_value) == 0.0:
        raise SystemExit("time loss was zero on the lock batch")
    lambda_time = abs(loss_l_value) / abs(loss_t_value)
    model.zero_grad(set_to_none=True)
    total, loss_g, loss_l, loss_t, stats = perceiver_ltc_loss(
        model,
        crops["mel1"],
        crops["mel2"],
        crops["frame_valid1"],
        crops["frame_valid2"],
        crops["start1"],
        crops["start2"],
        torch.Generator().manual_seed(seed + 91),
        1.0,
        lambda_time,
    )
    finite = bool(torch.isfinite(total))
    grad_norm = None
    grad_finite = False
    if finite:
        total.backward()
        square = 0.0
        saw = False
        grad_finite = True
        for param in model.student_parameters():
            if param.grad is None:
                continue
            saw = True
            if not bool(torch.isfinite(param.grad).all()):
                grad_finite = False
                break
            square += float(param.grad.detach().float().square().sum())
        grad_finite = bool(grad_finite and saw)
        if grad_finite:
            grad_norm = square ** 0.5
    after = sum(param.numel() for param in model.encoder.parameters())
    del model, reference
    if not finite or not grad_finite or after != PERCEIVER_ENCODER_PARAMS:
        raise SystemExit("lambda_t is not finite at initialization; it is not swept")
    return {
        "task": "loopear_pltc_lambda_time",
        "loss_sweep": False,
        "tau": float(TAU_LOCAL),
        "tau_changed": False,
        "neighborhood_patches": TIME_NEIGHBORHOOD_PATCHES,
        "lambda_local": 1.0,
        "loss_l": loss_l_value,
        "loss_t_at_unit_weight": loss_t_value,
        "lambda_time": float(lambda_time),
        "lock_batch_index": int(chosen_index),
        "n_queries": n_queries,
        "check_loss": float(total.detach()),
        "check_loss_g": float(loss_g.detach()),
        "check_loss_l": float(loss_l.detach()),
        "check_loss_t": float(loss_t.detach()),
        "grad_norm": grad_norm,
        "chosen_lambda": 1.0,
        "encoder_params": after,
        "time_alpha": float(time_alpha),
        "gpu_usd": 0.0,
    }


def smoke_ema_checkpoint(cache: MelCache, seed: int, time_alpha: float, lambda_time: float, path: Path) -> dict[str, Any]:
    if "smoke" not in path.name:
        raise SystemExit("the smoke checkpoint path must be marked smoke")
    model = init_perceiver_local(seed)
    model.set_time_alpha(time_alpha)
    opt = torch.optim.AdamW(model.student_parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    order = Order(len(cache), 8, seed)
    for step in range(2):
        crops = cache.crops_with_starts(order.indices(), order.crop_gen)
        model.train()
        loss, _loss_g, _loss_l, _loss_t, _stats = perceiver_ltc_loss(
            model,
            crops["mel1"],
            crops["mel2"],
            crops["frame_valid1"],
            crops["frame_valid2"],
            crops["start1"],
            crops["start2"],
            order.mask_gen,
            1.0,
            lambda_time,
        )
        if not torch.isfinite(loss):
            raise SystemExit("smoke loss was not finite")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.student_parameters(), GRAD_CLIP)
        opt.step()
        model.ema_update(ema_at(step, 2))
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": 2,
        "objective": "global_plus_local_plus_time",
        "lambda_local": 1.0,
        "lambda_time": float(lambda_time),
        "time_alpha": float(time_alpha),
        "encoder": model.encoder.state_dict(),
        "global_head": model.global_head.state_dict(),
        "predictor": model.predictor.state_dict(),
        "local_head": model.local_head.state_dict(),
        "teacher_encoder": model.teacher_encoder.state_dict(),
        "teacher_predictor": model.teacher_predictor.state_dict(),
        "teacher_local": model.teacher_local.state_dict(),
    }
    torch.save(payload, path)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    required = ("teacher_encoder", "teacher_predictor", "teacher_local")
    missing = [key for key in required if key not in blob or not isinstance(blob[key], dict)]
    if missing:
        raise SystemExit(f"smoke checkpoint is missing EMA tensors: {missing}")
    student_vec = torch.cat([tensor.detach().float().reshape(-1) for tensor in model.encoder.state_dict().values()])
    teacher_vec = torch.cat([tensor.detach().float().reshape(-1) for tensor in blob["teacher_encoder"].values()])
    if student_vec.shape != teacher_vec.shape:
        raise SystemExit("smoke teacher encoder shape does not match the student")
    delta = float((student_vec - teacher_vec).abs().sum())
    if delta <= 0.0:
        raise SystemExit("smoke EMA teacher matches the student exactly")
    keys = sorted(key for key in blob if key.endswith("_encoder") or key.startswith("teacher_"))
    del model, blob
    return {
        "path": str(path),
        "teacher_tensors_present": True,
        "teacher_keys": keys,
        "teacher_student_l1": delta,
        "steps": 2,
        "gpu_usd": 0.0,
    }


def run_perceiver_local(
    cache: MelCache,
    steps: int,
    batch: int,
    seed: int,
    log_path: Path,
    ckpt_path: Path,
    weight_path: Path,
    run_id: str,
    config: dict[str, Any],
    lambda_local: float,
    snapshot_every: int | None = None,
    time_alpha: float = 0.0,
    lambda_time: float = 0.0,
    time_neighborhood: int = TIME_NEIGHBORHOOD_PATCHES,
) -> dict[str, Any]:
    if torch.cuda.is_available():
        raise SystemExit("cuda is visible; refusing to start rather than risk GPU spend")
    if float(lambda_local) <= 0.0:
        raise SystemExit("P-L requires a positive local weight")
    probe_model = init_perceiver_local(seed + 1)
    probe_model.set_time_alpha(time_alpha)
    counted = sum(p.numel() for p in probe_model.encoder.parameters())
    if counted != PERCEIVER_ENCODER_PARAMS or float(probe_model.encoder.time_alpha) != float(time_alpha):
        raise SystemExit(f"encoder params {counted} != {PERCEIVER_ENCODER_PARAMS}")
    probe_opt = torch.optim.AdamW(
        probe_model.student_parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    probe_order = Order(len(cache), batch, seed + 1)
    write_run_manifest(run_id, config)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    probe_model.train()
    t_probe = time.perf_counter()
    if float(lambda_time) > 0.0:
        probe = cache.crops_with_starts(probe_order.indices(), probe_order.crop_gen)
        loss_probe, _probe_g, _probe_l, _probe_t, _probe_stats = perceiver_ltc_loss(
            probe_model,
            probe["mel1"],
            probe["mel2"],
            probe["frame_valid1"],
            probe["frame_valid2"],
            probe["start1"],
            probe["start2"],
            probe_order.mask_gen,
            lambda_local,
            lambda_time,
            time_neighborhood,
        )
    else:
        probe = cache.crops(probe_order.indices(), probe_order.crop_gen)
        loss_probe, _probe_g, _probe_l = perceiver_local_loss(
            probe_model,
            probe["mel1"],
            probe["mel2"],
            probe["frame_valid1"],
            probe["frame_valid2"],
            probe_order.mask_gen,
            lambda_local,
        )
    probe_opt.zero_grad(set_to_none=True)
    loss_probe.backward()
    torch.nn.utils.clip_grad_norm_(probe_model.student_parameters(), GRAD_CLIP)
    probe_opt.step()
    profile = {
        "sec_per_step": time.perf_counter() - t_probe,
        "peak_rss_gb": peak_rss_gb(),
        "loss": float(loss_probe.detach()),
        "batch": batch,
    }
    print(json.dumps({"profile": profile}), flush=True)
    del probe_model, probe_opt, probe
    model = init_perceiver_local(seed)
    model.set_time_alpha(time_alpha)
    if float(model.encoder.time_alpha) != float(time_alpha):
        raise SystemExit("local perceiver did not keep its time anchor")
    opt = torch.optim.AdamW(model.student_parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    order = Order(len(cache), batch, seed)
    init_vec = flat_params(model.encoder).clone()
    t0 = time.perf_counter()
    cpu0 = cpu_seconds()
    rows = []
    snapshots: list[str] = []
    stopped_collapse = False
    with log_path.open("w", encoding="utf-8") as handle:
        for step in range(int(steps)):
            model.train()
            if float(model.encoder.time_alpha) != float(time_alpha):
                raise RuntimeError("time anchor changed during training")
            lr = lr_at(step, steps)
            _set_lr(opt, lr)
            if float(lambda_time) > 0.0:
                batch_tensors = cache.crops_with_starts(order.indices(), order.crop_gen)
                loss, loss_g, loss_l, loss_t, time_stats = perceiver_ltc_loss(
                    model,
                    batch_tensors["mel1"],
                    batch_tensors["mel2"],
                    batch_tensors["frame_valid1"],
                    batch_tensors["frame_valid2"],
                    batch_tensors["start1"],
                    batch_tensors["start2"],
                    order.mask_gen,
                    lambda_local,
                    lambda_time,
                    time_neighborhood,
                )
            else:
                batch_tensors = cache.crops(order.indices(), order.crop_gen)
                loss, loss_g, loss_l = perceiver_local_loss(
                    model,
                    batch_tensors["mel1"],
                    batch_tensors["mel2"],
                    batch_tensors["frame_valid1"],
                    batch_tensors["frame_valid2"],
                    order.mask_gen,
                    lambda_local,
                )
                loss_t = loss.detach() * 0.0
                time_stats = {"n_queries": 0.0}
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite loss")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.student_parameters(), GRAD_CLIP)
            opt.step()
            model.ema_update(ema_at(step, steps))
            stats = {
                "step": step,
                "loss": float(loss.detach()),
                "loss_g": float(loss_g.detach()),
                "loss_l": float(loss_l.detach()),
                "loss_t": float(loss_t.detach()),
                "lambda_local": float(lambda_local),
                "lambda_time": float(lambda_time),
                "n_time_queries": float(time_stats["n_queries"]),
                "k": 1,
                "time_alpha": float(time_alpha),
                "lr": float(lr),
                "ema": float(ema_at(step, steps)),
                "grad_norm": float(grad_norm),
            }
            if (step + 1) % 100 == 0:
                snap = _encode_cache(model, cache, min(32, len(cache)))
                stats["diagnostics"] = snap
                if _collapsed(snap):
                    stopped_collapse = True
            rows.append(stats)
            handle.write(json.dumps(stats, sort_keys=True) + "\n")
            finished = step + 1
            if snapshot_every and finished % int(snapshot_every) == 0 and finished < int(steps):
                snapshots.append(_snapshot_encoder(model, ckpt_path, finished))
            if stopped_collapse:
                print(f"local perceiver collapsed at step {step}", flush=True)
                break
            if step % 10 == 0 or step + 1 == steps:
                handle.flush()
                print(
                    f"local perceiver step {step} loss {stats['loss']:.4f} "
                    f"g {stats['loss_g']:.4f} l {stats['loss_l']:.4f} t {stats['loss_t']:.4f}",
                    flush=True,
                )
    wall = time.perf_counter() - t0
    cpu = cpu_seconds() - cpu0
    diag = _encode_cache(model, cache, min(32, len(cache)))
    moved = param_delta_l2(flat_params(model.encoder), init_vec)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": len(rows),
            "objective": "global_plus_local_plus_time" if float(lambda_time) > 0.0 else "global_plus_local",
            "lambda_local": float(lambda_local),
            "lambda_time": float(lambda_time),
            "time_neighborhood": int(time_neighborhood),
            "tau_time": float(TAU_LOCAL),
            "k": 1,
            "time_alpha": float(time_alpha),
            "encoder": model.encoder.state_dict(),
            "global_head": model.global_head.state_dict(),
            "predictor": model.predictor.state_dict(),
            "local_head": model.local_head.state_dict(),
            "teacher_encoder": model.teacher_encoder.state_dict(),
            "teacher_predictor": model.teacher_predictor.state_dict(),
            "teacher_local": model.teacher_local.state_dict(),
            "config": config,
        },
        ckpt_path,
    )
    if weight_path.exists():
        raise FileExistsError(weight_path)
    counted = sum(p.numel() for p in model.encoder.parameters())
    if counted != PERCEIVER_ENCODER_PARAMS or float(model.encoder.time_alpha) != float(time_alpha):
        raise SystemExit("refusing to save an encoder outside the 685,728-parameter budget")
    weight_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(model.encoder.state_dict(), str(weight_path))
    losses = [row["loss"] for row in rows]
    return {
        "objective": "global_plus_local_plus_time" if float(lambda_time) > 0.0 else "global_plus_local",
        "lambda_local": float(lambda_local),
        "lambda_time": float(lambda_time),
        "time_neighborhood": int(time_neighborhood),
        "tau_time": float(TAU_LOCAL),
        "k": 1,
        "time_alpha": float(time_alpha),
        "anchored": float(time_alpha) != 0.0,
        "predictor": "position queries cross-attend to the 32 latents; acoustic patches are not an input",
        "teacher": "stop-grad EMA of the encoder, the latent predictor, and the local head",
        "steps": len(rows),
        "steps_requested": int(steps),
        "stopped_collapse": stopped_collapse,
        "batch_recordings": int(batch),
        "views_per_recording": 2,
        "negatives_per_anchor": int(2 * (batch - 1)),
        "denominator_terms_per_anchor": int(1 + 2 * (batch - 1)),
        "crop_presentations": int(len(rows) * batch),
        "presented_crop_hours": hours(len(rows) * batch * CROP_SECONDS),
        "loss_start": losses[0],
        "loss_end": losses[-1],
        "loss_min": min(losses),
        "loss_g_end": rows[-1]["loss_g"],
        "loss_l_end": rows[-1]["loss_l"],
        "loss_t_end": rows[-1].get("loss_t"),
        "weights_l2_change": moved,
        "weights_moved": moved > 0.0,
        "diagnostics": diag,
        "collapsed": _collapsed(diag) or stopped_collapse,
        "wall_s": wall,
        "cpu_s": cpu,
        "sec_per_step": wall / max(len(rows), 1),
        "cpu_hours": hours(cpu),
        "wall_hours": hours(wall),
        "profile_step": profile,
        "peak_rss_gb": peak_rss_gb(),
        "encoder_path": str(weight_path),
        "snapshots": snapshots,
        "encoder_params": counted,
        "flops": analytic_perceiver_flops(1, 200),
        "flops_batch": analytic_perceiver_flops(batch, 200),
        "loopear_d_flops": analytic_encoder_flops(1, 201),
    }


def _launch_prepared(which: str, seed: int) -> None:
    if torch.cuda.is_available():
        raise SystemExit("cuda is visible; refusing to start rather than risk GPU spend")
    torch.set_num_threads(4)
    trainer = _load_d_trainer()
    manifest = load_manifest(Path("data/manifests/fsd50k_loopear_d.json"))
    rows = items_for(manifest, "h20")
    assert_train_rows(rows)
    digest = sha256_of_ids(rows)
    if digest != "13c22b307322c88c09710b953c706b9432f800c21ff90bed834bd4eed82bf017":
        raise SystemExit(f"h20 id sha {digest}")
    missing = [
        row
        for row in rows
        if not (Path("cache/loopear_d_mel") / f"{int(row['freesound_id'])}.npy").exists()
    ]
    if missing:
        raise SystemExit(
            f"{len(missing)} of {len(rows)} 20-hour clips have no mel cache; "
            "this program does not build that cache and does not start the run"
        )
    if which == "loopear-p":
        log_path = Path("results/loopear_p_h20_global_5000.jsonl")
        ckpt_path = Path("checkpoints/loopear_p_h20_global_5000.pt")
        weight_path = Path("results/loopear_p_h20_global_5000_encoder.safetensors")
        report_path = Path("results/loopear_p_h20_train.json")
    elif which == "loopear-d":
        log_path = Path("results/loopear_d_h20_global_5000.jsonl")
        ckpt_path = Path("checkpoints/loopear_d_h20_global_5000.pt")
        weight_path = Path("results/loopear_d_h20_global_5000_encoder.safetensors")
        report_path = Path("results/loopear_d_h20_train.json")
    elif which == "loopear-pt":
        log_path = Path("results/loopear_pt_h20_global_5000.jsonl")
        ckpt_path = Path("checkpoints/loopear_pt_h20_global_5000.pt")
        weight_path = Path("results/loopear_pt_h20_global_5000_encoder.safetensors")
        report_path = Path("results/loopear_pt_h20_train.json")
    elif which == "loopear-pl":
        log_path = Path("results/loopear_pl_h20_global_5000.jsonl")
        ckpt_path = Path("checkpoints/loopear_pl_h20_global_5000.pt")
        weight_path = Path("results/loopear_pl_h20_global_5000_encoder.safetensors")
        report_path = Path("results/loopear_pl_h20_train.json")
    elif which == "loopear-plt":
        log_path = Path("results/loopear_plt_h20_global_5000.jsonl")
        ckpt_path = Path("checkpoints/loopear_plt_h20_global_5000.pt")
        weight_path = Path("results/loopear_plt_h20_global_5000_encoder.safetensors")
        report_path = Path("results/loopear_plt_h20_train.json")
    elif which == "loopear-pltc":
        log_path = Path("results/loopear_pltc_h20_global_5000.jsonl")
        ckpt_path = Path("checkpoints/loopear_pltc_h20_global_5000.pt")
        weight_path = Path("results/loopear_pltc_h20_global_5000_encoder.safetensors")
        report_path = Path("results/loopear_pltc_h20_train.json")
    else:
        raise SystemExit(f"unknown arm {which}")
    occupied = [path for path in (log_path, ckpt_path, weight_path, report_path) if path.exists()]
    if occupied:
        raise SystemExit(f"refusing to overwrite {[str(path) for path in occupied]}")
    cache = trainer._cache(
        manifest,
        "h20",
        Path("cache/loopear_d_mel"),
        Path("data/manifests/loopear_d_frontend.json"),
        None,
    )
    config = {
        "model": which,
        "subset": "h20",
        "steps": 5000,
        "batch": 64,
        "seed": seed,
        "objective": (
            "global_plus_local_plus_time"
            if which == "loopear-pltc"
            else "global_plus_local"
            if which in ("loopear-pl", "loopear-plt")
            else "global"
        ),
        "lambda_local": 0.0,
        "snapshot_every": 1000,
    }
    sweep = None
    penalty_scale = None
    lambda_check = None
    preserved_before = None
    time_rank = None
    softmax_alpha = None
    offset_crops = None
    lambda_time_lock = None
    smoke_ema = None
    locked_time_alpha = 0.0027896127554686775
    if which == "loopear-pt":
        sweep = select_time_alpha(cache, seed, 64)
        config["time_alpha"] = sweep["alpha"]
        print(json.dumps({"locked_alpha": sweep["alpha"], "factor": sweep["factor"]}), flush=True)
    before_train = None
    if which == "loopear-pl":
        preserved_before = {str(path): _file_sha256(path) for path in _preserved_weight_paths()}
        before_cpu = cpu_seconds()
        before_wall = time.perf_counter()
        scale_path = Path("results/loopear_pt_penalty_scale.json")
        if scale_path.exists():
            penalty_scale = json.loads(scale_path.read_text(encoding="utf-8"))
        else:
            penalty_scale = measure_locked_penalty(cache, seed, 8)
            write_result_json(scale_path, penalty_scale)
        print(
            json.dumps(
                {
                    "penalty_rms_over_logit_rms": penalty_scale["pooled"]["penalty_rms_over_logit_rms"],
                    "content_logit_rms": penalty_scale["pooled"]["content_logit_rms"],
                    "anchor_penalty_rms": penalty_scale["pooled"]["anchor_penalty_rms"],
                    "alpha": penalty_scale["alpha"],
                    "alpha_changed": False,
                }
            ),
            flush=True,
        )
        lambda_check = choose_local_lambda(cache, seed, 64)
        config["lambda_local"] = lambda_check["chosen_lambda"]
        config["k"] = 1
        config["time_alpha"] = 0.0
        print(json.dumps({"lambda_local": lambda_check["chosen_lambda"], "scaled": lambda_check["scaled"]}), flush=True)
        before_train = {
            "cpu_s": cpu_seconds() - before_cpu,
            "wall_s": time.perf_counter() - before_wall,
            "cpu_hours": hours(cpu_seconds() - before_cpu),
        }
    if which == "loopear-plt":
        preserved_before = {str(path): _file_sha256(path) for path in _preserved_weight_paths()}
        before_cpu = cpu_seconds()
        before_wall = time.perf_counter()
        rank_path = Path("results/loopear_pl_time_rank.json")
        if rank_path.exists():
            time_rank = json.loads(rank_path.read_text(encoding="utf-8"))
        else:
            time_rank = measure_pl_time_rank(cache, seed, 16)
            write_result_json(rank_path, time_rank)
        pooled_rank = time_rank["pooled"]["masked_queries"]
        print(
            json.dumps(
                {
                    "masked_queries": pooled_rank["n_queries_with_other_moment"],
                    "fraction_correct_strictly_best": pooled_rank["fraction_correct_strictly_best"],
                    "mean_correct_cosine": pooled_rank["mean_correct_cosine"],
                    "mean_other_time_cosine": pooled_rank["mean_other_time_cosine"],
                    "mean_teacher_offdiag_cosine": pooled_rank["mean_teacher_offdiag_cosine"],
                }
            ),
            flush=True,
        )
        alpha_path = Path("results/loopear_plt_alpha.json")
        if alpha_path.exists():
            softmax_alpha = json.loads(alpha_path.read_text(encoding="utf-8"))
        else:
            softmax_alpha = lock_softmax_alpha(cache, seed, 8)
            write_result_json(alpha_path, softmax_alpha)
        print(
            json.dumps(
                {
                    "alpha": softmax_alpha["alpha"],
                    "mean_total_variation": softmax_alpha["mean_total_variation"],
                    "fraction_argmax_changed": softmax_alpha["fraction_argmax_changed"],
                    "loss_sweep": False,
                }
            ),
            flush=True,
        )
        lambda_check = confirm_local_lambda(cache, seed, 64, float(softmax_alpha["alpha"]))
        config["lambda_local"] = 1.0
        config["k"] = 1
        config["time_alpha"] = float(softmax_alpha["alpha"])
        before_train = {
            "cpu_s": cpu_seconds() - before_cpu,
            "wall_s": time.perf_counter() - before_wall,
            "cpu_hours": hours(cpu_seconds() - before_cpu),
        }
    if which == "loopear-pltc":
        preserved_before = {str(path): _file_sha256(path) for path in _preserved_weight_paths()}
        before_cpu = cpu_seconds()
        before_wall = time.perf_counter()
        before_path = Path("results/loopear_pltc_before.json")
        smoke_path = Path("checkpoints/loopear_pltc_smoke_ema.pt")
        if before_path.exists():
            stored = json.loads(before_path.read_text(encoding="utf-8"))
            offset_crops = stored["offset_crops"]
            lambda_time_lock = stored["lambda_time"]
            smoke_ema = stored["smoke_ema"]
            if float(lambda_time_lock["lambda_time"]) <= 0.0 or bool(lambda_time_lock["loss_sweep"]):
                raise SystemExit("stored lambda_t is not a one-batch lock")
            blob = torch.load(smoke_path, map_location="cpu", weights_only=False)
            required = ("teacher_encoder", "teacher_predictor", "teacher_local")
            if any(key not in blob or not isinstance(blob[key], dict) for key in required):
                raise SystemExit("stored smoke checkpoint is missing EMA tensors")
            del blob
            before_train = stored.get("before_train")
        else:
            offset_crops = measure_offset_crops(cache, seed, 4)
            lambda_time_lock = lock_lambda_time(cache, seed, 64, locked_time_alpha)
            smoke_ema = smoke_ema_checkpoint(
                cache, seed, locked_time_alpha, float(lambda_time_lock["lambda_time"]), smoke_path
            )
            if not smoke_ema["teacher_tensors_present"]:
                raise SystemExit("smoke checkpoint does not contain EMA tensors")
            before_train = {
                "cpu_s": cpu_seconds() - before_cpu,
                "wall_s": time.perf_counter() - before_wall,
                "cpu_hours": hours(cpu_seconds() - before_cpu),
            }
            write_result_json(
                before_path,
                {
                    "offset_crops": offset_crops,
                    "lambda_time": lambda_time_lock,
                    "smoke_ema": smoke_ema,
                    "before_train": before_train,
                    "gpu_usd": 0.0,
                },
            )
        config["lambda_local"] = 1.0
        config["lambda_time"] = float(lambda_time_lock["lambda_time"])
        config["k"] = 1
        config["time_alpha"] = locked_time_alpha
        config["time_neighborhood"] = TIME_NEIGHBORHOOD_PATCHES
        config["tau_time"] = float(TAU_LOCAL)
        lambda_check = lambda_time_lock
        print(
            json.dumps(
                {
                    "lambda_time": float(lambda_time_lock["lambda_time"]),
                    "loss_l": lambda_time_lock["loss_l"],
                    "loss_t_at_unit_weight": lambda_time_lock["loss_t_at_unit_weight"],
                    "neighborhood_patches": TIME_NEIGHBORHOOD_PATCHES,
                    "neighborhood_seconds": offset_crops["neighborhood_seconds"],
                    "distinct_embedding_pairs": offset_crops["n_distinct_embedding_pairs"],
                    "shared_embedding_pairs": offset_crops["n_shared_embedding_pairs"],
                    "smoke_teacher_tensors": True,
                    "loss_sweep": False,
                }
            ),
            flush=True,
        )
        if before_train is None:
            before_train = {
                "cpu_s": cpu_seconds() - before_cpu,
                "wall_s": time.perf_counter() - before_wall,
                "cpu_hours": hours(cpu_seconds() - before_cpu),
            }
    ledger = SpendLedger(Path("ledger/loopear_h20_spend.jsonl"), cap_usd=0.0)
    ledger.record(which, "cpu", 0.0, "no paid GPU")
    if which == "loopear-pltc":
        trained = run_perceiver_local(
            cache,
            5000,
            64,
            seed,
            log_path,
            ckpt_path,
            weight_path,
            "loopear_pltc_h20_global_5000",
            config,
            1.0,
            snapshot_every=1000,
            time_alpha=locked_time_alpha,
            lambda_time=float(lambda_time_lock["lambda_time"]),
            time_neighborhood=TIME_NEIGHBORHOOD_PATCHES,
        )
        preserved_after = {str(path): _file_sha256(path) for path in _preserved_weight_paths()}
        if preserved_after != preserved_before:
            raise SystemExit("a preserved D, P, P-T, P-L, or P-LT encoder changed during P-LTC")
        saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        required = ("teacher_encoder", "teacher_predictor", "teacher_local")
        if any(key not in saved or not isinstance(saved[key], dict) for key in required):
            raise SystemExit("the trained checkpoint is missing EMA tensors")
        del saved
    elif which == "loopear-plt":
        trained = run_perceiver_local(
            cache,
            5000,
            64,
            seed,
            log_path,
            ckpt_path,
            weight_path,
            "loopear_plt_h20_global_5000",
            config,
            1.0,
            snapshot_every=1000,
            time_alpha=float(softmax_alpha["alpha"]),
        )
        preserved_after = {str(path): _file_sha256(path) for path in _preserved_weight_paths()}
        if preserved_after != preserved_before:
            raise SystemExit("a preserved D, P, P-T, or P-L encoder changed during P-LT")
    elif which == "loopear-pl":
        trained = run_perceiver_local(
            cache,
            5000,
            64,
            seed,
            log_path,
            ckpt_path,
            weight_path,
            "loopear_pl_h20_global_5000",
            config,
            float(lambda_check["chosen_lambda"]),
            snapshot_every=1000,
        )
        preserved_after = {str(path): _file_sha256(path) for path in _preserved_weight_paths()}
        if preserved_after != preserved_before:
            raise SystemExit("a preserved D, P, or P-T encoder changed during P-L")
    elif which in ("loopear-p", "loopear-pt"):
        trained = run_perceiver(
            cache,
            5000,
            64,
            seed,
            log_path,
            ckpt_path,
            weight_path,
            "loopear_pt_h20_global_5000" if which == "loopear-pt" else "loopear_p_h20_global_5000",
            config,
            snapshot_every=1000,
            time_alpha=0.0 if sweep is None else float(sweep["alpha"]),
        )
    else:
        trained = trainer.run_arm(
            cache,
            "global",
            5000,
            64,
            seed,
            log_path,
            ckpt_path,
            "loopear_d_h20_global_5000",
            config,
            snapshot_every=1000,
        )
        trained["crop_presentations"] = int(trained["steps"]) * 64
        trained["presented_crop_hours"] = hours(trained["crop_presentations"] * CROP_SECONDS)
        trained["batch_recordings"] = 64
        trained["negatives_per_anchor"] = 126
    report = {
        "task": "loopear_h20_train",
        "model": which,
        "gpu_usd": 0.0,
        "dataset_clips": len(rows),
        "dataset_hours": sum(float(row["duration_s"]) for row in rows) / 3600.0,
        "dataset_id_sha256": digest,
        "lambda_local": 0.0 if lambda_check is None else lambda_check["chosen_lambda"],
        "objective": (
            "global_plus_local_plus_time"
            if which == "loopear-pltc"
            else "global_plus_local"
            if which in ("loopear-pl", "loopear-plt")
            else "global"
        ),
        "equal_compute_extension": False,
        "time_alpha_sweep": sweep,
        "penalty_versus_logits": penalty_scale,
        "within_recording_time_rank": time_rank,
        "softmax_alpha": softmax_alpha,
        "offset_crops": offset_crops,
        "lambda_time_lock": lambda_time_lock,
        "smoke_ema": smoke_ema,
        "lambda_check": lambda_check,
        "before_train": before_train,
        "preserved_encoder_files": 0 if preserved_before is None else len(preserved_before),
        "run": trained,
        "fsd50k_test_audio_read": False,
        "sealed_realdesed_test_read": False,
    }
    write_result_json(report_path, report)
    print(json.dumps({"wrote": str(report_path), "steps": trained.get("steps"), "gpu_usd": 0.0}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--subset", default="h5")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--launch-prepared",
        choices=("loopear-d", "loopear-p", "loopear-pt", "loopear-pl", "loopear-plt", "loopear-pltc"),
    )
    args = parser.parse_args()
    if args.prepare_only:
        write_plan(None, Path("results/loopear_matched_h20_plan.json"))
        print(json.dumps({"launched": False, "plan": "results/loopear_matched_h20_plan.json"}))
        return
    if args.launch_prepared:
        _launch_prepared(args.launch_prepared, args.seed)
        return
    if args.subset != "h5" or args.steps != 1000 or args.batch != 64:
        raise SystemExit("this entry point only runs the matched 5-hour, 1000-step, batch-64 LoopEar-P job")
    if torch.cuda.is_available():
        raise SystemExit("cuda is visible; refusing to start rather than risk GPU spend")
    torch.set_num_threads(4)
    ledger = SpendLedger(Path("ledger/loopear_p_spend.jsonl"), cap_usd=0.0)
    ledger.record("loopear_p", "cpu", 0.0, "no paid GPU")
    trainer = _load_d_trainer()
    manifest = load_manifest(Path("data/manifests/fsd50k_loopear_d.json"))
    cache = trainer._cache(
        manifest,
        "h5",
        Path("cache/loopear_d_mel"),
        Path("data/manifests/loopear_d_frontend.json"),
        None,
    )
    data = json.loads(Path("results/loopear_d_data.json").read_text(encoding="utf-8"))
    cpu0 = cpu_seconds()
    wall0 = time.perf_counter()
    trained = run_perceiver(
        cache,
        args.steps,
        args.batch,
        args.seed,
        Path("results/loopear_p_global_1000.jsonl"),
        Path("checkpoints/loopear_p_global_1000.pt"),
        Path("results/loopear_p_encoder.safetensors"),
        "loopear_p_global_1000",
        {
            "model": "loopear-p",
            "subset": "h5",
            "steps": args.steps,
            "batch": args.batch,
            "seed": args.seed,
            "objective": "global",
            "lambda_local": 0.0,
        },
    )
    plan = write_plan(float(trained["sec_per_step"]), Path("results/loopear_matched_h20_plan.json"))
    report = {
        "task": "loopear_p_train",
        "gpu_usd": 0.0,
        "device": trainer._device_report(),
        "encoder_params": PERCEIVER_ENCODER_PARAMS,
        "loopear_d_encoder_params": ENCODER_PARAMS,
        "param_delta": int(PERCEIVER_ENCODER_PARAMS - ENCODER_PARAMS),
        "latents": N_LATENTS,
        "latent_width": 96,
        "cross_layers": 3,
        "self_layers": 3,
        "lambda_local": 0.0,
        "local_objective_removed": False,
        "dataset_hours": data["subsets"]["h5"]["hours"],
        "dataset_clips": data["subsets"]["h5"]["clips"],
        "dataset_id_sha256": data["subset_id_sha256"]["h5"],
        "run": trained,
        "h20_plan_launched": plan["launched"],
        "sealed_realdesed_test_read": False,
        "fsd50k_test_audio_read": False,
        "wall_hours": hours(time.perf_counter() - wall0),
        "cpu_hours": hours(cpu_seconds() - cpu0),
    }
    write_result_json("results/loopear_p_train.json", report)
    print(
        json.dumps(
            {
                "params": PERCEIVER_ENCODER_PARAMS,
                "sec_per_step": trained["sec_per_step"],
                "cpu_hours": report["cpu_hours"],
                "loss_end": trained["loss_end"],
                "collapsed": trained["collapsed"],
                "gpu_usd": 0.0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
