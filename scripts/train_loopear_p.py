import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

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
    broadcast_time_valid,
    global_ntxent,
    patch_valid_from_frames,
    sample_freq_mask,
    sample_time_mask,
)
from loopear.d.perceiver import analytic_perceiver_flops, init_perceiver
from loopear.d.spec import (
    CROP_SECONDS,
    ENCODER_PARAMS,
    GRAD_CLIP,
    LR,
    N_LATENTS,
    PERCEIVER_ENCODER_PARAMS,
    SEED,
    TAU_GLOBAL,
    WEIGHT_DECAY,
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
) -> dict[str, Any]:
    if torch.cuda.is_available():
        raise SystemExit("cuda is visible; refusing to start rather than risk GPU spend")
    probe_model = init_perceiver(seed)
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
    ledger = SpendLedger(Path("ledger/loopear_h20_spend.jsonl"), cap_usd=0.0)
    ledger.record(which, "cpu", 0.0, "no paid GPU")
    config = {
        "model": which,
        "subset": "h20",
        "steps": 5000,
        "batch": 64,
        "seed": seed,
        "objective": "global",
        "lambda_local": 0.0,
        "snapshot_every": 1000,
    }
    if which == "loopear-p":
        trained = run_perceiver(
            cache,
            5000,
            64,
            seed,
            log_path,
            ckpt_path,
            weight_path,
            "loopear_p_h20_global_5000",
            config,
            snapshot_every=1000,
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
        "lambda_local": 0.0,
        "objective": "global",
        "equal_compute_extension": False,
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
    parser.add_argument("--launch-prepared", choices=("loopear-d", "loopear-p"))
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
