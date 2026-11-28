import argparse
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from loopear.d.diagnostics import (
    cpu_seconds,
    embedding_diagnostics,
    flat_params,
    hours,
    param_delta_l2,
    peak_rss_gb,
)
from loopear.d.frontend import load_frontend_config
from loopear.d.fsd import MelCache, assert_train_rows, items_for, load_manifest
from loopear.d.loop import init_model, make_optimizer, save_checkpoint, train_step
from loopear.d.probe import encode_rows
from loopear.d.spec import DEVIATIONS, ENCODER_PARAMS, LR, SEED, STUDENT_PARAMS, TRAIN_ONLY_PARAMS
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


def _device_report() -> dict[str, Any]:
    mps = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    return {
        "device": "cpu",
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_available": mps,
        "threads": int(torch.get_num_threads()),
        "gpu_usd": 0.0,
    }


def _cache(manifest: dict[str, Any], subset: str, cache_dir: Path, frontend_path: Path, limit: int | None) -> MelCache:
    rows = items_for(manifest, subset)
    if limit is not None:
        rows = rows[: int(limit)]
    assert_train_rows(rows)
    payload = load_frontend_config(frontend_path)
    mean = torch.tensor(payload["mean"], dtype=torch.float32)
    std = torch.tensor(payload["std"], dtype=torch.float32)
    return MelCache(rows, cache_dir, mean, std)


def _diagnostics(model, cache: MelCache, n: int) -> dict[str, Any]:
    encoded = encode_rows(model, cache, list(range(min(n, len(cache)))))
    return {
        "cls": embedding_diagnostics(encoded["cls"]),
        "patch_mean": embedding_diagnostics(encoded["patch_mean"]),
    }


def _collapsed(stats: dict[str, Any]) -> bool:
    return bool(stats["cls"]["collapsed"] or stats["patch_mean"]["collapsed"] or not stats["cls"]["finite"])


def _snapshot_encoder(model, ckpt_path: Path, step: int) -> str:
    path = Path("results") / f"{ckpt_path.stem}_step{int(step)}_encoder.safetensors"
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(model.encoder.state_dict(), str(path))
    return str(path)


def run_arm(
    cache: MelCache,
    objective: str,
    steps: int,
    batch: int,
    seed: int,
    log_path: Path,
    ckpt_path: Path,
    run_id: str,
    config: dict[str, Any],
    deadline: datetime | None = None,
    snapshot_every: int | None = None,
) -> dict[str, Any]:
    if objective == "random":
        model = init_model(seed)
        diag = _diagnostics(model, cache, min(32, len(cache)))
        return {
            "objective": "random",
            "steps": 0,
            "diagnostics": diag,
            "collapsed": _collapsed(diag),
            "wall_s": 0.0,
            "cpu_s": 0.0,
        }
    model = init_model(seed)
    opt = make_optimizer(model, LR)
    order = Order(len(cache), batch, seed)
    init_vec = flat_params(model.encoder).clone()
    write_run_manifest(run_id, config)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    cpu0 = cpu_seconds()
    rows = []
    snapshots: list[str] = []
    stopped_early = False
    stopped_collapse = False
    with log_path.open("w", encoding="utf-8") as handle:
        for step in range(int(steps)):
            if deadline is not None and datetime.now(timezone.utc) >= deadline:
                stopped_early = True
                break
            batch_tensors = cache.crops(order.indices(), order.crop_gen)
            stats = train_step(
                model,
                opt,
                batch_tensors["mel1"],
                batch_tensors["mel2"],
                batch_tensors["frame_valid1"],
                batch_tensors["frame_valid2"],
                order.mask_gen,
                objective,
                step,
                steps,
            )
            stats["step"] = step
            if (step + 1) % 100 == 0:
                snap = _diagnostics(model, cache, min(32, len(cache)))
                stats["diagnostics"] = snap
                if _collapsed(snap):
                    stopped_collapse = True
            rows.append(stats)
            handle.write(json.dumps(stats, sort_keys=True) + "\n")
            finished = step + 1
            if snapshot_every and finished % int(snapshot_every) == 0 and finished < int(steps):
                snapshots.append(_snapshot_encoder(model, ckpt_path, finished))
            if stopped_collapse:
                print(f"{objective} collapsed at step {step}", flush=True)
                break
            if step % 10 == 0 or step + 1 == steps:
                handle.flush()
                print(
                    f"{objective} step {step} loss {stats['loss']:.4f} "
                    f"g {stats['loss_g']:.4f} l {stats['loss_l']:.4f}",
                    flush=True,
                )
    wall = time.perf_counter() - t0
    cpu = cpu_seconds() - cpu0
    done = len(rows)
    if done == 0:
        return {
            "objective": objective,
            "steps": 0,
            "steps_requested": int(steps),
            "stopped_early": True,
            "collapsed": False,
            "weights_moved": False,
            "reason": "deadline before the first step",
            "wall_s": wall,
            "cpu_s": cpu,
        }
    diag = _diagnostics(model, cache, min(32, len(cache)))
    moved = param_delta_l2(flat_params(model.encoder), init_vec)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(str(ckpt_path), model, opt, done, int(steps), objective, config)
    weight_path = Path("results") / f"{ckpt_path.stem}_encoder.safetensors"
    if weight_path.exists():
        raise FileExistsError(weight_path)
    weight_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(model.encoder.state_dict(), str(weight_path))
    losses = [row["loss"] for row in rows]
    return {
        "objective": objective,
        "steps": int(done),
        "steps_requested": int(steps),
        "stopped_early": stopped_early,
        "batch": int(batch),
        "seed": int(seed),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_min": min(losses),
        "wall_s": wall,
        "cpu_s": cpu,
        "sec_per_step": wall / float(done),
        "encoder_l2_delta": moved,
        "diagnostics": diag,
        "collapsed": bool(_collapsed(diag) or stopped_collapse),
        "weights_moved": moved > 0.0,
        "peak_rss_gb": peak_rss_gb(),
        "checkpoint": str(ckpt_path),
        "encoder_weights": str(weight_path),
        "snapshots": snapshots,
        "log": str(log_path),
    }


def profile_batch(cache: MelCache, batches: list[int], seed: int) -> list[dict[str, float]]:
    found = []
    for batch in batches:
        if batch > len(cache):
            continue
        model = init_model(seed)
        opt = make_optimizer(model, LR)
        order = Order(len(cache), batch, seed)
        warm = cache.crops(order.indices(), order.crop_gen)
        train_step(
            model, opt, warm["mel1"], warm["mel2"], warm["frame_valid1"], warm["frame_valid2"],
            order.mask_gen, "k1", 0, 1000,
        )
        t0 = time.perf_counter()
        timed = cache.crops(order.indices(), order.crop_gen)
        stats = train_step(
            model, opt, timed["mel1"], timed["mel2"], timed["frame_valid1"], timed["frame_valid2"],
            order.mask_gen, "k1", 1, 1000,
        )
        row = {
            "batch": float(batch),
            "sec_per_step": time.perf_counter() - t0,
            "peak_rss_gb": peak_rss_gb(),
            "loss": float(stats["loss"]),
        }
        found.append(row)
        print(json.dumps(row), flush=True)
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("profile", "smoke", "arm", "auto"), default="auto")
    parser.add_argument("--manifest", default="data/manifests/fsd50k_loopear_d.json")
    parser.add_argument("--cache-dir", default="cache/loopear_d_mel")
    parser.add_argument("--frontend", default="data/manifests/loopear_d_frontend.json")
    parser.add_argument("--objective", default="k1")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--smoke-steps", type=int, default=150)
    parser.add_argument("--smoke-clips", type=int, default=32)
    parser.add_argument("--smoke-batch", type=int, default=8)
    parser.add_argument("--report", default="results/loopear_d_train.json")
    parser.add_argument("--deadline-utc", default="2026-11-26T15:30:00+00:00")
    args = parser.parse_args()
    if args.objective not in ("random", "global", "center", "k1"):
        raise SystemExit(f"unknown objective {args.objective}")
    torch.set_num_threads(4)
    device = _device_report()
    if device["cuda_available"]:
        raise SystemExit("cuda is visible; refusing to start rather than risk GPU spend")
    manifest = load_manifest(Path(args.manifest))
    ledger = SpendLedger(Path("ledger/loopear_d_spend.jsonl"), cap_usd=0.0)
    ledger.record("loopear_d", "cpu", 0.0, "no paid GPU")
    cpu0 = cpu_seconds()
    wall0 = time.perf_counter()
    report: dict[str, Any] = {
        "task": "loopear_d_train",
        "stage": args.stage,
        "device": device,
        "encoder_params": ENCODER_PARAMS,
        "train_only_params": TRAIN_ONLY_PARAMS,
        "student_params": STUDENT_PARAMS,
        "deviations": list(DEVIATIONS),
        "gpu_usd": 0.0,
        "audioset_waveforms_in_repo": 0,
    }
    if args.stage == "profile":
        cache = _cache(manifest, "h5", Path(args.cache_dir), Path(args.frontend), None)
        report["profile"] = profile_batch(cache, [8, 16, 32, 64], args.seed)
    elif args.stage == "smoke":
        cache = _cache(manifest, "h5", Path(args.cache_dir), Path(args.frontend), args.smoke_clips)
        report["smoke"] = run_arm(
            cache,
            "k1",
            args.smoke_steps,
            args.smoke_batch,
            args.seed,
            Path("results/loopear_d_smoke.jsonl"),
            Path("checkpoints/loopear_d_smoke.pt"),
            "loopear_d_smoke",
            {
                "objective": "k1",
                "steps": args.smoke_steps,
                "batch": args.smoke_batch,
                "clips": args.smoke_clips,
                "seed": args.seed,
                "subset": "h5_prefix",
            },
        )
    elif args.stage == "arm":
        cache = _cache(manifest, "h5", Path(args.cache_dir), Path(args.frontend), None)
        name = args.objective
        report["arm"] = run_arm(
            cache,
            name,
            args.steps,
            args.batch,
            args.seed,
            Path(f"results/loopear_d_{name}.jsonl"),
            Path(f"checkpoints/loopear_d_{name}.pt"),
            f"loopear_d_{name}_{args.steps}",
            {
                "objective": name,
                "steps": args.steps,
                "batch": args.batch,
                "seed": args.seed,
                "subset": "h5",
            },
        )
    else:
        smoke_cache = _cache(
            manifest, "h5", Path(args.cache_dir), Path(args.frontend), args.smoke_clips
        )
        report["smoke"] = run_arm(
            smoke_cache,
            "k1",
            args.smoke_steps,
            args.smoke_batch,
            args.seed,
            Path("results/loopear_d_smoke.jsonl"),
            Path("checkpoints/loopear_d_smoke.pt"),
            "loopear_d_smoke",
            {
                "objective": "k1",
                "steps": args.smoke_steps,
                "batch": args.smoke_batch,
                "clips": args.smoke_clips,
                "seed": args.seed,
                "subset": "h5_prefix",
            },
        )
        write_result_json("results/loopear_d_train_partial.json", report)
        if report["smoke"]["collapsed"] or not report["smoke"]["weights_moved"]:
            report["stopped"] = "smoke_gate"
            write_result_json(args.report, report)
        else:
            full_cache = _cache(manifest, "h5", Path(args.cache_dir), Path(args.frontend), None)
            profile = []
            for candidate in (16, 32, 64):
                if profile and profile[-1]["peak_rss_gb"] >= 4.0:
                    break
                profile.extend(profile_batch(full_cache, [candidate], args.seed))
            report["profile"] = profile
            usable = [row for row in profile if row["peak_rss_gb"] < 8.0 and row["sec_per_step"] > 0.0]
            chosen = max(usable, key=lambda row: row["batch"]) if usable else {"batch": float(args.batch), "sec_per_step": 10.0}
            batch = int(chosen["batch"])
            sec = float(chosen["sec_per_step"])
            report["chosen_batch"] = batch
            report["profile_sec_per_step"] = sec
            deadline = datetime.fromisoformat(args.deadline_utc)
            if deadline.tzinfo is None:
                deadline = deadline.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            hard = deadline - timedelta(minutes=20)
            remain_s = max(0.0, (hard - now).total_seconds())
            step_cost = max(sec, 1e-6)
            report["hard_stop_utc"] = hard.isoformat()
            if remain_s >= float(args.steps) * step_cost:
                feasible_steps = int(args.steps)
            elif remain_s >= 50.0 * step_cost:
                feasible_steps = int(remain_s / step_cost)
            else:
                feasible_steps = 0
            report["deadline_utc"] = deadline.isoformat()
            report["remain_s_before_stage1"] = remain_s
            report["requested_steps"] = int(args.steps)
            report["feasible_steps"] = feasible_steps
            if feasible_steps < 50:
                report["stopped"] = "deadline_before_stage1"
                report["stage1"] = {"steps": 0, "reason": "fewer than 50 steps fit before the deadline"}
            else:
                stage_deadline = hard
                report["stage1"] = run_arm(
                    full_cache,
                    "k1",
                    feasible_steps,
                    batch,
                    args.seed,
                    Path("results/loopear_d_k1.jsonl"),
                    Path("checkpoints/loopear_d_k1.pt"),
                    f"loopear_d_k1_{feasible_steps}",
                    {
                        "objective": "k1",
                        "steps": feasible_steps,
                        "batch": batch,
                        "seed": args.seed,
                        "subset": "h5",
                    },
                    deadline=stage_deadline,
                )
            if report.get("stopped") == "deadline_before_stage1":
                pass
            elif report["stage1"].get("steps", 0) == 0:
                report["stopped"] = "stage1_deadline"
            elif report["stage1"]["collapsed"]:
                report["stopped"] = "stage1_collapsed"
            elif report["stage1"].get("stopped_early"):
                report["stopped"] = "stage1_deadline"
            else:
                measured = float(report["stage1"]["sec_per_step"])
                now = datetime.now(timezone.utc)
                remain = max(0.0, (hard - now).total_seconds())
                full_match = 2.0 * measured * float(feasible_steps)
                if remain >= full_match:
                    compare_steps = feasible_steps
                else:
                    compare_steps = int(remain / max(3.0 * measured, 1e-6))
                    compare_steps = max(0, min(feasible_steps, compare_steps))
                report["compare_steps"] = compare_steps
                report["compare"] = [
                    run_arm(
                        full_cache,
                        "random",
                        0,
                        batch,
                        args.seed,
                        Path("results/loopear_d_cmp_random.jsonl"),
                        Path("checkpoints/loopear_d_cmp_random.pt"),
                        "loopear_d_cmp_random",
                        {"objective": "random", "steps": 0, "seed": args.seed, "subset": "h5"},
                    )
                ]
                if compare_steps >= 50:
                    names = ["global", "center"]
                    if not (
                        compare_steps == feasible_steps
                        and batch == int(report["stage1"]["batch"])
                    ):
                        names.append("k1")
                    else:
                        reused = dict(report["stage1"])
                        reused["reused_stage1"] = True
                        report["compare"].append(reused)
                    for name in names:
                        report["compare"].append(
                            run_arm(
                                full_cache,
                                name,
                                compare_steps,
                                batch,
                                args.seed,
                                Path(f"results/loopear_d_cmp_{name}.jsonl"),
                                Path(f"checkpoints/loopear_d_cmp_{name}.pt"),
                                f"loopear_d_cmp_{name}_{compare_steps}",
                                {
                                    "objective": name,
                                    "steps": compare_steps,
                                    "batch": batch,
                                    "seed": args.seed,
                                    "subset": "h5",
                                    "matched": True,
                                },
                                deadline=hard,
                            )
                        )
                        if report["compare"][-1].get("stopped_early"):
                            report["stopped"] = "compare_deadline"
                            break
                        if report["compare"][-1]["collapsed"] and name != "global":
                            report["stopped"] = f"compare_{name}_collapsed"
                            break
                else:
                    report["compare_skipped"] = "remaining wall clock below 50 matched steps"
    report["wall_hours"] = hours(time.perf_counter() - wall0)
    report["cpu_hours"] = hours(cpu_seconds() - cpu0)
    report["gpu_usd"] = 0.0
    write_result_json(args.report, report)
    print(json.dumps({"wall_hours": report["wall_hours"], "cpu_hours": report["cpu_hours"], "stopped": report.get("stopped", "ok")}, indent=2))


if __name__ == "__main__":
    main()
