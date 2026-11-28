import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import torch

from loopear.d.diagnostics import cpu_seconds, hours, peak_rss_gb
from loopear.d.fsd import items_for, load_manifest
from loopear.d.objective import TIME_NEIGHBORHOOD_PATCHES, temporal_retrieval_sums
from loopear.d.perceiver import load_local_with_teacher, within_recording_time_sums
from loopear.d.spec import HOP_SAMPLES, PATCH_FRAMES, PERCEIVER_ENCODER_PARAMS, SR
from loopear.harness import write_result_json


LOCKED_TIME_ALPHA = 0.0027896127554686775
RETRIEVAL_KEYS = (
    "n_queries",
    "n_strictly_best",
    "n_tied_for_best",
    "n_worse",
    "rank_sum",
    "candidate_sum",
    "chance_top1_sum",
    "chance_rank_sum",
    "correct_cosine_sum",
    "n_same_index",
    "n_correct_beats_same_index",
    "same_index_cosine_sum",
    "n_cross_same_index",
    "n_correct_beats_cross_same_index",
    "cross_same_index_cosine_sum",
)


def _train_module():
    path = Path(__file__).resolve().parent / "train_loopear_p.py"
    spec = importlib.util.spec_from_file_location("train_loopear_p_time", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the LoopEar-P trainer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _blank() -> dict[str, float]:
    return {key: 0.0 for key in RETRIEVAL_KEYS}


def _add(total: dict[str, float], row: dict[str, float]) -> None:
    for key in RETRIEVAL_KEYS:
        total[key] += float(row[key])


def _divide(numerator: float, denominator: float) -> float | None:
    if denominator <= 0.0:
        return None
    return numerator / denominator


def _retrieval_means(total: dict[str, float]) -> dict[str, Any]:
    queries = total["n_queries"]
    return {
        **total,
        "fraction_strictly_best": _divide(total["n_strictly_best"], queries),
        "mean_rank": _divide(total["rank_sum"], queries),
        "chance_top1": _divide(total["chance_top1_sum"], queries),
        "chance_mean_rank": _divide(total["chance_rank_sum"], queries),
        "mean_candidates": _divide(total["candidate_sum"], queries),
        "mean_correct_cosine": _divide(total["correct_cosine_sum"], queries),
        "fraction_correct_beats_same_index": _divide(
            total["n_correct_beats_same_index"], total["n_same_index"]
        ),
        "mean_same_index_cosine": _divide(total["same_index_cosine_sum"], total["n_same_index"]),
        "fraction_correct_beats_cross_same_index": _divide(
            total["n_correct_beats_cross_same_index"], total["n_cross_same_index"]
        ),
        "mean_cross_same_index_cosine": _divide(
            total["cross_same_index_cosine_sum"], total["n_cross_same_index"]
        ),
    }


def _flat(module: torch.nn.Module) -> torch.Tensor:
    return torch.cat([param.detach().float().reshape(-1) for param in module.parameters()])


def _teacher_local(model, mel: torch.Tensor, valid: torch.Tensor, view: int) -> torch.Tensor:
    latents, _summary = model.teacher_encoder(mel, valid)
    return model.project_teacher(model.teacher_predictor(latents, view))


def _student_local(model, mel: torch.Tensor, valid: torch.Tensor, view: int) -> torch.Tensor:
    latents, _summary = model.encoder(mel, valid)
    return model.project_local(model.predictor(latents, view))


def measure(checkpoint: Path, batches: int, seed: int, time_alpha: float) -> dict[str, Any]:
    if torch.cuda.is_available():
        raise SystemExit("cuda is visible; refusing to start rather than risk GPU spend")
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    required = ("teacher_encoder", "teacher_predictor", "teacher_local")
    missing = [key for key in required if key not in blob or not isinstance(blob[key], dict)]
    if missing:
        raise SystemExit(f"checkpoint is missing EMA tensors: {missing}")
    model = load_local_with_teacher(str(checkpoint), time_alpha)
    counted = sum(param.numel() for param in model.encoder.parameters())
    if counted != PERCEIVER_ENCODER_PARAMS:
        raise SystemExit(f"encoder params {counted} != {PERCEIVER_ENCODER_PARAMS}")
    teacher_l1 = float((_flat(model.encoder) - _flat(model.teacher_encoder)).abs().sum())
    if teacher_l1 <= 0.0:
        raise SystemExit("saved EMA teacher matches the student exactly")
    helpers = _train_module()
    trainer = helpers._load_d_trainer()
    manifest = load_manifest(Path("data/manifests/fsd50k_loopear_d.json"))
    rows = items_for(manifest, "h20")
    cache = trainer._cache(
        manifest,
        "h20",
        Path("cache/loopear_d_mel"),
        Path("data/manifests/loopear_d_frontend.json"),
        None,
    )
    if len(cache) != len(rows):
        raise SystemExit("mel cache does not match the 20-hour manifest")
    order = helpers.Order(len(cache), 64, seed)
    pooled = {"all_valid": helpers._blank_rank(), "masked_queries": helpers._blank_rank()}
    views: dict[str, dict[str, Any]] = {}
    ema_retrieval = _blank()
    student_retrieval = _blank()
    with torch.no_grad():
        for _ in range(int(batches)):
            crops = cache.crops_with_starts(order.indices(), order.crop_gen)
            packed = helpers._view_pack(crops, order.mask_gen)
            students = []
            teachers = []
            for view in packed:
                student = _student_local(
                    model, view["mel"], ~view["mask"] & view["valid"], int(view["view"])
                )
                target = _teacher_local(model, view["mel"], view["valid"], int(view["view"]))
                students.append(student)
                teachers.append(target)
                slot = views.setdefault(
                    str(view["name"]),
                    {"all_valid": helpers._blank_rank(), "masked_queries": helpers._blank_rank()},
                )
                full = within_recording_time_sums(student, target, view["valid"])
                masked = within_recording_time_sums(student, target, view["valid"], view["mask"])
                helpers._add_rank(slot["all_valid"], full)
                helpers._add_rank(slot["masked_queries"], masked)
                helpers._add_rank(pooled["all_valid"], full)
                helpers._add_rank(pooled["masked_queries"], masked)
            _add(
                ema_retrieval,
                temporal_retrieval_sums(
                    teachers[0],
                    teachers[1],
                    packed[0]["valid"],
                    packed[1]["valid"],
                    crops["start1"],
                    crops["start2"],
                    TIME_NEIGHBORHOOD_PATCHES,
                ),
            )
            _add(
                ema_retrieval,
                temporal_retrieval_sums(
                    teachers[1],
                    teachers[0],
                    packed[1]["valid"],
                    packed[0]["valid"],
                    crops["start2"],
                    crops["start1"],
                    TIME_NEIGHBORHOOD_PATCHES,
                ),
            )
            _add(
                student_retrieval,
                temporal_retrieval_sums(
                    students[0],
                    teachers[1],
                    packed[0]["valid"],
                    packed[1]["valid"],
                    crops["start1"],
                    crops["start2"],
                    TIME_NEIGHBORHOOD_PATCHES,
                ),
            )
            _add(
                student_retrieval,
                temporal_retrieval_sums(
                    students[1],
                    teachers[0],
                    packed[1]["valid"],
                    packed[0]["valid"],
                    crops["start2"],
                    crops["start1"],
                    TIME_NEIGHBORHOOD_PATCHES,
                ),
            )
    reported_views = {
        name: {
            "all_valid": helpers._rank_means(slot["all_valid"]),
            "masked_queries": helpers._rank_means(slot["masked_queries"]),
        }
        for name, slot in views.items()
    }
    neighborhood_seconds = TIME_NEIGHBORHOOD_PATCHES * PATCH_FRAMES * HOP_SAMPLES / SR
    return {
        "task": "loopear_pltc_ema_time",
        "checkpoint": str(checkpoint),
        "ema_tensors_in_checkpoint": True,
        "teacher_keys": list(required),
        "teacher_student_encoder_l1": teacher_l1,
        "teacher_source": "saved EMA tensors, not a stop-grad copy of the student",
        "encoder_params": counted,
        "time_alpha": float(time_alpha),
        "batches": int(batches),
        "batch_recordings": 64,
        "seed": int(seed),
        "neighborhood_patches": TIME_NEIGHBORHOOD_PATCHES,
        "neighborhood_seconds_each_side": neighborhood_seconds,
        "moment_versus_recording": {
            "comparison": "masked student versus unmasked saved EMA, same crop, other times in that crop",
            "other_recording_role": "comparison only, not the claim",
            "views": reported_views,
            "pooled": {
                "all_valid": helpers._rank_means(pooled["all_valid"]),
                "masked_queries": helpers._rank_means(pooled["masked_queries"]),
            },
        },
        "temporal_retrieval_ema": {
            "query": "unmasked EMA local vector",
            "candidates": "positive absolute moment plus other moments outside the neighborhood",
            "position_control": "same crop-relative index, which is a different absolute time",
            "cross_recording_control": "same crop-relative index in the next batch row",
            **_retrieval_means(ema_retrieval),
        },
        "temporal_retrieval_student_to_ema": {
            "query": "masked student local vector",
            "target": "unmasked saved EMA",
            **_retrieval_means(student_retrieval),
        },
        "gpu_usd": 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-alpha", type=float, default=LOCKED_TIME_ALPHA)
    args = parser.parse_args()
    if args.report.exists():
        raise SystemExit(f"refusing to overwrite {args.report}")
    torch.set_num_threads(4)
    import time

    started = cpu_seconds()
    wall0 = time.perf_counter()
    payload = measure(args.checkpoint, args.batches, args.seed, args.time_alpha)
    payload["cpu_s"] = cpu_seconds() - started
    payload["wall_s"] = time.perf_counter() - wall0
    payload["cpu_hours"] = hours(payload["cpu_s"])
    payload["peak_rss_gb"] = peak_rss_gb()
    write_result_json(args.report, payload)
    print(json.dumps({"wrote": str(args.report), "ema_tensors_in_checkpoint": True, "gpu_usd": 0.0}))


if __name__ == "__main__":
    main()
