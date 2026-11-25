import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

from loopear.encoder import (
    D_E,
    NUM_ENCODER_LAYERS,
    SR,
    WIN_S,
    load_usad2_extractor,
)
from loopear.harness import write_result_json
from loopear.losses import MultiTaskHeads
from loopear.models import build_arm

ARM_SPECS: list[tuple[str, str, dict[str, Any]]] = [
    ("A_last", "A_last", {"stride": 8}),
    ("A_ml", "A_ml", {"stride": 8}),
    ("B_comp", "B_comp", {"stride": 8}),
    ("B_param", "B_param", {"stride": 8}),
    ("C_param", "C_param", {"stride": 8}),
    ("C_comp", "C_comp", {"stride": 8}),
    ("C_wide", "C_wide", {"stride": 8}),
    ("D", "D", {"stride": 8}),
    ("E_s8", "E", {"stride": 8}),
    ("E_s4", "E", {"stride": 4}),
]


def _peak_rss_gb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024.0 * 1024.0)


def _count_params_m(module: torch.nn.Module) -> float:
    return float(sum(p.numel() for p in module.parameters())) / 1e6


def _make_10s_wave() -> torch.Tensor:
    n = int(round(WIN_S * SR))
    t = torch.arange(n, dtype=torch.float32) / float(SR)
    wav = 0.2 * torch.sin(2.0 * np.pi * 220.0 * t) + 0.1 * torch.sin(
        2.0 * np.pi * 880.0 * t
    )
    return wav.unsqueeze(0)


def _measure_flops(fn) -> float:
    with FlopCounterMode(display=False) as fc:
        fn()
    return float(fc.get_total_flops()) / 1e9


def _worker_profile_encoder(threads: int, warmup: int, trials: int) -> dict[str, Any]:
    torch.set_num_threads(threads)
    enc, _ = load_usad2_extractor("cpu")
    wav = _make_10s_wave()
    with torch.no_grad():
        for _ in range(warmup):
            out = enc(wavs=wav)
        gflops = _measure_flops(lambda: enc(wavs=wav))
        times: list[float] = []
        for _ in range(trials):
            t0 = time.perf_counter()
            out = enc(wavs=wav)
            _ = out["x"].sum().item()
            times.append(time.perf_counter() - t0)
    med_s = float(np.median(times))
    return {
        "params_m": round(_count_params_m(enc), 4),
        "gflops_10s": round(gflops, 4),
        "median_ms": round(med_s * 1000.0, 2),
        "rtf": round(med_s / WIN_S, 5),
        "peak_rss_gb": round(_peak_rss_gb(), 4),
        "t_e": int(out["x"].shape[1]),
    }


def _worker_profile_readout(
    out_dim: int,
    t_z: int,
    threads: int,
    warmup: int,
    trials: int,
) -> dict[str, Any]:
    torch.set_num_threads(threads)
    heads = MultiTaskHeads(d=out_dim).eval()
    z = torch.randn(1, t_z, out_dim)
    pad = torch.zeros(1, t_z, dtype=torch.bool)
    with torch.no_grad():
        for _ in range(warmup):
            _ = heads.event_logits(z)
            _ = heads.speaker_embed(z, pad)
        gflops = _measure_flops(
            lambda: (heads.event_logits(z), heads.speaker_embed(z, pad))
        )
        times: list[float] = []
        for _ in range(trials):
            t0 = time.perf_counter()
            ev = heads.event_logits(z)
            spk = heads.speaker_embed(z, pad)
            _ = (ev.sum() + spk.sum()).item()
            times.append(time.perf_counter() - t0)
    med_s = float(np.median(times))
    return {
        "heads_params_m": round(_count_params_m(heads), 4),
        "gflops_10s": round(gflops, 4),
        "median_ms": round(med_s * 1000.0, 2),
        "rtf": round(med_s / WIN_S, 5),
        "peak_rss_gb": round(_peak_rss_gb(), 4),
    }


def _worker_profile_refine(
    arm_type: str,
    kwargs: dict[str, Any],
    threads: int,
    warmup: int,
    trials: int,
) -> dict[str, Any]:
    torch.set_num_threads(threads)
    ref = build_arm(arm_type, **kwargs).eval()
    t_e = 499
    stride = int(kwargs.get("stride", 8))
    t_z = (t_e + stride - 1) // stride
    e = torch.randn(1, t_e, D_E)
    e_len = torch.tensor([t_e], dtype=torch.long)
    layer_pool = (
        torch.randn(1, NUM_ENCODER_LAYERS, t_z, D_E) if arm_type == "A_ml" else None
    )
    refine_out: dict[str, Any] = {}
    with torch.no_grad():
        for k in ref.depths:
            for _ in range(warmup):
                states, _ = ref.forward_states(
                    e, e_len, depths=(k,), layer_pool=layer_pool
                )
            gflops_ref = _measure_flops(
                lambda depth=k: ref.forward_states(
                    e, e_len, depths=(depth,), layer_pool=layer_pool
                )
            )
            times: list[float] = []
            for _ in range(trials):
                t0 = time.perf_counter()
                states, _ = ref.forward_states(
                    e, e_len, depths=(k,), layer_pool=layer_pool
                )
                _ = states[k].sum().item()
                times.append(time.perf_counter() - t0)
            med_s = float(np.median(times))
            refine_out[str(k)] = {
                "gflops_10s": round(gflops_ref, 4),
                "median_ms": round(med_s * 1000.0, 2),
                "rtf": round(med_s / WIN_S, 5),
            }
    return {
        "out_dim": int(ref.out_dim),
        "t_z": int(t_z),
        "ref_params_m": round(_count_params_m(ref), 4),
        "depths": list(ref.depths),
        "refine": refine_out,
        "peak_rss_gb": round(_peak_rss_gb(), 4),
    }


def _worker_profile_e2e(
    arm_type: str,
    kwargs: dict[str, Any],
    threads: int,
    warmup: int,
    trials: int,
) -> dict[str, Any]:
    torch.set_num_threads(threads)
    enc, _ = load_usad2_extractor("cpu")
    ref = build_arm(arm_type, **kwargs).eval()
    heads = MultiTaskHeads(d=ref.out_dim).eval()
    wav = _make_10s_wave()
    e2e_by_k: dict[str, Any] = {}
    with torch.no_grad():
        for k in ref.depths:
            for _ in range(warmup):
                eo = enc(wavs=wav)
                hs = (
                    torch.stack([h.float() for h in eo["hidden_states"]], dim=1)
                    if arm_type == "A_ml"
                    else None
                )
                st, pad = ref.forward_states(
                    eo["x"].float(), eo["x_lengths"].long(), depths=(k,), hidden_states=hs
                )
                _ = heads.event_logits(st[k])
                _ = heads.speaker_embed(st[k], pad)
            times: list[float] = []
            for _ in range(trials):
                t0 = time.perf_counter()
                eo = enc(wavs=wav)
                hs = (
                    torch.stack([h.float() for h in eo["hidden_states"]], dim=1)
                    if arm_type == "A_ml"
                    else None
                )
                st, pad = ref.forward_states(
                    eo["x"].float(), eo["x_lengths"].long(), depths=(k,), hidden_states=hs
                )
                ev = heads.event_logits(st[k])
                spk = heads.speaker_embed(st[k], pad)
                _ = (ev.sum() + spk.sum()).item()
                times.append(time.perf_counter() - t0)
            med_s = float(np.median(times))
            e2e_by_k[str(k)] = {
                "median_ms": round(med_s * 1000.0, 2),
                "rtf": round(med_s / WIN_S, 5),
            }
    return {
        "e2e": e2e_by_k,
        "peak_rss_gb": round(_peak_rss_gb(), 4),
    }


def _run_subprocess_worker(payload: dict[str, Any], threads: int) -> dict[str, Any]:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    env["MKL_NUM_THREADS"] = str(threads)
    env["OPENBLAS_NUM_THREADS"] = str(threads)
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-json",
        json.dumps(payload),
    ]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
    return json.loads(proc.stdout.strip().splitlines()[-1])


def run_profile(
    out_path: Path,
    enc_trials: int = 3,
    ref_trials: int = 10,
    e2e_trials: int = 3,
) -> dict[str, Any]:
    nproc = os.cpu_count() or 1
    thread_counts = sorted({1, nproc})
    enc_results: dict[str, Any] = {}
    for nt in thread_counts:
        enc_results[f"{nt}c"] = _run_subprocess_worker(
            {"mode": "enc", "threads": nt, "warmup": 1, "trials": enc_trials}, nt
        )
    enc_params_m = enc_results["1c"]["params_m"]
    enc_gflops = enc_results["1c"]["gflops_10s"]

    readout_cache: dict[tuple[int, int, int], dict[str, Any]] = {}
    arms_out: dict[str, Any] = {}
    for label, arm_type, kwargs in ARM_SPECS:
        ref_by_t: dict[str, Any] = {}
        ro_by_t: dict[str, Any] = {}
        e2e_by_t: dict[str, Any] = {}
        for nt in thread_counts:
            ref_res = _run_subprocess_worker(
                {
                    "mode": "refine",
                    "arm_type": arm_type,
                    "kwargs": kwargs,
                    "threads": nt,
                    "warmup": 2,
                    "trials": ref_trials,
                },
                nt,
            )
            ref_by_t[f"{nt}c"] = ref_res
            ro_key = (ref_res["out_dim"], ref_res["t_z"], nt)
            if ro_key not in readout_cache:
                readout_cache[ro_key] = _run_subprocess_worker(
                    {
                        "mode": "readout",
                        "out_dim": ref_res["out_dim"],
                        "t_z": ref_res["t_z"],
                        "threads": nt,
                        "warmup": 2,
                        "trials": ref_trials,
                    },
                    nt,
                )
            ro_by_t[f"{nt}c"] = readout_cache[ro_key]
            e2e_by_t[f"{nt}c"] = _run_subprocess_worker(
                {
                    "mode": "e2e",
                    "arm_type": arm_type,
                    "kwargs": kwargs,
                    "threads": nt,
                    "warmup": 1,
                    "trials": e2e_trials,
                },
                nt,
            )
        ref1 = ref_by_t["1c"]
        ro1 = ro_by_t["1c"]
        depths = ref1["depths"]
        ro_gf = ro1["gflops_10s"]
        gflops_refine = {str(k): ref1["refine"][str(k)]["gflops_10s"] for k in depths}
        gflops_e2e = {
            str(k): round(enc_gflops + gflops_refine[str(k)] + ro_gf, 4) for k in depths
        }
        arms_out[label] = {
            "arm_type": arm_type,
            "kwargs": kwargs,
            "depths": depths,
            "params_m": {
                "enc": enc_params_m,
                "ref": ref1["ref_params_m"],
                "heads": ro1["heads_params_m"],
                "total": round(
                    enc_params_m + ref1["ref_params_m"] + ro1["heads_params_m"], 4
                ),
            },
            "gflops_10s": {
                "enc": enc_gflops,
                "readout": ro_gf,
                "refine": gflops_refine,
                "e2e": gflops_e2e,
            },
            "timings": {
                f"{nt}c": {
                    "readout": {
                        "median_ms": ro_by_t[f"{nt}c"]["median_ms"],
                        "rtf": ro_by_t[f"{nt}c"]["rtf"],
                    },
                    "refine": ref_by_t[f"{nt}c"]["refine"],
                    "enc": {
                        "median_ms": enc_results[f"{nt}c"]["median_ms"],
                        "rtf": enc_results[f"{nt}c"]["rtf"],
                    },
                    "e2e": e2e_by_t[f"{nt}c"]["e2e"],
                    "peak_rss_gb": {
                        "readout": ro_by_t[f"{nt}c"]["peak_rss_gb"],
                        "refine": ref_by_t[f"{nt}c"]["peak_rss_gb"],
                        "enc": enc_results[f"{nt}c"]["peak_rss_gb"],
                        "e2e": e2e_by_t[f"{nt}c"]["peak_rss_gb"],
                    },
                }
                for nt in thread_counts
            },
        }
    report = {
        "hardware": {"cpu_count": nproc, "threads_tested": thread_counts},
        "window_s": WIN_S,
        "encoder": enc_results,
        "arms": arms_out,
    }
    write_result_json(out_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Profile LoopEar arms in isolated subprocesses"
    )
    parser.add_argument("--out", default="results/stage0_profile.json")
    parser.add_argument("--enc-trials", type=int, default=3)
    parser.add_argument("--ref-trials", type=int, default=10)
    parser.add_argument("--e2e-trials", type=int, default=3)
    parser.add_argument("--worker-json", default=None)
    args = parser.parse_args()

    if args.worker_json is not None:
        spec = json.loads(args.worker_json)
        mode = spec["mode"]
        if mode == "enc":
            res = _worker_profile_encoder(
                spec["threads"], spec["warmup"], spec["trials"]
            )
        elif mode == "readout":
            res = _worker_profile_readout(
                spec["out_dim"],
                spec["t_z"],
                spec["threads"],
                spec["warmup"],
                spec["trials"],
            )
        elif mode == "refine":
            res = _worker_profile_refine(
                spec["arm_type"],
                spec["kwargs"],
                spec["threads"],
                spec["warmup"],
                spec["trials"],
            )
        elif mode == "e2e":
            res = _worker_profile_e2e(
                spec["arm_type"],
                spec["kwargs"],
                spec["threads"],
                spec["warmup"],
                spec["trials"],
            )
        else:
            raise ValueError(f"Unknown worker mode: {mode}")
        print(json.dumps(res))
        return 0

    report = run_profile(
        Path(args.out),
        enc_trials=args.enc_trials,
        ref_trials=args.ref_trials,
        e2e_trials=args.e2e_trials,
    )
    print(f"Wrote profile to {args.out}")
    all_key = f"{report['hardware']['threads_tested'][-1]}c"
    for label, info in report["arms"].items():
        kmax = str(info["depths"][-1])
        p_tot = info["params_m"]["total"]
        p_ref = info["params_m"]["ref"]
        gf_ref = info["gflops_10s"]["refine"][kmax]
        gf_e2e = info["gflops_10s"]["e2e"][kmax]
        rtf_1c = info["timings"]["1c"]["e2e"][kmax]["rtf"]
        rtf_ac = info["timings"][all_key]["e2e"][kmax]["rtf"]
        rss = info["timings"]["1c"]["peak_rss_gb"]["e2e"]
        print(
            f"{label:8s} k={kmax} ref={p_ref:.2f}M tot={p_tot:.2f}M "
            f"gf_ref={gf_ref:.3f} gf_e2e={gf_e2e:.2f} "
            f"1c_rtf={rtf_1c:.4f} {all_key}_rtf={rtf_ac:.4f} rss={rss:.3f}GB"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
