import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

from loopear.d.causal import (
    ATTENTION_ENCODER_PARAMS,
    CAUSAL_ENCODER_PARAMS,
    CausalAttentionEncoder,
    CausalLoopEarDEncoder,
)
from loopear.d.diagnostics import peak_rss_gb
from loopear.d.frontend import build_mel, load_frontend_config, normalize_logmel, waveform_logmel
from loopear.d.spec import GRID_F, HOP_SAMPLES, N_FFT, PATCH_FRAMES, SR
from loopear.harness import write_result_json


ROOT = Path(__file__).resolve().parent.parent
STREAM_SAMPLES = N_FFT + 999 * HOP_SAMPLES
REPORT = Path("results/loopear_dgated_orderhead_h20_20000_stream10s.json")


def _mel() -> torch.Tensor:
    generator = torch.Generator().manual_seed(42)
    wave = torch.randn(STREAM_SAMPLES, generator=generator)
    spec = waveform_logmel(wave, build_mel())
    if int(spec.shape[0]) != 1000:
        raise RuntimeError(f"10-second stream produced {spec.shape[0]} frames")
    config = load_frontend_config(ROOT / "data/manifests/loopear_d_frontend.json")
    mean = torch.tensor(config["mean"], dtype=torch.float32)
    std = torch.tensor(config["std"], dtype=torch.float32)
    return normalize_logmel(spec, mean, std).unsqueeze(0)


def _child(kind: str) -> dict[str, float | int | str]:
    if torch.cuda.is_available():
        raise SystemExit("cuda is visible; refusing to start rather than risk GPU spend")
    torch.set_num_threads(4)
    mel = _mel()
    time_steps = int(mel.shape[1]) // PATCH_FRAMES
    visible = torch.ones(int(mel.shape[0]), time_steps, GRID_F, dtype=torch.bool)
    encoder = CausalLoopEarDEncoder() if kind == "gated" else CausalAttentionEncoder()
    params = sum(p.numel() for p in encoder.parameters())
    encoder.eval()
    with torch.inference_mode():
        encoder(mel, visible)
        started = time.perf_counter()
        encoder(mel, visible)
        wall_s = time.perf_counter() - started
    return {
        "backbone": kind,
        "params": int(params),
        "wall_s": float(wall_s),
        "wall_ms": float(wall_s * 1000.0),
        "peak_rss_mb": float(peak_rss_gb() * 1024.0),
        "peak_rss_gb": float(peak_rss_gb()),
        "samples": int(STREAM_SAMPLES),
        "duration_s": float(STREAM_SAMPLES) / float(SR),
        "frames": int(mel.shape[1]),
        "time_steps": int(time_steps),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", choices=("gated", "attention"), default="")
    parser.add_argument("--report", default=str(REPORT))
    args = parser.parse_args()
    if args.child:
        print(json.dumps(_child(args.child), sort_keys=True), flush=True)
        return
    if torch.cuda.is_available():
        raise SystemExit("cuda is visible; refusing to start rather than risk GPU spend")
    report = Path(args.report)
    if report.exists():
        raise SystemExit(f"refusing to overwrite {report}")
    rows = []
    for kind in ("gated", "attention"):
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--child", kind],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise SystemExit(proc.stderr[-2000:] or proc.stdout[-2000:])
        rows.append(json.loads(proc.stdout.strip().splitlines()[-1]))
    by_name = {row["backbone"]: row for row in rows}
    payload = {
        "task": "loopear_dgated_stream10s",
        "gpu_usd": 0.0,
        "machine": "paired sequential subprocesses, 4 CPU threads, one warmup forward excluded from the timer",
        "stream_s": rows[0]["duration_s"],
        "gated_params": CAUSAL_ENCODER_PARAMS,
        "attention_params": ATTENTION_ENCODER_PARAMS,
        "gated": by_name["gated"],
        "attention": by_name["attention"],
    }
    path = write_result_json(report, payload)
    print(json.dumps({"wrote": str(path), "gated_ms": by_name["gated"]["wall_ms"], "attention_ms": by_name["attention"]["wall_ms"]}), flush=True)


if __name__ == "__main__":
    main()
