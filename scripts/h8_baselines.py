import time
from pathlib import Path

import torch

from loopear.efficientat import (
    load_efficientat,
    load_ssondo_student,
    parameter_count,
    ssondo_inference_params,
)
from loopear.encoder import load_usad2_extractor
from loopear.h8eval import evaluate_encoder, mixture_retention, time_encode
from loopear.harness import write_result_json


def _public(metrics: dict) -> dict:
    return {key: value for key, value in metrics.items() if key != "probes"}


def _timed(encode, threads: int) -> dict:
    return time_encode(encode, threads)


def main() -> int:
    t0 = time.perf_counter()
    root = Path(".").resolve()
    weight_root = root / "cache" / "h8" / "efficientat"
    rows = []
    print("usad2", flush=True)
    teacher, _ = load_usad2_extractor("cpu")

    def usad_encode(wav: torch.Tensor, net: torch.nn.Module = teacher) -> torch.Tensor:
        with torch.no_grad():
            return net(wavs=wav)["x"].float()

    usad_metrics = evaluate_encoder(usad_encode, str(root))
    usad_ret = mixture_retention(usad_encode, str(root), probes=usad_metrics["probes"])
    rows.append(
        {
            "name": "usad2_small",
            "params": int(sum(p.numel() for p in teacher.parameters())),
            "role": "frozen_reference",
            "rtf_1": _timed(usad_encode, 1),
            "rtf_4": _timed(usad_encode, 4),
            "dev": _public(usad_metrics),
            "retention": usad_ret,
        }
    )
    del teacher
    for name in ("mn01", "mn04", "mn05"):
        print(name, flush=True)
        model = load_efficientat(name, weight_root)

        def encode(wav: torch.Tensor, net: torch.nn.Module = model) -> torch.Tensor:
            with torch.no_grad():
                return net(wav)

        metrics = evaluate_encoder(encode, str(root))
        rows.append(
            {
                "name": name,
                "params": parameter_count(model),
                "role": "public_lightweight",
                "rtf_1": _timed(encode, 1),
                "rtf_4": _timed(encode, 4),
                "dev": _public(metrics),
                "retention": mixture_retention(encode, str(root), probes=metrics["probes"]),
            }
        )
        del model
    print("ssondo", flush=True)
    ssondo = load_ssondo_student()

    def ssondo_encode(wav: torch.Tensor, net: torch.nn.Module = ssondo) -> torch.Tensor:
        with torch.no_grad():
            return net(wav)

    ssondo_metrics = evaluate_encoder(ssondo_encode, str(root))
    rows.append(
        {
            "name": "ssondo_matpac_mobilenetv3",
            "params": ssondo_inference_params(ssondo),
            "role": "public_clip_student_frame_map",
            "rtf_1": _timed(ssondo_encode, 1),
            "rtf_4": _timed(ssondo_encode, 4),
            "dev": _public(ssondo_metrics),
            "retention": mixture_retention(ssondo_encode, str(root), probes=ssondo_metrics["probes"]),
        }
    )
    del ssondo
    payload = {
        "gpu_usd": 0.0,
        "wall_s": round(time.perf_counter() - t0, 1),
        "encoders": rows,
    }
    write_result_json(root / "results" / "h8_baselines.json", payload)
    print("baselines written", payload["wall_s"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
