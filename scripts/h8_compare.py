import time
from pathlib import Path

import torch
from safetensors.torch import save_file

from loopear.encoder import load_usad2_extractor
from loopear.h8eval import evaluate_encoder, mixture_retention
from loopear.h8train import ARMS, N_PAIRS, STEPS, build_pair_bank, cache_teachers, train_arm
from loopear.harness import write_result_json
from loopear.tiny import inference_parameter_count


def _public(metrics: dict) -> dict:
    return {key: value for key, value in metrics.items() if key != "probes"}


def main() -> int:
    t0 = time.perf_counter()
    root = Path(".").resolve()
    print("pairs", flush=True)
    pairs = build_pair_bank(str(root), n_pairs=N_PAIRS, seed=42)
    print("teacher", flush=True)
    teacher, _ = load_usad2_extractor("cpu")
    teachers = cache_teachers(teacher, pairs, batch=4)
    del teacher
    arms = []
    weight_dir = root / "results"
    weight_dir.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        print("train", arm, flush=True)
        model, trace = train_arm(pairs, teachers, arm, steps=STEPS)
        trace = {key: value for key, value in trace.items() if key != "losses"}
        model.eval()

        def encode(wav: torch.Tensor, net: torch.nn.Module = model) -> torch.Tensor:
            with torch.no_grad():
                return net.encode(wav)

        metrics = evaluate_encoder(encode, str(root))
        retention = mixture_retention(encode, str(root), probes=metrics["probes"])
        path = weight_dir / f"h8_{arm}.safetensors"
        save_file({name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}, path)
        arms.append(
            {
                "arm": arm,
                "inference_params": inference_parameter_count(model),
                "train": trace,
                "dev": _public(metrics),
                "retention": retention,
                "weights": str(path.name),
                "forwards_per_step": 3 if arm == "align" else (2 if arm == "distill" else 1),
            }
        )
        del model
    ranked = sorted(arms, key=lambda row: float(row["dev"]["composite"]), reverse=True)
    payload = {
        "gpu_usd": 0.0,
        "wall_s": round(time.perf_counter() - t0, 1),
        "pairs": N_PAIRS,
        "steps": STEPS,
        "arms": arms,
        "highest_composite": ranked[0]["arm"],
    }
    write_result_json(root / "results" / "h8_comparison.json", payload)
    print("comparison", payload["highest_composite"], payload["wall_s"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
