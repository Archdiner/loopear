import json
from pathlib import Path

import torch

from loopear.d.causal import CausalLoopEarD, causal_step_loss, time_valid_from_frames
from loopear.d.diagnostics import cpu_seconds, peak_rss_gb
from loopear.d.frontend import load_frontend_config
from loopear.d.fsd import MelCache, assert_train_rows, items_for, load_manifest
from loopear.d.spec import GRID_T, N_MELS, PATCH_FRAMES, SEED
from loopear.harness import write_result_json


def _shuffle_valid_times(mel: torch.Tensor, frame_valid: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    if mel.ndim != 3 or mel.shape[1] != GRID_T * PATCH_FRAMES:
        raise ValueError("mel must be [B, 200, bins]")
    time_valid = time_valid_from_frames(frame_valid)
    patches = mel.view(mel.shape[0], GRID_T, PATCH_FRAMES, N_MELS).clone()
    for row in range(int(mel.shape[0])):
        slots = time_valid[row].nonzero(as_tuple=False).flatten()
        if int(slots.numel()) < 2:
            continue
        order = torch.randperm(int(slots.numel()), generator=generator)
        patches[row, slots] = patches[row, slots[order]]
    return patches.view(mel.shape[0], GRID_T * PATCH_FRAMES, N_MELS)


def _mean(rows: list[dict[str, float]], key: str) -> float:
    return float(sum(row[key] for row in rows) / len(rows))


def main() -> None:
    if torch.cuda.is_available():
        raise SystemExit("cuda is visible; refusing to start rather than risk GPU spend")
    torch.set_num_threads(2)
    report = Path("results/loopear_dc_h20_step1000_position_shortcut.json")
    if report.exists():
        raise SystemExit(f"refusing to overwrite {report}")
    cpu0 = cpu_seconds()
    blob = torch.load(
        "checkpoints/loopear_dc_h20_5000_step1000.pt",
        map_location="cpu",
        weights_only=False,
    )
    if int(blob["step"]) != 1000:
        raise SystemExit(f"checkpoint step is {blob['step']}")
    model = CausalLoopEarD(separation=False)
    model.encoder.load_state_dict(blob["student_encoder"])
    model.teacher_encoder.load_state_dict(blob["teacher_encoder"])
    model.horizons.load_state_dict(blob["horizons"])
    model.eval()
    manifest = load_manifest(Path("data/manifests/fsd50k_loopear_d.json"))
    rows = items_for(manifest, "h20")
    assert_train_rows(rows)
    frontend = load_frontend_config(Path("data/manifests/loopear_d_frontend.json"))
    cache = MelCache(
        rows,
        Path("cache/loopear_d_mel"),
        torch.tensor(frontend["mean"], dtype=torch.float32),
        torch.tensor(frontend["std"], dtype=torch.float32),
    )
    order_gen = torch.Generator().manual_seed(SEED)
    crop_gen = torch.Generator().manual_seed(SEED + 17)
    shuffle_gen = torch.Generator().manual_seed(SEED + 23)
    order: list[int] = []
    cursor = 0
    batch = 64
    batches = 8
    found = {"real": [], "zeros": [], "shuffle": []}
    n_full = 0
    n_crops = 0
    for _ in range(batches):
        if cursor + batch > len(order):
            order = torch.randperm(len(cache), generator=order_gen).tolist()
            cursor = 0
        chosen = order[cursor : cursor + batch]
        cursor += batch
        crops = cache.crop_once(chosen, crop_gen)
        mel = crops["mel"]
        valid = crops["frame_valid"]
        n_crops += int(mel.shape[0])
        n_full += int(valid.all(dim=1).sum())
        variants = {
            "real": mel,
            "zeros": torch.zeros_like(mel),
            "shuffle": _shuffle_valid_times(mel, valid, shuffle_gen),
        }
        for name, clip in variants.items():
            with torch.no_grad():
                out = causal_step_loss(model, clip, valid)
            found[name].append(
                {
                    "loss": float(out["loss"]),
                    "loss_pred": float(out["loss_pred"]),
                    "loss_h1": float(out["loss_h1"]),
                    "loss_h4": float(out["loss_h4"]),
                    "loss_h12": float(out["loss_h12"]),
                    "loss_var": float(out["loss_var"]),
                    "n_h1": float(out["n_h1"]),
                    "n_h4": float(out["n_h4"]),
                    "n_h12": float(out["n_h12"]),
                }
            )
            print(name, found[name][-1]["loss_pred"], flush=True)
    summary = {}
    for name, rows_out in found.items():
        summary[name] = {
            "loss_pred": _mean(rows_out, "loss_pred"),
            "loss": _mean(rows_out, "loss"),
            "loss_h1": _mean(rows_out, "loss_h1"),
            "loss_h4": _mean(rows_out, "loss_h4"),
            "loss_h12": _mean(rows_out, "loss_h12"),
            "loss_var": _mean(rows_out, "loss_var"),
            "batches": rows_out,
        }
    real = summary["real"]["loss_pred"]
    zeros = summary["zeros"]["loss_pred"]
    shuffled = summary["shuffle"]["loss_pred"]
    destroyed_near = zeros < 0.05 and shuffled < 0.05
    destroyed_jump = zeros > 1.0 and shuffled > 1.0 and real < 0.05
    payload = {
        "task": "loopear_dc_position_shortcut",
        "checkpoint": "checkpoints/loopear_dc_h20_5000_step1000.pt",
        "step": 1000,
        "batches": batches,
        "batch": batch,
        "seed": SEED,
        "n_crops": n_crops,
        "n_fully_valid_crops": n_full,
        "gpu_usd": 0.0,
        "conditions": {
            "real": "cached normalized crop",
            "zeros": "mel replaced by zeros, frame validity unchanged",
            "shuffle": "within each clip, valid 40 ms time patches are permuted; invalid slots and the positional grid stay put",
        },
        "summary": summary,
        "real_loss_pred": real,
        "zeros_loss_pred": zeros,
        "shuffle_loss_pred": shuffled,
        "destroyed_near_0.004": destroyed_near,
        "destroyed_jumps_and_real_stays_low": destroyed_jump,
        "cpu_s": cpu_seconds() - cpu0,
        "peak_rss_gb": peak_rss_gb(),
    }
    path = write_result_json(report, payload)
    print(json.dumps({
        "wrote": str(path),
        "real_loss_pred": real,
        "zeros_loss_pred": zeros,
        "shuffle_loss_pred": shuffled,
        "destroyed_near_0.004": destroyed_near,
        "destroyed_jumps_and_real_stays_low": destroyed_jump,
    }), flush=True)


if __name__ == "__main__":
    main()
