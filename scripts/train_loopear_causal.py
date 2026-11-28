import argparse
import gc
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from loopear.d.causal import (
    CAUSAL_ENCODER_PARAMS,
    CausalLoopEarD,
    CausalLoopEarDEncoder,
    causal_step_loss,
    chance_nce_combined,
    mix_at_snr,
    read_crop_waveform,
    sample_snr,
    zero_mel_init_nce,
)
from loopear.d.causal_probe import score_event_order, teacher_temporal_stats
from loopear.d.diagnostics import cpu_seconds, hours, peak_rss_gb
from loopear.d.frontend import build_mel, load_frontend_config, normalize_logmel, waveform_logmel
from loopear.d.fsd import MelCache, assert_train_rows, audio_path, items_for, load_manifest, sha256_of_ids
from loopear.d.loop import capture_rng, set_lr
from loopear.d.spec import ENCODER_PARAMS, GRAD_CLIP, LR, N_FRAMES, SEED, WEIGHT_DECAY, ema_at, lr_at
from loopear.harness import SpendLedger, write_result_json
from loopear.losses import set_seed


ROOT = Path(__file__).resolve().parent.parent
H20_ID_SHA = "13c22b307322c88c09710b953c706b9432f800c21ff90bed834bd4eed82bf017"
FRONTEND_SHA = "defe151b53153cf1fd125668569d24e4da3a138ec8eedb1a9875c90e30b2570c"
PROBE_EVERY = 500
CHECKPOINT_STEPS = (10000, 20000)
PRESERVED_STEMS = (
    "loopear_dc_notime_h20_1000",
    "loopear_dc_h20_5000",
    "loopear_dcp_h20_5000",
    "loopear_dc_splice_h20_5000",
    "loopear_dc_ordergap_h20_5000",
)
_PREFETCH: subprocess.Popen | None = None


class Order:
    def __init__(self, n_items: int, batch: int, seed: int) -> None:
        if n_items < batch:
            raise ValueError(f"dataset {n_items} is smaller than batch {batch}")
        self.n_items = n_items
        self.batch = batch
        self.order_gen = torch.Generator().manual_seed(seed)
        self.crop_gen = torch.Generator().manual_seed(seed + 17)
        self.order: list[int] = []
        self.cursor = 0

    def indices(self) -> list[int]:
        if self.cursor + self.batch > len(self.order):
            self.order = torch.randperm(self.n_items, generator=self.order_gen).tolist()
            self.cursor = 0
        chosen = self.order[self.cursor : self.cursor + self.batch]
        self.cursor += self.batch
        return chosen


def _rss_gb() -> float:
    rss = 0
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            rss = int(line.split()[1])
    return rss / (1024.0 * 1024.0)


def _available_gb() -> float:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / (1024.0 * 1024.0)
    raise RuntimeError("MemAvailable is missing")


def _refuse_cuda() -> None:
    if torch.cuda.is_available():
        raise SystemExit("cuda is visible; refusing to start rather than risk GPU spend")


def _probe_step(batch: int, separation: bool) -> dict[str, Any]:
    script = (
        "import json,sys\n"
        "import torch\n"
        "from loopear.d.causal import CausalLoopEarD, causal_step_loss\n"
        "from loopear.d.diagnostics import peak_rss_gb\n"
        "from loopear.d.spec import N_FRAMES, N_MELS\n"
        "from loopear.losses import set_seed\n"
        "batch=int(sys.argv[1])\n"
        "separation=sys.argv[2]=='1'\n"
        "torch.set_num_threads(4)\n"
        "set_seed(42)\n"
        "model=CausalLoopEarD(separation=separation)\n"
        "mel=torch.randn(batch,N_FRAMES,N_MELS)\n"
        "valid=torch.ones(batch,N_FRAMES,dtype=torch.bool)\n"
        "mix=torch.roll(mel,1,0) if separation else None\n"
        "mix_valid=valid if separation else None\n"
        "import time\n"
        "t0=time.perf_counter()\n"
        "out=causal_step_loss(model,mel,valid,mix,mix_valid,500)\n"
        "out['loss'].backward()\n"
        "sec=time.perf_counter()-t0\n"
        "print(json.dumps({'ok':True,'rss_gb':peak_rss_gb(),'loss':float(out['loss'].detach()),'sec':sec}))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script, str(batch), "1" if separation else "0"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return {
            "ok": False,
            "batch": int(batch),
            "separation": bool(separation),
            "returncode": int(proc.returncode),
            "stderr": proc.stderr[-1500:],
        }
    row = json.loads(proc.stdout.strip().splitlines()[-1])
    row["batch"] = int(batch)
    row["separation"] = bool(separation)
    return row


def choose_batch(requested: int, separation: bool) -> tuple[int, list[dict[str, Any]]]:
    parent = _rss_gb()
    room = parent + _available_gb() * 0.75
    trial = int(requested)
    reports: list[dict[str, Any]] = []
    while trial >= 8:
        row = _probe_step(trial, separation)
        reports.append(row)
        print(json.dumps({"probe": row, "room_gb": room}), flush=True)
        if row.get("ok") and float(row.get("rss_gb", 99.0)) <= room:
            return trial, reports
        trial //= 2
    raise RuntimeError(f"no batch fit in memory: {reports}")


def _require_fresh(path: Path) -> None:
    if path.exists():
        raise FileExistsError(path)


def _save_encoders(stem: str, model: CausalLoopEarD) -> tuple[str, str]:
    encoder_path = Path("results") / f"{stem}_encoder.safetensors"
    teacher_path = Path("results") / f"{stem}_teacher.safetensors"
    _require_fresh(encoder_path)
    _require_fresh(teacher_path)
    encoder_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(model.encoder.state_dict(), str(encoder_path))
    save_file(model.teacher_encoder.state_dict(), str(teacher_path))
    return str(encoder_path), str(teacher_path)


def _save_checkpoint(
    path: Path,
    model: CausalLoopEarD,
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
    name: str,
    config: dict[str, Any],
) -> None:
    _require_fresh(path)
    payload: dict[str, Any] = {
        "step": int(step),
        "total_steps": int(total_steps),
        "name": name,
        "student_encoder": model.encoder.state_dict(),
        "teacher_encoder": model.teacher_encoder.state_dict(),
        "horizons": model.predictor.heads.state_dict(),
        "horizon_steps": list(model.predictor.horizons),
        "order_head": model.order_head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng": capture_rng(),
        "config": config,
    }
    if model.sep_heads is not None:
        payload["sep_heads"] = model.sep_heads.state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _float_row(stats: dict[str, torch.Tensor], step: int, lr: float, ema: float, grad_norm: float) -> dict[str, float | int]:
    row: dict[str, float | int] = {
        "step": int(step),
        "lr": float(lr),
        "ema": float(ema),
        "grad_norm": float(grad_norm),
    }
    for key, value in stats.items():
        row[key] = float(value.detach())
    return row


def _load_waves(cache: MelCache, indices: list[int], starts: torch.Tensor, audio_root: Path) -> torch.Tensor:
    waves = []
    for index, start in zip(indices, starts.tolist()):
        path = audio_path(audio_root, cache.rows[int(index)])
        waves.append(read_crop_waveform(path, int(start)))
    return torch.stack(waves, dim=0)


def _mix_batch(
    waves: torch.Tensor,
    frame_valid: torch.Tensor,
    snr_gen: torch.Generator,
    mel_bank: torch.nn.Module,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    snr = sample_snr(int(waves.shape[0]), snr_gen)
    mixed, _scale = mix_at_snr(waves, torch.roll(waves, 1, 0), snr)
    spec = waveform_logmel(mixed, mel_bank)
    if spec.ndim != 3 or int(spec.shape[1]) != N_FRAMES:
        raise RuntimeError(f"mixture log-mel shape {tuple(spec.shape)}")
    mix_valid = frame_valid | torch.roll(frame_valid, 1, 0)
    norm = normalize_logmel(spec, mean, std) * mix_valid.unsqueeze(-1)
    return norm, mix_valid


def _alignment_error(cache: MelCache, audio_root: Path, mel_bank: torch.nn.Module) -> float:
    gen = torch.Generator().manual_seed(0)
    chosen = list(range(min(4, len(cache))))
    crops = cache.crop_once(chosen, gen)
    worst = 0.0
    for row, index in enumerate(chosen):
        wave = read_crop_waveform(audio_path(audio_root, cache.rows[index]), int(crops["start"][row]))
        recon = normalize_logmel(waveform_logmel(wave, mel_bank), cache.mean, cache.std)
        mask = crops["frame_valid"][row].unsqueeze(-1)
        err = float(((recon - crops["mel"][row]).abs() * mask).max())
        worst = max(worst, err)
    return worst


def _prefetch_starss() -> None:
    from loopear.d.causal_probe import _event_module

    module = _event_module()
    manifest = module.load_manifest(Path("data/manifests/starss23.json"))
    rows = list(manifest["items"])
    onsets, pairs = module._collect(rows)
    needed = {item["id"] for item in onsets} | {item["id"] for item in pairs}
    cached = 0
    for row in rows:
        if row["id"] not in needed:
            continue
        module._wave(row)
        cached += 1
        if cached % 10 == 0:
            print(json.dumps({"starss_cached": cached, "needed": len(needed)}), flush=True)
    print(json.dumps({"starss_cached": cached, "needed": len(needed)}), flush=True)


def _wait_prefetch() -> None:
    global _PREFETCH
    if _PREFETCH is None:
        return
    code = int(_PREFETCH.wait())
    _PREFETCH = None
    print(json.dumps({"starss_prefetch_returncode": code}), flush=True)


def _event_gate(teacher_path: str, report_path: Path) -> dict[str, Any]:
    _wait_prefetch()
    _require_fresh(report_path)
    proc = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--gate-teacher",
            teacher_path,
            "--gate-report",
            str(report_path),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    if report_path.exists():
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        payload["subprocess_returncode"] = int(proc.returncode)
        return payload
    return {
        "error": "event order subprocess failed",
        "returncode": int(proc.returncode),
        "stderr": proc.stderr[-2000:],
        "stdout": proc.stdout[-500:],
    }


def _counts(model: CausalLoopEarD) -> dict[str, int]:
    sep = 0 if model.sep_heads is None else sum(p.numel() for p in model.sep_heads.parameters())
    return {
        "encoder": sum(p.numel() for p in model.encoder.parameters()),
        "order_head": sum(p.numel() for p in model.order_head.parameters()),
        "horizons": sum(p.numel() for p in model.predictor.parameters()),
        "sep_heads": sep,
        "student": sum(p.numel() for p in model.student_parameters()),
        "teacher": sum(p.numel() for p in model.teacher_encoder.parameters()),
    }


def _fixed_probe(cache: MelCache, batch: int, n_batches: int, seed: int) -> list[dict[str, torch.Tensor]]:
    order_gen = torch.Generator().manual_seed(int(seed))
    crop_gen = torch.Generator().manual_seed(int(seed) + 17)
    order: list[int] = []
    cursor = 0
    stored: list[dict[str, torch.Tensor]] = []
    for _ in range(int(n_batches)):
        if cursor + int(batch) > len(order):
            order = torch.randperm(len(cache), generator=order_gen).tolist()
            cursor = 0
        chosen = order[cursor : cursor + int(batch)]
        cursor += int(batch)
        crops = cache.crop_once(chosen, crop_gen)
        mel = crops["mel"]
        stored.append({"real": mel, "zeros": torch.zeros_like(mel), "valid": crops["frame_valid"]})
    return stored


def _mean_probe(rows: list[dict[str, torch.Tensor]], key: str) -> float:
    return float(sum(float(row[key]) for row in rows) / len(rows))


def _score_probe(model: CausalLoopEarD, stored: list[dict[str, torch.Tensor]], batch: int) -> dict[str, float]:
    training = model.training
    model.eval()
    real_rows: list[dict[str, torch.Tensor]] = []
    silent_rows: list[dict[str, torch.Tensor]] = []
    with torch.no_grad():
        for item in stored:
            real_rows.append(causal_step_loss(model, item["real"], item["valid"], score_gap=True))
            silent_rows.append(causal_step_loss(model, item["zeros"], item["valid"]))
    if training:
        model.train()
    real = _mean_probe(real_rows, "loss_pred")
    perm = _mean_probe(real_rows, "loss_perm")
    silent = _mean_probe(silent_rows, "loss_pred")
    return {
        "real_loss": real,
        "perm_loss": perm,
        "gap_nats": perm - real,
        "silent_loss": silent,
        "chance": chance_nce_combined(int(batch)),
        "silent_minus_chance": silent - chance_nce_combined(int(batch)),
    }


def _start_experiment_2(probe: dict[str, Any]) -> bool:
    forward = probe.get("event_forward")
    reversed_order = probe.get("event_reversed")
    if forward is None or reversed_order is None:
        return False
    return float(forward) > 0.5 and float(reversed_order) < 0.5 and float(reversed_order) < float(forward)


def run_experiment(
    name: str,
    stem: str,
    separation: bool,
    cache: MelCache,
    audio_root: Path,
    mel_bank: torch.nn.Module,
    steps: int,
    batch: int,
    seed: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    set_seed(seed)
    model = CausalLoopEarD(separation=separation)
    counts = _counts(model)
    if counts["encoder"] != CAUSAL_ENCODER_PARAMS:
        raise AssertionError(f"encoder params {counts['encoder']} != {CAUSAL_ENCODER_PARAMS}")
    opt = torch.optim.AdamW(model.student_parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999))
    order = Order(len(cache), batch, seed)
    snr_gen = torch.Generator().manual_seed(seed + 29)
    probe_batches = _fixed_probe(cache, batch, 8, seed)
    log_path = Path("results") / f"{stem}.jsonl"
    probe_path = Path("results") / f"{stem}_probes.jsonl"
    ckpt_path = Path("checkpoints") / f"{stem}.pt"
    _require_fresh(log_path)
    _require_fresh(probe_path)
    _require_fresh(ckpt_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    train_wall = 0.0
    train_cpu = 0.0
    gate_wall = 0.0
    gate_cpu = 0.0
    gate: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []
    loss_probes: list[dict[str, Any]] = []
    with log_path.open("w", encoding="utf-8") as handle, probe_path.open("w", encoding="utf-8") as probe_handle:
        for step in range(int(steps)):
            cpu0 = cpu_seconds()
            wall0 = time.perf_counter()
            chosen = order.indices()
            crops = cache.crop_once(chosen, order.crop_gen)
            mix_mel = None
            mix_valid = None
            mel = crops["mel"]
            frame_valid = crops["frame_valid"]
            if separation:
                waves = _load_waves(cache, chosen, crops["start"], audio_root)
                mix_mel, mix_valid = _mix_batch(
                    waves, crops["frame_valid"], snr_gen, mel_bank, cache.mean, cache.std
                )
            model.train()
            lr = lr_at(step, steps)
            set_lr(opt, lr)
            out = causal_step_loss(model, mel, frame_valid, mix_mel, mix_valid, step)
            opt.zero_grad(set_to_none=True)
            out["loss"].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.student_parameters(), GRAD_CLIP)
            opt.step()
            momentum = ema_at(step, steps)
            model.ema_update(momentum)
            record = _float_row(out, step, lr, momentum, float(grad_norm))
            record["wall_ms"] = (time.perf_counter() - wall0) * 1000.0
            record["peak_rss_mb"] = peak_rss_gb() * 1024.0
            rows.append(record)
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            for key, value in record.items():
                if str(key).startswith("loss") or str(key).startswith("bce"):
                    if not math.isfinite(float(value)):
                        raise RuntimeError(f"{name} non-finite {key} at step {step}: {value}")
            train_wall += time.perf_counter() - wall0
            train_cpu += cpu_seconds() - cpu0
            finished = step + 1
            if finished % 10 == 0 or finished == int(steps):
                handle.flush()
                print(
                    f"{name} step {finished} loss {record['loss']:.4f} "
                    f"real {record['loss_pred']:.4f} order {record['loss_order']:.4f} "
                    f"{record['wall_ms']:.0f} ms",
                    flush=True,
                )
            if finished % PROBE_EVERY == 0:
                rng_state = torch.get_rng_state()
                try:
                    scored = _score_probe(model, probe_batches, batch)
                finally:
                    torch.set_rng_state(rng_state)
                scored["step"] = finished
                scored["peak_rss_mb"] = peak_rss_gb() * 1024.0
                loss_probes.append(scored)
                probe_handle.write(json.dumps(scored, sort_keys=True) + "\n")
                probe_handle.flush()
                print(json.dumps({"probe": scored}), flush=True)
            saved_teacher = ""
            saved_encoder = ""
            if finished in CHECKPOINT_STEPS:
                saved_encoder, saved_teacher = _save_encoders(f"{stem}_step{finished}", model)
                _save_checkpoint(
                    Path("checkpoints") / f"{stem}_step{finished}.pt",
                    model,
                    opt,
                    finished,
                    steps,
                    name,
                    config,
                )
                print(json.dumps({"checkpoint": finished, "stem": stem}), flush=True)
            if finished == int(steps):
                gate_cpu0 = cpu_seconds()
                gate_wall0 = time.perf_counter()
                if saved_teacher:
                    enc_path, tea_path = saved_encoder, saved_teacher
                else:
                    enc_path, tea_path = _save_encoders(f"{stem}_step{finished}", model)
                temporal = teacher_temporal_stats(model.teacher_encoder, cache, 8, batch, seed)
                final_probe = loss_probes[-1] if loss_probes else {}
                gate = {
                    "step": finished,
                    "kept_training": False,
                    "experiment_2_allowed": False,
                    "experiment_2_started": False,
                    "temporal": temporal,
                    "probe": final_probe,
                    "encoder_weights": enc_path,
                    "teacher_weights": tea_path,
                    "wall_s": time.perf_counter() - gate_wall0,
                    "cpu_s": cpu_seconds() - gate_cpu0,
                }
                gate_path = Path("results") / f"{stem}_gate.json"
                _require_fresh(gate_path)
                write_result_json(gate_path, gate)
                gate_wall += float(gate["wall_s"])
                gate_cpu += float(gate["cpu_s"])
                print(json.dumps({"gate": name, "temporal": temporal, "probe": final_probe}), flush=True)
    encoder_path, teacher_path = _save_encoders(stem, model)
    _save_checkpoint(ckpt_path, model, opt, len(rows), steps, name, config)
    summary = {
        "name": name,
        "separation": bool(separation),
        "steps": len(rows),
        "steps_requested": int(steps),
        "batch": int(batch),
        "seed": int(seed),
        "params": counts,
        "loss_first": rows[0]["loss"] if rows else None,
        "loss_last": rows[-1]["loss"] if rows else None,
        "loss_min": min(float(row["loss"]) for row in rows) if rows else None,
        "wall_s": train_wall,
        "cpu_s": train_cpu,
        "gate_wall_s": gate_wall,
        "gate_cpu_s": gate_cpu,
        "sec_per_step": None if not rows else train_wall / float(len(rows)),
        "loss_probes": loss_probes,
        "probe_log": str(probe_path),
        "splice": False,
        "horizons": [4, 10, 20],
        "experiment_2_started": False,
        "gate": gate,
        "encoder_weights": encoder_path,
        "teacher_weights": teacher_path,
        "checkpoint": str(ckpt_path),
        "log": str(log_path),
        "peak_rss_gb": peak_rss_gb(),
        "gpu_usd": 0.0,
    }
    final_probe = loss_probes[-1] if loss_probes else None
    summary["final_probe"] = final_probe
    summary["experiment_2_allowed"] = False if final_probe is None else _start_experiment_2(final_probe)
    summary_path = Path("results") / f"{stem}.json"
    _require_fresh(summary_path)
    write_result_json(summary_path, summary)
    del model, opt
    gc.collect()
    return summary


def _prepare(manifest_path: Path, cache_dir: Path, frontend_path: Path) -> tuple[MelCache, dict[str, Any]]:
    manifest = load_manifest(manifest_path)
    rows = items_for(manifest, "h20")
    assert_train_rows(rows)
    digest = sha256_of_ids(rows)
    stored = manifest["subset_id_sha256"]["h20"]
    if digest != stored or digest != H20_ID_SHA:
        raise SystemExit(f"h20 id hash {digest} stored {stored}")
    if len(rows) != 7405:
        raise SystemExit(f"h20 clip count {len(rows)}")
    payload = load_frontend_config(frontend_path)
    if payload["config_sha256"] != FRONTEND_SHA:
        raise SystemExit(f"frontend hash {payload['config_sha256']}")
    missing = [
        int(row["freesound_id"])
        for row in rows
        if not (cache_dir / f"{int(row['freesound_id'])}.npy").exists()
    ]
    if missing:
        raise SystemExit(f"{len(missing)} h20 log-mels are missing from {cache_dir}")
    for row in rows:
        if row["split"] == "test" or "test" in str(row.get("shard", "")):
            raise PermissionError("training tried to open FSD50K test")
    mean = torch.tensor(payload["mean"], dtype=torch.float32)
    std = torch.tensor(payload["std"], dtype=torch.float32)
    cache = MelCache(rows, cache_dir, mean, std)
    info = {
        "clips": len(rows),
        "hours": sum(float(row["duration_s"]) for row in rows) / 3600.0,
        "subset_id_sha256": digest,
        "frontend_sha256": payload["config_sha256"],
    }
    return cache, info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifests/fsd50k_loopear_d.json")
    parser.add_argument("--cache-dir", default="cache/loopear_d_mel")
    parser.add_argument("--frontend", default="data/manifests/loopear_d_frontend.json")
    parser.add_argument("--audio-root", default="data/raw/fsd50k")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--only", choices=("both", "dc", "dcp"), default="both")
    parser.add_argument("--stem", default="")
    parser.add_argument("--gate-teacher", default="")
    parser.add_argument("--gate-report", default="")
    parser.add_argument("--prefetch-starss", action="store_true")
    args = parser.parse_args()
    _refuse_cuda()
    if args.prefetch_starss:
        torch.set_num_threads(1)
        _prefetch_starss()
        return
    if args.gate_teacher:
        torch.set_num_threads(2)
        encoder = CausalLoopEarDEncoder()
        encoder.load_state_dict(load_file(args.gate_teacher))
        encoder.eval()
        stats = score_event_order(encoder)
        dest = Path(args.gate_report)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({
            "accuracy": stats["accuracy"],
            "reversed": stats["reversed_audio"]["accuracy"],
            "n_pairs": stats["n_pairs"],
        }))
        return
    torch.set_num_threads(4)
    os.chdir(ROOT)
    init_nce = zero_mel_init_nce(int(args.batch), int(args.seed))
    init_chance = chance_nce_combined(int(args.batch))
    print(json.dumps({"init_zero_mel_nce": init_nce, "chance_nce": init_chance}), flush=True)
    if init_nce < 3.5 or abs(init_nce - init_chance) > 0.15:
        raise SystemExit(
            f"init control failed: zero-mel nce {init_nce:.4f} vs chance {init_chance:.4f}"
        )
    ledger = SpendLedger(Path("ledger/loopear_causal_spend.jsonl"), cap_usd=0.0)
    ledger.record("loopear_causal", "cpu", 0.0, "no paid GPU")
    wall0 = time.perf_counter()
    cpu0 = cpu_seconds()
    cache, corpus = _prepare(Path(args.manifest), Path(args.cache_dir), Path(args.frontend))
    audio_root = Path(args.audio_root)
    mel_bank = build_mel()
    align = _alignment_error(cache, audio_root, mel_bank)
    print(json.dumps({"waveform_mel_max_abs": align, "corpus": corpus}), flush=True)
    if align > 0.05:
        raise SystemExit(f"waveform crop does not match the cached mel ({align})")
    global _PREFETCH
    _PREFETCH = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "--prefetch-starss"],
        cwd=str(ROOT),
    )
    config = {
        "seed": int(args.seed),
        "steps": int(args.steps),
        "requested_batch": int(args.batch),
        "corpus": corpus,
        "horizons": [4, 10, 20],
        "temperature": 0.1,
        "variance_weight": 0.0,
        "neighbor": 1,
        "gpu_usd": 0.0,
        "objective": "0.5 within-clip InfoNCE plus 0.5 concatenated time and same-time cross-batch InfoNCE, plus 0.5 times the order BCE, no splice, no PIT, no attention, no learned time position",
        "splice": False,
        "init_zero_mel_nce": init_nce,
        "init_chance_nce": init_chance,
        "time_position": False,
    }
    runs: dict[str, Any] = {}
    probes: dict[str, Any] = {}
    dc_stem = args.stem or "loopear_dgated_orderhead_h20_20000"
    if dc_stem in PRESERVED_STEMS:
        raise SystemExit(f"refusing to overwrite a preserved run ({dc_stem})")
    if args.only == "dcp":
        raise SystemExit("this branch does not train the mixture run")
    experiment_2_started = False
    if args.only in ("both", "dc"):
        batch, probed = choose_batch(int(args.batch), False)
        probes["D-C"] = probed
        runs["D-C"] = run_experiment(
            "D-C", dc_stem, False, cache, audio_root, mel_bank,
            int(args.steps), batch, int(args.seed), config,
        )
    report = {
        "task": "loopear_causal_predictive",
        "gpu_usd": 0.0,
        "cpu_s": cpu_seconds() - cpu0,
        "cpu_hours": hours(cpu_seconds() - cpu0),
        "wall_s": time.perf_counter() - wall0,
        "corpus": corpus,
        "encoder_params": CAUSAL_ENCODER_PARAMS,
        "dense_encoder_params_with_class_token": ENCODER_PARAMS,
        "init_zero_mel_nce": init_nce,
        "init_chance_nce": init_chance,
        "probes": probes,
        "experiment_2_started": experiment_2_started,
        "runs": runs,
        "fsd50k_test_audio_read": False,
        "sealed_realdesed_test_read": False,
        "peak_rss_gb": peak_rss_gb(),
    }
    dest = Path("results/loopear_causal_h20.json")
    if args.only == "both":
        _require_fresh(dest)
        write_result_json(dest, report)
    print(json.dumps({"wrote": str(dest) if args.only == "both" else "partial", "gpu_usd": 0.0}), flush=True)


if __name__ == "__main__":
    main()
