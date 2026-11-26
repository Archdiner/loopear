import json
import subprocess
import sys
import time
from pathlib import Path

import torch

from loopear.data import sha256_file
from loopear.encoder import SR, load_usad2_extractor
from loopear.h8data import DEV_CORPORA, cross_group_pair_count, load_manifest, train_hours
from loopear.h8eval import time_encode
from loopear.h8train import BATCH, arm_loss, build_pair_bank, cache_teachers
from loopear.harness import write_result_json
from loopear.tiny import (
    TinyEncoder,
    assert_param_gate,
    causal_mismatch,
    teacher_pair_report,
    wav_to_frame_lengths,
)


def _sha_ok(root: Path) -> dict[str, object]:
    bad: list[str] = []
    checked = 0
    for corpus in DEV_CORPORA:
        manifest = load_manifest(root, corpus)
        for item in manifest["items"]:
            path = root / str(item["audio_path"])
            digest = sha256_file(path)
            checked += 1
            if digest != item["sha256"]:
                bad.append(str(item["audio_path"]))
    return {"checked": checked, "bad": bad, "ok": not bad}


def _data_gate(root: Path) -> dict[str, object]:
    hours = train_hours(root)
    rows = []
    for corpus in DEV_CORPORA:
        manifest = load_manifest(root, corpus)
        for item in manifest["items"]:
            if str(item["split"]) != "train":
                continue
            row = dict(item)
            row["corpus"] = corpus
            rows.append(row)
    pairs = cross_group_pair_count(rows)
    return {
        "train_hours": round(hours, 4),
        "cross_group_pairs": pairs,
        "ok": hours >= 1.0 and pairs >= 20,
    }


def _rtf_gate() -> dict[str, object]:
    reports = []
    for threads in (1, 4):
        proc = subprocess.run(
            [sys.executable, __file__, "--rtf", str(threads)],
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            return {"ok": False, "stderr": proc.stderr[-500:], "threads": threads}
        reports.append(json.loads(proc.stdout.strip().splitlines()[-1]))
    ok = reports[0]["rtf"] < 1.0 and reports[1]["rtf"] < 0.25
    return {"ok": ok, "runs": reports}


def _grad_gate(
    model: TinyEncoder,
    pairs: list[dict],
    teachers: dict,
) -> dict[str, object]:
    packed = None
    from loopear.h8train import _batch

    packed = _batch(pairs, teachers, 0, min(BATCH, len(pairs)))
    failed: list[str] = []
    for arm in ("distill", "mixture", "averaged", "align"):
        model.zero_grad(set_to_none=True)
        loss = arm_loss(model, arm, *packed)
        if not torch.isfinite(loss):
            failed.append(arm)
            continue
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is None or not torch.isfinite(param.grad).all():
                failed.append(f"{arm}:{name}")
            elif float(param.grad.abs().sum()) == 0.0:
                failed.append(f"{arm}:{name}")
    return {"ok": not failed, "failed": failed[:12]}


def _teacher_gate(teacher: torch.nn.Module, student: TinyEncoder) -> dict[str, object]:
    n = 2 * SR
    silence = torch.zeros(1, n)
    t = torch.arange(n, dtype=torch.float32) / float(SR)
    tone = (0.25 * torch.sin(2.0 * torch.pi * 440.0 * t)).unsqueeze(0)
    with torch.no_grad():
        sil = teacher(wavs=silence)["x"].float()
        ton = teacher(wavs=tone)["x"].float()
        student_steps = int(student.encode(tone).shape[1])
    report = teacher_pair_report(sil, ton, sil, ton, student_steps, int(ton.shape[1]))
    report["silence_tone_cosine"] = report["constituent_abs_cosine"]
    return report


def _real_pair_gate(
    teacher: torch.nn.Module,
    pairs: list[dict],
    teachers: dict,
) -> dict[str, object]:
    with torch.no_grad():
        student = TinyEncoder().eval()
        wav = pairs[0]["wav_a"].unsqueeze(0)
        student_steps = int(student.encode(wav).shape[1])
    report = teacher_pair_report(
        teachers["a"][0].unsqueeze(0),
        teachers["b"][0].unsqueeze(0),
        teachers["a"][0].unsqueeze(0),
        teachers["b"][0].unsqueeze(0),
        student_steps,
        int(teachers["a"][0].shape[0]),
    )
    del teacher
    return report


def main() -> int:
    if len(sys.argv) >= 3 and sys.argv[1] == "--rtf":
        model = TinyEncoder().eval()
        print(json.dumps(time_encode(model.encode, int(sys.argv[2]))))
        return 0
    t0 = time.perf_counter()
    root = Path(".").resolve()
    report: dict[str, object] = {"gpu_usd": 0.0}
    print("sha256", flush=True)
    report["sha256"] = _sha_ok(root)
    print("data", flush=True)
    report["data"] = _data_gate(root)
    print("params", flush=True)
    student = TinyEncoder()
    report["params"] = assert_param_gate(student)
    print("rtf", flush=True)
    report["rtf"] = _rtf_gate()
    print("causal", flush=True)
    causal = causal_mismatch(student.eval(), torch.randn(1, 2 * SR))
    report["causal"] = causal
    print("teacher", flush=True)
    teacher, _ = load_usad2_extractor("cpu")
    tone_gate = _teacher_gate(teacher, student)
    print("pairs", flush=True)
    pairs = build_pair_bank(str(root), n_pairs=BATCH, seed=42)
    teachers = cache_teachers(teacher, pairs, batch=BATCH)
    pair_gate = _real_pair_gate(teacher, pairs, teachers)
    tone_gate["real_pair_abs_cosine"] = pair_gate["constituent_abs_cosine"]
    tone_gate["real_pair_frame_gap"] = pair_gate["frame_gap"]
    tone_gate["ok"] = bool(
        tone_gate["ok"]
        and pair_gate["constituent_abs_cosine"] < 0.98
        and pair_gate["frame_gap"] <= 2
        and pair_gate["finite"]
    )
    report["teacher"] = tone_gate
    print("grads", flush=True)
    student.train()
    report["grads"] = _grad_gate(student, pairs, teachers)
    print("smoke", flush=True)
    from loopear.h8train import train_arm

    _, smoke = train_arm(
        pairs,
        teachers,
        "distill",
        steps=8,
        batch=BATCH,
        repeat_first=True,
    )
    smoke_ok = bool(
        smoke["finite"]
        and smoke["last_loss"] < smoke["first_loss"]
        and smoke["max_abs_param_move"] > 0.0
    )
    smoke["ok"] = smoke_ok
    report["smoke"] = smoke
    gates = {
        "sha256": report["sha256"]["ok"],
        "data": report["data"]["ok"],
        "params": report["params"]["ok"],
        "rtf": report["rtf"]["ok"],
        "causal": report["causal"]["ok"],
        "teacher": report["teacher"]["ok"],
        "grads": report["grads"]["ok"],
        "smoke": smoke_ok,
    }
    report["gates"] = gates
    report["ok"] = all(bool(v) for v in gates.values())
    report["wall_s"] = round(time.perf_counter() - t0, 1)
    lengths = wav_to_frame_lengths(torch.tensor([2 * SR]))
    report["student_frames_2s"] = int(lengths[0])
    write_result_json(root / "results" / "h8_gates.json", report)
    print(json.dumps({"ok": report["ok"], "gates": gates}, indent=2), flush=True)
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
