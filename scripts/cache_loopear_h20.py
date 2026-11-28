import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from loopear.d.frontend import build_mel, load_frontend_config, waveform_logmel
from loopear.d.fsd import (
    assert_train_rows,
    audio_path,
    extract_audio,
    items_for,
    load_manifest,
    sha256_of_ids,
)
from loopear.harness import write_result_json

EXPECTED_SHA = "13c22b307322c88c09710b953c706b9432f800c21ff90bed834bd4eed82bf017"
FRONTEND_SHA = "defe151b53153cf1fd125668569d24e4da3a138ec8eedb1a9875c90e30b2570c"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _frontend_sha(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    body = {key: value for key, value in payload.items() if key != "config_sha256"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def main() -> None:
    root = Path("data/raw/fsd50k")
    cache_dir = Path("cache/loopear_d_mel")
    frontend_path = Path("data/manifests/loopear_d_frontend.json")
    manifest = load_manifest(Path("data/manifests/fsd50k_loopear_d.json"))
    rows = items_for(manifest, "h20")
    assert_train_rows(rows)
    digest = sha256_of_ids(rows)
    if digest != EXPECTED_SHA:
        raise SystemExit(f"h20 id sha {digest}")
    if len(rows) != 7405:
        raise SystemExit(f"h20 clips {len(rows)}")
    before_frontend = _frontend_sha(frontend_path)
    if before_frontend != FRONTEND_SHA:
        raise SystemExit(f"frontend sha {before_frontend}")
    load_frontend_config(frontend_path)
    kept = rows[:8]
    kept_hash = {int(row["freesound_id"]): _sha256(cache_dir / f"{int(row['freesound_id'])}.npy") for row in kept}
    missing = [row for row in rows if not (cache_dir / f"{int(row['freesound_id'])}.npy").exists()]
    print(f"missing mel {len(missing)} of {len(rows)}", flush=True)
    written = 0
    reused = 0
    max_gap = 0.0
    for start in range(0, len(missing), 200):
        chunk = missing[start : start + 200]
        if not chunk:
            continue
        stats = extract_audio(chunk, root)
        written += int(stats["written"])
        reused += int(stats["reused"])
        max_gap = max(max_gap, float(stats["max_duration_gap_s"]))
        print(
            f"audio {min(start + 200, len(missing))}/{len(missing)} "
            f"written {written} reused {reused}",
            flush=True,
        )
    mel = build_mel()
    cache_dir.mkdir(parents=True, exist_ok=True)
    made = 0
    for index, row in enumerate(missing):
        fid = int(row["freesound_id"])
        dest = cache_dir / f"{fid}.npy"
        if dest.exists():
            continue
        wav, sr = sf.read(str(audio_path(root, row)), dtype="float32", always_2d=False)
        if int(sr) != 16000:
            raise AssertionError(f"sr {sr} for {fid}")
        if wav.ndim != 1:
            wav = np.mean(wav, axis=-1)
        spec = waveform_logmel(torch.from_numpy(np.ascontiguousarray(wav)), mel)
        if int(spec.shape[1]) != 64 or int(spec.shape[0]) < 1:
            raise AssertionError(f"mel shape {tuple(spec.shape)} for {fid}")
        if not bool(torch.isfinite(spec).all()):
            raise AssertionError(f"non-finite mel {fid}")
        np.save(dest, spec.detach().numpy().astype(np.float16))
        made += 1
        if made % 200 == 0:
            print(f"mel {made}/{len(missing)}", flush=True)
    present = [row for row in rows if (cache_dir / f"{int(row['freesound_id'])}.npy").exists()]
    if len(present) != 7405:
        raise SystemExit(f"cache has {len(present)} of 7405")
    for row in kept:
        fid = int(row["freesound_id"])
        if _sha256(cache_dir / f"{fid}.npy") != kept_hash[fid]:
            raise SystemExit(f"existing mel changed for {fid}")
    after_frontend = _frontend_sha(frontend_path)
    if after_frontend != before_frontend:
        raise SystemExit("frontend config was rewritten")
    check_ids = [int(row["freesound_id"]) for row in missing[:4]] + [int(row["freesound_id"]) for row in missing[-4:]]
    max_abs = 0.0
    for fid in check_ids:
        row = next(item for item in rows if int(item["freesound_id"]) == fid)
        wav, sr = sf.read(str(audio_path(root, row)), dtype="float32", always_2d=False)
        if wav.ndim != 1:
            wav = np.mean(wav, axis=-1)
        fresh = waveform_logmel(torch.from_numpy(np.ascontiguousarray(wav)), mel)
        stored = torch.from_numpy(np.load(cache_dir / f"{fid}.npy").astype(np.float32))
        if fresh.shape != stored.shape:
            raise AssertionError(f"recompute shape {tuple(fresh.shape)} != {tuple(stored.shape)} for {fid}")
        gap = float((fresh - stored).abs().max())
        max_abs = max(max_abs, gap)
        if gap > 0.02:
            raise AssertionError(f"recompute gap {gap} for {fid}")
    hours = sum(float(row["duration_s"]) for row in rows) / 3600.0
    report = {
        "task": "loopear_h20_mel_cache",
        "gpu_usd": 0.0,
        "clips": len(rows),
        "hours": hours,
        "id_sha256": digest,
        "mel_files": len(present),
        "mel_written": made,
        "audio_written": written,
        "audio_reused": reused,
        "max_duration_gap_s": max_gap,
        "existing_mel_unchanged": len(kept),
        "recompute_max_abs": max_abs,
        "recompute_ids": check_ids,
        "frontend_sha256": after_frontend,
        "frontend_rewritten": False,
        "norm_refit": False,
        "fsd50k_test_audio_read": False,
        "sealed_realdesed_test_read": False,
    }
    path = write_result_json(Path("results/loopear_h20_cache.json"), report)
    print(json.dumps({"wrote": str(path), "mel_written": made, "sha": digest}))


if __name__ == "__main__":
    main()
