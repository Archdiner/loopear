import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from loopear.data import PILOT_CORPUS_SOURCES, validate_manifest
from loopear.encoder import (
    HOP_S,
    TAP_LAYERS,
    WIN_S,
    ShardedFeatureCache,
    load_usad2_extractor,
    verify_extractor_invariants,
    write_json_atomic,
)
from loopear.harness import assert_zero_pretraining_overlap


def _split_rows(manifest: dict[str, Any], split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in manifest["items"]:
        if item["split"] != split:
            continue
        meta = {
            k: v
            for k, v in item.items()
            if k not in ("audio_path", "intervals", "speaker_intervals", "soft_labels")
        }
        rows.append(
            {
                "id": item["id"],
                "path": item["audio_path"],
                "channel": item.get("channel"),
                "meta": meta,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", default="data/manifests")
    parser.add_argument("--cache-root", default="cache/stage2_taps")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    torch.set_num_threads(int(args.threads))
    assert_zero_pretraining_overlap(PILOT_CORPUS_SOURCES)
    manifest_dir = Path(args.manifest_dir)
    cache_root = Path(args.cache_root)

    print(
        f"[stage2-taps] Loading frozen MIT-SLS/USAD2-Small (taps={TAP_LAYERS}) ...",
        flush=True,
    )
    enc, _ = load_usad2_extractor("cpu")
    inv = verify_extractor_invariants(enc, "cpu")
    cache = ShardedFeatureCache(cache_root)

    entries: dict[str, dict[str, Any]] = {}
    total_h = 0.0
    total_s = 0.0
    for corpus in PILOT_CORPUS_SOURCES:
        man = json.loads((manifest_dir / f"{corpus}.json").read_text(encoding="utf-8"))
        validate_manifest(man)
        splits = sorted({str(it["split"]) for it in man["items"]})
        entries[corpus] = {}
        for split in splits:
            rows = _split_rows(man, split)
            split_h = sum(
                float(it["duration_s"]) for it in man["items"] if it["split"] == split
            ) / 3600.0
            t0 = time.perf_counter()
            out_dir = cache.extract_tap_dataset(
                dataset_name=f"{corpus}_{split}_taps",
                rows=rows,
                model=enc,
                device="cpu",
                batch_size=int(args.batch_size),
                shard_size=int(args.shard_size),
                tap_layers=TAP_LAYERS,
                win_s=WIN_S,
                hop_s=HOP_S,
            )
            dt = time.perf_counter() - t0
            idx = cache.read_index(out_dir)
            s_per_h = dt / max(split_h, 1e-6)
            total_h += split_h
            total_s += dt
            entries[corpus][split] = {
                "cache_dir": str(out_dir),
                "n_windows": len(idx["items"]),
                "audio_hours": round(split_h, 4),
                "extract_wall_s": round(dt, 2),
                "extract_s_per_h": round(s_per_h, 2),
            }
            print(
                f"[stage2-taps] {corpus}/{split}: {len(idx['items'])} windows, "
                f"{split_h:.3f} h in {dt:.1f}s ({s_per_h:.1f} s/h)",
                flush=True,
            )

    summary = {
        "tap_layers": list(TAP_LAYERS),
        "encoder_invariants": inv,
        "extract_total_audio_hours": round(total_h, 4),
        "extract_total_wall_s": round(total_s, 2),
        "extract_throughput_s_per_h": round(total_s / max(total_h, 1e-6), 2),
        "caches": entries,
    }
    write_json_atomic(manifest_dir / "stage2_tap_cache_index.json", summary)
    print(
        json.dumps(
            {
                "tap_layers": summary["tap_layers"],
                "extract_total_audio_hours": summary["extract_total_audio_hours"],
                "extract_total_wall_s": summary["extract_total_wall_s"],
                "extract_throughput_s_per_h": summary["extract_throughput_s_per_h"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
