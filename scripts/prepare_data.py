import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from loopear.data import (
    PILOT_CORPUS_SOURCES,
    SEALED_SPLITS,
    prepare_ami_sdm,
    prepare_cremad,
    prepare_dipco,
    prepare_maestro_real,
    prepare_realdesed,
    prepare_starss23,
    prepare_vocalset,
    validate_manifest,
)
from loopear.encoder import (
    HOP_S,
    WIN_S,
    ShardedFeatureCache,
    load_usad2_extractor,
    verify_extractor_invariants,
    write_json_atomic,
)
from loopear.harness import SealedSplitGuard, assert_zero_pretraining_overlap


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
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--manifest-dir", default="data/manifests")
    parser.add_argument("--cache-root", default="cache/stage1_pilot")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--skip-extract", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(int(args.threads))
    assert_zero_pretraining_overlap(PILOT_CORPUS_SOURCES)
    sealed_guard = SealedSplitGuard(
        SEALED_SPLITS, audit_log=Path("ledger") / "sealed_audit.jsonl"
    )

    raw_root = Path(args.raw_root)
    manifest_dir = Path(args.manifest_dir)
    cache_root = Path(args.cache_root)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    t_dl0 = time.perf_counter()
    manifests: dict[str, dict[str, Any]] = {}
    prep_specs = [
        ("starss23", lambda: prepare_starss23(raw_root, manifest_dir, max_train_clips=24, max_val_clips=14)),
        ("dipco", lambda: prepare_dipco(raw_root, manifest_dir, train_min_per_sess=15.0, val_min_per_sess=9.0)),
        ("ami_sdm", lambda: prepare_ami_sdm(raw_root, manifest_dir)),
        ("realdesed", lambda: prepare_realdesed(raw_root, manifest_dir, max_train_clips=120, max_val_clips=60, sealed_guard=sealed_guard)),
        ("crema_d", lambda: prepare_cremad(raw_root, manifest_dir, max_clips_per_actor=24)),
        ("vocalset", lambda: prepare_vocalset(raw_root, manifest_dir, max_clips_per_singer=18)),
        ("maestro_real", lambda: prepare_maestro_real(raw_root, manifest_dir, train_per_scene=2, val_per_scene=1)),
    ]

    dataset_stats: dict[str, Any] = {}
    for name, fn in prep_specs:
        t0 = time.perf_counter()
        man_path = manifest_dir / f"{name}.json"
        if man_path.exists():
            man_cached = json.loads(man_path.read_text(encoding="utf-8"))
            if all(Path(it["audio_path"]).exists() for it in man_cached.get("items", [])):
                man = man_cached
                print(f"[prepare] Reusing validated manifest {name} ...", flush=True)
            else:
                print(f"[prepare] Downloading & validating {name} ...", flush=True)
                man = fn()
        else:
            print(f"[prepare] Downloading & validating {name} ...", flush=True)
            man = fn()
        summary = validate_manifest(man)
        dt = time.perf_counter() - t0
        split_durs: dict[str, float] = {}
        split_counts: dict[str, int] = {}
        for it in man["items"]:
            sp = str(it["split"])
            split_durs[sp] = split_durs.get(sp, 0.0) + float(it["duration_s"])
            split_counts[sp] = split_counts.get(sp, 0) + 1
        dataset_stats[name] = {
            "corpus": name,
            "source": man["source"],
            "license": man["license"],
            "task": man["task"],
            "split_axis": man["split_axis"],
            "join_coverage": summary["join_coverage"],
            "n_items": summary["n_items"],
            "total_hours": summary["total_hours"],
            "split_hours": {k: round(v / 3600.0, 4) for k, v in sorted(split_durs.items())},
            "split_clips": dict(sorted(split_counts.items())),
            "split_groups": summary["splits"],
            "download_wall_s": round(dt, 2),
        }
        manifests[name] = man
        print(
            f"[prepare] {name}: {summary['n_items']} clips, {summary['total_hours']:.3f} h, "
            f"splits={dataset_stats[name]['split_hours']}, join={summary['join_coverage']:.4f} ({dt:.1f}s)",
            flush=True,
        )
    dl_total_s = time.perf_counter() - t_dl0

    if args.skip_extract:
        return 0

    print("[extract] Loading frozen MIT-SLS/USAD2-Small on CPU ...", flush=True)
    enc, _ = load_usad2_extractor("cpu")
    inv = verify_extractor_invariants(enc, "cpu")
    print(f"[extract] Invariants verified: {inv}", flush=True)

    cache = ShardedFeatureCache(cache_root)
    cache_entries: dict[str, dict[str, Any]] = {}
    total_audio_h = 0.0
    total_extract_s = 0.0

    for name, man in manifests.items():
        splits = sorted({str(it["split"]) for it in man["items"]})
        cache_entries[name] = {}
        for split in splits:
            rows = _split_rows(man, split)
            split_h = sum(
                float(it["duration_s"]) for it in man["items"] if it["split"] == split
            ) / 3600.0
            t0 = time.perf_counter()
            out_dir = cache.extract_dataset(
                dataset_name=f"{name}_{split}",
                rows=rows,
                model=enc,
                device="cpu",
                batch_size=int(args.batch_size),
                shard_size=int(args.shard_size),
                stride=8,
                win_s=WIN_S,
                hop_s=HOP_S,
                save_layer_pool=True,
            )
            dt = time.perf_counter() - t0
            idx = cache.read_index(out_dir)
            s_per_h = dt / max(split_h, 1e-6)
            total_audio_h += split_h
            total_extract_s += dt
            cache_entries[name][split] = {
                "cache_dir": str(out_dir),
                "n_windows": len(idx["items"]),
                "audio_hours": round(split_h, 4),
                "extract_wall_s": round(dt, 2),
                "extract_s_per_h": round(s_per_h, 2),
            }
            print(
                f"[extract] {name}/{split}: {len(idx['items'])} windows, "
                f"{split_h:.3f} h in {dt:.1f}s ({s_per_h:.1f} s/h)",
                flush=True,
            )

    summary_payload = {
        "encoder_invariants": inv,
        "download_total_wall_s": round(dl_total_s, 2),
        "extract_total_audio_hours": round(total_audio_h, 4),
        "extract_total_wall_s": round(total_extract_s, 2),
        "extract_throughput_s_per_h": round(
            total_extract_s / max(total_audio_h, 1e-6), 2
        ),
        "datasets": dataset_stats,
        "caches": cache_entries,
    }
    write_json_atomic(manifest_dir / "pilot_cache_index.json", summary_payload)
    print(
        json.dumps(
            {
                "extract_total_audio_hours": summary_payload["extract_total_audio_hours"],
                "extract_total_wall_s": summary_payload["extract_total_wall_s"],
                "extract_throughput_s_per_h": summary_payload["extract_throughput_s_per_h"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
