import argparse
import json
from pathlib import Path

from loopear.d.fsd import (
    audio_path,
    build_manifest_payload,
    cache_logmels,
    extract_audio,
    fetch_metadata_pair,
    items_for,
    mark_metadata_duplicates,
    probe_revision,
    rank_eligible,
    scan_repack_metadata,
    verify_rows,
    zenodo_status,
)
from loopear.harness import write_result_json


def _val_sample(rows: list[dict], n: int) -> list[dict]:
    copies = []
    for row in rows:
        if row["split"] != "val":
            continue
        copied = dict(row)
        copied["split"] = "train"
        copied["duplicate_of"] = None
        copies.append(copied)
    ranked = rank_eligible(copies)
    chosen = {int(row["freesound_id"]) for row in ranked[:n]}
    return [row for row in rows if row["split"] == "val" and int(row["freesound_id"]) in chosen]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifests/fsd50k_loopear_d.json")
    parser.add_argument("--audio-root", default="data/raw/fsd50k")
    parser.add_argument("--cache-dir", default="cache/loopear_d_mel")
    parser.add_argument("--frontend", default="data/manifests/loopear_d_frontend.json")
    parser.add_argument("--val-clips", type=int, default=160)
    parser.add_argument("--report", default="results/loopear_d_data.json")
    args = parser.parse_args()
    revision = probe_revision()
    status = zenodo_status()
    pair = fetch_metadata_pair(revision)
    if not all(item["identical"] for item in pair["hashes"].values()):
        raise SystemExit("repack metadata is not byte-identical to the pinned upstream copy")
    rows = scan_repack_metadata(revision)
    verification = verify_rows(rows, pair["paths"])
    duplicates = mark_metadata_duplicates(rows)
    payload = build_manifest_payload(rows, verification, pair["hashes"], duplicates, status)
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    h5 = items_for(payload, "h5")
    audio_root = Path(args.audio_root)
    train_extract = extract_audio(h5, audio_root, revision)
    for row in h5:
        if not audio_path(audio_root, row).exists():
            raise SystemExit(f"missing audio for {row['freesound_id']}")
    cache_info = cache_logmels(h5, audio_root, Path(args.cache_dir), Path(args.frontend))
    val_rows = _val_sample(payload["items"], args.val_clips)
    val_extract = extract_audio(val_rows, audio_root, revision)
    report = {
        "task": "loopear_d_fsd50k_prepare",
        "revision": revision,
        "zenodo_status": status,
        "verification": verification,
        "duplicates_removed": duplicates["n_removed"],
        "subsets": payload["subsets"],
        "subset_id_sha256": payload["subset_id_sha256"],
        "train_extract": train_extract,
        "val_extract": val_extract,
        "val_ids": [int(row["freesound_id"]) for row in val_rows],
        "frontend": cache_info,
        "audioset_waveforms_in_repo": 0,
        "test_audio_read": False,
        "gpu_usd": 0.0,
    }
    write_result_json(args.report, report)
    print(json.dumps({k: report[k] for k in ("verification", "subsets", "train_extract", "zenodo_status")}, indent=2))


if __name__ == "__main__":
    main()
