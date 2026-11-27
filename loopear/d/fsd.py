import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
from huggingface_hub import HfApi, HfFileSystem, hf_hub_download

from loopear.data import sha256_file
from loopear.d.frontend import (
    RunningNorm,
    build_mel,
    crop_logmel,
    crop_logmel_with_start,
    normalize_logmel,
    waveform_logmel,
    write_frontend_config,
)
from loopear.d.spec import (
    CROP_SECONDS,
    H20_SECONDS,
    H5_SECONDS,
    N_MELS,
    REPACK_REPO,
    REPACK_REVISION,
    UPSTREAM_REPO,
    UPSTREAM_REVISION,
    ZENODO_DOI,
)

META_COLUMNS: tuple[str, ...] = (
    "freesound_id",
    "duration_seconds",
    "uploader",
    "license",
    "source_partition",
    "title",
    "labels",
    "mids",
)
SHARD_COUNTS: dict[str, int] = {"train": 13, "validation": 2, "test": 5}
SPLIT_NAME: dict[str, str] = {"train": "train", "validation": "val", "test": "test"}
GROUND_TRUTH_FILES: tuple[tuple[str, str], ...] = (
    ("dev_csv", "metadata/labels/dev.csv", "labels/dev.csv"),
    ("eval_csv", "metadata/labels/eval.csv", "labels/eval.csv"),
    ("vocabulary_csv", "metadata/labels/vocabulary.csv", "labels/vocabulary.csv"),
    (
        "dev_clips",
        "metadata/original/dev_clips_info_FSD50K.json",
        "metadata/dev_clips_info_FSD50K.json",
    ),
    (
        "eval_clips",
        "metadata/original/eval_clips_info_FSD50K.json",
        "metadata/eval_clips_info_FSD50K.json",
    ),
)


def _sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def shard_paths(split: str) -> list[str]:
    count = SHARD_COUNTS[split]
    return [
        f"data/{split}-{index:05d}-of-{count:05d}.parquet" for index in range(count)
    ]


def _hf_parquet(split_file: str, revision: str) -> str:
    return f"datasets/{REPACK_REPO}@{revision}/{split_file}"


def scan_repack_metadata(revision: str = REPACK_REVISION) -> list[dict[str, Any]]:
    fs = HfFileSystem()
    rows: list[dict[str, Any]] = []
    for hf_split, official_split in SPLIT_NAME.items():
        for shard in shard_paths(hf_split):
            pf = pq.ParquetFile(_hf_parquet(shard, revision), filesystem=fs)
            table = pf.read(columns=list(META_COLUMNS))
            data = table.to_pydict()
            bounds: list[tuple[int, int, int]] = []
            cursor = 0
            for group in range(pf.num_row_groups):
                n_rows = int(pf.metadata.row_group(group).num_rows)
                bounds.append((cursor, cursor + n_rows, group))
                cursor += n_rows
            if cursor != table.num_rows:
                raise AssertionError(f"row group sizes do not cover {shard}")
            group_of = [0] * table.num_rows
            for start, end, group in bounds:
                for offset in range(start, end):
                    group_of[offset] = group
            for index in range(table.num_rows):
                labels = [str(x) for x in data["labels"][index]]
                mids = [str(x) for x in data["mids"][index]]
                rows.append(
                    {
                        "freesound_id": int(data["freesound_id"][index]),
                        "split": official_split,
                        "hf_split": hf_split,
                        "shard": shard,
                        "row_group": int(group_of[index]),
                        "duration_s": float(data["duration_seconds"][index]),
                        "uploader": str(data["uploader"][index]),
                        "license": str(data["license"][index]),
                        "source_partition": str(data["source_partition"][index]),
                        "title": str(data["title"][index]),
                        "labels": labels,
                        "mids": mids,
                    }
                )
    return rows


def _read_ground_truth_csv(path: Path, default_split: str | None) -> dict[int, dict[str, Any]]:
    found: dict[int, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for rec in reader:
            freesound_id = int(rec["fname"])
            labels = [part for part in rec["labels"].split(",") if part]
            mids = [part for part in rec["mids"].split(",") if part]
            split = rec["split"] if "split" in rec and rec["split"] else default_split
            if freesound_id in found:
                raise AssertionError(f"duplicate freesound id in {path}: {freesound_id}")
            found[freesound_id] = {"labels": labels, "mids": mids, "split": split}
    return found


def _file_sha(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def fetch_metadata_pair(revision: str = REPACK_REVISION) -> dict[str, Any]:
    hashes: dict[str, dict[str, str]] = {}
    paths: dict[str, Path] = {}
    for key, repack_file, upstream_file in GROUND_TRUTH_FILES:
        repack_path = Path(
            hf_hub_download(
                REPACK_REPO, repack_file, repo_type="dataset", revision=revision
            )
        )
        upstream_path = Path(
            hf_hub_download(
                UPSTREAM_REPO,
                upstream_file,
                repo_type="dataset",
                revision=UPSTREAM_REVISION,
            )
        )
        repack_sha = _file_sha(repack_path)
        upstream_sha = _file_sha(upstream_path)
        hashes[key] = {
            "repack_sha256": repack_sha,
            "upstream_sha256": upstream_sha,
            "identical": repack_sha == upstream_sha,
        }
        paths[key] = repack_path
    return {"hashes": hashes, "paths": paths}


def verify_rows(
    rows: list[dict[str, Any]], meta_paths: dict[str, Path]
) -> dict[str, Any]:
    dev = _read_ground_truth_csv(meta_paths["dev_csv"], None)
    evl = _read_ground_truth_csv(meta_paths["eval_csv"], "test")
    dev_info = json.loads(meta_paths["dev_clips"].read_text(encoding="utf-8"))
    eval_info = json.loads(meta_paths["eval_clips"].read_text(encoding="utf-8"))
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_split[row["split"]].append(row)
    mismatches: list[str] = []
    seen: set[int] = set()
    for row in rows:
        freesound_id = int(row["freesound_id"])
        if freesound_id in seen:
            mismatches.append(f"duplicate parquet id {freesound_id}")
            continue
        seen.add(freesound_id)
        if row["split"] in ("train", "val"):
            official = dev.get(freesound_id)
            info = dev_info.get(str(freesound_id))
            expect_partition = "dev"
        else:
            official = evl.get(freesound_id)
            info = eval_info.get(str(freesound_id))
            expect_partition = "eval"
        if official is None or info is None:
            mismatches.append(f"missing official metadata for {freesound_id}")
            continue
        if official["split"] != row["split"]:
            mismatches.append(
                f"split {freesound_id}: parquet {row['split']} official {official['split']}"
            )
        if official["labels"] != row["labels"] or official["mids"] != row["mids"]:
            mismatches.append(f"labels {freesound_id}")
        if info.get("uploader") != row["uploader"] or info.get("license") != row["license"]:
            mismatches.append(f"uploader/license {freesound_id}")
        if info.get("title") != row["title"]:
            mismatches.append(f"title {freesound_id}")
        if row["source_partition"] != expect_partition:
            mismatches.append(f"partition {freesound_id}")
        if not (row["duration_s"] > 0.0):
            mismatches.append(f"duration {freesound_id}")
    train_ids = {int(r["freesound_id"]) for r in by_split["train"]}
    val_ids = {int(r["freesound_id"]) for r in by_split["val"]}
    test_ids = {int(r["freesound_id"]) for r in by_split["test"]}
    official_train = {i for i, rec in dev.items() if rec["split"] == "train"}
    official_val = {i for i, rec in dev.items() if rec["split"] == "val"}
    official_test = set(evl)
    if train_ids != official_train:
        mismatches.append("train id set mismatch")
    if val_ids != official_val:
        mismatches.append("val id set mismatch")
    if test_ids != official_test:
        mismatches.append("test id set mismatch")
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        mismatches.append("splits overlap")
    if mismatches:
        preview = mismatches[:8]
        raise AssertionError(f"{len(mismatches)} metadata mismatches, e.g. {preview}")
    return {
        "train": len(train_ids),
        "val": len(val_ids),
        "test": len(test_ids),
        "disjoint": True,
        "parquet_matches_ground_truth": True,
    }


def mark_metadata_duplicates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["split"] != "train":
            row["duplicate_of"] = None
            continue
        millis = int(round(float(row["duration_s"]) * 1000.0))
        grouped[(row["uploader"], row["title"], millis)].append(row)
    removed: list[int] = []
    for members in grouped.values():
        members.sort(key=lambda rec: int(rec["freesound_id"]))
        keeper = int(members[0]["freesound_id"])
        for rec in members:
            if int(rec["freesound_id"]) == keeper:
                rec["duplicate_of"] = None
            else:
                rec["duplicate_of"] = keeper
                removed.append(int(rec["freesound_id"]))
    return {"removed_ids": removed, "n_removed": len(removed)}


def rank_eligible(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["split"] != "train" or row.get("duplicate_of"):
            continue
        labels = row["labels"] or ["__none__"]
        groups[(str(row["uploader"]), sorted(labels)[0])].append(row)
    ranked: list[tuple[int, str, int, int, dict[str, Any]]] = []
    for (uploader, label), members in groups.items():
        del uploader
        members.sort(key=lambda rec: int(rec["freesound_id"]))
        for occurrence, rec in enumerate(members):
            long_enough = 0 if float(rec["duration_s"]) >= CROP_SECONDS else 1
            ranked.append(
                (occurrence, label, long_enough, int(rec["freesound_id"]), rec)
            )
    ranked.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    return [item[4] for item in ranked]


def assign_nested_subsets(ranked: list[dict[str, Any]]) -> dict[str, float]:
    consumed = 0.0
    totals = {"h5": 0.0, "h20": 0.0, "full": 0.0}
    counts = {"h5": 0, "h20": 0, "full": 0}
    for index, row in enumerate(ranked):
        if consumed < H5_SECONDS:
            bucket = "h5"
        elif consumed < H20_SECONDS:
            bucket = "h20"
        else:
            bucket = "full"
        row["subset"] = bucket
        row["rank"] = index
        consumed += float(row["duration_s"])
        totals["full"] += float(row["duration_s"])
        counts["full"] += 1
        if bucket == "h5":
            totals["h5"] += float(row["duration_s"])
            counts["h5"] += 1
        if bucket in ("h5", "h20"):
            totals["h20"] += float(row["duration_s"])
            counts["h20"] += 1
    for row in ranked:
        if "subset" not in row:
            raise AssertionError("ranked row missed a subset")
    return {
        "h5_seconds": totals["h5"],
        "h20_seconds": totals["h20"],
        "full_seconds": totals["full"],
        "h5_clips": counts["h5"],
        "h20_clips": counts["h20"],
        "full_clips": counts["full"],
    }


def _subset_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"clips": 0, "seconds": 0.0}
    durations = [float(r["duration_s"]) for r in rows]
    short = [d for d in durations if d < CROP_SECONDS]
    labels: set[str] = set()
    uploaders: set[str] = set()
    licenses: set[str] = set()
    for row in rows:
        labels.update(row["labels"])
        uploaders.add(str(row["uploader"]))
        licenses.add(str(row["license"]))
    pad_seconds = float(sum(CROP_SECONDS - d for d in short))
    return {
        "clips": len(rows),
        "seconds": float(sum(durations)),
        "hours": float(sum(durations) / 3600.0),
        "uploaders": len(uploaders),
        "labels": len(labels),
        "licenses": sorted(licenses),
        "short_clip_fraction": float(len(short) / len(rows)),
        "pad_seconds_one_crop": pad_seconds,
    }


def build_manifest_payload(
    rows: list[dict[str, Any]],
    verification: dict[str, Any],
    hashes: dict[str, Any],
    duplicates: dict[str, Any],
    zenodo_status: str,
) -> dict[str, Any]:
    ranked = rank_eligible(rows)
    totals = assign_nested_subsets(ranked)
    ranked_ids = {int(r["freesound_id"]) for r in ranked}
    for row in rows:
        if int(row["freesound_id"]) in ranked_ids:
            continue
        row["rank"] = -1
        if row["split"] == "train" and row.get("duplicate_of"):
            row["subset"] = "duplicate"
        else:
            row["subset"] = "heldout"
    h5 = [r for r in ranked if r["subset"] == "h5"]
    h20 = [r for r in ranked if r["subset"] in ("h5", "h20")]
    if not (totals["h5_seconds"] >= H5_SECONDS):
        raise AssertionError("5 hour subset is short")
    if not (totals["h20_seconds"] >= H20_SECONDS):
        raise AssertionError("20 hour subset is short")
    h5_ids = {int(r["freesound_id"]) for r in h5}
    h20_ids = {int(r["freesound_id"]) for r in h20}
    full_ids = {int(r["freesound_id"]) for r in ranked}
    if not h5_ids < h20_ids < full_ids:
        raise AssertionError("subsets are not strictly nested")
    heldout_train = [r for r in rows if r["split"] == "train" and r["subset"] == "heldout"]
    if heldout_train:
        raise AssertionError("eligible train clips were left unranked")
    items = sorted(rows, key=lambda rec: (rec["split"], int(rec["freesound_id"])))
    return {
        "corpus": "fsd50k",
        "repack_repo": REPACK_REPO,
        "repack_revision": REPACK_REVISION,
        "upstream_repo": UPSTREAM_REPO,
        "upstream_revision": UPSTREAM_REVISION,
        "zenodo_doi": ZENODO_DOI,
        "zenodo_status": zenodo_status,
        "metadata_sha256": hashes,
        "verification": verification,
        "duplicates": {"n_removed": duplicates["n_removed"]},
        "subsets": {
            "h5": _subset_stats(h5),
            "h20": _subset_stats(h20),
            "full": _subset_stats(ranked),
        },
        "counts": {
            "rows": len(items),
            "train_eligible": len(ranked),
        },
        "subset_id_sha256": {
            "h5": sha256_of_ids(h5),
            "h20": sha256_of_ids(h20),
            "full": sha256_of_ids(ranked),
        },
        "items": items,
    }


def audio_path(root: Path, row: dict[str, Any]) -> Path:
    if row["split"] == "test":
        raise PermissionError("FSD50K test audio is not materialized")
    return root / row["split"] / f"{int(row['freesound_id'])}.flac"


def extract_audio(
    rows: list[dict[str, Any]],
    root: Path,
    revision: str = REPACK_REVISION,
) -> dict[str, Any]:
    for row in rows:
        if row["split"] == "test":
            raise PermissionError("refusing to read FSD50K test audio")
        if row["split"] not in ("train", "val"):
            raise PermissionError(f"unexpected split {row['split']}")
    wanted: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        wanted[(row["shard"], int(row["row_group"]))].append(row)
    fs = HfFileSystem()
    written = 0
    reused = 0
    max_duration_gap = 0.0
    root.mkdir(parents=True, exist_ok=True)
    open_files: dict[str, Any] = {}
    for (shard, group), members in sorted(wanted.items(), key=lambda item: item[0]):
        if shard not in open_files:
            open_files[shard] = pq.ParquetFile(_hf_parquet(shard, revision), filesystem=fs)
        table = open_files[shard].read_row_group(group, columns=["freesound_id", "audio"])
        blob_of = {
            int(fid): audio
            for fid, audio in zip(table.column("freesound_id").to_pylist(), table.column("audio").to_pylist())
        }
        for row in members:
            dest = audio_path(root, row)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                audio = blob_of[int(row["freesound_id"])]
                blob = audio["bytes"] if isinstance(audio, dict) else audio
                if not isinstance(blob, (bytes, bytearray)):
                    raise TypeError(f"audio payload for {row['freesound_id']} is {type(blob)}")
                dest.write_bytes(blob)
                written += 1
                if written % 200 == 0:
                    print(json.dumps({"extracted_train_clips": written, "reused": reused}), flush=True)
            else:
                reused += 1
            info = sf.info(str(dest))
            if info.samplerate != 16000 or info.channels != 1:
                raise AssertionError(
                    f"{dest} is sr={info.samplerate} ch={info.channels}"
                )
            gap = abs(float(info.duration) - float(row["duration_s"]))
            max_duration_gap = max(max_duration_gap, gap)
            if gap > 0.05:
                raise AssertionError(
                    f"duration gap {gap:.4f}s for freesound {row['freesound_id']}"
                )
    return {
        "written": written,
        "reused": reused,
        "max_duration_gap_s": max_duration_gap,
        "n": len(rows),
    }


def cache_missing_logmels(
    rows: list[dict[str, Any]], audio_root: Path, cache_dir: Path
) -> dict[str, Any]:
    assert_train_rows(rows)
    cache_dir.mkdir(parents=True, exist_ok=True)
    mel = None
    written = 0
    reused = 0
    for row in rows:
        shard = str(row.get("shard", ""))
        if "test" in shard or "validation" in shard:
            raise PermissionError(f"refusing non-train shard {shard}")
        dest = cache_dir / f"{int(row['freesound_id'])}.npy"
        if dest.exists():
            reused += 1
            continue
        if mel is None:
            mel = build_mel()
        wav_path = audio_path(audio_root, row)
        wav, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
        if int(sr) != 16000:
            raise AssertionError(f"unexpected sr {sr}")
        if wav.ndim != 1:
            wav = np.mean(wav, axis=1)
        logmel = waveform_logmel(torch.from_numpy(np.ascontiguousarray(wav)), mel)
        np.save(dest, logmel.detach().numpy().astype(np.float16))
        written += 1
        if written % 500 == 0:
            print(json.dumps({"cached_logmel": written, "reused": reused}), flush=True)
    return {
        "written": written,
        "reused": reused,
        "n": len(rows),
        "frontend_rewritten": False,
    }


def cache_logmels(rows: list[dict[str, Any]], audio_root: Path, cache_dir: Path, config_path: Path) -> dict[str, Any]:
    mel = build_mel()
    stats = RunningNorm()
    cache_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        if row["split"] != "train":
            raise PermissionError("log-mel norm is fit on training audio only")
        dest = audio_path(audio_root, row)
        wav, sr = sf.read(str(dest), dtype="float32", always_2d=False)
        if int(sr) != 16000:
            raise AssertionError(f"unexpected sr {sr}")
        if wav.ndim != 1:
            wav = np.mean(wav, axis=1)
        logmel = waveform_logmel(torch.from_numpy(np.ascontiguousarray(wav)), mel)
        stats.update(logmel.detach().numpy())
        np.save(cache_dir / f"{int(row['freesound_id'])}.npy", logmel.detach().numpy().astype(np.float16))
    mean, std = stats.finalize()
    payload = write_frontend_config(config_path, mean, std)
    return {"frames": stats.count, "config_sha256": payload["config_sha256"]}


def assert_train_rows(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        if row["split"] != "train":
            raise PermissionError(f"pretraining row {row['freesound_id']} split={row['split']}")
        if row.get("duplicate_of"):
            raise PermissionError(f"duplicate recording {row['freesound_id']} is not eligible")


def zenodo_status() -> str:
    import urllib.error
    import urllib.request

    url = "https://zenodo.org/api/records/4060432"
    req = urllib.request.Request(url, headers={"User-Agent": "loopear-data/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            if int(resp.status) == 200:
                return "ok"
            return f"http_{int(resp.status)}"
    except urllib.error.HTTPError as exc:
        return f"http_{int(exc.code)}"
    except Exception as exc:
        return f"error_{type(exc).__name__}"


def probe_revision() -> str:
    info = HfApi().dataset_info(REPACK_REPO)
    if info.sha != REPACK_REVISION:
        raise AssertionError(
            f"repack revision moved: pinned {REPACK_REVISION} remote {info.sha}"
        )
    return str(info.sha)


class MelCache:
    def __init__(
        self,
        rows: list[dict[str, Any]],
        cache_dir: Path,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> None:
        assert_train_rows(rows)
        self.rows = list(rows)
        self.cache_dir = Path(cache_dir)
        self.mean = mean.float().view(1, -1)
        self.std = std.float().view(1, -1)
        if self.mean.shape[-1] != N_MELS:
            raise ValueError("frontend mean has the wrong width")

    def __len__(self) -> int:
        return len(self.rows)

    def _load(self, index: int) -> torch.Tensor:
        freesound_id = int(self.rows[index]["freesound_id"])
        array = np.load(self.cache_dir / f"{freesound_id}.npy")
        return torch.from_numpy(np.ascontiguousarray(array)).float()

    def crops(
        self, indices: list[int], generator: torch.Generator
    ) -> dict[str, torch.Tensor]:
        mel1s = []
        mel2s = []
        valid1s = []
        valid2s = []
        for index in indices:
            raw = self._load(index)
            crop1, valid1 = crop_logmel(raw, generator)
            crop2, valid2 = crop_logmel(raw, generator)
            mel1s.append(normalize_logmel(crop1, self.mean, self.std) * valid1.unsqueeze(-1))
            mel2s.append(normalize_logmel(crop2, self.mean, self.std) * valid2.unsqueeze(-1))
            valid1s.append(valid1)
            valid2s.append(valid2)
        return {
            "mel1": torch.stack(mel1s, dim=0),
            "mel2": torch.stack(mel2s, dim=0),
            "frame_valid1": torch.stack(valid1s, dim=0),
            "frame_valid2": torch.stack(valid2s, dim=0),
            "freesound_id": torch.tensor(
                [int(self.rows[index]["freesound_id"]) for index in indices],
                dtype=torch.long,
            ),
        }

    def crops_with_starts(
        self, indices: list[int], generator: torch.Generator
    ) -> dict[str, torch.Tensor]:
        mel1s = []
        mel2s = []
        valid1s = []
        valid2s = []
        start1s = []
        start2s = []
        for index in indices:
            raw = self._load(index)
            crop1, valid1, start1 = crop_logmel_with_start(raw, generator)
            crop2, valid2, start2 = crop_logmel_with_start(raw, generator)
            mel1s.append(normalize_logmel(crop1, self.mean, self.std) * valid1.unsqueeze(-1))
            mel2s.append(normalize_logmel(crop2, self.mean, self.std) * valid2.unsqueeze(-1))
            valid1s.append(valid1)
            valid2s.append(valid2)
            start1s.append(int(start1))
            start2s.append(int(start2))
        return {
            "mel1": torch.stack(mel1s, dim=0),
            "mel2": torch.stack(mel2s, dim=0),
            "frame_valid1": torch.stack(valid1s, dim=0),
            "frame_valid2": torch.stack(valid2s, dim=0),
            "start1": torch.tensor(start1s, dtype=torch.long),
            "start2": torch.tensor(start2s, dtype=torch.long),
            "freesound_id": torch.tensor(
                [int(self.rows[index]["freesound_id"]) for index in indices],
                dtype=torch.long,
            ),
        }


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def items_for(manifest: dict[str, Any], subset: str) -> list[dict[str, Any]]:
    items = manifest["items"]
    if subset == "h5":
        chosen = [row for row in items if row.get("subset") == "h5"]
    elif subset == "h20":
        chosen = [row for row in items if row.get("subset") in ("h5", "h20")]
    elif subset == "full":
        chosen = [
            row
            for row in items
            if row["split"] == "train" and not row.get("duplicate_of")
        ]
    elif subset == "val":
        chosen = [row for row in items if row["split"] == "val"]
    else:
        raise ValueError(f"unknown subset {subset}")
    chosen.sort(key=lambda row: int(row.get("rank", 10**12)))
    for row in chosen:
        if subset in ("h5", "h20", "full") and row["split"] != "train":
            raise PermissionError("training subset contains a non-train row")
        if subset == "val" and row["split"] != "val":
            raise PermissionError("val subset leaked")
    return chosen


def sha256_of_ids(rows: list[dict[str, Any]]) -> str:
    blob = ",".join(str(int(row["freesound_id"])) for row in rows).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def file_sha256(path: Path) -> str:
    return sha256_file(path)
