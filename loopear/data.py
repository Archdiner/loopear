import csv
import gc
import hashlib
import io
import json
import math
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import HfApi, hf_hub_download

from loopear.encoder import HOP_E, SR, load_mono_audio, write_json_atomic
from loopear.harness import (
    SealedSplitGuard,
    assert_group_disjoint,
    assert_zero_pretraining_overlap,
)

STARSS23_CLASSES: tuple[str, ...] = (
    "female_speech",
    "male_speech",
    "clapping",
    "telephone",
    "laughter",
    "domestic_sounds",
    "walk_footsteps",
    "door_open_close",
    "music",
    "musical_instrument",
    "water_tap",
    "bell",
    "knock",
)

REALDESED_CLASSES: tuple[str, ...] = (
    "bell_ringing",
    "coffee_machine",
    "cutlery_dishes",
    "door_open_close",
    "footsteps",
    "keyboard_typing",
    "keychain",
    "light_switch",
    "microwave",
    "phone_ringing",
    "running_water",
    "toilet_flushing",
    "vacuum_cleaner",
    "wardrobe_drawer_open_close",
    "window_open_close",
)

DIPCO_OVERLAP_CLASSES: tuple[str, ...] = (
    "silence",
    "single_speaker",
    "overlap_2plus",
)

CREMAD_EMOTIONS: tuple[str, ...] = (
    "ANG",
    "DIS",
    "FEA",
    "HAP",
    "NEU",
    "SAD",
)

VOCALSET_TECHNIQUES: tuple[str, ...] = (
    "belt",
    "breathy",
    "fast_forte",
    "inhaled",
    "lip_trill",
    "messa",
    "straight",
    "trillo",
    "vibrato",
    "vocal_fry",
)

MAESTRO_REAL_CLASSES: tuple[str, ...] = (
    "birds_singing",
    "brakes_squeaking",
    "car",
    "children voices",
    "cutlery and dishes",
    "footsteps",
    "large_vehicle",
    "metro approaching",
    "metro leaving",
    "people talking",
    "wind_blowing",
)

SEALED_SPLITS: tuple[str, ...] = ("realdesed_test",)

PILOT_CORPUS_SOURCES: tuple[str, ...] = (
    "starss23",
    "realdesed",
    "dipco",
    "ami_sdm",
    "crema_d",
    "vocalset",
    "maestro_real",
)


def assert_disjoint_groups(
    train_groups: list[str] | set[str],
    eval_groups: list[str] | set[str],
    label: str = "split",
) -> None:
    assert_group_disjoint(train_groups, eval_groups, label=label)


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


class RangeHTTPFile(io.RawIOBase):
    def __init__(
        self,
        url: str,
        timeout: float = 30.0,
        retries: int = 4,
    ):
        self.url = url
        self.timeout = float(timeout)
        self.retries = int(retries)
        self.pos = 0
        req = urllib.request.Request(
            url, method="HEAD", headers={"User-Agent": "loopear-data/0.1"}
        )
        with self._open_with_retry(req) as resp:
            self.length = int(resp.headers["Content-Length"])

    def _open_with_retry(self, req: urllib.request.Request):
        last_exc: Exception | None = None
        for attempt in range(self.retries):
            try:
                return urllib.request.urlopen(req, timeout=self.timeout)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_exc = exc
                time.sleep(min(2.0 ** attempt, 16.0))
        if last_exc is not None:
            raise last_exc
        raise RuntimeError(f"Failed to open {self.url}")

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.length + offset
        else:
            raise ValueError(f"Invalid whence: {whence}")
        self.pos = max(0, min(self.pos, self.length))
        return self.pos

    def tell(self) -> int:
        return self.pos

    def readinto(self, b: bytearray | memoryview) -> int:
        data = self.read(len(b))
        n = len(data)
        b[:n] = data
        return n

    def read(self, size: int = -1) -> bytes:
        if self.pos >= self.length or size == 0:
            return b""
        end = self.length - 1 if size < 0 else min(self.length - 1, self.pos + size - 1)
        req = urllib.request.Request(
            self.url,
            headers={
                "User-Agent": "loopear-data/0.1",
                "Range": f"bytes={self.pos}-{end}",
            },
        )
        with self._open_with_retry(req) as resp:
            data = resp.read()
        self.pos += len(data)
        return data


def open_remote_zip(url: str) -> zipfile.ZipFile:
    raw = RangeHTTPFile(url)
    buffered = io.BufferedReader(raw, buffer_size=262144)
    return zipfile.ZipFile(buffered)


def _extract_zip_member(
    zf: zipfile.ZipFile,
    member: str,
    dest: Path,
) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.tmp")
    data = zf.read(member)
    tmp.write_bytes(data)
    tmp.replace(dest)
    return dest


def _download_zip_members_parallel(
    url: str,
    member_dest_pairs: list[tuple[str, Path]],
    max_workers: int = 4,
) -> None:
    pending = [
        (m, p)
        for m, p in member_dest_pairs
        if not (p.exists() and p.stat().st_size > 0)
    ]
    if not pending:
        return
    n_workers = max(1, min(int(max_workers), len(pending)))
    chunks: list[list[tuple[str, Path]]] = [
        pending[i::n_workers] for i in range(n_workers)
    ]

    def _worker(items: list[tuple[str, Path]]) -> None:
        with open_remote_zip(url) as zf:
            for member, dest in items:
                _extract_zip_member(zf, member, dest)

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futs = [pool.submit(_worker, ch) for ch in chunks if ch]
        for fut in futs:
            fut.result()


def parse_hms_seconds(ts: str) -> float:
    parts = ts.strip().split(":")
    if len(parts) == 3:
        return float(parts[0]) * 3600.0 + float(parts[1]) * 60.0 + float(parts[2])
    if len(parts) == 2:
        return float(parts[0]) * 60.0 + float(parts[1])
    return float(parts[0])


def rasterize_intervals_50hz(
    intervals: list[tuple[float, float, int]],
    start_s: float,
    n_frames: int,
    n_classes: int,
    hop_e: float = HOP_E,
) -> np.ndarray:
    mat = np.zeros((n_frames, n_classes), dtype=np.float32)
    if n_frames <= 0:
        return mat
    centers = start_s + (np.arange(n_frames, dtype=np.float64) + 0.5) * hop_e
    for onset, offset, cls_idx in intervals:
        if cls_idx < 0 or cls_idx >= n_classes:
            continue
        if offset <= start_s or onset >= start_s + n_frames * hop_e:
            continue
        mask = (centers >= onset) & (centers <= offset)
        mat[mask, cls_idx] = 1.0
    return mat


def rasterize_starss_frames_50hz(
    frame_events: list[tuple[int, int]],
    start_s: float,
    n_frames: int,
    n_classes: int = len(STARSS23_CLASSES),
    hop_e: float = HOP_E,
    label_hop_s: float = 0.1,
) -> np.ndarray:
    mat = np.zeros((n_frames, n_classes), dtype=np.float32)
    if n_frames <= 0:
        return mat
    win_end = start_s + n_frames * hop_e
    for frame_idx, cls_idx in frame_events:
        if cls_idx < 0 or cls_idx >= n_classes:
            continue
        t0 = frame_idx * label_hop_s
        t1 = t0 + label_hop_s
        if t1 <= start_s or t0 >= win_end:
            continue
        f0 = max(0, int(math.floor((t0 - start_s) / hop_e)))
        f1 = min(n_frames, int(math.ceil((t1 - start_s) / hop_e)))
        if f1 > f0:
            mat[f0:f1, cls_idx] = 1.0
    return mat


def rasterize_overlap_50hz(
    speaker_intervals: list[tuple[float, float, str]],
    start_s: float,
    n_frames: int,
    hop_e: float = HOP_E,
) -> np.ndarray:
    if n_frames <= 0:
        return np.zeros((0,), dtype=np.int64)
    win_end = start_s + n_frames * hop_e
    by_spk: dict[str, np.ndarray] = {}
    for onset, offset, spk in speaker_intervals:
        if offset <= start_s or onset >= win_end:
            continue
        f0 = max(0, int(math.floor((onset - start_s) / hop_e)))
        f1 = min(n_frames, int(math.ceil((offset - start_s) / hop_e)))
        if f1 <= f0:
            continue
        arr = by_spk.get(spk)
        if arr is None:
            arr = np.zeros((n_frames,), dtype=np.int8)
            by_spk[spk] = arr
        arr[f0:f1] = 1
    if not by_spk:
        return np.zeros((n_frames,), dtype=np.int64)
    total = np.zeros((n_frames,), dtype=np.int64)
    for arr in by_spk.values():
        total += arr.astype(np.int64)
    return np.clip(total, 0, 2)


def rasterize_soft_labels_50hz(
    soft_rows: list[tuple[float, float, int, float]],
    start_s: float,
    n_frames: int,
    n_classes: int,
    hop_e: float = HOP_E,
) -> np.ndarray:
    mat = np.zeros((n_frames, n_classes), dtype=np.float32)
    if n_frames <= 0:
        return mat
    win_end = start_s + n_frames * hop_e
    for onset, offset, cls_idx, prob in soft_rows:
        if cls_idx < 0 or cls_idx >= n_classes:
            continue
        if offset <= start_s or onset >= win_end:
            continue
        f0 = max(0, int(math.floor((onset - start_s) / hop_e)))
        f1 = min(n_frames, int(math.ceil((offset - start_s) / hop_e)))
        if f1 > f0:
            mat[f0:f1, cls_idx] = np.maximum(mat[f0:f1, cls_idx], float(prob))
    return mat


def downsample_binary_targets(
    y_50hz: np.ndarray,
    n_frames: int,
    stride: int,
    t_z_max: int,
) -> np.ndarray:
    n_cls = y_50hz.shape[1]
    out = np.zeros((t_z_max, n_cls), dtype=np.float32)
    n_bins = min(t_z_max, math.ceil(n_frames / stride))
    for i in range(n_bins):
        t0 = i * stride
        t1 = min(t0 + stride, n_frames)
        if t1 > t0:
            out[i] = y_50hz[t0:t1].max(axis=0)
    return out


def downsample_overlap_targets(
    y_50hz: np.ndarray,
    n_frames: int,
    stride: int,
    t_z_max: int,
) -> np.ndarray:
    out = np.zeros((t_z_max,), dtype=np.int64)
    n_bins = min(t_z_max, math.ceil(n_frames / stride))
    for i in range(n_bins):
        t0 = i * stride
        t1 = min(t0 + stride, n_frames)
        if t1 > t0:
            seg = y_50hz[t0:t1]
            counts = np.bincount(seg, minlength=3)
            best = 0
            best_c = -1
            for cls_id in (0, 1, 2):
                if int(counts[cls_id]) >= best_c:
                    best_c = int(counts[cls_id])
                    best = cls_id
            out[i] = best
    return out


def upsample_frame_preds(
    preds_z: np.ndarray,
    stride: int,
    n_frames: int,
) -> np.ndarray:
    rep = np.repeat(preds_z, stride, axis=0)
    return np.ascontiguousarray(rep[:n_frames])


def _extract_onsets(binary_seq: np.ndarray) -> list[int]:
    if binary_seq.size == 0:
        return []
    prev = np.concatenate(([0], binary_seq[:-1]))
    return np.flatnonzero((binary_seq > 0) & (prev == 0)).tolist()


def _match_onsets_greedy(
    gt_onsets: list[int],
    pred_onsets: list[int],
    collar_frames: int,
) -> tuple[int, int, int]:
    if not gt_onsets:
        return 0, len(pred_onsets), 0
    if not pred_onsets:
        return 0, 0, len(gt_onsets)
    used_gt: set[int] = set()
    tp = 0
    for tp_idx in pred_onsets:
        best_j = -1
        best_dist = collar_frames + 1
        for j, tg_idx in enumerate(gt_onsets):
            if j in used_gt:
                continue
            d = abs(tp_idx - tg_idx)
            if d <= collar_frames and d < best_dist:
                best_dist = d
                best_j = j
        if best_j >= 0:
            used_gt.add(best_j)
            tp += 1
    fp = len(pred_onsets) - tp
    fn = len(gt_onsets) - tp
    return tp, fp, fn


def compute_onset_and_boundary_metrics(
    y_50hz_list: list[np.ndarray],
    p_50hz_list: list[np.ndarray],
    hop_e: float = HOP_E,
    thr: float = 0.5,
) -> dict[str, float]:
    collars = {
        "onset_f1_60ms": max(1, int(round(0.06 / hop_e))),
        "onset_f1_100ms": max(1, int(round(0.10 / hop_e))),
        "onset_f1_200ms": max(1, int(round(0.20 / hop_e))),
    }
    counts = {k: [0, 0, 0] for k in collars}
    bnd_60_rad = max(1, int(round(0.06 / hop_e)))
    bnd_100_rad = max(1, int(round(0.10 / hop_e)))
    bnd_60 = [0, 0, 0]
    bnd_100 = [0, 0, 0]
    interior = [0, 0, 0]
    for y_raw, p_raw in zip(y_50hz_list, p_50hz_list):
        t_n = min(y_raw.shape[0], p_raw.shape[0])
        if t_n == 0:
            continue
        y_bin = (y_raw[:t_n] >= 0.5).astype(np.int32)
        p_bin = (p_raw[:t_n] >= thr).astype(np.int32)
        n_cls = y_bin.shape[1]
        for c in range(n_cls):
            gt_on = _extract_onsets(y_bin[:, c])
            pr_on = _extract_onsets(p_bin[:, c])
            for name, col in collars.items():
                tp, fp, fn = _match_onsets_greedy(gt_on, pr_on, col)
                counts[name][0] += tp
                counts[name][1] += fp
                counts[name][2] += fn
            prev = np.concatenate(([0], y_bin[:-1, c]))
            trans = np.flatnonzero(y_bin[:, c] != prev)
            m60 = np.zeros(t_n, dtype=bool)
            m100 = np.zeros(t_n, dtype=bool)
            for idx in trans:
                m60[max(0, idx - bnd_60_rad) : min(t_n, idx + bnd_60_rad + 1)] = True
                m100[max(0, idx - bnd_100_rad) : min(t_n, idx + bnd_100_rad + 1)] = True
            for mask, acc in (
                (m60, bnd_60),
                (m100, bnd_100),
                (~m100, interior),
            ):
                if not np.any(mask):
                    continue
                ym = y_bin[mask, c]
                pm = p_bin[mask, c]
                acc[0] += int(np.sum((ym == 1) & (pm == 1)))
                acc[1] += int(np.sum((ym == 0) & (pm == 1)))
                acc[2] += int(np.sum((ym == 1) & (pm == 0)))
    out: dict[str, float] = {}
    for name, (tp, fp, fn) in counts.items():
        denom = 2 * tp + fp + fn
        out[name] = round(float(2 * tp / denom) if denom > 0 else 0.0, 4)
    for name, (tp, fp, fn) in (
        ("boundary_f1_60ms", bnd_60),
        ("boundary_f1_100ms", bnd_100),
        ("interior_f1", interior),
    ):
        denom = 2 * tp + fp + fn
        out[name] = round(float(2 * tp / denom) if denom > 0 else 0.0, 4)
    return out


def validate_manifest(
    manifest: dict[str, Any],
    min_join_coverage: float = 0.99,
) -> dict[str, Any]:
    corpus = str(manifest["corpus"])
    assert_zero_pretraining_overlap([corpus])
    items = manifest.get("items", [])
    if not items:
        raise ValueError(f"Manifest for {corpus} contains no items")
    join_cov = float(manifest.get("join_coverage", 0.0))
    if join_cov < min_join_coverage:
        raise AssertionError(
            f"Join coverage for {corpus} is {join_cov:.4f} < {min_join_coverage}"
        )
    split_groups: dict[str, set[str]] = defaultdict(set)
    total_dur = 0.0
    for item in items:
        dur = float(item["duration_s"])
        if not math.isfinite(dur) or dur <= 0.0:
            raise ValueError(f"Invalid duration {dur} in {corpus}:{item['id']}")
        total_dur += dur
        grp = str(item.get("group_id", "")).strip()
        if not grp:
            raise ValueError(f"Missing group_id in {corpus}:{item['id']}")
        if not str(item.get("sha256", "")).strip():
            raise ValueError(f"Missing sha256 in {corpus}:{item['id']}")
        split_groups[str(item["split"])].add(grp)
    if "train" in split_groups and "val" in split_groups:
        assert_disjoint_groups(
            split_groups["train"], split_groups["val"], label=f"{corpus}:train_vs_val"
        )
    if "probe_train" in split_groups and "probe_eval" in split_groups:
        assert_disjoint_groups(
            split_groups["probe_train"],
            split_groups["probe_eval"],
            label=f"{corpus}:probe_train_vs_probe_eval",
        )
    return {
        "corpus": corpus,
        "n_items": len(items),
        "total_hours": round(total_dur / 3600.0, 4),
        "join_coverage": round(join_cov, 4),
        "splits": {k: len(v) for k, v in sorted(split_groups.items())},
    }


def save_manifest(manifest_dir: Path | str, manifest: dict[str, Any]) -> Path:
    validate_manifest(manifest)
    out = Path(manifest_dir) / f"{manifest['corpus']}.json"
    write_json_atomic(out, manifest)
    return out


def _select_stratified_clips(
    candidates: list[dict[str, Any]],
    max_clips: int,
    group_key: str,
) -> list[dict[str, Any]]:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sorted(candidates, key=lambda r: str(r["id"])):
        by_group[str(row[group_key])].append(row)
    selected: list[dict[str, Any]] = []
    groups = sorted(by_group.keys())
    idx = 0
    while len(selected) < max_clips:
        progressed = False
        for grp in groups:
            bucket = by_group[grp]
            if idx < len(bucket) and len(selected) < max_clips:
                selected.append(bucket[idx])
                progressed = True
        if not progressed:
            break
        idx += 1
    return selected


def prepare_starss23(
    raw_root: Path | str,
    manifest_dir: Path | str,
    max_train_clips: int = 24,
    max_val_clips: int = 14,
) -> dict[str, Any]:
    assert_zero_pretraining_overlap(["starss23"])
    repo_id = "labhamlet/STARSS23"
    api = HfApi()
    info = api.dataset_info(repo_id, files_metadata=False)
    label_map: dict[str, str] = {}
    for sib in info.siblings or []:
        if sib.rfilename.startswith("labels/") and sib.rfilename.endswith(".csv"):
            stem = Path(sib.rfilename).stem
            label_map[stem] = sib.rfilename
    train_cands: list[dict[str, Any]] = []
    val_cands: list[dict[str, Any]] = []
    for sib in info.siblings or []:
        if not (
            sib.rfilename.startswith("original_audios/")
            and sib.rfilename.endswith(".wav")
        ):
            continue
        stem = Path(sib.rfilename).stem
        if stem not in label_map:
            continue
        parts = stem.split("_")
        fold = parts[0]
        room = parts[1]
        entry = {
            "id": stem,
            "wav_remote": sib.rfilename,
            "csv_remote": label_map[stem],
            "group_id": room,
            "fold": fold,
        }
        if fold == "fold3":
            train_cands.append(entry)
        elif fold == "fold4":
            val_cands.append(entry)
    sel_train = _select_stratified_clips(train_cands, max_train_clips, "group_id")
    sel_val = _select_stratified_clips(val_cands, max_val_clips, "group_id")
    out_dir = Path(raw_root) / "starss23"
    out_dir.mkdir(parents=True, exist_ok=True)
    items: list[dict[str, Any]] = []
    matched = 0
    total_selected = len(sel_train) + len(sel_val)
    for split, subset in (("train", sel_train), ("val", sel_val)):
        for rec in subset:
            wav_src = hf_hub_download(
                repo_id, rec["wav_remote"], repo_type="dataset"
            )
            csv_src = hf_hub_download(
                repo_id, rec["csv_remote"], repo_type="dataset"
            )
            wav_dest = out_dir / f"{rec['id']}_ch0_16k.wav"
            csv_dest = out_dir / f"{rec['id']}.csv"
            if not wav_dest.exists():
                mono = load_mono_audio(wav_src, target_sr=SR, channel=0)
                sf.write(str(wav_dest), mono, SR, subtype="PCM_16")
            if not csv_dest.exists():
                csv_dest.write_bytes(Path(csv_src).read_bytes())
            wav_info = sf.info(str(wav_dest))
            dur_s = float(wav_info.frames) / float(wav_info.samplerate)
            events: list[list[int]] = []
            for line in csv_dest.read_text(encoding="utf-8").splitlines():
                cols = [c.strip() for c in line.split(",")]
                if len(cols) >= 2 and cols[0].isdigit():
                    events.append([int(cols[0]), int(cols[1])])
            matched += 1
            items.append(
                {
                    "id": rec["id"],
                    "split": split,
                    "group_id": rec["group_id"],
                    "audio_path": str(wav_dest),
                    "label_path": str(csv_dest),
                    "duration_s": round(dur_s, 4),
                    "sr": SR,
                    "channel": 0,
                    "sha256": sha256_file(wav_dest),
                    "n_label_rows": len(events),
                }
            )
    manifest = {
        "corpus": "starss23",
        "source": "Zenodo 7880637 / HF labhamlet/STARSS23",
        "license": "MIT",
        "task": "polyphonic_speech_events_13cls",
        "classes": list(STARSS23_CLASSES),
        "split_axis": "room_disjoint (fold3 train / fold4 val)",
        "join_coverage": float(matched) / max(float(total_selected), 1.0),
        "items": items,
    }
    save_manifest(manifest_dir, manifest)
    return manifest


def prepare_realdesed(
    raw_root: Path | str,
    manifest_dir: Path | str,
    max_train_clips: int = 120,
    max_val_clips: int = 60,
    sealed_guard: SealedSplitGuard | None = None,
) -> dict[str, Any]:
    assert_zero_pretraining_overlap(["realdesed"])
    guard = sealed_guard or SealedSplitGuard(SEALED_SPLITS)
    guard.check_access("realdesed_train")
    guard.check_access("realdesed_val")
    urls = {
        "train": "https://zenodo.org/api/records/20056072/files/train.zip/content",
        "val": "https://zenodo.org/api/records/20056072/files/validation.zip/content",
    }
    limits = {"train": int(max_train_clips), "val": int(max_val_clips)}
    cls_to_idx = {c: i for i, c in enumerate(REALDESED_CLASSES)}
    out_dir = Path(raw_root) / "realdesed"
    out_dir.mkdir(parents=True, exist_ok=True)
    items: list[dict[str, Any]] = []
    matched = 0
    attempted = 0
    for split in ("train", "val"):
        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        with open_remote_zip(urls[split]) as zf:
            meta_txt = zf.read("metadata.csv").decode("utf-8")
            ann_txt = zf.read("annotations.csv").decode("utf-8")
            ann_by_file: dict[str, list[list[Any]]] = defaultdict(list)
            for row in csv.DictReader(io.StringIO(ann_txt)):
                cname = row["class"].strip()
                if cname in cls_to_idx:
                    ann_by_file[row["filename"].strip()].append(
                        [
                            round(float(row["onset"]), 4),
                            round(float(row["offset"]), 4),
                            cls_to_idx[cname],
                        ]
                    )
            zip_set = set(zf.namelist())
            cands: list[dict[str, Any]] = []
            for row in csv.DictReader(io.StringIO(meta_txt)):
                fn = row["filename"].strip()
                member = f"audio/{fn}"
                if member not in zip_set or fn not in ann_by_file:
                    continue
                primary_cls = (
                    row["target_classes"].split(";")[0].strip()
                    if row.get("target_classes")
                    else "other"
                )
                cands.append(
                    {
                        "id": Path(fn).stem,
                        "filename": fn,
                        "member": member,
                        "primary_cls": primary_cls,
                        "env": row.get("recording_environment", "").strip(),
                        "license": row.get("license", "").strip(),
                    }
                )
            selected = _select_stratified_clips(cands, limits[split], "primary_cls")
            attempted += len(selected)
            dl_pairs = [
                (rec["member"], split_dir / rec["filename"])
                for rec in selected
                if not (split_dir / f"{rec['id']}_16k.wav").exists()
            ]
            _download_zip_members_parallel(urls[split], dl_pairs, max_workers=4)
            for rec in selected:
                raw_wav = split_dir / rec["filename"]
                wav_16k = split_dir / f"{rec['id']}_16k.wav"
                if not wav_16k.exists():
                    mono = load_mono_audio(raw_wav, target_sr=SR)
                    sf.write(str(wav_16k), mono, SR, subtype="PCM_16")
                    if raw_wav.exists():
                        raw_wav.unlink()
                wav_info = sf.info(str(wav_16k))
                dur_s = float(wav_info.frames) / float(wav_info.samplerate)
                intervals = ann_by_file[rec["filename"]]
                matched += 1
                items.append(
                    {
                        "id": f"rd_{split}_{rec['id']}",
                        "split": split,
                        "group_id": f"{split}_{rec['id']}",
                        "audio_path": str(wav_16k),
                        "duration_s": round(dur_s, 4),
                        "sr": SR,
                        "sha256": sha256_file(wav_16k),
                        "intervals": intervals,
                        "recording_environment": rec["env"],
                        "clip_license": rec["license"],
                    }
                )
    manifest = {
        "corpus": "realdesed",
        "source": "Zenodo 20056072 (train.zip & validation.zip only; test.zip sealed)",
        "license": "CC0 / CC-BY-4.0",
        "task": "domestic_polyphonic_events_15cls",
        "classes": list(REALDESED_CLASSES),
        "split_axis": "collector_disjoint (official train vs validation; test sealed)",
        "join_coverage": float(matched) / max(float(attempted), 1.0),
        "items": items,
    }
    save_manifest(manifest_dir, manifest)
    return manifest


def prepare_dipco(
    raw_root: Path | str,
    manifest_dir: Path | str,
    train_min_per_sess: float = 15.0,
    val_min_per_sess: float = 9.0,
) -> dict[str, Any]:
    assert_zero_pretraining_overlap(["dipco"])
    repo_id = "huckiyang/DiPCo"
    api = HfApi()
    info = api.dataset_info(repo_id, files_metadata=False)
    array_wavs: dict[tuple[str, str], list[str]] = defaultdict(list)
    for sib in info.siblings or []:
        if sib.rfilename.startswith("audio/") and sib.rfilename.endswith(".wav"):
            parts = sib.rfilename.split("/")
            if len(parts) == 3 and "_U" in parts[2]:
                sp_name = parts[1]
                sess = parts[2].split("_")[0]
                array_wavs[(sp_name, sess)].append(sib.rfilename)
    sessions = {
        "train": ("dev", ("S02", "S04", "S05", "S09", "S10"), float(train_min_per_sess) * 60.0),
        "val": ("eval", ("S01", "S03", "S06", "S07", "S08"), float(val_min_per_sess) * 60.0),
    }
    out_dir = Path(raw_root) / "dipco"
    out_dir.mkdir(parents=True, exist_ok=True)
    items: list[dict[str, Any]] = []
    split_speakers: dict[str, set[str]] = {"train": set(), "val": set()}
    matched = 0
    attempted = 0
    for split, (hf_split, sess_list, max_sec) in sessions.items():
        for sess in sess_list:
            attempted += 1
            cands = sorted(array_wavs.get((hf_split, sess), []))
            if not cands:
                continue
            wav_remote = cands[0]
            array_id = Path(wav_remote).name.split("_")[1].split(".")[0]
            json_remote = f"transcriptions/{hf_split}/{sess}.json"
            wav_src = hf_hub_download(repo_id, wav_remote, repo_type="dataset")
            json_src = hf_hub_download(repo_id, json_remote, repo_type="dataset")
            wav_dest = out_dir / f"{sess}_{array_id}_16k.wav"
            if not wav_dest.exists():
                wav_arr = load_mono_audio(
                    wav_src, target_sr=SR, start_s=0.0, end_s=max_sec
                )
                sf.write(str(wav_dest), wav_arr, SR, subtype="PCM_16")
            wav_info = sf.info(str(wav_dest))
            dur_s = float(wav_info.frames) / float(wav_info.samplerate)
            raw_utts = json.loads(Path(json_src).read_text(encoding="utf-8"))
            spk_intervals: list[list[Any]] = []
            sess_spks: set[str] = set()
            for u in raw_utts:
                st_map = u.get("start_time", {})
                en_map = u.get("end_time", {})
                st_str = st_map.get(array_id) or st_map.get("close-talk")
                en_str = en_map.get(array_id) or en_map.get("close-talk")
                if not st_str or not en_str:
                    continue
                t0 = parse_hms_seconds(st_str)
                t1 = parse_hms_seconds(en_str)
                if t1 <= 0.0 or t0 >= dur_s or t1 <= t0:
                    continue
                spk = str(u["speaker_id"])
                sess_spks.add(spk)
                split_speakers[split].add(spk)
                spk_intervals.append(
                    [round(max(0.0, t0), 4), round(min(dur_s, t1), 4), spk]
                )
            matched += 1
            items.append(
                {
                    "id": f"dipco_{sess}_{array_id}",
                    "split": split,
                    "group_id": sess,
                    "speakers": sorted(sess_spks),
                    "audio_path": str(wav_dest),
                    "duration_s": round(dur_s, 4),
                    "sr": SR,
                    "sha256": sha256_file(wav_dest),
                    "speaker_intervals": spk_intervals,
                }
            )
    assert_disjoint_groups(
        split_speakers["train"], split_speakers["val"], label="dipco:speakers"
    )
    manifest = {
        "corpus": "dipco",
        "source": "Zenodo 8122551 / HF huckiyang/DiPCo",
        "license": "CDLA-Permissive-1.0",
        "task": "farfield_overlapped_speech_3cls",
        "classes": list(DIPCO_OVERLAP_CLASSES),
        "split_axis": "session_and_speaker_disjoint (dev train / eval val)",
        "join_coverage": float(matched) / max(float(attempted), 1.0),
        "items": items,
    }
    save_manifest(manifest_dir, manifest)
    return manifest


def prepare_ami_sdm(
    raw_root: Path | str,
    manifest_dir: Path | str,
    train_shards: tuple[int, ...] = (0, 3, 6, 9, 12, 15, 18, 21, 24),
    val_shards: tuple[int, ...] = (0, 1, 2, 3, 4),
    utts_per_train_spk: int = 12,
    utts_per_val_spk: int = 16,
    min_dur_s: float = 2.0,
    max_dur_s: float = 8.0,
) -> dict[str, Any]:
    assert_zero_pretraining_overlap(["ami_sdm"])
    repo_id = "edinburghcstr/ami"
    out_dir = Path(raw_root) / "ami_sdm"
    out_dir.mkdir(parents=True, exist_ok=True)

    def _collect_split(
        split_name: str,
        hf_split: str,
        n_total: int,
        shard_ids: tuple[int, ...],
        utts_per_spk: int,
        forbidden_spks: set[str],
        require_multi_meeting: bool,
    ) -> list[dict[str, Any]]:
        spk_to_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        spk_meetings: dict[str, set[str]] = defaultdict(set)
        for sid in shard_ids:
            remote = f"sdm/{hf_split}-{sid:05d}-of-{n_total:05d}.parquet"
            local_pq = hf_hub_download(repo_id, remote, repo_type="dataset")
            tbl = pq.read_table(
                local_pq,
                columns=["meeting_id", "audio_id", "speaker_id", "begin_time", "end_time", "audio"],
            )
            mtgs = tbl["meeting_id"].to_pylist()
            aids = tbl["audio_id"].to_pylist()
            spks = tbl["speaker_id"].to_pylist()
            begs = tbl["begin_time"].to_pylist()
            ends = tbl["end_time"].to_pylist()
            auds = tbl["audio"].to_pylist()
            for mtg, aid, spk, b, e, aud in zip(mtgs, aids, spks, begs, ends, auds):
                spk_s = str(spk)
                if spk_s in forbidden_spks:
                    continue
                dur = float(e) - float(b)
                if dur < min_dur_s or dur > max_dur_s:
                    continue
                if len(spk_to_rows[spk_s]) >= utts_per_spk * 2:
                    continue
                if aud is None or not aud.get("bytes"):
                    continue
                spk_meetings[spk_s].add(str(mtg))
                spk_to_rows[spk_s].append(
                    {
                        "id": str(aid),
                        "meeting_id": str(mtg),
                        "speaker_id": spk_s,
                        "bytes": aud["bytes"],
                    }
                )
            del tbl, mtgs, aids, spks, begs, ends, auds
            gc.collect()
            pa.default_memory_pool().release_unused()
        split_dir = out_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        out_items: list[dict[str, Any]] = []
        for spk_s in sorted(spk_to_rows.keys()):
            if require_multi_meeting and len(spk_meetings[spk_s]) < 2:
                continue
            rows = spk_to_rows[spk_s]
            if len(rows) < 4:
                continue
            picked = _select_stratified_clips(rows, utts_per_spk, "meeting_id")
            if require_multi_meeting and len({r["meeting_id"] for r in picked}) < 2:
                continue
            for r in picked:
                wav_dest = split_dir / f"{r['id']}.wav"
                if not wav_dest.exists():
                    arr, sr_in = sf.read(io.BytesIO(r["bytes"]), dtype="float32", always_2d=True)
                    mono = arr.mean(axis=1)
                    if int(sr_in) != SR:
                        tmp_raw = split_dir / f".{r['id']}_raw.wav"
                        sf.write(str(tmp_raw), mono, int(sr_in), subtype="PCM_16")
                        mono = load_mono_audio(tmp_raw, target_sr=SR)
                        tmp_raw.unlink()
                    sf.write(str(wav_dest), mono, SR, subtype="PCM_16")
                info_w = sf.info(str(wav_dest))
                dur_s = float(info_w.frames) / float(info_w.samplerate)
                out_items.append(
                    {
                        "id": r["id"],
                        "split": split_name,
                        "group_id": spk_s,
                        "speaker_id": spk_s,
                        "meeting_id": r["meeting_id"],
                        "audio_path": str(wav_dest),
                        "duration_s": round(dur_s, 4),
                        "sr": SR,
                        "sha256": sha256_file(wav_dest),
                    }
                )
        return out_items

    val_items = _collect_split(
        "val",
        "validation",
        5,
        val_shards,
        utts_per_val_spk,
        forbidden_spks=set(),
        require_multi_meeting=True,
    )
    val_spks = {it["speaker_id"] for it in val_items}
    val_mtgs = {it["meeting_id"] for it in val_items}
    train_items = _collect_split(
        "train",
        "train",
        27,
        train_shards,
        utts_per_train_spk,
        forbidden_spks=val_spks,
        require_multi_meeting=False,
    )
    train_items = [it for it in train_items if it["meeting_id"] not in val_mtgs]
    train_spks = sorted({it["speaker_id"] for it in train_items})
    spk_to_idx = {s: i for i, s in enumerate(train_spks)}
    for it in train_items:
        it["speaker_idx"] = spk_to_idx[it["speaker_id"]]
    for it in val_items:
        it["speaker_idx"] = -1
    assert_disjoint_groups(
        {it["speaker_id"] for it in train_items},
        {it["speaker_id"] for it in val_items},
        label="ami_sdm:speakers",
    )
    assert_disjoint_groups(
        {it["meeting_id"] for it in train_items},
        {it["meeting_id"] for it in val_items},
        label="ami_sdm:meetings",
    )
    items = train_items + val_items
    manifest = {
        "corpus": "ami_sdm",
        "source": "HF edinburghcstr/ami (sdm)",
        "license": "CC-BY-4.0",
        "task": "farfield_meeting_speaker_id",
        "n_train_speakers": len(train_spks),
        "n_val_speakers": len(val_spks),
        "split_axis": "speaker_and_meeting_disjoint",
        "join_coverage": 1.0,
        "items": items,
    }
    save_manifest(manifest_dir, manifest)
    return manifest


def prepare_cremad(
    raw_root: Path | str,
    manifest_dir: Path | str,
    max_clips_per_actor: int = 24,
) -> dict[str, Any]:
    assert_zero_pretraining_overlap(["crema_d"])
    repo_id = "myleslinder/crema-d"
    tar_path = hf_hub_download(repo_id, "data/crema_d.tar.gz", repo_type="dataset")
    out_dir = Path(raw_root) / "crema_d"
    out_dir.mkdir(parents=True, exist_ok=True)
    emo_to_idx = {e: i for i, e in enumerate(CREMAD_EMOTIONS)}
    by_actor: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with tarfile.open(tar_path, "r:gz") as tf:
        members = [m for m in tf.getmembers() if m.isfile() and m.name.endswith(".wav")]
        for m in sorted(members, key=lambda x: x.name):
            stem = Path(m.name).stem
            parts = stem.split("_")
            if len(parts) < 4:
                continue
            actor_id, sentence_id, emo = parts[0], parts[1], parts[2]
            if emo not in emo_to_idx:
                continue
            by_actor[actor_id].append(
                {
                    "id": stem,
                    "actor_id": actor_id,
                    "sentence_id": sentence_id,
                    "emotion": emo,
                    "emotion_idx": emo_to_idx[emo],
                    "member": m,
                }
            )
        actors = sorted(by_actor.keys())
        n_probe_train = int(round(len(actors) * 0.67))
        train_actors = set(actors[:n_probe_train])
        selected_all: list[tuple[str, str, dict[str, Any]]] = []
        for actor in actors:
            split = "probe_train" if actor in train_actors else "probe_eval"
            selected = _select_stratified_clips(
                by_actor[actor], max_clips_per_actor, "emotion"
            )
            for rec in selected:
                selected_all.append((actor, split, rec))
        selected_all.sort(key=lambda item: int(item[2]["member"].offset))
        items: list[dict[str, Any]] = []
        matched = 0
        attempted = 0
        for actor, split, rec in selected_all:
            attempted += 1
            wav_dest = out_dir / f"{rec['id']}.wav"
            if not wav_dest.exists():
                extracted = tf.extractfile(rec["member"])
                if extracted is None:
                    continue
                raw_bytes = extracted.read()
                arr, sr_in = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=True)
                if arr.shape[0] < int(0.1 * sr_in):
                    continue
                mono = arr.mean(axis=1)
                if int(sr_in) != SR:
                    tmp_raw = out_dir / f".{rec['id']}_raw.wav"
                    sf.write(str(tmp_raw), mono, int(sr_in), subtype="PCM_16")
                    mono = load_mono_audio(tmp_raw, target_sr=SR)
                    tmp_raw.unlink()
                sf.write(str(wav_dest), mono, SR, subtype="PCM_16")
            wav_info = sf.info(str(wav_dest))
            dur_s = float(wav_info.frames) / float(wav_info.samplerate)
            if dur_s < 0.1:
                continue
            matched += 1
            items.append(
                {
                    "id": rec["id"],
                    "split": split,
                    "group_id": actor,
                    "speaker_id": actor,
                    "sentence_id": rec["sentence_id"],
                    "emotion": rec["emotion"],
                    "emotion_idx": rec["emotion_idx"],
                    "audio_path": str(wav_dest),
                    "duration_s": round(dur_s, 4),
                    "sr": SR,
                    "sha256": sha256_file(wav_dest),
                }
            )
    manifest = {
        "corpus": "crema_d",
        "source": "HF myleslinder/crema-d (CheyneyComputerScience/CREMA-D)",
        "license": "ODbL-1.0 / DbCL-1.0",
        "task": "transfer_emotion_6cls_and_unseen_speaker_eer",
        "classes": list(CREMAD_EMOTIONS),
        "split_axis": "actor_disjoint (61 probe_train / 30 probe_eval)",
        "join_coverage": float(matched) / max(float(attempted), 1.0),
        "items": items,
    }
    save_manifest(manifest_dir, manifest)
    return manifest


def prepare_vocalset(
    raw_root: Path | str,
    manifest_dir: Path | str,
    max_clips_per_singer: int = 20,
) -> dict[str, Any]:
    assert_zero_pretraining_overlap(["vocalset"])
    url = "https://zenodo.org/api/records/1442513/files/VocalSet11.zip/content"
    tech_to_idx = {t: i for i, t in enumerate(VOCALSET_TECHNIQUES)}
    out_dir = Path(raw_root) / "vocalset"
    out_dir.mkdir(parents=True, exist_ok=True)
    items: list[dict[str, Any]] = []
    matched = 0
    attempted = 0
    with open_remote_zip(url) as zf:
        by_singer: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for name in sorted(zf.namelist()):
            if not name.endswith(".wav") or name.startswith("__MACOSX"):
                continue
            parts = name.split("/")
            if len(parts) < 5 or parts[0] != "FULL":
                continue
            singer, tech = parts[1], parts[3]
            if tech not in tech_to_idx:
                continue
            stem = Path(name).stem
            by_singer[singer].append(
                {
                    "id": f"{singer}_{tech}_{stem}",
                    "singer": singer,
                    "technique": tech,
                    "technique_idx": tech_to_idx[tech],
                    "member": name,
                }
            )
        singers = sorted(by_singer.keys())
        n_train = int(round(len(singers) * 0.70))
        train_singers = set(singers[:n_train])
        selected_all: list[tuple[str, str, dict[str, Any]]] = []
        for singer in singers:
            split = "probe_train" if singer in train_singers else "probe_eval"
            selected = _select_stratified_clips(
                by_singer[singer], max_clips_per_singer, "technique"
            )
            for rec in selected:
                selected_all.append((singer, split, rec))
    dl_pairs = [
        (rec["member"], out_dir / f".{rec['id']}_raw.wav")
        for _, _, rec in selected_all
        if not (out_dir / f"{rec['id']}_16k.wav").exists()
    ]
    _download_zip_members_parallel(url, dl_pairs, max_workers=4)
    for singer, split, rec in selected_all:
        attempted += 1
        wav_dest = out_dir / f"{rec['id']}_16k.wav"
        raw_wav = out_dir / f".{rec['id']}_raw.wav"
        if not wav_dest.exists():
            mono = load_mono_audio(raw_wav, target_sr=SR)
            sf.write(str(wav_dest), mono, SR, subtype="PCM_16")
            if raw_wav.exists():
                raw_wav.unlink()
        wav_info = sf.info(str(wav_dest))
        dur_s = float(wav_info.frames) / float(wav_info.samplerate)
        if dur_s < 0.2:
            continue
        matched += 1
        items.append(
            {
                "id": rec["id"],
                "split": split,
                "group_id": singer,
                "singer": singer,
                "technique": rec["technique"],
                "technique_idx": rec["technique_idx"],
                "audio_path": str(wav_dest),
                "duration_s": round(dur_s, 4),
                "sr": SR,
                "sha256": sha256_file(wav_dest),
            }
        )
    manifest = {
        "corpus": "vocalset",
        "source": "Zenodo 1442513 (VocalSet11.zip)",
        "license": "CC-BY-4.0",
        "task": "transfer_vocal_technique_10cls",
        "classes": list(VOCALSET_TECHNIQUES),
        "split_axis": "singer_disjoint (14 probe_train / 6 probe_eval)",
        "join_coverage": float(matched) / max(float(attempted), 1.0),
        "items": items,
    }
    save_manifest(manifest_dir, manifest)
    return manifest


def prepare_maestro_real(
    raw_root: Path | str,
    manifest_dir: Path | str,
    train_per_scene: int = 2,
    val_per_scene: int = 1,
    max_sec_per_recording: float = 180.0,
) -> dict[str, Any]:
    assert_zero_pretraining_overlap(["maestro_real"])
    annot_url = "https://zenodo.org/api/records/7244360/files/development_annotation.zip/content"
    audio_url = "https://zenodo.org/api/records/7244360/files/development_audio.zip/content"
    cls_to_idx = {c: i for i, c in enumerate(MAESTRO_REAL_CLASSES)}
    out_dir = Path(raw_root) / "maestro_real"
    out_dir.mkdir(parents=True, exist_ok=True)
    soft_by_stem: dict[str, list[list[Any]]] = {}
    with open_remote_zip(annot_url) as zf_ann:
        for name in sorted(zf_ann.namelist()):
            if not name.endswith(".txt"):
                continue
            stem = Path(name).stem
            rows: list[list[Any]] = []
            for line in zf_ann.read(name).decode("utf-8").splitlines():
                parts = line.strip().split("\t")
                if len(parts) < 4:
                    continue
                cname = parts[2].strip()
                if cname not in cls_to_idx:
                    continue
                t0 = float(parts[0])
                t1 = float(parts[1])
                prob = float(parts[3])
                if t0 < max_sec_per_recording and prob > 0.0:
                    rows.append(
                        [
                            round(t0, 4),
                            round(min(t1, max_sec_per_recording), 4),
                            cls_to_idx[cname],
                            round(prob, 4),
                        ]
                    )
            soft_by_stem[stem] = rows
    items: list[dict[str, Any]] = []
    matched = 0
    attempted = 0
    selected_mr: list[tuple[str, str, str, str]] = []
    with open_remote_zip(audio_url) as zf_aud:
        by_scene: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for name in sorted(zf_aud.namelist()):
            if not name.endswith(".wav"):
                continue
            parts = name.split("/")
            if len(parts) < 3:
                continue
            scene = parts[1]
            stem = Path(name).stem
            if stem in soft_by_stem:
                by_scene[scene].append((stem, name))
        for scene in sorted(by_scene.keys()):
            recs = by_scene[scene][: train_per_scene + val_per_scene]
            for idx, (stem, member) in enumerate(recs):
                split = "probe_train" if idx < train_per_scene else "probe_eval"
                selected_mr.append((scene, split, stem, member))
    dl_pairs = [
        (member, out_dir / f".{stem}_raw.wav")
        for _, _, stem, member in selected_mr
        if not (out_dir / f"{stem}_16k.wav").exists()
    ]
    _download_zip_members_parallel(audio_url, dl_pairs, max_workers=4)
    for scene, split, stem, _ in selected_mr:
        attempted += 1
        wav_dest = out_dir / f"{stem}_16k.wav"
        raw_wav = out_dir / f".{stem}_raw.wav"
        if not wav_dest.exists():
            mono = load_mono_audio(
                raw_wav, target_sr=SR, start_s=0.0, end_s=max_sec_per_recording
            )
            sf.write(str(wav_dest), mono, SR, subtype="PCM_16")
            if raw_wav.exists():
                raw_wav.unlink()
        wav_info = sf.info(str(wav_dest))
        dur_s = float(wav_info.frames) / float(wav_info.samplerate)
        matched += 1
        items.append(
            {
                "id": stem,
                "split": split,
                "group_id": stem,
                "scene": scene,
                "audio_path": str(wav_dest),
                "duration_s": round(dur_s, 4),
                "sr": SR,
                "sha256": sha256_file(wav_dest),
                "soft_labels": soft_by_stem[stem],
            }
        )
    manifest = {
        "corpus": "maestro_real",
        "source": "Zenodo 7244360 (development_audio & development_annotation)",
        "license": "Non-Commercial Academic (Tampere University)",
        "task": "transfer_continuous_scene_soft_events_11cls",
        "classes": list(MAESTRO_REAL_CLASSES),
        "split_axis": "recording_disjoint (2 probe_train / 1 probe_eval per scene)",
        "join_coverage": float(matched) / max(float(attempted), 1.0),
        "items": items,
    }
    save_manifest(manifest_dir, manifest)
    return manifest
