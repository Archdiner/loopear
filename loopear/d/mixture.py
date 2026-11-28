from __future__ import annotations

from collections import defaultdict
from typing import Any

CAP_SECONDS = 600.0
TARGET_HOURS = 20.0
REALDESED_SOURCE_HOURS = 3.0
LIMITATION = (
    "Two random crops from a long recording can be unrelated events, so the "
    "recording-level contrastive term is a weaker match for long scenes than it "
    "was for short FSD50K clips."
)

AMI_EVAL_MEETINGS = frozenset(
    {
        "ES2011a",
        "ES2011c",
        "IB4001",
        "IB4002",
        "IB4003",
        "IB4004",
        "IS1008a",
        "IS1008c",
        "TS3004a",
        "TS3004c",
    }
)
DIPCO_EVAL_SESSIONS = frozenset({"S01", "S03", "S06", "S07", "S08"})
DIPCO_TRAIN_SESSIONS = ("S02", "S04", "S05", "S09", "S10")
STARSS_PROBE_ROOMS = frozenset(
    {
        "room12",
        "room13",
        "room14",
        "room21",
        "room22",
        "room4",
        "room6",
        "room7",
        "room9",
        "room10",
        "room15",
        "room16",
        "room2",
        "room23",
        "room24",
        "room8",
    }
)

NATURAL_PREFIXES = ("EN", "IB", "IN")
SCENARIO_PREFIXES = ("ES", "IS", "TS")


def meeting_kind(meeting_id: str) -> str:
    prefix = str(meeting_id)[:2]
    if prefix in NATURAL_PREFIXES:
        return "natural"
    if prefix in SCENARIO_PREFIXES:
        return "scenario"
    return "other"


def cap_seconds(duration_s: float, cap_s: float = CAP_SECONDS) -> float:
    duration = float(duration_s)
    if duration <= 0.0:
        raise ValueError("duration must be positive")
    return min(duration, float(cap_s))


def select_ami_meetings(
    meeting_ids: list[str],
    n_meetings: int,
    excluded: frozenset[str] = AMI_EVAL_MEETINGS,
) -> list[str]:
    natural = sorted(
        meeting
        for meeting in meeting_ids
        if meeting not in excluded and meeting_kind(meeting) == "natural"
    )
    scenario = sorted(
        meeting
        for meeting in meeting_ids
        if meeting not in excluded and meeting_kind(meeting) == "scenario"
    )
    chosen = natural + scenario
    if len(chosen) < int(n_meetings):
        raise ValueError(f"only {len(chosen)} AMI meetings remain after exclusions")
    return chosen[: int(n_meetings)]


def ami_meetings_for_remainder(remainder_s: float, available: list[str]) -> list[str]:
    if remainder_s <= 0.0:
        return []
    n_full = int(remainder_s // CAP_SECONDS)
    if remainder_s - n_full * CAP_SECONDS > 1.0:
        n_full += 1
    return select_ami_meetings(available, n_full)


def round_robin_until(rows: list[dict[str, Any]], key: str, seconds: float) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row[key])].append(row)
    names = sorted(buckets)
    for name in names:
        buckets[name].sort(key=lambda row: str(row.get("id", "")))
    chosen: list[dict[str, Any]] = []
    total = 0.0
    index = 0
    while total < float(seconds) and any(buckets[name] for name in names):
        name = names[index % len(names)]
        index += 1
        if not buckets[name]:
            continue
        row = buckets[name].pop(0)
        chosen.append(row)
        total += float(row["duration_s"])
    if total < float(seconds):
        raise ValueError(f"only {total} seconds available toward {seconds}")
    return chosen


def source_hours(rows: list[dict[str, Any]]) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for row in rows:
        totals[str(row["source"])] += float(row["duration_s"]) / 3600.0
    return dict(sorted(totals.items()))


def assert_mixture_row(row: dict[str, Any]) -> None:
    path = str(row.get("audio_path", "")).replace("\\", "/").lower()
    source = str(row.get("source", ""))
    session = str(row.get("session", ""))
    if row.get("split") != "train":
        raise PermissionError(f"{row.get('id')} split={row.get('split')}")
    if "realdesed" in path and ("/test" in path or "test.zip" in path or "/val" in path):
        raise PermissionError(path)
    if source == "realdesed" and str(row.get("split_name", "train")) != "train":
        raise PermissionError(source)
    if "fsd50k" in path and ("/val" in path or "/test" in path):
        raise PermissionError(path)
    if source == "fsd50k":
        raise PermissionError("FSD50K stays out of the mixture")
    if source == "starss23" or "starss" in path:
        raise PermissionError("STARSS probe rooms stay out")
    if session in STARSS_PROBE_ROOMS:
        raise PermissionError(session)
    if source == "dipco" and session in DIPCO_EVAL_SESSIONS:
        raise PermissionError(session)
    if source == "ami" and session in AMI_EVAL_MEETINGS:
        raise PermissionError(session)
    if float(row["duration_s"]) > CAP_SECONDS + 0.05:
        raise PermissionError(f"{row.get('id')} exceeds the per-recording cap")
    for banned in ("crema", "vocalset", "maestro", "audioset", "ego4d"):
        if banned in path or banned in source:
            raise PermissionError(source or path)


def cache_mixture_logmels(rows: list[dict[str, Any]], cache_dir: Any) -> dict[str, Any]:
    from pathlib import Path

    import numpy as np
    import soundfile as sf
    import torch

    from loopear.d.frontend import build_mel, waveform_logmel
    from loopear.d.fsd import assert_train_rows

    assert_train_rows(rows)
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    mel = None
    written = 0
    reused = 0
    for row in rows:
        assert_mixture_row(row)
        dest = cache / f"{int(row['freesound_id'])}.npy"
        if dest.exists():
            reused += 1
            continue
        wav_path = Path(row["audio_path"])
        info = sf.info(str(wav_path))
        if int(info.samplerate) != 16000 or int(info.channels) != 1:
            raise AssertionError(f"{wav_path} is sr={info.samplerate} ch={info.channels}")
        if abs(float(info.duration) - float(row["duration_s"])) > 0.05:
            raise AssertionError(f"duration gap for {row.get('id')}")
        wav, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
        if int(sr) != 16000 or wav.ndim != 1:
            raise AssertionError(str(wav_path))
        if mel is None:
            mel = build_mel()
        logmel = waveform_logmel(torch.from_numpy(np.ascontiguousarray(wav)), mel)
        np.save(dest, logmel.detach().numpy().astype(np.float16))
        written += 1
    return {
        "written": written,
        "reused": reused,
        "n": len(rows),
        "frontend_rewritten": False,
    }
