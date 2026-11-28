from __future__ import annotations

import io
import json
import subprocess
import time
import zlib
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

from loopear.d.mixture import (
    AMI_EVAL_MEETINGS,
    CAP_SECONDS,
    DIPCO_EVAL_SESSIONS,
    LIMITATION,
    DIPCO_TRAIN_SESSIONS,
    REALDESED_SOURCE_HOURS,
    STARSS_PROBE_ROOMS,
    TARGET_HOURS,
    ami_meetings_for_remainder,
    assert_mixture_row,
    cap_seconds,
    meeting_kind,
    source_hours,
)
from loopear.d.spec import CROP_SECONDS, SR
from loopear.encoder import load_mono_audio

ROOT = Path("data/raw/loopear_pltn")
INCOMING = ROOT / "incoming"
KEPT = ROOT / "kept"
DEMAND_NAMES = (
    "DLIVING_16k",
    "DKITCHEN_16k",
    "DWASHING_16k",
    "NFIELD_16k",
    "NPARK_16k",
    "NRIVER_16k",
    "OHALLWAY_16k",
    "OMEETING_16k",
    "OOFFICE_16k",
    "PCAFETER_16k",
    "PRESTO_16k",
    "PSTATION_16k",
    "SPSQUARE_16k",
    "STRAFFIC_16k",
    "TBUS_16k",
    "TCAR_16k",
    "TMETRO_16k",
)
USER_AGENT = "loopear/0.1"


def _curl(url: str, dest: Path, range_end: int | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "curl",
        "-fsSL",
        "-L",
        "--fail",
        "--retry",
        "4",
        "--retry-delay",
        "2",
        "-A",
        USER_AGENT,
        "-o",
        str(dest),
        url,
    ]
    if range_end is not None:
        cmd[1:1] = ["-r", f"0-{int(range_end)}"]
    subprocess.run(cmd, check=True)


def _ami_ids() -> list[str]:
    from huggingface_hub import HfApi

    info = HfApi().dataset_info("ggfox00000/dia-AMICorpus-all", files_metadata=False)
    ids = sorted(
        {
            sib.rfilename.split("/")[1]
            for sib in info.siblings or []
            if sib.rfilename.endswith(".Mix-Headset.wav")
        }
    )
    if len(ids) < 100:
        raise RuntimeError(f"AMI mirror returned {len(ids)} meetings")
    return ids


def _patch_wav_header(path: Path) -> None:
    data = path.read_bytes()
    if data[0:4] != b"RIFF" or data[36:40] != b"data":
        raise RuntimeError(f"{path} is not a plain PCM wav")
    payload = len(data) - 44
    patched = bytearray(data)
    patched[4:8] = (len(data) - 8).to_bytes(4, "little")
    patched[40:44] = payload.to_bytes(4, "little")
    path.write_bytes(patched)


def _write_excerpt(wav: np.ndarray, dest: Path) -> float:
    dest.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dest), np.ascontiguousarray(wav, dtype=np.float32), SR, subtype="PCM_16")
    info = sf.info(str(dest))
    if info.samplerate != SR or info.channels != 1:
        raise RuntimeError(f"{dest} was not written as 16 kHz mono")
    return float(info.duration)


def _row(
    source: str,
    session: str,
    audio_path: Path,
    duration_s: float,
    native_sr: int,
    license_name: str,
    experience: str,
    original_s: float,
    note: str,
) -> dict:
    kept = cap_seconds(duration_s)
    return {
        "id": f"{source}_{session}",
        "source": source,
        "session": session,
        "split": "train",
        "split_name": "train",
        "hf_split": "train",
        "shard": f"mixture/{source}",
        "duplicate_of": None,
        "audio_path": str(audio_path),
        "duration_s": kept,
        "original_duration_s": float(original_s),
        "native_sr": int(native_sr),
        "license": license_name,
        "experience": experience,
        "note": note,
        "cap_s": CAP_SECONDS,
    }


def _one_ami(index: int, meeting: str, take_s: float) -> dict:
    dest = KEPT / "ami" / f"{meeting}.wav"
    url = (
        "https://huggingface.co/datasets/ggfox00000/dia-AMICorpus-all/resolve/main/"
        f"amicorpus/{meeting}/audio/{meeting}.Mix-Headset.wav"
    )
    header_path = INCOMING / f"{meeting}.hdr"
    if not header_path.exists() or header_path.stat().st_size < 44:
        _curl(url, header_path, 43)
    header = header_path.read_bytes()[:44]
    data_bytes = int.from_bytes(header[40:44], "little")
    original_s = data_bytes / float(SR * 2)
    nbytes = 44 + int(min(data_bytes, round(take_s * SR * 2)))
    if not dest.exists():
        tmp = INCOMING / f"{meeting}.Mix-Headset.part.wav"
        _curl(url, tmp, max(nbytes - 1, 43))
        _patch_wav_header(tmp)
        wav = load_mono_audio(tmp, target_sr=SR, end_s=take_s)
        duration = _write_excerpt(wav, dest)
        tmp.unlink(missing_ok=True)
    else:
        duration = float(sf.info(str(dest)).duration)
    if (index + 1) % 10 == 0:
        print(json.dumps({"ami_kept": index + 1}), flush=True)
    return _row(
        "ami",
        meeting,
        dest,
        duration,
        SR,
        "CC-BY-4.0",
        "conversation",
        original_s,
        "AMI Mix-Headset from the CC-BY-4.0 Hugging Face mirror, close-talk mix of a meeting. "
        + ("Natural meeting." if meeting_kind(meeting) == "natural" else "Elicited scenario meeting."),
    )


def fetch_ami(meetings: list[str], tail_s: float) -> list[dict]:
    takes = [
        CAP_SECONDS if index < len(meetings) - 1 else min(CAP_SECONDS, tail_s)
        for index in range(len(meetings))
    ]
    with ThreadPoolExecutor(max_workers=6) as pool:
        rows = list(pool.map(_one_ami, range(len(meetings)), meetings, takes))
    return rows


def fetch_dipco() -> tuple[list[dict], dict[str, int]]:
    from huggingface_hub import hf_hub_download

    repo = "huckiyang/DiPCo"
    rows = []
    native: dict[str, int] = {}
    for session in DIPCO_TRAIN_SESSIONS:
        if session in DIPCO_EVAL_SESSIONS:
            raise PermissionError(session)
        remote = f"audio/dev/{session}_U01.CH1.wav"
        try:
            src = Path(hf_hub_download(repo, remote, repo_type="dataset"))
        except Exception:
            src = _dipco_fallback(repo, session)
        info = sf.info(str(src))
        native[session] = int(info.samplerate)
        dest = KEPT / "dipco" / f"{session}.wav"
        if not dest.exists():
            wav = load_mono_audio(src, target_sr=SR, channel=0, end_s=CAP_SECONDS)
            duration = _write_excerpt(wav, dest)
        else:
            duration = float(sf.info(str(dest)).duration)
        rows.append(
            _row(
                "dipco",
                session,
                dest,
                duration,
                native[session],
                "CDLA-Permissive-1.0",
                "overlapping_conversation",
                float(info.duration),
                "DiPCo dev session, one distant-array channel. Eval sessions were not read.",
            )
        )
    return rows, native


def _dipco_fallback(repo: str, session: str) -> Path:
    from huggingface_hub import HfApi, hf_hub_download

    info = HfApi().dataset_info(repo, files_metadata=False)
    names = sorted(
        sib.rfilename
        for sib in info.siblings or []
        if sib.rfilename.startswith(f"audio/dev/{session}_U") and sib.rfilename.endswith(".CH1.wav")
    )
    if not names:
        raise FileNotFoundError(session)
    return Path(hf_hub_download(repo, names[0], repo_type="dataset"))


def _demand_source_info(zpath: Path) -> tuple[int, float, bytes, str]:
    with zipfile.ZipFile(zpath) as zf:
        wavs = sorted(name for name in zf.namelist() if name.lower().endswith(".wav"))
        if not wavs:
            raise RuntimeError(f"{zpath} has no wav")
        chosen = next(
            (name for name in wavs if "ch01" in name.lower() or "channel01" in name.lower()),
            wavs[0],
        )
        blob = zf.read(chosen)
    tmp = zpath.with_suffix(".ch01.wav")
    tmp.write_bytes(blob)
    info = sf.info(str(tmp))
    native = int(info.samplerate)
    original = float(info.duration)
    tmp.unlink()
    return native, original, blob, chosen


def _one_demand(name: str) -> dict:
    zpath = INCOMING / f"{name}.zip"
    if not zipfile.is_zipfile(zpath):
        if zpath.exists():
            zpath.unlink()
        _curl(
            f"https://zenodo.org/records/1227121/files/{name}.zip?download=1",
            zpath,
        )
    dest = KEPT / "demand" / f"{name}.wav"
    dest.parent.mkdir(parents=True, exist_ok=True)
    native, original, blob, _chosen = _demand_source_info(zpath)
    if not dest.exists():
        tmp = dest.with_suffix(".src.wav")
        tmp.write_bytes(blob)
        wav = load_mono_audio(tmp, target_sr=SR, channel=0, end_s=CAP_SECONDS)
        duration = _write_excerpt(wav, dest)
        tmp.unlink()
    else:
        duration = float(sf.info(str(dest)).duration)
    scene = name.replace("_16k", "").lower()
    return _row(
        "demand",
        scene,
        dest,
        duration,
        native,
        "CC-BY-4.0",
        "environmental_ambience",
        original,
        "DEMAND 16 kHz release, channel 1 of one environment.",
    )


def fetch_demand() -> list[dict]:
    with ThreadPoolExecutor(max_workers=3) as pool:
        return list(pool.map(_one_demand, DEMAND_NAMES))


def _license_ok(text: str) -> bool:
    lowered = text.lower().replace(" ", "")
    return "cc0" in lowered or "cc-by" in lowered or "ccby" in lowered


REALDESED_URL = "https://zenodo.org/records/20056072/files/train.zip?download=1"
REALDESED_SIZE = 5647895421


def _span_bytes(url: str, start: int, end: int) -> bytes:
    dest = INCOMING / f"span_{start}_{end}.part"
    cmd = [
        "curl",
        "-fsSL",
        "--retry",
        "5",
        "--retry-delay",
        "3",
        "--retry-all-errors",
        "-A",
        USER_AGENT,
        "-r",
        f"{int(start)}-{int(end)}",
        "-o",
        str(dest),
        url,
    ]
    last = None
    for attempt in range(6):
        last = subprocess.run(cmd)
        if last.returncode == 0 and dest.exists() and dest.stat().st_size > 0:
            blob = dest.read_bytes()
            dest.unlink()
            return blob
        time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(f"range {start}-{end} failed: {last.returncode if last else 'none'}")


def _zip64_value(extra: bytes, local: int, comp: int, uncomp: int) -> int:
    if local != 0xFFFFFFFF:
        return local
    pos = 0
    while pos + 4 <= len(extra):
        hid = int.from_bytes(extra[pos : pos + 2], "little")
        hsz = int.from_bytes(extra[pos + 2 : pos + 4], "little")
        body = extra[pos + 4 : pos + 4 + hsz]
        pos += 4 + hsz
        if hid != 1:
            continue
        cursor = 0
        if uncomp == 0xFFFFFFFF:
            cursor += 8
        if comp == 0xFFFFFFFF:
            cursor += 8
        return int.from_bytes(body[cursor : cursor + 8], "little")
    raise RuntimeError("zip64 local offset missing")


def _realdesed_catalog() -> list[dict]:
    tail = _span_bytes(REALDESED_URL, REALDESED_SIZE - 262144, REALDESED_SIZE - 1)
    loc = tail.rfind(b"PK\x06\x07")
    z64_off = int.from_bytes(tail[loc + 8 : loc + 16], "little")
    rel = z64_off - (REALDESED_SIZE - len(tail))
    cd_size = int.from_bytes(tail[rel + 40 : rel + 48], "little")
    cd_off = int.from_bytes(tail[rel + 48 : rel + 56], "little")
    cd_rel = cd_off - (REALDESED_SIZE - len(tail))
    cd = tail[cd_rel : cd_rel + cd_size]
    entries = []
    index = 0
    while index + 46 <= len(cd) and cd[index : index + 4] == b"PK\x01\x02":
        method = int.from_bytes(cd[index + 10 : index + 12], "little")
        comp = int.from_bytes(cd[index + 20 : index + 24], "little")
        uncomp = int.from_bytes(cd[index + 24 : index + 28], "little")
        nlen = int.from_bytes(cd[index + 28 : index + 30], "little")
        elen = int.from_bytes(cd[index + 30 : index + 32], "little")
        clen = int.from_bytes(cd[index + 32 : index + 34], "little")
        local = int.from_bytes(cd[index + 42 : index + 46], "little")
        name = cd[index + 46 : index + 46 + nlen].decode("utf-8", "replace")
        extra = cd[index + 46 + nlen : index + 46 + nlen + elen]
        entries.append(
            {
                "name": name,
                "method": method,
                "comp": comp,
                "uncomp": uncomp,
                "local": _zip64_value(extra, local, comp, uncomp),
            }
        )
        index += 46 + nlen + elen + clen
    return entries


def _read_zip_member(entry: dict) -> bytes:
    header = _span_bytes(REALDESED_URL, entry["local"], entry["local"] + 29)
    nlen = int.from_bytes(header[26:28], "little")
    elen = int.from_bytes(header[28:30], "little")
    start = entry["local"] + 30 + nlen + elen
    blob = _span_bytes(REALDESED_URL, start, start + int(entry["comp"]) - 1)
    if entry["method"] == 0:
        return blob
    if entry["method"] == 8:
        return zlib.decompress(blob, -15)
    raise RuntimeError(f"zip method {entry['method']}")


def fetch_realdesed() -> list[dict]:
    import csv

    catalog = {item["name"]: item for item in _realdesed_catalog()}
    meta = _read_zip_member(catalog["metadata.csv"]).decode("utf-8")
    queued: dict[str, list[dict]] = {}
    for rec in csv.DictReader(io.StringIO(meta)):
        filename = rec["filename"].strip()
        member = f"audio/{filename}"
        if member not in catalog:
            continue
        license_name = rec.get("license", "").strip()
        if not _license_ok(license_name):
            continue
        environment = rec.get("recording_environment", "").strip() or "unknown"
        queued.setdefault(environment, []).append(
            {
                "id": Path(filename).stem,
                "member": member,
                "environment": environment,
                "license": license_name,
            }
        )
    names = sorted(queued)
    for name in names:
        queued[name].sort(key=lambda row: row["id"])
    rows = []
    total = 0.0
    target = REALDESED_SOURCE_HOURS * 3600.0
    cursor = 0
    while total < target and any(queued[name] for name in names):
        environment = names[cursor % len(names)]
        cursor += 1
        if not queued[environment]:
            continue
        rec = queued[environment].pop(0)
        raw = _read_zip_member(catalog[rec["member"]])
        tmp = INCOMING / f"rd_{rec['id']}.wav"
        tmp.write_bytes(raw)
        info = sf.info(str(tmp))
        if float(info.duration) < CROP_SECONDS:
            tmp.unlink()
            continue
        dest = KEPT / "realdesed" / f"{rec['id']}.wav"
        wav = load_mono_audio(tmp, target_sr=SR, end_s=CAP_SECONDS)
        duration = _write_excerpt(wav, dest)
        tmp.unlink()
        rows.append(
            _row(
                "realdesed",
                str(rec["id"]),
                dest,
                duration,
                int(info.samplerate),
                rec["license"],
                "short_domestic_scene",
                float(info.duration),
                "RealDESED train clip, a short real-home recording. Validation and test were not opened.",
            )
        )
        total += duration
        if len(rows) % 25 == 0:
            print(json.dumps({"realdesed_kept_s": total, "clips": len(rows)}), flush=True)
    if total < target:
        raise RuntimeError(f"RealDESED train only yielded {total} seconds")
    return rows


def exclusions() -> dict:
    return {
        "cap_s": CAP_SECONDS,
        "cap_rule": "Each kept recording is at most 600 seconds. A longer file contributes only that prefix.",
        "sampling_rate": (
            "Kept files are 16 kHz mono PCM. AMI Mix-Headset is already 16 kHz mono, so the prefix is cut in samples. The Edinburgh mirror returned 503, so these prefixes came from the public Hugging Face copy of the same files. "
            "Other files use loopear.encoder.load_mono_audio, which reads one channel and resamples with "
            "torchaudio.functional.resample when the native rate is not 16000."
        ),
        "kept_out": [
            {
                "source": "FSD50K",
                "reason": "Isolated event clips, and the 20.003 h control corpus. Validation and test stay out. None of it fills this mixture.",
            },
            {
                "source": "STARSS23",
                "reason": "Every room in the public development audio appears in the current probe, so the whole session list stays out.",
                "rooms": sorted(STARSS_PROBE_ROOMS),
            },
            {
                "source": "RealDESED validation and test",
                "reason": "Validation is the official eval split. Test stays sealed. Those zips were not opened.",
            },
            {
                "source": "DiPCo eval",
                "reason": "Sessions used as the eval split in this repository stay out, with the whole session.",
                "sessions": sorted(DIPCO_EVAL_SESSIONS),
            },
            {
                "source": "AMI validation meetings",
                "reason": "Meetings in the AMI SDM validation split stay out, with the whole meeting.",
                "meetings": sorted(AMI_EVAL_MEETINGS),
            },
            {
                "source": "AudioSet",
                "reason": "YouTube terms. The audio was not downloaded.",
            },
            {
                "source": "EPIC-SOUNDS",
                "reason": "Annotations are CC-BY-NC, but the continuous kitchen audio is an email request or a full video download. It was not obtained on this machine.",
            },
            {
                "source": "Ego4D",
                "reason": "The license requires an agreement. No credentials are on this machine, so it was skipped.",
            },
            {
                "source": "DCASE bioacoustics",
                "reason": "Animal recordings, including an ultrasonic subset, are not the everyday mixture asked for. Skipped.",
            },
            {
                "source": "MUSAN",
                "reason": "The noise subset is mostly short effects, the speech subset is read speech and hearings, and the music subset is not this mixture. Skipped so short clips do not fill it.",
            },
            {
                "source": "CREMA-D",
                "reason": "Acted emotion evaluation audio stays out of pretraining.",
            },
            {
                "source": "VocalSet",
                "reason": "Sung-voice evaluation audio stays out of pretraining.",
            },
            {
                "source": "MAESTRO",
                "reason": "The piano and MAESTRO Real evaluation audio stays out of pretraining.",
            },
            {
                "source": "CHiME",
                "reason": "Not obtained. No open file URL was used, so it was skipped rather than accepted through a registration form.",
            },
        ],
    }


def main() -> None:
    INCOMING.mkdir(parents=True, exist_ok=True)
    KEPT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        dipco_job = pool.submit(fetch_dipco)
        demand_job = pool.submit(fetch_demand)
        dipco_rows, dipco_sr = dipco_job.result()
        demand_rows = demand_job.result()
    real_rows = fetch_realdesed()
    used = sum(float(row["duration_s"]) for row in dipco_rows + demand_rows + real_rows)
    remainder = TARGET_HOURS * 3600.0 - used
    meetings = ami_meetings_for_remainder(remainder, _ami_ids())
    full = int(remainder // CAP_SECONDS)
    tail = remainder - full * CAP_SECONDS
    if tail <= 1.0:
        tail = CAP_SECONDS
        meetings = meetings[:full]
    ami_rows = fetch_ami(meetings, tail if tail > 1.0 else CAP_SECONDS)
    rows = ami_rows + dipco_rows + demand_rows + real_rows
    for index, row in enumerate(rows):
        row["freesound_id"] = 900000 + index
        row["cache_id"] = row["freesound_id"]
        assert_mixture_row(row)
    hours = source_hours(rows)
    unique = sum(hours.values())
    if not 19.5 <= unique <= 20.5:
        raise RuntimeError(f"mixture hours {unique} are outside 19.5 to 20.5")
    manifest = {
        "corpus": "loopear_pltn_mixture",
        "recipe": "P-LT",
        "run_name": "P-LT-N",
        "cap_s": CAP_SECONDS,
        "target_hours": TARGET_HOURS,
        "hours_by_source": hours,
        "unique_hours": unique,
        "clips": len(rows),
        "native_sr_by_dipco_session": dipco_sr,
        "sampling_rate": exclusions()["sampling_rate"],
        "limitation": LIMITATION,
        "items": rows,
    }
    out = Path("data/manifests/loopear_pltn_mixture.json")
    exc = Path("data/manifests/loopear_pltn_exclusions.json")
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    exc.write_text(json.dumps(exclusions(), indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": str(out), "hours": hours, "unique_hours": manifest["unique_hours"], "clips": len(rows)}), flush=True)


if __name__ == "__main__":
    main()
