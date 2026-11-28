import io
import math
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio

from loopear.d.causal import CausalLoopEarDEncoder, l2_normalize, teacher_target, visible_from_frames
from loopear.d.causal_probe import _event_module
from loopear.d.frontend import normalize_logmel, waveform_logmel
from loopear.d.spec import CROP_SAMPLES, GRID_T, HOP_SAMPLES, N_FRAMES, PATCH_FRAMES, SR, WIN_SAMPLES


BACKGROUND_CLASSES = ("rain", "vacuum_cleaner", "washing_machine", "pouring_water")
FOREGROUND_CLASSES = ("door_wood_knock", "glass_breaking", "clock_tick", "can_opening")
STRESS_CLASSES = BACKGROUND_CLASSES + FOREGROUND_CLASSES
SNRS = (-5.0, 0.0, 5.0)
N_SOURCE_PAIRS = 40
N_CLIPS = N_SOURCE_PAIRS * len(SNRS)
ONSET_LOW_S = 0.45
ONSET_HIGH_S = 1.20
PROTO_CLIPS = 16
PEAK = 0.99


def _cell_slots() -> list[tuple[str, str, int]]:
    cells = [(bg, fg) for bg in BACKGROUND_CLASSES for fg in FOREGROUND_CLASSES]
    counts = [2] * len(cells)
    extra = N_SOURCE_PAIRS - 2 * len(cells)
    if extra < 0:
        raise RuntimeError("the stress grid does not divide into source pairs")
    for index in range(extra):
        counts[index] += 1
    if sum(counts) != N_SOURCE_PAIRS:
        raise RuntimeError("source-pair counts do not add to 40")
    return [(bg, fg, count) for (bg, fg), count in zip(cells, counts)]


def patch_support(patch: int) -> tuple[int, int]:
    start = int(patch) * PATCH_FRAMES * HOP_SAMPLES
    end = start + (PATCH_FRAMES - 1) * HOP_SAMPLES + WIN_SAMPLES
    return start, end


def onset_patch(sample: int) -> int:
    frame = int(sample) // HOP_SAMPLES
    patch = frame // PATCH_FRAMES
    if patch < 0 or patch >= GRID_T:
        raise ValueError("onset falls outside the crop grid")
    return int(patch)


def pre_onset_patch(sample: int) -> int:
    support = (PATCH_FRAMES - 1) * HOP_SAMPLES + WIN_SAMPLES
    width = PATCH_FRAMES * HOP_SAMPLES
    latest = (int(sample) - support) // width
    if latest < 0:
        raise ValueError("onset has no patch that ends before it")
    return int(latest)


def plan_stress_clips(seed: int = 42) -> list[dict[str, Any]]:
    generator = torch.Generator().manual_seed(int(seed))
    low = int(ONSET_LOW_S * SR)
    high = int(ONSET_HIGH_S * SR)
    clips: list[dict[str, Any]] = []
    for background, foreground, n_slots in _cell_slots():
        for slot in range(n_slots):
            onset = int(torch.randint(low, high, (1,), generator=generator))
            pre_onset_patch(onset)
            onset_patch(onset)
            for snr in SNRS:
                clips.append(
                    {
                        "background": background,
                        "foreground": foreground,
                        "pair_slot": int(slot),
                        "snr_db": float(snr),
                        "onset_sample": int(onset),
                        "onset_s": int(onset) / float(SR),
                    }
                )
    if len(clips) != N_CLIPS:
        raise RuntimeError(f"stress plan has {len(clips)} clips")
    return clips


def detect_onset(wave: torch.Tensor, relative: float = 0.2) -> int:
    flat = wave.reshape(-1).float()
    if int(flat.shape[0]) == 0:
        raise ValueError("empty waveform")
    peak = float(flat.abs().max())
    if peak <= 0.0 or relative <= 0.0:
        return 0
    hits = flat.abs() >= (float(relative) * peak)
    return int(torch.argmax(hits.to(torch.int64)))


def fit_background(wave: torch.Tensor) -> torch.Tensor:
    flat = wave.reshape(-1).float()
    if int(flat.shape[0]) == 0:
        raise ValueError("empty background")
    if int(flat.shape[0]) >= CROP_SAMPLES:
        return flat[:CROP_SAMPLES].contiguous()
    reps = (CROP_SAMPLES + int(flat.shape[0]) - 1) // int(flat.shape[0])
    return flat.repeat(reps)[:CROP_SAMPLES].contiguous()


def foreground_crop(wave: torch.Tensor) -> tuple[torch.Tensor, int]:
    flat = wave.reshape(-1).float()
    src = detect_onset(flat)
    lead = int(0.50 * SR)
    start = max(0, src - lead)
    if int(flat.shape[0]) >= CROP_SAMPLES and start + CROP_SAMPLES > int(flat.shape[0]):
        start = int(flat.shape[0]) - CROP_SAMPLES
    crop = flat[start : start + CROP_SAMPLES]
    if int(crop.shape[0]) < CROP_SAMPLES:
        crop = torch.nn.functional.pad(crop, (0, CROP_SAMPLES - int(crop.shape[0])))
    onset = int(src - start)
    if onset < 0 or onset >= CROP_SAMPLES:
        raise ValueError("clean transient onset falls outside its crop")
    return crop.contiguous(), onset


def mix_transient(
    background: torch.Tensor,
    foreground: torch.Tensor,
    onset_sample: int,
    snr_db: float,
) -> tuple[torch.Tensor, float]:
    bg = fit_background(background)
    fg = foreground.reshape(-1).float()
    src = detect_onset(fg)
    excerpt = fg[src:]
    onset = int(onset_sample)
    if onset < 0 or onset >= CROP_SAMPLES:
        raise ValueError("onset is outside the crop")
    excerpt = excerpt[: CROP_SAMPLES - onset]
    if int(excerpt.shape[0]) == 0:
        raise ValueError("transient does not fit in the crop")
    segment = bg[onset : onset + int(excerpt.shape[0])]
    power_bg = segment.pow(2).mean().clamp_min(1e-8)
    power_fg = excerpt.pow(2).mean().clamp_min(1e-8)
    ratio = bg.new_tensor(10.0).pow(float(snr_db) / 10.0)
    scale = torch.sqrt(power_bg * ratio / power_fg)
    added = excerpt * scale
    measured = 10.0 * math.log10(float(added.pow(2).mean() / power_bg))
    mixed = bg.clone()
    mixed[onset : onset + int(excerpt.shape[0])] = segment + added
    peak = float(mixed.abs().max())
    if peak > PEAK:
        mixed = mixed * (PEAK / peak)
    return mixed, measured


def assign_sources(
    plan: list[dict[str, Any]],
    examples: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    grouped: dict[str, list[dict[str, Any]]] = {name: [] for name in STRESS_CLASSES}
    for row in examples:
        category = str(row["category"])
        if category in grouped:
            grouped[category].append(row)
    for name, rows in grouped.items():
        rows.sort(key=lambda rec: str(rec["filename"]))
        if not rows:
            raise RuntimeError(f"ESC-50 train has no {name}")
    used = {name: 0 for name in STRESS_CLASSES}
    chosen: dict[tuple[str, str, int], tuple[dict[str, Any], dict[str, Any]]] = {}
    assigned: list[dict[str, Any]] = []
    for clip in plan:
        key = (str(clip["background"]), str(clip["foreground"]), int(clip["pair_slot"]))
        if key not in chosen:
            bg_rows = grouped[key[0]]
            fg_rows = grouped[key[1]]
            bg_i = used[key[0]]
            fg_i = used[key[1]]
            if bg_i >= len(bg_rows) or fg_i >= len(fg_rows):
                raise RuntimeError("not enough ESC-50 train clips for the stress set")
            chosen[key] = (bg_rows[bg_i], fg_rows[fg_i])
            used[key[0]] += 1
            used[key[1]] += 1
        bg_row, fg_row = chosen[key]
        item = dict(clip)
        item["background_file"] = str(bg_row["filename"])
        item["foreground_file"] = str(fg_row["filename"])
        item["background_wave"] = bg_row["wave"]
        item["foreground_wave"] = fg_row["wave"]
        assigned.append(item)
    held = {name: rows[used[name] :] for name, rows in grouped.items()}
    for name, rows in held.items():
        if not rows:
            raise RuntimeError(f"no held-out ESC-50 clips for {name}")
    return assigned, held


def _resample(wave: torch.Tensor, sample_rate: int) -> torch.Tensor:
    flat = wave.reshape(-1).float().contiguous()
    if int(sample_rate) == SR:
        return flat
    return torchaudio.functional.resample(flat.unsqueeze(0), int(sample_rate), SR).squeeze(0).contiguous()


def _read_audio(audio: dict[str, Any]) -> torch.Tensor:
    blob = audio.get("bytes")
    if blob is not None:
        data, sample_rate = sf.read(io.BytesIO(blob), dtype="float32", always_2d=True)
    else:
        data, sample_rate = sf.read(str(audio["path"]), dtype="float32", always_2d=True)
    mono = np.mean(data, axis=1)
    return _resample(torch.from_numpy(np.ascontiguousarray(mono)), int(sample_rate))


def load_esc50_train(cache_dir: str | Any) -> list[dict[str, Any]]:
    from datasets import Audio, load_dataset

    dataset = load_dataset("ashraq/esc50", split="train", cache_dir=str(cache_dir))
    dataset = dataset.cast_column("audio", Audio(decode=False))
    wanted = set(STRESS_CLASSES)
    rows: list[dict[str, Any]] = []
    for row in dataset:
        category = str(row["category"])
        if category not in wanted:
            continue
        rows.append(
            {
                "category": category,
                "filename": str(row["filename"]),
                "fold": int(row["fold"]),
                "wave": _read_audio(row["audio"]),
            }
        )
    found = {row["category"] for row in rows}
    missing = [name for name in STRESS_CLASSES if name not in found]
    if missing:
        raise RuntimeError(f"ESC-50 train is missing {missing}")
    return rows


def _crop_mel(wave: torch.Tensor, mel_bank, mean: torch.Tensor, std: torch.Tensor):
    spec = waveform_logmel(wave, mel_bank)
    if int(spec.shape[0]) != N_FRAMES:
        raise RuntimeError(f"stress crop has {int(spec.shape[0])} frames")
    valid = torch.ones(N_FRAMES, dtype=torch.bool)
    norm = normalize_logmel(spec, mean, std) * valid.unsqueeze(-1)
    return norm, valid


def _encode(encoder: CausalLoopEarDEncoder, waves: list[torch.Tensor], mel_bank, mean, std) -> torch.Tensor:
    mels = []
    valids = []
    for wave in waves:
        mel, valid = _crop_mel(wave, mel_bank, mean, std)
        mels.append(mel)
        valids.append(valid)
    visible = visible_from_frames(torch.stack(valids, dim=0))
    grid, _summary = encoder(torch.stack(mels, dim=0), visible)
    return teacher_target(grid)


def _class_prototype(
    encoder: CausalLoopEarDEncoder,
    rows: list[dict[str, Any]],
    kind: str,
    mel_bank,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    chosen = rows[:PROTO_CLIPS]
    crops = []
    onsets: list[int | None] = []
    for row in chosen:
        if kind == "background":
            crops.append(fit_background(row["wave"]))
            onsets.append(None)
        else:
            crop, onset = foreground_crop(row["wave"])
            crops.append(crop)
            onsets.append(int(onset))
    pooled = _encode(encoder, crops, mel_bank, mean, std)
    vectors = []
    for index, onset in enumerate(onsets):
        if onset is None:
            vectors.append(l2_normalize(pooled[index].mean(dim=0)))
        else:
            vectors.append(pooled[index, onset_patch(onset)])
    return l2_normalize(torch.stack(vectors, dim=0).mean(dim=0))


@torch.no_grad()
def score_esc50_ood(
    encoder: CausalLoopEarDEncoder,
    examples: list[dict[str, Any]],
    mean: torch.Tensor,
    std: torch.Tensor,
    mel_bank,
    seed: int = 42,
) -> dict[str, Any]:
    encoder.eval()
    plan = plan_stress_clips(seed)
    assigned, held = assign_sources(plan, examples)
    prefers = _event_module()._prefers_first
    prototypes = {}
    for name in BACKGROUND_CLASSES:
        prototypes[name] = _class_prototype(encoder, held[name], "background", mel_bank, mean, std)
    for name in FOREGROUND_CLASSES:
        prototypes[name] = _class_prototype(encoder, held[name], "foreground", mel_bank, mean, std)
    mixtures = []
    records = []
    for clip in assigned:
        mixed, measured = mix_transient(
            clip["background_wave"],
            clip["foreground_wave"],
            int(clip["onset_sample"]),
            float(clip["snr_db"]),
        )
        mixtures.append(mixed)
        records.append(
            {
                "background": clip["background"],
                "foreground": clip["foreground"],
                "background_file": clip["background_file"],
                "foreground_file": clip["foreground_file"],
                "pair_slot": int(clip["pair_slot"]),
                "snr_db": float(clip["snr_db"]),
                "onset_sample": int(clip["onset_sample"]),
                "onset_s": int(clip["onset_sample"]) / float(SR),
                "measured_snr_db": float(measured),
            }
        )
    correct = 0
    ties = 0
    by_snr = {str(snr): {"n_correct": 0, "n_clips": 0} for snr in SNRS}
    step = 16
    cursor = 0
    while cursor < len(mixtures):
        batch = mixtures[cursor : cursor + step]
        pooled = _encode(encoder, batch, mel_bank, mean, std)
        for local, record in enumerate(records[cursor : cursor + step]):
            pre = pooled[local, pre_onset_patch(int(record["onset_sample"]))]
            at = pooled[local, onset_patch(int(record["onset_sample"]))]
            decision = prefers(pre, at, prototypes[record["background"]], prototypes[record["foreground"]])
            record["decision"] = None if decision is None else bool(decision)
            bucket = by_snr[str(float(record["snr_db"]))]
            bucket["n_clips"] += 1
            if decision is None:
                ties += 1
            elif decision:
                correct += 1
                bucket["n_correct"] += 1
        cursor += step
    return {
        "label": "ESC-50 polyphonic stress, continuous background plus a transient at -5, 0, and +5 dB",
        "dataset": "ashraq/esc50",
        "split": "train",
        "cache_dir": "data/esc50_cache",
        "seed": int(seed),
        "n_clips": len(records),
        "n_correct": correct,
        "n_ties": ties,
        "accuracy": None if not records else correct / len(records),
        "chance": 0.5,
        "preference": "pre-onset patch versus onset patch, same _prefers_first comparison to class-mean prototypes",
        "by_snr": by_snr,
        "clips": records,
    }
