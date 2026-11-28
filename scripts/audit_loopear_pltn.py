from __future__ import annotations

import json
import random
import struct
import zlib
from pathlib import Path

import numpy as np
import soundfile as sf

from loopear.d.spec import SR

MIXTURE = Path("data/manifests/loopear_pltn_mixture.json")
OUT_DIR = Path("/tmp/loopear_pltn_listen")
CLIP_S = 12.0


def _clip(path: Path, start_s: float) -> tuple[np.ndarray, float]:
    info = sf.info(str(path))
    start = min(float(start_s), max(0.0, float(info.duration) - 0.2))
    rate = int(info.samplerate)
    wav, sr = sf.read(
        str(path),
        dtype="float32",
        always_2d=False,
        start=int(start * rate),
        frames=int(CLIP_S * rate),
    )
    if int(sr) != SR:
        raise RuntimeError(f"{path} sr {sr}")
    if wav.ndim != 1:
        wav = np.mean(wav, axis=1)
    return np.ascontiguousarray(wav, dtype=np.float32), start


def _measures(wav: np.ndarray) -> dict[str, float]:
    if wav.size == 0:
        raise RuntimeError("empty clip")
    abs_wav = np.abs(wav)
    spectrum = np.abs(np.fft.rfft(wav * np.hanning(wav.size)))
    freqs = np.fft.rfftfreq(wav.size, d=1.0 / SR)
    weight = float(spectrum.sum()) + 1e-12
    return {
        "seconds": float(wav.size / SR),
        "rms": float(np.sqrt(np.mean(wav * wav))),
        "peak": float(abs_wav.max()),
        "fraction_below_0p01": float(np.mean(abs_wav < 0.01)),
        "fraction_near_clip": float(np.mean(abs_wav > 0.98)),
        "spectral_centroid_hz": float(np.sum(freqs * spectrum) / weight),
    }


def _png(path: Path, image: np.ndarray) -> None:
    height, width = image.shape
    raw = b"".join(b"\x00" + image[row].tobytes() for row in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    payload = b"\x89PNG\r\n\x1a\n"
    payload += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
    payload += chunk(b"IDAT", zlib.compress(raw, 9))
    payload += chunk(b"IEND", b"")
    path.write_bytes(payload)


def _spectrogram(wav: np.ndarray, dest: Path, title: str) -> None:
    del title
    nfft = 512
    hop = 160
    window = np.hanning(nfft).astype(np.float32)
    frames = 1 + max(0, (wav.size - nfft) // hop)
    spec = np.empty((nfft // 2, frames), dtype=np.float32)
    for index in range(frames):
        start = index * hop
        spec[:, index] = np.abs(np.fft.rfft(wav[start : start + nfft] * window))[:-1]
    view = np.log1p(spec)
    view = view / max(float(view.max()), 1e-6)
    image = (255.0 * (1.0 - view[::-1])).astype(np.uint8)
    _png(dest, image)


def main() -> None:
    mixture = json.loads(MIXTURE.read_text(encoding="utf-8"))
    rows = list(mixture["items"])
    by_source: dict[str, list[dict]] = {}
    for row in rows:
        by_source.setdefault(str(row["source"]), []).append(row)
    rng = random.Random(42)
    chosen: list[dict] = []
    for source in sorted(by_source):
        pool = list(by_source[source])
        rng.shuffle(pool)
        chosen.append(pool[0])
    rest = [row for row in rows if row not in chosen]
    rng.shuffle(rest)
    while len(chosen) < 12 and rest:
        chosen.append(rest.pop())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    notes = []
    for index, row in enumerate(chosen):
        path = Path(row["audio_path"])
        span = max(0.0, float(row["duration_s"]) - CLIP_S)
        start_s = rng.random() * span
        wav, used_start = _clip(path, start_s)
        image = OUT_DIR / f"{index:02d}_{row['source']}_{row['session']}.png"
        _spectrogram(wav, image, f"{row['source']} {row['session']} @ {used_start:.1f}s")
        sf.write(str(OUT_DIR / f"{index:02d}_{row['source']}_{row['session']}.wav"), wav, SR)
        notes.append(
            {
                "source": row["source"],
                "session": row["session"],
                "experience": row["experience"],
                "license": row["license"],
                "start_s": used_start,
                "image": str(image),
                "measures": _measures(wav),
                "note": row.get("note"),
            }
        )
    payload = {"seed": 42, "clip_s": CLIP_S, "sample": notes}
    Path("/tmp/loopear_pltn_listen.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"clips": len(notes), "dir": str(OUT_DIR)}))


if __name__ == "__main__":
    main()
