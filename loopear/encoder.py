import hashlib
import importlib.util
import json
import math
import os
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio
from huggingface_hub import snapshot_download
from safetensors.torch import load_file as load_safetensors
from torch import nn

MODEL_ID: str = "MIT-SLS/USAD2-Small"
SR: int = 16000
WIN_S: float = 10.0
HOP_S: float = 10.0
EVAL_HOP_S: float = 5.0
MIN_WIN_S: float = 0.1
HOP_E: float = 0.02
D_E: int = 384
NUM_ENCODER_LAYERS: int = 12
TAP_LAYERS: tuple[int, ...] = (2, 5, 8, 11)
T_E_MAX: int = 504


def _load_hf_usad_module(snap_dir: Path) -> types.ModuleType:
    pkg_name = "_loopear_usad2_hf"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(snap_dir)]
        sys.modules[pkg_name] = pkg
    for mod_short in ("usad_modules", "usad_model"):
        full_name = f"{pkg_name}.{mod_short}"
        if full_name not in sys.modules:
            mod_path = snap_dir / f"{mod_short}.py"
            spec = importlib.util.spec_from_file_location(full_name, mod_path)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Failed to load module spec from {mod_path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[full_name] = module
            spec.loader.exec_module(module)
    return sys.modules[f"{pkg_name}.usad_model"]


def load_usad2_extractor(
    device: torch.device | str = "cpu",
    repo_id: str = MODEL_ID,
    chunk_seconds: float = WIN_S,
) -> tuple[nn.Module, torch.dtype]:
    dev = torch.device(device)
    snap_dir = Path(
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=[
                "config.json",
                "usad_model.py",
                "usad_modules.py",
                "model.safetensors",
            ],
        )
    )
    cfg_dict = json.loads((snap_dir / "config.json").read_text(encoding="utf-8"))
    cfg = types.SimpleNamespace(**cfg_dict)
    usad_mod = _load_hf_usad_module(snap_dir)
    model = usad_mod.UsadModel(cfg)
    raw_state = load_safetensors(str(snap_dir / "model.safetensors"))
    cleaned_state = {
        (k[6:] if k.startswith("model.") else k): v for k, v in raw_state.items()
    }
    model.load_state_dict(cleaned_state, strict=True)
    model.set_audio_chunk_size(chunk_seconds)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
    model = model.to(device=dev, dtype=dtype)
    return model, dtype


def verify_extractor_invariants(
    model: nn.Module,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    dev = torch.device(device)
    n_samples = 2 * SR
    sil = torch.zeros(1, n_samples, device=dev, dtype=torch.float32)
    t = torch.arange(n_samples, device=dev, dtype=torch.float32) / float(SR)
    tone = (0.25 * torch.sin(2.0 * math.pi * 440.0 * t)).unsqueeze(0)
    with torch.no_grad():
        out_s = model(wavs=sil)
        out_a = model(wavs=tone)
        out_b = model(wavs=tone)
    x_s = out_s["x"].float()
    x_a = out_a["x"].float()
    x_b = out_b["x"].float()
    hs_a = out_a["hidden_states"]
    if tuple(x_a.shape) != (1, 99, D_E):
        raise AssertionError(f"Expected (1, 99, {D_E}), got {tuple(x_a.shape)}")
    if len(hs_a) != NUM_ENCODER_LAYERS:
        raise AssertionError(
            f"Expected {NUM_ENCODER_LAYERS} hidden states, got {len(hs_a)}"
        )
    if not torch.isfinite(x_s).all() or not torch.isfinite(x_a).all():
        raise AssertionError("Non-finite encoder output detected")
    for idx, h in enumerate(hs_a):
        if tuple(h.shape) != (1, 99, D_E):
            raise AssertionError(
                f"Hidden state {idx} shape {tuple(h.shape)} != (1, 99, {D_E})"
            )
        if not torch.isfinite(h).all():
            raise AssertionError(f"Non-finite values in hidden state {idx}")
    if not torch.equal(x_a, x_b):
        raise AssertionError("Extractor is not deterministic on identical inputs")
    if torch.allclose(x_s, x_a, atol=1e-4):
        raise AssertionError("Silence and tone representations are indistinguishable")
    return {
        "shape": list(x_a.shape),
        "num_layers": len(hs_a),
        "max_abs_diff_sil_tone": float((x_a - x_s).abs().max().item()),
    }


def load_mono_audio(
    path: Path | str,
    target_sr: int = SR,
    channel: int | None = None,
    start_s: float | None = None,
    end_s: float | None = None,
) -> np.ndarray:
    info = sf.info(str(path))
    sr = int(info.samplerate)
    start_frame = 0 if start_s is None else max(0, int(round(start_s * sr)))
    if end_s is None:
        frames = -1
    else:
        end_frame = min(int(info.frames), int(round(end_s * sr)))
        frames = max(0, end_frame - start_frame)
    wav, sr = sf.read(
        str(path),
        start=start_frame,
        frames=frames,
        dtype="float32",
        always_2d=True,
    )
    if channel is not None:
        mono = wav[:, int(channel)]
    else:
        mono = wav.mean(axis=1)
    if sr != target_sr and mono.size > 0:
        tensor = torch.from_numpy(np.ascontiguousarray(mono)).unsqueeze(0)
        mono = (
            torchaudio.functional.resample(tensor, sr, target_sr)
            .squeeze(0)
            .numpy()
            .astype(np.float32)
        )
    return np.ascontiguousarray(mono, dtype=np.float32)


def expand_recording_windows(
    rows: list[dict[str, Any]],
    win_s: float = WIN_S,
    hop_s: float = HOP_S,
    min_win_s: float = MIN_WIN_S,
    sr: int = SR,
) -> list[dict[str, Any]]:
    win_n = int(round(win_s * sr))
    hop_n = int(round(hop_s * sr))
    min_n = int(round(min_win_s * sr))
    expanded: list[dict[str, Any]] = []
    for row in rows:
        wav: np.ndarray | None = None
        base_start_s = float(row.get("start_s", 0.0))
        if "wav" in row and row["wav"] is not None:
            wav = np.asarray(row["wav"], dtype=np.float32).reshape(-1)
            total_n = int(wav.shape[0])
        elif "duration_s" in row and row["duration_s"] is not None:
            total_n = int(round(float(row["duration_s"]) * sr))
        elif "path" in row and row["path"] is not None:
            info = sf.info(str(row["path"]))
            file_dur = float(info.frames) / float(info.samplerate)
            clip_end_s = float(row.get("end_s", file_dur))
            clip_dur_s = max(0.0, min(file_dur, clip_end_s) - base_start_s)
            total_n = int(round(clip_dur_s * sr))
        else:
            raise ValueError("Row must specify wav, duration_s, or path")
        if total_n < min_n:
            continue
        if total_n <= win_n:
            starts = [0]
        else:
            starts = list(range(0, total_n - min_n + 1, hop_n))
        for w_idx, s0 in enumerate(starts):
            s1 = min(s0 + win_n, total_n)
            if s1 - s0 < min_n:
                continue
            entry: dict[str, Any] = {
                "id": f"{row['id']}__w{w_idx:04d}",
                "source_id": str(row["id"]),
                "start_sample": int(s0),
                "end_sample": int(s1),
                "start_s": base_start_s + float(s0) / float(sr),
                "end_s": base_start_s + float(s1) / float(sr),
                "meta": dict(row.get("meta", {})),
            }
            if wav is not None:
                entry["wav"] = wav[s0:s1]
            elif "path" in row:
                entry["path"] = str(row["path"])
                if row.get("channel") is not None:
                    entry["channel"] = int(row["channel"])
            expanded.append(entry)
    return expanded


def random_crop_window(
    e: np.ndarray,
    length: int,
    crop_frames: int,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, int]:
    valid_len = min(int(length), int(e.shape[0]))
    if valid_len <= crop_frames:
        out = np.zeros((crop_frames, e.shape[1]), dtype=e.dtype)
        out[:valid_len] = e[:valid_len]
        return out, valid_len
    gen = rng if rng is not None else np.random.default_rng()
    start = int(gen.integers(0, valid_len - crop_frames + 1))
    return np.ascontiguousarray(e[start : start + crop_frames]), crop_frames


def pool_layers_stride(
    hidden_states: torch.Tensor,
    stride: int = 8,
    max_out: int | None = None,
) -> np.ndarray:
    num_layers, t_valid, dim = hidden_states.shape
    out_len = max_out if max_out is not None else math.ceil(t_valid / stride)
    out = np.zeros((num_layers, out_len, dim), dtype=np.float32)
    hs_np = hidden_states.detach().cpu().float().numpy()
    for i in range(out_len):
        t0 = i * stride
        if t0 >= t_valid:
            break
        t1 = min(t0 + stride, t_valid)
        out[:, i, :] = hs_np[:, t0:t1, :].mean(axis=1)
    return out


def extract_window_features(
    model: nn.Module,
    wavs: torch.Tensor,
    wav_lengths: torch.Tensor | None = None,
) -> dict[str, Any]:
    with torch.no_grad():
        out = model(wavs=wavs, wav_lengths=wav_lengths)
    x = out["x"].float()
    hs = torch.stack([h.float() for h in out["hidden_states"]], dim=1)
    x_lengths = out["x_lengths"].long()
    if not torch.isfinite(x).all() or not torch.isfinite(hs).all():
        raise ValueError("Non-finite features produced by encoder")
    return {"x": x, "hidden_states": hs, "x_lengths": x_lengths}


def _assert_finite_arrays(arrays: dict[str, np.ndarray]) -> None:
    for name, arr in arrays.items():
        if np.issubdtype(arr.dtype, np.floating) and not np.isfinite(arr).all():
            raise ValueError(f"Non-finite values found in array {name}")


def save_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    _assert_finite_arrays(arrays)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}.npz")
    try:
        np.savez(tmp_path, **arrays)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}.json")
    try:
        tmp_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def make_cache_key(**params: Any) -> tuple[str, dict[str, Any]]:
    canonical = json.dumps(params, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return digest, dict(params)


class ShardedFeatureCache:
    def __init__(self, root: Path | str):
        self.root = Path(root)

    def _rows_digest(self, rows: list[dict[str, Any]]) -> str:
        h = hashlib.sha256()
        for row in rows:
            h.update(str(row["id"]).encode("utf-8"))
            if "wav" in row and row["wav"] is not None:
                arr = np.asarray(row["wav"], dtype=np.float32)
                h.update(arr.tobytes())
            elif "path" in row and row["path"] is not None:
                p = Path(row["path"])
                h.update(str(p).encode("utf-8"))
                if p.exists():
                    h.update(str(p.stat().st_size).encode("utf-8"))
                if row.get("channel") is not None:
                    h.update(f"ch={row['channel']}".encode("utf-8"))
                if row.get("start_s") is not None or row.get("end_s") is not None:
                    h.update(
                        f"span={row.get('start_s')}:{row.get('end_s')}".encode("utf-8")
                    )
        return h.hexdigest()[:16]

    def extract_dataset(
        self,
        dataset_name: str,
        rows: list[dict[str, Any]],
        model: nn.Module,
        device: torch.device | str = "cpu",
        batch_size: int = 8,
        shard_size: int = 256,
        stride: int = 8,
        win_s: float = WIN_S,
        hop_s: float = HOP_S,
        min_win_s: float = MIN_WIN_S,
        save_layer_pool: bool = False,
    ) -> Path:
        dev = torch.device(device)
        key, meta = make_cache_key(
            dataset=dataset_name,
            source_hash=self._rows_digest(rows),
            encoder=MODEL_ID,
            layers=NUM_ENCODER_LAYERS if save_layer_pool else 1,
            sr=SR,
            win_s=win_s,
            hop_s=hop_s,
            stride=stride,
            save_layer_pool=bool(save_layer_pool),
            dtype="float16",
        )
        out_dir = self.root / key
        index_path = out_dir / "index.json"
        if index_path.exists():
            return out_dir
        windows = expand_recording_windows(
            rows, win_s=win_s, hop_s=hop_s, min_win_s=min_win_s, sr=SR
        )
        if not windows:
            raise ValueError("No valid audio windows produced for caching")
        out_dir.mkdir(parents=True, exist_ok=True)
        max_z = math.ceil(T_E_MAX / stride)
        index_items: list[dict[str, Any]] = []
        for s0 in range(0, len(windows), shard_size):
            sw = windows[s0 : s0 + shard_size]
            e_buf = np.zeros((len(sw), T_E_MAX, D_E), dtype=np.float16)
            l_buf = np.zeros((len(sw),), dtype=np.int16)
            lp_buf = (
                np.zeros(
                    (len(sw), NUM_ENCODER_LAYERS, max_z, D_E), dtype=np.float16
                )
                if save_layer_pool
                else None
            )
            for b0 in range(0, len(sw), batch_size):
                batch = sw[b0 : b0 + batch_size]
                wav_list: list[np.ndarray] = []
                for item in batch:
                    if "wav" in item and item["wav"] is not None:
                        wav_list.append(np.asarray(item["wav"], dtype=np.float32))
                    else:
                        wav_list.append(
                            load_mono_audio(
                                item["path"],
                                target_sr=SR,
                                channel=item.get("channel"),
                                start_s=item["start_s"],
                                end_s=item["end_s"],
                            )
                        )
                lens = [int(w.shape[0]) for w in wav_list]
                max_len = max(lens)
                wav_tensor = torch.zeros(
                    (len(batch), max_len), dtype=torch.float32, device=dev
                )
                for idx, w in enumerate(wav_list):
                    wav_tensor[idx, : w.shape[0]] = torch.from_numpy(w)
                len_tensor = torch.tensor(lens, dtype=torch.long, device=dev)
                feats = extract_window_features(model, wav_tensor, len_tensor)
                x = feats["x"]
                hs = feats["hidden_states"]
                x_lens = feats["x_lengths"].cpu().tolist()
                for idx in range(len(batch)):
                    n_frames = min(int(x_lens[idx]), T_E_MAX)
                    e_np = x[idx, :n_frames].cpu().numpy().astype(np.float16)
                    _assert_finite_arrays({"E": e_np})
                    e_buf[b0 + idx, :n_frames] = e_np
                    l_buf[b0 + idx] = n_frames
                    if lp_buf is not None:
                        lp_np = pool_layers_stride(
                            hs[idx, :, :n_frames], stride=stride, max_out=max_z
                        ).astype(np.float16)
                        _assert_finite_arrays({"layer_pool": lp_np})
                        lp_buf[b0 + idx] = lp_np
            shard_name = f"shard_{s0 // shard_size:04d}.npz"
            shard_arrays: dict[str, np.ndarray] = {"E": e_buf, "lengths": l_buf}
            if lp_buf is not None:
                shard_arrays["layer_pool"] = lp_buf
            save_npz_atomic(out_dir / shard_name, **shard_arrays)
            for local_idx, item in enumerate(sw):
                index_items.append(
                    {
                        "id": item["id"],
                        "source_id": item["source_id"],
                        "shard": shard_name,
                        "row": local_idx,
                        "n_frames": int(l_buf[local_idx]),
                        "start_s": item["start_s"],
                        "end_s": item["end_s"],
                        "meta": item["meta"],
                    }
                )
        write_json_atomic(
            index_path,
            {"key": key, "meta": meta, "items": index_items},
        )
        return out_dir

    def read_index(self, cache_dir: Path | str) -> dict[str, Any]:
        path = Path(cache_dir) / "index.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def extract_tap_dataset(
        self,
        dataset_name: str,
        rows: list[dict[str, Any]],
        model: nn.Module,
        device: torch.device | str = "cpu",
        batch_size: int = 8,
        shard_size: int = 128,
        tap_layers: tuple[int, ...] = TAP_LAYERS,
        win_s: float = WIN_S,
        hop_s: float = HOP_S,
        min_win_s: float = MIN_WIN_S,
    ) -> Path:
        dev = torch.device(device)
        taps = tuple(int(i) for i in tap_layers)
        key, meta = make_cache_key(
            dataset=dataset_name,
            source_hash=self._rows_digest(rows),
            encoder=MODEL_ID,
            tap_layers=list(taps),
            layer_clip=True,
            sr=SR,
            win_s=win_s,
            hop_s=hop_s,
            dtype="float16",
        )
        out_dir = self.root / key
        index_path = out_dir / "index.json"
        if index_path.exists():
            return out_dir
        windows = expand_recording_windows(
            rows, win_s=win_s, hop_s=hop_s, min_win_s=min_win_s, sr=SR
        )
        if not windows:
            raise ValueError("No valid audio windows produced for tap caching")
        out_dir.mkdir(parents=True, exist_ok=True)
        index_items: list[dict[str, Any]] = []
        for s0 in range(0, len(windows), shard_size):
            sw = windows[s0 : s0 + shard_size]
            tap_buf = np.zeros(
                (len(sw), len(taps), T_E_MAX, D_E), dtype=np.float16
            )
            lc_buf = np.zeros(
                (len(sw), NUM_ENCODER_LAYERS, D_E), dtype=np.float16
            )
            l_buf = np.zeros((len(sw),), dtype=np.int16)
            for b0 in range(0, len(sw), batch_size):
                batch = sw[b0 : b0 + batch_size]
                wav_list: list[np.ndarray] = []
                for item in batch:
                    if "wav" in item and item["wav"] is not None:
                        wav_list.append(np.asarray(item["wav"], dtype=np.float32))
                    else:
                        wav_list.append(
                            load_mono_audio(
                                item["path"],
                                target_sr=SR,
                                channel=item.get("channel"),
                                start_s=item["start_s"],
                                end_s=item["end_s"],
                            )
                        )
                lens = [int(w.shape[0]) for w in wav_list]
                max_len = max(lens)
                wav_tensor = torch.zeros(
                    (len(batch), max_len), dtype=torch.float32, device=dev
                )
                for idx, w in enumerate(wav_list):
                    wav_tensor[idx, : w.shape[0]] = torch.from_numpy(w)
                len_tensor = torch.tensor(lens, dtype=torch.long, device=dev)
                feats = extract_window_features(model, wav_tensor, len_tensor)
                hs = feats["hidden_states"]
                x_lens = feats["x_lengths"].cpu().tolist()
                for idx in range(len(batch)):
                    n_frames = min(int(x_lens[idx]), T_E_MAX)
                    tap_np = (
                        hs[idx, list(taps), :n_frames]
                        .cpu()
                        .numpy()
                        .astype(np.float16)
                    )
                    lc_np = (
                        hs[idx, :, :n_frames]
                        .mean(dim=1)
                        .cpu()
                        .numpy()
                        .astype(np.float16)
                    )
                    _assert_finite_arrays({"E_taps": tap_np, "layer_clip": lc_np})
                    tap_buf[b0 + idx, :, :n_frames] = tap_np
                    lc_buf[b0 + idx] = lc_np
                    l_buf[b0 + idx] = n_frames
            shard_name = f"shard_{s0 // shard_size:04d}.npz"
            save_npz_atomic(
                out_dir / shard_name,
                E_taps=tap_buf,
                layer_clip=lc_buf,
                lengths=l_buf,
            )
            for local_idx, item in enumerate(sw):
                index_items.append(
                    {
                        "id": item["id"],
                        "source_id": item["source_id"],
                        "shard": shard_name,
                        "row": local_idx,
                        "n_frames": int(l_buf[local_idx]),
                        "start_s": item["start_s"],
                        "end_s": item["end_s"],
                        "meta": item["meta"],
                    }
                )
        write_json_atomic(
            index_path,
            {"key": key, "meta": meta, "items": index_items},
        )
        return out_dir

    def load_shard(self, cache_dir: Path | str, shard_name: str) -> dict[str, np.ndarray]:
        shard_path = Path(cache_dir) / shard_name
        with np.load(shard_path) as data:
            arrays = {k: data[k] for k in data.files}
        _assert_finite_arrays(arrays)
        return arrays
