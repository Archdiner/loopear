import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from loopear.encoder import D_E, HOP_E, SR, TAP_LAYERS, load_usad2_extractor
from loopear.halting import HaltingGate, extract_halting_features
from loopear.losses import MultiTaskHeads
from loopear.models import LoopEarModel

DEFAULT_EVENT_LABELS: tuple[str, ...] = tuple(f"event_{i:02d}" for i in range(15))


class AudioRingBuffer:
    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self._buf = np.zeros(0, dtype=np.float32)
        self._total_written = 0
        self._next_end: int | None = None

    def append(self, pcm: np.ndarray) -> None:
        arr = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return
        self._total_written += int(arr.size)
        if arr.size >= self.capacity:
            self._buf = np.ascontiguousarray(arr[-self.capacity :], dtype=np.float32)
        elif self._buf.size + arr.size > self.capacity:
            keep = self.capacity - arr.size
            self._buf = np.concatenate((self._buf[-keep:], arr))
        else:
            self._buf = np.concatenate((self._buf, arr))

    @property
    def buffered_samples(self) -> int:
        return int(self._buf.size)

    def has_step(self, win: int, step: int) -> bool:
        if self._buf.size < win:
            return False
        target = win if self._next_end is None else self._next_end
        return self._total_written >= target

    def consume_window(self, win: int, step: int) -> np.ndarray:
        if not self.has_step(win, step):
            raise RuntimeError("Insufficient audio buffered for window consumption")
        target = win if self._next_end is None else self._next_end
        oldest_idx = self._total_written - self._buf.size
        if target - win < oldest_idx:
            target = oldest_idx + win
        end_rel = target - oldest_idx
        start_rel = end_rel - win
        window = np.ascontiguousarray(self._buf[start_rel:end_rel], dtype=np.float32)
        self._next_end = target + step
        return window


class OnlinePatternVocab:
    def __init__(self, sim_threshold: float = 0.80, max_clusters: int = 64):
        self.sim_threshold = float(sim_threshold)
        self.max_clusters = int(max_clusters)
        self._clusters: list[dict[str, Any]] = []

    def add(self, timestamp_s: float, vec: np.ndarray) -> dict[str, Any]:
        v = np.asarray(vec, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(v))
        if norm < 1e-8:
            return {"cluster_id": -1, "similarity": 0.0, "novelty": 1.0}
        unit = v / norm
        best_idx = -1
        best_sim = -1.0
        for idx, cluster in enumerate(self._clusters):
            sim = float(np.dot(unit, cluster["centroid"]))
            if sim > best_sim:
                best_sim = sim
                best_idx = idx
        if best_idx >= 0 and best_sim >= self.sim_threshold:
            cl = self._clusters[best_idx]
            cl["sum"] = cl["sum"] + unit
            cl["count"] += 1
            cl["last_s"] = float(timestamp_s)
            cl["centroid"] = cl["sum"] / max(float(np.linalg.norm(cl["sum"])), 1e-8)
            return {
                "cluster_id": cl["id"],
                "similarity": round(best_sim, 4),
                "novelty": round(max(0.0, 1.0 - best_sim), 4),
            }
        if len(self._clusters) < self.max_clusters:
            cid = len(self._clusters)
            self._clusters.append(
                {
                    "id": cid,
                    "sum": unit.copy(),
                    "centroid": unit.copy(),
                    "count": 1,
                    "first_s": float(timestamp_s),
                    "last_s": float(timestamp_s),
                }
            )
            return {"cluster_id": cid, "similarity": 0.0, "novelty": 1.0}
        cl = self._clusters[best_idx]
        cl["sum"] = cl["sum"] + unit
        cl["count"] += 1
        cl["last_s"] = float(timestamp_s)
        cl["centroid"] = cl["sum"] / max(float(np.linalg.norm(cl["sum"])), 1e-8)
        return {
            "cluster_id": cl["id"],
            "similarity": round(best_sim, 4),
            "novelty": round(max(0.0, 1.0 - best_sim), 4),
        }

    def types(self, top_k: int = 10) -> list[dict[str, Any]]:
        ranked = sorted(self._clusters, key=lambda c: (-c["count"], c["id"]))
        return [
            {
                "id": c["id"],
                "count": c["count"],
                "first_s": round(c["first_s"], 2),
                "last_s": round(c["last_s"], 2),
            }
            for c in ranked[:top_k]
        ]


class EventTimeline:
    def __init__(
        self,
        labels: tuple[str, ...] = DEFAULT_EVENT_LABELS,
        threshold: float = 0.5,
        hop_s: float = HOP_E * 8,
        max_history: int = 200,
    ):
        self.labels = tuple(labels)
        self.threshold = float(threshold)
        self.hop_s = float(hop_s)
        self.now = 0.0
        self._events: deque[dict[str, Any]] = deque(maxlen=max_history)

    def update(self, probs: np.ndarray) -> None:
        arr = np.asarray(probs, dtype=np.float32)
        if arr.ndim != 2:
            return
        for row in arr:
            self.now += self.hop_s
            active = [
                {
                    "label": self.labels[i] if i < len(self.labels) else f"class_{i}",
                    "prob": round(float(row[i]), 3),
                }
                for i in range(row.shape[0])
                if float(row[i]) >= self.threshold
            ]
            if active:
                self._events.append({"t_s": round(self.now, 2), "active": active})

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        items = list(self._events)
        return items[-limit:]


class LoopEarStream:
    def __init__(
        self,
        encoder: nn.Module,
        refiner: nn.Module,
        heads: MultiTaskHeads,
        depth: int = 2,
        win_s: float = 4.0,
        step_s: float = 1.0,
        device: torch.device | str = "cpu",
        labels: tuple[str, ...] = DEFAULT_EVENT_LABELS,
        tap_layers: tuple[int, ...] = TAP_LAYERS,
        halting_gate: HaltingGate | None = None,
        adaptive: bool = False,
        halt_threshold: float = 0.5,
    ):
        self.dev = torch.device(device)
        self.enc = encoder.to(self.dev).eval()
        self.ref = refiner.to(self.dev).eval()
        self.heads = heads.to(self.dev).eval()
        self.valid_depths = tuple(getattr(self.ref, "depths", (1, 2, 4)))
        if depth not in self.valid_depths:
            raise ValueError(f"depth must be in {self.valid_depths}, got {depth}")
        self.depth = int(depth)
        self.tap_layers = tuple(int(i) for i in tap_layers)
        self.halting_gate = (
            halting_gate.to(self.dev).eval() if halting_gate is not None else None
        )
        self.adaptive = bool(adaptive and self.halting_gate is not None)
        self.halt_threshold = float(halt_threshold)
        self.win_s = float(win_s)
        self.step_s = float(step_s)
        self.win = int(round(win_s * SR))
        self.step = int(round(step_s * SR))
        self.stride = int(getattr(self.ref, "stride", 8))
        self.ring = AudioRingBuffer(max(self.win * 4, 120 * SR))
        self.enrolled: dict[str, dict[str, Any]] = {}
        self.vocab = OnlinePatternVocab()
        self.timeline = EventTimeline(labels=labels, hop_s=HOP_E * self.stride)
        self.ms: deque[float] = deque(maxlen=50)
        self.windows_processed = 0
        self.depth_counts: dict[int, int] = {k: 0 for k in self.valid_depths}
        self._last_depth_used = self.depth
        self._last_hits: dict[str, float] = {}

    @classmethod
    def from_checkpoint(
        cls,
        ckpt: Path | str | None = None,
        depth: int = 2,
        win_s: float = 4.0,
        step_s: float = 1.0,
        device: torch.device | str = "cpu",
        adaptive: bool = False,
        halt_threshold: float | None = None,
    ) -> "LoopEarStream":
        dev = torch.device(device)
        enc, _ = load_usad2_extractor(dev)
        ref_kwargs: dict[str, Any] = {
            "d_e": D_E,
            "d": 384,
            "h": 6,
            "d_ff": 768,
            "stride": 8,
        }
        head_kwargs: dict[str, Any] = {}
        labels = DEFAULT_EVENT_LABELS
        tap_layers = TAP_LAYERS
        gate: HaltingGate | None = None
        default_thr = 0.5
        payload: dict[str, Any] | None = None
        if ckpt is not None:
            payload = torch.load(Path(ckpt), map_location=dev, weights_only=False)
            if "refiner_kwargs" in payload:
                ref_kwargs.update(payload["refiner_kwargs"])
            if "head_kwargs" in payload:
                head_kwargs.update(payload["head_kwargs"])
            if "labels" in payload:
                labels = tuple(payload["labels"])
            if "tap_layers" in payload:
                tap_layers = tuple(int(i) for i in payload["tap_layers"])
            if "default_halt_threshold" in payload:
                default_thr = float(payload["default_halt_threshold"])
            if "halting_gate" in payload and payload["halting_gate"] is not None:
                gate_kwargs = dict(payload.get("halting_gate_kwargs", {}))
                gate = HaltingGate(**gate_kwargs)
                gate.load_state_dict(payload["halting_gate"], strict=True)
        ref = LoopEarModel(**ref_kwargs)
        head_kwargs.setdefault("d", ref.out_dim)
        heads = MultiTaskHeads(**head_kwargs)
        if payload is not None:
            if "refiner" in payload:
                ref.load_state_dict(payload["refiner"], strict=True)
            if "heads" in payload:
                heads.load_state_dict(payload["heads"], strict=True)
        return cls(
            encoder=enc,
            refiner=ref,
            heads=heads,
            depth=depth,
            win_s=win_s,
            step_s=step_s,
            device=dev,
            labels=labels,
            tap_layers=tap_layers,
            halting_gate=gate,
            adaptive=adaptive,
            halt_threshold=default_thr if halt_threshold is None else float(halt_threshold),
        )

    def set_depth(self, k: int) -> int:
        if k not in self.valid_depths:
            raise ValueError(f"depth must be in {self.valid_depths}, got {k}")
        self.depth = int(k)
        return self.depth

    def set_adaptive(
        self, enabled: bool, threshold: float | None = None
    ) -> dict[str, Any]:
        if enabled and self.halting_gate is None:
            raise ValueError("Adaptive halting requires a loaded halting_gate")
        self.adaptive = bool(enabled)
        if threshold is not None:
            self.halt_threshold = float(threshold)
        return {
            "adaptive": self.adaptive,
            "halt_threshold": round(self.halt_threshold, 4),
        }

    def _run_encoder(
        self, wav_t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        need_hs = bool(
            getattr(self.ref, "use_multilayer", False)
            or getattr(self.heads, "use_layer_clip_spk", False)
        )
        if need_hs:
            try:
                enc_out = self.enc(wavs=wav_t, return_hidden_states=True)
            except TypeError:
                enc_out = self.enc(wavs=wav_t)
        else:
            enc_out = self.enc(wavs=wav_t)
        x = enc_out["x"].float()
        x_len = enc_out["x_lengths"].long()
        hs_raw = enc_out.get("hidden_states")
        hidden_taps: torch.Tensor | None = None
        layer_clip: torch.Tensor | None = None
        if hs_raw is not None:
            stacked = torch.stack(list(hs_raw), dim=1).float()
            t_valid = min(int(x_len[0].item()), stacked.shape[2])
            if getattr(self.heads, "use_layer_clip_spk", False):
                layer_clip = stacked[:, :, : max(1, t_valid), :].mean(dim=2)
            if getattr(self.ref, "use_multilayer", False):
                idx = [i for i in self.tap_layers if i < stacked.shape[1]]
                if idx:
                    hidden_taps = stacked[:, idx, :, :]
        return x, x_len, hidden_taps, layer_clip

    @torch.no_grad()
    def encode(
        self,
        wav: np.ndarray,
        k: int | None = None,
        adaptive: bool | None = None,
    ) -> dict[str, Any]:
        use_adapt = (
            (self.adaptive if adaptive is None else bool(adaptive))
            and k is None
            and self.halting_gate is not None
            and isinstance(self.ref, LoopEarModel)
            and self.valid_depths == (1, 2, 4)
        )
        arr = np.asarray(wav, dtype=np.float32).reshape(-1)
        if arr.size < int(0.1 * SR):
            raise ValueError("Audio clip must be at least 0.1s (1600 samples)")
        wav_t = torch.from_numpy(arr).unsqueeze(0).to(self.dev)
        x, x_len, hidden_taps, layer_clip = self._run_encoder(wav_t)
        halt_probs: dict[int, float] = {}
        if use_adapt and isinstance(self.ref, LoopEarModel) and self.halting_gate is not None:
            ctx = self.ref.prepare_evidence(
                x, x_len, hidden_states=hidden_taps, use_kv_cache=True
            )
            pad = ctx["z_pad"]
            z_cur, z1 = self.ref.step_refine(ctx["z0"], 1, ctx, use_kv_cache=True)
            f1, p1 = extract_halting_features(
                ctx["e_in"], x_len, ctx["z0"], z1, pad, self.heads, step_k=1
            )
            cont1 = float(self.halting_gate.continue_prob(f1).item())
            halt_probs[1] = round(cont1, 4)
            if cont1 < self.halt_threshold:
                z = z1
                d_used = 1
            else:
                z_cur, z2 = self.ref.step_refine(z_cur, 2, ctx, use_kv_cache=True)
                f2, _ = extract_halting_features(
                    ctx["e_in"],
                    x_len,
                    z1,
                    z2,
                    pad,
                    self.heads,
                    step_k=2,
                    prev_probs=p1,
                )
                cont2 = float(self.halting_gate.continue_prob(f2).item())
                halt_probs[2] = round(cont2, 4)
                if cont2 < self.halt_threshold:
                    z = z2
                    d_used = 2
                else:
                    z_cur, _ = self.ref.step_refine(z_cur, 3, ctx, use_kv_cache=True)
                    _, z4 = self.ref.step_refine(z_cur, 4, ctx, use_kv_cache=True)
                    z = z4
                    d_used = 4
        else:
            d_used = self.depth if k is None else int(k)
            if d_used not in self.valid_depths:
                raise ValueError(f"depth must be in {self.valid_depths}, got {d_used}")
            states, pad = self.ref.forward_states(
                x, x_len, depths=(d_used,), hidden_states=hidden_taps
            )
            z = states[d_used]
        valid = (~pad[0]).float().unsqueeze(-1)
        pooled = (z[0] * valid).sum(dim=0) / valid.sum().clamp(min=1.0)
        return {
            "z": z[0],
            "pad": pad[0],
            "vec": F.normalize(pooled, dim=-1),
            "ev": torch.sigmoid(self.heads.event_logits(z))[0],
            "spk": self.heads.speaker_embed(z, pad, layer_clip=layer_clip)[0],
            "depth_used": int(d_used),
            "halt_probs": halt_probs,
        }

    @torch.no_grad()
    def enroll(self, label: str, wav: np.ndarray) -> dict[str, Any]:
        clean_label = str(label).strip()
        if not clean_label:
            raise ValueError("Enrollment label must be non-empty")
        arr = np.asarray(wav, dtype=np.float32).reshape(-1)
        if arr.size < int(0.1 * SR):
            raise ValueError("Enrollment clip must be at least 0.1s (1600 samples)")
        wav_t = torch.from_numpy(arr).unsqueeze(0).to(self.dev)
        x, x_len, hidden_taps, _ = self._run_encoder(wav_t)
        states, pad = self.ref.forward_states(
            x, x_len, depths=self.valid_depths, hidden_states=hidden_taps
        )
        valid = (~pad[0]).float().unsqueeze(-1)
        denom = valid.sum().clamp(min=1.0)
        entry = self.enrolled.setdefault(
            clean_label,
            {
                "sum": {
                    k: np.zeros(self.ref.out_dim, dtype=np.float32)
                    for k in self.valid_depths
                },
                "n": 0,
                "proto": {},
            },
        )
        for k in self.valid_depths:
            pooled = (states[k][0] * valid).sum(dim=0) / denom
            unit = F.normalize(pooled, dim=-1).cpu().numpy().astype(np.float32)
            entry["sum"][k] = entry["sum"][k] + unit
        entry["n"] += 1
        entry["proto"] = {
            k: entry["sum"][k] / max(float(np.linalg.norm(entry["sum"][k])), 1e-8)
            for k in self.valid_depths
        }
        return {"label": clean_label, "n": entry["n"]}

    def compare(
        self,
        wav_a: np.ndarray,
        wav_b: np.ndarray,
        k: int | None = None,
    ) -> dict[str, Any]:
        a = self.encode(wav_a, k=k)
        b = self.encode(wav_b, k=k)
        return {
            "depth": int(a["depth_used"]),
            "depth_b": int(b["depth_used"]),
            "acoustic_cos": round(float((a["vec"] * b["vec"]).sum().item()), 6),
            "speaker_cos": round(float((a["spk"] * b["spk"]).sum().item()), 6),
        }

    def feed(self, pcm: np.ndarray) -> dict[str, Any]:
        self.ring.append(pcm)
        step_frames = max(1, int(round(self.step_s / (HOP_E * self.stride))))
        while self.ring.has_step(self.win, self.step):
            window = self.ring.consume_window(self.win, self.step)
            t0 = time.perf_counter()
            res = self.encode(window)
            d_used = int(res["depth_used"])
            self._last_depth_used = d_used
            self.depth_counts[d_used] = self.depth_counts.get(d_used, 0) + 1
            tail_ev = res["ev"][-step_frames:].cpu().numpy()
            self.timeline.update(tail_ev)
            tail_z = res["z"][-step_frames:].mean(dim=0)
            tail_vec = F.normalize(tail_z, dim=-1).cpu().numpy().astype(np.float32)
            self._last_hits = {
                lbl: round(float(np.dot(tail_vec, e["proto"][d_used])), 4)
                for lbl, e in self.enrolled.items()
            }
            self.vocab.add(self.timeline.now, tail_vec)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self.ms.append(elapsed_ms)
            self.windows_processed += 1
        return self.state()

    def state(self) -> dict[str, Any]:
        mean_ms = float(np.mean(self.ms)) if self.ms else 0.0
        last_ms = float(self.ms[-1]) if self.ms else 0.0
        total_w = sum(self.depth_counts.values())
        avg_depth = (
            float(sum(k * c for k, c in self.depth_counts.items())) / float(total_w)
            if total_w > 0
            else float(self.depth)
        )
        return {
            "depth": self.depth,
            "adaptive": self.adaptive,
            "halt_threshold": round(self.halt_threshold, 4),
            "last_depth_used": self._last_depth_used,
            "avg_depth": round(avg_depth, 4),
            "depth_counts": {str(k): int(v) for k, v in sorted(self.depth_counts.items())},
            "valid_depths": list(self.valid_depths),
            "windows_processed": self.windows_processed,
            "buffered_s": round(float(self.ring.buffered_samples) / float(SR), 3),
            "enrolled_labels": {lbl: e["n"] for lbl, e in self.enrolled.items()},
            "enrolled_hits": dict(self._last_hits),
            "events": self.timeline.recent(20),
            "recurring": self.vocab.types(10),
            "mean_ms": round(mean_ms, 2),
            "last_ms": round(last_ms, 2),
            "rtf": round((mean_ms / 1000.0) / max(self.step_s, 1e-6), 4),
        }
