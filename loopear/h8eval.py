import time
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score

from loopear.data import compute_onset_and_boundary_metrics
from loopear.encoder import SR
from loopear.h8data import DEV_CORPORA, frame_targets, load_manifest, load_wav, window_spans
from loopear.losses import compute_cross_chapter_eer

EncodeFn = Callable[[torch.Tensor], torch.Tensor]
PROBE_STEPS: int = 1200
PROBE_BATCH: int = 2048
PROBE_LR: float = 1e-2
PROBE_SEED: int = 0


def to_rate(frames: np.ndarray, n_out: int) -> np.ndarray:
    if n_out <= 0:
        return np.zeros((0, frames.shape[-1]), dtype=np.float32)
    if frames.shape[0] == n_out:
        return np.ascontiguousarray(frames, dtype=np.float32)
    if frames.shape[0] == 0:
        return np.zeros((n_out, frames.shape[-1]), dtype=np.float32)
    if frames.shape[0] == 1:
        return np.repeat(frames.astype(np.float32), n_out, axis=0)
    tensor = torch.from_numpy(np.ascontiguousarray(frames, dtype=np.float32))
    tensor = tensor.transpose(0, 1).unsqueeze(0)
    resized = F.interpolate(tensor, size=n_out, mode="linear", align_corners=True)
    return resized.squeeze(0).transpose(0, 1).contiguous().numpy()


def n_frames_for(duration_s: float, hop_s: float = 0.02) -> int:
    return max(1, int(round(float(duration_s) / hop_s)))


def _pos_weight(y: torch.Tensor) -> torch.Tensor:
    pos = y.sum(dim=0)
    neg = y.shape[0] - pos
    weight = torch.ones_like(pos)
    active = pos > 0
    weight[active] = (neg[active] / pos[active]).clamp(max=20.0)
    return weight


def fit_multilabel(
    x_train: np.ndarray,
    y_train: np.ndarray,
    steps: int = PROBE_STEPS,
    batch: int = PROBE_BATCH,
    lr: float = PROBE_LR,
    seed: int = PROBE_SEED,
) -> dict[str, np.ndarray]:
    xt = torch.from_numpy(np.ascontiguousarray(x_train, dtype=np.float32))
    yt = torch.from_numpy(np.ascontiguousarray(y_train, dtype=np.float32))
    weight = torch.zeros(xt.shape[1], yt.shape[1], dtype=torch.float32, requires_grad=True)
    bias = torch.zeros(yt.shape[1], dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([weight, bias], lr=lr)
    pos_weight = _pos_weight(yt)
    generator = torch.Generator()
    generator.manual_seed(seed)
    n = int(xt.shape[0])
    take = min(batch, n)
    for _ in range(steps):
        idx = torch.randint(0, n, (take,), generator=generator)
        logit = xt[idx] @ weight + bias
        loss = F.binary_cross_entropy_with_logits(logit, yt[idx], pos_weight=pos_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return {
        "weight": weight.detach().cpu().numpy(),
        "bias": bias.detach().cpu().numpy(),
        "kind": np.array(["multilabel"]),
    }


def fit_multiclass(
    x_train: np.ndarray,
    y_train: np.ndarray,
    n_classes: int,
    steps: int = PROBE_STEPS,
    batch: int = PROBE_BATCH,
    lr: float = PROBE_LR,
    seed: int = PROBE_SEED,
) -> dict[str, np.ndarray]:
    xt = torch.from_numpy(np.ascontiguousarray(x_train, dtype=np.float32))
    yt = torch.from_numpy(np.ascontiguousarray(y_train, dtype=np.int64))
    weight = torch.zeros(xt.shape[1], n_classes, dtype=torch.float32, requires_grad=True)
    bias = torch.zeros(n_classes, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([weight, bias], lr=lr)
    generator = torch.Generator()
    generator.manual_seed(seed)
    n = int(xt.shape[0])
    take = min(batch, n)
    for _ in range(steps):
        idx = torch.randint(0, n, (take,), generator=generator)
        logit = xt[idx] @ weight + bias
        loss = F.cross_entropy(logit, yt[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return {
        "weight": weight.detach().cpu().numpy(),
        "bias": bias.detach().cpu().numpy(),
        "kind": np.array(["multiclass"]),
    }


def apply_probe(probe: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    xt = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    weight = torch.from_numpy(probe["weight"])
    bias = torch.from_numpy(probe["bias"])
    logit = xt @ weight + bias
    if str(probe["kind"][0]) == "multiclass":
        return torch.softmax(logit, dim=-1).detach().cpu().numpy()
    return torch.sigmoid(logit).detach().cpu().numpy()


def macro_average_precision(y: np.ndarray, p: np.ndarray) -> float | None:
    active = [c for c in range(y.shape[1]) if float(y[:, c].sum()) > 0.0]
    if not active:
        return None
    scores = [float(average_precision_score(y[:, c], p[:, c])) for c in active]
    return float(np.mean(scores))


def collect_split(
    encode: EncodeFn,
    root: str,
    corpus: str,
    split: str,
    win_s: float = 10.0,
    hop_s: float = 10.0,
) -> dict[str, Any]:
    manifest = load_manifest(root, corpus)
    frames: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    pooled: list[np.ndarray] = []
    speakers: list[str] = []
    meetings: list[str] = []
    native_hops: list[float] = []
    audio_s = 0.0
    n_windows = 0
    with torch.no_grad():
        for item in manifest["items"]:
            if str(item["split"]) != split:
                continue
            spans = window_spans(float(item["duration_s"]), win_s, hop_s, 10_000)
            for start, end in spans:
                wav_np = load_wav(root, item, start, end)
                if wav_np.size < int(0.1 * SR):
                    continue
                wav = torch.from_numpy(np.ascontiguousarray(wav_np)).unsqueeze(0)
                native = encode(wav)[0].detach().cpu().numpy().astype(np.float32)
                dur = float(wav_np.shape[0]) / float(SR)
                audio_s += dur
                n_windows += 1
                if native.shape[0] > 0:
                    native_hops.append(dur / float(native.shape[0]))
                    pooled.append(native.mean(axis=0))
                    if corpus == "ami_sdm":
                        speakers.append(str(item["speaker_id"]))
                        meetings.append(str(item["meeting_id"]))
                n_out = n_frames_for(dur)
                target = frame_targets(corpus, item, start, n_out, root)
                if target is not None:
                    frames.append(to_rate(native, n_out))
                    labels.append(target)
    out: dict[str, Any] = {
        "audio_s": audio_s,
        "native_hop_s": float(np.median(native_hops)) if native_hops else None,
        "n_windows": n_windows,
    }
    if frames:
        out["x"] = np.concatenate(frames, axis=0)
    if labels:
        out["y"] = np.concatenate(labels, axis=0)
        out["y_windows"] = labels
        out["x_windows"] = frames
    if pooled and corpus == "ami_sdm":
        out["embed"] = np.stack(pooled, axis=0)
        out["speaker_id"] = np.array(speakers)
        out["meeting_id"] = np.array(meetings)
    return out


def dev_readout(bundle: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    starss_probe = fit_multilabel(bundle["starss23"]["train"]["x"], bundle["starss23"]["train"]["y"])
    real_probe = fit_multilabel(bundle["realdesed"]["train"]["x"], bundle["realdesed"]["train"]["y"])
    dipco_y = np.argmax(bundle["dipco"]["train"]["y"], axis=1).astype(np.int64)
    dipco_probe = fit_multiclass(bundle["dipco"]["train"]["x"], dipco_y, n_classes=3)
    st_p = apply_probe(starss_probe, bundle["starss23"]["val"]["x"])
    rd_p = apply_probe(real_probe, bundle["realdesed"]["val"]["x"])
    dp_p = apply_probe(dipco_probe, bundle["dipco"]["val"]["x"])
    st_y = bundle["starss23"]["val"]["y"]
    rd_y = bundle["realdesed"]["val"]["y"]
    dp_y = np.argmax(bundle["dipco"]["val"]["y"], axis=1).astype(np.int64)
    st_map = macro_average_precision(st_y, st_p)
    rd_map = macro_average_precision(rd_y, rd_p)
    dp_f1 = float(f1_score(dp_y, np.argmax(dp_p, axis=1), average="macro", zero_division=0))
    eer = compute_cross_chapter_eer(
        bundle["ami_sdm"]["val"]["embed"],
        bundle["ami_sdm"]["val"]["speaker_id"],
        bundle["ami_sdm"]["val"]["meeting_id"],
    )
    st_windows_p = [apply_probe(starss_probe, x) for x in bundle["starss23"]["val"]["x_windows"]]
    rd_windows_p = [apply_probe(real_probe, x) for x in bundle["realdesed"]["val"]["x_windows"]]
    onset = compute_onset_and_boundary_metrics(
        bundle["starss23"]["val"]["y_windows"] + bundle["realdesed"]["val"]["y_windows"],
        st_windows_p + rd_windows_p,
    )
    ami_score = 1.0 - float(eer["eer"])
    composite = 0.25 * (
        float(st_map or 0.0) + float(rd_map or 0.0) + dp_f1 + ami_score
    )
    return {
        "starss_macro_ap": None if st_map is None else round(st_map, 4),
        "realdesed_macro_ap": None if rd_map is None else round(rd_map, 4),
        "dipco_macro_f1": round(dp_f1, 4),
        "ami_eer": round(float(eer["eer"]), 4),
        "ami_1_minus_eer": round(ami_score, 4),
        "onset_f1_100ms": round(float(onset["onset_f1_100ms"]), 4),
        "composite": round(composite, 4),
        "probes": {
            "starss23": starss_probe,
            "realdesed": real_probe,
            "dipco": dipco_probe,
        },
    }


def evaluate_encoder(
    encode: EncodeFn,
    root: str,
    corpora: tuple[str, ...] = DEV_CORPORA,
) -> dict[str, Any]:
    bundle: dict[str, dict[str, dict[str, Any]]] = {}
    hops = []
    audio_s = 0.0
    for corpus in corpora:
        bundle[corpus] = {}
        for split in ("train", "val"):
            got = collect_split(encode, root, corpus, split)
            bundle[corpus][split] = got
            audio_s += float(got["audio_s"])
            if got["native_hop_s"] is not None:
                hops.append(float(got["native_hop_s"]))
            print(
                f"encoded {corpus} {split} windows={got['n_windows']} audio_s={got['audio_s']:.1f}",
                flush=True,
            )
    metrics = dev_readout(bundle)
    probes = metrics.pop("probes")
    metrics["audio_s"] = round(audio_s, 1)
    metrics["native_hop_s"] = None if not hops else round(float(np.median(hops)), 4)
    metrics["probes"] = probes
    return metrics


def time_encode(
    encode: EncodeFn,
    threads: int,
    seconds: float = 10.0,
    repeats: int = 5,
    warmup: int = 1,
) -> dict[str, float]:
    torch.set_num_threads(int(threads))
    wav = torch.randn(1, int(round(seconds * SR)))
    with torch.no_grad():
        for _ in range(warmup):
            encode(wav).sum().item()
        times: list[float] = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            encode(wav).sum().item()
            times.append(time.perf_counter() - t0)
    med = float(np.median(times))
    return {
        "threads": float(threads),
        "median_s": med,
        "rtf": med / float(seconds),
        "audio_s": float(seconds),
    }


def mixture_retention(
    encode: EncodeFn,
    root: str,
    n_pairs: int = 8,
    seed: int = 1,
    crop_s: float = 4.0,
    gain_db: float = -12.0,
    probes: dict[str, dict[str, np.ndarray]] | None = None,
) -> dict[str, float]:
    from loopear.tiny import mix_waveforms

    rows: list[dict[str, Any]] = []
    for corpus in ("starss23", "realdesed", "dipco"):
        manifest = load_manifest(root, corpus)
        for item in manifest["items"]:
            if str(item["split"]) != "val" or float(item["duration_s"]) < crop_s:
                continue
            row = dict(item)
            row["corpus"] = corpus
            rows.append(row)
    rng = np.random.default_rng(seed)
    quiet_cos: list[float] = []
    loud_cos: list[float] = []
    quiet_ap: list[float] = []
    used = 0
    guard = 0
    with torch.no_grad():
        while used < n_pairs and guard < n_pairs * 50:
            guard += 1
            i, j = (int(v) for v in rng.integers(0, len(rows), size=2))
            loud, quiet = rows[i], rows[j]
            if (loud["corpus"], str(loud["group_id"])) == (quiet["corpus"], str(quiet["group_id"])):
                continue
            start_l = float(rng.random()) * (float(loud["duration_s"]) - crop_s)
            start_q = float(rng.random()) * (float(quiet["duration_s"]) - crop_s)
            wav_l = torch.from_numpy(load_wav(root, loud, start_l, start_l + crop_s)).unsqueeze(0)
            wav_q = torch.from_numpy(load_wav(root, quiet, start_q, start_q + crop_s)).unsqueeze(0)
            n = min(int(wav_l.shape[-1]), int(wav_q.shape[-1]))
            mixed = mix_waveforms(wav_l[:, :n], wav_q[:, :n], torch.tensor([gain_db]))
            emb_m = encode(mixed)[0].mean(dim=0).cpu().numpy()
            emb_l = encode(wav_l[:, :n])[0].mean(dim=0).cpu().numpy()
            emb_q = encode(wav_q[:, :n])[0].mean(dim=0).cpu().numpy()
            loud_cos.append(cosine(emb_m, emb_l))
            quiet_cos.append(cosine(emb_m, emb_q))
            if probes is not None and quiet["corpus"] in probes:
                n_out = n_frames_for(n / float(SR))
                feat = to_rate(encode(mixed)[0].cpu().numpy(), n_out)
                pred = apply_probe(probes[quiet["corpus"]], feat)
                target = frame_targets(quiet["corpus"], quiet, start_q, n_out, root)
                score = None if target is None else macro_average_precision(target, pred)
                if score is not None:
                    quiet_ap.append(score)
            used += 1
    return {
        "n_pairs": float(used),
        "gain_db": float(gain_db),
        "cosine_to_loud": float(np.mean(loud_cos)) if loud_cos else 0.0,
        "cosine_to_quiet": float(np.mean(quiet_cos)) if quiet_cos else 0.0,
        "quiet_label_ap": float(np.mean(quiet_ap)) if quiet_ap else 0.0,
    }


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    av = a.reshape(-1).astype(np.float64)
    bv = b.reshape(-1).astype(np.float64)
    denom = np.linalg.norm(av) * np.linalg.norm(bv)
    if denom <= 0.0:
        return 0.0
    return float(np.dot(av, bv) / denom)
