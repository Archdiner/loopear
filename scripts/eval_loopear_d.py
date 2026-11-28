import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

from loopear.data import STARSS23_CLASSES, rasterize_starss_frames_50hz, sha256_file
from loopear.d.diagnostics import cpu_seconds, embedding_diagnostics, hours
from loopear.d.frontend import build_mel, load_frontend_config, waveform_logmel
from loopear.d.fsd import audio_path, items_for, load_manifest
from loopear.d.loop import init_model
from loopear.d.probe import encode_logmel_windows, fit_binary, fit_linear_probe, fixed_project
from loopear.d.spec import CROP_SECONDS, D_MODEL, GRID_T, N_FRAMES, N_MELS
from loopear.encoder import load_mono_audio
from loopear.harness import write_result_json
from safetensors.torch import load_file


def _primary(labels: list[str]) -> str:
    return labels[0] if labels else ""


def _multi_hot(rows: list[dict[str, Any]], classes: list[str]) -> np.ndarray:
    index = {name: i for i, name in enumerate(classes)}
    out = np.zeros((len(rows), len(classes)), dtype=np.float32)
    for row_i, row in enumerate(rows):
        for label in row["labels"]:
            if label in index:
                out[row_i, index[label]] = 1.0
    return out


def _load_encoder(path: Path):
    model = init_model(0)
    state = load_file(str(path))
    model.encoder.load_state_dict(state)
    model.eval()
    return model


def _clip_features(model, rows, audio_root, mel, mean, std, max_windows: int) -> dict[str, torch.Tensor]:
    cls_rows = []
    patch_rows = []
    dsp_rows = []
    for row in rows:
        wav, sr = sf.read(str(audio_path(audio_root, row)), dtype="float32", always_2d=False)
        if int(sr) != 16000:
            raise AssertionError(f"sr {sr}")
        if wav.ndim != 1:
            wav = np.mean(wav, axis=-1)
        logmel = waveform_logmel(torch.from_numpy(np.ascontiguousarray(wav)), mel)
        encoded = encode_logmel_windows(model, logmel, mean, std, max_windows)
        cls_rows.append(encoded["cls"])
        patch_rows.append(encoded["patch_mean"])
        dsp_rows.append(encoded["dsp"])
    return {
        "cls": torch.stack(cls_rows),
        "patch_mean": torch.stack(patch_rows),
        "dsp": torch.stack(dsp_rows),
    }


def _probe_block(features, y_train, y_val, seed: int) -> dict[str, Any]:
    out = {}
    for name, matrix in features.items():
        x_train = fixed_project(matrix["train"], D_MODEL, seed)
        x_val = fixed_project(matrix["val"], D_MODEL, seed)
        result = fit_linear_probe(x_train, y_train, x_val, y_val, steps=250, lr=1e-2, seed=seed)
        perm_gen = torch.Generator().manual_seed(seed + 7)
        shuffled = y_train[torch.randperm(y_train.shape[0], generator=perm_gen)]
        control = fit_linear_probe(x_train, shuffled, x_val, y_val, steps=250, lr=1e-2, seed=seed)
        out[name] = {"probe": result, "shuffled_label_probe": control, "input_dim": int(matrix["train"].shape[1])}
    return out


def _few_shot(rows_train, rows_val, feat_train, feat_val, seed: int) -> dict[str, Any]:
    train_primary = {_primary(row["labels"]) for row in rows_train}
    by_class: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows_val):
        label = _primary(row["labels"])
        if label and label not in train_primary:
            by_class[label].append(index)
    usable = {name: idxs for name, idxs in by_class.items() if len(idxs) >= 6}
    if not usable:
        return {
            "status": "null",
            "reason": "no primary val label is absent from the 5h subset with at least 6 clips",
            "n_unfamiliar_primaries": len(by_class),
        }
    label = sorted(usable, key=lambda name: (-len(usable[name]), name))[0]
    positives = usable[label]
    negatives = [i for i in range(len(rows_val)) if i not in set(positives)]
    gen = np.random.default_rng(seed)
    pos_take = positives[:4]
    neg_take = list(gen.choice(negatives, size=4, replace=False))
    train_idx = pos_take + neg_take
    val_idx = [i for i in positives[4:]] + [i for i in negatives if i not in set(neg_take)][: max(8, len(positives))]
    y_train = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.float32)
    y_val = torch.tensor([1 if i in set(positives) else 0 for i in val_idx], dtype=torch.float32)
    x_train = fixed_project(feat_train[train_idx], D_MODEL, seed)
    x_val = fixed_project(feat_val[val_idx], D_MODEL, seed)
    scored = fit_binary(x_train, y_train, x_val, y_val, steps=200, lr=1e-2, seed=seed)
    return {"status": "measured", "label": label, "k_pos": 4, "k_neg": 4, **scored}


def _similar_pair(rows_val, feat_val, seed: int) -> dict[str, Any]:
    groups: dict[str, list[int]] = defaultdict(list)
    ancestors: dict[str, set[str]] = defaultdict(set)
    for index, row in enumerate(rows_val):
        primary = _primary(row["labels"])
        if not primary:
            continue
        groups[primary].append(index)
        ancestors[primary].update(row["labels"][1:])
    candidates = []
    names = [name for name, idxs in groups.items() if len(idxs) >= 6]
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            inter = ancestors[left] & ancestors[right]
            union = ancestors[left] | ancestors[right]
            if not union or len(inter) < 2:
                continue
            score = len(inter) / len(union)
            candidates.append((score, left, right))
    if not candidates:
        return {"status": "null", "reason": "no sibling primaries with 6 val clips"}
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    _score, left, right = candidates[0]
    left_idx = groups[left]
    right_idx = groups[right]
    train_idx = left_idx[:4] + right_idx[:4]
    val_idx = left_idx[4:12] + right_idx[4:12]
    if len(val_idx) < 4:
        return {"status": "null", "reason": "pair has too few held-out clips", "pair": [left, right]}
    y_train = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.float32)
    y_val = torch.tensor([1 if i in set(left_idx) else 0 for i in val_idx], dtype=torch.float32)
    scored = fit_binary(
        fixed_project(feat_val[train_idx], D_MODEL, seed),
        y_train,
        fixed_project(feat_val[val_idx], D_MODEL, seed),
        y_val,
        steps=200,
        lr=1e-2,
        seed=seed,
    )
    return {"status": "measured", "pair": [left, right], "jaccard_ancestors": candidates[0][0], **scored}


def _refit_scores(x_train, x_val, y_train, y_val, seed: int):
    torch.manual_seed(seed)
    x_tr = fixed_project(x_train, D_MODEL, seed)
    x_va = fixed_project(x_val, D_MODEL, seed)
    head = torch.nn.Linear(D_MODEL, int(y_train.shape[1]))
    opt = torch.optim.AdamW(head.parameters(), lr=1e-2, weight_decay=0.0)
    for _ in range(250):
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(head(x_tr), y_train)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return torch.sigmoid(head(x_va)).numpy()


def _overlap(rows_val, y_val, scores, classes) -> dict[str, Any]:
    primaries = { _primary(row["labels"]) for row in rows_val if row["labels"] }
    multi = []
    single = []
    for index, row in enumerate(rows_val):
        leaves = [label for label in row["labels"] if label in primaries]
        if len(set(leaves)) >= 2:
            multi.append(index)
        elif len(set(leaves)) == 1:
            single.append(index)
    from loopear.d.probe import macro_average_precision

    def _slice(indices):
        if len(indices) < 2:
            return None
        return macro_average_precision(y_val.numpy()[indices], scores[indices])

    return {
        "definition": "a leaf is any label that is the first label of some val clip; overlap means two such labels on one clip",
        "n_multi": len(multi),
        "n_single": len(single),
        "multi": _slice(multi),
        "single": _slice(single),
        "limit": "weak clip labels, not time-aligned polyphony",
        "classes": len(classes),
    }


def _parse_starss_csv(text: str) -> list[tuple[int, int]]:
    parsed = []
    for line in text.splitlines():
        cols = [part.strip() for part in line.split(",")]
        if len(cols) >= 2 and cols[0].isdigit():
            parsed.append((int(cols[0]), int(cols[1])))
    return parsed


def _onset_lift(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    inactive = scores[labels.sum(axis=1) == 0]
    background = float(inactive.mean()) if inactive.size else float(scores.mean())
    peaks = []
    above = 0
    for t in range(labels.shape[0]):
        for cls in range(labels.shape[1]):
            if labels[t, cls] < 0.5:
                continue
            if t > 0 and labels[t - 1, cls] >= 0.5:
                continue
            lo = max(0, t - 2)
            hi = min(labels.shape[0], t + 3)
            peak = float(scores[lo:hi, cls].max())
            peaks.append(peak - background)
            if peak > background:
                above += 1
    if not peaks:
        return {"n_onsets": 0, "mean_peak_minus_background": None, "fraction_above_background": None}
    return {
        "n_onsets": len(peaks),
        "mean_peak_minus_background": float(np.mean(peaks)),
        "fraction_above_background": float(above / len(peaks)),
        "background_mean": background,
        "tolerance_frames": 2,
        "tolerance_s": 0.08,
    }


def _starss(model, mel, mean, std, max_files: int) -> dict[str, Any]:
    from loopear.d.frontend import normalize_logmel
    from loopear.d.objective import broadcast_time_valid, patch_valid_from_frames

    manifest = json.loads(Path("data/manifests/starss23.json").read_text(encoding="utf-8"))
    chosen = [row for row in manifest["items"] if row["split"] == "val"][:max_files]
    if len(chosen) < 4:
        return {"status": "null", "reason": "need 4 val STARSS23 rows in the existing manifest"}
    from huggingface_hub import hf_hub_download

    clips = []
    for row in chosen:
        try:
            wav_src = hf_hub_download(
                "labhamlet/STARSS23", f"original_audios/{row['id']}.wav", repo_type="dataset"
            )
            csv_src = None
            last_error: Exception | None = None
            for prefix in ("labels/test/", "labels/valid/", "labels/"):
                try:
                    csv_src = hf_hub_download(
                        "labhamlet/STARSS23", f"{prefix}{row['id']}.csv", repo_type="dataset"
                    )
                    break
                except Exception as exc:
                    last_error = exc
            if csv_src is None:
                raise last_error if last_error is not None else FileNotFoundError(row["id"])
        except Exception as exc:
            return {"status": "fail", "reason": f"{type(exc).__name__}: {exc}", "id": row["id"]}
        wav = load_mono_audio(wav_src, target_sr=16000, channel=0)
        logmel = waveform_logmel(torch.from_numpy(wav), mel)
        events = _parse_starss_csv(Path(csv_src).read_text(encoding="utf-8"))
        n_frames = int(logmel.shape[0])
        starts = list(range(0, max(n_frames - N_FRAMES, 0) + 1, N_FRAMES))[:6]
        windows = []
        for start in starts:
            chunk = logmel[start : start + N_FRAMES]
            valid = torch.ones(N_FRAMES, dtype=torch.bool)
            if chunk.shape[0] < N_FRAMES:
                padded = torch.zeros(N_FRAMES, chunk.shape[1], dtype=chunk.dtype)
                padded[: chunk.shape[0]] = chunk
                valid[chunk.shape[0] :] = False
                chunk = padded
            norm = normalize_logmel(chunk, mean, std) * valid.unsqueeze(-1)
            visible = broadcast_time_valid(patch_valid_from_frames(valid.unsqueeze(0)))
            grid, cls = model.encoder(norm.unsqueeze(0), visible)
            patch = grid[0].mean(dim=1).detach().float().cpu()
            grouped = norm.view(GRID_T, 4, N_MELS)
            dsp = torch.cat(
                [grouped.mean(dim=1), grouped.std(dim=1, unbiased=False)], dim=-1
            ).detach().float().cpu()
            start_s = float(start) * 0.01
            labels = rasterize_starss_frames_50hz(
                events,
                start_s,
                GRID_T,
                len(STARSS23_CLASSES),
                hop_e=0.04,
                label_hop_s=0.1,
            )
            windows.append({"patch": patch, "dsp": dsp, "cls": cls[0].detach().float().cpu(), "y": labels})
        clips.append({"id": row["id"], "group": row["group_id"], "windows": windows})
    same = []
    diff = []
    for i in range(len(clips)):
        for j in range(i + 1, len(clips)):
            a = torch.nn.functional.normalize(clips[i]["windows"][0]["cls"], dim=0)
            b = torch.nn.functional.normalize(clips[j]["windows"][0]["cls"], dim=0)
            cos = float(torch.dot(a, b))
            if clips[i]["group"] == clips[j]["group"]:
                same.append(cos)
            else:
                diff.append(cos)
    train_clips = clips[:2]
    val_clips = clips[2:]

    def _stack(group, key):
        xs = [win[key] for clip in group for win in clip["windows"]]
        ys = [win["y"] for clip in group for win in clip["windows"]]
        return torch.cat(xs, dim=0), np.concatenate(ys, axis=0)

    x_tr, y_tr = _stack(train_clips, "patch")
    x_va, y_va = _stack(val_clips, "patch")
    d_tr, _ = _stack(train_clips, "dsp")
    d_va, _ = _stack(val_clips, "dsp")
    y_tr_t = torch.tensor(y_tr)
    y_va_t = torch.tensor(y_va)
    enc = fit_linear_probe(
        fixed_project(x_tr, D_MODEL, 42), y_tr_t, fixed_project(x_va, D_MODEL, 42), y_va_t, 250, 1e-2, 42
    )
    dsp = fit_linear_probe(
        fixed_project(d_tr, D_MODEL, 42), y_tr_t, fixed_project(d_va, D_MODEL, 42), y_va_t, 250, 1e-2, 42
    )
    torch.manual_seed(42)
    head = torch.nn.Linear(D_MODEL, y_tr_t.shape[1])
    opt = torch.optim.AdamW(head.parameters(), lr=1e-2, weight_decay=0.0)
    x_fit = fixed_project(x_tr, D_MODEL, 42)
    x_eval = fixed_project(x_va, D_MODEL, 42)
    for _ in range(250):
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(head(x_fit), y_tr_t)
        loss.backward()
        opt.step()
    with torch.no_grad():
        enc_scores = torch.sigmoid(head(x_eval)).numpy()
    torch.manual_seed(42)
    dsp_head = torch.nn.Linear(D_MODEL, y_tr_t.shape[1])
    dsp_opt = torch.optim.AdamW(dsp_head.parameters(), lr=1e-2, weight_decay=0.0)
    d_fit = fixed_project(d_tr, D_MODEL, 42)
    d_eval = fixed_project(d_va, D_MODEL, 42)
    for _ in range(250):
        dsp_opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(dsp_head(d_fit), y_tr_t)
        loss.backward()
        dsp_opt.step()
    with torch.no_grad():
        dsp_scores = torch.sigmoid(dsp_head(d_eval)).numpy()
    same_room_extra = None
    try:
        extra_wav = hf_hub_download(
            "labhamlet/STARSS23", "original_audios/fold4_room10_mix003.wav", repo_type="dataset"
        )
        extra_mel = waveform_logmel(torch.from_numpy(load_mono_audio(extra_wav, target_sr=16000, channel=0)), mel)
        extra_feat = encode_logmel_windows(model, extra_mel, mean, std, max_windows=1)
        anchor = clips[0]["windows"][0]["cls"]
        pair = torch.nn.functional.normalize(extra_feat["cls"], dim=0)
        base = torch.nn.functional.normalize(anchor, dim=0)
        same_room_extra = {
            "ids": [clips[0]["id"], "fold4_room10_mix003"],
            "cosine": float(torch.dot(base, pair)),
            "same_group": clips[0]["group"] == "room10",
        }
    except Exception as exc:
        same_room_extra = {"status": "fail", "reason": f"{type(exc).__name__}: {exc}"}
    return {
        "status": "measured",
        "files": [clip["id"] for clip in clips],
        "rooms": [clip["group"] for clip in clips],
        "train_files": [clip["id"] for clip in train_clips],
        "val_files": [clip["id"] for clip in val_clips],
        "n_train_frames": int(y_tr.shape[0]),
        "n_val_frames": int(y_va.shape[0]),
        "same_room_cosine_mean": float(np.mean(same)) if same else None,
        "different_room_cosine_mean": float(np.mean(diff)) if diff else None,
        "n_same_pairs": len(same),
        "n_diff_pairs": len(diff),
        "same_room_extra": same_room_extra,
        "frame_macro_ap": {"encoder": enc["val"], "dsp": dsp["val"]},
        "onset": {"encoder": _onset_lift(y_va, enc_scores), "dsp": _onset_lift(y_va, dsp_scores)},
        "limit": (
            "four fold4 files, at most six non-overlapping 2.015 s windows each; "
            "frame hop is the 40 ms patch; labels are 100 ms STARSS frames rasterized onto that hop; "
            "an onset is a 0-to-1 class change inside a window, so a window boundary can split an event; "
            "the committed STARSS sha256 is not checked because this render is a fresh channel-0 16 kHz copy"
        ),
        "manifest_rewritten": False,
    }


def _usad_probe(train_rows, val_rows, y_train, y_val, audio_root, seed: int, cap: int) -> dict[str, Any]:
    try:
        from loopear.encoder import load_usad2_extractor
    except Exception as exc:
        return {"status": "fail", "reason": f"import {type(exc).__name__}: {exc}"}
    try:
        encoder, _dtype = load_usad2_extractor("cpu")
    except Exception as exc:
        return {"status": "fail", "reason": f"load {type(exc).__name__}: {exc}"}
    def _embed(rows):
        vecs = []
        for row in rows[:cap]:
            wav, _sr = sf.read(str(audio_path(audio_root, row)), dtype="float32", always_2d=False)
            if wav.ndim != 1:
                wav = np.mean(wav, axis=-1)
            if wav.shape[0] > 16000 * 4:
                start = max((wav.shape[0] - 16000 * 4) // 2, 0)
                wav = wav[start : start + 16000 * 4]
            tensor = torch.from_numpy(np.ascontiguousarray(wav)).unsqueeze(0)
            with torch.no_grad():
                out = encoder(wavs=tensor)
            vecs.append(out["x"].float().mean(dim=1).squeeze(0).cpu())
        return torch.stack(vecs)
    try:
        x_train = _embed(train_rows)
        x_val = _embed(val_rows)
    except Exception as exc:
        return {"status": "fail", "reason": f"forward {type(exc).__name__}: {exc}"}
    n_train = x_train.shape[0]
    n_val = x_val.shape[0]
    result = fit_linear_probe(
        fixed_project(x_train, D_MODEL, seed),
        y_train[:n_train],
        fixed_project(x_val, D_MODEL, seed),
        y_val[:n_val],
        steps=250,
        lr=1e-2,
        seed=seed,
    )
    return {
        "status": "measured",
        "n_train": int(n_train),
        "n_val": int(n_val),
        "cap": cap,
        "probe": result,
        "limit": "USAD2-Small mean-pooled 4s center crop, then a fixed projection to 96 and the same linear probe",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--manifest", default="data/manifests/fsd50k_loopear_d.json")
    parser.add_argument("--frontend", default="data/manifests/loopear_d_frontend.json")
    parser.add_argument("--audio-root", default="data/raw/fsd50k")
    parser.add_argument("--data-report", default="results/loopear_d_data.json")
    parser.add_argument("--train-clips", type=int, default=256)
    parser.add_argument("--max-windows", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", required=True)
    parser.add_argument("--skip-usad", action="store_true")
    parser.add_argument("--skip-starss", action="store_true")
    args = parser.parse_args()
    cpu0 = cpu_seconds()
    wall0 = time.perf_counter()
    manifest = load_manifest(Path(args.manifest))
    data_report = json.loads(Path(args.data_report).read_text(encoding="utf-8"))
    val_ids = set(int(i) for i in data_report["val_ids"])
    train_rows = items_for(manifest, "h5")[: args.train_clips]
    val_rows = [row for row in manifest["items"] if int(row["freesound_id"]) in val_ids]
    for row in val_rows:
        if row["split"] != "val":
            raise PermissionError("val id resolved to a non-val row")
    for row in train_rows:
        if row["split"] != "train":
            raise PermissionError("probe train row is not train")
    frontend = load_frontend_config(Path(args.frontend))
    mean = torch.tensor(frontend["mean"], dtype=torch.float32)
    std = torch.tensor(frontend["std"], dtype=torch.float32)
    mel = build_mel()
    model = _load_encoder(Path(args.encoder))
    audio_root = Path(args.audio_root)
    train_feat = _clip_features(model, train_rows, audio_root, mel, mean, std, args.max_windows)
    val_feat = _clip_features(model, val_rows, audio_root, mel, mean, std, args.max_windows)
    counts = Counter()
    for row in train_rows + val_rows:
        counts.update(row["labels"])
    classes = sorted(label for label, n in counts.items() if n >= 3)
    y_train = torch.tensor(_multi_hot(train_rows, classes))
    y_val = torch.tensor(_multi_hot(val_rows, classes))
    keep = (y_train.sum(dim=0) >= 2) & (y_val.sum(dim=0) >= 2)
    y_train = y_train[:, keep]
    y_val = y_val[:, keep]
    classes = [name for name, flag in zip(classes, keep.tolist()) if flag]
    if not classes:
        raise SystemExit("no class has both train and val support in this probe split")
    packed = {name: {"train": train_feat[name], "val": val_feat[name]} for name in ("cls", "patch_mean", "dsp")}
    probes = _probe_block(packed, y_train, y_val, args.seed)
    cls_scores = _refit_scores(train_feat["cls"], val_feat["cls"], y_train, y_val, args.seed)
    report = {
        "task": "loopear_d_dev_eval",
        "encoder": args.encoder,
        "n_train_clips": len(train_rows),
        "n_val_clips": len(val_rows),
        "n_classes": len(classes),
        "val_label_prevalence": float(y_val.mean()) if y_val.numel() else None,
        "crop_s": CROP_SECONDS,
        "max_windows": args.max_windows,
        "window_limit": f"at most {args.max_windows} non-overlapping {N_FRAMES}-frame crops, mean pooled",
        "representation": {
            "cls": embedding_diagnostics(val_feat["cls"]),
            "patch_mean": embedding_diagnostics(val_feat["patch_mean"]),
        },
        "linear_probe": probes,
        "few_shot_unfamiliar": _few_shot(train_rows, val_rows, train_feat["cls"], val_feat["cls"], args.seed),
        "similar_sounds": _similar_pair(val_rows, val_feat["cls"], args.seed),
        "overlap": _overlap(val_rows, y_val, cls_scores, classes),
        "efficientat": {
            "status": "not_run",
            "reason": "no local EfficientAT checkpoint; a mismatched frontend would not be a fair readout",
        },
        "gpu_usd": 0.0,
        "sealed_realdesed_test_read": False,
        "fsd50k_test_audio_read": False,
    }
    if args.skip_usad:
        report["usad"] = {"status": "not_run", "reason": "skipped"}
    else:
        report["usad"] = _usad_probe(train_rows, val_rows, y_train, y_val, audio_root, args.seed, cap=24)
    if args.skip_starss:
        report["starss"] = {"status": "not_run", "reason": "skipped"}
    else:
        try:
            report["starss"] = _starss(model, mel, mean, std, max_files=4)
        except Exception as exc:
            report["starss"] = {"status": "fail", "reason": f"{type(exc).__name__}: {exc}"}
    report["wall_hours"] = hours(time.perf_counter() - wall0)
    report["cpu_hours"] = hours(cpu_seconds() - cpu0)
    report["audio_sha256_spotcheck"] = {
        "n": min(3, len(train_rows)),
        "sha256": [sha256_file(audio_path(audio_root, row)) for row in train_rows[:3]],
    }
    write_result_json(args.report, report)
    print(json.dumps({"n_val": len(val_rows), "n_classes": len(classes), "probe_cls": probes["cls"]["probe"]["val"]}, indent=2))


if __name__ == "__main__":
    main()
