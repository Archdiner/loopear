import importlib.util
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from loopear.data import prepare_cremad, prepare_maestro_real, prepare_vocalset, sha256_file
from loopear.h8data import load_manifest, load_wav
from loopear.h8eval import n_frames_for
from loopear.harness import write_result_json
from loopear.tiny import TinyEncoder


def _transfer_helpers():
    path = Path(__file__).resolve().parent / "eval_transfer.py"
    spec = importlib.util.spec_from_file_location("h8_eval_transfer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._fit_linear_multiclass, module._eval_maestro_segments

TRANSFER = ("crema_d", "vocalset", "maestro_real")
ARMS = ("distill", "mixture", "averaged", "align")


def _sha_bad(root: Path, corpus: str) -> list[str]:
    manifest = load_manifest(root, corpus)
    return [
        str(item["id"])
        for item in manifest["items"]
        if not (root / item["audio_path"]).exists()
        or sha256_file(root / item["audio_path"]) != item["sha256"]
    ]


def _prepare(root: Path) -> None:
    if all(not _sha_bad(root, corpus) for corpus in TRANSFER):
        print("transfer audio present", flush=True)
        return
    raw = root / "data" / "raw"
    with tempfile.TemporaryDirectory(prefix="h8_transfer_") as tmp:
        manifest_dir = Path(tmp)
        print("fetch crema_d", flush=True)
        prepare_cremad(raw, manifest_dir)
        print("fetch vocalset", flush=True)
        prepare_vocalset(raw, manifest_dir)
        print("fetch maestro_real", flush=True)
        prepare_maestro_real(raw, manifest_dir)
    for corpus in TRANSFER:
        bad = _sha_bad(root, corpus)
        print(f"{corpus} sha_bad={len(bad)}", flush=True)
        if bad:
            raise RuntimeError(f"{corpus} sha mismatch {bad[:3]}")


def _load_student(path: Path) -> TinyEncoder:
    model = TinyEncoder()
    model.load_state_dict(load_file(path), strict=True)
    model.eval()
    return model


def _clip_embed(model: TinyEncoder, root: Path, item: dict) -> np.ndarray:
    wav_np = load_wav(root, item, 0.0, float(item["duration_s"]))
    wav = torch.from_numpy(np.ascontiguousarray(wav_np)).unsqueeze(0)
    with torch.no_grad():
        frames = model.encode(wav)[0]
    return frames.mean(dim=0).cpu().numpy()


def _frames(model: TinyEncoder, root: Path, item: dict) -> np.ndarray:
    wav_np = load_wav(root, item, 0.0, float(item["duration_s"]))
    wav = torch.from_numpy(np.ascontiguousarray(wav_np)).unsqueeze(0)
    with torch.no_grad():
        frames = model.encode(wav)[0].cpu().numpy()
    return frames


def _classify(model: TinyEncoder, root: Path, corpus: str, label_key: str, fit) -> dict:
    manifest = load_manifest(root, corpus)
    train_x, train_y, eval_x, eval_y = [], [], [], []
    for item in manifest["items"]:
        vec = _clip_embed(model, root, item)
        label = int(item[label_key])
        if item["split"] == "probe_train":
            train_x.append(vec)
            train_y.append(label)
        elif item["split"] == "probe_eval":
            eval_x.append(vec)
            eval_y.append(label)
    return fit(
        np.stack(train_x),
        np.array(train_y),
        np.stack(eval_x),
        np.array(eval_y),
    )


def _maestro(model: TinyEncoder, root: Path, evaluate) -> dict:
    manifest = load_manifest(root, "maestro_real")
    by_id = {item["id"]: item for item in manifest["items"]}
    tr_items, tr_frames, ev_items, ev_frames = [], [], [], []
    for item in manifest["items"]:
        raw = _frames(model, root, item)
        n_out = n_frames_for(float(item["duration_s"]))
        if raw.shape[0] != n_out:
            grid = torch.from_numpy(raw).transpose(0, 1).unsqueeze(0)
            raw = (
                torch.nn.functional.interpolate(grid, size=n_out, mode="linear", align_corners=True)
                .squeeze(0)
                .transpose(0, 1)
                .numpy()
            )
        packed = {"source_id": item["id"], "start_s": 0.0}
        if item["split"] == "probe_train":
            tr_items.append(packed)
            tr_frames.append(raw)
        elif item["split"] == "probe_eval":
            ev_items.append(packed)
            ev_frames.append(raw)
    return evaluate(tr_items, tr_frames, ev_items, ev_frames, by_id)


def main() -> int:
    t0 = time.perf_counter()
    root = Path(".").resolve()
    _prepare(root)
    fit, evaluate = _transfer_helpers()
    rows = []
    for arm in ARMS:
        print("transfer", arm, flush=True)
        model = _load_student(root / "results" / f"h8_{arm}.safetensors")
        crema = _classify(model, root, "crema_d", "emotion_idx", fit)
        vocal = _classify(model, root, "vocalset", "technique_idx", fit)
        maestro = _maestro(model, root, evaluate)
        composite = (crema["acc"] + vocal["acc"] + maestro["macro_map"]) / 3.0
        rows.append(
            {
                "arm": arm,
                "crema_acc": crema["acc"],
                "crema_shuffled_acc": crema["shuffled_acc"],
                "vocalset_acc": vocal["acc"],
                "vocalset_shuffled_acc": vocal["shuffled_acc"],
                "maestro_macro_ap": maestro["macro_map"],
                "maestro_shuffled_macro_ap": maestro["shuffled_macro_map"],
                "composite": round(float(composite), 4),
            }
        )
        del model
    payload = {
        "gpu_usd": 0.0,
        "wall_s": round(time.perf_counter() - t0, 1),
        "used_to_choose_arm": False,
        "arms": rows,
    }
    write_result_json(root / "results" / "h8_transfer.json", payload)
    print("transfer written", payload["wall_s"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
