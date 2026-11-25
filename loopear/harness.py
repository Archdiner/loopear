import hashlib
import json
import os
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SPEND_CAP_USD: float = 5.0

USAD2_SMALL_PRETRAINING_SOURCES: frozenset[str] = frozenset(
    {
        "librivox",
        "librispeech",
        "librilight",
        "mls_en",
        "mls_non_en",
        "mls",
        "common_voice_17",
        "voxpopuli",
        "voxpopuli_en",
        "mms_v2",
        "gigaspeech",
        "voxlingua107",
        "fisher",
        "babel",
        "fleurs",
        "aishell1",
        "chime3",
        "zeroth_korean",
        "expresso",
        "soundnet",
        "audioset",
        "audioset_strong",
        "laion_audio_630k",
        "vggsound",
        "fma",
        "mtg_jamendo",
        "music4all",
        "nsynth",
        "magnatagatune",
        "piano_maestro",
    }
)

USAD2_SMALL_FREESOUND_OVERLAP_SOURCES: frozenset[str] = frozenset(
    {
        "fsd50k",
        "esc50",
        "urbansound8k",
        "clotho",
        "nigens_freesound",
    }
)


def _normalize_source_name(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def assert_zero_pretraining_overlap(
    eval_sources: Iterable[str],
    pretraining_sources: Iterable[str] = USAD2_SMALL_PRETRAINING_SOURCES,
    indirect_overlap_sources: Iterable[str] = USAD2_SMALL_FREESOUND_OVERLAP_SOURCES,
    block_indirect: bool = True,
    encoder_name: str = "MIT-SLS/USAD2-Small",
) -> None:
    norm_eval = {_normalize_source_name(s) for s in eval_sources}
    norm_direct = {_normalize_source_name(s) for s in pretraining_sources}
    norm_indirect = {_normalize_source_name(s) for s in indirect_overlap_sources}
    direct_hits = sorted(norm_eval.intersection(norm_direct))
    if direct_hits:
        raise AssertionError(
            f"Pretraining leakage for {encoder_name}: direct overlap in held-out eval {direct_hits}"
        )
    indirect_hits = sorted(norm_eval.intersection(norm_indirect))
    if block_indirect and indirect_hits:
        raise AssertionError(
            f"Indirect Freesound pretraining overlap for {encoder_name}: {indirect_hits}"
        )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_sha256(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def write_run_manifest(
    run_id: str,
    config: dict[str, Any],
    ledger_dir: Path | str = "ledger",
) -> Path:
    runs_dir = Path(ledger_dir) / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"{run_id}.json"
    config_sha = _canonical_sha256(config)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("config_sha256") != config_sha:
            raise FileExistsError(
                f"Immutable run manifest {path} already exists with a different config"
            )
        return path
    record = {
        "run_id": run_id,
        "created_utc": _utc_now_iso(),
        "config_sha256": config_sha,
        "config": config,
    }
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path


class SpendLedger:
    def __init__(
        self,
        path: Path | str = Path("ledger") / "spend.jsonl",
        cap_usd: float = SPEND_CAP_USD,
    ):
        self.path = Path(path)
        self.cap_usd = float(cap_usd)

    def entries(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows

    def total_usd(self) -> float:
        return float(sum(float(r["usd"]) for r in self.entries()))

    def check_budget(self, estimated_usd: float) -> None:
        if estimated_usd < 0.0:
            raise ValueError("estimated_usd must be non-negative")
        projected = self.total_usd() + float(estimated_usd)
        if projected > self.cap_usd + 1e-9:
            raise RuntimeError(
                f"Spend cap exceeded: projected ${projected:.4f} > cap ${self.cap_usd:.2f}"
            )

    def record(
        self,
        run_id: str,
        backend: str,
        usd: float,
        detail: str = "",
    ) -> dict[str, Any]:
        self.check_budget(usd)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts_utc": _utc_now_iso(),
            "run_id": run_id,
            "backend": backend,
            "usd": round(float(usd), 6),
            "detail": detail,
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, sort_keys=True) + "\n")
        return entry


class SealedSplitGuard:
    def __init__(
        self,
        sealed_splits: Iterable[str],
        audit_log: Path | str | None = None,
    ):
        self.sealed_splits = {str(s) for s in sealed_splits}
        self.audit_log = Path(audit_log) if audit_log is not None else None
        self._opened: set[str] = self._load_opened()

    def _load_opened(self) -> set[str]:
        if self.audit_log is None or not self.audit_log.exists():
            return set()
        opened: set[str] = set()
        for line in self.audit_log.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rec = json.loads(line)
                opened.add(str(rec["split"]))
        return opened

    def check_access(
        self,
        split_name: str,
        allow_sealed: bool = False,
        run_id: str = "",
    ) -> None:
        if split_name not in self.sealed_splits:
            return
        if not allow_sealed:
            raise PermissionError(
                f"Access blocked: split '{split_name}' is sealed"
            )
        if split_name in self._opened:
            raise PermissionError(
                f"Access blocked: sealed split '{split_name}' was already spent"
            )
        self._opened.add(split_name)
        if self.audit_log is not None:
            self.audit_log.parent.mkdir(parents=True, exist_ok=True)
            entry = {
                "ts_utc": _utc_now_iso(),
                "split": split_name,
                "run_id": run_id,
            }
            with self.audit_log.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, sort_keys=True) + "\n")


def assert_group_disjoint(
    train_groups: Iterable[str],
    eval_groups: Iterable[str],
    label: str = "split",
) -> None:
    overlap = sorted(set(train_groups).intersection(set(eval_groups)))
    if overlap:
        sample = overlap[:5]
        raise AssertionError(
            f"Group leakage in {label}: {len(overlap)} overlapping groups (e.g. {sample})"
        )


def write_result_json(path: Path | str, payload: dict[str, Any]) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    envelope = dict(payload)
    envelope.setdefault("written_utc", _utc_now_iso())
    envelope["payload_sha256"] = _canonical_sha256(payload)
    tmp = dest.with_name(f".{dest.name}.tmp.{os.getpid()}")
    try:
        tmp.write_text(
            json.dumps(envelope, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(tmp, dest)
    finally:
        if tmp.exists():
            tmp.unlink()
    return dest
