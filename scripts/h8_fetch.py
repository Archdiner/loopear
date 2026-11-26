import shutil
import tempfile
from pathlib import Path

from loopear.data import prepare_ami_sdm, prepare_dipco, prepare_realdesed, prepare_starss23
from loopear.harness import SealedSplitGuard
from loopear.h8data import DEV_CORPORA, audio_ready, load_manifest


def main() -> int:
    root = Path(".").resolve()
    raw = root / "data" / "raw"
    if audio_ready(root, DEV_CORPORA):
        print("dev audio already present", flush=True)
        return 0
    guard = SealedSplitGuard(("realdesed_test",))
    guard.check_access("realdesed_train")
    guard.check_access("realdesed_val")
    with tempfile.TemporaryDirectory(prefix="h8_manifests_") as tmp:
        manifest_dir = Path(tmp)
        print("fetch starss23", flush=True)
        prepare_starss23(raw, manifest_dir)
        print("fetch dipco", flush=True)
        prepare_dipco(raw, manifest_dir)
        print("fetch ami_sdm", flush=True)
        prepare_ami_sdm(raw, manifest_dir)
        print("fetch realdesed train/val", flush=True)
        prepare_realdesed(raw, manifest_dir, sealed_guard=guard)
        shutil.rmtree(manifest_dir, ignore_errors=True)
    for corpus in DEV_CORPORA:
        manifest = load_manifest(root, corpus)
        missing = [
            item["audio_path"]
            for item in manifest["items"]
            if not (root / item["audio_path"]).exists()
        ]
        print(f"{corpus}: items={len(manifest['items'])} missing={len(missing)}", flush=True)
        if missing:
            print("missing example", missing[0], flush=True)
            return 1
    print("dev audio ready", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
