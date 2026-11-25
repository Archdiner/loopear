# LoopEar

Standalone implementation of LoopEar: a compact (~27M total parameters) evidence-grounded iterative auditory perception model built on a frozen 50 Hz `MIT-SLS/USAD2-Small` encoder (`24.76M` params, `384`-dim) and a shared recurrent refinement block (`2.22M` params) supporting multi-depth inference (`k in {1, 2, 4}`) from a single checkpoint.

## Layout

- `loopear/encoder.py`: Frozen `MIT-SLS/USAD2-Small` extractor, invariant checks, FOA/channel-selective mono loading, and content-addressed sharded fp16 feature cache.
- `loopear/data.py`: Zero-USAD2-overlap real-world dataset loaders (`STARSS23`, `RealDESED`, `DiPCo`, `AMI SDM`, `CREMA-D`, `VocalSet`, `MAESTRO Real`), HTTP-Range selective Zenodo ZIP extraction, sealed-split & group-disjoint manifest validation, and 50 Hz frame label rasterization/resampling.
- `loopear/models.py`: Comparison arms (`A_last`, `A_ml`, `B_comp`, `B_param`, `C_param`, `C_comp`, `C_wide`, `D`, `E`) with `ContinuousTimeRoPE`, cached encoder K/V, FiLM depth conditioning, configurable temporal compression (`stride in {1, 4, 8}`, `pool_mode in {avg, strided_conv, attn}`), and optional 12-layer evidence fusion.
- `loopear/losses.py`: Multi-depth event BCE (`RealDESED`, `STARSS23`), 3-class overlapped-speech CE (`DiPCo`), cross-meeting AAM-Softmax speaker loss (`AMI SDM`), masked pooling, and deterministic seeding.
- `loopear/oracle.py`: H6 oracle depth-allocation knapsack headroom, overthinking rate, query-independent collapse check, and training-free AudioTETRIS heuristic.
- `loopear/harness.py`: Immutable SHA-256 run manifests (`ledger/runs/`), spend ledger (`ledger/spend.jsonl`, `$5.00` cap), sealed-split guard, and group-disjoint split checks.
- `loopear/stream.py` and `loopear/demo.py`: Continuous local inference (`LoopEarStream`) and `aiohttp` microphone/upload prototype (`/api/pcm`, `/api/depth`, `/api/enroll`, `/api/compare`, `/api/state`).
- `scripts/verify.py`: Enforces `pyflakes`, zero `#` comments in `.py` files, `README.md`-only repo docs, and `pytest`.
- `scripts/profile_arms.py`: Isolated subprocess profiler reporting params, 10s GFLOPs, 1-core and all-core CPU RTF, and Peak RSS split into encoder, refinement, readout, and end-to-end.
- `scripts/prepare_data.py`: Downloads, validates (`join_coverage >= 0.99`, group-disjoint, 0% USAD2 overlap, `realdesed_test` sealed), writes `data/manifests/*.json`, and extracts sharded fp16 `USAD2-Small` features.
- `scripts/train_pilot.py` and `scripts/eval_transfer.py`: Trains and evaluates the Stage 1 isolating arms, paired bootstrap 95% CIs, H6 oracle headroom, frozen transfer probes with shuffled-label controls, and real-audio `LoopEarStream` verification.

## Status

- **Stage 0** (`results/stage0_profile.json`): Verified `USAD2-Small` 4-core CPU RTF `0.0123` (`44 s/h`) and `E_s8@4` refinement `6.67 ms`/10s.
- **Stage 1 Real-Audio Pilot** (`results/stage1_pilot.json`, `$0.00` cloud GPU spend, CPU-only):
  - **Data (`9.11 h` total, 0% USAD2 overlap, `45.85 s/h` CPU extraction)**: Train/Val (`4.31 h` train / `1.87 h` val) across `STARSS23`, `RealDESED` (`test` sealed), `DiPCo`, and `AMI SDM`; zero-overlap frozen transfer (`2.94 h`) across `CREMA-D`, `VocalSet`, and `MAESTRO Real`.
  - **Dev Composite (4-task equal-weight, `k_max`)**: `B_comp` `0.6151` > `E_s8_conv@4` `0.6069` > `A_ml` `0.5938` > `D@4` `0.5936` > `E_s8@4` `0.5895` (`+1.33 pp` vs `k=1`) > `E_s4@4` `0.5851` (`+1.38 pp` vs `k=1`) > `C_param@4` `0.5618` > `A_last` `0.5568`.
  - **Zero-Overlap Frozen Transfer Composite (`k_max`, shuffled controls at chance)**: `E_s8@4` `0.6793` (`+1.96 pp` vs `k=1`) > `E_s4@4` `0.6787` > `D@4` `0.6745` > `A_ml` `0.6595` > `C_param@4` `0.6516` > `A_last` `0.6512` > `B_comp` `0.6467` > `E_s8_conv@4` `0.6427`.
  - **H6 Oracle Headroom**: `E_s8` `48.1%` savings (`improved=31.9%`, `overthinking=28.3%`, `h6_go=True`), `E_s4` `57.5%` savings (`improved=32.9%`, `overthinking=29.7%`, `h6_go=True`), non-collapsed (`cos=0.62–0.64`).
  - **Live `LoopEarStream` (`E_s8@4`, real val WAVs)**: `46.19 ms`/window (`RTF 0.0462`), cross-meeting same-speaker `speaker_cos=0.746` vs different-speaker `0.665`, enrolled-scene hit `0.688` vs `0.552`.

## Quickstart

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python scripts/verify.py
.venv/bin/python scripts/profile_arms.py --out results/stage0_profile.json
.venv/bin/python scripts/prepare_data.py --batch-size 8 --threads 4
.venv/bin/python scripts/train_pilot.py --epochs 10 --steps-per-epoch 28 --batch-size 16 --seed 42
.venv/bin/python -m loopear.demo --ckpt checkpoints/stage1_pilot_best.pt --host 127.0.0.1 --port 8765
```
