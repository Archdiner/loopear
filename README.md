# LoopEar

Standalone implementation of LoopEar: a compact (~27M total parameters) evidence-grounded iterative auditory perception model built on a frozen 50 Hz `MIT-SLS/USAD2-Small` encoder (`24.76M` params, `384`-dim) and a shared recurrent refinement block (`2.22M` params) supporting multi-depth inference (`k in {1, 2, 4}`) from a single checkpoint.

## Layout

- `loopear/encoder.py`: Frozen `MIT-SLS/USAD2-Small` extractor, invariant checks, and content-addressed sharded fp16 feature cache.
- `loopear/models.py`: Comparison arms (`A_last`, `A_ml`, `B_comp`, `B_param`, `C_param`, `C_comp`, `C_wide`, `D`, `E`) with `ContinuousTimeRoPE`, cached encoder K/V, FiLM depth conditioning, configurable temporal compression (`stride in {1, 4, 8}`, `pool_mode in {avg, strided_conv, attn}`), and optional 12-layer evidence fusion.
- `loopear/losses.py`: Multi-depth event BCE + cross-chapter AAM-Softmax speaker loss, masked pooling, and deterministic seeding.
- `loopear/oracle.py`: H6 oracle depth-allocation knapsack headroom, overthinking rate, query-independent collapse check, and training-free AudioTETRIS heuristic.
- `loopear/harness.py`: Immutable SHA-256 run manifests (`ledger/runs/`), spend ledger (`ledger/spend.jsonl`, `$5.00` cap), sealed-split guard, and group-disjoint split checks.
- `loopear/stream.py` and `loopear/demo.py`: Continuous local inference (`LoopEarStream`) and `aiohttp` microphone/upload prototype (`/api/pcm`, `/api/depth`, `/api/enroll`, `/api/compare`, `/api/state`).
- `scripts/verify.py`: Enforces `pyflakes`, zero `#` comments in `.py` files, `README.md`-only repo docs, and `pytest`.
- `scripts/profile_arms.py`: Isolated subprocess profiler reporting params, 10s GFLOPs, 1-core and all-core CPU RTF, and Peak RSS split into encoder, refinement, readout, and end-to-end.

## Quickstart

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python scripts/verify.py
.venv/bin/python scripts/profile_arms.py --out results/stage0_profile.json
.venv/bin/python -m loopear.demo --host 127.0.0.1 --port 8765
```
