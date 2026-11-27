# LoopEar

Standalone implementation of LoopEar: a compact (~27M total parameters) evidence-grounded iterative auditory perception model built on a frozen 50 Hz `MIT-SLS/USAD2-Small` encoder (`24.76M` params, `384`-dim) and a shared recurrent refinement block (`2.22M` params) supporting multi-depth inference (`k in {1, 2, 4}`) from a single checkpoint.

## Layout

- `loopear/encoder.py`: Frozen `MIT-SLS/USAD2-Small` extractor, invariant checks, FOA/channel-selective mono loading, and content-addressed sharded fp16 feature cache.
- `loopear/data.py`: Zero-USAD2-overlap real-world dataset loaders (`STARSS23`, `RealDESED`, `DiPCo`, `AMI SDM`, `CREMA-D`, `VocalSet`, `MAESTRO Real`), HTTP-Range selective Zenodo ZIP extraction, sealed-split & group-disjoint manifest validation, and 50 Hz frame label rasterization/resampling.
- `loopear/models.py`: Comparison arms (`A_last`, `A_ml`, `B_comp`, `B_param`, `C_param`, `C_comp`, `C_wide`, `D`, `E`) with `ContinuousTimeRoPE`, cached encoder K/V, FiLM depth conditioning, configurable temporal compression (`stride in {1, 4, 8}`, `pool_mode in {avg, strided_conv, attn}`), a learned softmax over USAD2 layers, and a zero-initialized residual local 50 Hz depthwise-separable stem (`x + alpha * DWConv1d(x)`, `alpha` initialized at `0`).
- `loopear/losses.py`: Multi-depth event BCE (`RealDESED`, `STARSS23`), 3-class overlapped-speech CE (`DiPCo`), cross-meeting AAM-Softmax speaker loss (`AMI SDM`) with a separate 12-layer speaker mixture, masked pooling, cached-feature time masking, and deterministic seeding.
- `loopear/oracle.py`: H6 oracle depth-allocation knapsack headroom, overthinking rate, query-independent collapse check, and training-free AudioTETRIS heuristic.
- `loopear/halting.py`: Learned halting gate (`k in {1, 2, 4}`) from step confidence, entropy, representation delta, and acoustic complexity, plus a compute-budget Pareto comparison against uniform depth, AudioTETRIS, and the H6 oracle.
- `loopear/harness.py`: Immutable SHA-256 run manifests (`ledger/runs/`), spend ledger (`ledger/spend.jsonl`, `$5.00` cap), sealed-split guard, and group-disjoint split checks.
- `loopear/stream.py` and `loopear/demo.py`: Continuous local inference (`LoopEarStream`, including `adaptive=True`) and `aiohttp` microphone/upload prototype (`/api/pcm`, `/api/depth`, `/api/adaptive`, `/api/enroll`, `/api/compare`, `/api/state`).
- `scripts/verify.py`: Enforces `pyflakes`, zero `#` comments in `.py` files, `README.md`-only repo docs, and `pytest`.
- `scripts/profile_arms.py`: Isolated subprocess profiler reporting params, 10s GFLOPs, 1-core and all-core CPU RTF, and Peak RSS split into encoder, refinement, readout, and end-to-end.
- `scripts/prepare_data.py`: Downloads, validates (`join_coverage >= 0.99`, group-disjoint, 0% USAD2 overlap, `realdesed_test` sealed), writes `data/manifests/*.json`, and extracts sharded fp16 `USAD2-Small` features.
- `scripts/train_pilot.py` and `scripts/eval_transfer.py`: Trains and evaluates the Stage 1 isolating arms, paired bootstrap 95% CIs, H6 oracle headroom, frozen transfer probes with shuffled-label controls, and real-audio `LoopEarStream` verification.
- `scripts/prepare_stage2_taps.py` and `scripts/train_stage2.py`: Extract a 4-layer 50 Hz tap cache and train the fair multi-layer Stage 2 arms, onset diagnostics, learned halting Pareto curves, and adaptive streaming verification.
- `scripts/train_stage3.py`: One-seed depth probe. Later steps are trained against a stop-grad `k=1` target with a penalty when their loss is worse, and the event head keeps a full-rate 50 Hz residual.

## Status

- **Stage 0** (`results/stage0_profile.json`): Verified `USAD2-Small` 4-core CPU RTF `0.0123` (`44 s/h`) and `E_s8@4` refinement `6.67 ms`/10s.
- **Stage 1 Real-Audio Pilot** (`results/stage1_pilot.json`, `$0.00` cloud GPU spend, CPU-only):
  - **Data (`9.11 h` total, 0% USAD2 overlap, `45.85 s/h` CPU extraction)**: Train/Val (`4.31 h` train / `1.87 h` val) across `STARSS23`, `RealDESED` (`test` sealed), `DiPCo`, and `AMI SDM`; zero-overlap frozen transfer (`2.94 h`) across `CREMA-D`, `VocalSet`, and `MAESTRO Real`.
  - **Dev Composite (4-task equal-weight, `k_max`)**: `B_comp` `0.6151` > `E_s8_conv@4` `0.6069` > `A_ml` `0.5938` > `D@4` `0.5936` > `E_s8@4` `0.5895` (`+1.33 pp` vs `k=1`) > `E_s4@4` `0.5851` (`+1.38 pp` vs `k=1`) > `C_param@4` `0.5618` > `A_last` `0.5568`.
  - **Zero-Overlap Frozen Transfer Composite (`k_max`, shuffled controls at chance)**: `E_s8@4` `0.6793` (`+1.96 pp` vs `k=1`) > `E_s4@4` `0.6787` > `D@4` `0.6745` > `A_ml` `0.6595` > `C_param@4` `0.6516` > `A_last` `0.6512` > `B_comp` `0.6467` > `E_s8_conv@4` `0.6427`.
  - **H6 Oracle Headroom**: `E_s8` `48.1%` savings (`improved=31.9%`, `overthinking=28.3%`, `h6_go=True`), `E_s4` `57.5%` savings (`improved=32.9%`, `overthinking=29.7%`, `h6_go=True`), non-collapsed (`cos=0.62–0.64`).
  - **Live `LoopEarStream` (`E_s8@4`, real val WAVs)**: `46.19 ms`/window (`RTF 0.0462`), cross-meeting same-speaker `speaker_cos=0.746` vs different-speaker `0.665`, enrolled-scene hit `0.688` vs `0.552`.
- **Stage 2 Fair Multi-Layer Fusion + Learned Halting** (`results/stage2_multilayer_halting.json`, `$0.00` cloud GPU, CPU only, seeds `42/43/44`, 26 epochs, best checkpoint restored):
  - **Shared inputs**: 50 Hz taps from USAD2 layers `[2, 5, 8, 11]` for `B_comp_ml`, `D_ml`, `E_ml_res_s8`, and `E_ml_res_s4`, plus a separate 12-layer speaker mixture on `layer_clip`. `A_ml` stays the linear 12-layer baseline. The local stem is `x + alpha * DWConv1d(x)` with `alpha` initialized at `0` (LayerScale `0.3`). After training, mean `|alpha|` stayed about `0.003`, so the stem remained close to the USAD2 identity.
  - **Dev composite** (`STARSS23` macro-AP, `RealDESED` macro-AP, `DiPCo` macro-F1, `AMI` `1-EER`): `B_comp_ml` `0.6604±0.0041` > `D_ml@4` `0.6539±0.0075` > `E_ml_res_s4@4` `0.6538±0.0047` > `E_ml_res_s8@4` `0.6510±0.0032` > `A_ml` `0.6318±0.0019`. Paired bootstrap 95% CIs on seed-averaged window scores: `E_s8@4 − B` `-1.50 pp` `[-2.34, -0.80]`, `E_s4@4 − B` `-0.20 pp` `[-0.78, +0.36]`, `E_s4@4 − E_s8@4` `+1.31 pp` `[+0.87, +1.77]`. Extra depth did not help: `E_s8@4 − E_s8@1` `-1.31 pp` `[-1.77, -0.84]` and `E_s4@4 − E_s4@1` `-0.76 pp` `[-1.16, -0.38]`. `E_s8` validation peaked at epoch 16 on every seed. `D_ml` seed 42 and `B_comp_ml` seed 44 were still best on the final epoch.
  - **Fine timing** (`STARSS23` + `RealDESED`, mean onset event-F1 / boundary-zone frame-F1): onset-F1 at `±100 ms` is `B` `0.193` > `D@4` `0.191` > `E_s8@4` `0.168` > `E_s4@4` `0.143` > `A` `0.120`. `±60 ms` onset-F1 has the same order. Boundary-F1 inside `±60 ms` of a label change is highest for `A_ml` (`0.595`), not for the finer `s=4` pool (`0.542`). Stride 4 did not preserve sharper onsets than stride 8.
  - **Zero-overlap frozen transfer** (CREMA-D emotion accuracy + speaker `1-EER`, VocalSet accuracy, MAESTRO Real macro-AP; shuffled probes stay near chance): `A_ml` `0.6395±0.0043` > `B_comp_ml` `0.6189±0.0051` > `E_ml_res_s4@4` `0.6120±0.0085` > `D_ml@4` `0.6073±0.0044` > `E_ml_res_s8@4` `0.6055±0.0066`. `k=4` is below `k=1` on transfer for `D` (`-1.94 pp`), `E_s8` (`-0.96 pp`), and `E_s4` (`-0.71 pp`).
  - **Learned halting vs uniform / AudioTETRIS / H6 oracle**: On dev, Pareto-AUC of the train-fit gate is above uniform and AudioTETRIS for both strides (`E_s8` `0.6668` vs `0.6646` vs `0.6610`; `E_s4` `0.6777` vs `0.6754` vs `0.6726`). At matched `avg k ≈ 2` the learned score beats uniform by `+0.12` to `+0.43 pp` and AudioTETRIS by `+0.45` to `+0.98 pp`. On frozen transfer the same gate wins for `E_s8` (`+0.33 pp` AUC vs uniform, `+0.18 pp` vs AudioTETRIS) and loses for `E_s4` (`-0.17 pp` vs uniform, `-0.08 pp` vs AudioTETRIS). It does cut overthinking versus always-`k=4` (`+28` to `+38 pp` on dev, `+10` to `+13 pp` on transfer) because `k=4` is usually worse, so the gate stops early. H6 oracle savings remain `50%` (`E_s8`) and `60%` (`E_s4`), representations are not collapsed (`delta cosine 0.27–0.31`), and `h6_go` is `False` because mean `k=4` is below mean `k=1`.
  - **Live `LoopEarStream(adaptive=True)`** on `checkpoints/stage2_best.pt` (`E_ml_res_s4`, seed 44, the best LoopEar arm by the average of dev and transfer, not the best overall arm): `reload_max_abs_diff = 0.0`. AMI cross-meeting same-speaker cosine `0.461` vs different-speaker `0.296`. CPU latency on 4 s DiPCo windows is `50.7 ms` adaptive, `50.0 ms` at `k=1`, and `54.8 ms` at `k=4` (the frozen encoder dominates). Quiet windows used `k=1/2/4` at `42% / 25% / 33%` (avg `2.25`); dense overlap used `58% / 42% / 0%` (avg `1.42`). A STARSS23 + RealDESED stream was `79% k=1`, `21% k=2`, `0% k=4` (`56.3 ms`/window, `RTF 0.056`). The learned policy does not spend more depth on overlapped audio.
- **Stage 3 depth probe** (`results/stage3_depth_probe.json`, seed `42` only, 8 epochs, `$0.00` GPU, `A_ml` and `B_comp_ml` not retrained): one LoopEar (`stride 8`) with a stop-grad `k=1` target, `relu(loss_k - loss_k1)` on every task, and a full-rate 50 Hz residual into the STARSS and RealDESED heads. The pre-specified gate **passes**, narrowly. On the best `k=4` checkpoint (epoch 8, still rising): dev composite `k=4` `0.6287` vs `k=1` `0.6239` (`+0.48 pp`); onset event-F1 at `±100 ms` `0.0540` vs `0.0534` (`+0.06 pp`; STARSS `0.0651` vs `0.0656`, RealDESED `0.0429` vs `0.0412`); transfer composite `0.6400` vs `0.6381` (`+0.19 pp`). Shuffled probes stay near chance (CREMA-D emotion `0.201` vs chance `0.167`, VocalSet `0.148` vs `0.100`, MAESTRO `0.128`). Epoch 2 had `k=4` behind `k=1` by `0.94 pp` before the later epochs recovered. Absolute dev composite and onset remain below the frozen seed-42 baselines (`A_ml` dev `0.6342`, onset `0.122`; `B_comp_ml` dev `0.6660`, onset `0.195`). The later step can edge its own first pass under this loss. It does not match the 50 Hz baseline's timing.

## Campaign closeout

- **H5: CLOSED — NO ROBUST ARCHITECTURAL ADVANTAGE ESTABLISHED**
- **H6: DEFERRED — INSUFFICIENT EVIDENCE OF USEFUL ADDITIONAL DEPTH**

No separate research-registry file exists in this repository. These two lines are the registry record. The full closeout, including the pre-written criteria, the Stage 1 transfer win, the Stage 2 reversal, the Stage 3 within-model gate, and the reopen bar, is outside the git tree:

`/cursor/stores/bc-01a213ed-7a8a-7981-b509-7012340964c4/docs/h5-h6-campaign-report.md`

`realdesed_test` stays sealed. Checkpoints, caches, and the sealed split were kept. This closeout does not train a follow-on model.

## Quickstart

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python scripts/verify.py
.venv/bin/python scripts/profile_arms.py --out results/stage0_profile.json
.venv/bin/python scripts/prepare_data.py --batch-size 8 --threads 4
.venv/bin/python scripts/train_pilot.py --epochs 10 --steps-per-epoch 28 --batch-size 16 --seed 42
.venv/bin/python scripts/prepare_stage2_taps.py --batch-size 8 --threads 4
.venv/bin/python scripts/train_stage2.py --epochs 26 --steps-per-epoch 28 --batch-size 16 --seeds 42 43 44
.venv/bin/python scripts/train_stage3.py --epochs 8 --steps-per-epoch 28 --batch-size 16 --seed 42
.venv/bin/python -m loopear.demo --ckpt checkpoints/stage2_best.pt --adaptive --host 127.0.0.1 --port 8765
```

## LoopEar-D

Separate from the frozen USAD2 stack above. `loopear/d` is a 682,752-parameter encoder trained from scratch on the official FSD50K training split. The deployed file is the encoder. The predictor, the projection heads, and the EMA teacher are training-only. The 1,000-step comparison in this section used the nested 5.001-hour prefix (1,680 clips). A later matched run trains this encoder for 5,000 steps on the 20.003-hour prefix and writes `results/loopear_d_h20_global_5000_encoder.safetensors`. CPU only. Cloud GPU spend `$0.00`.

```bash
.venv/bin/python scripts/prepare_loopear_d.py
.venv/bin/python scripts/train_loopear_d.py --stage auto
.venv/bin/python scripts/eval_loopear_d.py --encoder results/loopear_d_k1_encoder.safetensors --report results/loopear_d_eval_k1.json
```

After 1,000 steps of global contrastive loss plus masked latent discrimination (`K=1`, batch 64, seed 42), validation class tokens are not collapsed (effective rank 29.3, mean off-diagonal cosine 0.327). A frozen 2,716-parameter probe on mean patch tokens scores macro AP 0.316 on 160 official validation clips and 28 classes. Random initialization scores 0.262 and log-mel mean/std scores 0.246. Shuffled labels sit near 0.17. Global-only is slightly higher (0.328). The same budget does not beat a mel baseline on a four-file STARSS23 onset check. `results/loopear_d_train.json` and `results/loopear_d_eval_k1.json` hold the raw numbers.

```bash
.venv/bin/python scripts/eval_loopear_d_baselines.py
```

That command does not train. It scores frozen USAD2-Small and EfficientAT mn04_as with the same 2,716-parameter readout, the same 256/160 clips, and the same four STARSS23 files. USAD2-Small scores macro AP 0.312. EfficientAT mn04_as, AudioSet-pretrained and using its own 32 kHz frontend, scores 0.526. Shuffled labels for both sit near 0.16. On the STARSS23 frame grid both raw scores sit above global-only and match their shuffled-label controls, because the scored validation classes have no positive training frames. `results/loopear_d_baselines.json` holds those numbers. AudioSet audio is not downloaded.

## LoopEar-P

Same frontend and the same global contrastive loss as LoopEar-D. The encoder keeps 32 learned latents of width 96, with three input cross-attention blocks and three latent self-attention blocks. The deployed encoder has 685,728 parameters, 2,976 above LoopEar-D. The local masked loss remains in `loopear/d/objective.py` and this comparison leaves it off. CPU only. Cloud GPU spend `$0.00`.

```bash
.venv/bin/python scripts/train_loopear_p.py
.venv/bin/python scripts/eval_loopear_protocol.py
```

The training command is the existing 5.001-hour subset, 1,000 steps, batch 64, seed 42. It writes `results/loopear_p_encoder.safetensors` and `results/loopear_matched_h20_plan.json`. That plan file still records the 20-hour runs as unlaunched. The runs that followed keep every 1,000-step file and write new ones:

```bash
.venv/bin/python scripts/cache_loopear_h20.py
.venv/bin/python scripts/train_loopear_p.py --launch-prepared loopear-d
.venv/bin/python scripts/train_loopear_p.py --launch-prepared loopear-p
.venv/bin/python scripts/eval_loopear_protocol.py --d-global results/loopear_d_h20_global_5000_encoder.safetensors --p results/loopear_p_h20_global_5000_encoder.safetensors --skip-k1 --coordinate --report results/loopear_h20_protocol.json
```

The 5,000-step weights are `results/loopear_d_h20_global_5000_encoder.safetensors` and `results/loopear_p_h20_global_5000_encoder.safetensors`. `results/loopear_h20_protocol.json` holds that probe, including the coordinate readout. The 1,000-step eval command fits one readout on 1,680 development clips and 640 official validation clips, scores STARSS23 only where both sides have enough positives, runs a 4-example probe, and checks a 10-bin log-mel profile. `results/loopear_protocol.json` holds those 1,000-step numbers. The 160-clip scores above are a different probe.

The equal-wall-clock scores use the existing LoopEar-D snapshots at 2,000 and 3,000 steps: `results/loopear_h20_equal_time_d2000.json` and `results/loopear_h20_equal_time_d3000.json`. P-T keeps the same 32 latents and the same global loss, and changes only the input cross-attention logits to `q·k/sqrt(d) − α(t−μ)^2`. Alpha 0 is ordinary cross-attention. The locked alpha is in `results/loopear_pt_h20_train.json`.

```bash
.venv/bin/python scripts/train_loopear_p.py --launch-prepared loopear-pt
```

`results/loopear_pt_h20_protocol.json` holds the P-T probe. GPU spend `$0.00`.

P-L is that same unanchored encoder, 32 latents and 685,728 parameters, trained with the global loss plus the existing masked local loss at λ = 1. The local predictor queries the latent bank and does not take the acoustic patches. Teacher targets are stop-grad. The locked P-T penalty, measured on the trained logits before this run and left unchanged, is in `results/loopear_pt_penalty_scale.json`.

```bash
.venv/bin/python scripts/train_loopear_p.py --launch-prepared loopear-pl
.venv/bin/python scripts/eval_loopear_protocol.py --d-global results/loopear_d_h20_global_5000_encoder.safetensors --p results/loopear_pl_h20_global_5000_encoder.safetensors --time-alpha 0 --p-reference results/loopear_p_h20_global_5000_encoder.safetensors --skip-k1 --coordinate --coordinate-name perceiver_l --report results/loopear_pl_h20_protocol.json
```

`results/loopear_pl_h20_protocol.json` holds the P-L probe. Four-example recognition is the primary score. The energy profile is a diagnostic.

P-LT keeps that local objective and adds the temporal penalty on the input cross-attention logits. Alpha is the centered content-logit RMS of the trained P-L encoder divided by the centered `(t − μ)^2` RMS. The initialization's own centered RMS is recorded and is not the scale. There is no loss sweep. The locked alpha is in `results/loopear_plt_alpha.json`. The within-recording ranking of the saved P-L student is in `results/loopear_pl_time_rank.json`.

```bash
.venv/bin/python scripts/train_loopear_p.py --launch-prepared loopear-plt
.venv/bin/python scripts/eval_loopear_protocol.py --d-global results/loopear_d_h20_global_5000_encoder.safetensors --p results/loopear_plt_h20_global_5000_encoder.safetensors --time-alpha 0.0027896127554686775 --p-reference results/loopear_pl_h20_global_5000_encoder.safetensors --skip-k1 --coordinate --coordinate-name perceiver_lt --report results/loopear_plt_h20_protocol.json
```

`results/loopear_plt_h20_protocol.json` holds the P-LT probe. Four-example recognition stays the primary score. The energy profile stays a diagnostic.

P-LTC keeps that model, including the locked alpha `0.0027896127554686775`, and adds one correspondence term. The positive pair is the same absolute time in two crops whose offsets differ. The denominator is the other aligned moments in that recording, excluding one patch (0.04 s) on each side of the positive. Temperature stays 0.1. The locked weight is `λ_t = 0.8774499635938284`, from one initialization batch, in `results/loopear_pltc_before.json`. The run saves the EMA teacher.

```bash
.venv/bin/python scripts/train_loopear_p.py --launch-prepared loopear-pltc
.venv/bin/python scripts/eval_loopear_protocol.py --d-global results/loopear_d_h20_global_5000_encoder.safetensors --p results/loopear_pltc_h20_global_5000_encoder.safetensors --time-alpha 0.0027896127554686775 --p-reference results/loopear_plt_h20_global_5000_encoder.safetensors --p-reference-time-alpha 0.0027896127554686775 --skip-k1 --coordinate --coordinate-name perceiver_ltc --report results/loopear_pltc_h20_protocol.json
```

`results/loopear_pltc_h20_protocol.json` holds the probe. `results/loopear_pltc_time_rank.json` holds the EMA retrieval. `results/loopear_pltc_event_order.json` holds the STARSS event-order counts. Four-example recognition, onset localization, and event order are the primary scores. The energy profile and the 970-parameter readout stay diagnostics.

P-LT-F is the P-LT recipe again, not a new loss. It is the calibrated anchored Perceiver before temporal correspondence: global NT-Xent plus the masked local loss, λ = 1, K = 1, priors 0.1/0.8/0.1, locked alpha `0.0027896127554686775`. No `L_time`. Random initialization. 5,000 steps, batch 64, seed 42, 640,000 windows. The corpus is the full eligible official FSD50K train split. The run records how many of those recordings the sampler actually drew. It saves the EMA teacher.

```bash
.venv/bin/python scripts/train_loopear_p.py --launch-prepared loopear-plt-f
.venv/bin/python scripts/eval_loopear_protocol.py --p results/loopear_pltf_full_5000_encoder.safetensors --time-alpha 0.0027896127554686775 --p-reference results/loopear_plt_h20_global_5000_encoder.safetensors --p-reference-time-alpha 0.0027896127554686775 --d-global results/loopear_d_h20_global_5000_encoder.safetensors --skip-k1 --coordinate --coordinate-name perceiver_ltf --report results/loopear_pltf_full_protocol.json
```

`results/loopear_pltf_full_train.json` holds the run, including the recordings actually drawn. `results/loopear_pltf_full_protocol.json` holds the same probe as P-LT. `results/loopear_pltf_event_order.json` holds the STARSS event-order counts against the saved P-LT EMA. In that file the `pltc` block is P-LT-F.
