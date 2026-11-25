import asyncio
import io
import math
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
import torch.nn.functional as F
from aiohttp.test_utils import TestClient, TestServer

from loopear.data import (
    PILOT_CORPUS_SOURCES,
    assert_disjoint_groups,
    downsample_binary_targets,
    downsample_overlap_targets,
    parse_hms_seconds,
    rasterize_intervals_50hz,
    rasterize_overlap_50hz,
    rasterize_soft_labels_50hz,
    rasterize_starss_frames_50hz,
    upsample_frame_preds,
    validate_manifest,
)
from loopear.demo import create_app
from loopear.encoder import (
    D_E,
    EVAL_HOP_S,
    NUM_ENCODER_LAYERS,
    SR,
    T_E_MAX,
    ShardedFeatureCache,
    load_mono_audio,
    load_usad2_extractor,
    random_crop_window,
    save_npz_atomic,
    verify_extractor_invariants,
)
from loopear.harness import (
    USAD2_SMALL_FREESOUND_OVERLAP_SOURCES,
    USAD2_SMALL_PRETRAINING_SOURCES,
    SealedSplitGuard,
    SpendLedger,
    assert_group_disjoint,
    assert_zero_pretraining_overlap,
    write_result_json,
    write_run_manifest,
)
from loopear.losses import (
    MultiTaskHeads,
    compute_cross_chapter_eer,
    masked_mean_pool,
    multidepth_event_loss,
    multidepth_loss,
    multidepth_overlap_loss,
    set_seed,
)
from loopear.models import (
    LoopEarModel,
    build_arm,
)
from loopear.oracle import (
    audiotetris_scores,
    evaluate_h6_gate,
    knapsack_oracle,
    overthinking_rate,
    query_independence_check,
)
from loopear.stream import LoopEarStream


@pytest.fixture(scope="module")
def usad2_encoder():
    model, dtype = load_usad2_extractor("cpu")
    assert dtype == torch.float32
    return model


def _sine_wave(freq_hz: float, dur_s: float, amp: float = 0.25) -> np.ndarray:
    n = int(round(dur_s * SR))
    t = np.arange(n, dtype=np.float32) / float(SR)
    return (amp * np.sin(2.0 * math.pi * freq_hz * t)).astype(np.float32)


def test_usad2_live_invariants_and_cache_roundtrip(usad2_encoder, tmp_path: Path):
    summary = verify_extractor_invariants(usad2_encoder, "cpu")
    assert summary["shape"] == [1, 99, D_E]
    assert summary["num_layers"] == NUM_ENCODER_LAYERS
    assert summary["max_abs_diff_sil_tone"] > 1e-3

    rows = [
        {"id": "clip_short", "wav": _sine_wave(440.0, 2.0), "meta": {"group": "g1"}},
        {"id": "clip_long", "wav": _sine_wave(660.0, 12.0), "meta": {"group": "g2"}},
    ]
    cache = ShardedFeatureCache(tmp_path / "feat_cache")
    out_dir = cache.extract_dataset(
        "unit_synth", rows, usad2_encoder, device="cpu", batch_size=2, shard_size=8
    )
    out_dir_repeat = cache.extract_dataset(
        "unit_synth", rows, usad2_encoder, device="cpu", batch_size=2, shard_size=8
    )
    assert out_dir == out_dir_repeat
    idx = cache.read_index(out_dir)
    assert len(idx["items"]) == 3
    shard = cache.load_shard(out_dir, idx["items"][0]["shard"])
    assert shard["E"].dtype == np.float16
    assert shard["E"].shape == (3, T_E_MAX, D_E)
    assert "layer_pool" not in shard
    assert shard["lengths"][0] == 99
    assert np.isfinite(shard["E"]).all()

    crop, crop_len = random_crop_window(
        shard["E"][1], int(shard["lengths"][1]), 128, rng=np.random.default_rng(0)
    )
    assert crop.shape == (128, D_E)
    assert crop_len == 128

    out_dir_ml = cache.extract_dataset(
        "unit_synth_eval",
        rows,
        usad2_encoder,
        device="cpu",
        batch_size=2,
        shard_size=8,
        hop_s=EVAL_HOP_S,
        save_layer_pool=True,
    )
    idx_ml = cache.read_index(out_dir_ml)
    assert len(idx_ml["items"]) == 4
    shard_ml = cache.load_shard(out_dir_ml, idx_ml["items"][0]["shard"])
    assert shard_ml["layer_pool"].shape == (4, NUM_ENCODER_LAYERS, T_E_MAX // 8, D_E)
    assert np.isfinite(shard_ml["layer_pool"]).all()

    bad = np.array([1.0, np.nan], dtype=np.float32)
    with pytest.raises(ValueError):
        save_npz_atomic(tmp_path / "bad_nan.npz", arr=bad)
    bad_inf = np.array([1.0, np.inf], dtype=np.float32)
    with pytest.raises(ValueError):
        save_npz_atomic(tmp_path / "bad_inf.npz", arr=bad_inf)


def test_all_arms_shapes_padding_invariance_and_params():
    set_seed(0)
    b, t_e = 2, 104
    e = torch.randn(b, t_e, D_E)
    e_len = torch.tensor([99, 64], dtype=torch.long)
    hs = torch.randn(b, NUM_ENCODER_LAYERS, t_e, D_E)
    e_corrupt = e.clone()
    hs_corrupt = hs.clone()
    e_corrupt[0, 99:] = 1e6
    e_corrupt[1, 64:] = -1e6
    hs_corrupt[0, :, 99:] = 1e6
    hs_corrupt[1, :, 64:] = -1e6

    expected_ranges = {
        "A_last": (0.10, 0.25),
        "A_ml": (0.10, 0.25),
        "B_comp": (0.90, 1.40),
        "B_param": (1.90, 2.50),
        "C_param": (1.90, 2.50),
        "C_comp": (8.00, 9.00),
        "C_wide": (8.00, 9.20),
        "D": (2.20, 2.80),
        "E": (2.00, 2.50),
    }

    for arm, (p_lo, p_hi) in expected_ranges.items():
        model = build_arm(arm, stride=8).eval()
        params_m = sum(p.numel() for p in model.parameters()) / 1e6
        assert p_lo <= params_m <= p_hi, f"{arm} params {params_m:.3f}M not in [{p_lo}, {p_hi}]"
        with torch.no_grad():
            st1, pad1 = model.forward_states(e, e_len, hidden_states=hs)
            st2, pad2 = model.forward_states(e_corrupt, e_len, hidden_states=hs_corrupt)
        assert torch.equal(pad1, pad2)
        expected_tz = math.ceil(t_e / 8)
        assert tuple(pad1.shape) == (b, expected_tz)
        for k in model.depths:
            assert k in st1
            assert tuple(st1[k].shape) == (b, expected_tz, model.out_dim)
            assert torch.isfinite(st1[k]).all()
            diff = (st1[k] - st2[k]).abs().max().item()
            assert diff < 1e-5, f"{arm} k={k} padding invariance diff={diff}"


def test_loopear_strides_pooling_multilayer_and_kv_cache_parity():
    set_seed(7)
    b, t_e = 2, 100
    e = torch.randn(b, t_e, D_E)
    e_len = torch.tensor([97, 61], dtype=torch.long)
    hs = torch.randn(b, NUM_ENCODER_LAYERS, t_e, D_E)
    e_corrupt = e.clone()
    e_corrupt[0, 97:] = 1e5
    e_corrupt[1, 61:] = -1e5

    for stride in (1, 4, 8):
        for pool_mode in ("avg", "strided_conv", "attn"):
            model = LoopEarModel(
                d_e=D_E,
                d=96,
                h=4,
                d_ff=192,
                stride=stride,
                pool_mode=pool_mode,
                use_multilayer=(stride == 4 and pool_mode == "avg"),
            ).eval()
            with torch.no_grad():
                st_cached, pad_c = model.forward_states(
                    e, e_len, depths=(1, 2, 4), hidden_states=hs, use_kv_cache=True
                )
                st_fresh, pad_f = model.forward_states(
                    e, e_len, depths=(1, 2, 4), hidden_states=hs, use_kv_cache=False
                )
                st_pad, _ = model.forward_states(
                    e_corrupt, e_len, depths=(1, 2, 4), hidden_states=None, use_kv_cache=True
                )
            assert torch.equal(pad_c, pad_f)
            expected_tz = math.ceil(t_e / stride)
            for k in (1, 2, 4):
                assert tuple(st_cached[k].shape) == (b, expected_tz, 96)
                kv_diff = (st_cached[k] - st_fresh[k]).abs().max().item()
                assert kv_diff < 1e-5, (
                    f"stride={stride} pool={pool_mode} k={k} kv_diff={kv_diff}"
                )
            if not model.use_multilayer:
                for k in (1, 2, 4):
                    pad_diff = (st_cached[k] - st_pad[k]).abs().max().item()
                    assert pad_diff < 1e-5, (
                        f"stride={stride} pool={pool_mode} k={k} pad_diff={pad_diff}"
                    )


def _make_transient_order_dataset(
    n_samples: int,
    t_e: int,
    d_e: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    u_a = torch.zeros(d_e)
    u_b = torch.zeros(d_e)
    u_a[: d_e // 2] = 3.0
    u_b[d_e // 2 :] = 3.0
    e = 0.05 * torch.randn(n_samples, t_e, d_e, generator=g)
    y = torch.zeros(n_samples, dtype=torch.long)
    n_bins = t_e // 8
    for i in range(n_samples):
        label = i % 2
        y[i] = label
        b_idx = int(torch.randint(0, n_bins, (1,), generator=g).item())
        base = b_idx * 8
        if label == 1:
            e[i, base + 1] += u_a
            e[i, base + 6] += u_b
        else:
            e[i, base + 1] += u_b
            e[i, base + 6] += u_a
    lengths = torch.full((n_samples,), t_e, dtype=torch.long)
    return e, lengths, y


def _train_transient_probe(
    model: torch.nn.Module,
    e_tr: torch.Tensor,
    l_tr: torch.Tensor,
    y_tr: torch.Tensor,
    e_va: torch.Tensor,
    l_va: torch.Tensor,
    y_va: torch.Tensor,
    steps: int = 80,
) -> dict[int, float]:
    head = torch.nn.Linear(model.out_dim, 2)
    opt = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()), lr=3e-3, weight_decay=1e-4
    )
    model.train()
    for _ in range(steps):
        opt.zero_grad()
        states, pad = model.forward_states(e_tr, l_tr, depths=model.depths)
        loss = sum(
            F.cross_entropy(head(masked_mean_pool(states[k], pad)), y_tr)
            for k in model.depths
        ) / float(len(model.depths))
        loss.backward()
        opt.step()
    model.eval()
    accs: dict[int, float] = {}
    with torch.no_grad():
        states, pad = model.forward_states(e_va, l_va, depths=model.depths)
        for k in model.depths:
            preds = head(masked_mean_pool(states[k], pad)).argmax(dim=-1)
            accs[k] = float((preds == y_va).float().mean().item())
    return accs


def test_tiny_overfit_and_transient_temporal_compression_diagnostic():
    set_seed(11)
    model = LoopEarModel(d_e=64, d=64, h=4, d_ff=128, stride=8, drop=0.0)
    heads = MultiTaskHeads(d=64, n_ev=4, d_spk=32, n_spk=6)
    w0 = torch.cat([p.detach().flatten().clone() for p in model.parameters()])
    e_ev = torch.randn(4, 64, 64)
    l_ev = torch.tensor([64, 56, 48, 64], dtype=torch.long)
    y_f = (torch.rand(4, 8, 4) > 0.5).float()
    y_c = y_f.amax(dim=1)
    e_spk = torch.randn(6, 64, 64)
    l_spk = torch.full((6,), 64, dtype=torch.long)
    spk_id = torch.arange(6, dtype=torch.long)
    ev_batch = {"E": e_ev, "lengths": l_ev, "Y_f": y_f, "Y_c": y_c}
    spk_batch = {"E": e_spk, "lengths": l_spk, "spk_id": spk_id}
    opt = torch.optim.AdamW(
        list(model.parameters()) + list(heads.parameters()), lr=5e-3
    )
    losses: list[float] = []
    for _ in range(35):
        opt.zero_grad()
        out = multidepth_loss(model, heads, ev_batch, spk_batch)
        out["total"].backward()
        opt.step()
        losses.append(float(out["total"].item()))
    w1 = torch.cat([p.detach().flatten() for p in model.parameters()])
    assert float(torch.norm(w1 - w0).item()) > 1e-3
    assert losses[-1] < losses[0] * 0.45

    emb = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.98, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.1, 0.98, 0.0],
        ],
        dtype=np.float32,
    )
    eer_res = compute_cross_chapter_eer(
        emb, np.array([0, 0, 1, 1]), np.array([10, 11, 20, 21])
    )
    assert eer_res["eer"] == 0.0

    d_t = 32
    e_tr, l_tr, y_tr = _make_transient_order_dataset(48, 64, d_t, seed=101)
    e_va, l_va, y_va = _make_transient_order_dataset(32, 64, d_t, seed=202)

    set_seed(1)
    acc_a = _train_transient_probe(
        build_arm("A_last", d_e=d_t, d=d_t, stride=8, pool_mode="avg"),
        e_tr,
        l_tr,
        y_tr,
        e_va,
        l_va,
        y_va,
    )
    set_seed(1)
    acc_e_avg_s8 = _train_transient_probe(
        LoopEarModel(d_e=d_t, d=d_t, h=4, d_ff=64, stride=8, pool_mode="avg", drop=0.0),
        e_tr,
        l_tr,
        y_tr,
        e_va,
        l_va,
        y_va,
    )
    set_seed(1)
    acc_d_avg_s8 = _train_transient_probe(
        build_arm("D", d_e=d_t, d=d_t, h=4, d_ff=64, stride=8, pool_mode="avg", drop=0.0),
        e_tr,
        l_tr,
        y_tr,
        e_va,
        l_va,
        y_va,
    )
    set_seed(1)
    acc_e_conv_s8 = _train_transient_probe(
        LoopEarModel(
            d_e=d_t, d=d_t, h=4, d_ff=64, stride=8, pool_mode="strided_conv", drop=0.0
        ),
        e_tr,
        l_tr,
        y_tr,
        e_va,
        l_va,
        y_va,
    )
    set_seed(1)
    acc_e_avg_s4 = _train_transient_probe(
        LoopEarModel(d_e=d_t, d=d_t, h=4, d_ff=64, stride=4, pool_mode="avg", drop=0.0),
        e_tr,
        l_tr,
        y_tr,
        e_va,
        l_va,
        y_va,
    )
    set_seed(1)
    acc_b = _train_transient_probe(
        build_arm(
            "B_comp",
            d_e=d_t,
            d=d_t,
            h=4,
            d_ff=64,
            stride=8,
            pool_mode="avg",
            drop=0.0,
        ),
        e_tr,
        l_tr,
        y_tr,
        e_va,
        l_va,
        y_va,
    )

    assert acc_a[1] <= 0.65
    assert acc_e_avg_s8[4] >= 0.85
    assert acc_d_avg_s8[4] >= 0.80
    assert acc_e_conv_s8[4] >= 0.85
    assert acc_e_avg_s4[4] >= 0.85
    assert acc_b[1] >= 0.85


def test_oracle_h6_and_audiotetris():
    rng = np.random.default_rng(42)
    n = 100
    s1 = np.full(n, 0.80, dtype=np.float64)
    s2 = s1.copy()
    s4 = s1.copy()
    hard = np.arange(0, 30)
    s2[hard] = 0.88
    s4[hard] = 0.98
    scores = {1: s1, 2: s2, 4: s4}
    flops_go = {1: 0.50, 2: 0.90, 4: 1.60}
    flops_tight = {1: 0.72, 2: 0.99, 4: 1.54}

    e_seq = []
    for i in range(n):
        scale = 1.5 if i in hard else 0.1
        e_seq.append(rng.normal(scale=scale, size=(20, 8)).astype(np.float32))
    tetris = audiotetris_scores(e_seq)
    assert tetris.shape == (n,)
    assert tetris[hard].mean() > tetris[30:].mean() * 5.0
    assert audiotetris_scores([np.zeros((1, 8), dtype=np.float32)])[0] == 0.0

    alloc = knapsack_oracle(scores, flops_go, budget=0.90)
    assert alloc["avg_flops"] <= 0.90 + 1e-9
    assert alloc["score"] > s4.mean() - 1e-6

    gate_go = evaluate_h6_gate(scores, flops_go, e_seq)
    assert gate_go["oracle_savings"] >= 0.40
    assert gate_go["improved_rate"] == pytest.approx(0.30, abs=1e-6)
    assert gate_go["overthinking_rate"] == 0.0
    assert gate_go["h6_go"] is True

    gate_tight = evaluate_h6_gate(scores, flops_tight, e_seq)
    assert gate_tight["oracle_savings"] < 0.40
    assert gate_tight["h6_go"] is False

    assert overthinking_rate(np.array([0.8, 0.8]), np.array([0.7, 0.9])) == 0.5

    z1 = rng.normal(size=(20, 16))
    z4_collapsed = z1 + np.ones((1, 16)) * 2.0
    z4_diverse = z1 + rng.normal(size=(20, 16))
    assert query_independence_check({1: z1, 4: z4_collapsed})["collapsed"] is True
    assert query_independence_check({1: z1, 4: z4_diverse})["collapsed"] is False


def test_research_harness_guardrails(tmp_path: Path):
    m1 = write_run_manifest("exp_01", {"arm": "E", "seed": 0}, ledger_dir=tmp_path)
    m2 = write_run_manifest("exp_01", {"arm": "E", "seed": 0}, ledger_dir=tmp_path)
    assert m1 == m2
    with pytest.raises(FileExistsError):
        write_run_manifest("exp_01", {"arm": "E", "seed": 1}, ledger_dir=tmp_path)

    ledger = SpendLedger(tmp_path / "spend.jsonl", cap_usd=5.0)
    ledger.record("pilot", "local", 0.0, detail="stage0")
    ledger.record("run1", "akash", 2.5)
    assert ledger.total_usd() == pytest.approx(2.5)
    with pytest.raises(RuntimeError):
        ledger.record("run2", "akash", 2.6)

    audit = tmp_path / "sealed.jsonl"
    guard = SealedSplitGuard({"realdesed_test"}, audit_log=audit)
    guard.check_access("realdesed_val")
    with pytest.raises(PermissionError):
        guard.check_access("realdesed_test")
    guard.check_access("realdesed_test", allow_sealed=True, run_id="final_01")
    with pytest.raises(PermissionError):
        guard.check_access("realdesed_test", allow_sealed=True, run_id="final_02")
    reloaded = SealedSplitGuard({"realdesed_test"}, audit_log=audit)
    with pytest.raises(PermissionError):
        reloaded.check_access("realdesed_test", allow_sealed=True, run_id="final_03")

    assert_group_disjoint(["a", "b"], ["c", "d"])
    with pytest.raises(AssertionError):
        assert_group_disjoint(["a", "b"], ["b", "c"])

    assert len(USAD2_SMALL_PRETRAINING_SOURCES) >= 24
    assert {"fsd50k", "esc50", "urbansound8k", "clotho"}.issubset(
        USAD2_SMALL_FREESOUND_OVERLAP_SOURCES
    )
    assert_zero_pretraining_overlap(
        [
            "realdesed",
            "starss23",
            "chime6",
            "cn_celeb1",
            "cremad",
            "vocalset",
            "maestro_real",
            "epic_sounds",
            "dipco",
            "dcase_bioacoustics",
        ]
    )
    for leaked in ("librispeech", "audioset", "nsynth", "piano_maestro", "fma"):
        with pytest.raises(AssertionError):
            assert_zero_pretraining_overlap([leaked])
    for freesound_hit in ("fsd50k", "esc50", "urbansound8k", "clotho"):
        with pytest.raises(AssertionError):
            assert_zero_pretraining_overlap([freesound_hit])
        assert_zero_pretraining_overlap([freesound_hit], block_indirect=False)

    res_path = write_result_json(tmp_path / "out.json", {"metric": 0.91})
    assert res_path.exists()


def test_stream_and_demo_http_endpoints(usad2_encoder):
    set_seed(3)
    ref = LoopEarModel(d_e=D_E, d=384, h=6, d_ff=768, stride=8).eval()
    heads = MultiTaskHeads(d=384).eval()
    stream = LoopEarStream(
        encoder=usad2_encoder,
        refiner=ref,
        heads=heads,
        depth=2,
        win_s=2.0,
        step_s=1.0,
        device="cpu",
    )
    tone_a = _sine_wave(440.0, 2.0)
    tone_b = _sine_wave(880.0, 2.0)

    async def _run_http():
        app = create_app(stream)
        async with TestClient(TestServer(app)) as client:
            r_idx = await client.get("/")
            assert r_idx.status == 200
            assert "LoopEar" in await r_idx.text()

            r_st = await client.get("/api/state")
            assert r_st.status == 200
            assert (await r_st.json())["depth"] == 2

            r_bad = await client.post("/api/depth", json={"depth": 3})
            assert r_bad.status == 400
            r_ok = await client.post("/api/depth", json={"depth": 4})
            assert r_ok.status == 200
            assert (await r_ok.json())["depth"] == 4

            wav_buf = io.BytesIO()
            sf.write(wav_buf, tone_a, SR, format="WAV")
            r_enr = await client.post(
                "/api/enroll?label=tone440",
                data=wav_buf.getvalue(),
                headers={"Content-Type": "audio/wav"},
            )
            assert r_enr.status == 200
            assert (await r_enr.json()) == {"label": "tone440", "n": 1}

            r_cmp_same = await client.post(
                "/api/compare",
                json={"pcm_a": tone_a.tolist(), "pcm_b": tone_a.tolist(), "depth": 2},
            )
            assert r_cmp_same.status == 200
            d_same = await r_cmp_same.json()
            assert d_same["acoustic_cos"] > 0.999

            r_cmp_diff = await client.post(
                "/api/compare",
                json={"pcm_a": tone_a.tolist(), "pcm_b": tone_b.tolist(), "depth": 2},
            )
            assert r_cmp_diff.status == 200
            d_diff = await r_cmp_diff.json()
            assert d_same["acoustic_cos"] > d_diff["acoustic_cos"]

            chunk_3s = np.concatenate((tone_a, _sine_wave(440.0, 1.0)))
            r_pcm = await client.post(
                "/api/pcm",
                data=chunk_3s.astype("<f4").tobytes(),
                headers={"Content-Type": "application/octet-stream"},
            )
            assert r_pcm.status == 200
            st_after = await r_pcm.json()
            assert st_after["windows_processed"] == 2
            assert "tone440" in st_after["enrolled_hits"]
            assert len(st_after["recurring"]) >= 1
            assert st_after["mean_ms"] > 0.0

    asyncio.run(_run_http())


def test_data_rasterization_downsampling_and_upsampling():
    assert abs(parse_hms_seconds("01:02:03.5") - 3723.5) < 1e-6
    assert abs(parse_hms_seconds("02:10.25") - 130.25) < 1e-6

    y_iv = rasterize_intervals_50hz(
        [(0.10, 0.30, 1), (0.20, 0.40, 3)], start_s=0.0, n_frames=25, n_classes=5
    )
    assert y_iv.shape == (25, 5)
    assert y_iv[6, 1] == 1.0 and y_iv[0, 1] == 0.0
    assert y_iv[12, 1] == 1.0 and y_iv[12, 3] == 1.0

    y_st = rasterize_starss_frames_50hz(
        [(1, 0), (1, 4), (3, 7)], start_s=0.0, n_frames=25, n_classes=13
    )
    assert y_st.shape == (25, 13)
    assert y_st[5:10, 0].sum() == 5.0
    assert y_st[5:10, 4].sum() == 5.0
    assert y_st[15:20, 7].sum() == 5.0

    y_ovl = rasterize_overlap_50hz(
        [(0.0, 0.40, "P1"), (0.20, 0.60, "P2"), (0.24, 0.36, "P3")],
        start_s=0.0,
        n_frames=40,
    )
    assert y_ovl.shape == (40,)
    assert int(y_ovl[2]) == 1
    assert int(y_ovl[14]) == 2
    assert int(y_ovl[24]) == 1
    assert int(y_ovl[35]) == 0

    y_soft = rasterize_soft_labels_50hz(
        [(0.0, 1.0, 2, 0.75)], start_s=0.0, n_frames=50, n_classes=11
    )
    assert np.allclose(y_soft[:, 2], 0.75)

    for stride in (4, 8):
        tz = math.ceil(40 / stride)
        ds_bin = downsample_binary_targets(y_iv, 25, stride, tz)
        assert ds_bin.shape == (tz, 5)
        assert ds_bin[:, 1].max() == 1.0
        ds_ovl = downsample_overlap_targets(y_ovl, 40, stride, tz)
        assert ds_ovl.shape == (tz,)
        assert set(ds_ovl.tolist()).issubset({0, 1, 2})
        up = upsample_frame_preds(ds_bin, stride, 25)
        assert up.shape == (25, 5)


def test_data_manifest_validation_channel_select_and_stage1_heads(tmp_path: Path):
    assert_zero_pretraining_overlap(PILOT_CORPUS_SOURCES)
    assert_disjoint_groups(["r1", "r2"], ["r3", "r4"], label="rooms")

    good_manifest = {
        "corpus": "starss23",
        "join_coverage": 1.0,
        "items": [
            {"id": "a", "split": "train", "group_id": "room1", "duration_s": 10.0, "sha256": "abc"},
            {"id": "b", "split": "val", "group_id": "room2", "duration_s": 8.0, "sha256": "def"},
        ],
    }
    summary = validate_manifest(good_manifest)
    assert summary["n_items"] == 2

    bad_leak = dict(good_manifest, corpus="librispeech")
    with pytest.raises(AssertionError):
        validate_manifest(bad_leak)

    bad_cov = dict(good_manifest, join_coverage=0.80)
    with pytest.raises(AssertionError):
        validate_manifest(bad_cov)

    bad_grp = dict(
        good_manifest,
        items=[
            {"id": "a", "split": "train", "group_id": "room1", "duration_s": 10.0, "sha256": "abc"},
            {"id": "b", "split": "val", "group_id": "room1", "duration_s": 8.0, "sha256": "def"},
        ],
    )
    with pytest.raises(AssertionError):
        validate_manifest(bad_grp)

    foa_path = tmp_path / "foa.wav"
    ch0 = _sine_wave(440.0, 1.0, amp=0.4)
    ch_zero = np.zeros_like(ch0)
    foa_wav = np.stack([ch0, -ch0, ch_zero, ch_zero], axis=1)
    sf.write(str(foa_path), foa_wav, SR, subtype="FLOAT")
    mono_ch0 = load_mono_audio(foa_path, target_sr=SR, channel=0)
    mono_avg = load_mono_audio(foa_path, target_sr=SR, channel=None)
    assert np.allclose(mono_ch0, ch0, atol=1e-5)
    assert float(np.abs(mono_avg).max()) < 1e-5

    set_seed(7)
    model = build_arm("E", stride=4, max_k=4)
    heads = MultiTaskHeads(d=model.out_dim, n_ev=15, n_spk=20, n_starss=13, n_ovl=3)
    e = torch.randn(2, 64, D_E)
    lens = torch.tensor([64, 48], dtype=torch.long)
    states, pad = model.forward_states(e, lens, depths=(1, 2, 4))
    y_st = torch.randint(0, 2, (2, pad.shape[1], 13)).float()
    y_st_c = y_st.max(dim=1).values
    y_ovl = torch.randint(0, 3, (2, pad.shape[1]))
    loss = multidepth_event_loss(
        states, pad, heads, y_st, y_st_c, head_name="starss"
    ) + multidepth_overlap_loss(states, pad, heads, y_ovl)
    assert torch.isfinite(loss)
    loss.backward()
    assert heads.starss_head is not None and heads.starss_head.weight.grad is not None
    assert heads.ovl_head is not None and heads.ovl_head.weight.grad is not None
