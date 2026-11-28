import math
from pathlib import Path

import numpy as np
import torch

from loopear.d.diagnostics import analytic_encoder_flops, embedding_diagnostics
from loopear.d.frontend import build_mel, frame_count, waveform_logmel
from loopear.d.fsd import (
    MelCache,
    assert_train_rows,
    assign_nested_subsets,
    mark_metadata_duplicates,
    rank_eligible,
)
from loopear.d.loop import (
    forward_batch,
    init_model,
    load_checkpoint,
    make_optimizer,
    save_checkpoint,
    snapshot_teacher,
    teacher_matches_formula,
    train_step,
)
from loopear.d.model import patchify
from loopear.d.objective import (
    broadcast_time_valid,
    global_ntxent,
    local_mld,
    patch_valid_from_frames,
    sample_freq_mask,
    sample_time_mask,
    view_masks,
)
from loopear.d.spec import (
    CROP_SAMPLES,
    D_MODEL,
    ENCODER_PARAMS,
    GLOBAL_HEAD_PARAMS,
    GRID_F,
    GRID_T,
    LOCAL_HEAD_PARAMS,
    N_FRAMES,
    N_MELS,
    PREDICTOR_PARAMS,
    STUDENT_PARAMS,
    TEACHER_PARAMS,
    ema_at,
    lr_at,
    warmup_steps,
)


def _pair(batch: int = 4, valid_frames: int = N_FRAMES, seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    mel1 = torch.randn(batch, N_FRAMES, N_MELS, generator=gen)
    mel2 = torch.randn(batch, N_FRAMES, N_MELS, generator=gen)
    valid = torch.zeros(batch, N_FRAMES, dtype=torch.bool)
    valid[:, :valid_frames] = True
    mel1 = mel1 * valid.unsqueeze(-1)
    mel2 = mel2 * valid.unsqueeze(-1)
    return mel1, mel2, valid, valid.clone()


def _manual_global(z1: torch.Tensor, z2: torch.Tensor, tau: float) -> float:
    losses = []
    batch = z1.shape[0]
    for view_a, view_b in ((z1, z2), (z2, z1)):
        for i in range(batch):
            numer = torch.exp(torch.dot(view_a[i], view_b[i]) / tau)
            denom = numer.clone()
            for j in range(batch):
                if j != i:
                    denom = denom + torch.exp(torch.dot(view_a[i], view_a[j]) / tau)
                if j != i:
                    denom = denom + torch.exp(torch.dot(view_a[i], view_b[j]) / tau)
            losses.append(float(-torch.log(numer / denom)))
    return float(sum(losses) / len(losses))


def test_locked_parameter_counts_and_shapes():
    model = init_model(42)
    encoder_n = sum(p.numel() for p in model.encoder.parameters())
    predictor_n = sum(p.numel() for p in model.predictor.parameters())
    global_n = sum(p.numel() for p in model.global_head.parameters())
    local_n = sum(p.numel() for p in model.local_head.parameters())
    teacher_n = sum(p.numel() for p in model.teacher_encoder.parameters()) + sum(
        p.numel() for p in model.teacher_local.parameters()
    )
    student_n = encoder_n + predictor_n + global_n + local_n
    assert encoder_n == ENCODER_PARAMS == 682752
    assert predictor_n == PREDICTOR_PARAMS == 66192
    assert global_n == GLOBAL_HEAD_PARAMS == 20864
    assert local_n == LOCAL_HEAD_PARAMS == 15712
    assert student_n == STUDENT_PARAMS
    assert teacher_n == TEACHER_PARAMS
    assert "pos" not in dict(model.predictor.named_parameters())
    mel, _, valid, _ = _pair(2)
    visible = broadcast_time_valid(patch_valid_from_frames(valid))
    grid, cls = model.encoder(mel, visible)
    assert tuple(patchify(mel).shape) == (2, GRID_T, GRID_F, 64)
    assert tuple(grid.shape) == (2, GRID_T, GRID_F, D_MODEL)
    assert tuple(cls.shape) == (2, D_MODEL)
    flops = analytic_encoder_flops(1, GRID_T * GRID_F + 1)
    assert flops["gemm_gflops"] > 0.2


def test_student_teacher_init_ema_and_grad_isolation():
    model = init_model(7)
    for left, right in zip(model.encoder.parameters(), model.teacher_encoder.parameters()):
        assert torch.equal(left, right)
    for left, right in zip(model.local_head.parameters(), model.teacher_local.parameters()):
        assert torch.equal(left, right)
    for param in list(model.teacher_encoder.parameters()) + list(model.teacher_local.parameters()):
        assert param.requires_grad is False
    student_ids = {id(p) for p in model.student_parameters()}
    teacher_ids = {id(p) for p in model.teacher_encoder.parameters()}
    teacher_ids.update(id(p) for p in model.teacher_local.parameters())
    assert student_ids.isdisjoint(teacher_ids)
    before = snapshot_teacher(model)
    with torch.no_grad():
        model.encoder.cls.add_(0.25)
        model.local_head.fc2.weight.add_(0.1)
    model.ema_update(0.5)
    assert teacher_matches_formula(model, 0.5, before)
    mel1, mel2, v1, v2 = _pair(2, seed=3)
    gen = torch.Generator().manual_seed(9)
    out = forward_batch(model, mel1, mel2, v1, v2, gen, "k1")
    model.zero_grad(set_to_none=True)
    out["loss_g"].backward(retain_graph=True)
    grad_g = model.encoder.patch.weight.grad.detach().clone()
    model.zero_grad(set_to_none=True)
    out["loss_l"].backward()
    grad_l = model.encoder.patch.weight.grad.detach().clone()
    assert float(grad_g.abs().sum()) > 0.0
    assert float(grad_l.abs().sum()) > 0.0
    assert not torch.allclose(grad_g, grad_l)
    for param in list(model.teacher_encoder.parameters()) + list(model.teacher_local.parameters()):
        assert param.grad is None


def test_hand_checked_losses_and_negative_set():
    z1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    z2 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    got = float(global_ntxent(z1, z2, 0.1))
    assert abs(got - _manual_global(z1, z2, 0.1)) < 1e-5
    tau = 0.1
    pos = math.exp(1.0 / tau)
    neg = math.exp(0.0)
    expected = -math.log(pos / (pos + neg))
    pred = torch.tensor([[[[1.0, 0.0]]], [[[0.0, 1.0]]]])
    target = pred.clone()
    anchor = torch.ones(2, 1, 1, dtype=torch.bool)
    valid = torch.ones(2, 1, 1, dtype=torch.bool)
    got_local = float(local_mld(pred, target, anchor, valid, tau, 0))
    assert abs(got_local - expected) < 1e-5
    pred2 = torch.zeros(2, 2, 1, 2)
    target2 = torch.zeros(2, 2, 1, 2)
    pred2[:, 0, 0, 0] = 1.0
    pred2[1, 0, 0] = torch.tensor([0.0, 1.0])
    target2[:, 0, 0, 0] = 1.0
    target2[1, 0, 0] = torch.tensor([0.0, 1.0])
    target2[:, 1, 0] = torch.tensor([1.0, 0.0])
    mask = torch.zeros(2, 2, 1, dtype=torch.bool)
    mask[:, 0, 0] = True
    valid2 = torch.ones(2, 2, 1, dtype=torch.bool)
    got_coord = float(local_mld(pred2, target2, mask, valid2, tau, 0))
    assert abs(got_coord - expected) < 1e-5
    leaked = target2.clone()
    leaked[:, 1] = torch.tensor([0.0, 1.0])
    got_leaked = float(local_mld(pred2, leaked, mask, valid2, tau, 0))
    assert abs(got_leaked - got_coord) < 1e-5


def test_neighbor_eligibility_and_padding():
    tau = 0.1
    pred = torch.randn(2, 3, 1, 4)
    pred = torch.nn.functional.normalize(pred, dim=-1)
    target = torch.nn.functional.normalize(torch.randn(2, 3, 1, 4), dim=-1)
    valid = torch.ones(2, 3, 1, dtype=torch.bool)
    center = torch.zeros(2, 3, 1, dtype=torch.bool)
    center[:, 1, 0] = True
    loss_center = local_mld(pred, target, center, valid, tau, 1)
    shifted = target.clone()
    shifted[:, 0] = torch.nn.functional.normalize(torch.randn(2, 1, 4), dim=-1)
    shifted[:, 2] = torch.nn.functional.normalize(torch.randn(2, 1, 4), dim=-1)
    loss_shift = local_mld(pred, shifted, center, valid, tau, 1)
    assert torch.allclose(loss_center, loss_shift, atol=1e-6)
    changed_center = shifted.clone()
    changed_center[:, 1] = torch.nn.functional.normalize(torch.randn(2, 1, 4), dim=-1)
    loss_changed = local_mld(pred, changed_center, center, valid, tau, 1)
    assert not torch.allclose(loss_center, loss_changed, atol=1e-5)
    all_masked = torch.ones(2, 3, 1, dtype=torch.bool)
    base = local_mld(pred, target, all_masked, valid, tau, 1)
    neighbor = target.clone()
    neighbor[:, 0] = torch.nn.functional.normalize(torch.randn(2, 1, 4), dim=-1)
    moved = local_mld(pred, neighbor, all_masked, valid, tau, 1)
    assert not torch.allclose(base, moved, atol=1e-5)
    only_center = local_mld(pred, target, center, valid, tau, 0)
    only_center_k = local_mld(pred, target, center, valid, tau, 1)
    assert torch.allclose(only_center, only_center_k, atol=1e-6)
    pred_b = torch.nn.functional.normalize(torch.randn(3, 1, 1, 4), dim=-1)
    target_b = torch.nn.functional.normalize(torch.randn(3, 1, 1, 4), dim=-1)
    anchor = torch.ones(3, 1, 1, dtype=torch.bool)
    valid_b = torch.tensor([[[True]], [[True]], [[False]]])
    target_pad = target_b.clone()
    target_pad[2] = 0
    loss_drop = local_mld(pred_b, target_b, anchor, valid_b, tau, 0)
    loss_noise = local_mld(pred_b, target_pad, anchor, valid_b, tau, 0)
    assert torch.allclose(loss_drop, loss_noise, atol=1e-6)
    manual_pred = pred_b[:2, :, :, :]
    manual_tgt = target_b[:2, :, :, :]
    manual = local_mld(
        manual_pred,
        manual_tgt,
        torch.ones(2, 1, 1, dtype=torch.bool),
        torch.ones(2, 1, 1, dtype=torch.bool),
        tau,
        0,
    )
    assert torch.allclose(loss_drop, manual, atol=1e-5)


def test_masks_pad_and_no_predictor_bypass():
    gen = torch.Generator().manual_seed(4)
    valid_time = torch.ones(3, GRID_T, dtype=torch.bool)
    valid_time[0, 20:] = False
    time_mask = sample_time_mask(valid_time, gen)
    assert int(time_mask[1].sum()) == 30
    assert int(time_mask[0].sum()) == int(round(0.6 * 20))
    assert not bool(time_mask[0, 20:].any())
    freq = sample_freq_mask(3, gen)
    assert int(freq[0].sum()) == 2
    pad = torch.zeros(2, GRID_T, GRID_F, dtype=torch.bool)
    pad[:, 40:] = True
    gen2 = torch.Generator().manual_seed(5)
    masked_t, masked_f = view_masks(pad, gen2)
    assert not bool((masked_t & pad).any())
    assert not bool((masked_f & pad).any())
    assert bool(masked_t[:, :, 0].equal(masked_t[:, :, 1]))
    assert bool(masked_f[:, 0, :].equal(masked_f[:, 1, :]))
    model = init_model(11)
    mel = torch.randn(2, N_FRAMES, N_MELS)
    visible = torch.ones(2, GRID_T, GRID_F, dtype=torch.bool)
    visible[:, 10:20] = False
    pad_grid = torch.zeros_like(visible)
    masked = ~visible
    grid, cls = model.encoder(mel, visible)
    assert torch.count_nonzero(grid[masked]) == 0
    mel_bad = mel.clone()
    mel_bad[:, 40:80] += 4.0
    grid_b, cls_b = model.encoder(mel_bad, visible)
    assert torch.allclose(grid, grid_b, atol=0.0, rtol=0.0)
    assert torch.allclose(cls, cls_b, atol=0.0, rtol=0.0)
    pred_in = model.predictor.inputs(grid, masked, pad_grid)
    assert torch.allclose(
        pred_in[masked], model.predictor.mask_token.view(1, 1, 1, -1).expand_as(pred_in)[masked]
    )
    junk = grid.clone()
    junk[masked] = 8.0
    pred_in_b = model.predictor.inputs(junk, masked, pad_grid)
    assert torch.allclose(pred_in, pred_in_b)
    short = torch.ones(2, N_FRAMES, dtype=torch.bool)
    short[:, 80:] = False
    mel_a = torch.randn(2, N_FRAMES, N_MELS)
    mel_c = mel_a.clone()
    mel_a[:, 80:] = 0
    mel_c[:, 80:] = 5
    vis = broadcast_time_valid(patch_valid_from_frames(short))
    ga, ca = model.encoder(mel_a, vis)
    gb, cb = model.encoder(mel_c, vis)
    assert torch.allclose(ca, cb)
    assert torch.allclose(ga[vis], gb[vis])
    gen_a = torch.Generator().manual_seed(19)
    gen_b = torch.Generator().manual_seed(19)
    loss_a = forward_batch(model, mel_a, mel_a, short, short, gen_a, "center")["loss"]
    loss_b = forward_batch(model, mel_c, mel_c, short, short, gen_b, "center")["loss"]
    assert torch.allclose(loss_a, loss_b)


def test_checkpoint_resume_and_deterministic_eval(tmp_path: Path):
    mel1, mel2, v1, v2 = _pair(2, seed=15)
    model = init_model(0)
    opt = make_optimizer(model, 3e-4)
    gen = torch.Generator().manual_seed(123)
    train_step(model, opt, mel1, mel2, v1, v2, gen, "k1", 0, 20)
    gen_state = gen.get_state()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(str(path), model, opt, 1, 20, "k1", {"seed": 0})
    continued = train_step(model, opt, mel1, mel2, v1, v2, gen, "k1", 1, 20)
    student_after = {k: v.detach().clone() for k, v in model.encoder.state_dict().items()}
    teacher_after = {
        k: v.detach().clone() for k, v in model.teacher_encoder.state_dict().items()
    }
    resumed = init_model(99)
    opt_r = make_optimizer(resumed, 1e-2)
    load_checkpoint(str(path), resumed, opt_r)
    gen_r = torch.Generator()
    gen_r.set_state(gen_state)
    resumed_stats = train_step(resumed, opt_r, mel1, mel2, v1, v2, gen_r, "k1", 1, 20)
    for key, value in student_after.items():
        assert torch.allclose(resumed.encoder.state_dict()[key], value, atol=1e-7, rtol=0.0)
    for key, value in teacher_after.items():
        assert torch.allclose(resumed.teacher_encoder.state_dict()[key], value, atol=1e-7, rtol=0.0)
    assert abs(continued["loss"] - resumed_stats["loss"]) < 1e-6
    resumed.eval()
    visible = torch.ones(2, GRID_T, GRID_F, dtype=torch.bool)
    with torch.no_grad():
        a1, b1 = resumed.encoder(mel1, visible)
        a2, b2 = resumed.encoder(mel1, visible)
    assert torch.equal(a1, a2)
    assert torch.equal(b1, b2)
    assert warmup_steps(1000) == 50
    assert abs(lr_at(0, 1000) - (3e-4 / 50.0)) < 1e-12
    assert abs(ema_at(0, 1000) - 0.99) < 1e-12
    assert abs(ema_at(999, 1000) - 0.9995) < 1e-12


def test_frontend_frame_count_and_cache_has_no_labels(tmp_path: Path):
    assert CROP_SAMPLES == 32240
    assert frame_count(CROP_SAMPLES) == N_FRAMES
    mel = build_mel()
    wav = torch.randn(CROP_SAMPLES)
    spec = waveform_logmel(wav, mel)
    assert tuple(spec.shape) == (N_FRAMES, N_MELS)
    assert torch.isfinite(spec).all()
    cache = tmp_path / "mel"
    cache.mkdir()
    long = np.random.default_rng(0).standard_normal((240, N_MELS)).astype(np.float16)
    short = np.random.default_rng(1).standard_normal((20, N_MELS)).astype(np.float16)
    np.save(cache / "10.npy", long)
    np.save(cache / "11.npy", short)
    rows = [
        {"freesound_id": 10, "split": "train", "duplicate_of": None, "labels": ["Bark"]},
        {"freesound_id": 11, "split": "train", "duplicate_of": None, "labels": ["Dog"]},
    ]
    store = MelCache(
        rows,
        cache,
        mean=torch.zeros(N_MELS),
        std=torch.ones(N_MELS),
    )
    batch = store.crops([0, 1], torch.Generator().manual_seed(2))
    assert set(batch) == {"mel1", "mel2", "frame_valid1", "frame_valid2", "freesound_id"}
    assert "labels" not in batch
    assert tuple(batch["mel1"].shape) == (2, N_FRAMES, N_MELS)
    assert int(batch["frame_valid1"][1].sum()) == 20
    assert int(patch_valid_from_frames(batch["frame_valid1"])[1].sum()) == 5
    bad = list(rows)
    bad[0]["split"] = "test"
    try:
        assert_train_rows(bad)
    except PermissionError:
        pass
    else:
        raise AssertionError("test split was accepted")


def test_nested_manifest_ranking_and_duplicates():
    rows = []
    for index in range(30):
        rows.append(
            {
                "freesound_id": 1000 + index,
                "split": "train",
                "duration_s": 2000.0 if index % 2 == 0 else 1.0,
                "uploader": f"u{index % 3}",
                "title": f"clip-{index}",
                "labels": ["Bark" if index % 2 == 0 else "Meow"],
                "duplicate_of": None,
            }
        )
    rows.append(
        {
            "freesound_id": 50,
            "split": "train",
            "duration_s": 3.0,
            "uploader": "same",
            "title": "dup",
            "labels": ["Bark"],
            "duplicate_of": None,
        }
    )
    rows.append(
        {
            "freesound_id": 80,
            "split": "train",
            "duration_s": 3.0,
            "uploader": "same",
            "title": "dup",
            "labels": ["Bark"],
            "duplicate_of": None,
        }
    )
    rows.append(
        {
            "freesound_id": 9,
            "split": "val",
            "duration_s": 4.0,
            "uploader": "v",
            "title": "held",
            "labels": ["Bark"],
            "duplicate_of": None,
        }
    )
    report = mark_metadata_duplicates(rows)
    assert report["n_removed"] == 1
    kept = [r for r in rows if r["freesound_id"] == 50][0]
    dropped = [r for r in rows if r["freesound_id"] == 80][0]
    assert kept["duplicate_of"] is None
    assert dropped["duplicate_of"] == 50
    ranked = rank_eligible(rows)
    assert all(r["split"] == "train" and not r["duplicate_of"] for r in ranked)
    assert 80 not in {int(r["freesound_id"]) for r in ranked}
    assert 9 not in {int(r["freesound_id"]) for r in ranked}
    totals = assign_nested_subsets(ranked)
    h5 = [r for r in ranked if r["subset"] == "h5"]
    h20 = [r for r in ranked if r["subset"] in ("h5", "h20")]
    assert totals["h5_seconds"] >= 5 * 3600
    assert totals["h5_seconds"] - float(h5[-1]["duration_s"]) < 5 * 3600
    assert {int(r["freesound_id"]) for r in h5} < {int(r["freesound_id"]) for r in h20}
    assert h20[-1]["rank"] == len(h20) - 1


def test_collapse_diagnostics_flags_a_constant_embedding():
    alive = embedding_diagnostics(torch.randn(16, 8))
    assert alive["collapsed"] is False
    assert alive["finite"] is True
    assert float(alive["effective_rank"]) > 2.0
    dead = embedding_diagnostics(torch.ones(16, 8))
    assert dead["collapsed"] is True
    assert float(dead["offdiag_cosine_mean"]) > 0.99
