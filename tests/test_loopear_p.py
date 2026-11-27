import torch

from loopear.d.diagnostics import analytic_encoder_flops
from loopear.d.model import patchify
from loopear.d.objective import (
    broadcast_time_valid,
    global_ntxent,
    local_mld,
    patch_valid_from_frames,
    sample_freq_mask,
    sample_time_mask,
)
from loopear.d.perceiver import (
    LatentBankPredictor,
    LoopEarPEncoder,
    analytic_perceiver_flops,
    apply_time_bias,
    attention_shift_sums,
    centered_variation_sums,
    content_and_anchor_sums,
    encoder_parameter_count,
    init_perceiver,
    init_perceiver_local,
    patch_and_latent_time,
    perceiver_local_loss,
    visible_cross_logit_stats,
    within_recording_time_sums,
)
from loopear.d.spec import (
    D_MODEL,
    ENCODER_PARAMS,
    GRID_F,
    GRID_T,
    N_FRAMES,
    N_LATENTS,
    N_MELS,
    PERCEIVER_ENCODER_PARAMS,
    TAU_GLOBAL,
)


def _mel(batch: int = 2, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    mel = torch.randn(batch, N_FRAMES, N_MELS, generator=gen)
    valid = torch.ones(batch, N_FRAMES, dtype=torch.bool)
    visible = broadcast_time_valid(patch_valid_from_frames(valid))
    return mel, visible


def test_perceiver_parameter_count_is_near_loopear_d():
    model = init_perceiver(42)
    counted = sum(p.numel() for p in model.encoder.parameters())
    assert counted == encoder_parameter_count() == PERCEIVER_ENCODER_PARAMS
    assert counted == 685728
    assert abs(counted - ENCODER_PARAMS) == (N_LATENTS - 1) * D_MODEL
    assert counted / float(ENCODER_PARAMS) < 1.01


def test_perceiver_shapes_and_summary_is_the_latent_mean():
    model = init_perceiver(3)
    mel, visible = _mel()
    latents, summary = model.encoder(mel, visible)
    assert tuple(patchify(mel).shape) == (2, GRID_T, GRID_F, 64)
    assert tuple(latents.shape) == (2, N_LATENTS, D_MODEL)
    assert tuple(summary.shape) == (2, D_MODEL)
    assert torch.allclose(summary, latents.mean(dim=1))
    assert torch.isfinite(latents).all()


def test_masked_patches_do_not_change_the_latents():
    model = init_perceiver(5)
    model.eval()
    mel, visible = _mel(1, seed=9)
    hidden = visible.clone()
    hidden[:, GRID_T // 2 :, :] = False
    edited = mel.clone()
    edited[:, (GRID_T // 2) * 4 :, :] = 50.0
    with torch.no_grad():
        left, _ = model.encoder(mel, hidden)
        right, _ = model.encoder(edited, hidden)
    assert torch.allclose(left, right, atol=1e-5, rtol=1e-5)


def test_visible_audio_does_change_the_latents():
    model = init_perceiver(5)
    model.eval()
    mel, visible = _mel(1, seed=9)
    edited = mel.clone()
    edited[:, :20, :] = edited[:, :20, :] + 3.0
    with torch.no_grad():
        left, _ = model.encoder(mel, visible)
        right, _ = model.encoder(edited, visible)
    assert not torch.allclose(left, right, atol=1e-4, rtol=1e-4)


def test_global_objective_matches_and_local_loss_stays_off():
    model = init_perceiver(11)
    mel1, visible = _mel(4, seed=1)
    mel2, _ = _mel(4, seed=2)
    valid = patch_valid_from_frames(torch.ones(4, N_FRAMES, dtype=torch.bool))
    gen = torch.Generator().manual_seed(91)
    time_mask = sample_time_mask(valid, gen)
    freq_mask = sample_freq_mask(4, gen)
    see1 = broadcast_time_valid(~time_mask) & visible
    see2 = broadcast_time_valid(torch.ones_like(valid)) 
    see2 = see2.clone()
    see2[:, :, :] = True
    freq = freq_mask[:, None, :].expand_as(visible)
    see2 = see2 & ~freq
    lat1, sum1 = model.encoder(mel1, see1)
    _lat2, sum2 = model.encoder(mel2, see2)
    z1 = model.project_global(sum1)
    z2 = model.project_global(sum2)
    loss = global_ntxent(z1, z2, TAU_GLOBAL)
    assert torch.isfinite(loss)
    assert float(loss) > 0.0
    dummy_grid = lat1[:, :1, :].view(4, 1, 1, D_MODEL).repeat(1, GRID_T, GRID_F, 1)
    kept = local_mld(
        dummy_grid,
        dummy_grid.detach(),
        torch.zeros(4, GRID_T, GRID_F, dtype=torch.bool),
        visible,
        TAU_GLOBAL,
        0,
    )
    assert float(kept) == 0.0
    loss.backward()
    assert model.encoder.patch.weight.grad is not None
    assert not hasattr(model, "predictor")
    assert not hasattr(model, "local_head")


def test_one_step_moves_the_encoder_on_cpu():
    model = init_perceiver(42)
    before = torch.nn.utils.parameters_to_vector(model.encoder.parameters()).detach().clone()
    opt = torch.optim.AdamW(model.student_parameters(), lr=3e-4, weight_decay=0.05)
    mel1, visible = _mel(4, seed=4)
    mel2, _ = _mel(4, seed=5)
    _lat1, sum1 = model.encoder(mel1, visible)
    _lat2, sum2 = model.encoder(mel2, visible)
    loss = global_ntxent(model.project_global(sum1), model.project_global(sum2), TAU_GLOBAL)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.student_parameters(), 1.0)
    opt.step()
    after = torch.nn.utils.parameters_to_vector(model.encoder.parameters()).detach()
    assert float(torch.linalg.vector_norm(after - before)) > 0.0


def test_perceiver_flops_are_below_the_token_transformer():
    ours = analytic_perceiver_flops(1, GRID_T * GRID_F)
    theirs = analytic_encoder_flops(1, GRID_T * GRID_F + 1)
    assert ours["gemm_gflops"] > 0.01
    assert ours["gemm_gflops"] < theirs["gemm_gflops"]
    assert ours["latents"] == float(N_LATENTS)
    assert ours["cross_score_bytes_one_layer"] < theirs["gemm_flops_mul_add"]
    encoder = LoopEarPEncoder()
    assert sum(p.numel() for p in encoder.parameters()) == PERCEIVER_ENCODER_PARAMS


def test_zero_alpha_matches_ordinary_cross_attention_and_keeps_the_parameter_count():
    torch.manual_seed(0)
    left = LoopEarPEncoder()
    right = LoopEarPEncoder()
    right.load_state_dict(left.state_dict())
    left.time_alpha = 0.0
    right.time_alpha = 0.0
    mel, visible = _mel(2, seed=1)
    with torch.no_grad():
        latents, summary = left(mel, visible)
        other_latents, other_summary = right(mel, visible)
    assert torch.equal(latents, other_latents)
    assert torch.equal(summary, other_summary)
    assert sum(p.numel() for p in left.parameters()) == PERCEIVER_ENCODER_PARAMS
    scores = torch.zeros(1, 1, N_LATENTS, GRID_T * GRID_F)
    assert torch.equal(apply_time_bias(scores, 0.0), scores)
    patch_time, latent_time = patch_and_latent_time()
    assert float(latent_time[0]) == 0.0
    assert float(latent_time[-1]) == float(GRID_T - 1)
    assert float(patch_time[0]) == 0.0
    assert float(patch_time[-1]) == float(GRID_T - 1)
    biased = apply_time_bias(scores, 0.5)
    far = (GRID_T - 1) * GRID_F
    assert float(biased[0, 0, 0, 0]) == 0.0
    assert torch.allclose(biased[0, 0, 0, far], torch.tensor(-0.5 * float(GRID_T - 1) ** 2))
    assert torch.allclose(biased[0, 0, -1, far], torch.tensor(0.0), atol=1e-5)
    left.time_alpha = 4.0
    with torch.no_grad():
        moved, _summary = left(mel, visible)
    assert not torch.allclose(moved, latents, atol=1e-5, rtol=1e-5)
    assert sum(p.numel() for p in left.parameters()) == PERCEIVER_ENCODER_PARAMS
    stats = visible_cross_logit_stats(left, mel, visible)
    assert stats["count"] > 0.0
    assert stats["logit_sq"] > 0.0
    assert stats["penalty_sq"] > 0.0


def test_local_predictor_queries_latents_and_stops_the_teacher():
    model = init_perceiver_local(4)
    assert sum(p.numel() for p in model.encoder.parameters()) == PERCEIVER_ENCODER_PARAMS
    assert model.encoder.time_alpha == 0.0
    assert model.teacher_encoder.time_alpha == 0.0
    for module in model.predictor.modules():
        if isinstance(module, torch.nn.Linear):
            assert module.in_features != 64
    latents = torch.randn(2, N_LATENTS, D_MODEL)
    grid = model.predictor(latents, 0)
    assert grid.shape == (2, GRID_T, GRID_F, D_MODEL)
    shifted = latents.clone()
    shifted[0, 0, 0] = shifted[0, 0, 0] + 3.0
    assert not torch.allclose(grid, model.predictor(shifted, 0))
    try:
        model.predictor(torch.randn(2, GRID_T * GRID_F, D_MODEL), 0)
    except ValueError:
        rejected = True
    else:
        rejected = False
    assert rejected
    mel, visible = _mel(2, seed=2)
    hidden = visible.clone()
    hidden[:, 8:18, :] = False
    encoded, _summary = model.encoder(mel, hidden)
    prediction = model.project_local(model.predictor(encoded, 0))
    with torch.no_grad():
        target = model.project_teacher(model.teacher_predictor(encoded.detach(), 0))
    anchor = (~hidden) & visible
    loss_l = local_mld(prediction, target, anchor, visible, 0.1, 1)
    loss_l.backward()
    assert model.encoder.latents.grad is not None
    assert model.predictor.q.weight.grad is not None
    assert all(param.grad is None for param in model.teacher_encoder.parameters())
    assert all(param.grad is None for param in model.teacher_predictor.parameters())
    assert all(param.grad is None for param in model.teacher_local.parameters())
    fresh = init_perceiver_local(5)
    frame_valid = torch.ones(2, N_FRAMES, dtype=torch.bool)
    loss, loss_g, loss_local = perceiver_local_loss(
        fresh,
        mel,
        mel,
        frame_valid,
        frame_valid,
        torch.Generator().manual_seed(9),
        1.0,
    )
    assert torch.isfinite(loss)
    assert torch.isfinite(loss_g)
    assert torch.isfinite(loss_local)
    loss.backward()
    assert fresh.encoder.patch.weight.grad is not None
    assert all(param.grad is None for param in fresh.teacher_encoder.parameters())
    assert isinstance(model.predictor, LatentBankPredictor)


def test_anchor_penalty_sums_follow_the_locked_alpha_on_every_cross_block():
    torch.manual_seed(0)
    encoder = LoopEarPEncoder()
    encoder.time_alpha = 0.0
    mel, visible = _mel(2, seed=3)
    alpha = 3.1816019504258156e-06
    rows = content_and_anchor_sums(encoder, mel, visible, alpha)
    reference = visible_cross_logit_stats(encoder, mel, visible)
    assert len(rows) == 3
    assert rows[0]["count"] == reference["count"]
    assert abs(rows[0]["logit_sq"] - reference["logit_sq"]) / reference["logit_sq"] < 1e-5
    assert abs(rows[0]["penalty_sq"] - (alpha ** 2) * reference["penalty_sq"]) < 1e-5
    assert all(row["count"] > 0.0 and row["logit_sq"] > 0.0 for row in rows)


def test_within_recording_rank_counts_the_correct_moment():
    batch, time_n, freq_n, width = 2, 4, 3, 4
    teacher = torch.zeros(batch, time_n, freq_n, width)
    for index in range(time_n):
        teacher[:, index, :, index] = 1.0
    student = teacher.clone()
    valid = torch.ones(batch, time_n, freq_n, dtype=torch.bool)
    ranked = within_recording_time_sums(student, teacher, valid)
    assert ranked["n_queries_with_other_moment"] == float(batch * time_n * freq_n)
    assert ranked["n_correct_strictly_best"] == ranked["n_queries_with_other_moment"]
    assert ranked["n_teacher_pairs"] == float(batch * freq_n * (time_n * (time_n - 1) // 2))
    assert ranked["teacher_cosine_sum"] == 0.0
    assert ranked["n_correct_beats_other_recording_max"] == 0.0
    assert ranked["n_other_recording_pairs"] == float(batch * (batch - 1) * time_n * freq_n)
    flat = teacher[:, :1].expand(batch, time_n, freq_n, width).contiguous()
    tied = within_recording_time_sums(flat, flat, valid)
    assert tied["n_correct_strictly_best"] == 0.0
    assert tied["n_tied_for_best"] == tied["n_queries_with_other_moment"]
    assert tied["n_teacher_pairs_below_0_99"] == 0.0
    reversed_student = torch.flip(teacher, dims=(1,))
    missed = within_recording_time_sums(reversed_student, teacher, valid)
    assert missed["n_correct_strictly_best"] == 0.0
    assert missed["n_worse_than_another_moment"] == missed["n_queries_with_other_moment"]
    total = (
        ranked["n_correct_strictly_best"]
        + ranked["n_tied_for_best"]
        + ranked["n_worse_than_another_moment"]
    )
    assert total == ranked["n_queries_with_other_moment"]


def test_softmax_alpha_matches_centered_variation_and_moves_attention():
    torch.manual_seed(0)
    encoder = LoopEarPEncoder()
    mel, visible = _mel(2, seed=4)
    scale = centered_variation_sums(encoder, mel, visible)
    assert scale["count"] > 0.0
    assert scale["penalty_centered_sq"] > 0.0
    content_rms = (scale["content_centered_sq"] / scale["count"]) ** 0.5
    penalty_rms = (scale["penalty_centered_sq"] / scale["count"]) ** 0.5
    matched = content_rms / penalty_rms
    matched_shift = attention_shift_sums(encoder, mel, visible, matched)
    assert matched_shift["n_queries"] > 0.0
    assert matched_shift["constant_offset_tv_sum"] / matched_shift["n_queries"] < 1e-5
    assert matched_shift["removed_mean_tv_sum"] / matched_shift["n_queries"] < 1e-5
    alpha = 1.0 / penalty_rms
    shift = attention_shift_sums(encoder, mel, visible, alpha)
    assert shift["tv_sum"] / shift["n_queries"] > 0.05
    still = attention_shift_sums(encoder, mel, visible, 0.0)
    assert still["tv_sum"] == 0.0
    model = init_perceiver_local(1)
    model.set_time_alpha(float(alpha))
    model.train()
    assert model.encoder.time_alpha == alpha
    assert model.teacher_encoder.time_alpha == alpha
    assert sum(p.numel() for p in model.encoder.parameters()) == PERCEIVER_ENCODER_PARAMS
    frame_valid = torch.ones(2, N_FRAMES, dtype=torch.bool)
    loss, _loss_g, _loss_l = perceiver_local_loss(
        model, mel, mel, frame_valid, frame_valid, torch.Generator().manual_seed(2), 1.0
    )
    assert torch.isfinite(loss)
    encoder.time_alpha = 0.0
    with torch.no_grad():
        plain, _summary = encoder(mel, visible)
        encoder.time_alpha = float(alpha)
        moved, _summary = encoder(mel, visible)
    assert not torch.allclose(plain, moved)
