import torch
import torch.nn.functional as F

from loopear.d.causal import (
    CAUSAL_ENCODER_PARAMS,
    HORIZONS,
    TAU,
    VAR_WEIGHT,
    CausalLoopEarD,
    CausalLoopEarDEncoder,
    batch_variance_hinge,
    causal_step_loss,
    crop_waveform,
    lambda_mix_at,
    mix_at_snr,
    pit_cosine_loss,
    sample_snr,
    temporal_infonce,
    visible_from_frames,
)
from loopear.d.frontend import build_mel, waveform_logmel
from loopear.d.spec import D_MODEL, ENCODER_PARAMS, GRID_F, GRID_T, N_FRAMES, N_MELS, SR
from loopear.losses import set_seed


def _mel(batch: int = 2, seed: int = 0, frames: int = N_FRAMES) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    mel = torch.randn(batch, N_FRAMES, N_MELS, generator=gen)
    valid = torch.zeros(batch, N_FRAMES, dtype=torch.bool)
    valid[:, :frames] = True
    return mel * valid.unsqueeze(-1), valid


def _manual_infonce(query, target, valid_time, horizon, tau, neighbor):
    losses = []
    b, t, _d = query.shape
    for row in range(b):
        for q in range(t):
            pos = q + horizon
            if pos >= t or not bool(valid_time[row, q]) or not bool(valid_time[row, pos]):
                continue
            numer = torch.exp(torch.dot(query[row, q], target[row, pos]) / tau)
            denom = numer.new_zeros(())
            for s in range(t):
                if not bool(valid_time[row, s]):
                    continue
                if abs(s - pos) == neighbor:
                    continue
                denom = denom + torch.exp(torch.dot(query[row, q], target[row, s]) / tau)
            losses.append(float(-torch.log(numer / denom)))
    return float(sum(losses) / len(losses)), len(losses)


def test_waveform_slice_matches_the_mel_crop():
    gen = torch.Generator().manual_seed(8)
    wav = torch.randn(SR * 3, generator=gen)
    bank = build_mel()
    full = waveform_logmel(wav, bank)
    start = 11
    recon = waveform_logmel(crop_waveform(wav, start), bank)
    assert int(recon.shape[0]) == N_FRAMES
    assert torch.allclose(recon, full[start : start + N_FRAMES], atol=1e-5)
    short = wav[:5000]
    short_full = waveform_logmel(short, bank)
    short_recon = waveform_logmel(crop_waveform(short, 0), bank)
    n = int(short_full.shape[0])
    assert n < N_FRAMES
    assert torch.allclose(short_recon[:n], short_full, atol=1e-5)


def test_causal_encoder_drops_the_class_token_and_keeps_the_grid():
    encoder = CausalLoopEarDEncoder()
    n_params = sum(p.numel() for p in encoder.parameters())
    assert n_params == CAUSAL_ENCODER_PARAMS == ENCODER_PARAMS - D_MODEL
    assert "cls" not in dict(encoder.named_parameters())
    mel, frame_valid = _mel(2)
    visible = visible_from_frames(frame_valid)
    grid, summary = encoder(mel, visible)
    assert tuple(grid.shape) == (2, GRID_T, GRID_F, D_MODEL)
    assert tuple(summary.shape) == (2, D_MODEL)
    assert torch.allclose(summary, grid.mean(dim=(1, 2)))


def test_attention_is_block_causal_and_keeps_every_patch():
    set_seed(3)
    encoder = CausalLoopEarDEncoder()
    encoder.eval()
    mel, frame_valid = _mel(1, seed=3)
    visible = visible_from_frames(frame_valid)
    base, _summary = encoder(mel, visible)
    future = mel.clone()
    future[:, 100:, :] += 2.5
    future_out, _summary = encoder(future, visible)
    assert torch.allclose(base[:, :25], future_out[:, :25], atol=1e-5)
    assert not torch.allclose(base[:, 30], future_out[:, 30], atol=1e-4)
    same = mel.clone()
    same[:, 40:44, 16:32] += 3.0
    same_out, _summary = encoder(same, visible)
    assert torch.allclose(base[:, :10], same_out[:, :10], atol=1e-5)
    assert not torch.allclose(base[:, 10, 0], same_out[:, 10, 0], atol=1e-4)
    assert not torch.allclose(base[:, 11], same_out[:, 11], atol=1e-4)
    holed = visible.clone()
    holed[:, 5, :] = False
    past = mel.clone()
    past[:, 16:20, :] += 2.0
    later = mel.clone()
    later[:, 28:32, :] += 2.0
    with_past, _summary = encoder(past, holed)
    with_later, _summary = encoder(later, holed)
    reference, _summary = encoder(mel, holed)
    assert not torch.allclose(reference[:, 6], with_past[:, 6], atol=1e-4)
    assert torch.allclose(reference[:, 6], with_later[:, 6], atol=1e-5)
    assert torch.count_nonzero(reference[:, 5]) == 0


def test_within_clip_infonce_masks_distance_one_neighbors():
    gen = torch.Generator().manual_seed(4)
    query = F.normalize(torch.randn(2, 8, 4, generator=gen), dim=-1)
    target = F.normalize(torch.randn(2, 8, 4, generator=gen), dim=-1)
    valid = torch.ones(2, 8, dtype=torch.bool)
    valid[1, 6:] = False
    loss, count = temporal_infonce(query, target, valid, horizon=4, tau=TAU, neighbor=1)
    manual, manual_count = _manual_infonce(query, target, valid, 4, TAU, 1)
    assert count == manual_count == 6
    assert abs(float(loss) - manual) < 1e-5
    skipped, skipped_count = temporal_infonce(query, target, valid, horizon=12)
    assert skipped_count == 0
    assert float(skipped) == 0.0
    short = torch.zeros(1, 50, dtype=torch.bool)
    short[:, :8] = True
    q50 = F.normalize(torch.randn(1, 50, 4, generator=gen), dim=-1)
    t50 = F.normalize(torch.randn(1, 50, 4, generator=gen), dim=-1)
    _loss, n_fit = temporal_infonce(q50, t50, short, horizon=12)
    assert n_fit == 0


def test_variance_hinge_and_pit_and_snr_and_ramp():
    dead = torch.ones(4, 3, 2, 6)
    visible = torch.ones(4, 3, 2, dtype=torch.bool)
    assert float(batch_variance_hinge(dead, visible)) > 0.9
    alive = torch.randn(8, 3, 2, 6) * 3
    alive_visible = torch.ones(8, 3, 2, dtype=torch.bool)
    assert float(batch_variance_hinge(alive, alive_visible)) < 0.05
    gen = torch.Generator().manual_seed(5)
    tgt_a = F.normalize(torch.randn(3, 5, 4, generator=gen), dim=-1)
    tgt_b = F.normalize(torch.randn(3, 5, 4, generator=gen), dim=-1)
    valid = torch.ones(3, 5, dtype=torch.bool)
    matched = pit_cosine_loss(tgt_a, tgt_b, tgt_a, tgt_b, valid)
    swapped = pit_cosine_loss(tgt_b, tgt_a, tgt_a, tgt_b, valid)
    flipped = pit_cosine_loss(tgt_a, tgt_b, tgt_b, tgt_a, valid)
    assert float(matched) < 1e-5
    assert abs(float(matched) - float(swapped)) < 1e-5
    assert abs(float(matched) - float(flipped)) < 1e-5
    wave = torch.randn(4, 2000, generator=gen)
    other = torch.roll(wave, 1, 0)
    snr = torch.tensor([-5.0, -1.0, 0.0, 5.0])
    mixed, scale = mix_at_snr(wave, other, snr)
    interferer = mixed - wave
    measured = 10.0 * torch.log10(wave.pow(2).mean(-1) / interferer.pow(2).mean(-1).clamp_min(1e-8))
    assert torch.allclose(measured, snr, atol=1e-4)
    assert torch.allclose(interferer, scale[:, None] * other, atol=1e-5)
    drawn = sample_snr(64, torch.Generator().manual_seed(9))
    assert float(drawn.min()) >= -5.0 and float(drawn.max()) <= 5.0
    assert lambda_mix_at(0) == 0.0
    assert abs(lambda_mix_at(500) - 0.25) < 1e-12
    assert lambda_mix_at(1000) == 0.5
    assert lambda_mix_at(4000) == 0.5
    assert VAR_WEIGHT == 0.1
    assert HORIZONS == (1, 4, 12)


def test_step_isolates_the_teacher_and_has_no_global_head():
    set_seed(11)
    model = CausalLoopEarD(separation=False)
    assert not hasattr(model, "teacher_local")
    assert model.sep_heads is None
    for left, right in zip(model.encoder.parameters(), model.teacher_encoder.parameters()):
        assert torch.equal(left, right)
        assert right.requires_grad is False
    mel, frame_valid = _mel(4, seed=11)
    out = causal_step_loss(model, mel, frame_valid)
    assert torch.isfinite(out["loss"])
    assert float(out["lambda_mix"]) == 0.0
    assert float(out["n_h1"]) == 4 * 49
    assert float(out["n_h4"]) == 4 * 46
    assert float(out["n_h12"]) == 4 * 38
    out["loss"].backward()
    assert model.encoder.patch.weight.grad is not None
    assert all(p.grad is None for p in model.teacher_encoder.parameters())
    with torch.no_grad():
        model.encoder.patch.weight.add_(0.05)
    before = [p.detach().clone() for p in model.teacher_encoder.parameters()]
    model.ema_update(0.5)
    moved = False
    for old, teacher, student in zip(before, model.teacher_encoder.parameters(), model.encoder.parameters()):
        assert teacher.requires_grad is False
        if not torch.equal(old, teacher):
            moved = True
        expect = 0.5 * old + 0.5 * student.detach()
        assert torch.allclose(teacher, expect)
    assert moved


def test_mixture_step_uses_two_heads_and_a_ramped_weight():
    set_seed(13)
    model = CausalLoopEarD(separation=True)
    mel, frame_valid = _mel(4, seed=13)
    mix = mel + 0.2 * torch.roll(mel, 1, 0)
    early = causal_step_loss(model, mel, frame_valid, mix, frame_valid, step=0)
    late = causal_step_loss(model, mel, frame_valid, mix, frame_valid, step=1000)
    assert float(early["lambda_mix"]) == 0.0
    assert float(late["lambda_mix"]) == 0.5
    assert torch.isfinite(late["loss_pit"])
    late["loss"].backward()
    assert model.sep_heads[0].fc1.weight.grad is not None
    assert model.sep_heads[1].fc2.weight.grad is not None
    assert all(p.grad is None for p in model.teacher_encoder.parameters())
