import torch

from loopear.slots import (
    PARAM_MAX,
    PARAM_MIN,
    ProvisionalListener,
    degenerate_report,
    measure_runtime,
    measure_support,
    parameter_count,
    straight_through_gate,
    variation_loss,
)


def _model() -> ProvisionalListener:
    torch.manual_seed(0)
    return ProvisionalListener()


def test_parameter_count_is_inside_the_provisional_band():
    count = parameter_count(_model())
    assert PARAM_MIN <= count <= PARAM_MAX


def test_zero_visible_slots_reconstruct_silence():
    model = _model()
    visible = torch.zeros(1, 5, model.slots, model.latent)
    assert torch.count_nonzero(model.to_mel(visible)) == 0


def test_inactive_slot_cannot_change_the_decoder():
    model = _model()
    raw = torch.randn(1, 6, model.slots, model.latent)
    gates = torch.tensor([0.2, 0.9, 0.9, 0.9, 0.9, 0.9]).view(1, 1, model.slots)
    gate = straight_through_gate(gates)
    visible = gate.unsqueeze(-1) * raw
    changed = raw.clone()
    changed[:, :, 0] += 3.0
    visible_changed = gate.unsqueeze(-1) * changed
    original = model.to_mel(visible).sum(dim=2)
    edited = model.to_mel(visible_changed).sum(dim=2)
    assert torch.allclose(original, edited, atol=1e-6, rtol=0.0)
    opened = straight_through_gate(torch.ones_like(gates)).unsqueeze(-1) * raw
    assert not torch.allclose(original, model.to_mel(opened).sum(dim=2), atol=1e-5, rtol=0.0)


def test_inactive_value_gets_no_evidence_gradient():
    slots = torch.randn(2, 4, requires_grad=True)
    gate = straight_through_gate(torch.tensor([0.2, 0.8]))
    visible = gate.unsqueeze(-1) * slots
    visible.square().sum().backward()
    assert slots.grad is not None
    assert float(slots.grad[0].abs().sum()) == 0.0
    assert float(slots.grad[1].abs().sum()) > 0.0


def test_evidence_gradient_reaches_the_backbone():
    model = _model()
    wav = torch.randn(1, 8000)
    init = torch.zeros(1, model.slots, model.latent)
    out = model(wav, slot_init=init)
    (out["reconstruction"] - out["target"]).abs().mean().backward()
    assert model.frame.weight.grad is not None
    assert float(model.frame.weight.grad.abs().sum()) > 0.0
    assert model.to_mel.weight.grad is not None
    assert float(model.to_mel.weight.grad.abs().sum()) > 0.0
    assert model.blocks[0].pw1.weight.grad is not None
    assert float(model.blocks[0].pw1.weight.grad.abs().sum()) > 0.0


def test_slot_init_is_not_stored_state():
    model = _model()
    wav = torch.randn(1, 8000)
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    first = torch.randn(1, model.slots, model.latent)
    second = torch.randn(1, model.slots, model.latent)
    left = model(wav, slot_init=first)["reconstruction"]
    right = model(wav, slot_init=second)["reconstruction"]
    assert not torch.allclose(left, right, atol=1e-6, rtol=0.0)
    for name, tensor in model.state_dict().items():
        assert torch.equal(before[name], tensor)


def test_decoder_is_permutation_invariant():
    model = _model()
    visible = torch.randn(1, 7, model.slots, model.latent)
    order = torch.randperm(model.slots)
    left = model.to_mel(visible).sum(dim=2)
    right = model.to_mel(visible[:, :, order]).sum(dim=2)
    assert torch.allclose(left, right, atol=1e-5, rtol=0.0)


def test_early_frame_ignores_later_audio_and_last_frame_ignores_distant_past():
    model = _model()
    generator = torch.Generator().manual_seed(4)
    wav = torch.randn(1, 32000, generator=generator)
    init = torch.zeros(1, model.slots, model.latent)
    base = model(wav, slot_init=init)["reconstruction"]
    later = wav.clone()
    later[:, 16000:] = torch.randn(1, 16000)
    early = model(later, slot_init=init)["reconstruction"]
    assert torch.allclose(base[0, 2], early[0, 2], atol=1e-5, rtol=0.0)
    distant = wav.clone()
    distant[:, :4000] = 0
    tail = model(distant, slot_init=init)["reconstruction"]
    assert torch.allclose(base[0, -1], tail[0, -1], atol=1e-5, rtol=0.0)


def test_support_is_bounded_in_both_directions():
    support = measure_support(_model(), n_samples=48000)
    assert support["lookahead_samples"] < 1600
    assert support["last_affecting_sample"] < 4000
    assert support["first_affecting_sample_of_last_frame"] > 16000


def test_runtime_is_faster_than_realtime_on_cpu():
    runtime = measure_runtime(_model(), seconds=2.0, repeats=2)
    assert runtime["rtf_1thread"] < 1.0
    assert runtime["rtf_4thread"] < 1.0
    assert runtime["conv_linear_macs_per_s"] > 0.0


def test_candidate_loss_blocks_the_listed_collapses():
    report = degenerate_report()
    assert report["selective_intervene"] < report["single_slot_intervene"]
    assert report["selective_intervene"] < report["duplicate_intervene"]
    assert report["selective_intervene"] < report["constant_intervene"]
    assert report["selective_intervene"] < report["hidden_inactive_intervene"]
    assert abs(report["selective_intervene"] - report["selective_swapped"]) < 1e-5
    assert abs(report["selective_intervene"] - report["selective_permuted"]) < 1e-5
    assert report["natural_same"] < report["shifted_same"]
    assert report["appear_represented"] < report["appear_missed"]
    assert report["within_as_intervene"] < report["within_as_same"]
    assert report["gain_as_global"] < report["gain_as_intervene"]
    assert report["quiet_keep_intervene"] < report["quiet_drop_intervene"]
    assert report["one_slot_moves"] < report["two_slots_move"]
    assert report["time_sustained_32"] < report["time_brief_32"]
    assert report["time_brief_32"] < report["time_no_move_32"]
    assert abs(report["time_brief_32"] - report["time_matched_smear_32"]) < 1e-5


def test_candidate_loss_fails_correlated_merge_nuisance_and_untrained_quiet_drop():
    report = degenerate_report()
    assert report["correlated_merged"] < report["correlated_split"]
    assert report["nuisance_intervene"] <= report["clean_intervene"] + 1e-6
    assert report["quiet_drop_same"] < report["quiet_keep_same"]


def test_pair_loss_matches_a_gradient_into_the_moving_slot():
    z = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]], requires_grad=True)
    z2 = torch.tensor([[[[1.0, 0.0]], [[0.0, 2.0]]]], requires_grad=True)
    g = torch.tensor([[1.0, 1.0]])
    y = torch.zeros(1, 1, 2)
    loss = variation_loss(
        z=z,
        g=g,
        yhat=y,
        y=y,
        z2=z2,
        g2=g,
        yhat2=y,
        y2=y,
        kind="intervene",
        unchanged_present=True,
    )["total"]
    loss.backward()
    assert z.grad is not None and z2.grad is not None
    assert bool(torch.isfinite(z.grad).all())
    assert bool(torch.isfinite(z2.grad).all())
    assert float(z2.grad[0, 1].abs().sum()) > float(z2.grad[0, 0].abs().sum())
