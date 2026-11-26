import torch

from loopear.encoder import SR
from loopear.h8data import assert_not_sealed, cross_group_pair_count
from loopear.tiny import (
    PARAM_MAX,
    PARAM_MIN,
    TinyEncoder,
    assert_param_gate,
    averaged_constituent_loss,
    causal_mismatch,
    constituent_align_loss,
    distill_pair_loss,
    inference_parameter_count,
    mix_waveforms,
    mixture_augment_loss,
    wav_to_frame_lengths,
)


def test_tiny_param_gate_and_causal_inference():
    torch.manual_seed(0)
    model = TinyEncoder()
    report = assert_param_gate(model)
    assert PARAM_MIN <= report["inference_params"] <= PARAM_MAX
    assert report["ok"]
    assert inference_parameter_count(model) == report["inference_params"]
    wav = torch.randn(2, 2 * SR)
    frames = model.encode(wav)
    lengths = wav_to_frame_lengths(torch.tensor([wav.shape[-1], wav.shape[-1]]))
    assert frames.shape[0] == 2
    assert frames.shape[-1] == model.out_dim
    assert int(frames.shape[1]) == int(lengths[0])
    mismatch = causal_mismatch(model, wav[:1])
    assert mismatch["ok"], mismatch


def test_four_losses_train_every_parameter():
    torch.manual_seed(1)
    model = TinyEncoder()
    model.train()
    n = 2 * SR
    wav_a = torch.randn(2, n)
    wav_b = torch.randn(2, n)
    mixed = mix_waveforms(wav_a, wav_b, torch.tensor([-12.0, 6.0]))
    lengths = wav_to_frame_lengths(torch.tensor([n, n]))
    probe = model.encode(wav_a)
    teacher = torch.randn(2, probe.shape[1], model.teacher_dim)
    teacher_b = torch.randn_like(teacher)
    teacher_m = torch.randn_like(teacher)
    builders = {
        "distill": lambda fa, fb, fm: distill_pair_loss(
            model, fa, fb, teacher, teacher_b, lengths, lengths
        ),
        "mix": lambda fa, fb, fm: mixture_augment_loss(model, fm, teacher_m, lengths),
        "avg": lambda fa, fb, fm: averaged_constituent_loss(
            model, fm, teacher, teacher_b, lengths
        ),
        "align": lambda fa, fb, fm: constituent_align_loss(
            model, fm, fa, fb, teacher, teacher_b, lengths
        ),
    }
    for name, build in builders.items():
        model.zero_grad(set_to_none=True)
        frames_a = model.encode(wav_a)
        frames_b = model.encode(wav_b)
        frames_m = model.encode(mixed)
        loss = build(frames_a, frames_b, frames_m)
        assert torch.isfinite(loss), name
        loss.backward()
        missing = []
        for pname, param in model.named_parameters():
            if param.grad is None or not torch.isfinite(param.grad).all():
                missing.append(pname)
            elif float(param.grad.abs().sum()) == 0.0 and not pname.startswith("teacher_proj"):
                missing.append(pname)
        assert not missing, (name, missing[:8])


def test_sealed_items_and_pair_count_stay_closed():
    try:
        assert_not_sealed({"split": "test", "audio_path": "data/raw/realdesed/train/a.wav"})
        raised = False
    except PermissionError:
        raised = True
    assert raised
    try:
        assert_not_sealed(
            {"split": "train", "audio_path": "data/raw/realdesed/test/a.wav"}
        )
        raised_path = False
    except PermissionError:
        raised_path = True
    assert raised_path
    count = cross_group_pair_count(
        [
            {"corpus": "a", "group_id": "1"},
            {"corpus": "a", "group_id": "1"},
            {"corpus": "b", "group_id": "2"},
        ]
    )
    assert count == 1
