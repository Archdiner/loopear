import torch

from loopear.d.esc50_stress import (
    BACKGROUND_CLASSES,
    FOREGROUND_CLASSES,
    N_CLIPS,
    SNRS,
    assign_sources,
    detect_onset,
    mix_transient,
    onset_patch,
    patch_support,
    plan_stress_clips,
    pre_onset_patch,
)
from loopear.d.spec import CROP_SAMPLES, SR


def _examples(n: int) -> list[dict]:
    rows = []
    for name in BACKGROUND_CLASSES + FOREGROUND_CLASSES:
        for index in range(n):
            rows.append(
                {
                    "category": name,
                    "filename": f"{index:02d}-{name}.wav",
                    "wave": torch.zeros(8),
                }
            )
    return rows


def test_stress_plan_is_120_clips_with_exact_onsets():
    first = plan_stress_clips(42)
    second = plan_stress_clips(42)
    assert first == second
    assert len(first) == N_CLIPS == 120
    counts = {float(snr): 0 for snr in SNRS}
    groups: dict[tuple, list] = {}
    for clip in first:
        assert clip["background"] in BACKGROUND_CLASSES
        assert clip["foreground"] in FOREGROUND_CLASSES
        counts[float(clip["snr_db"])] += 1
        onset = int(clip["onset_sample"])
        assert clip["onset_s"] == onset / float(SR)
        assert int(0.45 * SR) <= onset < int(1.20 * SR)
        pre_start, pre_end = patch_support(pre_onset_patch(onset))
        at_start, at_end = patch_support(onset_patch(onset))
        assert pre_end <= onset
        assert at_start <= onset < at_end
        key = (clip["background"], clip["foreground"], clip["pair_slot"])
        groups.setdefault(key, []).append(clip)
    assert counts == {-5.0: 40, 0.0: 40, 5.0: 40}
    assert len(groups) == 40
    for members in groups.values():
        assert [item["snr_db"] for item in members] == [-5.0, 0.0, 5.0]
        assert len({item["onset_sample"] for item in members}) == 1


def test_source_assignment_holds_out_unused_train_clips():
    assigned, held = assign_sources(plan_stress_clips(42), _examples(20))
    assert len(assigned) == 120
    rain = [row["background_file"] for row in assigned if row["background"] == "rain"]
    assert len(set(rain)) == 12
    door = [row["foreground_file"] for row in assigned if row["foreground"] == "door_wood_knock"]
    assert len(set(door)) == 10
    assert len(held["rain"]) == 8
    assert len(held["door_wood_knock"]) == 10
    trio = [row for row in assigned if row["background"] == "rain" and row["foreground"] == "door_wood_knock" and row["pair_slot"] == 0]
    assert len(trio) == 3
    assert len({row["background_file"] for row in trio}) == 1
    assert len({row["foreground_file"] for row in trio}) == 1
    assert len({row["onset_sample"] for row in trio}) == 1


def test_transient_onset_is_the_recorded_sample_at_each_snr():
    click = torch.zeros(200)
    click[40] = 1.0
    assert detect_onset(click) == 40
    generator = torch.Generator().manual_seed(1)
    background = torch.randn(CROP_SAMPLES, generator=generator) * 0.02
    foreground = torch.randn(CROP_SAMPLES + 80, generator=generator) * 0.02
    foreground[:90] = 0
    onset = 8000
    for snr in SNRS:
        mixed, measured = mix_transient(background, foreground, onset, float(snr))
        assert int(mixed.shape[0]) == CROP_SAMPLES
        assert abs(measured - float(snr)) < 1e-4
        assert torch.allclose(mixed[:onset], background[:onset])
        assert not torch.allclose(mixed[onset : onset + 160], background[onset : onset + 160])
