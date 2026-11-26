import numpy as np
import torch

from loopear.d.protocol import (
    clip_window_starts,
    crop_energy_profile,
    event_window_starts,
    few_example_classes,
    mean_pearson,
    starss_keep,
    tag_class_names,
)


def test_window_starts_pick_a_center_crop():
    starts, center = clip_window_starts(600, 200, 4)
    assert starts == [0, 200, 400]
    assert starts[center] == 200
    short, short_center = clip_window_starts(80, 200, 4)
    assert short == [0] and short_center == 0


def test_energy_profile_ignores_a_constant_and_keeps_a_ramp():
    flat = torch.ones(200, 64)
    assert torch.allclose(crop_energy_profile(flat), torch.zeros(10), atol=1e-6)
    ramp = torch.arange(200, dtype=torch.float32).view(200, 1).expand(200, 64).contiguous()
    profile = crop_energy_profile(ramp)
    assert torch.all(profile[1:] > profile[:-1])
    edited = ramp.clone()
    edited[:20] += 50
    changed = crop_energy_profile(edited)
    delta = (changed - profile).abs()
    assert int(delta.argmax()) == 0
    assert float(delta[0]) > float(delta[1:].max())


def test_pearson_is_one_for_a_copy_and_skips_flat_rows():
    target = np.array([[0.0, 1.0, 2.0], [3.0, 3.0, 3.0]], dtype=np.float64)
    scored = mean_pearson(target, target)
    assert scored["n"] == 1
    assert scored["mean_r"] == 1.0
    assert scored["n_skipped_flat"] == 1


def test_tag_and_few_example_rules():
    train = [
        {"labels": ["a", "b"]},
        {"labels": ["a"]},
        {"labels": ["a"]},
        {"labels": ["a"]},
        {"labels": ["c"]},
        {"labels": ["b"]},
        {"labels": ["d"]},
        {"labels": ["d"]},
    ]
    val = [
        {"labels": ["a"]},
        {"labels": ["a"]},
        {"labels": ["a"]},
        {"labels": ["a"]},
        {"labels": ["b"]},
        {"labels": ["c", "b"]},
    ]
    assert tag_class_names(train, val, min_total=3, min_each=2) == ["a", "b"]
    few = few_example_classes(train, val, k=4)
    assert [row["label"] for row in few] == ["a"]
    assert few[0]["train_positive"] == [0, 1, 2, 3]
    assert len(few[0]["train_negative"]) == 4
    assert few_example_classes(train, [{"labels": ["a"]}] * 3, k=4) == []


def test_starss_requires_positives_on_both_sides():
    y_train = np.zeros((80, 2), dtype=np.float32)
    y_val = np.zeros((80, 2), dtype=np.float32)
    y_train[:40, 0] = 1
    y_val[:40, 0] = 1
    y_val[:40, 1] = 1
    file_train = np.array([0] * 20 + [1] * 60)
    file_val = np.array([2] * 20 + [3] * 60)
    keep, detail = starss_keep(y_train, y_val, file_train, file_val, min_files=2, min_frames=40)
    assert keep.tolist() == [True, False]
    assert detail[1]["train_positive_files"] == 0
    assert detail[1]["scored"] is False


def test_event_windows_cover_the_class_and_stay_inside_the_file():
    starts = event_window_starts(60.0, [(10.0, 1), (40.0, 4)], max_windows=6)
    assert any(start <= 10.0 < start + 2.015 for start in starts)
    assert all(0.0 <= start <= 60.0 - 2.015 + 1e-6 for start in starts)
    tail = event_window_starts(30.0, [(29.5, 0)], max_windows=4)
    assert max(tail) <= 30.0 - 2.015 + 1e-6
    assert event_window_starts(1.0, [(0.2, 0)]) == [0.0]
