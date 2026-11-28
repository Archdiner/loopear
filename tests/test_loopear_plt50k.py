from loopear.d.fsd import fixed_withhold
from loopear.d.spec import LR, MIN_LR, lr_at, warmup_steps


def test_fifty_thousand_step_schedule_reaches_the_end_of_the_run():
    total = 50000
    warm = warmup_steps(total)
    assert warm == 2500
    assert lr_at(0, total) == LR / float(warm)
    assert lr_at(warm - 1, total) == LR
    assert lr_at(10000, total) < LR
    assert lr_at(25000, total) < lr_at(10000, total)
    assert lr_at(total - 1, total) < lr_at(25000, total)
    assert lr_at(total - 1, total) > MIN_LR
    assert abs(lr_at(total - 1, total) - MIN_LR) < 1e-8


def test_fixed_withhold_is_disjoint_and_covers_the_train_rows():
    rows = [
        {"freesound_id": 1000 + index, "duration_s": 10.0, "split": "train"}
        for index in range(128)
    ]
    train_rows, held_rows = fixed_withhold(rows, 64)
    train_ids = {int(row["freesound_id"]) for row in train_rows}
    held_ids = {int(row["freesound_id"]) for row in held_rows}
    assert held_ids == {1000, 1064}
    assert not train_ids & held_ids
    assert train_ids | held_ids == {int(row["freesound_id"]) for row in rows}
    assert len(train_rows) == 126
    assert sum(float(row["duration_s"]) for row in held_rows) / 3600.0 == 20.0 / 3600.0
