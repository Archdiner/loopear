from loopear.d.mixture import (
    AMI_EVAL_MEETINGS,
    CAP_SECONDS,
    DIPCO_EVAL_SESSIONS,
    LIMITATION,
    STARSS_PROBE_ROOMS,
    ami_meetings_for_remainder,
    assert_mixture_row,
    cap_seconds,
    round_robin_until,
    select_ami_meetings,
    source_hours,
)
from loopear.d.spec import CROP_SECONDS


def test_mixture_cap_excludes_eval_sessions_and_balances_sources():
    meetings = [
        "EN2001a",
        "ES2011a",
        "IB4005",
        "IS1008a",
        "TS3003a",
        "IN1001",
    ]
    chosen = select_ami_meetings(meetings, 3)
    assert chosen == ["EN2001a", "IB4005", "IN1001"]
    assert "ES2011a" not in chosen
    assert cap_seconds(1800.0) == CAP_SECONDS
    assert cap_seconds(12.0) == 12.0
    filled = ami_meetings_for_remainder(1500.0, meetings)
    assert len(filled) == 3
    rows = [
        {"id": "a1", "environment": "kitchen", "duration_s": 20.0},
        {"id": "a2", "environment": "kitchen", "duration_s": 20.0},
        {"id": "b1", "environment": "hall", "duration_s": 30.0},
    ]
    picked = round_robin_until(rows, "environment", 50.0)
    assert [row["id"] for row in picked] == ["b1", "a1"]
    hours = source_hours(
        [
            {"source": "ami", "duration_s": 3600.0},
            {"source": "demand", "duration_s": 1800.0},
        ]
    )
    assert hours["ami"] == 1.0
    assert hours["demand"] == 0.5
    safe = {
        "id": "ami_EN2002a",
        "source": "ami",
        "session": "EN2002a",
        "split": "train",
        "audio_path": "data/raw/loopear_pltn/kept/ami/EN2002a.wav",
        "duration_s": 600.0,
    }
    assert_mixture_row(safe)
    for session in list(AMI_EVAL_MEETINGS) + list(DIPCO_EVAL_SESSIONS) + list(STARSS_PROBE_ROOMS):
        blocked = dict(safe)
        blocked["session"] = session
        if session in DIPCO_EVAL_SESSIONS:
            blocked["source"] = "dipco"
        if session in STARSS_PROBE_ROOMS:
            blocked["source"] = "starss23"
        try:
            assert_mixture_row(blocked)
        except PermissionError:
            refused = True
        else:
            refused = False
        assert refused
    try:
        assert_mixture_row({**safe, "source": "fsd50k", "audio_path": "data/raw/fsd50k/train/1.flac"})
    except PermissionError:
        refused_fsd = True
    else:
        refused_fsd = False
    assert refused_fsd
    assert 5000 * 64 * 2 == 640000
    assert LIMITATION == (
        "Two random crops from a long recording can be unrelated events, so the "
        "recording-level contrastive term is a weaker match for long scenes than it "
        "was for short FSD50K clips."
    )
    assert 640000 * CROP_SECONDS / 3600.0 > 358.0
