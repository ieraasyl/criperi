from kassa_pipeline.config import ScanConfig
from kassa_pipeline.vlm_scan import Stage1, _merge, _runs, door_rule, fixed_windows, laptop_rule, Finding


def _frame(t, zone, screen="unknown"):
    return {"t": t, "people": [], "objects": [], "facts": {"cashier_zone": zone, "cashier_at_workplace": zone == "cashier_seat", "screen": screen}}


def _stage1(frames):
    return Stage1({"frames": frames, "layouts": [{"start": 0.0}], "people": []})


def test_windows_cover_the_video_with_overlap():
    w = fixed_windows(20.0, ScanConfig())
    assert w[0][0] == 0.0 and w[-1][1] >= 19.9
    assert all(b[0] < a[1] for a, b in zip(w, w[1:]))  # consecutive windows overlap


def test_state_needs_two_agreeing_windows():
    yes = {"state": {"cash_unattended_on_desk": ("yes", 5.0)}, "t0": 4.0, "t1": 12.0, "events": []}
    no = {"state": {"cash_unattended_on_desk": ("no", None)}, "t0": 8.0, "t1": 16.0, "events": []}
    assert _runs([yes, no], "cash_unattended_on_desk", 2) == []
    runs = _runs([yes, dict(yes, t0=8.0, t1=16.0)], "cash_unattended_on_desk", 2)
    assert len(runs) == 1 and runs[0][0] == 5.0


def test_laptop_rule_fires_only_when_the_cashier_leaves_a_work_screen():
    seated = [_frame(t / 4, "cashier_seat", "work_screen") for t in range(0, 20)]
    away_work = [_frame(5 + t / 4, "absent", "work_screen") for t in range(0, 20)]
    away_locked = [_frame(5 + t / 4, "absent", "locked") for t in range(0, 20)]
    sc = ScanConfig()
    assert [f.t for f in laptop_rule(_stage1(seated + away_work), sc)] == [5.0]
    assert laptop_rule(_stage1(seated + away_locked), sc) == []
    assert laptop_rule(_stage1(away_work), sc) == []  # absent from the start: she never left


def test_door_must_stay_open_long_enough():
    sc = ScanConfig()
    brief = [{"t": float(t), "door": "open" if 3 <= t <= 5 else "closed", "person": "yes"} for t in range(12)]
    long = [{"t": float(t), "door": "open" if t >= 2 else "closed", "person": "no"} for t in range(12)]
    assert door_rule(brief, sc) == []
    f = door_rule(long, sc)
    assert len(f) == 1 and f[0].t == 2.0 and f[0].kind == "door_open"


def test_close_findings_of_one_rule_merge():
    m = _merge([Finding("R-CASH-LOCKED", 9.0, 0.7, "a"), Finding("R-CASH-LOCKED", 12.0, 0.6, "b"),
                Finding("R-IS-LOCK", 10.0, 0.65, "c")], gap=10.0)
    assert [f.rule for f in m] == ["R-CASH-LOCKED", "R-IS-LOCK"] and m[0].confidence > 0.7
