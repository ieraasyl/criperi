from kassa_pipeline.config import ScanConfig
from kassa_pipeline.vlm_scan import Finding, Stage1, _merge, _runs, _session_rules, door_rule, fixed_windows, laptop_rule


def _frame(t, zone, screen="unknown", people=1):
    return {"t": t, "people": [], "objects": [],
            "facts": {"cashier_zone": zone, "cashier_at_workplace": zone == "cashier_seat", "screen": screen, "people_in_room": people}}


def _stage1(frames):
    return Stage1({"frames": frames, "layouts": [{"start": 0.0}], "people": []})


def test_windows_cover_the_video_with_overlap():
    w = fixed_windows(20.0, ScanConfig())
    assert w[0][0] == 0.0 and w[-1][1] >= 19.9
    assert all(b[0] < a[1] for a, b in zip(w, w[1:]))


def test_state_needs_two_agreeing_windows():
    yes = {"state": {"cash_on_open_desk": ("yes", 5.0)}, "t0": 4.0, "t1": 12.0, "events": []}
    no = {"state": {"cash_on_open_desk": ("no", None)}, "t0": 8.0, "t1": 16.0, "events": []}
    assert _runs([yes, no], "cash_on_open_desk", 2) == []
    runs = _runs([yes, dict(yes, t0=8.0, t1=16.0)], "cash_on_open_desk", 2)
    assert len(runs) == 1 and runs[0][0] == 5.0


def test_v14_only_when_the_cashier_leaves_the_room_with_a_work_screen():
    seated = [_frame(t / 4, "cashier_seat", "work_screen") for t in range(20)]
    out_work = [_frame(5 + t / 4, "absent", "work_screen") for t in range(20)]
    out_locked = [_frame(5 + t / 4, "absent", "locked") for t in range(20)]
    at_safe = [_frame(5 + t / 4, "near_safe", "work_screen") for t in range(20)]
    sc = ScanConfig()
    assert [(f.code, f.t) for f in laptop_rule(_stage1(seated + out_work), sc)] == [("V14", 5.0)]
    assert laptop_rule(_stage1(seated + out_locked), sc) == []
    assert laptop_rule(_stage1(seated + at_safe), sc) == []  # the safe is inside the room: not leaving
    assert laptop_rule(_stage1(out_work), sc) == []


def test_v18_door_must_stay_open_long_enough():
    sc = ScanConfig()
    brief = [{"t": float(t), "door": "open" if 3 <= t <= 5 else "closed", "person": "yes"} for t in range(12)]
    long = [{"t": float(t), "door": "open" if t >= 2 else "closed", "person": "no"} for t in range(12)]
    assert door_rule(brief, sc) == []
    f = door_rule(long, sc)
    assert len(f) == 1 and f[0].t == 2.0 and f[0].code == "V18" and f[0].rule == "R-ENTRY-ISOLATION"


def test_taxonomy_priorities_drop_the_weaker_duplicate():
    m = _merge([Finding("V01", 9.0, 0.75, "a"), Finding("V11", 12.0, 0.6, "b"), Finding("V17", 30.0, 0.55, "c"),
                Finding("V15", 31.0, 0.6, "d"), Finding("V14", 10.0, 0.65, "e")], gap=10.0)
    assert [f.code for f in m] == ["V01", "V14", "V17"]


def test_not_visible_is_never_a_violation():
    base = {"t0": 0.0, "t1": 20.0, "answer": {"start_fully_visible": "no", "money_taken": "yes", "money_t": 5,
                                               "document_shown": "no", "open_cash": "yes", "counted_on_big_counter": "not_visible",
                                               "money_put_away": "yes", "operation_finished": "yes"}}
    assert _session_rules([base], [], _stage1([]), ScanConfig()) == []
    seen = dict(base, answer={**base["answer"], "start_fully_visible": "yes"})
    assert [f.code for f in _session_rules([seen], [], _stage1([]), ScanConfig())] == ["V04"]
