from kassa_pipeline.facts import FactTimeline
from kassa_pipeline.scenarios.stage1_facts import clean_bool, facts_from_stage1

DESK = [100.0, 100.0, 300.0, 200.0]
DOOR = [0.0, 100.0, 100.0, 300.0]  # foot (50, 300) is in the door
DOOR_FOOT_BOX = [20.0, 100.0, 80.0, 300.0]
INSIDE_BOX = [500.0, 100.0, 580.0, 400.0]
COVER_BOX = [90.0, 90.0, 310.0, 210.0]  # covers the whole desk


def person(track, box, role="visitor"):
    return {"track": track, "role": role, "box": box, "score": 0.9, "zone": "elsewhere",
            "hands": {}, "holding": {"left": None, "right": None}}


def obj(cls, score=0.9, on_desk=True, held_by=None):
    return {"track": 1, "cls": cls, "det_cls": cls, "box": [150, 120, 180, 150], "score": score,
            "on_desk": on_desk, "held_by": held_by}


def rec(t, people=(), objects=(), **facts):
    f = {"cashier_zone": "absent", "safe": "unknown", "screen": "unknown"}
    f.update(facts)
    return {"i": int(t * 4), "t": t, "people": list(people), "objects": list(objects), "facts": f}


def stage1(frames, duration=8.0, cashier=None, layouts=None):
    lay = {"start": 0.0, "width": 1000, "height": 600, "desk": DESK, "workstation": None,
           "laptop": [110, 110, 150, 140], "safe": [700, 100, 800, 300], "doors": [DOOR]}
    return {"meta": {"name": "Х-20.mov", "duration": duration, "sample_fps": 4},
            "layouts": layouts or [lay], "cashier_track": cashier, "people": [], "objects": [],
            "frames": frames}


def per_sec(seconds, **kw):
    """Four records per second for the listed seconds; kw values may be callables of the second."""
    out = []
    for s in seconds:
        for k in range(4):
            out.append(rec(s + k * 0.25, **{a: (v(s) if callable(v) else v) for a, v in kw.items()}))
    return out


def test_clean_bool_closes_gap_and_drops_short_run():
    assert clean_bool([True, False, True], 2, 2) == [True, True, True]
    assert clean_bool([False, True, False, False, False], 2, 2) == [False] * 5
    assert clean_bool([True, True, False, False, False, True, True], 2, 2) == [True, True, False, False, False, True, True]


def test_valid_timeline_basics():
    ft = facts_from_stage1(stage1(per_sec(range(8))), "Х-20.mov")
    again = FactTimeline.from_dict(ft.to_dict())
    assert again.clip == "X-20" and ft.duration == 8.0 and ft.status == "draft"
    assert ft.observable == ["room_entry", "room_exit", "safe_access_start", "safe_access_end"]
    assert ft.zones["desk"] == DESK and ft.zones["door"] == DOOR
    assert ft.states["laptop_lid"] == ["unknown"] * 8
    assert ft.states["desk_docs"] == ["unknown"] * 8
    assert all(o == "cv" for o in ft.state_origin["screen"])


def test_desk_cash_yes_no_unknown():
    cash = [obj("cash")]
    frames = per_sec([0, 1, 2], objects=cash) + per_sec([3, 4])  # 5..7 unsampled
    ft = facts_from_stage1(stage1(frames), "x")
    assert ft.states["desk_cash"] == ["yes", "yes", "yes", "no", "no", "unknown", "unknown", "unknown"]
    assert ft.states["desk_bag"][:5] == ["no"] * 5


def test_desk_floors_held_and_off_desk_are_not_on_desk():
    objs = [obj("cash", score=0.05), obj("cash_bag", held_by=[1, "left"]), obj("phone", score=0.2),
            obj("personal", on_desk=False)]
    ft = facts_from_stage1(stage1(per_sec(range(3), objects=objs), duration=3.0), "x")
    assert ft.states["desk_cash"] == ["no"] * 3
    assert ft.states["desk_bag"] == ["no"] * 3
    assert ft.states["desk_personal"] == ["no"] * 3


def test_desk_personal_classes_and_bag_floor():
    objs = [obj("food_drink", score=0.3), obj("cash_bag", score=0.09)]
    ft = facts_from_stage1(stage1(per_sec(range(3), objects=objs), duration=3.0), "x")
    assert ft.states["desk_personal"] == ["yes"] * 3
    assert ft.states["desk_bag"] == ["yes"] * 3


def test_desk_any_record_in_second_counts_and_hysteresis():
    frames = []
    for s in range(8):
        for k in range(4):
            on = s in (0, 1, 3, 4) and k == 2  # one hit per second, gap at second 2
            frames.append(rec(s + k * 0.25, objects=[obj("cash")] if on else []))
    ft = facts_from_stage1(stage1(frames), "x")
    assert ft.states["desk_cash"] == ["yes"] * 5 + ["no"] * 3  # gap of 1 closed


def test_desk_short_blip_is_dropped():
    frames = per_sec([0, 1]) + per_sec([2], objects=[obj("cash")]) + per_sec([3, 4])
    ft = facts_from_stage1(stage1(frames, duration=5.0), "x")
    assert ft.states["desk_cash"] == ["no"] * 5


def test_desk_occluded_second_is_unknown_not_no():
    covering = per_sec([1], people=[person(2, COVER_BOX)])
    frames = per_sec([0]) + covering + per_sec([2])
    ft = facts_from_stage1(stage1(frames, duration=3.0), "x")
    assert ft.states["desk_cash"] == ["no", "unknown", "no"]
    partial = per_sec([0], people=[person(2, [100, 100, 300, 140])])  # covers 0.4 of desk
    ft = facts_from_stage1(stage1(partial, duration=1.0), "x")
    assert ft.states["desk_cash"] == ["no"]


def test_no_desk_box_means_unknown_for_no():
    s1 = stage1(per_sec(range(2)), duration=2.0)
    s1["layouts"][0]["desk"] = None
    ft = facts_from_stage1(s1, "x")
    assert ft.states["desk_cash"] == ["unknown"] * 2
    assert ft.zones["desk"] == []


def test_layout_switch_uses_layout_at_second_middle():
    far = {"start": 4.0, "desk": [600.0, 300.0, 700.0, 400.0], "doors": []}
    s1 = stage1(per_sec(range(6), people=[person(2, COVER_BOX)]), duration=6.0)
    s1["layouts"].append(far)
    ft = facts_from_stage1(s1, "x")
    assert ft.states["desk_cash"] == ["unknown"] * 4 + ["no"] * 2


def test_screen_and_safe_majority():
    frames = []
    for s, (scr, safe) in enumerate([("work_screen", "open"), ("locked", "closed"), ("off", "unknown"), ("work_screen", "open")]):
        frames += [rec(s, screen=scr, safe=safe), rec(s + 0.25, screen=scr, safe=safe),
                   rec(s + 0.5, screen="off", safe="closed")]
    frames += [rec(4.0, screen="work_screen", safe="open"), rec(4.25, screen="off", safe="closed")]  # tie
    ft = facts_from_stage1(stage1(frames, duration=6.0), "x")
    assert ft.states["screen"] == ["work", "locked", "off", "work", "unknown", "unknown"]
    assert ft.states["safe_door"] == ["open", "closed", "unknown", "open", "unknown", "unknown"]


def walk(track, spec, role="visitor"):
    """spec: list of (t, 'd'|'i') -> list of (t, person)."""
    return [(t, person(track, DOOR_FOOT_BOX if s == "d" else INSIDE_BOX, role)) for t, s in spec]


def run(a, b, s, step=0.25):
    n = int(round((b - a) / step))
    return [(a + k * step, s) for k in range(n + 1)]


def frames_from(*tracks, duration=8.0):
    by_t = {}
    for tr in tracks:
        for t, p in tr:
            by_t.setdefault(t, []).append(p)
    return [rec(t, people=by_t.get(t, [])) for t in sorted(by_t)]


def test_entry_exit_and_numbering():
    cashier = walk(5, run(0, 3, "i") + run(3.25, 4.25, "d"), role="cashier")
    late = walk(7, run(5, 6, "d") + run(6.25, 7.5, "i"))
    early = walk(6, run(0.5, 1.5, "d") + run(1.75, 3, "i"))
    ft = facts_from_stage1(stage1(frames_from(cashier, late, early), cashier=5), "x")
    got = [(e.type, e.who, e.t, e.origin) for e in ft.events if e.type.startswith("room_")]
    assert got == [("room_entry", "client-1", 1.75, "cv"), ("room_exit", "cashier", 4.25, "cv"),
                   ("room_entry", "client-2", 6.25, "cv")]


def test_crossing_thresholds():
    first_inside = walk(1, run(0, 3, "i"))
    jitter = walk(2, run(0, 3, "i") + [(3.25, "d")] + run(3.5, 5, "i"))
    one_obs_door = walk(3, [(0, "d")] + run(0.25, 2, "i"))
    ft = facts_from_stage1(stage1(frames_from(first_inside, jitter, one_obs_door)), "x")
    assert [e for e in ft.events if e.type.startswith("room_")] == []


def test_people_inside_counts_tracks_with_two_inside_obs():
    a = walk(1, run(0, 3, "i"))
    b = walk(2, [(1.0, "i"), (2.25, "i"), (2.5, "i")])  # one obs in sec 1, two in sec 2
    ft = facts_from_stage1(stage1(frames_from(a, b) + per_sec([4, 5, 6, 7]), duration=8.0), "x")
    assert ft.states["people_inside"][:4] == [1, 1, 2, 0]
    assert ft.states["people_inside"][4:] == [0, 0, 0, 0]


def test_people_inside_unsampled_is_none():
    ft = facts_from_stage1(stage1(per_sec([0]), duration=3.0), "x")
    assert ft.states["people_inside"] == [0, 0, None]


def safe_frames(door, zones, extra_sec=None):
    frames = []
    for s, (d, z) in enumerate(zip(door, zones)):
        frames += [rec(s + k * 0.25, safe=d, cashier_zone=z) for k in range(4)]
    return frames


def test_safe_access_start_and_end_on_close():
    door = ["closed", "open", "open", "open", "closed", "closed"]
    zones = ["cashier_seat"] * 6
    ft = facts_from_stage1(stage1(safe_frames(door, zones), duration=6.0), "x")
    ev = [(e.type, e.t, e.who, e.note) for e in ft.events if e.type.startswith("safe")]
    assert ev == [("safe_access_start", 1.0, "cashier", "draft"), ("safe_access_end", 4.0, "cashier", "draft")]


def test_safe_access_ends_when_cashier_leaves_safe():
    door = ["open"] * 6
    zones = ["near_safe", "near_safe", "elsewhere", "elsewhere", "elsewhere", "elsewhere"]
    ft = facts_from_stage1(stage1(safe_frames(door, zones), duration=6.0), "x")
    ev = [(e.type, e.t) for e in ft.events if e.type.startswith("safe")]
    assert ev == [("safe_access_start", 0.0), ("safe_access_end", 2.0)]


def test_safe_access_seated_fallback():
    door = ["open"] * 8
    zones = ["cashier_seat"] * 8
    ft = facts_from_stage1(stage1(safe_frames(door, zones), duration=8.0), "x")
    ev = [(e.type, e.t, e.note) for e in ft.events if e.type.startswith("safe")]
    assert ev == [("safe_access_start", 0.0, "draft"), ("safe_access_end", 4.0, "draft-fallback")]


def test_events_sorted_by_time():
    cashier = walk(5, run(0, 1, "i") + run(1.25, 2, "d"), role="cashier")
    frames = frames_from(cashier)
    for r in frames:
        r["facts"]["safe"] = "open"
    ft = facts_from_stage1(stage1(frames, duration=4.0, cashier=5), "x")
    ts = [e.t for e in ft.events]
    assert ts == sorted(ts) and {e.type for e in ft.events} >= {"room_exit", "safe_access_start"}


def test_tracking_dropout_does_not_end_safe_access():
    frames = [rec(k * 0.25, safe="open", cashier_zone="near_safe") for k in range(4)]
    for s in (1, 2, 3):
        frames += [rec(s + k * 0.25, safe="open", cashier_zone="near_safe" if k < 2 else "absent") for k in range(4)]
    ft = facts_from_stage1(stage1(frames, duration=4.0), "x")
    assert [(e.type, e.t) for e in ft.events if e.type.startswith("safe")] == [("safe_access_start", 0.0)]


def test_people_inside_trailing_partial_second_borrows_previous():
    a = walk(1, run(0, 7.75, "i"))
    ft = facts_from_stage1(stage1(frames_from(a), duration=8.008), "x")
    assert len(ft.states["people_inside"]) == 9
    assert ft.states["people_inside"] == [1] * 8 + [1]


def test_people_inside_long_gap_stays_none_and_leading_borrows_next():
    frames = per_sec([0]) + per_sec([4])
    ft = facts_from_stage1(stage1(frames, duration=6.0), "x")
    assert ft.states["people_inside"] == [0, 0, None, 0, 0, 0]
    ft = facts_from_stage1(stage1(per_sec([1, 2]), duration=3.0), "x")
    assert ft.states["people_inside"] == [0, 0, 0]


def test_person_at_desk_is_inside_even_with_foot_in_door_box():
    # a seated client's feet are hidden by the desk, so the box bottom lands in the tall glass-door box
    seated = dict(person(1, DOOR_FOOT_BOX), zone="at_desk")
    cashier = dict(person(2, DOOR_FOOT_BOX, role="cashier"), zone="cashier_seat")
    frames = [rec(s + k * 0.25, people=[seated, cashier]) for s in range(3) for k in range(4)]
    ft = facts_from_stage1(stage1(frames, duration=3.0, cashier=2), "x")
    assert ft.states["people_inside"] == [2, 2, 2]
