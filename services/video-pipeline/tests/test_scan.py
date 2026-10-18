from kassa_pipeline.config import ScanConfig
from kassa_pipeline.vlm_scan import Finding, Stage1, _merge, _runs, _session_rules, door_rule, fixed_windows, laptop_rule

SESS = ScanConfig(sessions_from_vlm_events=True)  # event sessions are off by default


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


def _doors(first, last, total=30, person="yes"):
    return [{"t": float(t), "door": "open" if first <= t <= last else "closed", "person": person} for t in range(total)]


def _win(events, t0=0.0, t1=8.0):
    return {"kind": "window", "t0": t0, "t1": t1, "events": events, "state": {}}


def _ev(what, t):
    return {"t": t, "who": "#1", "what": what, "note": ""}


def test_v18_door_run_explained_by_a_client_entry_is_ignored():
    # real X-3: door open 1-7 s, the VLM logged client_entered_room at 1.5, 4.0 and 8.0
    doors = _doors(1, 7)
    assert door_rule(doors, ScanConfig(), transits=[1.5, 4.0, 8.0]) == []
    # real X-12: door open 19-23 s, client_left_room at 19.0
    assert door_rule(_doors(19, 23), ScanConfig(), transits=[19.0]) == []


def test_v18_reports_the_unexplained_rest_of_the_open_run():
    f = door_rule(_doors(1, 12), ScanConfig(), transits=[2.0])
    assert len(f) == 1 and f[0].t == 5.0 and f[0].evidence == ["door open 5.0-12.0s"]


def test_v18_without_a_transit_is_unchanged():
    f = door_rule(_doors(2, 11), ScanConfig(), transits=[20.0])
    assert [x.t for x in f] == [2.0]


def test_apply_rules_takes_transits_from_vlm_events_and_the_flag_turns_it_off():
    from kassa_pipeline.vlm_scan import apply_rules
    wins = [_win([_ev("client_entered_room", 1.5)])]
    s1, doors = _s1_with_visitor(1.5, 12.0, {t: "at_desk" for t in range(4, 12)}), _doors(1, 7)
    assert [f for f in apply_rules(wins, [], doors, s1, ScanConfig()) if f.code == "V18"] == []
    off = ScanConfig(door_ignore_transits=False)
    assert [f.code for f in apply_rules(wins, [], doors, s1, off) if f.code == "V18"] == ["V18"]


def test_vlm_entry_not_covered_by_a_session_gets_one_until_the_next_exit():
    from kassa_pipeline.vlm_scan import add_event_sessions
    wins = [_win([_ev("client_entered_room", 10.0), _ev("client_left_room", 20.0)])]
    out = add_event_sessions([(30.0, 35.0, [4])], wins, 60.0, SESS)
    assert out == [(8.5, 21.5, []), (30.0, 35.0, [4])]


def test_vlm_entry_without_exit_runs_to_the_cap_and_pad_clips_at_zero():
    from kassa_pipeline.vlm_scan import add_event_sessions
    out = add_event_sessions([], [_win([_ev("client_entered_room", 1.0)])], 40.0, SESS)
    assert out == [(0.0, 31.0, [])]  # capped at event_session_max_sec
    out = add_event_sessions([], [_win([_ev("client_entered_room", 1.0)])], 20.0, SESS)
    assert out == [(0.0, 20.0, [])]  # no exit, short clip: to the clip end


def test_vlm_entry_already_inside_a_session_adds_nothing():
    from kassa_pipeline.vlm_scan import add_event_sessions
    base = [(0.0, 20.0, [1, 2])]
    wins = [_win([_ev("client_entered_room", 1.5)]), _win([_ev("client_entered_room", 4.0)])]
    assert add_event_sessions(base, wins, 20.0, SESS) == base


def test_vlm_sessions_merge_and_the_flag_turns_them_off():
    from kassa_pipeline.vlm_scan import add_event_sessions
    wins = [_win([_ev("client_entered_room", 10.0), _ev("client_entered_room", 14.0), _ev("client_left_room", 30.0)])]
    assert add_event_sessions([], wins, 60.0, SESS) == [(8.5, 31.5, [])]
    assert add_event_sessions([], wins, 60.0, ScanConfig(sessions_from_vlm_events=False)) == []


def test_session_prompt_tolerates_no_visitor_ids():
    from kassa_pipeline.vlm_scan import visitors_text
    assert visitors_text([]) == "the client(s) seen in these frames"
    assert visitors_text([1, 2]) == "#1, #2"


def test_scan_asks_for_a_session_after_the_windows_reported_an_untracked_entry(monkeypatch):
    import json
    import types
    import kassa_pipeline.vlm_scan as vs
    monkeypatch.setattr(vs, "_build", lambda *a, **k: ([], []))
    calls = []

    def ask_many(items, **k):
        calls.append([p for _, p in items])
        if len(calls) == 1:
            return [json.dumps({"events": [{"t": 5.0, "who": "#1", "what": "client_entered_room", "note": ""}]})] * len(items)
        return ["{}"] * len(items)

    s1 = {"frames": [], "layouts": [{"start": 0.0}], "people": []}
    from kassa_pipeline.config import PipelineConfig
    res = vs.scan(types.SimpleNamespace(duration=20.0), s1, types.SimpleNamespace(ask_many=ask_many), PipelineConfig(scan=SESS))
    assert [(s["kind"], s["visitors"]) for s in res["sessions"]] == [("session", [])]
    assert "the client(s) seen in these frames" in calls[1][0]


def test_money_handover_outside_every_session_opens_a_session_on_real_x13_windows():
    import json
    from pathlib import Path
    from kassa_pipeline.vlm_scan import add_event_sessions
    d = json.loads(Path("/home/dauka/Desktop/code/research/baseline/debug/Х-13.json").read_text("utf8"))
    out = add_event_sessions([], d["windows"], d["duration"], SESS)
    assert out == [(11.0, 63.0, [])]  # money at 21, 24, 35 -> [11, 41]; 43 -> [33, 63]; merged
    assert add_event_sessions([], d["windows"], d["duration"], ScanConfig(sessions_from_vlm_events=False)) == []


def test_money_handover_inside_a_stage1_session_adds_nothing():
    from kassa_pipeline.vlm_scan import add_event_sessions
    base = [(5.0, 30.0, [1])]
    assert add_event_sessions(base, [_win([_ev("money_taken_by_cashier", 21.0)])], 65.0, SESS) == base


def test_bad_vlm_event_times_are_dropped_not_clamped():
    from kassa_pipeline.vlm_scan import add_event_sessions, apply_rules
    bad = ["abc", float("nan"), float("inf"), float("-inf"), -5, 1e9, 31.5, None, True]
    wins = [_win([_ev(w, t) for w in ("client_entered_room", "client_left_room", "money_taken_by_cashier") for t in bad])]
    assert add_event_sessions([], wins, 30.0, SESS) == []
    apply_rules(wins, [], _doors(1, 7), _stage1([_frame(0.0, "cashier_seat")]), ScanConfig())
    near = [_win([_ev("client_entered_room", -0.5), _ev("client_left_room", 30.8)])]
    out = add_event_sessions([], near, 30.0, SESS)
    assert out == [(0.0, 30.0, [])]


def test_repeated_entry_events_do_not_chain_cuts_over_a_long_open_door():
    f = door_rule(_doors(0, 20), ScanConfig(), transits=_transits_of([1.5, 4.0, 8.0, 12.0]))
    assert [x.t for x in f] == [4.5]
    single = door_rule(_doors(1, 12), ScanConfig(), transits=_transits_of([2.0]))
    assert [x.t for x in single] == [5.0]


def _transits_of(ts):
    from kassa_pipeline.vlm_scan import _transits
    return _transits([_win([_ev("client_entered_room", t) for t in ts], 0.0, 30.0)])


def test_entry_without_exit_is_capped_and_event_sessions_stay_apart_from_stage1_ones():
    from kassa_pipeline.vlm_scan import add_event_sessions
    out = add_event_sessions([], [_win([_ev("client_entered_room", 10.0)])], 100.0, SESS)
    assert out == [(8.5, 40.0, [])]
    base = [(0.0, 9.0, [1])]
    out = add_event_sessions(base, [_win([_ev("client_entered_room", 12.0)])], 100.0, SESS)
    assert out == [(0.0, 9.0, [1]), (10.5, 42.0, [])] and base == [(0.0, 9.0, [1])]


def test_event_sessions_are_off_by_default():
    from kassa_pipeline.vlm_scan import add_event_sessions
    assert ScanConfig().sessions_from_vlm_events is False
    assert add_event_sessions([], [_win([_ev("client_entered_room", 5.0)])], 20.0, ScanConfig()) == []


def _s1_with_visitor(t0, t1, zones):
    """Stage 1 with one visitor track; `zones` maps a second to the zone it is in (default doorway)."""
    frames = []
    for i in range(int(t0 * 2), int(t1 * 2) + 1):
        t = i / 2
        f = _frame(t, "cashier_seat")
        f["people"] = [{"track": 7, "role": "visitor", "zone": zones.get(int(t), "doorway"), "box": [0, 0, 1, 1]}]
        frames.append(f)
    return _stage1(frames)


def test_transits_only_cut_when_a_client_track_reaching_the_desk_confirms_them():
    from kassa_pipeline.vlm_scan import apply_rules
    x3 = [_win([_ev("client_entered_room", 1.5), _ev("client_entered_room", 4.0)])]
    client = _s1_with_visitor(1.5, 12.0, {t: "at_desk" for t in range(4, 12)})
    assert [f for f in apply_rules(x3, [], _doors(1, 7), client, ScanConfig()) if f.code == "V18"] == []
    # X-33: the VLM calls a doorway outsider a client; his track never reaches the desk -> no cut
    outsider = _s1_with_visitor(1.5, 12.0, {})
    assert [f.t for f in apply_rules(x3, [], _doors(1, 7), outsider, ScanConfig()) if f.code == "V18"] == [1.0]
    # X-12: no visitor track near the event at all -> trust the VLM and cut
    assert [f for f in apply_rules(x3, [], _doors(1, 7), _stage1([_frame(0.0, "cashier_seat")]), ScanConfig()) if f.code == "V18"] == []
    # an outsider track elsewhere in the clip does not block a cut far from it
    elsewhere = _s1_with_visitor(40.0, 45.0, {})
    assert [f for f in apply_rules(x3, [], _doors(1, 7), elsewhere, ScanConfig()) if f.code == "V18"] == []


def test_x33_real_events_with_an_outsider_track_keep_v18():
    import json
    from pathlib import Path
    from kassa_pipeline.vlm_scan import apply_rules
    d = json.loads(Path("/home/dauka/Desktop/code/research/baseline/hybrid/Х-33.json").read_text("utf8"))
    s1 = _s1_with_visitor(2.0, 14.0, {})
    out = [f for f in apply_rules(d["windows"], [], d["door"]["frames"], s1, ScanConfig()) if f.code == "V18"]
    assert [f.t for f in out] == [4.0]


def _s1_tracks(tracks):
    """Stage 1 with visitor tracks given as (track, t0, t1, reaches_desk)."""
    ts = sorted({round(t * 2) / 2 for _, a, b, _ in tracks for t in (a, b)} | {0.0})
    frames = []
    for t in ts:
        f = _frame(t, "cashier_seat")
        f["people"] = [{"track": k, "role": "visitor", "zone": "at_desk" if desk else "doorway", "box": [0, 0, 1, 1]}
                       for k, a, b, desk in tracks if a <= t <= b]
        frames.append(f)
    return _stage1(frames)


def _confirmed(times, tracks, what="client_entered_room"):
    from kassa_pipeline.vlm_scan import _confirmed_transits
    return _confirmed_transits([_win([_ev(what, t) for t in times], 0.0, 40.0)], _s1_tracks(tracks), ScanConfig())


def test_outsider_event_does_not_swallow_a_later_client_transit():
    from kassa_pipeline.vlm_scan import apply_rules
    tracks = [(1, 0.0, 0.9, False), (2, 5.0, 12.0, True)]
    assert _confirmed([1.5, 5.0], tracks) == [5.0]
    wins = [_win([_ev("client_entered_room", 1.5), _ev("client_entered_room", 5.0)])]
    out = [f.t for f in apply_rules(wins, [], _doors(1, 12), _s1_tracks(tracks), ScanConfig()) if f.code == "V18"]
    assert out == [8.0]  # the 5.0 client transit cut [2, 8]; the rest of the run stays open


def test_repeated_client_logs_still_collapse_after_confirmation():
    tracks = [(2, 4.0, 12.0, True)]
    assert _confirmed([4.0, 6.0, 8.0], tracks) == [4.0]


def test_far_client_track_without_outsider_still_cuts_and_outsider_near_blocks():
    from kassa_pipeline.vlm_scan import apply_rules
    wins = [_win([_ev("client_entered_room", 4.0)])]
    far_client = _s1_tracks([(2, 20.0, 30.0, True)])
    assert [f for f in apply_rules(wins, [], _doors(1, 7), far_client, ScanConfig()) if f.code == "V18"] == []
    near_outsider = _s1_tracks([(2, 20.0, 30.0, True), (3, 3.0, 5.0, False)])
    assert [f.t for f in apply_rules(wins, [], _doors(1, 7), near_outsider, ScanConfig()) if f.code == "V18"] == [1.0]


def test_confirmation_boundary_is_door_transit_sec():
    client_only = [(1, 10.0, 12.0, True)]
    outsider_too = [(1, 10.0, 12.0, True), (2, 0.0, 40.0, False)]
    assert _confirmed([7.1], client_only) == [7.1] and _confirmed([6.9], client_only) == [6.9]
    assert _confirmed([7.1], outsider_too) == [7.1]  # client end within 3 s: rule 1 wins over the outsider
    assert _confirmed([6.9], outsider_too) == []  # 3.1 s from the client, outsider present: no cut
    outsider = [(2, 10.0, 12.0, False)]
    assert _confirmed([7.1], outsider) == []  # outsider span 2.9 s away
    assert _confirmed([6.9], outsider) == [6.9]  # 3.1 s away: no track near, trust the VLM


# ---- zoomed safe-door check (V12)

def _safe_frames(first, last, total=20, v="yes"):
    return [{"t": float(t), "open": v if first <= t <= last else "no"} for t in range(total)]


def _safe_s1(away_from=10):
    return _stage1([_frame(float(t), "near_safe" if t < away_from else "cashier_seat") for t in range(30)])


def _v12(wins, s1, safe, sc=None):
    from kassa_pipeline.vlm_scan import apply_rules
    return [f for f in apply_rules(wins, [], [], s1, sc or ScanConfig(), safe=safe) if f.code == "V12"]


def _safe_win(t0, t1, ans):
    from kassa_pipeline.vlm_scan import STATES
    st = {s: ("no", None) for s in STATES}
    st["safe_door_open"] = (ans, None)
    return {"kind": "window", "t0": t0, "t1": t1, "events": [], "state": st}


def test_safe_zoom_crops_the_safe_box_and_needs_one():
    import types
    import numpy as np
    from kassa_pipeline.vlm_scan import safe_zoom
    fr = types.SimpleNamespace(t=3.0, rgb=np.zeros((400, 600, 3), np.uint8))
    assert safe_zoom(fr, {}, 448) is None
    img = safe_zoom(fr, {"safe": [200, 100, 300, 250]}, 448)
    assert img.shape[1] == 448 and abs(img.shape[0] - 672) <= 1  # box 100x150 padded by 0.6 -> 220x330


def _fake_scan(monkeypatch, sc, answer, duration=20.0, lay=None):
    import json
    import types
    import numpy as np
    import kassa_pipeline.vlm_scan as vs
    from kassa_pipeline.config import PipelineConfig
    monkeypatch.setattr(vs, "_build", lambda *a, **k: ([], []))
    monkeypatch.setattr(vs, "_frames_at", lambda v, ts: [types.SimpleNamespace(t=t, rgb=np.zeros((400, 600, 3), np.uint8)) for t in ts])
    prompts = []

    def ask_many(items, **k):
        prompts.extend(p for _, p in items)
        return [answer if "zoomed views of the safe" in p else "{}" for _, p in items]

    s1 = {"frames": [], "layouts": [dict({"start": 0.0}, **(lay or {}))], "people": []}
    res = vs.scan(types.SimpleNamespace(duration=duration), s1, types.SimpleNamespace(ask_many=ask_many), PipelineConfig(scan=sc))
    return res, prompts


def test_scan_builds_one_safe_item_per_window_next_to_the_other_items(monkeypatch):
    lay = {"safe": [200, 100, 300, 250]}
    res, prompts = _fake_scan(monkeypatch, ScanConfig(), "{}", lay=lay)
    assert sum('zoomed views of the safe' in p for p in prompts) == 3  # 20 s / 8 s windows
    assert res["safe"]["frames"] == [] and len(res["safe"]["calls"]) == 3
    assert not any('zoomed views of the safe' in p for p in _fake_scan(monkeypatch, ScanConfig(safe_zoom=False), "{}", lay=lay)[1])
    assert not any('zoomed views of the safe' in p for p in _fake_scan(monkeypatch, ScanConfig(), "{}")[1])


def test_scan_parses_safe_answers_by_position_with_tolerant_values(monkeypatch):
    import json
    ans = json.dumps({"frames": [{"t": 99, "open": "yes"}, {"t": 98, "open": "No"}, {"t": 97, "open": "not_visible"},
                                 {"t": 96, "open": "maybe"}, {"t": 95, "open": True}, {"t": 94, "open": " NO "},
                                 {"t": 93}, "junk", {"t": 0, "open": "yes"}, {"t": 1, "open": "yes"}]})
    res, _ = _fake_scan(monkeypatch, ScanConfig(), ans, duration=8.0, lay={"safe": [200, 100, 300, 250]})
    got = res["safe"]["frames"]  # 8 views were sent: extra answers are dropped, t comes from the request
    assert [f["t"] for f in got] == [float(i) for i in range(8)]
    assert [f["open"] for f in got] == ["yes", "no", "not_visible", "not_visible", "yes", "no", "not_visible", "not_visible"]
    res2, _ = _fake_scan(monkeypatch, ScanConfig(), "not json", duration=8.0, lay={"safe": [200, 100, 300, 250]})
    assert res2["safe"]["frames"] == [] and res2["vlm"]["unparsed"] >= 1


def test_v12_from_zoom_open_run_then_cashier_away():
    f = _v12([], _safe_s1(10), _safe_frames(4, 14))  # open from 4 s, cashier leaves the safe at 10 s
    assert [x.t for x in f] == [10.0] and f[0].source.endswith("(zoom)") and f[0].evidence == ["safe open 4.0-15.0s"]


def test_v12_from_zoom_ignores_a_short_open_run():
    assert _v12([], _safe_s1(5), _safe_frames(4, 5)) == []


def test_v12_zoom_closed_overrides_window_state_open():  # real X-5 shape
    wins = [_safe_win(0, 8, "yes"), _safe_win(4, 12, "yes"), _safe_win(8, 16, "yes")]
    assert _v12(wins, _safe_s1(5), None) != []  # old behaviour fires
    assert _v12(wins, _safe_s1(5), _safe_frames(99, 99)) == []


def test_v12_zoom_open_overrides_window_state_closed():  # real X-20 shape
    wins = [_safe_win(0, 8, "no"), _safe_win(4, 12, "no"), _safe_win(8, 16, "no")]
    assert [x.t for x in _v12(wins, _safe_s1(12), _safe_frames(2, 14))] == [12.0]


def test_v12_falls_back_to_window_state_without_usable_zoom_data():
    wins = [_safe_win(0, 8, "yes"), _safe_win(4, 12, "yes"), _safe_win(8, 16, "yes")]
    old = _v12(wins, _safe_s1(5), None)
    assert old and [(x.t, x.source) for x in _v12(wins, _safe_s1(5), [])] == [(x.t, x.source) for x in old]
    nv = [{"t": float(t), "open": "not_visible"} for t in range(20)]
    assert [x.t for x in _v12(wins, _safe_s1(5), nv)] == [x.t for x in old]
    assert [x.t for x in _v12(wins, _safe_s1(5), _safe_frames(99, 99), ScanConfig(safe_zoom=False))] == [x.t for x in old]


def _wins(ans, spans=((0, 8), (4, 12), (8, 16), (12, 20))):
    return [_safe_win(a, b, ans) for a, b in spans]


def test_v12_zoom_covering_only_the_first_half_leaves_the_second_to_window_state():
    wins = _wins("yes")
    f = _v12(wins, _safe_s1(5), [{"t": float(t), "open": "no"} for t in range(10)])
    assert [x.t for x in f] == [12.0] and not f[0].source.endswith("(zoom)")


def test_v12_mostly_not_visible_zoom_falls_back_for_that_span():
    nv = [{"t": float(t), "open": "not_visible"} for t in range(20)]
    nv[3] = {"t": 3.0, "open": "no"}
    assert [x.t for x in _v12(_wins("yes"), _safe_s1(5), nv)] == [x.t for x in _v12(_wins("yes"), _safe_s1(5), None)] != []


def test_safe_open_run_survives_one_not_visible_frame_but_not_a_no():
    from kassa_pipeline.vlm_scan import _safe_open_runs
    sc = ScanConfig()
    one = [{"t": float(t), "open": "yes"} for t in range(6)]
    one[3]["open"] = "not_visible"
    assert _safe_open_runs(one, sc) == [(0.0, 6.0)]
    two = [dict(f) for f in one]
    two[2]["open"] = "not_visible"
    assert _safe_open_runs(two, sc) == []  # 0-1 and 4-5 are 2 s each
    no = [dict(f) for f in one]
    no[2]["open"] = "no"
    assert _safe_open_runs(no, sc) == []


def _seq(s):
    return [{"t": float(i), "open": {"Y": "yes", "n": "no", ".": "not_visible"}[c]} for i, c in enumerate(s)]


X19 = "n" * 38 + "YYYY" + "n" * 16
X39 = "n" * 30 + "YYYYY" + "n" * 30 + "YYYY" + "n" * 8
X20 = "nn" + "Y" * 10


def test_safe_access_that_is_closed_again_is_not_v12_on_real_x19_x39():
    from kassa_pipeline.vlm_scan import _safe_open_runs
    for s in (X19, X39):
        assert _safe_open_runs(_seq(s), ScanConfig()) == []
        assert _v12([], _safe_s1(0), _seq(s)) == []


def test_safe_left_open_to_the_end_is_v12_on_real_x20():
    from kassa_pipeline.vlm_scan import _safe_open_runs
    assert _safe_open_runs(_seq(X20), ScanConfig()) == [(2.0, 12.0)]


def test_safe_run_closed_afterwards_counts_only_from_safe_left_open_sec():
    from kassa_pipeline.vlm_scan import _safe_open_runs
    sc = ScanConfig()
    assert _safe_open_runs(_seq("n" * 5 + "Y" * 7 + "n" * 5), sc) == []
    assert _safe_open_runs(_seq("n" * 5 + "Y" * 8 + "n" * 5), sc) == [(5.0, 13.0)]
    assert _safe_open_runs(_seq("n" * 5 + "YY" + "n" * 2), sc) == []  # at the end but shorter than safe_open_sec
    assert _safe_open_runs(_seq("n" * 5 + "YYY"), sc) == [(5.0, 8.0)]


def test_safe_run_followed_by_a_final_no_is_closed_not_open_to_the_end():
    from kassa_pipeline.vlm_scan import _safe_open_runs
    sc = ScanConfig()
    assert _safe_open_runs(_seq("nn" + "Y" * 5 + "n"), sc) == []
    assert _safe_open_runs(_seq("nn" + "Y" * 5 + "."), sc) == [(2.0, 7.0)]  # unseen tail: the run still holds the last usable view
