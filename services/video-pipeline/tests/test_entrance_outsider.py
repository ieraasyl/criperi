import kassa_pipeline.vlm_scan as vs
from kassa_pipeline.config import ScanConfig
from kassa_pipeline.scenarios.entrance import entrance_findings
from kassa_pipeline.vlm_scan import Finding, Stage1, apply_rules


def _p(track, zone, role="visitor", h=2.0):
    return {"track": track, "role": role, "box": [0, 0, 1, h], "zone": zone}


def _s1(visitors, secs=10, cashier="cashier_seat", cashier_by_sec=None, vh=2.0, has_cashier=True):
    """visitors: {track: {second: zone}}; 2 records per second."""
    frames = []
    for s in range(secs):
        for k in range(2):
            people = [_p(tid, z[s], h=vh) for tid, z in visitors.items() if s in z]
            if has_cashier:
                people.append(_p(0, "cashier_seat", "cashier", 1.0))
            cz = (cashier_by_sec or {}).get(s, cashier)
            frames.append({"t": s + k * 0.5, "people": people, "objects": [],
                           "facts": {"cashier_zone": cz, "cashier_at_workplace": cz == "cashier_seat", "screen": "unknown", "people_in_room": 1}})
    return {"frames": frames, "layouts": [{"start": 0.0}], "people": [],
            "meta": {"name": "X-1.mov", "duration": float(secs)}}


def _sc(**kw):
    return ScanConfig(entrance_outsider=True, **kw)


def test_outsider_in_doorway_three_seconds_flagged():
    s1 = _s1({7: {s: "doorway" for s in range(2, 6)}})
    (f,) = entrance_findings(s1, _sc())
    assert (f.code, f.t, f.confidence) == ("V18", 2.0, 0.45)
    assert f.source.startswith("scenario ") and "track 7" in f.evidence[0]


def test_short_doorway_figure_not_flagged_and_no_cashier_fails_closed():
    z = {7: {s: "doorway" for s in range(2, 6)}}
    assert entrance_findings(_s1(z, vh=0.6), _sc()) == []
    assert len(entrance_findings(_s1(z, vh=1.0), _sc())) == 1  # as tall as the seated cashier
    assert entrance_findings(_s1(z, has_cashier=False), _sc()) == []


def test_seated_height_falls_back_to_all_cashier_records():
    s1 = _s1({7: {s: "doorway" for s in range(2, 6)}})
    for f in s1["frames"]:
        for p in f["people"]:
            if p["role"] == "cashier":
                p["zone"] = "elsewhere"
    assert len(entrance_findings(s1, _sc())) == 1


def test_two_seconds_not_enough():
    assert entrance_findings(_s1({7: {2: "doorway", 3: "doorway"}}), _sc()) == []


def test_client_entering_through_door_then_desk_not_flagged():
    z = {0: "doorway", 1: "doorway", 2: "doorway", 3: "doorway", 4: "at_desk", 5: "at_desk"}
    assert entrance_findings(_s1({7: z}), _sc()) == []


def test_reid_fragment_of_client_not_flagged():
    client = {s: "at_desk" for s in range(0, 3)}  # client t1 = 2.5
    frag = {s: "doorway" for s in range(4, 9)}  # fragment t0 = 4 within 3 s of client t1
    assert entrance_findings(_s1({7: client, 8: frag}), _sc()) == []
    far = {s: "doorway" for s in range(7, 10)}  # 4.5 s after client t1 -> real outsider
    (f,) = entrance_findings(_s1({7: client, 8: far}, secs=11), _sc())
    assert f.t == 7.0


def test_fragment_before_client_start_not_flagged():
    frag = {s: "doorway" for s in range(0, 4)}  # t1 = 3.5
    client = {s: "at_desk" for s in range(5, 9)}  # t0 = 5, within 3 s
    assert entrance_findings(_s1({8: frag, 7: client}), _sc()) == []


def test_cashier_out_of_room_not_flagged():
    z = {s: "doorway" for s in range(2, 6)}
    assert entrance_findings(_s1({7: z}, cashier_by_sec={s: "absent" for s in range(10)}), _sc()) == []
    assert entrance_findings(_s1({7: z}, cashier="doorway"), _sc()) == []


def test_cashier_leaving_mid_run_breaks_it():
    z = {s: "doorway" for s in range(2, 7)}
    assert entrance_findings(_s1({7: z}, cashier_by_sec={4: "absent"}), _sc()) == []


def test_apply_rules_default_off_and_gate(monkeypatch):
    s1 = Stage1(_s1({7: {s: "doorway" for s in range(2, 6)}}))
    assert apply_rules([], [], [], s1, ScanConfig()) == []
    assert apply_rules([], [], [], s1, _sc()) == []  # lone scenario finding is dropped
    monkeypatch.setattr(vs, "door_rule", lambda *a: [Finding("V18", 3.0, 0.6, "door open")])
    (g,) = apply_rules([], [], [], s1, _sc())
    assert g.confidence > 0.6 and "scenario entrance-outsider" in g.source


def test_errors_recorded(monkeypatch):
    import kassa_pipeline.scenarios.entrance as en
    monkeypatch.setattr(en, "entrance_findings", lambda *a: (_ for _ in ()).throw(ValueError("bad")))
    errs = []
    assert apply_rules([], [], [], Stage1(_s1({})), _sc(), errors=errs) == []
    assert errs == ["entrance: ValueError: bad"]
