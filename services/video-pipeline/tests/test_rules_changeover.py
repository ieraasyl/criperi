from factsutil import make_facts
from kassa_pipeline.scenarios.changeover_rules import check_people_inside, check_previous_cash
from kassa_pipeline.scenarios.episodes import build_episodes


def _ev(t, type, who=None, **kw):
    return {"t": t, "type": type, "who": who, **kw}


def _prev(desk, extra=(), n=20):
    events = [
        _ev(2, "cash_received", "cashier", batch="b1", **{"from": "client-1"}),
        _ev(6, "room_exit", "client-1"),
        _ev(8, "room_entry", "client-2"),
        *extra,
    ]
    ft = make_facts(n, states={"desk_cash": desk}, events=events)
    return check_previous_cash(ft, build_episodes(ft))


def test_wt21_previous_cash_on_desk_when_b_enters():
    (r,) = _prev("n" * 8 + "y" + "n" * 11)
    assert (r.rule_id, r.basis, r.status) == ("W05", "project", "violation")
    assert (r.timecode_sec, r.batch, r.episode) == (8.0, "b1", "client-1")
    assert r.evidence_sec == [7, 8]


def test_wt22_stored_before_entry_passes():
    (r,) = _prev("n" * 8 + "y" + "n" * 11, extra=[_ev(5, "cash_stored", "cashier", batch="b1")])
    assert r.status == "pass"


def test_returned_before_entry_passes():
    (r,) = _prev("y" * 20, extra=[_ev(7, "cash_returned", "cashier", batch="b1")])
    assert r.status == "pass"


def test_desk_no_passes():
    (r,) = _prev("n" * 20)
    assert r.status == "pass"


def test_wt37_desk_unknown_gives_unknown():
    (r,) = _prev("?" * 20)
    assert r.status == "unknown"


def test_entry_not_observed_is_unknown():
    ft = make_facts(20, events=[
        _ev(2, "cash_received", "cashier", batch="b1", **{"from": "client-1"}),
        _ev(8, "bag_received", "cashier", batch="b2", **{"from": "client-2"}),
    ])
    (r,) = check_previous_cash(ft, build_episodes(ft))
    assert r.status == "unknown" and r.unknown_reason == "entry not observed"


def test_previous_without_batches_not_applicable():
    ft = make_facts(20, events=[_ev(1, "room_entry", "client-1"), _ev(8, "room_entry", "client-2")])
    (r,) = check_previous_cash(ft, build_episodes(ft))
    assert (r.status, r.episode) == ("not_applicable", "client-1")


def test_first_client_not_applicable():
    ft = make_facts(20, events=[_ev(1, "room_entry", "client-1")])
    (r,) = check_previous_cash(ft, build_episodes(ft))
    assert r.rule_id == "W05" and r.status == "not_applicable"


def test_several_outstanding_batches_one_result_each():
    ft = make_facts(20, states={"desk_cash": "y" * 20}, events=[
        _ev(2, "cash_received", "cashier", batch="b1", **{"from": "client-1"}),
        _ev(3, "cash_received", "cashier", batch="b2", **{"from": "client-1"}),
        _ev(8, "room_entry", "client-2"),
    ])
    rs = check_previous_cash(ft, build_episodes(ft))
    assert [(r.batch, r.status) for r in rs] == [("b1", "violation"), ("b2", "violation")]


def _people(arr):
    return check_people_inside(make_facts(len(arr), states={"people_inside": arr}))


def test_wt23_third_person_short_still_violates():
    (r,) = _people([1, 2, 3, 2])
    assert (r.rule_id, r.basis, r.status, r.timecode_sec) == ("W06", "project", "violation", 2)
    assert r.evidence_sec == [2, 3]


def test_wt24_doorway_overlap_not_counted():
    (r,) = _people([1, 2, 2, 2])
    assert r.status == "pass"


def test_two_runs_two_violations():
    rs = _people([1, 3, 3, 2, 4, 1])
    assert [(r.status, r.timecode_sec) for r in rs] == [("violation", 1), ("violation", 4)]


def test_null_people_unknown():
    (r,) = _people([None, None, None])
    assert r.status == "unknown"


def test_none_breaks_a_run():
    rs = _people([3, None, 3])
    assert [r.timecode_sec for r in rs] == [0, 2]


def test_evidence_clipped_to_duration():
    (r,) = _people([1, 3])
    assert r.evidence_sec == [1, 2]


def test_batch_received_after_entry_not_outstanding():
    ft = make_facts(20, states={"desk_cash": "y" * 20}, events=[
        _ev(1, "room_entry", "client-1"),
        _ev(8, "room_entry", "client-2"),
        _ev(10, "cash_received", "cashier", batch="b1", **{"from": "client-1"}),
    ])
    (r,) = check_previous_cash(ft, build_episodes(ft))
    assert r.status == "not_applicable"


def test_partial_unknown_people_is_unknown():
    (r,) = _people([None, 2, None])
    assert r.status == "unknown"
    assert r.unknown_reason == "people count not available for the whole clip"


def test_bag_batch_judged_on_desk_bag():
    ft = make_facts(20, states={"desk_bag": "y" * 20, "desk_cash": "n" * 20}, events=[
        _ev(2, "bag_received", "cashier", batch="b1", **{"from": "client-1"}),
        _ev(8, "room_entry", "client-2"),
    ])
    (r,) = check_previous_cash(ft, build_episodes(ft))
    assert (r.status, r.batch) == ("violation", "b1")
