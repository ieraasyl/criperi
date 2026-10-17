from factsutil import make_facts
from kassa_pipeline.scenarios.safe_rules import SAFE_SETTLE_SEC, check_safe


def end(t):
    return {"t": t, "type": "safe_access_end"}


def start(t):
    return {"t": t, "type": "safe_access_start"}


def exit_(t):
    return {"t": t, "type": "room_exit", "who": "cashier"}


def statuses(results):
    return [r.status for r in results]


def test_settle_constant():
    assert SAFE_SETTLE_SEC == 2


def test_wt01_close_after_access_passes():
    ft = make_facts(5, {"safe_door": "ooocc"}, [end(2.5)])
    (r,) = check_safe(ft)
    assert r.rule_id == "W01" and r.basis == "project"
    assert r.status == "pass"
    assert r.timecode_sec == 2.5


def test_wt02_left_open_after_access_violates():
    ft = make_facts(6, {"safe_door": "oooooo"}, [end(1.5)])
    (r,) = check_safe(ft)
    assert r.status == "violation"
    assert r.timecode_sec == 1.5
    assert r.evidence_sec == [1.5, 3.5]
    assert any("open" in f for f in r.facts)


def test_wt03_no_access_end_no_result():
    ft = make_facts(5, {"safe_door": "ooooo"})
    (r,) = check_safe(ft)
    assert r.rule_id == "W01"
    assert r.status == "not_applicable"
    assert r.subject is None and r.timecode_sec is None


def test_wt04_closed_door_passes_lock_unknown():
    ft = make_facts(6, {"safe_door": "cccccc"}, [end(1.5), exit_(4.5)])
    results = check_safe(ft)
    assert statuses(results) == ["pass", "pass"]
    assert [r.timecode_sec for r in results] == [1.5, 4.5]


def test_open_at_exit_violates():
    ft = make_facts(10, {"safe_door": "o" * 10}, [exit_(8.4)])
    (r,) = check_safe(ft)
    assert r.status == "violation"
    assert r.timecode_sec == 8.4


def test_one_open_run_reported_once():
    ft = make_facts(8, {"safe_door": "o" * 8}, [end(1.5), exit_(4.5)])
    results = check_safe(ft)
    assert statuses(results) == ["violation"]
    assert results[0].timecode_sec == 1.5


def test_exit_in_separate_open_run_still_reported():
    ft = make_facts(10, {"safe_door": "ooocccooco"}, [end(0.5), exit_(7.5)])
    assert statuses(check_safe(ft)) == ["violation", "violation"]


def test_cash_stored_sets_subject():
    ft = make_facts(6, {"safe_door": "ooocco"}, [start(0.5), {"t": 1.0, "type": "cash_stored"}, end(2.5)])
    (r,) = check_safe(ft)
    assert r.subject == "cash_in_safe"


def test_no_cash_stored_subject_none():
    ft = make_facts(6, {"safe_door": "ooocco"}, [start(0.5), end(2.5)])
    (r,) = check_safe(ft)
    assert r.subject is None


def test_unknown_after_access_is_unknown():
    ft = make_facts(5, {"safe_door": "oo???"}, [end(1.5)])
    (r,) = check_safe(ft)
    assert r.status == "unknown"
    assert r.unknown_reason


def test_access_end_with_clipped_empty_window_is_unknown():
    ft = make_facts(5, {"safe_door": "ooooo"}, [end(4.5)])
    (r,) = check_safe(ft)
    assert r.status == "unknown"
    assert r.unknown_reason


def test_two_exits_in_one_open_run_report_once():
    ft = make_facts(10, {"safe_door": "o" * 10}, [exit_(4.5), exit_(8.4)])
    results = check_safe(ft)
    assert statuses(results) == ["violation"]
    assert results[0].timecode_sec == 4.5


def test_exit_with_unknown_door_is_unknown():
    ft = make_facts(10, {}, [exit_(8.4)])
    (r,) = check_safe(ft)
    assert r.status == "unknown"
    assert r.unknown_reason
