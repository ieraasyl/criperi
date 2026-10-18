from factsutil import make_facts
from kassa_pipeline.scenarios.episodes import build_episodes
from kassa_pipeline.scenarios.service_rules import check_bag_opened, check_counted, check_id_first


def _ev(t, type, who=None, batch=None, frm=None):
    e = {"t": t, "type": type, "who": who, "batch": batch}
    if frm:
        e["from"] = frm
    return e


def _run(rule, events, observable=None, duration=60):
    ft = make_facts(duration, events=events, observable=observable)
    return rule(ft, build_episodes(ft))


def _cash(t, batch="b1", client="client-1"):
    return _ev(t, "cash_received", who="cashier", batch=batch, frm=client)


def _enter(t=1, client="client-1"):
    return _ev(t, "room_entry", who=client)


def test_id_before_cash_passes():
    res = _run(check_id_first, [_enter(), _ev(5, "id_presented", "client-1"), _cash(10)])
    assert [(r.rule_id, r.basis, r.status, r.episode) for r in res] == [("P01", "project", "pass", "client-1")]


def test_cash_before_id_violates_at_cash_time():
    res = _run(check_id_first, [_enter(), _cash(10), _ev(20, "id_presented", "client-1")])
    assert len(res) == 1 and res[0].status == "violation"
    assert res[0].timecode_sec == 10 and res[0].evidence_sec == [9, 10]


def test_id_by_other_client_does_not_count():
    res = _run(check_id_first, [_enter(), _enter(2, "client-2"), _ev(5, "id_presented", "client-2"), _cash(10)])
    ep1 = [r for r in res if r.episode == "client-1"]
    assert ep1[0].status == "violation"


def test_recording_starts_mid_operation_is_unknown():
    res = _run(check_id_first, [_ev(5, "id_presented", "client-1"), _cash(10)])
    assert res[0].status == "unknown" and res[0].unknown_reason == "start of service not observed"


def test_id_not_observable_is_unknown():
    obs = [t for t in ("room_entry", "cash_received") ]
    res = _run(check_id_first, [_enter(), _cash(10)], observable=obs)
    assert res[0].status == "unknown" and res[0].unknown_reason == "ID presentation not observable"


def test_no_episodes_or_batches_not_applicable():
    res = _run(check_id_first, [])
    assert [(r.status, r.episode) for r in res] == [("not_applicable", None)]
    res = _run(check_id_first, [_enter()])
    assert [(r.status, r.episode) for r in res] == [("not_applicable", "client-1")]
    assert [r.status for r in _run(check_counted, [])] == ["not_applicable"]
    assert [r.status for r in _run(check_bag_opened, [_enter(), _cash(5)])] == ["not_applicable"]


def test_two_batches_one_uncounted_flags_only_that_batch():
    ev = [_enter(), _cash(5, "b1"), _ev(8, "machine_count", batch="b1"), _ev(12, "cash_stored", batch="b1"),
          _cash(15, "b2"), _ev(20, "cash_stored", batch="b2")]
    res = _run(check_counted, ev)
    by = {r.batch: r for r in res}
    assert by["b1"].status == "pass"
    assert by["b2"].status == "violation" and by["b2"].timecode_sec == 20
    assert by["b2"].evidence_sec == [19, 20] and by["b2"].episode == "client-1"


def test_counted_after_storage_is_violation():
    ev = [_enter(), _cash(5), _ev(12, "cash_stored", batch="b1"), _ev(14, "machine_count", batch="b1")]
    res = _run(check_counted, ev)
    assert res[0].status == "violation" and res[0].timecode_sec == 12


def test_unfinished_batch_unknown():
    res = _run(check_counted, [_enter(), _cash(5)])
    assert res[0].status == "unknown" and res[0].unknown_reason == "batch not finished in clip"


def test_exit_is_decision_moment_when_not_stored():
    res = _run(check_counted, [_enter(), _cash(5), _ev(30, "room_exit", "client-1")])
    assert res[0].status == "violation" and res[0].timecode_sec == 30


def test_machine_count_not_observable_is_unknown():
    obs = ["room_entry", "cash_received", "cash_stored"]
    res = _run(check_counted, [_enter(), _cash(5), _ev(12, "cash_stored", batch="b1")], observable=obs)
    assert res[0].status == "unknown"


def test_bag_stored_unopened_violates():
    ev = [_enter(), _ev(5, "bag_received", "cashier", "b1", "client-1"), _ev(12, "cash_stored", batch="b1")]
    res = _run(check_bag_opened, ev)
    assert res[0].rule_id == "P03" and res[0].status == "violation" and res[0].timecode_sec == 12


def test_bag_opened_and_counted_passes():
    ev = [_enter(), _ev(5, "bag_received", "cashier", "b1", "client-1"), _ev(7, "bag_opened", batch="b1"),
          _ev(9, "machine_count", batch="b1"), _ev(12, "cash_stored", batch="b1")]
    assert _run(check_bag_opened, ev)[0].status == "pass"
    assert _run(check_counted, ev)[0].status == "pass"


def test_bag_opened_not_observable_is_unknown():
    obs = ["room_entry", "bag_received", "cash_stored"]
    ev = [_enter(), _ev(5, "bag_received", "cashier", "b1", "client-1"), _ev(12, "cash_stored", batch="b1")]
    assert _run(check_bag_opened, ev, observable=obs)[0].status == "unknown"


def test_returned_batch_not_applicable():
    ev = [_enter(), _cash(5), _ev(8, "cash_returned", batch="b1"), _ev(12, "cash_stored", batch="b1")]
    assert _run(check_counted, ev)[0].status == "not_applicable"
    ev = [_enter(), _cash(5), _ev(8, "cash_returned", batch="b1")]
    assert _run(check_counted, ev)[0].status == "not_applicable"


def test_exit_before_reentry_is_not_decision_moment():
    ev = [_enter(1), _ev(5, "room_exit", "client-1"), _enter(8), _cash(10), _ev(12, "machine_count", batch="b1")]
    res = _run(check_counted, ev)
    assert res[0].status == "unknown" and res[0].unknown_reason == "batch not finished in clip"
    bag = [_enter(1), _ev(5, "room_exit", "client-1"), _enter(8),
           _ev(10, "bag_received", "cashier", "b1", "client-1"), _ev(11, "bag_opened", batch="b1")]
    assert _run(check_bag_opened, bag)[0].status == "unknown"


def test_events_without_batch_ids_link_and_pass():
    ev = [_enter(), _cash(5, batch=None), _ev(8, "machine_count"), _ev(12, "cash_stored")]
    res = _run(check_counted, ev)
    assert [(r.status, r.timecode_sec) for r in res] == [("pass", 12)]


def test_id_shown_at_door_just_before_entry_passes():
    res = _run(check_id_first, [_ev(7, "id_presented", "client-1"), _enter(10), _cash(15)])
    assert res[0].status == "pass" and "ID presented at 7 s" in res[0].facts
    res = _run(check_id_first, [_ev(6.5, "id_presented", "client-1"), _enter(10), _cash(15)])
    assert res[0].status == "violation"
