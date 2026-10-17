from factsutil import make_facts
from kassa_pipeline.scenarios.episodes import build_episodes


def _ev(t, type, who=None, **kw):
    return {"t": t, "type": type, "who": who, **kw}


def test_batches_link_to_their_client():
    ft = make_facts(30, events=[
        _ev(1, "room_entry", "client-1"),
        _ev(2, "cash_received", "cashier", batch="b1", **{"from": "client-1"}),
        _ev(5, "room_exit", "client-1"),
        _ev(8, "room_entry", "client-2"),
        _ev(9, "bag_received", "cashier", batch="b2", **{"from": "client-2"}),
        _ev(12, "room_exit", "client-2"),
    ])
    eps = build_episodes(ft)
    assert [e.client for e in eps] == ["client-1", "client-2"]
    assert (eps[0].entry_t, eps[0].exit_t, eps[0].first_seen_t) == (1, 5, 1)
    assert [(b.id, b.kind, b.received_t) for b in eps[0].batches] == [("b1", "cash", 2)]
    assert [(b.id, b.kind, b.client) for b in eps[1].batches] == [("b2", "bag", "client-2")]


def test_count_and_store_matched_by_batch_id():
    ft = make_facts(30, events=[
        _ev(2, "cash_received", "cashier", batch="b1", **{"from": "client-1"}),
        _ev(4, "machine_count", "cashier", batch="b1"),
        _ev(6, "cash_stored", "cashier", batch="b1"),
        _ev(7, "cash_returned", "cashier", batch="b1"),
        _ev(8, "cash_stored", "cashier", batch="other"),
        _ev(3, "bag_received", "cashier", batch="b2", **{"from": "client-1"}),
        _ev(5, "bag_opened", "cashier", batch="b2"),
    ])
    b1, b2 = build_episodes(ft)[0].batches
    assert (b1.counted_t, b1.stored_t, b1.returned_t, b1.opened_t) == (4, 6, 7, None)
    assert (b2.opened_t, b2.counted_t, b2.stored_t) == (5, None, None)


def test_missing_batch_id_is_generated():
    ft = make_facts(30, events=[
        _ev(2, "cash_received", "cashier", **{"from": "client-1"}),
        _ev(4, "bag_received", "cashier", **{"from": "client-1"}),
    ])
    assert [b.id for b in build_episodes(ft)[0].batches] == ["client-1-b1", "client-1-b2"]


def test_stage_without_batch_links_to_latest_open_batch():
    ft = make_facts(30, events=[
        _ev(1, "machine_count", "cashier"),  # before any handover: ignored
        _ev(5, "cash_received", "cashier", **{"from": "client-1"}),
        _ev(6, "machine_count", "cashier"),
        _ev(10, "bag_received", "cashier", **{"from": "client-2"}),
        _ev(11, "bag_opened", "cashier"),
        _ev(12, "cash_stored", "cashier"),
        _ev(14, "cash_stored", "cashier"),
        _ev(15, "machine_count", "cashier"),
        _ev(16, "cash_returned", "cashier"),
    ])
    (b1,), (b2,) = [ep.batches for ep in build_episodes(ft)]
    assert (b1.counted_t, b1.opened_t, b1.stored_t, b1.returned_t) == (6, None, 14, None)
    assert (b2.counted_t, b2.opened_t, b2.stored_t, b2.returned_t) == (15, 11, 12, 16)


def test_stage_with_explicit_batch_is_not_reassigned():
    ft = make_facts(30, events=[
        _ev(2, "cash_received", "cashier", batch="b1", **{"from": "client-1"}),
        _ev(4, "cash_received", "cashier", batch="b2", **{"from": "client-1"}),
        _ev(6, "cash_stored", "cashier", batch="b1"),
        _ev(8, "cash_stored", "cashier"),
    ])
    b1, b2 = build_episodes(ft)[0].batches
    assert (b1.stored_t, b2.stored_t) == (6, 8)
