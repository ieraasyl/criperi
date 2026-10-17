from factsutil import make_facts
from kassa_pipeline.scenarios.exit_rules import check_exit_items, check_exit_laptop

EXIT = {"t": 8.4, "type": "room_exit", "who": "cashier"}
ITEMS = ("desk_cash", "desk_bag", "desk_docs", "desk_personal")


def clean(**over):
    states = {k: "n" * 10 for k in ITEMS}
    states["laptop_lid"] = "c" * 10
    states.update(over)
    return states


def by_subject(results):
    return {r.subject: r for r in results}


def test_wt07_cash_left_at_exit():
    ft = make_facts(10, clean(desk_cash="y" * 10), [EXIT])
    r = by_subject(check_exit_items(ft))["desk_cash"]
    assert r.rule_id == "W02" and r.basis == "project"
    assert r.status == "violation"
    assert r.timecode_sec == 8.4
    assert r.evidence_sec == [7.4, 8.4]


def test_wt08_bag_left_reports_bag_not_cash():
    ft = make_facts(10, clean(desk_bag="y" * 10), [EXIT])
    r = by_subject(check_exit_items(ft))
    assert r["desk_bag"].status == "violation"
    assert r["desk_cash"].status == "pass"


def test_wt10_clean_exit_passes():
    ft = make_facts(10, clean(), [EXIT])
    results = check_exit_items(ft) + check_exit_laptop(ft)
    assert len(results) == 5
    assert all(r.status == "pass" for r in results)


def test_wt11_wt13_lid_open_or_partial_violates():
    for ch in "op":
        ft = make_facts(10, {"laptop_lid": ch * 10}, [EXIT])
        (r,) = check_exit_laptop(ft)
        assert r.rule_id == "W03" and r.status == "violation"


def test_wt14_closed_lid_passes_even_if_screen_unknown():
    ft = make_facts(10, {"laptop_lid": "c" * 10, "screen": "?" * 10}, [EXIT])
    (r,) = check_exit_laptop(ft)
    assert r.status == "pass"


def test_wt15_occluded_laptop_is_unknown():
    ft = make_facts(10, {"laptop_lid": "?" * 10}, [EXIT])
    (r,) = check_exit_laptop(ft)
    assert r.status == "unknown" and r.unknown_reason


def test_unknown_item_reason():
    ft = make_facts(10, {}, [EXIT])
    r = by_subject(check_exit_items(ft))["desk_cash"]
    assert r.status == "unknown"
    assert r.unknown_reason == "desk_cash not visible before exit"


def test_wt35_no_exit_no_violation():
    ft = make_facts(10, {"laptop_lid": "o" * 10})
    items, laptop = check_exit_items(ft), check_exit_laptop(ft)
    assert [r.status for r in items + laptop] == ["not_applicable"] * 2
    assert items[0].subject is None and items[0].timecode_sec is None


def test_wt36_last_state_counts():
    ft = make_facts(6, {"laptop_lid": "ccccoo"}, [{"t": 5.5, "type": "room_exit", "who": "cashier"}])
    (r,) = check_exit_laptop(ft)
    assert r.status == "violation"


def test_wt06_lost_track_is_not_exit():
    ft = make_facts(10, {"laptop_lid": "o" * 10}, [{"t": 5.0, "type": "room_exit", "who": "client-1"}])
    assert [r.status for r in check_exit_laptop(ft) + check_exit_items(ft)] == ["not_applicable"] * 2


def test_exit_at_clip_start_window_is_clipped():
    ev = [{"t": 0.5, "type": "room_exit", "who": "cashier"}]
    (r,) = check_exit_laptop(make_facts(2, {"laptop_lid": "?o"}, ev))
    assert r.status == "unknown"
    (r,) = check_exit_laptop(make_facts(2, {"laptop_lid": "o?"}, ev))
    assert r.status == "violation"


def test_exit_past_last_second_is_clipped():
    ft = make_facts(4, {"laptop_lid": "ccco"}, [{"t": 4.0, "type": "room_exit", "who": "cashier"}])
    (r,) = check_exit_laptop(ft)  # t == duration: start clipped to last second, no IndexError
    assert r.status == "violation"


def test_each_exit_judged_separately():
    ev = [
        {"t": 3.5, "type": "room_exit", "who": "cashier"},
        {"t": 5.0, "type": "room_entry", "who": "cashier"},
        {"t": 7.5, "type": "room_exit", "who": "cashier"},
    ]
    ft = make_facts(8, {"laptop_lid": "ccccoooo"}, ev)
    first, second = check_exit_laptop(ft)
    assert (first.status, first.timecode_sec) == ("pass", 3.5)
    assert (second.status, second.timecode_sec) == ("violation", 7.5)
