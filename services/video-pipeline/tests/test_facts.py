import json

import pytest

from factsutil import make_facts
from kassa_pipeline.facts import FactTimeline, clip_id


def test_clip_id_normalises_cyrillic_and_latin():
    assert clip_id("Х-20") == clip_id("X-20.mov") == clip_id("X-20 (2)") == "X-20"
    assert clip_id("X-20 (2).mov") == "X-20"
    assert clip_id("demo") == "demo"


def test_roundtrip_preserves_everything(tmp_path):
    facts = make_facts(
        5,
        {"laptop_lid": "ccoo?"},
        [
            {"t": 3.2, "type": "room_exit", "who": "cashier", "origin": "cv"},
            {"t": 1.0, "type": "cash_returned", "who": "client-1", "from_": "cashier", "t_end": 2.0},
        ],
    )
    path = tmp_path / "X-1.json"
    facts.save(path)
    loaded = FactTimeline.load(path)
    assert loaded.to_dict() == facts.to_dict()
    assert loaded.events[1].from_ == "cashier"
    assert '"from"' in path.read_text(encoding="utf-8")
    assert not list(tmp_path.glob("*.tmp"))


def test_state_out_of_range_is_unknown():
    facts = make_facts(5, {"laptop_lid": "ccoo?"})
    assert facts.state("laptop_lid", 99) == "unknown"
    assert facts.state("laptop_lid", 2) == "open"
    assert facts.state("screen", 0) == "unknown"
    assert facts.state("people_inside", -1) is None
    assert facts.state("people_inside", 0) is None


def test_validate_rejects_bad_length():
    d = make_facts(5, {"laptop_lid": "ccoo?"}).to_dict()
    d["states"]["laptop_lid"] = ["closed"] * 4
    with pytest.raises(ValueError, match="laptop_lid"):
        FactTimeline.from_dict(d)


def test_validate_rejects_unknown_value_and_event_type():
    d = make_facts(5).to_dict()
    d["states"]["laptop_lid"] = ["ajar"] * 5
    with pytest.raises(ValueError, match="ajar"):
        FactTimeline.from_dict(d)
    d = make_facts(5).to_dict()
    d["events"] = [{"t": 1, "type": "teleport"}]
    with pytest.raises(ValueError, match="teleport"):
        FactTimeline.from_dict(d)


def test_validate_other_fields():
    base = make_facts(5).to_dict()
    for mutate in (
        lambda d: d["states"].update(people_inside=[0, -1, 1, 1, 1]),
        lambda d: d.update(status="final"),
        lambda d: d["events"].append({"t": 9, "type": "room_exit"}),
        lambda d: d["events"].append({"t": 2, "type": "room_exit", "t_end": 1}),
        lambda d: d["state_origin"].update(laptop_lid=["vlm"]),
        lambda d: d["state_origin"].update(screen=["x"] * 5),
        lambda d: d["zones"].update(desk=[1, 2, 3]),
    ):
        d = json.loads(json.dumps(base))
        mutate(d)
        with pytest.raises(ValueError):
            FactTimeline.from_dict(d)


def test_empty_defaults():
    facts = FactTimeline.empty("X-1", 2.5)
    assert facts.n_seconds == 3
    assert facts.states["people_inside"] == [None] * 3
    assert facts.states["safe_door"] == ["unknown"] * 3
    assert len(facts.observable) == 12
    d = facts.to_dict()
    del d["observable"]
    assert len(FactTimeline.from_dict(d).observable) == 12


def test_events_of_filters_and_sorts():
    facts = make_facts(
        10,
        events=[
            {"t": 5, "type": "room_exit", "who": "cashier"},
            {"t": 2, "type": "room_exit", "who": "client-1"},
            {"t": 1, "type": "room_exit", "who": "cashier"},
            {"t": 3, "type": "room_entry", "who": "cashier"},
        ],
    )
    assert [e.t for e in facts.events_of("room_exit")] == [1, 2, 5]
    assert [e.t for e in facts.events_of("room_exit", "cashier")] == [1, 5]


@pytest.mark.parametrize("patch", [
    {"events": [{"t": 1, "type": "room_exit", "t_end": "abc"}]},
    {"events": [{"t": 1, "type": "room_exit", "t_end": float("nan")}]},
    {"events": [{"t": 1, "type": "room_exit", "t_end": float("inf")}]},
    {"events": [{"t": 1, "type": "room_exit", "t_end": True}]},
    {"duration": "inf"},
    {"duration": float("inf")},
    {"duration": "nan"},
    {"duration": -3},
    {"duration": 0},
    {"duration": 10 ** 400},
    {"events": [{"t": 1, "type": "room_exit", "who": "client"}]},
    {"events": [{"t": 1, "type": "room_exit", "who": "Client-1"}]},
    {"events": [{"t": 1, "type": "room_exit", "who": "bob"}]},
    {"events": [{"t": 1, "type": "room_exit", "who": "client-1 "}]},
    {"events": [{"t": 1, "type": "cash_received", "who": "cashier", "from": "client-x"}]},
    {"events": [{"t": 1, "type": "cash_received", "who": "cashier", "from": 3}]},
])
def test_from_dict_malformed_always_value_error(patch):
    base = FactTimeline.empty("X-1", 5).to_dict()
    with pytest.raises(ValueError):
        FactTimeline.from_dict({**base, **patch})


def test_who_and_from_accept_known_actors():
    base = FactTimeline.empty("X-1", 5).to_dict()
    events = [{"t": 1, "type": "cash_received", "who": "cashier", "from": "client-12"},
              {"t": 2, "type": "room_entry", "who": "other-3"}, {"t": 3, "type": "machine_count"}]
    assert len(FactTimeline.from_dict({**base, "events": events}).events) == 3
