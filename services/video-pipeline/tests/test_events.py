"""Event rules and timeline smoothing: pure functions, no models needed."""

from kassa_pipeline.config import EventsConfig
from kassa_pipeline.events import find_candidates, merge, Candidate
from kassa_pipeline.timeline import Second, Timeline, Visitor, clean_bool

CFG = EventsConfig()


def tl(seconds: list[Second]) -> Timeline:
    return Timeline(seconds, float(len(seconds)))


def sec(t, zone="cashier_seat", **kw) -> Second:
    at = zone in ("cashier_seat", "at_desk")
    return Second(t=t, cashier_zone=zone, cashier_at_workplace=at, **kw)


def types(c):
    return sorted({x.type for x in c})


def test_clean_bool_fills_gaps_and_drops_blips():
    assert clean_bool([True, False, True, True], close_gap=1, min_run=2) == [True] * 4
    assert clean_bool([False, True, False, False], close_gap=1, min_run=2) == [False] * 4


def test_cash_left_when_cashier_gets_up_is_dated_to_getting_up():
    secs = [sec(i) for i in range(4)]
    secs += [sec(4, "standing_at_seat"), sec(5, "standing_at_seat")]
    secs += [sec(6 + i, "doorway", cash_on_desk=True, cash_score=0.3) for i in range(4)]
    secs[5].cash_on_desk = True
    c = [x for x in find_candidates(tl(secs), CFG) if x.type == "T1"]
    assert len(c) == 1 and c[0].timecode == 4  # cashier got up at 4, cash seen from 5


def test_walking_away_without_cash_or_unlocked_screen_is_not_a_violation():
    """Four no-violation training clips do exactly this."""
    secs = [sec(i) for i in range(3)] + [sec(3 + i, "doorway") for i in range(6)]
    assert find_candidates(tl(secs), CFG) == []


def test_unlocked_screen_only_flags_when_screen_stays_on():
    on = [sec(i, screen="work_screen") for i in range(3)] + [
        sec(3 + i, "doorway", screen="work_screen") for i in range(5)
    ]
    locked = [sec(i, screen="work_screen") for i in range(3)] + [
        sec(3 + i, "doorway", screen="locked") for i in range(5)
    ]
    assert types(find_candidates(tl(on), CFG)) == ["T5"]
    assert find_candidates(tl(locked), CFG) == []


def test_leaving_room_during_service_requires_a_client():
    client = [Visitor(1, "at_desk", 1.5)]
    with_client = [sec(i, visitors=client) for i in range(2)] + [
        sec(2 + i, "doorway", visitors=client) for i in range(4)
    ]
    assert "T6" in types(find_candidates(tl(with_client), CFG))
    alone = [sec(i) for i in range(2)] + [sec(2 + i, "doorway") for i in range(4)]
    assert find_candidates(tl(alone), CFG) == []


def test_personal_items_need_to_persist():
    brief = [sec(i, personal_on_desk=(i == 3)) for i in range(8)]
    long = [sec(i, personal_on_desk=(2 <= i <= 6)) for i in range(8)]
    assert find_candidates(tl(brief), CFG) == []
    assert types(find_candidates(tl(long), CFG)) == ["T2"]


def test_new_visitor_arriving_while_old_cash_is_on_desk():
    secs = [sec(i, cash_on_desk=True, cash_score=0.2) for i in range(8)]
    secs[6].arrivals = [5]
    c = [x for x in find_candidates(tl(secs), CFG) if x.type == "T1"]
    assert c and any(x.facts.get("mode") == "new_visitor_arrives_with_cash_on_desk" for x in c)


def test_merge_same_type_close_candidates():
    a = Candidate("T1", 5, 7, 5, 0.6)
    b = Candidate("T1", 9, 12, 9, 0.8)
    m = merge([a, b], gap=10)
    assert len(m) == 1 and m[0].timecode == 5 and m[0].score == 0.8 and m[0].t_end == 12
