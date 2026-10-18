from factsutil import make_facts
from kassa_pipeline.scenarios.base import decisive, decisive_sec


def test_decisive_sec_returns_the_deciding_second():
    ft = make_facts(10, {"safe_door": "ooo??????c"})
    assert decisive_sec(ft, "safe_door", 5.5) == 2
    assert decisive(ft, "safe_door", 5.5) == "open"


def test_decisive_sec_none_when_all_unknown():
    ft = make_facts(10, {"safe_door": "o?????????"})
    assert decisive_sec(ft, "safe_door", 8.0) is None
    assert decisive(ft, "safe_door", 8.0) == "unknown"


def test_decisive_sec_clipped_at_clip_start_and_end():
    ft = make_facts(5, {"safe_door": "?c???"})
    assert decisive_sec(ft, "safe_door", 1.2) == 1
    assert decisive_sec(ft, "safe_door", 0.5) is None
    assert decisive_sec(ft, "safe_door", 99) is None
