"""Test helper: build FactTimeline objects from compact strings."""

import math

from kassa_pipeline.facts import UNKNOWN, Event, FactTimeline

_LID = {"c": "closed", "o": "open", "p": "partial", "?": UNKNOWN}
_YESNO = {"y": "yes", "n": "no", "?": UNKNOWN}
_SAFE = {"o": "open", "c": "closed", "?": UNKNOWN}
_SCREEN = {"w": "work", "l": "locked", "f": "off", "?": UNKNOWN}
_CHARS = {
    "laptop_lid": _LID,
    "safe_door": _SAFE,
    "screen": _SCREEN,
    "desk_cash": _YESNO,
    "desk_bag": _YESNO,
    "desk_personal": _YESNO,
    "desk_docs": _YESNO,
}


def _expand(name: str, value) -> list:
    if isinstance(value, str):
        return [_CHARS[name][ch] for ch in value]
    return list(value)


def make_facts(
    duration: float,
    states: dict[str, list] | None = None,
    events: list[dict] | None = None,
    observable: list[str] | None = None,
) -> FactTimeline:
    """Timeline with the given states (strings like "cco?" or lists); the rest unknown."""
    facts = FactTimeline.empty("test", duration)
    n = math.ceil(duration)
    for name, value in (states or {}).items():
        arr = _expand(name, value) if name in _CHARS else list(value)
        if len(arr) != n:
            raise ValueError(f"{name}: expected {n} values, got {len(arr)}")
        facts.states[name] = arr
    facts.events = [Event.from_dict({**e, "from": e["from"]}) if "from" in e else Event(**e)
                    for e in (events or [])]
    if observable is not None:
        facts.observable = list(observable)
    return facts
