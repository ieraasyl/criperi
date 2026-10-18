"""Rule result type and shared helpers for scenario rules."""

from dataclasses import asdict, dataclass, field

from ..facts import UNKNOWN, Event, FactTimeline

EXIT_WINDOW_SEC = 3


@dataclass
class RuleResult:
    rule_id: str
    basis: str  # law | project | project_assumption
    status: str  # pass | violation | unknown | not_applicable
    timecode_sec: float | None = None
    subject: str | None = None
    episode: str | None = None
    batch: str | None = None
    evidence_sec: list[float] = field(default_factory=list)
    facts: list[str] = field(default_factory=list)
    unknown_reason: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def decisive_sec(ft: FactTimeline, name: str, t: float, window: int = EXIT_WINDOW_SEC) -> int | None:
    """Second of the latest non-unknown value among seconds int(t) down to int(t)-window, clipped; None if none."""
    missing = None if name == "people_inside" else UNKNOWN
    hi = min(int(t), ft.n_seconds - 1)
    lo = max(int(t) - window, 0)
    for sec in range(hi, lo - 1, -1):
        if ft.state(name, sec) != missing:
            return sec
    return None


def decisive(ft: FactTimeline, name: str, t: float, window: int = EXIT_WINDOW_SEC) -> str | int | None:
    """Latest non-unknown value among seconds int(t) down to int(t)-window, clipped to the clip."""
    sec = decisive_sec(ft, name, t, window)
    if sec is None:
        return None if name == "people_inside" else UNKNOWN
    return ft.state(name, sec)


def cashier_exits(ft: FactTimeline) -> list[Event]:
    return ft.events_of("room_exit", who="cashier")
