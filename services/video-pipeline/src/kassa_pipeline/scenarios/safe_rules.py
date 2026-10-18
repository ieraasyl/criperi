"""Safe rule W01: the safe door must not be left open after access ends or when the cashier leaves."""

from ..facts import FactTimeline
from .base import RuleResult, cashier_exits, decisive_sec

SAFE_SETTLE_SEC = 2
SUBJECT_CASH = "cash_in_safe"


def _open_run(ft: FactTimeline, sec: int) -> tuple[int, int]:
    """Maximal run of consecutive open seconds containing sec."""
    lo = hi = sec
    while ft.state("safe_door", lo - 1) == "open":
        lo -= 1
    while ft.state("safe_door", hi + 1) == "open":
        hi += 1
    return lo, hi


def _windows(ft: FactTimeline) -> list[tuple[float, float]]:
    """(access start, access end) per safe_access_end; start 0 if none precedes it."""
    starts = [e.t for e in ft.events_of("safe_access_start")]
    return [(max([s for s in starts if s <= e.t], default=0), e.t)
            for e in ft.events_of("safe_access_end")]


def _subject(ft: FactTimeline, window: tuple[float, float] | None) -> str | None:
    if window and any(window[0] <= e.t <= window[1] for e in ft.events_of("cash_stored")):
        return SUBJECT_CASH
    return None


def check_safe(ft: FactTimeline) -> list[RuleResult]:
    """W01: safe door closed after access ends and at every cashier room exit."""
    windows = _windows(ft)
    exits = cashier_exits(ft)
    if not windows and not exits:
        return [RuleResult("W01", "project", "not_applicable")]
    results = []
    reported: set[tuple[int, int]] = set()
    for window in windows:
        ta = window[1]
        secs = range(int(ta) + 1, min(int(ta) + SAFE_SETTLE_SEC, ft.n_seconds - 1) + 1)
        values = [ft.state("safe_door", s) for s in secs]
        subject = _subject(ft, window)
        if values and all(v == "open" for v in values):
            reported.add(_open_run(ft, secs[0]))
            results.append(RuleResult(
                "W01", "project", "violation", timecode_sec=ta, subject=subject,
                evidence_sec=[ta, ta + SAFE_SETTLE_SEC],
                facts=[f"safe access ended at {ta} s", "safe door stayed open after access ended"]))
        elif "closed" in values:
            results.append(RuleResult(
                "W01", "project", "pass", timecode_sec=ta, subject=subject,
                evidence_sec=[ta, ta + SAFE_SETTLE_SEC],
                facts=[f"safe access ended at {ta} s", "safe door closed after access"]))
        else:
            results.append(RuleResult(
                "W01", "project", "unknown", timecode_sec=ta, subject=subject,
                evidence_sec=[ta, ta + SAFE_SETTLE_SEC],
                facts=[f"safe access ended at {ta} s", "safe door state after access not fully visible"],
                unknown_reason="safe door not visible after access ended"))
    for ev in exits:
        sec = decisive_sec(ft, "safe_door", ev.t)
        door = ft.state("safe_door", sec) if sec is not None else "unknown"
        subject = _subject(ft, max((w for w in windows if w[1] <= ev.t), key=lambda w: w[1], default=None))
        facts = [f"cashier crossed the room exit at {ev.t} s", f"safe_door: {door}"]
        evidence = [max(0, ev.t - 1), ev.t]
        if door == "open":
            run = _open_run(ft, sec)
            if run in reported:
                continue
            reported.add(run)
            status = "violation"
        else:
            status = "pass" if door == "closed" else "unknown"
        results.append(RuleResult(
            "W01", "project", status, timecode_sec=ev.t, subject=subject, evidence_sec=evidence,
            facts=facts,
            unknown_reason="safe door not visible before exit" if status == "unknown" else None))
    return results
