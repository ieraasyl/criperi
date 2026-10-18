"""Room-exit rules: W02 (desk items) and W03 (laptop lid)."""

from ..facts import FactTimeline
from .base import RuleResult, cashier_exits, decisive

DESK_ITEMS = ("desk_cash", "desk_bag", "desk_docs", "desk_personal")


def check_exit_items(ft: FactTimeline) -> list[RuleResult]:
    """W02: nothing left on the desk when the cashier leaves the room."""
    exits = cashier_exits(ft)
    if not exits:
        return [RuleResult("W02", "project", "not_applicable")]
    results = []
    for ev in exits:
        for item in DESK_ITEMS:
            value = decisive(ft, item, ev.t)
            status = {"yes": "violation", "no": "pass"}.get(value, "unknown")
            results.append(RuleResult(
                "W02", "project", status, timecode_sec=ev.t, subject=item,
                evidence_sec=[max(0, ev.t - 1), ev.t],
                facts=[f"cashier crossed the room exit at {ev.t} s", f"{item}: {value}"],
                unknown_reason=f"{item} not visible before exit" if status == "unknown" else None))
    return results


def check_exit_laptop(ft: FactTimeline) -> list[RuleResult]:
    """W03: laptop lid closed when the cashier leaves the room."""
    exits = cashier_exits(ft)
    if not exits:
        return [RuleResult("W03", "project", "not_applicable")]
    results = []
    for ev in exits:
        lid = decisive(ft, "laptop_lid", ev.t)
        status = {"open": "violation", "partial": "violation", "closed": "pass"}.get(lid, "unknown")
        results.append(RuleResult(
            "W03", "project", status, timecode_sec=ev.t,
            evidence_sec=[max(0, ev.t - 1), ev.t],
            facts=[f"cashier crossed the room exit at {ev.t} s", f"laptop_lid: {lid}"],
            unknown_reason="laptop lid not visible before exit" if status == "unknown" else None))
    return results
