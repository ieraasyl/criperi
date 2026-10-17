"""Changeover rules: W05 previous client's cash left on the desk, W06 three or more people inside."""

from ..facts import FactTimeline
from .base import RuleResult, decisive
from .episodes import Episode

MAX_PEOPLE = 2


def check_previous_cash(ft: FactTimeline, episodes: list[Episode]) -> list[RuleResult]:
    """W05: when client k enters, earlier clients' cash must not still lie on the desk."""
    if len(episodes) < 2:
        return [RuleResult("W05", "project", "not_applicable")]
    results = []
    for k in range(1, len(episodes)):
        ep, previous = episodes[k], episodes[:k]
        prev_id = episodes[k - 1].client
        tb = ep.entry_t
        if not any(b.received_t <= (tb if tb is not None else float("inf"))
                   for p in previous for b in p.batches):
            results.append(RuleResult("W05", "project", "not_applicable", episode=prev_id))
            continue
        if tb is None:
            results.append(RuleResult(
                "W05", "project", "unknown", episode=prev_id,
                facts=[f"{ep.client} entry not observed"], unknown_reason="entry not observed"))
            continue
        outstanding = [b for p in previous for b in p.batches
                       if b.received_t <= tb and (b.stored_t is None or b.stored_t > tb)
                       and (b.returned_t is None or b.returned_t > tb)]
        evidence = [max(0, tb - 1), tb]
        if not outstanding:
            results.append(RuleResult(
                "W05", "project", "pass", timecode_sec=tb, episode=prev_id, evidence_sec=evidence,
                facts=[f"{ep.client} entered at {tb} s", "previous cash already stored or returned"]))
            continue
        for b in outstanding:
            desk_name = "desk_bag" if b.kind == "bag" else "desk_cash"
            desk = decisive(ft, desk_name, tb, 1)
            status = {"yes": "violation", "no": "pass"}.get(desk, "unknown")
            results.append(RuleResult(
                "W05", "project", status, timecode_sec=tb, episode=b.client, batch=b.id,
                evidence_sec=evidence,
                facts=[f"{ep.client} entered at {tb} s", f"batch {b.id} not stored or returned",
                       f"{desk_name}: {desk}"],
                unknown_reason=f"{desk_name} not visible at entry" if status == "unknown" else None))
    return results


def check_people_inside(ft: FactTimeline) -> list[RuleResult]:
    """W06: one violation at the first second of every run with more than two people inside."""
    values = [ft.state("people_inside", s) for s in range(ft.n_seconds)]
    results = []
    prev_over = False
    for s, v in enumerate(values):
        over = v is not None and v > MAX_PEOPLE
        if over and not prev_over:
            results.append(RuleResult(
                "W06", "project", "violation", timecode_sec=s,
                evidence_sec=[s, min(s + 1, ft.duration)],
                facts=[f"people_inside: {v} at {s} s"]))
        prev_over = over
    if results:
        return results
    if values and all(v is not None for v in values):
        return [RuleResult("W06", "project", "pass", facts=[f"people_inside never above {MAX_PEOPLE}"])]
    return [RuleResult("W06", "project", "unknown", unknown_reason="people count not available for the whole clip")]
