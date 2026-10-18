"""Service-order rules: P01 ID before money, P02 cash counted before storage, P03 bag opened before storage."""

from ..facts import FactTimeline
from .base import RuleResult
from .episodes import Batch, Episode

ID_AT_DOOR_SEC = 3  # an ID shown at the door this long before the inward crossing counts


def _evidence(t: float) -> list[float]:
    return [max(0, t - 1), t]


def check_id_first(ft: FactTimeline, episodes: list[Episode]) -> list[RuleResult]:
    """P01: the client presents ID no later than the first money/bag handover."""
    if not episodes:
        return [RuleResult("P01", "project", "not_applicable")]
    results = []
    for ep in episodes:
        if not ep.batches:
            results.append(RuleResult("P01", "project", "not_applicable", episode=ep.client))
            continue
        tc = ep.batches[0].received_t
        if "id_presented" not in ft.observable:
            results.append(RuleResult("P01", "project", "unknown", episode=ep.client,
                                      unknown_reason="ID presentation not observable"))
            continue
        if ep.entry_t is None:
            results.append(RuleResult("P01", "project", "unknown", episode=ep.client,
                                      unknown_reason="start of service not observed"))
            continue
        lo = ep.entry_t - ID_AT_DOOR_SEC
        shown = [e.t for e in ft.events_of("id_presented", who=ep.client) if lo <= e.t <= tc]
        facts = [f"first handover at {tc} s"]
        if shown:
            results.append(RuleResult("P01", "project", "pass", timecode_sec=tc, episode=ep.client,
                                      evidence_sec=_evidence(tc),
                                      facts=facts + [f"ID presented at {shown[0]} s"]))
        else:
            results.append(RuleResult("P01", "project", "violation", timecode_sec=tc, episode=ep.client,
                                      evidence_sec=_evidence(tc), facts=facts + ["no ID presented before"]))
    return results


def _decision_time(ep: Episode, b: Batch) -> float | None:
    """Storage time, else the client's exit if it came after the handover (an earlier exit was before re-entry)."""
    if b.stored_t is not None:
        return b.stored_t
    return ep.exit_t if ep.exit_t is not None and ep.exit_t >= b.received_t else None


def _per_batch(ft: FactTimeline, episodes: list[Episode], rule_id: str, kind: str | None,
               needs: tuple[str, ...], done_t, what: str) -> list[RuleResult]:
    pairs = [(ep, b) for ep in episodes for b in ep.batches if kind is None or b.kind == kind]
    if not pairs:
        return [RuleResult(rule_id, "project", "not_applicable")]
    results = []
    for ep, b in pairs:
        td = _decision_time(ep, b)
        base = {"episode": ep.client, "batch": b.id}
        if b.returned_t is not None and (td is None or b.returned_t < td):
            results.append(RuleResult(rule_id, "project", "not_applicable", **base))
        elif td is None:
            results.append(RuleResult(rule_id, "project", "unknown", **base,
                                      unknown_reason="batch not finished in clip"))
        elif any(n not in ft.observable for n in needs):
            results.append(RuleResult(rule_id, "project", "unknown", **base,
                                      unknown_reason=f"{what} not observable"))
        else:
            t = done_t(b)
            ok = t is not None and t < td
            facts = [f"batch {b.id} decided at {td} s", f"{what} at {t} s" if t is not None else f"no {what}"]
            results.append(RuleResult(rule_id, "project", "pass" if ok else "violation", timecode_sec=td,
                                      evidence_sec=_evidence(td), facts=facts, **base))
    return results


def check_counted(ft: FactTimeline, episodes: list[Episode]) -> list[RuleResult]:
    """P02: every batch is counted on the machine before it is stored."""
    return _per_batch(ft, episodes, "P02", None, ("machine_count",), lambda b: b.counted_t, "machine count")


def check_bag_opened(ft: FactTimeline, episodes: list[Episode]) -> list[RuleResult]:
    """P03: every bag is opened before its contents are stored."""
    return _per_batch(ft, episodes, "P03", "bag", ("bag_opened",), lambda b: b.opened_t, "bag opening")
