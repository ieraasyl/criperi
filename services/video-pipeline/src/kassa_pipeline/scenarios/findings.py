"""Scenario rule violations as VLM-scan findings (merged with the VLM ones by vlm_scan._merge)."""

from collections.abc import Iterable

from ..vlm_scan import Finding
from .base import RuleResult

CODES = {"W03": "V14", "W01": "V12", "W05": "V02", "W06": "V20", "P01": "V04", "P02": "V05", "P03": "V05"}
W02_CODES = {"desk_cash": "V01", "desk_bag": "V01", "desk_personal": "V15"}  # desk_docs: no finding


def _code(r: RuleResult) -> str | None:
    return W02_CODES.get(r.subject or "") if r.rule_id == "W02" else CODES.get(r.rule_id)


def scenario_findings(results: Iterable[RuleResult], enabled: Iterable[str], confidence: float) -> list[Finding]:
    """One finding per violation result of an enabled rule; pass, unknown and not_applicable give none."""
    on = set(enabled)
    out = []
    for r in results:
        code = _code(r)
        if r.status == "violation" and r.rule_id in on and code and r.timecode_sec is not None:
            out.append(Finding(code, float(r.timecode_sec), confidence, f"scenario {r.rule_id}", list(r.facts)))
    return out
