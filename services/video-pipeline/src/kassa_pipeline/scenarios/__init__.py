"""Scenario engine: rules over a FactTimeline."""

from ..facts import FactTimeline
from .base import RuleResult
from .changeover_rules import check_people_inside, check_previous_cash
from .episodes import build_episodes
from .exit_rules import check_exit_items, check_exit_laptop
from .safe_rules import check_safe
from .service_rules import check_counted, check_id_first


def run_scenarios(ft: FactTimeline) -> list[RuleResult]:
    """Run every first-wave check (W02, W03, W01, W05, W06, P01, P02). P03 (open every ordinary bag) is retired:
    no law backs it, and only law-backed rules may raise a violation beyond the project assumptions kept."""
    episodes = build_episodes(ft)
    return [
        *check_exit_items(ft),
        *check_exit_laptop(ft),
        *check_safe(ft),
        *check_previous_cash(ft, episodes),
        *check_people_inside(ft),
        *check_id_first(ft, episodes),
        *check_counted(ft, episodes),
    ]
