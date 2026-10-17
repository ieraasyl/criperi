"""Stage 5: candidate violations from timeline rules. Pure functions of the timeline.

"Cashier left the desk" alone is never a violation: four no-violation training clips do exactly
that. Every rule combines the leaving with a state (cash on desk, screen unlocked, client present).
"""

from dataclasses import dataclass, field

from kassa_pipeline.config import EventsConfig
from kassa_pipeline.timeline import Second, Timeline

AWAY_FROM_ROOM = ("doorway", "elsewhere", "absent")

# type id -> rule id in rules_kb.json
RULE_OF = {
    "T1": "R-CASH-LOCKED",
    "T2": "R-PERSONAL-ITEMS",
    "T3": "R-PERSONAL-ITEMS",
    "T4": "R-ENTRY-ISOLATION",
    "T5": "R-IS-LOCK",
    "T6": "R-LEAVE",
    "T7": "R-CASH-LOCKED",
    "T9": "R-PERSONAL-ITEMS",
}

TYPE_TITLE = {
    "T1": "Наличные оставлены на рабочем столе",
    "T2": "Личные вещи на рабочем месте кассира",
    "T3": "Еда или напитки на рабочем месте кассира",
    "T4": "Посторонний в рабочей зоне кассира",
    "T5": "Рабочий компьютер не заблокирован при уходе кассира",
    "T6": "Кассир покинул рабочее место при клиенте",
    "T7": "Сейф открыт, кассир отсутствует",
    "T9": "Телефон на рабочем месте кассира",
}


@dataclass
class Candidate:
    type: str
    t_start: float
    t_end: float
    timecode: float  # first second at which the state becomes violating
    score: float  # detector-side confidence, 0..1
    facts: dict = field(default_factory=dict)  # what the rule saw: feeds the description


def _runs(flags: list[bool]) -> list[tuple[int, int]]:
    """(start, end) of consecutive True values, end inclusive."""
    out, i, n = [], 0, len(flags)
    while i < n:
        if flags[i]:
            j = i
            while j < n and flags[j]:
                j += 1
            out.append((i, j - 1))
            i = j
        else:
            i += 1
    return out


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def _away(sec: Second) -> bool:
    return not sec.cashier_at_workplace


def find_candidates(tl: Timeline, cfg: EventsConfig) -> list[Candidate]:
    secs = tl.seconds
    n = len(secs)
    out: list[Candidate] = []

    # T1: cash on the desk with nobody dealing with it
    cash_away = [s.cash_on_desk and _away(s) for s in secs]
    for a, b in _runs(cash_away):
        dur = b - a + 1
        if dur >= cfg.cash_away_sec:
            conf = max(s.cash_score for s in secs[a : b + 1])
            # cash is detected a little after the cashier gets up: date the event to the getting up
            left = a
            while left > 0 and _away(secs[left - 1]):
                left -= 1
            start = left if a - left <= cfg.cash_lag_sec else a
            out.append(
                Candidate(
                    "T1", start, b + 1, start,
                    _clamp(0.45 + 0.05 * min(dur, 6) + 0.2 * min(1.0, conf / 0.3)),
                    {"mode": "cashier_away", "seconds": dur, "cashier_zone": secs[a].cashier_zone, "cash_seen_from": a},
                )
            )
    for s in range(1, n):
        if secs[s].arrivals and (secs[s].cash_on_desk or secs[s - 1].cash_on_desk):
            before = secs[max(0, s - 3) : s]
            if not any(x.client_at_desk for x in before):
                out.append(
                    Candidate(
                        "T1", s, s + 1, s,
                        _clamp(0.55 + 0.2 * min(1.0, secs[s].cash_score / 0.3)),
                        {"mode": "new_visitor_arrives_with_cash_on_desk"},
                    )
                )
    idle = [s.cash_on_desk and s.cashier_at_workplace and not s.client_at_desk for s in secs]
    for a, b in _runs(idle):
        if b - a + 1 >= cfg.cash_no_client_sec:
            out.append(
                Candidate("T1", a, b + 1, a, 0.4, {"mode": "cash_on_desk_without_client", "seconds": b - a + 1})
            )

    # T2 / T3 / T9: items on the desk
    for key, typ, need, base in (
        ("personal_on_desk", "T2", cfg.personal_item_sec, 0.55),
        ("food_on_desk", "T3", cfg.food_sec, 0.45),
        ("phone_on_desk", "T9", cfg.personal_item_sec, 0.4),
    ):
        for a, b in _runs([getattr(s, key) for s in secs]):
            dur = b - a + 1
            if dur >= need:
                out.append(
                    Candidate(typ, a, b + 1, a, _clamp(base + 0.04 * min(dur - need, 5)), {"seconds": dur})
                )

    # T4: someone other than the cashier right beside the cashier's seat
    near = [
        any(v.seat_dist is not None and v.seat_dist < cfg.outsider_seat_dist for v in s.visitors)
        for s in secs
    ]
    for a, b in _runs(near):
        dur = b - a + 1
        if dur >= cfg.outsider_sec:
            out.append(Candidate("T4", a, b + 1, a, _clamp(0.45 + 0.05 * min(dur, 6)), {"seconds": dur}))

    # T5: cashier leaves the desk while the work screen stays on
    for a, b in _runs([_away(s) for s in secs]):
        dur = b - a + 1
        if dur < cfg.unlocked_after_leave_sec:
            continue
        window = secs[a : a + cfg.unlocked_after_leave_sec]
        before_ok = secs[max(0, a - 1)].screen == "work_screen" or secs[a].screen == "work_screen"
        if before_ok and all(s.screen == "work_screen" for s in window):
            out.append(
                Candidate("T5", a, b + 1, a, 0.6, {"seconds_away": dur, "cashier_zone": secs[a].cashier_zone})
            )

    # T6: cashier leaves the room while a client sits at the desk
    left = [s.cashier_zone in AWAY_FROM_ROOM and s.client_at_desk for s in secs]
    for a, b in _runs(left):
        if b - a + 1 >= cfg.leave_during_service_sec:
            out.append(Candidate("T6", a, b + 1, a, 0.7, {"seconds": b - a + 1}))

    # T7: safe door open while the cashier is out of the room
    safe_open = [s.safe == "open" and s.cashier_zone in AWAY_FROM_ROOM for s in secs]
    for a, b in _runs(safe_open):
        if b - a + 1 >= cfg.safe_open_away_sec:
            out.append(Candidate("T7", a, b + 1, a, 0.5, {"seconds": b - a + 1}))

    return merge(out, cfg.merge_gap_sec)


def merge(cands: list[Candidate], gap: float) -> list[Candidate]:
    """Same-type candidates closer than `gap` seconds become one: earliest time, best score."""
    merged: list[Candidate] = []
    for c in sorted(cands, key=lambda c: (c.type, c.timecode)):
        last = next((m for m in reversed(merged) if m.type == c.type), None)
        if last is not None and c.timecode - last.t_end <= gap:
            last.t_end = max(last.t_end, c.t_end)
            if c.score > last.score:
                last.score, last.facts = c.score, c.facts
        else:
            merged.append(c)
    return sorted(merged, key=lambda c: c.timecode)
