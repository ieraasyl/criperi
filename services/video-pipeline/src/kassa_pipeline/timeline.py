"""Stage 4: one smoothed state record per second, merged from people and object stages."""

from collections import Counter
from dataclasses import asdict, dataclass, field

import numpy as np

from kassa_pipeline import geometry as g
from kassa_pipeline.config import TimelineConfig
from kassa_pipeline.layout import LayoutTimeline
from kassa_pipeline.objects import ObjectsResult
from kassa_pipeline.people import PeopleResult, zone_of
from kassa_pipeline.video import Video

AT_WORKPLACE = ("cashier_seat", "at_desk")


@dataclass
class Visitor:
    track: int
    zone: str
    seat_dist: float | None


@dataclass
class Second:
    t: int
    cashier_zone: str  # cashier_seat | at_desk | near_safe | doorway | elsewhere | absent
    cashier_at_workplace: bool
    visitors: list[Visitor] = field(default_factory=list)
    arrivals: list[int] = field(default_factory=list)  # visitor tracks first seen this second
    cash_on_desk: bool = False
    cash_score: float = 0.0
    personal_on_desk: bool = False
    food_on_desk: bool = False
    phone_on_desk: bool = False
    screen: str = "unknown"
    safe: str = "unknown"
    lighting_event: bool = False

    @property
    def client_at_desk(self) -> bool:
        return any(v.zone in ("at_desk", "cashier_seat") for v in self.visitors)


@dataclass
class Timeline:
    seconds: list[Second]
    duration: float

    def to_dict(self) -> list[dict]:
        return [asdict(s) for s in self.seconds]


def clean_bool(series: list[bool], close_gap: int, min_run: int) -> list[bool]:
    """Fill short gaps, then drop short runs. Detections flicker; real states last seconds."""
    out = list(series)
    n = len(out)
    i = 0
    while i < n:  # close gaps shorter than or equal to close_gap that sit between true runs
        if not out[i]:
            j = i
            while j < n and not out[j]:
                j += 1
            if i > 0 and j < n and (j - i) <= close_gap:
                for k in range(i, j):
                    out[k] = True
            i = j
        else:
            i += 1
    i = 0
    while i < n:
        if out[i]:
            j = i
            while j < n and out[j]:
                j += 1
            if (j - i) < min_run:
                for k in range(i, j):
                    out[k] = False
            i = j
        else:
            i += 1
    return out


def _mode(values: list[str], skip: str = "unknown") -> str:
    vals = [v for v in values if v != skip]
    return Counter(vals).most_common(1)[0][0] if vals else skip


def build_timeline(
    video: Video,
    layouts: LayoutTimeline,
    people: PeopleResult,
    objects: ObjectsResult,
    cfg: TimelineConfig,
    min_cash_score: float = 0.08,
    min_item_score: float = 0.25,
) -> Timeline:
    n = int(np.ceil(video.duration))
    cashier = people.cashier()
    seconds: list[Second] = []

    # arrival second per visitor track = first sighting
    first_seen = {tr.id: tr.t0 for tr in people.tracks if tr.role != "cashier"}

    for s in range(n):
        zone, at_wp = "absent", False
        if cashier is not None:
            o = cashier.at(s + 0.5, tol=0.6)
            if o:
                zone = zone_of(o.t, o.box, layouts, people).name
                at_wp = zone in AT_WORKPLACE
        sec = Second(t=s, cashier_zone=zone, cashier_at_workplace=at_wp)
        for tr in people.tracks:
            if tr.role == "cashier":
                continue
            o = tr.at(s + 0.5, tol=0.6)
            if o:
                z = zone_of(o.t, o.box, layouts, people)
                sec.visitors.append(Visitor(tr.id, z.name, z.seat_dist))
            if int(first_seen[tr.id]) == s:
                sec.arrivals.append(tr.id)
        seconds.append(sec)

    # object flags, raw then cleaned
    def raw(cats: tuple[str, ...], thr: float) -> tuple[list[bool], list[float]]:
        flags, scores = [], []
        for s in range(n):
            hits = [
                o
                for o in objects.objects.get(s, [])
                if o.cat in cats and o.on_desk and not o.in_hand and o.score >= thr
            ]
            flags.append(bool(hits))
            scores.append(max((o.score for o in hits), default=0.0))
        return flags, scores

    cash, cash_sc = raw(("cash", "cash_bag"), min_cash_score)
    pers, _ = raw(("personal",), min_item_score)
    food, _ = raw(("food_drink",), min_item_score)
    phone, _ = raw(("phone",), min_item_score)
    cash = clean_bool(cash, cfg.min_off_sec, cfg.min_on_sec)
    pers = clean_bool(pers, cfg.min_off_sec, cfg.min_on_sec)
    food = clean_bool(food, cfg.min_off_sec, cfg.min_on_sec)
    phone = clean_bool(phone, cfg.min_off_sec, cfg.min_on_sec)

    screens = [objects.screen.get(s).state if s in objects.screen else "unknown" for s in range(n)]
    safes = [objects.safe.get(s).state if s in objects.safe else "unknown" for s in range(n)]

    luma = {int(f.t): f.luma for f in video.frames}
    for s, sec in enumerate(seconds):
        sec.cash_on_desk, sec.cash_score = cash[s], cash_sc[s]
        sec.personal_on_desk, sec.food_on_desk, sec.phone_on_desk = pers[s], food[s], phone[s]
        win = slice(max(0, s - 1), min(n, s + 2))
        sec.screen = _mode(screens[win])
        sec.safe = _mode(safes[win])
        if s in luma and s - 1 in luma:
            sec.lighting_event = abs(luma[s] - luma[s - 1]) > cfg.lighting_jump
    return Timeline(seconds, video.duration)
