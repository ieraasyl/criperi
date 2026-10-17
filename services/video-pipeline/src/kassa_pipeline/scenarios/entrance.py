"""V18 from Stage 1: an outsider (never reached the desk, not a client's re-ID fragment) standing in the doorway."""

import math
import statistics
from collections import defaultdict

from ..vlm_scan import Finding
from .stage1_facts import _by_second, _cashier_zones

DESK_ZONES = ("cashier_seat", "at_desk", "near_safe")
REID_GAP_SEC = 3.0  # a track starting/ending this close to a client's end/start is that client's fragment
CASHIER_OUT = ("absent", "doorway")
MIN_HEIGHT = 1.0  # doorway box at least this many seated-cashier heights tall: in the opening, not behind the glass


def _seated_height(frames: list[dict]) -> float | None:
    """Median cashier box height while seated (fallback: any cashier record); None without cashier records."""
    rec = [(p["zone"], p["box"][3] - p["box"][1]) for f in frames for p in f["people"] if p["role"] == "cashier"]
    hs = [h for z, h in rec if z == "cashier_seat"] or [h for _, h in rec]
    return statistics.median(hs) if hs else None


def _outsiders(frames: list[dict]) -> set[int]:
    span: dict[int, list[float]] = {}
    desk: set[int] = set()
    for f in frames:
        for p in f["people"]:
            if p["role"] != "visitor":
                continue
            tid, t = p["track"], f["t"]
            lo, hi = span.get(tid, (t, t))
            span[tid] = [min(lo, t), max(hi, t)]
            if p["zone"] in DESK_ZONES:
                desk.add(tid)
    return {tid for tid, (t0, t1) in span.items() if tid not in desk
            and not any(abs(t0 - span[c][1]) <= REID_GAP_SEC or abs(t1 - span[c][0]) <= REID_GAP_SEC for c in desk)}


def entrance_findings(stage1: dict, sc) -> list[Finding]:
    """One V18 per run of >= sc.entrance_sec consecutive seconds with an outsider in the doorway and the cashier in the room."""
    frames = stage1["frames"]
    outsiders = _outsiders(frames)
    seat = _seated_height(frames)
    if not outsiders or not seat or seat <= 0:
        return []  # no cashier height to compare against: fail closed
    n = math.ceil(float(stage1["meta"]["duration"]))
    secs = _by_second(frames)
    cashier = _cashier_zones(secs, n)
    who: dict[int, set[int]] = defaultdict(set)
    for s in range(n):
        if cashier[s] not in CASHIER_OUT:
            who[s] = {p["track"] for f in secs.get(s, []) for p in f["people"]
                      if p["track"] in outsiders and p["zone"] == "doorway"
                      and p["box"][3] - p["box"][1] >= MIN_HEIGHT * seat}
    out, s = [], 0
    while s < n:
        e = s
        while e < n and who[e]:
            e += 1
        if e - s >= sc.entrance_sec:
            ids = sorted(set().union(*(who[k] for k in range(s, e))))
            out.append(Finding("V18", float(s), sc.scenario_confidence, "scenario entrance-outsider: outsider standing in the doorway",
                               [f"track {', '.join(map(str, ids))} in the doorway {s}-{e}s"]))
        s = max(e, s + 1)
    return out
