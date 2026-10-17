"""V18 from Stage 1: an outsider (never reached the desk, not a client's re-ID fragment) standing in the doorway."""

import math
from collections import defaultdict

from ..vlm_scan import Finding
from .stage1_facts import _by_second, _cashier_zones

DESK_ZONES = ("cashier_seat", "at_desk", "near_safe")
REID_GAP_SEC = 3.0  # a track starting/ending this close to a client's end/start is that client's fragment
CASHIER_OUT = ("absent", "doorway")


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
    if not outsiders:
        return []
    n = math.ceil(float(stage1["meta"]["duration"]))
    secs = _by_second(frames)
    cashier = _cashier_zones(secs, n)
    who: dict[int, set[int]] = defaultdict(set)
    for s in range(n):
        if cashier[s] not in CASHIER_OUT:
            who[s] = {p["track"] for f in secs.get(s, []) for p in f["people"]
                      if p["track"] in outsiders and p["zone"] == "doorway"}
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
