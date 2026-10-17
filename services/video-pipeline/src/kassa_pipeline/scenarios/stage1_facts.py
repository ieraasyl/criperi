"""Build a FactTimeline from a Stage 1 result dict (frame records, layouts, cashier track)."""

import math
from collections import Counter, defaultdict
from itertools import groupby

from .. import geometry as g
from ..facts import UNKNOWN, FactTimeline, clip_id

DOOR_PAD = 0.05  # same padding as people.zone_of's doorway test
MIN_SIDE_SEC = 1.0  # a door run this long (or ending the track) after inside is an exit
MIN_ENTRY_DOOR_OBS = 2  # a brisk walk-in at 4 fps leaves only 2-4 door observations
MIN_INSIDE_OBS = 2  # inside observations within one second for a track to count as a person there
OCCLUSION_SHARE = 0.5
SAFE_IDLE_SEC = 5  # door open this long with the cashier seated and never near the safe: access over
SEATED = ("cashier_seat", "at_desk")
CLOSE_GAP, MIN_RUN = 2, 2
PERSONAL_CATS = ("personal", "food_drink", "phone")
CASH_FLOOR, ITEM_FLOOR = 0.08, 0.25
SCREEN = {"work_screen": "work", "locked": "locked", "off": "off"}
OBSERVABLE = ["room_entry", "room_exit", "safe_access_start", "safe_access_end"]


def clean_bool(series: list[bool], close_gap: int, min_run: int) -> list[bool]:
    """Fill short gaps, then drop short runs. Detections flicker; real states last seconds."""
    out = list(series)
    n = len(out)
    i = 0
    while i < n:  # close gaps no longer than close_gap that sit between true runs
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


def _layout_at(layouts: list[dict], t: float) -> dict:
    cur = layouts[0]
    for lay in layouts:
        if t + 1e-6 >= lay["start"]:
            cur = lay
    return cur


def _box(b) -> g.Box | None:
    return tuple(float(x) for x in b) if b else None


def _by_second(frames: list[dict]) -> dict[int, list[dict]]:
    secs: dict[int, list[dict]] = defaultdict(list)
    for f in frames:
        secs[int(f["t"])].append(f)
    return secs


def _majority(values: list[str]) -> str | None:
    """Most common value; None on a tie or no values."""
    top = Counter(values).most_common(2)
    if not top or (len(top) > 1 and top[0][1] == top[1][1]):
        return None
    return top[0][0]


def _desk_series(secs, layouts, n, cats, floor) -> list[str]:
    raw = [
        any(o["cls"] in cats and o["on_desk"] and o.get("held_by") is None and o["score"] >= floor
            for f in secs.get(s, []) for o in f["objects"])
        for s in range(n)
    ]
    out = []
    for s, on in enumerate(clean_bool(raw, CLOSE_GAP, MIN_RUN)):
        if s not in secs:
            out.append(UNKNOWN)
        elif on:
            out.append("yes")
        else:
            desk = _box(_layout_at(layouts, s + 0.5).get("desk"))
            covered = desk is not None and any(
                g.inside_share(desk, tuple(p["box"])) > OCCLUSION_SHARE for f in secs[s] for p in f["people"]
            )
            out.append(UNKNOWN if desk is None or covered else "no")
    return out


def _in_door(box, layout: dict) -> bool:
    foot = ((box[0] + box[2]) / 2, box[3])
    return any(g.contains(g.expand(tuple(d), DOOR_PAD), foot) for d in (layout.get("doors") or [])[:2])


def _tracks(frames, layouts) -> dict[int, list[tuple[float, str]]]:
    """Per track id: (t, 'door'|'inside') sorted by time."""
    obs: dict[int, list[tuple[float, str]]] = defaultdict(list)
    for f in frames:
        lay = _layout_at(layouts, f["t"])
        for p in f["people"]:
            obs[p["track"]].append((f["t"], "door" if _in_door(p["box"], lay) else "inside"))
    return {tid: sorted(v) for tid, v in obs.items()}


def _crossings(tracks, cashier_id) -> list[dict]:
    clients = sorted((tid for tid in tracks if tid != cashier_id), key=lambda tid: tracks[tid][0][0])
    names = {tid: f"client-{k}" for k, tid in enumerate(clients, 1)}
    events = []
    for tid, obs in tracks.items():
        who = "cashier" if tid == cashier_id else names[tid]
        runs = []
        for side, grp in groupby(obs, key=lambda o: o[1]):
            ts = [t for t, _ in grp]
            runs.append((side, ts[0], ts[-1], len(ts)))
        for i, (side, t0, t1, n) in enumerate(runs):
            prev = runs[i - 1] if i else None
            if side == "door":
                if prev is None or prev[0] != "inside":
                    continue
                if i == len(runs) - 1 or t1 - t0 >= MIN_SIDE_SEC:
                    events.append(("room_exit", t1, who))
            elif prev and prev[0] == "door" and prev[3] >= MIN_ENTRY_DOOR_OBS:
                # jitter (inside, brief door, inside) emitted no exit, so it gets no entry either
                if i == 1 or prev[2] - prev[1] >= MIN_SIDE_SEC:
                    events.append(("room_entry", t0, who))
    events.sort(key=lambda e: e[1])
    return [{"type": ty, "t": round(t, 2), "who": w, "origin": "cv"} for ty, t, w in events]


def _people_inside(tracks, secs, n) -> list[int | None]:
    """Tracks with >= MIN_INSIDE_OBS inside observations in [s, s+1).

    An unsampled second (e.g. a trailing partial one) borrows the adjacent sampled second's count,
    previous first, else next; None only when neither neighbour was sampled.
    """
    counts = [
        sum(sum(s <= t < s + 1 and side == "inside" for t, side in obs) >= MIN_INSIDE_OBS
            for obs in tracks.values()) if s in secs else None
        for s in range(n)
    ]
    out = list(counts)
    for s in range(n):
        if counts[s] is None:
            prev = counts[s - 1] if s > 0 else None
            nxt = counts[s + 1] if s + 1 < n else None
            out[s] = prev if prev is not None else nxt
    return out


def _cashier_zones(secs, n) -> list[str]:
    """Majority zone among records where the cashier is tracked; "absent" if none or tied."""
    out = []
    for s in range(n):
        zones = [z for f in secs.get(s, []) if (z := f["facts"].get("cashier_zone", "absent")) != "absent"]
        out.append(_majority(zones) or "absent")
    return out


def _safe_access(safe_door: list[str], zones: list[str]) -> list[dict]:
    """Safe access start when the door opens; end when it closes or the cashier leaves the safe.

    Fallback when the cashier was never near the safe (safe beside the seat): end once the door has been
    open SAFE_IDLE_SEC seconds in a row and the cashier sat at the desk for the last 3 seconds.
    """
    def zone(s):
        return zones[s] if 0 <= s < len(zones) else "absent"

    events = []
    in_access = left_open = was_near = False  # left_open: access already ended, door still open
    open_run = 0
    for s, door in enumerate(safe_door):
        def ev(kind, note="draft"):
            events.append({"type": kind, "t": float(s), "who": "cashier", "origin": "cv", "note": note})

        open_run = open_run + 1 if door == "open" else 0
        if door == "open":
            if not in_access and not left_open:
                ev("safe_access_start")
                in_access, was_near = True, False
            if in_access:
                near = zone(s) == "near_safe"
                was_near = was_near or near
                if was_near and not near:
                    ev("safe_access_end")
                    in_access, left_open = False, True
                elif (not was_near and open_run >= SAFE_IDLE_SEC
                      and all(zone(k) in SEATED for k in range(s - 2, s + 1))):
                    ev("safe_access_end", "draft-fallback")
                    in_access, left_open = False, True
        elif door == "closed":
            if in_access:
                ev("safe_access_end")
            in_access = left_open = False
    return events


def _vote(secs, n, key, mapping) -> list[str]:
    out = []
    for s in range(n):
        top = _majority([f["facts"].get(key) for f in secs.get(s, [])])
        out.append(mapping.get(top, UNKNOWN))
    return out


def facts_from_stage1(stage1: dict, clip: str) -> FactTimeline:
    duration = float(stage1["meta"]["duration"])
    n = math.ceil(duration)
    layouts = sorted(stage1["layouts"], key=lambda x: x["start"])
    frames = stage1["frames"]
    secs = _by_second(frames)

    def desk(cats, floor):
        return _desk_series(secs, layouts, n, cats, floor)

    parts = [desk((c,), ITEM_FLOOR) for c in PERSONAL_CATS]
    personal = ["yes" if "yes" in v else UNKNOWN if UNKNOWN in v else "no" for v in zip(*parts)]
    tracks = _tracks(frames, layouts)
    safe_door = _vote(secs, n, "safe", {"open": "open", "closed": "closed"})
    states = {
        "desk_cash": desk(("cash",), CASH_FLOOR),
        "desk_bag": desk(("cash_bag",), CASH_FLOOR),
        "desk_personal": personal,
        "desk_docs": [UNKNOWN] * n,
        "screen": _vote(secs, n, "screen", SCREEN),
        "safe_door": safe_door,
        "people_inside": _people_inside(tracks, secs, n),
        "laptop_lid": [UNKNOWN] * n,
    }
    events = _crossings(tracks, stage1.get("cashier_track")) + _safe_access(safe_door, _cashier_zones(secs, n))
    first = layouts[0]
    zones = {z: [float(x) for x in (_box(first.get(key)) or [])]
             for z, key in (("laptop", "laptop"), ("safe", "safe"), ("desk", "desk"))}
    doors = first.get("doors") or []
    zones["door"] = [float(x) for x in doors[0]] if doors else []
    return FactTimeline.from_dict({
        "clip": clip_id(clip), "duration": duration, "status": "draft", "zones": zones,
        "states": states, "state_origin": {k: ["cv"] * n for k in states},
        "events": sorted(events, key=lambda e: e["t"]), "observable": list(OBSERVABLE),
    })
