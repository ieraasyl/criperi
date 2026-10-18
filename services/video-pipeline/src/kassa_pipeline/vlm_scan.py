"""Sliding-window VLM scan: the VLM logs what it sees, code decides which case class (V01-V21) applies.

The classes and their conditions are the case taxonomy (violation_taxonomy.md, data/violation_catalog.json).
1. Fixed windows (8 s, stride 4 s), 16 full frames + 8 zoomed desk views with the Stage 1 track boxes and
   the time on every image: events (document shown, money taken, counted, put in the safe, cashier left
   the room...) and states (money on the open desk, safe open, key visible, personal items, clothing,
   food, visitor on the cashier's side, people in the room).
2. Session windows (one per client visit): the order of the deposit procedure (V02-V11, V21). A step the
   VLM could not see is "not_visible", never a violation.
3. Door: zoomed doorway views at 1 fps; open for `door_open_sec` = V18.
4. Safe: zoomed safe views at 1 fps decide V12 (the window state is the fallback).
5. Laptop: the Stage 1 pixel classifier after the cashier left the room = V14.
A state needs `min_agree` overlapping windows; duplicates follow the taxonomy's priorities. The VLM never
names a class or a norm.
"""

import bisect
import math
import json
import time
from functools import lru_cache
from importlib import resources
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from kassa_pipeline import geometry as g
from kassa_pipeline import video as V
from kassa_pipeline.config import PipelineConfig, ScanConfig
from kassa_pipeline.vlm import parse_json

# Observations the VLM reports. They follow the case classes in violation_taxonomy.md (V01-V21); the
# VLM never names a class, the rules below do.
STATES = [
    "cashier_in_room", "service_in_progress", "cash_on_open_desk", "cash_bag_on_open_desk", "safe_door_open",
    "key_left_visible", "personal_items_on_desk", "clothing_on_desk_or_in_safe", "food_or_drink_on_desk",
    "visitor_on_cashier_side",
]
EVENTS = [
    "client_entered_room", "client_left_room", "document_shown", "money_taken_by_cashier", "bag_taken_by_cashier",
    "counted_on_big_counter", "rejected_notes_separated", "checked_on_small_machine", "batches_combined",
    "typing_or_printing_on_laptop", "money_put_in_safe", "safe_opened", "safe_closed", "key_taken_by_cashier",
    "cashier_left_room", "cashier_returned", "eating_or_drinking",
]

WINDOW_PROMPT = """These are {n} frames from a top-down surveillance camera in a bank cashier's room, from {t0:.1f} s to {t1:.1f} s, followed by {m} zoomed views of the cashier's desk at the same moments. Every image shows its time (t=...) in the top-left corner. Boxes mark people found by a tracker: "#N cashier" is the bank cashier, "#N visitor" is anyone else (usually a client). People seen through the glass wall, reflections and people outside the room do not count.
Scene: {scene}
Report only what is clearly visible. If you cannot tell, answer "not_visible". Do not guess; an action hidden by a hand or a body is "not_visible".

Reply with ONLY this JSON, no other text:
{{"events": [{{"t": <seconds>, "who": "#N", "what": "<event>", "note": "<at most 8 words>"}}],
 "people_in_room": <number of real people inside the room in the last frames, or null>,
 "state": {{{state_fields}}}}}
Allowed events, at most 8, in time order: {events}.
Event meanings: document_shown = the client shows or hands over an ID document, physical or on a phone screen; counted_on_big_counter = banknotes go through the big black counting machine; rejected_notes_separated = the cashier puts aside notes the machine did not accept; checked_on_small_machine = notes are checked on the small low machine; batches_combined = two separate piles of money are put together; typing_or_printing_on_laptop = the cashier types on the laptop or prints; key_taken_by_cashier = the cashier takes the safe key with her.
Each state is [answer, since]: answer is "yes", "no" or "not_visible" for the LAST frames of the window; since is the first second it was "yes" in this window, or null.
State meanings: cashier_in_room = the cashier is inside the room (at the desk, at the safe or elsewhere inside); service_in_progress = a client is being served at the desk; cash_on_open_desk = banknotes lie on the open desk and nobody is holding or counting them; cash_bag_on_open_desk = a money bag lies on the open desk; safe_door_open = the safe door stands open; key_left_visible = a key sticks out of the safe lock or lies on the desk; personal_items_on_desk = a personal book, notebook, handbag, backpack or other non-work item lies on the desk (work papers, the cash bag, clothing and food do not count); clothing_on_desk_or_in_safe = a coat, jacket or other clothing lies on the desk or is put in the safe; food_or_drink_on_desk = food or a drink is on the desk, is being eaten or drunk at the desk, or is put in the safe; visitor_on_cashier_side = a visitor has crossed to the cashier's side of the desk, next to her seat or equipment."""

SESSION_PROMPT = """These are {n} frames from a top-down surveillance camera in a bank cashier's room covering one client visit, from {t0:.1f} s to {t1:.1f} s, followed by {m} zoomed views of the cashier's desk. Every image shows its time (t=...) in the top-left corner. Boxes mark people found by a tracker: "#N cashier" is the bank cashier, "#N visitor" is anyone else. The visitor(s) in this visit: {visitors}.
Scene: {scene}
The normal cash deposit: the client shows an ID document (physical or on a phone; handing it over from the hand is fine) -> the cashier takes the money -> counts it on the big black counting machine while the client is present -> if the machine rejected some notes, those are put aside, checked on the small low machine and recounted before being added to the rest -> the cashier types on the laptop and puts the money into the safe (in either order) -> closes the safe -> the service ends. A closed money bag is not an open-cash deposit.
Answer about THIS visit only, from what is visible. If you cannot tell, answer "not_visible". Do not guess: a step you did not see because it was hidden, out of frame or between frames is "not_visible", not "no".

Reply with ONLY this JSON, no other text:
{{"start_fully_visible": "yes|no", "document_shown": "yes|no|not_visible", "document_t": <seconds or null>,
 "money_taken": "yes|no|not_visible", "money_t": <seconds or null>, "open_cash": "yes|no|not_visible",
 "counted_on_big_counter": "yes|no|not_visible", "count_t": <seconds or null>, "client_present_during_count": "yes|no|not_visible",
 "rejected_part_separated": "yes|no|not_visible", "rejected_checked_on_small_machine": "yes|no|not_visible",
 "rejected_part_recounted": "yes|no|not_visible", "rejected_combined_before_checks": "yes|no|not_visible",
 "typed_or_printed_on_laptop": "yes|no|not_visible", "money_put_away": "yes|no|not_visible", "put_away_t": <seconds or null>,
 "operation_finished": "yes|no|not_visible", "finished_t": <seconds or null>,
 "money_left_on_open_desk_after_finish": "yes|no|not_visible",
 "previous_batch_on_desk_when_client_entered": "yes|no|not_visible",
 "batches_of_different_clients_combined": "yes|no|not_visible",
 "cashier_left_room_before_service_finished": "yes|no|not_visible", "left_t": <seconds or null>,
 "note": "<at most 25 words on what happened>"}}"""

DOOR_PROMPT = """These are {n} zoomed views of the doorway area of a bank cashier's room (a glass wall with a glass door), one per second from {t0:.1f} s to {t1:.1f} s. Every image shows its time (t=...) in the top-left corner.
For EACH image: is the door open (the door leaf is swung away from the glass wall and the doorway is a gap) or closed (the door leaf lies flat in line with the glass wall)? Is a person standing in or passing through the doorway?
If you cannot tell, answer "not_visible". Do not guess.
Reply with ONLY this JSON, no other text:
{{"frames": [{{"t": <seconds>, "door": "open|closed|not_visible", "person_in_doorway": "yes|no"}}]}}"""
SAFE_PROMPT = """These are {n} zoomed views of the safe area of a bank cashier's room (a grey metal cabinet with a door), one per second from {t0:.1f} s to {t1:.1f} s. Every image shows its time (t=...) in the top-left corner.
For EACH image: is the safe door open (the door leaf is swung away from the cabinet and the inside of the safe is visible) or closed (the door leaf covers the front of the cabinet)?
If you cannot tell, answer "not_visible". Do not guess.
Reply with ONLY this JSON, no other text:
{{"frames": [{{"t": <seconds>, "open": "yes|no|not_visible"}}]}}"""
SESSION_TIMES = ("document_t", "money_t", "count_t", "put_away_t", "finished_t", "left_t")


@lru_cache(maxsize=None)
def catalog() -> dict[str, dict]:
    data = json.loads(resources.files("kassa_pipeline.data").joinpath("violation_catalog.json").read_text("utf8"))
    return {c["code"]: c for c in data["codes"]}


@dataclass
class Finding:
    code: str  # case class V01-V21 (violation_catalog.json)
    t: float
    confidence: float
    source: str  # which check produced it
    evidence: list[str] = field(default_factory=list)

    @property
    def rule(self) -> str:  # the verified norm in rules_kb.json, "" when the class has no open norm
        return catalog()[self.code]["rule"] or ""


# ---------------------------------------------------------------- Stage 1 helpers


class Stage1:
    def __init__(self, result: dict):
        self.r = result
        self.frames = result["frames"]
        self.times = [f["t"] for f in self.frames]
        self.layouts = sorted(result["layouts"], key=lambda x: x["start"])

    def record(self, t: float) -> dict | None:
        if not self.frames:
            return None
        i = bisect.bisect_left(self.times, t)
        cands = [j for j in (i - 1, i) if 0 <= j < len(self.frames)]
        return self.frames[min(cands, key=lambda j: abs(self.times[j] - t))]

    def layout(self, t: float) -> dict:
        cur = self.layouts[0]
        for lay in self.layouts:
            if t + 1e-6 >= lay["start"]:
                cur = lay
        return cur

    def visitors_present(self, t0: float, t1: float) -> bool:
        return any(p["role"] == "visitor" for f in self.frames if t0 <= f["t"] <= t1 for p in f["people"])

    def sessions(self, duration: float, pad: float) -> list[tuple[float, float, list[int]]]:
        """Client visits: spans of visitor tracks, merged when they overlap."""
        spans = sorted((p["t0"], p["t1"], p["id"]) for p in self.r["people"] if p["role"] == "visitor" and p["t1"] - p["t0"] >= 1.0)
        out: list[list] = []
        for a, b, tid in spans:
            a, b = max(0.0, a - pad), min(duration, b + pad)
            if out and a <= out[-1][1]:
                out[-1][1] = max(out[-1][1], b)
                out[-1][2].append(tid)
            else:
                out.append([a, b, [tid]])
        return [(a, b, ids) for a, b, ids in out]


def _event_times(wins: list[dict], what: str, duration: float) -> list[float]:
    """Times of one event type. VLM output is untrusted: bools, non-numbers, non-finite values and times
    outside [-1, duration + 1] are dropped; the rest is clipped into [0, duration]."""
    return sorted({min(max(float(e["t"]), 0.0), duration) for w in wins for e in w["events"]
                   if e["what"] == what and isinstance(e["t"], (int, float)) and not isinstance(e["t"], bool)
                   and math.isfinite(e["t"]) and -1.0 <= e["t"] <= duration + 1.0})


def add_event_sessions(visits: list[tuple[float, float, list[int]]], wins: list[dict], duration: float,
                       sc: ScanConfig) -> list[tuple[float, float, list[int]]]:
    """A client the tracker missed still gets a session: from each VLM client_entered_room outside every
    session to the next client_left_room (at most `event_session_max_sec`, else the clip end), padded; a
    money_taken_by_cashier outside every session gets [t-10, t+20]. Event sessions merge only among
    themselves; the Stage 1 sessions stay as they are."""
    if not sc.sessions_from_vlm_events:
        return visits
    exits = _event_times(wins, "client_left_room", duration)
    new: list[list] = []
    covered = lambda t: any(v[0] <= t <= v[1] for v in [*visits, *new])  # noqa: E731
    for t in _event_times(wins, "client_entered_room", duration):
        if not covered(t):
            end = min(next((x for x in exits if x > t), duration) + sc.session_pad_sec, t + sc.event_session_max_sec)
            new.append([max(0.0, t - sc.session_pad_sec), min(duration, end)])
    for t in _event_times(wins, "money_taken_by_cashier", duration):  # no entry logged (X-13)
        if not covered(t):
            new.append([max(0.0, t - 10.0), min(duration, t + 20.0)])
    merged: list[list] = []
    for a, b in sorted(v for v in new if v[0] < v[1]):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return sorted([*visits, *((a, b, []) for a, b in merged)], key=lambda v: v[0])


def visitors_text(ids: list[int]) -> str:
    return ", ".join(f"#{i}" for i in ids) or "the client(s) seen in these frames"


def scene_text(lay: dict) -> str:
    desk = lay.get("desk") or lay.get("workstation")
    if not desk:
        return "a desk with a laptop and a banknote counting machine; a grey metal safe stands near the desk."
    cx = (desk[0] + desk[2]) / 2

    def side(b):
        x = (b[0] + b[2]) / 2
        return "left of" if x < cx - 0.15 * (desk[2] - desk[0]) else "right of" if x > cx + 0.15 * (desk[2] - desk[0]) else "on"

    parts = ["the cashier's desk is in the middle"]
    if lay.get("laptop"):
        parts.append(f"the laptop is {side(lay['laptop'])} the desk")
    if lay.get("counter"):
        parts.append(f"the banknote counting machine is {side(lay['counter'])} the desk")
    if lay.get("safe"):
        parts.append(f"the safe (a grey metal cabinet with a door) is {side(lay['safe'])} the desk")
    if lay.get("doors"):
        d = lay["doors"][0]
        horiz = "left" if (d[0] + d[2]) / 2 < lay["width"] / 2 else "right"
        parts.append(f"the room's door is in the upper {horiz} part of the image")
    return "; ".join(parts) + "."


# ---------------------------------------------------------------- images


def _label(img: np.ndarray, text: str):
    s = max(0.5, img.shape[1] / 700)
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, s, 2)
    cv2.rectangle(img, (0, 0), (w + 10, h + 12), (0, 0, 0), -1)
    cv2.putText(img, text, (5, h + 5), cv2.FONT_HERSHEY_SIMPLEX, s, (255, 255, 255), 2, cv2.LINE_AA)


def annotate(frame: V.Frame, rec: dict | None, long_side: int) -> np.ndarray:
    img = frame.rgb.copy()
    for p in (rec or {}).get("people", []):
        b = [int(v) for v in p["box"]]
        col = (60, 220, 60) if p["role"] == "cashier" else (255, 150, 40)
        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), col, 2)
        txt = f"#{p['track']} {p['role']}"
        cv2.putText(img, txt, (b[0] + 2, max(14, b[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, txt, (b[0] + 2, max(14, b[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)
    h, w = img.shape[:2]
    k = long_side / max(h, w)
    img = cv2.resize(img, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
    _label(img, f"t={frame.t:.1f}s")
    return img


def desk_zoom(frame: V.Frame, lay: dict, width: int) -> np.ndarray | None:
    ws = lay.get("workstation") or lay.get("desk")
    if not ws:
        return None
    h, w = frame.rgb.shape[:2]
    c = g.crop(frame.rgb, g.expand(tuple(ws), 0.12, w, h), min_size=20)
    if c is None:
        return None
    k = width / c.shape[1]
    img = cv2.resize(c, (width, int(c.shape[0] * k)), interpolation=cv2.INTER_CUBIC)
    _label(img, f"t={frame.t:.1f}s desk")
    return img


def door_zoom(frame: V.Frame, lay: dict, width: int) -> np.ndarray | None:
    doors = lay.get("doors") or []
    if not doors:
        return None
    h, w = frame.rgb.shape[:2]
    box = g.expand(g.union([tuple(d) for d in doors[:2]]), 0.05, w, h)
    c = g.crop(frame.rgb, box, min_size=20)
    if c is None:
        return None
    k = width / c.shape[1]
    img = cv2.resize(c, (width, int(c.shape[0] * k)), interpolation=cv2.INTER_CUBIC)
    _label(img, f"t={frame.t:.1f}s door")
    return img


def safe_zoom(frame: V.Frame, lay: dict, width: int) -> np.ndarray | None:
    safe = lay.get("safe")
    if not safe:
        return None
    h, w = frame.rgb.shape[:2]
    c = g.crop(frame.rgb, g.expand(tuple(safe), 0.6, w, h), min_size=20)
    if c is None:
        return None
    k = width / c.shape[1]
    img = cv2.resize(c, (width, int(c.shape[0] * k)), interpolation=cv2.INTER_CUBIC)
    _label(img, f"t={frame.t:.1f}s safe")
    return img


def _frames_at(video: V.Video, times: list[float]) -> list[V.Frame]:
    return [video.nearest(t) for t in times]


def _times(t0: float, t1: float, fps: float, cap: int | None = None) -> list[float]:
    ts = list(np.arange(t0, t1 + 1e-6, 1.0 / fps))
    if cap and len(ts) > cap:
        ts = list(np.linspace(t0, t1, cap))
    return [round(float(t), 2) for t in ts]


# ---------------------------------------------------------------- windows


def fixed_windows(duration: float, sc: ScanConfig) -> list[tuple[float, float]]:
    if duration <= sc.window_sec:
        return [(0.0, max(duration - 0.05, 0.0))]
    starts = list(np.arange(0.0, duration - sc.window_sec + 1e-6, sc.stride_sec))
    if starts[-1] + sc.window_sec < duration - 0.5:
        starts.append(duration - sc.window_sec)
    return [(round(float(s), 2), round(float(min(s + sc.window_sec, duration - 0.05)), 2)) for s in starts]


def _build(video: V.Video, s1: Stage1, t0: float, t1: float, fps: float, cap: int | None, sc: ScanConfig):
    times = _times(t0, t1, fps, cap)
    frames = _frames_at(video, times)
    full = [annotate(f, s1.record(f.t), sc.frame_long_side) for f in frames]
    ctimes = _times(t0, t1, min(sc.crop_fps, fps), (cap // 2) if cap else None)
    crops = [desk_zoom(f, s1.layout(f.t), sc.crop_width) for f in _frames_at(video, ctimes)]
    crops = [c for c in crops if c is not None]
    return full, crops


def _ans(v) -> tuple[str, float | None]:
    if isinstance(v, list) and v:
        a = str(v[0]).lower()
        since = v[1] if len(v) > 1 else None
    else:
        a, since = str(v).lower(), None
    try:
        since = float(since) if since is not None else None
    except (TypeError, ValueError):
        since = None
    return ("yes" if a.startswith("y") else "no" if a.startswith("n") and a != "not_visible" else "not_visible"), since


def _num(v) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- rules over the log


def _runs(wins: list[dict], state: str, min_agree: int) -> list[tuple[float, float, list[dict]]]:
    """Groups of consecutive windows reporting `state` = yes; (first second, last second, windows)."""
    out, cur = [], []
    for w in wins:
        if w["state"].get(state, ("not_visible", None))[0] == "yes":
            cur.append(w)
        else:
            if len(cur) >= min_agree:
                out.append(cur)
            cur = []
    if len(cur) >= min_agree:
        out.append(cur)
    res = []
    for grp in out:
        since = grp[0]["state"][state][1]
        first = since if since is not None and grp[0]["t0"] <= since <= grp[0]["t1"] else grp[0]["t0"] + (grp[0]["t1"] - grp[0]["t0"]) / 2
        res.append((first, grp[-1]["t1"], grp))
    return res


def _event(wins: list[dict], what: str, t0: float, t1: float) -> float | None:
    ts = [e["t"] for w in wins for e in w["events"] if e["what"] == what and e["t"] is not None and t0 - 2 <= e["t"] <= t1 + 2]
    return min(ts) if ts else None


OUT_OF_ROOM = ("absent", "doorway")  # "leaving" in the taxonomy is leaving the room, not walking to the safe
WORK_ZONES = ("cashier_seat", "at_desk", "standing_at_seat")


def _stage1_run(s1: Stage1, t0: float, t1: float, pred, min_sec: float) -> float | None:
    """First second in [t0, t1] where `pred(frame)` holds for at least `min_sec`."""
    start = None
    for f in s1.frames:
        if f["t"] < t0 or f["t"] > t1:
            continue
        if pred(f):
            start = f["t"] if start is None else start
            if f["t"] - start >= min_sec:
                return start
        else:
            start = None
    return None


def _left_room(wins: list[dict], s1: Stage1, grp: list[dict], first: float) -> float | None:
    """Earliest second (not before `first`) when the cashier is out of the room: the VLM says so at the end
    of a window, the VLM logged her leaving, or Stage 1 has her absent or in the doorway for 2 s."""
    for w in grp:
        if w["state"].get("cashier_in_room", ("not_visible", None))[0] == "no":
            return max(first, w["t0"] + (w["t1"] - w["t0"]) / 2)
    t = _event(wins, "cashier_left_room", grp[0]["t0"], grp[-1]["t1"])
    if t is not None:
        return max(first, t)
    return _stage1_run(s1, first, grp[-1]["t1"], lambda f: f["facts"]["cashier_zone"] in OUT_OF_ROOM, 2.0)


def laptop_rule(s1: Stage1, sc: ScanConfig) -> list[Finding]:
    """V14: the cashier leaves the room after being at the workplace, and the Stage 1 pixel classifier sees
    the active work screen in the first seconds of the absence (lock screen, closed lid or dark screen are fine)."""
    F, frames, i, was_at_work = [], s1.frames, 0, False
    while i < len(frames):
        f = frames[i]
        zone = f["facts"]["cashier_zone"]
        if zone in WORK_ZONES:
            was_at_work = True
        if was_at_work and zone in OUT_OF_ROOM:
            j = i
            while j + 1 < len(frames) and frames[j + 1]["facts"]["cashier_zone"] in OUT_OF_ROOM:
                j += 1
            if frames[j]["t"] - f["t"] >= sc.away_min_sec:
                win = [x["facts"]["screen"] for x in frames[i : j + 1] if x["t"] <= f["t"] + sc.away_min_sec]
                known = [x for x in win if x != "unknown"]
                share = sum(x == "work_screen" for x in known) / len(known) if known else 0.0
                if share >= 0.6:
                    F.append(Finding("V14", f["t"], 0.65, "work screen (pixel classifier) after the cashier left the room",
                                     [f"cashier out from {f['t']:.1f}s to {frames[j]['t']:.1f}s", f"work screen in {share:.0%} of frames"]))
            was_at_work = False
            i = j + 1
            continue
        i += 1
    return F


def door_rule(doors: list[dict], sc: ScanConfig, transits: list[float] | None = None) -> list[Finding]:
    """V18: the door stays open for at least `door_open_sec` (people walking through open it only briefly).
    Seconds within `door_transit_sec` of a client entering or leaving (`transits`) are not counted: that
    is the client opening the door, not the door being left open."""
    F, run = [], []
    seq = sorted((d for d in doors if d.get("t") is not None), key=lambda d: d["t"])
    step, cut = 1.0 / sc.door_fps, sorted(transits or []) if sc.door_ignore_transits else []

    def close(run):
        if not run:
            return
        parts, x = [], run[0]["t"]
        end = run[-1]["t"] + step
        for c in cut:
            lo, hi = c - sc.door_transit_sec, c + sc.door_transit_sec
            if hi <= x or lo >= end:
                continue
            if lo > x:
                parts.append((x, lo))
            x = max(x, hi)
        if x < end:
            parts.append((x, end))
        for x, y in parts:
            if y - x >= sc.door_open_sec:
                someone = any(d["person"] == "yes" for d in run if x - 1e-6 <= d["t"] < y)
                F.append(Finding("V18", x, 0.65 if someone else 0.6,
                                 "door left open" + (" with a person in the doorway" if someone else ""),
                                 [f"door open {x:.1f}-{y - step:.1f}s"]))
                return

    for d in seq:
        if d["door"] == "open" and (not run or d["t"] - run[-1]["t"] <= 1.5 / sc.door_fps):
            run.append(d)
        else:
            close(run)
            run = [d] if d["door"] == "open" else []
    close(run)
    return F


TRANSIT_EVENTS = ("client_entered_room", "client_left_room")


def _dedupe(times: list[float]) -> list[float]:
    """The VLM repeats an event in every window that sees it: a time less than 6 s after the previous one
    (chained) joins its cluster, and only the earliest of a cluster is kept."""
    out: list[float] = []
    for i, t in enumerate(times):
        if i == 0 or t - times[i - 1] >= 6.0:
            out.append(t)
    return out


def _window_end(wins: list[dict]) -> float:
    return max((w["t1"] for w in wins), default=0.0)


def _transits(wins: list[dict]) -> list[float]:
    """Seconds when the VLM saw a client walk in or out (unconfirmed), repeats of one type collapsed."""
    end = _window_end(wins)
    return sorted(t for what in TRANSIT_EVENTS for t in _dedupe(_event_times(wins, what, end)))


CLIENT_ZONES = ("cashier_seat", "at_desk", "standing_at_seat", "near_safe")


def _confirmed_transits(wins: list[dict], s1: Stage1, sc: ScanConfig) -> list[float]:
    """VLM transit events that may cut the door run. The VLM also calls a doorway outsider
    "client_entered_room", so each raw event is judged first, by Stage 1 track spans:
    1. a non-cashier track that reaches the desk zones starts or ends within `door_transit_sec`: cut;
    2. else a track that never reaches the desk has a span within `door_transit_sec` of it (an outsider in
       the doorway): no cut;
    3. else no non-cashier track is near it (the client went untracked, or no people records): cut.
    Only then are repeats of one type collapsed, among the events that cut."""
    span: dict[int, list[float]] = {}
    desk: set[int] = set()
    for f in s1.frames:
        for p in f.get("people", []):
            if p["role"] == "cashier":
                continue
            lo, hi = span.get(p["track"], (f["t"], f["t"]))
            span[p["track"]] = [min(lo, f["t"]), max(hi, f["t"])]
            if p["zone"] in CLIENT_ZONES:
                desk.add(p["track"])
    w, end = sc.door_transit_sec, _window_end(wins)

    def cuts(t: float) -> bool:
        if any(abs(t - e) <= w for tid in desk for e in span[tid]):
            return True
        return not any(lo - w <= t <= hi + w for tid, (lo, hi) in span.items() if tid not in desk)

    return sorted(t for what in TRANSIT_EVENTS for t in _dedupe([t for t in _event_times(wins, what, end) if cuts(t)]))


def people_rule(wins: list[dict], s1: Stage1, sc: ScanConfig) -> list[Finding]:
    """V20: three or more real people inside the room. Stage 1 tracks and the VLM count must both say so;
    either alone stays below the threshold."""
    F = []
    vlm = [w for w in wins if isinstance(w.get("people_in_room"), (int, float)) and w["people_in_room"] >= 3]
    t_vlm = vlm[0]["t0"] + (vlm[0]["t1"] - vlm[0]["t0"]) / 2 if len(vlm) >= sc.min_agree else None
    t_s1 = _stage1_run(s1, 0.0, 1e9, lambda f: f["facts"]["people_in_room"] >= 3, 2.0)
    if t_vlm is not None:
        F.append(Finding("V20", t_vlm, 0.45, "VLM counts three or more people", [f"{len(vlm)} windows"]))
    if t_s1 is not None:
        F.append(Finding("V20", t_s1, 0.45, "three or more people tracked", [f"from {t_s1:.1f}s"]))
    return F


def _session_rules(sessions: list[dict], wins: list[dict], s1: Stage1, sc: ScanConfig) -> list[Finding]:
    F = []
    for s_ in sessions:
        a = s_["answer"]
        if not a:
            continue
        yes = lambda k: a.get(k) == "yes"  # noqa: E731
        no = lambda k: a.get(k) == "no"  # noqa: E731
        t = lambda k, d=None: _num(a.get(k)) if _num(a.get(k)) is not None else (d if d is not None else s_["t0"])  # noqa: E731
        note = [f"visit {s_['t0']:.1f}-{s_['t1']:.1f}s: {a.get('note', '')}"]
        done = yes("money_put_away") or yes("operation_finished")
        if yes("previous_batch_on_desk_when_client_entered"):
            F.append(Finding("V02", s_["t0"] + sc.session_pad_sec, 0.6, "previous batch on the desk when the client entered", note))
        if yes("batches_of_different_clients_combined"):
            F.append(Finding("V03", t("money_t"), 0.5, "batches of different clients combined", note))
        if yes("start_fully_visible") and yes("money_taken"):
            doc_t, money_t = _num(a.get("document_t")), _num(a.get("money_t"))
            if no("document_shown") or (doc_t is not None and money_t is not None and doc_t > money_t + 0.5):
                F.append(Finding("V04", t("money_t"), 0.55, "money taken before a document was shown", note))
        if yes("open_cash") and no("counted_on_big_counter") and done:
            F.append(Finding("V05", t("put_away_t", t("finished_t")), 0.5, "open cash stored without a primary count", note))
        if yes("counted_on_big_counter") and no("client_present_during_count"):
            F.append(Finding("V06", t("count_t"), 0.55, "primary count after the client left", note))
        if yes("rejected_combined_before_checks"):
            F.append(Finding("V09", t("put_away_t", t("count_t")), 0.5, "rejected notes combined before the checks", note))
        elif yes("rejected_part_separated") and done:
            if no("rejected_checked_on_small_machine"):
                F.append(Finding("V07", t("put_away_t", t("finished_t")), 0.45, "rejected notes accepted without the small machine", note))
            if no("rejected_part_recounted"):
                F.append(Finding("V08", t("put_away_t", t("finished_t")), 0.45, "rejected part not recounted", note))
        if yes("operation_finished") and yes("start_fully_visible") and no("typed_or_printed_on_laptop"):
            F.append(Finding("V10", t("finished_t"), 0.45, "operation finished without a laptop record", note))
        if yes("operation_finished") and yes("money_left_on_open_desk_after_finish"):
            fin = t("finished_t", s_["t1"])
            seen = any(w["t1"] >= fin and w["t0"] <= fin + 8 and (w["state"]["cash_on_open_desk"][0] == "yes" or w["state"]["cash_bag_on_open_desk"][0] == "yes") for w in wins)
            if seen:  # the visit answer alone over-reported this on the training clips
                F.append(Finding("V11", fin, 0.6, "accepted money left on the open desk after the operation", note))
        if yes("cashier_left_room_before_service_finished"):
            lt = t("left_t")
            out = _stage1_run(s1, lt - 2, lt + 6, lambda f: f["facts"]["cashier_zone"] in OUT_OF_ROOM, 2.0)
            F.append(Finding("V21", lt, 0.55 if out is not None else 0.45, "cashier left the room before the service ended", note))
    return F


PRIORITY = {"V11": ("V01", "V02"), "V15": ("V16", "V17"), "V07": ("V09",), "V08": ("V09",)}  # taxonomy: duplicates


def _scenario_results(s1: Stage1) -> tuple[list, str | None]:
    """Scenario rule results for a Stage 1 run; never raises: (results, "<Type>: <msg>" or None)."""
    try:
        from .facts import clip_id
        from .scenarios import run_scenarios
        from .scenarios.stage1_facts import facts_from_stage1
        name = (s1.r.get("meta") or {}).get("name")
        return run_scenarios(facts_from_stage1(s1.r, clip_id(name) if name else "clip")), None
    except Exception as e:
        return [], f"{type(e).__name__}: {e}"


def _safe_open_runs(safe: list[dict], sc: ScanConfig) -> list[tuple[float, float]]:
    """(first second, end) of runs of consecutive zoomed views with the safe door open, at least `safe_open_sec` long."""
    seq = sorted((d for d in safe if d.get("t") is not None), key=lambda d: d["t"])
    step, runs, cur = 1.0 / sc.safe_fps, [], []
    for d in seq + [{"t": math.inf, "open": "no"}]:
        if d["open"] == "yes" and (not cur or d["t"] - cur[-1] <= 1.5 * step):
            cur.append(d["t"])
            continue
        if cur and cur[-1] - cur[0] + step >= sc.safe_open_sec:
            runs.append((cur[0], cur[-1] + step))
        cur = [d["t"]] if d["open"] == "yes" else []
    return runs


def apply_rules(wins: list[dict], sessions: list[dict], doors: list[dict], s1: Stage1, sc: ScanConfig,
                scenarios: list | None = None, errors: list | None = None, safe: list[dict] | None = None) -> list[Finding]:
    F: list[Finding] = []
    k = sc.min_agree

    for state in ("cash_on_open_desk", "cash_bag_on_open_desk"):
        for first, last, grp in _runs(wins, state, k):
            left = _left_room(wins, s1, grp, first)
            entered = _event(wins, "client_entered_room", first, last)
            if left is not None:
                F.append(Finding("V01", max(first, left), 0.75, f"{state} while the cashier is out of the room",
                                 [f"{state} {first:.1f}-{last:.1f}s", f"cashier out at {left:.1f}s"]))
            elif entered is not None and entered > first:
                F.append(Finding("V02", entered, 0.7, f"{state} when the next client entered",
                                 [f"{state} since {first:.1f}s", f"client entered {entered:.1f}s"]))

    zoomed = sc.safe_zoom and any(d.get("open") in ("yes", "no") for d in safe or [])
    if zoomed:
        open_runs, v12_source = _safe_open_runs(safe, sc), "safe door open after the cashier finished at the safe (zoom)"
    else:
        open_runs, v12_source = [(a, b) for a, b, _ in _runs(wins, "safe_door_open", k)], "safe door open after the cashier finished at the safe"
    for first, last in open_runs:
        ended = _stage1_run(s1, first, last, lambda f: f["facts"]["cashier_zone"] != "near_safe", 3.0)
        if ended is not None:
            F.append(Finding("V12", ended, 0.55, v12_source, [f"safe open {first:.1f}-{last:.1f}s"]))
    for first, last, grp in _runs(wins, "key_left_visible", k):
        left = _left_room(wins, s1, grp, first)
        if left is not None and _event(wins, "key_taken_by_cashier", first - 8, left) is None:
            F.append(Finding("V13", left, 0.55, "key left visible after the cashier left the room", [f"key visible {first:.1f}-{last:.1f}s"]))
    for first, last, grp in _runs(wins, "personal_items_on_desk", k):
        busy = any(w["state"]["service_in_progress"][0] == "yes" or w["state"]["cash_on_open_desk"][0] == "yes"
                   or w["state"]["cash_bag_on_open_desk"][0] == "yes" for w in grp)
        if busy:  # personal items are allowed with no client and no money on the desk
            F.append(Finding("V15", first, 0.6, "personal item on the desk during service or with money on the desk", [f"{first:.1f}-{last:.1f}s"]))
    for first, last, grp in _runs(wins, "clothing_on_desk_or_in_safe", k):
        F.append(Finding("V16", first, 0.6, "clothing on the desk or in the safe", [f"{first:.1f}-{last:.1f}s"]))
    for first, last, grp in _runs(wins, "food_or_drink_on_desk", k):
        F.append(Finding("V17", first, 0.55, "food or drink at the workplace", [f"{first:.1f}-{last:.1f}s"]))
    eat = _event(wins, "eating_or_drinking", 0.0, 1e9)
    if eat is not None:
        F.append(Finding("V17", eat, 0.5, "eating or drinking at the workplace", [f"at {eat:.1f}s"]))
    for first, last, grp in _runs(wins, "visitor_on_cashier_side", k):
        F.append(Finding("V19", first, 0.45, "VLM: visitor on the cashier's side", [f"{first:.1f}-{last:.1f}s"]))
    t19 = _stage1_run(s1, 0.0, 1e9, lambda f: any(p["role"] == "visitor" and p["zone"] in ("cashier_seat", "standing_at_seat") for p in f["people"]), 2.0)
    if t19 is not None:
        F.append(Finding("V19", t19, 0.45, "visitor tracked at the cashier's seat", [f"from {t19:.1f}s"]))
    F += people_rule(wins, s1, sc)
    F += laptop_rule(s1, sc)
    F += door_rule(doors, sc, _confirmed_transits(wins, s1, sc))
    F += _session_rules(sessions, wins, s1, sc)
    if sc.scenario_rules:
        from .scenarios.findings import scenario_findings  # lazy: findings imports Finding from this module
        try:
            results = scenarios if scenarios is not None else _scenario_results(s1)[0]
            F += scenario_findings(results, sc.scenario_rules, sc.scenario_confidence)
        except Exception as e:  # scenarios are advisory: never break the scan
            if errors is not None:
                errors.append(f"{type(e).__name__}: {e}")
    if sc.entrance_outsider:
        from .scenarios.entrance import entrance_findings  # lazy: entrance imports Finding from this module
        try:
            F += entrance_findings(s1.r, sc)
        except Exception as e:  # advisory, like the scenario rules
            if errors is not None:
                errors.append(f"entrance: {type(e).__name__}: {e}")
    return _merge(F, sc.merge_gap_sec)


def _scenario_only(f: Finding) -> bool:
    return f.source.startswith("scenario ")


def _merge(F: list[Finding], gap: float) -> list[Finding]:
    """One record per class and continuous event: close findings of the same class merge (and gain
    confidence when two checks agree); taxonomy priorities drop the weaker duplicate.
    Scenario-only findings never shift or re-split VLM groups and never suppress a finding by priority;
    one that agrees with no VLM group is dropped (it stays visible in the scan's "scenarios")."""
    out: list[Finding] = []
    for f in sorted((f for f in F if not _scenario_only(f)), key=lambda f: (f.code, f.t)):
        if out and out[-1].code == f.code and f.t - out[-1].t <= gap:
            o = out[-1]
            o.confidence = min(0.95, max(o.confidence, f.confidence) + 0.1)
            o.evidence += f.evidence
            o.source += " + " + f.source
        else:
            out.append(f)
    boosted: set[int] = set()
    for f in sorted((f for f in F if _scenario_only(f)), key=lambda f: (f.code, f.t)):
        o = next((g for g in out if g.code == f.code and abs(f.t - g.t) <= gap), None)
        if o is None:  # scenario rules alone never report
            continue
        if id(o) not in boosted:  # VLM agreement: one boost per group, however many rules agree
            boosted.add(id(o))
            o.confidence = min(0.95, max(o.confidence, f.confidence) + 0.1)
        o.evidence += f.evidence
        o.source += " + " + f.source
    keep = [f for f in out if not any(g.code in PRIORITY.get(f.code, ()) and abs(g.t - f.t) <= gap for g in out)]
    return sorted(keep, key=lambda f: f.t)


# ---------------------------------------------------------------- run


def _parse_window(meta: dict, raw: str, d: dict) -> dict:
    st = d.get("state", {}) if isinstance(d.get("state"), dict) else {}
    evs = []
    for e in d.get("events", []) if isinstance(d.get("events"), list) else []:
        if isinstance(e, dict) and e.get("what") in EVENTS:
            evs.append({"t": _num(e.get("t")), "who": str(e.get("who", "")), "what": e["what"], "note": str(e.get("note", ""))[:80]})
    return {**meta, "state": {s: _ans(st.get(s, "not_visible")) for s in STATES}, "events": evs,
            "people_in_room": _num(d.get("people_in_room")), "parsed": bool(d), "raw": raw[:1500]}


def scan(video: V.Video, stage1: dict, vlm, cfg: PipelineConfig | None = None, log=lambda m: None,
         started: float | None = None) -> dict:
    cfg = cfg or PipelineConfig()
    sc = cfg.scan
    s1 = Stage1(stage1)
    t_begin = time.time()
    calls0, gpu0 = getattr(vlm, "calls", 0), getattr(vlm, "gpu_seconds", 0.0)
    state_fields = ", ".join(f'"{s}": [answer, since]' for s in STATES)

    wins = fixed_windows(video.duration, sc)
    items, metas = [], []
    for t0, t1 in wins:
        full, crops = _build(video, s1, t0, t1, sc.window_fps, None, sc)
        prompt = WINDOW_PROMPT.format(n=len(full), m=len(crops), t0=t0, t1=t1, scene=scene_text(s1.layout(t0)),
                                      state_fields=state_fields, events=", ".join(EVENTS))
        items.append((full + crops, prompt))
        metas.append({"kind": "window", "t0": t0, "t1": t1, "images": len(full) + len(crops)})

    # window answers come first: client entries the tracker missed define extra session windows
    t = time.time()
    deadline = started + sc.time_budget_sec - sc.finish_reserve_sec if sc.time_budget_sec and started is not None else None
    raws = list(vlm.ask_many(items, max_new_tokens=sc.max_new_tokens, batch=sc.batch, deadline=deadline))
    t_vlm = time.time() - t
    windows = [_parse_window(m, r or "", parse_json(r or "") or {}) for m, r in zip(metas, raws)]
    n_window_images = sum(m["images"] for m in metas)
    items, metas = [], []

    visits = add_event_sessions(s1.sessions(video.duration, sc.session_pad_sec), windows, video.duration, sc)
    for t0, t1, ids in visits:
        full, crops = _build(video, s1, t0, t1, sc.session_fps, sc.session_max_frames, sc)
        prompt = SESSION_PROMPT.format(n=len(full), m=len(crops), t0=t0, t1=t1, scene=scene_text(s1.layout(t0)),
                                       visitors=visitors_text(ids))
        items.append((full + crops, prompt))
        metas.append({"kind": "session", "t0": t0, "t1": t1, "visitors": ids, "images": len(full) + len(crops)})
    lay0 = s1.layout(0.0)
    if lay0.get("doors"):
        for t0 in np.arange(0.0, video.duration, sc.door_window_sec):
            t1 = min(t0 + sc.door_window_sec - 1.0 / sc.door_fps, video.duration - 0.05)
            times = _times(float(t0), float(t1), sc.door_fps)
            crops = [door_zoom(f, s1.layout(f.t), sc.crop_width) for f in _frames_at(video, times)]
            crops = [c for c in crops if c is not None]
            if crops:
                items.append((crops, DOOR_PROMPT.format(n=len(crops), t0=float(t0), t1=float(t1))))
                metas.append({"kind": "door", "t0": float(t0), "t1": float(t1), "images": len(crops)})
    if sc.safe_zoom and lay0.get("safe"):
        for t0 in np.arange(0.0, video.duration, sc.safe_window_sec):
            t1 = min(t0 + sc.safe_window_sec - 1.0 / sc.safe_fps, video.duration - 0.05)
            times = _times(float(t0), float(t1), sc.safe_fps)
            crops = [safe_zoom(f, s1.layout(f.t), sc.crop_width) for f in _frames_at(video, times)]
            crops = [c for c in crops if c is not None]
            if crops:
                items.append((crops, SAFE_PROMPT.format(n=len(crops), t0=float(t0), t1=float(t1))))
                metas.append({"kind": "safe", "t0": float(t0), "t1": float(t1), "images": len(crops)})
    t_build = time.time() - t_begin - t_vlm
    log(f"{len(wins)} windows + {len(visits)} visits, {n_window_images + sum(m['images'] for m in metas)} images")

    t = time.time()
    raws2 = vlm.ask_many(items, max_new_tokens=sc.max_new_tokens, batch=sc.batch, deadline=deadline) if items else []
    skipped = sum(r is None for r in raws + list(raws2))
    t_vlm += time.time() - t

    sessions, doors, doors_raw, safe, safe_raw = [], [], [], [], []
    for meta, raw in zip(metas, raws2):
        raw = raw or ""  # None: not asked before the deadline, handled like an unparseable answer
        d = parse_json(raw) or {}
        if meta["kind"] == "session":
            ans = {k: (v if k == "note" or k in SESSION_TIMES else _ans(v)[0]) for k, v in d.items()}
            sessions.append({**meta, "answer": ans, "parsed": bool(d), "raw": raw[:1500]})
        elif meta["kind"] == "safe":
            for fr in d.get("frames", []) if isinstance(d.get("frames"), list) else []:
                if isinstance(fr, dict):
                    v = str(fr.get("open", "")).lower()
                    safe.append({"t": _num(fr.get("t")), "open": "yes" if v.startswith("yes") else "no" if v == "no" else "not_visible"})
            safe_raw.append({**meta, "parsed": bool(d), "raw": raw[:800]})
        else:
            for fr in d.get("frames", []) if isinstance(d.get("frames"), list) else []:
                if isinstance(fr, dict):
                    v = str(fr.get("door", "")).lower()
                    state = "open" if v.startswith("open") else "closed" if v.startswith("clos") else "not_visible"
                    doors.append({"t": _num(fr.get("t")), "door": state, "person": _ans(fr.get("person_in_doorway", "no"))[0]})
            doors_raw.append({**meta, "parsed": bool(d), "raw": raw[:800]})

    scenarios, scenarios_error = _scenario_results(s1) if sc.scenario_rules else ([], None)
    map_errors: list[str] = []
    findings = apply_rules(windows, sessions, doors, s1, sc, scenarios, map_errors, safe)
    scenarios_error = scenarios_error or (map_errors[0] if map_errors else None)
    result = {
        "duration": video.duration,
        "windows": windows,
        "sessions": sessions,
        "door": {"frames": doors, "calls": doors_raw},
        "safe": {"frames": safe, "calls": safe_raw},
        "findings": [{**asdict(f), "rule": f.rule} for f in findings],
        "verdict": "violation" if any(f.confidence >= sc.threshold for f in findings) else "no_violation",
        "timings": {"build": round(t_build, 2), "vlm": round(t_vlm, 2), "total": round(time.time() - t_begin, 2)},
        "truncated": skipped > 0,
        "vlm_skipped": skipped,
        "vlm": {"calls": getattr(vlm, "calls", 0) - calls0, "gpu_seconds": round(getattr(vlm, "gpu_seconds", 0.0) - gpu0, 1),
                "unparsed": sum(not w["parsed"] for w in windows + sessions + doors_raw + safe_raw)},
    }
    result["scenarios"] = [r.to_dict() for r in scenarios]
    if scenarios_error:
        result["scenarios_error"] = scenarios_error
    return result
