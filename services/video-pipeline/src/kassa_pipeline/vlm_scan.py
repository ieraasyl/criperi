"""Sliding-window VLM scan: the VLM logs what it sees, code decides which rule is violated.

Each signal comes from the source that was reliable on the 42 training clips (runs/scan/report.md):
1. Fixed windows (8 s, stride 4 s), 16 full frames + 8 zoomed desk views with the Stage 1 track boxes and
   the time drawn on every image: the VLM reports cash, cash bags, personal items and food on the desk.
2. Session windows (one per client visit): the VLM says whether money was left on the desk across a visit.
3. Door: zoomed views of the doorway at 1 fps; the VLM labels each view open or closed. Clients open the
   door briefly when they walk in, so only a door left open for `door_open_sec` counts.
4. Laptop: the Stage 1 pixel classifier (the VLM read the screen wrong on most controls).
Rules run over the merged log; the VLM never names a rule. Rule ids map to rules_kb.json.
Not checked here, because the VLM could not see them at this resolution: ID shown, money counted, key in the safe.
"""

import bisect
import time
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from kassa_pipeline import geometry as g
from kassa_pipeline import video as V
from kassa_pipeline.config import PipelineConfig, ScanConfig
from kassa_pipeline.vlm import parse_json

STATES = ["cashier_at_desk", "cash_unattended_on_desk", "cash_bag_on_desk", "personal_items_on_desk", "food_or_drink_on_desk"]
EVENTS = [
    "client_arrived", "client_left", "money_handed_to_cashier", "bag_handed_to_cashier", "money_counted",
    "money_put_in_safe", "money_given_to_client", "cashier_left_room", "cashier_returned",
]

WINDOW_PROMPT = """These are {n} frames from a top-down surveillance camera in a bank cashier's room, from {t0:.1f} s to {t1:.1f} s, followed by {m} zoomed views of the cashier's desk at the same moments. Every image shows its time (t=...) in the top-left corner. Boxes mark people found by a tracker: "#N cashier" is the bank cashier, "#N visitor" is anyone else (usually a client).
Scene: {scene}
Report only what is clearly visible. If you cannot tell, answer "not_visible". Do not guess.

Reply with ONLY this JSON, no other text:
{{"events": [{{"t": <seconds>, "who": "#N", "what": "<event>", "note": "<at most 8 words>"}}],
 "state": {{{state_fields}}}}}
Allowed events, at most 6, in time order: {events}.
Each state is [answer, since]: answer is "yes", "no" or "not_visible" for the LAST frames of the window; since is the first second it was "yes" in this window, or null.
State meanings: cash_unattended_on_desk = banknotes lie on the desk and nobody is holding or counting them; cash_bag_on_desk = a money bag lies on the desk; personal_items_on_desk = a coat, handbag or backpack lies on the desk; food_or_drink_on_desk = a cup, bottle, can or food is on the desk; cashier_at_desk = the cashier sits or stands at the desk."""

SESSION_PROMPT = """These are {n} frames from a top-down surveillance camera in a bank cashier's room covering one client visit, from {t0:.1f} s to {t1:.1f} s, followed by {m} zoomed views of the cashier's desk. Every image shows its time (t=...) in the top-left corner. Boxes mark people found by a tracker: "#N cashier" is the bank cashier, "#N visitor" is anyone else. The visitor(s) in this visit: {visitors}.
Scene: {scene}
Answer about THIS visit only, from what is visible. If you cannot tell, answer "not_visible". Do not guess.

Reply with ONLY this JSON, no other text:
{{"previous_money_on_desk_when_client_arrived": "yes|no|not_visible",
 "money_left_on_desk_after_client_left": "yes|no|not_visible",
 "note": "<at most 20 words on what happened>"}}
previous_money_on_desk_when_client_arrived = banknotes or a money bag already lay on the desk, unattended, when this client came in.
money_left_on_desk_after_client_left = banknotes or a money bag still lie on the desk, unattended, after this client left."""

DOOR_PROMPT = """These are {n} zoomed views of the doorway area of a bank cashier's room (a glass wall with a glass door), one per second from {t0:.1f} s to {t1:.1f} s. Every image shows its time (t=...) in the top-left corner.
For EACH image: is the door open (the door leaf is swung away from the glass wall and the doorway is a gap) or closed (the door leaf lies flat in line with the glass wall)? Is a person standing in the doorway?
If you cannot tell, answer "not_visible". Do not guess.
Reply with ONLY this JSON, no other text:
{{"frames": [{{"t": <seconds>, "door": "open|closed|not_visible", "person_in_doorway": "yes|no"}}]}}"""


@dataclass
class Finding:
    rule: str
    t: float
    confidence: float
    source: str  # which check produced it
    evidence: list[str] = field(default_factory=list)
    kind: str = ""  # what was seen; report.py words the description from it


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


AWAY_ZONES = ("absent", "doorway", "elsewhere")


def _away(wins: list[dict], s1: Stage1, grp: list[dict], first: float, min_sec: float = 2.0) -> float | None:
    """Earliest second (not before `first`) when the cashier is away: the VLM says so at the end of a
    window, the VLM logged the cashier leaving, or Stage 1 has the cashier away for `min_sec`."""
    for w in grp:
        if w["state"].get("cashier_at_desk", ("not_visible", None))[0] == "no":
            return max(first, w["t0"] + (w["t1"] - w["t0"]) / 2)
    t = _event(wins, "cashier_left_room", grp[0]["t0"], grp[-1]["t1"])
    if t is not None:
        return max(first, t)
    run_start = None
    for f in s1.frames:
        if f["t"] < first or f["t"] > grp[-1]["t1"]:
            continue
        if f["facts"]["cashier_zone"] in AWAY_ZONES:
            run_start = f["t"] if run_start is None else run_start
            if f["t"] - run_start >= min_sec:
                return run_start
        else:
            run_start = None
    return None


WORK_ZONES = ("cashier_seat", "at_desk", "standing_at_seat")


def laptop_rule(s1: Stage1, sc: ScanConfig) -> list[Finding]:
    """The cashier leaves (away for `away_min_sec` after being at the workplace) and the Stage 1 pixel
    classifier sees the work screen during the first seconds of the absence."""
    F, frames, i, was_at_work = [], s1.frames, 0, False
    while i < len(frames):
        f = frames[i]
        zone = f["facts"]["cashier_zone"]
        if zone in WORK_ZONES:
            was_at_work = True
        if was_at_work and zone in AWAY_ZONES:
            j = i
            while j + 1 < len(frames) and frames[j + 1]["facts"]["cashier_zone"] in AWAY_ZONES:
                j += 1
            if frames[j]["t"] - f["t"] >= sc.away_min_sec:
                win = [x["facts"]["screen"] for x in frames[i : j + 1] if x["t"] <= f["t"] + sc.away_min_sec]
                known = [x for x in win if x != "unknown"]
                share = sum(x == "work_screen" for x in known) / len(known) if known else 0.0
                if share >= 0.6:
                    F.append(Finding("R-IS-LOCK", f["t"], 0.65, "work screen (pixel classifier) after the cashier left",
                                     [f"cashier away from {f['t']:.1f}s to {frames[j]['t']:.1f}s", f"work screen in {share:.0%} of frames"],
                                     kind="laptop_away"))
            was_at_work = False
            i = j + 1
            continue
        i += 1
    return F


def door_rule(doors: list[dict], sc: ScanConfig) -> list[Finding]:
    """The door stands open for at least `door_open_sec` (a client walking in opens it only briefly)."""
    F, run = [], []
    seq = sorted((d for d in doors if d.get("t") is not None), key=lambda d: d["t"])

    def close(run):
        if run and run[-1]["t"] - run[0]["t"] + 1.0 / sc.door_fps >= sc.door_open_sec:
            someone = any(d["person"] == "yes" for d in run)
            F.append(Finding("R-ENTRY-ISOLATION", run[0]["t"], 0.65 if someone else 0.6,
                             "door left open" + (" with a person in the doorway" if someone else ""),
                             [f"door open {run[0]['t']:.1f}-{run[-1]['t']:.1f}s"],
                             kind="door_open_person" if someone else "door_open"))

    for d in seq:
        if d["door"] == "open" and (not run or d["t"] - run[-1]["t"] <= 1.5 / sc.door_fps):
            run.append(d)
        else:
            close(run)
            run = [d] if d["door"] == "open" else []
    close(run)
    return F


def apply_rules(wins: list[dict], sessions: list[dict], doors: list[dict], s1: Stage1, sc: ScanConfig) -> list[Finding]:
    F: list[Finding] = []
    k = sc.min_agree

    # money or a money bag left lying: cashier away, or a new client arrives while it is there
    for state in ("cash_unattended_on_desk", "cash_bag_on_desk"):
        for first, last, grp in _runs(wins, state, k):
            away = _away(wins, s1, grp, first)
            arrived = _event(wins, "client_arrived", first, last)
            if away is not None:
                F.append(Finding("R-CASH-LOCKED", max(first, away), 0.75, f"{state} while cashier away",
                                 [f"{state} from {first:.1f}s to {last:.1f}s", f"cashier away at {away:.1f}s"], kind="cash_away"))
            elif arrived is not None and arrived > first:
                F.append(Finding("R-CASH-LOCKED", arrived, 0.7, f"{state} when a new client arrived",
                                 [f"{state} since {first:.1f}s", f"client arrived {arrived:.1f}s"], kind="cash_new_client"))
            elif last - first >= 12 and not s1.visitors_present(first, last):
                F.append(Finding("R-CASH-LOCKED", first, 0.55, f"{state} with no client present",
                                 [f"{state} from {first:.1f}s to {last:.1f}s, no visitor tracked"], kind="cash_no_client"))

    for state, conf in (("personal_items_on_desk", 0.65), ("food_or_drink_on_desk", 0.55)):
        for first, last, grp in _runs(wins, state, k):
            F.append(Finding("R-PERSONAL-ITEMS", first, conf, state, [f"{state} {first:.1f}-{last:.1f}s"],
                             kind="personal_items" if state.startswith("personal") else "food"))
    F += laptop_rule(s1, sc)
    F += door_rule(doors, sc)

    # money across a client visit
    for s_ in sessions:
        a = s_["answer"]
        note = [f"visit {s_['t0']:.1f}-{s_['t1']:.1f}s: {a.get('note', '')}"]
        if a.get("previous_money_on_desk_when_client_arrived") == "yes":
            F.append(Finding("R-CASH-LOCKED", s_["t0"] + sc.session_pad_sec, 0.65, "previous money on the desk when the client arrived", note, kind="cash_prev_visit"))
        if a.get("money_left_on_desk_after_client_left") == "yes":
            F.append(Finding("R-CASH-LOCKED", max(s_["t1"] - sc.session_pad_sec, s_["t0"]), 0.6, "money left on the desk after the client left", note, kind="cash_after_visit"))
    return _merge(F, sc.merge_gap_sec)


def _merge(F: list[Finding], gap: float) -> list[Finding]:
    out: list[Finding] = []
    for f in sorted(F, key=lambda f: (f.rule, f.t)):
        if out and out[-1].rule == f.rule and f.t - out[-1].t <= gap:
            o = out[-1]
            o.confidence = min(0.95, max(o.confidence, f.confidence) + 0.1)  # two checks agree
            o.evidence += f.evidence
            o.source += " + " + f.source
        else:
            out.append(f)
    return sorted(out, key=lambda f: f.t)


# ---------------------------------------------------------------- run


def scan(video: V.Video, stage1: dict, vlm, cfg: PipelineConfig | None = None, log=lambda m: None) -> dict:
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

    visits = s1.sessions(video.duration, sc.session_pad_sec)
    for t0, t1, ids in visits:
        full, crops = _build(video, s1, t0, t1, sc.session_fps, sc.session_max_frames, sc)
        prompt = SESSION_PROMPT.format(n=len(full), m=len(crops), t0=t0, t1=t1, scene=scene_text(s1.layout(t0)),
                                       visitors=", ".join(f"#{i}" for i in ids))
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
    t_build = time.time() - t_begin
    log(f"{len(wins)} windows + {len(visits)} visits, {sum(m['images'] for m in metas)} images")

    t = time.time()
    raws = vlm.ask_many(items, max_new_tokens=sc.max_new_tokens, batch=sc.batch)
    t_vlm = time.time() - t

    windows, sessions, doors, doors_raw = [], [], [], []
    for meta, raw in zip(metas, raws):
        d = parse_json(raw) or {}
        if meta["kind"] == "window":
            st = d.get("state", {}) if isinstance(d.get("state"), dict) else {}
            evs = []
            for e in d.get("events", []) if isinstance(d.get("events"), list) else []:
                if isinstance(e, dict) and e.get("what") in EVENTS:
                    evs.append({"t": _num(e.get("t")), "who": str(e.get("who", "")), "what": e["what"], "note": str(e.get("note", ""))[:80]})
            windows.append({**meta, "state": {s: _ans(st.get(s, "not_visible")) for s in STATES}, "events": evs,
                            "parsed": bool(d), "raw": raw[:1500]})
        elif meta["kind"] == "session":
            ans = {k: (v if k == "note" else _ans(v)[0]) for k, v in d.items()}
            sessions.append({**meta, "answer": ans, "parsed": bool(d), "raw": raw[:1500]})
        else:
            for fr in d.get("frames", []) if isinstance(d.get("frames"), list) else []:
                if isinstance(fr, dict):
                    v = str(fr.get("door", "")).lower()
                    state = "open" if v.startswith("open") else "closed" if v.startswith("clos") else "not_visible"
                    doors.append({"t": _num(fr.get("t")), "door": state, "person": _ans(fr.get("person_in_doorway", "no"))[0]})
            doors_raw.append({**meta, "parsed": bool(d), "raw": raw[:800]})

    findings = apply_rules(windows, sessions, doors, s1, sc)
    return {
        "duration": video.duration,
        "windows": windows,
        "sessions": sessions,
        "door": {"frames": doors, "calls": doors_raw},
        "findings": [asdict(f) for f in findings],
        "verdict": "violation" if any(f.confidence >= sc.threshold for f in findings) else "no_violation",
        "timings": {"build": round(t_build, 2), "vlm": round(t_vlm, 2), "total": round(time.time() - t_begin, 2)},
        "vlm": {"calls": getattr(vlm, "calls", 0) - calls0, "gpu_seconds": round(getattr(vlm, "gpu_seconds", 0.0) - gpu0, 1),
                "unparsed": sum(not w["parsed"] for w in windows + sessions + doors_raw)},
    }
