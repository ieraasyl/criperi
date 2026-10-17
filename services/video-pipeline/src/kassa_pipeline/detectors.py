"""One detector function per rule.

Every detector reads the shared Context (timeline from the CV stages, plus the VLM) and returns
Detections. The pattern is the same everywhere:

  CV finds a candidate and its timing  ->  the VLM answers one closed question about the frames
  around it  ->  the two scores are fused  ->  the pipeline reports detections above a threshold.

Some detectors also let the VLM propose candidates the CV missed (a coat the detector never saw).
If no VLM is available, detectors fall back to CV-only scores; a failing VLM never fails the video.

The rule text and citation are never produced here: each Detection only carries a rule id that
report.py resolves in data/rules_kb.json.
"""

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from kassa_pipeline import events as E
from kassa_pipeline import geometry as g
from kassa_pipeline.config import PipelineConfig
from kassa_pipeline.events import Candidate
from kassa_pipeline.layout import LayoutTimeline
from kassa_pipeline.objects import ObjectsResult
from kassa_pipeline.people import PeopleResult
from kassa_pipeline.timeline import AT_WORKPLACE, Timeline
from kassa_pipeline.video import Video
from kassa_pipeline.vlm import VLM, Answer, ask_check, parse_json

import cv2


@dataclass
class Verdict:
    """Aggregate of the VLM's answers over a few frames."""

    confirmed: bool | None = None  # None = could not tell, or no VLM
    confidence: float = 0.0
    observation: str = ""
    yes: int = 0
    no: int = 0
    n: int = 0
    answers: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"confirmed": self.confirmed, "confidence": round(self.confidence, 2), "observation": self.observation, "yes": self.yes, "no": self.no}


@dataclass
class Detection(Candidate):
    rule: str = ""  # rule id in rules_kb.json
    cv_score: float = 0.0
    vlm: list[dict] = field(default_factory=list)
    source: str = "cv"  # cv | vlm | cv+vlm


# ---------------------------------------------------------------- context


def _zoom(rgb: np.ndarray, box: g.Box | None, pad: float, min_width: int = 480) -> np.ndarray | None:
    if box is None:
        return None
    h, w = rgb.shape[:2]
    c = g.crop(rgb, g.expand(box, pad, w, h), min_size=8)
    if c is None:
        return None
    s = max(1.0, min_width / c.shape[1])
    return cv2.resize(c, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)


SWEEP_PROMPT = """You are checking a top-down surveillance video frame of a bank cashier's workplace.
Image 1 is the whole scene; image 2 is the cashier's desk, enlarged. Look only at what is visible.
For each key answer true, false or "unknown":
 personal: a coat, jacket, handbag, purse or backpack lying on the cashier's desk
 food: a cup, mug, bottle or food on the desk or held by the cashier
 bag: a green or dark money bag / cash bag lying on the desk
Reply with ONLY this JSON: {"personal": ..., "food": ..., "bag": ...}"""


@dataclass
class Context:
    video: Video
    layouts: LayoutTimeline
    people: PeopleResult
    objects: ObjectsResult
    timeline: Timeline
    cfg: PipelineConfig
    vlm: VLM | None = None
    vlm_calls: int = 0
    deadline: float | None = None  # time.time() after which no more VLM calls are made
    _cv: list[Candidate] | None = None
    _sweep: dict[float, dict] | None = None

    # ---- frames
    def rgb(self, t: float) -> np.ndarray:
        return self.video.nearest(max(0.0, min(t, self.video.duration))).rgb

    def images(self, t: float, kind: str = "desk", box: g.Box | None = None) -> list[np.ndarray]:
        rgb, lay = self.rgb(t), self.layouts.at(t)
        desk = _zoom(rgb, lay.workstation, 0.25, 640)
        if kind == "scene" or desk is None and kind == "desk":
            return [rgb]
        if kind == "cash":
            boxes = [o.box for o in self.objects.objects.get(int(t), []) if o.cat in ("cash", "cash_bag") and o.on_desk]
            near = _zoom(rgb, g.union(boxes), 1.5, 480) if boxes else None
            return [x for x in (near, desk, rgb) if x is not None][:2]
        if kind == "safe":
            z = _zoom(rgb, lay.safe, 0.6, 480)
            return [x for x in (z, rgb) if x is not None]
        if kind == "seat":
            return [x for x in (desk, rgb) if x is not None]
        if kind == "box":
            z = _zoom(rgb, box, 0.5, 480)
            return [x for x in (z, rgb) if x is not None]
        return [rgb, desk] if desk is not None else [rgb]

    # ---- CV candidates (existing rule logic over the timeline)
    def cv(self, type_id: str) -> list[Candidate]:
        if self._cv is None:
            self._cv = E.find_candidates(self.timeline, self.cfg.events)
        return [c for c in self._cv if c.type == type_id]

    # ---- VLM
    def expired(self) -> bool:
        return self.deadline is not None and time.time() > self.deadline

    def verify(self, question: str, times: list[float], kind: str = "desk", context: str = "", box: g.Box | None = None) -> Verdict:
        if self.vlm is None or not self.cfg.vlm.enabled or self.expired():
            return Verdict()
        answers: list[Answer] = []
        for t in times[: max(1, self.cfg.vlm.verify_frames)]:
            imgs = self.images(t, kind, box)
            if not imgs:
                continue
            answers.append(ask_check(self.vlm, imgs, question, context))
            self.vlm_calls += 1
        yes = [a for a in answers if a.answer is True]
        no = [a for a in answers if a.answer is False]
        if yes and len(yes) > len(no):
            conf, obs, c = float(np.mean([a.confidence for a in yes])), yes[0].observation, True
        elif no and len(no) > len(yes):
            conf, obs, c = float(np.mean([a.confidence for a in no])), no[0].observation, False
        else:
            conf, obs, c = 0.0, (answers[0].observation if answers else ""), None
        return Verdict(c, conf, obs, len(yes), len(no), len(answers),
                       [{"t": round(t, 1), "q": question[:60], "a": a.answer, "conf": a.confidence, "obs": a.observation} for t, a in zip(times, answers)])

    def sweep(self) -> dict[float, dict]:
        """VLM look at the desk every few seconds: finds personal items, food and bags the CV missed."""
        if self._sweep is not None:
            return self._sweep
        self._sweep = {}
        if self.vlm is None or not self.cfg.vlm.enabled:
            return self._sweep
        step = max(1.0, self.cfg.vlm.sweep_step)
        times = list(np.arange(1.0, self.video.duration - 0.5, step))
        if len(times) > self.cfg.vlm.max_sweep_calls:
            times = list(np.linspace(1.0, self.video.duration - 0.5, self.cfg.vlm.max_sweep_calls))
        for t in times:
            if self.expired():
                break
            imgs = self.images(t, "desk")
            d = parse_json(self.vlm.ask(imgs, SWEEP_PROMPT, max_new_tokens=40)) or {}
            self.vlm_calls += 1
            self._sweep[round(float(t), 2)] = {k: (v if isinstance(v, bool) else None) for k, v in d.items()}
        return self._sweep

    def sweep_runs(self, key: str, min_samples: int = 2) -> list[tuple[float, float, int]]:
        """Runs of consecutive sweep frames where `key` was true: (first t, last t, samples)."""
        ts = sorted(self.sweep())
        runs, cur = [], []
        for t in ts:
            if self.sweep()[t].get(key) is True:
                cur.append(t)
            else:
                if len(cur) >= min_samples:
                    runs.append((cur[0], cur[-1], len(cur)))
                cur = []
        if len(cur) >= min_samples:
            runs.append((cur[0], cur[-1], len(cur)))
        return runs

    # ---- scoring
    def fuse(self, cv: float, v: Verdict | None, require_vlm: bool = False, cap: float = 1.0) -> float:
        c = self.cfg.vlm
        if v is None or v.n == 0:
            s = cv * c.require_unknown_factor if require_vlm and self.vlm is not None else cv
            if require_vlm and self.vlm is None:
                s = cv * 0.5  # a rule that needs the VLM, without one
        elif v.confirmed is True:
            s = min(1.0, cv + c.confirm_bonus * v.confidence)
        elif v.confirmed is False:
            s = cv * c.deny_factor
        else:
            s = cv * (c.require_unknown_factor if require_vlm else c.unknown_factor)
        return min(s, cap)

    def detection(self, c: Candidate, rule: str, v: Verdict | None = None, require_vlm: bool = False, cap: float = 1.0, type_id: str | None = None) -> Detection:
        score = self.fuse(c.score, v, require_vlm, cap)
        facts = dict(c.facts)
        if v is not None and v.n:
            facts["vlm"] = v.to_dict()
        return Detection(
            type_id or c.type, c.t_start, c.t_end, c.timecode, round(score, 3), facts,
            rule=rule, cv_score=c.score, vlm=v.answers if v else [],
            source="cv+vlm" if v is not None and v.n else "cv",
        )

    def discovered(self, type_id: str, rule: str, first: float, last: float, n: int, mode: str, cap: float = 1.0) -> Detection:
        """A candidate that only the VLM sweep found."""
        score = min(cap, self.cfg.vlm.discovery_base + 0.05 * min(n - 2, 4))
        return Detection(type_id, first, last + self.cfg.vlm.sweep_step, first, round(score, 3), {"mode": mode, "samples": n},
                         rule=rule, cv_score=0.0, source="vlm")

    def away_at(self, t: float) -> bool:
        s = self.timeline.seconds
        return not s[min(int(t), len(s) - 1)].cashier_at_workplace


def _span(c: Candidate, ctx: Context, lo: float = 0.5, hi: float = 5.0) -> list[float]:
    a = c.t_start + lo
    b = min(c.t_end - 0.2, c.t_start + hi, ctx.video.duration - 0.3)
    n = max(1, ctx.cfg.vlm.verify_frames)
    if b <= a:
        return [max(0.0, min(c.timecode + 0.5, ctx.video.duration - 0.3))]
    return [float(x) for x in np.linspace(a, b, n)]


# ---------------------------------------------------------------- detectors
# Each returns a list[Detection]. Rule ids refer to data/rules_kb.json.


def detect_cash_left(ctx: Context) -> list[Detection]:
    """НБРК № 231 п. 20 ч. 2: cash or a cash bag left on the desk (T1)."""
    out = []
    for c in ctx.cv("T1"):
        t0 = float(c.facts.get("cash_seen_from", c.timecode))
        times = [t0 + 0.5, min(t0 + 3.0, max(c.t_end - 0.2, t0 + 0.5))]
        v = ctx.verify(
            "Are banknotes (paper money) or a cash bag lying on the desk surface? Ignore money held in a hand.",
            times, "cash", context="The cashier has stood up or left the desk.",
        )
        out.append(ctx.detection(c, "R-CASH-LOCKED", v))
    for first, last, n in ctx.sweep_runs("bag"):
        if ctx.away_at(first):
            out.append(ctx.discovered("T1", "R-CASH-LOCKED", first, last, n, "cash_bag_seen_by_vlm"))
    return E.merge(out, ctx.cfg.events.merge_gap_sec)  # type: ignore[return-value]


def detect_personal_items(ctx: Context) -> list[Detection]:
    """НБРК № 231 п. 12: personal belongings on the desk (T2)."""
    out = []
    for c in ctx.cv("T2"):
        v = ctx.verify("Is there a personal item (coat, jacket, handbag, purse or backpack) lying on the work desk?", _span(c, ctx))
        out.append(ctx.detection(c, "R-PERSONAL-ITEMS", v))
    for first, last, n in ctx.sweep_runs("personal"):
        out.append(ctx.discovered("T2", "R-PERSONAL-ITEMS", first, last, n, "personal_item_seen_by_vlm"))
    return E.merge(out, ctx.cfg.events.merge_gap_sec)  # type: ignore[return-value]


def detect_food_drink(ctx: Context) -> list[Detection]:
    """НБРК № 231 п. 12, weak fit: food or drink at the workstation (T3)."""
    cap = ctx.cfg.vlm.soft_cap
    out = []
    for c in ctx.cv("T3"):
        v = ctx.verify("Is there a cup, mug, bottle or food on the desk, or is the cashier eating or drinking?", _span(c, ctx))
        out.append(ctx.detection(c, "R-PERSONAL-ITEMS", v, cap=cap))
    for first, last, n in ctx.sweep_runs("food"):
        out.append(ctx.discovered("T3", "R-PERSONAL-ITEMS", first, last, n, "food_seen_by_vlm", cap=cap))
    return E.merge(out, ctx.cfg.events.merge_gap_sec)  # type: ignore[return-value]


def detect_safe_open_unattended(ctx: Context) -> list[Detection]:
    """НБРК № 231 п. 20: safe door open while the cashier is out of the room (T7).

    The CV safe state flaps, so this detector requires the VLM to confirm."""
    out = []
    for c in ctx.cv("T7"):
        v = ctx.verify("Is the door of the grey metal safe (cabinet) swung open, with its inner side visible?", _span(c, ctx), "safe")
        out.append(ctx.detection(c, "R-CASH-LOCKED", v, require_vlm=True))
    return out


def detect_leave_during_service(ctx: Context) -> list[Detection]:
    """НБРК № 231 п. 22: the cashier leaves while a client is being served (T6)."""
    out = []
    for c in ctx.cv("T6"):
        v = ctx.verify(
            "Is there a person (a client) at the desk while the cashier is not at the desk?",
            _span(c, ctx), "scene",
        )
        out.append(ctx.detection(c, "R-LEAVE", v))
    return out


def detect_outsider_on_cashier_side(ctx: Context) -> list[Detection]:
    """НБРК № 14 п. 8 / № 231 п. 75, indirect: someone beside the cashier (T4)."""
    out = []
    for c in ctx.cv("T4"):
        v = ctx.verify(
            "Is there a person standing on the SAME side of the desk as the cashier, right next to the cashier? "
            "A person sitting across the desk facing the cashier does NOT count.",
            _span(c, ctx, 0.2, 3.0), "seat",
        )
        out.append(ctx.detection(c, "R-ENTRY-ISOLATION", v, cap=ctx.cfg.vlm.soft_cap))
    return out


def detect_screen_unlocked(ctx: Context) -> list[Detection]:
    """АРРФР № 53 п. 71 пп. 9, weak fit: laptop left unlocked when leaving (T5).

    CV only: pixel statistics on the laptop screen. Both zero-shot SigLIP and the VLM misread the
    screen state (X-29's lock screen came back as a work window)."""
    return [ctx.detection(c, "R-IS-LOCK", None, cap=ctx.cfg.vlm.soft_cap) for c in ctx.cv("T5")]


def detect_count_without_client(ctx: Context) -> list[Detection]:
    """НБРК № 231 п. 20 ч. 1, п. 31: cash handled with no client at the desk (T10).

    Low confidence by design: counting after a client leaves is also normal work."""
    S = ctx.timeline.seconds
    handling = []
    for s in S:
        objs = ctx.objects.objects.get(s.t, [])
        in_hand = any(o.cat in ("cash", "cash_bag") and o.in_hand for o in objs)
        handling.append(in_hand and s.cashier_at_workplace and not s.client_at_desk)
    out = []
    for a, b in E._runs(handling):
        if b - a + 1 >= 3 and not any(x.client_at_desk for x in S[max(0, a - 2) : a]):
            c = Candidate("T10", a, b + 1, a, 0.4, {"seconds": b - a + 1})
            v = ctx.verify("Is the cashier counting or handling banknotes while nobody sits at the desk?", _span(c, ctx), "desk")
            out.append(ctx.detection(c, "R-COUNT-CLIENT", v, cap=0.45))
    return out


def detect_counting_alone(ctx: Context) -> list[Detection]:
    """НБРК № 231 п. 42: counting without a controlling employee (T11). Very low confidence: a second
    employee's role cannot be told from video, so this never reaches the reporting threshold alone."""
    lay = ctx.layouts.first
    cashier = ctx.people.cashier()
    if cashier is None or lay.counter is None:
        return []
    S, flags = ctx.timeline.seconds, []
    zone = g.expand(lay.counter, 0.4)
    for s in S:
        o = cashier.at(s.t + 0.5, tol=0.6)
        near = bool(o and any(g.contains(zone, w) for w in o.wrists))
        flags.append(near and not s.visitors)
    return [
        ctx.detection(Candidate("T11", a, b + 1, a, 0.3, {"seconds": b - a + 1}), "R-CONTROLLER", None, cap=0.35)
        for a, b in E._runs(flags) if b - a + 1 >= 3
    ]


def detect_collector_issues(ctx: Context) -> list[Detection]:
    """НБРК № 231 пп. 98, 103: a collector who arrives alone (T13) or without body armour (T14).

    Hypotheses, not verified in the training data: capped below the reporting threshold."""
    if ctx.vlm is None:
        return []
    out = []
    for tr in ctx.people.tracks:
        if tr.role == "cashier":
            continue
        t = tr.t0 + 1.0
        o = tr.at(t, tol=1.0)
        if o is None:
            continue
        has_bag = ctx.verify("Is this person carrying a bag (a cash bag, money bag or briefcase)?", [t], "box", box=o.box)
        if has_bag.confirmed is not True:
            continue
        armour = ctx.verify("Is this person wearing body armour, a bulletproof vest or a helmet?", [t], "box", box=o.box)
        alone = ctx.verify("Is this person alone, with no second person arriving together with them?", [t], "scene")
        if armour.confirmed is False:
            out.append(ctx.detection(Candidate("T14", tr.t0, tr.t0 + 2, tr.t0, 0.3 + 0.15 * has_bag.confidence, {"track": tr.id}), "R-COLLECTOR-KIT", armour, cap=0.45))
        if alone.confirmed is True:
            out.append(ctx.detection(Candidate("T13", tr.t0, tr.t0 + 2, tr.t0, 0.3 + 0.15 * has_bag.confidence, {"track": tr.id}), "R-COLLECTOR-CREW", alone, cap=0.45))
    return out


def detect_shared_laptop(ctx: Context) -> list[Detection]:
    """АРРФР № 53 п. 73: someone other than the cashier works at the cashier's laptop (T17)."""
    lay = ctx.layouts.first
    if lay.laptop is None:
        return []
    zone = g.expand(lay.laptop, 0.4)
    S, flags = ctx.timeline.seconds, []
    for s in S:
        hit = False
        for tr in ctx.people.tracks:
            if tr.role == "cashier":
                continue
            o = tr.at(s.t + 0.5, tol=0.6)
            hit = hit or bool(o and any(g.contains(zone, w) for w in o.wrists))
        flags.append(hit)
    out = []
    for a, b in E._runs(flags):
        if b - a + 1 >= 2:
            c = Candidate("T17", a, b + 1, a, 0.45, {"seconds": b - a + 1})
            v = ctx.verify("Is a person other than the cashier touching or typing on the laptop?", _span(c, ctx), "box", box=lay.laptop)
            out.append(ctx.detection(c, "R-IS-NO-SHARED-ACCOUNT", v, cap=0.6))
    return out


# ---------------------------------------------------------------- registry


@dataclass
class Detector:
    id: str
    types: tuple[str, ...]
    rule: str
    fn: Callable[[Context], list[Detection]]
    uses_vlm: bool
    note: str = ""


DETECTORS: list[Detector] = [
    Detector("cash_left", ("T1",), "R-CASH-LOCKED", detect_cash_left, True, "CV + VLM verify + VLM bag sweep"),
    Detector("personal_items", ("T2",), "R-PERSONAL-ITEMS", detect_personal_items, True, "CV + VLM verify + VLM sweep"),
    Detector("food_drink", ("T3",), "R-PERSONAL-ITEMS", detect_food_drink, True, "weak norm, capped"),
    Detector("safe_open", ("T7",), "R-CASH-LOCKED", detect_safe_open_unattended, True, "requires VLM"),
    Detector("leave_during_service", ("T6",), "R-LEAVE", detect_leave_during_service, True, "no training clip shows it"),
    Detector("outsider", ("T4",), "R-ENTRY-ISOLATION", detect_outsider_on_cashier_side, True, "indirect norm, capped"),
    Detector("screen_unlocked", ("T5",), "R-IS-LOCK", detect_screen_unlocked, False, "CV pixel statistics only"),
    Detector("count_without_client", ("T10",), "R-COUNT-CLIENT", detect_count_without_client, True, "low confidence by design"),
    Detector("counting_alone", ("T11",), "R-CONTROLLER", detect_counting_alone, False, "never above the threshold alone"),
    Detector("collector", ("T13", "T14"), "R-COLLECTOR-CREW", detect_collector_issues, True, "unverified hypothesis"),
    Detector("shared_laptop", ("T17",), "R-IS-NO-SHARED-ACCOUNT", detect_shared_laptop, True, "not seen in any training clip"),
]

# Rules in rules_kb.json that this pipeline deliberately does not detect, and why.
UNSUPPORTED = {
    "R-HANDOVER-DOC": "a handover between two cashiers cannot be told from a cashier-client handover; the document is not visible",
    "R-DOUBTFUL-ACT": "needs a second employee and a paper act, not reliably visible at this resolution",
    "R-KEYS": "keys are too small to detect reliably",
    "R-CLIENT-ID": "absence of an ID check is not a visible event",
    "R-CASH-ROOM-DOOR": "the door is opened by every visitor; the VLM's door state is unreliable",
    "R-CLIENT-COUNTS": "client behaviour after receiving cash is not separable from normal flow",
    "R-COUNT-ISSUE-CLIENT": "covered approximately by count_without_client",
    "R-COLLECTOR-CHECKS-BAG": "a seal check is not visible at this resolution",
    "R-ESCORT-CREW": "no escort scenario in the data",
    "phone (R-PERSONAL-ITEMS)": "the VLM answers 'phone visible' on almost every frame and zero-shot CV is noisy",
}


def run_detectors(ctx: Context, only: list[str] | None = None) -> tuple[list[Detection], dict]:
    """Run every detector; one failing detector never fails the video."""
    out: list[Detection] = []
    info: dict = {"timings": {}, "errors": {}}
    for d in DETECTORS:
        if only and d.id not in only:
            continue
        t = time.time()
        try:
            out += d.fn(ctx)
        except Exception as e:  # a VLM or model hiccup must degrade, not crash
            info["errors"][d.id] = f"{type(e).__name__}: {e}"
        info["timings"][d.id] = round(time.time() - t, 2)
    info["vlm_calls"] = ctx.vlm_calls
    return sorted(out, key=lambda x: x.timecode), info
