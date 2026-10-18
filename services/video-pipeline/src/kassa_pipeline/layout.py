"""Stage 1: scene layout found per video, so nothing depends on a fixed camera angle."""

from dataclasses import asdict, dataclass, field

import numpy as np

from kassa_pipeline import geometry as g
from kassa_pipeline.config import LayoutConfig
from kassa_pipeline.models import CropEncoder, Detection, OpenVocabDetector
from kassa_pipeline.video import Frame

# Keyword order matters: Grounding DINO can merge adjacent phrases ("safe metal cabinet").
_CANON = [
    ("counting machine", "counter"),
    ("banknote", "counter"),
    ("laptop", "laptop"),
    ("telephone", "telephone"),
    ("safe", "safe"),
    ("cabinet", "safe"),
    ("desk", "desk"),
    ("door", "door"),
    ("chair", "chair"),
]

_SAFE_PROMPTS = [
    "a grey metal safe",
    "a metal cabinet for storing money",
    "a white office drawer pedestal",
    "a desk",
    "a wall",
]
_SAFE_POSITIVE = {0, 1}

_COUNTER_PROMPTS = [
    "a banknote counting machine",
    "a stack of banknotes or paper money",
    "a desk",
    "a laptop",
]
_COUNTER_POSITIVE = {0}


def canon(label: str) -> str | None:
    label = label.lower()
    for key, name in _CANON:
        if key in label:
            return name
    return None


@dataclass
class Instance:
    cls: str
    box: g.Box
    score: float
    presence: float  # share of layout frames where it was found


@dataclass
class Layout:
    width: int
    height: int
    desk: g.Box | None = None
    workstation: g.Box | None = None  # desk grown by desk_pad: "at the desk"
    laptop: g.Box | None = None
    counter: g.Box | None = None
    telephone: g.Box | None = None
    safe: g.Box | None = None
    doors: list[g.Box] = field(default_factory=list)
    chairs: list[g.Box] = field(default_factory=list)
    instances: list[Instance] = field(default_factory=list)  # everything found, for debugging
    rejected_counters: list[Instance] = field(default_factory=list)  # failed the SigLIP counter check

    def to_dict(self) -> dict:
        d = asdict(self)
        d["instances"] = [asdict(i) for i in self.instances]
        d["rejected_counters"] = [asdict(i) for i in self.rejected_counters]
        return d


def _cluster(per_frame: list[list[tuple[str, Detection]]], n_frames: int) -> list[Instance]:
    """Group detections of one class across frames by IoU; median box per group."""
    groups: list[dict] = []
    for dets in per_frame:
        for cls, d in dets:
            best, best_iou = None, 0.3
            for grp in groups:
                if grp["cls"] != cls:
                    continue
                o = g.iou(d.box, g.median_box(grp["boxes"]))
                if o > best_iou:
                    best, best_iou = grp, o
            if best is None:
                groups.append({"cls": cls, "boxes": [d.box], "scores": [d.score], "frames": 1})
            else:
                best["boxes"].append(d.box)
                best["scores"].append(d.score)
                best["frames"] += 1
    return [
        Instance(
            grp["cls"],
            g.median_box(grp["boxes"]),
            float(np.mean(grp["scores"])),
            min(1.0, grp["frames"] / n_frames),
        )
        for grp in groups
    ]


def detect_layout(
    frames: list[Frame],
    detector: OpenVocabDetector,
    encoder: CropEncoder,
    cfg: LayoutConfig,
) -> Layout:
    h, w = frames[0].norm.shape[:2]
    raw = detector([f.norm for f in frames], cfg.prompt, cfg.box_threshold, cfg.text_threshold)
    per_frame: list[list[tuple[str, Detection]]] = []
    for dets in raw:
        kept = []
        for d in dets:
            c = canon(d.label)
            if c:
                kept.append((c, d))
        per_frame.append(kept)
    insts = [i for i in _cluster(per_frame, len(frames)) if i.presence >= cfg.min_presence]
    # "banknote" labels also map to counter: keep only those SigLIP agrees are a counting machine.
    ref = frames[len(frames) // 2].norm
    rejected_insts: list[Instance] = []
    counters = [i for i in insts if i.cls == "counter"]
    crops = [g.crop(ref, i.box) for i in counters]
    valid = [(i, c) for i, c in zip(counters, crops) if c is not None]
    if valid:
        probs = encoder.zero_shot(encoder.embed([c for _, c in valid]), _COUNTER_PROMPTS)
        rejected = {
            id(i)
            for (i, _), p in zip(valid, probs)
            if float(sum(p[k] for k in _COUNTER_POSITIVE)) < cfg.counter_min_prob
        }
        rejected_insts = [i for i in insts if id(i) in rejected]
        insts = [i for i in insts if id(i) not in rejected]
    by = lambda c: [i for i in insts if i.cls == c]  # noqa: E731
    layout = Layout(width=w, height=h, instances=insts, rejected_counters=rejected_insts)

    anchors = by("laptop") + by("counter") + by("telephone")

    # Desk: the desk box that holds the most work equipment, not simply the largest/most confident.
    def desk_score(d: Instance) -> float:
        grown = g.expand(d.box, 0.1)
        held = sum(1 for a in anchors if g.contains(grown, g.center(a.box)))
        return held + 0.1 * d.score * d.presence

    desks = sorted(by("desk"), key=desk_score, reverse=True)
    if desks and desk_score(desks[0]) >= 1:
        layout.desk = desks[0].box
    elif anchors:
        layout.desk = g.expand(g.union([a.box for a in anchors]), 0.3, w, h)
    elif desks:
        layout.desk = desks[0].box

    def on_desk(i: Instance) -> bool:
        return layout.desk is not None and g.contains(g.expand(layout.desk, 0.1), g.center(i.box))

    for cls in ("laptop", "counter", "telephone"):
        cands = [i for i in by(cls) if on_desk(i)] or by(cls)
        if cands:
            best = max(cands, key=lambda i: i.score * i.presence)
            setattr(layout, cls, best.box)

    # Safe: a cabinet near the desk that SigLIP agrees looks like a safe, not a drawer unit.
    safes = [i for i in by("safe") if not (layout.desk and g.inside_share(i.box, layout.desk) > 0.8)]
    if safes:
        crops = [g.crop(ref, i.box) for i in safes]
        valid = [(i, c) for i, c in zip(safes, crops) if c is not None]
        if valid:
            probs = encoder.zero_shot(encoder.embed([c for _, c in valid]), _SAFE_PROMPTS)
            diag = float(np.hypot(w, h))
            scored = []
            for (inst, _), p in zip(valid, probs):
                safe_p = float(sum(p[k] for k in _SAFE_POSITIVE))
                dist = g.gap(inst.box, layout.desk) / diag if layout.desk else 0.5
                scored.append((safe_p - dist, safe_p, inst))
            best = max(scored, key=lambda s: s[0])
            if best[1] >= 0.4:
                layout.safe = best[2].box

    layout.doors = [i.box for i in sorted(by("door"), key=lambda i: -i.score * i.presence)]
    layout.chairs = [i.box for i in by("chair")]
    if layout.desk:
        layout.workstation = g.expand(layout.desk, cfg.desk_pad, w, h)
    return layout


def layout_frames(frames: list[Frame], n: int, t0: float = 0.0, t1: float | None = None) -> list[Frame]:
    seg = [f for f in frames if f.t >= t0 and (t1 is None or f.t < t1)]
    if len(seg) <= n:
        return seg
    idx = np.linspace(0, len(seg) - 1, n).round().astype(int)
    return [seg[i] for i in idx]


@dataclass
class LayoutTimeline:
    """One layout per camera segment. A segment starts at a scene cut (camera moved or switched)."""

    starts: list[float]
    layouts: list[Layout]

    def index(self, t: float) -> int:
        i = 0
        for k, s0 in enumerate(self.starts):
            if t + 1e-6 >= s0:
                i = k
        return i

    def at(self, t: float) -> Layout:
        return self.layouts[self.index(t)]

    @property
    def first(self) -> Layout:
        return self.layouts[0]

    def to_dict(self) -> list[dict]:
        return [{"start": s, **lay.to_dict()} for s, lay in zip(self.starts, self.layouts)]


def detect_layout_timeline(
    frames: list[Frame],
    cuts: list[float],
    detector: OpenVocabDetector,
    encoder: CropEncoder,
    cfg: LayoutConfig,
    n_frames: int,
    min_segment_sec: float = 3.0,
) -> LayoutTimeline:
    starts = [0.0]
    for c in cuts:
        if c - starts[-1] >= min_segment_sec:
            starts.append(c)
    end = frames[-1].t + 1e-3
    bounds = starts + [end]
    layouts = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        # skip the first second after a cut: exposure is still settling
        seg = layout_frames(frames, n_frames, a + (1.0 if a > 0 else 0.0), b) or layout_frames(
            frames, n_frames, a, b
        )
        layouts.append(detect_layout(seg, detector, encoder, cfg))
    return LayoutTimeline(starts, layouts)
