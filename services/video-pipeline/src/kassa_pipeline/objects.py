"""Stage 3: object states on and around the desk (cash, bags, items, laptop screen, safe door).

Candidates come from open-vocabulary detection on the desk region. Their category and the screen/safe
states come from SigLIP: zero-shot by default, or from small trained heads in `heads/` when present.
"""

from dataclasses import dataclass, field

import numpy as np

from kassa_pipeline import geometry as g
from kassa_pipeline.config import ObjectsConfig
from kassa_pipeline.layout import LayoutTimeline
from kassa_pipeline.models import CropEncoder, OpenVocabDetector, TrainedHeads, threshold_for
from kassa_pipeline.people import PeopleResult
from kassa_pipeline.video import Video

# SigLIP zero-shot prompts. Keys are the categories the pipeline understands.
OBJECT_PROMPTS = {
    "cash": "a photo of banknotes or a stack of paper money on a desk",
    "cash_bag": "a photo of a green cash bag",
    "personal": "a photo of a handbag, purse, backpack or a coat",
    "food_drink": "a photo of a cup, mug, bottle or food",
    "phone": "a photo of a mobile phone",
    "paper": "a photo of paper documents, a notebook or a book",
    "device": "a photo of a laptop, tablet, telephone or office equipment",
    "surface": "a photo of an empty wooden desk surface",
}
IGNORED = {"paper", "device", "surface"}

SCREEN_PROMPTS = {
    "work_screen": "a laptop with a bright white screen showing a spreadsheet or application window",
    "locked": "a laptop showing a dark blue lock screen with a landscape wallpaper",
    "off": "a laptop with a black turned-off screen",
}
SAFE_PROMPTS = {
    "closed": "a closed grey metal safe",
    "open": "an open safe with its door swung open",
}


def categories_agree(dino_cat: str, siglip_cat: str) -> bool:
    cash = {"cash", "cash_bag"}
    return dino_cat == siglip_cat or (dino_cat in cash and siglip_cat in cash)


def _category_from_label(label: str) -> str | None:
    label = label.lower()
    if "bag" in label and "cash" in label:
        return "cash_bag"
    if any(k in label for k in ("banknote", "money", "cash")):
        return "cash"
    if any(k in label for k in ("handbag", "coat", "jacket")):
        return "personal"
    if any(k in label for k in ("cup", "mug", "food")):
        return "food_drink"
    if "phone" in label:
        return "phone"
    return None


@dataclass
class ObjObs:
    t: float
    cat: str
    box: g.Box
    score: float  # detector score x classifier probability
    on_desk: bool
    in_hand: bool


@dataclass
class StateObs:
    state: str  # work_screen | locked | off | unknown   or   open | closed | unknown
    prob: float


@dataclass
class ObjectsResult:
    objects: dict[int, list[ObjObs]] = field(default_factory=dict)  # by whole second
    screen: dict[int, StateObs] = field(default_factory=dict)
    safe: dict[int, StateObs] = field(default_factory=dict)


def _wrists_near(people: PeopleResult, t: float) -> list[tuple[float, float]]:
    pts = []
    for tr in people.tracks:
        o = tr.at(t, tol=0.6)
        if o and o.wrists:
            pts += o.wrists
    return pts


def _classify(
    encoder: CropEncoder, heads: TrainedHeads, task: str, prompts: dict[str, str], crops: list
) -> list[tuple[str, float]]:
    """Return (class, probability) per crop; trained head if available, else zero-shot."""
    if not crops:
        return []
    emb = encoder.embed(crops)
    if heads.has(task):
        classes, probs = heads.predict(task, emb)
    else:
        classes = list(prompts)
        probs = encoder.zero_shot(emb, list(prompts.values()))
    return [(classes[int(np.argmax(p))], float(np.max(p))) for p in probs]


@dataclass
class RawCandidate:
    t: float
    sec: int
    dino_cat: str
    det_score: float
    box: g.Box
    on_desk: bool
    in_hand: bool
    probs: dict[str, float]  # SigLIP / head class -> probability


@dataclass
class RawObjects:
    """Threshold-free output of the expensive part of the objects stage."""

    candidates: list[RawCandidate] = field(default_factory=list)
    screen: dict[int, StateObs] = field(default_factory=dict)
    safe: dict[int, StateObs] = field(default_factory=dict)


def _classify_probs(
    encoder: CropEncoder, heads: TrainedHeads, task: str, prompts: dict[str, str], crops: list
) -> list[dict[str, float]]:
    if not crops:
        return []
    emb = encoder.embed(crops)
    if heads.has(task):
        classes, probs = heads.predict(task, emb)
    else:
        classes = list(prompts)
        probs = encoder.zero_shot(emb, list(prompts.values()))
    return [{c: float(x) for c, x in zip(classes, p)} for p in probs]


def collect_object_candidates(
    video: Video,
    layouts: LayoutTimeline,
    people: PeopleResult,
    detector: OpenVocabDetector,
    encoder: CropEncoder,
    heads: TrainedHeads,
    cfg: ObjectsConfig,
    fps: float,
    floor: float | None = None,
) -> RawObjects:
    """Run DINO at `floor` and classify every candidate; thresholds are applied later by select_objects."""
    raw = RawObjects()
    if floor is None:
        floor = min([cfg.box_threshold, *cfg.category_thresholds.values()])
    for f in video.every(fps):
        sec = int(f.t)
        lay = layouts.at(f.t)
        h, w = f.norm.shape[:2]

        # 3a. candidates on the desk region (cropped, so small items get more pixels)
        if lay.workstation is not None:
            region = g.expand(lay.workstation, cfg.desk_crop_pad, w, h)
            crop = g.crop(f.norm, region)
            cands = []
            if crop is not None:
                dets = detector([crop], cfg.prompt, floor, cfg.text_threshold)[0]
                for d in dets:
                    cat = _category_from_label(d.label)
                    if cat is None:
                        continue
                    box = (
                        d.box[0] + region[0],
                        d.box[1] + region[1],
                        d.box[2] + region[0],
                        d.box[3] + region[1],
                    )
                    equipment = [b for b in (lay.laptop, lay.counter, lay.telephone) if b]
                    if any(g.iou(box, e) > 0.5 for e in equipment):
                        continue  # the work equipment itself, not an item on the desk
                    if g.area(box) > 0.5 * g.area(lay.workstation):
                        continue
                    cands.append((cat, d.score, box))

            # 3b. classify from a crop with context
            ctx = [g.crop(f.norm, g.expand(b, 0.4, w, h), min_size=12) for _, _, b in cands]
            valid = [(c, x) for c, x in zip(cands, ctx) if x is not None]
            probs = _classify_probs(encoder, heads, "object", OBJECT_PROMPTS, [x for _, x in valid])
            wrists = _wrists_near(people, f.t)
            desk = lay.desk or lay.workstation
            for (cat0, sc, box), pr in zip((c for c, _ in valid), probs):
                cx, cy = g.center(box)
                on_desk = g.contains(g.expand(desk, 0.05), (cx, cy))
                in_hand = any(
                    g.gap(box, (wx - 1, wy - 1, wx + 1, wy + 1)) < cfg.in_hand_px for wx, wy in wrists
                )
                raw.candidates.append(RawCandidate(f.t, sec, cat0, sc, box, on_desk, in_hand, pr))

        # 3c. laptop screen state
        raw.screen[sec] = _state(f.norm, lay.laptop, 0.3, encoder, heads, "screen", SCREEN_PROMPTS)
        # 3d. safe door state
        raw.safe[sec] = _state(f.norm, lay.safe, 0.5, encoder, heads, "safe", SAFE_PROMPTS)
    return raw


def select_objects(raw: RawObjects, cfg: ObjectsConfig) -> ObjectsResult:
    """Cheap part: per-category threshold, category from probs, min score, DINO/SigLIP agreement."""
    res = ObjectsResult(screen=dict(raw.screen), safe=dict(raw.safe))
    for c in raw.candidates:
        if c.det_score < threshold_for(c.dino_cat, cfg.category_thresholds, cfg.box_threshold):
            continue
        cat = max(c.probs, key=c.probs.get)
        prob = c.probs[cat]
        if cat in IGNORED or prob < cfg.min_category_score:
            continue
        agree = 1.0 if categories_agree(c.dino_cat, cat) else cfg.disagree_factor
        res.objects.setdefault(c.sec, []).append(
            ObjObs(c.t, cat, c.box, c.det_score * prob * agree, c.on_desk, c.in_hand)
        )
    return res


def detect_objects(
    video: Video,
    layouts: LayoutTimeline,
    people: PeopleResult,
    detector: OpenVocabDetector,
    encoder: CropEncoder,
    heads: TrainedHeads,
    cfg: ObjectsConfig,
    fps: float,
) -> ObjectsResult:
    return select_objects(
        collect_object_candidates(video, layouts, people, detector, encoder, heads, cfg, fps), cfg
    )


def _state(img, box, pad, encoder, heads, task, prompts) -> StateObs:
    if box is None:
        return StateObs("unknown", 0.0)
    h, w = img.shape[:2]
    c = g.crop(img, g.expand(box, pad, w, h), min_size=12)
    if c is None:
        return StateObs("unknown", 0.0)
    (cls, p), = _classify(encoder, heads, task, prompts, [c])
    return StateObs(cls, p)
