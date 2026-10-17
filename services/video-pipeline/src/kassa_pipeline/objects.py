"""Stage 3: object states on and around the desk (cash, bags, items, laptop screen, safe door).

Candidates come from open-vocabulary detection on the desk region. Their category and the screen/safe
states come from SigLIP: zero-shot by default, or from small trained heads in `heads/` when present.
"""

from dataclasses import dataclass, field

import numpy as np

from kassa_pipeline import geometry as g
from kassa_pipeline.config import ObjectsConfig
from kassa_pipeline.layout import LayoutTimeline
from kassa_pipeline.models import CropEncoder, OpenVocabDetector, TrainedHeads
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
    res = ObjectsResult()
    frames = video.every(fps)
    for f in frames:
        sec = int(f.t)
        lay = layouts.at(f.t)
        h, w = f.norm.shape[:2]

        # 3a. candidates on the desk region (cropped, so small items get more pixels)
        if lay.workstation is not None:
            region = g.expand(lay.workstation, cfg.desk_crop_pad, w, h)
            crop = g.crop(f.norm, region)
            cands = []
            if crop is not None:
                dets = detector([crop], cfg.prompt, cfg.box_threshold, cfg.text_threshold)[0]
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

            # 3b. verify category from a crop with context
            ctx = [g.crop(f.norm, g.expand(b, 0.4, w, h), min_size=12) for _, _, b in cands]
            valid = [(c, x) for c, x in zip(cands, ctx) if x is not None]
            labels = _classify(encoder, heads, "object", OBJECT_PROMPTS, [x for _, x in valid])
            wrists = _wrists_near(people, f.t)
            for (cat0, sc, box), (cat, prob) in zip((c for c, _ in valid), labels):
                if cat in IGNORED or prob < cfg.min_category_score:
                    continue
                cx, cy = g.center(box)
                desk = lay.desk or lay.workstation
                on_desk = g.contains(g.expand(desk, 0.05), (cx, cy))
                in_hand = any(
                    g.gap(box, (wx - 1, wy - 1, wx + 1, wy + 1)) < cfg.in_hand_px for wx, wy in wrists
                )
                res.objects.setdefault(sec, []).append(
                    ObjObs(f.t, cat, box, sc * prob, on_desk, in_hand)
                )

        # 3c. laptop screen state
        res.screen[sec] = _state(f.norm, lay.laptop, 0.3, encoder, heads, "screen", SCREEN_PROMPTS)
        # 3d. safe door state
        res.safe[sec] = _state(f.norm, lay.safe, 0.5, encoder, heads, "safe", SAFE_PROMPTS)
    return res


def _state(img, box, pad, encoder, heads, task, prompts) -> StateObs:
    if box is None:
        return StateObs("unknown", 0.0)
    h, w = img.shape[:2]
    c = g.crop(img, g.expand(box, pad, w, h), min_size=12)
    if c is None:
        return StateObs("unknown", 0.0)
    (cls, p), = _classify(encoder, heads, task, prompts, [c])
    return StateObs(cls, p)
