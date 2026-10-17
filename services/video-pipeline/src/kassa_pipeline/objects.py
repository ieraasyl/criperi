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
    "device": "a photo of a laptop, tablet, telephone, banknote counting machine or office equipment",
    "hand": "a photo of a person's hand, fingers or arm",
    "surface": "a photo of an empty wooden desk surface",
}
IGNORED = {"paper", "device", "surface", "hand"}

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


# ---------------------------------------------------------------- laptop screen from pixels
# Zero-shot SigLIP and a VLM both failed on the screen state (X-29 lock screen read as "work_app").
# Plain pixel statistics separate the states: a work window is mostly bright, a lock screen and an
# off screen are dark, and a lock screen has a blue tint.

WORK_BRIGHT_SHARE = 0.30  # share of bright pixels in the screen area
LOCK_BLUE_MINUS_RED = 5.0  # lock screens (landscape wallpaper) lean blue; an off screen does not


def screen_features(rgb: np.ndarray, laptop: g.Box, scene_gain: float = 1.0) -> dict | None:
    """Bright-pixel share, mean brightness and blue-minus-red of the laptop's screen area."""
    import cv2

    x1, y1, x2, y2 = laptop
    h, w = y2 - y1, x2 - x1
    area = g.crop(rgb, (x1 + 0.1 * w, y1, x2 - 0.1 * w, y1 + 0.65 * h), min_size=6)
    if area is None:
        return None
    a = np.clip(area.astype(np.float32) * scene_gain, 0, 255)  # undo scene-wide lighting changes
    v = a.max(axis=2)
    return {
        "bright": float((v > 170).mean()),
        "v_mean": float(v.mean()),
        "blue_minus_red": float(a[..., 2].mean() - a[..., 0].mean()),
    }


def classify_screen(f: dict | None) -> StateObs:
    if f is None:
        return StateObs("unknown", 0.0)
    if f["bright"] >= WORK_BRIGHT_SHARE:
        return StateObs("work_screen", min(1.0, 0.6 + f["bright"]))
    if f["blue_minus_red"] >= LOCK_BLUE_MINUS_RED:
        return StateObs("locked", 0.7)
    return StateObs("off", 0.7)


def pixel_screen_states(video: Video, layouts: LayoutTimeline, fps: float = 1.0) -> dict[int, StateObs]:
    """Per-second laptop screen state; CPU only, so it never invalidates the expensive stages."""
    frames = video.every(fps)
    base = float(np.median([f.luma for f in video.frames])) or 1.0
    out: dict[int, StateObs] = {}
    for f in frames:
        lay = layouts.at(f.t)
        gain = base / max(f.luma, 1.0)  # lights off / auto-exposure: bring the frame back to the median
        out[int(f.t)] = classify_screen(screen_features(f.rgb, lay.laptop, gain)) if lay.laptop else StateObs("unknown", 0.0)
    return out
