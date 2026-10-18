"""Object classes and states used by the extract stage: SigLIP prompts for desk objects and the safe
door, and the laptop screen state from pixel statistics.
"""

from dataclasses import dataclass

import numpy as np

from kassa_pipeline import geometry as g

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
SAFE_PROMPTS = {
    "closed": "a closed grey metal safe",
    "open": "an open safe with its door swung open",
}


@dataclass
class StateObs:
    state: str  # work_screen | locked | off | unknown   or   open | closed | unknown
    prob: float


def _classify(encoder, heads, task: str, prompts: dict[str, str], crops: list) -> list[tuple[str, float]]:
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


def _state(img, box, pad, encoder, heads, task, prompts) -> StateObs:
    if box is None:
        return StateObs("unknown", 0.0)
    h, w = img.shape[:2]
    c = g.crop(img, g.expand(box, pad, w, h), min_size=12)
    if c is None:
        return StateObs("unknown", 0.0)
    (cls, p), = _classify(encoder, heads, task, prompts, [c])
    return StateObs(cls, p)


# ---------------------------------------------------------------- safe door from edges
# The camera is fixed within a segment, so an open door shows as a change of the safe's edge structure
# against its usual (closed, most of the time) look. Recorded as a number; not yet turned into a state.


def safe_signature(img: np.ndarray, box: g.Box | None) -> np.ndarray | None:
    """Edge-structure signature of the safe crop, insensitive to brightness shifts (like video._structure)."""
    import cv2

    c = None if box is None else g.crop(img, box, min_size=12)
    if c is None:
        return None
    gray = cv2.GaussianBlur(cv2.resize(cv2.cvtColor(c, cv2.COLOR_RGB2GRAY), (32, 32), interpolation=cv2.INTER_AREA), (3, 3), 0)
    m = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1)).flatten()
    m -= m.mean()
    return m / (np.linalg.norm(m) + 1e-6)


def safe_reference(sigs: list[np.ndarray]) -> np.ndarray | None:
    if not sigs:
        return None
    m = np.median(np.stack(sigs), axis=0)
    return m / (np.linalg.norm(m) + 1e-6)


def safe_change(img: np.ndarray, box: g.Box | None, ref: np.ndarray | None) -> float | None:
    """0 = looks like the reference, up to 2 = opposite edges."""
    sig = safe_signature(img, box) if ref is not None else None
    return None if sig is None else float(1.0 - sig @ ref)


# ---------------------------------------------------------------- laptop screen from pixels
# Zero-shot SigLIP and a VLM both failed on the screen state (X-29 lock screen read as "work_app").
# Plain pixel statistics separate the states: a work window is mostly bright, a lock screen and an
# off screen are dark, and a lock screen has a blue tint.

WORK_BRIGHT_SHARE = 0.30  # share of bright pixels in the screen area
LOCK_BLUE_MINUS_RED = 5.0  # lock screens (landscape wallpaper) lean blue; an off screen does not


def screen_area(laptop: g.Box) -> g.Box:
    """The screen part of the laptop box: the upper 65 %, without the side bezels."""
    x1, y1, x2, y2 = laptop
    h, w = y2 - y1, x2 - x1
    return (x1 + 0.1 * w, y1, x2 - 0.1 * w, y1 + 0.65 * h)


def screen_features(rgb: np.ndarray, laptop: g.Box, scene_gain: float = 1.0) -> dict | None:
    """Bright-pixel share, mean brightness and blue-minus-red of the laptop's screen area."""
    area = g.crop(rgb, screen_area(laptop), min_size=6)
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
