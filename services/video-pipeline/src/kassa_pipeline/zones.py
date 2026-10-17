"""Zone crops and closed per-zone questions on Stage-1 layouts and records (leaf: no pipeline import)."""

import math
from pathlib import Path

import cv2
import numpy as np

from . import geometry as g
from .video import read_frame_at
from .vlm import ask_check

YES_NO = {"yes": "yes", "no": "no"}

# state -> (zone, question, answer map)
QUESTIONS: dict[str, tuple[str, str, dict]] = {
    "laptop_lid": ("laptop", "Is the laptop lid fully closed?", {"yes": "closed", "no": "open"}),
    "screen": (
        "laptop",
        "Is the laptop screen showing working content (not black and not a lock screen)?",
        {"yes": "work", "no": "off"},
    ),
    "safe_door": ("safe", "Is the door of the grey metal safe open?", {"yes": "open", "no": "closed"}),
    "desk_cash": ("desk", "Are there banknotes lying on the desk?", YES_NO),
    "desk_bag": ("desk", "Is there a cash bag or money pouch lying on the desk?", YES_NO),
    "desk_personal": (
        "desk",
        "Is there a personal item (handbag, wallet, phone, cup, food or clothing) on the desk?",
        YES_NO,
    ),
    "desk_docs": ("desk", "Are there paper documents or forms lying on the desk?", YES_NO),
}

OCCLUSION_SHARE = 0.5
UNKNOWN = "unknown"


def zone_box(layout: dict, zone: str) -> g.Box | None:
    """Box of `zone` in a Stage-1 layout dict; "door" is the first door."""
    if zone == "door":
        b = (layout.get("doors") or [None])[0]
    elif zone in ("laptop", "safe", "desk"):
        b = layout.get(zone)
    else:
        b = None
    return tuple(float(x) for x in b) if b else None


def zone_crop(rgb: np.ndarray, box: g.Box, min_side: int = 448) -> np.ndarray | None:
    """Zone crop grown by 0.15 and upscaled (cubic) so the short side is at least `min_side`."""
    h, w = rgb.shape[:2]
    c = g.crop(rgb, g.expand(box, 0.15, w, h))
    if c is None:
        return None
    short = min(c.shape[:2])
    if short >= min_side:
        return c
    k = min_side / short
    return cv2.resize(c, (math.ceil(c.shape[1] * k), math.ceil(c.shape[0] * k)), interpolation=cv2.INTER_CUBIC)


def occluded(box: g.Box, people_boxes: list) -> bool:
    """Some person's box covers more than OCCLUSION_SHARE of the zone."""
    return any(g.inside_share(box, tuple(p)) > OCCLUSION_SHARE for p in people_boxes)


def despeckle(series: list[str]) -> list[str]:
    """One flip out and back between two equal known neighbours is flicker: mark it unknown."""
    out = list(series)
    for i in range(1, len(series) - 1):
        a, b, c = series[i - 1], series[i], series[i + 1]
        if a != UNKNOWN and a == c and b != UNKNOWN and b != a:
            out[i] = UNKNOWN
    return out


def ask_zone_state(
    video_path: str | Path,
    duration: float,
    layout: dict,
    people_boxes: list,
    vlm,
    state: str,
    s: int,
    rgb: np.ndarray | None = None,
) -> str | None:
    """Ask the closed question for `state` at second `s`; None when skipped or unanswered."""
    t = s + 0.5
    zone, question, mapping = QUESTIONS[state]
    box = zone_box(layout, zone)
    if box is None or occluded(box, people_boxes):
        return None
    if rgb is None:
        try:
            rgb = read_frame_at(video_path, min(t, duration - 0.05))
        except ValueError:
            return None
    c = zone_crop(rgb, box)
    if c is None:
        return None
    ans = ask_check(vlm, [c], question).answer
    return None if ans is None else mapping["yes" if ans else "no"]
