"""Client for the Modal VLM service plus the per-frame state prompt.

The VLM only reports what it sees in a frame (closed fields, 'unknown' allowed). Timing, tracking
and the decision stay in the CV pipeline; the VLM never selects or words a rule.
"""

import json
import re

import cv2
import numpy as np

from kassa_pipeline import geometry as g

STATE_PROMPT = """You are analysing a top-down surveillance image of a bank cashier's workplace.
Image 1 is the whole scene. Image 2 is the cashier's desk area, enlarged.
Look only at what is visible. If something is not clearly visible, answer "unknown".
Reply with ONLY a JSON object with exactly these keys:
{
 "cash_on_desk": true/false/"unknown"        (banknotes or a stack of money lying on the desk surface; NOT held in a hand),
 "cash_bag_on_desk": true/false/"unknown"    (a green or dark cash bag / money bag on the desk),
 "personal_item_on_desk": true/false/"unknown" (a coat, jacket, handbag, purse or backpack lying on the desk),
 "food_or_drink": true/false/"unknown"       (a cup, mug, bottle or food on the desk or in the cashier's hand),
 "phone_visible": true/false/"unknown"       (a mobile phone on the desk or in someone's hand),
 "laptop_screen": "work_app" | "lock_screen" | "off" | "not_visible",
 "safe_door": "open" | "closed" | "not_visible"   (the grey metal safe/cabinet),
 "cashier_state": "seated_at_desk" | "standing" | "not_in_room",
 "other_people_in_room": integer              (people other than the cashier inside the room, not behind the glass wall),
 "other_person_on_cashier_side": true/false   (someone other than the cashier standing behind the desk next to the cashier),
 "room_door": "open" | "closed" | "not_visible"
}"""


def to_jpeg(rgb: np.ndarray, quality: int = 92) -> bytes:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def desk_view(rgb: np.ndarray, workstation: g.Box | None, min_width: int = 640) -> np.ndarray:
    """The desk region enlarged: small objects (banknotes) need the pixels."""
    h, w = rgb.shape[:2]
    c = g.crop(rgb, g.expand(workstation, 0.25, w, h)) if workstation else None
    if c is None:
        return rgb
    s = max(1.0, min_width / c.shape[1])
    return cv2.resize(c, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)


def parse_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def ask_many(jobs: list[dict], batch: int = 24) -> list[dict]:
    import modal

    cls = modal.Cls.from_name("kassa-vlm", "VLM")()
    out = []
    for i in range(0, len(jobs), batch):
        out += cls.ask.remote(jobs[i : i + batch])
    return out
