"""Dataset for the cash / ID-card detector: desk crops from the training clips, in COCO format.

Frames are sampled once per second plus every Stage 1 frame where a cashier hand and a visitor hand
are close (hand-offs, where ID cards and money change hands). Near-identical crops are dropped. Each
crop is the desk area grown by `pad` and upscaled `scale` times, because a card is ~12x10 px and a
banknote ~25x10 px in the 464x848 video.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from kassa_pipeline import geometry as g
from kassa_pipeline.vlm_scan import Stage1

CLASSES = ["cash", "cash_bag", "id_card"]  # category ids 1..3 in COCO files
HANDOFF_PX = 60.0  # cashier and visitor hand points closer than this = a possible hand-off


@dataclass
class Sample:
    image_id: str  # "<clip>_<t>"
    clip: str
    t: float
    region: tuple[float, float, float, float]  # desk crop in video pixels
    scale: float
    handoff: bool


def handoff_times(s1: Stage1) -> list[float]:
    out = []
    for f in s1.frames:
        cash = [h for p in f["people"] if p["role"] == "cashier" for h in p["hands"].values() if h]
        vis = [h for p in f["people"] if p["role"] != "cashier" for h in p["hands"].values() if h]
        if any(np.hypot(a[0] - b[0], a[1] - b[1]) < HANDOFF_PX for a in cash for b in vis):
            out.append(f["t"])
    return out


def desk_region(lay: dict, w: int, h: int, pad: float) -> tuple[float, float, float, float] | None:
    ws = lay.get("workstation") or lay.get("desk")
    return g.expand(tuple(ws), pad, w, h) if ws else None


def sample_clip(video_path: Path, clip: str, s1: Stage1, pad: float = 0.25, scale: float = 2.0,
                min_diff: float = 3.0) -> list[tuple[Sample, np.ndarray]]:
    """(sample, RGB crop) pairs: 1 fps + hand-off frames, near-duplicates removed."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    dur = n / fps
    hand = set(round(t, 2) for t in handoff_times(s1))
    times = sorted(set([round(float(t), 2) for t in np.arange(0.0, dur - 0.1, 1.0)]) | hand)
    out, last = [], None
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, bgr = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        region = desk_region(s1.layout(t), w, h, pad)
        if region is None:
            continue
        crop = g.crop(rgb, region, min_size=20)
        if crop is None:
            continue
        small = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY), (64, 64)).astype(np.float32)
        is_hand = round(t, 2) in hand
        if last is not None and float(np.abs(small - last).mean()) < (min_diff / 2 if is_hand else min_diff):
            continue
        last = small
        big = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        out.append((Sample(f"{clip}_{t:06.2f}", clip, t, tuple(float(v) for v in region), scale, is_hand), big))
    cap.release()
    return out


def coco(samples: list[Sample], sizes: dict[str, tuple[int, int]], boxes: dict[str, list[dict]]) -> dict:
    """COCO detection file. boxes[image_id] = [{"bbox": [x, y, w, h], "cls": "cash", "score": ..., ...}]."""
    images, anns, k = [], [], 1
    for i, s in enumerate(samples, 1):
        w, h = sizes[s.image_id]
        images.append({"id": i, "file_name": f"{s.image_id}.jpg", "width": w, "height": h, "clip": s.clip, "t": s.t,
                       "region": s.region, "scale": s.scale, "handoff": s.handoff})
        for b in boxes.get(s.image_id, []):
            x, y, bw, bh = b["bbox"]
            anns.append({"id": k, "image_id": i, "category_id": CLASSES.index(b["cls"]) + 1, "bbox": [x, y, bw, bh],
                         "area": bw * bh, "iscrowd": 0, **{kk: v for kk, v in b.items() if kk not in ("bbox", "cls")}})
            k += 1
    return {"images": images, "annotations": anns, "categories": [{"id": i + 1, "name": c} for i, c in enumerate(CLASSES)]}


def split_by_clip(images: list[dict], val_clips: set[str]) -> tuple[list[int], list[int]]:
    tr = [im["id"] for im in images if im["clip"] not in val_clips]
    va = [im["id"] for im in images if im["clip"] in val_clips]
    return tr, va


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False))
