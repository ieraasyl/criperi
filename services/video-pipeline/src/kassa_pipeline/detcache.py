"""Disk cache for raw open-vocabulary detections, so thresholds can be re-applied offline."""
from __future__ import annotations

import hashlib
import json
import os
from importlib.metadata import version
from pathlib import Path
from typing import Sequence

import numpy as np

from kassa_pipeline.models import Detection, format_prompt


class CachedDetector:
    """Caches per image at a low box threshold; text_threshold is in the key because in HF Grounding DINO the box threshold only filters boxes while labels depend on text_threshold."""

    def __init__(self, inner, cache_dir: str | Path, model_id: str, floor: float = 0.1):
        self.inner = inner
        self.dir = Path(cache_dir)
        self.model_id = model_id
        self.floor = floor
        self.hits = 0
        self.misses = 0

    def _key(self, img: np.ndarray, prompt: str, text_threshold: float) -> str:
        h = hashlib.sha1()
        h.update(f"{self.model_id}|tf{version('transformers')}".encode())
        h.update(f"|{img.shape}|{img.dtype}|".encode())
        h.update(np.ascontiguousarray(img).tobytes())
        h.update(f"|{prompt}|{round(text_threshold, 4)}".encode())
        return h.hexdigest()

    def __call__(
        self,
        images: list[np.ndarray],
        phrases: Sequence[str],
        box_threshold: float,
        text_threshold: float,
        batch: int = 1,
    ) -> list[list[Detection]]:
        prompt = format_prompt(phrases)
        out: list[list[Detection]] = []
        for img in images:
            key = self._key(img, prompt, text_threshold)
            path = self.dir / key[:2] / f"{key}.json"
            entry = None
            if path.exists():
                try:
                    entry = json.loads(path.read_text(encoding="utf8"))
                    if box_threshold < entry["computed_at"]:
                        entry = None
                    else:  # validate shape now so a partial file is a miss, not a crash
                        entry["detections"] = [
                            (d["label"], float(d["score"]), tuple(d["box"])) for d in entry["detections"]
                        ]
                except (ValueError, KeyError, TypeError):
                    entry = None  # corrupt or partial file: recompute
            if entry is None:
                self.misses += 1
                low = min(self.floor, box_threshold)
                dets = self.inner([img], phrases, low, text_threshold, batch)[0]
                payload = {
                    "computed_at": low,
                    "detections": [
                        {"label": d.label, "score": d.score, "box": list(d.box)} for d in dets
                    ],
                }
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(f".{os.getpid()}.tmp")
                tmp.write_text(json.dumps(payload), encoding="utf8")
                os.replace(tmp, path)
                entry = {"detections": [(d.label, d.score, tuple(d.box)) for d in dets]}
            else:
                self.hits += 1
            out.append(
                [
                    Detection(label, score, box)
                    for label, score, box in entry["detections"]
                    if score >= box_threshold
                ]
            )
        return out
