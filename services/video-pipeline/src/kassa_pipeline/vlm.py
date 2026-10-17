"""VLM access. The model runs inside the Modal container (Qwen3.5-9B); detectors only see the
`VLM` protocol, so tests use a fake and a missing VLM degrades to CV-only results.

The VLM answers closed questions about what is visible. It never selects or words a rule.
"""

import hashlib
import io
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

CHECK_PROMPT = """You are checking a top-down surveillance video frame of a bank cashier's workplace.
{context}Look only at what is visible. If it is not clearly visible, answer "unknown".
Question: {question}
Reply with ONLY this JSON: {{"answer": "yes" | "no" | "unknown", "confidence": 0.0-1.0, "observation": "<one short sentence about what is visible>"}}"""


class VLM(Protocol):
    def ask(self, images: list[np.ndarray], prompt: str, max_new_tokens: int = 100) -> str: ...


@dataclass
class Answer:
    answer: bool | None  # True = yes, False = no, None = unknown / unparseable
    confidence: float
    observation: str
    raw: str = ""


def parse_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def ask_check(vlm: VLM, images: list[np.ndarray], question: str, context: str = "") -> Answer:
    prompt = CHECK_PROMPT.format(question=question, context=(context + "\n") if context else "")
    raw = vlm.ask(images, prompt, max_new_tokens=90)
    d = parse_json(raw) or {}
    a = str(d.get("answer", "")).strip().lower()
    conf = d.get("confidence", 0.5)
    try:
        conf = max(0.0, min(1.0, float(conf)))
    except (TypeError, ValueError):
        conf = 0.5
    ans = True if a == "yes" else False if a == "no" else None
    return Answer(ans, conf if ans is not None else 0.0, str(d.get("observation", ""))[:240], raw)


def _jpeg(rgb: np.ndarray, q: int = 92) -> bytes:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
    return buf.tobytes()


class QwenVLM:
    """Qwen3.5-9B via transformers. Responses are cached on disk keyed by (prompt, images), so
    re-running detectors over the same video costs no GPU time."""

    def __init__(self, model_id: str = "Qwen/Qwen3.5-9B", cache_path: str | Path | None = None, device: str = "cuda"):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        t = time.time()
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16, device_map=device).eval()
        self.calls = 0
        self.hits = 0
        self.gpu_seconds = 0.0
        self._cache: dict[str, str] = {}
        self._cache_path: Path | None = None
        if cache_path:
            self.use_cache(cache_path)
        print(f"VLM {model_id} ready in {time.time() - t:.0f}s")

    def use_cache(self, path: str | Path | None):
        self.flush()
        self._cache_path = Path(path) if path else None
        self._cache = {}
        if self._cache_path and self._cache_path.exists():
            self._cache = json.loads(self._cache_path.read_text())

    def flush(self):
        if self._cache_path and self._cache:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(json.dumps(self._cache))

    def ask(self, images: list[np.ndarray], prompt: str, max_new_tokens: int = 100) -> str:
        import torch
        from PIL import Image

        jpegs = [_jpeg(im) for im in images]
        key = hashlib.sha1(prompt.encode() + b"".join(hashlib.sha1(j).digest() for j in jpegs)).hexdigest()
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        t = time.time()
        pil = [Image.open(io.BytesIO(j)).convert("RGB") for j in jpegs]
        content = [{"type": "image", "image": im} for im in pil] + [{"type": "text", "text": prompt}]
        inputs = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt", enable_thinking=False,
        ).to(self.device)
        with torch.inference_mode():
            gen = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        text = self.processor.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        self._cache[key] = text
        self.calls += 1
        self.gpu_seconds += time.time() - t
        return text
