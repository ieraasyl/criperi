"""Qwen3.5-9B inside the Modal container. The scan only needs `ask_many`, so tests use a fake.

The VLM answers closed questions about what is visible. It never selects or words a rule.
"""

import hashlib
import io
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np


def parse_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


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

    @staticmethod
    def _prepare(images: list[np.ndarray], prompt: str):
        from PIL import Image

        jpegs = [_jpeg(im) for im in images]
        key = hashlib.sha1(prompt.encode() + b"".join(hashlib.sha1(j).digest() for j in jpegs)).hexdigest()
        pil = [Image.open(io.BytesIO(j)).convert("RGB") for j in jpegs]
        content = [{"type": "image", "image": im} for im in pil] + [{"type": "text", "text": prompt}]
        return key, [{"role": "user", "content": content}]

    def _generate(self, convs: list[list[dict]], max_new_tokens: int) -> list[str]:
        import torch

        self.processor.tokenizer.padding_side = "left"  # batched generation continues from the right edge
        inputs = self.processor.apply_chat_template(
            convs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
            processor_kwargs={"padding": True}, enable_thinking=False,
        ).to(self.device)
        with torch.inference_mode():
            gen = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return [t.strip() for t in self.processor.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)]

    def ask(self, images: list[np.ndarray], prompt: str, max_new_tokens: int = 100) -> str:
        return self.ask_many([(images, prompt)], max_new_tokens)[0]

    def ask_many(self, items: list[tuple[list[np.ndarray], str]], max_new_tokens: int = 100, batch: int = 4) -> list[str]:
        """Several (images, prompt) requests, generated `batch` at a time. Cached per request."""
        out: list[str | None] = [None] * len(items)
        todo = []
        for i, (images, prompt) in enumerate(items):
            key, conv = self._prepare(images, prompt)
            if key in self._cache:
                self.hits += 1
                out[i] = self._cache[key]
            else:
                todo.append((i, key, conv))
        for s in range(0, len(todo), batch):
            chunk = todo[s : s + batch]
            t = time.time()
            try:
                texts = self._generate([c for _, _, c in chunk], max_new_tokens)
            except Exception:  # batching failed (e.g. memory): fall back to one at a time
                texts = [self._generate([c], max_new_tokens)[0] for _, _, c in chunk]
            self.gpu_seconds += time.time() - t
            for (i, key, _), text in zip(chunk, texts):
                self._cache[key] = text
                out[i] = text
                self.calls += 1
        return out  # type: ignore[return-value]
