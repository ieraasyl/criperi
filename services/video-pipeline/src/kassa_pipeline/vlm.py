"""Qwen3.5-9B inside the Modal container. The scan only needs `ask_many`, so tests use a fake.

The VLM answers closed questions about what is visible. It never selects or words a rule.
"""

import hashlib
import io
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

CHECK_PROMPT = """You are checking a top-down surveillance video frame of a bank cashier's workplace.
{context}Look only at what is visible. If it is not clearly visible, answer "unknown".
Question: {question}
Reply with ONLY this JSON: {{"answer": "yes" | "no" | "unknown", "confidence": 0.0-1.0, "observation": "<one short sentence about what is visible>"}}"""


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


def ask_check(vlm, images: list[np.ndarray], question: str, context: str = "") -> Answer:
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

    def __init__(self, model_id: str = "Qwen/Qwen3.5-9B", cache_path: str | Path | None = None, device: str = "cuda", quant: str | None = None):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        t = time.time()
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id)
        if quant == "nf4":
            from transformers import BitsAndBytesConfig

            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                llm_int8_skip_modules=["visual", "lm_head"],
            )
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id, dtype=torch.bfloat16, device_map={"": 0}, quantization_config=bnb
            ).eval()
        elif quant is None:
            self.model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16, device_map=device).eval()
        else:
            raise ValueError(f"unknown quant: {quant}")
        self.calls = 0
        self.hits = 0
        self.skipped = 0  # items left unasked because the deadline passed
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

    def ask_many(self, items: list[tuple[list[np.ndarray], str]], max_new_tokens: int = 100, batch: int = 4,
                 deadline: float | None = None) -> list[str | None]:
        """Several (images, prompt) requests, generated `batch` at a time. Cached per request.
        Once `deadline` (epoch seconds) passes, no new batch starts: unanswered items come back as None."""
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
            if deadline is not None and time.time() >= deadline:
                self.skipped += len(todo) - s
                break
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


LOCAL_MODELS = ("Qwen/Qwen3-VL-4B-Instruct", "Qwen/Qwen3-VL-2B-Instruct")
MIN_FREE_GB_4B = 3.2


def pick_local_model(free_gb: float) -> str:
    """The 4B model when it fits in the free VRAM, else the 2B."""
    return LOCAL_MODELS[0] if free_gb >= MIN_FREE_GB_4B else LOCAL_MODELS[1]


def _free_vram_gb() -> float:
    import torch

    return torch.cuda.mem_get_info()[0] / 2**30


def local_vlm(cache_path: str | Path | None, free_gb_fn=None) -> "QwenVLM":
    model_id = pick_local_model((free_gb_fn or _free_vram_gb)())
    if cache_path is not None:  # the answer cache key ignores the model, so keep one file per model
        p = Path(cache_path)
        cache_path = p.with_name(f"{p.stem}-{model_id.rsplit('/', 1)[-1].lower()}{p.suffix}")
    return QwenVLM(model_id=model_id, quant="nf4", cache_path=cache_path, device="cuda")
