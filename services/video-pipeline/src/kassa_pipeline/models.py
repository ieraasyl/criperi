"""Thin wrappers around the pretrained models. Loaded lazily and cached per process.

All weights are open source with commercial-use licenses:
RT-DETRv2 (Apache-2.0), Grounding DINO (Apache-2.0), SigLIP (Apache-2.0), RTMPose (Apache-2.0).
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import os

import cv2
import numpy as np
import torch
from PIL import Image

# python.org builds of Python ship without a CA bundle; rtmlib downloads weights with urllib.
try:
    import certifi

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except ImportError:
    pass

RTMPOSE_URLS = {
    "rtmpose-m": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
    "rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip",
    "rtmpose-s": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
    "rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.zip",
}

# COCO-17 keypoint indices used downstream
NOSE, L_WRIST, R_WRIST, L_HIP, R_HIP, L_KNEE, R_KNEE = 0, 9, 10, 11, 12, 13, 14


def pick_device(pref: str = "auto") -> str:
    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class Detection:
    label: str
    score: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in the input image


def _pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(img)


class PersonDetector:
    def __init__(self, model_id: str, device: str):
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForObjectDetection.from_pretrained(model_id).to(device).eval()
        self.person_ids = [i for i, n in self.model.config.id2label.items() if n == "person"]

    @torch.inference_mode()
    def __call__(self, images: list[np.ndarray], threshold: float, batch: int = 8):
        out: list[list[Detection]] = []
        for s in range(0, len(images), batch):
            chunk = [_pil(im) for im in images[s : s + batch]]
            inputs = self.processor(images=chunk, return_tensors="pt").to(self.device)
            res = self.processor.post_process_object_detection(
                self.model(**inputs),
                target_sizes=[im.size[::-1] for im in chunk],
                threshold=threshold,
            )
            for r in res:
                dets = [
                    Detection("person", float(sc), tuple(float(v) for v in bx))
                    for sc, lb, bx in zip(r["scores"], r["labels"], r["boxes"])
                    if int(lb) in self.person_ids
                ]
                out.append(dets)
        return out


class OpenVocabDetector:
    def __init__(self, model_id: str, device: str):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()

    @torch.inference_mode()
    def __call__(
        self,
        images: list[np.ndarray],
        prompt: str,
        box_threshold: float,
        text_threshold: float,
        batch: int = 1,  # batching pads the text and is slower than single images on MPS
    ) -> list[list[Detection]]:
        out: list[list[Detection]] = []
        for s in range(0, len(images), batch):
            chunk = [_pil(im) for im in images[s : s + batch]]
            inputs = self.processor(
                images=chunk, text=[prompt] * len(chunk), return_tensors="pt", padding=True
            ).to(self.device)
            res = self.processor.post_process_grounded_object_detection(
                self.model(**inputs),
                inputs.input_ids,
                threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=[im.size[::-1] for im in chunk],
            )
            for r in res:
                labels = r.get("text_labels", r["labels"])
                out.append(
                    [
                        Detection(str(lb), float(sc), tuple(float(v) for v in bx))
                        for lb, sc, bx in zip(labels, r["scores"], r["boxes"])
                    ]
                )
        return out


class CropEncoder:
    """SigLIP image/text encoder: embeddings for trained heads, text prompts for zero-shot."""

    def __init__(self, model_id: str, device: str):
        from transformers import AutoModel, AutoProcessor

        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(device).eval()
        self._text_cache: dict[tuple[str, ...], torch.Tensor] = {}

    @torch.inference_mode()
    def embed(self, crops: list[np.ndarray], batch: int = 32) -> np.ndarray:
        if not crops:
            return np.zeros((0, self.model.config.vision_config.hidden_size), np.float32)
        feats = []
        for s in range(0, len(crops), batch):
            inputs = self.processor(
                images=[_pil(c) for c in crops[s : s + batch]], return_tensors="pt"
            ).to(self.device)
            f = self.model.get_image_features(**inputs)
            f = getattr(f, "pooler_output", f)
            feats.append(torch.nn.functional.normalize(f, dim=-1).float().cpu())
        return torch.cat(feats).numpy()

    @torch.inference_mode()
    def _text(self, texts: tuple[str, ...]) -> torch.Tensor:
        if texts not in self._text_cache:
            inputs = self.processor(text=list(texts), padding="max_length", return_tensors="pt").to(
                self.device
            )
            f = self.model.get_text_features(**inputs)
            f = getattr(f, "pooler_output", f)
            self._text_cache[texts] = torch.nn.functional.normalize(f, dim=-1).float().cpu()
        return self._text_cache[texts]

    def zero_shot(self, embeddings: np.ndarray, texts: list[str]) -> np.ndarray:
        """Softmax over prompts for already-computed image embeddings. Returns (N, len(texts))."""
        if len(embeddings) == 0:
            return np.zeros((0, len(texts)), np.float32)
        t = self._text(tuple(texts))
        scale = float(self.model.logit_scale.exp())
        logits = torch.from_numpy(embeddings) @ t.T * scale
        return torch.softmax(logits, dim=-1).numpy()


class PoseEstimator:
    def __init__(self, name: str):
        from rtmlib import RTMPose

        self.model = RTMPose(RTMPOSE_URLS[name], model_input_size=(192, 256), backend="onnxruntime")

    def __call__(self, rgb: np.ndarray, boxes: list[tuple]) -> tuple[np.ndarray, np.ndarray]:
        """Returns keypoints (N, 17, 2) and scores (N, 17) in image coordinates."""
        if not boxes:
            return np.zeros((0, 17, 2)), np.zeros((0, 17))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return self.model(bgr, bboxes=[list(b) for b in boxes])


class TrainedHeads:
    """Optional logistic-regression heads trained by tools/train_heads.py on SigLIP embeddings."""

    def __init__(self, heads_dir: str | Path):
        import joblib

        self.heads = {}
        d = Path(heads_dir)
        if d.is_dir():
            for p in d.glob("*.joblib"):
                self.heads[p.stem] = joblib.load(p)

    def has(self, task: str) -> bool:
        return task in self.heads

    def predict(self, task: str, embeddings: np.ndarray) -> tuple[list[str], np.ndarray]:
        clf = self.heads[task]
        return list(clf.classes_), clf.predict_proba(embeddings)


@lru_cache(maxsize=None)
def person_detector(model_id: str, device: str) -> PersonDetector:
    return PersonDetector(model_id, device)


@lru_cache(maxsize=None)
def open_vocab_detector(model_id: str, device: str) -> OpenVocabDetector:
    return OpenVocabDetector(model_id, device)


@lru_cache(maxsize=None)
def crop_encoder(model_id: str, device: str) -> CropEncoder:
    return CropEncoder(model_id, device)


@lru_cache(maxsize=None)
def pose_estimator(name: str) -> PoseEstimator:
    return PoseEstimator(name)


@lru_cache(maxsize=None)
def trained_heads(heads_dir: str) -> TrainedHeads:
    return TrainedHeads(heads_dir)
