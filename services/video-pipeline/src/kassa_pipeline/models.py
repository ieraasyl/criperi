"""Thin wrappers around the pretrained models. Loaded lazily and cached per process.

All weights are open source with commercial-use licenses:
RT-DETRv2 (Apache-2.0), Grounding DINO (Apache-2.0), SigLIP (Apache-2.0), RTMPose/RTMW (Apache-2.0).
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence
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

_MMPOSE = "https://download.openmmlab.com/mmpose/v1/projects/"
RTMPOSE_URLS = {  # name -> (onnx zip, model input size (w, h))
    "rtmpose-m": (_MMPOSE + "rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip", (192, 256)),
    "rtmpose-s": (_MMPOSE + "rtmposev1/onnx_sdk/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.zip", (192, 256)),
    "rtmpose-x": (_MMPOSE + "rtmposev1/onnx_sdk/rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.zip", (288, 384)),
    # COCO-WholeBody: body 0-16 (same indices as COCO-17), left hand 91-111, right hand 112-132
    "rtmw-x": (_MMPOSE + "rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip", (288, 384)),
}

# COCO-17 keypoint indices used downstream
NOSE, L_WRIST, R_WRIST, L_HIP, R_HIP, L_KNEE, R_KNEE = 0, 9, 10, 11, 12, 13, 14
L_HAND, R_HAND = range(91, 112), range(112, 133)  # whole-body models only


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


def format_prompt(phrases: Sequence[str]) -> str:
    """Grounding DINO text: lowercase phrases separated by ". " and ended with "."."""
    parts = [p.strip().lower() for p in phrases]
    return ". ".join(p for p in parts if p) + "."


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
        phrases: Sequence[str],
        box_threshold: float,
        text_threshold: float,
        batch: int = 1,  # batching pads the text and is slower than single images on MPS
    ) -> list[list[Detection]]:
        prompt = format_prompt(phrases)
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
    def __init__(self, name: str, device: str = "cpu"):
        from rtmlib import RTMPose

        url, size = RTMPOSE_URLS[name]
        self.n_kpts = 133 if name.startswith("rtmw") else 17
        dev = "cuda" if device == "cuda" else "cpu"
        if dev == "cuda":
            try:  # onnxruntime-gpu finds the CUDA/cuDNN libraries that the torch wheels installed
                import onnxruntime

                if hasattr(onnxruntime, "preload_dlls"):
                    onnxruntime.preload_dlls()
                if "CUDAExecutionProvider" not in onnxruntime.get_available_providers():
                    dev = "cpu"
            except Exception:
                dev = "cpu"
        self.device = dev
        self.model = RTMPose(url, model_input_size=size, backend="onnxruntime", device=dev)

    def __call__(self, rgb: np.ndarray, boxes: list[tuple]) -> tuple[np.ndarray, np.ndarray]:
        """Returns keypoints (N, K, 2) and scores (N, K) in image coordinates (K = 17 or 133)."""
        if not boxes:
            return np.zeros((0, self.n_kpts, 2)), np.zeros((0, self.n_kpts))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return self.model(bgr, bboxes=[list(b) for b in boxes])


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
def pose_estimator(name: str, device: str = "cpu") -> PoseEstimator:
    return PoseEstimator(name, device)
