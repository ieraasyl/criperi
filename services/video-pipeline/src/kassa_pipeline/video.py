"""Stage 0: decode, sample and normalize frames."""

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Frame:
    index: int  # index in the source stream
    t: float  # seconds from start
    rgb: np.ndarray  # original colours, HxWx3 uint8
    norm: np.ndarray  # lighting-normalized copy used for model input
    luma: float  # mean brightness of the original frame


@dataclass
class Video:
    path: Path
    fps: float
    duration: float
    width: int
    height: int
    frames: list[Frame] = field(default_factory=list)  # sampled at sampling.people_fps
    scene_cuts: list[float] = field(default_factory=list)

    def every(self, fps: float) -> list[Frame]:
        """Subset of the sampled frames at a lower rate (e.g. 1 fps for objects)."""
        step = 1.0 / fps
        out, next_t = [], 0.0
        for f in self.frames:
            if f.t + 1e-6 >= next_t:
                out.append(f)
                next_t = f.t + step
        return out

    def nearest(self, t: float) -> Frame:
        return min(self.frames, key=lambda f: abs(f.t - t))


_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def normalize_lighting(rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[..., 0] = _clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def _structure(rgb: np.ndarray) -> np.ndarray:
    """Edge-structure signature, insensitive to brightness and colour shifts."""
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    g = cv2.GaussianBlur(cv2.resize(g, (48, 84), interpolation=cv2.INTER_AREA), (3, 3), 0)
    m = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)).flatten()
    m -= m.mean()
    return m / (np.linalg.norm(m) + 1e-6)


def load_video(path: str | Path, fps: float, cut_threshold: float) -> Video:
    """Decode the whole file once, keeping frames at `fps`.

    OpenCV applies the container's rotation tag by default, so phone footage comes out upright.
    """
    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    frames: list[Frame] = []
    step = 1.0 / fps
    next_t, i = 0.0, 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        t = i / src_fps
        if t + 1e-6 >= next_t:
            ok, bgr = cap.retrieve()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            luma = float(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).mean())
            frames.append(Frame(i, round(t, 3), rgb, normalize_lighting(rgb), luma))
            next_t += step
        i += 1
    cap.release()
    if not frames:
        raise ValueError(f"no frames decoded: {path}")

    h, w = frames[0].rgb.shape[:2]
    video = Video(path, src_fps, (n or i) / src_fps, w, h, frames)

    prev = _structure(frames[0].rgb)
    for f in frames[1:]:
        cur = _structure(f.rgb)
        if float(prev @ cur) < cut_threshold:
            video.scene_cuts.append(f.t)
        prev = cur
    return video


def read_frame_at(path: str | Path, t: float) -> np.ndarray:
    """Full-resolution RGB frame at time `t` (for evidence images)."""
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"cannot read frame at {t}s from {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
