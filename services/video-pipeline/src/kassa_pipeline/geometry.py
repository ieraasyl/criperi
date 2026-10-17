"""Box helpers. Boxes are (x1, y1, x2, y2) in pixels."""

import numpy as np

Box = tuple[float, float, float, float]


def area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def inside_share(inner: Box, outer: Box) -> float:
    """Share of `inner` covered by `outer`."""
    ix = max(0.0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    iy = max(0.0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    a = area(inner)
    return ix * iy / a if a > 0 else 0.0


def center(b: Box) -> tuple[float, float]:
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def contains(b: Box, p: tuple[float, float]) -> bool:
    return b[0] <= p[0] <= b[2] and b[1] <= p[1] <= b[3]


def expand(b: Box, pad: float, w: int | None = None, h: int | None = None) -> Box:
    dx, dy = (b[2] - b[0]) * pad, (b[3] - b[1]) * pad
    x1, y1, x2, y2 = b[0] - dx, b[1] - dy, b[2] + dx, b[3] + dy
    if w is not None and h is not None:
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    return (x1, y1, x2, y2)


def gap(a: Box, b: Box) -> float:
    """Smallest distance between two boxes (0 if they overlap)."""
    dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    return float(np.hypot(dx, dy))


def median_box(boxes: list[Box]) -> Box:
    arr = np.array(boxes, dtype=float)
    return tuple(float(v) for v in np.median(arr, axis=0))


def union(boxes: list[Box]) -> Box:
    arr = np.array(boxes, dtype=float)
    return (arr[:, 0].min(), arr[:, 1].min(), arr[:, 2].max(), arr[:, 3].max())


def crop(img: np.ndarray, b: Box, min_size: int = 8) -> np.ndarray | None:
    h, w = img.shape[:2]
    x1, y1 = int(max(0, b[0])), int(max(0, b[1]))
    x2, y2 = int(min(w, b[2])), int(min(h, b[3]))
    if x2 - x1 < min_size or y2 - y1 < min_size:
        return None
    return img[y1:y2, x1:x2]
