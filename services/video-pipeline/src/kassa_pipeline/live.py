"""Inference on frames captured during browser playback. No saved detections or disk caches."""

import base64
import io
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from kassa_pipeline import video as V
from kassa_pipeline.config import PipelineConfig
from kassa_pipeline.report import annotate, describe, type_label
from kassa_pipeline.vlm_scan import Stage1, scan

MAX_IMAGE = 350_000
MAX_PIXELS = 960 * 960


def decode_window(frames: list[dict]) -> tuple[V.Video, float]:
    """Validate before allocating pixels; preserve playback time without inventing missing frames."""
    if not isinstance(frames, list) or not 2 <= len(frames) <= 12:
        raise ValueError("expected 2-12 frames")
    decoded, previous, offset, size = [], None, None, None
    for i, frame in enumerate(frames):
        t = frame.get("timestamp_sec")
        image = frame.get("image_base64")
        if not isinstance(t, (int, float)) or isinstance(t, bool) or not math.isfinite(t) or t < 0:
            raise ValueError("invalid timestamp")
        if previous is not None and not 0.5 <= t - previous <= 1.75:
            raise ValueError("nonconsecutive frames")
        if not isinstance(image, str) or len(image) > MAX_IMAGE:
            raise ValueError("image too large")
        data = base64.b64decode(image, validate=True)
        with Image.open(io.BytesIO(data)) as im:
            if im.format != "JPEG" or max(im.size) > 960 or im.width * im.height > MAX_PIXELS:
                raise ValueError("expected a JPEG no larger than 960 pixels per side")
            if size is not None and im.size != size:
                raise ValueError("frame dimensions changed")
            size = im.size
            rgb = np.array(im.convert("RGB"))
        if offset is None:
            offset = float(t)
        local = round(t - offset, 3)
        luma = float(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).mean())
        decoded.append(V.Frame(i, local, rgb, V.normalize_lighting(rgb), luma))
        previous = t
    width, height = size
    # Duration ends at the last captured frame, so neither evidence nor the VLM
    # can refer to a moment the browser has not supplied.
    video = V.Video(Path("live-window"), 1.0, decoded[-1].t, width, height, decoded)
    prev = V._structure(decoded[0].rgb)
    for frame in decoded[1:]:
        cur = V._structure(frame.rgb)
        if float(prev @ cur) < 0.5:
            video.scene_cuts.append(frame.t)
        prev = cur
    return video, offset


def jpeg(rgb: np.ndarray) -> str:
    ok, data = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 75])
    if not ok:
        raise ValueError("cannot encode frame")
    value = base64.b64encode(data).decode()
    if len(value) > MAX_IMAGE:
        raise ValueError("annotated frame too large")
    return value


def live_result(video: V.Video, offset: float, stage1: dict, result: dict, threshold: float) -> dict:
    s1 = Stage1(stage1)
    findings = []
    for finding in result["findings"]:
        t = finding["t"]
        if finding["confidence"] < threshold or not math.isfinite(t) or not 0 <= t <= video.duration:
            continue
        frame = video.nearest(t)
        absolute = round(offset + t, 3)
        findings.append({
            "type": type_label(finding["code"]),
            "timecode_sec": absolute,
            "description": describe(finding["code"], absolute),
            "evidence_frames": [{"timestamp_sec": round(offset + frame.t, 3),
                                 "image_base64": jpeg(annotate(frame.rgb.copy(), frame.t, finding["code"], s1))}],
        })
    return {"type": "result", "from_sec": offset, "to_sec": round(offset + video.duration, 3),
            "violations": sorted(findings, key=lambda f: f["timecode_sec"])}


def stream(frames: list[dict], models, vlm):
    from kassa_pipeline.extract import stream_video
    from kassa_pipeline.report import _label, OBJ_TEXT

    video, offset = decode_window(frames)
    cfg = PipelineConfig()
    cfg.extract.fps = 1.0
    cfg.sampling.layout_frames = 3
    cfg.scan.window_fps = 1.0
    cfg.scan.batch = 2
    # Repeated windows must still run inference. Live inputs and outputs never
    # enter the file analysis cache or the persistent Modal work volume.
    vlm.use_cache(None)
    stage1 = None
    yield {"type": "stage", "stage": "layout"}
    for event in stream_video(video, cfg, models):
        if event["type"] == "stage" and event["stage"] == "layout":
            yield {"type": "stage", "stage": "detecting"}
        elif event["type"] == "frame":
            rec = event["rec"]
            # The snapshot is tied to its own timestamp, never drawn over a
            # newer playback frame where a delayed box could be misleading.
            rgb = video.nearest(rec["t"]).rgb.copy()
            for person in rec["people"]:
                rgb = _label(rgb, person["box"], "кассир" if person["role"] == "cashier" else "посетитель", (80, 220, 120))
            for obj in rec["objects"]:
                rgb = _label(rgb, obj["box"], OBJ_TEXT.get(obj["cls"], obj["cls"]), (255, 200, 0))
            yield {"type": "frame", "timestamp_sec": round(offset + rec["t"], 3), "image_base64": jpeg(rgb),
                   "people_count": len(rec["people"]), "objects_count": len(rec["objects"])}
        elif event["type"] == "final":
            stage1 = event["result"]
    if stage1 is None:
        raise RuntimeError("extraction did not complete")
    yield {"type": "stage", "stage": "verifying"}
    result = scan(video, stage1, vlm, cfg, include_sessions=False)
    if result["vlm"]["unparsed"]:
        raise RuntimeError("the verifier returned an unreadable observation")
    yield live_result(video, offset, stage1, result, cfg.scan.threshold)
