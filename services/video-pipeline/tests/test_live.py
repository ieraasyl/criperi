import base64
import io
import json
import sys
import types

import pytest
from PIL import Image

from kassa_pipeline.live import decode_window, live_result
from kassa_pipeline.config import PipelineConfig
from kassa_pipeline.vlm_scan import scan
from kassa_pipeline import live


def frame(t, size=(32, 48)):
    data = io.BytesIO()
    Image.new("RGB", size, (120, 100, 90)).save(data, format="JPEG")
    return {"timestamp_sec": t, "image_base64": base64.b64encode(data.getvalue()).decode()}


def test_decode_uses_only_supplied_frames_and_preserves_absolute_timestamps():
    video, offset = decode_window([frame(42), frame(43), frame(44)])
    assert offset == 42
    assert video.duration == 2
    assert [f.t for f in video.frames] == [0, 1, 2]
    assert (video.width, video.height) == (32, 48)


@pytest.mark.parametrize("frames", [
    [frame(0)], [frame(1), frame(0)], [frame(0), frame(3)],
    [frame(0), frame(float("nan"))], [frame(0), frame(1, (961, 1))],
    [frame(0), frame(1, (30, 48))],
    [frame(0), {"timestamp_sec": 1, "image_base64": "not jpeg"}],
])
def test_invalid_windows_are_rejected(frames):
    with pytest.raises((ValueError, OSError)):
        decode_window(frames)


def test_evidence_and_descriptions_refer_to_original_playback_time():
    video, offset = decode_window([frame(42), frame(43), frame(44)])
    stage1 = {"frames": [], "layouts": [{"start": 0}], "people": []}
    result = live_result(video, offset, stage1, {"findings": [
        {"code": "V17", "t": 1, "confidence": 0.8},
        {"code": "V17", "t": 4, "confidence": 0.8},
        {"code": "V17", "t": 0, "confidence": 0.2},
    ]}, 0.5)
    assert result["from_sec"] == 42 and result["to_sec"] == 44
    assert len(result["violations"]) == 1
    finding = result["violations"][0]
    assert finding["timecode_sec"] == 43
    assert "43 с" in finding["description"]
    assert finding["evidence_frames"][0]["timestamp_sec"] == 43
    assert base64.b64decode(finding["evidence_frames"][0]["image_base64"]).startswith(b"\xff\xd8\xff")


def test_live_scan_does_not_infer_missing_steps_from_an_incomplete_visit():
    video, _ = decode_window([frame(t) for t in range(12)])
    stage1 = {"frames": [], "layouts": [{"start": 0}], "people": [{"role": "visitor", "id": 2, "t0": 0, "t1": 11}]}

    class Verifier:
        calls = 0
        gpu_seconds = 0

        def ask_many(self, items, **kwargs):
            assert all("covering one client visit" not in prompt for _, prompt in items)
            return [json.dumps({"state": {"food_or_drink_on_desk": ["yes", 4]}, "events": []}) for _ in items]

    result = scan(video, stage1, Verifier(), PipelineConfig(), include_sessions=False)
    assert result["sessions"] == []
    assert [f["code"] for f in result["findings"]] == ["V17"]


def test_live_generator_streams_detections_before_running_verification(monkeypatch):
    stage1 = {"frames": [], "layouts": [{"start": 0}], "people": []}

    def extract(video, cfg, models):
        yield {"type": "stage", "stage": "layout"}
        for frame in video.frames:
            yield {"type": "frame", "rec": {"t": frame.t, "people": [], "objects": []}}
        yield {"type": "final", "result": stage1}

    module = types.ModuleType("kassa_pipeline.extract")
    module.stream_video = extract
    monkeypatch.setitem(sys.modules, "kassa_pipeline.extract", module)
    calls = []

    def verify(video, result, vlm, cfg, *, include_sessions):
        calls.append(include_sessions)
        return {"vlm": {"unparsed": 0}, "findings": [{"code": "V17", "t": 1, "confidence": 0.8}]}

    monkeypatch.setattr(live, "scan", verify)
    caches = []
    vlm = types.SimpleNamespace(use_cache=caches.append)
    events = live.stream([frame(42), frame(43)], object(), vlm)
    assert next(events) == {"type": "stage", "stage": "layout"}
    assert next(events) == {"type": "stage", "stage": "detecting"}
    assert next(events)["timestamp_sec"] == 42
    assert next(events)["timestamp_sec"] == 43
    assert calls == []
    assert next(events) == {"type": "stage", "stage": "verifying"}
    result = next(events)
    assert calls == [False]
    assert caches == [None]
    assert result["violations"][0]["timecode_sec"] == 43
