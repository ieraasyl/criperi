import json

import cv2
import numpy as np
import pytest

from kassa_pipeline import label, vlm, zones

W, H = 200, 150
LAPTOP, SAFE, DESK = [10, 10, 60, 50], [120, 10, 190, 90], [10, 80, 110, 140]


class FakeVLM:
    """Answers by the question text found in the prompt; records every question asked."""

    def __init__(self, answers):
        self.answers = answers  # question -> "yes" | "no" | "unknown" | list per call
        self.asked = []

    def ask(self, images, prompt, max_new_tokens=100):
        q = next(q for _, q, _ in zones.QUESTIONS.values() if q in prompt)
        self.asked.append(q)
        a = self.answers.get(q, "no")
        if isinstance(a, list):
            a = a[self.asked.count(q) - 1]
        return json.dumps({"answer": a, "confidence": 0.9, "observation": "x"})


def q(state):
    return zones.QUESTIONS[state][1]


def make_video(tmp_path, seconds=3, fps=10):
    p = tmp_path / "X-1.mp4"
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for i in range(seconds * fps):
        w.write(np.full((H, W, 3), 30 + i, np.uint8))
    w.release()
    return p


def layout(**kw):
    lay = {"start": 0.0, "width": W, "height": H, "desk": DESK, "laptop": LAPTOP, "safe": SAFE, "doors": []}
    lay.update(kw)
    return lay


def rec(t, people=(), **facts):
    f = {"cashier_zone": "absent", "safe": "open", "screen": "work_screen"}
    f.update(facts)
    return {"i": int(t * 4), "t": t, "people": [{"track": 1, "role": "cashier", "box": b, "score": 0.9,
                                                  "zone": "x", "hands": {}, "holding": {}} for b in people],
            "objects": [], "facts": f}


def stage1(seconds=3, people=(), **lay):
    frames = [rec(s + k * 0.25, people) for s in range(seconds) for k in range(4)]
    return {"meta": {"name": "X-1.mov", "duration": float(seconds), "sample_fps": 4},
            "layouts": [layout(**lay)], "cashier_track": None, "people": [], "objects": [], "frames": frames}


def draft(tmp_path, answers, s1=None, seconds=3, **kw):
    fake = FakeVLM(answers)
    ft = label.build_draft(make_video(tmp_path, seconds), s1 or stage1(seconds), fake, **kw)
    return ft, fake


def test_local_vlm_falls_back_when_vram_low(monkeypatch, tmp_path):
    assert vlm.pick_local_model(2.0) == "Qwen/Qwen3-VL-2B-Instruct"
    assert vlm.pick_local_model(3.5) == "Qwen/Qwen3-VL-4B-Instruct"
    seen = {}

    class FakeQwen:
        def __init__(self, **kw):
            seen.update(kw)

    monkeypatch.setattr(vlm, "QwenVLM", FakeQwen)
    vlm.local_vlm(tmp_path / "c.json", free_gb_fn=lambda: 2.0)
    assert seen == {"model_id": "Qwen/Qwen3-VL-2B-Instruct", "quant": "nf4",
                    "cache_path": tmp_path / "c-qwen3-vl-2b-instruct.json", "device": "cuda"}
    vlm.local_vlm(tmp_path / "c.json", free_gb_fn=lambda: 3.5)
    assert seen["model_id"] == "Qwen/Qwen3-VL-4B-Instruct"
    assert seen["cache_path"] == tmp_path / "c-qwen3-vl-4b-instruct.json"
    vlm.local_vlm(None, free_gb_fn=lambda: 3.5)
    assert seen["cache_path"] is None


def test_ask_check_parses_answers():
    class V:
        def ask(self, images, prompt, max_new_tokens=100):
            return 'noise {"answer": "Yes", "confidence": 2, "observation": "ok"}'

    a = vlm.ask_check(V(), [np.zeros((4, 4, 3), np.uint8)], "Is it?")
    assert (a.answer, a.confidence, a.observation) == (True, 1.0, "ok")

    class Bad:
        def ask(self, images, prompt, max_new_tokens=100):
            return "no json"

    assert vlm.ask_check(Bad(), [], "Is it?").answer is None


def test_answers_replace_cv_with_vlm_origin(tmp_path):
    ft, _ = draft(tmp_path, {q("laptop_lid"): "no", q("safe_door"): "no", q("desk_cash"): "yes"})
    assert ft.states["laptop_lid"] == ["open"] * 3
    assert ft.states["safe_door"] == ["closed"] * 3  # CV said open
    assert ft.states["desk_cash"] == ["yes"] * 3
    assert ft.states["desk_docs"] == ["no"] * 3
    assert ft.state_origin["safe_door"] == ["vlm"] * 3
    assert ft.state_origin["people_inside"] == ["cv"] * 3
    assert ft.status == "draft" and ft.clip == "X-1"


def test_screen_only_asked_when_lid_open(tmp_path):
    ft, fake = draft(tmp_path, {q("laptop_lid"): ["yes", "yes", "no"], q("screen"): "no"})
    assert ft.states["laptop_lid"] == ["closed", "closed", "open"]
    assert ft.states["screen"] == ["off", "off", "off"]  # a closed lid means the screen is off
    assert ft.state_origin["screen"] == ["vlm", "vlm", "vlm"]
    assert fake.asked.count(q("screen")) == 1


def test_missing_zone_keeps_cv_value(tmp_path):
    ft, fake = draft(tmp_path, {}, s1=stage1(safe=None))
    assert ft.states["safe_door"] == ["open"] * 3
    assert ft.state_origin["safe_door"] == ["cv"] * 3
    assert q("safe_door") not in fake.asked


def test_occluded_zone_keeps_cv_value(tmp_path):
    ft, fake = draft(tmp_path, {q("laptop_lid"): "yes"}, s1=stage1(people=[[0, 0, 70, 60]]))
    assert ft.states["laptop_lid"] == ["unknown"] * 3
    assert ft.state_origin["laptop_lid"] == ["cv"] * 3
    assert q("laptop_lid") not in fake.asked
    assert ft.states["safe_door"] == ["closed"] * 3


def test_unknown_answer_keeps_cv_value(tmp_path):
    ft, _ = draft(tmp_path, {q("safe_door"): "unknown"})
    assert ft.states["safe_door"] == ["open"] * 3
    assert ft.state_origin["safe_door"] == ["cv"] * 3


def test_despeckle_marks_single_flip_in_vlm_series(tmp_path):
    ft, _ = draft(tmp_path, {q("safe_door"): ["yes", "no", "yes"]})
    assert ft.states["safe_door"] == ["open", "unknown", "open"]


def test_events_and_people_inside_kept_from_stage1(tmp_path):
    ft, _ = draft(tmp_path, {q("safe_door"): "yes"})
    assert [e.type for e in ft.events] == ["safe_access_start"]
    assert ft.states["people_inside"] == [0, 0, 0]


def test_despeckle_unit():
    assert zones.despeckle(["closed", "open", "closed"]) == ["closed", "unknown", "closed"]
    assert zones.despeckle(["closed", "open", "open"]) == ["closed", "open", "open"]
    assert zones.despeckle(["closed", "unknown", "closed"]) == ["closed", "unknown", "closed"]


def test_zone_helpers():
    c = zones.zone_crop(np.zeros((H, W, 3), np.uint8), tuple(LAPTOP))
    assert min(c.shape[:2]) >= 448
    assert zones.zone_box({"doors": [SAFE]}, "door") == tuple(SAFE)
    assert zones.zone_box({"doors": []}, "door") is None
    assert zones.zone_box({"laptop": None}, "laptop") is None
    assert zones.occluded(tuple(LAPTOP), [(0, 0, 70, 60)])
    assert not zones.occluded(tuple(SAFE), [(0, 0, 70, 60)])


def test_unreadable_video_raises(tmp_path):
    with pytest.raises(ValueError, match="no frames could be read"):
        label.build_draft(tmp_path / "missing.mp4", stage1(), FakeVLM({}))


def test_single_unreadable_second_keeps_cv(tmp_path):
    ft, _ = draft(tmp_path, {q("safe_door"): "no"}, s1=stage1(5), seconds=3)
    assert ft.states["safe_door"][:3] == ["closed"] * 3
    assert ft.states["safe_door"][4] == "open"
    assert ft.state_origin["safe_door"][4] == "cv"


def test_closed_lid_forces_screen_off(tmp_path):
    ft, fake = draft(tmp_path, {q("laptop_lid"): ["no", "yes", "yes"], q("screen"): "no"})
    assert ft.states["laptop_lid"] == ["open", "closed", "closed"]
    assert ft.states["screen"] == ["off", "off", "off"]
    assert ft.state_origin["screen"] == ["vlm"] * 3
    ft, _ = draft(tmp_path, {q("laptop_lid"): "yes"}, s1=stage1(people=[[0, 0, 70, 60]]))  # lid occluded: CV kept
    assert ft.states["screen"] == ["work"] * 3 and ft.state_origin["screen"] == ["cv"] * 3

