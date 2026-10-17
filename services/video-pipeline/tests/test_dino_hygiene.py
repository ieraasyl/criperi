import numpy as np

from kassa_pipeline.config import LayoutConfig
from kassa_pipeline.layout import _COUNTER_PROMPTS, detect_layout
from kassa_pipeline.models import Detection, format_prompt, threshold_for
from kassa_pipeline.objects import categories_agree
from kassa_pipeline.video import Frame

MACHINE, STACK = 200, 100  # crop brightness the fake encoder maps to a class


class FakeDetector:
    def __init__(self, dets):
        self.dets, self.calls = dets, []

    def __call__(self, images, phrases, box_threshold, text_threshold, batch=1):
        self.calls.append((tuple(phrases), box_threshold))
        return [[d for d in self.dets if d.score >= box_threshold] for _ in images]


class FakeEncoder:
    def embed(self, crops):
        return np.array([[float(c.mean())] for c in crops])

    def zero_shot(self, emb, texts):
        out = np.full((len(emb), len(texts)), 0.0)
        for r, (v,) in enumerate(emb):
            out[r, 0 if v > 150 else 1] = 1.0
        return out


def _frames(boxes_vals):
    img = np.zeros((200, 300, 3), np.uint8)
    for (x1, y1, x2, y2), v in boxes_vals:
        img[y1:y2, x1:x2] = v
    return [Frame(i, float(i), img, img, 0.0) for i in range(3)]


def test_format_prompt():
    assert format_prompt(("Desk", " safe ", "")) == "desk. safe."


def test_threshold_for():
    assert threshold_for("cash", {}, 0.25) == 0.25
    assert threshold_for("cash", {"cash": 0.4}, 0.25) == 0.4
    assert threshold_for("cup", {"cash": 0.4}, 0.25) == 0.25


def test_categories_agree():
    assert categories_agree("cash", "cash")
    assert categories_agree("cash", "cash_bag")
    assert categories_agree("cash_bag", "cash")
    assert not categories_agree("cash", "personal")
    assert not categories_agree("phone", "cash")


def test_counter_prompt_positive_is_machine():
    assert "counting machine" in _COUNTER_PROMPTS[0]


def test_counter_verification():
    box = (20, 20, 100, 100)
    for val, expected in ((STACK, None), (MACHINE, box)):
        det = FakeDetector([Detection("banknote", 0.6, tuple(map(float, box)))])
        lay = detect_layout(_frames([(box, val)]), det, FakeEncoder(), LayoutConfig())
        assert lay.counter == (None if expected is None else tuple(map(float, box)))
        assert (not lay.instances) == (expected is None)
        assert (len(lay.rejected_counters) == 1) == (expected is None)
        assert len(lay.to_dict()["rejected_counters"]) == len(lay.rejected_counters)
    assert det.calls[0][0] == LayoutConfig().prompt


def test_class_thresholds():
    dets = [
        Detection("laptop", 0.3, (10.0, 10.0, 80.0, 80.0)),
        Detection("telephone", 0.3, (120.0, 10.0, 190.0, 80.0)),
    ]
    cfg = LayoutConfig(class_thresholds={"telephone": 0.5, "laptop": 0.2})
    det = FakeDetector(dets)
    lay = detect_layout(_frames([]), det, FakeEncoder(), cfg)
    assert det.calls[0][1] == 0.2  # lowest threshold goes to the detector
    assert lay.laptop is not None
    assert lay.telephone is None
