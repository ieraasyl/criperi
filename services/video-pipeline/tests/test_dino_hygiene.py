import numpy as np

from kassa_pipeline.config import ExtractConfig, LayoutConfig
from kassa_pipeline.layout import _COUNTER_PROMPTS, detect_layout
from kassa_pipeline.models import Detection, format_prompt
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


def test_detection_cache_off_by_default():
    assert ExtractConfig().detection_cache_dir is None


def test_counter_prompt_positive_is_machine():
    assert "counting machine" in _COUNTER_PROMPTS[0]
    assert LayoutConfig().counter_min_prob == 0.4


def test_counter_verification():
    box = (20, 20, 100, 100)
    for val, expected in ((STACK, None), (MACHINE, box)):
        det = FakeDetector([Detection("banknote", 0.6, tuple(map(float, box)))])
        lay = detect_layout(_frames([(box, val)]), det, FakeEncoder(), LayoutConfig())
        assert lay.counter == (None if expected is None else tuple(map(float, box)))
        assert (not lay.instances) == (expected is None)
        assert (len(lay.rejected_counters) == 1) == (expected is None)
        assert len(lay.to_dict()["rejected_counters"]) == len(lay.rejected_counters)
    assert det.calls[0][0] == tuple(LayoutConfig().prompt)


def test_format_prompt_accepts_strings():
    assert format_prompt("banknotes . money . cash bag") == "banknotes. money. cash bag."
    assert format_prompt("desk. safe.") == "desk. safe."
    assert format_prompt(("desk", "safe")) == "desk. safe."


def test_open_vocab_detector_formats_string_prompt():
    from types import SimpleNamespace

    from kassa_pipeline.models import OpenVocabDetector

    seen = {}

    class Inputs(dict):
        input_ids = None

        def to(self, _):
            return self

    class Proc:
        def __call__(self, images, text, **kw):
            seen["text"] = text
            return Inputs()

        def post_process_grounded_object_detection(self, *a, **kw):
            return [{"labels": [], "scores": [], "boxes": []}]

    det = OpenVocabDetector.__new__(OpenVocabDetector)
    det.device, det.processor, det.model = "cpu", Proc(), lambda **kw: None
    det([np.zeros((4, 4, 3), np.uint8)], "banknotes . money", 0.25, 0.2)
    assert seen["text"] == ["banknotes. money."]
