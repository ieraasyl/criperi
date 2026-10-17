import numpy as np

from kassa_pipeline import extract as E
from kassa_pipeline.config import ExtractConfig, PipelineConfig
from kassa_pipeline.layout import Layout
from kassa_pipeline.models import Detection
from kassa_pipeline.video import Frame

WS = (100.0, 100.0, 300.0, 200.0)


class CountingOV:
    def __init__(self):
        self.calls = 0

    def __call__(self, images, phrases, box_threshold, text_threshold, batch=1):
        self.calls += 1
        return [[Detection("cup", 0.6, (150.0, 75.0, 170.0, 95.0))] for _ in images]


class Enc:
    def embed(self, crops):
        return np.zeros((len(crops), 1))

    def zero_shot(self, emb, texts):
        out = np.zeros((len(emb), len(texts)))
        out[:, list(E.CLASS_PROMPTS).index("food_drink")] = 1.0
        return out


class Mdl:
    def __init__(self):
        self.ov, self.enc, self.desk = CountingOV(), Enc(), None


def _frame(t, value=60):
    img = np.full((300, 400, 3), value, np.uint8)
    return Frame(int(t * 4), t, img, img, float(value))


def _run(cfg, frames):
    mdl, memo, lay = Mdl(), E._DeskMemo(), Layout(400, 300, desk=WS, workstation=WS)
    out = [E._desk_kept(f, lay, mdl, cfg, {}, memo) for f in frames]
    return mdl.ov.calls, out


def test_desk_reuse_off_by_default():
    assert ExtractConfig().desk_reuse_diff is None
    calls, _ = _run(PipelineConfig(), [_frame(t / 4) for t in range(4)])
    assert calls == 4


def test_unchanged_desk_reuses_detections_until_max_age():
    cfg = PipelineConfig()
    cfg.extract.desk_reuse_diff, cfg.extract.desk_reuse_max_sec = 2.0, 1.0
    calls, out = _run(cfg, [_frame(t / 4) for t in range(9)])  # 0 .. 2 s
    assert calls == 3  # at 0, 1.0 and 2.0 s
    assert all(o == out[0] for o in out) and out[0][0][2] == "food_drink"


def test_changed_desk_detects_again():
    cfg = PipelineConfig()
    cfg.extract.desk_reuse_diff = 2.0
    calls, _ = _run(cfg, [_frame(0.0), _frame(0.25), _frame(0.5, 120), _frame(0.75, 120)])
    assert calls == 2
