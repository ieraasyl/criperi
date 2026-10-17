import numpy as np

from kassa_pipeline import extract as E
from kassa_pipeline import objects as O
from kassa_pipeline.config import ExtractConfig
from kassa_pipeline.layout import Layout, LayoutTimeline
from kassa_pipeline.video import Frame

SAFE = (20.0, 20.0, 80.0, 100.0)


class Enc:
    def __init__(self, p_open):
        self.p_open = p_open

    def embed(self, crops):
        return np.zeros((len(crops), 1))

    def zero_shot(self, emb, texts):
        return np.array([[1 - self.p_open, self.p_open]] * len(emb))  # SAFE_PROMPTS order: closed, open


class Mdl:
    def __init__(self, p_open):
        self.enc, self.heads = Enc(p_open), E._NoHeads()


def _img(door_open=False):
    img = np.full((120, 120, 3), 90, np.uint8)
    img[30:90, 30:70] = 200  # the safe door
    if door_open:
        img[30:90, 30:70] = 20  # dark interior where the door was
        img[30:90, 70:76] = 230  # door leaf seen edge-on
    return img


def test_safe_defaults_unchanged():
    e = ExtractConfig()
    assert e.safe_min_prob is None and e.safe_reference is False


def test_low_confidence_safe_is_unknown():
    f, lay = Frame(0, 0.0, _img(), _img(), 90.0), Layout(120, 120, safe=SAFE)
    assert E._states(f, lay, Mdl(0.55), 1.0, {})[0].state == "open"
    assert E._states(f, lay, Mdl(0.55), 1.0, {}, safe_min_prob=0.7)[0] == O.StateObs("unknown", 0.55)
    assert E._states(f, lay, Mdl(0.9), 1.0, {}, safe_min_prob=0.7)[0].state == "open"


def test_safe_change_against_segment_reference():
    frames = [Frame(i, float(i), _img(), _img(), 90.0) for i in range(5)]
    frames.append(Frame(5, 5.0, _img(True), _img(True), 90.0))
    lays = LayoutTimeline([0.0], [Layout(120, 120, safe=SAFE)])
    refs = E._safe_refs(frames, lays)
    closed = O.safe_change(frames[0].norm, SAFE, refs[0])
    opened = O.safe_change(frames[5].norm, SAFE, refs[0])
    assert closed < 0.05 < 0.3 < opened


def test_safe_refs_none_without_safe():
    frames = [Frame(0, 0.0, _img(), _img(), 90.0)]
    assert E._safe_refs(frames, LayoutTimeline([0.0], [Layout(120, 120)])) == [None]
    assert O.safe_change(frames[0].norm, None, None) is None


def test_record_carries_safe_change_only_when_measured():
    from kassa_pipeline import people as P
    from kassa_pipeline.config import PipelineConfig

    lays = LayoutTimeline([0.0], [Layout(120, 120, safe=SAFE)])
    res = P.PeopleResult([], None, [None])
    closed, off = O.StateObs("closed", 0.9), O.StateObs("off", 0.7)
    plain = E._record(E._Raw(0, 0.0, [], [], closed, off), {}, res, lays, {}, PipelineConfig())
    measured = E._record(E._Raw(0, 0.0, [], [], closed, off, 0.41234), {}, res, lays, {}, PipelineConfig())
    assert "safe_change" not in plain["facts"] and measured["facts"]["safe_change"] == 0.412
