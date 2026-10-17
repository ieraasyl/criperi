"""Detector functions with a fake VLM: pure logic, no models."""

import numpy as np

from kassa_pipeline import detectors as D
from kassa_pipeline.config import PipelineConfig
from kassa_pipeline.layout import Layout, LayoutTimeline
from kassa_pipeline.objects import ObjectsResult
from kassa_pipeline.people import PeopleResult
from kassa_pipeline.timeline import Second, Timeline
from kassa_pipeline.video import Frame, Video


class FakeVLM:
    """Answers every closed question with `answer`; sweep questions with `sweep`."""

    def __init__(self, answer="yes", sweep='{"personal": false, "food": false, "bag": false}'):
        self.answer, self.sweep, self.calls = answer, sweep, 0

    def ask(self, images, prompt, max_new_tokens=100):
        self.calls += 1
        if '"personal"' in prompt:
            return self.sweep
        return '{"answer": "%s", "confidence": 0.9, "observation": "fake"}' % self.answer


def make_ctx(seconds, vlm=None, objects=None):
    n = len(seconds)
    frames = [Frame(i * 30, float(i), np.zeros((120, 80, 3), np.uint8), np.zeros((120, 80, 3), np.uint8), 100.0) for i in range(n)]
    video = Video(__import__("pathlib").Path("x.mov"), 30.0, float(n), 80, 120, frames)
    lay = Layout(80, 120, desk=(5, 60, 70, 110), workstation=(2, 50, 75, 118), laptop=(40, 55, 55, 70))
    tl = LayoutTimeline([0.0], [lay])
    return D.Context(video, tl, PeopleResult([], None, [None]), objects or ObjectsResult(), Timeline(seconds, float(n)), PipelineConfig(), vlm)


def sec(t, zone="cashier_seat", **kw):
    return Second(t=t, cashier_zone=zone, cashier_at_workplace=zone in ("cashier_seat", "at_desk"), **kw)


CASH_LEFT = [sec(i) for i in range(3)] + [sec(3 + i, "doorway", cash_on_desk=True, cash_score=0.3) for i in range(6)]


def test_vlm_confirmation_raises_and_denial_lowers_the_score():
    base = D.detect_cash_left(make_ctx(CASH_LEFT, None))[0]
    yes = D.detect_cash_left(make_ctx(CASH_LEFT, FakeVLM("yes")))[0]
    no = D.detect_cash_left(make_ctx(CASH_LEFT, FakeVLM("no")))[0]
    assert no.score < base.score < yes.score
    assert yes.source == "cv+vlm" and base.source == "cv"
    assert yes.rule == "R-CASH-LOCKED" and yes.type == "T1"


def test_safe_detector_requires_the_vlm():
    secs = [sec(i, safe="open") for i in range(2)] + [sec(2 + i, "doorway", safe="open") for i in range(5)]
    assert D.detect_safe_open_unattended(make_ctx(secs, None))[0].score < 0.5  # no VLM -> not reported
    assert D.detect_safe_open_unattended(make_ctx(secs, FakeVLM("yes")))[0].score >= 0.5
    assert D.detect_safe_open_unattended(make_ctx(secs, FakeVLM("no")))[0].score < 0.3


def test_sweep_finds_a_personal_item_the_cv_missed():
    secs = [sec(i) for i in range(12)]
    vlm = FakeVLM(sweep='{"personal": true, "food": false, "bag": false}')
    found = D.detect_personal_items(make_ctx(secs, vlm))
    assert found and found[0].type == "T2" and found[0].source == "vlm" and found[0].score >= 0.5
    assert D.detect_personal_items(make_ctx(secs, None)) == []  # no VLM, no CV signal: nothing


def test_walking_away_alone_triggers_nothing():
    secs = [sec(i) for i in range(3)] + [sec(3 + i, "doorway") for i in range(6)]
    ctx = make_ctx(secs, FakeVLM("yes"))
    found, info = D.run_detectors(ctx)
    assert found == [] and info["errors"] == {}


def test_a_failing_detector_does_not_fail_the_video():
    class Broken(FakeVLM):
        def ask(self, *a, **k):
            raise RuntimeError("GPU gone")

    found, info = D.run_detectors(make_ctx(CASH_LEFT, Broken()))
    assert "cash_left" in info["errors"] and "GPU gone" in info["errors"]["cash_left"]


def test_every_detector_type_has_a_rule_and_title():
    from kassa_pipeline.events import RULE_OF, TYPE_TITLE
    from kassa_pipeline.report import rule_for

    for d in D.DETECTORS:
        for t in d.types:
            assert t in RULE_OF and t in TYPE_TITLE and rule_for(t) is not None, t
