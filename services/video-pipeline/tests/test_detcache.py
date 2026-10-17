import numpy as np

from kassa_pipeline.detcache import CachedDetector
from kassa_pipeline.models import Detection


class FakeInner:
    def __init__(self):
        self.calls = []

    def __call__(self, images, phrases, box_threshold, text_threshold, batch=1):
        self.calls.append((len(images), tuple(phrases), box_threshold, text_threshold))
        dets = [Detection("desk", 0.9, (0, 0, 1, 1)), Detection("safe", 0.3, (1, 1, 2, 2))]
        return [[d for d in dets if d.score >= box_threshold] for _ in images]


IMG = np.zeros((4, 4, 3), np.uint8)


def make(tmp_path):
    inner = FakeInner()
    return inner, CachedDetector(inner, tmp_path, "m")


def test_second_call_is_hit(tmp_path):
    inner, c = make(tmp_path)
    a = c([IMG], ["desk", "safe"], 0.2, 0.25)
    b = c([IMG], ["desk", "safe"], 0.2, 0.25)
    assert a == b and len(inner.calls) == 1
    assert (c.hits, c.misses) == (1, 1)


def test_higher_threshold_filters_cached(tmp_path):
    inner, c = make(tmp_path)
    assert len(c([IMG], ["desk"], 0.2, 0.25)[0]) == 2
    assert [d.label for d in c([IMG], ["desk"], 0.5, 0.25)[0]] == ["desk"]
    assert len(inner.calls) == 1


def test_different_prompt_or_text_threshold_misses(tmp_path):
    inner, c = make(tmp_path)
    c([IMG], ["desk"], 0.2, 0.25)
    c([IMG], ["safe"], 0.2, 0.25)
    c([IMG], ["desk"], 0.2, 0.3)
    assert len(inner.calls) == 3


def test_inner_called_with_floor_and_lower_threshold_recomputes(tmp_path):
    inner, c = make(tmp_path)
    c([IMG], ["desk"], 0.4, 0.25)
    assert inner.calls[-1][2] == 0.1
    c([IMG], ["desk"], 0.05, 0.25)  # below computed_at -> miss
    assert inner.calls[-1][2] == 0.05 and len(inner.calls) == 2
    c([IMG], ["desk"], 0.05, 0.25)
    assert len(inner.calls) == 2


def test_corrupt_cache_file_is_a_miss(tmp_path):
    inner, c = make(tmp_path)
    c([IMG], ["desk"], 0.2, 0.25)
    (f,) = tmp_path.glob("*/*.json")
    for bad in ("{truncated", '{"computed_at": 0.1}', '{"computed_at": 0.1, "detections": [{"label": "x"}]}'):
        f.write_text(bad, encoding="utf8")
        assert len(c([IMG], ["desk"], 0.2, 0.25)[0]) == 2
    assert len(inner.calls) == 4


def test_key_includes_transformers_version(tmp_path, monkeypatch):
    import kassa_pipeline.detcache as D

    _, c = make(tmp_path)
    k1 = c._key(IMG, "p", 0.2)
    monkeypatch.setattr(D, "version", lambda name: "0.0.0")
    assert c._key(IMG, "p", 0.2) != k1
