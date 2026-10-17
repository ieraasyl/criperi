import numpy as np
import pytest

from kassa_pipeline import extract as E
from kassa_pipeline import heads as H
from kassa_pipeline.config import PipelineConfig
from kassa_pipeline.facts import FactTimeline
from kassa_pipeline.layout import Layout
from kassa_pipeline.video import Frame

ENC = "google/siglip-so400m-patch14-384"


def _blobs(n=40, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal([3, 0], 0.5, (n, 2))
    b = rng.normal([0, 3], 0.5, (n, 2))
    return np.vstack([a, b]).astype(np.float32), ["open"] * n + ["closed"] * n


def test_fit_and_predict_separate_two_classes(tmp_path):
    x, y = _blobs()
    heads = H.Heads(ENC, {"safe": H.fit(x, y)})
    classes, p = heads.predict("safe", np.array([[3.0, 0.0], [0.0, 3.0]], np.float32))
    assert [classes[i] for i in p.argmax(axis=1)] == ["open", "closed"]
    assert np.allclose(p.sum(axis=1), 1.0)
    heads.save(tmp_path / "h.npz")
    back = H.Heads.load(tmp_path / "h.npz")
    assert back.encoder == ENC and back.has("safe") and not back.has("screen")
    assert np.allclose(back.predict("safe", x)[1], heads.predict("safe", x)[1], atol=1e-6)


def test_train_reports_held_out_clips_and_skips_one_class_tasks():
    x, y = _blobs()
    rows = [(f"X-{i % 8}", e, lb) for i, (e, lb) in enumerate(zip(x, y))]
    heads, report = H.train({"safe": rows, "screen": [(r[0], r[1], "off") for r in rows]}, ENC)
    assert heads.has("safe") and not heads.has("screen")
    assert report["safe"]["val"]["accuracy"] > 0.9 and report["safe"]["val_clips"] == ["X-0", "X-4"]
    assert "skipped" in report["screen"]


def test_samples_crop_labelled_seconds_only():
    ft = FactTimeline.empty("X-1", 3.0)
    ft.status = "reviewed"
    ft.states["safe_door"] = ["open", "unknown", "closed"]
    ft.states["screen"] = ["unknown", "work", "unknown"]
    stage1 = {"layouts": [{"start": 0.0, "safe": [10, 10, 40, 50], "laptop": [50, 50, 90, 80]}]}
    times = []

    def read(t):
        times.append(t)
        return np.full((100, 100, 3), 120, np.uint8)

    out = H.samples(ft, stage1, read)
    assert [lb for _, lb in out["safe"]] == ["open", "closed"]
    assert [lb for _, lb in out["screen"]] == ["work_screen"]
    assert times == [0.5, 1.5, 2.5]
    assert out["screen"][0][0].shape == (19, 32, 3)  # screen area: upper 65 % of 30 px, 80 % of 40 px


class Enc:
    def embed(self, crops):
        return np.array([[3.0, 0.0] for _ in crops], np.float32)


class Mdl:
    def __init__(self, heads):
        self.enc, self.heads = Enc(), heads


def test_stage1_uses_the_screen_and_safe_heads():
    x, y = _blobs()
    heads = H.Heads(ENC, {"safe": H.fit(x, y), "screen": H.fit(x, ["work_screen" if v == "open" else "off" for v in y])})
    img = np.full((120, 120, 3), 90, np.uint8)
    lay = Layout(120, 120, safe=(20.0, 20.0, 60.0, 60.0), laptop=(60.0, 60.0, 110.0, 100.0))
    safe, screen = E._states(Frame(0, 0.0, img, img, 90.0), lay, Mdl(heads), 1.0, {})
    assert (safe.state, screen.state) == ("open", "work_screen") and screen.prob > 0.9


def test_models_reject_heads_of_another_encoder(tmp_path, monkeypatch):
    from kassa_pipeline import models as M

    x, y = _blobs()
    H.Heads("google/siglip-base-patch16-224", {"safe": H.fit(x, y)}).save(tmp_path / "h.npz")

    class Stub:
        device = "cpu"

    for f in ("person_detector", "open_vocab_detector", "crop_encoder"):
        monkeypatch.setattr(M, f, lambda *a: Stub())
    cfg = PipelineConfig()
    cfg.extract.pose_model, cfg.extract.heads_path = "none", str(tmp_path / "h.npz")
    with pytest.raises(ValueError, match="trained on"):
        E.Models(cfg)
    cfg.extract.crop_encoder = "google/siglip-base-patch16-224"
    assert E.Models(cfg).heads.has("safe")
