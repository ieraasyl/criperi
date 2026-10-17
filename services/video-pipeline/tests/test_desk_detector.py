import numpy as np

from kassa_pipeline import extract as E
from kassa_pipeline.config import ExtractConfig, PipelineConfig
from kassa_pipeline.layout import Layout
from kassa_pipeline.models import Detection
from kassa_pipeline.video import Frame

WS = (100.0, 100.0, 300.0, 200.0)  # workstation; desk_crop_pad 0.25 grows it to (50, 75, 350, 225)


class FakeOV:
    def __init__(self, dets):
        self.dets = dets

    def __call__(self, images, phrases, box_threshold, text_threshold, batch=1):
        return [list(self.dets) for _ in images]


class FakeEnc:
    """Classifies a crop by its mean brightness (the 0.4 padding dims it): bright = cash, dim = food_drink."""

    def embed(self, crops):
        return np.array([[float(c.mean())] for c in crops])

    def zero_shot(self, emb, texts):
        names = list(E.CLASS_PROMPTS)
        out = np.zeros((len(emb), len(texts)))
        for r, (v,) in enumerate(emb):
            out[r, names.index("cash" if v > 45 else "food_drink")] = 1.0
        return out


class FakeDesk:
    def __init__(self, dets):
        self.dets, self.images = dets, []

    def __call__(self, images, threshold):
        self.images += images
        return [[d for d in self.dets if d.score >= threshold] for _ in images]


class Mdl:
    def __init__(self, ov, desk=None):
        self.ov, self.enc, self.desk = ov, FakeEnc(), desk


def _frame():
    rgb = np.zeros((300, 400, 3), np.uint8)
    rgb[100:110, 100:120] = 200  # bright patch: SigLIP says cash
    rgb[150:170, 200:220] = 100  # dim patch: SigLIP says food_drink
    norm = rgb.copy()
    norm[0, 0] = 1  # tell rgb and norm apart
    return Frame(0, 0.0, rgb, norm, 50.0)


# DINO boxes are in crop pixels: the crop starts at (50, 75)
DINO = [Detection("banknotes", 0.6, (50.0, 25.0, 70.0, 35.0)), Detection("cup", 0.6, (150.0, 75.0, 170.0, 95.0))]


def test_desk_detector_off_by_default():
    e = ExtractConfig()
    assert e.desk_detector is None and e.desk_detector_scale == 2.0


def test_dino_only_without_desk_detector():
    kept = E._candidates(_frame(), Layout(400, 300, desk=WS, workstation=WS), Mdl(FakeOV(DINO)), PipelineConfig())
    assert sorted(k[2] for k in kept) == ["cash", "food_drink"]


def test_desk_detector_owns_money_and_cards():
    # detector input is the raw RGB desk crop upscaled 2x: (300 x 150) -> (600 x 300)
    desk = FakeDesk([Detection("cash_bag", 0.8, (100.0, 100.0, 140.0, 120.0)), Detection("cash", 0.3, (0.0, 0.0, 10.0, 10.0))])
    cfg = PipelineConfig()
    cfg.extract.desk_detector = "models/v2"
    kept = E._candidates(_frame(), Layout(400, 300, desk=WS, workstation=WS), Mdl(FakeOV(DINO), desk), cfg)
    assert desk.images[0].shape == (300, 600, 3)
    assert desk.images[0][0, 0, 0] == 0  # rgb, not the lighting-normalized copy
    by = {k[2]: k for k in kept}
    assert sorted(by) == ["cash_bag", "food_drink"]  # DINO's cash dropped, low-score detector box dropped
    assert by["cash_bag"][0] == (100.0, 125.0, 120.0, 135.0)  # back in video pixels
    assert by["cash_bag"][1] == 0.8
