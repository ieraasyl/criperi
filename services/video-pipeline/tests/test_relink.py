import numpy as np

from kassa_pipeline import people as P
from kassa_pipeline.config import PeopleConfig
from kassa_pipeline.video import Frame, Video


def _track(tid, t0, t1, hist=None, emb=None):
    tr = P.Track(tid, [P.Obs(t0, (0, 0, 10, 20), 0.9), P.Obs(t1, (0, 0, 10, 20), 0.9)], hist)
    tr.emb = emb
    return tr


def _unit(*v):
    a = np.array(v, np.float32)
    return a / np.linalg.norm(a)


def test_relink_embed_off_by_default():
    cfg = PeopleConfig()
    assert cfg.relink_embed is False and cfg.relink_min_embed_similarity == 0.85


def test_relink_by_embedding_overrides_colour():
    same_colour = np.array([1, 2, 3, 4], np.float32)
    a = _track(1, 0, 5, same_colour, _unit(1, 0))
    b = _track(2, 10, 15, same_colour, _unit(0, 1))  # same clothes colour, different person
    assert len(P._relink([a, b], PeopleConfig())) == 1  # colour histograms merge them
    cfg = PeopleConfig(relink_embed=True)
    a, b = _track(1, 0, 5, same_colour, _unit(1, 0)), _track(2, 10, 15, same_colour, _unit(0, 1))
    assert len(P._relink([a, b], cfg)) == 2


def test_relink_by_embedding_merges_same_person():
    cfg = PeopleConfig(relink_embed=True)
    a, b = _track(1, 0, 5, None, _unit(1, 0.1)), _track(2, 10, 15, None, _unit(1, 0.2))
    merged = P._relink([a, b], cfg)
    assert len(merged) == 1 and len(merged[0].obs) == 4
    assert abs(float(np.linalg.norm(merged[0].emb)) - 1.0) < 1e-5


def test_relink_falls_back_to_colour_without_embeddings():
    cfg = PeopleConfig(relink_embed=True)
    h = np.array([1, 2, 3, 4], np.float32)
    assert len(P._relink([_track(1, 0, 5, h), _track(2, 10, 15, h)], cfg)) == 1


class Enc:
    def __init__(self):
        self.crops = []

    def embed(self, crops):
        self.crops += crops
        return np.array([_unit(float(c.mean()) + 1.0, 1.0) for c in crops])


def test_track_embedding_mean_of_person_crops():
    rgb = np.full((100, 100, 3), 50, np.uint8)
    frames = [Frame(i, float(i), rgb, rgb, 50.0) for i in range(3)]
    video = Video(None, 4.0, 3.0, 100, 100, frames)
    tr = P.Track(1, [P.Obs(float(i), (10, 10, 40, 80), 0.9) for i in range(3)])
    enc = Enc()
    emb = P._track_embedding(video, tr, enc)
    assert len(enc.crops) == 3 and enc.crops[0].shape == (70, 30, 3)
    assert abs(float(np.linalg.norm(emb)) - 1.0) < 1e-5
    assert P._track_embedding(video, P.Track(2, [P.Obs(9.0, (10, 10, 40, 80), 0.9)]), enc) is None
