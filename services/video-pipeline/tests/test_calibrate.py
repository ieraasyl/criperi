from types import SimpleNamespace

import numpy as np

from kassa_pipeline import calibrate as C
from kassa_pipeline import objects as O
from kassa_pipeline.config import ObjectsConfig
from kassa_pipeline.models import Detection
from kassa_pipeline.objects import RawCandidate, RawObjects
from kassa_pipeline.video import Frame, Video


def _cand(cat, score, probs, sec=0):
    return RawCandidate(float(sec), sec, cat, score, (0, 0, 10, 10), True, False, probs)


def test_select_thresholds_ignored_agreement():
    raw = RawObjects(
        candidates=[
            _cand("cash", 0.3, {"cash": 0.9, "paper": 0.1}),  # kept, agree
            _cand("cash", 0.3, {"cash_bag": 0.8, "cash": 0.2}),  # cash/cash_bag agree
            _cand("cup", 0.3, {"personal": 0.9, "food_drink": 0.1}),  # kept, disagree x0.5
            _cand("cash", 0.2, {"cash": 0.9}),  # below cash threshold
            _cand("cash", 0.3, {"paper": 0.9, "cash": 0.1}),  # IGNORED
            _cand("cash", 0.3, {"cash": 0.3, "surface": 0.2}),  # below min_category_score
        ]
    )
    cfg = ObjectsConfig(category_thresholds={"cash": 0.25})
    cfg.box_threshold = 0.25
    out = O.select_objects(raw, cfg).objects[0]
    assert [o.cat for o in out] == ["cash", "cash_bag", "personal"]
    assert np.allclose([o.score for o in out], [0.3 * 0.9, 0.3 * 0.8, 0.3 * 0.9 * cfg.disagree_factor])
    cfg2 = ObjectsConfig(category_thresholds={"cash": 0.35})
    assert [o.cat for o in O.select_objects(raw, cfg2).objects[0]] == ["personal"]


class _Lay:
    workstation = (0, 0, 100, 100)
    desk = (0, 0, 100, 100)
    laptop = counter = telephone = safe = None


def test_detect_equals_select_of_collect():
    img = np.full((100, 100, 3), 200, np.uint8)
    video = Video("x", 1.0, 2.0, 100, 100, [Frame(i, float(i), img, img, 0.0) for i in range(2)])
    layouts = SimpleNamespace(at=lambda t: _Lay())
    people = SimpleNamespace(tracks=[])
    dets = [
        Detection("banknotes", 0.3, (10, 10, 30, 30)),
        Detection("cup", 0.18, (50, 50, 70, 70)),
        Detection("tablet", 0.9, (1, 1, 5, 5)),  # no category
    ]

    def detector(images, phrases, thr, ttr, batch=1):
        return [[d for d in dets if d.score >= thr] for _ in images]

    enc = SimpleNamespace(
        embed=lambda crops: np.zeros((len(crops), 1)),
        zero_shot=lambda emb, texts: np.tile(np.eye(len(texts))[0], (len(emb), 1)),
    )
    heads = SimpleNamespace(has=lambda task: False)
    cfg = ObjectsConfig()
    raw = O.collect_object_candidates(video, layouts, people, detector, enc, heads, cfg, 1.0, floor=0.1)
    assert len(raw.candidates) == 4 and {c.dino_cat for c in raw.candidates} == {"cash", "food_drink"}
    assert raw.candidates[0].probs["cash"] == 1.0
    res = O.detect_objects(video, layouts, people, detector, enc, heads, cfg, 1.0)
    assert res.objects == O.select_objects(
        O.collect_object_candidates(video, layouts, people, detector, enc, heads, cfg, 1.0), cfg
    ).objects
    # cup (0.18) is under the global 0.25 threshold; only the banknote survives
    assert [o.cat for o in res.objects[0]] == ["cash"]


def test_score_with_tolerance():
    labels = {1: (True, 30.0), 2: (True, 10.0), 3: (False, None), 4: (False, None), 5: (True, 50.0), 9: (True, 1.0)}
    preds = {
        1: (True, [34.0]),  # tp, within tol
        2: (True, [40.0]),  # tp, wrong time
        3: (True, [5.0]),  # fp
        4: (False, []),  # tn
        5: (False, []),  # fn
    }
    m = C.score(preds, labels, tol=5.0)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 1)
    assert np.isclose(m["precision"], 2 / 3) and np.isclose(m["recall"], 2 / 3)
    assert np.isclose(m["f1"], 2 / 3) and np.isclose(m["accuracy"], 3 / 5)
    assert m["ts_hit_rate"] == 0.5
    assert C.score(preds, labels, tol=35.0)["ts_hit_rate"] == 1.0


def test_grouped_cv_keeps_groups_together(monkeypatch):
    prepared = [SimpleNamespace(clip=i, group=f"g{i // 2}") for i in range(12)]  # 6 groups of 2
    labels = {i: (i % 2 == 0, 1.0) for i in range(12)}
    grid = [C.Params(cash_box=0.1), C.Params(cash_box=0.2)]
    table = [{i: (i % 2 == 0, [1.0]) for i in range(12)}, {i: (True, [1.0]) for i in range(12)}]
    cv = C.grouped_cv(prepared, labels, grid, k=3, table=table)
    assert cv["k"] == 3
    group_of = {p.clip: p.group for p in prepared}
    seen = []
    for fold in cv["folds"]:
        test_groups = {group_of[c] for c in fold["test_clips"]}
        train_groups = {group_of[c] for c in group_of if c not in fold["test_clips"]}
        assert not test_groups & train_groups
        assert fold["params"]["cash_box"] == 0.1  # the perfect param set wins every fold
        seen += fold["test_clips"]
    assert sorted(seen) == list(range(12)) and cv["pooled"]["f1"] == 1.0
