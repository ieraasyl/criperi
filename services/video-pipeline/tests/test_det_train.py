import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from kassa_pipeline import det_train as T


def _coco(tmp_path, cats=((1, "cash"), (2, "id_card"))):
    data = {"images": [{"id": 1, "file_name": "a.jpg", "clip": "Х_3"}, {"id": 2, "file_name": "b.jpg", "clip": "Х_5"},
                       {"id": 3, "file_name": "c.jpg", "clip": "X-17"}],
            "annotations": [{"id": 1, "image_id": 2, "category_id": 1, "bbox": [0, 0, 5, 5]}],
            "categories": [{"id": i, "name": n} for i, n in cats]}
    p = tmp_path / "c.json"
    p.write_text(json.dumps(data))
    return p


def test_load_coco_and_split_by_clip(tmp_path):
    ims, anns, classes = T.load_coco(_coco(tmp_path))
    assert classes == ["cash", "id_card"] and list(anns) == [2]
    tr, va = T.split(ims)
    assert tr == [2] and va == [1, 3]  # "Х_3" (Cyrillic, underscore) is the held-out X-3


def test_load_coco_rejects_gaps_in_category_ids(tmp_path):
    with pytest.raises(ValueError):
        T.load_coco(_coco(tmp_path, cats=((1, "cash"), (3, "id_card"))))


def test_scores_ap_precision_recall():
    gt = {"boxes": np.array([[0, 0, 10, 10], [20, 20, 30, 30]]), "labels": np.array([0, 0])}
    pred = {"boxes": np.array([[0, 0, 10, 10], [50, 50, 60, 60], [21, 21, 30, 30]]),
            "labels": np.array([0, 0, 0]), "scores": np.array([0.9, 0.8, 0.3])}
    s = T.scores([pred], [gt], ["cash", "id_card"])["per_class"]
    assert s["cash"]["precision@0.5"] == 0.5 and s["cash"]["recall@0.5"] == 0.5
    assert 0.5 < s["cash"]["ap50"] < 1.0  # third box finds the second object, after one false positive
    assert s["id_card"]["ap50"] is None and s["id_card"]["gt"] == 0


def test_scores_perfect_and_duplicate():
    gt = {"boxes": np.array([[0, 0, 10, 10]]), "labels": np.array([0])}
    dup = {"boxes": np.array([[0, 0, 10, 10], [0, 0, 10, 10]]), "labels": np.array([0, 0]), "scores": np.array([0.9, 0.8])}
    s = T.scores([dup], [gt], ["person"])
    assert s["map50"] == 1.0 and s["per_class"]["person"]["precision@0.5"] == 0.5  # the duplicate is a false positive


def test_person_dataset_to_coco():
    spec = importlib.util.spec_from_file_location("person_dataset", Path(__file__).parents[1] / "scripts" / "person_dataset.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    data = {"file_name": "X-1_000.00.jpg", "width": 200, "height": 100, "clip": "X-1", "t": 0.0}
    box = {"type": "rectanglelabels", "original_width": 200, "original_height": 100,
           "value": {"x": 10, "y": 20, "width": 50, "height": 50, "rectanglelabels": ["person"]}}
    coco = mod.to_coco([{"data": data, "annotations": [{"result": [box]}]}, {"data": data, "annotations": []}])
    assert len(coco["images"]) == 1 and coco["annotations"][0]["bbox"] == [20.0, 20.0, 100.0, 50.0]


def test_base_class_ids_keep_known_heads():
    coco80 = {0: "person", 1: "bicycle", 41: "cup"}
    assert T.base_class_ids(coco80, ["person"]) == [0]
    assert T.base_class_ids(coco80, ["cup", "person"]) == [41, 0]
    assert T.base_class_ids(coco80, ["cash", "person"]) is None
