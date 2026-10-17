from kassa_pipeline import extract as E
from kassa_pipeline import objects as O
from kassa_pipeline import people as P
from kassa_pipeline.config import PipelineConfig
from kassa_pipeline.layout import Layout, LayoutTimeline

DESK = (100.0, 100.0, 300.0, 200.0)
CASH = (150.0, 150.0, 160.0, 155.0)


def _record(hand_x: float, body_height: float, cfg: PipelineConfig) -> dict:
    lay = Layout(400, 400, desk=DESK, workstation=DESK)
    people = P.PeopleResult([], 1, [None], body_height)
    person = E._Person(P.Obs(0.0, (100.0, 50.0, 200.0, 250.0), 0.9), {"left": (hand_x, 152.0), "right": None})
    raw = E._Raw(0, 0.0, [person], [E._Object(7, "cash", CASH, 0.5)], O.StateObs("closed", 0.9), O.StateObs("off", 0.7))
    return E._record(raw, {id(person.obs): 1}, people, LayoutTimeline([0.0], [lay]), {7: "cash"}, cfg)


def test_in_hand_distance_defaults_to_pixels():
    cfg = PipelineConfig()
    assert cfg.objects.in_hand_body is None
    assert _record(160.0 + 25, 400.0, cfg)["objects"][0]["held_by"] == [1, "left"]
    assert _record(160.0 + 35, 400.0, cfg)["objects"][0]["held_by"] is None


def test_in_hand_distance_scales_with_body_height():
    cfg = PipelineConfig()
    cfg.objects.in_hand_body = 0.3
    near_small, far_small = _record(160.0 + 25, 50.0, cfg), _record(160.0 + 25, 400.0, cfg)
    assert near_small["objects"][0]["held_by"] is None  # 25 px > 0.3 x 50 px
    assert far_small["objects"][0]["held_by"] == [1, "left"]  # 25 px < 0.3 x 400 px
    assert far_small["facts"]["cash_on_desk"] is False and near_small["facts"]["cash_on_desk"] is True


def test_in_hand_distance_falls_back_to_pixels_without_body_height():
    cfg = PipelineConfig()
    cfg.objects.in_hand_body = 0.3
    assert _record(160.0 + 25, 0.0, cfg)["objects"][0]["held_by"] == [1, "left"]
