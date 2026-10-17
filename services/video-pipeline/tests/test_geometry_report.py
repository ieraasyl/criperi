from kassa_pipeline import geometry as g
from kassa_pipeline.events import Candidate
from kassa_pipeline.report import describe, rule_for, type_label


def test_geometry_basics():
    a, b = (0, 0, 10, 10), (5, 5, 15, 15)
    assert abs(g.iou(a, b) - 25 / 175) < 1e-9
    assert g.inside_share((2, 2, 4, 4), a) == 1.0
    assert g.gap((0, 0, 1, 1), (4, 5, 6, 7)) == 5.0


def test_every_type_has_a_verified_rule_with_a_quote():
    for t in ("T1", "T2", "T3", "T4", "T5", "T6", "T7", "T9"):
        r = rule_for(t)
        assert r and r["quote"] and r["point"] and r["url"].startswith("https://adilet.zan.kz")


def test_indirect_rules_are_flagged_and_worded_softly():
    assert rule_for("T5")["indirect"] and rule_for("T4")["indirect"]
    assert not rule_for("T1")["indirect"]
    text = describe(Candidate("T5", 7, 15, 7, 0.6))
    assert "косвенная" in text and "подсказкой" in text


def test_type_label_names_the_norm():
    assert "п. 20" in type_label("T1")
