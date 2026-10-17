from kassa_pipeline import geometry as g
from kassa_pipeline.report import describe, rule, rule_base, type_label


def test_geometry_basics():
    a, b = (0, 0, 10, 10), (5, 5, 15, 15)
    assert abs(g.iou(a, b) - 25 / 175) < 1e-9
    assert g.inside_share((2, 2, 4, 4), a) == 1.0
    assert g.gap((0, 0, 1, 1), (4, 5, 6, 7)) == 5.0


def test_every_rule_has_a_title_and_a_verified_quote():
    for r in rule_base()["rules"]:
        x = rule(r["id"])
        assert x["title"] and x["quote"] and x["point"] and x["url"].startswith("https://adilet.zan.kz")


def test_indirect_rules_are_flagged_and_worded_softly():
    assert rule("R-IS-LOCK")["indirect"] and rule("R-ENTRY-ISOLATION")["indirect"]
    assert not rule("R-CASH-LOCKED")["indirect"]
    text = describe("R-IS-LOCK", "laptop_away", 7)
    assert "косвенная" in text and "подсказкой" in text and "В 7 с" in text


def test_type_label_names_the_norm():
    assert "п. 20" in type_label("R-CASH-LOCKED") and "НБРК № 231" in type_label("R-CASH-LOCKED")
