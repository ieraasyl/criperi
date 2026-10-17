from kassa_pipeline import geometry as g
from kassa_pipeline.report import describe, rule, rule_base, type_label
from kassa_pipeline.vlm_scan import catalog


def test_geometry_basics():
    a, b = (0, 0, 10, 10), (5, 5, 15, 15)
    assert abs(g.iou(a, b) - 25 / 175) < 1e-9
    assert g.inside_share((2, 2, 4, 4), a) == 1.0
    assert g.gap((0, 0, 1, 1), (4, 5, 6, 7)) == 5.0


def test_every_rule_has_a_title_and_a_verified_quote():
    for r in rule_base()["rules"]:
        x = rule(r["id"])
        assert x["title"] and x["quote"] and x["point"] and x["url"].startswith("https://adilet.zan.kz")


def test_catalog_has_21_classes_linked_to_existing_rules():
    cat = catalog()
    assert sorted(cat) == [f"V{i:02d}" for i in range(1, 22)]
    ids = {r["id"] for r in rule_base()["rules"]}
    assert all(c["rule"] is None or c["rule"] in ids for c in cat.values())


def test_descriptions_cite_the_norm_or_say_there_is_none():
    text = describe("V14", 7)
    assert "В 7 с" in text and "п. 71" in text and "косвенная" in text
    assert "не найдено" in describe("V20", 3)
    assert type_label("V01").startswith("V01.") and "п. 20" in type_label("V01")
    assert type_label("V20") == "V20. " + catalog()["V20"]["title_ru"]
