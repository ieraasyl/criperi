import json

from factsutil import make_facts
from kassa_pipeline import cli, facts_eval

ITEMS = ("desk_cash", "desk_bag", "desk_docs", "desk_personal")
EXIT = {"t": 8.4, "type": "room_exit", "who": "cashier"}


def timeline(clip, cash="n", status="reviewed", exit_t=8.4):
    states = {k: "n" * 10 for k in ITEMS}
    states["desk_cash"] = cash * 10
    states["laptop_lid"] = "c" * 10
    ft = make_facts(10, states, [{**EXIT, "t": exit_t}])
    ft.clip = clip
    ft.status = status
    return ft


def write_set(tmp_path):
    d = tmp_path / "facts"
    d.mkdir()
    timeline("X-1", cash="y").save(d / "X-1.json")  # TP, violation at 8.4
    timeline("X-2", cash="y").save(d / "X-2.json")  # FP
    timeline("X-3").save(d / "X-3.json")  # FN
    timeline("X-4").save(d / "X-4.json")  # TN
    timeline("X-5", cash="y", status="draft").save(d / "X-5.json")  # ignored
    timeline("X-9").save(d / "X-9.json")  # no label -> unlabelled
    return d


LABELS = {1: (True, 9.0), 2: (False, None), 3: (True, 5.0), 4: (False, None), 5: (True, 1.0)}


def test_rules_on_truth_counts_confusion(tmp_path):
    r = facts_eval.rules_on_truth(write_set(tmp_path), LABELS)
    t = r["totals"]
    assert (t["tp"], t["fp"], t["fn"], t["tn"]) == (1, 1, 1, 1)
    assert t["precision"] == 0.5 and t["recall"] == 0.5 and t["accuracy"] == 0.5
    assert sorted(c["clip"] for c in r["clips"]) == ["X-1", "X-2", "X-3", "X-4"]
    assert r["unlabelled"] == ["X-9"]
    c1 = next(c for c in r["clips"] if c["clip"] == "X-1")
    assert c1["predicted_violation"] and c1["label_violation"] and c1["label_sec"] == 9.0
    assert c1["violations"][0]["rule_id"] == "W02"
    assert c1["violations"][0]["timecode_sec"] == 8.4


def test_timecode_error_uses_nearest_violation(tmp_path):
    d = tmp_path / "facts"
    d.mkdir()
    ft = timeline("X-1", cash="y")
    ft.states["desk_bag"] = ["yes"] * 10
    ft.events = [
        facts_eval_event(2.0), facts_eval_event(8.4),
    ]
    ft.save(d / "X-1.json")
    r = facts_eval.rules_on_truth(d, {1: (True, 9.0)})
    tc = r["totals"]["timecode"]
    assert abs(tc["median_error_sec"] - 0.6) < 1e-9
    assert tc["within_2"] == 1.0 and tc["within_5"] == 1.0
    assert facts_eval.rules_on_truth(d, {1: (True, None)})["totals"]["timecode"]["median_error_sec"] is None


def facts_eval_event(t):
    from kassa_pipeline.facts import Event
    return Event(t=t, type="room_exit", who="cashier")


def test_perception_state_accuracy_ignores_unknown_truth():
    truth = make_facts(4, {"safe_door": "oc??"})
    pred = make_facts(4, {"safe_door": "o?oc"})
    p = facts_eval.perception_vs_truth(pred, truth)
    s = p["states"]["safe_door"]
    assert s["n"] == 1 and s["accuracy"] == 1.0  # only second 0 has both known; sec 1 pred unknown
    assert s["pred_unknown_rate"] == 0.5  # 2 truth-known seconds, 1 pred unknown
    assert "screen" not in p["states"]  # absent from both (all unknown arrays still present -> n == 0 skipped)


def test_event_matching_within_tolerance_one_to_one():
    truth = make_facts(30, events=[
        {"t": 5.0, "type": "room_exit", "who": "cashier"},
        {"t": 6.0, "type": "room_exit", "who": "cashier"},
        {"t": 10.0, "type": "room_entry", "who": "client-1"},
    ])
    pred = make_facts(30, events=[
        {"t": 5.5, "type": "room_exit", "who": "cashier"},
        {"t": 20.0, "type": "room_exit", "who": "cashier"},
        {"t": 10.5, "type": "room_entry", "who": "other-2"},
        {"t": 11.0, "type": "cash_received", "who": "cashier"},
    ], observable=["room_exit", "room_entry"])
    ev = facts_eval.perception_vs_truth(pred, truth)["events"]
    assert set(ev) == {"room_exit", "room_entry"}
    assert ev["room_exit"]["tp"] == 1 and ev["room_exit"]["precision"] == 0.5
    assert ev["room_exit"]["recall"] == 0.5
    assert ev["room_entry"]["tp"] == 1  # client-1 and other-2 are both "client"


def test_cli_facts_mode_writes_report(tmp_path, monkeypatch):
    d = write_set(tmp_path)
    monkeypatch.setattr(cli, "load_labels", lambda p: LABELS)
    pred = tmp_path / "pred"
    pred.mkdir()
    (pred / "Х-1.json").write_text(json.dumps({"facts": timeline("X-1", cash="y").to_dict()}), "utf8")
    (pred / "X-2.json").write_text(json.dumps({"facts": None}), "utf8")
    out = tmp_path / "sub" / "r.md"
    rc = cli.main(["facts-eval", "--facts", str(d), "--labels", "x.xlsx", "--pred", str(pred), "--out", str(out)])
    assert rc == 0
    text = out.read_text("utf8")
    assert "X-1" in text and "precision" in text.lower() and "X-2" in text
    assert "missing" in text.lower()


def _stage1_json(seconds=10):
    lay = {"start": 0.0, "width": 200, "height": 150, "desk": [10, 80, 110, 140], "laptop": None, "safe": None, "doors": []}
    frames = [{"i": i, "t": i * 0.5, "people": [], "objects": [],
               "facts": {"cashier_zone": "absent", "safe": "unknown", "screen": "unknown"}} for i in range(seconds * 2)]
    return {"meta": {"name": "X-3.mov", "duration": float(seconds), "sample_fps": 2},
            "layouts": [lay], "cashier_track": None, "people": [], "objects": [], "frames": frames}


def test_perception_accepts_timeline_debug_and_stage1_json(tmp_path):
    d = write_set(tmp_path)
    pred = tmp_path / "pred"
    pred.mkdir()
    (pred / "Х-1.json").write_text(json.dumps(timeline("X-1", cash="y").to_dict()), "utf8")  # (a) timeline as-is
    (pred / "X-2 (2).json").write_text(json.dumps({"facts": timeline("X-2", cash="y").to_dict()}), "utf8")  # (b) debug
    (pred / "X-3.json").write_text(json.dumps(_stage1_json()), "utf8")  # (c) Stage 1, converted
    (pred / "X-4.json").write_text("[1, 2]", "utf8")  # not a dict
    (pred / "X-9.json").write_text("{not json", "utf8")  # malformed
    p = facts_eval.perception_on_dir(d, pred)
    assert p["n_clips"] == 3
    assert p["states"]["desk_cash"]["correct"] == 30  # each clip agrees with its truth for all 10 s
    missing = dict(m.split(": ", 1) for m in p["missing"])
    assert set(missing) == {"X-4", "X-9"} and all(missing.values())


def test_perception_bad_timeline_and_bad_stage1_are_missing_with_reason(tmp_path):
    d = write_set(tmp_path)
    pred = tmp_path / "pred"
    pred.mkdir()
    bad = timeline("X-1").to_dict()
    bad["states"]["desk_cash"] = ["maybe"] * 10
    (pred / "X-1.json").write_text(json.dumps(bad), "utf8")
    (pred / "X-2.json").write_text(json.dumps({"frames": [], "layouts": []}), "utf8")  # Stage 1 without meta
    (pred / "X-3.json").write_text(json.dumps({"facts": None}), "utf8")
    p = facts_eval.perception_on_dir(d, pred)
    assert p["n_clips"] == 0
    missing = dict(m.split(": ", 1) if ": " in m else (m, "") for m in p["missing"])
    assert set(missing) == {"X-1", "X-2", "X-3", "X-4", "X-9"}
    assert missing["X-1"] and missing["X-2"] and missing["X-3"] and missing["X-4"] == ""


def test_cli_facts_eval_help_mentions_stage1_dir(capsys):
    import pytest
    with pytest.raises(SystemExit):
        cli.main(["facts-eval", "--help"])
    assert ".stage1" in capsys.readouterr().out
