import subprocess
import sys

import pytest

import kassa_pipeline.vlm_scan as vs
from kassa_pipeline.config import ScanConfig
from kassa_pipeline.scenarios.base import RuleResult
from kassa_pipeline.vlm_scan import Finding, Stage1, apply_rules


def rr(rule, status="violation", subject=None, t=10.0):
    return RuleResult(rule, "project", status, timecode_sec=t, subject=subject, facts=[f"{rule} fact"])


ALL = ("W01", "W02", "W03", "W05", "W06", "P01", "P02", "P03")


def _map(results, enabled=ALL, conf=0.45):
    from kassa_pipeline.scenarios.findings import scenario_findings
    return scenario_findings(results, enabled, conf)


@pytest.mark.parametrize("rule,subject,code", [
    ("W02", "desk_cash", "V01"), ("W02", "desk_bag", "V01"), ("W02", "desk_personal", "V15"),
    ("W03", None, "V14"), ("W01", None, "V12"), ("W05", None, "V02"), ("W06", None, "V20"),
    ("P01", None, "V04"), ("P02", None, "V05"), ("P03", None, "V05"),
])
def test_mapping_table(rule, subject, code):
    (f,) = _map([rr(rule, subject=subject)])
    assert (f.code, f.t, f.confidence, f.source) == (code, 10.0, 0.45, f"scenario {rule}")
    assert f.evidence == [f"{rule} fact"]


def test_docs_and_non_violations_and_disabled_give_nothing():
    assert _map([rr("W02", subject="desk_docs")]) == []
    assert _map([rr("W03", status=s) for s in ("pass", "unknown", "not_applicable")]) == []
    assert _map([rr("W03")], enabled=("W01",)) == []
    assert _map([rr("W03", t=None)]) == []


def test_fresh_interpreter_imports():
    for mod in ("kassa_pipeline.scenarios.findings", "kassa_pipeline.vlm_scan"):
        subprocess.run([sys.executable, "-c", f"import {mod}"], check=True)


def _frame(t, zone="cashier_seat"):
    return {"t": t, "people": [], "objects": [],
            "facts": {"cashier_zone": zone, "cashier_at_workplace": zone == "cashier_seat", "screen": "unknown", "people_in_room": 1}}


def _s1():
    return Stage1({"frames": [_frame(t / 4) for t in range(20)], "layouts": [{"start": 0.0}], "people": [],
                   "meta": {"name": "X-20.mov", "duration": 5.0}})


def _win(t0, t1, state):
    st = {"cash_on_open_desk": ("no", None)}
    st.update(state)
    return {"t0": t0, "t1": t1, "state": st, "events": []}


def _key(fs):
    return [(f.code, f.t, f.confidence, f.source, f.evidence) for f in fs]


def test_default_off_never_touches_scenarios(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("scenarios must not run")
    monkeypatch.setattr("kassa_pipeline.scenarios.run_scenarios", boom, raising=False)
    wins = [_win(0, 8, {"food_or_drink_on_desk": ("yes", 1.0)}), _win(4, 12, {"food_or_drink_on_desk": ("yes", 1.0)})]
    out = apply_rules(wins, [], [], _s1(), ScanConfig())
    assert _key(out) == [("V17", 1.0, 0.55, "food or drink at the workplace", ["1.0-12.0s"])]  # pre-change output


def test_enabled_scenario_boosts_vlm_finding_and_lone_stays_below_threshold(monkeypatch):
    sc = ScanConfig(scenario_rules=("W03",))
    res = [rr("W03", t=2.0), rr("W01", t=3.0)]
    monkeypatch.setattr(vs, "_scenario_results", lambda s1: (res, None))
    vlm = Finding("V14", 4.0, 0.6, "vlm")
    monkeypatch.setattr(vs, "people_rule", lambda *a: [vlm])
    out = apply_rules([], [], [], _s1(), sc)
    (f,) = out
    assert f.code == "V14" and f.confidence > 0.6 and "scenario W03" in f.source
    monkeypatch.setattr(vs, "people_rule", lambda *a: [])
    (lone,) = apply_rules([], [], [], _s1(), sc)
    assert lone.confidence == 0.45 < sc.threshold


def test_adapter_exception_is_recorded_and_findings_unchanged(monkeypatch):
    import kassa_pipeline.scenarios.stage1_facts as sf
    monkeypatch.setattr(sf, "facts_from_stage1", lambda *a: (_ for _ in ()).throw(ValueError("bad")))
    res, err = vs._scenario_results(_s1())
    assert res == [] and err == "ValueError: bad"
    out = apply_rules([], [], [], _s1(), ScanConfig(scenario_rules=("W03",)))
    assert out == apply_rules([], [], [], _s1(), ScanConfig())


def test_clip_id_from_meta_with_fallback(monkeypatch):
    seen = []
    import kassa_pipeline.scenarios.stage1_facts as sf
    monkeypatch.setattr(sf, "facts_from_stage1", lambda s, clip: seen.append(clip) or (_ for _ in ()).throw(RuntimeError()))
    vs._scenario_results(_s1())
    s = _s1()
    s.r["meta"] = {"duration": 5.0}
    vs._scenario_results(s)
    assert seen == ["X-20", "clip"]
