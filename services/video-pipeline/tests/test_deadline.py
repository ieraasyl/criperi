import types

import kassa_pipeline.vlm_scan as vs
from kassa_pipeline.config import PipelineConfig, ScanConfig
from kassa_pipeline.vlm import QwenVLM


def _fake_qwen(cached=()):
    q = QwenVLM.__new__(QwenVLM)
    q.calls = q.hits = q.skipped = 0
    q.gpu_seconds = 0.0
    q._cache = {f"k{i}": f"cached{i}" for i in cached}
    q._prepare = lambda images, prompt: (f"k{prompt}", prompt)
    q.batches = []

    def gen(convs, max_new_tokens):
        q.batches.append(list(convs))
        return [f"ans{c}" for c in convs]

    q._generate = gen
    return q


ITEMS = [([], str(i)) for i in range(6)]


def test_ask_many_no_deadline_is_unchanged():
    q = _fake_qwen()
    assert q.ask_many(ITEMS, batch=4) == [f"ans{i}" for i in range(6)]
    assert len(q.batches) == 2 and q.skipped == 0


def test_ask_many_past_deadline_returns_none_and_counts_skipped():
    q = _fake_qwen(cached=[1])
    out = q.ask_many(ITEMS, batch=4, deadline=0.0)
    assert out == [None, "cached1", None, None, None, None]
    assert q.batches == [] and q.skipped == 5


def test_ask_many_stops_between_batches(monkeypatch):
    q = _fake_qwen()
    clock = iter([100.0, 100.0, 100.0])  # check, start, end of batch 1; then past the deadline
    monkeypatch.setattr("kassa_pipeline.vlm.time.time", lambda: next(clock, 300.0))
    out = q.ask_many(ITEMS, batch=4, deadline=200.0)
    assert out[:4] == [f"ans{i}" for i in range(4)] and out[4:] == [None, None]
    assert q.skipped == 2 and len(q.batches) == 1


def _scan(monkeypatch, cfg, answers, **kw):
    monkeypatch.setattr(vs, "_build", lambda *a, **k: ([], []))
    seen = {}

    def ask_many(items, **k):
        seen.update(k)
        return answers(len(items))

    vlm = types.SimpleNamespace(ask_many=ask_many, skipped=0)
    s1 = {"frames": [], "layouts": [{"start": 0.0}], "people": [], "meta": {"name": "X-1.mov", "duration": 20.0}}
    res = vs.scan(types.SimpleNamespace(duration=20.0), s1, vlm, cfg, **kw)
    return res, seen


def test_scan_passes_deadline(monkeypatch):
    cfg = PipelineConfig(scan=ScanConfig(time_budget_sec=270.0, finish_reserve_sec=25.0))
    _, seen = _scan(monkeypatch, cfg, lambda n: ["{}"] * n, started=1000.0)
    assert seen["deadline"] == 1000.0 + 270.0 - 25.0


def test_scan_without_started_or_budget_has_no_deadline(monkeypatch):
    _, seen = _scan(monkeypatch, PipelineConfig(), lambda n: ["{}"] * n)
    assert seen["deadline"] is None
    cfg = PipelineConfig(scan=ScanConfig(time_budget_sec=None))
    _, seen = _scan(monkeypatch, cfg, lambda n: ["{}"] * n, started=1.0)
    assert seen["deadline"] is None


def test_scan_truncated_missing_answers_make_no_findings(monkeypatch):
    res, _ = _scan(monkeypatch, PipelineConfig(), lambda n: [None] * n, started=1.0)
    assert res["findings"] == [] and res["verdict"] == "no_violation"
    assert res["truncated"] is True and res["vlm_skipped"] == len(res["windows"]) > 0
    assert all(not w["parsed"] for w in res["windows"])


def test_scan_complete_is_not_truncated(monkeypatch):
    res, _ = _scan(monkeypatch, PipelineConfig(), lambda n: ["{}"] * n)
    assert res["truncated"] is False and res["vlm_skipped"] == 0


def test_defaults_and_service_wiring():
    sc = ScanConfig()
    assert sc.time_budget_sec == 270.0 and sc.finish_reserve_sec == 25.0
    from pathlib import Path
    src = (Path(__file__).parents[1] / "vlm_service" / "kassa_service.py").read_text()
    assert "started=started" in src and "started = time.time()" in src
