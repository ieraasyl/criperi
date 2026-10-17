import json
from pathlib import Path

from kassa_pipeline import cli, label, remote, vlm
from kassa_pipeline.facts import FactTimeline, clip_id


def s1(name):
    return {"meta": {"name": name, "duration": 2.0}, "layouts": [], "frames": []}


def stub_remote(monkeypatch, calls, fail=()):
    def fake(path, options=None):
        calls.append(("stage1", Path(path).name))
        if Path(path).name in fail:
            raise RuntimeError("stage boom")
        yield {"type": "stage", "stage": "decode", "sec": 1.0}
        yield {"type": "final", "result": s1(Path(path).name)}

    monkeypatch.setattr(remote, "extract_stream", fake)


def stub_build(monkeypatch, calls, fail=()):
    def fake(path, stage1, vlm_, step=1):
        calls.append(("build", Path(path).name, stage1["meta"]["name"], step))
        if Path(path).name in fail:
            raise ValueError("boom")
        return FactTimeline.empty(clip_id(Path(path).name), 2.0)

    monkeypatch.setattr(label, "build_draft", fake)


def test_two_phase_stage1_cached_then_vlm_once(monkeypatch, tmp_path, capsys):
    for n in ("X-1.mov", "X-2.mov", "X-3.mov"):
        (tmp_path / n).write_bytes(b"")
    out = tmp_path / "f"
    calls = []
    stub_remote(monkeypatch, calls, fail=("X-2.mov",))
    stub_build(monkeypatch, calls)
    monkeypatch.setattr(vlm, "local_vlm", lambda p: calls.append(("vlm", p)) or object())
    files = [str(tmp_path / n) for n in ("X-1.mov", "X-2.mov", "X-3.mov")]
    assert cli.main(["label", *files, "--out", str(out), "--step", "2"]) == 1
    assert calls == [("stage1", "X-1.mov"), ("stage1", "X-2.mov"), ("stage1", "X-3.mov"),
                     ("vlm", out / ".vlm_cache" / "answers.json"),
                     ("build", "X-1.mov", "X-1.mov", 2), ("build", "X-3.mov", "X-3.mov", 2)]
    assert "X-2.mov: ERROR stage boom" in capsys.readouterr().out
    assert json.loads((out / ".stage1" / "X-1.json").read_text())["meta"]["name"] == "X-1.mov"
    assert not (out / "X-2.json").exists() and (out / "X-3.json").exists()


def test_cached_stage1_skips_remote(monkeypatch, tmp_path):
    (tmp_path / "X-1.mov").write_bytes(b"")
    out = tmp_path / "f"
    (out / ".stage1").mkdir(parents=True)
    (out / ".stage1" / "X-1.json").write_text(json.dumps(s1("cached")))
    calls = []
    stub_remote(monkeypatch, calls)
    stub_build(monkeypatch, calls)
    monkeypatch.setattr(vlm, "local_vlm", lambda p: object())
    assert cli.main(["label", str(tmp_path / "X-1.mov"), "--out", str(out)]) == 0
    assert calls == [("build", "X-1.mov", "cached", 1)]


def test_skips_reviewed_and_existing_drafts(monkeypatch, tmp_path, capsys):
    vids = tmp_path / "vids"
    vids.mkdir()
    for n in ("X-1.MOV", "X-2.mov", "X-3.mov"):
        (vids / n).write_bytes(b"")
    out = tmp_path / "facts"
    out.mkdir()
    FactTimeline.empty("X-1", 2.0).save(out / "X-1.json")
    reviewed = FactTimeline.empty("X-2", 2.0)
    reviewed.status = "reviewed"
    reviewed.save(out / "X-2.json")
    before = (out / "X-2.json").read_text()
    calls = []
    stub_remote(monkeypatch, calls)
    stub_build(monkeypatch, calls)
    monkeypatch.setattr(vlm, "local_vlm", lambda p: object())
    assert cli.main(["label", str(vids), "--out", str(out)]) == 0
    assert [c for c in calls if c[0] == "build"] == [("build", "X-3.mov", "X-3.mov", 1)]
    assert (out / "X-2.json").read_text() == before
    assert "X-3" in capsys.readouterr().out
    calls.clear()
    assert cli.main(["label", str(vids), "--out", str(out), "--overwrite"]) == 0
    assert sorted(c[1] for c in calls if c[0] == "build") == ["X-1.MOV", "X-3.mov"]
    assert (out / "X-2.json").read_text() == before


def test_continues_after_draft_failure(monkeypatch, tmp_path, capsys):
    (tmp_path / "X-1.mov").write_bytes(b"")
    (tmp_path / "X-2.mov").write_bytes(b"")
    out = tmp_path / "f"
    calls = []
    stub_remote(monkeypatch, calls)
    stub_build(monkeypatch, calls, fail=("X-1.mov",))
    monkeypatch.setattr(vlm, "local_vlm", lambda p: object())
    assert cli.main(["label", str(tmp_path / "X-1.mov"), str(tmp_path / "X-2.mov"), "--out", str(out)]) == 1
    assert "X-1.mov: ERROR boom" in capsys.readouterr().out
    assert (out / "X-2.json").exists()


def test_stream_without_final_is_an_error(monkeypatch, tmp_path, capsys):
    (tmp_path / "X-1.mov").write_bytes(b"")
    monkeypatch.setattr(remote, "extract_stream", lambda p, o=None: iter([{"type": "stage"}]))
    assert cli.main(["label", str(tmp_path / "X-1.mov"), "--out", str(tmp_path / "f")]) == 1
    assert "X-1.mov: ERROR" in capsys.readouterr().out


def test_duplicate_clip_ids_fail_before_stage1(monkeypatch, tmp_path, capsys):
    vids = tmp_path / "vids"
    vids.mkdir()
    for n in ("X-1.mov", "Х-1 (2).mov", "X-2.mov"):
        (vids / n).write_bytes(b"")
    calls = []
    stub_remote(monkeypatch, calls)
    stub_build(monkeypatch, calls)
    monkeypatch.setattr(vlm, "local_vlm", lambda p: calls.append("vlm") or object())
    assert cli.main(["label", str(vids), "--out", str(tmp_path / "f")]) == 2
    assert calls == []
    cap = capsys.readouterr()
    text = cap.out + cap.err
    assert "X-1.mov" in text and "Х-1 (2).mov" in text and "X-1" in text


def test_unreadable_existing_draft_does_not_abort_the_batch(monkeypatch, tmp_path, capsys):
    vids = tmp_path / "vids"
    vids.mkdir()
    for n in ("X-1.mov", "X-2.mov", "X-3.mov"):
        (vids / n).write_bytes(b"")
    out = tmp_path / "f"
    out.mkdir()
    (out / "X-1.json").write_text("{broken", "utf8")
    (out / "X-2.json").mkdir()  # not even a file: load raises OSError, not ValueError
    calls = []
    stub_remote(monkeypatch, calls)
    stub_build(monkeypatch, calls)
    monkeypatch.setattr(vlm, "local_vlm", lambda p: object())
    assert cli.main(["label", str(vids), "--out", str(out)]) == 1
    text = capsys.readouterr().out
    assert "X-1.mov: ERROR existing draft unreadable:" in text and "X-2.mov: ERROR existing draft unreadable:" in text
    assert [c[1] for c in calls if c[0] == "build"] == ["X-3.mov"]
    assert (out / "X-1.json").read_text("utf8") == "{broken"
    calls.clear()
    (out / "X-2.json").rmdir()
    (out / "X-2.json").write_bytes(b"\xff\xfe")  # undecodable
    assert cli.main(["label", str(vids), "--out", str(out), "--overwrite"]) == 0
    assert sorted(c[1] for c in calls if c[0] == "build") == ["X-1.mov", "X-2.mov", "X-3.mov"]
