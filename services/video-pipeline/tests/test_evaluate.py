import io
import zipfile

from kassa_pipeline.cli import load_labels
from kassa_pipeline.evaluate import ClipResult, per_type, summarize, sweep


def R(name, label, sec=None, cands=(), error=None):
    return ClipResult(name, None, label, sec, 20.0, [dict(type=t, timecode=tc, score=s) for t, tc, s in cands], {}, error)


RESULTS = [
    R("a", True, 9, [("T1", 10, 0.9)]),                       # hit, 1 s late
    R("b", True, 20, [("T5", 7, 0.6), ("T2", 18, 0.45)]),     # hit only above/below depending on thr
    R("c", True, 5, [("T3", 2, 0.4)]),                        # miss at 0.5
    R("d", False, None, [("T1", 12, 0.7)]),                   # false alarm
    R("e", False, None, []),                                  # correct
    R("f", None, None, [("T1", 1, 0.9)]),                     # unlabeled: ignored
    R("g", True, 3, error="boom"),                            # failed: ignored
]


def test_confusion_and_lists():
    s = summarize(RESULTS, 0.5)
    assert (s["tp"], s["fp"], s["fn"], s["tn"]) == (2, 1, 1, 1)
    assert s["fn_clips"] == ["c"] and s["fp_clips"] == ["d"]
    assert s["clips"] == 5
    assert abs(s["precision"] - 2 / 3) < 1e-9 and abs(s["recall"] - 2 / 3) < 1e-9


def test_timecode_uses_first_prediction_and_nearest():
    s = summarize(RESULTS, 0.5)["timecode"]
    assert s["n"] == 2
    # clip a: +1 s; clip b: first prediction at 7 vs label 20 = -13 s, nearest is 7 (|13|) since 18 is below thr
    assert s["median_abs_first"] == 7.0 and s["within_2s_first"] == 0.5


def test_lower_threshold_catches_more_but_raises_false_alarms():
    low, high = summarize(RESULTS, 0.3), summarize(RESULTS, 0.8)
    assert low["fn"] == 0 and high["fn"] == 2  # at 0.8 only clip a's 0.9 survives... and d's 0.7 drops
    assert high["fp"] == 0
    assert len(sweep(RESULTS)) == 7


def test_per_type_counts_split_by_clip_kind():
    t = per_type(RESULTS, 0.5)
    assert t["T1"] == {"on_violation_clips": 1, "on_control_clips": 1}
    assert "T3" not in t  # below threshold


def test_load_labels_reads_minutes_and_seconds():
    rows = [
        ("наименование файла", "таймстемп", "нарушение / без нарушения"),
        ("X-1", "22 секунда", "нарушение"),
        ("Х-3", None, "без нарушения"),
        ("Х-47", "1минута 8 секунда", "нарушение"),
    ]

    def cell(v):
        return f'<c t="inlineStr"><is><t>{v}</t></is></c>' if v is not None else ""

    xml = "<sheetData>" + "".join(f"<row>{''.join(cell(v) for v in r)}</row>" for r in rows) + "</sheetData>"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("xl/worksheets/sheet1.xml", xml)
    p = io.BytesIO(buf.getvalue())
    import tempfile, pathlib

    f = pathlib.Path(tempfile.mkdtemp()) / "l.xlsx"
    f.write_bytes(p.getvalue())
    lab = load_labels(f)
    assert lab == {1: (True, 22.0), 3: (False, None), 47: (True, 68.0)}
