"""Command line: analyze one video or a folder, dump JSON, and compare with open_labels.xlsx."""

import argparse
import json
import time
import re
import sys
import zipfile
from pathlib import Path

from kassa_pipeline.config import PipelineConfig
from kassa_pipeline.pipeline import analyze_video


def load_labels(xlsx: Path) -> dict[int, tuple[bool, float | None]]:
    """clip number -> (has violation, labeled second). Reads the xlsx without extra dependencies."""
    z = zipfile.ZipFile(xlsx)
    xml = z.read("xl/worksheets/sheet1.xml").decode("utf8")
    labels = {}
    for row in re.findall(r"<row.*?</row>", xml, flags=re.S)[1:]:
        cells = [re.sub(r"<.*?>", "", c) for c in re.findall(r"<c .*?</c>", row, flags=re.S)]
        m = re.search(r"(\d+)\s*$", cells[0])
        if not m:
            continue
        viol = "без" not in cells[-1]
        sec = None
        if viol and len(cells) >= 3:
            mm = re.search(r"(?:(\d+)\s*минут\w*\s*)?(\d+)\s*секунд", cells[1])
            if mm:
                sec = float(mm.group(2)) + 60 * float(mm.group(1) or 0)
        labels[int(m.group(1))] = (viol, sec)
    return labels


def clip_number(p: Path) -> int | None:
    m = re.search(r"-(\d+)$", p.stem)
    return int(m.group(1)) if m else None


def _demo(args) -> int:
    import platform
    import subprocess

    from kassa_pipeline.render import play_window, render_replay

    f = Path(args.path)
    t_start = time.time()

    def log(stage: str, sec: float, info: dict):
        extra = ""
        if stage == "decode":
            extra = f"{info['frames']} кадров, {info['duration']:.0f} с видео"
        elif stage == "layout":
            extra = f"сегментов камеры: {len(info['layouts'].layouts)}"
        elif stage == "people":
            p = info["people"]
            extra = f"треков: {len(p.tracks)}, кассир: трек {p.cashier_id}"
        elif stage == "objects":
            extra = f"объектов на столе: {sum(len(v) for v in info['objects'].objects.values())}"
        elif stage == "events":
            extra = f"кандидатов: {len(info['candidates'])}"
        elif stage == "report":
            extra = f"нарушений: {len(info['violations'])}"
        print(f"  [{stage:8s}] {sec:5.1f} с  {extra}", flush=True)

    print(f"Анализ {f.name}", flush=True)
    res = analyze_video(f, progress=log)
    print(f"Готово за {time.time() - t_start:.0f} с (видео {res.video.duration:.0f} с). Вердикт: {res.verdict}")
    for v in res.violations:
        print(f"  {v.candidate.timecode:5.1f} с  {v.type_label}  (уверенность {v.candidate.score:.2f})")
        print(f"        {v.description[:200]}...")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{f.stem}.json").write_text(json.dumps(res.to_debug(), ensure_ascii=False, indent=1), "utf8")
    if args.window:
        play_window(res)
    else:
        mp4 = render_replay(res, out / f"{f.stem}_replay.mp4")
        print(f"Реплей: {mp4}")
        if not args.no_open:
            opener = "open" if platform.system() == "Darwin" else "xdg-open"
            subprocess.run([opener, str(mp4)], check=False)
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "demo":
        dp = argparse.ArgumentParser(prog="kassa demo", description="analyze one video and play the replay")
        dp.add_argument("path")
        dp.add_argument("--out", default="runs")
        dp.add_argument("--window", action="store_true", help="play in a window at real-time speed instead of writing an mp4")
        dp.add_argument("--no-open", action="store_true", help="write the mp4 but do not open it")
        return _demo(dp.parse_args(argv[1:]))
    if argv and argv[0] == "export-crops":
        from kassa_pipeline.config import PipelineConfig
        from kassa_pipeline.heads import export_crops

        ep = argparse.ArgumentParser(prog="kassa export-crops")
        ep.add_argument("videos", help="folder of .mov files")
        ep.add_argument("--out", default="crops")
        a = ep.parse_args(argv[1:])
        n = export_crops(sorted(Path(a.videos).glob("*.mov")), Path(a.out), PipelineConfig())
        print(f"{n} crops written to {a.out}/<task>/unsorted; move them into crops/<task>/<class>/")
        return 0
    if argv and argv[0] == "train-heads":
        from kassa_pipeline.config import PipelineConfig
        from kassa_pipeline.heads import train_heads

        tp = argparse.ArgumentParser(prog="kassa train-heads")
        tp.add_argument("--crops", default="crops")
        tp.add_argument("--out", default="heads")
        a = tp.parse_args(argv[1:])
        print(json.dumps(train_heads(Path(a.crops), Path(a.out), PipelineConfig()), ensure_ascii=False, indent=1))
        return 0
    if argv and argv[0] == "calibrate":
        return _calibrate(argv[1:])
    if argv and argv[0] == "run":
        argv = argv[1:]
    return _main_run(argv)


def _calibrate(argv: list[str]) -> int:
    from kassa_pipeline import calibrate as C

    ap = argparse.ArgumentParser(prog="kassa calibrate")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("prepare", help="run the expensive stages once per clip and cache them")
    pp.add_argument("videos", help="folder of .mov files")
    pp.add_argument("--out", default="calib")
    pp.add_argument("--det-cache", help="folder caching raw DINO detections between runs")
    sp = sub.add_parser("sweep", help="sweep thresholds offline over prepared clips")
    sp.add_argument("calib", help="folder written by `calibrate prepare`")
    sp.add_argument("--labels", required=True, help="open_labels.xlsx")
    sp.add_argument("--tol", type=float, default=5.0, help="timestamp tolerance, seconds")
    sp.add_argument("--out", default="calib/report.json")
    a = ap.parse_args(argv)

    if a.cmd == "prepare":
        out = Path(a.out)
        cfg = PipelineConfig()
        for f in sorted(Path(a.videos).glob("*.mov")):
            if (out / f"{f.stem}.pkl").exists():
                print(f"{f.name}: cached", file=sys.stderr)
                continue
            print(f.name, file=sys.stderr, flush=True)
            C.save(C.prepare(f, cfg, C.FLOOR, a.det_cache), out, f.stem)
        return 0

    prepared = [C.load(f) for f in sorted(Path(a.calib).glob("*.pkl"))]
    rep = C.sweep(prepared, load_labels(Path(a.labels)), tol=a.tol)
    Path(a.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf8")
    cols = ("tp", "fp", "fn", "tn", "precision", "recall", "f1", "accuracy", "ts_hit_rate")
    print(f"{'':16s}" + "".join(f"{c:>10s}" for c in cols))
    for name, m in (
        ("default", rep["default"]),
        ("best in-sample", rep["in_sample"][0]),
        ("grouped CV", rep["cv"]["pooled"]),
    ):
        print(f"{name:16s}" + "".join(f"{m[c]:10.2f}" if isinstance(m[c], float) else f"{m[c]:10d}" for c in cols))
    print(f"best params: {rep['in_sample'][0]['params']}")
    print(f"report: {a.out}")
    return 0


def _main_run(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="kassa run")
    ap.add_argument("path", help="a .mov file or a folder of them")
    ap.add_argument("--out", default="runs", help="folder for debug JSON per video")
    ap.add_argument("--labels", help="open_labels.xlsx, to print a comparison")
    ap.add_argument("--no-evidence", action="store_true")
    ap.add_argument("--det-cache", help="folder caching raw DINO detections between runs")
    args = ap.parse_args(argv)

    p = Path(args.path)
    files = sorted(p.glob("*.mov")) if p.is_dir() else [p]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    labels = load_labels(Path(args.labels)) if args.labels else {}
    cfg = PipelineConfig()
    cfg.models.detection_cache_dir = args.det_cache

    for f in files:
        def log(stage: str, sec: float, info: dict):
            print(f"  {stage:9s} {sec:5.1f}s", file=sys.stderr)

        print(f"{f.name}", file=sys.stderr)
        res = analyze_video(f, cfg, progress=log, with_evidence=not args.no_evidence)
        (out / f"{f.stem}.json").write_text(
            json.dumps(res.to_debug(), ensure_ascii=False, indent=1), encoding="utf8"
        )
        n = clip_number(f)
        truth = labels.get(n)
        pred = [f"{v.candidate.type}@{v.candidate.timecode:.0f}s" for v in res.violations]
        line = f"{f.name}: {res.verdict} {pred}"
        if truth:
            line += f" | label: {'violation' if truth[0] else 'no_violation'}" + (
                f" @{truth[1]:.0f}s" if truth[1] is not None else ""
            )
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
