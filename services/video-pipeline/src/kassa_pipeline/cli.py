"""Command line: analyze one video or a folder, dump JSON, and compare with open_labels.xlsx."""

import argparse
import json
import time
import re
import sys
import zipfile
from pathlib import Path

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

    from kassa_pipeline.pipeline import AnalysisResult
    from kassa_pipeline.render import play_window, render_replay

    f = Path(args.path)
    t0 = time.time()
    print(f"Анализ {f.name} на Modal (CV + Qwen3.5-9B)...", flush=True)
    if args.local:
        res = analyze_video(f, progress=lambda s, sec, i: print(f"  [{s:8s}] {sec:5.1f} с", flush=True))
        out_dbg = res.to_debug()
    else:
        from kassa_pipeline.remote import analyze_remote

        r = analyze_remote(f, {"with_evidence": False})
        res, out_dbg = AnalysisResult.from_blob(r["blob"], f), r["debug"]
        for k, v in out_dbg["timings"].items():
            print(f"  [{k:8s}] {v:5.1f} с", flush=True)
        det = out_dbg["detectors"]
        print(f"  VLM: {out_dbg['vlm']}  ошибки детекторов: {det.get('errors') or 'нет'}")
    print(f"Готово за {time.time() - t0:.0f} с (видео {res.video.duration:.0f} с). Вердикт: {res.verdict}")
    for v in res.violations:
        c = v.candidate
        print(f"  {c.timecode:5.1f} с  {c.type}  {v.type_label}  [{getattr(c, 'source', 'cv')}] уверенность {c.score:.2f}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{f.stem}.json").write_text(json.dumps(out_dbg, ensure_ascii=False, indent=1), "utf8")
    if args.window:
        play_window(res)
    else:
        mp4 = render_replay(res, out / f"{f.stem}_replay.mp4")
        print(f"Реплей: {mp4}")
        if not args.no_open:
            subprocess.run(["open" if platform.system() == "Darwin" else "xdg-open", str(mp4)], check=False)
    return 0


def _crops(cmd: str, argv: list[str]) -> int:
    from kassa_pipeline import heads as H
    from kassa_pipeline.config import PipelineConfig

    ap = argparse.ArgumentParser(prog=f"kassa {cmd}")
    ap.add_argument("--crops", default="crops")
    if cmd == "export-crops":
        ap.add_argument("videos", help="folder of .mov files")
        ap.add_argument("--cache", default="cache", help="reuse cached layouts from kassa eval")
        ap.add_argument("--only", help="comma-separated clip numbers")
    if cmd in ("review", "relabel", "accept"):
        ap.add_argument("--task", required=True, choices=list(H.TASKS))
    if cmd in ("review", "accept"):
        ap.add_argument("--cls", required=(cmd == "accept"))
    if cmd == "review":
        ap.add_argument("--all", action="store_true", help="include already reviewed crops")
    if cmd == "relabel":
        ap.add_argument("--ids", required=True, help="comma-separated crop ids, e.g. 12,15,19")
        ap.add_argument("--to", required=True)
    if cmd == "train-heads":
        ap.add_argument("--out", default="heads")
    a = ap.parse_args(argv)
    crops = Path(a.crops)
    if cmd == "export-crops":
        files = sorted(Path(a.videos).glob("*.mov"))
        if a.only:
            want = {int(x) for x in a.only.split(",")}
            files = [f for f in files if clip_number(f) in want]
        n = H.export_crops(files, crops, PipelineConfig(), Path(a.cache) if a.cache else None, log=lambda m: print(m, flush=True))
        print(f"{n} new crops; next: kassa crops-status, then kassa review --task object --cls cash")
    elif cmd == "review":
        for p in H.review_sheets(crops, a.task, a.cls, crops / "review", only_unreviewed=not a.all):
            print(p)
    elif cmd == "relabel":
        print(H.relabel(crops, a.task, [int(x) for x in a.ids.split(",")], a.to), "crops moved")
    elif cmd == "accept":
        print(H.accept(crops, a.task, a.cls), "crops marked reviewed")
    elif cmd == "crops-status":
        print(json.dumps(H.status(crops), ensure_ascii=False, indent=1))
    elif cmd == "train-heads":
        print(json.dumps(H.train_heads(crops, Path(a.out), PipelineConfig()), ensure_ascii=False, indent=1))
    return 0


def _eval(argv: list[str]) -> int:
    from kassa_pipeline import evaluate as EV
    from kassa_pipeline.config import PipelineConfig

    ap = argparse.ArgumentParser(prog="kassa eval", description="score the pipeline on labeled clips")
    ap.add_argument("videos", help="folder of .mov files")
    ap.add_argument("--labels", required=True, help="open_labels.xlsx")
    ap.add_argument("--cache", default="cache", help="per-clip stage cache (makes re-scoring fast)")
    ap.add_argument("--out", default="runs/eval", help="per-clip results and report.md")
    ap.add_argument("--only", help="comma-separated clip numbers, e.g. 26,28")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--score-only", action="store_true", help="re-score stored results, run nothing")
    ap.add_argument("--local", action="store_true", help="debug: run the CV stages on this machine, no VLM")
    ap.add_argument("--detectors", help="comma-separated detector ids to run (remote mode)")
    a = ap.parse_args(argv)

    out = Path(a.out)
    if a.score_only:
        results = EV.load_results(out)
    else:
        labels = load_labels(Path(a.labels))
        files = sorted(Path(a.videos).glob("*.mov"))
        if a.only:
            want = {int(x) for x in a.only.split(",")}
            files = [f for f in files if clip_number(f) in want]
        if a.local:
            cfg = PipelineConfig()
            cfg.events.provisional_threshold = 0.0  # keep every candidate; thresholds are applied when scoring
            results = EV.run_all(files, labels, cfg, Path(a.cache), out, log=lambda m: print(m, flush=True))
        else:
            only = a.detectors.split(",") if a.detectors else None
            results = EV.run_all_remote(files, labels, out, log=lambda m: print(m, flush=True), only_detectors=only)
        if a.only:  # score everything stored so far, not just this subset
            results = EV.load_results(out)
    EV.print_summary(results, a.threshold)
    (out / "report.md").write_text(EV.report_markdown(results, a.threshold), "utf8")
    print(f"report: {out / 'report.md'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "demo":
        dp = argparse.ArgumentParser(prog="kassa demo", description="analyze one video and play the replay")
        dp.add_argument("path")
        dp.add_argument("--out", default="runs")
        dp.add_argument("--window", action="store_true", help="play in a window at real-time speed instead of writing an mp4")
        dp.add_argument("--no-open", action="store_true", help="write the mp4 but do not open it")
        dp.add_argument("--local", action="store_true", help="debug: run CV locally without the VLM")
        return _demo(dp.parse_args(argv[1:]))
    if argv and argv[0] == "eval":
        return _eval(argv[1:])
    if argv and argv[0] in ("export-crops", "review", "relabel", "accept", "crops-status", "train-heads"):
        return _crops(argv[0], argv[1:])
    if argv and argv[0] == "run":
        argv = argv[1:]
    return _main_run(argv)


def _main_run(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="kassa run")
    ap.add_argument("path", help="a .mov file or a folder of them")
    ap.add_argument("--out", default="runs", help="folder for debug JSON per video")
    ap.add_argument("--labels", help="open_labels.xlsx, to print a comparison")
    ap.add_argument("--no-evidence", action="store_true")
    args = ap.parse_args(argv)

    p = Path(args.path)
    files = sorted(p.glob("*.mov")) if p.is_dir() else [p]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    labels = load_labels(Path(args.labels)) if args.labels else {}

    for f in files:
        def log(stage: str, sec: float, info: dict):
            print(f"  {stage:9s} {sec:5.1f}s", file=sys.stderr)

        print(f"{f.name}", file=sys.stderr)
        res = analyze_video(f, progress=log, with_evidence=not args.no_evidence)
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
