"""Command line: watch Stage 1 live (extract) and analyze clips on Modal (scan), compared with the labels."""

import argparse
import json
import time
import re
import sys
import zipfile
from pathlib import Path

from kassa_pipeline.facts import clip_id



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


def _expected_rules() -> dict[int, list[str]]:
    """clip number -> rules the team's analysis links to it (rules_kb.json observed_in)."""
    from kassa_pipeline.report import rule_base

    out: dict[int, list[str]] = {}
    for r in rule_base()["rules"]:
        for clip in r.get("observed_in", {}):
            m = re.search(r"(\d+)$", clip)
            if m:
                out.setdefault(int(m.group(1)), []).append(r["id"])
    return out


def _scan(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="kassa scan", description="sliding-window VLM scan on Modal")
    ap.add_argument("videos", nargs="+", help=".mov files")
    ap.add_argument("--labels", help="open_labels.xlsx")
    ap.add_argument("--out", default="runs/scan")
    ap.add_argument("--no-vlm-cache", action="store_true")
    ap.add_argument("--fresh-stage1", action="store_true", help="re-run stage 1 even if the volume has it")
    a = ap.parse_args(argv)
    from kassa_pipeline.remote import analyze_many

    files = [Path(v) for v in a.videos]
    labels = load_labels(Path(a.labels)) if a.labels else {}
    expected = _expected_rules()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"scanning {len(files)} clips on Modal...", flush=True)
    for r in analyze_many(files, {"no_vlm_cache": a.no_vlm_cache, "fresh_stage1": a.fresh_stage1, "with_evidence": False}):
        if isinstance(r, Exception):
            print(f"  ERROR {type(r).__name__}: {r}", flush=True)
            continue
        name = r["name"]
        (out / f"{Path(name).stem}.json").write_text(json.dumps(r, ensure_ascii=False, indent=1), "utf8")
        n = clip_number(Path(name))
        lab = labels.get(n)
        truth = ("violation" + (f" @{lab[1]:.0f}s" if lab[1] is not None else "")) if lab and lab[0] else ("none" if lab else "?")
        tm = r["timings"]
        print(f"\n{name}  ({r['duration']:.0f}s video)  label: {truth}  expected rules: {expected.get(n, [])}", flush=True)
        print(f"  verdict {r['verdict']}   stage1 {tm['stage1']}s{' (cached)' if r['stage1']['cached'] else ''}  "
              f"vlm {tm['vlm']}s for {r['vlm']['calls']} calls, unparsed {r['vlm']['unparsed']}", flush=True)
        for f in r["findings"]:
            print(f"  {f['t']:5.1f}s  {f['code']} {f['rule'] or '-':18s} {f['confidence']:.2f}  {f['source']}", flush=True)
        for s in r["sessions"]:
            print(f"  visit {s['t0']:.1f}-{s['t1']:.1f}s: {s['answer']}", flush=True)
    print(f"\nall done in {time.time() - t0:.0f}s; per-clip JSON in {out}")
    return 0


def _run(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="kassa run", description="full analysis of one video on Modal, the case's JSON")
    ap.add_argument("path", help=".mov file")
    ap.add_argument("--out", default="runs/run")
    ap.add_argument("--fresh-stage1", action="store_true", help="re-run stage 1 even if the volume has it")
    a = ap.parse_args(argv)
    import base64

    from kassa_pipeline.remote import analyze

    f = Path(a.path)
    out = Path(a.out) / f.stem
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"{f.name}: uploading and analyzing on Modal...", flush=True)
    r = analyze(f, {"with_evidence": True, "fresh_stage1": a.fresh_stage1})
    api = r["api"]
    for i, v in enumerate(api["violations"], 1):
        for fr in v["evidence_frames"]:
            (out / f"v{i}_{fr['timestamp_sec']:.1f}s.jpg").write_bytes(base64.b64decode(fr["image_base64"]))
    (out / "result.json").write_text(json.dumps(api, ensure_ascii=False, indent=1), "utf8")
    (out / "debug.json").write_text(json.dumps({k: v for k, v in r.items() if k != "api"}, ensure_ascii=False, indent=1), "utf8")
    tm = r["timings"]
    print(f"done in {time.time() - t0:.0f}s (video {r['duration']:.0f}s; stage1 {tm['stage1']}s"
          f"{' cached' if r['stage1']['cached'] else ''}, vlm {tm['vlm']}s)")
    print(f"verdict: {api['verdict']}")
    for i, v in enumerate(api["violations"], 1):
        print(f"\n{i}. {v['timecode_sec']:.1f}s  {v['type']}\n   {v['description']}")
    print(f"\nJSON and evidence frames: {out}")
    return 0


def _stage1(f: Path, cache: Path) -> dict:
    """Stage 1 for one clip: the cached JSON, else from Modal (final event's result), then cached."""
    target = cache / f"{clip_id(f.name)}.json"
    if target.exists():
        return json.loads(target.read_text("utf8"))
    from kassa_pipeline import remote

    final = None
    for ev in remote.extract_stream(f):
        if ev.get("type") == "final":
            final = ev["result"]
    if final is None:
        raise RuntimeError("stage 1 stream ended without a result")
    cache.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(final, ensure_ascii=False), "utf8")
    return final


def _label(argv: list[str]) -> int:
    from kassa_pipeline import label as LB
    from kassa_pipeline import vlm as VM
    from kassa_pipeline.facts import FactTimeline

    ap = argparse.ArgumentParser(prog="kassa label", description="draft fact timelines from Stage 1 and a local VLM")
    ap.add_argument("paths", nargs="+", help=".mov files or folders of them")
    ap.add_argument("--out", default="labels/facts")
    ap.add_argument("--step", type=int, default=1, help="ask the VLM every N seconds")
    ap.add_argument("--overwrite", action="store_true", help="redraft existing drafts (reviewed files are never touched)")
    a = ap.parse_args(argv)

    files = []
    for p in map(Path, a.paths):
        files += sorted(f for f in p.iterdir() if f.suffix.lower() == ".mov") if p.is_dir() else [p]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    def skip(f: Path) -> bool:
        target = out / f"{clip_id(f.name)}.json"
        if not target.exists():
            return False
        if not a.overwrite:
            return True
        try:
            return FactTimeline.load(target).status == "reviewed"
        except ValueError:
            return False  # unreadable draft: safe to redo

    todo = []
    for f in files:
        if skip(f):
            print(f"{f.name}: skipped")
        else:
            todo.append(f)
    if not todo:
        return 0
    failed = 0
    ready = []  # phase 1: Stage 1 for every clip
    for f in todo:
        t0 = time.time()
        try:
            ready.append((f, _stage1(f, out / ".stage1")))
            print(f"{f.name}: stage 1 in {time.time() - t0:.0f} s", flush=True)
        except Exception as exc:  # one bad clip must not stop the batch
            failed += 1
            print(f"{f.name}: ERROR {exc}")
    if not ready:
        return 1 if failed else 0
    model = VM.local_vlm(out / ".vlm_cache" / "answers.json")  # phase 2: one VLM for all drafts
    for f, stage1 in ready:
        t0 = time.time()
        try:
            facts = LB.build_draft(f, stage1, model, step=a.step)
            facts.save(out / f"{facts.clip}.json")
            print(f"{f.name}: {facts.clip} drafted in {time.time() - t0:.0f} s", flush=True)
        except Exception as exc:
            failed += 1
            print(f"{f.name}: ERROR {exc}")
    return 1 if failed else 0


def _facts_eval(argv: list[str]) -> int:
    from kassa_pipeline import facts_eval as FE

    ap = argparse.ArgumentParser(prog="kassa facts-eval", description="score scenario rules on reviewed facts")
    ap.add_argument("--facts", required=True, help="folder of reviewed fact timelines")
    ap.add_argument("--labels", required=True, help="open_labels.xlsx")
    ap.add_argument("--pred", help="folder of per-clip predictions (perception vs truth): fact timelines, debug JSON "
                    "with a \"facts\" key, or Stage 1 JSON such as <labels>/.stage1/ from kassa label")
    ap.add_argument("--out", default="runs/facts_eval.md")
    a = ap.parse_args(argv)
    rules = FE.rules_on_truth(Path(a.facts), load_labels(Path(a.labels)))
    perception = FE.perception_on_dir(Path(a.facts), Path(a.pred)) if a.pred else None
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(FE.report_markdown(rules, perception), "utf8")
    t = rules["totals"]
    print(f"facts eval: tp={t['tp']} fp={t['fp']} fn={t['fn']} tn={t['tn']} report: {out}")
    return 0


def _facts_review(argv: list[str]) -> int:
    from kassa_pipeline.review_server import make_server

    ap = argparse.ArgumentParser(prog="kassa facts-review", description="click-to-correct page for fact timelines")
    ap.add_argument("--facts", default="labels/facts", help="folder of fact timeline JSON files")
    ap.add_argument("--videos", default="videos", help="folder of .mov files")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args(argv)
    try:
        server = make_server(Path(a.facts), Path(a.videos), a.port)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    print(f"http://127.0.0.1:{server.server_address[1]}/  (Ctrl-C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "extract":
        xp = argparse.ArgumentParser(prog="kassa extract", description="stage 1 on Modal, drawn live as frames arrive")
        xp.add_argument("path")
        xp.add_argument("--out", default="runs/extract")
        xp.add_argument("--no-window", action="store_true", help="only write the live recording and the JSON")
        a = xp.parse_args(argv[1:])
        from kassa_pipeline.remote import extract_stream
        from kassa_pipeline.watch import watch

        f = Path(a.path)
        print(f"{f.name}: extract on Modal", flush=True)
        watch(f, extract_stream(f), Path(a.out), window=not a.no_window, log=lambda m: print(m, flush=True))
        return 0
    if argv and argv[0] == "scan":
        return _scan(argv[1:])
    if argv and argv[0] == "run":
        return _run(argv[1:])
    if argv and argv[0] == "label":
        return _label(argv[1:])
    if argv and argv[0] == "facts-review":
        return _facts_review(argv[1:])
    if argv and argv[0] == "facts-eval":
        return _facts_eval(argv[1:])
    print("usage: kassa run <video.mov>                    full analysis, the case's JSON + evidence frames\n"
          "       kassa extract <video.mov> [--no-window]  stage 1, drawn live\n"
          "       kassa scan <video.mov>... [--labels open_labels.xlsx]  several clips, compared with the labels\n"
          "       kassa label <video.mov|dir>... [--out DIR --step N --overwrite]  draft fact timelines (Stage 1 + local VLM)\n"
          "       kassa facts-review [--facts DIR --videos DIR --port N]  click-to-correct fact timelines\n"
          "       kassa facts-eval --facts DIR --labels XLSX [--pred DIR] [--out PATH]  score rules on reviewed facts")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
