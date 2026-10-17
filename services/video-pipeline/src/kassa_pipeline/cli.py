"""Command line: watch Stage 1 live (extract) and analyze clips on Modal (scan), compared with the labels."""

import argparse
import json
import time
import re
import sys
import zipfile
from pathlib import Path



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
            print(f"  {f['t']:5.1f}s  {f['rule']:18s} {f['confidence']:.2f}  {f['source']}", flush=True)
        for s in r["sessions"]:
            print(f"  visit {s['t0']:.1f}-{s['t1']:.1f}s: {s['answer']}", flush=True)
    print(f"\nall done in {time.time() - t0:.0f}s; per-clip JSON in {out}")
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
    print("usage: kassa extract <video.mov> [--no-window]\n       kassa scan <video.mov>... [--labels open_labels.xlsx]")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
