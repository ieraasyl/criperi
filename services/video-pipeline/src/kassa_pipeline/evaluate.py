"""Evaluation harness: run the pipeline over labeled clips and score it.

Expensive stages are cached per clip (see pipeline._cached), so after the first pass a threshold or
rule change is re-scored in seconds. Scoring works from stored candidates, which is why the
threshold sweep needs no re-run.
"""

import json
import re
import statistics
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

from kassa_pipeline.cli import clip_number, load_labels
from kassa_pipeline.config import PipelineConfig
from kassa_pipeline.pipeline import analyze_video

REAL_TIME_BUDGET_SEC = 300.0  # the case gives 5 minutes per video
STAGES = ["decode", "layout", "people", "objects", "timeline", "events", "report"]
COLD_STAGES = ["decode", "layout", "people", "objects"]  # the ones that cost real time


@dataclass
class ClipResult:
    name: str
    n: int | None
    label_violation: bool | None
    label_sec: float | None
    duration: float = 0.0
    candidates: list[dict] = field(default_factory=list)  # {type, timecode, score}
    cold_timings: dict[str, float] = field(default_factory=dict)
    error: str | None = None

    def predicted(self, thr: float) -> list[dict]:
        return [c for c in self.candidates if c["score"] >= thr]


# ------------------------------------------------------------------ running


def _merge_cold(cache_dir: Path, clip_id: str, stage: str, sec: float):
    p = cache_dir / clip_id / "cold_timings.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(p.read_text()) if p.exists() else {}
    data[stage] = sec
    p.write_text(json.dumps(data))


def run_clip(path: Path, cfg: PipelineConfig, cache_dir: Path, label) -> ClipResult:
    clip_id = re.sub(r"\W+", "_", path.stem)
    res = ClipResult(path.name, clip_number(path), label[0] if label else None, label[1] if label else None)

    def on_stage(stage: str, sec: float, info: dict):
        # a stage served from cache took ~0 s: keep its original cold time instead
        if not info.get("cached"):
            _merge_cold(cache_dir, clip_id, stage, sec)

    try:
        out = analyze_video(path, cfg, progress=on_stage, with_evidence=False, cache_dir=cache_dir)
        res.duration = out.video.duration
        res.candidates = [
            {"type": c.type, "timecode": round(c.timecode, 2), "score": round(c.score, 3)}
            for c in out.candidates
        ]
    except Exception as e:  # keep going: one bad clip must not stop the batch
        res.error = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    cold = cache_dir / clip_id / "cold_timings.json"
    if cold.exists():
        res.cold_timings = json.loads(cold.read_text())
    return res


def run_all(
    videos: list[Path], labels: dict, cfg: PipelineConfig, cache_dir: Path, out_dir: Path, log=print
) -> list[ClipResult]:
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for i, v in enumerate(videos, 1):
        t = time.time()
        r = run_clip(v, cfg, cache_dir, labels.get(clip_number(v)))
        (out_dir / f"{Path(v).stem}.json").write_text(json.dumps(asdict(r), ensure_ascii=False, indent=1), "utf8")
        results.append(r)
        log(f"[{i}/{len(videos)}] {v.name}  {time.time() - t:.0f}s  "
            f"{'ERROR ' + r.error if r.error else [(c['type'], c['timecode'], c['score']) for c in r.candidates]}")
    return results


def run_all_remote(videos: list[Path], labels: dict, out_dir: Path, log=print, only_detectors: list[str] | None = None) -> list[ClipResult]:
    """Run every clip on Modal (CV + Qwen3.5-9B), in parallel. All candidates are kept (threshold 0)
    so the same run can be re-scored at any threshold."""
    from kassa_pipeline.remote import analyze_many

    out_dir.mkdir(parents=True, exist_ok=True)
    by_name = {v.name: v for v in videos}
    opts = {"with_evidence": False, "use_vlm": True, "cfg": {"events.provisional_threshold": 0.0}}
    if only_detectors:
        opts["only"] = only_detectors
    results: list[ClipResult] = []
    t0, done = time.time(), 0
    for r in analyze_many(videos, opts):
        done += 1
        if isinstance(r, Exception):
            log(f"[{done}/{len(videos)}] ERROR {type(r).__name__}: {r}")
            continue
        name = r["name"]
        n = clip_number(by_name[name])
        lab = labels.get(n)
        dbg = r["debug"]
        res = ClipResult(
            name, n, lab[0] if lab else None, lab[1] if lab else None, dbg["duration"],
            [{"type": c["type"], "timecode": round(c["timecode"], 2), "score": c["score"], "cv_score": c["cv_score"],
              "source": c["source"], "rule": c["rule"], "obs": (c["vlm"][0]["obs"] if c["vlm"] else "")} for c in dbg["candidates"]],
            {**dbg["timings"], "wall_seconds": r["wall_seconds"]},
            "; ".join(f"{k}: {v}" for k, v in dbg["detectors"].get("errors", {}).items()) or None,
        )
        (out_dir / f"{Path(name).stem}.json").write_text(json.dumps(asdict(res), ensure_ascii=False, indent=1), "utf8")
        results.append(res)
        shown = [(c["type"], c["timecode"], c["score"], c["source"]) for c in res.candidates if c["score"] >= 0.3]
        log(f"[{done}/{len(videos)}] {name}  {r['wall_seconds']:.0f}s  elapsed {time.time() - t0:.0f}s  {shown}" + (f"  DETECTOR ERRORS: {res.error}" if res.error else ""))
    return sorted(results, key=lambda r: (r.n is None, r.n or 0))


def load_results(out_dir: Path) -> list[ClipResult]:
    res = []
    for p in sorted(out_dir.glob("*.json")):
        d = json.loads(p.read_text("utf8"))
        res.append(ClipResult(**d))
    return sorted(res, key=lambda r: (r.n is None, r.n or 0))


# ------------------------------------------------------------------ scoring


def summarize(results: list[ClipResult], thr: float) -> dict:
    """Verdict confusion and timecode error at threshold `thr`. Failed clips are excluded."""
    ok = [r for r in results if r.error is None and r.label_violation is not None]
    tp = fp = fn = tn = 0
    fp_clips, fn_clips = [], []
    first_err, near_err = [], []
    for r in ok:
        pred = r.predicted(thr)
        if r.label_violation and pred:
            tp += 1
            if r.label_sec is not None:
                first_err.append(min(c["timecode"] for c in pred) - r.label_sec)
                near_err.append(min(abs(c["timecode"] - r.label_sec) for c in pred))
        elif r.label_violation:
            fn += 1
            fn_clips.append(r.name)
        elif pred:
            fp += 1
            fp_clips.append(r.name)
        else:
            tn += 1
    n = len(ok)

    def within(errs, s):
        return sum(abs(e) <= s for e in errs) / len(errs) if errs else None

    return {
        "threshold": thr,
        "clips": n,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": (tp + tn) / n if n else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "fp_clips": fp_clips,
        "fn_clips": fn_clips,
        "timecode": {
            "n": len(first_err),
            "median_abs_first": statistics.median(abs(e) for e in first_err) if first_err else None,
            "median_signed_first": statistics.median(first_err) if first_err else None,
            "median_abs_nearest": statistics.median(near_err) if near_err else None,
            "within_2s_first": within(first_err, 2),
            "within_3s_first": within(first_err, 3),
            "within_5s_first": within(first_err, 5),
            "within_3s_nearest": within(near_err, 3),
        },
    }


def sweep(results: list[ClipResult], thresholds=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)) -> list[dict]:
    return [summarize(results, t) for t in thresholds]


def per_type(results: list[ClipResult], thr: float) -> dict[str, dict[str, int]]:
    """For each violation type: number of violation clips and control clips where it fired."""
    out: dict[str, dict[str, int]] = {}
    for r in results:
        if r.error or r.label_violation is None:
            continue
        for t in {c["type"] for c in r.predicted(thr)}:
            d = out.setdefault(t, {"on_violation_clips": 0, "on_control_clips": 0})
            d["on_violation_clips" if r.label_violation else "on_control_clips"] += 1
    return dict(sorted(out.items()))


def runtime(results: list[ClipResult]) -> dict:
    rows = []
    for r in results:
        if r.error or not r.duration:
            continue
        total = sum(r.cold_timings.get(s, 0.0) for s in STAGES)
        rows.append((r.name, r.duration, total))
    stage_med = {
        s: statistics.median([r.cold_timings[s] for r in results if s in r.cold_timings] or [0.0])
        for s in STAGES
    }
    ratios = [t / d for _, d, t in rows if d]
    return {
        "median_seconds_per_video_second": statistics.median(ratios) if ratios else None,
        "slowest": sorted(rows, key=lambda x: -x[2])[:3],
        "over_budget": [n for n, _, t in rows if t > REAL_TIME_BUDGET_SEC],
        "median_stage_seconds": stage_med,
    }


# ------------------------------------------------------------------ report


def _pct(x):
    return "n/a" if x is None else f"{100 * x:.0f}%"


def _f(x, nd=1):
    return "n/a" if x is None else f"{x:.{nd}f}"


def report_markdown(results: list[ClipResult], thr: float) -> str:
    s = summarize(results, thr)
    tc = s["timecode"]
    failed = [r for r in results if r.error]
    lines = [
        f"# Evaluation report (threshold {thr})",
        "",
        f"Clips scored: {s['clips']} of {len(results)}" + (f"; **failed: {', '.join(r.name for r in failed)}**" if failed else ""),
        "",
        "## Verdict",
        "",
        "| | predicted violation | predicted no violation |",
        "|---|---|---|",
        f"| **label violation** | TP {s['tp']} | FN {s['fn']} (misses) |",
        f"| **label no violation** | FP {s['fp']} (false alarms) | TN {s['tn']} |",
        "",
        f"Accuracy {_pct(s['accuracy'])}, precision {_pct(s['precision'])}, recall {_pct(s['recall'])}.",
        f"- Missed violations: {', '.join(s['fn_clips']) or 'none'}",
        f"- False alarms: {', '.join(s['fp_clips']) or 'none'}",
        "",
        "## Timecode (detected violations with a labeled second)",
        "",
        f"- n = {tc['n']}; median |error| of the first prediction: {_f(tc['median_abs_first'])} s "
        f"(signed median {_f(tc['median_signed_first'])} s; negative = we are early)",
        f"- within ±2 s: {_pct(tc['within_2s_first'])}, ±3 s: {_pct(tc['within_3s_first'])}, ±5 s: {_pct(tc['within_5s_first'])}",
        f"- closest prediction: median |error| {_f(tc['median_abs_nearest'])} s, within ±3 s: {_pct(tc['within_3s_nearest'])}",
        "",
        "## Threshold sweep",
        "",
        "| threshold | TP | FP | FN | TN | precision | recall |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in sweep(results):
        lines.append(f"| {r['threshold']} | {r['tp']} | {r['fp']} | {r['fn']} | {r['tn']} | {_pct(r['precision'])} | {_pct(r['recall'])} |")
    lines += ["", "## Types fired", "", "| type | on violation clips | on control clips |", "|---|---|---|"]
    for t, d in per_type(results, thr).items():
        lines.append(f"| {t} | {d['on_violation_clips']} | {d['on_control_clips']} |")
    rt = runtime(results)
    lines += [
        "",
        "## Runtime (cold, this machine)",
        "",
        f"- median {_f(rt['median_seconds_per_video_second'], 2)} s of compute per second of video",
        "- median seconds per stage: " + ", ".join(f"{k} {v:.1f}" for k, v in rt["median_stage_seconds"].items()),
        "- slowest: " + ", ".join(f"{n} ({d:.0f}s video, {t:.0f}s)" for n, d, t in rt["slowest"]),
        f"- over the {REAL_TIME_BUDGET_SEC:.0f}s budget: {', '.join(rt['over_budget']) or 'none'}",
        "",
        "## Per clip",
        "",
        "| clip | label | predicted (type @ time, score) |",
        "|---|---|---|",
    ]
    for r in results:
        lab = "-" if r.label_violation is None else ("violation" + (f" @{r.label_sec:.0f}s" if r.label_sec is not None else "") if r.label_violation else "none")
        pred = "ERROR " + r.error if r.error else ", ".join(f"{c['type']} @{c['timecode']:.0f}s ({c['score']:.2f})" for c in r.candidates) or "-"
        lines.append(f"| {r.name} | {lab} | {pred} |")
    return "\n".join(lines) + "\n"


def print_summary(results: list[ClipResult], thr: float, log=print):
    s = summarize(results, thr)
    log(f"\nVerdict @ {thr}: TP {s['tp']}  FP {s['fp']}  FN {s['fn']}  TN {s['tn']}  "
        f"accuracy {_pct(s['accuracy'])}  precision {_pct(s['precision'])}  recall {_pct(s['recall'])}")
    log(f"  misses: {s['fn_clips'] or 'none'}")
    log(f"  false alarms: {s['fp_clips'] or 'none'}")
    tc = s["timecode"]
    log(f"  timecode (n={tc['n']}): median |err| {_f(tc['median_abs_first'])}s, within 3s {_pct(tc['within_3s_first'])}")
