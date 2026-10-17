"""Evaluate scenario rules and perception against human-reviewed FactTimelines."""

import json
import re
import statistics
from pathlib import Path

from .facts import STATE_VALUES, UNKNOWN, FactTimeline, clip_id
from .scenarios import run_scenarios


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def _clip_num(clip: str) -> int | None:
    m = re.fullmatch(r"X-(\d+)", clip)
    return int(m.group(1)) if m else None


def rules_on_truth(facts_dir: Path, labels: dict[int, tuple[bool, float | None]]) -> dict:
    """Run the scenario rules on reviewed truth files and score them against the xlsx labels."""
    clips, unlabelled = [], []
    for path in sorted(Path(facts_dir).glob("*.json")):
        ft = FactTimeline.load(path)
        if ft.status != "reviewed":
            continue
        num = _clip_num(ft.clip)
        if num is None or num not in labels:
            unlabelled.append(ft.clip)
            continue
        label_violation, label_sec = labels[num]
        results = run_scenarios(ft)
        violations = [{"rule_id": r.rule_id, "subject": r.subject, "timecode_sec": r.timecode_sec}
                      for r in results if r.status == "violation"]
        clips.append({
            "clip": ft.clip, "label_violation": label_violation, "label_sec": label_sec,
            "predicted_violation": bool(violations), "violations": violations,
            "unknown": [r.rule_id for r in results if r.status == "unknown"],
        })
    tp = sum(c["label_violation"] and c["predicted_violation"] for c in clips)
    fp = sum(not c["label_violation"] and c["predicted_violation"] for c in clips)
    fn = sum(c["label_violation"] and not c["predicted_violation"] for c in clips)
    tn = len(clips) - tp - fp - fn
    errors = []
    for c in clips:
        times = [v["timecode_sec"] for v in c["violations"] if v["timecode_sec"] is not None]
        if c["label_violation"] and c["predicted_violation"] and c["label_sec"] is not None and times:
            errors.append(min(abs(t - c["label_sec"]) for t in times))
    timecode = {
        "n": len(errors),
        "median_error_sec": statistics.median(errors) if errors else None,
        "within_2": _ratio(sum(e <= 2 for e in errors), len(errors)),
        "within_5": _ratio(sum(e <= 5 for e in errors), len(errors)),
    }
    totals = {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": _ratio(tp, tp + fp),
              "recall": _ratio(tp, tp + fn), "accuracy": _ratio(tp + tn, len(clips)),
              "timecode": timecode}
    return {"clips": clips, "unlabelled": unlabelled, "totals": totals}


def _who_kind(who: str | None) -> str | None:
    if who is None:
        return None
    return "cashier" if who == "cashier" else "client"


def _match_events(truth: list, pred: list, tol_sec: float) -> int:
    """Greedy one-to-one matches in time order: each truth event takes the nearest free pred within tol."""
    free = sorted(pred, key=lambda e: e.t)
    matched = 0
    for te in sorted(truth, key=lambda e: e.t):
        best = None
        for pe in free:
            if _who_kind(pe.who) == _who_kind(te.who) and abs(pe.t - te.t) <= tol_sec:
                if best is None or abs(pe.t - te.t) < abs(best.t - te.t):
                    best = pe
        if best is not None:
            free.remove(best)
            matched += 1
    return matched


def perception_vs_truth(pred: FactTimeline, truth: FactTimeline, tol_sec: float = 2.0) -> dict:
    """Per-state agreement and per-event-type precision/recall of a predicted timeline."""
    states = {}
    for name in STATE_VALUES:
        missing = None if name == "people_inside" else UNKNOWN
        n = correct = truth_known = pred_unknown = 0
        for sec in range(truth.n_seconds):
            t, p = truth.state(name, sec), pred.state(name, sec)
            if t == missing:
                continue
            truth_known += 1
            if p == missing:
                pred_unknown += 1
                continue
            n += 1
            correct += t == p
        if truth_known:
            states[name] = {"n": n, "correct": correct, "truth_known": truth_known,
                            "pred_unknown": pred_unknown, "accuracy": _ratio(correct, n),
                            "pred_unknown_rate": _ratio(pred_unknown, truth_known)}
    events = {}
    for etype in pred.observable:
        t_ev = [e for e in truth.events if e.type == etype]
        p_ev = [e for e in pred.events if e.type == etype]
        tp = _match_events(t_ev, p_ev, tol_sec)
        events[etype] = {"tp": tp, "n_pred": len(p_ev), "n_truth": len(t_ev),
                         "precision": _ratio(tp, len(p_ev)), "recall": _ratio(tp, len(t_ev))}
    return {"states": states, "events": events}


def aggregate_perception(parts: list[dict]) -> dict:
    """Sum per-clip perception counts and recompute the rates."""
    states: dict[str, dict] = {}
    events: dict[str, dict] = {}
    for part in parts:
        for name, s in part["states"].items():
            a = states.setdefault(name, {"n": 0, "correct": 0, "truth_known": 0, "pred_unknown": 0})
            for k in a:
                a[k] += s[k]
        for etype, e in part["events"].items():
            a = events.setdefault(etype, {"tp": 0, "n_pred": 0, "n_truth": 0})
            for k in a:
                a[k] += e[k]
    for s in states.values():
        s["accuracy"] = _ratio(s["correct"], s["n"])
        s["pred_unknown_rate"] = _ratio(s["pred_unknown"], s["truth_known"])
    for e in events.values():
        e["precision"] = _ratio(e["tp"], e["n_pred"])
        e["recall"] = _ratio(e["tp"], e["n_truth"])
    return {"states": states, "events": events}


def perception_on_dir(facts_dir: Path, pred_dir: Path) -> dict:
    """Compare debug-JSON facts in pred_dir with every reviewed truth file; collect missing clips."""
    pred_files = {}
    for p in Path(pred_dir).glob("*.json"):
        pred_files.setdefault(clip_id(p.stem), p)
    parts, missing = [], []
    for path in sorted(Path(facts_dir).glob("*.json")):
        truth = FactTimeline.load(path)
        if truth.status != "reviewed":
            continue
        pf = pred_files.get(clip_id(truth.clip)) or pred_files.get(clip_id(path.stem))
        data = json.loads(pf.read_text("utf8")).get("facts") if pf else None
        if not data:
            missing.append(truth.clip)
            continue
        parts.append(perception_vs_truth(FactTimeline.from_dict(data), truth))
    result = aggregate_perception(parts)
    result["missing"] = missing
    result["n_clips"] = len(parts)
    return result


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.0f}%"


def report_markdown(rules: dict, perception: dict | None) -> str:
    t = rules["totals"]
    tc = t["timecode"]
    med = "n/a" if tc["median_error_sec"] is None else f"{tc['median_error_sec']:.1f} s"
    lines = [
        "# Facts evaluation", "", "## Rules on reviewed truth", "",
        "| tp | fp | fn | tn | precision | recall | accuracy |", "|---|---|---|---|---|---|---|",
        f"| {t['tp']} | {t['fp']} | {t['fn']} | {t['tn']} | {_pct(t['precision'])} | "
        f"{_pct(t['recall'])} | {_pct(t['accuracy'])} |", "",
        f"Timecode error over {tc['n']} TP clips: median {med}, within 2 s {_pct(tc['within_2'])}, "
        f"within 5 s {_pct(tc['within_5'])}", "",
        "| clip | label | predicted | violations (rule@time) | unknown rules |", "|---|---|---|---|---|",
    ]
    for c in rules["clips"]:
        label = "violation" if c["label_violation"] else "no violation"
        if c["label_sec"] is not None:
            label += f" @{c['label_sec']:g}s"
        viol = ", ".join(f"{v['rule_id']}@{v['timecode_sec']}" for v in c["violations"]) or "-"
        lines.append(f"| {c['clip']} | {label} | {'violation' if c['predicted_violation'] else 'none'} | "
                     f"{viol} | {', '.join(c['unknown']) or '-'} |")
    if rules["unlabelled"]:
        lines += ["", "Unlabelled (excluded): " + ", ".join(rules["unlabelled"])]
    if perception is not None:
        lines += ["", "## Perception vs truth", "", f"Clips compared: {perception.get('n_clips', 0)}"]
        if perception.get("missing"):
            lines.append("Missing predictions: " + ", ".join(perception["missing"]))
        lines += ["", "| state | seconds compared | accuracy | pred unknown rate |", "|---|---|---|---|"]
        for name, s in perception["states"].items():
            lines.append(f"| {name} | {s['n']} | {_pct(s['accuracy'])} | {_pct(s['pred_unknown_rate'])} |")
        lines += ["", "| event | truth | predicted | tp | precision | recall |", "|---|---|---|---|---|---|"]
        for etype, e in perception["events"].items():
            lines.append(f"| {etype} | {e['n_truth']} | {e['n_pred']} | {e['tp']} | "
                         f"{_pct(e['precision'])} | {_pct(e['recall'])} |")
    return "\n".join(lines) + "\n"
