"""Score `kassa scan` results against open_labels.xlsx and the team's per-clip rules (rules_kb.json observed_in).

    PYTHONPATH=src .venv/bin/python scripts/score_scan.py runs/scan --labels ../../../open/open_labels.xlsx
"""

import argparse
import json
import statistics as st
from collections import Counter
from pathlib import Path

from kassa_pipeline.cli import _expected_rules, clip_number, load_labels

TIME_TOL = 5.0  # a finding counts as "on time" within this many seconds of the labeled second


def load(d: Path) -> list[dict]:
    out = []
    for p in sorted(d.glob("*.json")):
        try:
            r = json.loads(p.read_text())
        except Exception:
            continue
        if "findings" in r:
            out.append(r)
    return out


def score(results: list[dict], labels: dict, expected: dict, thr: float) -> dict:
    rows, conf = [], Counter()
    rule_tp, rule_fp = Counter(), Counter()
    errs = []
    for r in results:
        n = clip_number(Path(r["name"]))
        lab = labels.get(n)
        if lab is None:
            continue
        viol, sec = lab
        exp = set(expected.get(n, []))
        fs = [f for f in r["findings"] if f["confidence"] >= thr]
        pred = bool(fs)
        conf[("TP" if pred else "FN") if viol else ("FP" if pred else "TN")] += 1
        right = [f for f in fs if f["rule"] in exp]
        wrong = [f for f in fs if f["rule"] not in exp]
        for f in right:
            rule_tp[f["rule"]] += 1
        for f in wrong:
            rule_fp[f["rule"]] += 1
        err = None
        if right and sec is not None:
            err = min((f["t"] - sec for f in right), key=abs)
            errs.append(err)
        rows.append({
            "clip": r["name"], "n": n, "label": (f"violation @{sec:.0f}s" if sec is not None else "violation") if viol else "none",
            "expected": sorted(exp), "found": [f"{f['rule']}@{f['t']:.0f}s({f['confidence']:.2f})" for f in fs],
            "right_rule": bool(right), "on_time": err is not None and abs(err) <= TIME_TOL, "err": err,
            "wrong": len(wrong), "duration": r["duration"], "timings": r["timings"],
        })
    tp, fp, fn, tn = conf["TP"], conf["FP"], conf["FN"], conf["TN"]
    vrows = [x for x in rows if x["label"] != "none" and x["expected"]]
    return {
        "thr": thr, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1), "accuracy": (tp + tn) / max(len(rows), 1),
        "right_rule": sum(x["right_rule"] for x in vrows), "on_time": sum(x["on_time"] for x in vrows), "n_rule_clips": len(vrows),
        "wrong_findings": sum(x["wrong"] for x in rows), "median_abs_err": st.median(abs(e) for e in errs) if errs else None,
        "rule_tp": rule_tp, "rule_fp": rule_fp, "rows": rows,
    }


FMT = "{code}@{t:.0f}s"
CODES_FILE = Path(__file__).resolve().parents[3] / "docs" / "clip_codes.json"


def code_section(results: list[dict], labels: dict, thr: float) -> list[str]:
    """Scores by case class (V01-V21) against docs/clip_codes.json (inferred from the team's notes)."""
    if not CODES_FILE.exists():
        return []
    exp = {int(k): v for k, v in json.loads(CODES_FILE.read_text("utf8"))["clips"].items()}
    tp, fp = Counter(), Counter()
    rows, hit, n, timed = [], 0, 0, 0
    for r in sorted(results, key=lambda r: clip_number(Path(r["name"])) or 0):
        num = clip_number(Path(r["name"]))
        lab = labels.get(num)
        if lab is None:
            continue
        found = [f for f in r["findings"] if f["confidence"] >= thr and f.get("code")]
        e = exp.get(num, {"codes": [], "unsure": []})
        ok_codes = set(e.get("codes", [])) | set(e.get("unsure", []))
        for f in found:
            (tp if f["code"] in ok_codes else fp)[f["code"]] += 1
        if lab[0] and e.get("codes"):
            n += 1
            right = [f for f in found if f["code"] in ok_codes]
            hit += bool(right)
            if right and lab[1] is not None and min(abs(f["t"] - lab[1]) for f in right) <= TIME_TOL:
                timed += 1
        rows.append(f"| X-{num} | {'violation' if lab[0] else 'none'} | {', '.join(e.get('codes', [])) or '-'}"
                    f"{' (' + ', '.join(e['unsure']) + '?)' if e.get('unsure') else ''} | "
                    f"{', '.join(FMT.format(**f) for f in found) or '-'} |")
    L = ["## Case classes (V-codes)", "", f"Expected codes are inferred from the team's notes (`docs/clip_codes.json`).", "",
         f"- Violation clips with an expected code: {n}; right code found: **{hit}**, within ±{TIME_TOL:.0f} s: **{timed}**", "",
         "| code | on a clip where expected | elsewhere |", "|---|---|---|"]
    for c in sorted(set(tp) | set(fp)):
        L.append(f"| {c} | {tp[c]} | {fp[c]} |")
    L += ["", "| clip | label | expected | found |", "|---|---|---|---|"] + rows + [""]
    return L


def report(results: list[dict], labels: dict, expected: dict, thr: float) -> str:
    s = score(results, labels, expected, thr)
    L = [f"# Sliding-window VLM scan: {len(s['rows'])} clips (threshold {thr})", ""]
    L += ["## Verdict", "", "| | predicted violation | predicted none |", "|---|---|---|",
          f"| **label violation** | TP {s['TP']} | FN {s['FN']} |", f"| **label none** | FP {s['FP']} | TN {s['TN']} |", "",
          f"Accuracy {s['accuracy']:.0%}, precision {s['precision']:.0%}, recall {s['recall']:.0%}.", ""]
    L += ["## Right rule", "",
          f"- Violation clips with a team-assigned rule: {s['n_rule_clips']}",
          f"- Right rule found: **{s['right_rule']}**; and within ±{TIME_TOL:.0f} s of the label: **{s['on_time']}**",
          f"- Median |time error| of the right rule: {s['median_abs_err']:.1f} s" if s["median_abs_err"] is not None else "- no timed hits",
          f"- Findings with a rule the team did not assign (wrong or extra): {s['wrong_findings']}", ""]
    L += ["## Threshold sweep", "", "| thr | TP | FP | FN | TN | precision | recall | right rule | on time | wrong findings |", "|---|---|---|---|---|---|---|---|---|---|"]
    for t in (0.4, 0.5, 0.6, 0.7):
        x = score(results, labels, expected, t)
        L.append(f"| {t} | {x['TP']} | {x['FP']} | {x['FN']} | {x['TN']} | {x['precision']:.0%} | {x['recall']:.0%} | {x['right_rule']} | {x['on_time']} | {x['wrong_findings']} |")
    L += ["", "## Per rule (findings at threshold)", "", "| rule | on a clip the team links to it | elsewhere |", "|---|---|---|"]
    for rule in sorted(set(s["rule_tp"]) | set(s["rule_fp"])):
        L.append(f"| {rule} | {s['rule_tp'][rule]} | {s['rule_fp'][rule]} |")
    tot = [x["timings"]["total"] + x["timings"].get("stage1", 0) for x in s["rows"]]
    per = [t / max(x["duration"], 1) for t, x in zip(tot, s["rows"])]
    slow = sorted(zip(tot, s["rows"]), key=lambda z: -z[0])[:4]
    L += ["", "## Runtime on Modal (stage 1 + scan, excluding upload)", "",
          f"- median {st.median(tot):.0f} s per clip, {st.median(per):.2f} s per second of video",
          "- slowest: " + ", ".join(f"{x['clip']} ({x['duration']:.0f}s video, {t:.0f}s)" for t, x in slow),
          f"- over 300 s: {sum(t > 300 for t in tot)}", ""]
    L += code_section(results, labels, thr)
    L += ["## Per clip (rules)", "", "| clip | label | expected rules | found | right rule | time error |", "|---|---|---|---|---|---|"]
    for x in sorted(s["rows"], key=lambda x: x["n"]):
        right = "yes" if x["right_rule"] else ("-" if not x["expected"] else "no")
        err = "" if x["err"] is None else f"{x['err']:+.0f}s"
        L.append(f"| {x['clip']} | {x['label']} | {', '.join(x['expected']) or '-'} | {', '.join(x['found']) or '-'} | {right} | {err} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    a = ap.parse_args()
    md = report(load(Path(a.dir)), load_labels(Path(a.labels)), _expected_rules(), a.threshold)
    (Path(a.dir) / "report.md").write_text(md, "utf8")
    print(md)
