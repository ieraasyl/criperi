"""Trained heads: logistic regression on SigLIP crop embeddings, used by Stage 1 in place of zero-shot prompts.

Trained from human-reviewed FactTimelines: every labelled second gives one crop of the safe (safe_door) and of
the laptop screen (screen), cut with the Stage 1 layout of that clip. A head only fits the encoder it was
trained with, so the file records the encoder id and Stage 1 refuses a mismatch.

    python -m kassa_pipeline.heads --facts labels/facts --videos open/videos --out heads.npz
(Stage 1 per clip is read from <facts>/.stage1/<clip>.json, where `kassa label` caches it.)
"""

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np

from kassa_pipeline import geometry as g
from kassa_pipeline import objects as O
from kassa_pipeline.facts import FactTimeline, clip_id

# task -> (FactTimeline state, label -> Stage 1 class). Labels not listed (unknown, partial) are skipped.
TASKS = {
    "safe": ("safe_door", {"open": "open", "closed": "closed"}),
    "screen": ("screen", {"work": "work_screen", "locked": "locked", "off": "off"}),
}
SAFE_PAD = 0.5  # as extract._states crops the safe


class Heads:
    def __init__(self, encoder: str, heads: dict[str, tuple[np.ndarray, np.ndarray, list[str]]]):
        self.encoder, self.heads = encoder, heads

    def has(self, task: str) -> bool:
        return task in self.heads

    def predict(self, task: str, emb: np.ndarray) -> tuple[list[str], np.ndarray]:
        w, b, classes = self.heads[task]
        z = np.asarray(emb, np.float32) @ w + b
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        return classes, p / p.sum(axis=1, keepdims=True)

    def save(self, path: str | Path) -> None:
        arrays = {"encoder": np.array(self.encoder)}
        for task, (w, b, classes) in self.heads.items():
            arrays[f"{task}__w"], arrays[f"{task}__b"], arrays[f"{task}__classes"] = w, b, np.array(classes)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            np.savez(fh, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "Heads":
        z = np.load(path, allow_pickle=False)
        tasks = {k.split("__")[0] for k in z.files if "__" in k}
        return cls(str(z["encoder"]), {t: (z[f"{t}__w"], z[f"{t}__b"], [str(c) for c in z[f"{t}__classes"]]) for t in tasks})


def fit(emb: np.ndarray, labels: list[str], l2: float = 1e-3, steps: int = 500):
    """Multinomial logistic regression with class-balanced weights (the safe is closed most of the time)."""
    import torch

    classes = sorted(set(labels))
    y = torch.tensor([classes.index(c) for c in labels])
    x = torch.tensor(np.asarray(emb, np.float32))
    counts = torch.bincount(y, minlength=len(classes)).float()
    weight = counts.sum() / (len(classes) * counts.clamp(min=1))
    w = torch.zeros((x.shape[1], len(classes)), requires_grad=True)
    b = torch.zeros(len(classes), requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(x @ w + b, y, weight=weight) + l2 * (w * w).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach().numpy(), b.detach().numpy(), classes


def _layout_at(layouts: list[dict], t: float) -> dict:
    cur = layouts[0]
    for lay in sorted(layouts, key=lambda x: x["start"]):
        if t + 1e-6 >= lay["start"]:
            cur = lay
    return cur


def task_crop(task: str, img: np.ndarray, lay: dict) -> np.ndarray | None:
    """The crop Stage 1 classifies for `task`, from the lighting-normalized frame."""
    h, w = img.shape[:2]
    if task == "safe":
        box = lay.get("safe")
        return None if not box else g.crop(img, g.expand(tuple(box), SAFE_PAD, w, h), min_size=12)
    box = lay.get("laptop")
    return None if not box else g.crop(img, O.screen_area(tuple(box)), min_size=12)


def samples(ft: FactTimeline, stage1: dict, read_frame: Callable[[float], np.ndarray]) -> dict[str, list]:
    """task -> [(crop, label)] for every second a reviewed timeline labels. Frames at the middle of the second."""
    from kassa_pipeline.video import normalize_lighting

    out: dict[str, list] = {t: [] for t in TASKS}
    for sec in range(ft.n_seconds):
        wanted = {task: m.get(ft.state(state, sec)) for task, (state, m) in TASKS.items()}
        if not any(wanted.values()):
            continue
        t = min(sec + 0.5, max(ft.duration - 0.05, 0.0))
        img = normalize_lighting(read_frame(t))
        lay = _layout_at(stage1["layouts"], t)
        for task, label in wanted.items():
            c = task_crop(task, img, lay) if label else None
            if c is not None:
                out[task].append((c, label))
    return out


def _accuracy(heads: Heads, task: str, emb: np.ndarray, labels: list[str]) -> dict:
    classes, p = heads.predict(task, emb)
    pred = [classes[i] for i in p.argmax(axis=1)]
    per = {c: round(float(np.mean([a == c for a, b in zip(pred, labels) if b == c])), 3) for c in sorted(set(labels))}
    return {"n": len(labels), "accuracy": round(float(np.mean([a == b for a, b in zip(pred, labels)])), 3), "recall": per}


def train(data: dict[str, list[tuple[str, np.ndarray, str]]], encoder: str, val_every: int = 4) -> tuple[Heads, dict]:
    """data: task -> [(clip, embedding, label)]. Held-out report on every `val_every`-th clip, then fit on all."""
    report, final = {}, {}
    for task, rows in data.items():
        if len({r[2] for r in rows}) < 2:
            report[task] = {"skipped": f"needs two classes, has {sorted({r[2] for r in rows})}"}
            continue
        clips = sorted({r[0] for r in rows})
        val = set(clips[::val_every]) if len(clips) >= 2 else set()
        tr = [r for r in rows if r[0] not in val]
        va = [r for r in rows if r[0] in val]
        rep: dict = {"clips": len(clips), "val_clips": sorted(val), "train": len(tr)}
        if va and len({r[2] for r in tr}) >= 2:
            held = Heads(encoder, {task: fit(np.stack([r[1] for r in tr]), [r[2] for r in tr])})
            rep["val"] = _accuracy(held, task, np.stack([r[1] for r in va]), [r[2] for r in va])
        final[task] = fit(np.stack([r[1] for r in rows]), [r[2] for r in rows])
        report[task] = rep
    return Heads(encoder, final), report


def main(argv: list[str] | None = None) -> None:
    from kassa_pipeline import models as M
    from kassa_pipeline import video as V
    from kassa_pipeline.config import ExtractConfig

    ap = argparse.ArgumentParser(prog="python -m kassa_pipeline.heads", description="train SigLIP heads on reviewed facts")
    ap.add_argument("--facts", required=True, help="reviewed FactTimelines; Stage 1 in <facts>/.stage1/<clip>.json")
    ap.add_argument("--videos", required=True)
    ap.add_argument("--out", required=True, help="heads file (.npz) for ExtractConfig.heads_path")
    ap.add_argument("--encoder", default=ExtractConfig().crop_encoder)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)

    facts = Path(a.facts)
    videos = {clip_id(p.name): p for p in Path(a.videos).iterdir() if p.is_file()}
    enc = M.crop_encoder(a.encoder, M.pick_device(a.device))
    data: dict[str, list] = {t: [] for t in TASKS}
    for path in sorted(facts.glob("*.json")):
        ft = FactTimeline.load(path)
        s1_path = facts / ".stage1" / f"{ft.clip}.json"
        if ft.status != "reviewed" or ft.clip not in videos or not s1_path.exists():
            continue
        video = videos[ft.clip]
        for task, rows in samples(ft, json.loads(s1_path.read_text()), lambda t: V.read_frame_at(video, t)).items():
            if rows:
                emb = enc.embed([c for c, _ in rows])
                data[task] += [(ft.clip, e, lb) for e, (_, lb) in zip(emb, rows)]
        print(ft.clip, {t: len(r) for t, r in data.items()}, flush=True)
    heads, report = train(data, a.encoder)
    if not heads.heads:
        raise SystemExit(f"nothing to train: {json.dumps(report)}")
    heads.save(a.out)
    print(json.dumps(report, indent=1))
    print(f"heads {sorted(heads.heads)} -> {a.out}")


if __name__ == "__main__":
    main()
