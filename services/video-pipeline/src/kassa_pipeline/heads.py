"""Labeled-crop workflow for the small state classifiers (objects, laptop screen, safe door).

    kassa export-crops <videos>      crops + manifest.csv, pre-sorted into the zero-shot guess
    kassa review --task T --cls C    contact sheets of one class folder, to eyeball and fix
    kassa relabel --task T --ids .. --to C    move wrongly sorted crops
    kassa accept --task T --cls C    mark every remaining crop in that folder as reviewed
    kassa train-heads                fit heads on REVIEWED crops only -> heads/<task>.joblib

Crops use exactly the same padding as the pipeline's inference, so training matches deployment.
Unreviewed crops carry only a zero-shot guess and are never used for training.
"""

import csv
import pickle
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from kassa_pipeline import geometry as g
from kassa_pipeline import models
from kassa_pipeline import objects as O
from kassa_pipeline.config import PipelineConfig

TASKS = {
    "object": O.OBJECT_PROMPTS,
    "screen": O.SCREEN_PROMPTS,
    "safe": O.SAFE_PROMPTS,
}
PAD = {"screen": 0.3, "safe": 0.5, "object": 0.4}  # must match objects.py
FIELDS = ["id", "task", "clip", "sec", "file", "x1", "y1", "x2", "y2", "guess", "guess_prob", "gdino_label", "label"]
DEDUP_COS, DEDUP_IOU, DEDUP_SEC = 0.96, 0.5, 6


def _read(manifest: Path) -> list[dict]:
    if not manifest.exists():
        return []
    with manifest.open(newline="", encoding="utf8") as f:
        return list(csv.DictReader(f))


def _write(manifest: Path, rows: list[dict]):
    with manifest.open("w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def _cached_layout(cache: Path | None, clip_id: str):
    if cache is None:
        return None
    files = sorted((cache / clip_id).glob("layout-*.pkl"), key=lambda p: p.stat().st_mtime)
    return pickle.loads(files[-1].read_bytes()) if files else None


def export_crops(
    videos: list[Path], out: Path, cfg: PipelineConfig, cache: Path | None = None, every_sec: int = 1, log=print
) -> int:
    """Append crops for clips not yet in the manifest. Returns the number of new crops."""
    from kassa_pipeline import layout as L
    from kassa_pipeline import video as V

    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.csv"
    rows = _read(manifest)
    done_clips = {r["clip"] for r in rows}
    next_id = max((int(r["id"]) for r in rows), default=0) + 1

    dev = models.pick_device(cfg.models.device)
    ov = models.open_vocab_detector(cfg.models.open_vocab_detector, dev)
    enc = models.crop_encoder(cfg.models.crop_encoder, dev)
    added = 0
    for v in videos:
        clip = re.sub(r"\W+", "_", v.stem)
        if clip in done_clips:
            continue
        vid = V.load_video(v, 1.0, cfg.sampling.scene_cut_threshold)
        tl = _cached_layout(cache, clip) or L.detect_layout_timeline(
            vid.frames, vid.scene_cuts, ov, enc, cfg.layout, cfg.sampling.layout_frames
        )
        items: list[tuple[str, int, g.Box, str, np.ndarray]] = []  # task, sec, box, gdino label, crop
        for f in vid.frames[::every_sec]:
            lay = tl.at(f.t)
            h, w = f.norm.shape[:2]
            if lay.laptop:
                c = g.crop(f.norm, g.expand(lay.laptop, PAD["screen"], w, h), min_size=12)
                if c is not None:
                    items.append(("screen", int(f.t), lay.laptop, "", c))
            if lay.safe:
                c = g.crop(f.norm, g.expand(lay.safe, PAD["safe"], w, h), min_size=12)
                if c is not None:
                    items.append(("safe", int(f.t), lay.safe, "", c))
            if lay.workstation:
                region = g.expand(lay.workstation, cfg.objects.desk_crop_pad, w, h)
                rc = g.crop(f.norm, region)
                if rc is not None:
                    dets = ov([rc], cfg.objects.prompt, cfg.objects.box_threshold, cfg.objects.text_threshold)[0]
                    equipment = [b for b in (lay.laptop, lay.counter, lay.telephone) if b]
                    for d in dets:
                        box = (d.box[0] + region[0], d.box[1] + region[1], d.box[2] + region[0], d.box[3] + region[1])
                        if any(g.iou(box, e) > 0.5 for e in equipment) or g.area(box) > 0.5 * g.area(lay.workstation):
                            continue
                        c = g.crop(f.norm, g.expand(box, PAD["object"], w, h), min_size=12)
                        if c is not None:
                            items.append(("object", int(f.t), box, d.label, c))

        # zero-shot guess + near-duplicate removal, per task
        kept = 0
        for task, prompts in TASKS.items():
            sel = [it for it in items if it[0] == task]
            if not sel:
                continue
            emb = enc.embed([it[4] for it in sel])
            probs = enc.zero_shot(emb, list(prompts.values()))
            keys = list(prompts)
            last: list[tuple[int, g.Box, np.ndarray]] = []
            for it, e, p in zip(sel, emb, probs):
                _, sec, box, label, crop = it
                if any(
                    sec - s2 <= DEDUP_SEC and g.iou(box, b2) > DEDUP_IOU and float(e @ e2) > DEDUP_COS
                    for s2, b2, e2 in last
                ):
                    continue
                last.append((sec, box, e))
                guess = keys[int(np.argmax(p))]
                d = out / task / guess
                d.mkdir(parents=True, exist_ok=True)
                rel = f"{task}/{guess}/{next_id:06d}.jpg"
                cv2.imwrite(str(out / rel), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
                rows.append({
                    "id": next_id, "task": task, "clip": clip, "sec": sec, "file": rel,
                    "x1": int(box[0]), "y1": int(box[1]), "x2": int(box[2]), "y2": int(box[3]),
                    "guess": guess, "guess_prob": f"{float(np.max(p)):.3f}", "gdino_label": label, "label": "",
                })
                next_id += 1
                kept += 1
        added += kept
        _write(manifest, rows)  # after every clip: the export can be interrupted and resumed
        log(f"{v.name}: {kept} crops")
    return added


# ---------------------------------------------------------------- review and relabel

_FONT = None


def _font(size=13):
    global _FONT
    for p in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def review_sheets(crops: Path, task: str, cls: str | None, out: Path, per_sheet: int = 30, only_unreviewed=True) -> list[Path]:
    """Contact sheets of the crops currently sorted into task/cls, each tile labeled with its id."""
    rows = [r for r in _read(crops / "manifest.csv") if r["task"] == task and r["file"].split("/")[1] == (cls or r["file"].split("/")[1])]
    if only_unreviewed:
        rows = [r for r in rows if not r["label"]]
    out.mkdir(parents=True, exist_ok=True)
    tile_w, tile_h, cols = 200, 150, 6
    font = _font()
    paths = []
    for s in range(0, len(rows), per_sheet):
        chunk = rows[s : s + per_sheet]
        n_rows = (len(chunk) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * tile_w, n_rows * tile_h), (20, 20, 24))
        for i, r in enumerate(chunk):
            im = Image.open(crops / r["file"]).convert("RGB")
            scale = min((tile_w - 4) / im.width, (tile_h - 4) / im.height, 8.0)  # enlarge tiny crops too
            im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.LANCZOS)
            tile = Image.new("RGB", (tile_w, tile_h), (40, 40, 48))
            tile.paste(im, ((tile_w - im.width) // 2, (tile_h - im.height) // 2))
            d = ImageDraw.Draw(tile)
            d.rectangle([0, 0, 78, 16], fill=(0, 0, 0))
            d.text((3, 1), f"#{r['id']} {r['clip'].split('_')[-1]}@{r['sec']}", font=font, fill=(255, 255, 0))
            sheet.paste(tile, ((i % cols) * tile_w, (i // cols) * tile_h))
        p = out / f"{task}_{cls or 'all'}_{s // per_sheet + 1:02d}.png"
        sheet.save(p)
        paths.append(p)
    return paths


def relabel(crops: Path, task: str, ids: list[int], to: str) -> int:
    """Move crops to another class folder and mark them reviewed with that class."""
    if to not in TASKS[task]:
        raise ValueError(f"{to!r} is not a {task} class: {list(TASKS[task])}")
    manifest = crops / "manifest.csv"
    rows = _read(manifest)
    n = 0
    for r in rows:
        if r["task"] == task and int(r["id"]) in ids:
            src = crops / r["file"]
            new_rel = f"{task}/{to}/{Path(r['file']).name}"
            (crops / task / to).mkdir(parents=True, exist_ok=True)
            if src.exists():
                src.rename(crops / new_rel)
            r["file"], r["label"] = new_rel, to
            n += 1
    _write(manifest, rows)
    return n


def accept(crops: Path, task: str, cls: str) -> int:
    """After fixing a class folder: mark all its still-unreviewed crops as correct."""
    manifest = crops / "manifest.csv"
    rows = _read(manifest)
    n = 0
    for r in rows:
        if r["task"] == task and r["file"].split("/")[1] == cls and not r["label"]:
            r["label"] = cls
            n += 1
    _write(manifest, rows)
    return n


def status(crops: Path) -> dict:
    rows = _read(crops / "manifest.csv")
    out: dict = {}
    for task in TASKS:
        t = [r for r in rows if r["task"] == task]
        lab = [r for r in t if r["label"]]
        out[task] = {
            "crops": len(t),
            "reviewed": len(lab),
            "by_class": {c: sum(r["label"] == c for r in lab) for c in TASKS[task] if any(r["label"] == c for r in lab)},
            "guessed_unreviewed": {c: sum(1 for r in t if not r["label"] and r["guess"] == c) for c in TASKS[task] if any(not r["label"] and r["guess"] == c for r in t)},
        }
    return out


# ---------------------------------------------------------------- training


def train_heads(crops: Path, heads_dir: Path, cfg: PipelineConfig, min_per_class: int = 5) -> dict[str, dict]:
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    from sklearn.model_selection import GroupKFold, cross_val_predict

    dev = models.pick_device(cfg.models.device)
    enc = models.crop_encoder(cfg.models.crop_encoder, dev)
    heads_dir.mkdir(parents=True, exist_ok=True)
    rows = [r for r in _read(crops / "manifest.csv") if r["label"]]
    report = {}
    for task in TASKS:
        t = [r for r in rows if r["task"] == task]
        counts = {c: sum(r["label"] == c for r in t) for c in TASKS[task]}
        usable = {c for c, n in counts.items() if n >= min_per_class}
        t = [r for r in t if r["label"] in usable]
        if len(usable) < 2:
            report[task] = {"skipped": "need >= 2 classes with >= %d reviewed crops" % min_per_class, "counts": counts}
            continue
        imgs = [cv2.cvtColor(cv2.imread(str(crops / r["file"])), cv2.COLOR_BGR2RGB) for r in t]
        X = enc.embed(imgs)
        y = np.array([r["label"] for r in t])
        groups = np.array([r["clip"] for r in t])  # never split one clip across folds
        clf = LogisticRegression(max_iter=3000, C=2.0, class_weight="balanced")
        info = {"n": len(y), "counts": {c: int(n) for c, n in counts.items() if n}}
        if len(set(groups)) >= 3:
            pred = cross_val_predict(clf, X, y, groups=groups, cv=GroupKFold(n_splits=min(5, len(set(groups)))))
            info["clip_cv_accuracy"] = round(float((pred == y).mean()), 3)
            info["per_class"] = classification_report(y, pred, output_dict=True, zero_division=0)
            info["per_class"] = {k: {m: round(v, 2) for m, v in d.items()} for k, d in info["per_class"].items() if isinstance(d, dict)}
        clf.fit(X, y)
        joblib.dump(clf, heads_dir / f"{task}.joblib")
        report[task] = info
    return report
