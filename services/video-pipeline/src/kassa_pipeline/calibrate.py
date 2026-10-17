"""Offline calibration: cache the expensive stages once per clip, then sweep thresholds in seconds.

Not swept: `text_threshold` (part of the DINO cache key) and layout-stage thresholds (they change the
desk crop, which invalidates the cached objects stage).
"""

import dataclasses
import hashlib
import itertools
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kassa_pipeline import events as E
from kassa_pipeline import layout as L
from kassa_pipeline import models, objects as O, people as P, timeline as T
from kassa_pipeline import video as V
from kassa_pipeline.config import PipelineConfig
from kassa_pipeline.detcache import CachedDetector
from kassa_pipeline.layout import LayoutTimeline
from kassa_pipeline.objects import RawObjects
from kassa_pipeline.people import PeopleResult

FLOOR = 0.15
Pred = tuple[bool, list[float]]  # (verdict, timecodes of candidates above the provisional threshold)

DEFAULT_GRID = {
    "cash_box": [0.15, 0.2, 0.25, 0.3],
    "item_box": [0.2, 0.25, 0.3, 0.35],
    "min_category_score": [0.25, 0.35, 0.45],
    "disagree_factor": [0.25, 0.5, 1.0],
    "min_cash_score": [0.05, 0.08, 0.12],
    "provisional_threshold": [0.4, 0.5, 0.6],
}


@dataclass(frozen=True)
class Params:
    cash_box: float = 0.25  # category threshold for cash and cash_bag
    item_box: float = 0.25  # ... for personal, food_drink and phone
    min_category_score: float = 0.35
    disagree_factor: float = 0.5
    min_cash_score: float = 0.08
    provisional_threshold: float = 0.5


@dataclass
class Prepared:
    clip: int
    group: str  # md5 of the video file: byte-identical duplicates share a group
    duration: float
    video: V.Video  # frame-free copy: only t, luma, index kept
    layouts: LayoutTimeline
    people: PeopleResult
    raw: RawObjects


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare(path: Path, cfg: PipelineConfig, floor: float = FLOOR, det_cache: str | None = None) -> Prepared:
    from kassa_pipeline.cli import clip_number

    dev = models.pick_device(cfg.models.device)
    video = V.load_video(path, cfg.sampling.people_fps, cfg.sampling.scene_cut_threshold)
    ov = models.open_vocab_detector(cfg.models.open_vocab_detector, dev)
    if det_cache:
        ov = CachedDetector(ov, det_cache, cfg.models.open_vocab_detector)
    enc = models.crop_encoder(cfg.models.crop_encoder, dev)
    layouts = L.detect_layout_timeline(
        video.frames, video.scene_cuts, ov, enc, cfg.layout, cfg.sampling.layout_frames
    )
    pose = None if cfg.models.pose_model == "none" else models.pose_estimator(cfg.models.pose_model)
    pdet = models.person_detector(cfg.models.person_detector, dev)
    people = P.track_people(video, layouts, pdet, pose, cfg.people, cfg.sampling.people_fps)
    heads = models.trained_heads(cfg.models.heads_dir)
    raw = O.collect_object_candidates(
        video, layouts, people, ov, enc, heads, cfg.objects, cfg.sampling.objects_fps, floor
    )
    slim = dataclasses.replace(
        video, frames=[dataclasses.replace(f, rgb=None, norm=None) for f in video.frames]
    )
    return Prepared(clip_number(path), file_md5(path), video.duration, slim, layouts, people, raw)


def save(p: Prepared, out: Path, stem: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"{stem}.pkl"
    with open(f, "wb") as fh:
        pickle.dump(p, fh)
    return f


def load(path: Path) -> Prepared:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def default_params(base: PipelineConfig) -> Params:
    o = base.objects
    return Params(
        cash_box=o.category_thresholds.get("cash", o.box_threshold),
        item_box=o.category_thresholds.get("personal", o.box_threshold),
        min_category_score=o.min_category_score,
        disagree_factor=o.disagree_factor,
        provisional_threshold=base.events.provisional_threshold,
    )


def _objects_cfg(params: Params, base: PipelineConfig):
    th = dict(base.objects.category_thresholds)
    for c in ("cash", "cash_bag"):
        th[c] = params.cash_box
    for c in ("personal", "food_drink", "phone"):
        th[c] = params.item_box
    return dataclasses.replace(
        base.objects,
        category_thresholds=th,
        min_category_score=params.min_category_score,
        disagree_factor=params.disagree_factor,
    )


def _candidates(p: Prepared, params: Params, base: PipelineConfig) -> list[E.Candidate]:
    objects = O.select_objects(p.raw, _objects_cfg(params, base))
    tl = T.build_timeline(
        p.video, p.layouts, p.people, objects, base.timeline, min_cash_score=params.min_cash_score
    )
    return E.find_candidates(tl, base.events)


def _verdict(cands: list[E.Candidate], threshold: float) -> Pred:
    codes = [c.timecode for c in cands if c.score >= threshold]
    return bool(codes), codes


def evaluate(p: Prepared, params: Params, base: PipelineConfig) -> Pred:
    return _verdict(_candidates(p, params, base), params.provisional_threshold)


def score(preds: dict[int, Pred], labels: dict[int, tuple[bool, float | None]], tol: float = 5.0) -> dict:
    tp = fp = fn = tn = hit = hit_n = 0
    for clip, (verdict, codes) in preds.items():
        if clip not in labels:
            continue
        viol, sec = labels[clip]
        if verdict and viol:
            tp += 1
            if sec is not None:
                hit_n += 1
                hit += any(abs(c - sec) <= tol for c in codes)
        elif verdict:
            fp += 1
        elif viol:
            fn += 1
        else:
            tn += 1
    n = tp + fp + fn + tn
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": prec,
        "recall": rec,
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "accuracy": (tp + tn) / n if n else 0.0,
        "ts_hit_rate": hit / hit_n if hit_n else 0.0,
    }


def make_grid(grid: dict[str, list[float]] | None = None) -> list[Params]:
    grid = grid or DEFAULT_GRID
    keys = list(grid)
    return [Params(**dict(zip(keys, v))) for v in itertools.product(*grid.values())]


def predictions_table(
    prepared: list[Prepared], grid: list[Params], base: PipelineConfig
) -> list[dict[int, Pred]]:
    """preds[i][clip] for every param set, each (clip, params) computed once.

    Timelines are shared by param sets differing only in provisional_threshold (a plain filter).
    """
    table: list[dict[int, Pred]] = [{} for _ in grid]
    for n, p in enumerate(prepared, 1):
        _log(f"  sweep {n}/{len(prepared)} clip {p.clip}")
        memo: dict[Params, list[E.Candidate]] = {}
        for i, prm in enumerate(grid):
            key = dataclasses.replace(prm, provisional_threshold=0.0)
            if key not in memo:
                memo[key] = _candidates(p, prm, base)
            table[i][p.clip] = _verdict(memo[key], prm.provisional_threshold)
    return table


def _rank(m: dict) -> tuple:
    return (m["f1"], m["accuracy"], m["ts_hit_rate"])


def _subset(preds: dict[int, Pred], clips) -> dict[int, Pred]:
    return {c: preds[c] for c in clips}


def grouped_cv(
    prepared: list[Prepared],
    labels: dict,
    grid: list[Params],
    k: int = 5,
    tol: float = 5.0,
    table: list[dict[int, Pred]] | None = None,
    base: PipelineConfig | None = None,
) -> dict:
    """Pick the best params on train folds, score on the held-out fold; folds never split a group."""
    from sklearn.model_selection import GroupKFold

    if table is None:
        table = predictions_table(prepared, grid, base or PipelineConfig())
    ps = [p for p in prepared if p.clip in labels]
    clips = np.array([p.clip for p in ps])
    groups = np.array([p.group for p in ps])
    k = min(k, len(set(groups.tolist())))
    folds, pooled = [], {}
    for tr, te in GroupKFold(n_splits=k).split(clips, groups=groups):
        train, test = clips[tr].tolist(), clips[te].tolist()
        best = max(range(len(grid)), key=lambda i: _rank(score(_subset(table[i], train), labels, tol)))
        pooled.update(_subset(table[best], test))
        folds.append(
            {
                "test_clips": test,
                "params": dataclasses.asdict(grid[best]),
                "train": score(_subset(table[best], train), labels, tol),
                "test": score(_subset(table[best], test), labels, tol),
            }
        )
    return {"k": k, "folds": folds, "pooled": score(pooled, labels, tol)}


def sweep(
    prepared: list[Prepared],
    labels: dict,
    base: PipelineConfig | None = None,
    grid: dict[str, list[float]] | None = None,
    tol: float = 5.0,
    k: int = 5,
    top: int = 5,
) -> dict:
    base = base or PipelineConfig()
    missing = [p.clip for p in prepared if p.clip not in labels]
    if missing:
        _log(f"no label for clips {missing}: skipped")
    prepared = [p for p in prepared if p.clip in labels]
    params = make_grid(grid)
    table = predictions_table(prepared, params, base)
    clips = [p.clip for p in prepared]
    rows = [
        {"params": dataclasses.asdict(prm), **score(table[i], labels, tol)} for i, prm in enumerate(params)
    ]
    order = sorted(range(len(rows)), key=lambda i: _rank(rows[i]), reverse=True)
    dflt = default_params(base)
    dpreds = {p.clip: evaluate(p, dflt, base) for p in prepared}
    return {
        "n_clips": len(clips),
        "tol": tol,
        "default": {"params": dataclasses.asdict(dflt), **score(dpreds, labels, tol)},
        "in_sample": [rows[i] for i in order],
        "top_predictions": [
            {"params": rows[i]["params"], "verdicts": {str(c): table[i][c][0] for c in clips}}
            for i in order[:top]
        ],
        "cv": grouped_cv(prepared, labels, params, k, tol, table),
    }
