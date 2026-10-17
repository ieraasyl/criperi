"""Crop export and head training for the small classifiers (objects, screen, safe).

Workflow: `kassa export-crops <videos>` writes unsorted crops; move them into
`crops/<task>/<class>/`; `kassa train-heads` fits one logistic-regression head per task on frozen
SigLIP embeddings and saves `heads/<task>.joblib`. The pipeline picks heads up automatically.
"""

import re
from pathlib import Path

import cv2
import numpy as np

from kassa_pipeline import geometry as g
from kassa_pipeline import models
from kassa_pipeline import objects as O
from kassa_pipeline.config import PipelineConfig

TASKS = {"object": list(O.OBJECT_PROMPTS), "screen": list(O.SCREEN_PROMPTS), "safe": list(O.SAFE_PROMPTS)}


def export_crops(videos: list[Path], out: Path, cfg: PipelineConfig, every_sec: int = 1) -> int:
    """Dump object candidates, laptop screens and safes at ~1 fps, named <clip>_<sec>_<i>.jpg.

    The zero-shot guess goes into the file name as a hint to speed up sorting.
    """
    from kassa_pipeline import layout as L
    from kassa_pipeline import video as V

    dev = models.pick_device(cfg.models.device)
    ov = models.open_vocab_detector(cfg.models.open_vocab_detector, dev)
    enc = models.crop_encoder(cfg.models.crop_encoder, dev)
    n = 0
    for v in videos:
        clip = re.sub(r"\W+", "", v.stem)
        vid = V.load_video(v, 1.0, cfg.sampling.scene_cut_threshold)
        tl = L.detect_layout_timeline(vid.frames, vid.scene_cuts, ov, enc, cfg.layout, cfg.sampling.layout_frames)
        for f in vid.frames[::every_sec]:
            lay = tl.at(f.t)
            h, w = f.norm.shape[:2]
            jobs = []
            if lay.laptop:
                jobs.append(("screen", g.expand(lay.laptop, 0.3, w, h)))
            if lay.safe:
                jobs.append(("safe", g.expand(lay.safe, 0.5, w, h)))
            if lay.workstation:
                region = g.expand(lay.workstation, cfg.objects.desk_crop_pad, w, h)
                crop = g.crop(f.norm, region)
                if crop is not None:
                    for i, d in enumerate(ov([crop], cfg.objects.prompt, cfg.objects.box_threshold, cfg.objects.text_threshold)[0]):
                        b = (d.box[0] + region[0], d.box[1] + region[1], d.box[2] + region[0], d.box[3] + region[1])
                        jobs.append((f"object", g.expand(b, 0.4, w, h)))
            for i, (task, box) in enumerate(jobs):
                c = g.crop(f.norm, box, min_size=12)
                if c is None:
                    continue
                d = out / task / "unsorted"
                d.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(d / f"{clip}_{int(f.t):03d}_{i}.jpg"), cv2.cvtColor(c, cv2.COLOR_RGB2BGR))
                n += 1
    return n


def train_heads(crops_dir: Path, heads_dir: Path, cfg: PipelineConfig) -> dict[str, dict]:
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_predict

    dev = models.pick_device(cfg.models.device)
    enc = models.crop_encoder(cfg.models.crop_encoder, dev)
    heads_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for task in TASKS:
        files, labels = [], []
        for cls_dir in sorted((crops_dir / task).glob("*")):
            if cls_dir.name == "unsorted" or not cls_dir.is_dir():
                continue
            for p in cls_dir.glob("*.jpg"):
                files.append(p)
                labels.append(cls_dir.name)
        if len(set(labels)) < 2:
            continue
        imgs = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in files]
        X = enc.embed(imgs)
        y = np.array(labels)
        # group by clip: a clip's neighbouring seconds look alike, so never split a clip across folds
        groups = np.array([p.name.split("_")[0] for p in files])
        clf = LogisticRegression(max_iter=2000, C=2.0, class_weight="balanced")
        info = {"n": len(y), "classes": {c: int((y == c).sum()) for c in sorted(set(y))}}
        if len(set(groups)) >= 3:
            pred = cross_val_predict(clf, X, y, groups=groups, cv=GroupKFold(n_splits=min(5, len(set(groups)))))
            info["clip_cv_accuracy"] = float((pred == y).mean())
        clf.fit(X, y)
        joblib.dump(clf, heads_dir / f"{task}.joblib")
        report[task] = info
    return report
