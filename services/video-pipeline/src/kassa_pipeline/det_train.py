"""Fine-tune an RT-DETRv2 detector on a COCO file, locally (a 4 GB GPU is enough with the r18 base).

The same loop as vlm_service/kassa_train.py::train, without Modal: small batches with gradient accumulation,
bf16, and an evaluation that needs no pycocotools (AP50 and precision / recall at IoU 0.5, per class).
Images whose `clip` is in the validation set are held out, so frames of one clip never sit on both sides.

    python -m kassa_pipeline.det_train --coco det/pseudo_v2.json --images det/images --out models/desk-v1
    python -m kassa_pipeline.det_train --coco persons/coco.json --images persons/images --out models/person-v1

The saved folder goes into ExtractConfig.desk_detector (cash, cash_bag, id_card) or person_detector ("person").
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

VAL_CLIPS = ("X-3", "X-26", "X-38", "X-17", "X-21", "X-40", "X-46", "X-12")  # as kassa_train.VAL_CLIPS


def load_coco(path: str | Path) -> tuple[dict[int, dict], dict[int, list[dict]], list[str]]:
    """images by id, annotations by image id, class names in category-id order (ids 1..n)."""
    data = json.loads(Path(path).read_text())
    cats = sorted(data["categories"], key=lambda c: c["id"])
    if [c["id"] for c in cats] != list(range(1, len(cats) + 1)):
        raise ValueError(f"category ids must be 1..n, got {[c['id'] for c in cats]}")
    anns: dict[int, list[dict]] = {}
    for a in data["annotations"]:
        anns.setdefault(a["image_id"], []).append(a)
    return {im["id"]: im for im in data["images"]}, anns, [c["name"] for c in cats]


def _canon(clip: str) -> str:
    from kassa_pipeline.facts import clip_id

    return clip_id(clip.replace("_", "-"))  # det_data names clips "Х_3"


def split(images: dict[int, dict], val_clips: tuple[str, ...] = VAL_CLIPS) -> tuple[list[int], list[int]]:
    val = {_canon(c) for c in val_clips}
    tr = [i for i, im in images.items() if _canon(im.get("clip", "")) not in val]
    va = [i for i, im in images.items() if _canon(im.get("clip", "")) in val]
    return tr, va


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix of xyxy boxes (N, 4) x (M, 4)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def scores(preds: list[dict], targets: list[dict], classes: list[str], at: float = 0.5) -> dict:
    """preds/targets per image: {"boxes": (N, 4) xyxy, "labels": (N,), "scores": (N,) for preds}.
    Per class: AP50 (all-point interpolation) and precision / recall of the boxes scoring >= `at`."""
    out = {}
    for ci, name in enumerate(classes):
        dets, n_gt = [], 0
        for p, t in zip(preds, targets):
            gt = np.asarray(t["boxes"], np.float64).reshape(-1, 4)[np.asarray(t["labels"]) == ci]
            n_gt += len(gt)
            m = np.asarray(p["labels"]) == ci
            boxes = np.asarray(p["boxes"], np.float64).reshape(-1, 4)[m]
            sc = np.asarray(p["scores"], np.float64)[m]
            order = np.argsort(-sc)
            iou, used = _iou(boxes[order], gt), np.zeros(len(gt), bool)
            for r, k in enumerate(order):
                j = int(np.argmax(iou[r])) if len(gt) else -1
                hit = j >= 0 and iou[r, j] >= 0.5 and not used[j]
                if hit:
                    used[j] = True
                dets.append((float(sc[k]), hit))
        dets.sort(key=lambda d: -d[0])
        tp = np.cumsum([d[1] for d in dets]) if dets else np.zeros(0)
        prec = tp / np.arange(1, len(dets) + 1) if dets else np.zeros(0)
        rec = tp / n_gt if n_gt else np.zeros(len(dets))
        ap = None
        if n_gt:
            r = np.concatenate([[0.0], rec, [1.0]])
            pr = np.concatenate([[1.0], prec, [0.0]])
            pr = np.maximum.accumulate(pr[::-1])[::-1]
            ap = round(float(np.sum((r[1:] - r[:-1]) * pr[1:])), 3)
        kept = [d for d in dets if d[0] >= at]
        tp_at = sum(d[1] for d in kept)
        out[name] = {"ap50": ap, "gt": n_gt, f"precision@{at}": round(tp_at / len(kept), 3) if kept else None,
                     f"recall@{at}": round(tp_at / n_gt, 3) if n_gt else None}
    aps = [v["ap50"] for v in out.values() if v["ap50"] is not None]
    return {"map50": round(float(np.mean(aps)), 3) if aps else None, "per_class": out}


def base_class_ids(base_id2label: dict, classes: list[str]) -> list[int] | None:
    """Label ids of `classes` in the base model when it already knows every one of them, else None."""
    by_name = {str(n): int(i) for i, n in base_id2label.items()}
    return [by_name[c] for c in classes] if all(c in by_name for c in classes) else None


def train(coco: str, images_dir: str, out: str, base: str = "PekingU/rtdetr_v2_r18vd", epochs: int = 40,
          batch: int = 2, accum: int = 4, lr: float = 1e-4, val_clips: tuple[str, ...] = VAL_CLIPS,
          eval_every: int = 5, device: str = "auto") -> dict:
    import torch
    from PIL import Image
    from torchvision import tv_tensors
    from torchvision.transforms import v2
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForObjectDetection

    from kassa_pipeline.models import pick_device

    dev = pick_device(device)
    ims, anns, classes = load_coco(coco)
    tr, va = split(ims, val_clips)
    if not tr:
        raise ValueError("no training images left after holding out the validation clips")
    print(f"train {len(tr)} images, val {len(va)} images, classes {classes}", flush=True)

    proc = AutoImageProcessor.from_pretrained(base)
    base_ids = base_class_ids(AutoConfig.from_pretrained(base).id2label, classes)
    if base_ids is not None:  # every class is already in the base model ("person"): keep its trained head
        model = AutoModelForObjectDetection.from_pretrained(base).to(dev)
        print(f"keeping the base class head, classes at ids {base_ids}", flush=True)
    else:
        id2label = dict(enumerate(classes))
        model = AutoModelForObjectDetection.from_pretrained(
            base, id2label=id2label, label2id={v: k for k, v in id2label.items()}, ignore_mismatched_sizes=True
        ).to(dev)
        base_ids = list(range(len(classes)))
    ours = {b: k for k, b in enumerate(base_ids)}  # model label id -> index in `classes`
    aug = v2.Compose([v2.RandomPhotometricDistort(p=0.8), v2.RandomHorizontalFlip(),
                      v2.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.8, 1.25)), v2.SanitizeBoundingBoxes()])
    root = Path(images_dir)
    amp = dict(device_type="cuda", dtype=torch.bfloat16) if dev == "cuda" else dict(device_type="cpu", enabled=False)

    def load(i, train_mode):
        img = np.array(Image.open(root / ims[i]["file_name"]).convert("RGB"))
        al = anns.get(i, [])
        boxes = torch.tensor([[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
                              for a in al], dtype=torch.float32).reshape(-1, 4)
        labels = torch.tensor([base_ids[a["category_id"] - 1] for a in al], dtype=torch.int64)
        if train_mode:
            o = aug({"image": tv_tensors.Image(torch.from_numpy(img).permute(2, 0, 1)),
                     "boxes": tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=img.shape[:2]), "labels": labels})
            img, boxes, labels = o["image"].permute(1, 2, 0).numpy(), o["boxes"].as_subclass(torch.Tensor), o["labels"]
        coco_anns = [{"bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])], "category_id": int(c),
                      "area": float((b[2] - b[0]) * (b[3] - b[1])), "iscrowd": 0} for b, c in zip(boxes, labels)]
        return img, {"image_id": i, "annotations": coco_anns}, boxes, labels

    def batches(ids, train_mode):
        ids = list(ids)
        if train_mode:
            np.random.shuffle(ids)
        for s in range(0, len(ids), batch):
            yield [load(i, train_mode) for i in ids[s : s + batch]]

    def evaluate():
        model.eval()
        preds, tgts = [], []
        with torch.no_grad():
            for b in batches(va, False):
                enc = proc(images=[x[0] for x in b], return_tensors="pt").to(dev)
                with torch.autocast(**amp):
                    o = model(pixel_values=enc["pixel_values"])
                res = proc.post_process_object_detection(o, threshold=0.05, target_sizes=[x[0].shape[:2] for x in b])
                for r in res:  # back to indices in `classes`; other base classes are not scored
                    lab = r["labels"].cpu().numpy()
                    keep = np.array([int(v) in ours for v in lab], bool)
                    preds.append({"boxes": r["boxes"].float().cpu().numpy()[keep], "scores": r["scores"].float().cpu().numpy()[keep],
                                  "labels": np.array([ours[int(v)] for v in lab[keep]], np.int64)})
                tgts += [{"boxes": x[2].numpy(), "labels": np.array([ours[int(v)] for v in x[3]], np.int64)} for x in b]
        model.train()
        return scores(preds, tgts, classes)

    head = [p for n, p in model.named_parameters() if "backbone" not in n]
    bb = [p for n, p in model.named_parameters() if "backbone" in n]
    opt = torch.optim.AdamW([{"params": head, "lr": lr}, {"params": bb, "lr": lr / 10}], weight_decay=1e-4)
    steps = max(1, epochs * ((len(tr) + batch * accum - 1) // (batch * accum)))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr, lr / 10], total_steps=steps, pct_start=0.05)

    t0, hist, k = time.time(), [], 0
    model.train()
    for ep in range(epochs):
        tot, n = 0.0, 0
        for b in batches(tr, True):
            enc = proc(images=[x[0] for x in b], annotations=[x[1] for x in b], return_tensors="pt")
            labels = [{kk: v.to(dev) for kk, v in lab.items()} for lab in enc["labels"]]
            with torch.autocast(**amp):
                loss = model(pixel_values=enc["pixel_values"].to(dev), labels=labels).loss
            (loss / accum).backward()
            tot, n, k = tot + float(loss), n + 1, k + 1
            if k % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                opt.step()
                opt.zero_grad()
                if sched.last_epoch + 1 < steps:
                    sched.step()
        row = {"epoch": ep + 1, "loss": round(tot / max(n, 1), 3), "sec": round(time.time() - t0)}
        if va and ((ep + 1) % eval_every == 0 or ep == epochs - 1):
            row.update(evaluate())
        hist.append(row)
        print(json.dumps(row), flush=True)

    dst = Path(out)
    model.save_pretrained(dst)
    proc.save_pretrained(dst)
    (dst / "metrics.json").write_text(json.dumps({"base": base, "classes": classes, "val_clips": list(val_clips),
                                                  "train_images": len(tr), "val_images": len(va), "history": hist}, indent=1))
    return hist[-1]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m kassa_pipeline.det_train", description="fine-tune RT-DETRv2 on a COCO file")
    ap.add_argument("--coco", required=True)
    ap.add_argument("--images", required=True, help="folder the COCO file_name entries are relative to")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="PekingU/rtdetr_v2_r18vd", help="r50vd for more accuracy if memory allows")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4, help="gradient accumulation: effective batch = batch x accum")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val-clips", default=",".join(VAL_CLIPS), help="comma-separated clips held out for evaluation")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)
    last = train(a.coco, a.images, a.out, a.base, a.epochs, a.batch, a.accum, a.lr,
                 tuple(c for c in a.val_clips.split(",") if c), device=a.device)
    print(json.dumps(last, indent=1))


if __name__ == "__main__":
    main()
