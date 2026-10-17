"""Cash / ID-card detector: dataset building and fine-tuning on Modal (separate from the jury service).

    modal volume put kassa-work <open/videos> /videos          # once: the training clips
    modal run vlm_service/kassa_train.py::build                 # desk crops + pre-labels -> /work/det
    modal run vlm_service/kassa_train.py::train --name v1       # RT-DETRv2-r50 -> /work/det/models/v1

Pre-labels: Grounding DINO proposes boxes on the 2x desk crops, Qwen3.5-9B names what is inside each box
(banknotes / money bag / ID card / paper / phone / hand / other). Human corrections, when present in
/work/det/human.json (same COCO layout), replace the pre-labels of the images they cover.
"""

import json
import re
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
PKG = HERE.parent / "src" / "kassa_pipeline"

app = modal.App("kassa-train")
hf_vol = modal.Volume.from_name("kassa-hf-cache", create_if_missing=True)
work_vol = modal.Volume.from_name("kassa-work", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch", "torchvision", "transformers>=5.5", "accelerate", "pillow", "numpy", "opencv-python-headless",
        "certifi", "sentencepiece", "protobuf", "torchmetrics", "pycocotools",
    )
    .env({"HF_HOME": "/cache", "PYTHONPATH": "/root"})
    .add_local_dir(str(PKG), remote_path="/root/kassa_pipeline", ignore=["__pycache__", "*.pyc"])
)

DET = Path("/work/det")
PROPOSE_PROMPT = "banknotes . money . stack of banknotes . cash bag . money bag . id card . plastic card . passport"
VERIFY_PROMPT = (
    "The red rectangle marks one object on a bank cashier's desk, seen from a ceiling camera. What is inside the red rectangle?\n"
    "A) banknotes or a stack of paper money\nB) a money bag (cash bag)\nC) an ID card, bank card or passport\n"
    "D) a sheet of paper, receipt or document\nE) a phone\nF) a hand or arm\nG) something else\nH) cannot tell\n"
    "Reply with one letter only."
)
LETTER = {"A": "cash", "B": "cash_bag", "C": "id_card"}
VAL_CLIPS = {"Х_3", "Х_26", "Х_38", "Х_17", "Х_21", "Х_40", "Х_46", "Х_12"}  # held out, by clip


def _stage1(clip: str) -> dict | None:
    cands = sorted(Path("/work/extract").glob(f"{clip}-*.json"), key=lambda p: p.stat().st_mtime)
    return json.loads(cands[-1].read_text()) if cands else None


def _nms(dets, iou=0.5):
    from kassa_pipeline import geometry as g

    keep = []
    for d in sorted(dets, key=lambda d: -d.score):
        if all(g.iou(d.box, k.box) < iou for k in keep):
            keep.append(d)
    return keep


def _verify_crop(img, box):
    import cv2

    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    H, W = img.shape[:2]
    cx1, cy1, cx2, cy2 = int(max(0, x1 - w)), int(max(0, y1 - h)), int(min(W, x2 + w)), int(min(H, y2 + h))
    c = img[cy1:cy2, cx1:cx2].copy()
    cv2.rectangle(c, (int(x1 - cx1), int(y1 - cy1)), (int(x2 - cx1), int(y2 - cy1)), (255, 0, 0), 2)
    k = max(1.0, 256 / max(1, min(c.shape[:2])))
    k = min(k, 448 / max(c.shape[:2]))
    return cv2.resize(c, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)


@app.function(gpu="L40S", image=image, volumes={"/cache": hf_vol, "/work": work_vol}, timeout=3 * 3600)
def build(box_threshold: float = 0.2, text_threshold: float = 0.2):
    import cv2

    from kassa_pipeline import det_data as D
    from kassa_pipeline import models as M
    from kassa_pipeline.vlm import QwenVLM
    from kassa_pipeline.vlm_scan import Stage1

    work_vol.reload()
    t0 = time.time()
    (DET / "images").mkdir(parents=True, exist_ok=True)
    samples, sizes, crops = [], {}, {}
    for v in sorted(Path("/work/videos").glob("*.mov")):
        clip = re.sub(r"\W+", "_", v.stem)
        st = _stage1(clip)
        if st is None:
            print("no stage 1 for", v.name)
            continue
        got = D.sample_clip(v, clip, Stage1(st))
        for s, img in got:
            cv2.imwrite(str(DET / "images" / f"{s.image_id}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
            samples.append(s)
            sizes[s.image_id] = (img.shape[1], img.shape[0])
            crops[s.image_id] = img
        print(f"{v.name}: {len(got)} crops ({sum(s.handoff for s, _ in got)} at hand-offs)")
    print(f"{len(samples)} images in {time.time() - t0:.0f}s")

    ov = M.open_vocab_detector("IDEA-Research/grounding-dino-tiny", "cuda")
    props = {}
    for s in samples:
        dets = ov([crops[s.image_id]], PROPOSE_PROMPT, box_threshold, text_threshold)[0]
        W, H = sizes[s.image_id]
        dets = [d for d in dets if (d.box[2] - d.box[0]) * (d.box[3] - d.box[1]) < 0.3 * W * H]
        props[s.image_id] = _nms(dets)
    n_props = sum(len(v) for v in props.values())
    print(f"{n_props} proposals in {time.time() - t0:.0f}s")

    vlm = QwenVLM("Qwen/Qwen3.5-9B", cache_path=str(DET / "vlm_cache.json"))
    items, keys = [], []
    for iid, dets in props.items():
        for d in dets:
            items.append(([_verify_crop(crops[iid], d.box)], VERIFY_PROMPT))
            keys.append((iid, d))
    answers = vlm.ask_many(items, max_new_tokens=4, batch=16)
    vlm.flush()
    boxes, letters = {}, {}
    for (iid, d), a in zip(keys, answers):
        m = re.search(r"[A-H]", a.upper())
        letter = m.group(0) if m else "H"
        letters[letter] = letters.get(letter, 0) + 1
        if letter in LETTER:
            x1, y1, x2, y2 = d.box
            boxes.setdefault(iid, []).append({"bbox": [x1, y1, x2 - x1, y2 - y1], "cls": LETTER[letter],
                                              "score": round(d.score, 3), "phrase": d.label, "vlm": letter})
    data = D.coco(samples, sizes, boxes)
    D.save_json(data, DET / "pseudo.json")
    work_vol.commit()
    per = {c: sum(a["category_id"] == i + 1 for a in data["annotations"]) for i, c in enumerate(D.CLASSES)}
    summary = {"images": len(samples), "proposals": n_props, "vlm_letters": letters, "boxes_per_class": per,
               "images_with_boxes": len(boxes), "seconds": round(time.time() - t0)}
    print(json.dumps(summary, ensure_ascii=False))
    return summary


@app.function(gpu="L40S", image=image, volumes={"/cache": hf_vol, "/work": work_vol}, timeout=3 * 3600)
def train(name: str = "v1", labels: str = "pseudo_v2.json", epochs: int = 40, batch: int = 8, lr: float = 1e-4,
          base: str = "PekingU/rtdetr_v2_r50vd"):
    import numpy as np
    import torch
    from PIL import Image
    from torchmetrics.detection import MeanAveragePrecision
    from torchvision import tv_tensors
    from torchvision.transforms import v2
    from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection

    from kassa_pipeline.det_data import CLASSES

    work_vol.reload()
    data = json.loads((DET / labels).read_text())
    human_path = DET / "human.json"
    if human_path.exists():  # human corrections replace pre-labels image by image
        human = json.loads(human_path.read_text())
        covered = {im["file_name"] for im in human["images"]}
        by_file = {im["file_name"]: im["id"] for im in data["images"]}
        hid_to_pid = {im["id"]: by_file[im["file_name"]] for im in human["images"] if im["file_name"] in by_file}
        data["annotations"] = [a for a in data["annotations"] if next(im["file_name"] for im in data["images"] if im["id"] == a["image_id"]) not in covered]
        data["annotations"] += [{**a, "image_id": hid_to_pid[a["image_id"]]} for a in human["annotations"] if a["image_id"] in hid_to_pid]
        print(f"human labels on {len(covered)} images")
    anns: dict[int, list] = {}
    for a in data["annotations"]:
        anns.setdefault(a["image_id"], []).append(a)
    ims = {im["id"]: im for im in data["images"]}
    tr = [i for i, im in ims.items() if im["clip"] not in VAL_CLIPS]
    va = [i for i, im in ims.items() if im["clip"] in VAL_CLIPS]
    print(f"train {len(tr)} images, val {len(va)} images (clips {sorted(VAL_CLIPS)})")

    proc = RTDetrImageProcessor.from_pretrained(base)
    id2label = dict(enumerate(CLASSES))
    model = RTDetrV2ForObjectDetection.from_pretrained(base, id2label=id2label, label2id={v: k for k, v in id2label.items()},
                                                       ignore_mismatched_sizes=True).cuda()
    aug = v2.Compose([v2.RandomPhotometricDistort(p=0.8), v2.RandomHorizontalFlip(),
                      v2.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.8, 1.25)), v2.SanitizeBoundingBoxes()])

    def load(i, train_mode):
        im = ims[i]
        img = np.array(Image.open(DET / "images" / im["file_name"]).convert("RGB"))
        al = anns.get(i, [])
        boxes = torch.tensor([[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]] for a in al],
                             dtype=torch.float32).reshape(-1, 4)
        labels = torch.tensor([a["category_id"] - 1 for a in al], dtype=torch.int64)
        if train_mode:
            out = aug({"image": tv_tensors.Image(torch.from_numpy(img).permute(2, 0, 1)),
                       "boxes": tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=img.shape[:2]), "labels": labels})
            img = out["image"].permute(1, 2, 0).numpy()
            boxes, labels = out["boxes"].as_subclass(torch.Tensor), out["labels"]
        coco_anns = [{"bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])], "category_id": int(c),
                      "area": float((b[2] - b[0]) * (b[3] - b[1])), "iscrowd": 0} for b, c in zip(boxes, labels)]
        return img, {"image_id": i, "annotations": coco_anns}, boxes, labels

    def batches(ids, train_mode):
        ids = list(ids)
        if train_mode:
            np.random.shuffle(ids)
        for s in range(0, len(ids), batch):
            yield [load(i, train_mode) for i in ids[s : s + batch]]

    head = [p for n, p in model.named_parameters() if "backbone" not in n]
    bb = [p for n, p in model.named_parameters() if "backbone" in n]
    opt = torch.optim.AdamW([{"params": head, "lr": lr}, {"params": bb, "lr": lr / 10}], weight_decay=1e-4)
    steps = epochs * ((len(tr) + batch - 1) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr, lr / 10], total_steps=steps, pct_start=0.05)

    def evaluate():
        model.eval()
        metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
        pres = {c: [0, 0, 0] for c in CLASSES}  # image-level presence at score 0.5: TP, FP, FN
        with torch.no_grad():
            for b in batches(va, False):
                enc = proc(images=[x[0] for x in b], return_tensors="pt").to("cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    outs = model(pixel_values=enc["pixel_values"])
                res = proc.post_process_object_detection(outs, threshold=0.05, target_sizes=[x[0].shape[:2] for x in b])
                preds = [{k: r[k].float().cpu() if k != "labels" else r[k].cpu() for k in ("boxes", "scores", "labels")} for r in res]
                tgts = [{"boxes": x[2], "labels": x[3]} for x in b]
                metric.update(preds, tgts)
                for p, t in zip(preds, tgts):
                    for ci, c in enumerate(CLASSES):
                        has_p = bool(((p["labels"] == ci) & (p["scores"] >= 0.5)).any())
                        has_t = bool((t["labels"] == ci).any())
                        pres[c][0] += has_p and has_t
                        pres[c][1] += has_p and not has_t
                        pres[c][2] += has_t and not has_p
        m = metric.compute()
        model.train()
        per = {}
        for ci, c in enumerate(CLASSES):
            idx = (m["classes"] == ci).nonzero()
            ap50 = None
            if "map_50_per_class" in m and len(idx):
                ap50 = float(m["map_50_per_class"][idx[0, 0]])
            tp, fp, fn = pres[c]
            per[c] = {"image_precision": round(tp / max(tp + fp, 1), 3), "image_recall": round(tp / max(tp + fn, 1), 3),
                      "tp": tp, "fp": fp, "fn": fn, "ap50": ap50}
        return {"map": round(float(m["map"]), 3), "map_50": round(float(m["map_50"]), 3), "per_class": per}

    t0, step, hist = time.time(), 0, []
    model.train()
    for ep in range(epochs):
        tot = 0.0
        for b in batches(tr, True):
            enc = proc(images=[x[0] for x in b], annotations=[x[1] for x in b], return_tensors="pt")
            labels = [{k: v.cuda() for k, v in lab.items()} for lab in enc["labels"]]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(pixel_values=enc["pixel_values"].cuda(), labels=labels).loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            opt.step()
            sched.step()
            tot += float(loss)
            step += 1
        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            ev = evaluate()
            hist.append({"epoch": ep + 1, "loss": round(tot / max(1, len(tr) // batch), 3), **ev})
            print(json.dumps(hist[-1]), f"{time.time() - t0:.0f}s", flush=True)
    out = DET / "models" / name
    model.save_pretrained(out)
    proc.save_pretrained(out)
    (out / "metrics.json").write_text(json.dumps({"history": hist, "val_clips": sorted(VAL_CLIPS), "train_images": len(tr),
                                                  "val_images": len(va), "human_labels": human_path.exists()}, indent=1))
    work_vol.commit()
    return hist[-1]


@app.function(image=image, volumes={"/work": work_vol}, timeout=1800)
def sheets(labels_file: str = "pseudo.json", per_class: int = 40, seed: int = 0):
    """Contact sheets of random boxes per class (box drawn in red, with context) -> /work/det/review."""
    import random

    import cv2
    import numpy as np

    from kassa_pipeline.det_data import CLASSES

    work_vol.reload()
    data = json.loads((DET / labels_file).read_text())
    ims = {im["id"]: im for im in data["images"]}
    rnd = random.Random(seed)
    out = DET / "review"
    out.mkdir(parents=True, exist_ok=True)
    made = []
    for ci, c in enumerate(CLASSES):
        anns = [a for a in data["annotations"] if a["category_id"] == ci + 1]
        pick = rnd.sample(anns, min(per_class, len(anns)))
        tiles = []
        for a in pick:
            im = ims[a["image_id"]]
            img = cv2.imread(str(DET / "images" / im["file_name"]))
            x, y, w, h = a["bbox"]
            H, W = img.shape[:2]
            x1, y1, x2, y2 = int(max(0, x - w)), int(max(0, y - h)), int(min(W, x + 2 * w)), int(min(H, y + 2 * h))
            t = img[y1:y2, x1:x2].copy()
            cv2.rectangle(t, (int(x - x1), int(y - y1)), (int(x + w - x1), int(y + h - y1)), (0, 0, 255), 2)
            t = cv2.resize(t, (200, 200))
            cv2.putText(t, f"{im['clip']} {im['t']:.0f}s {a.get('score', 0):.2f}", (3, 14), 0, 0.4, (0, 255, 255), 1)
            tiles.append(t)
        while len(tiles) % 8:
            tiles.append(np.zeros((200, 200, 3), np.uint8))
        sheet = np.vstack([np.hstack(tiles[i : i + 8]) for i in range(0, len(tiles), 8)])
        p = out / f"{Path(labels_file).stem}_{c}.jpg"
        cv2.imwrite(str(p), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
        made.append(str(p))
    work_vol.commit()
    return made


VERIFY_PROMPT_V2 = (
    "{view} from a ceiling camera over a bank cashier's desk. What is {where}?\n"
    "A) banknotes: a stack or bundle of paper money (a thick stack, often with a paper band), or loose banknotes\n"
    "B) a money bag: a soft green fabric bag for cash\n"
    "C) an ID card or plastic card: one thin rectangular card, often with a small photo; or a passport\n"
    "D) a sheet of paper, receipt or document\n"
    "E) a mobile phone\n"
    "F) a hand or arm with nothing in it\n"
    "G) a banknote counting machine, its display or its tray\n"
    "H) a desk telephone\n"
    "I) a laptop or keyboard\n"
    "J) a handbag, backpack, coat or other personal item\n"
    "K) part of the desk, chair, door, wall, cable or socket\n"
    "L) something else\n"
    "M) cannot tell\n"
    "Reply with one letter only."
)
VIEWS = [("This picture shows part of a scene", "inside the red rectangle"), ("This is a close-up of one object", "this object")]
EQUIPMENT = ("laptop", "counter", "telephone", "safe")


def _to_crop(box, region, scale):
    return ((box[0] - region[0]) * scale, (box[1] - region[1]) * scale, (box[2] - region[0]) * scale, (box[3] - region[1]) * scale)


def _tight(img, box, pad=0.25):
    import cv2

    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    H, W = img.shape[:2]
    c = img[int(max(0, y1 - pad * h)) : int(min(H, y2 + pad * h)), int(max(0, x1 - pad * w)) : int(min(W, x2 + pad * w))]
    k = min(max(1.0, 224 / max(1, min(c.shape[:2]))), 448 / max(c.shape[:2]))
    return cv2.resize(c, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)


@app.function(gpu="L40S", image=image, volumes={"/cache": hf_vol, "/work": work_vol}, timeout=3 * 3600)
def relabel(out_name: str = "pseudo_v2.json", box_threshold: float = 0.2, text_threshold: float = 0.2):
    """Pre-labels v2 on the saved crops: equipment and edge filters from Stage 1, a 13-way question, and
    two views (context, close-up) that must agree."""
    import cv2

    from kassa_pipeline import det_data as D
    from kassa_pipeline import geometry as g
    from kassa_pipeline import models as M
    from kassa_pipeline.vlm import QwenVLM
    from kassa_pipeline.vlm_scan import Stage1

    work_vol.reload()
    t0 = time.time()
    meta = json.loads((DET / "pseudo.json").read_text())["images"]
    stage1 = {}
    ov = M.open_vocab_detector("IDEA-Research/grounding-dino-tiny", "cuda")
    drops = {"equipment": 0, "edge": 0, "size": 0}
    cands = []  # (image meta, crop image, box)
    for im in meta:
        if im["clip"] not in stage1:
            stage1[im["clip"]] = Stage1(_stage1(im["clip"]))
        lay = stage1[im["clip"]].layout(im["t"])
        img = cv2.cvtColor(cv2.imread(str(DET / "images" / im["file_name"])), cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        equip = [_to_crop(lay[k], im["region"], im["scale"]) for k in EQUIPMENT if lay.get(k)]
        equip += [_to_crop(b, im["region"], im["scale"]) for k in ("doors", "chairs") for b in lay.get(k, [])]
        for d in _nms(ov([img], PROPOSE_PROMPT, box_threshold, text_threshold)[0]):
            x1, y1, x2, y2 = d.box
            area = (x2 - x1) * (y2 - y1)
            if area > 0.15 * W * H:
                drops["size"] += 1
                continue
            if area > 0.02 * W * H and (x1 < 3 or y1 < 3 or x2 > W - 3 or y2 > H - 3):
                drops["edge"] += 1
                continue
            c = g.center(d.box)
            if any(g.contains(e, c) or g.iou(d.box, e) > 0.3 for e in equip):
                drops["equipment"] += 1
                continue
            cands.append((im, img, d))
    print(f"{len(cands)} proposals kept, dropped {drops}, {time.time() - t0:.0f}s", flush=True)

    vlm = QwenVLM("Qwen/Qwen3.5-9B", cache_path=str(DET / "vlm_cache_v2.json"))
    items = []
    for im, img, d in cands:
        items.append(([_verify_crop(img, d.box)], VERIFY_PROMPT_V2.format(view=VIEWS[0][0], where=VIEWS[0][1])))
        items.append(([_tight(img, d.box)], VERIFY_PROMPT_V2.format(view=VIEWS[1][0], where=VIEWS[1][1])))
    answers = vlm.ask_many(items, max_new_tokens=4, batch=16)
    vlm.flush()

    def letter(a):
        m = re.search(r"[A-M]", a.strip().upper())
        return m.group(0) if m else "M"

    boxes, pairs, agree = {}, {}, 0
    for k, (im, img, d) in enumerate(cands):
        a, b = letter(answers[2 * k]), letter(answers[2 * k + 1])
        pairs[f"{a}{b}"] = pairs.get(f"{a}{b}", 0) + 1
        if a == b and a in LETTER:
            agree += 1
            x1, y1, x2, y2 = d.box
            boxes.setdefault(im["file_name"], []).append({"bbox": [x1, y1, x2 - x1, y2 - y1], "cls": LETTER[a],
                                                          "score": round(d.score, 3), "phrase": d.label, "vlm": a + b})
    samples = [D.Sample(Path(im["file_name"]).stem, im["clip"], im["t"], tuple(im["region"]), im["scale"], im["handoff"]) for im in meta]
    sizes = {Path(im["file_name"]).stem: (im["width"], im["height"]) for im in meta}
    data = D.coco(samples, sizes, {Path(f).stem: v for f, v in boxes.items()})
    D.save_json(data, DET / out_name)
    work_vol.commit()
    per = {c: sum(a["category_id"] == i + 1 for a in data["annotations"]) for i, c in enumerate(D.CLASSES)}
    summary = {"proposals_kept": len(cands), "dropped": drops, "agreeing_boxes": agree, "boxes_per_class": per,
               "images_with_boxes": len(boxes), "top_answer_pairs": dict(sorted(pairs.items(), key=lambda x: -x[1])[:12]),
               "seconds": round(time.time() - t0)}
    print(json.dumps(summary), flush=True)
    return summary
