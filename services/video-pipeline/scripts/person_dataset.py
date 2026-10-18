"""Person boxes from the ceiling camera for fine-tuning the person detector, labelled locally in Label Studio.

1. Frames at 1 fps (near-duplicates dropped) with the current detector's boxes as pre-labels:
       python scripts/person_dataset.py export <videos...> --out persons
2. Label Studio (local files, as in det_labelstudio.py): labeling config persons/config.xml, source storage
   persons/images, import persons/tasks.json. Fix the boxes: add missed people (half hidden, seen from behind,
   at the door), delete reflections in the glass and people outside the room.
3. Export as JSON and convert it back to COCO:
       python scripts/person_dataset.py import <export.json> --out persons
4. Train: python -m kassa_pipeline.det_train --coco persons/coco.json --images persons/images --out models/person-v1
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

CONFIG = """<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true"/>
  <RectangleLabels name="label" toName="image">
    <Label value="person" background="#43a047"/>
  </RectangleLabels>
</View>
"""


def frames(path: Path, fps: float = 1.0, min_diff: float = 1.5):
    """(t, RGB) at `fps`, skipping frames nearly identical to the last kept one."""
    cap = cv2.VideoCapture(str(path))
    src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step, i, next_t, last = 1.0 / fps, 0, 0.0, None
    while cap.grab():
        t = i / src
        i += 1
        if t + 1e-6 < next_t:
            continue
        next_t += step
        ok, bgr = cap.retrieve()
        if not ok:
            break
        small = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (64, 64)).astype(np.float32)
        if last is not None and float(np.abs(small - last).mean()) < min_diff:
            continue
        last = small
        yield round(t, 2), cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    cap.release()


def export(videos: list[Path], out: Path, detector: str, threshold: float, device: str):
    from kassa_pipeline import models as M
    from kassa_pipeline.facts import clip_id

    det = M.person_detector(detector, M.pick_device(device))
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    for v in videos:
        clip = clip_id(v.name)
        for t, rgb in frames(v):
            name = f"{clip}_{t:06.2f}.jpg"
            cv2.imwrite(str(img_dir / name), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            h, w = rgb.shape[:2]
            res = [{"from_name": "label", "to_name": "image", "type": "rectanglelabels", "original_width": w,
                    "original_height": h, "score": round(d.score, 3),
                    "value": {"x": 100 * d.box[0] / w, "y": 100 * d.box[1] / h, "width": 100 * (d.box[2] - d.box[0]) / w,
                              "height": 100 * (d.box[3] - d.box[1]) / h, "rotation": 0, "rectanglelabels": ["person"]}}
                   for d in det([rgb], threshold)[0]]
            tasks.append({"data": {"image": f"/data/local-files/?d=images/{name}", "file_name": name, "clip": clip, "t": t,
                                   "width": w, "height": h},
                          "predictions": [{"model_version": detector, "result": res}]})
        print(clip, sum(1 for x in tasks if x["data"]["clip"] == clip), "frames", flush=True)
    (out / "tasks.json").write_text(json.dumps(tasks, ensure_ascii=False))
    (out / "config.xml").write_text(CONFIG)
    print(f"{len(tasks)} tasks -> {out / 'tasks.json'}; labeling config -> {out / 'config.xml'}")


def to_coco(export_json: list[dict]) -> dict:
    """Label Studio JSON export -> COCO with one class "person". Tasks without an annotation are skipped."""
    images, anns = [], []
    for task in export_json:
        done = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
        if not done:
            continue
        d = task["data"]
        iid = len(images) + 1
        images.append({"id": iid, "file_name": d["file_name"], "width": d["width"], "height": d["height"],
                       "clip": d["clip"], "t": d["t"]})
        for r in done[-1]["result"]:
            if r.get("type") != "rectanglelabels":
                continue
            v, W, H = r["value"], r.get("original_width", d["width"]), r.get("original_height", d["height"])
            x, y, bw, bh = v["x"] * W / 100, v["y"] * H / 100, v["width"] * W / 100, v["height"] * H / 100
            anns.append({"id": len(anns) + 1, "image_id": iid, "category_id": 1, "bbox": [x, y, bw, bh],
                         "area": bw * bh, "iscrowd": 0})
    return {"images": images, "annotations": anns, "categories": [{"id": 1, "name": "person"}]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("videos", nargs="+", type=Path)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--detector", default="PekingU/rtdetr_v2_r101vd")
    e.add_argument("--threshold", type=float, default=0.3, help="low on purpose: deleting a box is quicker than drawing one")
    e.add_argument("--device", default="auto")
    i = sub.add_parser("import")
    i.add_argument("export_json", type=Path)
    i.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == "export":
        export(a.videos, a.out, a.detector, a.threshold, a.device)
    else:
        coco = to_coco(json.loads(a.export_json.read_text()))
        (a.out / "coco.json").write_text(json.dumps(coco, ensure_ascii=False))
        print(f"{len(coco['images'])} images, {len(coco['annotations'])} boxes -> {a.out / 'coco.json'}")


if __name__ == "__main__":
    main()
