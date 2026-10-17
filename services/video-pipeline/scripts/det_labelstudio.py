"""Correct the detector's pre-labels in Label Studio (local, the images never leave the laptop).

1. Get the dataset and make Label Studio tasks with the pre-labels as predictions:
       python scripts/det_labelstudio.py export --out det_labels
2. Start Label Studio with local file serving and import det_labels/tasks.json:
       LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=$PWD/det_labels \\
           label-studio start
   Project -> Settings -> Labeling Interface -> paste det_labels/config.xml;
   Settings -> Cloud Storage -> Add Source Storage -> Local files, path $PWD/det_labels/images (do not sync);
   Import -> det_labels/tasks.json. Correct boxes; accept a task as is when the pre-labels are right.
3. Export from Label Studio as JSON (not JSON-MIN) and send the corrections to Modal:
       python scripts/det_labelstudio.py import <export.json> --out det_labels
   `modal run vlm_service/kassa_train.py::train --name v2` then trains on them.
"""

import argparse
import json
import subprocess
from pathlib import Path

CLASSES = ["cash", "cash_bag", "id_card"]
CONFIG = """<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true"/>
  <RectangleLabels name="label" toName="image">
    <Label value="cash" background="#e53935"/>
    <Label value="cash_bag" background="#fb8c00"/>
    <Label value="id_card" background="#1e88e5"/>
  </RectangleLabels>
</View>
"""


def modal_get(remote: str, local: Path):
    subprocess.run(["modal", "volume", "get", "kassa-work", remote, str(local), "--force"], check=True)


def export(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    modal_get("/det/pseudo.json", out / "pseudo.json")
    modal_get("/det/images", out)
    data = json.loads((out / "pseudo.json").read_text())
    anns: dict[int, list] = {}
    for a in data["annotations"]:
        anns.setdefault(a["image_id"], []).append(a)
    tasks = []
    for im in data["images"]:
        W, H = im["width"], im["height"]
        res = [{
            "from_name": "label", "to_name": "image", "type": "rectanglelabels", "original_width": W, "original_height": H,
            "value": {"x": 100 * a["bbox"][0] / W, "y": 100 * a["bbox"][1] / H, "width": 100 * a["bbox"][2] / W,
                      "height": 100 * a["bbox"][3] / H, "rotation": 0, "rectanglelabels": [CLASSES[a["category_id"] - 1]]},
        } for a in anns.get(im["id"], [])]
        tasks.append({"data": {"image": f"/data/local-files/?d=images/{im['file_name']}", "file_name": im["file_name"],
                               "clip": im["clip"], "t": im["t"], "handoff": im["handoff"]},
                      "predictions": [{"model_version": "gdino+qwen", "result": res}]})
    # hand-off frames first: that is where ID cards and money change hands
    tasks.sort(key=lambda t: (not t["data"]["handoff"], t["data"]["clip"], t["data"]["t"]))
    (out / "tasks.json").write_text(json.dumps(tasks, ensure_ascii=False))
    (out / "config.xml").write_text(CONFIG)
    print(f"{len(tasks)} tasks -> {out / 'tasks.json'}; labeling config -> {out / 'config.xml'}")


def import_(export_json: Path, out: Path):
    ls = json.loads(export_json.read_text())
    pseudo = json.loads((out / "pseudo.json").read_text())
    by_file = {im["file_name"]: im for im in pseudo["images"]}
    images, annotations, k = [], [], 1
    for task in ls:
        done = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
        if not done:
            continue  # not reviewed: keep the pre-labels for this image
        im = by_file.get(task["data"].get("file_name"))
        if im is None:
            continue
        images.append(im)
        for r in done[-1]["result"]:
            if r.get("type") != "rectanglelabels":
                continue
            v, W, H = r["value"], r["original_width"], r["original_height"]
            box = [v["x"] * W / 100, v["y"] * H / 100, v["width"] * W / 100, v["height"] * H / 100]
            annotations.append({"id": k, "image_id": im["id"], "category_id": CLASSES.index(v["rectanglelabels"][0]) + 1,
                                "bbox": box, "area": box[2] * box[3], "iscrowd": 0})
            k += 1
    human = {"images": images, "annotations": annotations, "categories": pseudo["categories"]}
    p = out / "human.json"
    p.write_text(json.dumps(human, ensure_ascii=False))
    subprocess.run(["modal", "volume", "put", "kassa-work", str(p), "/det/human.json", "--force"], check=True)
    print(f"{len(images)} reviewed images, {len(annotations)} boxes -> /det/human.json on the volume")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--out", default="det_labels")
    i = sub.add_parser("import")
    i.add_argument("export_json")
    i.add_argument("--out", default="det_labels")
    a = ap.parse_args()
    export(Path(a.out)) if a.cmd == "export" else import_(Path(a.export_json), Path(a.out))
