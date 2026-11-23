"""STEP 6b - Balanced visual questions on REAL dataset images (needs your trained YOLO weights).

Design (one question = one person):
  * take a test image, clean or degraded (darkness / blur / occlusion, mild and medium),
  * run the detector on it,
  * mark ONE mobility-aid user from the ground-truth labels with a yellow box,
  * ask: "Did the detector report a person at the marked location?"
The truth is "yes" if a detection overlaps that person (IoU >= vision.iou_match, any person class),
otherwise "no". Questions are sampled so that half the answers are "yes" and half are "no".

Why one person per question: in a crowded frame almost every image has at least one missed
person, so a whole-image question would be "yes" nearly every time and could not test anything.

Only the picture shows WHERE the marked person is, and only the detection list shows WHAT the
detector reported, so the question needs both ("image_log"). With one of them the correct answer
is "cannot_determine".

    python scripts/06b_make_real_visual_questions.py --weights weights/pmma_yolo11s_seed0.pt --per-answer 60

The questions REPLACE any older visual_real questions in results/questions/questions.json.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from safecross.config import load_config, resolve
from safecross.vision import degrade
from safecross.vision.dataset import load_dataset_yaml, list_images
from safecross.vision.detect import Detector, load_yolo_labels, canonical_class
from safecross.vision.evaluate import match

CONDITIONS = [("clean", 0), ("darkness", 1), ("darkness", 2), ("blur", 1), ("blur", 2), ("occlusion", 1)]
OUT_WIDTH = 1280


def main():
    cfg = load_config()
    v = cfg["vision"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--per-answer", type=int, default=60, help="questions with answer 'yes' and with 'no' (each)")
    ap.add_argument("--data", default=None, help="dataset .yaml (default: vision.dataset_yaml in config.yaml)")
    ap.add_argument("--device", default=None, help="auto (default), mps (Mac GPU), 0 (NVIDIA GPU) or cpu")
    args = ap.parse_args()
    from safecross.vision.detect import pick_device
    from safecross.vision.dataset import dataset_yaml_path
    args.device = pick_device(args.device or v.get("device", "auto"))
    print(f"Using device: {args.device}")

    yml = dataset_yaml_path(cfg, args.data)
    d = load_dataset_yaml(yml)
    pairs = list_images(d, yml, "test")
    aid = set(v["mobility_aid_classes"])
    person_classes = {canonical_class(n) for n in d["names"]} | {"pedestrian"}
    rng = np.random.default_rng(cfg["seed"])
    det = Detector("yolo", args.weights, conf=v["conf"], imgsz=v["imgsz"], device=args.device)

    qdir = resolve(cfg["paths"]["results_dir"]) / "questions"
    img_dir = qdir / "real_images"
    img_dir.mkdir(parents=True, exist_ok=True)
    for old in img_dir.glob("*.jpg"):
        old.unlink()

    # 1. candidates: every mobility-aid user in every image x condition, with detected yes/no
    cands = {"yes": [], "no": []}
    for n, (img_path, lab) in enumerate(pairs):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        gts = load_yolo_labels(lab, w, h, d["names"])
        if not any(g["cls"] in aid for g in gts):
            continue
        for kind, sev in CONDITIONS:
            dimg = degrade.apply(img, kind, sev, seed=n * 10 + sev, boxes=[g["box"] for g in gts])
            dets = [x for x in det.predict(dimg) if x["cls"] in person_classes]
            found, _, _ = match(dets, gts, v["iou_match"], class_aware=False)
            found_ids = {id(g) for _, g in found}
            for j, g in enumerate(gts):
                if g["cls"] not in aid:
                    continue
                ans = "yes" if id(g) in found_ids else "no"
                cands[ans].append({"img_path": img_path, "lab": lab, "kind": kind, "sev": sev, "seed": n * 10 + sev,
                                   "gt_index": j, "cls": g["cls"], "box": g["box"], "dets": dets})
        if (n + 1) % 25 == 0:
            print(f"  {n + 1}/{len(pairs)} images checked")
    print(f"Candidates: {len(cands['yes'])} detected, {len(cands['no'])} missed mobility-aid users")

    # 2. balanced sample
    k = min(args.per_answer, len(cands["yes"]), len(cands["no"]))
    if k == 0:
        sys.exit("Not enough detected AND missed people to build a balanced set.")
    chosen = []
    for ans in ("yes", "no"):
        idx = rng.choice(len(cands[ans]), size=k, replace=False)
        chosen += [(ans, cands[ans][i]) for i in sorted(idx)]

    # 3. write images (resized, target marked) and questions
    items = []
    cache = {}
    for ans, c in chosen:
        key = (str(c["img_path"]), c["kind"], c["sev"])
        if key not in cache:
            img = cv2.imread(str(c["img_path"]))
            h, w = img.shape[:2]
            gts = load_yolo_labels(c["lab"], w, h, d["names"])
            cache[key] = degrade.apply(img, c["kind"], c["sev"], seed=c["seed"], boxes=[g["box"] for g in gts])
        dimg = cache[key].copy()
        h, w = dimg.shape[:2]
        s = OUT_WIDTH / w
        dimg = cv2.resize(dimg, (OUT_WIDTH, int(round(h * s))))
        x1, y1, x2, y2 = [int(round(t * s)) for t in c["box"]]
        pad = 6
        cv2.rectangle(dimg, (x1 - pad, y1 - pad), (x2 + pad, y2 + pad), (0, 230, 255), 4)   # yellow (BGR)
        name = f"{Path(c['img_path']).stem}_{c['kind']}{c['sev']}_p{c['gt_index']}"
        out_img = img_dir / f"{name}.jpg"
        cv2.imwrite(str(out_img), dimg, [cv2.IMWRITE_JPEG_QUALITY, 90])
        listing = [f"image size {dimg.shape[1]} x {dimg.shape[0]} pixels; boxes are [x1, y1, x2, y2]"]
        listing += [f"{x['cls']} (confidence {x['conf']:.2f}) at box "
                    f"[{int(x['box'][0] * s)}, {int(x['box'][1] * s)}, {int(x['box'][2] * s)}, {int(x['box'][3] * s)}]"
                    for x in c["dets"]] or ["no detections"]
        items.append({
            "qid": f"realbox_{name}", "qtype": "visual_real", "episode_id": None, "controller": None,
            "condition_image": c["kind"], "severity": c["sev"], "target_class": c["cls"],
            "question": "In the picture, one person who uses a mobility aid is marked with a yellow box. "
                        "Did the detector report a person at the marked location?",
            "fields": {"answer": ["yes", "no", "cannot_determine"]},
            "truth": {"answer": ans},
            "log": [], "detections": listing, "image": str(out_img)})

    qfile = qdir / "questions.json"
    all_items = json.loads(qfile.read_text()) if qfile.exists() else []
    all_items = [i for i in all_items if i["qtype"] != "visual_real"] + items
    qfile.write_text(json.dumps(all_items, indent=1, default=float))
    from collections import Counter
    print(f"Wrote {len(items)} balanced visual_real questions ({k} yes / {k} no).")
    print("By image condition:", dict(Counter(f"{i['condition_image']}{i['severity']}" for i in items)))
    print(f"Total questions now: {len(all_items)}")


if __name__ == "__main__":
    main()
