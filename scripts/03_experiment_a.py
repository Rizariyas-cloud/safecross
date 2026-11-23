"""STEP 3 - Experiment A: visual robustness (answers RQ1).

Runs each detector on clean and degraded test images and measures how
detection of mobility-aid users changes.

Two recall numbers per class:
    recall           found AND given the right class (e.g. 'wheelchair')
    found_as_person  found as any person class (what a signal controller needs to hold the phase).
                     Use this one for a COCO model such as yolo11s.pt, which only knows 'person'.

Examples:
    python scripts/03_experiment_a.py --yolo weights/icg_yolo11s_seed0.pt --world --limit 300
    python scripts/03_experiment_a.py --yolo weights/icg_yolo11s_seed0.pt --yolo weights/icg_yolo11s_seed1.pt

Outputs (in results/experiment_a/):
    per_image.csv        one row per image x condition x model (counts)
    summary.csv          recall / precision per class with 95% bootstrap CIs
    examples/            a few side-by-side pictures for the report
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from safecross.config import load_config, resolve, ROOT
from safecross.metrics import cluster_bootstrap
from safecross.vision import degrade
from safecross.vision.dataset import load_dataset_yaml, list_images
from safecross.vision.detect import Detector, load_yolo_labels, draw_detections
from safecross.vision.evaluate import image_counts


def block_of(image_name: str, block: int) -> str:
    """Cluster id for the bootstrap: video frames are grouped into blocks of `block` frame numbers
    (e.g. pmma_v2_02436.jpg -> 'pmma_v2_81'). Images without a frame number are their own cluster."""
    import re
    m = re.search(r"^(.*?)(\d+)\.\w+$", image_name)
    if not m or block <= 1:
        return image_name
    return f"{m.group(1)}{int(m.group(2)) // block}"


def main():
    cfg = load_config()
    v = cfg["vision"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo", action="append", default=[], help="fine-tuned YOLO weights (repeat for seeds)")
    ap.add_argument("--world", action="store_true", help="also evaluate zero-shot YOLO-World")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N images (0 = all)")
    ap.add_argument("--block-frames", type=int, default=30,
                    help="frames per bootstrap cluster: neighbouring video frames are nearly identical, so the "
                         "confidence interval resamples blocks of frames (30 = 2 s at 15 fps), not single images")
    ap.add_argument("--data", default=None, help="dataset .yaml (default: vision.dataset_yaml in config.yaml)")
    ap.add_argument("--out", default="experiment_a", help="results sub-folder (keeps earlier results safe)")
    ap.add_argument("--device", default=None, help="auto (default), mps (Mac GPU), 0 (NVIDIA GPU) or cpu")
    args = ap.parse_args()
    from safecross.vision.detect import pick_device
    args.device = pick_device(args.device or cfg["vision"].get("device", "auto"))
    print(f"Using device: {args.device}")

    from safecross.vision.dataset import dataset_yaml_path
    from safecross.vision.detect import canonical_class
    yml = dataset_yaml_path(cfg, args.data)
    if not yml.exists():
        sys.exit(f"{yml} not found. Run scripts/01 (ICG) or scripts/11 (PMMA) first.")
    print(f"Dataset: {yml.name}")
    d = load_dataset_yaml(yml)
    person_classes = {canonical_class(n) for n in d["names"]} | {"pedestrian"}
    pairs = list_images(d, yml, args.split)
    if args.limit:
        rng = np.random.default_rng(cfg["seed"])
        idx = sorted(rng.choice(len(pairs), size=min(args.limit, len(pairs)), replace=False))
        pairs = [pairs[i] for i in idx]
    print(f"{len(pairs)} images from split '{args.split}'")

    models = {}
    for w in args.yolo:
        models[Path(w).stem] = Detector("yolo", w, conf=v["conf"], imgsz=v["imgsz"], device=args.device)
    if args.world:
        models["yolo_world_zeroshot"] = Detector("yolo_world", v["yolo_world_model"], prompts=v["world_prompts"],
                                                 conf=v["conf"], imgsz=v["imgsz"], device=args.device)
    if not models:
        sys.exit("Give at least one --yolo weights file or --world")

    conditions = [("clean", 0)] + [(k, s) for k in v["degradations"] for s in v["severities"]]
    out_dir = resolve(cfg["paths"]["results_dir"]) / args.out
    (out_dir / "examples").mkdir(parents=True, exist_ok=True)
    rows = []
    for n, (img_path, lab_path) in enumerate(pairs):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        gts = load_yolo_labels(lab_path, w, h, d["names"])
        for kind, sev in conditions:
            seed = cfg["seed"] * 100000 + n * 10 + sev
            dimg = degrade.apply(img, kind, sev, seed=seed, boxes=[g["box"] for g in gts])
            for mname, det in models.items():
                dets = [x for x in det.predict(dimg) if x["cls"] in person_classes]   # ignore cars, bags, ...
                c = image_counts(dets, gts, v["iou_match"])
                rows.append({"image": str(img_path.name), "condition": kind, "severity": sev, "model": mname,
                             "counts": json.dumps(c)})
                if n < 4 and sev in (0, 3):
                    from safecross.vision.evaluate import match
                    _, missed, _ = match(dets, gts, v["iou_match"], class_aware=False)
                    vis = draw_detections(dimg, dets, missed)
                    cv2.imwrite(str(out_dir / "examples" / f"{n}_{mname}_{kind}{sev}.jpg"), vis)
        if (n + 1) % 25 == 0:
            print(f"  {n + 1}/{len(pairs)} images done")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_image.csv", index=False)

    # summary with clustered bootstrap (cluster = image)
    aid = v["mobility_aid_classes"]
    summ = []
    for (m, k, s), g in df.groupby(["model", "condition", "severity"]):
        recs = [dict(image=r.image, cluster=block_of(r.image, args.block_frames), **json.loads(r.counts))
                for r in g.itertuples()]
        classes = sorted({c for r in recs for c in r["per_class"]})
        for cname in classes + ["ALL_MOBILITY_AIDS"]:
            members = aid if cname == "ALL_MOBILITY_AIDS" else [cname]

            def recall(rs, members=members):
                tp = sum(r["per_class"].get(c, {}).get("tp", 0) for r in rs for c in members)
                fn = sum(r["per_class"].get(c, {}).get("fn", 0) for r in rs for c in members)
                return tp / (tp + fn) if tp + fn else float("nan")

            def precision(rs, members=members):
                tp = sum(r["per_class"].get(c, {}).get("tp", 0) for r in rs for c in members)
                fp = sum(r["per_class"].get(c, {}).get("fp", 0) for r in rs for c in members)
                return tp / (tp + fp) if tp + fp else float("nan")

            def recall_any(rs, members=members):
                tp = sum(r["per_class"].get(c, {}).get("tp_any", 0) for r in rs for c in members)
                fn = sum(r["per_class"].get(c, {}).get("fn_any", 0) for r in rs for c in members)
                return tp / (tp + fn) if tp + fn else float("nan")

            r_pt, r_lo, r_hi = cluster_bootstrap(recs, recall, "cluster", n_boot=1000)
            a_pt, a_lo, a_hi = cluster_bootstrap(recs, recall_any, "cluster", n_boot=1000)
            p_pt, p_lo, p_hi = cluster_bootstrap(recs, precision, "cluster", n_boot=1000)
            summ.append({"model": m, "condition": k, "severity": s, "class": cname,
                         "recall": r_pt, "recall_lo": r_lo, "recall_hi": r_hi,
                         "found_as_person": a_pt, "found_as_person_lo": a_lo, "found_as_person_hi": a_hi,
                         "precision": p_pt, "precision_lo": p_lo, "precision_hi": p_hi,
                         "misclassified_people": sum(r["misclassified"] for r in recs)})
    sdf = pd.DataFrame(summ)
    sdf.to_csv(out_dir / "summary.csv", index=False)
    n_clusters = len({block_of(p.name, args.block_frames) for p, _ in pairs})
    print(f"Confidence intervals resample {n_clusters} blocks of {args.block_frames} frames (not single images).")
    cols = ["model", "condition", "severity", "recall", "recall_lo", "recall_hi",
            "found_as_person", "found_as_person_lo", "found_as_person_hi"]
    print(sdf[sdf["class"] == "ALL_MOBILITY_AIDS"][cols].round(3).to_string(index=False))
    print("\nPer class, clean images (found_as_person):")
    print(sdf[(sdf["severity"] == 0)].pivot_table(index="class", columns="model", values="found_as_person").round(3))
    print(f"\nSaved results to {out_dir}")


if __name__ == "__main__":
    main()
