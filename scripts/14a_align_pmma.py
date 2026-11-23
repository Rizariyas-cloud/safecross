"""STEP 14a - find which annotation number belongs to which video frame (automatic alignment).

Why: the mp4 files have more frames than the annotations (video 1: 12,628 frames vs 10,700 labels;
video 3: 17,294 vs 14,638), so "frame N = label N" is not safe. This script does not guess:

  1. it picks ~40 frames spread over each video and detects people with off-the-shelf YOLO11s;
  2. for every picked frame it scores EVERY annotated frame of that video
     (score = soft F1: summed overlap (IoU) of labelled boxes with detected people, peaks at the right frame);
  3. it prints, per picked frame, the best-matching annotation number and the difference
     (annotation number - frame number). A constant difference = a simple offset.

Self-test: video 2 (zip images) has a KNOWN answer (annotation 20000 = frame 1800, difference 18200).
If the script finds 18200 for video 2, trust its answer for videos 1 and 3.

    python scripts/14a_align_pmma.py            # ~5-10 minutes (reads the videos once)
Output: printed table + results/pmma_alignment.csv
"""
import argparse
import json
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from safecross.config import load_config, resolve, ROOT

FRAME_RE = re.compile(r"frame_(\d+)\.(png|jpg|jpeg)$", re.I)
SKIP = {"PushWheel1", "PushWheelEmpty"}          # group boxes / empty chairs: not one person
RANGES = {"v1": (0, 19999), "v2": (20000, 29999), "v3": (30000, 49999)}


def load_boxes(pmma: Path):
    """All person boxes as flat arrays: ids (annotation number), boxes (x1,y1,x2,y2 in original pixels)."""
    ids, boxes, size = [], [], {}
    for name in ("train.json", "val.json", "test.json"):
        d = json.load(open(pmma / name))
        cats = {c["id"]: c["name"] for c in d["categories"]}
        num = {im["id"]: int(Path(im["file_name"]).stem) for im in d["images"]}
        for im in d["images"]:
            size[num[im["id"]]] = (im["width"], im["height"])
        for a in d["annotations"]:
            if cats[a["category_id"]] in SKIP:
                continue
            x, y, w, h = a["bbox"]
            ids.append(num[a["image_id"]])
            boxes.append((x, y, x + w, y + h))
    return np.array(ids), np.array(boxes, float), size


def scores_for(det: np.ndarray, ids: np.ndarray, boxes: np.ndarray, n_per_id: np.ndarray, max_id: int):
    """F1 between one frame's detections and every annotated frame at once."""
    if len(det) == 0:
        return np.zeros(max_id + 1)
    x1 = np.maximum(det[:, None, 0], boxes[None, :, 0]); y1 = np.maximum(det[:, None, 1], boxes[None, :, 1])
    x2 = np.minimum(det[:, None, 2], boxes[None, :, 2]); y2 = np.minimum(det[:, None, 3], boxes[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a_det = ((det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1]))[:, None]
    a_box = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))[None, :]
    iou = inter / np.maximum(a_det + a_box - inter, 1e-9)
    best = iou.max(axis=0)                                            # per labelled box
    matched = np.where(best >= 0.3, best, 0.0)                        # continuous: peaks at the exact frame
    m_per_id = np.bincount(ids, weights=matched, minlength=max_id + 1)
    return 2 * m_per_id / np.maximum(n_per_id + len(det), 1)


def sample_video(paths, n):
    total = 0
    for p in paths:
        cap = cv2.VideoCapture(str(p)); c = 0
        while cap.grab():
            c += 1
        cap.release(); total += c
    picks = set(np.linspace(total * 0.02, total * 0.98, n).astype(int).tolist())
    fno = 0
    for p in paths:
        cap = cv2.VideoCapture(str(p))
        while cap.grab():
            if fno in picks:
                ok, img = cap.retrieve()
                if ok:
                    yield fno, img
            fno += 1
        cap.release()


def sample_zip(zip_dir: Path, n):
    frames = {}
    for zp in sorted(zip_dir.glob("*.zip")):
        try:
            zf = zipfile.ZipFile(zp)
        except zipfile.BadZipFile:
            continue
        for m in zf.namelist():
            mm = FRAME_RE.search(m)
            if mm and "__MACOSX" not in m:
                frames[int(mm.group(1))] = (zp, m)
    keys = sorted(frames)
    for k in np.linspace(len(keys) * 0.02, len(keys) * 0.98, n).astype(int):
        fno = keys[k]; zp, m = frames[fno]
        img = cv2.imdecode(np.frombuffer(zipfile.ZipFile(zp).read(m), np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            yield fno, img


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--pmma", default="data/pmma")
    ap.add_argument("--samples", type=int, default=40)
    ap.add_argument("--weights", default="yolo11s.pt")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    pmma = resolve(args.pmma)
    from ultralytics import YOLO
    model = YOLO(str(ROOT / args.weights) if (ROOT / args.weights).exists() else args.weights)

    print("Loading annotations ...")
    ids_all, boxes_all, size = load_boxes(pmma)
    sources = {
        "v2": ("video 2 (zip) - SELF-TEST, expected difference 18200", lambda: sample_zip(pmma / "zips", args.samples)),
        "v1": ("video 1", lambda: sample_video([pmma / "videos" / "video_1.mp4"], args.samples)),
        "v3": ("video 3", lambda: sample_video([pmma / "videos" / "video_3_part1.mp4",
                                                 pmma / "videos" / "video_3_part2.mp4"], args.samples)),
    }
    rows = []
    for key, (title, gen) in sources.items():
        lo, hi = RANGES[key]
        sel = (ids_all >= lo) & (ids_all <= hi)
        ids, boxes = ids_all[sel], boxes_all[sel]
        if not len(ids):
            print(f"\n{title}: no annotations in range {lo}-{hi}"); continue
        n_per_id = np.bincount(ids, minlength=hi + 1).astype(float)
        print(f"\n== {title}")
        for fno, full in gen():
            ow, oh = size[int(ids[0])]
            fh, fw = full.shape[:2]
            # PMMA mp4 files are side-by-side stereo (left | right camera): try each half
            halves = {"left": full[:, :fw // 2], "right": full[:, fw // 2:]} if fw >= 2 * ow * 0.95 else {"full": full}
            line = f"   {fno:6d}"
            for eye, img in halves.items():
                h, w = img.shape[:2]
                r = model.predict(img, imgsz=1280, conf=0.3, classes=[0], verbose=False, device=args.device)[0]
                det = r.boxes.xyxy.cpu().numpy() * np.array([ow / w, oh / h, ow / w, oh / h])
                sc = scores_for(det, ids, boxes, n_per_id, hi)
                best = int(np.argmax(sc))
                rows.append(dict(video=key, eye=eye, frame=fno, best_label=best, score=round(float(sc[best]), 3),
                                 difference=best - fno, n_detected=len(det)))
                line += f" | {eye:5s} label {best:6d} score {sc[best]:4.2f} diff {best - fno:6d}"
            print(line)
        for eye in ("full", "left", "right"):
            good = [r["difference"] for r in rows if r["video"] == key and r["eye"] == eye and r["score"] >= 0.4]
            n_eye = sum(1 for r in rows if r["video"] == key and r["eye"] == eye)
            if n_eye:
                top = Counter(good).most_common(3)
                med = int(np.median(good)) if good else None
                print(f"   [{eye}] matched well: {len(good)} of {n_eye} | median difference: {med} | most common: {top}")

    import csv
    out = resolve(cfg["paths"]["results_dir"]) / "pmma_alignment.csv"
    with open(out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    print(f"\nSaved {out}. Send the whole printed output to Claude.")


if __name__ == "__main__":
    main()
