"""STEP 28 (demo) - put a REAL frame with OUR model's detections on the dashboard's Camera monitoring card.

Picks a frame from PMMA video 3 (the camera position never seen in training) that contains mobility-aid users,
runs our fine-tuned YOLO11s on it, blurs every face, draws the boxes, and saves:
    app/web/camera.jpg    the frame with our detections (faces blurred)
    app/web/camera.json   the real counts per group, shown on the dashboard

    python scripts/28_demo_camera_frame.py                 # picks a good frame automatically
    python scripts/28_demo_camera_frame.py --pick 3        # try the 3rd-best frame instead
    python scripts/28_demo_camera_frame.py --image path/to/frame.jpg

Then run the dashboard with:  python app/run_demo.py
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
GROUP = {"pedestrian": "pedestrian", "wheelchair": "wheelchair", "rollator": "walker", "cane": "cane"}
COL = {"pedestrian": (245, 245, 245), "wheelchair": (190, 184, 242), "walker": (190, 184, 242), "cane": (190, 184, 242)}  # BGR


def best_frames(img_dir: Path, lab_dir: Path, names: dict):
    aid_ids = {i for i, n in names.items() if n != "pedestrian"}
    scored = []
    for lab in sorted(lab_dir.glob("*.txt")):
        rows = [l.split() for l in lab.read_text().splitlines() if l.strip()]
        if not rows:
            continue
        aids = sum(int(r[0]) in aid_ids for r in rows)
        kinds = len({int(r[0]) for r in rows if int(r[0]) in aid_ids})
        if aids:
            scored.append((kinds, aids, min(len(rows), 12), lab.stem))
    scored.sort(reverse=True)
    return [img_dir / f"{s[3]}.jpg" for s in scored]


def blur_faces(img, boxes):
    h, w = img.shape[:2]
    for x1, y1, x2, y2 in boxes:
        bw, bh = x2 - x1, y2 - y1
        fx1, fx2 = int(max(0, x1 + 0.1 * bw)), int(min(w, x2 - 0.1 * bw))
        fy1, fy2 = int(max(0, y1)), int(min(h, y1 + 0.24 * bh))
        if fx2 - fx1 < 4 or fy2 - fy1 < 4:
            continue
        roi = img[fy1:fy2, fx1:fx2]
        small = cv2.resize(roi, (max(1, roi.shape[1] // 8), max(1, roi.shape[0] // 8)), interpolation=cv2.INTER_LINEAR)
        img[fy1:fy2, fx1:fx2] = cv2.GaussianBlur(cv2.resize(small, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST), (0, 0), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/pmma_v2_yolo11s_seed0.pt")
    ap.add_argument("--data", default="data/pmma_v2")
    ap.add_argument("--pick", type=int, default=1, help="1 = best frame, 2 = second best, ...")
    ap.add_argument("--image", default=None, help="use this image instead of picking one")
    ap.add_argument("--conf", type=float, default=0.35)
    args = ap.parse_args()

    from ultralytics import YOLO
    w = ROOT / args.weights
    if not w.exists():
        sys.exit(f"Weights not found: {w}")
    model = YOLO(str(w))
    names = {int(k): v for k, v in model.names.items()}

    if args.image:
        img_path = Path(args.image)
    else:
        d = ROOT / args.data
        frames = best_frames(d / "images" / "test_new", d / "labels" / "test_new", names)
        if not frames:
            sys.exit(f"No labelled test_new frames with mobility-aid users found in {d}")
        img_path = frames[min(args.pick, len(frames)) - 1]
    img = cv2.imread(str(img_path))
    if img is None:
        sys.exit(f"Could not read {img_path}")
    print(f"Frame: {img_path.name}  ({img.shape[1]} x {img.shape[0]})")

    r = model.predict(img, imgsz=1280, conf=args.conf, verbose=False)[0]
    dets = []
    for b, c, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
        dets.append((b.astype(int).tolist(), GROUP.get(names[int(c)], names[int(c)]), float(s)))

    blur_faces(img, [d[0] for d in dets])
    for (x1, y1, x2, y2), g, s in dets:
        col = COL.get(g, (245, 245, 245))
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2, cv2.LINE_AA)
        lab = f"{g} {s:.2f}"
        (tw, th), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(img, (x1 - 1, y1 - th - 8), (x1 + tw + 8, y1), col, -1)
        cv2.putText(img, lab, (x1 + 4, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 28, 27), 1, cv2.LINE_AA)

    out = ROOT / "app" / "web"
    out.mkdir(parents=True, exist_ok=True)
    scale = 1600 / img.shape[1]
    if scale < 1:
        img = cv2.resize(img, (1600, int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out / "camera.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    counts = {g: sum(d[1] == g for d in dets) for g in ("pedestrian", "wheelchair", "walker", "cane")}
    info = {"frame": img_path.name, "video": "PMMA video 3 (camera position never seen in training)",
            "model": Path(args.weights).stem, "total": len(dets), "counts": counts,
            "aids": counts["wheelchair"] + counts["walker"] + counts["cane"]}
    (out / "camera.json").write_text(json.dumps(info, indent=1))
    print(f"Detections: {counts}")
    print(f"Saved {out / 'camera.jpg'} and camera.json. Faces are blurred.")
    print("Now run:  python app/run_demo.py   (if the frame is not good, try --pick 2, 3, ...)")


if __name__ == "__main__":
    main()
