"""Run YOLO + ByteTrack on a video and return annotated frames for the Live Monitoring replay."""
from __future__ import annotations

import cv2
import numpy as np

from .detect import canonical_class, pick_device

MOBILITY = ("wheelchair", "rollator", "crutches", "cane")
VEHICLES = ("car", "bus", "truck", "motorcycle")


def _one_eye(img):
    """PMMA-style side-by-side stereo video (width >= 3 x height): keep the left camera, as in training."""
    h, w = img.shape[:2]
    return img[:, : w // 2] if w >= 3 * h else img


def _frames(video_path: str):
    cap = cv2.VideoCapture(video_path)
    while True:
        ok, img = cap.read()
        if not ok:
            break
        yield _one_eye(img)
    cap.release()


def track_video(video_path: str, weights: str, device: str | None = None, conf: float = 0.25,
                max_seconds: float = 20.0, out_fps: float = 4.0, width: int = 720, imgsz: int | None = None):
    """Returns (frames_rgb, infos, rows).
    frames_rgb : list of annotated RGB frames (sampled at about out_fps)
    infos      : per sampled frame {"t", "people", "mobility_aid", "vehicles", "track_ids"}
    rows       : one row per track: id, class, first/last time seen, frames seen
    """
    from ultralytics import YOLO
    model = YOLO(weights)
    names = model.names
    if imgsz is None:                                   # PMMA-trained models were trained at 1280 px
        imgsz = 1280 if "pmma" in str(weights).lower() else 640
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    cap.release()
    step = max(1, int(round(fps / out_fps)))
    max_frames = int(max_seconds * fps)
    frames, infos, tracks = [], [], {}
    dev = pick_device(device)
    for i, frame in enumerate(_frames(video_path)):
        if i >= max_frames:
            break
        r = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=conf, imgsz=imgsz, device=dev,
                        verbose=False)[0]
        t = i / fps
        boxes = r.boxes
        dets, vehs = [], []
        if boxes is not None and len(boxes):
            ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(boxes)
            for b, c, s, tid in zip(boxes.xyxy.cpu().numpy(), boxes.cls.cpu().numpy(), boxes.conf.cpu().numpy(), ids):
                if names[int(c)].lower() in VEHICLES:
                    vehs.append(b)
                    continue
                cls = canonical_class(names[int(c)])
                if cls not in ("pedestrian",) + MOBILITY:
                    continue
                dets.append((b, cls, float(s), tid))
                if tid is not None:
                    tr = tracks.setdefault(tid, {"track": tid, "class": cls, "first_s": round(t, 1), "frames": 0})
                    tr["last_s"] = round(t, 1)
                    tr["frames"] += 1
        if i % step:
            continue
        img = r.orig_img.copy()
        for b in vehs:
            x1, y1, x2, y2 = [int(v) for v in b]
            cv2.rectangle(img, (x1, y1), (x2, y2), (150, 150, 150), 2)
        for b, cls, s, tid in dets:
            x1, y1, x2, y2 = [int(v) for v in b]
            col = (45, 30, 139) if cls in MOBILITY else (80, 170, 60)          # BGR: maroon / green
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 3)
            label = f"#{tid} {cls}" if tid is not None else cls
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), col, -1)
            cv2.putText(img, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        h, w = img.shape[:2]
        if w > width:
            img = cv2.resize(img, (width, int(h * width / w)))
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        infos.append({"t": round(t, 1), "people": len(dets),
                      "mobility_aid": sum(1 for d in dets if d[1] in MOBILITY), "vehicles": len(vehs),
                      "track_ids": [d[3] for d in dets if d[3] is not None]})
    rows = sorted(tracks.values(), key=lambda r: r["track"])
    return frames, infos, rows
