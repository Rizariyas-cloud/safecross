"""Detector wrappers: a fine-tuned YOLO model and zero-shot YOLO-World.

Both return the same format: a list of dicts
    {"box": (x1, y1, x2, y2), "cls": "wheelchair", "conf": 0.87}
so the rest of the project does not care which model produced them.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def pick_device(preferred: str | None = None) -> str:
    """'auto' -> the best available: NVIDIA GPU ('0'), Apple Silicon GPU ('mps') or 'cpu'."""
    if preferred and preferred != "auto":
        return preferred
    try:
        import torch
        if torch.cuda.is_available():
            return "0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def canonical_class(name: str) -> str:
    """Map dataset- or model-specific class names onto SafeCross names, e.g.
    'walking_cane' -> 'cane', 'person_wheelchair' -> 'wheelchair', 'person' -> 'pedestrian'."""
    n = name.lower().replace("-", "_").replace(" ", "_")
    if "wheel" in n:
        return "wheelchair"
    if "rollator" in n or "walker" in n or "walking_frame" in n:
        return "rollator"
    if "crutch" in n:
        return "crutches"
    if "cane" in n or "stick" in n:
        return "cane"
    if "pedestrian" in n or n in ("person", "people", "human"):
        return "pedestrian"
    return n


class Detector:
    def __init__(self, kind: str, weights: str, class_names: list[str] | None = None,
                 prompts: dict[str, str] | None = None, conf: float = 0.25, imgsz: int = 640,
                 device: str | None = None):
        """
        kind     : "yolo" (fine-tuned or pre-trained) or "yolo_world" (zero-shot with text prompts)
        weights  : path or name of the .pt file, e.g. weights/icg_yolo11s_seed0.pt or yolov8s-worldv2.pt
        prompts  : YOLO-World only. {our_class_name: "text description"}
        """
        from ultralytics import YOLO, YOLOWorld   # imported here so the rest of the project works without it
        self.kind, self.conf, self.imgsz, self.device = kind, conf, imgsz, pick_device(device)
        if kind == "yolo_world":
            prompts = prompts or {"pedestrian": "a pedestrian"}
            self.class_names = list(prompts.keys())
            self.model = YOLOWorld(weights)
            self.model.set_classes(list(prompts.values()))
        else:
            self.model = YOLO(weights)
            names = self.model.names
            self.class_names = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
            if class_names:
                self.class_names = class_names

    def predict(self, img_bgr: np.ndarray) -> list[dict]:
        r = self.model.predict(img_bgr, conf=self.conf, imgsz=self.imgsz, verbose=False, device=self.device)[0]
        out = []
        if r.boxes is None:
            return out
        for b, c, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            ci = int(c)
            name = self.class_names[ci] if ci < len(self.class_names) else str(ci)
            out.append({"box": tuple(float(v) for v in b), "cls": canonical_class(name), "conf": float(s)})
        return out


def load_yolo_labels(label_path: str | Path, img_w: int, img_h: int, class_names: list[str]) -> list[dict]:
    """Read a YOLO-format label file (class cx cy w h, normalised) into pixel boxes."""
    out = []
    p = Path(label_path)
    if not p.exists():
        return out
    for line in p.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        c, cx, cy, w, h = int(parts[0]), *map(float, parts[1:5])
        x1, y1 = (cx - w / 2) * img_w, (cy - h / 2) * img_h
        x2, y2 = (cx + w / 2) * img_w, (cy + h / 2) * img_h
        name = class_names[c] if c < len(class_names) else str(c)
        out.append({"box": (x1, y1, x2, y2), "cls": canonical_class(name)})
    return out


def draw_detections(img_bgr: np.ndarray, dets: list[dict], missed: list[dict] | None = None,
                    mobility_classes=("wheelchair", "rollator", "crutches", "cane")) -> np.ndarray:
    """Draw boxes: burgundy for mobility-aid classes, green for pedestrians, dashed red for missed people."""
    import cv2
    out = img_bgr.copy()
    for d in dets:
        x1, y1, x2, y2 = [int(v) for v in d["box"]]
        col = (44, 29, 140) if any(m in d["cls"] for m in mobility_classes) else (78, 143, 47)
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        label = f'{d["cls"]} {d.get("conf", 1):.2f}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), col, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    for d in missed or []:
        x1, y1, x2, y2 = [int(v) for v in d["box"]]
        for x in range(x1, x2, 12):
            cv2.line(out, (x, y1), (min(x + 6, x2), y1), (40, 40, 220), 2)
            cv2.line(out, (x, y2), (min(x + 6, x2), y2), (40, 40, 220), 2)
        for y in range(y1, y2, 12):
            cv2.line(out, (x1, y), (x1, min(y + 6, y2)), (40, 40, 220), 2)
            cv2.line(out, (x2, y), (x2, min(y + 6, y2)), (40, 40, 220), 2)
        cv2.putText(out, "MISSED " + d["cls"], (x1, y2 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 220), 2)
    return out
