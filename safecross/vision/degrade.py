"""Image corruptions for Experiment A (visual robustness).

Three conditions, each at severity 1 (mild), 2 (moderate) or 3 (strong):
  darkness  - lower brightness, gamma curve and sensor noise, like a dim night camera
  blur      - motion blur along a random direction, like a fast shutter problem or vibration
  occlusion - grey blocks covering part of each person (or random patches if no boxes are given)

All functions are deterministic for a given seed, so the same degraded images
can be recreated exactly.
"""
from __future__ import annotations

import cv2
import numpy as np

DARKNESS = {1: (0.55, 1.4, 4), 2: (0.35, 1.8, 7), 3: (0.2, 2.2, 10)}     # (brightness, gamma, noise sd)
BLUR = {1: 9, 2: 17, 3: 27}                                               # kernel length in pixels
OCCLUSION = {1: 0.25, 2: 0.45, 3: 0.65}                                   # share of each box covered


def darkness(img: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    scale, gamma, noise = DARKNESS[severity]
    x = img.astype(np.float32) / 255.0
    x = np.power(x, gamma) * scale
    x = x * 255.0 + rng.normal(0, noise, x.shape)
    return np.clip(x, 0, 255).astype(np.uint8)


def motion_blur(img: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    k = BLUR[severity]
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0
    angle = float(rng.uniform(0, 180))
    rot = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, rot, (k, k))
    kernel /= max(kernel.sum(), 1e-6)
    return cv2.filter2D(img, -1, kernel)


def occlusion(img: np.ndarray, severity: int, rng: np.random.Generator,
              boxes: list[tuple[float, float, float, float]] | None = None) -> np.ndarray:
    """boxes are (x1, y1, x2, y2) in pixels. Each box gets a grey block over part of it."""
    out = img.copy()
    share = OCCLUSION[severity]
    h, w = img.shape[:2]
    if not boxes:                                          # random patches over the image
        n = 3 * severity
        boxes = []
        for _ in range(n):
            bw, bh = rng.uniform(0.05, 0.15) * w, rng.uniform(0.1, 0.3) * h
            x1, y1 = rng.uniform(0, w - bw), rng.uniform(0, h - bh)
            boxes.append((x1, y1, x1 + bw, y1 + bh))
        share = 1.0
    for (x1, y1, x2, y2) in boxes:
        bw, bh = x2 - x1, y2 - y1
        if bw <= 1 or bh <= 1:
            continue
        if rng.random() < 0.5:                             # vertical strip (e.g. a pole or another person)
            cw = bw * share
            cx = rng.uniform(x1, x2 - cw)
            rect = (cx, y1, cx + cw, y2)
        else:                                              # horizontal band (e.g. a parked car)
            ch = bh * share
            cy = rng.uniform(y1, y2 - ch)
            rect = (x1, cy, x2, cy + ch)
        a, b, c, d = [int(round(v)) for v in rect]
        grey = int(rng.integers(70, 140))
        cv2.rectangle(out, (max(a, 0), max(b, 0)), (min(c, w - 1), min(d, h - 1)), (grey, grey, grey), -1)
    return out


def apply(img: np.ndarray, kind: str, severity: int, seed: int = 0, boxes=None) -> np.ndarray:
    """Apply one corruption. kind = 'clean' | 'darkness' | 'blur' | 'occlusion'."""
    if kind == "clean" or severity == 0:
        return img.copy()
    rng = np.random.default_rng(seed)
    if kind == "darkness":
        return darkness(img, severity, rng)
    if kind == "blur":
        return motion_blur(img, severity, rng)
    if kind == "occlusion":
        return occlusion(img, severity, rng, boxes)
    raise ValueError(f"unknown corruption: {kind}")
