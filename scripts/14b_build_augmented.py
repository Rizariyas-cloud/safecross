"""STEP 14b - robustness training data: every training image + one darkened OR blurred copy.

Why: the model trained on clean video-1 frames finds 75% of mobility-aid users at a new camera position,
but only ~50% in medium darkness and 18% in medium blur (Experiment A on video 3). This builds a training
set that also contains degraded images, to test whether training with them makes the camera more robust.

Fair-test design (say this in the viva):
  * the copies use RANDOM, CONTINUOUS settings (not the exact 3 test levels of Experiment A);
  * only DARKNESS and BLUR are used for training. OCCLUSION is held out, so Experiment A also shows whether
    robustness transfers to a degradation the model never saw during training;
  * validation and test images stay exactly the same (clean video 1 end / video 2 / video 3).

    python scripts/14b_build_augmented.py
    python scripts/02_train_yolo.py --data configs/pmma_v2_aug.yaml --model yolo11s.pt --epochs 15 --imgsz 1280 --batch 8 --seed 0
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from safecross.config import ROOT, resolve


def dark(img, rng):
    scale, gamma, noise = rng.uniform(0.2, 0.6), rng.uniform(1.3, 2.3), rng.uniform(3, 10)
    x = np.power(img.astype(np.float32) / 255.0, gamma) * scale * 255.0 + rng.normal(0, noise, img.shape)
    return np.clip(x, 0, 255).astype(np.uint8), f"dark s={scale:.2f} g={gamma:.2f} n={noise:.1f}"


def blur(img, rng):
    k = int(rng.integers(4, 14)) * 2 + 1                     # 9 ... 27 px, any odd length
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0
    rot = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), float(rng.uniform(0, 180)), 1.0)
    kernel = cv2.warpAffine(kernel, rot, (k, k))
    kernel /= max(kernel.sum(), 1e-6)
    return cv2.filter2D(img, -1, kernel), f"blur k={k}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/pmma_v2")
    ap.add_argument("--out", default="data/pmma_v2_aug")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    src, out = resolve(args.src), resolve(args.out)
    tr_img, tr_lab = src / "images" / "train", src / "labels" / "train"
    if not tr_img.exists():
        sys.exit(f"{tr_img} not found - run scripts/14_build_pmma_v2.py first.")
    shutil.rmtree(out, ignore_errors=True)
    (out / "images" / "train").mkdir(parents=True)
    (out / "labels" / "train").mkdir(parents=True)
    rng = np.random.default_rng(args.seed)
    imgs = sorted(tr_img.glob("*.jpg"))
    log = []
    for k, im in enumerate(imgs):
        lab = tr_lab / f"{im.stem}.txt"
        os.symlink(im.resolve(), out / "images" / "train" / im.name)        # the clean original (no copy)
        shutil.copy(lab, out / "labels" / "train" / lab.name)
        img = cv2.imread(str(im))
        aug, how = (dark if rng.random() < 0.5 else blur)(img, rng)
        cv2.imwrite(str(out / "images" / "train" / f"{im.stem}_aug.jpg"), aug, [cv2.IMWRITE_JPEG_QUALITY, 92])
        shutil.copy(lab, out / "labels" / "train" / f"{im.stem}_aug.txt")
        log.append(f"{im.stem}_aug.jpg,{how}")
        if (k + 1) % 250 == 0:
            print(f"  {k + 1}/{len(imgs)}")
    (out / "augmentations.csv").write_text("image,setting\n" + "\n".join(log) + "\n")
    s = src.resolve()
    yml = ROOT / "configs" / "pmma_v2_aug.yaml"
    yml.write_text("\n".join([
        "# Made by scripts/14b_build_augmented.py: train = video 1 clean + one darkened/blurred copy each",
        f"path: {out.resolve()}",
        "train: images/train",
        f"val: {s / 'images' / 'val'}",
        f"test: {s / 'images' / 'test_new'}",
        f"test_same: {s / 'images' / 'test_same'}",
        f"test_new: {s / 'images' / 'test_new'}",
        "names:", "  0: pedestrian", "  1: wheelchair", "  2: rollator", "  3: cane"]) + "\n")
    n_dark = sum("dark" in l for l in log)
    print(f"\n{len(imgs)} clean + {len(imgs)} degraded training images ({n_dark} dark, {len(imgs) - n_dark} blurred)")
    print(f"Validation/test images unchanged (taken from {s}). Wrote {yml}")


if __name__ == "__main__":
    main()
