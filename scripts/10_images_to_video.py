"""Make a short video from a folder of consecutive images (for example ICG or PMMA frames),
so you can show REAL mobility-aid users on the Live Monitoring page.

    python scripts/10_images_to_video.py data/icg/mobility/images/test --fps 10 --max 300

Output: results/demo_video.mp4  (upload it on the Live Monitoring page)
Tip: pick a folder where the images are consecutive frames of the same camera.
"""
import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from safecross.config import load_config, resolve


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--fps", type=float, default=10)
    ap.add_argument("--max", type=int, default=300, help="maximum number of images")
    ap.add_argument("--start", type=int, default=0, help="skip this many images first")
    args = ap.parse_args()
    imgs = sorted(p for p in Path(args.folder).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    imgs = imgs[args.start:args.start + args.max]
    if not imgs:
        sys.exit("No images found in that folder.")
    first = cv2.imread(str(imgs[0]))
    h, w = first.shape[:2]
    out = resolve(cfg["paths"]["results_dir"]) / "demo_video.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
    for p in imgs:
        im = cv2.imread(str(p))
        if im is None:
            continue
        if im.shape[:2] != (h, w):
            im = cv2.resize(im, (w, h))
        vw.write(im)
    vw.release()
    print(f"Wrote {len(imgs)} frames -> {out}")


if __name__ == "__main__":
    main()
