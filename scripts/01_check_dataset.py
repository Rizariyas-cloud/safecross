"""STEP 1 - Check the ICG dataset after you download and unzip it.

Run from the project folder:
    python scripts/01_check_dataset.py

What it does:
  * finds the dataset .yaml file inside data/icg
  * prints the class names and how many images / boxes each split has
  * writes configs/icg_dataset.yaml with the correct absolute path (used for training)
  * warns you if the class names don't match the prompts in configs/config.yaml
"""
import sys
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from safecross.config import load_config, resolve, ROOT
from safecross.vision.dataset import find_dataset_yaml, load_dataset_yaml, list_images
from safecross.vision.detect import canonical_class


def main():
    cfg = load_config()
    icg = resolve(cfg["paths"]["icg_dir"])
    if not icg.exists():
        sys.exit(f"Folder not found: {icg}\nDownload mobility.zip from the TU Graz repository and unzip it there.")
    y = find_dataset_yaml(icg)
    if y is None:
        sys.exit(f"No dataset .yaml found inside {icg}. Check that the zip was fully extracted.")
    d = load_dataset_yaml(y)
    print(f"Dataset file : {y}")
    print(f"Classes      : {d['names']}")
    print(f"As SafeCross : {[canonical_class(n) for n in d['names']]}")

    for split in ("train", "val", "test"):
        pairs = list_images(d, y, split)
        boxes = Counter()
        missing = 0
        for _, lab in pairs:
            if not lab.exists():
                missing += 1
                continue
            for line in lab.read_text().splitlines():
                if line.strip():
                    c = int(line.split()[0])
                    boxes[d["names"][c] if c < len(d["names"]) else c] += 1
        print(f"\n[{split}] images: {len(pairs)}   label files missing: {missing}")
        for k, v in boxes.most_common():
            print(f"    {k:>20}: {v}")

    # write a copy with an absolute path so ultralytics finds the images from anywhere
    out = dict(d)
    out["path"] = str((Path(d.get("path") or y.parent) if Path(d.get("path") or y.parent).is_absolute()
                       else (y.parent / (d.get("path") or ".")).resolve()))
    out["names"] = {i: n for i, n in enumerate(d["names"])}
    dst = ROOT / "configs" / "icg_dataset.yaml"
    with open(dst, "w", encoding="utf-8") as f:
        yaml.safe_dump(out, f, sort_keys=False)
    print(f"\nWrote {dst}")

    canon = {canonical_class(n) for n in d["names"]}
    missing_prompts = canon - set(cfg["vision"]["world_prompts"])
    if missing_prompts:
        print(f"\nWARNING: add YOLO-World prompts for {missing_prompts} in configs/config.yaml (vision.world_prompts)")
    else:
        print("All classes have YOLO-World prompts. Good.")


if __name__ == "__main__":
    main()
