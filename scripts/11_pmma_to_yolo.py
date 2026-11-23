"""STEP 11 (backup dataset) - turn PMMA video-2 frames into a YOLO dataset for Experiments A / training.

Use this when the ICG dataset cannot be downloaded. It needs:
  * data/pmma/test.json                (COCO annotations, from Borealis)
  * data/pmma/zips/video_2_1.zip ...   (frame images, from Borealis; one zip is enough to start)

    python scripts/11_pmma_to_yolo.py                       # all frames in the zips -> test set only
    python scripts/11_pmma_to_yolo.py --every 5             # keep every 5th frame (frames 1/15 s apart look almost identical)
    python scripts/11_pmma_to_yolo.py --every 5 --train-share 0.6   # also make train/val sets for fine-tuning

How file names match (checked visually on frame_01800.png):
    annotation "020000.png"  ==  zip image "frame_01800.png"
    i.e. annotation number = 20000 + (frame number - 1800)

Classes are grouped into the SafeCross classes (see 'pmma_class_map' in configs/config.yaml):
    Ped -> pedestrian, WheelChair / PushWheel3 -> wheelchair, WalkerWalking / WalkerRest -> rollator,
    Cane -> cane, PushWheel2 (the person pushing) -> pedestrian, PushWheel1 (group box) and
    PushWheelEmpty -> left out (they are boxes around a group or an empty chair, not one person).

Output:
    data/pmma_yolo/images/{train,val,test}/*.jpg  and  data/pmma_yolo/labels/{...}/*.txt
    configs/pmma_dataset.yaml        (and configs/config.yaml -> vision.dataset_yaml is switched to it)
    results/pmma_check_first.jpg / pmma_check_last.jpg   (boxes drawn on the first and last frame: check them!)
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

NAMES = ["pedestrian", "wheelchair", "rollator", "cane"]
FRAME_RE = re.compile(r"frame_(\d+)\.(png|jpg|jpeg)$", re.I)


def occlusion_of(a: dict):
    """PMMA stores occlusion as an attribute (0 none, 1 partial, 2 full, 3 tree shadow). Key names vary."""
    for src in (a, a.get("attributes") or {}):
        for k, v in src.items():
            if "occlu" in str(k).lower():
                try:
                    return int(v)
                except (TypeError, ValueError):
                    return None
    return None


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", default="data/pmma/test.json")
    ap.add_argument("--zips", default="data/pmma/zips")
    ap.add_argument("--base", type=int, default=20000, help="annotation number of the first frame (020000.png)")
    ap.add_argument("--offset", type=int, default=1800, help="frame number that matches --base (frame_01800.png)")
    ap.add_argument("--every", type=int, default=1, help="keep every N-th frame")
    ap.add_argument("--train-share", type=float, default=0.0,
                    help="0 = everything is test. 0.6 = first 60%% train, next 10%% val, rest test (in time order)")
    ap.add_argument("--gap", type=int, default=150, help="frames skipped between train/val/test (avoids near-duplicates)")
    ap.add_argument("--max-occlusion", type=int, default=1,
                    help="drop people with occlusion above this (2 = fully hidden, 3 = tree shadow)")
    ap.add_argument("--out", default="data/pmma_yolo")
    ap.add_argument("--no-switch", action="store_true", help="do not switch configs/config.yaml to this dataset")
    args = ap.parse_args()

    class_map = cfg["vision"].get("pmma_class_map") or {}
    ann_path, zip_dir, out = resolve(args.annotations), resolve(args.zips), resolve(args.out)
    if not ann_path.exists():
        sys.exit(f"Not found: {ann_path}. Put test.json in data/pmma first.")
    zips = sorted(zip_dir.glob("*.zip"))
    if not zips:
        sys.exit(f"No .zip files in {zip_dir}. Download video_2_1.zip there first.")

    coco = json.load(open(ann_path))
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    by_img = defaultdict(list)
    for a in coco["annotations"]:
        by_img[a["image_id"]].append(a)
    info_by_name = {i["file_name"]: i for i in coco["images"]}
    unknown = sorted(set(cats.values()) - set(class_map))
    if unknown:
        print("WARNING: these PMMA classes are not in pmma_class_map and will be left out:", unknown)

    # 1. collect frames that exist in the zips AND have annotations
    frames = []                                    # (frame_no, zip_path, member_name, annotation info)
    for zp in zips:
        try:
            zf = zipfile.ZipFile(zp)
        except zipfile.BadZipFile:
            print(f"SKIP {zp.name}: not a complete zip yet (still downloading?)")
            continue
        for m in zf.namelist():
            mm = FRAME_RE.search(m)
            if not mm or "__MACOSX" in m:
                continue
            fno = int(mm.group(1))
            name = f"{args.base + fno - args.offset:06d}.png"
            if name in info_by_name:
                frames.append((fno, zp, m, info_by_name[name]))
    frames.sort(key=lambda r: r[0])
    frames = frames[::max(1, args.every)]
    if not frames:
        sys.exit("No zip frames matched the annotations. Check --base/--offset (see the docstring).")
    print(f"{len(frames)} annotated frames found (frame {frames[0][0]} .. {frames[-1][0]}) in {len(zips)} zip(s)")

    # 2. split in time order
    n = len(frames)
    split_of = {}
    if args.train_share > 0:
        f0, f1 = frames[0][0], frames[-1][0]
        span = f1 - f0
        t_end = f0 + span * args.train_share
        v_end = t_end + span * 0.10
        for fno, *_ in frames:
            if fno < t_end:
                split_of[fno] = "train"
            elif t_end + args.gap <= fno < v_end:
                split_of[fno] = "val"
            elif fno >= v_end + args.gap:
                split_of[fno] = "test"
    else:
        split_of = {fno: "test" for fno, *_ in frames}

    # 3. write images + YOLO labels
    counts = defaultdict(Counter)
    dropped = Counter()
    open_zips = {}
    first_last = []
    for k, (fno, zp, member, info) in enumerate(frames):
        split = split_of.get(fno)
        if split is None:
            continue
        zf = open_zips.setdefault(zp, zipfile.ZipFile(zp))
        img = cv2.imdecode(np.frombuffer(zf.read(member), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        sx, sy = w / info["width"], h / info["height"]
        lines, boxes = [], []
        for a in by_img.get(info["id"], []):
            cname = cats[a["category_id"]]
            target = class_map.get(cname)
            if target is None:
                dropped[cname] += 1
                continue
            occ = occlusion_of(a)
            if occ is not None and occ > args.max_occlusion:
                dropped[f"{cname} (occluded {occ})"] += 1
                continue
            x, y, bw, bh = a["bbox"]
            x, y, bw, bh = x * sx, y * sy, bw * sx, bh * sy
            if bw < 4 or bh < 4:
                continue
            cid = NAMES.index(target)
            lines.append(f"{cid} {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}")
            boxes.append((int(x), int(y), int(x + bw), int(y + bh), target))
            counts[split][target] += 1
        stem = f"pmma_v2_{fno:05d}"
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / "images" / split / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        (out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines))
        counts[split]["_images"] += 1
        if k in (0, n - 1):
            first_last.append((img.copy(), boxes))
        if (k + 1) % 100 == 0:
            print(f"  {k + 1}/{n} frames written")

    # 4. check pictures: boxes on the first and last frame (they must sit on the people)
    res = resolve(cfg["paths"]["results_dir"])
    res.mkdir(parents=True, exist_ok=True)
    for tag, (img, boxes) in zip(("first", "last"), first_last):
        for x1, y1, x2, y2, c in boxes:
            col = (78, 143, 47) if c == "pedestrian" else (44, 29, 140)
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 3)
            cv2.putText(img, c, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2)
        cv2.imwrite(str(res / f"pmma_check_{tag}.jpg"), img)

    # 5. dataset yaml
    splits = [s for s in ("train", "val", "test") if counts[s]["_images"]]
    yml = ROOT / "configs" / "pmma_dataset.yaml"
    test_or_any = "images/test" if "test" in splits else f"images/{splits[0]}"
    lines = [f"# Made by scripts/11_pmma_to_yolo.py from {ann_path.name} + {', '.join(z.name for z in zips)}",
             f"path: {out.resolve()}",
             f"train: {'images/train' if 'train' in splits else test_or_any}",
             f"val: {'images/val' if 'val' in splits else test_or_any}",
             f"test: {test_or_any}",
             "names:"] + [f"  {i}: {nm}" for i, nm in enumerate(NAMES)]
    yml.write_text("\n".join(lines) + "\n")

    print("\nObjects per split:")
    for s in splits:
        c = counts[s]
        print(f"  {s:5s} {c['_images']:5d} images | " + ", ".join(f"{k} {c[k]}" for k in NAMES))
    if dropped:
        print("Left out:", dict(dropped))
    if "train" not in splits:
        print("NOTE: no train split (use --train-share 0.6 if you want to fine-tune YOLO on PMMA).")

    if not args.no_switch:
        cpath = ROOT / "configs" / "config.yaml"
        txt = cpath.read_text()
        txt2 = re.sub(r"(\n\s*dataset_yaml:\s*)\S+", r"\1configs/pmma_dataset.yaml", txt, count=1)
        if txt2 != txt:
            cpath.write_text(txt2)
            print("\nconfigs/config.yaml now uses configs/pmma_dataset.yaml for training and Experiment A.")
            print("(To go back to ICG later, change vision -> dataset_yaml to configs/icg_dataset.yaml.)")
    print(f"\nWrote {yml}\nCHECK these pictures before continuing: {res / 'pmma_check_first.jpg'} and pmma_check_last.jpg")


if __name__ == "__main__":
    main()
