"""STEP 14 - build the bigger PMMA dataset from videos 1, 2 and 3 (cross-scene design).

Design (PMMA paper: videos 1 and 2 are filmed from camera Position 1, video 3 from Position 2):
    train / val   = video 1                 (Position 1)
    test_same     = video 2 (the zip frames) (Position 1, another recording)  -> "same place"
    test_new      = video 3                 (Position 2, never seen in training) -> "new place"

Needs (all in data/pmma/):
    train.json, val.json, test.json            COCO annotations from Borealis
    videos/video_1.mp4, videos/video_3_part1.mp4, videos/video_3_part2.mp4
    zips/video_2_1.zip                         (already used before)

FIRST run the check (writes pictures with the boxes drawn, nothing else):
    python scripts/14_build_pmma_v2.py --check
Open results/pmma_v2_check_*.jpg : the boxes must sit on the people. Then build:
    python scripts/14_build_pmma_v2.py

How annotation names map to video frames (annotation "NNNNNN.png" = one global counter):
MEASURED with scripts/14a_align_pmma.py (left camera of the stereo mp4; 30-40 of 40 sampled frames agree):
    video 1: annotation number = frame - 901           (mp4 frames 901-9900 are labelled)
    video 2: annotation number = 20000 + frame - 1800  (self-test of 14a: found exactly 18200)
    video 3: annotation number = frame + 28493 in part1 (frames 0-7531)
             annotation number = frame + 28520 in part2 (part2 starts 27 labelled frames later)
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
COLORS = {"pedestrian": (78, 143, 47), "wheelchair": (44, 29, 140), "rollator": (0, 140, 255), "cane": (200, 60, 200)}


# ----------------------------------------------------------------------------- annotations
def load_annotations(pmma: Path):
    by_img, info, cats, src_of = defaultdict(list), {}, {}, {}
    n_occ = n_total = 0
    for name in ("train.json", "val.json", "test.json"):
        p = pmma / name
        if not p.exists():
            sys.exit(f"Missing {p}")
        d = json.load(open(p))
        cats.update({c["id"]: c["name"] for c in d["categories"]})
        for im in d["images"]:
            info[im["id"]] = im
            src_of[im["id"]] = name
        for a in d["annotations"]:
            by_img[a["image_id"]].append(a)
            n_total += 1
            if any("occlu" in str(k).lower() for k in list(a) + list((a.get("attributes") or {}))):
                n_occ += 1
    name_to_id = {im["file_name"]: i for i, im in info.items()}
    return by_img, info, cats, src_of, name_to_id, n_occ, n_total


def id_blocks(ids, src_of):
    """Contiguous runs of annotation numbers (helps confirm which run is which video)."""
    ids = sorted(ids)
    runs, start, prev = [], ids[0], ids[0]
    for i in ids[1:]:
        if i != prev + 1:
            runs.append((start, prev))
            start = i
        prev = i
    runs.append((start, prev))
    out = []
    for a, b in runs:
        srcs = Counter(src_of[i] for i in range(a, b + 1) if i in src_of)
        out.append((a, b, b - a + 1, dict(srcs)))
    return out


def yolo_lines(anns, cats, class_map, w0, h0, w, h, counts, split):
    sx, sy = w / w0, h / h0
    lines, boxes = [], []
    for a in anns:
        target = class_map.get(cats[a["category_id"]])
        if target is None:
            counts["_dropped"][cats[a["category_id"]]] += 1
            continue
        x, y, bw, bh = a["bbox"]
        x, y, bw, bh = x * sx, y * sy, bw * sx, bh * sy
        if bw < 4 or bh < 4:
            continue
        lines.append(f"{NAMES.index(target)} {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}")
        boxes.append((int(x), int(y), int(x + bw), int(y + bh), target))
        counts[split][target] += 1
    return lines, boxes


def draw(img, boxes, title):
    img = img.copy()
    for x1, y1, x2, y2, c in boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), COLORS[c], 3)
        cv2.putText(img, c, (x1, max(22, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLORS[c], 2)
    cv2.putText(img, title, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 4)
    cv2.putText(img, title, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 2)
    return img


# ----------------------------------------------------------------------------- video reading
EYE = "left"   # PMMA mp4 files are side-by-side stereo (4416 x 1242 = left | right); labels are 2208 x 1242


def one_eye(img):
    h, w = img.shape[:2]
    if w >= 3 * h:                                   # side-by-side stereo -> keep one camera
        return img[:, : w // 2] if EYE == "left" else img[:, w // 2:]
    return img


def iter_video(paths, wanted):
    """Read the parts one after another; frame numbers continue across parts.
    Yields (frame_no, image) only for frame numbers in `wanted` (decoding only those).
    Stereo frames are cut to one camera (EYE) so they match the 2208 x 1242 labels."""
    fno = 0
    for p in paths:
        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            sys.exit(f"Cannot open {p}")
        while True:
            if not cap.grab():
                break
            if fno in wanted:
                ok, img = cap.retrieve()
                if ok:
                    yield fno, one_eye(img)
            fno += 1
        cap.release()
        print(f"    {p.name}: frames counted so far = {fno}")


def count_frames(paths):
    counts = []
    for p in paths:
        cap = cv2.VideoCapture(str(p))
        n = 0
        while cap.grab():
            n += 1
        cap.release()
        counts.append(n)
    return counts


def zip_frames(zip_dir: Path):
    out = {}
    for zp in sorted(zip_dir.glob("*.zip")):
        try:
            zf = zipfile.ZipFile(zp)
        except zipfile.BadZipFile:
            print(f"SKIP {zp.name}: not a complete zip")
            continue
        for m in zf.namelist():
            mm = FRAME_RE.search(m)
            if mm and "__MACOSX" not in m:
                out[int(mm.group(1))] = (zp, m)
    return out


# ----------------------------------------------------------------------------- main
def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--pmma", default="data/pmma")
    ap.add_argument("--check", action="store_true", help="only write check pictures and print the numbering")
    ap.add_argument("--every", type=int, default=3, help="keep every N-th frame of video 1 (train/val)")
    ap.add_argument("--every-test", type=int, default=6, help="keep every N-th frame of video 3 (test_new)")
    ap.add_argument("--every-same", type=int, default=3, help="keep every N-th frame of video 2 (test_same)")
    ap.add_argument("--v1-max", type=int, default=8999,
                    help="last annotation number of video 1 (labels 10000-11699 are NOT video 1: 14a found no match)")
    ap.add_argument("--only", choices=["all", "v1", "v2", "v3"], default="all", help="rebuild only one video's split(s)")
    ap.add_argument("--val-share", type=float, default=0.15, help="last part of video 1 used as validation")
    ap.add_argument("--gap", type=int, default=150, help="frames skipped between train and val")
    ap.add_argument("--eye", choices=["left", "right"], default="left",
                    help="which half of the side-by-side stereo mp4 matches the labels (from script 14a)")
    ap.add_argument("--v1-base", type=int, default=-901)
    ap.add_argument("--v2-base", type=int, default=20000)
    ap.add_argument("--v2-offset", type=int, default=1800)
    ap.add_argument("--v3-base", type=int, default=28493, help="offset in video 3 part1")
    ap.add_argument("--v3-base2", type=int, default=28520, help="offset in video 3 part2")
    ap.add_argument("--v3-split", type=int, default=7532, help="number of frames in video_3_part1.mp4")
    ap.add_argument("--out", default="data/pmma_v2")
    args = ap.parse_args()

    global EYE
    EYE = args.eye
    pmma, out, res = resolve(args.pmma), resolve(args.out), resolve(cfg["paths"]["results_dir"])
    res.mkdir(parents=True, exist_ok=True)
    class_map = cfg["vision"]["pmma_class_map"]
    v1 = [pmma / "videos" / "video_1.mp4"]
    v3 = [pmma / "videos" / "video_3_part1.mp4", pmma / "videos" / "video_3_part2.mp4"]
    for p in v1 + v3:
        if not p.exists():
            sys.exit(f"Missing {p}")

    print("Reading annotations (train + val + test json) ...")
    by_img, info, cats, src_of, name_to_id, n_occ, n_total = load_annotations(pmma)
    print(f"  {len(info)} annotated images, {n_total} boxes")
    print(f"  boxes with an occlusion field: {n_occ} of {n_total}")

    def ann_id(video, frame):
        if video == "v1" and not (0 <= args.v1_base + frame <= args.v1_max):
            return None
        name = {"v1": f"{args.v1_base + frame:06d}.png",
                "v2": f"{args.v2_base + frame - args.v2_offset:06d}.png",
                "v3": f"{(args.v3_base if frame < args.v3_split else args.v3_base2) + frame:06d}.png"}[video]
        return name_to_id.get(name)

    if args.check:
        print("\nAnnotation numbers come in these continuous runs (start - end, count, which json):")
        for a, b, n, s in id_blocks(info.keys(), src_of):
            print(f"  {a:06d} - {b:06d}  ({n:5d} images)  {s}")
        print("\nCounting video frames (takes a minute) ...")
        c1, c3 = count_frames(v1), count_frames(v3)
        print(f"  video 1: {c1[0]} frames | video 3: part1 {c3[0]} + part2 {c3[1]} = {sum(c3)} frames")
        print("  (PMMA paper: video 1 = 9,000 frames, video 3 = 14,637-14,638 frames)")
        tot3 = sum(c3)
        picks = {"v1": [1500, 5000, 9500], "v3": [2000, 7500, 7600, 12000, 16000]}   # labelled frames
        for video, paths in (("v1", v1), ("v3", v3)):
            wanted = set(picks[video])
            for fno, img in iter_video(paths, wanted):
                aid = ann_id(video, fno)
                h, w = img.shape[:2]
                boxes = []
                if aid is not None:
                    im = info[aid]
                    _, boxes = yolo_lines(by_img.get(aid, []), cats, class_map, im["width"], im["height"],
                                          w, h, defaultdict(Counter), "check")
                title = f"{video} frame {fno} -> annotation {aid if aid is not None else 'NONE'} ({len(boxes)} boxes)"
                cv2.imwrite(str(res / f"pmma_v2_check_{video}_{fno:05d}.jpg"), draw(img, boxes, title))
                print(f"  wrote pmma_v2_check_{video}_{fno:05d}.jpg  -> {title}")
        zf = zip_frames(pmma / "zips")
        if zf:
            fno = sorted(zf)[len(zf) // 2]
            zp, m = zf[fno]
            img = cv2.imdecode(np.frombuffer(zipfile.ZipFile(zp).read(m), np.uint8), cv2.IMREAD_COLOR)
            aid = ann_id("v2", fno)
            boxes = []
            if aid is not None:
                im = info[aid]
                _, boxes = yolo_lines(by_img.get(aid, []), cats, class_map, im["width"], im["height"],
                                      img.shape[1], img.shape[0], defaultdict(Counter), "check")
            cv2.imwrite(str(res / f"pmma_v2_check_v2_{fno:05d}.jpg"),
                        draw(img, boxes, f"v2 frame {fno} -> annotation {aid} ({len(boxes)} boxes)"))
            print(f"  wrote pmma_v2_check_v2_{fno:05d}.jpg")
        print(f"\nOpen the pictures in {res} named pmma_v2_check_*.jpg - the boxes must sit on the people.")
        return

    # ---------------------------------------------------------------- build
    counts = defaultdict(Counter)

    def write(split, stem, img, aid):
        im = info[aid]
        h, w = img.shape[:2]
        lines, _ = yolo_lines(by_img.get(aid, []), cats, class_map, im["width"], im["height"], w, h, counts, split)
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / "images" / split / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        (out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines))
        counts[split]["_images"] += 1

    import shutil

    def fresh(*splits):                     # remove old files of a split before rebuilding it
        for sp in splits:
            for kind in ("images", "labels"):
                shutil.rmtree(out / kind / sp, ignore_errors=True)

    # video 1 -> train / val (time order, with a gap); only labels 0-8999 belong to video 1
    if args.only in ("all", "v1"):
        fresh("train", "val")
        print("\nVideo 1 -> train / val")
        c1 = count_frames(v1)[0]
        lab1 = [f for f in range(c1) if ann_id("v1", f) is not None][::args.every]
        if not lab1:
            sys.exit("No video 1 frames matched the annotations - run --check and send the output.")
        cut = lab1[0] + (lab1[-1] - lab1[0]) * (1 - args.val_share)
        split_of = {f: ("train" if f < cut else "val") for f in lab1 if f < cut or f >= cut + args.gap}
        for k, (fno, img) in enumerate(iter_video(v1, set(split_of))):
            write(split_of[fno], f"pmma_v1_{fno:05d}", img, ann_id("v1", fno))
            if (k + 1) % 250 == 0:
                print(f"  {k + 1}/{len(split_of)}")

    # video 2 (zip) -> test_same
    if args.only in ("all", "v2"):
        fresh("test_same")
        print("Video 2 (zip) -> test_same")
        zf = zip_frames(pmma / "zips")
        lab2 = [f for f in sorted(zf) if ann_id("v2", f) is not None][::args.every_same]
        open_z = {}
        for fno in lab2:
            zp, m = zf[fno]
            z = open_z.setdefault(zp, zipfile.ZipFile(zp))
            img = cv2.imdecode(np.frombuffer(z.read(m), np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                write("test_same", f"pmma_v2_{fno:05d}", img, ann_id("v2", fno))

    # video 3 -> test_new
    if args.only in ("all", "v3"):
        fresh("test_new")
        print("Video 3 -> test_new (camera position never seen in training)")
        c3 = sum(count_frames(v3))
        lab3 = [f for f in range(c3) if ann_id("v3", f) is not None][::args.every_test]
        for k, (fno, img) in enumerate(iter_video(v3, set(lab3))):
            write("test_new", f"pmma_v3_{fno:05d}", img, ann_id("v3", fno))
            if (k + 1) % 250 == 0:
                print(f"  {k + 1}/{len(lab3)}")

    yml = ROOT / "configs" / "pmma_v2.yaml"
    yml.write_text("\n".join([
        "# Made by scripts/14_build_pmma_v2.py: train/val = video 1, test_same = video 2, test_new = video 3",
        f"path: {out.resolve()}",
        "train: images/train",
        "val: images/val",
        "test: images/test_new",
        "test_same: images/test_same",
        "test_new: images/test_new",
        "names:"] + [f"  {i}: {n}" for i, n in enumerate(NAMES)]) + "\n")

    print("\nImages and people per split (counted on disk):")
    for sp in ("train", "val", "test_same", "test_new"):
        c = Counter()
        lab_dir = out / "labels" / sp
        files = list(lab_dir.glob("*.txt")) if lab_dir.exists() else []
        for f in files:
            for line in f.read_text().splitlines():
                if line.strip():
                    c[NAMES[int(line.split()[0])]] += 1
        print(f"  {sp:9s} {len(files):5d} images | " + ", ".join(f"{n} {c[n]}" for n in NAMES))
    print("Left out (group boxes / empty wheelchairs):", dict(counts["_dropped"]))
    print(f"\nWrote {yml}")


if __name__ == "__main__":
    main()
