"""STEP 20 - tracking evaluation on REAL video (CV) + calibration of the Experiment B simulation.

Experiment A scores single pictures. The traffic signal, however, sees a VIDEO through a tracker, and
Experiment B assumed how that tracker fails (misses in bursts of ~0.8 s; lost tracks for 3-15 s that come
back with a new ID). This script measures those things on real PMMA video:

  * MOTA, IDF1, ID switches, recall, precision   (standard multi-object-tracking scores)
  * per person: share of frames found, miss bursts (how long the camera "loses" them), ID changes
  * separately for mobility-aid users and pedestrians
  * calibration.json: the measured values in the format of the config 'failures' block
    -> python scripts/04_experiment_b.py --calibration results/tracking/<tag>/calibration.json

Fair-test design: the detector was trained on video 1 only. Default = video 3 (the NEW camera position,
never seen in training). --video 2 = same position as training, different day.

    python scripts/20_eval_tracking.py --weights weights/pmma_v2_yolo11s_seed0.pt --max-frames 300   # quick test
    python scripts/20_eval_tracking.py --weights weights/pmma_v2_yolo11s_seed0.pt                   # all of video 3
    python scripts/20_eval_tracking.py --weights weights/pmma_v2_aug_yolo11s_seed0.pt --tag robust_v3

Ground-truth identities (who is who across frames):
  1. a track id inside the COCO labels, if PMMA has one;  2. the CVAT XML (--xml, or found under data/pmma);
  3. otherwise linked frame-to-frame by box overlap (labels exist for every frame, so this is reliable;
     the script says which source it used - write it in the report).
"""
import argparse
import importlib.util
import json
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

ROOTDIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOTDIR))
from safecross.config import load_config, resolve
from safecross.vision.track_eval import AID, calibration, evaluate, iou_matrix

_spec = importlib.util.spec_from_file_location("pmma_v2", ROOTDIR / "scripts" / "14_build_pmma_v2.py")
B14 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B14)                      # reuse the proven frame <-> label mapping helpers

IGNORE_CLASSES = ("PushWheel1", "PushWheelEmpty")  # group box / person pushing an empty chair: not FP, not GT
TRACK_KEYS = ("track_id", "trackId", "instance_id", "object_id")


# ----------------------------------------------------------------------------- ground truth
def coco_track_id(a):
    for k in TRACK_KEYS:
        if k in a:
            return a[k]
        if k in (a.get("attributes") or {}):
            return a["attributes"][k]
    return None


def load_cvat(xml_path: Path, class_map):
    """{cvat_frame: [(track_id, box, label)]} from a CVAT 'video' XML (tracks of boxes)."""
    out = defaultdict(list)
    for tr in ET.parse(xml_path).getroot().iter("track"):
        tid, label = tr.get("id"), tr.get("label")
        for b in tr.iter("box"):
            if b.get("outside") == "1":
                continue
            box = tuple(float(b.get(k)) for k in ("xtl", "ytl", "xbr", "ybr"))
            out[int(b.get("frame"))].append((f"cvat{tid}", box, label))
    return out


def align_cvat(cvat, coco_boxes, hint):
    """Find the offset label_number = cvat_frame + off by box overlap (no guessing).
    Tries the expected offset (and +-60) first; searches every possible offset only if that fails."""
    sample = [n for n in sorted(coco_boxes) if len(coco_boxes[n]) >= 2]
    sample = sample[:: max(1, len(sample) // 40)][:40]
    if not sample or not cvat:
        return None, 0.0

    def score(off):
        sc = []
        for n in sample:
            C = cvat.get(n - off, [])
            M = iou_matrix(coco_boxes[n], [c[1] for c in C])
            sc.append(M.max(axis=1).mean() if M.size else 0.0)
        return float(np.mean(sc))

    best = max(((o, score(o)) for o in range(hint - 60, hint + 61)), key=lambda x: x[1])
    if best[1] < 0.8:
        print("    expected offset did not fit - searching all offsets (a minute) ...")
        lo, hi = min(sample) - max(cvat), max(sample) - min(cvat)
        best = max(((o, score(o)) for o in range(lo, hi + 1)), key=lambda x: x[1])
    return best


def link_by_overlap(frames_boxes, max_gap=5, thr=0.3):
    """Fallback identities: connect boxes frame to frame (Hungarian on IoU, may skip up to max_gap frames)."""
    ids, next_id, active = {}, 0, []        # active: [id, box, last_index]
    for n, (f, boxes) in enumerate(frames_boxes):
        active = [a for a in active if n - a[2] <= max_gap]
        M = iou_matrix([a[1] for a in active], boxes)
        got = {}
        if M.size:
            r, c = linear_sum_assignment(-M)
            got = {j: i for i, j in zip(r, c) if M[i, j] >= thr}
        for j, b in enumerate(boxes):
            if j in got:
                a = active[got[j]]
                a[1], a[2] = b, n
                ids[(f, j)] = a[0]
            else:
                ids[(f, j)] = f"link{next_id}"
                active.append([ids[(f, j)], b, n])
                next_id += 1
    return ids


# ----------------------------------------------------------------------------- frames
def frames_for(video, pmma, args):
    """(label_of_frame, sequence_of_frame, source). A new sequence = new mp4 part (the tracker restarts)."""
    if video == 3:
        paths = [pmma / "videos" / "video_3_part1.mp4", pmma / "videos" / "video_3_part2.mp4"]
        for p in paths:
            if not p.exists():
                sys.exit(f"Missing {p}")
        return (lambda f: f + (args.v3_base if f < args.v3_split else args.v3_base2),
                lambda f: 0 if f < args.v3_split else 1, paths)
    zf = B14.zip_frames(pmma / "zips")
    if not zf:
        sys.exit(f"No video-2 zips found in {pmma / 'zips'}")
    return (lambda f: args.v2_base + f - args.v2_offset), (lambda f: 0), zf


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/pmma_v2_yolo11s_seed0.pt")
    ap.add_argument("--video", type=int, choices=[2, 3], default=3)
    ap.add_argument("--pmma", default="data/pmma")
    ap.add_argument("--xml", default=None, help="CVAT XML of this video (optional; found automatically if possible)")
    ap.add_argument("--tracker", default="bytetrack.yaml")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=None, help="leave empty: ByteTrack needs the low-score boxes")
    ap.add_argument("--iou", type=float, default=0.5, help="IoU needed for a match")
    ap.add_argument("--start", type=int, default=0, help="skip this many labelled frames")
    ap.add_argument("--max-frames", type=int, default=None, help="only this many labelled frames (quick test)")
    ap.add_argument("--fps", type=float, default=None, help="video frame rate (read from the mp4 if empty)")
    ap.add_argument("--save-video", type=int, default=0, help="also write an annotated mp4 of the first N frames")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--v2-base", type=int, default=20000)
    ap.add_argument("--v2-offset", type=int, default=1800)
    ap.add_argument("--v3-base", type=int, default=28493)
    ap.add_argument("--v3-base2", type=int, default=28520)
    ap.add_argument("--v3-split", type=int, default=7532)
    args = ap.parse_args()

    pmma = resolve(args.pmma)
    weights = resolve(args.weights)
    if not weights.exists():
        sys.exit(f"{weights} not found")
    tag = args.tag or f"{weights.stem}_v{args.video}"
    out = resolve(cfg["paths"]["results_dir"]) / "tracking" / tag
    out.mkdir(parents=True, exist_ok=True)
    class_map = cfg["vision"]["pmma_class_map"]

    # ---- labels
    print("Reading PMMA labels (train/val/test json) ...")
    by_img, info, cats, src_of, name_to_id, _, _ = B14.load_annotations(pmma)
    num_to_img = {int(Path(im["file_name"]).stem): iid for iid, im in info.items()
                  if Path(im["file_name"]).stem.isdigit()}
    label_of, seq_of, source = frames_for(args.video, pmma, args)
    rng_lab = (30000, 44637) if args.video == 3 else (20000, 24356)
    if args.video == 3:
        cap = cv2.VideoCapture(str(source[0]))
        fps = args.fps or cap.get(cv2.CAP_PROP_FPS) or 12.55
        cap.release()
        first_f = rng_lab[0] - args.v3_base
        last_f = rng_lab[1] - args.v3_base2
        all_frames = [f for f in range(first_f, last_f + 1) if label_of(f) in num_to_img]
    else:
        fps = args.fps or 15.0
        all_frames = [f for f in sorted(source) if label_of(f) in num_to_img]
        print(f"  video 2 comes as pictures: using {fps} fps (set --fps if you know the exact rate)")
    frames = all_frames[args.start:]
    if args.max_frames:
        frames = frames[: args.max_frames]
    if not frames:
        sys.exit("No labelled frames in this range - check the offsets (scripts/14a).")
    print(f"  video {args.video}: {len(frames)} labelled frames to track ({len(frames) / fps:.0f} s at {fps:.2f} fps)")

    # GT boxes per frame (in label pixel units, 2208 x 1242), plus ignore boxes
    gt_raw, ignore, has_tid = {}, {}, 0
    for f in frames:
        iid = num_to_img[label_of(f)]
        G, I = [], []
        for a in by_img.get(iid, []):
            name = cats[a["category_id"]]
            x, y, w, h = a["bbox"]
            box = (x, y, x + w, y + h)
            if name in IGNORE_CLASSES:
                I.append(box)
            elif name in class_map and w >= 4 and h >= 4:
                tid = coco_track_id(a)
                has_tid += tid is not None
                G.append((tid, box, class_map[name]))
        gt_raw[f], ignore[f] = G, I

    # ---- identities
    n_boxes = sum(len(v) for v in gt_raw.values())
    id_source = None
    if n_boxes and has_tid == n_boxes:
        id_source = "COCO track ids"
        gt = {f: [(f"coco{t}", b, c) for t, b, c in G] for f, G in gt_raw.items()}
    else:
        xmls = [resolve(args.xml)] if args.xml else sorted(pmma.rglob(f"*{args.video}*.xml"))
        for xp in xmls:
            print(f"  trying CVAT file {xp.name} ...")
            cvat = load_cvat(xp, class_map)
            coco_boxes = {label_of(f): [b for _, b, _ in G] for f, G in gt_raw.items()}
            off, score = align_cvat(cvat, coco_boxes, rng_lab[0])
            print(f"    best alignment: label = CVAT frame + {off}  (mean box overlap {score:.2f})")
            if off is None or score < 0.8:
                continue
            gt = {}
            for f, G in gt_raw.items():
                C = cvat.get(label_of(f) - off, [])
                M = iou_matrix([b for _, b, _ in G], [c[1] for c in C])
                got = {}
                if M.size:
                    r, c = linear_sum_assignment(-M)
                    got = {i: j for i, j in zip(r, c) if M[i, j] >= 0.7}
                gt[f] = [(C[got[i]][0] if i in got else f"un{f}_{i}", b, cl) for i, (_, b, cl) in enumerate(G)]
            unmatched = sum(1 for G in gt.values() for g in G if str(g[0]).startswith("un"))
            id_source = f"CVAT XML {xp.name} (offset {off}, {unmatched} of {n_boxes} boxes without a CVAT id)"
            break
        if id_source is None:
            ids = link_by_overlap([(f, [b for _, b, _ in gt_raw[f]]) for f in frames])
            gt = {f: [(ids[(f, j)], b, c) for j, (_, b, c) in enumerate(G)] for f, G in gt_raw.items()}
            id_source = "linked frame-to-frame by box overlap (no track ids in the labels or XML)"
    print(f"  ground-truth identities: {id_source}")

    # ---- run detector + tracker
    from ultralytics import YOLO
    from safecross.vision.detect import canonical_class, pick_device
    device = pick_device(cfg["vision"].get("device", "auto"))
    evaluate_on = set(frames)
    if args.video == 3:
        wanted = set(range(frames[0], frames[-1] + 1))          # the tracker sees EVERY frame, labelled or not
    else:
        wanted = {f for f in source if frames[0] <= f <= frames[-1]}
    pred, seqs, cur_seq = {}, [], None
    model = None
    writer = None
    t0 = time.time()
    if args.video == 3:
        it = B14.iter_video(source, wanted)
    else:
        def _zip_iter():
            import zipfile
            for f in sorted(wanted):
                zp, m = source[f]
                img = cv2.imdecode(np.frombuffer(zipfile.ZipFile(zp).read(m), np.uint8), cv2.IMREAD_COLOR)
                yield f, B14.one_eye(img)
        it = _zip_iter()
    for f, img in it:
        s = seq_of(f)
        if s != cur_seq:                                                 # new mp4 part: restart the tracker
            model, cur_seq = YOLO(str(weights)), s
            seqs.append([])
        kw = dict(persist=True, tracker=args.tracker, imgsz=args.imgsz, device=device, verbose=False)
        if args.conf is not None:
            kw["conf"] = args.conf
        r = model.track(img, **kw)[0]
        if f not in evaluate_on:                                         # unlabelled frame: only tracked
            continue
        h, w = img.shape[:2]
        im = info[num_to_img[label_of(f)]]
        sx, sy = im["width"] / w, im["height"] / h                     # tracker pixels -> label pixels
        P = []
        if r.boxes is not None and r.boxes.id is not None:
            for b, c, cf, tid in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(),
                                     r.boxes.conf.cpu().numpy(), r.boxes.id.int().cpu().tolist()):
                P.append((tid, (b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy), canonical_class(model.names[int(c)]),
                          float(cf)))
        pred[f] = P
        seqs[-1].append(f)
        if args.save_video and len(pred) <= args.save_video:
            if writer is None:
                writer = cv2.VideoWriter(str(out / "tracking_sample.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                         fps, (w // 2, h // 2))
            vis = img.copy()
            for _, b, _ in gt[f]:
                cv2.rectangle(vis, (int(b[0] / sx), int(b[1] / sy)), (int(b[2] / sx), int(b[3] / sy)), (255, 255, 255), 2)
            for tid, b, c, _ in P:
                col = (45, 30, 139) if c in AID else (80, 170, 60)
                p1, p2 = (int(b[0] / sx), int(b[1] / sy)), (int(b[2] / sx), int(b[3] / sy))
                cv2.rectangle(vis, p1, p2, col, 4)
                cv2.putText(vis, f"#{tid} {c}", (p1[0], max(25, p1[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 3)
            cv2.putText(vis, "white = labels, colour = tracker", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 5)
            cv2.putText(vis, "white = labels, colour = tracker", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            writer.write(cv2.resize(vis, (w // 2, h // 2)))
        if len(pred) % 250 == 0:
            el, k = time.time() - t0, len(pred)
            print(f"  {k}/{len(frames)} frames, {el / k:.2f} s each, about {el / k * (len(frames) - k) / 60:.0f} min left")
    if writer is not None:
        writer.release()

    # ---- scores
    summary, tracks, bursts, changes = evaluate(gt, pred, seqs, fps, ignore, iou_thr=args.iou)
    persistence = cfg["controller"]["persistence_s"]
    cal = calibration(summary, tracks, bursts, changes, sim_dt=cfg["crossing"]["dt"],
                      persistence_s=persistence)
    cal["measured_on"] = {"video": args.video, "weights": weights.name, "tracker": args.tracker,
                          "frames": len(pred), "fps": fps, "identities": id_source}
    row = {k: v for k, v in summary.items() if k != "phantom_lengths_s"}
    row.update({"video": args.video, "weights": weights.name, "tracker": args.tracker, "identities": id_source})
    pd.DataFrame([row]).to_csv(out / "summary.csv", index=False)
    pd.DataFrame(tracks).to_csv(out / "per_person.csv", index=False)
    pd.DataFrame(bursts).to_csv(out / "miss_bursts.csv", index=False)
    pd.DataFrame(changes).to_csv(out / "id_changes.csv", index=False)
    (out / "calibration.json").write_text(json.dumps(cal, indent=2))

    print("\n=== Tracking on real video (" + ("NEW camera position, video 3" if args.video == 3 else "video 2") + ") ===")
    print(f"MOTA {summary['MOTA']:.3f}   IDF1 {summary['IDF1']:.3f}   ID switches {summary['IDSW']}   "
          f"recall {summary['recall']:.3f}   precision {summary['precision']:.3f}")
    print(f"GT people {summary['gt_people']}, tracker IDs {summary['tracker_ids']}, phantom tracker IDs "
          f"{summary['phantom_tracks']}, {summary['seconds_of_video']:.0f} s of video")
    for g in ("mobility_aid", "pedestrian"):
        c = cal.get(g)
        if not c:
            print(f"\n{g}: none in this segment")
            continue
        print(f"\n{'Mobility-aid users' if g == 'mobility_aid' else 'Pedestrians'}: {c['people']} people, found in {c['recall']:.1%} of frames")
        print(f"  miss bursts: mean {c['mean_miss_burst_s']:.2f} s, median {c['median_miss_burst_s']:.2f} s, "
              f"longest {c['longest_miss_s']:.1f} s; {c['share_of_bursts_longer_than_persistence']:.0%} last longer "
              f"than the baseline's {persistence:.1f} s memory")
        print(f"  people whose ID changed: {c['share_people_with_id_change']:.0%}  "
              f"(gap before the new ID, middle half: {c['id_change_gap_s_p25_p75']})")
        if g == "mobility_aid":
            print(f"  labelled 'pedestrian' in {c['misclassification_prob']:.1%} of found frames")
    fb = cal["failures_for_experiment_b"]
    print("\nSimulation assumed: misses in bursts of 0.8 s; lost tracks for 3-15 s, then a new ID.")
    print(f"Measured (mobility-aid users): recall {fb['recall']}, bursts {fb['miss_burst_frames'] * cfg['crossing']['dt']:.2f} s, "
          f"ID change in {fb['track_loss_prob']:.0%} of people, gap {fb['track_loss_duration_s']} s, "
          f"phantoms {fb['false_positive_rate']}/s")
    print(f"\nSaved to {out}\nNext: python scripts/04_experiment_b.py --calibration {out / 'calibration.json'}")


if __name__ == "__main__":
    main()
