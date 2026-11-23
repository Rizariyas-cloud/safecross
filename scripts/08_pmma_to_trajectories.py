"""STEP 8 (optional, after PMMA is downloaded) - turn PMMA tracks into real walking-speed
profiles for the simulated crossing.

How the mapping works (write this in your methodology):
  1. For every tracked person in PMMA we take their box centre in each frame.
  2. We convert pixel movement to metres with --pixels-per-metre (measure it once from
     objects of known size in the video, e.g. the spacing of the traffic cones).
  3. We smooth the path and compute the walking SPEED at each moment.
  4. Only the speed profile is kept. In the simulation, each speed profile is "replayed"
     along the virtual crosswalk, from the entry line to the exit line.
  So the crossing is simulated, but how fast (and how unevenly) people move is real.

    python scripts/08_pmma_to_trajectories.py --inspect          # look at the annotation format first
    python scripts/08_pmma_to_trajectories.py --pixels-per-metre 60

Output: data/pmma_trajectories.json (use it with scripts/04 --trajectories ...)
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from safecross.config import load_config, resolve
from safecross.vision.detect import canonical_class


def _track_id(ann):
    for key in ("track_id", "trackId", "instance_id"):
        if key in ann:
            return ann[key]
    attrs = ann.get("attributes") or {}
    for key in ("track_id", "trackId", "id"):
        if key in attrs:
            return attrs[key]
    return None


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true", help="print the structure of the first annotation file")
    ap.add_argument("--pixels-per-metre", type=float, default=None)
    ap.add_argument("--fps", type=float, default=15.0, help="PMMA videos are 15 fps (one is 13 fps)")
    ap.add_argument("--min-seconds", type=float, default=3.0)
    args = ap.parse_args()

    files = sorted(resolve(cfg["paths"]["pmma_dir"]).rglob("*.json"))
    if not files:
        sys.exit("No COCO .json annotation files found in data/pmma")
    if args.inspect:
        d = json.loads(files[0].read_text())
        print(files[0])
        print("top-level keys:", list(d))
        print("categories:", d.get("categories"))
        print("first image:", (d.get("images") or [None])[0])
        print("first annotation:", (d.get("annotations") or [None])[0])
        return
    if not args.pixels_per_metre:
        sys.exit("Give --pixels-per-metre (see the explanation at the top of this file).")

    dt = cfg["crossing"]["dt"]
    bank = []
    for fpath in files:
        d = json.loads(fpath.read_text())
        cats = {c["id"]: canonical_class(c["name"]) for c in d.get("categories", [])}
        frame_of = {}
        for im in d.get("images", []):
            frame_of[im["id"]] = im.get("frame_id", im.get("frame", im["id"]))
        tracks = defaultdict(list)
        for a in d.get("annotations", []):
            tid = _track_id(a)
            if tid is None:
                continue
            x, y, w, h = a["bbox"]
            tracks[(fpath.name, tid)].append((frame_of.get(a["image_id"], a["image_id"]), x + w / 2, y + h,
                                              cats.get(a["category_id"], "pedestrian")))
        for key, pts in tracks.items():
            pts.sort()
            if len(pts) < args.min_seconds * args.fps:
                continue
            f = np.array([p[0] for p in pts], float)
            xy = np.array([[p[1], p[2]] for p in pts], float) / args.pixels_per_metre
            k = max(1, int(args.fps // 3))                          # smooth over ~1/3 s
            ker = np.ones(k) / k
            xs, ys = np.convolve(xy[:, 0], ker, "valid"), np.convolve(xy[:, 1], ker, "valid")
            ts = f[:len(xs)] / args.fps
            step = np.hypot(np.diff(xs), np.diff(ys))
            sp = step / np.maximum(np.diff(ts), 1e-6)
            grid = np.arange(ts[0], ts[-2], dt)                     # resample to the simulation time step
            speeds = np.interp(grid, ts[:-1], sp)
            speeds = np.clip(speeds, 0.0, 2.5)
            cls = max(set(p[3] for p in pts), key=[p[3] for p in pts].count)
            bank.append({"source": f"{key[0]}#{key[1]}", "cls": cls, "speeds": speeds.round(3).tolist(),
                         "mean_speed": float(speeds.mean())})
    out = resolve(cfg["paths"]["data_dir"]) / "pmma_trajectories.json"
    out.write_text(json.dumps(bank))
    by = defaultdict(list)
    for b in bank:
        by[b["cls"]].append(b["mean_speed"])
    for c, v in by.items():
        print(f"{c:>12}: {len(v)} tracks, mean speed {np.mean(v):.2f} m/s")
    print(f"Saved {len(bank)} speed profiles -> {out}")


if __name__ == "__main__":
    main()
