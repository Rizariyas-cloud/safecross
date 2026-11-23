"""Tracking evaluation on real video (CV): CLEAR-MOT (MOTA, ID switches), IDF1, and the per-person
miss-burst / identity-change statistics that the Experiment B simulation ASSUMES.

Why this exists
---------------
Experiment B simulates camera failures with chosen settings (misses in bursts of ~0.8 s, track loss of
3-15 s with a new ID). This module MEASURES those quantities on real PMMA video, so the simulation can be
calibrated (scripts/20 writes results/tracking/<tag>/calibration.json; scripts/04 --calibration uses it).

Definitions (standard, so they can be cited):
  * a ground-truth (GT) box and a tracker box MATCH when IoU >= 0.5 (one-to-one, Hungarian).
    Matches from the previous frame are kept first when still valid (CLEAR-MOT rule, Bernardin & Stiefelhagen 2008).
  * FN = GT boxes with no match, FP = tracker boxes with no match (boxes on "ignore" people, e.g. someone
    pushing an EMPTY wheelchair or a whole-group box, are not counted as FP), IDSW = a GT person matched to a
    different tracker ID than the last time they were matched.
  * MOTA = 1 - (FN + FP + IDSW) / number of GT boxes.
  * IDF1 (Ristani et al. 2016): one global one-to-one assignment between GT people and tracker IDs that
    maximises the frames they agree on; IDF1 = 2 * IDTP / (GT boxes + tracker boxes).
  * miss burst = a run of consecutive frames in which a GT person is in the labels but NOT matched.

Inputs (all plain Python, so the metrics are unit-tested without a GPU):
  gt[f]     = list of (gt_id, (x1, y1, x2, y2), cls)
  pred[f]   = list of (track_id, (x1, y1, x2, y2), cls, conf)
  ignore[f] = list of (x1, y1, x2, y2)
  seqs      = list of lists of frame keys, in time order (a new sequence = tracker restarted)
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

AID = ("wheelchair", "rollator", "crutches", "cane")


def iou_matrix(a, b) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a, b = np.asarray(a, float), np.asarray(b, float)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def _runs(flags):
    """[(start_index, length)] of consecutive True values."""
    out, start = [], None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        if not f and start is not None:
            out.append((start, i - start))
            start = None
    if start is not None:
        out.append((start, len(flags) - start))
    return out


def evaluate(gt, pred, seqs, fps: float, ignore=None, iou_thr: float = 0.5):
    ignore = ignore or {}
    n_gt = n_fp = n_fn = n_idsw = n_match = n_pred = n_ignored = 0
    pair_counts = defaultdict(int)          # (gt key, pred key) -> frames with IoU >= thr (for IDF1)
    gt_len, pred_len = defaultdict(int), defaultdict(int)
    per = {}                                # gt key -> dict of per-frame lists
    pred_matched = defaultdict(int)         # pred key -> frames matched to a GT person
    for si, frames in enumerate(seqs):
        last = {}                           # gt id -> pred id it was last matched to (this sequence)
        for f in frames:
            G, P, I = gt.get(f, []), pred.get(f, []), ignore.get(f, [])
            gk = [(si, g[0]) for g in G]
            pk = [(si, p[0]) for p in P]
            M = iou_matrix([g[1] for g in G], [p[1] for p in P])
            # IDF1 pair counts: every GT/pred pair overlapping enough in this frame
            for i, j in zip(*np.where(M >= iou_thr)):
                pair_counts[(gk[i], pk[j])] += 1
            # CLEAR-MOT matching: keep last frame's correspondences first, then Hungarian on the rest
            cost = np.where(M >= iou_thr, 1.0 - M, 1e6)
            for i, g in enumerate(G):
                for j, p in enumerate(P):
                    if M[i, j] >= iou_thr and last.get(g[0]) == p[0]:
                        cost[i, j] = -1.0
            matches = []
            if len(G) and len(P):
                r, c = linear_sum_assignment(cost)
                matches = [(i, j) for i, j in zip(r, c) if cost[i, j] < 1e5]
            mg = {i: j for i, j in matches}
            mp = {j for _, j in matches}
            # unmatched predictions on ignore people are neither FP nor counted as tracker boxes
            Ig = iou_matrix([p[1] for p in P], I) if I else np.zeros((len(P), 0))
            ign = {j for j in range(len(P)) if j not in mp and Ig.shape[1] and Ig[j].max() >= iou_thr}
            n_ignored += len(ign)
            for j, p in enumerate(P):
                if j in ign:
                    continue
                n_pred += 1
                pred_len[pk[j]] += 1
                if j in mp:
                    pred_matched[pk[j]] += 1
            n_fp += len(P) - len(mp) - len(ign)
            for i, g in enumerate(G):
                n_gt += 1
                gt_len[gk[i]] += 1
                d = per.setdefault(gk[i], {"frames": [], "matched": [], "pid": [], "pcls": [], "cls": []})
                d["frames"].append(f)
                d["cls"].append(g[2])
                if i in mg:
                    p = P[mg[i]]
                    n_match += 1
                    if g[0] in last and last[g[0]] != p[0]:
                        n_idsw += 1
                        d.setdefault("switch_at", []).append(len(d["frames"]) - 1)
                    last[g[0]] = p[0]
                    d["matched"].append(True)
                    d["pid"].append(p[0])
                    d["pcls"].append(p[2])
                else:
                    n_fn += 1
                    d["matched"].append(False)
                    d["pid"].append(None)
                    d["pcls"].append(None)
    # ---- IDF1: global one-to-one assignment GT people <-> tracker IDs
    gks, pks = sorted(gt_len), sorted(pred_len)
    idtp = 0
    if gks and pks:
        gi, pi = {k: i for i, k in enumerate(gks)}, {k: i for i, k in enumerate(pks)}
        W = np.zeros((len(gks), len(pks)))
        for (g, p), n in pair_counts.items():
            if g in gi and p in pi:
                W[gi[g], pi[p]] = n
        r, c = linear_sum_assignment(-W)
        idtp = int(W[r, c].sum())
    idf1 = 2 * idtp / max(n_gt + n_pred, 1)
    summary = {
        "gt_boxes": n_gt, "tracker_boxes": n_pred, "matched": n_match, "FN": n_fn, "FP": n_fp, "IDSW": n_idsw,
        "ignored_tracker_boxes": n_ignored,
        "MOTA": 1 - (n_fn + n_fp + n_idsw) / max(n_gt, 1),
        "IDF1": idf1, "IDTP": idtp,
        "recall": n_match / max(n_gt, 1), "precision": n_match / max(n_pred, 1),
        "gt_people": len(gt_len), "tracker_ids": len(pred_len),
        "fps": fps,
    }
    tracks, bursts, changes = _per_track(per, fps)
    # false-positive "people": tracker IDs that (almost) never sit on a real person
    fp_tracks = [(k, n) for k, n in pred_len.items() if pred_matched[k] < 0.5 * n]
    summary["phantom_tracks"] = len(fp_tracks)
    summary["phantom_lengths_s"] = [n / fps for _, n in fp_tracks]
    summary["seconds_of_video"] = sum(len(s) for s in seqs) / fps
    return summary, tracks, bursts, changes


def _per_track(per, fps):
    tracks, bursts, changes = [], [], []
    for (si, gid), d in sorted(per.items(), key=lambda x: (x[0][0], str(x[0][1]))):
        cls = max(set(d["cls"]), key=d["cls"].count)
        group = "mobility_aid" if cls in AID else "pedestrian"
        m = d["matched"]
        n = len(m)
        miss_runs = _runs([not x for x in m])
        for start, length in miss_runs:
            where = "start" if start == 0 else ("end" if start + length == n else "middle")
            bursts.append({"seq": si, "gt_id": gid, "group": group, "class": cls, "start_frame": d["frames"][start],
                           "frames": length, "seconds": length / fps, "where": where})
        pids = [p for p in d["pid"] if p is not None]
        # identity changes: matched to tracker ID a, then later to a different ID b; gap = unmatched frames between
        last_pid, last_k = None, None
        for k, p in enumerate(d["pid"]):
            if p is None:
                continue
            if last_pid is not None and p != last_pid:
                changes.append({"seq": si, "gt_id": gid, "group": group, "class": cls, "frame": d["frames"][k],
                                "from_id": last_pid, "to_id": p, "gap_frames": k - last_k - 1,
                                "gap_seconds": (k - last_k - 1) / fps})
            last_pid, last_k = p, k
        mpcls = [pc for pc in d["pcls"] if pc is not None]
        tracks.append({
            "seq": si, "gt_id": gid, "class": cls, "group": group, "frames": n, "seconds": n / fps,
            "detected_share": sum(m) / n,
            "miss_bursts": len(miss_runs),
            "longest_miss_s": max((l for _, l in miss_runs), default=0) / fps,
            "tracker_ids_used": len(set(pids)),
            "id_changes": sum(1 for c in changes if c["seq"] == si and c["gt_id"] == gid),
            "labelled_as_pedestrian_share": (sum(1 for pc in mpcls if pc == "pedestrian") / len(mpcls)) if mpcls else None,
            "correct_class_share": (sum(1 for pc, gc in zip(d["pcls"], d["cls"]) if pc is not None and pc == gc)
                                    / len(mpcls)) if mpcls else None,
        })
    return tracks, bursts, changes


def calibration(summary, tracks, bursts, changes, sim_dt: float = 0.1, persistence_s: float = 1.0) -> dict:
    """Turn the measurements into the Experiment B failure settings (configs/config.yaml 'failures')."""
    out = {"note": ("Measured on real PMMA video with the trained detector + ByteTrack. Values per group. "
                    "'failures_for_experiment_b' uses the MOBILITY-AID group because Experiment B measures "
                    "cut-offs of mobility-aid users. False positives are counted over the whole image, not only "
                    "on a crossing, so the phantom rate is an upper bound for the simulation.")}
    for group in ("mobility_aid", "pedestrian", "all"):
        T = [t for t in tracks if group == "all" or t["group"] == group]
        B = [b for b in bursts if group == "all" or b["group"] == group]
        C = [c for c in changes if group == "all" or c["group"] == group]
        if not T:
            out[group] = None
            continue
        frames = sum(t["frames"] for t in T)
        det = sum(t["detected_share"] * t["frames"] for t in T) / frames
        mid = [b["seconds"] for b in B if b["where"] == "middle"]
        allb = [b["seconds"] for b in B]
        mis = [t["labelled_as_pedestrian_share"] for t in T if t["labelled_as_pedestrian_share"] is not None]
        gaps = [c["gap_seconds"] for c in C]
        out[group] = {
            "people": len(T), "frames": frames,
            "recall": det,
            "mean_miss_burst_s": float(np.mean(allb)) if allb else 0.0,
            "median_miss_burst_s": float(np.median(allb)) if allb else 0.0,
            "mean_mid_track_miss_burst_s": float(np.mean(mid)) if mid else 0.0,
            "share_of_bursts_longer_than_persistence": (sum(1 for s in allb if s > persistence_s) / len(allb))
            if allb else 0.0,
            "longest_miss_s": max(allb, default=0.0),
            "misclassification_prob": (sum(t["labelled_as_pedestrian_share"] * t["frames"] for t in T
                                           if t["labelled_as_pedestrian_share"] is not None) /
                                       max(sum(t["frames"] for t in T if t["labelled_as_pedestrian_share"] is not None), 1))
            if mis else 0.0,
            "share_people_with_id_change": sum(1 for t in T if t["id_changes"] > 0) / len(T),
            "id_changes_per_person": sum(t["id_changes"] for t in T) / len(T),
            "id_change_gap_s_p25_p75": [float(np.percentile(gaps, 25)), float(np.percentile(gaps, 75))] if gaps else None,
        }
    g = out.get("mobility_aid") or out.get("all")
    lengths = summary.get("phantom_lengths_s") or []
    out["failures_for_experiment_b"] = {
        "recall": round(g["recall"], 3),
        "miss_burst_frames": round(max(1.0, g["mean_miss_burst_s"] / sim_dt), 1),
        "misclassification_prob": round(g["misclassification_prob"], 3),
        "track_loss_prob": round(g["share_people_with_id_change"], 3),
        "track_loss_duration_s": ([round(max(sim_dt, g["id_change_gap_s_p25_p75"][0]), 2),
                                   round(max(2 * sim_dt, g["id_change_gap_s_p25_p75"][1]), 2)]
                                  if g["id_change_gap_s_p25_p75"] else [3.0, 15.0]),
        "false_positive_rate": round(summary["phantom_tracks"] / max(summary["seconds_of_video"], 1e-9), 4),
        "false_positive_duration_s": ([round(float(np.percentile(lengths, 25)), 2),
                                       round(max(float(np.percentile(lengths, 75)), sim_dt), 2)]
                                      if lengths else [0.3, 1.5]),
    }
    out["simulation_assumptions_before"] = {"miss_burst_s": 8 * sim_dt, "track_loss_duration_s": [3.0, 15.0]}
    return out
