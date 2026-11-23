"""Match detections to ground truth and compute per-class precision / recall.

We use our own simple matcher (IoU >= 0.5, same class, greedy by confidence)
so that YOLO and YOLO-World are evaluated in exactly the same way.
"""
from __future__ import annotations

from collections import defaultdict


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def match(dets: list[dict], gts: list[dict], thr: float = 0.5, class_aware: bool = True):
    """Return (true_positive_pairs, missed_gts, false_positive_dets)."""
    used = set()
    pairs = []
    fps = []
    for d in sorted(dets, key=lambda x: -x.get("conf", 1.0)):
        best, best_j = 0.0, -1
        for j, g in enumerate(gts):
            if j in used or (class_aware and g["cls"] != d["cls"]):
                continue
            v = iou(d["box"], g["box"])
            if v > best:
                best, best_j = v, j
        if best >= thr:
            used.add(best_j)
            pairs.append((d, gts[best_j]))
        else:
            fps.append(d)
    missed = [g for j, g in enumerate(gts) if j not in used]
    return pairs, missed, fps


def image_counts(dets, gts, thr=0.5) -> dict:
    """Per-class TP / FN / FP counts for one image, plus class-agnostic person counts
    and the number of mobility-aid users detected as some person but with the wrong class."""
    counts = defaultdict(lambda: {"tp": 0, "fn": 0, "fp": 0})
    pairs, missed, fps = match(dets, gts, thr, class_aware=True)
    for d, g in pairs:
        counts[g["cls"]]["tp"] += 1
    for g in missed:
        counts[g["cls"]]["fn"] += 1
    for d in fps:
        counts[d["cls"]]["fp"] += 1
    p_any, m_any, _ = match(dets, gts, thr, class_aware=False)
    for d, g in p_any:
        counts[g["cls"]]["tp_any"] = counts[g["cls"]].get("tp_any", 0) + 1
    for g in m_any:
        counts[g["cls"]]["fn_any"] = counts[g["cls"]].get("fn_any", 0) + 1
    misclassified = sum(1 for d, g in p_any if d["cls"] != g["cls"])
    return {"per_class": dict(counts), "person_tp_any_class": len(p_any),
            "person_missed_any_class": len(m_any), "misclassified": misclassified}


def summarise(per_image: list[dict], mobility_classes) -> dict:
    tot = defaultdict(lambda: {"tp": 0, "fn": 0, "fp": 0})
    for r in per_image:
        for c, v in r["per_class"].items():
            for k in v:
                tot[c][k] = tot[c].get(k, 0) + v[k]
    out = {}
    for c, v in tot.items():
        rec = v["tp"] / (v["tp"] + v["fn"]) if v["tp"] + v["fn"] else float("nan")
        prec = v["tp"] / (v["tp"] + v["fp"]) if v["tp"] + v["fp"] else float("nan")
        out[c] = {"recall": rec, "precision": prec, **v}
    aid = {k: sum(tot[c][k] for c in mobility_classes if c in tot) for k in ("tp", "fn", "fp")}
    out["ALL_MOBILITY_AIDS"] = {
        "recall": aid["tp"] / (aid["tp"] + aid["fn"]) if aid["tp"] + aid["fn"] else float("nan"),
        "precision": aid["tp"] / (aid["tp"] + aid["fp"]) if aid["tp"] + aid["fp"] else float("nan"), **aid}
    return out
