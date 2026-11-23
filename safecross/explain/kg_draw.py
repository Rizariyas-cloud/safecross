"""Draw the decision knowledge graph for the app (Ask SafeCross page).

Layout (left -> right):   controller decisions (event log)  ->  walk phase  <-  camera tracks
                                                   rules on top
Facts the AI cited are outlined in gold, so you can see which part of the graph an answer used.
"""
from __future__ import annotations

import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

MAR, GREEN, INK, MUTE, GOLD, LINE = "#8C1D2C", "#2F8F4E", "#262626", "#6B6B6B", "#C9A227", "#D9D9D9"


def _box(ax, x, y, text, face, edge, w=0.2, h=0.075, gold=False, color="white", size=8.5):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
                                fc=face, ec=GOLD if gold else edge, lw=3.2 if gold else 1.2, zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=size, color=color, zorder=4, linespacing=1.25)


def _arrow(ax, a, b, label="", color=MUTE, lw=1.1, gold=False, rad=0.0, size=7.5):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11, lw=3 if gold else lw,
                                 color=GOLD if gold else color, connectionstyle=f"arc3,rad={rad}", zorder=2))
    if label:
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.018, label, ha="center", va="bottom", fontsize=size,
                color=GOLD if gold else color, zorder=5,
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.6))


def draw_kg(facts, path: str, cited: set[str] | None = None, title: str = "") -> str:
    cited = cited or set()
    by_s: dict[str, list] = {}
    for f in facts:
        by_s.setdefault(f.s, []).append(f)
    gold_nodes = {f.s for f in facts if f.gid in cited} | {f.o for f in facts if f.gid in cited}

    decisions = [s for s in by_s if re.match(r"L\d+ ", s)]
    decisions.sort(key=lambda s: int(s.split()[0][1:]))
    involved = set()
    for f in facts:
        if f.p in ("caused_by", "extended_by") and f.o.startswith("track"):
            involved.add(f.o)
        if f.s.startswith("track") and f.p in ("only_occupant", "kept_crossing_occupied", "lost_by_camera"):
            involved.add(f.s)
    tracks = sorted(involved, key=lambda s: int(s.split()[1]))[:9]

    fig, ax = plt.subplots(figsize=(12.5, 7.2), dpi=130)
    ax.set_xlim(0, 1); ax.set_ylim(-0.03, 1.04); ax.axis("off")
    ax.text(0.01, 1.035, title or "Decision knowledge graph of this walk phase", fontsize=12, color=INK,
            fontweight="bold", va="top")
    ax.text(0.01, 0.995, "built automatically from the controller trace  ·  thin grey line = caused by  ·  gold = facts the answer cited",
            fontsize=8.5, color=MUTE, va="top")

    # rules (top centre)
    rules = [f for f in facts if "(rule" in f.o or "rule)" in f.o]
    rule_txt = "\n".join(re.sub(r"\s*\(rule.*?\)$", "", f"{f.p.replace('_', ' ')}: {f.o}")[:72] for f in rules[:4])
    _box(ax, 0.5, 0.875, "RULES\n" + rule_txt, "#F6F6F6", LINE, w=0.46, h=0.13, color=INK, size=7.2,
         gold=any(f.gid in cited for f in rules))

    # walk phase (centre)
    ended = next((f.o for f in facts if f.p == "ended_at"), "")
    phase = (0.5, 0.45)
    end_t, _, why = ended.split(" (see")[0].partition(" because ")
    _box(ax, *phase, f"WALK PHASE\nended at {end_t}\n{why.replace('_', ' ')}", INK, INK, w=0.2, h=0.13,
         gold="walk phase" in gold_nodes, size=9)
    _arrow(ax, (0.5, 0.808), (0.5, 0.517), "rules", color=MUTE)

    # decisions (left column)
    n = max(1, len(decisions))
    dpos = {}
    for i, d in enumerate(decisions):
        y = 0.72 - i * (0.64 / max(1, n - 1)) if n > 1 else 0.45
        t = next((f.o for f in by_s[d] if f.p == "at_time"), "")
        r = next((f.o.split(" (")[0] for f in by_s[d] if f.p == "reason"), "")
        conf = next((f.o for f in by_s[d] if f.p == "confidence"), None)
        txt = f"{d}  t={t}\n{r.replace('_', ' ')}" + (f"\nconfidence {conf}" if conf else "")
        face = MAR if "END" in d else ("#B5555F" if ("EXTEND" in d or "HOLD" in d) else "#7A7A7A")
        dpos[d] = (0.14, y)
        _box(ax, 0.14, y, txt, face, face, w=0.25, h=0.085, gold=d in gold_nodes, size=7.6)
        if "PHASE_END" in d:
            _arrow(ax, (0.265, y), (0.4, 0.45), "ended the phase", color=MAR,
                   gold=any(f.gid in cited and f.p == "ended_at" for f in facts))

    # tracks (right column)
    m = max(1, len(tracks))
    for i, tname in enumerate(tracks):
        y = 0.72 - i * (0.64 / max(1, m - 1)) if m > 1 else 0.45
        fs = by_s.get(tname, [])
        lab = next((f.o for f in fs if f.p == "labelled_as"), "")
        aid = lab.startswith("mobility")
        lost = next((f.o for f in fs if f.p == "last_seen_at"), None)
        status = f"lost after {lost.split(' at ')[0]}" if any(f.p == "lost_by_camera" for f in fs) else \
            ("reached exit" if any(f.p == "reached_exit_at" for f in fs) else "seen")
        txt = f"{tname}  ·  {'mobility aid' if aid else 'pedestrian'}\n{status}"
        _box(ax, 0.86, y, txt, MAR if aid else GREEN, MAR if aid else GREEN, w=0.25, h=0.075,
             gold=tname in gold_nodes, size=7.8)
        only = next((f for f in fs if f.p == "only_occupant"), None)
        keep = next((f for f in fs if f.p == "kept_crossing_occupied"), None)
        if only or keep:
            f = only or keep
            lbl = ("ONLY occupant " if only else "kept occupied ") + f.o.split(" (")[0]
            _arrow(ax, (0.735, y), (0.6, 0.45), lbl, color=MAR if only else GREEN, lw=2.2 if only else 1.2,
                   gold=f.gid in cited, size=7.2)
        for d, (dx, dy) in dpos.items():
            cb = [f for f in by_s[d] if f.p == "caused_by" and f.o == tname]
            if cb:
                g = any(f.gid in cited for f in cb)
                ax.add_patch(FancyArrowPatch((0.265, dy), (0.735, y), arrowstyle="-|>", mutation_scale=9,
                                             lw=2.6 if g else 0.8, color=GOLD if g else LINE, zorder=1,
                                             connectionstyle="arc3,rad=-0.12"))
    ax.text(0.14, -0.02, "controller decisions (event log)", ha="center", fontsize=8.5, color=MUTE)
    ax.text(0.86, -0.02, "camera tracks (detector + tracker)", ha="center", fontsize=8.5, color=MUTE)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path
