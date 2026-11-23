"""STEP 13 (follow-up) - the crossing inside a simulated junction. Runs on any laptop (no GPU, ~10 min).

Question: when the controller changes the walk phase, what happens to the REST of the junction -
cars on the cross road, and cars turning over the crossing? Does it create conflicts with cars?

For the same simulated walk phases as Experiment B, each controller's decision is placed inside a
two-stage junction (configs/junction.yaml) and we measure:
  * cut-off rate                 (as in Experiment B)
  * cross-road conflict rate     share of mobility-aid users still on the crossing when cross-road cars reach it
  * turning conflict rate        share of mobility-aid users who meet a turning car on the crossing,
                                 with permissive turns and with turns gated by the controller
  * extra delay per cross-road car, cross-road green lost, longer cycles, turning-car delay

Three controllers: baseline, uncertainty_aware, and uncertainty_aware_filtered (with a ghost filter
against phantom detections). Everything here is SIMULATED; report it as a follow-up simulation.

    python scripts/13_junction_experiment.py
    python scripts/13_junction_experiment.py --episodes 50     # quick test

Outputs in results/junction/: episodes.csv, summary.csv, junction_safety.png, junction_cost.png
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from safecross.config import load_config, resolve
from safecross.controller import run_controller
from safecross.junction import load_junction_config, junction_metrics
from safecross.metrics import episode_metrics, cluster_bootstrap
from safecross.perception import perceive
from safecross.sim import generate_episode, load_trajectory_bank

FAILURES = {"recall_levels": ("recall", "Detection recall"),
            "false_positive_levels": ("false_positive_rate", "False positives per second"),
            "track_loss_levels": ("track_loss_prob", "Track-loss probability")}
LABELS = {"baseline": "Baseline", "uncertainty_aware": "Uncertainty-aware",
          "uncertainty_aware_filtered": "Uncertainty-aware + ghost filter"}
COLOURS = {"baseline": "#B9482A", "uncertainty_aware": "#3C7A4E", "uncertainty_aware_filtered": "#2F5D8A"}


def ratio(num: str, den: str = "aid_crossings"):
    def f(rows):
        d = sum(r[den] for r in rows)
        return sum(r[num] for r in rows) / d if d else float("nan")
    return f


def main():
    cfg = load_config()
    j = load_junction_config()
    cfg["controller"].update(j.get("ghost_filter", {}))
    xj = j["experiment"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=xj["episodes"])
    ap.add_argument("--trajectories", default=None, help="JSON from scripts/08 (real walking speeds)")
    args = ap.parse_args()
    bank = load_trajectory_bank(args.trajectories) if args.trajectories else None

    rows = []
    for level_key, (fkey, _) in FAILURES.items():
        for level in xj[level_key]:
            fcfg = dict(cfg["failures"])
            fcfg[fkey] = level
            rng = np.random.default_rng(cfg["seed"])            # same walk phases as Experiment B
            for e in range(args.episodes):
                ep = generate_episode(e, cfg, rng, bank)
                obs = perceive(ep, fcfg, np.random.default_rng(cfg["seed"] * 7919 + e))
                for cname in xj["controllers"]:
                    res = run_controller(cname, cfg, ep, obs)
                    m = episode_metrics(ep, res, cfg)
                    jm = junction_metrics(ep, res, j, seed=cfg["seed"] * 104729 + e)
                    jm.pop("aid_crossings"), jm.pop("all_crossings")
                    rows.append({"failure": fkey, "level": level, **m, **jm})
            print(f"done: {fkey} = {level}")

    out = resolve(cfg["paths"]["results_dir"]) / "junction"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "episodes.csv", index=False)

    stats = {"cutoff_rate": ratio("aid_cutoffs"),
             "cross_conflict_rate": ratio("aid_cross_conflicts")}
    for mode in j["turn_modes"]:
        stats[f"{mode}_turn_conflict_rate"] = ratio(f"{mode}_aid_turn_conflicts")
    summ = []
    for (f, lvl, c), g in df.groupby(["failure", "level", "controller"], sort=False):
        recs = g.to_dict("records")
        row = {"failure": f, "level": lvl, "controller": c}
        for name, fn in stats.items():
            v, lo, hi = cluster_bootstrap(recs, fn, "episode_id", xj["bootstrap_samples"], cfg["seed"])
            row[name], row[name + "_lo"], row[name + "_hi"] = v, lo, hi
        row.update({"walk_end_s": g["walk_end"].mean(),
                    "cross_extra_delay_s": g["cross_extra_delay_s"].mean(),
                    "cross_green_lost_s": g["cross_green_lost_s"].mean(),
                    "longer_cycle_share": (g["cycle_extra_s"] > 0).mean(),
                    **{f"{m}_turn_delay_s": g[f"{m}_turn_delay_s"].mean() for m in j["turn_modes"]},
                    "aid_crossings": int(g["aid_crossings"].sum())})
        summ.append(row)
    sdf = pd.DataFrame(summ)
    sdf.to_csv(out / "summary.csv", index=False)
    show = ["failure", "level", "controller", "cutoff_rate", "cross_conflict_rate"] + \
           [f"{m}_turn_conflict_rate" for m in j["turn_modes"]] + ["cross_extra_delay_s", "cross_green_lost_s"] + \
           [f"{m}_turn_delay_s" for m in j["turn_modes"]]
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(sdf[show].round(3).to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def panel(ax, fkey, metric, ylabel, invert=False):
        for c in xj["controllers"]:
            s = sdf[(sdf.failure == fkey) & (sdf.controller == c)].sort_values("level")
            ax.plot(s.level, s[metric], "o-", color=COLOURS.get(c), label=LABELS.get(c, c))
            if metric + "_lo" in s:
                ax.fill_between(s.level, s[metric + "_lo"], s[metric + "_hi"], color=COLOURS.get(c), alpha=0.12)
        ax.set_ylabel(ylabel)
        if invert:
            ax.invert_xaxis()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=120)
    panel(axes[0], "recall", "cross_conflict_rate", "still on crossing when cross-road cars arrive", True)
    panel(axes[1], "recall", "permissive_turn_conflict_rate", "meets a turning car (permissive turns)", True)
    panel(axes[2], "recall", "gated_turn_conflict_rate", "meets a turning car (turns gated by controller)", True)
    for ax in axes:
        ax.set_xlabel("Detection recall")
    axes[0].legend(fontsize=8)
    fig.suptitle("Junction study (simulated): conflicts between mobility-aid users and cars")
    fig.tight_layout()
    fig.savefig(out / "junction_safety.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=120)
    panel(axes[0], "false_positive_rate", "cross_extra_delay_s", "extra delay per cross-road car (s)")
    panel(axes[1], "false_positive_rate", "gated_turn_delay_s", "turning-car delay, gated (s)")
    for ax in axes:
        ax.set_xlabel("False positives per second")
    axes[0].legend(fontsize=8)
    fig.suptitle("Junction study (simulated): what the walk-phase decision costs other traffic")
    fig.tight_layout()
    fig.savefig(out / "junction_cost.png")
    print(f"\nSaved results to {out}")


if __name__ == "__main__":
    main()
