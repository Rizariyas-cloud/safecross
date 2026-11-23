"""STEP 4 - Experiment B: decision sensitivity (answers RQ2). Runs on any laptop (no GPU).

Injects each failure type at increasing levels into the controller's input and
measures the cut-off rate (safety) and added time / false extensions (efficiency)
for the baseline and the uncertainty-aware controller.

    python scripts/04_experiment_b.py                     # synthetic walkers
    python scripts/04_experiment_b.py --episodes 50       # quicker test
    python scripts/04_experiment_b.py --trajectories data/pmma_trajectories.json   # real PMMA speeds
    python scripts/04_experiment_b.py --calibration results/tracking/<tag>/calibration.json --only-calibrated
        # CALIBRATED run: all failures at the levels MEASURED on real video by scripts/20 (all at once)

Outputs in results/experiment_b/: episodes.csv, summary.csv, cutoff_vs_level.png
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from safecross.config import load_config, resolve
from safecross.controller import run_controller
from safecross.metrics import episode_metrics, cutoff_rate, cluster_bootstrap
from safecross.perception import perceive
from safecross.sim import generate_episode, load_trajectory_bank

FAILURES = {   # name in config -> (failure key, label for plots)
    "recall_levels": ("recall", "False negatives (detection recall)"),
    "misclassification_levels": ("misclassification_prob", "Misclassification probability"),
    "track_loss_levels": ("track_loss_prob", "Track-loss probability"),
    "false_positive_levels": ("false_positive_rate", "False positives per second"),
}


def main():
    cfg = load_config()
    xb = cfg["experiment_b"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=xb["episodes"])
    ap.add_argument("--trajectories", default=None, help="JSON from scripts/08 (real walking speeds)")
    ap.add_argument("--calibration", default=None, help="calibration.json from scripts/20 (measured failures)")
    ap.add_argument("--only-calibrated", action="store_true", help="skip the level sweeps, run only the calibrated setting")
    ap.add_argument("--out", default="experiment_b", help="results sub-folder")
    args = ap.parse_args()
    bank = load_trajectory_bank(args.trajectories) if args.trajectories else None

    settings = []                                   # (failure name, level, failure settings)
    if not args.only_calibrated:
        for level_key, (fkey, _) in FAILURES.items():
            for level in xb[level_key]:
                fcfg = dict(cfg["failures"])
                fcfg[fkey] = level
                settings.append((fkey, level, fcfg))
    if args.calibration:
        import json
        cal = json.load(open(resolve(args.calibration)))["failures_for_experiment_b"]
        fcfg = dict(cfg["failures"])
        fcfg.update(cal)
        print("Calibrated failure settings (measured on real video):", cal)
        settings.append(("calibrated", 1.0, fcfg))
        nofp = dict(fcfg)
        nofp["false_positive_rate"] = 0.0       # phantoms were counted over the whole image: an upper bound
        settings.append(("calibrated_without_phantoms", 1.0, nofp))
        for k in ("recall", "track_loss_prob"):    # sensitivity: each measured failure alone
            one = dict(cfg["failures"])
            one[k] = cal[k]
            if k == "track_loss_prob":
                one["track_loss_duration_s"] = cal["track_loss_duration_s"]
            else:
                one["miss_burst_frames"] = cal["miss_burst_frames"]
            settings.append((f"calibrated_{k}_only", 1.0, one))

    controllers = ["baseline", "uncertainty_aware"]
    if args.calibration:                        # the ghost filter matters once real phantom rates are used
        controllers.append("uncertainty_aware_filtered")
    rows = []
    for fkey, level, fcfg in settings:
        # the SAME episodes are used for every level and controller (paired comparison)
        rng = np.random.default_rng(cfg["seed"])
        for e in range(args.episodes):
            ep = generate_episode(e, cfg, rng, bank)
            obs = perceive(ep, fcfg, np.random.default_rng(cfg["seed"] * 7919 + e))
            for cname in controllers:
                res = run_controller(cname, cfg, ep, obs)
                m = episode_metrics(ep, res, cfg, xb["false_extension_tolerance_s"])
                rows.append({"failure": fkey, "level": level, **m})
        print(f"done: {fkey} = {level}")

    out = resolve(cfg["paths"]["results_dir"]) / args.out
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "episodes.csv", index=False)

    summ = []
    for (f, lvl, c), g in df.groupby(["failure", "level", "controller"]):
        recs = g.to_dict("records")
        cr, lo, hi = cluster_bootstrap(recs, cutoff_rate, "episode_id", xb["bootstrap_samples"], cfg["seed"])
        summ.append({"failure": f, "level": lvl, "controller": c, "cutoff_rate": cr, "ci_low": lo, "ci_high": hi,
                     "mean_added_time_s": g["added_time"].mean(),
                     "false_extension_rate": g["false_extension"].mean(),
                     "aid_crossings": int(g["aid_crossings"].sum())})
    sdf = pd.DataFrame(summ)
    sdf.to_csv(out / "summary.csv", index=False)
    print(sdf.round(3).to_string(index=False))
    if args.only_calibrated:
        print(f"\nSaved results to {out}")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), dpi=120)
    colours = {"baseline": "#B9482A", "uncertainty_aware": "#3C7A4E"}
    for ax, (level_key, (fkey, label)) in zip(axes, FAILURES.items()):
        for c in ("baseline", "uncertainty_aware"):
            s = sdf[(sdf.failure == fkey) & (sdf.controller == c)].sort_values("level")
            ax.plot(s.level, s.cutoff_rate, "o-", color=colours[c], label=c)
            ax.fill_between(s.level, s.ci_low, s.ci_high, color=colours[c], alpha=0.15)
        ax.set_xlabel(label)
        ax.set_ylabel("cut-off rate")
        if fkey == "recall":
            ax.invert_xaxis()
    axes[0].legend(fontsize=8)
    fig.suptitle("Experiment B (simulated): how perception failures change the cut-off rate of mobility-aid users")
    fig.tight_layout()
    fig.savefig(out / "cutoff_vs_level.png")
    print(f"\nSaved results to {out}")


if __name__ == "__main__":
    main()
