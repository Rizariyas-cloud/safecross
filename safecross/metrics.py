"""Safety and efficiency metrics, plus clustered bootstrap confidence intervals.

Cut-off rate (main safety metric):
    number of mobility-aid crossings where the phase ended before the person
    passed the exit line, divided by the number of mobility-aid crossings.

A crossing is "eligible" if the person stepped onto the crossing before the
phase ended (people who start after the phase ended are not counted).

Efficiency metrics:
    false extension = phase ended more than `tol` seconds after it needed to
    added time      = seconds the phase ran longer than needed (0 if not longer)
"""
from __future__ import annotations

import numpy as np

from .controller import EpisodeResult, base_phase_length
from .sim import Episode


def episode_metrics(ep: Episode, res: EpisodeResult, cfg: dict, tol: float = 2.0) -> dict:
    base = base_phase_length(cfg)
    exits, aid_total, aid_cut, all_total, all_cut = [], 0, 0, 0, 0
    for i, p in enumerate(ep.pedestrians):
        if p.t_start >= res.end_time:
            continue
        t_exit = ep.exit_time(i)
        exits.append(t_exit)
        cut = t_exit > res.end_time + 1e-9
        all_total += 1
        all_cut += int(cut)
        if p.is_mobility_aid:
            aid_total += 1
            aid_cut += int(cut)
    needed = max([base] + [e for e in exits if np.isfinite(e)])
    needed = min(needed, cfg["controller"]["max_phase_s"])
    added = max(0.0, res.end_time - needed)
    return {
        "episode_id": ep.episode_id, "controller": res.controller, "end_time": res.end_time,
        "needed_time": round(needed, 1), "added_time": round(added, 2),
        "false_extension": int(added > tol),
        "aid_crossings": aid_total, "aid_cutoffs": aid_cut,
        "all_crossings": all_total, "all_cutoffs": all_cut,
    }


def cutoff_rate(rows: list[dict]) -> float:
    total = sum(r["aid_crossings"] for r in rows)
    return sum(r["aid_cutoffs"] for r in rows) / total if total else float("nan")


def cluster_bootstrap(rows: list[dict], stat_fn, cluster_key: str = "episode_id",
                      n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """95% CI that resamples whole clusters (episodes or video sequences), not single frames,
    because rows from the same cluster are correlated."""
    rng = np.random.default_rng(seed)
    clusters: dict = {}
    for r in rows:
        clusters.setdefault(r[cluster_key], []).append(r)
    keys = list(clusters)
    point = stat_fn(rows)
    if len(keys) < 2:
        return point, float("nan"), float("nan")
    stats = []
    for _ in range(n_boot):
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        sample = [r for j in pick for r in clusters[keys[j]]]
        v = stat_fn(sample)
        if np.isfinite(v):
            stats.append(v)
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)]) if stats else (np.nan, np.nan)
    return float(point), float(lo), float(hi)


def mcnemar(a_correct: list[bool], b_correct: list[bool]) -> dict:
    """McNemar's test for two systems answering the SAME items (exact binomial version)."""
    from scipy.stats import binomtest
    b = sum(1 for x, y in zip(a_correct, b_correct) if x and not y)
    c = sum(1 for x, y in zip(a_correct, b_correct) if y and not x)
    p = binomtest(b, b + c, 0.5).pvalue if (b + c) else 1.0
    return {"a_only": b, "b_only": c, "p_value": float(p)}
