"""Simulated junction around the SafeCross crossing (follow-up study).

The single-crossing simulation (sim.py + controller.py) decides WHEN the walk phase ends.
This module places that walk phase inside a simple two-stage junction and measures what the
decision means for the rest of the junction:

  Stage A  parallel road green + our walk phase. Cars from the parallel road may turn into the
           cross road and drive over our crossing (permissive or gated by the controller).
  clear    all-red intergreen
  Stage B  cross road green; its cars drive straight over our crossing.
  clear    all-red intergreen

Rules (configs/junction.yaml):
  * Stage A lasts max(normal parallel green, end of the walk phase). A longer walk phase only costs
    the junction time when it runs past the normal parallel green.
  * The extra time comes out of stage B, but stage B never drops below its minimum green; if it would,
    the cycle gets longer instead.

Safety measures (ground truth, hidden from the controllers):
  * cross-road conflict : a person is still on the crossing when the first cross-road car reaches it
  * turning conflict    : a turning car reaches the crossing while a person is on it
Efficiency measures:
  * extra delay per cross-road car compared with the normal fixed timing
  * cross-road green lost, cycle lengthened, mean delay of turning cars
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from .controller import EpisodeResult
from .sim import Episode

ROOT = Path(__file__).resolve().parent.parent


def load_junction_config(path: str | Path | None = None) -> dict:
    p = Path(path) if path else ROOT / "configs" / "junction.yaml"
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def stage_times(walk_end: float, j: dict) -> dict:
    """Signal timings of one cycle, given when the controller ended the walk phase."""
    I = j["intergreen_s"]
    a_end = max(j["parallel_green_s"], walk_end)
    b_start = a_end + I
    b_green = max(j["min_green_cross_s"], j["cycle_s"] - a_end - 2 * I)
    cycle = a_end + I + b_green + I
    b_normal = j["cycle_s"] - j["parallel_green_s"] - 2 * I
    return {"walk_end": walk_end, "a_end": a_end, "b_start": b_start, "b_green": b_green,
            "b_end": b_start + b_green, "cycle": cycle, "cycle_extra": cycle - j["cycle_s"],
            "green_lost": max(0.0, b_normal - b_green),
            "next_b_start": cycle + j["parallel_green_s"] + I, "b_normal": b_normal}


def _arrivals(rng: np.random.Generator, rate_vph: float, t0: float, t1: float) -> np.ndarray:
    if t1 <= t0 or rate_vph <= 0:
        return np.zeros(0)
    n = rng.poisson(rate_vph / 3600.0 * (t1 - t0))
    return np.sort(rng.uniform(t0, t1, n))


def cross_road_delay(arrivals: np.ndarray, st: dict, j: dict) -> float:
    """Mean stop-line delay (s) of cross-road cars that arrive during this cycle (one lane)."""
    if len(arrivals) == 0:
        return 0.0
    h = 3600.0 / j["cross_road"]["saturation_vph"]
    windows = [(st["b_start"], st["b_end"]), (st["next_b_start"], st["next_b_start"] + st["b_normal"])]
    nxt = windows[0][0]
    delays = []
    for a in arrivals:
        d = max(a, nxt)
        for w0, w1 in windows:
            if d < w0:
                d = w0
            if d < w1:
                break
        else:                                  # still unserved after two green windows (rare)
            d = windows[-1][1]
        delays.append(d - a)
        nxt = d + h
    return float(np.mean(delays))


def _on_crossing(ep: Episode, i: int, t: float) -> bool:
    k = min(len(ep.times) - 1, max(0, int(round(t / ep.dt))))
    return ep.on_crossing(i, k)


def turning_cars(ep: Episode, res: EpisodeResult, st: dict, j: dict, mode: str,
                 arrivals: np.ndarray) -> tuple[np.ndarray, float]:
    """Times at which turning cars reach the crossing during stage A, and their mean delay."""
    tj = j["turning"]
    h = tj["headway_s"]
    occ = res.occupancy
    releases, delays = [], []
    nxt = 0.0
    for a in arrivals:
        t = max(a, nxt)
        if mode == "gated":
            # arrow green only after the leading interval, and while the controller reports CLEAR
            # (or after the walk phase has ended)
            while t < st["a_end"]:
                k = int(round(t / ep.dt))
                state = occ[k] if k < len(occ) else "ended"
                if t >= tj["leading_interval_s"] and state in ("clear", "ended"):
                    break
                t += ep.dt
        if t < st["a_end"]:
            releases.append(t)
            delays.append(t - a)
            nxt = t + h
        else:                                   # not served in this stage: waits for the next cycle
            delays.append(st["cycle"] + h * (len(delays) - len(releases)) - a)
    return np.asarray(releases), (float(np.mean(delays)) if delays else 0.0)


def junction_metrics(ep: Episode, res: EpisodeResult, j: dict, seed: int) -> dict:
    """All junction measures for one walk phase decided by one controller."""
    st = stage_times(res.end_time, j)
    ref = stage_times(0.0, j)                   # normal fixed timing (walk inside the normal parallel green)
    rng = np.random.default_rng(seed)           # same cars for every controller (paired comparison)
    cross = _arrivals(rng, j["cross_road"]["arrival_vph"], 0.0, j["cycle_s"])
    turns = _arrivals(rng, j["turning"]["arrival_vph"], 0.0, 40.0 + j["parallel_green_s"])
    turns = turns[turns < st["a_end"]]

    out = {"walk_end": res.end_time, "stage_a_end": st["a_end"], "cross_green_start": st["b_start"],
           "cross_green_s": st["b_green"], "cross_green_lost_s": st["green_lost"],
           "cycle_extra_s": st["cycle_extra"],
           "cross_delay_s": cross_road_delay(cross, st, j),
           "cross_delay_ref_s": cross_road_delay(cross, ref, j)}
    out["cross_extra_delay_s"] = out["cross_delay_s"] - out["cross_delay_ref_s"]

    reach = st["b_start"] + j["startup_s"]
    eligible = [i for i, p in enumerate(ep.pedestrians) if p.t_start < res.end_time]
    aid = [i for i in eligible if ep.pedestrians[i].is_mobility_aid]
    out["aid_crossings"] = len(aid)
    out["all_crossings"] = len(eligible)
    out["aid_cross_conflicts"] = sum(_on_crossing(ep, i, reach) for i in aid)
    out["all_cross_conflicts"] = sum(_on_crossing(ep, i, reach) for i in eligible)

    for mode in j["turn_modes"]:
        rel, delay = turning_cars(ep, res, st, j, mode, turns)
        exposed = [i for i in aid if any(_on_crossing(ep, i, t) for t in rel)]
        out[f"{mode}_aid_turn_conflicts"] = len(exposed)
        out[f"{mode}_cars_conflicting"] = int(sum(any(_on_crossing(ep, i, t) for i in eligible) for t in rel))
        out[f"{mode}_turn_cars"] = int(len(turns))
        out[f"{mode}_turn_delay_s"] = delay
    return out


def junction_figure(ep: Episode, results: dict, j: dict, seed: int, mode: str = "gated"):
    """One row per controller: the junction's signal stages, people on the crossing, turning cars."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results)
    fig, axes = plt.subplots(len(names), 1, figsize=(10, 1.9 * len(names) + 0.6), dpi=110, sharex=True)
    axes = np.atleast_1d(axes)
    rng = np.random.default_rng(seed)
    _arrivals(rng, j["cross_road"]["arrival_vph"], 0.0, j["cycle_s"])
    turns_all = _arrivals(rng, j["turning"]["arrival_vph"], 0.0, 40.0 + j["parallel_green_s"])
    n = len(ep.pedestrians)
    for ax, c in zip(axes, names):
        res = results[c]
        st = stage_times(res.end_time, j)
        y0 = n + 0.6
        ax.barh(y0, res.end_time, left=0, height=0.5, color="#3C7A4E")
        ax.barh(y0, st["a_end"] - res.end_time, left=res.end_time, height=0.5, color="#B8B8B8")
        ax.barh(y0, j["intergreen_s"], left=st["a_end"], height=0.5, color="#C0392B")
        ax.barh(y0, st["b_green"], left=st["b_start"], height=0.5, color="#2F5D8A")
        ax.text(0.3, y0, "walk", va="center", color="white", fontsize=8)
        ax.text(st["b_start"] + 0.3, y0, "cross-road cars go", va="center", color="white", fontsize=8)
        for i, p in enumerate(ep.pedestrians):
            t_exit = min(ep.exit_time(i), 60.0)
            col = "#7A1F2B" if p.is_mobility_aid else "#9A9A9A"
            ax.barh(i, t_exit - p.t_start, left=p.t_start, height=0.55, color=col)
            ax.text(p.t_start, i, f" P{p.pid} {p.cls}", va="center", fontsize=7,
                    color="white" if p.is_mobility_aid else "black")
        rel, _ = turning_cars(ep, res, st, j, mode, turns_all[turns_all < st["a_end"]])
        for t in rel:
            hit = any(_on_crossing(ep, i, t) for i, p in enumerate(ep.pedestrians) if p.is_mobility_aid)
            ax.plot(t, -0.9, marker="v", color="#C0392B" if hit else "#444444", ms=8 if hit else 6)
        reach = st["b_start"] + j["startup_s"]
        ax.axvline(reach, color="#2F5D8A", ls="--", lw=1)
        ax.set_yticks([])
        ax.set_ylim(-1.5, n + 1.1)
        ax.set_title(f"{c}: walk ends {res.end_time:.1f}s · cross-road green {st['b_green']:.0f}s "
                     f"· turning cars ({mode}) ▼, red = meets a mobility-aid user", fontsize=9, loc="left")
    axes[-1].set_xlabel("time since the walk phase started (s)")
    axes[-1].set_xlim(0, 60)
    fig.tight_layout()
    return fig
