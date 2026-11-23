"""Simulated camera output with controlled perception failures (Experiment B).

Instead of running a real detector, we take the ground truth of an episode and
deliberately damage it in four defined ways:

1. False negatives   - a person is not detected in some frames (controlled by recall).
2. Misclassification - a mobility-aid user is labelled as an ordinary pedestrian.
3. Track loss        - the tracker loses a person and later gives them a NEW id.
4. False positives   - a person who does not exist is "detected" on the crossing.

The result is a list of observations per time step, exactly what a detector +
tracker would hand to the signal controller.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sim import Episode, MOBILITY_AID_CLASSES


@dataclass
class Observation:
    track_id: int        # id given by the (simulated) tracker
    cls: str             # predicted class
    y: float             # position along the crossing (m)
    x: float
    conf: float          # detector confidence
    gt_pid: int          # true person id (-1 for a phantom detection); never shown to controllers


def _miss_mask(rng: np.random.Generator, n: int, recall: float, burst: float) -> np.ndarray:
    """True where the person is MISSED. Misses come in bursts (two-state Markov chain)
    so that the long-run detection rate equals `recall`."""
    if recall >= 1.0:
        return np.zeros(n, dtype=bool)
    if recall <= 0.0:
        return np.ones(n, dtype=bool)
    burst = max(1.0, burst)
    p_recover = 1.0 / burst
    p_miss = (1.0 - recall) * p_recover / recall
    p_miss = min(p_miss, 1.0)
    mask = np.zeros(n, dtype=bool)
    missed = rng.random() < (1.0 - recall)
    for k in range(n):
        mask[k] = missed
        missed = (rng.random() >= p_recover) if missed else (rng.random() < p_miss)
    return mask


def perceive(ep: Episode, fcfg: dict, rng: np.random.Generator) -> list[list[Observation]]:
    """Return observations[k] = list of Observation at time step k."""
    n_steps = len(ep.times)
    obs: list[list[Observation]] = [[] for _ in range(n_steps)]
    next_id = 1
    for i, p in enumerate(ep.pedestrians):
        # visible = on the kerb about to cross, on the crossing, or just past the exit
        visible = (ep.positions[i] > -1.0) & (ep.positions[i] < ep.length + 1.5)
        missed = _miss_mask(rng, n_steps, fcfg.get("recall", 1.0), fcfg.get("miss_burst_frames", 8))

        # track loss: pick a moment while the person is on the crossing
        lost = np.zeros(n_steps, dtype=bool)
        new_id_from = None
        if rng.random() < fcfg.get("track_loss_prob", 0.0):
            on = np.where((ep.positions[i] >= 0.5) & (ep.positions[i] < ep.length - 1.0))[0]
            if len(on):
                k_lost = int(rng.choice(on))
                lo, hi = fcfg.get("track_loss_duration_s", [3.0, 15.0])
                dur = int(rng.uniform(lo, hi) / ep.dt)
                lost[k_lost:k_lost + dur] = True
                new_id_from = k_lost + dur                 # re-detected with a new id afterwards

        track_id = next_id
        next_id += 1
        base_conf = rng.uniform(0.55, 0.95)
        for k in range(n_steps):
            if not visible[k] or missed[k] or lost[k]:
                continue
            if new_id_from is not None and k >= new_id_from:
                track_id = next_id                         # tracker starts a new identity
                next_id += 1
                new_id_from = None
            cls = p.cls
            conf = float(np.clip(base_conf + rng.normal(0, 0.05), 0.05, 0.99))
            if p.cls in MOBILITY_AID_CLASSES and rng.random() < fcfg.get("misclassification_prob", 0.0):
                cls = "pedestrian"
                conf *= 0.8
            obs[k].append(Observation(track_id, cls, float(ep.positions[i, k]), p.x, conf, p.pid))

    # phantom detections
    rate = fcfg.get("false_positive_rate", 0.0)
    if rate > 0:
        n_fp = rng.poisson(rate * n_steps * ep.dt)
        lo, hi = fcfg.get("false_positive_duration_s", [0.3, 1.5])
        for _ in range(n_fp):
            k0 = int(rng.integers(0, n_steps))
            dur = max(1, int(rng.uniform(lo, hi) / ep.dt))
            y = rng.uniform(0.5, ep.length - 1.0)
            x = rng.uniform(0.3, ep.width - 0.3)
            tid = next_id
            next_id += 1
            conf = rng.uniform(0.3, 0.6)
            for k in range(k0, min(n_steps, k0 + dur)):
                obs[k].append(Observation(tid, "pedestrian", float(y), float(x), float(conf), -1))
    return obs


def perfect_perception(ep: Episode) -> list[list[Observation]]:
    """Observations with no failures (used as the reference for 'what should have happened')."""
    return perceive(ep, {"recall": 1.0}, np.random.default_rng(0))


def remove_track(observations: list[list[Observation]], track_id: int) -> list[list[Observation]]:
    """Copy of the observations without one track (used for counterfactual questions)."""
    return [[o for o in frame if o.track_id != track_id] for frame in observations]


def remove_person(observations: list[list[Observation]], gt_pid: int) -> list[list[Observation]]:
    """Copy of the observations without one real person (all of their track ids)."""
    return [[o for o in frame if o.gt_pid != gt_pid] for frame in observations]
