"""Simulated pedestrian crossing.

Coordinates: y is the distance along the crossing in metres. y < 0 means the
person is waiting on the kerb, 0 <= y <= length means the person is on the
crossing, and y > length means the person has reached the other side.
x is the sideways position inside the crossing (0 .. width).

Each pedestrian has a speed profile (one speed value per time step). This
lets us use synthetic walkers OR real walking speeds taken from PMMA tracks.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

MOBILITY_AID_CLASSES = ("wheelchair", "rollator", "crutches", "cane")


@dataclass
class Pedestrian:
    pid: int
    cls: str                      # "pedestrian", "wheelchair", "rollator", "crutches", "cane"
    t_start: float                # time (s) the person steps onto the crossing
    x: float                      # sideways position in the crossing (m)
    speeds: np.ndarray = field(repr=False)  # speed (m/s) for each step after t_start

    @property
    def is_mobility_aid(self) -> bool:
        return self.cls in MOBILITY_AID_CLASSES


@dataclass
class Episode:
    """One walk phase: the crossing, the people and their ground-truth positions."""
    episode_id: int
    length: float
    width: float
    dt: float
    times: np.ndarray             # time of each step
    pedestrians: list[Pedestrian]
    positions: np.ndarray         # shape (n_peds, n_steps): y of each person
    exit_margin: float = 0.3

    def on_crossing(self, i: int, k: int) -> bool:
        y = self.positions[i, k]
        return 0.0 <= y < self.length - self.exit_margin

    def exit_time(self, i: int) -> float:
        """First time person i reaches the exit line (np.inf if never)."""
        cleared = np.where(self.positions[i] >= self.length - self.exit_margin)[0]
        return float(self.times[cleared[0]]) if len(cleared) else float("inf")

    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode_id, "length": self.length, "width": self.width, "dt": self.dt,
            "pedestrians": [{"pid": p.pid, "cls": p.cls, "t_start": p.t_start, "x": p.x,
                             "exit_time": self.exit_time(i)} for i, p in enumerate(self.pedestrians)],
        }


def _positions(peds: list[Pedestrian], times: np.ndarray, dt: float) -> np.ndarray:
    pos = np.full((len(peds), len(times)), -0.8)          # waiting on the kerb
    for i, p in enumerate(peds):
        k0 = int(round(p.t_start / dt))
        n = len(times) - k0
        if n <= 0:
            continue
        sp = p.speeds[:n]
        if len(sp) < n:                                    # keep walking at the last speed
            sp = np.concatenate([sp, np.full(n - len(sp), sp[-1] if len(sp) else 1.0)])
        pos[i, k0:] = np.cumsum(sp) * dt
    return pos


def build_episode(episode_id: int, peds: list[Pedestrian], cfg: dict) -> Episode:
    c = cfg["crossing"]
    times = np.arange(0.0, c["horizon_s"], c["dt"])
    return Episode(episode_id, c["length_m"], c["width_m"], c["dt"], times, peds,
                   _positions(peds, times, c["dt"]), c["exit_margin_m"])


def _speed_profile(rng: np.random.Generator, lo: float, hi: float, n: int, dt: float) -> np.ndarray:
    """A realistic-looking speed profile: a base speed with gentle variation."""
    base = rng.uniform(lo, hi)
    noise = np.cumsum(rng.normal(0, 0.02, n))              # slow drift
    noise -= np.linspace(0, noise[-1], n)                  # keep the average close to base
    sp = np.clip(base + noise, 0.25, 2.2)
    accel_steps = max(1, int(1.0 / dt))                    # about 1 s to reach full speed
    sp[:accel_steps] *= np.linspace(0.4, 1.0, accel_steps)
    return sp


def generate_episode(episode_id: int, cfg: dict, rng: np.random.Generator,
                     trajectory_bank: list[dict] | None = None) -> Episode:
    """Create one random walk phase.

    If a trajectory_bank (real speed profiles from PMMA) is given, people's
    speeds are taken from it; otherwise synthetic speeds from the config are used.
    """
    s = cfg["scenario"]
    dt = cfg["crossing"]["dt"]
    n_steps = int(cfg["crossing"]["horizon_s"] / dt)
    n = int(rng.integers(s["n_pedestrians"][0], s["n_pedestrians"][1] + 1))
    peds = []
    for pid in range(n):
        if rng.random() < s["mobility_aid_share"]:
            cls = str(rng.choice(MOBILITY_AID_CLASSES))
        else:
            cls = "pedestrian"
        if rng.random() < s["late_arrival_share"]:
            t_start = rng.uniform(0.5 * s["walk_interval_s"], s["walk_interval_s"])
        else:
            t_start = rng.uniform(0.0, 2.0)
        t_start = round(t_start / dt) * dt
        x = rng.uniform(0.4, cfg["crossing"]["width_m"] - 0.4)
        speeds = None
        if trajectory_bank:
            same = [t for t in trajectory_bank if t["cls"] == cls] or trajectory_bank
            pick = same[int(rng.integers(len(same)))]
            speeds = np.asarray(pick["speeds"], dtype=float)
        if speeds is None or len(speeds) < 5:
            lo, hi = cfg["speeds"].get(cls, cfg["speeds"]["pedestrian"])
            speeds = _speed_profile(rng, lo, hi, n_steps, dt)
        peds.append(Pedestrian(pid, cls, float(t_start), float(x), speeds))
    return build_episode(episode_id, peds, cfg)


def load_trajectory_bank(path: str | Path) -> list[dict]:
    """Load real speed profiles produced by scripts/08_pmma_to_trajectories.py."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
