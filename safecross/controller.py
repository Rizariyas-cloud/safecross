"""Two pedestrian-signal controllers that receive ONLY camera observations.

Both controllers share the same rules for timing:
  * normal phase length  = walk interval + crossing length / design speed
  * the phase can be extended while the crossing seems occupied
  * the phase can never be longer than max_phase_s

BaselineController
    "Occupied" means: someone was detected on the crossing during the last
    `persistence_s` seconds. A missed detection longer than that looks like an
    empty crossing.

UncertaintyAwareController
    Keeps a memory of every track. A track that disappears INSIDE the crossing
    (and not at the exit) is put in an "uncertain" state instead of being
    forgotten. The phase is held until that person would have had time to finish
    crossing at a slow assumed speed, until they are seen at the exit, or until a
    safety timeout. Occupancy therefore has three states: confirmed, uncertain, clear.

Each controller writes an event log that the explanation module uses later.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .perception import Observation
from .sim import Episode, MOBILITY_AID_CLASSES


@dataclass
class EpisodeResult:
    controller: str
    end_time: float
    log: list[dict]
    occupancy: list[str] = field(repr=False)   # state at each time step: confirmed / uncertain / clear / ended
    # Provenance for the knowledge graph (does not change any decision):
    # trace[k] = (t, [tracks counted as confirmed], [tracks counted as uncertain]) for every step until the end
    trace: list = field(default_factory=list, repr=False)
    # per-track summary at the end: first/last time seen, last position, mobility-aid share, speed, hold time
    tracks: dict = field(default_factory=dict, repr=False)


def base_phase_length(cfg: dict) -> float:
    c = cfg["controller"]
    return cfg["scenario"]["walk_interval_s"] + cfg["crossing"]["length_m"] / c["design_speed_mps"]


class _Controller:
    name = "controller"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.c = cfg["controller"]
        self.length = cfg["crossing"]["length_m"]
        self.margin = cfg["crossing"]["exit_margin_m"]
        self.base = base_phase_length(cfg)
        self.max_phase = self.c["max_phase_s"]

    def _on_crossing(self, y: float) -> bool:
        return 0.0 <= y < self.length - self.margin

    def run(self, ep: Episode, observations: list[list[Observation]]) -> EpisodeResult:
        raise NotImplementedError


class BaselineController(_Controller):
    name = "baseline"

    def run(self, ep, observations):
        persist = self.c["persistence_s"]
        last_seen_on = -1e9
        last_conf = 0.0
        last_tracks: list[int] = []
        log = [{"t": 0.0, "event": "PHASE_START", "reason": "walk_requested", "confidence": None,
                "details": f"normal phase length {self.base:.1f}s, maximum {self.max_phase:.1f}s"}]
        occ = []
        trace, info = [], {}
        last_on: dict[int, float] = {}
        extended = False
        end_time = self.max_phase
        for k, t in enumerate(ep.times):
            on = [o for o in observations[k] if self._on_crossing(o.y)]
            if on:
                last_seen_on = t
                last_conf = max(o.conf for o in on)
                last_tracks = sorted({o.track_id for o in on})
            for o in observations[k]:
                d = info.setdefault(o.track_id, {"first_t": float(t), "n": 0, "aid": 0})
                d.update(last_t=float(t), last_y=float(o.y))
                d["n"] += 1
                d["aid"] += int(o.cls in MOBILITY_AID_CLASSES)
            for o in on:
                last_on[o.track_id] = t
            occupied = (t - last_seen_on) <= persist
            occ.append("confirmed" if occupied else "clear")
            trace.append((round(float(t), 1), sorted(i for i, lt in last_on.items() if t - lt <= persist), []))
            if t + 1e-9 >= self.max_phase:
                end_time = self.max_phase
                log.append({"t": round(float(t), 1), "event": "PHASE_END", "reason": "max_phase_reached",
                            "confidence": None, "details": "maximum phase length reached"})
                break
            if t + 1e-9 >= self.base:
                if occupied and not extended:
                    extended = True
                    log.append({"t": round(float(t), 1), "event": "EXTEND", "reason": "confirmed_occupancy",
                                "confidence": round(last_conf, 2),
                                "details": f"tracks {last_tracks} detected on the crossing"})
                if not occupied:
                    end_time = t
                    log.append({"t": round(float(t), 1), "event": "PHASE_END", "reason": "crossing_clear",
                                "confidence": None,
                                "details": f"no detection on the crossing for {persist:.1f}s"})
                    break
        occ += ["ended"] * (len(ep.times) - len(occ))
        tracks = {tid: {"first_t": d["first_t"], "last_t": d["last_t"], "last_y": d["last_y"],
                        "aid_share": d["aid"] / d["n"], "exited": d["last_y"] >= self.length - self.margin}
                  for tid, d in info.items()}
        return EpisodeResult(self.name, float(round(end_time, 1)), log, occ, trace, tracks)


@dataclass
class _Track:
    tid: int
    last_t: float
    last_y: float
    ys: list = field(default_factory=list)
    ts: list = field(default_factory=list)
    aid_votes: int = 0
    n: int = 0
    exited: bool = False
    max_conf: float = 0.0

    def speed(self) -> float:
        if len(self.ts) < 3:
            return float("nan")
        t0 = self.ts[-1] - 1.0                      # speed over the last second
        idx = [i for i, t in enumerate(self.ts) if t >= t0]
        i0 = idx[0] if idx else 0
        dt = self.ts[-1] - self.ts[i0]
        return (self.ys[-1] - self.ys[i0]) / dt if dt > 0 else float("nan")

    def aid_share(self) -> float:
        return self.aid_votes / self.n if self.n else 0.0


class UncertaintyAwareController(_Controller):
    name = "uncertainty_aware"

    def _expected_clear_time(self, tr: _Track) -> float:
        """When a lost person should have finished crossing, assuming they may be slow."""
        v = tr.speed()
        slow = self.c["assumed_slow_speed_mps"]
        if tr.aid_share() >= self.c["mobility_aid_threshold"] or not np.isfinite(v):
            v = slow
        v = max(min(v, 1.5), slow)
        remaining = max(0.0, (self.length - tr.last_y)) / v * self.c["remaining_time_margin"]
        return tr.last_t + min(remaining, self.c["safety_timeout_s"])

    def _counts(self, tr: _Track) -> bool:
        """Every track counts for this controller (the filtered version below overrides this)."""
        return True

    def run(self, ep, observations):
        persist = self.c["persistence_s"]
        tracks: dict[int, _Track] = {}
        log = [{"t": 0.0, "event": "PHASE_START", "reason": "walk_requested", "confidence": None,
                "details": f"normal phase length {self.base:.1f}s, maximum {self.max_phase:.1f}s"}]
        occ = []
        trace = []
        last_reason = None
        end_time = self.max_phase
        for k, t in enumerate(ep.times):
            for o in observations[k]:
                tr = tracks.get(o.track_id)
                if tr is None:
                    tr = tracks[o.track_id] = _Track(o.track_id, t, o.y)
                tr.last_t, tr.last_y = t, o.y
                tr.ys.append(o.y)
                tr.ts.append(t)
                tr.n += 1
                tr.aid_votes += int(o.cls in MOBILITY_AID_CLASSES)
                tr.max_conf = max(tr.max_conf, o.conf)
                if o.y >= self.length - self.margin:
                    tr.exited = True

            confirmed = [tr for tr in tracks.values()
                         if t - tr.last_t <= persist and self._on_crossing(tr.last_y) and not tr.exited
                         and self._counts(tr)]
            uncertain = [tr for tr in tracks.values()
                         if t - tr.last_t > persist and not tr.exited and self._counts(tr)
                         and 0.0 <= tr.last_y < self.length - self.margin
                         and t < self._expected_clear_time(tr)]
            state = "confirmed" if confirmed else ("uncertain" if uncertain else "clear")
            occ.append(state)
            trace.append((round(float(t), 1), sorted(tr.tid for tr in confirmed), sorted(tr.tid for tr in uncertain)))

            if t + 1e-9 >= self.max_phase:
                end_time = self.max_phase
                log.append({"t": round(float(t), 1), "event": "PHASE_END", "reason": "max_phase_reached",
                            "confidence": None, "details": "maximum phase length reached"})
                break
            if t + 1e-9 >= self.base:
                if state == "clear":
                    end_time = t
                    log.append({"t": round(float(t), 1), "event": "PHASE_END", "reason": "crossing_clear",
                                "confidence": None, "details": "no confirmed or uncertain occupancy"})
                    break
                if state == "confirmed":
                    slow = [tr for tr in confirmed if tr.aid_share() >= self.c["mobility_aid_threshold"]]
                    reason = "mobility_aid_on_crossing" if slow else "confirmed_occupancy"
                    conf = max(tr.max_conf for tr in confirmed)
                    details = f"tracks {[tr.tid for tr in confirmed]} detected on the crossing"
                else:
                    reason = "uncertain_occupancy_lost_track"
                    tr = max(uncertain, key=self._expected_clear_time)
                    lost_for = t - tr.last_t
                    exp_clear = self._expected_clear_time(tr)
                    total = max(1e-6, exp_clear - tr.last_t)
                    conf = 1.0 - lost_for / total          # falls as the hold continues
                    details = (f"track {tr.tid} lost at y={tr.last_y:.1f}m, {lost_for:.1f}s ago; "
                               f"holding until {exp_clear:.1f}s")
                if reason != last_reason:
                    log.append({"t": round(float(t), 1), "event": "EXTEND" if last_reason is None else "HOLD",
                                "reason": reason, "confidence": round(float(conf), 2), "details": details})
                    last_reason = reason
        occ += ["ended"] * (len(ep.times) - len(occ))
        summary = {}
        for tr in tracks.values():
            v = tr.speed()
            summary[tr.tid] = {"first_t": float(tr.ts[0]) if tr.ts else float(tr.last_t), "last_t": float(tr.last_t),
                               "last_y": float(tr.last_y), "aid_share": tr.aid_share(), "exited": tr.exited,
                               "speed": None if not np.isfinite(v) else float(v),
                               "hold_until": float(self._expected_clear_time(tr)),
                               "counts": bool(self._counts(tr))}
        return EpisodeResult(self.name, float(round(end_time, 1)), log, occ, trace, summary)


class FilteredUncertaintyAwareController(UncertaintyAwareController):
    """Uncertainty-aware controller with a ghost filter (follow-up, added for the junction study).

    A track only counts as a person once it has been seen for at least `confirm_min_s` seconds AND has
    moved at least `confirm_min_move_m` metres. Phantom detections are short and do not move, so they
    no longer extend the walk phase. Real people walk, so they pass the filter within about a second.
    """
    name = "uncertainty_aware_filtered"

    def _counts(self, tr: _Track) -> bool:
        if len(tr.ts) < 2:
            return False
        seen_for = tr.ts[-1] - tr.ts[0]
        moved = max(tr.ys) - min(tr.ys)
        return (seen_for >= self.c.get("confirm_min_s", 0.5)
                and moved >= self.c.get("confirm_min_move_m", 0.3))


CONTROLLERS = {"baseline": BaselineController, "uncertainty_aware": UncertaintyAwareController,
               "uncertainty_aware_filtered": FilteredUncertaintyAwareController}


def run_controller(name: str, cfg: dict, ep: Episode, observations) -> EpisodeResult:
    return CONTROLLERS[name](cfg).run(ep, observations)
