"""Turn an episode into evidence an explainer can read: numbered log lines,
a detection list, and a picture of the scene."""
from __future__ import annotations

from ..controller import EpisodeResult
from ..perception import Observation

REASON_CODES = {
    "walk_requested": "the walk phase started",
    "confirmed_occupancy": "a person was detected on the crossing",
    "mobility_aid_on_crossing": "a person using a mobility aid was detected on the crossing",
    "uncertain_occupancy_lost_track": "a person was lost by the tracker while still on the crossing, so the phase was held",
    "crossing_clear": "the crossing was judged to be clear",
    "max_phase_reached": "the maximum phase length was reached",
}


def format_log(res: EpisodeResult) -> list[str]:
    """['[L1] t=0.0s PHASE_START reason=walk_requested ...', ...]"""
    lines = []
    for i, e in enumerate(res.log, start=1):
        conf = "" if e.get("confidence") is None else f" confidence={e['confidence']:.2f}"
        lines.append(f"[L{i}] t={e['t']:.1f}s {e['event']} reason={e['reason']}{conf} ({e['details']})")
    return lines


def format_detections(observations: list[list[Observation]], k: int, length: float) -> list[str]:
    """What the camera reported at time step k (never reveals ground truth)."""
    out = []
    for o in sorted(observations[k], key=lambda o: o.track_id):
        where = "on the crossing" if 0 <= o.y < length else ("waiting at the kerb" if o.y < 0 else "past the exit")
        out.append(f"track {o.track_id}: {o.cls}, confidence {o.conf:.2f}, {where} at y={o.y:.1f} m")
    return out or ["no detections"]
