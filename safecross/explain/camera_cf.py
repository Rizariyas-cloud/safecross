"""A new what-if question that links NLP to the tracking failures measured in CV (script 20):

    "If the camera had not lost track 4, would the walk phase have ended earlier than, later than,
     or at the same time as 26.7 s?"

Truth is computed by RE-RUNNING the controller with the camera repaired for that person: from the moment the
track was lost, the person is tracked under the SAME id on every frame until they leave the crossing
(no more misses, no new id). The model gets the usual knowledge-graph facts (last position, estimated speed,
hold time, how the phase ended) and must reason from them, or say "cannot_determine".

This question type never appears in the explainer's training data, so it is also an out-of-distribution test.
"""
from __future__ import annotations

from collections import Counter

from ..controller import run_controller
from ..perception import Observation

CAM_Q = "If the camera had not lost track {n}, would the walk phase have ended earlier than, later than, or at the same time as {t:.1f} s?"
FIELDS = {"answer": ["earlier", "later", "same"]}


def lost_tracks(res, ep) -> list[int]:
    """Tracks the camera lost on the crossing before the phase ended (did not reach the exit)."""
    return sorted(tid for tid, d in res.tracks.items()
                  if not d.get("exited") and d["last_t"] < res.end_time - 1e-6 and 0 <= d["last_y"] < ep.length)


def repair(ep, obs, tid: int):
    """Observations with track `tid` never lost again after its last sighting (same id until the person leaves)."""
    frames = [k for k, fr in enumerate(obs) if any(o.track_id == tid for o in fr)]
    if not frames:
        return None
    k_last = frames[-1]
    mine = [o for o in obs[k_last] if o.track_id == tid][0]
    pid = mine.gt_pid
    if pid < 0:                                            # a phantom: nothing real to repair
        return None
    i = next(j for j, p in enumerate(ep.pedestrians) if p.pid == pid)
    cls = Counter(o.cls for fr in obs for o in fr if o.track_id == tid).most_common(1)[0][0]
    out = [list(fr) for fr in obs]
    for k in range(k_last + 1, len(obs)):
        out[k] = [o for o in out[k] if o.gt_pid != pid]    # drop the new-id sightings of the same person
        y = float(ep.positions[i, k])
        if -1.0 < y < ep.length + 1.5:
            out[k].append(Observation(tid, cls, y, ep.pedestrians[i].x, mine.conf, pid))
    return out


def truth(cfg, controller: str, ep, obs, res, tid: int):
    rep = repair(ep, obs, tid)
    if rep is None:
        return None
    end2 = run_controller(controller, cfg, ep, rep).end_time
    if abs(end2 - res.end_time) < 0.05:
        return "same", end2
    return ("earlier" if end2 < res.end_time else "later"), end2
