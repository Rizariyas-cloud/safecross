"""Generate benchmark questions whose correct answers are computed exactly.

Three question families (evaluated separately):
  decision        - why / when the controller acted. Truth = the controller's own log.
  counterfactual  - "what if this track had not been detected?" Truth = re-running the controller.
  visual          - "did the camera miss someone?" Truth = ground-truth positions vs detections.
(regulation questions are written by the team, see configs/regulation_questions.yaml)

Every question stores three kinds of evidence so the SAME question can be asked
under the three input conditions: image+log, log only, image only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..controller import run_controller
from ..perception import perceive, remove_track
from ..render import draw_frame
from ..sim import generate_episode
from .evidence import format_log, format_detections, REASON_CODES

REASON_LIST = [r for r in REASON_CODES if r != "walk_requested"]


def _image(ep, k, obs, res, img_dir: Path, name: str, show_detections: bool) -> str:
    """Scene picture. For visual questions we draw ONLY the real people (what the camera
    should see); the model must compare it with the detection list."""
    img = draw_frame(ep, k, obs if show_detections else None, res, title="Camera view (schematic)")
    path = img_dir / f"{name}.png"
    img.save(path)
    return str(path)


def questions_for_episode(ep, cfg, fail_cfg, rng, img_dir: Path, controllers=("baseline", "uncertainty_aware"),
                          keep_context: bool = False):
    obs = perceive(ep, fail_cfg, np.random.default_rng(int(rng.integers(1e9))))
    items = []
    length = ep.length
    for cname in controllers:
        res = run_controller(cname, cfg, ep, obs)
        log_lines = format_log(res)
        k_end = min(len(ep.times) - 1, int(round(res.end_time / ep.dt)))
        base = f"ep{ep.episode_id}_{cname}"
        n_before = len(items)
        end_img = _image(ep, k_end, obs, res, img_dir, base + "_end", show_detections=True)

        # --- decision: why did the phase end?
        end_evt = [e for e in res.log if e["event"] == "PHASE_END"][-1]
        items.append({
            "qid": f"{base}_why_end", "qtype": "decision", "episode_id": ep.episode_id, "controller": cname,
            "question": f"Why did the pedestrian phase end at {res.end_time:.1f} s?",
            "fields": {"reason_code": REASON_LIST},
            "truth": {"reason_code": end_evt["reason"]},
            "log": log_lines, "detections": format_detections(obs, k_end, length), "image": end_img})

        # --- decision: why was it extended (if it was)?
        ext = [e for e in res.log if e["event"] == "EXTEND"]
        if ext:
            e = ext[0]
            k = min(len(ep.times) - 1, int(round(e["t"] / ep.dt)))
            items.append({
                "qid": f"{base}_why_extend", "qtype": "decision", "episode_id": ep.episode_id, "controller": cname,
                "question": f"Why was the pedestrian phase extended at {e['t']:.1f} s?",
                "fields": {"reason_code": REASON_LIST},
                "truth": {"reason_code": e["reason"]},
                "log": log_lines, "detections": format_detections(obs, k, length),
                "image": _image(ep, k, obs, res, img_dir, base + "_extend", show_detections=True)})

        # --- counterfactual: remove one track that was on the crossing near the end
        on_tracks = sorted({o.track_id for f in obs[max(0, k_end - 30):k_end + 1] for o in f
                            if 0 <= o.y < length and o.gt_pid >= 0})
        if on_tracks:
            tid = int(rng.choice(on_tracks))
            res_cf = run_controller(cname, cfg, ep, remove_track(obs, tid))
            earlier = res_cf.end_time < res.end_time - 0.05
            items.append({
                "qid": f"{base}_cf_track{tid}", "qtype": "counterfactual", "episode_id": ep.episode_id,
                "controller": cname,
                "question": f"If track {tid} had never been detected, would the pedestrian phase have ended earlier "
                            f"than {res.end_time:.1f} s?",
                "fields": {"answer": ["yes", "no"]},
                "truth": {"answer": "yes" if earlier else "no", "new_end_time": res_cf.end_time},
                "log": log_lines, "detections": format_detections(obs, k_end, length), "image": end_img})
        if keep_context:                            # for the knowledge graph (script 15); never saved to json
            for it in items[n_before:]:
                it["_ctx"] = (ep, res)

    # --- visual: did the camera miss anyone at a chosen moment? (controller-independent)
    on_any = [k for k in range(len(ep.times)) if any(ep.on_crossing(i, k) for i in range(len(ep.pedestrians)))]
    if on_any:
        k = int(rng.choice(on_any))
        on_people = [p.pid for i, p in enumerate(ep.pedestrians) if ep.on_crossing(i, k)]
        seen = {o.gt_pid for o in obs[k]}
        missed = [pid for pid in on_people if pid not in seen]
        name = f"ep{ep.episode_id}_visual_t{k}"
        items.append({
            "qid": name, "qtype": "visual", "episode_id": ep.episode_id, "controller": None,
            "question": f"At t = {ep.times[k]:.1f} s, compare the picture with the camera's detection list. "
                        f"Did the camera miss any person who is on the crossing?",
            "fields": {"answer": ["yes", "no", "cannot_determine"], "people_on_crossing": "integer"},
            "truth": {"answer": "yes" if missed else "no", "people_on_crossing": len(on_people),
                      "missed_pids": missed},
            "log": [], "detections": format_detections(obs, k, length),
            "image": _image(ep, k, obs, None, img_dir, name, show_detections=False)})
    return items


def build_question_set(cfg: dict, n_episodes: int, out_dir: Path, fail_cfg: dict | None = None, seed: int = 0,
                       keep_context: bool = False):
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    fail = dict(cfg["failures"])
    fail.update(fail_cfg or {"recall": 0.85, "track_loss_prob": 0.3, "misclassification_prob": 0.1,
                             "false_positive_rate": 0.05})
    rng = np.random.default_rng(seed)
    items = []
    for e in range(n_episodes):
        ep = generate_episode(10000 + e, cfg, rng)
        items += questions_for_episode(ep, cfg, fail, rng, img_dir, keep_context=keep_context)
    return items
