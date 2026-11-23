"""Quick checks that the core logic behaves as designed. Run:  python -m pytest -q"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from safecross.config import load_config
from safecross.controller import run_controller, base_phase_length
from safecross.explain.scoring import score
from safecross.metrics import episode_metrics
from safecross.perception import _miss_mask, perceive
from safecross.sim import Pedestrian, build_episode

CFG = load_config()


def _wheelchair_episode(speed=0.6):
    n = int(CFG["crossing"]["horizon_s"] / CFG["crossing"]["dt"])
    p = Pedestrian(0, "wheelchair", 0.0, 2.0, np.full(n, speed))
    return build_episode(0, [p], CFG)


def test_miss_rate_matches_recall():
    m = _miss_mask(np.random.default_rng(0), 200000, 0.8, 8)
    assert abs(m.mean() - 0.2) < 0.01


def test_perfect_perception_no_cutoff():
    ep = _wheelchair_episode()
    obs = perceive(ep, {"recall": 1.0}, np.random.default_rng(0))
    for c in ("baseline", "uncertainty_aware"):
        r = run_controller(c, CFG, ep, obs)
        assert episode_metrics(ep, r, CFG)["aid_cutoffs"] == 0
        assert r.end_time >= ep.exit_time(0)


def test_lost_track_baseline_cuts_off_uncertainty_aware_holds():
    ep = _wheelchair_episode(0.6)                       # needs ~19.5 s, normal phase is 17 s
    obs = perceive(ep, {"recall": 1.0}, np.random.default_rng(0))
    k_lost = int(10.0 / ep.dt)                          # camera loses the person at 10 s, for good
    obs = [f if k < k_lost else [] for k, f in enumerate(obs)]
    rb = run_controller("baseline", CFG, ep, obs)
    ru = run_controller("uncertainty_aware", CFG, ep, obs)
    assert rb.end_time == base_phase_length(CFG)
    assert episode_metrics(ep, rb, CFG)["aid_cutoffs"] == 1
    assert episode_metrics(ep, ru, CFG)["aid_cutoffs"] == 0
    assert any(e["reason"] == "uncertain_occupancy_lost_track" for e in ru.log)


def test_scoring_three_conditions():
    item = {"qid": "q", "qtype": "decision", "truth": {"reason_code": "crossing_clear"},
            "log": ["[L1] t=0.0s PHASE_START reason=walk_requested", "[L2] t=17.0s PHASE_END reason=crossing_clear"]}
    good = json.dumps({"reason_code": "crossing_clear", "evidence": ["L2"]})
    fake = json.dumps({"reason_code": "crossing_clear", "evidence": ["L9"]})
    assert score(item, "log_only", good)["controller_faithful"] is True
    assert score(item, "log_only", fake)["hallucinated_citation"] is True
    abstain = json.dumps({"reason_code": "cannot_determine"})
    assert score(item, "image_only", abstain)["controller_faithful"] is True   # no log -> abstaining is right
