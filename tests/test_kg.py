"""Knowledge graph: provenance must agree with replaying the controller; the checker must catch bad citations."""
import json

import numpy as np

from safecross.config import load_config
from safecross.controller import run_controller, CONTROLLERS
from safecross.explain.kg import build_kg, lines, subgraph, phase_ends_earlier_without
from safecross.explain.verify import verify
from safecross.perception import perceive, remove_track
from safecross.sim import generate_episode


def _case(seed=3):
    cfg = load_config()
    ep = generate_episode(5, cfg, np.random.default_rng(seed))
    obs = perceive(ep, dict(cfg["failures"], recall=0.6, track_loss_prob=0.4), np.random.default_rng(1))
    return cfg, ep, obs


def test_graph_agrees_with_controller_replay():
    cfg = load_config()
    rng = np.random.default_rng(11)
    for e in range(25):
        ep = generate_episode(e, cfg, rng)
        obs = perceive(ep, dict(cfg["failures"], recall=0.7, track_loss_prob=0.3, false_positive_rate=0.05),
                       np.random.default_rng(e))
        for c in CONTROLLERS:
            res = run_controller(c, cfg, ep, obs)
            assert build_kg(ep, res, cfg)
            for tid in {o.track_id for f in obs for o in f}:
                replay = run_controller(c, cfg, ep, remove_track(obs, tid)).end_time < res.end_time - 0.05
                assert phase_ends_earlier_without(res, cfg, tid) == replay


def test_subgraph_is_smaller_and_keeps_the_asked_track():
    cfg, ep, obs = _case()
    res = run_controller("uncertainty_aware", cfg, ep, obs)
    facts = build_kg(ep, res, cfg)
    sub = subgraph(facts, "If track 1 had never been detected, would the pedestrian phase have ended earlier?")
    assert 0 < len(sub) < len(facts)
    assert any("(track 1) --" in f.line() for f in sub)


def test_checker_rejects_unsupported_answers():
    cfg, ep, obs = _case()
    res = run_controller("uncertainty_aware", cfg, ep, obs)
    facts = lines(build_kg(ep, res, cfg))
    item = {"qtype": "counterfactual", "question": "If track 1 had never been detected, would ... ended earlier?"}
    only = next(l for l in facts if "(track 1) --only_occupant-->" in l)
    gid = only.split("]")[0][1:]
    ok = json.dumps({"answer": "yes", "evidence": [gid]})
    assert verify(item, ok, facts)[1] is True
    assert verify(item, json.dumps({"answer": "yes", "evidence": ["G999"]}), facts)[1] is False     # invented id
    assert verify(item, json.dumps({"answer": "no", "evidence": [gid]}), facts)[1] is False        # contradicts graph
    assert verify(item, json.dumps({"answer": "yes", "evidence": []}), facts)[1] is False          # no citation
    assert verify(item, json.dumps({"answer": "cannot_determine", "evidence": []}), facts)[1] is True
