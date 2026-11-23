"""Training data for the explainer: every cited fact must exist, and answers must follow the graph."""
import json

from safecross.config import load_config
from safecross.explain.sft import build_examples


def test_training_answers_are_grounded():
    ex, _ = build_examples(load_config(), 25, seed=5)
    assert ex
    for x in ex:
        t = json.loads(x["target"])
        ids = {l.split("]")[0][1:] for l in x["facts"]}
        assert all(e in ids for e in t["evidence"])                      # never cites a missing fact
        val = t.get("answer", t.get("reason_code"))
        if x["kind"].startswith("abstain"):
            assert val == "cannot_determine" and t["evidence"] == []
        if x["kind"] == "counterfactual" and val == "yes":
            cited = [l for l in x["facts"] if l.split("]")[0][1:] in t["evidence"]]
            assert any("--only_occupant-->" in l for l in cited)
        if x["kind"] == "decision":
            cited = [l for l in x["facts"] if l.split("]")[0][1:] in t["evidence"]]
            assert any(val in l for l in cited)
