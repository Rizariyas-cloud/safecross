"""STEP 19 (NLP) - two HARDER test sets that the fine-tuned explainer never saw in training.

  ood_wording   : the usual controllers, but every question is WORDED differently from all training templates
                  ("For what reason did the walk signal stop at 19.6 s?", "Suppose the camera had missed track 4
                  completely...") -> does the model understand the question, or only one sentence pattern?
  ood_controller: questions about a controller that is NOT in the training data (uncertainty_aware_filtered,
                  the ghost-filter version) -> does the model read the graph, or memorise controller habits?

New simulated walk phases (a third seed: not the benchmark's, not the training's). Correct answers are exact,
as in the benchmark (the controller's log; re-running the controller for "what if").

    python scripts/19_make_ood_questions.py
    python scripts/16_run_kg_explanations.py --backend mlx --questions results/questions_ood_wording
    python scripts/16_run_kg_explanations.py --backend mlx --questions results/questions_ood_controller
(and the same with --backend mlx-base / ollama for comparison)
"""
import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from safecross.config import load_config, resolve
from safecross.explain.kg import build_kg, lines, subgraph, phase_ends_earlier_without
from safecross.explain.questions import questions_for_episode
from safecross.explain.sft import END_Q, EXT_Q, CF_Q
from safecross.sim import generate_episode

END_OOD = ["For what reason did the walk signal stop at {t:.1f} s?", "Why did the green man turn off at {t:.1f} s?",
           "What led the controller to finish the pedestrian phase at {t:.1f} s?"]
EXT_OOD = ["Why did the walk signal stay green longer at {t:.1f} s?",
           "What kept the pedestrian signal going beyond its normal length at {t:.1f} s?"]
CF_OOD = ["Suppose the camera had missed track {n} completely. Would the walk signal have stopped sooner than {t:.1f} s?",
          "Imagine track {n} was never seen by the camera: would the pedestrian phase have ended before {t:.1f} s?"]


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--seed", type=int, default=777001)
    args = ap.parse_args()
    for t in END_OOD + EXT_OOD + CF_OOD:                       # never a training template
        assert t not in END_Q + EXT_Q + CF_Q
    rng = np.random.default_rng(args.seed)
    fail = dict(cfg["failures"])
    fail.update({"recall": 0.85, "track_loss_prob": 0.3, "misclassification_prob": 0.1, "false_positive_rate": 0.05})
    sets = {"ood_wording": [], "ood_controller": []}
    with tempfile.TemporaryDirectory() as tmp:
        for e in range(args.episodes):
            ep = generate_episode(90000 + e, cfg, rng)
            items = questions_for_episode(ep, cfg, fail, rng, Path(tmp),
                                          controllers=("baseline", "uncertainty_aware", "uncertainty_aware_filtered"),
                                          keep_context=True)
            for it in items:
                if it["qtype"] not in ("decision", "counterfactual"):
                    continue
                ep_, res = it.pop("_ctx")
                it.pop("image", None)
                if it["controller"] == "uncertainty_aware_filtered":
                    it["qid"] = "oodc_" + it["qid"]
                    target = "ood_controller"
                else:
                    t = float(re.search(r"(\d+\.\d+) s", it["question"]).group(1))
                    if it["qtype"] == "counterfactual":
                        n = re.search(r"track (\d+)", it["question"]).group(1)
                        it["question"] = str(rng.choice(CF_OOD)).format(n=n, t=t)
                    elif it["qid"].endswith("why_end"):
                        it["question"] = str(rng.choice(END_OOD)).format(t=t)
                    else:
                        it["question"] = str(rng.choice(EXT_OOD)).format(t=t)
                    it["qid"] = "oodw_" + it["qid"]
                    target = "ood_wording"
                facts = build_kg(ep_, res, cfg)
                it["_kg"] = {"kg_full": lines(facts), "kg_sub": lines(subgraph(facts, it["question"]))}
                if it["qtype"] == "counterfactual":         # graph must agree with the replayed answer
                    tid = int(re.search(r"track (\d+)", it["question"]).group(1))
                    assert phase_ends_earlier_without(res, cfg, tid) == (it["truth"]["answer"] == "yes")
                sets[target].append(it)
    for name, items in sets.items():
        out = resolve(cfg["paths"]["results_dir"]) / f"questions_{name}"
        out.mkdir(parents=True, exist_ok=True)
        kg = {it["qid"]: it.pop("_kg") for it in items}
        (out / "questions.json").write_text(json.dumps(items, indent=1, default=float))
        (out / "kg.json").write_text(json.dumps(kg, indent=1))
        n_cf = sum(it["qtype"] == "counterfactual" for it in items)
        yes = sum(it["qtype"] == "counterfactual" and it["truth"]["answer"] == "yes" for it in items)
        print(f"{name}: {len(items)} questions ({len(items) - n_cf} decision, {n_cf} what-if, {yes} of them 'yes') -> {out}")
    ex = sets["ood_wording"][:3]
    for it in ex:
        print("  e.g.", it["question"])


if __name__ == "__main__":
    main()
