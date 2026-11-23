"""STEP 6c (follow-up) - the same "what if" questions, but the model is also told HOW the controller works.

In the first run the model was given the event log but not the controller's rules, and it answered
"no" to every counterfactual question. This follow-up tests whether that was missing knowledge:
each counterfactual question is copied (qtype = counterfactual_rules) with a short, exact description
of the controller's rules added to the question. The original questions and answers are kept.

Report it as a follow-up experiment that was added after seeing the first results.

    python scripts/06c_add_counterfactual_with_rules.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from safecross.config import load_config, resolve
from safecross.controller import base_phase_length


def rules_text(cfg: dict, controller: str) -> str:
    c = cfg["controller"]
    base = base_phase_length(cfg)
    common = (f"HOW THE SIGNAL CONTROLLER WORKS: the walk phase normally lasts {base:.1f} s. After that it is "
              f"extended while the crossing counts as occupied, and it can never last longer than "
              f"{c['max_phase_s']:.0f} s. It ends at the first moment after {base:.1f} s when the crossing "
              f"no longer counts as occupied.")
    if controller == "baseline":
        return common + (f" This controller counts the crossing as occupied if ANY tracked person was detected on "
                         f"it during the last {c['persistence_s']:.1f} s.")
    return common + (f" This controller counts the crossing as occupied if a person is detected on it, OR if a "
                     f"person who was seen on the crossing disappeared before reaching the exit. Such a lost "
                     f"person is held as 'uncertain' until they would have finished crossing at "
                     f"{c['assumed_slow_speed_mps']} m/s (x{c['remaining_time_margin']} safety margin), until "
                     f"they are seen at the exit, or for at most {c['safety_timeout_s']:.0f} s.")


def main():
    cfg = load_config()
    qfile = resolve(cfg["paths"]["results_dir"]) / "questions" / "questions.json"
    items = json.loads(qfile.read_text())
    items = [i for i in items if i["qtype"] != "counterfactual_rules"]
    new = []
    for it in items:
        if it["qtype"] != "counterfactual":
            continue
        cp = json.loads(json.dumps(it))
        cp["qid"] = it["qid"] + "_rules"
        cp["qtype"] = "counterfactual_rules"
        cp["question"] = rules_text(cfg, it["controller"]) + "\n\n" + it["question"]
        new.append(cp)
    qfile.write_text(json.dumps(items + new, indent=1, default=float))
    print(f"Added {len(new)} counterfactual questions with the controller rules. Total questions: {len(items) + len(new)}")


if __name__ == "__main__":
    main()
