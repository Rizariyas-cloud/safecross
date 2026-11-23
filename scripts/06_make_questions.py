"""STEP 6 - Build the explanation benchmark (decision, counterfactual and visual questions).

    python scripts/06_make_questions.py                # uses explain.n_decision_episodes from the config
    python scripts/06_make_questions.py --episodes 20  # small test set

Outputs: results/questions/questions.json and results/questions/images/*.png
Regulation questions are separate: write them in configs/regulation_questions.yaml.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from safecross.config import load_config, resolve
from safecross.explain.questions import build_question_set


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=cfg["explain"]["n_decision_episodes"])
    args = ap.parse_args()
    out = resolve(cfg["paths"]["results_dir"]) / "questions"
    items = build_question_set(cfg, args.episodes, out, seed=cfg["seed"])
    with open(out / "questions.json", "w", encoding="utf-8") as f:
        json.dump(items, f, indent=1, default=float)
    print(f"{len(items)} questions -> {out / 'questions.json'}")
    print("By type:", dict(Counter(i["qtype"] for i in items)))
    print("Truth balance:", dict(Counter((i["qtype"], str(list(i["truth"].values())[0])) for i in items)))


if __name__ == "__main__":
    main()
