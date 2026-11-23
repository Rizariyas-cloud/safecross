"""STEP 17 (NLP step 2) - make the training data for our own fine-tuned explainer.

    python scripts/17_make_explainer_data.py            # ~1,000 simulated walk phases -> ~8,000 examples
    python scripts/17_make_explainer_data.py --full-share 0.3 --out data/explainer_sft_v2
        # v2: 30% of questions show the WHOLE graph (fixes the weakness found in the OOD tests:
        #     the v1 model, trained only on question-focused subgraphs, did worse when given the whole graph)

Writes data/explainer_sft/train.jsonl and valid.jsonl (mlx-lm chat format) and checks that NONE of the
training walk phases is one of the benchmark's walk phases (a different random seed, and a fingerprint check).
"""
import argparse
import collections
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from safecross.config import load_config, resolve
from safecross.explain.questions import build_question_set
from safecross.explain.sft import build_examples, to_chat, fingerprint


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261121, help="must differ from the benchmark seed")
    ap.add_argument("--valid-share", type=float, default=0.08)
    ap.add_argument("--out", default="data/explainer_sft")
    ap.add_argument("--full-share", type=float, default=0.0,
                    help="share of questions shown with the whole graph instead of the subgraph (v2: 0.3)")
    args = ap.parse_args()
    if args.seed == cfg["seed"]:
        sys.exit("Use a different seed from the benchmark (config seed).")

    print(f"Building examples from {args.episodes} new simulated walk phases (seed {args.seed}) ...")
    ex, prints = build_examples(cfg, args.episodes, args.seed, full_share=args.full_share)

    # ---- leakage check against the benchmark walk phases
    n_bench = cfg["explain"]["n_decision_episodes"]
    with tempfile.TemporaryDirectory() as tmp:
        bench = build_question_set(cfg, n_bench, Path(tmp), seed=cfg["seed"], keep_context=True)
    bench_prints = {fingerprint(it["_ctx"][0]) for it in bench if "_ctx" in it}
    overlap = len(set(prints) & bench_prints)
    print(f"Leakage check: {overlap} of {len(prints)} training walk phases also appear in the benchmark "
          f"({len(bench_prints)} benchmark walk phases)")
    if overlap:
        sys.exit("Overlap found - change --seed.")

    # ---- split by WALK PHASE (never the same walk phase in train and valid)
    rng = np.random.default_rng(args.seed)
    eps = sorted({x["episode"] for x in ex})
    valid_eps = set(rng.choice(eps, size=max(1, int(len(eps) * args.valid_share)), replace=False).tolist())
    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", [x for x in ex if x["episode"] not in valid_eps]),
                       ("valid", [x for x in ex if x["episode"] in valid_eps])):
        order = rng.permutation(len(part))
        with open(out / f"{name}.jsonl", "w") as f:
            for i in order:
                f.write(json.dumps(to_chat(part[i])) + "\n")
        kinds = collections.Counter(x["kind"] for x in part)
        answers = collections.Counter(json.loads(x["target"]).get("answer", json.loads(x["target"]).get("reason_code"))
                                      for x in part)
        print(f"{name}: {len(part)} examples | {dict(kinds)}")
        print(f"       graph shown: {dict(collections.Counter(x.get('graph', 'sub') for x in part))}")
        print(f"       answers: {dict(answers)}")
    (out / "README.txt").write_text(
        f"Made by scripts/17_make_explainer_data.py, seed {args.seed}, {args.episodes} walk phases, "
        f"full-graph share {args.full_share}.\n"
        "Each line: system prompt + (knowledge-graph facts + question) -> JSON answer with cited fact ids.\n")
    print(f"Saved {out}/train.jsonl and valid.jsonl")


if __name__ == "__main__":
    main()
