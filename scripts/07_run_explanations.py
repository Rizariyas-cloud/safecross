"""STEP 7 - Ask a vision-language model every benchmark question under the three
input conditions and score the answers (answers RQ3). Run on the GPU laptop.

    python scripts/07_run_explanations.py --backend offline                 # test the pipeline (no AI)
    python scripts/07_run_explanations.py --backend ollama --model qwen2.5vl:7b --limit 30   # timing test
    python scripts/07_run_explanations.py --backend ollama --model qwen2.5vl:7b             # full run
    python scripts/07_run_explanations.py --backend ollama --model llava:7b                 # second model

Answers are saved as they come in (answers_<model>.jsonl), so if it stops you can
re-run the same command and it continues where it left off.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from safecross.config import load_config, resolve
from safecross.explain.backends import make_backend
from safecross.explain.prompts import SYSTEM, CONDITIONS, build_user_prompt, evidence_for_condition
from safecross.explain.scoring import score
from safecross.metrics import mcnemar


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=cfg["explain"]["backend"], choices=["offline", "ollama", "hf"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--limit", type=int, default=0, help="only the first N questions (for a timing test)")
    args = ap.parse_args()

    qdir = resolve(cfg["paths"]["results_dir"]) / "questions"
    items = json.loads((qdir / "questions.json").read_text())
    if args.limit:
        items = items[:args.limit]
    backend = make_backend(cfg, args.backend, args.model)
    tag = backend.name.replace(":", "_").replace("/", "_")
    out_dir = resolve(cfg["paths"]["results_dir"]) / "explanations"
    out_dir.mkdir(parents=True, exist_ok=True)
    ans_path = out_dir / f"answers_{tag}.jsonl"

    done = set()
    if ans_path.exists():
        for line in ans_path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["qid"], r["condition"]))
    print(f"{len(items)} questions x {len(CONDITIONS)} conditions, {len(done)} already done. Model: {backend.name}")

    t0 = time.time()
    n_new = 0
    with open(ans_path, "a", encoding="utf-8") as f:
        for it in items:
            for cond in CONDITIONS:
                if (it["qid"], cond) in done:
                    continue
                log, dets, img = evidence_for_condition(it, cond)
                user = build_user_prompt(it["question"], log, dets, None, it["fields"], has_image=img is not None)
                try:
                    raw = backend.generate(SYSTEM, user, img, json_mode=True)
                except Exception as e:                      # keep going if one call fails
                    raw = json.dumps({"_error": str(e)})
                f.write(json.dumps({"qid": it["qid"], "condition": cond, "raw": raw}) + "\n")
                f.flush()
                n_new += 1
                if n_new % 20 == 0:
                    rate = (time.time() - t0) / n_new
                    left = (len(items) * len(CONDITIONS) - len(done) - n_new) * rate
                    print(f"  {n_new} answers, {rate:.1f}s each, about {left / 60:.0f} min left")

    # ---- scoring
    by_id = {it["qid"]: it for it in items}
    scored = []
    for line in ans_path.read_text().splitlines():
        r = json.loads(line)
        if r["qid"] in by_id:
            scored.append(score(by_id[r["qid"]], r["condition"], r["raw"]))
    df = pd.DataFrame(scored)
    df.to_csv(out_dir / f"scores_{tag}.csv", index=False)
    summary = (df.groupby(["qtype", "condition"])
                 [["factual_correct", "controller_faithful", "visually_grounded", "abstained",
                   "hallucinated_citation", "parsed"]]
                 .mean(numeric_only=False).round(3))
    print("\nShare of answers (1.0 = 100%):")
    print(summary.to_string())
    summary.to_csv(out_dir / f"summary_{tag}.csv")

    # Does the picture matter? Compare image+log vs log only on the SAME questions.
    for q in ("visual", "visual_real", "decision"):
        a = df[(df.qtype == q) & (df.condition == "image_log")].set_index("qid")["factual_correct"]
        b = df[(df.qtype == q) & (df.condition == "log_only")].set_index("qid")["factual_correct"]
        common = a.index.intersection(b.index)
        if len(common):
            t = mcnemar(list(a[common]), list(b[common]))
            print(f"\n{q}: image+log vs log-only on {len(common)} questions -> McNemar {t}")

    # Follow-up: does telling the model the controller rules change the "what if" answers?
    for cond in ("image_log", "log_only"):
        a = df[(df.qtype == "counterfactual_rules") & (df.condition == cond)].copy()
        b = df[(df.qtype == "counterfactual") & (df.condition == cond)].set_index("qid")["factual_correct"]
        if len(a):
            a["base"] = a["qid"].str.replace("_rules$", "", regex=True)
            a = a.set_index("base")["factual_correct"]
            common = a.index.intersection(b.index)
            t = mcnemar(list(a[common]), list(b[common]))
            print(f"\ncounterfactual ({cond}): with rules {a[common].mean():.3f} vs without {b[common].mean():.3f} "
                  f"on {len(common)} questions -> McNemar {t}")

    # Answer balance: a model that always says the same thing is not reasoning
    def _ans(raw):
        from safecross.explain.backends import parse_json
        j = parse_json(raw)
        return str(j.get("answer", j.get("reason_code")))
    raw_by = {(json.loads(l)["qid"], json.loads(l)["condition"]): json.loads(l)["raw"]
              for l in ans_path.read_text().splitlines()}
    print("\nWhat the model answered (yes/no questions, picture + log):")
    for q in ("counterfactual", "counterfactual_rules", "visual_real"):
        qs = [i for i in items if i["qtype"] == q]
        if not qs:
            continue
        from collections import Counter
        truth = Counter(i["truth"]["answer"] for i in qs)
        given = Counter(_ans(raw_by[(i["qid"], "image_log")]) for i in qs if (i["qid"], "image_log") in raw_by)
        print(f"  {q:22s} truth {dict(truth)}  |  model {dict(given)}")
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
