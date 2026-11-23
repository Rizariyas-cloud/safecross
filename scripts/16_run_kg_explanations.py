"""STEP 16 (NLP-RQ1) - ask the model the decision / "what if" questions from the KNOWLEDGE GRAPH
instead of the flat event log, then apply the citation checker.

Compared on the SAME questions (from the earlier run, answers_<model>.jsonl):
    log_only          the flat event log + detections           (earlier run)
    kg_full           the whole decision knowledge graph        (new)
    kg_sub            the question-focused subgraph             (new)
    kg_sub+checker    kg_sub, answers without valid support are withheld (no extra model calls)

    python scripts/16_run_kg_explanations.py --backend offline                       # pipeline test
    python scripts/16_run_kg_explanations.py --backend ollama --model qwen2.5vl:7b --limit 20   # timing
    python scripts/16_run_kg_explanations.py --backend ollama --model qwen2.5vl:7b
    python scripts/16_run_kg_explanations.py --backend ollama --model llava:7b
    python scripts/16_run_kg_explanations.py --backend mlx-base      # small Qwen 1.5B, NOT fine-tuned (control)
    python scripts/16_run_kg_explanations.py --backend mlx           # small Qwen 1.5B + our LoRA (fine-tuned)

Extra condition kg_sub_missing: the subgraph with the key facts REMOVED (the decision asked about, or every
fact about the track asked about). The only correct answer is "cannot_determine" (tests honest abstention).
Resumable: re-run the same command to continue.
Output: results/explanations/answers_kg_<model>.jsonl, scores_kg_<model>.csv, summary_kg_<model>.csv
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from safecross.config import load_config, resolve
from safecross.explain.backends import make_backend
from safecross.explain.prompts import SYSTEM_KG, KG_CONDITIONS, build_kg_prompt
from safecross.explain.scoring import score
from safecross.explain.verify import verify
from safecross.metrics import mcnemar


def missing_evidence(item: dict, facts: list[str]) -> list[str]:
    """The subgraph without the facts needed to answer (for the abstention test)."""
    if item["qtype"] == "counterfactual":
        tid = re.search(r"track\s+(\d+)", item["question"]).group(1)
        return [l for l in facts if f"(track {tid})" not in l]
    kind = "PHASE_END" if item["qid"].endswith("why_end") else "EXTEND"
    nodes = [re.match(r"\[\w+\] \((L\d+ \w+)\)", l).group(1) for l in facts if re.match(r"\[\w+\] \(L\d+ " + kind, l)]
    node = nodes[0] if nodes else None
    return [l for l in facts if not (node and l.split("] ", 1)[1].startswith(f"({node})"))
            and not (kind == "PHASE_END" and "--ended_at-->" in l)]


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=cfg["explain"]["backend"], choices=["offline", "ollama", "hf", "mlx", "mlx-base"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--adapter", default=None, help="mlx only: LoRA folder (default weights/explainer_lora; v2: weights/explainer_lora_v2)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--questions", default=None, help="question folder (default results/questions; OOD sets: script 19)")
    args = ap.parse_args()

    qdir = resolve(args.questions) if args.questions else resolve(cfg["paths"]["results_dir"]) / "questions"
    suffix = "" if not args.questions else "_" + Path(args.questions).name.replace("questions_", "")
    kg = json.loads((qdir / "kg.json").read_text())
    items = [it for it in json.loads((qdir / "questions.json").read_text()) if it["qid"] in kg]
    if args.limit:
        items = items[:args.limit]
    backend = make_backend(cfg, args.backend, args.model, args.adapter)
    tag = backend.name.replace(":", "_").replace("/", "_") + suffix
    out_dir = resolve(cfg["paths"]["results_dir"]) / "explanations"
    out_dir.mkdir(parents=True, exist_ok=True)
    ans_path = out_dir / f"answers_kg_{tag}.jsonl"
    done = set()
    if ans_path.exists():
        done = {(json.loads(l)["qid"], json.loads(l)["condition"]) for l in ans_path.read_text().splitlines()}
    for v in kg.values():
        v["kg_sub_missing"] = None
    conditions = list(KG_CONDITIONS) + ["kg_sub_missing"]
    todo = len(items) * len(conditions) - len(done)
    print(f"{len(items)} questions x {len(KG_CONDITIONS)} graph conditions, {len(done)} already done. Model: {backend.name}")

    t0, n_new = time.time(), 0
    with open(ans_path, "a", encoding="utf-8") as f:
        for it in items:
            for cond in conditions:
                if (it["qid"], cond) in done:
                    continue
                facts = (missing_evidence(it, kg[it["qid"]]["kg_sub"]) if cond == "kg_sub_missing"
                         else kg[it["qid"]][cond])
                user = build_kg_prompt(it["question"], facts, it["fields"])
                try:
                    raw = backend.generate(SYSTEM_KG, user, None, json_mode=True)
                except Exception as e:
                    raw = json.dumps({"_error": str(e)})
                f.write(json.dumps({"qid": it["qid"], "condition": cond, "raw": raw}) + "\n")
                f.flush()
                n_new += 1
                if n_new % 20 == 0:
                    rate = (time.time() - t0) / n_new
                    print(f"  {n_new} answers, {rate:.1f}s each, about {(todo - n_new) * rate / 60:.0f} min left")

    # ---------------- scoring
    by_id = {it["qid"]: it for it in items}
    rows, check_log = [], []
    for line in ans_path.read_text().splitlines():
        r = json.loads(line)
        it = by_id.get(r["qid"])
        if it is None:
            continue
        if r["condition"] == "kg_sub_missing":
            facts = missing_evidence(it, kg[r["qid"]]["kg_sub"])
            s = score(it, r["condition"], r["raw"], evidence_lines=facts)
            s["factual_correct"] = s["controller_faithful"] = bool(s["abstained"])   # only declining is correct
            rows.append(s)
            continue
        facts = kg[r["qid"]][r["condition"]]
        rows.append(score(it, r["condition"], r["raw"], evidence_lines=facts))
        if r["condition"] == "kg_sub":                       # citation checker on the subgraph answers
            raw2, kept, why = verify(it, r["raw"], facts)
            s2 = score(it, "kg_sub+checker", raw2, evidence_lines=facts)
            s2["withheld"] = not kept
            rows.append(s2)
            check_log.append({"qid": r["qid"], "kept": kept, "why": why})
    old_path = out_dir / f"answers_{tag}.jsonl"
    if old_path.exists():                                    # the earlier flat-log answers on the same questions
        for line in old_path.read_text().splitlines():
            r = json.loads(line)
            if r["qid"] in by_id and r["condition"] == "log_only":
                rows.append(score(by_id[r["qid"]], "log_only", r["raw"]))
    else:
        print(f"(no earlier answers file {old_path.name}: log_only comparison skipped)")

    df = pd.DataFrame(rows)
    df["answered"] = df["parsed"] & ~df["abstained"]
    df.to_csv(out_dir / f"scores_kg_{tag}.csv", index=False)
    order = ["log_only", "kg_full", "kg_sub", "kg_sub+checker", "kg_sub_missing"]
    summ = (df.groupby(["qtype", "condition"])[["factual_correct", "controller_faithful", "hallucinated_citation",
                                                 "abstained", "answered", "parsed"]].mean().round(3))
    summ = summ.reindex([(q, c) for q in ("decision", "counterfactual") for c in order if (q, c) in summ.index])
    # selective accuracy: correct among the answers that were actually given (not declined)
    sel = (df[df["answered"]].groupby(["qtype", "condition"])["factual_correct"].mean().round(3)
           .rename("accuracy_when_answered"))
    summ = summ.join(sel)
    print("\nShare of answers (1.0 = 100%), same questions in every row:")
    print(summ.to_string())
    summ.to_csv(out_dir / f"summary_kg_{tag}.csv")

    for q in ("decision", "counterfactual"):
        for a_c, b_c in (("kg_full", "log_only"), ("kg_sub", "log_only"), ("kg_sub", "kg_full")):
            a = df[(df.qtype == q) & (df.condition == a_c)].set_index("qid")["factual_correct"]
            b = df[(df.qtype == q) & (df.condition == b_c)].set_index("qid")["factual_correct"]
            common = a.index.intersection(b.index)
            if len(common):
                print(f"{q:15s} {a_c} {a[common].mean():.3f} vs {b_c} {b[common].mean():.3f} "
                      f"on {len(common)} questions -> McNemar {mcnemar(list(a[common]), list(b[common]))}")
    if check_log:
        kept = sum(c["kept"] for c in check_log)
        print(f"\nCitation checker: kept {kept} of {len(check_log)} subgraph answers "
              f"({len(check_log) - kept} withheld as unsupported)")
        pd.DataFrame(check_log).to_csv(out_dir / f"checker_kg_{tag}.csv", index=False)
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
