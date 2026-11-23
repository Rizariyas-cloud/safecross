"""STEP 15 (NLP-RQ1) - build the decision knowledge graph for every decision / counterfactual question.

It re-creates the same simulated walk phases that produced results/questions/questions.json
(same seed, same number of episodes), CHECKS that every question and its correct answer come out
identical, and then stores two evidence versions per question:
    kg_full : every fact of that walk phase's knowledge graph
    kg_sub  : only the facts linked to the question (question-focused subgraph)

It also checks the graph against the controller itself: for every "what if track N had never been
detected?" question, the graph's answer (only_occupant fact) must equal the replayed answer.

    python scripts/15_build_kg_questions.py
Output: results/questions/kg.json  (+ results/questions/kg_example.txt, one graph to read)
"""
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from safecross.config import load_config, resolve
from safecross.explain.questions import build_question_set
from safecross.explain.kg import build_kg, lines, subgraph, phase_ends_earlier_without
from safecross.explain.rag import load_retriever


def main():
    cfg = load_config()
    qdir = resolve(cfg["paths"]["results_dir"]) / "questions"
    qfile = qdir / "questions.json"
    if not qfile.exists():
        sys.exit(f"{qfile} not found - run scripts/06_make_questions.py first.")
    old = {it["qid"]: it for it in json.loads(qfile.read_text())}
    sim_ids = [it["episode_id"] for it in old.values() if it["qtype"] in ("decision", "counterfactual", "visual")]
    n_ep = max(sim_ids) - 10000 + 1
    n_ep = max(n_ep, cfg["explain"]["n_decision_episodes"])
    print(f"Re-creating {n_ep} simulated walk phases (seed {cfg['seed']}) ...")
    with tempfile.TemporaryDirectory() as tmp:
        new = build_question_set(cfg, n_ep, Path(tmp), seed=cfg["seed"], keep_context=True)

    retriever = load_retriever(resolve(cfg["paths"]["results_dir"]) / "rag_index.json")
    kg, check, cf_agree = {}, Counter(), Counter()
    example = None
    for it in new:
        if it["qtype"] not in ("decision", "counterfactual"):
            continue
        o = old.get(it["qid"])
        if o is None:
            check["not in questions.json"] += 1
            continue
        same = (o["question"] == it["question"] and
                json.dumps(o["truth"], sort_keys=True, default=float) == json.dumps(it["truth"], sort_keys=True, default=float))
        check["identical" if same else "DIFFERENT"] += 1
        if not same:
            continue
        ep, res = it["_ctx"]
        facts = build_kg(ep, res, cfg, retriever)
        sub = subgraph(facts, it["question"])
        kg[it["qid"]] = {"kg_full": lines(facts), "kg_sub": lines(sub)}
        if it["qtype"] == "counterfactual":
            tid = int(it["qid"].split("_cf_track")[1])
            cf_agree[phase_ends_earlier_without(res, cfg, tid) == (it["truth"]["answer"] == "yes")] += 1
        if example is None and it["qtype"] == "counterfactual":
            example = (it["question"], lines(facts), lines(sub))

    print("Questions re-created:", dict(check))
    if check["DIFFERENT"] or not kg:
        sys.exit("The re-created questions do not match questions.json - do NOT use kg.json; send this output.")
    print(f"Graph vs controller replay on 'what if' questions: agree {cf_agree[True]}, disagree {cf_agree[False]}")
    (qdir / "kg.json").write_text(json.dumps(kg, indent=1))
    full = [len(v["kg_full"]) for v in kg.values()]
    sub = [len(v["kg_sub"]) for v in kg.values()]
    print(f"{len(kg)} questions | facts per graph: mean {sum(full) / len(full):.0f} | "
          f"subgraph: mean {sum(sub) / len(sub):.0f}")
    if example:
        q, f, s = example
        (qdir / "kg_example.txt").write_text(f"QUESTION: {q}\n\nFULL GRAPH ({len(f)} facts)\n" + "\n".join(f) +
                                             f"\n\nSUBGRAPH ({len(s)} facts)\n" + "\n".join(s) + "\n")
    print(f"Saved {qdir / 'kg.json'} and kg_example.txt")


if __name__ == "__main__":
    main()
