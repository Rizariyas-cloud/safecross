"""STEP 25 (NLP) - pick the items for the blinded human evaluation.

12 questions from the main benchmark, each answered by TWO systems from the saved answers:
    our fine-tuned explainer (v1)  and  Qwen2.5-VL-7B   (both given the question-focused facts)
  * 4 decision questions + 5 what-if questions (random, stratified by type)
  * 3 questions with the key facts REMOVED (the correct behaviour is to decline)
For every question the two answers are shown as "Answer A" and "Answer B" in a random order (blinding).

Writes results/human_eval/items.json   (what raters see: no system names) -> send this file to Claude
       results/human_eval/key.json     (which system is A or B, and the true answer) -> keep, used by step 26

    python scripts/25_make_human_eval_items.py
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOTDIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOTDIR))
import importlib.util

from safecross.config import load_config, resolve
from safecross.explain.backends import parse_json
from safecross.explain.evidence import REASON_CODES

_spec = importlib.util.spec_from_file_location("s16", ROOTDIR / "scripts" / "16_run_kg_explanations.py")
S16 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S16)
SYSTEMS = {"ours_v1": "answers_kg_mlx_Qwen2.5-1.5B-Instruct+lora.jsonl",
           "qwen7b": "answers_kg_ollama_qwen2.5vl_7b.jsonl"}


def readable_fact(line: str) -> str:
    m = re.match(r"\[(G\d+)\] \((.*?)\) --(\w+)--> \((.*)\)$", line)
    if not m:
        return line
    gid, s, p, o = m.groups()
    return f"{gid}: {s} | {p.replace('_', ' ')} | {o}"


def readable_answer(item, raw):
    a = parse_json(raw)
    key = "reason_code" if item["qtype"] == "decision" else "answer"
    val = str(a.get(key, "")).strip()
    norm = val.lower().replace(" ", "_")
    if norm in ("cannot_determine", "unknown"):
        shown = "Cannot determine from the facts"
    elif item["qtype"] == "decision":
        shown = REASON_CODES.get(norm, val or "(no answer)")
    else:
        shown = {"yes": "Yes", "no": "No"}.get(norm, val or "(no answer)")
    ev = a.get("evidence", [])
    ev = [str(e).strip("[] ") for e in ev] if isinstance(ev, list) else []
    return {"answer": shown, "explanation": str(a.get("explanation", "")).strip(), "evidence": ev}, norm


def main():
    cfg = load_config()
    res = resolve(cfg["paths"]["results_dir"])
    exp = res / "explanations"
    qd = res / "questions"
    items = {it["qid"]: it for it in json.loads((qd / "questions.json").read_text())}
    kg = json.loads((qd / "kg.json").read_text())
    answers = {}
    for name, fname in SYSTEMS.items():
        path = exp / fname
        if not path.exists():
            sys.exit(f"Missing {path} - run scripts/16_run_kg_explanations.py for this model first.")
        for l in path.read_text().splitlines():
            r = json.loads(l)
            answers[(name, r["qid"], r["condition"])] = r["raw"]
    rng = np.random.default_rng(20261122)
    ok = [q for q in kg if q in items and all((s, q, "kg_sub") in answers and (s, q, "kg_sub_missing") in answers
                                              for s in SYSTEMS)]
    dec = [q for q in ok if items[q]["qtype"] == "decision"]
    cf = [q for q in ok if items[q]["qtype"] == "counterfactual"]
    pick = [("kg_sub", q) for q in rng.choice(dec, 4, replace=False)] + \
           [("kg_sub", q) for q in rng.choice(cf, 5, replace=False)]
    rest = [q for q in dec + cf if ("kg_sub", q) not in pick]
    pick += [("kg_sub_missing", q) for q in rng.choice([q for q in rest if items[q]["qtype"] == "counterfactual"], 2,
                                                       replace=False)]
    pick += [("kg_sub_missing", q) for q in rng.choice([q for q in rest if items[q]["qtype"] == "decision"], 1,
                                                       replace=False)]
    order = rng.permutation(len(pick))
    shown, key = [], {}
    for n, i in enumerate(order, 1):
        cond, q = pick[i]
        it = items[q]
        facts = S16.missing_evidence(it, kg[q]["kg_sub"]) if cond == "kg_sub_missing" else kg[q]["kg_sub"]
        a_first = bool(rng.random() < 0.5)
        sysA, sysB = ("ours_v1", "qwen7b") if a_first else ("qwen7b", "ours_v1")
        ansA, normA = readable_answer(it, answers[(sysA, q, cond)])
        ansB, normB = readable_answer(it, answers[(sysB, q, cond)])
        iid = f"Q{n:02d}"
        shown.append({"id": iid, "question": it["question"], "facts": [readable_fact(l) for l in facts],
                      "A": ansA, "B": ansB})
        truth = "cannot_determine" if cond == "kg_sub_missing" else \
            str(it["truth"]["reason_code" if it["qtype"] == "decision" else "answer"]).lower()
        key[iid] = {"qid": q, "condition": cond, "qtype": it["qtype"], "A": sysA, "B": sysB, "truth": truth,
                    "A_correct": normA == truth or (truth == "cannot_determine" and normA in ("cannot_determine", "unknown")),
                    "B_correct": normB == truth or (truth == "cannot_determine" and normB in ("cannot_determine", "unknown"))}
    out = res / "human_eval"
    out.mkdir(parents=True, exist_ok=True)
    (out / "items.json").write_text(json.dumps(shown, indent=1, ensure_ascii=False))
    (out / "key.json").write_text(json.dumps(key, indent=1))
    print(f"{len(shown)} items: {sum(k['condition'] == 'kg_sub' for k in key.values())} normal, "
          f"{sum(k['condition'] == 'kg_sub_missing' for k in key.values())} with the key facts removed")
    print(f"Answer A is ours in {sum(k['A'] == 'ours_v1' for k in key.values())} of {len(key)} (random)")
    print(f"Send {out / 'items.json'} to Claude. Keep key.json (do not share it with raters).")


if __name__ == "__main__":
    main()
