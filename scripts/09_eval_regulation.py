"""STEP 9 - Evaluate retrieval for regulation questions (Dubai Universal Design Code).

Fill in configs/regulation_questions.yaml first (questions + the correct page numbers).

    python scripts/09_eval_regulation.py
    python scripts/09_eval_regulation.py --backend ollama --model qwen2.5vl:7b   # also check answers cite passages
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from safecross.config import load_config, resolve, ROOT
from safecross.explain.backends import make_backend
from safecross.explain.prompts import SYSTEM, build_user_prompt
from safecross.explain.rag import load_retriever


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=None, choices=[None, "ollama", "hf"])
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    ret = load_retriever(resolve(cfg["paths"]["results_dir"]) / "rag_index.json")
    if ret is None:
        sys.exit("Run scripts/05_build_rag_index.py first.")
    qs = yaml.safe_load((ROOT / "configs" / "regulation_questions.yaml").read_text())["questions"]
    backend = make_backend(cfg, args.backend, args.model) if args.backend else None
    k = cfg["explain"]["rag_top_k"]
    rows = []
    for q in qs:
        hits = ret.search(q["question"], k)
        pages = [h["page"] for h in hits]
        exp = set(q.get("expected_pages") or [])
        row = {"id": q["id"], "retrieved_pages": pages,
               "hit_at_k": bool(exp & set(pages)) if exp else None}
        if backend:
            reply = backend.generate(SYSTEM, build_user_prompt(q["question"], None, None, hits))
            row["cites_passage"] = bool(re.search(r"\[R\d\]", reply))
            row["answer"] = reply[:400]
        rows.append(row)
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    if df["hit_at_k"].notna().any():
        print(f"\nRetrieval hit@{k}: {df['hit_at_k'].dropna().mean():.2f}")
    out = resolve(cfg["paths"]["results_dir"]) / "regulation_eval.csv"
    df.to_csv(out, index=False)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
