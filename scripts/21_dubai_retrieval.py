"""STEP 21 (NLP) - which search method finds the right page of the Dubai Universal Design guide?

35 questions written from AI drafts and checked by the team (configs/dubai_questions.csv): 30 answerable,
5 the guide does not cover (signal timing, walking speed, countdowns, audible push buttons, refuge islands).

    pip install sentence-transformers                 # once (meaning-based search + re-ranker)
    python scripts/21_dubai_retrieval.py              # all methods (first run downloads ~220 MB of models)
    python scripts/21_dubai_retrieval.py --no-models  # keyword methods only (no download)

The guide PDF is looked for in data/rag_docs/ (or give --pdf). Results: results/dubai_retrieval/
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from safecross.config import ROOT, load_config, resolve
from safecross.explain import retrieval_eval as R
from safecross.metrics import mcnemar


def find_pdf(arg):
    if arg:
        return resolve(arg)
    docs = sorted(resolve(load_config()["paths"]["rag_docs_dir"]).glob("*.pdf"))
    dubai = [p for p in docs if "dubai" in p.name.lower() or "universal" in p.name.lower()]
    if not (dubai or docs):
        sys.exit("Put the Dubai guide PDF in data/rag_docs/ (or use --pdf path/to/guide.pdf).")
    return (dubai or docs)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default=None)
    ap.add_argument("--questions", default="configs/dubai_questions.csv")
    ap.add_argument("--no-models", action="store_true", help="skip dense search and the re-ranker")
    ap.add_argument("--dense-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--rerank-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--out", default="dubai_retrieval")
    args = ap.parse_args()

    pdf = find_pdf(args.pdf)
    qs = pd.read_csv(resolve(args.questions))
    raw = R.load_pages(pdf, fix=False)
    fixed = R.load_pages(pdf, fix=True)
    chunks = R.char_chunks(raw)
    print(f"{pdf.name}: {len(fixed)} pages with text; version-1 chunks: {len(chunks)}; "
          f"{len(qs)} questions ({int((qs['answerable'] == 'yes').sum())} answerable)")
    changed = sum(a["text"] != b["text"] for a, b in zip(raw, fixed))
    print(f"Ligature repair changed {changed} pages (e.g. 'foor' -> 'floor', 'refectance' -> 'reflectance')")

    methods = {
        "tfidf_chunks (v1 baseline)": (chunks, R.TFIDF(chunks).search),
        "tfidf_pages": (fixed, R.TFIDF(fixed).search),
        "bm25_raw (no repair)": (raw, R.BM25(raw).search),
        "bm25": (fixed, R.BM25(fixed).search),
    }
    if not args.no_models:
        try:
            dense = R.Dense(fixed, args.dense_model)
            bm25 = R.BM25(fixed)
            rer = R.Reranker(args.rerank_model)
            methods["dense (BGE)"] = (fixed, dense.search)
            methods["hybrid (BM25 + BGE, RRF)"] = (fixed, lambda q: R.rrf(bm25.search(q), dense.search(q)))
            methods["hybrid + re-ranker"] = (fixed, lambda q: rer.rerank(q, R.rrf(bm25.search(q), dense.search(q)), fixed))
        except ImportError:
            print('\nsentence-transformers is not installed: run  pip install sentence-transformers'
                  "  then run this script again. Continuing with keyword methods only.\n")

    rows, retrieved = [], []
    for name, (units, search) in methods.items():
        for _, q in qs.iterrows():
            hits = search(q["question"])
            pages = R.page_ranking(hits, units)
            row = {"method": name, "id": q["id"], "answerable": q["answerable"] == "yes",
                   "top_score": hits[0][1] if hits else 0.0, "top_pages": pages[:5]}
            if row["answerable"]:
                row.update(R.score_ranking(pages, R.parse_pages(q["pages"])))
            rows.append(row)
            retrieved.append({"method": name, "id": q["id"], "question": q["question"], "pages": pages[:10]})
    df = pd.DataFrame(rows)

    summ = []
    for name in methods:
        d = df[df.method == name]
        a = d[d.answerable]
        summ.append({"method": name, "n": len(a),
                     **{m: a[m].mean() for m in ("recall@1", "recall@3", "recall@5", "mrr", "ndcg@5")},
                     "not_covered_AUROC": R.auroc(list(a.top_score), list(d[~d.answerable].top_score))})
    summ = pd.DataFrame(summ)
    out = resolve(load_config()["paths"]["results_dir"]) / args.out
    out.mkdir(parents=True, exist_ok=True)
    summ.to_csv(out / "summary.csv", index=False)
    df.to_csv(out / "per_question.csv", index=False)
    with open(out / "retrieved.jsonl", "w") as f:
        for r in retrieved:
            f.write(json.dumps(r) + "\n")

    print("\nRetrieval on 30 answerable questions (1.0 = 100%); AUROC on the 5 not-covered questions:")
    print(summ.round(3).to_string(index=False))
    base = df[(df.method == "tfidf_chunks (v1 baseline)") & df.answerable].set_index("id")["recall@3"]
    for name in list(methods)[1:]:
        other = df[(df.method == name) & df.answerable].set_index("id")["recall@3"]
        print(f"Recall@3  {name} {other.mean():.3f} vs v1 baseline {base.mean():.3f} -> McNemar "
              f"{mcnemar(other.astype(bool).values, base.astype(bool).values)}")
    misses = df[df.answerable & (df["recall@5"] == 0)].groupby("method")["id"].apply(list)
    print("\nQuestions with no correct page in the top 5:")
    for name, ids in misses.items():
        print(f"  {name}: {', '.join(ids)}")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
