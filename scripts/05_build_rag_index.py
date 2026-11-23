"""STEP 5 - Build the search index over the Dubai Universal Design Code.

1. Download the Dubai Universal Design Code PDF from Dubai Municipality's website.
2. Put it in data/rag_docs/  (you can add more official PDFs there too).
3. Run:  python scripts/05_build_rag_index.py
4. Test: python scripts/05_build_rag_index.py --ask "requirements for pedestrian crossings"
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from safecross.config import load_config, resolve
from safecross.explain.rag import build_index, load_retriever


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--ask", default=None, help="test a search after building")
    args = ap.parse_args()
    docs = resolve(cfg["paths"]["rag_docs_dir"])
    docs.mkdir(parents=True, exist_ok=True)
    index = resolve(cfg["paths"]["results_dir"]) / "rag_index.json"
    n = build_index(docs, index, cfg["explain"]["rag_chunk_chars"], cfg["explain"]["rag_chunk_overlap"])
    if n == 0:
        sys.exit(f"No text found. Put the Dubai Universal Design Code PDF in {docs} and run again.")
    print(f"Indexed {n} chunks -> {index}")
    if args.ask:
        for i, h in enumerate(load_retriever(index).search(args.ask, cfg["explain"]["rag_top_k"]), 1):
            print(f"\n[R{i}] {h['source']} page {h['page']} (score {h['score']:.2f})\n{h['text'][:500]}")


if __name__ == "__main__":
    main()
