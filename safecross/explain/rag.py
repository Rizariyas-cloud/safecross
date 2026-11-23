"""Retrieval over the Dubai Universal Design Code (and any other PDFs you add).

Simple and transparent: the PDF text is split into overlapping chunks and
searched with TF-IDF. No internet or extra model download is needed.
Every chunk keeps its file name and page number so answers can cite it.
"""
from __future__ import annotations

import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _chunks(text: str, size: int, overlap: int) -> list[str]:
    text = " ".join(text.split())
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + size])
        i += max(1, size - overlap)
    return [c for c in out if len(c) > 80]


def build_index(docs_dir: str | Path, out_path: str | Path, size: int = 900, overlap: int = 150) -> int:
    from pypdf import PdfReader
    docs_dir = Path(docs_dir)
    chunks = []
    for pdf in sorted(docs_dir.glob("*.pdf")):
        reader = PdfReader(str(pdf))
        for pno, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            for c in _chunks(text, size, overlap):
                chunks.append({"source": pdf.name, "page": pno, "text": c})
    for txt in sorted(docs_dir.glob("*.txt")):                     # plain-text notes also work
        for c in _chunks(txt.read_text(errors="ignore"), size, overlap):
            chunks.append({"source": txt.name, "page": None, "text": c})
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f)
    return len(chunks)


class Retriever:
    def __init__(self, index_path: str | Path):
        with open(index_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.mat = self.vec.fit_transform([c["text"] for c in self.chunks]) if self.chunks else None

    def search(self, query: str, k: int = 3) -> list[dict]:
        if self.mat is None:
            return []
        sims = cosine_similarity(self.vec.transform([query]), self.mat)[0]
        top = sims.argsort()[::-1][:k]
        return [{**self.chunks[i], "score": float(sims[i])} for i in top if sims[i] > 0]


def load_retriever(index_path: str | Path) -> Retriever | None:
    p = Path(index_path)
    return Retriever(p) if p.exists() else None
