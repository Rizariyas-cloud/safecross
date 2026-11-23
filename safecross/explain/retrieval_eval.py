"""Retrieval over the Dubai Universal Design guide: methods + evaluation (NLP, regulation questions).

Why: SafeCross answers regulation questions from the guide. Version 1 used TF-IDF over 900-character chunks.
Here we test, on 35 team-checked questions (30 answerable + 5 the guide does not cover):

  tfidf_chunks   version 1 (baseline): TF-IDF over 900-character overlapping chunks
  tfidf_pages    TF-IDF over page units (the guide prints each requirement on its own page, with its code)
  bm25_raw       BM25 keyword search, text as extracted (fi/fl ligatures missing: "foor", "refectance")
  bm25           BM25 after ligature repair (text normalisation)
  dense          meaning-based search with BGE embeddings (BAAI/bge-small-en-v1.5)
  hybrid         BM25 + dense combined by Reciprocal Rank Fusion (RRF, k = 60)
  hybrid_rerank  hybrid top 20 re-ordered by a cross-encoder re-ranker

Scores (answerable questions): Recall@1/3/5 = a correct page is in the top 1/3/5; MRR = 1 / rank of the first
correct page (0 if not in the top 10); nDCG@5 = ranking quality with all correct pages counted.
"Not covered" questions: the best search score should be LOW. AUROC = how well the top score separates
answerable from not-covered questions (1.0 = perfectly, 0.5 = no better than chance), threshold-free.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

# Words the PDF breaks because the fi / fl / ff ligature glyphs are dropped during text extraction.
LIGATURE_FIXES = {
    "foor": "floor", "foors": "floors", "fooring": "flooring", "refectance": "reflectance", "refective": "reflective",
    "fxed": "fixed", "fight": "flight", "fights": "flights", "frmly": "firmly", "fush": "flush", "identifed": "identified",
    "felds": "fields", "fexibility": "flexibility", "fexible": "flexible", "traffc": "traffic", "suffcient": "sufficient",
    "fller": "filler", "fre": "fire", "fashes": "flashes", "fat": "flat", "frst": "first", "fnish": "finish",
    "fnished": "finished", "specifed": "specified", "signifcant": "significant", "defned": "defined",
    "confguration": "configuration", "fttings": "fittings", "fow": "flow", "fows": "flows", "fuid": "fluid",
}
_LIG = re.compile(r"\b(" + "|".join(sorted(LIGATURE_FIXES, key=len, reverse=True)) + r")\b", re.IGNORECASE)
STOP = set("a an the of to in on at for and or is are be by with what how which where when does do must should "
           "can it its this that as from there their than any much many allowed required need needed".split())


def repair(text: str) -> str:
    return _LIG.sub(lambda m: LIGATURE_FIXES[m.group(0).lower()], text)


def tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP]


def load_pages(pdf_path, fix: bool = True, min_chars: int = 30) -> list[dict]:
    from pypdf import PdfReader
    pages = []
    for pno, page in enumerate(PdfReader(str(pdf_path)).pages, start=1):
        t = " ".join((page.extract_text() or "").split())
        if len(t) >= min_chars:
            pages.append({"page": pno, "text": repair(t) if fix else t})
    return pages


def char_chunks(pages_raw: list[dict], size: int = 900, overlap: int = 150) -> list[dict]:
    """Version-1 chunking (as in rag.py): fixed-size overlapping character windows inside each page."""
    out = []
    for p in pages_raw:
        t, i = p["text"], 0
        while i < len(t):
            c = t[i:i + size]
            if len(c) > 80 or i == 0:
                out.append({"page": p["page"], "text": c})
            i += max(1, size - overlap)
    return out


# ------------------------------------------------------------------ retrievers: search(q, k) -> [(unit_index, score)]
class TFIDF:
    def __init__(self, units):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.mat = self.vec.fit_transform([u["text"] for u in units])

    def search(self, q, k=20):
        from sklearn.metrics.pairwise import cosine_similarity
        s = cosine_similarity(self.vec.transform([q]), self.mat)[0]
        top = s.argsort()[::-1][:k]
        return [(int(i), float(s[i])) for i in top]


class BM25:
    """Okapi BM25 (k1 = 1.5, b = 0.75), written out so every step is visible."""

    def __init__(self, units, k1=1.5, b=0.75):
        self.docs = [tokens(u["text"]) for u in units]
        self.k1, self.b = k1, b
        self.avg = np.mean([len(d) for d in self.docs])
        df = Counter(t for d in self.docs for t in set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.tf = [Counter(d) for d in self.docs]

    def search(self, q, k=20):
        qt = tokens(q)
        s = np.zeros(len(self.docs))
        for i, (tf, d) in enumerate(zip(self.tf, self.docs)):
            norm = self.k1 * (1 - self.b + self.b * len(d) / self.avg)
            s[i] = sum(self.idf.get(t, 0) * tf[t] * (self.k1 + 1) / (tf[t] + norm) for t in qt if t in tf)
        top = s.argsort()[::-1][:k]
        return [(int(i), float(s[i])) for i in top]


class Dense:
    PREFIX = "Represent this sentence for searching relevant passages: "      # BGE's query instruction

    def __init__(self, units, model="BAAI/bge-small-en-v1.5", encoder=None):
        if encoder is None:
            from sentence_transformers import SentenceTransformer
            encoder = SentenceTransformer(model)
        self.enc = encoder
        self.emb = np.asarray(self.enc.encode([u["text"] for u in units], normalize_embeddings=True))

    def search(self, q, k=20):
        v = np.asarray(self.enc.encode([self.PREFIX + q], normalize_embeddings=True))[0]
        s = self.emb @ v
        top = s.argsort()[::-1][:k]
        return [(int(i), float(s[i])) for i in top]


def rrf(*rankings, k: int = 60, top: int = 20):
    """Reciprocal Rank Fusion: score = sum over methods of 1 / (k + rank). Needs no score calibration."""
    agg = Counter()
    for r in rankings:
        for rank, (i, _) in enumerate(r, start=1):
            agg[i] += 1.0 / (k + rank)
    return [(i, s) for i, s in agg.most_common(top)]


class Reranker:
    def __init__(self, model="cross-encoder/ms-marco-MiniLM-L-6-v2", scorer=None):
        if scorer is None:
            from sentence_transformers import CrossEncoder
            scorer = CrossEncoder(model)
        self.ce = scorer

    def rerank(self, q, cand, units, k=20):
        if not cand:
            return []
        s = self.ce.predict([(q, units[i]["text"]) for i, _ in cand])
        order = np.argsort(-np.asarray(s))
        return [(cand[j][0], float(s[j])) for j in order][:k]


# ------------------------------------------------------------------ evaluation
def page_ranking(hits, units) -> list[int]:
    """Unit hits -> pages in rank order (first occurrence; several chunks can share a page)."""
    seen, out = set(), []
    for i, _ in hits:
        p = units[i]["page"]
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def score_ranking(pages: list[int], relevant: set[int]) -> dict:
    first = next((r for r, p in enumerate(pages[:10], start=1) if p in relevant), None)
    dcg = sum(1 / math.log2(r + 1) for r, p in enumerate(pages[:5], start=1) if p in relevant)
    idcg = sum(1 / math.log2(r + 1) for r in range(1, min(len(relevant), 5) + 1))
    return {"recall@1": float(first == 1), "recall@3": float(first is not None and first <= 3),
            "recall@5": float(first is not None and first <= 5), "mrr": 1 / first if first else 0.0,
            "ndcg@5": dcg / idcg if idcg else 0.0, "first_correct_rank": first}


def auroc(pos: list[float], neg: list[float]) -> float:
    """P(score of an answerable question > score of a not-covered one); ties count half (Mann-Whitney)."""
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def parse_pages(cell) -> set[int]:
    return {int(x) for x in re.findall(r"\d+", str(cell))}
