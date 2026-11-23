"""Retrieval evaluation: metrics on hand-checked cases + the dense / hybrid / re-rank plumbing with stand-in models."""
import numpy as np
import pytest

from safecross.explain import retrieval_eval as R

UNITS = [{"page": 10, "text": "Illumination level at least 10 lux on the accessible path"},
         {"page": 11, "text": "Curb cut gradient at crossing at most 8 percent"},
         {"page": 12, "text": "Handrail mounted at a height of 900 mm"},
         {"page": 12, "text": "Additional handrail between 650 and 750 mm"}]


class FakeEncoder:                     # bag of words -> vector (stands in for BGE)
    vocab = ["lux", "illumination", "curb", "gradient", "handrail", "height", "light"]

    def encode(self, texts, normalize_embeddings=True):
        m = np.array([[t.lower().count(w) for w in self.vocab] for t in texts], float) + 1e-6
        return m / np.linalg.norm(m, axis=1, keepdims=True)


class FakeScorer:
    def predict(self, pairs):
        return [len(set(q.lower().split()) & set(t.lower().split())) for q, t in pairs]


def test_metrics():
    s = R.score_ranking([5, 12, 3], {12, 40})
    assert s["recall@1"] == 0 and s["recall@3"] == 1 and s["mrr"] == pytest.approx(0.5)
    assert s["ndcg@5"] == pytest.approx((1 / np.log2(3)) / (1 + 1 / np.log2(3)))
    assert R.score_ranking([1, 2], {9})["mrr"] == 0
    assert R.auroc([3, 2], [1]) == 1.0 and R.auroc([1], [1]) == 0.5
    assert R.parse_pages("61, 70") == {61, 70}


def test_page_ranking_merges_chunks_of_one_page():
    assert R.page_ranking([(2, 0.9), (3, 0.8), (0, 0.1)], UNITS) == [12, 10]


def test_repair_and_bm25():
    assert R.repair("Non slip foor with refectance contrast") == "Non slip floor with reflectance contrast"
    assert R.BM25(UNITS).search("curb gradient", 1)[0][0] == 1


def test_dense_hybrid_rerank_plumbing():
    d = R.Dense(UNITS, encoder=FakeEncoder())
    assert d.search("illumination lux", 1)[0][0] == 0
    fused = R.rrf(R.BM25(UNITS).search("handrail height"), d.search("handrail height"))
    assert {i for i, _ in fused[:2]} == {2, 3}
    rr = R.Reranker(scorer=FakeScorer()).rerank("handrail height of 900 mm", fused, UNITS)
    assert rr[0][0] == 2
