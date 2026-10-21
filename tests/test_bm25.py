import math

import numpy as np
import pytest

from sift.index.bm25 import BM25Index, top_k
from sift.text import Analyzer

DOCS = {
    "d1": "the quick brown fox jumps over the lazy dog",
    "d2": "a quick brown dog",
    "d3": "lazy cats sleep all day long in the sun",
    "d4": "foxes and dogs are not friends; the fox is quick",
}


@pytest.fixture()
def index():
    # No stemming / stopwords so the hand computation below is transparent.
    return BM25Index(k1=1.2, b=0.75, analyzer=Analyzer(stopwords=frozenset(), stem=False)).build(
        list(DOCS), list(DOCS.values())
    )


def reference_bm25(query: str, k1=1.2, b=0.75):
    toks = {d: t.replace(";", " ").split() for d, t in DOCS.items()}
    n = len(toks)
    avgdl = sum(len(t) for t in toks.values()) / n
    scores = {}
    for d, terms in toks.items():
        s = 0.0
        for q in query.split():
            df = sum(1 for t in toks.values() if q in t)
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            tf = terms.count(q)
            s += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(terms) / avgdl))
        scores[d] = s
    return scores


@pytest.mark.parametrize("query", ["quick fox", "lazy", "dog dog quick", "sun", "missing words"])
def test_scores_match_reference_formula(index, query):
    ref = reference_bm25(query)
    got = dict(zip(index.ids, index.score(query)))
    for d in DOCS:
        assert got[d] == pytest.approx(ref[d], rel=1e-5, abs=1e-6)


def test_search_ranking_and_zero_filter(index):
    hits = index.search("lazy", k=10)
    assert {pid for pid, _ in hits} == {"d1", "d3"}
    assert all(s > 0 for _, s in hits)
    assert index.search("zebra") == []


def test_set_params_and_roundtrip(index, tmp_path):
    before = index.score("quick fox").copy()
    index.set_params(0.9, 0.4)
    after = index.score("quick fox")
    assert not np.allclose(before, after)
    index.save(tmp_path / "bm25")
    loaded = BM25Index.load(tmp_path / "bm25", analyzer=index.analyzer)
    assert loaded.k1 == 0.9 and loaded.b == 0.4
    np.testing.assert_allclose(loaded.score("quick fox"), after, rtol=1e-6)


def test_explain_sums_to_score(index):
    contrib = index.explain("quick fox", "d4")
    total = dict(zip(index.ids, index.score("quick fox")))["d4"]
    assert sum(contrib.values()) == pytest.approx(total, rel=1e-5)


def test_top_k_ordering():
    scores = np.array([0.1, 3.0, 2.0, 3.0, -1.0])
    assert top_k(scores, list("abcde"), 3) == [("b", 3.0), ("d", 3.0), ("c", 2.0)]
    assert top_k(scores, list("abcde"), 50)[-1] == ("e", -1.0)
    assert top_k(np.array([]), [], 5) == []


def test_build_validates_lengths():
    with pytest.raises(ValueError):
        BM25Index().build(["a"], [])
