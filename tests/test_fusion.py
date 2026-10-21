import numpy as np
import pytest

from sift.fusion import convex, normalize_scores, rrf


def test_rrf_hand_computed():
    runs = {"bm25": [("a", 10.0), ("b", 5.0), ("c", 1.0)], "dense": [("b", 0.9), ("d", 0.8)]}
    hits = rrf(runs, k=60)
    scores = {h.pid: h.score for h in hits}
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["a"] == pytest.approx(1 / 61)
    assert scores["d"] == pytest.approx(1 / 62)
    assert [h.pid for h in hits] == ["b", "a", "d", "c"]
    assert hits[0].sources == {"bm25": 2, "dense": 1}
    assert [h.rank for h in hits] == [1, 2, 3, 4]


def test_rrf_weights_and_limit():
    runs = {"x": [("a", 1.0)], "y": [("b", 1.0)]}
    hits = rrf(runs, weights={"x": 1.0, "y": 2.0}, limit=1)
    assert len(hits) == 1 and hits[0].pid == "b"


def test_normalize_scores():
    np.testing.assert_allclose(normalize_scores(np.array([1.0, 3.0, 2.0])), [0.0, 1.0, 0.5])
    np.testing.assert_allclose(normalize_scores(np.array([4.0, 4.0])), [1.0, 1.0])
    z = normalize_scores(np.array([1.0, 2.0, 3.0]), "zscore")
    assert z.mean() == pytest.approx(0.0) and z.std() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        normalize_scores(np.array([1.0]), "nope")


def test_convex_combination():
    runs = {"bm25": [("a", 10.0), ("b", 0.0)], "dense": [("b", 1.0), ("c", 0.5), ("a", 0.0)]}
    hits = convex(runs, {"dense": 0.5, "bm25": 0.5})
    s = {h.pid: h.score for h in hits}
    assert s["a"] == pytest.approx(0.5 * 1.0 + 0.5 * 0.0)
    assert s["b"] == pytest.approx(0.5 * 0.0 + 0.5 * 1.0)
    assert s["c"] == pytest.approx(0.5 * 0.0 + 0.5 * 0.5)  # missing from bm25 -> floor 0
    only_dense = convex(runs, {"dense": 1.0, "bm25": 0.0})
    assert only_dense[0].pid == "b"


def test_convex_single_candidate_still_outranks_missing_docs():
    runs = {"bm25": [("only", 3.2)], "dense": [("a", 0.9), ("b", 0.8), ("only", 0.1)]}
    hits = convex(runs, {"dense": 0.0, "bm25": 1.0})
    assert hits[0].pid == "only"
