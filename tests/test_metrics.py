import math

import pytest

from sift.eval.metrics import ap_at_k, evaluate, mrr_at_k, ndcg_at_k, recall_at_k

JUDGED = {"a": 2, "b": 1, "c": 0, "d": 1}  # relevant: a (2), b (1), d (1)


def test_ndcg_hand_computed():
    ranking = ["c", "a", "x", "b"]
    dcg = 0 / math.log2(2) + 2 / math.log2(3) + 0 / math.log2(4) + 1 / math.log2(5)
    idcg = 2 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(ranking, JUDGED, 10) == pytest.approx(dcg / idcg)
    # Cut-off at 2 only sees "c", "a"; ideal@2 is (2, 1)
    assert ndcg_at_k(ranking, JUDGED, 2) == pytest.approx((2 / math.log2(3)) / (2 + 1 / math.log2(3)))
    assert ndcg_at_k(["a", "b", "d"], JUDGED, 3) == pytest.approx(1.0)


def test_recall_mrr_ap():
    ranking = ["c", "a", "x", "b"]
    assert recall_at_k(ranking, JUDGED, 2) == pytest.approx(1 / 3)
    assert recall_at_k(ranking, JUDGED, 10) == pytest.approx(2 / 3)
    assert mrr_at_k(ranking, JUDGED, 10) == pytest.approx(1 / 2)
    assert mrr_at_k(ranking, JUDGED, 1) == 0.0
    # hits at ranks 2 and 4: P@2 = 1/2, P@4 = 2/4; divided by |relevant| = 3
    assert ap_at_k(ranking, JUDGED, 10) == pytest.approx((0.5 + 0.5) / 3)


def test_evaluate_averages_and_penalises_missing():
    qrels = {"q1": {"a": 1}, "q2": {"b": 1}, "q3": {"z": 0}}  # q3 has no relevant docs -> skipped
    run = {"q1": ["a", "x"]}  # q2 missing -> zeros
    m = evaluate(run, qrels, ks=(1, 10))
    assert m["ndcg@10"] == pytest.approx(0.5)
    assert m["recall@1"] == pytest.approx(0.5)
    assert m["mrr@10"] == pytest.approx(0.5)


def test_empty():
    assert evaluate({}, {}) == {f"{m}@{k}": 0.0 for m in ("ndcg", "recall", "mrr", "map") for k in (1, 10, 100)}
