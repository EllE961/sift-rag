import pytest

from sift.datasets import Claim
from sift.eval.claims import cited_sentences, confusion, macro_f1, per_class_f1, sample_balanced, score_condition
from sift.generate import Citation
from sift.types import Passage
from sift.verify import NLIVerifier, SentenceScore, decide, tune_thresholds

SUP, REF, NEI = "SUPPORTED", "REFUTED", "NOT ENOUGH INFO"


def test_decide_rule():
    assert decide(0.9, 0.1, 0.5, 0.5) == SUP
    assert decide(0.4, 0.9, 0.5, 0.5) == REF
    assert decide(0.9, 0.95, 0.5, 0.5) == REF  # stronger contradiction wins
    assert decide(0.3, 0.3, 0.5, 0.5) == NEI


def test_relevance_gate_blocks_unrelated_contradictions():
    scores = [
        SentenceScore(0, 0, 10, entail=0.01, contradict=0.99, neutral=0.0, sim=0.2),  # unrelated
        SentenceScore(0, 11, 20, entail=0.85, contradict=0.05, neutral=0.1, sim=0.8),  # on-topic support
    ]
    assert NLIVerifier.aggregate(scores, tau_s=0.0) == (0.85, 0.99)
    assert NLIVerifier.aggregate(scores, tau_s=0.5) == (0.85, 0.05)
    assert NLIVerifier.aggregate(scores, tau_s=0.9) == (0.0, 0.0)


def test_explain_cites_the_deciding_sentences():
    p = Passage("d", "d", "Aging reduces autophagy. Mice live longer.", end=42)
    scores = [SentenceScore(0, 0, 24, 0.95, 0.01, 0.04, 0.9), SentenceScore(0, 25, 42, 0.2, 0.1, 0.7, 0.4)]
    cits = NLIVerifier.explain(SUP, scores, [p])
    assert [c.cited_text for c in cits] == ["Aging reduces autophagy."]
    assert NLIVerifier.explain(NEI, scores, [p]) == []


def test_tune_thresholds_separates_toy_data():
    def sc(e, c, sim=0.9):
        return [SentenceScore(0, 0, 1, e, c, 1 - e - c, sim)]

    scores = [sc(0.9, 0.05), sc(0.8, 0.1), sc(0.05, 0.9), sc(0.1, 0.8), sc(0.2, 0.3), sc(0.01, 0.97, sim=0.1)]
    gold = [SUP, SUP, REF, REF, NEI, NEI]
    best = tune_thresholds(scores, gold)
    assert best["macro_f1"] == pytest.approx(1.0)
    assert best["tau_sim"] > 0.1  # must learn to ignore the unrelated "contradiction"


def test_f1_and_confusion():
    gold = [SUP, SUP, REF, NEI]
    pred = [SUP, REF, REF, NEI]
    f1 = per_class_f1(gold, pred)
    assert f1[SUP] == pytest.approx(2 / 3) and f1[REF] == pytest.approx(2 / 3) and f1[NEI] == pytest.approx(1.0)
    assert macro_f1(gold, pred) == pytest.approx((2 / 3 + 2 / 3 + 1) / 3)
    assert confusion(gold, pred) == [[1, 1, 0], [0, 1, 0], [0, 0, 1]]


def test_cited_sentences_maps_spans_to_gold_segmentation():
    p = Passage("d", "d", "Aa. Bb. Cc.", metadata={"sentence_spans": [(0, 3), (4, 7), (8, 11)]})
    cit = Citation(n=1, passage_id="d", doc_id="d", title="", cited_text="Bb. Cc", start=4, end=10, doc_start=4, doc_end=10)
    assert cited_sentences(cit, p) == {1, 2}


def test_score_condition_metrics():
    claims = [
        Claim("1", "c1", SUP, evidence_doc_ids=["a"], rationales={"a": [0]}),
        Claim("2", "c2", REF, evidence_doc_ids=["b"], rationales={"b": [2]}),
        Claim("3", "c3", NEI),
    ]
    records = [
        {"label": SUP, "retrieved": ["a", "x"], "cited_docs": ["a", "x"], "cited_sentences": {"a": [0], "x": [1]}},
        {"label": NEI, "retrieved": ["z"], "cited_docs": [], "cited_sentences": {}},
        {"label": NEI, "retrieved": ["y"], "cited_docs": [], "cited_sentences": {}},
    ]
    m = score_condition(claims, records)
    assert m["accuracy"] == pytest.approx(2 / 3, abs=1e-3)  # metrics are rounded to 4 dp
    assert m["evidence_recall"] == pytest.approx(0.5)  # claim 1 found its doc, claim 2 did not
    assert m["citation_precision"] == pytest.approx(0.5)  # cited a (gold) and x (not)
    assert m["citation_recall"] == pytest.approx(0.5)  # a cited, b not
    assert m["rationale_precision"] == pytest.approx(0.5)
    assert m["nei_abstention"] == pytest.approx(1.0)


def test_sample_balanced():
    claims = [Claim(str(i), "c", [SUP, REF, NEI][i % 3]) for i in range(60)]
    s = sample_balanced(claims, 30)
    assert len(s) == 30 and {c.label for c in s} == {SUP, REF, NEI}
    assert sum(c.label == SUP for c in s) == 10
    assert sample_balanced(claims, 30) == s  # deterministic
