r"""Ranking metrics, implemented from scratch with trec_eval semantics.

For a ranked list :math:`r_1..r_k` and graded judgements :math:`rel(d)`:

* nDCG@k  = DCG@k / IDCG@k,  DCG@k = \sum_i rel(r_i) / \log_2(i + 1)
  (linear gain, as in trec_eval / pytrec_eval which BEIR uses)
* Recall@k = |relevant \cap top-k| / |relevant|
* MRR@k   = 1 / rank of the first relevant document (0 if none in top-k)
* MAP@k   = (1/|relevant|) \sum_{i<=k, r_i relevant} Precision@i

Documents with judgement 0 are non-relevant. Queries without any relevant
document are skipped (they would make every metric undefined).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

Qrels = Mapping[str, Mapping[str, int]]
Run = Mapping[str, Sequence[str]]  # qid -> ranked doc ids


def dcg(gains: Sequence[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranking: Sequence[str], judgements: Mapping[str, int], k: int) -> float:
    gains = [float(judgements.get(d, 0)) for d in ranking[:k]]
    ideal = sorted((float(v) for v in judgements.values() if v > 0), reverse=True)[:k]
    idcg = dcg(ideal)
    return dcg(gains) / idcg if idcg > 0 else 0.0


def recall_at_k(ranking: Sequence[str], judgements: Mapping[str, int], k: int) -> float:
    relevant = {d for d, v in judgements.items() if v > 0}
    if not relevant:
        return 0.0
    return len(relevant.intersection(ranking[:k])) / len(relevant)


def mrr_at_k(ranking: Sequence[str], judgements: Mapping[str, int], k: int) -> float:
    for i, d in enumerate(ranking[:k]):
        if judgements.get(d, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def ap_at_k(ranking: Sequence[str], judgements: Mapping[str, int], k: int) -> float:
    relevant = {d for d, v in judgements.items() if v > 0}
    if not relevant:
        return 0.0
    hits, total = 0, 0.0
    for i, d in enumerate(ranking[:k]):
        if d in relevant:
            hits += 1
            total += hits / (i + 1)
    return total / len(relevant)


METRICS = {"ndcg": ndcg_at_k, "recall": recall_at_k, "mrr": mrr_at_k, "map": ap_at_k}


def evaluate(
    run: Run,
    qrels: Qrels,
    ks: Sequence[int] = (1, 10, 100),
    metrics: Sequence[str] = ("ndcg", "recall", "mrr", "map"),
) -> dict[str, float]:
    """Mean of each ``metric@k`` over judged queries; keys like ``ndcg@10``.

    Queries in ``qrels`` that are missing from ``run`` score 0 (a retriever
    that returns nothing should be penalised, not ignored).
    """
    per_query = evaluate_per_query(run, qrels, ks, metrics)
    if not per_query:
        return {f"{m}@{k}": 0.0 for m in metrics for k in ks}
    keys = next(iter(per_query.values())).keys()
    return {key: sum(q[key] for q in per_query.values()) / len(per_query) for key in keys}


def evaluate_per_query(
    run: Run,
    qrels: Qrels,
    ks: Sequence[int] = (1, 10, 100),
    metrics: Sequence[str] = ("ndcg", "recall", "mrr", "map"),
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for qid, judgements in qrels.items():
        if not any(v > 0 for v in judgements.values()):
            continue
        ranking = list(run.get(qid, []))
        out[qid] = {f"{m}@{k}": METRICS[m](ranking, judgements, k) for m in metrics for k in ks}
    return out
