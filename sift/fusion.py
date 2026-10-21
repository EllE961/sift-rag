r"""Rank fusion of lexical and dense candidate lists.

**Reciprocal Rank Fusion** (Cormack et al., 2009) ignores scores entirely:

.. math:: \mathrm{RRF}(d) = \sum_{r \in R} \frac{w_r}{k + \mathrm{rank}_r(d)}

It needs no calibration between retrievers whose scores live on different
scales (BM25 is unbounded, cosine is in [-1, 1]) — which is why it is the
robust default. ``k = 60`` damps the influence of top ranks.

**Convex combination** normalizes each retriever's scores over its own
candidate list and mixes them: ``s(d) = α·norm(s_dense) + (1-α)·norm(s_bm25)``.
It can beat RRF when α is tuned — but α must be tuned on held-out queries, not
the test set (see ``sift.eval.retrieval``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .types import Hit

Candidates = Sequence[tuple[str, float]]  # ranked (pid, score)


def rrf(
    runs: Mapping[str, Candidates],
    k: int = 60,
    weights: Mapping[str, float] | None = None,
    limit: int | None = None,
) -> list[Hit]:
    scores: dict[str, float] = {}
    sources: dict[str, dict[str, float]] = {}
    for name, candidates in runs.items():
        w = 1.0 if weights is None else weights.get(name, 1.0)
        for rank, (pid, _score) in enumerate(candidates, start=1):
            scores[pid] = scores.get(pid, 0.0) + w / (k + rank)
            sources.setdefault(pid, {})[name] = rank
    return _to_hits(scores, sources, limit)


def normalize_scores(scores: np.ndarray, method: str = "minmax") -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0:
        return scores
    if method == "minmax":
        lo, hi = scores.min(), scores.max()
        return np.ones_like(scores) if hi - lo < 1e-12 else (scores - lo) / (hi - lo)
    if method == "zscore":
        sd = scores.std()
        return np.zeros_like(scores) if sd < 1e-12 else (scores - scores.mean()) / sd
    raise ValueError(f"unknown normalization {method!r}")


def convex(
    runs: Mapping[str, Candidates],
    weights: Mapping[str, float],
    norm: str = "minmax",
    limit: int | None = None,
) -> list[Hit]:
    """Weighted sum of per-retriever normalized scores.

    A document missing from one retriever's candidate list receives that
    list's minimum normalized score (0 for min-max) — it was, at best, just
    below the cutoff.
    """
    normed: dict[str, dict[str, float]] = {}
    floors: dict[str, float] = {}
    for name, candidates in runs.items():
        if not candidates:
            normed[name], floors[name] = {}, 0.0
            continue
        raw = np.array([s for _, s in candidates])
        values = normalize_scores(raw, norm)
        normed[name] = {pid: float(v) for (pid, _), v in zip(candidates, values)}
        # A constant-score list normalizes to a single value; missing documents
        # must still rank strictly below every retrieved one.
        floors[name] = float(values.min()) - (1.0 if np.ptp(raw) < 1e-12 else 0.0)
    all_pids = {pid for cands in runs.values() for pid, _ in cands}
    scores: dict[str, float] = {}
    sources: dict[str, dict[str, float]] = {}
    for pid in all_pids:
        total = 0.0
        for name in runs:
            v = normed[name].get(pid, floors[name])
            total += weights.get(name, 0.0) * v
            if pid in normed[name]:
                sources.setdefault(pid, {})[name] = round(normed[name][pid], 6)
        scores[pid] = total
    return _to_hits(scores, sources, limit)


def _to_hits(scores: dict[str, float], sources: dict[str, dict[str, float]], limit: int | None) -> list[Hit]:
    items = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    if limit is not None:
        items = items[:limit]
    return [Hit(pid=pid, score=s, rank=i + 1, sources=sources.get(pid, {})) for i, (pid, s) in enumerate(items)]
