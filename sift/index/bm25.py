r"""Okapi BM25 over a sparse inverted index.

For query :math:`q` and document :math:`d`:

.. math::

    \mathrm{BM25}(q, d) = \sum_{t \in q} \mathrm{idf}(t)\,
        \frac{tf_{t,d}\,(k_1 + 1)}{tf_{t,d} + k_1\,(1 - b + b\,|d| / \overline{|d|})}

    \mathrm{idf}(t) = \ln\!\Big(1 + \frac{N - df_t + 0.5}{df_t + 0.5}\Big)

(the Lucene idf, which is always positive). Because the per-(term, doc) factor
does not depend on the query, the whole right-hand side is precomputed into a
``terms x docs`` CSR matrix ``W`` at build time. Scoring a query is then a
single sparse row-gather + sum: ``scores = q_tf @ W[q_terms]`` — no Python
loop over postings, which makes it ~100x faster than a naive implementation.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from ..text import Analyzer


class BM25Index:
    def __init__(self, k1: float = 0.9, b: float = 0.4, analyzer: Analyzer | None = None) -> None:
        self.k1 = k1
        self.b = b
        self.analyzer = analyzer or Analyzer()
        self.vocab: dict[str, int] = {}
        self.ids: list[str] = []
        self.W: sp.csr_matrix | None = None  # terms x docs
        self.idf: np.ndarray | None = None
        self.doc_len: np.ndarray | None = None
        self.avgdl: float = 0.0

    # ----------------------------------------------------------------- build
    def build(self, ids: Sequence[str], texts: Sequence[str]) -> BM25Index:
        if len(ids) != len(texts):
            raise ValueError("ids and texts must have the same length")
        self.ids = list(ids)
        vocab: dict[str, int] = {}
        rows: list[int] = []
        cols: list[int] = []
        vals: list[int] = []
        doc_len = np.zeros(len(texts), dtype=np.float32)
        for d, text in enumerate(texts):
            counts = Counter(self.analyzer(text))
            doc_len[d] = sum(counts.values())
            for term, tf in counts.items():
                t = vocab.setdefault(term, len(vocab))
                rows.append(t)
                cols.append(d)
                vals.append(tf)
        self.vocab = vocab
        n_docs, n_terms = len(texts), len(vocab)
        tf = sp.csr_matrix(
            (np.asarray(vals, dtype=np.float32), (np.asarray(rows), np.asarray(cols))),
            shape=(n_terms, n_docs),
        )
        df = np.diff(tf.indptr).astype(np.float32)  # docs per term (one entry per (t, d))
        self.idf = np.log1p((n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
        self.doc_len = doc_len
        self.avgdl = float(doc_len.mean()) if n_docs else 0.0
        self._compute_weights(tf)
        return self

    def _compute_weights(self, tf: sp.csr_matrix) -> None:
        assert self.idf is not None and self.doc_len is not None
        k1, b = self.k1, self.b
        norm = k1 * (1.0 - b + b * self.doc_len / max(self.avgdl, 1e-9))  # per doc
        data = tf.data
        doc_of_entry = tf.indices
        term_of_entry = np.repeat(np.arange(tf.shape[0]), np.diff(tf.indptr))
        w = self.idf[term_of_entry] * data * (k1 + 1.0) / (data + norm[doc_of_entry])
        self.W = sp.csr_matrix((w.astype(np.float32), tf.indices.copy(), tf.indptr.copy()), shape=tf.shape)
        self._tf = tf  # kept so k1/b can be re-tuned without re-tokenizing

    def set_params(self, k1: float, b: float) -> None:
        """Re-weight in place (cheap: no re-tokenization)."""
        self.k1, self.b = k1, b
        self._compute_weights(self._tf)

    # ---------------------------------------------------------------- search
    def query_vector(self, query: str) -> tuple[np.ndarray, np.ndarray]:
        counts = Counter(t for t in self.analyzer(query) if t in self.vocab)
        term_ids = np.fromiter((self.vocab[t] for t in counts), dtype=np.int64, count=len(counts))
        weights = np.fromiter(counts.values(), dtype=np.float32, count=len(counts))
        return term_ids, weights

    def score(self, query: str) -> np.ndarray:
        """Dense vector of BM25 scores for every document."""
        assert self.W is not None, "index not built"
        term_ids, weights = self.query_vector(query)
        if term_ids.size == 0:
            return np.zeros(len(self.ids), dtype=np.float32)
        return np.asarray(weights @ self.W[term_ids]).ravel()

    def search(self, query: str, k: int = 100) -> list[tuple[str, float]]:
        scores = self.score(query)
        return [(pid, s) for pid, s in top_k(scores, self.ids, k) if s > 0.0]

    def explain(self, query: str, pid: str) -> dict[str, float]:
        """Per-term contribution to a document's score (for the UI trace)."""
        assert self.W is not None
        d = self.ids.index(pid)
        term_ids, weights = self.query_vector(query)
        inv = {v: k for k, v in self.vocab.items() if v in set(term_ids.tolist())}
        return {inv[int(t)]: float(w * self.W[int(t), d]) for t, w in zip(term_ids, weights)}

    # ----------------------------------------------------------- persistence
    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        sp.save_npz(path / "bm25_tf.npz", self._tf)
        np.save(path / "bm25_doclen.npy", self.doc_len)
        meta = {"k1": self.k1, "b": self.b, "ids": self.ids, "vocab": self.vocab}
        (path / "bm25_meta.json").write_text(json.dumps(meta), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, analyzer: Analyzer | None = None) -> BM25Index:
        meta = json.loads((path / "bm25_meta.json").read_text(encoding="utf-8"))
        index = cls(k1=meta["k1"], b=meta["b"], analyzer=analyzer)
        index.ids = meta["ids"]
        index.vocab = meta["vocab"]
        tf = sp.load_npz(path / "bm25_tf.npz").tocsr()
        index.doc_len = np.load(path / "bm25_doclen.npy")
        index.avgdl = float(index.doc_len.mean()) if len(index.doc_len) else 0.0
        n_docs = len(index.ids)
        df = np.diff(tf.indptr).astype(np.float32)
        index.idf = np.log1p((n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
        index._compute_weights(tf)
        return index

    @property
    def n_terms(self) -> int:
        return len(self.vocab)


def top_k(scores: np.ndarray, ids: Sequence[str], k: int) -> list[tuple[str, float]]:
    """Top-k (id, score) pairs by descending score using an O(n) partition."""
    n = scores.shape[0]
    if n == 0:
        return []
    k = min(k, n)
    idx = np.argpartition(-scores, k - 1)[:k] if k < n else np.arange(n)
    idx = idx[np.argsort(-scores[idx], kind="stable")]
    return [(ids[i], float(scores[i])) for i in idx]
