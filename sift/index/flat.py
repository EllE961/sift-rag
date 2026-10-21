"""Exact (brute-force) inner-product index — the ground truth for ANN recall."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np


class FlatIndex:
    """Stores L2-normalized vectors; search is one GEMV/GEMM + partial sort."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.vectors = np.zeros((0, dim), dtype=np.float32)
        self.ids: list[str] = []

    def add(self, ids: Sequence[str], vectors: np.ndarray) -> FlatIndex:
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.shape[1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {vectors.shape[1]}")
        self.vectors = np.vstack([self.vectors, vectors]) if len(self.ids) else vectors
        self.ids.extend(ids)
        return self

    def __len__(self) -> int:
        return len(self.ids)

    def search_ids(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Batched search returning (indices, scores), each ``(n_queries, k)``."""
        queries = np.atleast_2d(np.asarray(queries, dtype=np.float32))
        n = len(self.ids)
        k = min(k, n)
        if k == 0:
            empty = np.zeros((queries.shape[0], 0))
            return empty.astype(np.int64), empty.astype(np.float32)
        scores = queries @ self.vectors.T
        if k < n:
            part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
        else:
            part = np.tile(np.arange(n), (queries.shape[0], 1))
        part_scores = np.take_along_axis(scores, part, axis=1)
        order = np.argsort(-part_scores, axis=1, kind="stable")
        idx = np.take_along_axis(part, order, axis=1)
        return idx, np.take_along_axis(part_scores, order, axis=1)

    def search(self, query: np.ndarray, k: int = 100) -> list[tuple[str, float]]:
        idx, scores = self.search_ids(query[None, :] if query.ndim == 1 else query, k)
        return [(self.ids[i], float(s)) for i, s in zip(idx[0], scores[0])]

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "flat_vectors.npy", self.vectors)
        (path / "flat_ids.txt").write_text("\n".join(self.ids), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> FlatIndex:
        vectors = np.load(path / "flat_vectors.npy")
        index = cls(vectors.shape[1])
        ids = (path / "flat_ids.txt").read_text(encoding="utf-8").split("\n") if len(vectors) else []
        return index.add(ids, vectors) if len(ids) else index
