"""Second-stage reranking with a cross-encoder.

A bi-encoder embeds query and passage independently (cheap, indexable); a
cross-encoder reads them *together* through full self-attention, so it can
model term interactions — far more accurate, but O(N) transformer passes per
query, so it only runs over the top-N fused candidates.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from .config import resolve_device


@runtime_checkable
class Reranker(Protocol):
    name: str

    def score(self, query: str, passages: Sequence[str]) -> np.ndarray: ...


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "auto",
        batch_size: int = 64,
        max_length: int = 512,
    ) -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.name = model_name.split("/")[-1]
        self.device = resolve_device(device)
        self.model = CrossEncoder(model_name, device=self.device, max_length=max_length)
        if self.device.startswith("cuda"):
            self.model.model.half()
        self.batch_size = batch_size

    def score(self, query: str, passages: Sequence[str]) -> np.ndarray:
        if not passages:
            return np.zeros(0, dtype=np.float32)
        pairs = [(query, p) for p in passages]
        out = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False, convert_to_numpy=True)
        return np.asarray(out, dtype=np.float32)


class OverlapReranker:
    """Query-term overlap scorer: a dependency-free stand-in for tests/offline use."""

    name = "overlap"

    def score(self, query: str, passages: Sequence[str]) -> np.ndarray:
        q = set(re.findall(r"[a-z0-9]+", query.lower()))
        out = []
        for p in passages:
            toks = re.findall(r"[a-z0-9]+", p.lower())
            out.append(len(q.intersection(toks)) / (1 + len(q)) + 1e-4 / (1 + len(toks)))
        return np.asarray(out, dtype=np.float32)
