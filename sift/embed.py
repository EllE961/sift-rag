"""Dense text encoders.

``Embedder`` is a small protocol so the pipeline, server and tests can swap a
real transformer encoder for a deterministic fake. Embeddings are always
L2-normalized float32, so inner product == cosine similarity.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from .config import CACHE_DIR, resolve_device

# Asymmetric retrieval models expect an instruction on the query side only.
QUERY_PREFIX = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
}
SHORT_NAMES = {
    "BAAI/bge-small-en-v1.5": "bge-small",
    "sentence-transformers/all-MiniLM-L6-v2": "minilm",
}


@runtime_checkable
class Embedder(Protocol):
    name: str
    dim: int

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray: ...

    def encode_docs(self, texts: Sequence[str]) -> np.ndarray: ...


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


class TransformerEmbedder:
    """Sentence-transformers bi-encoder with fp16 inference on GPU."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: str = "auto",
        batch_size: int = 128,
        max_seq_length: int | None = 512,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.name = SHORT_NAMES.get(model_name, model_name.split("/")[-1])
        self.device = resolve_device(device)
        self.model = SentenceTransformer(model_name, device=self.device)
        if max_seq_length is not None:
            self.model.max_seq_length = min(max_seq_length, self.model.max_seq_length or max_seq_length)
        if self.device.startswith("cuda"):
            self.model.half()
        self.batch_size = batch_size
        self.dim = int(self.model.get_sentence_embedding_dimension())
        self.query_prefix = QUERY_PREFIX.get(model_name, "")

    def _encode(self, texts: Sequence[str], show_progress: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        emb = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )
        return _normalize(emb)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode([self.query_prefix + t for t in texts])

    def encode_docs(self, texts: Sequence[str], show_progress: bool = False) -> np.ndarray:
        return self._encode(texts, show_progress=show_progress)


class HashingEmbedder:
    """Deterministic bag-of-words random-projection embedder.

    Not a toy: it is a legitimate (if weak) lexical embedding, which makes it a
    fast stand-in for tests and for running the service without model weights.
    """

    def __init__(self, dim: int = 256, seed: int = 0) -> None:
        self.dim = dim
        self.name = f"hashing-{dim}"
        self._seed = seed

    def _vec(self, token: str) -> np.ndarray:
        h = int.from_bytes(hashlib.blake2b(f"{self._seed}:{token}".encode(), digest_size=8).digest(), "little")
        return np.random.default_rng(h).standard_normal(self.dim).astype(np.float32)

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in re.findall(r"[a-z0-9]+", text.lower()):
                out[i] += self._vec(tok)
        return _normalize(out + 1e-6)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_docs(self, texts: Sequence[str], show_progress: bool = False) -> np.ndarray:
        return self._encode(texts)


def corpus_fingerprint(ids: Sequence[str], texts: Sequence[str]) -> str:
    h = hashlib.sha1()
    for i, t in zip(ids, texts):
        h.update(i.encode())
        h.update(b"\0")
        h.update(t.encode())
        h.update(b"\1")
    return h.hexdigest()[:16]


def cached_doc_embeddings(
    embedder: Embedder,
    ids: Sequence[str],
    texts: Sequence[str],
    cache_dir: Path = CACHE_DIR / "embeddings",
    show_progress: bool = False,
) -> np.ndarray:
    """Encode a corpus once; reuse the ``.npy`` on disk keyed by (model, corpus hash)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{embedder.name.replace('/', '_')}-{corpus_fingerprint(ids, texts)}.npy"
    if path.exists():
        emb = np.load(path)
        if emb.shape == (len(ids), embedder.dim):
            return emb
    emb = embedder.encode_docs(texts, show_progress=show_progress)  # type: ignore[call-arg]
    np.save(path, emb.astype(np.float32))
    return emb
