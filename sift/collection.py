"""A searchable collection: passages + lexical index + dense ANN index(es).

Collections persist to ``data/indexes/<name>/`` and load lazily, so the
service warm-starts in about a second instead of re-embedding the corpus.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .config import INDEX_DIR
from .datasets import load_corpus
from .embed import Embedder, cached_doc_embeddings
from .index.bm25 import BM25Index
from .index.flat import FlatIndex
from .index.hnsw import HNSWIndex
from .types import Document, Passage

DenseIndex = FlatIndex | HNSWIndex


class Collection:
    def __init__(self, name: str, passages: Sequence[Passage], docs: dict[str, Document] | None = None) -> None:
        self.name = name
        self.passages: dict[str, Passage] = {p.id: p for p in passages}
        self.docs: dict[str, Document] = docs or {}
        self.bm25: BM25Index | None = None
        self.dense: dict[str, DenseIndex] = {}
        self.meta: dict[str, Any] = {"name": name}

    # ------------------------------------------------------------- building
    @property
    def ids(self) -> list[str]:
        return list(self.passages)

    def __len__(self) -> int:
        return len(self.passages)

    def build_lexical(self, k1: float = 0.9, b: float = 0.4) -> Collection:
        ids = self.ids
        if not ids:
            self.bm25 = None
            return self
        self.bm25 = BM25Index(k1=k1, b=b).build(ids, [self.passages[i].index_text for i in ids])
        return self

    def build_dense(self, embedder: Embedder, ann: str = "hnsw", m: int = 16, ef_construction: int = 200,
                    show_progress: bool = False) -> Collection:  # fmt: skip
        ids = self.ids
        if not ids:
            return self
        texts = [self.passages[i].index_text for i in ids]
        vectors = cached_doc_embeddings(embedder, ids, texts, show_progress=show_progress)
        if ann == "hnsw" and len(ids) >= 64:
            index: DenseIndex = HNSWIndex(embedder.dim, m=m, ef_construction=ef_construction).build(ids, vectors)
        else:
            index = FlatIndex(embedder.dim).add(ids, vectors)
        self.dense[embedder.name] = index
        return self

    def add(self, docs: Iterable[Document], passages: Sequence[Passage], embedder: Embedder | None) -> int:
        """Incremental add (workspace collections): passages are appended, the
        BM25 index is rebuilt (milliseconds at workspace scale) and the dense
        vectors are appended to a flat index."""
        for d in docs:
            self.docs[d.id] = d
        new = [p for p in passages if p.id not in self.passages]
        for p in new:
            self.passages[p.id] = p
        self.build_lexical()
        if embedder is not None and new:
            vecs = embedder.encode_docs([p.index_text for p in new])
            index = self.dense.get(embedder.name)
            if not isinstance(index, FlatIndex):
                index = FlatIndex(embedder.dim)
                existing = [pid for pid in self.passages if pid not in {p.id for p in new}]
                if existing:
                    index.add(existing, embedder.encode_docs([self.passages[i].index_text for i in existing]))
                self.dense[embedder.name] = index
            index.add([p.id for p in new], vecs)
        return len(new)

    # --------------------------------------------------------------- lookup
    def passage(self, pid: str) -> Passage:
        return self.passages[pid]

    def source_text(self, doc_id: str) -> str | None:
        doc = self.docs.get(doc_id)
        return doc.text if doc else None

    # ---------------------------------------------------------- persistence
    def save(self, root: Path = INDEX_DIR) -> Path:
        path = root / self.name
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "passages.jsonl", "w", encoding="utf-8") as fh:
            for p in self.passages.values():
                fh.write(json.dumps(p.to_dict()) + "\n")
        with open(path / "docs.jsonl", "w", encoding="utf-8") as fh:
            for d in self.docs.values():
                fh.write(json.dumps({"id": d.id, "title": d.title, "text": d.text, "metadata": d.metadata}) + "\n")
        if self.bm25 is not None:
            self.bm25.save(path / "bm25")
        for name, index in self.dense.items():
            sub = path / "dense" / name.replace("/", "_")
            index.save(sub)
            (sub / "kind.txt").write_text("hnsw" if isinstance(index, HNSWIndex) else "flat")
        self.meta.update({"n_passages": len(self.passages), "n_docs": len(self.docs), "dense": list(self.dense),
                          "saved_at": time.time()})  # fmt: skip
        (path / "meta.json").write_text(json.dumps(self.meta, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, name: str, root: Path = INDEX_DIR) -> Collection:
        path = root / name
        passages = []
        with open(path / "passages.jsonl", encoding="utf-8") as fh:
            passages = [Passage(**json.loads(line)) for line in fh if line.strip()]
        docs: dict[str, Document] = {}
        if (path / "docs.jsonl").exists():
            with open(path / "docs.jsonl", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        row = json.loads(line)
                        docs[row["id"]] = Document(**row)
        col = cls(name, passages, docs)
        if (path / "meta.json").exists():
            col.meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        if (path / "bm25").exists():
            col.bm25 = BM25Index.load(path / "bm25")
        dense_root = path / "dense"
        if dense_root.exists():
            for sub in sorted(dense_root.iterdir()):
                kind = (sub / "kind.txt").read_text().strip()
                col.dense[sub.name] = HNSWIndex.load(sub) if kind == "hnsw" else FlatIndex.load(sub)
        return col

    @staticmethod
    def exists(name: str, root: Path = INDEX_DIR) -> bool:
        return (root / name / "passages.jsonl").exists()


def beir_collection(name: str, embedders: Sequence[Embedder] = (), ann: str = "hnsw") -> Collection:
    """Doc-level collection over a BEIR corpus (abstracts are short: no chunking)."""
    corpus = load_corpus(name)
    passages = [Passage(id=d.id, doc_id=d.id, text=d.text, title=d.title, start=0, end=len(d.text))
                for d in corpus.values()]  # fmt: skip
    col = Collection(name, passages, docs=corpus)
    col.meta.update({"kind": "beir", "dataset": name})
    col.build_lexical()
    for emb in embedders:
        col.build_dense(emb, ann=ann)
    return col


def unit_vectors(n: int, dim: int, seed: int = 0) -> np.ndarray:
    """Random unit vectors (test helper)."""
    x = np.random.default_rng(seed).standard_normal((n, dim)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def load_or_build_beir(name: str, embedder: Embedder, root: Path = INDEX_DIR, ann: str = "hnsw") -> Collection:
    """Load a persisted BEIR collection, building (and saving) it on first use."""
    if Collection.exists(name, root):
        col = Collection.load(name, root)
        if embedder.name in col.dense:
            return col
        col.build_dense(embedder, ann=ann)
        col.save(root)
        return col
    col = beir_collection(name, [embedder], ann=ann)
    col.save(root)
    return col
