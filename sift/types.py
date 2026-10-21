"""Core value types shared across the retrieval and generation stack."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Document:
    """A source document (an abstract, a file, a web page)."""

    id: str
    text: str
    title: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        """Title and body joined — the text that gets indexed."""
        return f"{self.title}\n{self.text}" if self.title else self.text


@dataclass
class Passage:
    """The unit of retrieval: a document or a chunk of one.

    ``start``/``end`` are character offsets into the parent document's ``text``
    so that a citation inside the passage can be mapped back to an exact span
    of the original source.
    """

    id: str
    doc_id: str
    text: str
    title: str = ""
    start: int = 0
    end: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def index_text(self) -> str:
        return f"{self.title}\n{self.text}" if self.title else self.text

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Hit:
    """A scored passage returned by a retriever, fuser or reranker."""

    pid: str
    score: float
    rank: int = 0
    sources: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"pid": self.pid, "score": float(self.score), "rank": self.rank, "sources": self.sources}


def ranked(scores: dict[str, float], k: int | None = None) -> list[Hit]:
    """Turn a ``{pid: score}`` map into a rank-ordered list of hits (ties by id)."""
    items = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    if k is not None:
        items = items[:k]
    return [Hit(pid=pid, score=float(score), rank=i + 1) for i, (pid, score) in enumerate(items)]
