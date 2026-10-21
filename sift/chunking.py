"""Chunking strategies with exact character offsets.

Every chunker returns ``Chunk(start, end)`` such that
``chunk.text == source[chunk.start:chunk.end]`` — a hard invariant (tested)
that lets a citation inside a chunk be mapped back to the exact span of the
original document.

Sizes are measured in whitespace-delimited words ("tokens" below). That is a
model-agnostic proxy: for English, one word is ~1.3 subword tokens.

* ``FixedTokenChunker`` — sliding window with overlap. Simple, predictable,
  blind to structure.
* ``RecursiveChunker`` — split on the coarsest separator that yields
  pieces under the limit (paragraph → line → sentence → word), then pack.
* ``SentenceChunker`` — pack whole sentences, overlapping by N sentences.
* ``SemanticChunker`` — embed sentences and cut where adjacent-sentence
  similarity drops (topic shifts), bounded by a max size.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from .text import split_sentences

_WORD_RE = re.compile(r"\S+")


@dataclass
class Chunk:
    text: str
    start: int
    end: int
    index: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_tokens(self) -> int:
        return count_tokens(self.text)


def count_tokens(text: str) -> int:
    return sum(1 for _ in _WORD_RE.finditer(text))


def _word_spans(text: str, start: int = 0, end: int | None = None) -> list[tuple[int, int]]:
    end = len(text) if end is None else end
    return [(m.start(), m.end()) for m in _WORD_RE.finditer(text, start, end)]


def _make(text: str, start: int, end: int, index: int, **meta: Any) -> Chunk | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if end <= start:
        return None
    return Chunk(text=text[start:end], start=start, end=end, index=index, meta=dict(meta))


def _reindex(chunks: list[Chunk]) -> list[Chunk]:
    for i, c in enumerate(chunks):
        c.index = i
    return chunks


class Chunker(Protocol):
    name: str

    def split(self, text: str) -> list[Chunk]: ...


class FixedTokenChunker:
    name = "fixed"

    def __init__(self, chunk_size: int = 200, overlap: int = 40) -> None:
        if not 0 <= overlap < chunk_size:
            raise ValueError("need 0 <= overlap < chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, text: str) -> list[Chunk]:
        spans = _word_spans(text)
        chunks: list[Chunk] = []
        step = self.chunk_size - self.overlap
        for i in range(0, max(len(spans), 1), step):
            window = spans[i : i + self.chunk_size]
            if not window:
                break
            c = _make(text, window[0][0], window[-1][1], len(chunks))
            if c:
                chunks.append(c)
            if i + self.chunk_size >= len(spans):
                break
        return chunks


class RecursiveChunker:
    name = "recursive"

    def __init__(self, chunk_size: int = 200, separators: Sequence[str] = ("\n\n", "\n", ". ", " ")) -> None:
        self.chunk_size = chunk_size
        self.separators = tuple(separators)

    def _pieces(self, text: str, start: int, end: int, level: int) -> list[tuple[int, int]]:
        """Split [start, end) into ranges of <= chunk_size tokens (separators kept)."""
        if count_tokens(text[start:end]) <= self.chunk_size:
            return [(start, end)]
        if level >= len(self.separators):
            spans = _word_spans(text, start, end)
            out = []
            for i in range(0, len(spans), self.chunk_size):
                w = spans[i : i + self.chunk_size]
                out.append((w[0][0], w[-1][1]))
            return out
        sep = self.separators[level]
        ranges: list[tuple[int, int]] = []
        pos = start
        while pos < end:
            hit = text.find(sep, pos, end)
            cut = end if hit == -1 else hit + len(sep)
            ranges.append((pos, cut))
            pos = cut
        if len(ranges) == 1:
            return self._pieces(text, start, end, level + 1)
        out: list[tuple[int, int]] = []
        for a, b in ranges:
            out.extend(self._pieces(text, a, b, level + 1))
        return out

    def split(self, text: str) -> list[Chunk]:
        pieces = self._pieces(text, 0, len(text), 0)
        chunks: list[Chunk] = []
        cur_start, cur_end, cur_tokens = None, None, 0
        for a, b in pieces:
            n = count_tokens(text[a:b])
            if cur_start is not None and cur_tokens + n > self.chunk_size:
                c = _make(text, cur_start, cur_end, len(chunks))
                if c:
                    chunks.append(c)
                cur_start, cur_tokens = None, 0
            if cur_start is None:
                cur_start = a
            cur_end = b
            cur_tokens += n
        if cur_start is not None:
            c = _make(text, cur_start, cur_end, len(chunks))
            if c:
                chunks.append(c)
        return chunks


class SentenceChunker:
    name = "sentence"

    def __init__(self, chunk_size: int = 200, overlap_sentences: int = 1) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap_sentences
        self._fallback = FixedTokenChunker(chunk_size, 0)

    def _sentences(self, text: str) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        for a, b in split_sentences(text):
            if count_tokens(text[a:b]) <= self.chunk_size:
                out.append((a, b))
            else:  # a single monster "sentence" (tables, code): hard-split it
                out.extend((a + c.start, a + c.end) for c in self._fallback.split(text[a:b]))
        return out

    def split(self, text: str) -> list[Chunk]:
        sents = self._sentences(text)
        chunks: list[Chunk] = []
        i = 0
        while i < len(sents):
            j, tokens = i, 0
            while j < len(sents):
                n = count_tokens(text[sents[j][0] : sents[j][1]])
                if j > i and tokens + n > self.chunk_size:
                    break
                tokens += n
                j += 1
            c = _make(text, sents[i][0], sents[j - 1][1], len(chunks), sentences=j - i)
            if c:
                chunks.append(c)
            if j >= len(sents):
                break
            i = max(j - self.overlap, i + 1)
        return chunks


class SemanticChunker:
    """Cut at topic shifts: where cosine distance between adjacent sentence
    embeddings exceeds the given percentile of all adjacent distances.

    ``meta["break_score"]`` records the distance at each chunk's start boundary
    (the UI renders it as boundary strength).
    """

    name = "semantic"

    def __init__(self, embedder: Any, chunk_size: int = 250, percentile: float = 85.0, min_tokens: int = 30):
        self.embedder = embedder
        self.chunk_size = chunk_size
        self.percentile = percentile
        self.min_tokens = min_tokens
        self._sentence = SentenceChunker(chunk_size, 0)

    def split(self, text: str) -> list[Chunk]:
        sents = self._sentence._sentences(text)
        if len(sents) <= 1:
            return _reindex([c for c in [_make(text, 0, len(text), 0)] if c])
        emb = self.embedder.encode_docs([text[a:b] for a, b in sents])
        dist = 1.0 - np.sum(emb[1:] * emb[:-1], axis=1)  # distance between sentence i and i+1
        threshold = float(np.percentile(dist, self.percentile))
        chunks: list[Chunk] = []
        start_i, tokens, break_score = 0, 0, 0.0
        for i, (a, b) in enumerate(sents):
            n = count_tokens(text[a:b])
            is_last = i == len(sents) - 1
            overflow = tokens + n > self.chunk_size and i > start_i
            if overflow:
                c = _make(text, sents[start_i][0], sents[i - 1][1], len(chunks), break_score=break_score)
                if c:
                    chunks.append(c)
                start_i, tokens, break_score = i, 0, float(dist[i - 1])
            tokens += n
            if is_last:
                break
            if dist[i] > threshold and tokens >= self.min_tokens:
                c = _make(text, sents[start_i][0], b, len(chunks), break_score=break_score)
                if c:
                    chunks.append(c)
                start_i, tokens, break_score = i + 1, 0, float(dist[i])
        if start_i < len(sents):
            c = _make(text, sents[start_i][0], sents[-1][1], len(chunks), break_score=break_score)
            if c:
                chunks.append(c)
        return _reindex(chunks)


def make_chunker(name: str, chunk_size: int = 200, embedder: Any = None) -> Chunker:
    if name == "fixed":
        return FixedTokenChunker(chunk_size, max(0, chunk_size // 5))
    if name == "recursive":
        return RecursiveChunker(chunk_size)
    if name == "sentence":
        return SentenceChunker(chunk_size, 1)
    if name == "semantic":
        if embedder is None:
            raise ValueError("semantic chunking needs an embedder")
        return SemanticChunker(embedder, chunk_size)
    raise ValueError(f"unknown chunker {name!r}")


CHUNKERS = ("fixed", "recursive", "sentence", "semantic")
