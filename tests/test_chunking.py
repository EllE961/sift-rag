import pytest

from sift.chunking import (
    FixedTokenChunker,
    RecursiveChunker,
    SemanticChunker,
    SentenceChunker,
    count_tokens,
    make_chunker,
)
from sift.embed import HashingEmbedder

TEXT = (
    "# Retrieval-augmented generation\n\n"
    "Retrieval-augmented generation (RAG) grounds a language model in external documents. "
    "A retriever selects passages; the generator conditions on them. This reduces hallucination.\n\n"
    "## Lexical retrieval\n\n"
    "BM25 scores documents by term frequency with saturation and length normalization. "
    "It is fast, robust and needs no training data. It fails on vocabulary mismatch.\n\n"
    "## Dense retrieval\n\n"
    "Bi-encoders embed queries and passages into a shared vector space. Similar meanings land close together. "
    "Approximate nearest-neighbor indexes such as HNSW make search sublinear.\n"
    "Cats are unrelated to any of this, but they are nice. Dogs are also nice animals.\n"
) * 3


def chunkers():
    return [
        FixedTokenChunker(40, 10),
        RecursiveChunker(40),
        SentenceChunker(40, 1),
        SemanticChunker(HashingEmbedder(64), chunk_size=60, percentile=70, min_tokens=5),
    ]


@pytest.mark.parametrize("chunker", chunkers(), ids=lambda c: c.name)
def test_offsets_reconstruct_source_exactly(chunker):
    chunks = chunker.split(TEXT)
    assert chunks
    for i, c in enumerate(chunks):
        assert TEXT[c.start : c.end] == c.text
        assert c.text == c.text.strip()
        assert c.index == i


@pytest.mark.parametrize("chunker", chunkers(), ids=lambda c: c.name)
def test_every_word_is_covered(chunker):
    chunks = chunker.split(TEXT)
    covered = [False] * len(TEXT)
    for c in chunks:
        for j in range(c.start, c.end):
            covered[j] = True
    missing = [ch for ch, cov in zip(TEXT, covered) if not cov and not ch.isspace()]
    assert missing == []


def test_size_limits():
    for c in FixedTokenChunker(40, 10).split(TEXT):
        assert count_tokens(c.text) <= 40
    for c in RecursiveChunker(40).split(TEXT):
        assert count_tokens(c.text) <= 40
    for c in SemanticChunker(HashingEmbedder(64), chunk_size=60).split(TEXT):
        assert count_tokens(c.text) <= 60


def test_fixed_overlap():
    words = " ".join(f"w{i}" for i in range(100))
    chunks = FixedTokenChunker(30, 10).split(words)
    assert chunks[0].text.split()[-10:] == chunks[1].text.split()[:10]
    assert chunks[-1].text.endswith("w99")


def test_sentence_chunker_overlap_and_long_sentence():
    text = "Alpha beta. Gamma delta. Epsilon zeta. Eta theta."
    chunks = SentenceChunker(4, overlap_sentences=1).split(text)
    assert [c.text for c in chunks] == ["Alpha beta. Gamma delta.", "Gamma delta. Epsilon zeta.", "Epsilon zeta. Eta theta."]
    long = " ".join(["word"] * 25) + "."
    assert all(count_tokens(c.text) <= 10 for c in SentenceChunker(10, 0).split(long))


def test_recursive_prefers_paragraph_boundaries():
    chunks = RecursiveChunker(60).split(TEXT)
    # With a generous budget, chunks should start at headings/paragraph starts, never mid-word.
    for c in chunks:
        assert c.start == 0 or TEXT[c.start - 1].isspace()


def test_semantic_records_break_scores():
    chunks = SemanticChunker(HashingEmbedder(64), chunk_size=60, percentile=70, min_tokens=5).split(TEXT)
    assert len(chunks) > 1
    assert all("break_score" in c.meta for c in chunks)


def test_factory_and_empty():
    assert make_chunker("recursive", 50).name == "recursive"
    with pytest.raises(ValueError):
        make_chunker("semantic")
    with pytest.raises(ValueError):
        make_chunker("bogus")
    with pytest.raises(ValueError):
        FixedTokenChunker(10, 10)
    for c in chunkers():
        assert c.split("") == []
