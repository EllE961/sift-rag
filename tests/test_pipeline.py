import pytest

from sift.chunking import RecursiveChunker
from sift.collection import Collection
from sift.embed import HashingEmbedder
from sift.ingest import load_bytes, to_passages
from sift.pipeline import MODES, Retriever, SearchConfig
from sift.rerank import OverlapReranker
from sift.types import Document, Passage

DOCS = [
    Document("d1", "BM25 ranks documents with term frequency saturation and document length normalization.", "BM25"),
    Document("d2", "Dense retrievers embed queries and documents into one vector space.", "Dense retrieval"),
    Document("d3", "HNSW is a graph index for approximate nearest neighbor search.", "HNSW"),
    Document("d4", "Cross-encoders rerank candidates by reading query and passage jointly.", "Rerankers"),
    Document("d5", "Reciprocal rank fusion merges ranked lists without score calibration.", "RRF"),
] + [Document(f"x{i}", f"Unrelated filler document number {i} about cooking pasta and gardening.", f"Filler {i}")
     for i in range(80)]  # fmt: skip


@pytest.fixture(scope="module")
def retriever():
    emb = HashingEmbedder(64)
    col = Collection("t", [Passage(id=d.id, doc_id=d.id, text=d.text, title=d.title, end=len(d.text)) for d in DOCS],
                     {d.id: d for d in DOCS})  # fmt: skip
    col.build_lexical().build_dense(emb, ann="hnsw")
    return Retriever(col, emb, OverlapReranker())


@pytest.mark.parametrize("mode", MODES)
def test_every_mode_returns_ranked_results_and_a_trace(retriever, mode):
    res = retriever.search("graph index for nearest neighbor search", SearchConfig(mode=mode, k=3, depth=20))
    assert [h.rank for h in res.hits] == list(range(1, len(res.hits) + 1))
    # BM25 alone only returns documents sharing a term with the query
    assert len(res.hits) == 3 or mode in ("bm25", "bm25_rerank")
    assert res.hits[0].pid == "d3"
    names = [s.name for s in res.stages]
    assert ("bm25" in names) == (mode in ("bm25", "hybrid", "hybrid_rerank", "bm25_rerank"))
    assert ("fusion" in names) == mode.startswith("hybrid")
    assert ("rerank" in names) == mode.endswith("_rerank")
    assert res.total_ms >= sum(s.latency_ms for s in res.stages) * 0.5


def test_trace_dict_is_ui_ready(retriever):
    res = retriever.search("rank fusion of ranked lists", SearchConfig(mode="hybrid_rerank", k=5, depth=30, rerank_depth=10))
    d = res.to_dict(retriever.collection)
    assert d["stage_order"] == ["bm25", "dense", "fusion", "rerank"]
    assert len(d["hits"]) == 5 and d["hits"][0]["passage"]["id"] == "d5"
    assert all("stage_ranks" in h for h in d["hits"])
    flow_ids = {r["pid"] for r in d["rank_flow"]}
    assert {h["pid"] for h in d["hits"]} <= flow_ids
    assert all(s["n"] >= len(s["hits"]) for s in d["stages"])


def test_convex_fusion_alpha_extremes(retriever):
    q = "vector space embeddings"
    lex = retriever.search(q, SearchConfig(mode="hybrid", fusion="convex", alpha=0.0, k=5)).ranking()
    dense = retriever.search(q, SearchConfig(mode="hybrid", fusion="convex", alpha=1.0, k=5)).ranking()
    lexical = retriever.search(q, SearchConfig(mode="bm25", k=3)).ranking()
    assert lex[: len(lexical)] == lexical
    assert dense[0] == retriever.search(q, SearchConfig(mode="dense", k=1)).ranking()[0]


def test_unknown_mode_and_missing_components():
    col = Collection("empty", [Passage("a", "a", "alpha beta", end=10)])
    col.build_lexical()
    r = Retriever(col)
    assert r.available_modes() == ["bm25"]
    with pytest.raises(ValueError):
        r.search("x", SearchConfig(mode="nope"))
    with pytest.raises(RuntimeError):
        r.search("x", SearchConfig(mode="dense"))
    with pytest.raises(RuntimeError):
        r.search("x", SearchConfig(mode="bm25_rerank"))


def test_collection_roundtrip_and_incremental_add(tmp_path, retriever):
    retriever.collection.save(tmp_path)
    loaded = Collection.load("t", tmp_path)
    assert len(loaded) == len(retriever.collection)
    r2 = Retriever(loaded, retriever.embedder, retriever.reranker)
    q = "cross encoders rerank candidates"
    assert r2.search(q, SearchConfig(mode="hybrid")).ranking() == retriever.search(q, SearchConfig(mode="hybrid")).ranking()

    ws = Collection("ws", [])
    doc = load_bytes("notes.md", b"# Sieve notes\n\nA sieve separates fine grains from coarse ones.\n\nMesh size matters.")
    added = ws.add([doc], to_passages([doc], RecursiveChunker(8)), HashingEmbedder(64))
    assert added >= 2 and ws.bm25 is not None
    hits = Retriever(ws, HashingEmbedder(64)).search("mesh size", SearchConfig(mode="hybrid", k=1)).hits
    p = ws.passage(hits[0].pid)
    assert doc.text[p.start : p.end] == p.text  # passages keep exact source offsets


def test_blended_rerank_interpolates_between_first_stage_and_cross_encoder(retriever):
    q = "rank fusion of ranked lists"
    base = SearchConfig(mode="hybrid_rerank", k=10, depth=30, rerank_depth=10)
    fused = retriever.search(q, base.with_(mode="hybrid")).ranking()
    pure_ce = retriever.search(q, base).ranking()
    assert retriever.search(q, base.with_(blend=0.0)).ranking() == fused  # w=0: first stage order kept
    assert retriever.search(q, base.with_(blend=1.0)).ranking() == pure_ce  # w=1: cross-encoder order
    res = retriever.search(q, base.with_(blend=0.5))
    rerank = next(s for s in res.stages if s.name == "rerank")
    assert rerank.meta["blend"] == 0.5 and all("ce" in h.sources for h in rerank.hits)
