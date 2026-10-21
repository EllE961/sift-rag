# Scaling Sift to 100M documents

The current build indexes a few thousand documents in memory. This is how each piece would change at 100M documents.

| Concern | At 5k docs (this repo) | At 100M docs |
|---|---|---|
| Lexical index | in-memory CSR matrix | sharded inverted index (Lucene/Elasticsearch/OpenSearch, or Tantivy), BM25 with block-max WAND for top-k early termination |
| Vectors | 384-d float32, 7.5 MB | 100M × 384 × 4 B = 154 GB raw → product quantization (IVF-PQ / ScaNN) or int8 + HNSW on disk (DiskANN); shard by doc id, fan out, merge top-k |
| ANN build | 2 min single-thread numba | parallel/incremental builds; segment-based indexes merged in background (like Lucene segments) |
| Embedding | one GPU, minutes | batch inference fleet, ~10–20k docs/s per A10-class GPU for a small encoder; re-embed only changed docs |
| Freshness | rebuild | write path: new docs go to a small mutable segment searched alongside the large immutable one |
| Rerank | GPU, 50 candidates | dedicated GPU pool, smaller/distilled reranker, cap depth by latency SLO, cache by (query, doc) |
| Fusion | in-process | in the query coordinator after the shard merge; RRF needs ranks only, so it's shard-friendly |
| Evaluation | BEIR qrels | offline judged sets per domain, plus online signals (click-through, citation clicks, thumbs) and LLM-judge on a sample |
| Cost | — | the reranker and LLM dominate; cache answers for repeated queries, use prompt caching for a stable system prompt, route easy queries to a smaller model |

The architecture doesn't change, only the implementations behind each interface (`Embedder`,
`Reranker`, `Generator`, and the dense index's `search_ids`).
