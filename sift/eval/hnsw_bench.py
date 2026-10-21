"""HNSW vs exact search: recall@10 / QPS Pareto curves on real embeddings.

    python -m sift.eval hnsw

Corpus: every *sentence* of SciFact + NFCorpus (~80k bge-small vectors) — real
text, large enough for sub-linear search to matter. Queries: 1,000 real BEIR
queries (bge query prefix), i.e. genuinely out-of-corpus.

Reported per (N, M, ef): recall@10 against brute force, single-thread QPS,
distance evaluations per query; plus build time. Brute force is measured both
single-threaded (fair comparison) and with multi-threaded BLAS.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..config import CACHE_DIR, RESULTS_DIR
from ..datasets import load_beir
from ..embed import TransformerEmbedder, cached_doc_embeddings
from ..index.flat import FlatIndex
from ..index.hnsw import HNSWIndex
from ..text import split_sentences

EFS = (10, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384)


def sentence_corpus() -> tuple[list[str], list[str]]:
    ids, texts = [], []
    for name in ("scifact", "nfcorpus"):
        ds = load_beir(name, "test")
        for doc in ds.corpus.values():
            for i, (a, b) in enumerate(split_sentences(doc.text)):
                s = doc.text[a:b]
                if len(s.split()) >= 4:
                    ids.append(f"{name}:{doc.id}:{i}")
                    texts.append(s)
    return ids, texts


def recall_at(approx: np.ndarray, exact: np.ndarray) -> float:
    k = exact.shape[1]
    return float(np.mean([len(set(a) & set(e)) / k for a, e in zip(approx, exact)]))


def bench_flat(flat: FlatIndex, q: np.ndarray, single_thread: bool) -> float:
    from threadpoolctl import threadpool_limits

    ctx = threadpool_limits(limits=1) if single_thread else threadpool_limits(limits=None)
    with ctx:
        for i in range(10):
            flat.search_ids(q[i : i + 1], 10)
        t = time.perf_counter()
        for i in range(len(q)):
            flat.search_ids(q[i : i + 1], 10)
        return len(q) / (time.perf_counter() - t)


def sweep(index: HNSWIndex, q: np.ndarray, exact: np.ndarray) -> list[dict[str, float]]:
    rows = []
    index.search_ids(q[:20], 10, ef=EFS[0])  # JIT warm-up
    for ef in EFS:
        t = time.perf_counter()
        approx, _ = index.search_ids(q, 10, ef=ef)
        dt = time.perf_counter() - t
        rows.append({
            "ef": ef,
            "recall@10": round(recall_at(approx, exact), 4),
            "qps": round(len(q) / dt, 1),
            "latency_us": round(dt / len(q) * 1e6, 1),
            "distance_evals_per_query": round(index.last_search_distance_evals / len(q), 1),
        })
    return rows


def main(out: Path = RESULTS_DIR / "hnsw.json", n_queries: int = 1000) -> dict[str, Any]:
    emb = TransformerEmbedder("BAAI/bge-small-en-v1.5", batch_size=128, max_seq_length=128)
    ids, texts = sentence_corpus()
    print(f"sentence corpus: {len(ids)}")
    vectors = cached_doc_embeddings(emb, ids, texts, cache_dir=CACHE_DIR / "embeddings")
    queries = []
    for name in ("scifact", "nfcorpus"):
        queries.extend(load_beir(name, "test").queries.values())
    rng = np.random.default_rng(0)
    q_text = [queries[i] for i in rng.permutation(len(queries))[:n_queries]]
    q = emb.encode_queries(q_text)

    perm = rng.permutation(len(ids))
    vectors, ids = vectors[perm], [ids[i] for i in perm]
    results: dict[str, Any] = {"n_total": len(ids), "dim": int(vectors.shape[1]), "n_queries": len(q),
                               "scaling": [], "m_sweep": []}  # fmt: skip
    sizes = [n for n in (5_000, 10_000, 20_000, 40_000) if n < len(ids)] + [len(ids)]
    for n in sizes:
        flat = FlatIndex(vectors.shape[1]).add(ids[:n], vectors[:n])
        exact, _ = flat.search_ids(q, 10)
        t = time.perf_counter()
        index = HNSWIndex(vectors.shape[1], m=16, ef_construction=200).build(ids[:n], vectors[:n])
        build_s = time.perf_counter() - t
        row = {
            "n": n, "M": 16, "ef_construction": 200, "build_s": round(build_s, 2),
            "flat_qps_1thread": round(bench_flat(flat, q, True), 1),
            "flat_qps_blas": round(bench_flat(flat, q, False), 1),
            "levels": index.stats()["nodes_per_level"],
            "curve": sweep(index, q, exact),
        }  # fmt: skip
        results["scaling"].append(row)
        best = next((r for r in row["curve"] if r["recall@10"] >= 0.95), row["curve"][-1])
        print(f"N={n:>6d} build={build_s:6.1f}s flat1t={row['flat_qps_1thread']:8.0f}qps "
              f"hnsw@R≥.95: ef={best['ef']} qps={best['qps']:.0f} "
              f"evals={best['distance_evals_per_query']:.0f}")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    n = len(ids)
    flat = FlatIndex(vectors.shape[1]).add(ids, vectors)
    exact, _ = flat.search_ids(q, 10)
    for m in (8, 32):
        t = time.perf_counter()
        index = HNSWIndex(vectors.shape[1], m=m, ef_construction=200).build(ids, vectors)
        build_s = time.perf_counter() - t
        results["m_sweep"].append({"n": n, "M": m, "build_s": round(build_s, 2), "curve": sweep(index, q, exact),
                                   "memory_mb": round(index.stats()["memory_mb"], 1)})  # fmt: skip
        print(f"M={m} build={build_s:.1f}s")
    results["m_sweep"].insert(1, {"n": n, "M": 16, "build_s": results["scaling"][-1]["build_s"],
                                  "curve": results["scaling"][-1]["curve"]})  # fmt: skip
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return results
