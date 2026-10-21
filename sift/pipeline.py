"""The retrieval pipeline: BM25 ∥ dense → fusion → cross-encoder rerank.

Every stage is timed and its candidate list recorded, producing a ``trace``
that the UI renders as a latency waterfall and a rank-flow chart. Tracing is
cheap (lists of ids and floats), so it is always on.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np

from .collection import Collection
from .embed import Embedder
from .fusion import convex, normalize_scores, rrf
from .rerank import Reranker
from .types import Hit

MODES = ("bm25", "dense", "hybrid", "hybrid_rerank", "bm25_rerank")


@dataclass(frozen=True)
class SearchConfig:
    mode: str = "hybrid_rerank"
    fusion: str = "rrf"  # "rrf" | "convex"
    alpha: float = 0.5  # dense weight for convex fusion
    rrf_k: int = 60
    depth: int = 100  # candidates per first-stage retriever
    rerank_depth: int = 50
    k: int = 10
    ef: int = 128  # HNSW beam width at query time
    # Weight on the (min-max normalized) cross-encoder score when reranking;
    # the rest goes to the first-stage score. None = pure cross-encoder order.
    # Tuned per collection on train queries (see `python -m sift.eval blend`).
    blend: float | None = None

    def with_(self, **overrides: Any) -> SearchConfig:
        clean = {k: v for k, v in overrides.items() if v is not None}
        return replace(self, **clean)


@dataclass
class Stage:
    name: str
    latency_ms: float
    hits: list[Hit]
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    query: str
    config: SearchConfig
    hits: list[Hit]
    stages: list[Stage]
    total_ms: float

    def ranking(self) -> list[str]:
        return [h.pid for h in self.hits]

    def to_dict(self, collection: Collection, top_stage: int = 50, flow_k: int = 10) -> dict[str, Any]:
        """JSON trace for the UI: per-stage candidates, timings and rank flow."""
        rank_stages = [s for s in self.stages if s.name != "embed"]
        ranks: dict[str, dict[str, int]] = {}
        for s in rank_stages:
            for h in s.hits:
                ranks.setdefault(h.pid, {})[s.name] = h.rank
        flow_ids: list[str] = []
        for s in rank_stages:
            for h in s.hits[:flow_k]:
                if h.pid not in flow_ids:
                    flow_ids.append(h.pid)

        def passage_json(pid: str) -> dict[str, Any]:
            p = collection.passage(pid)
            return {"id": p.id, "doc_id": p.doc_id, "title": p.title, "text": p.text, "start": p.start, "end": p.end}

        return {
            "query": self.query,
            "config": asdict(self.config),
            "total_ms": round(self.total_ms, 3),
            "stages": [
                {
                    "name": s.name,
                    "n": len(s.hits),
                    "latency_ms": round(s.latency_ms, 3),
                    "meta": s.meta,
                    "hits": [h.to_dict() for h in s.hits[:top_stage]],
                }
                for s in self.stages
            ],
            "hits": [
                {**h.to_dict(), "passage": passage_json(h.pid), "stage_ranks": ranks.get(h.pid, {})}
                for h in self.hits
            ],
            "rank_flow": [
                {"pid": pid, "title": collection.passage(pid).title[:90], "ranks": ranks.get(pid, {})}
                for pid in flow_ids
            ],
            "stage_order": [s.name for s in rank_stages],
        }


class Retriever:
    def __init__(
        self,
        collection: Collection,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        config: SearchConfig | None = None,
    ) -> None:
        self.collection = collection
        self.embedder = embedder
        self.reranker = reranker
        self.config = config or SearchConfig()

    def _dense_index(self):
        if self.embedder is None:
            return None
        return self.collection.dense.get(self.embedder.name)

    def available_modes(self) -> list[str]:
        modes = ["bm25"] if self.collection.bm25 is not None else []
        if self._dense_index() is not None:
            modes.append("dense")
            if modes[0] == "bm25":
                modes.append("hybrid")
        if self.reranker is not None:
            modes += [m + "_rerank" for m in ("bm25", "hybrid") if m in modes]
        return modes

    def search(self, query: str, config: SearchConfig | None = None) -> SearchResult:
        cfg = config or self.config
        if cfg.mode not in MODES:
            raise ValueError(f"unknown mode {cfg.mode!r}; expected one of {MODES}")
        t_start = time.perf_counter()
        stages: list[Stage] = []
        use_bm25 = cfg.mode in ("bm25", "hybrid", "hybrid_rerank", "bm25_rerank")
        use_dense = cfg.mode in ("dense", "hybrid", "hybrid_rerank")
        runs: dict[str, list[tuple[str, float]]] = {}

        if use_bm25:
            if self.collection.bm25 is None:
                raise RuntimeError(f"collection {self.collection.name!r} has no lexical index")
            t = time.perf_counter()
            runs["bm25"] = self.collection.bm25.search(query, cfg.depth)
            stages.append(Stage("bm25", _ms(t), _hits(runs["bm25"]), {"k1": self.collection.bm25.k1,
                                                                      "b": self.collection.bm25.b}))  # fmt: skip
        if use_dense:
            index = self._dense_index()
            if index is None or self.embedder is None:
                raise RuntimeError(f"collection {self.collection.name!r} has no dense index for this embedder")
            t = time.perf_counter()
            qv = self.embedder.encode_queries([query])[0]
            stages.append(Stage("embed", _ms(t), [], {"model": self.embedder.name, "dim": self.embedder.dim}))
            t = time.perf_counter()
            if hasattr(index, "ef_search"):
                runs["dense"] = index.search(qv, cfg.depth, ef=max(cfg.ef, cfg.depth))
                evals = index.last_search_distance_evals
                meta = {"index": "hnsw", "ef": max(cfg.ef, cfg.depth), "distance_evals": evals}
            else:
                runs["dense"] = index.search(qv, cfg.depth)
                meta = {"index": "flat"}
            stages.append(Stage("dense", _ms(t), _hits(runs["dense"]), meta))

        if use_bm25 and use_dense:
            t = time.perf_counter()
            if cfg.fusion == "rrf":
                fused = rrf(runs, k=cfg.rrf_k, limit=cfg.depth)
                meta = {"method": "rrf", "k": cfg.rrf_k}
            else:
                fused = convex(runs, {"dense": cfg.alpha, "bm25": 1.0 - cfg.alpha}, limit=cfg.depth)
                meta = {"method": "convex", "alpha": cfg.alpha}
            stages.append(Stage("fusion", _ms(t), fused, meta))
            current = fused
        else:
            current = stages[-1].hits

        if cfg.mode.endswith("_rerank"):
            if self.reranker is None:
                raise RuntimeError("no reranker configured")
            t = time.perf_counter()
            pool = current[: cfg.rerank_depth]
            texts = [self.collection.passage(h.pid).index_text for h in pool]
            ce = np.asarray(self.reranker.score(query, texts), dtype=np.float64)
            if cfg.blend is not None and len(pool) > 1:
                first = np.array([h.score for h in pool])
                final = cfg.blend * normalize_scores(ce) + (1 - cfg.blend) * normalize_scores(first)
            else:
                final = ce
            order = sorted(range(len(pool)), key=lambda i: (-float(final[i]), pool[i].rank))
            reranked = [
                Hit(pid=pool[i].pid, score=float(final[i]), rank=r + 1,
                    sources={"prev_rank": pool[i].rank, "ce": round(float(ce[i]), 4)})
                for r, i in enumerate(order)
            ]  # fmt: skip
            meta = {"model": self.reranker.name, "depth": len(pool), "blend": cfg.blend}
            stages.append(Stage("rerank", _ms(t), reranked, meta))
            current = reranked

        final = [Hit(pid=h.pid, score=h.score, rank=i + 1, sources=h.sources) for i, h in enumerate(current[: cfg.k])]
        return SearchResult(query, cfg, final, stages, _ms(t_start))


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


def _hits(cands: list[tuple[str, float]]) -> list[Hit]:
    return [Hit(pid=pid, score=s, rank=i + 1) for i, (pid, s) in enumerate(cands)]
