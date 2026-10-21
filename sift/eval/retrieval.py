"""Retrieval ablation on BEIR test splits.

    python -m sift.eval retrieval --datasets scifact nfcorpus

Methodology
-----------
* First-stage candidate lists (depth 100) are computed once per retriever.
  Dense retrieval in the quality ablation uses *exact* search so ANN error does
  not confound the comparison; the HNSW row shows the production index loses
  essentially nothing.
* Convex-fusion weights (and the normalization) are tuned on the **train**
  qrels and frozen before touching **test**. RRF has no tuned parameters.
* Cross-encoder scores are cached per (query, passage) pair, so the rerank
  variants share work.
* Latency is measured end-to-end through ``Retriever.search`` one query at a
  time (the serving path), excluding a warm-up.
"""

from __future__ import annotations

import json
import platform
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..collection import Collection
from ..config import RESULTS_DIR
from ..datasets import BeirDataset, load_beir
from ..embed import TransformerEmbedder, cached_doc_embeddings
from ..fusion import convex, rrf
from ..index.bm25 import BM25Index
from ..index.flat import FlatIndex
from ..index.hnsw import HNSWIndex
from ..pipeline import Retriever, SearchConfig
from ..rerank import CrossEncoderReranker
from ..types import Passage
from .metrics import evaluate

Cands = dict[str, list[tuple[str, float]]]
KS = (1, 5, 10, 100)
DEPTH = 100
TRAIN_SPLIT = {"scifact": "train", "nfcorpus": "train", "fiqa": "train"}
BLEND_LABEL = "Hybrid RRF → rerank, blended (train-tuned)"


def _first_stage_bm25(bm25: BM25Index, queries: Mapping[str, str], depth: int = DEPTH) -> Cands:
    return {qid: bm25.search(text, depth) for qid, text in queries.items()}


def _first_stage_dense(index: FlatIndex | HNSWIndex, embedder: TransformerEmbedder, queries: Mapping[str, str],
                       depth: int = DEPTH, ef: int | None = None) -> Cands:  # fmt: skip
    qids = list(queries)
    q = embedder.encode_queries([queries[x] for x in qids])
    if isinstance(index, HNSWIndex):
        idx, sims = index.search_ids(q, depth, ef=ef)
    else:
        idx, sims = index.search_ids(q, depth)
    return {qid: [(index.ids[j], float(s)) for j, s in zip(idx[n], sims[n]) if j >= 0] for n, qid in enumerate(qids)}


def _to_run(cands: Cands) -> dict[str, list[str]]:
    return {qid: [pid for pid, _ in c] for qid, c in cands.items()}


def _fuse(per_retriever: Mapping[str, Cands], fn: Callable[..., Any], **kw: Any) -> Cands:
    qids = next(iter(per_retriever.values())).keys()
    out: Cands = {}
    for qid in qids:
        hits = fn({name: c.get(qid, []) for name, c in per_retriever.items()}, limit=DEPTH, **kw)
        out[qid] = [(h.pid, h.score) for h in hits]
    return out


class RerankCache:
    def __init__(self, reranker: CrossEncoderReranker, passages: Mapping[str, Passage]) -> None:
        self.reranker = reranker
        self.passages = passages
        self.scores: dict[tuple[str, str], float] = {}

    def rerank(self, queries: Mapping[str, str], cands: Cands, depth: int) -> Cands:
        out: Cands = {}
        for qid, c in cands.items():
            pool = [pid for pid, _ in c[:depth]]
            missing = [pid for pid in pool if (qid, pid) not in self.scores]
            if missing:
                s = self.reranker.score(queries[qid], [self.passages[p].index_text for p in missing])
                for pid, v in zip(missing, s):
                    self.scores[(qid, pid)] = float(v)
            # Stable: ties keep first-stage order. Docs past `depth` keep their tail order.
            head = sorted(pool, key=lambda p: -self.scores[(qid, p)])
            out[qid] = [(p, self.scores[(qid, p)]) for p in head] + [(p, -1e9) for p, _ in c[depth:]]
        return out


def _metrics(cands: Cands, qrels: Mapping[str, Mapping[str, int]]) -> dict[str, float]:
    m = evaluate(_to_run(cands), qrels, ks=KS)
    return {k: round(v, 5) for k, v in m.items()}


def tune_convex(train_runs: Mapping[str, Cands], qrels: Mapping[str, Mapping[str, int]]) -> dict[str, Any]:
    sweep = []
    for norm in ("minmax", "zscore"):
        for alpha in np.round(np.linspace(0, 1, 21), 3):
            fused = _fuse(train_runs, convex, weights={"dense": float(alpha), "bm25": 1 - float(alpha)}, norm=norm)
            ndcg = evaluate(_to_run(fused), qrels, ks=(10,), metrics=("ndcg",))["ndcg@10"]
            sweep.append({"norm": norm, "alpha": float(alpha), "ndcg@10": round(ndcg, 5)})
    best = max(sweep, key=lambda r: r["ndcg@10"])
    return {"best": best, "sweep": sweep}


def tune_rrf_k(train_runs: Mapping[str, Cands], qrels: Mapping[str, Mapping[str, int]]) -> list[dict[str, float]]:
    out = []
    for k in (1, 5, 10, 20, 40, 60, 100, 200):
        fused = _fuse(train_runs, rrf, k=k)
        ndcg = evaluate(_to_run(fused), qrels, ks=(10,), metrics=("ndcg",))["ndcg@10"]
        out.append({"k": k, "ndcg@10": round(ndcg, 5)})
    return out


def _pct(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values), q)) if values else 0.0


def latency_profile(col: Collection, embedder: TransformerEmbedder, reranker: CrossEncoderReranker,
                    queries: Sequence[str], cfg: SearchConfig, warmup: int = 10) -> dict[str, Any]:  # fmt: skip
    retriever = Retriever(col, embedder, reranker, cfg)
    per_stage: dict[str, list[float]] = {}
    totals: list[float] = []
    for i, q in enumerate(queries):
        res = retriever.search(q)
        if i < warmup:
            continue
        totals.append(res.total_ms)
        for s in res.stages:
            per_stage.setdefault(s.name, []).append(s.latency_ms)
    return {
        "config": cfg.__dict__,
        "n_queries": len(totals),
        "total_ms": {
            "p50": round(_pct(totals, 50), 3),
            "p95": round(_pct(totals, 95), 3),
            "mean": round(float(np.mean(totals)), 3),
        },
        "stages": {
            name: {"p50": round(_pct(v, 50), 3), "p95": round(_pct(v, 95), 3), "mean": round(float(np.mean(v)), 3)}
            for name, v in per_stage.items()
        },
    }


def run_dataset(name: str, bge: TransformerEmbedder, minilm: TransformerEmbedder,
                reranker: CrossEncoderReranker) -> dict[str, Any]:  # fmt: skip
    t0 = time.time()
    test: BeirDataset = load_beir(name, "test")
    train: BeirDataset = load_beir(name, TRAIN_SPLIT[name])
    ids = list(test.corpus)
    texts = [test.corpus[i].full_text for i in ids]
    passages = {i: Passage(id=i, doc_id=i, text=test.corpus[i].text, title=test.corpus[i].title) for i in ids}
    print(f"[{name}] docs={len(ids)} test_q={len(test.eval_queries)} train_q={len(train.eval_queries)}")

    bm25 = BM25Index().build(ids, texts)
    flat: dict[str, FlatIndex] = {}
    for emb in (bge, minilm):
        flat[emb.name] = FlatIndex(emb.dim).add(ids, cached_doc_embeddings(emb, ids, texts))
    t = time.time()
    hnsw = HNSWIndex(bge.dim, m=16, ef_construction=200).build(ids, flat[bge.name].vectors)
    hnsw_build_s = time.time() - t

    tq, trq = test.eval_queries, train.eval_queries
    test_runs = {
        "bm25": _first_stage_bm25(bm25, tq),
        "dense": _first_stage_dense(flat[bge.name], bge, tq),
    }
    minilm_run = _first_stage_dense(flat[minilm.name], minilm, tq)
    hnsw_run = _first_stage_dense(hnsw, bge, tq, ef=128)
    train_runs = {"bm25": _first_stage_bm25(bm25, trq), "dense": _first_stage_dense(flat[bge.name], bge, trq)}

    tuned = tune_convex(train_runs, train.qrels)
    best = tuned["best"]
    rrf_sweep = tune_rrf_k(train_runs, train.qrels)
    test_alpha_sweep = [  # reported for the figure only — never used for selection
        {"alpha": float(a), "ndcg@10": round(evaluate(_to_run(_fuse(test_runs, convex, weights={"dense": float(a), "bm25": 1 - float(a)}, norm=best["norm"])), test.qrels, ks=(10,), metrics=("ndcg",))["ndcg@10"], 5)}  # noqa: E501
        for a in np.round(np.linspace(0, 1, 21), 3)
    ]

    fused_rrf = _fuse(test_runs, rrf, k=60)
    fused_cvx = _fuse(test_runs, convex, weights={"dense": best["alpha"], "bm25": 1 - best["alpha"]}, norm=best["norm"])

    rc = RerankCache(reranker, passages)
    t = time.time()
    rr_bm25 = rc.rerank(tq, test_runs["bm25"], DEPTH)
    rr_rrf = rc.rerank(tq, fused_rrf, DEPTH)
    rr_cvx = rc.rerank(tq, fused_cvx, DEPTH)
    rerank_wall_s = time.time() - t

    methods = {
        "BM25": test_runs["bm25"],
        "Dense · MiniLM-L6": minilm_run,
        "Dense · bge-small": test_runs["dense"],
        "Dense · bge-small (HNSW ef=128)": hnsw_run,
        "Hybrid · RRF": fused_rrf,
        "Hybrid · convex (train-tuned)": fused_cvx,
        "BM25 → rerank": rr_bm25,
        "Hybrid RRF → rerank": rr_rrf,
        "Hybrid convex → rerank": rr_cvx,
    }
    table = {m: _metrics(c, test.qrels) for m, c in methods.items()}
    for m, row in table.items():
        print(f"  {m:34s} nDCG@10={row['ndcg@10']:.4f} R@100={row['recall@100']:.4f}")

    # Rerank depth sweep (hybrid RRF first stage) with per-query GPU latency.
    depth_sweep = []
    sample_q = list(tq.items())[:100]
    for n in (10, 25, 50, 100):
        reranked = rc.rerank(tq, fused_rrf, n)
        ndcg = evaluate(_to_run(reranked), test.qrels, ks=(10,), metrics=("ndcg",))["ndcg@10"]
        lat = []
        for qid, qtext in sample_q:
            pool = [passages[p].index_text for p, _ in fused_rrf[qid][:n]]
            t = time.perf_counter()
            reranker.score(qtext, pool)
            lat.append((time.perf_counter() - t) * 1000)
        depth_sweep.append({"depth": n, "ndcg@10": round(ndcg, 5), "latency_ms_p50": round(_pct(lat, 50), 3),
                            "latency_ms_p95": round(_pct(lat, 95), 3)})  # fmt: skip

    # Serving-path latency through the real Retriever (HNSW + CE@50).
    col = Collection(name, list(passages.values()))
    col.bm25 = bm25
    col.dense[bge.name] = hnsw
    lat = latency_profile(col, bge, reranker, list(tq.values()), SearchConfig(mode="hybrid_rerank", rerank_depth=50, ef=128))  # noqa: E501
    lat_nr = latency_profile(col, bge, reranker, list(tq.values()), SearchConfig(mode="hybrid", ef=128))

    return {
        "dataset": name,
        "n_docs": len(ids),
        "n_test_queries": len(tq),
        "n_train_queries": len(trq),
        "metrics": table,
        "convex_tuning": tuned,
        "rrf_k_sweep_train": rrf_sweep,
        "convex_alpha_sweep_test_reference_only": test_alpha_sweep,
        "rerank_depth_sweep": depth_sweep,
        "rerank_pairs_scored": len(rc.scores),
        "rerank_wall_s": round(rerank_wall_s, 2),
        "hnsw": {"build_s": round(hnsw_build_s, 3), **hnsw.stats()},
        "latency_hybrid_rerank": lat,
        "latency_hybrid": lat_nr,
        "wall_s": round(time.time() - t0, 1),
    }


def environment() -> dict[str, Any]:
    import torch

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cpu": platform.processor(),
        "note": "GPU shared with another training job during measurement; latencies are conservative.",
    }


def main(datasets: Sequence[str], out: Path = RESULTS_DIR / "retrieval.json") -> dict[str, Any]:
    bge = TransformerEmbedder("BAAI/bge-small-en-v1.5", batch_size=64)
    minilm = TransformerEmbedder("sentence-transformers/all-MiniLM-L6-v2", batch_size=64)
    reranker = CrossEncoderReranker()
    results = {"environment": environment(), "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "datasets": {}}
    for name in datasets:
        results["datasets"][name] = run_dataset(name, bge, minilm, reranker)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return results


# --------------------------------------------------------------------------- blend
def _blend(first: Cands, ce: Cands, depth: int, method: str, w: float, k: int = 60) -> Cands:
    """Combine first-stage and cross-encoder evidence instead of replacing one with the other."""
    from ..fusion import normalize_scores

    out: Cands = {}
    for qid, cands in first.items():
        pool = cands[:depth]
        ce_scores = dict(ce[qid][:depth])
        pids = [p for p, _ in pool]
        if method == "score":
            f = normalize_scores(np.array([s for _, s in pool]), "minmax")
            c = normalize_scores(np.array([ce_scores[p] for p in pids]), "minmax")
            mixed = w * c + (1 - w) * f
        else:  # weighted reciprocal-rank blend
            ce_rank = {p: r for r, p in enumerate(sorted(pids, key=lambda p: -ce_scores[p]), start=1)}
            mixed = np.array([w / (k + ce_rank[p]) + (1 - w) / (k + r) for r, p in enumerate(pids, start=1)])
        order = np.argsort(-mixed, kind="stable")
        out[qid] = [(pids[i], float(mixed[i])) for i in order] + list(cands[depth:])
    return out


def blend_study(datasets: Sequence[str] = ("scifact", "nfcorpus"), depth: int = 50, max_train: int = 600,
                path: Path = RESULTS_DIR / "retrieval.json") -> dict[str, Any]:  # fmt: skip
    """Tune a first-stage/cross-encoder blend on train, report on test; append to retrieval.json."""
    results = json.loads(path.read_text(encoding="utf-8"))
    bge = TransformerEmbedder("BAAI/bge-small-en-v1.5", batch_size=64)
    reranker = CrossEncoderReranker()
    rng = np.random.default_rng(0)
    for name in datasets:
        test, train = load_beir(name, "test"), load_beir(name, TRAIN_SPLIT[name])
        ids = list(test.corpus)
        texts = [test.corpus[i].full_text for i in ids]
        passages = {i: Passage(id=i, doc_id=i, text=test.corpus[i].text, title=test.corpus[i].title) for i in ids}
        bm25 = BM25Index().build(ids, texts)
        flat = FlatIndex(bge.dim).add(ids, cached_doc_embeddings(bge, ids, texts))
        trq_all = train.eval_queries
        keep = sorted(rng.choice(list(trq_all), size=min(max_train, len(trq_all)), replace=False).tolist())
        trq = {q: trq_all[q] for q in keep}
        tq = test.eval_queries
        rc = RerankCache(reranker, passages)
        study: dict[str, Any] = {"depth": depth, "n_train_queries": len(trq), "sweep": []}
        fused = {}
        for split, queries in (("train", trq), ("test", tq)):
            runs = {"bm25": _first_stage_bm25(bm25, queries), "dense": _first_stage_dense(flat, bge, queries)}
            f = _fuse(runs, rrf, k=60)
            fused[split] = (f, rc.rerank(queries, f, depth))
        qrels_train = {q: train.qrels[q] for q in trq}
        for method in ("score", "rank"):
            for w in np.round(np.linspace(0, 1, 21), 3):
                f, ce = fused["train"]
                run = _to_run(_blend(f, ce, depth, method, float(w)))
                ndcg = evaluate(run, qrels_train, ks=(10,), metrics=("ndcg",))
                study["sweep"].append({"method": method, "w": float(w), "ndcg@10_train": round(ndcg["ndcg@10"], 5)})
        best = max(study["sweep"], key=lambda r: r["ndcg@10_train"])
        f, ce = fused["test"]
        blended = _blend(f, ce, depth, best["method"], best["w"])
        study["best"] = best
        study["test_metrics"] = _metrics(blended, test.qrels)
        study["test_plain_rerank@depth"] = _metrics(ce, test.qrels)
        study["test_hybrid_rrf"] = _metrics(f, test.qrels)
        results["datasets"][name]["rerank_blend"] = study
        label = BLEND_LABEL
        results["datasets"][name]["metrics"][label] = study["test_metrics"]
        print(f"[{name}] best on train: {best} | test nDCG@10 "
              f"blended={study['test_metrics']['ndcg@10']:.4f} "
              f"plain-rerank={study['test_plain_rerank@depth']['ndcg@10']:.4f} "
              f"hybrid={study['test_hybrid_rrf']['ndcg@10']:.4f}")
        path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results
