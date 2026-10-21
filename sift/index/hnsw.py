r"""Hierarchical Navigable Small World graphs, from scratch (Malkov & Yashunin, 2016).

Idea: a skip-list made of proximity graphs. Every vector lives in layer 0;
a geometrically shrinking random subset is promoted to layers 1, 2, ...
(``level ~ floor(-ln U · m_L)``, ``m_L = 1/ln M``). Search greedily descends
the sparse upper layers to find a good entry point, then runs a beam search of
width ``ef`` on the dense bottom layer. Expected search cost is
:math:`O(\log N)` distance evaluations instead of :math:`O(N)`.

Implementation notes
--------------------
* Everything hot is a ``numba.njit`` kernel over flat NumPy arrays — no Python
  objects per node, so a 100k-vector build takes seconds, not hours.
* Layer-0 adjacency is a dense ``(N, M0)`` int32 matrix (``M0 = 2M``, as in
  hnswlib); upper layers use a compact slot table addressed through a per-node
  offset, since only ~N/(M-1) nodes ever appear above layer 0.
* Neighbor selection uses the paper's heuristic (Algorithm 4): a candidate is
  kept only if it is closer to the base node than to every neighbor already
  kept. This keeps "long-range" edges that bridge clusters and is the main
  reason HNSW beats plain k-NN graphs on clustered data.
* "Visited" bookkeeping uses a generation counter instead of clearing a
  boolean array per query.
* Distance is cosine distance ``1 - <q, x>`` on L2-normalized float32 vectors.
"""

from __future__ import annotations

import heapq
import json
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numba import njit


# ----------------------------------------------------------------------------
# numba kernels
# ----------------------------------------------------------------------------
@njit(fastmath=True, cache=True, inline="always")
def _dist_q(vectors, a, q):
    s = np.float32(0.0)
    for j in range(q.shape[0]):
        s += vectors[a, j] * q[j]
    return 1.0 - np.float64(s)


@njit(fastmath=True, cache=True, inline="always")
def _dist_pair(vectors, a, b):
    s = np.float32(0.0)
    for j in range(vectors.shape[1]):
        s += vectors[a, j] * vectors[b, j]
    return 1.0 - np.float64(s)


@njit(cache=True, inline="always")
def _neighbors(node, layer, links0, count0, upper_offset, links_up, count_up):
    if layer == 0:
        return links0[node, : count0[node]]
    slot = upper_offset[node] + layer - 1
    return links_up[slot, : count_up[slot]]


@njit(cache=True)
def _greedy(q, ep, ep_d, layer, vectors, links0, count0, upper_offset, links_up, count_up, counter):
    changed = True
    while changed:
        changed = False
        nb = _neighbors(ep, layer, links0, count0, upper_offset, links_up, count_up)
        for t in range(nb.shape[0]):
            e = np.int64(nb[t])
            d = _dist_q(vectors, e, q)
            counter[0] += 1
            if d < ep_d:
                ep_d = d
                ep = e
                changed = True
    return ep, ep_d


@njit(cache=True)
def _search_layer(q, ep, ep_d, ef, layer, vectors, links0, count0, upper_offset, links_up, count_up,
                  visited, tag, counter):  # fmt: skip
    cand = [(ep_d, ep)]  # min-heap on distance
    res = [(-ep_d, ep)]  # max-heap on distance (negated)
    visited[ep] = tag
    while len(cand) > 0:
        d_c, c = heapq.heappop(cand)
        if d_c > -res[0][0]:
            break
        nb = _neighbors(c, layer, links0, count0, upper_offset, links_up, count_up)
        for t in range(nb.shape[0]):
            e = np.int64(nb[t])
            if visited[e] == tag:
                continue
            visited[e] = tag
            d_e = _dist_q(vectors, e, q)
            counter[0] += 1
            if len(res) < ef or d_e < -res[0][0]:
                heapq.heappush(cand, (d_e, e))
                heapq.heappush(res, (-d_e, e))
                if len(res) > ef:
                    heapq.heappop(res)
    m = len(res)
    ids = np.empty(m, np.int64)
    ds = np.empty(m, np.float64)
    for i in range(m):
        ds[i] = -res[i][0]
        ids[i] = res[i][1]
    order = np.argsort(ds)
    return ids[order], ds[order]


@njit(cache=True)
def _select_heuristic(vectors, cand_ids, cand_d, m):
    """Algorithm 4 (without extendCandidates / keepPrunedConnections, as hnswlib)."""
    selected = np.empty(m, np.int64)
    n_sel = 0
    for i in range(cand_ids.shape[0]):
        if n_sel >= m:
            break
        e = cand_ids[i]
        de = cand_d[i]
        good = True
        for j in range(n_sel):
            if _dist_pair(vectors, e, selected[j]) < de:
                good = False
                break
        if good:
            selected[n_sel] = e
            n_sel += 1
    return selected[:n_sel]


@njit(cache=True)
def _set_links(node, layer, sel, links0, count0, upper_offset, links_up, count_up):
    if layer == 0:
        for t in range(sel.shape[0]):
            links0[node, t] = sel[t]
        count0[node] = sel.shape[0]
    else:
        slot = upper_offset[node] + layer - 1
        for t in range(sel.shape[0]):
            links_up[slot, t] = sel[t]
        count_up[slot] = sel.shape[0]


@njit(cache=True)
def _connect(node, new, layer, cap, vectors, links0, count0, upper_offset, links_up, count_up):
    """Add the back-edge node -> new, shrinking node's adjacency if it overflows."""
    nb = _neighbors(node, layer, links0, count0, upper_offset, links_up, count_up)
    cnt = nb.shape[0]
    if cnt < cap:
        if layer == 0:
            links0[node, cnt] = new
            count0[node] = cnt + 1
        else:
            slot = upper_offset[node] + layer - 1
            links_up[slot, cnt] = new
            count_up[slot] = cnt + 1
        return
    cand = np.empty(cnt + 1, np.int64)
    cd = np.empty(cnt + 1, np.float64)
    for t in range(cnt):
        cand[t] = nb[t]
        cd[t] = _dist_pair(vectors, node, nb[t])
    cand[cnt] = new
    cd[cnt] = _dist_pair(vectors, node, new)
    order = np.argsort(cd)
    sel = _select_heuristic(vectors, cand[order], cd[order], cap)
    _set_links(node, layer, sel, links0, count0, upper_offset, links_up, count_up)


@njit(cache=True)
def _build(vectors, levels, upper_offset, m, m0, ef_construction, links0, count0, links_up, count_up):
    n = vectors.shape[0]
    visited = np.zeros(n, np.int32)
    counter = np.zeros(1, np.int64)
    tag = 0
    entry = 0
    max_level = levels[0]
    for i in range(1, n):
        q = vectors[i]
        lvl = levels[i]
        ep = np.int64(entry)
        ep_d = _dist_q(vectors, ep, q)
        for layer in range(max_level, lvl, -1):
            ep, ep_d = _greedy(q, ep, ep_d, layer, vectors, links0, count0, upper_offset, links_up, count_up,
                               counter)  # fmt: skip
        for layer in range(min(lvl, max_level), -1, -1):
            tag += 1
            ids, ds = _search_layer(q, ep, ep_d, ef_construction, layer, vectors, links0, count0, upper_offset,
                                    links_up, count_up, visited, tag, counter)  # fmt: skip
            sel = _select_heuristic(vectors, ids, ds, m)
            _set_links(i, layer, sel, links0, count0, upper_offset, links_up, count_up)
            cap = m0 if layer == 0 else m
            for t in range(sel.shape[0]):
                _connect(sel[t], i, layer, cap, vectors, links0, count0, upper_offset, links_up, count_up)
            ep = ids[0]
            ep_d = ds[0]
        if lvl > max_level:
            max_level = lvl
            entry = i
    return entry, max_level, counter[0]


@njit(cache=True)
def _search_batch(queries, k, ef, entry, max_level, vectors, links0, count0, upper_offset, links_up, count_up):
    nq = queries.shape[0]
    n = vectors.shape[0]
    out_i = np.full((nq, k), -1, np.int64)
    out_d = np.full((nq, k), np.inf, np.float64)
    visited = np.zeros(n, np.int32)
    counter = np.zeros(1, np.int64)
    for qi in range(nq):
        q = queries[qi]
        ep = np.int64(entry)
        ep_d = _dist_q(vectors, ep, q)
        counter[0] += 1
        for layer in range(max_level, 0, -1):
            ep, ep_d = _greedy(q, ep, ep_d, layer, vectors, links0, count0, upper_offset, links_up, count_up,
                               counter)  # fmt: skip
        ids, ds = _search_layer(q, ep, ep_d, max(ef, k), 0, vectors, links0, count0, upper_offset, links_up,
                                count_up, visited, qi + 1, counter)  # fmt: skip
        mm = min(k, ids.shape[0])
        out_i[qi, :mm] = ids[:mm]
        out_d[qi, :mm] = ds[:mm]
    return out_i, out_d, counter[0]


# ----------------------------------------------------------------------------
# Python API
# ----------------------------------------------------------------------------
class HNSWIndex:
    """Approximate nearest-neighbor index over L2-normalized vectors."""

    def __init__(self, dim: int, m: int = 16, ef_construction: int = 200, ef_search: int = 64, seed: int = 42):
        if m < 2:
            raise ValueError("M must be >= 2")
        self.dim = dim
        self.m = m
        self.m0 = 2 * m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.seed = seed
        self.ids: list[str] = []
        self.vectors = np.zeros((0, dim), np.float32)
        self.last_search_distance_evals = 0
        self.build_distance_evals = 0

    # ------------------------------------------------------------- build
    def build(self, ids: Sequence[str], vectors: np.ndarray) -> HNSWIndex:
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        n = vectors.shape[0]
        if n == 0:
            raise ValueError("cannot build an empty index")
        if vectors.shape[1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {vectors.shape[1]}")
        self.ids = list(ids)
        self.vectors = vectors
        rng = np.random.default_rng(self.seed)
        m_l = 1.0 / math.log(self.m)
        u = rng.random(n)
        self.levels = np.floor(-np.log(np.maximum(u, 1e-12)) * m_l).astype(np.int64)
        self.upper_offset = np.zeros(n, np.int64)
        self.upper_offset[1:] = np.cumsum(self.levels)[:-1]
        n_slots = int(self.levels.sum())
        self.links0 = np.zeros((n, self.m0), np.int32)
        self.count0 = np.zeros(n, np.int32)
        self.links_up = np.zeros((max(n_slots, 1), self.m), np.int32)
        self.count_up = np.zeros(max(n_slots, 1), np.int32)
        entry, max_level, evals = _build(
            self.vectors, self.levels, self.upper_offset, self.m, self.m0, self.ef_construction,
            self.links0, self.count0, self.links_up, self.count_up,
        )  # fmt: skip
        self.entry = int(entry)
        self.max_level = int(max_level)
        self.build_distance_evals = int(evals)
        return self

    def __len__(self) -> int:
        return len(self.ids)

    # ------------------------------------------------------------ search
    def search_ids(self, queries: np.ndarray, k: int, ef: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Returns (indices, similarities) of shape ``(n_queries, k)``; -1 pads."""
        queries = np.ascontiguousarray(np.atleast_2d(queries), dtype=np.float32)
        k = min(k, len(self.ids))
        idx, dist, evals = _search_batch(
            queries, k, ef or self.ef_search, self.entry, self.max_level, self.vectors,
            self.links0, self.count0, self.upper_offset, self.links_up, self.count_up,
        )  # fmt: skip
        self.last_search_distance_evals = int(evals)
        return idx, (1.0 - dist).astype(np.float32)

    def search(self, query: np.ndarray, k: int = 100, ef: int | None = None) -> list[tuple[str, float]]:
        idx, sims = self.search_ids(query, k, ef)
        return [(self.ids[i], float(s)) for i, s in zip(idx[0], sims[0]) if i >= 0]

    # ------------------------------------------------------------- stats
    def stats(self) -> dict:
        hist = np.bincount(self.levels).tolist()
        return {
            "n": len(self.ids),
            "dim": self.dim,
            "M": self.m,
            "M0": self.m0,
            "ef_construction": self.ef_construction,
            "max_level": self.max_level,
            "nodes_per_level": hist,
            "avg_degree_layer0": float(self.count0.mean()),
            "build_distance_evals": self.build_distance_evals,
            "memory_mb": (self.vectors.nbytes + self.links0.nbytes + self.links_up.nbytes) / 2**20,
        }

    # ------------------------------------------------------- persistence
    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        np.savez(
            path / "hnsw.npz",
            vectors=self.vectors, levels=self.levels, upper_offset=self.upper_offset,
            links0=self.links0, count0=self.count0, links_up=self.links_up, count_up=self.count_up,
        )  # fmt: skip
        meta = {
            "dim": self.dim, "m": self.m, "ef_construction": self.ef_construction, "ef_search": self.ef_search,
            "seed": self.seed, "entry": self.entry, "max_level": self.max_level, "ids": self.ids,
        }  # fmt: skip
        (path / "hnsw_meta.json").write_text(json.dumps(meta), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> HNSWIndex:
        meta = json.loads((path / "hnsw_meta.json").read_text(encoding="utf-8"))
        index = cls(meta["dim"], meta["m"], meta["ef_construction"], meta["ef_search"], meta["seed"])
        arrays = np.load(path / "hnsw.npz")
        for key in ("vectors", "levels", "upper_offset", "links0", "count0", "links_up", "count_up"):
            setattr(index, key, arrays[key])
        index.ids = meta["ids"]
        index.entry = meta["entry"]
        index.max_level = meta["max_level"]
        return index
