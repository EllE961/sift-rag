"""2-D embedding maps for the UI (PCA → t-SNE) with labeled topic clusters.

    python -m sift.eval projection

For each collection: bge-small document embeddings → PCA(50) → t-SNE(2).
Clusters come from k-means in the *original* embedding space (not the 2-D
map, which distorts distances) and are labeled by their most distinctive title
words (class-based TF-IDF). A query is placed on the map at the
similarity-weighted mean of its nearest neighbours' coordinates — t-SNE has no
out-of-sample transform, and this is the standard cheap approximation.

Cluster colours. Seven topics get seven hues from the dark-mode categorical
palette (yellow is left out: it would read as the amber "retrieved" marker).
Some hue pairs are nearly indistinguishable under colour-vision deficiency
(blue/violet, aqua/magenta, orange/green) or even with full colour vision
(orange/red), so hues are not assigned by rank: ``assign_colors`` measures
which clusters actually border each other on the map and picks the
permutation that minimises the length of borders between confusable hues.

On a t-SNE map almost every pair of clusters touches, which makes this an
all-pairs problem, and no 7-hue palette passes all pairs. The residual is
recorded in ``color_check`` rather than hidden. Colour is therefore never the
only identity channel: every topic is labelled on the map, the hover tooltip
names the topic, and hovering a topic in the sidebar isolates its points.
"""

from __future__ import annotations

import itertools
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .collection import Collection
from .config import INDEX_DIR
from .embed import cached_doc_embeddings

_STOP = frozenset(
    """a about above after again against all am an and any are as at be because been before being below between
    both but by can could did do does doing down during each few for from further had has have having he her here
    hers herself him himself his how i if in into is it its itself just me more most my myself no nor not now of
    off on once only or other our ours ourselves out over own same she should so some such than that the their
    theirs them themselves then there these they this those through to too under until up very was we were what
    when where which while who whom why will with you your yours yourself yourselves study studies patients patient
    effect effects associated association using use based among analysis results risk human vs versus new role
    may also two one three via without within across versus trial trials review evidence clinical cell cells""".split()
)


# Dark-mode categorical slots (dataviz reference palette) minus yellow.
CLUSTER_PALETTE = ("#3987e5", "#d95926", "#199e70", "#d55181", "#008300", "#9085e9", "#e66767")
# Pairwise OKLab ΔE×100 between those hues, computed with the dataviz skill's
# validator (scripts/validate_palette.js): worst of protan/deutan simulation, and unsimulated.
_CVD = (
    (0, 26.8, 19.6, 15.9, 27.3, 1.9, 19.2), (26.8, 0, 9.4, 10.5, 2.7, 26.0, 6.6),
    (19.6, 9.4, 0, 1.6, 11.5, 17.3, 6.5), (15.9, 10.5, 1.6, 0, 13.0, 16.0, 7.5),
    (27.3, 2.7, 11.5, 13.0, 0, 26.9, 8.6), (1.9, 26.0, 17.3, 16.0, 26.9, 0, 19.5),
    (19.2, 6.6, 6.5, 7.5, 8.6, 19.5, 0),
)  # fmt: skip
_NORMAL = (
    (0, 31.8, 20.9, 26.5, 29.9, 9.8, 29.0), (31.8, 0, 26.5, 11.6, 29.0, 27.0, 7.1),
    (20.9, 26.5, 0, 29.6, 11.9, 24.6, 27.5), (26.5, 11.6, 29.6, 0, 34.5, 19.7, 7.8),
    (29.9, 29.0, 11.9, 34.5, 0, 34.0, 32.6), (9.8, 27.0, 24.6, 19.7, 34.0, 0, 22.5),
    (29.0, 7.1, 27.5, 7.8, 32.6, 22.5, 0),
)  # fmt: skip
CVD_TARGET, CVD_FLOOR, NORMAL_FLOOR = 8.0, 6.0, 15.0


def pair_separation(a: int, b: int) -> float:
    """How distinguishable two palette slots are, as a fraction of the passing thresholds (>= 1 passes)."""
    return min(_CVD[a][b] / CVD_TARGET, _NORMAL[a][b] / NORMAL_FLOOR)


def cluster_adjacency(xy: np.ndarray, labels: np.ndarray, k: int = 10) -> dict[tuple[int, int], int]:
    """Count k-NN edges in the 2-D map that cross between clusters: a proxy for shared border length."""
    from sklearn.neighbors import NearestNeighbors

    _, idx = NearestNeighbors(n_neighbors=k + 1).fit(xy).kneighbors(xy)
    edges: Counter = Counter()
    for i, row in enumerate(idx):
        for j in row[1:]:
            a, b = int(labels[i]), int(labels[j])
            if a != b:
                edges[(min(a, b), max(a, b))] += 1
    return dict(edges)


def border_penalty(a: int, b: int) -> float:
    """0 for a distinguishable pair of slots; 2 if it fails the normal-vision floor; +1 if it fails CVD."""
    return 2.0 * (_NORMAL[a][b] < NORMAL_FLOOR) + 1.0 * (_CVD[a][b] < CVD_FLOOR)


def assign_colors(n_clusters: int, adjacency: dict[tuple[int, int], int]) -> list[int]:
    """Palette slot per cluster minimising the border length shared by confusable hues.

    The objective is ``Σ border_weight · border_penalty``. Ties are broken by border-weighted separation.
    """
    n_slots = len(CLUSTER_PALETTE)
    if n_clusters > n_slots:
        raise ValueError(f"at most {n_slots} clusters can get distinct palette hues")
    borders = list(adjacency.items())
    best, best_key = list(range(n_clusters)), (float("inf"), 0.0)
    for perm in itertools.permutations(range(n_slots), n_clusters):
        cost = sum(w * border_penalty(perm[a], perm[b]) for (a, b), w in borders)
        sep = sum(w * min(pair_separation(perm[a], perm[b]), 2.0) for (a, b), w in borders)
        key = (cost, -sep)
        if key < best_key:
            best, best_key = list(perm), key
    return best


def color_check(slots: list[int], adjacency: dict[tuple[int, int], int]) -> dict[str, float]:
    """Share of shared border (by k-NN edge count) between hues that pass each check."""
    total = sum(adjacency.values()) or 1
    ok_normal = sum(w for (a, b), w in adjacency.items() if _NORMAL[slots[a]][slots[b]] >= NORMAL_FLOOR)
    ok_cvd = sum(w for (a, b), w in adjacency.items() if _CVD[slots[a]][slots[b]] >= CVD_FLOOR)
    return {"border_share_normal_vision_ok": round(ok_normal / total, 3),
            "border_share_cvd_floor_ok": round(ok_cvd / total, 3)}  # fmt: skip


def _words(text: str) -> list[str]:
    words = (w.strip("-") for w in re.findall(r"[a-z][a-z\-]{2,}", text.lower()))
    return [w for w in words if len(w) >= 3 and w not in _STOP]


def label_clusters(titles: list[str], labels: np.ndarray, n_words: int = 3) -> dict[int, str]:
    """c-TF-IDF: words frequent in a cluster but rare across clusters."""
    k = int(labels.max()) + 1
    counts = [Counter() for _ in range(k)]
    for t, c in zip(titles, labels):
        counts[int(c)].update(set(_words(t)))
    df = Counter()
    for c in counts:
        df.update(c.keys())
    out = {}
    for ci, c in enumerate(counts):
        size = max(1, int((labels == ci).sum()))
        scored = sorted(c.items(), key=lambda kv: -(kv[1] / size) * np.log(1 + k / df[kv[0]]))
        out[ci] = " · ".join(w for w, _ in scored[:n_words])
    return out


def project(col: Collection, vectors: np.ndarray, n_clusters: int = 7, seed: int = 0) -> dict[str, Any]:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    t0 = time.time()
    pca = PCA(n_components=50, random_state=seed).fit_transform(vectors)
    xy = TSNE(n_components=2, perplexity=35, init="pca", learning_rate="auto", random_state=seed,
              n_jobs=4).fit_transform(pca)  # fmt: skip
    xy = (xy - xy.min(0)) / (xy.max(0) - xy.min(0))  # unit square
    km = KMeans(n_clusters=n_clusters, n_init=4, random_state=seed).fit(vectors)
    titles = [col.passage(pid).title or col.passage(pid).text[:120] for pid in col.ids]
    names = label_clusters(titles, km.labels_)
    centers = {
        int(c): [float(v) for v in np.median(xy[km.labels_ == c], axis=0)] for c in range(n_clusters)
    }
    adjacency = cluster_adjacency(xy, km.labels_)
    slots = assign_colors(n_clusters, adjacency)
    return {
        "collection": col.name,
        "n": len(col),
        "ids": col.ids,
        "xy": np.round(xy, 4).tolist(),
        "cluster": km.labels_.astype(int).tolist(),
        "clusters": [{"id": c, "label": names[c], "center": centers[c], "size": int((km.labels_ == c).sum()),
                      "color": CLUSTER_PALETTE[slots[c]]} for c in range(n_clusters)],  # fmt: skip
        "color_check": color_check(slots, adjacency),
        "titles": [t[:110] for t in titles],
        "seconds": round(time.time() - t0, 1),
    }


def place_query(projection: dict[str, Any], neighbor_ids: list[str], sims: list[float]) -> list[float]:
    index = {pid: i for i, pid in enumerate(projection["ids"])}
    pts, ws = [], []
    for pid, s in zip(neighbor_ids, sims):
        if pid in index:
            pts.append(projection["xy"][index[pid]])
            ws.append(np.exp(12.0 * s))
    if not pts:
        return [0.5, 0.5]
    w = np.asarray(ws) / np.sum(ws)
    return [float(v) for v in (np.asarray(pts) * w[:, None]).sum(0)]


def projection_path(name: str, root: Path = INDEX_DIR) -> Path:
    return root / name / "projection.json"


def main(names: tuple[str, ...] = ("scifact", "nfcorpus")) -> None:
    from .collection import load_or_build_beir
    from .embed import TransformerEmbedder

    emb = TransformerEmbedder("BAAI/bge-small-en-v1.5", batch_size=64)
    for name in names:
        col = load_or_build_beir(name, emb)
        vecs = cached_doc_embeddings(emb, col.ids, [col.passage(p).index_text for p in col.ids])
        proj = project(col, vecs)
        path = projection_path(name)
        path.write_text(json.dumps(proj), encoding="utf-8")
        print(f"{name}: {proj['n']} points, {len(proj['clusters'])} clusters in {proj['seconds']}s -> {path}")
        for c in proj["clusters"]:
            print(f"   [{c['id']:2d}] {c['size']:5d}  {c['color']}  {c['label']}")
        print(f"   colour check: {proj['color_check']}")
