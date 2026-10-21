import numpy as np
import pytest

from sift.index.flat import FlatIndex
from sift.index.hnsw import HNSWIndex


def clustered(n: int, dim: int, n_clusters: int = 20, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, dim))
    x = centers[rng.integers(0, n_clusters, n)] + 0.35 * rng.standard_normal((n, dim))
    x = x.astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


@pytest.fixture(scope="module")
def data():
    x = clustered(3000, 32)
    rng = np.random.default_rng(5)
    # In-distribution queries: perturbed corpus points (how real queries relate to a corpus).
    q = x[rng.integers(0, len(x), 200)] + 0.15 * rng.standard_normal((200, 32)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    ids = [f"v{i}" for i in range(len(x))]
    return ids, x, q


@pytest.fixture(scope="module")
def built(data):
    ids, x, _ = data
    return HNSWIndex(32, m=12, ef_construction=100).build(ids, x)


def recall(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean([len(set(a[i]) & set(b[i])) / b.shape[1] for i in range(len(a))]))


def test_recall_vs_exact(data, built):
    ids, x, q = data
    exact, _ = FlatIndex(32).add(ids, x).search_ids(q, 10)
    approx, sims = built.search_ids(q, 10, ef=64)
    assert recall(approx, exact) >= 0.95
    assert np.all(np.diff(sims, axis=1) <= 1e-6), "results must be sorted by similarity"
    # A self-query must return itself first.
    self_idx, _ = built.search_ids(x[:50], 1, ef=32)
    assert np.mean(self_idx[:, 0] == np.arange(50)) >= 0.98


def test_out_of_distribution_queries_need_larger_ef(data, built):
    """Queries far from every cluster are the hard case for graph search: the
    beam can settle in the wrong region. A wider beam recovers recall."""
    ids, x, _ = data
    q_ood = clustered(200, 32, seed=1)  # different cluster centers
    exact, _ = FlatIndex(32).add(ids, x).search_ids(q_ood, 10)
    narrow, _ = built.search_ids(q_ood, 10, ef=16)
    wide, _ = built.search_ids(q_ood, 10, ef=256)
    assert recall(wide, exact) >= 0.97
    assert recall(wide, exact) > recall(narrow, exact)


def test_ef_improves_recall_and_costs_more_distance_evals(data, built):
    ids, x, q = data
    exact, _ = FlatIndex(32).add(ids, x).search_ids(q, 10)
    low, _ = built.search_ids(q, 10, ef=10)
    evals_low = built.last_search_distance_evals
    high, _ = built.search_ids(q, 10, ef=128)
    evals_high = built.last_search_distance_evals
    assert recall(high, exact) >= recall(low, exact)
    assert evals_high > evals_low
    assert evals_high / len(q) < len(x), "HNSW must touch far fewer vectors than brute force"


def test_graph_invariants(built):
    assert built.count0.max() <= built.m0
    assert built.levels.max() == built.max_level
    assert built.levels[built.entry] == built.max_level
    # no self-loops on layer 0
    for i in range(0, len(built), 97):
        assert i not in built.links0[i, : built.count0[i]]
    stats = built.stats()
    assert stats["n"] == 3000 and stats["avg_degree_layer0"] > 4


def test_save_load_roundtrip(tmp_path, data, built):
    _, _, q = data
    built.save(tmp_path / "h")
    loaded = HNSWIndex.load(tmp_path / "h")
    a, _ = built.search_ids(q[:20], 5, ef=40)
    b, _ = loaded.search_ids(q[:20], 5, ef=40)
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("n", [1, 2, 5])
def test_tiny_indexes(n):
    x = clustered(n, 8, n_clusters=1)
    idx = HNSWIndex(8, m=4).build([str(i) for i in range(n)], x)
    hits = idx.search(x[0], k=10)
    assert len(hits) == n and hits[0][0] == "0"


def test_validation():
    with pytest.raises(ValueError):
        HNSWIndex(8, m=1)
    with pytest.raises(ValueError):
        HNSWIndex(8).build([], np.zeros((0, 8), np.float32))
    with pytest.raises(ValueError):
        HNSWIndex(8).build(["a"], np.zeros((1, 4), np.float32))
