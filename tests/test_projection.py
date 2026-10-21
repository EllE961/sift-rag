import numpy as np
import pytest

from sift.projection import (
    CLUSTER_PALETTE,
    assign_colors,
    border_penalty,
    cluster_adjacency,
    color_check,
    label_clusters,
    place_query,
)


def test_cluster_adjacency_counts_cross_cluster_neighbours():
    rng = np.random.default_rng(0)
    a = rng.normal((0, 0), 0.1, (50, 2))
    b = rng.normal((0.5, 0), 0.1, (50, 2))  # touches a
    c = rng.normal((5, 5), 0.1, (50, 2))  # isolated
    xy = np.vstack([a, b, c])
    labels = np.array([0] * 50 + [1] * 50 + [2] * 50)
    adj = cluster_adjacency(xy, labels, k=5)
    assert adj.get((0, 1), 0) > 0
    assert (0, 2) not in adj and (1, 2) not in adj


def test_assign_colors_keeps_confusable_hues_apart():
    # A path 0-1-2-3-4-5-6 of heavy borders: an assignment with zero penalty exists.
    adjacency = {(i, i + 1): 100 for i in range(6)}
    slots = assign_colors(7, adjacency)
    assert sorted(slots) == list(range(7))  # every hue used once
    assert all(border_penalty(slots[a], slots[b]) == 0 for a, b in adjacency)
    check = color_check(slots, adjacency)
    assert check == {"border_share_normal_vision_ok": 1.0, "border_share_cvd_floor_ok": 1.0}


def test_known_confusable_pairs_are_penalised():
    blue, violet = CLUSTER_PALETTE.index("#3987e5"), CLUSTER_PALETTE.index("#9085e9")
    aqua, magenta = CLUSTER_PALETTE.index("#199e70"), CLUSTER_PALETTE.index("#d55181")
    assert border_penalty(blue, violet) > 0
    assert border_penalty(aqua, magenta) > 0
    assert border_penalty(blue, aqua) == 0
    with pytest.raises(ValueError):
        assign_colors(len(CLUSTER_PALETTE) + 1, {})


def test_label_clusters_and_query_placement():
    titles = ["Malaria transmission in Africa", "Malaria vaccine trial", "Stem cell pluripotency", "Embryonic stem cells"]
    names = label_clusters(titles, np.array([0, 0, 1, 1]), n_words=1)
    assert names == {0: "malaria", 1: "stem"}
    proj = {"ids": ["a", "b"], "xy": [[0.0, 0.0], [1.0, 1.0]]}
    x, y = place_query(proj, ["a", "b"], [0.9, 0.1])
    assert x < 0.5 and y < 0.5  # pulled toward the more similar neighbour
    assert place_query(proj, ["zzz"], [1.0]) == [0.5, 0.5]
