from __future__ import annotations

import importlib

import numpy as np
import pytest


def _api(name: str):
    try:
        module = importlib.import_module("soilmatch.clustering")
    except ModuleNotFoundError:
        pytest.fail("soilmatch.clustering has not been implemented")
    function = getattr(module, name, None)
    assert callable(function), f"clustering.{name} has not been implemented"
    return function


def test_clustering_is_deterministic_and_keeps_soil_types_in_separate_strata():
    cluster_environments = _api("cluster_environments")
    matrix = np.array(
        [[0.0], [0.1], [10.0], [10.1], [100.0], [100.1], [110.0], [110.1]]
    )
    soil = np.array(["A"] * 4 + ["B"] * 4)

    labels1, centers1 = cluster_environments(matrix, soil, n_clusters=2, random_state=17)
    labels2, centers2 = cluster_environments(matrix, soil, n_clusters=2, random_state=17)

    np.testing.assert_array_equal(labels1, labels2)
    assert centers1.equals(centers2)
    assert len(np.unique(labels1[:4])) == 2
    assert len(np.unique(labels1[4:])) == 2
    assert set(centers1["soil_type"]) == {"A", "B"}
    assert set(labels1[:4]).isdisjoint(set(labels1[4:]))


def test_clustering_handles_singleton_strata_and_rejects_invalid_cluster_count():
    cluster_environments = _api("cluster_environments")
    labels, centers = cluster_environments(
        np.array([[1.0], [2.0], [3.0]]), np.array(["A", "A", "B"]), n_clusters=8
    )

    assert labels.shape == (3,)
    assert centers["n_cells"].sum() == 3
    with pytest.raises(ValueError, match="n_clusters"):
        cluster_environments(np.array([[1.0]]), np.array(["A"]), n_clusters=0)
