from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest


def _api(name: str):
    try:
        module = importlib.import_module("soilmatch.environment_vectors")
    except ModuleNotFoundError:
        pytest.fail("soilmatch.environment_vectors has not been implemented")
    function = getattr(module, name, None)
    assert callable(function), f"environment_vectors.{name} has not been implemented"
    return function


def test_environment_vectors_standardize_units_and_keep_categories_outside_distance():
    build_environment_vectors = _api("build_environment_vectors")
    frame = pd.DataFrame(
        {"Elevation": [0.0, 1000.0], "LST": [10.0, 30.0], "SoilType": ["A", "B"]}
    )

    matrix, metadata = build_environment_vectors(
        frame,
        continuous=["Elevation", "LST"],
        categorical=["SoilType"],
        weights={"Elevation": 4.0, "LST": 1.0},
    )

    np.testing.assert_allclose(matrix, [[-2.0, -1.0], [2.0, 1.0]])
    assert metadata["columns"] == ["Elevation", "LST"]
    assert metadata["categorical_columns"] == ["SoilType"]
    assert metadata["strata"]["SoilType"].tolist() == ["A", "B"]
    assert frame["Elevation"].tolist() == [0.0, 1000.0]


def test_environment_vectors_reject_missing_values_and_nonpositive_weights():
    build_environment_vectors = _api("build_environment_vectors")
    with pytest.raises(ValueError, match="missing|finite"):
        build_environment_vectors(
            pd.DataFrame({"Elevation": [1.0, np.nan]}), continuous=["Elevation"]
        )
    with pytest.raises(ValueError, match="weight"):
        build_environment_vectors(
            pd.DataFrame({"Elevation": [1.0, 2.0]}),
            continuous=["Elevation"],
            weights={"Elevation": 0.0},
        )


def test_environment_distances_support_weighted_rowwise_euclidean_distance():
    environment_distances = _api("environment_distances")

    distances = environment_distances(
        np.array([[0.0, 0.0], [1.0, 1.0]]),
        np.array([[3.0, 4.0], [4.0, 5.0]]),
        weights=[4.0, 1.0],
    )

    np.testing.assert_allclose(distances, [np.sqrt(52.0), np.sqrt(52.0)])

