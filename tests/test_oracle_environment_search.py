from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest


def _api(name: str):
    try:
        module = importlib.import_module("experiments.oracle_exploration.environment_search")
    except ModuleNotFoundError:
        pytest.fail("experiments.oracle_exploration.environment_search has not been implemented")
    function = getattr(module, name, None)
    assert callable(function), f"environment_search.{name} has not been implemented"
    return function


def _columns():
    return [
        "Elevation",
        "Cur",
        "TWI",
        "LST",
        "NDWI",
        "NDBSI",
        "EVI",
        "NDVI",
        "Precipitation",
        "SoilType",
        "Aa_20yr",
        "SHI_candidate",
        "AN",
        "cell_id",
        "x",
        "y",
    ]


def test_environment_sets_include_named_vectors_and_seeded_nonhealth_subsets():
    candidate_environment_sets = _api("candidate_environment_sets")

    first = candidate_environment_sets(_columns(), seed=7, n_random=20)
    second = candidate_environment_sets(_columns(), seed=7, n_random=20)
    named = dict(first[:8])
    random_sets = first[8:]

    assert list(named) == [f"V{i}" for i in range(1, 9)]
    assert named["V1"] == ("Elevation", "Cur", "TWI", "LST")
    assert named["V7"] == ("Elevation", "Cur", "TWI", "LST", "NDWI")
    assert named["V8"] == ("Elevation", "Cur", "TWI", "LST", "EVI")
    assert first == second
    assert len(random_sets) == 20
    allowed = {
        "Elevation",
        "Cur",
        "TWI",
        "LST",
        "NDWI",
        "NDBSI",
        "EVI",
        "NDVI",
        "Precipitation",
    }
    assert all(set(variables) <= allowed for _, variables in first)
    assert all(len(variables) == len(set(variables)) for _, variables in first)
    assert all(sum(name in variables for name in ("NDWI", "NDBSI", "EVI", "NDVI")) <= 1 for _, variables in random_sets)
    assert all(not {"Elevation", "Precipitation"} <= set(variables) for _, variables in random_sets)


def test_random_environment_sets_obey_correlation_groups_from_config():
    candidate_environment_sets = _api("candidate_environment_sets")

    sets = candidate_environment_sets(
        _columns(),
        seed=4,
        n_random=32,
        correlated_variable_groups=(("Cur", "TWI"),),
    )

    assert all(not {"Cur", "TWI"} <= set(variables) for _, variables in sets[8:])


def test_distance_transforms_are_finite_regularized_and_environment_only():
    transform_environment_features = _api("transform_environment_features")
    frame = pd.DataFrame(
        {
            "Elevation": np.arange(20, dtype=float),
            "TWI": np.arange(20, dtype=float) * 2.0 + 0.01,
            "LST": np.sin(np.arange(20, dtype=float)),
            "AN": np.arange(20, dtype=float),
        }
    )

    standard, standard_metadata = transform_environment_features(
        frame, ["Elevation", "TWI", "LST"], method="standardized_euclidean"
    )
    weighted, _ = transform_environment_features(
        frame, ["Elevation", "TWI", "LST"], method="weighted_euclidean", weights={"Elevation": 4.0}
    )
    mahalanobis, mahalanobis_metadata = transform_environment_features(
        frame, ["Elevation", "TWI", "LST"], method="mahalanobis"
    )
    pca, pca_metadata = transform_environment_features(
        frame, ["Elevation", "TWI", "LST"], method="pca_euclidean"
    )
    nearest, nearest_metadata = transform_environment_features(
        frame, ["Elevation", "TWI", "LST"], method="direct_nearest_neighbor"
    )

    assert standard.shape == weighted.shape == mahalanobis.shape == nearest.shape == (20, 3)
    assert pca.shape[0] == 20 and 1 <= pca.shape[1] <= 3
    assert np.isfinite(standard).all() and np.isfinite(weighted).all()
    assert np.isfinite(mahalanobis).all() and np.isfinite(pca).all() and np.isfinite(nearest).all()
    assert weighted.std(axis=0)[0] > standard.std(axis=0)[0]
    assert standard_metadata["features"] == ["Elevation", "TWI", "LST"]
    assert mahalanobis_metadata["regularization"] > 0
    assert pca_metadata["retained_components"] == pca.shape[1]
    assert nearest_metadata["method"] == "direct_nearest_neighbor"

    with pytest.raises(ValueError, match="health|environment"):
        transform_environment_features(frame, ["AN"], method="standardized_euclidean")


def test_soil_counts_and_labeling_exclude_water_and_variable_missing_rows():
    soil_category_counts = _api("soil_category_counts")
    fit_environment_labels = _api("fit_environment_labels")
    frame = pd.DataFrame(
        {
            "Elevation": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "LST": [10.0, 11.0, 12.0, 13.0, np.nan, 15.0],
            "SoilType": [26, 26, 27, 42, 42, 85],
        }
    )

    counts = soil_category_counts(frame)
    labels = fit_environment_labels(
        frame,
        ["Elevation", "LST"],
        method="none",
        k=0,
        by_soil=True,
        seed=11,
    )

    category_counts = counts.set_index("soil_type")
    assert category_counts.loc[26, "n_cells"] == 2
    assert category_counts.loc[85, "n_cells"] == 1
    assert not bool(category_counts.loc[85, "primary_soil_stratum"])
    assert labels.tolist() == [0, 0, 0, 0, -1, -1]


def test_soil_stratified_minibatch_labels_are_reproducible_and_disjoint():
    fit_environment_labels = _api("fit_environment_labels")
    rng = np.random.default_rng(15)
    frames = []
    for soil_code, offset in ((26, 0.0), (27, 20.0), (42, 40.0)):
        values = np.repeat(np.arange(3, dtype=float) * 10.0 + offset, 20)
        frames.append(
            pd.DataFrame(
                {
                    "Elevation": values + rng.normal(0, 0.05, len(values)),
                    "LST": values + rng.normal(0, 0.05, len(values)),
                    "SoilType": soil_code,
                }
            )
        )
    frame = pd.concat(frames, ignore_index=True)

    first = fit_environment_labels(
        frame, ["Elevation", "LST"], method="minibatch_kmeans", k=3, by_soil=True, seed=13
    )
    second = fit_environment_labels(
        frame, ["Elevation", "LST"], method="minibatch_kmeans", k=3, by_soil=True, seed=13
    )

    np.testing.assert_array_equal(first, second)
    for soil_code in (26, 27, 42):
        labels = set(first[frame["SoilType"].eq(soil_code)])
        assert len(labels) == 3
    assert set(first[frame["SoilType"].eq(26)]).isdisjoint(set(first[frame["SoilType"].eq(27)]))


@pytest.mark.parametrize("method", ["gaussian_mixture", "agglomerative"])
def test_bounded_cluster_methods_assign_labels_to_all_rows(method):
    fit_environment_labels = _api("fit_environment_labels")
    frame = pd.DataFrame(
        {
            "Elevation": np.r_[np.linspace(0, 1, 20), np.linspace(8, 9, 20)],
            "LST": np.r_[np.linspace(0, 1, 20), np.linspace(8, 9, 20)],
            "SoilType": [26] * 40,
        }
    )

    labels = fit_environment_labels(
        frame, ["Elevation", "LST"], method=method, k=2, by_soil=True, seed=9
    )

    assert labels.shape == (40,)
    assert set(np.unique(labels)) == {0, 1}
