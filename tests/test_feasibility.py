from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest


def _ranker():
    try:
        module = importlib.import_module("soilmatch.feasibility")
    except ModuleNotFoundError:
        pytest.fail("soilmatch.feasibility has not been implemented")
    function = getattr(module, "add_within_group_exposure_rank", None)
    assert callable(function), "feasibility.add_within_group_exposure_rank has not been implemented"
    return function


def test_exposure_ranks_are_computed_within_environment_and_soil_groups():
    frame = pd.DataFrame(
        {
            "environment_group": [1, 1, 1, 1, 1, 1],
            "SoilType": [26, 26, 26, 27, 27, 27],
            "erosion": [0.0, 5.0, 10.0, 100.0, 200.0, 300.0],
        }
    )

    ranked = _ranker()(
        frame,
        exposure_column="erosion",
        group_columns=["environment_group", "SoilType"],
    )

    np.testing.assert_allclose(ranked["exposure_percentile"], [1 / 3, 2 / 3, 1, 1 / 3, 2 / 3, 1])
    assert frame.columns.tolist() == ["environment_group", "SoilType", "erosion"]


def test_exposure_rank_ties_share_a_percentile_and_missing_values_stay_missing():
    frame = pd.DataFrame(
        {
            "group": [1, 1, 1, 2],
            "erosion": [0.0, 0.0, 10.0, np.nan],
        }
    )

    ranked = _ranker()(
        frame,
        exposure_column="erosion",
        group_columns=["group"],
        output_column="rank",
    )

    np.testing.assert_allclose(ranked["rank"].iloc[:3], [0.5, 0.5, 1.0])
    assert np.isnan(ranked["rank"].iloc[3])


def test_exposure_rank_requires_existing_groups_and_nonnegative_values():
    frame = pd.DataFrame({"group": [1, 1], "erosion": [1.0, -1.0]})

    with np.testing.assert_raises_regex(ValueError, "nonnegative"):
        _ranker()(frame, exposure_column="erosion", group_columns=["group"])

    with np.testing.assert_raises_regex(ValueError, "Missing group columns"):
        _ranker()(frame, exposure_column="erosion", group_columns=["soil"])
