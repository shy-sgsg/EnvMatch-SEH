from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest


def _api(name: str):
    try:
        module = importlib.import_module("soilmatch.matching")
    except ModuleNotFoundError:
        pytest.fail("soilmatch.matching has not been implemented")
    function = getattr(module, name, None)
    assert callable(function), f"matching.{name} has not been implemented"
    return function


def _frame():
    return pd.DataFrame(
        {
            "cell_id": [1, 2, 3, 4, 5, 6],
            "environment_group": [1, 1, 1, 1, 1, 2],
            "SoilType": ["A", "A", "A", "A", "B", "A"],
            "Aa": [10.0, 9.0, 2.0, 1.0, 0.0, 0.0],
            "Elevation": [0.0, 0.1, 0.05, 0.2, 0.0, 0.0],
            "x": [0.0, 10.0, 5.0, 15.0, 5.0, 5.0],
            "y": [0.0] * 6,
            "SHI": [1000.0, -1000.0, 500.0, -500.0, 1.0, 2.0],
        }
    )


def _match(frame, **overrides):
    match_cells = _api("match_cells")
    arguments = {
        "environment_columns": ["Elevation"],
        "group_col": "environment_group",
        "erosion_col": "Aa",
        "soil_type_col": "SoilType",
        "x_col": "x",
        "y_col": "y",
        "distance_limit": 100.0,
        "environment_distance_limit": 0.5,
        "erosion_gap": 5.0,
        "allow_reuse": False,
    }
    arguments.update(overrides)
    return match_cells(frame, **arguments)


def test_matching_respects_group_soil_environment_erosion_and_one_to_one_constraints():
    pairs, diagnostics = _match(_frame())

    assert len(pairs) == 2
    assert pairs["high_erosion_cell_id"].tolist() == [1, 2]
    assert pairs["low_erosion_cell_id"].tolist() == [3, 4]
    assert pairs["environment_group"].tolist() == [1, 1]
    assert (pairs["erosion_difference"] >= 5.0).all()
    assert pairs["environmental_distance"].max() <= 0.5
    assert pairs["geographic_distance"].max() <= 100.0
    assert len(set(pairs["high_erosion_cell_id"]) | set(pairs["low_erosion_cell_id"])) == 4
    assert diagnostics["matched_pairs"] == 2


def test_health_columns_never_change_pair_selection():
    frame = _frame()
    baseline, _ = _match(frame)
    frame["SHI"] = np.arange(len(frame)) * 1e12
    frame["AN"] = np.arange(len(frame))[::-1]
    changed, _ = _match(frame)

    pd.testing.assert_frame_equal(baseline, changed)
    assert not {"SHI", "AN"}.intersection(changed.columns)


def test_matching_rejects_soil_group_or_environment_mismatches():
    frame = pd.DataFrame(
        {
            "cell_id": [1, 2, 3],
            "environment_group": [1, 1, 1],
            "SoilType": ["A", "B", "A"],
            "Aa": [10.0, 0.0, 0.0],
            "Elevation": [0.0, 0.0, 10.0],
            "x": [0.0, 0.0, 0.0],
            "y": [0.0, 0.0, 0.0],
        }
    )

    pairs, diagnostics = _match(frame)

    assert pairs.empty
    assert diagnostics["matched_pairs"] == 0
    assert diagnostics["unmatched_high_cells"] == 1


def test_reference_reuse_is_configurable_but_high_cells_are_not_reused():
    frame = pd.DataFrame(
        {
            "cell_id": [1, 2, 3],
            "environment_group": [1, 1, 1],
            "SoilType": ["A", "A", "A"],
            "Aa": [10.0, 9.0, 0.0],
            "Elevation": [0.0, 0.0, 0.0],
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 0.0, 0.0],
        }
    )

    unique_pairs, _ = _match(frame)
    reused_pairs, _ = _match(frame, allow_reuse=True)

    assert len(unique_pairs) == 1
    assert len(reused_pairs) == 2
    assert reused_pairs["low_erosion_cell_id"].tolist() == [3, 3]
    assert reused_pairs["high_erosion_cell_id"].is_unique


def test_matching_requires_explicit_finite_distance_constraints():
    with pytest.raises(ValueError, match="environment_distance_limit"):
        _match(_frame(), environment_distance_limit=None)
