from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest


def _api(name: str):
    try:
        module = importlib.import_module("soilmatch.balance")
    except ModuleNotFoundError:
        pytest.fail("soilmatch.balance has not been implemented")
    function = getattr(module, name, None)
    assert callable(function), f"balance.{name} has not been implemented"
    return function


def test_balance_reports_unmatched_and_matched_standardized_differences():
    assess_balance = _api("assess_balance")
    frame = pd.DataFrame(
        {
            "cell_id": [1, 2, 3, 4],
            "Aa": [10.0, 9.0, 2.0, 1.0],
            "Elevation": [0.0, 0.0, 0.0, 10.0],
        }
    )
    pairs = pd.DataFrame(
        {
            "high_erosion_cell_id": [1],
            "low_erosion_cell_id": [3],
            "erosion_high": [10.0],
            "erosion_low": [2.0],
        }
    )

    result = assess_balance(frame, pairs, ["Elevation"]).iloc[0]

    assert abs(result["matched_smd"]) == pytest.approx(0.0)
    assert abs(result["unmatched_smd"]) > 0.1
    assert result["n_pairs"] == 1
    assert result["mean_absolute_pair_difference"] == pytest.approx(0.0)


def test_balance_returns_explicit_empty_result_for_no_pairs():
    assess_balance = _api("assess_balance")
    result = assess_balance(pd.DataFrame({"cell_id": [1], "Elevation": [1.0]}), pd.DataFrame(), ["Elevation"])

    assert result.empty
    assert list(result.columns) == [
        "covariate",
        "unmatched_smd",
        "matched_smd",
        "mean_absolute_pair_difference",
        "n_pairs",
    ]
