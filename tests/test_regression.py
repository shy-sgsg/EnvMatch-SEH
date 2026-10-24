from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest


def _api(name: str):
    try:
        module = importlib.import_module("soilmatch.regression")
    except ModuleNotFoundError:
        pytest.fail("soilmatch.regression has not been implemented")
    function = getattr(module, name, None)
    assert callable(function), f"regression.{name} has not been implemented"
    return function


def test_group_regression_recovers_known_log_erosion_slope_and_zero_values():
    fit_group_models = _api("fit_group_models")
    erosion = np.array([0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 20.0, 30.0])
    exposure = np.log1p(erosion)
    frame = pd.DataFrame(
        {
            "environment_group": [1] * len(erosion),
            "Aa": erosion,
            "SHI": 0.7 + 1.8 * exposure + np.array([0.02, -0.03, 0.01, 0.0, 0.03, -0.01, 0.02, -0.02]),
        }
    )

    result = fit_group_models(frame, ["SHI"], "Aa", "environment_group").iloc[0]

    assert result["status"] == "ok"
    assert result["n_cells"] == len(frame)
    assert result["coefficient"] == pytest.approx(1.8, abs=0.02)
    assert result["ci_lower"] < result["coefficient"] < result["ci_upper"]
    assert np.isfinite(result["standardized_coefficient"])


def test_group_regression_reports_insufficient_sample_without_estimate():
    fit_group_models = _api("fit_group_models")
    frame = pd.DataFrame(
        {"environment_group": [1, 1], "Aa": [0.0, 1.0], "SHI": [0.2, 0.4]}
    )

    result = fit_group_models(frame, ["SHI"], "Aa", "environment_group").iloc[0]

    assert result["status"] == "insufficient_sample"
    assert np.isnan(result["coefficient"])


def test_pair_regression_uses_high_low_log_erosion_and_health_differences():
    fit_pair_models = _api("fit_pair_models")
    high_erosion = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0])
    delta_erosion = np.log1p(high_erosion) - np.log1p(np.zeros_like(high_erosion))
    pairs = pd.DataFrame(
        {
            "erosion_high": high_erosion,
            "erosion_low": np.zeros_like(high_erosion),
            "high_SHI": 0.4 + 2.5 * delta_erosion,
            "low_SHI": np.full_like(high_erosion, 0.4),
        }
    )

    result = fit_pair_models(pairs, ["SHI"]).iloc[0]

    assert result["status"] == "ok"
    assert result["coefficient"] == pytest.approx(2.5)
    assert result["n_pairs"] == len(pairs)


def test_group_interaction_formally_tests_different_slopes():
    fit_group_interaction = _api("fit_group_interaction")
    erosion = np.tile(np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]), 2)
    exposure = np.log1p(erosion)
    groups = np.array(["A"] * 6 + ["B"] * 6)
    outcome = np.r_[1 + exposure[:6], 1 + 3 * exposure[6:]]
    frame = pd.DataFrame(
        {"environment_group": groups, "Aa": erosion, "SHI": outcome}
    )

    result = fit_group_interaction(frame, "SHI", "Aa", "environment_group")

    assert result["status"] == "ok"
    assert result["df_num"] == 1
    assert result["p_value"] < 0.05
