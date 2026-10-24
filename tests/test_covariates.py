from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest


def _api(name: str):
    try:
        module = importlib.import_module("soilmatch.covariates")
    except ModuleNotFoundError:
        pytest.fail("soilmatch.covariates has not been implemented")
    function = getattr(module, name, None)
    assert callable(function), f"covariates.{name} has not been implemented"
    return function


def test_covariate_screen_uses_environment_values_and_reports_redundancy():
    screen_covariates = _api("screen_covariates")
    frame = pd.DataFrame(
        {
            "Elevation": [1.0, 2.0, 3.0, 4.0],
            "LST": [10.0, 20.0, 30.0, 40.0],
            "SoilType": ["A", "A", "B", "B"],
            "SHI": [1000.0, -1000.0, 0.0, 0.0],
        }
    )

    result = screen_covariates(frame, ["Elevation", "LST", "SoilType", "SHI"])

    rows = result.set_index("logical_name")
    assert rows.loc["Elevation", "status"] == "candidate"
    assert rows.loc["LST", "most_correlated_with"] == "Elevation"
    assert rows.loc["SoilType", "status"] == "non_numeric"
    assert rows.loc["SHI", "status"] == "excluded_health_result"


def test_covariate_screen_reports_missing_constant_and_high_missingness():
    screen_covariates = _api("screen_covariates")
    frame = pd.DataFrame(
        {
            "Elevation": [1.0, 1.0, 1.0],
            "TWI": [1.0, np.nan, np.nan],
        }
    )

    result = screen_covariates(
        frame, ["Elevation", "TWI", "Missing"], max_missing_fraction=0.5
    ).set_index("logical_name")

    assert result.loc["Elevation", "status"] == "constant"
    assert result.loc["TWI", "status"] == "high_missingness"
    assert result.loc["Missing", "status"] == "missing_column"
