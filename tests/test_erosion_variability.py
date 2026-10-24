from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest


def _api(name: str):
    try:
        module = importlib.import_module("soilmatch.erosion_variability")
    except ModuleNotFoundError:
        pytest.fail("soilmatch.erosion_variability has not been implemented")
    function = getattr(module, name, None)
    assert callable(function), f"erosion_variability.{name} has not been implemented"
    return function


def test_erosion_summary_reports_quantiles_zero_share_and_log_values():
    summarize_erosion = _api("summarize_erosion")
    frame = pd.DataFrame(
        {"environment_group": [1, 1, 1, 1, 2, 2], "Aa": [0.0, 1.0, 3.0, np.nan, 0.0, 3.0]}
    )

    summary = summarize_erosion(frame, "environment_group", "Aa").set_index(
        "environment_group"
    )

    assert summary.loc[1, "valid_count"] == 3
    assert summary.loc[1, "total_count"] == 4
    assert summary.loc[1, "zero_proportion"] == pytest.approx(1 / 3)
    assert summary.loc[1, "p25"] == pytest.approx(0.5)
    assert summary.loc[1, "p95"] == pytest.approx(2.8)
    assert summary.loc[2, "log1p_mean"] == pytest.approx(np.log(4) / 2)


def test_erosion_summary_rejects_negative_finite_exposure():
    summarize_erosion = _api("summarize_erosion")
    with pytest.raises(ValueError, match="nonnegative"):
        summarize_erosion(pd.DataFrame({"group": [1], "Aa": [-1.0]}), "group", "Aa")


def test_erosion_class_composition_uses_explicit_edges_only():
    summarize_erosion = _api("summarize_erosion")
    summary = summarize_erosion(
        pd.DataFrame({"group": [1, 1, 1], "Aa": [0.0, 0.5, 2.5]}),
        "group",
        "Aa",
        class_edges=[0.0, 1.0, 2.0, 4.0],
        class_labels=["low", "moderate", "high"],
    ).iloc[0]

    assert summary["class_low_proportion"] == pytest.approx(2 / 3)
    assert summary["class_moderate_proportion"] == pytest.approx(0.0)
    assert summary["class_high_proportion"] == pytest.approx(1 / 3)
