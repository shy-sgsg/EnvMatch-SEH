from __future__ import annotations

import importlib
import warnings

import numpy as np
import pandas as pd
import pytest


def _api(name: str):
    try:
        module = importlib.import_module("soilmatch.visualization")
    except ModuleNotFoundError:
        pytest.fail("soilmatch.visualization has not been implemented")
    function = getattr(module, name, None)
    assert callable(function), f"visualization.{name} has not been implemented"
    return function


def test_requested_figure_functions_write_academic_style_outputs(tmp_path):
    plot_environment_groups = _api("plot_environment_groups")
    plot_environment_profile = _api("plot_environment_profile")
    plot_erosion_distribution = _api("plot_erosion_distribution")
    plot_matching_balance = _api("plot_matching_balance")
    plot_group_regression = _api("plot_group_regression")
    plot_pair_differences = _api("plot_pair_differences")
    plot_regression_forest = _api("plot_regression_forest")
    plot_indicator_response_heatmap = _api("plot_indicator_response_heatmap")
    plot_pcnm_comparison = _api("plot_pcnm_comparison")

    frame = pd.DataFrame(
        {
            "x": [0, 1, 2, 3, 4, 5],
            "y": [0, 0, 1, 1, 2, 2],
            "environment_group": [1, 1, 1, 2, 2, 2],
            "Aa": [0, 1, 2, 3, 4, 5],
            "SHI": [0.1, 0.2, 0.4, 0.3, 0.5, 0.8],
            "AN": [0.4, 0.3, 0.2, 0.1, 0.3, 0.4],
        }
    )
    centers = pd.DataFrame(
        {"environment_group": [1, 2], "center_0": [-0.5, 0.5], "center_1": [0.2, -0.2]}
    )
    balance = pd.DataFrame(
        {"covariate": ["Elevation", "TWI"], "unmatched_smd": [0.8, 0.4], "matched_smd": [0.1, 0.05]}
    )
    group_results = pd.DataFrame(
        {
            "outcome": ["SHI", "SHI"],
            "environment_group": [1, 2],
            "coefficient": [0.2, 0.3],
            "status": ["ok", "ok"],
            "ci_lower": [0.1, 0.15],
            "ci_upper": [0.3, 0.45],
            "standardized_coefficient": [0.4, 0.5],
        }
    )
    pairs = pd.DataFrame(
        {
            "erosion_high": [4.0, 5.0],
            "erosion_low": [1.0, 2.0],
            "high_SHI": [0.5, 0.8],
            "low_SHI": [0.2, 0.4],
        }
    )
    comparison = pd.DataFrame(
        {"outcome": ["SHI"], "coefficient_pcnm": [-0.2], "coefficient_environmental": [-0.1]}
    )
    outputs = [
        plot_environment_groups(frame, "x", "y", "environment_group", tmp_path / "groups.png"),
        plot_environment_profile(centers, tmp_path / "profile.png"),
        plot_erosion_distribution(frame, "environment_group", "Aa", tmp_path / "erosion.png"),
        plot_matching_balance(balance, tmp_path / "balance.png"),
        plot_group_regression(frame, group_results, "SHI", "Aa", "environment_group", tmp_path / "regression.png"),
        plot_pair_differences(pairs, "SHI", tmp_path / "pairs.png"),
        plot_regression_forest(group_results, tmp_path / "forest.png"),
        plot_indicator_response_heatmap(group_results.assign(outcome="SHI"), tmp_path / "health.png"),
        plot_pcnm_comparison(comparison, tmp_path / "pcnm.png"),
    ]

    assert len(outputs) == 9
    assert all(path.is_file() and path.stat().st_size > 100 for path in outputs)


def test_plotting_uses_an_installed_chinese_font_without_missing_glyph_warnings(tmp_path):
    plot_environment_groups = _api("plot_environment_groups")
    frame = pd.DataFrame({"x": [0, 1], "y": [0, 1], "group": ["环境甲", "环境乙"]})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        plot_environment_groups(frame, "x", "y", "group", tmp_path / "cjk.png")

    glyph_warnings = [warning for warning in caught if "Glyph" in str(warning.message)]
    assert not glyph_warnings
