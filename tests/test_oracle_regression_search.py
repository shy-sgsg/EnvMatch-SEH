from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest


def _api(module_name: str, name: str):
    module = importlib.import_module(f"experiments.oracle_exploration.{module_name}")
    function = getattr(module, name, None)
    assert callable(function), f"{module_name}.{name} has not been implemented"
    return function


def _frame():
    rng = np.random.default_rng(41)
    n = 90
    exposure = np.linspace(0.0, 25.0, n)
    elevation = rng.normal(size=n)
    lst = rng.normal(size=n)
    outcome = 2.0 * np.log1p(exposure) + 0.4 * elevation - 0.2 * lst + rng.normal(0, 0.1, n)
    frame = pd.DataFrame(
        {
            "cell_id": np.arange(n),
            "Aa_20yr": exposure,
            "AN": outcome,
            "Elevation": elevation,
            "LST": lst,
            "environment_group": np.repeat([0, 1, 2], n // 3),
            "spatial_role": ["train"] * 54 + ["validation"] * 18 + ["test"] * 18,
        }
    )
    frame.loc[0, "AN"] = np.nan
    frame.loc[5, "LST"] = np.nan
    return frame


def test_candidate_models_use_complete_cases_and_keep_spatial_test_out_of_fit():
    fit_candidate_models = _api("regression_search", "fit_candidate_models")
    frame = _frame()
    result = fit_candidate_models(
        frame,
        "AN",
        "Aa_20yr",
        ["Elevation", "LST"],
        group="environment_group",
        model_families=["ols", "quadratic", "statsmodels_gam"],
    )

    assert set(result["evaluation_set"]) == {"apparent", "train", "validation", "test"}
    assert result.loc[result["evaluation_set"].eq("train"), "n"].eq(52).all()
    assert result.loc[result["evaluation_set"].eq("validation"), "n"].eq(18).all()
    assert result.loc[result["evaluation_set"].eq("test"), "n"].eq(18).all()
    assert result["outcome"].eq("AN").all()
    assert result["outcome_guided_selection"].eq(True).all()
    train_ols = result.loc[
        result["model_name"].eq("ols") & result["evaluation_set"].eq("train")
    ].iloc[0]
    assert train_ols["coefficient"] > 0
    assert train_ols["direction"] == "positive"

    changed_test = frame.copy()
    test_mask = changed_test["spatial_role"].eq("test")
    changed_test.loc[test_mask, "AN"] += 1000.0
    changed = fit_candidate_models(
        changed_test,
        "AN",
        "Aa_20yr",
        ["Elevation", "LST"],
        group="environment_group",
        model_families=["ols", "quadratic", "statsmodels_gam"],
    )
    before_train = result.loc[result["evaluation_set"].eq("train")].sort_values("model_name")
    after_train = changed.loc[changed["evaluation_set"].eq("train")].sort_values("model_name")
    pd.testing.assert_series_equal(before_train["r_squared"].reset_index(drop=True), after_train["r_squared"].reset_index(drop=True))
    before_test = result.loc[result["evaluation_set"].eq("test")].sort_values("model_name")
    after_test = changed.loc[changed["evaluation_set"].eq("test")].sort_values("model_name")
    assert not np.allclose(before_test["r_squared"], after_test["r_squared"])


def test_feature_regimes_share_complete_cases_and_delta_r2_uses_same_split():
    compare_feature_regimes = _api("regression_search", "compare_feature_regimes")
    compute_delta_r2 = _api("regression_search", "compute_delta_r2")
    frame = _frame()
    pairs = pd.DataFrame(
        {
            "erosion_high": [8.0, 10.0, 15.0, 20.0],
            "erosion_low": [1.0, 2.0, 4.0, 6.0],
            "AN_high": [2.1, 2.6, 3.2, 3.8],
            "AN_low": [1.0, 1.2, 1.8, 2.0],
        }
    )

    result = compare_feature_regimes(
        frame,
        "AN",
        "Aa_20yr",
        ["Elevation", "LST"],
        group="environment_group",
        model_families=["ols"],
        pairs=pairs,
    )

    apparent = result.loc[result["evaluation_set"].eq("apparent")]
    nonpaired = apparent.loc[~apparent["regime"].eq("paired_difference")]
    assert set(nonpaired["regime"]) == {
        "exposure_only",
        "environment_only",
        "environment_plus_exposure",
        "group_plus_exposure",
    }
    assert nonpaired["n"].nunique() == 1
    env_r2 = apparent.loc[apparent["regime"].eq("environment_only"), "r_squared"].iloc[0]
    full_r2 = apparent.loc[apparent["regime"].eq("environment_plus_exposure"), "r_squared"].iloc[0]
    delta = apparent.loc[apparent["regime"].eq("environment_plus_exposure"), "delta_r2"].iloc[0]
    assert delta == pytest.approx(compute_delta_r2(full_r2, env_r2))
    assert delta == pytest.approx(full_r2 - env_r2)
    assert apparent.loc[apparent["regime"].eq("paired_difference"), "n"].iloc[0] == 4


def test_delta_r2_is_full_minus_environment_only():
    compute_delta_r2 = _api("regression_search", "compute_delta_r2")
    assert compute_delta_r2(0.72, 0.31) == pytest.approx(0.41)
    with pytest.raises(ValueError, match="finite"):
        compute_delta_r2(np.nan, 0.2)


def test_validation_selected_test_result_records_exposure_effect():
    fit_selected_model_on_test = _api("regression_search", "fit_selected_model_on_test")
    frame = _frame()

    result = fit_selected_model_on_test(
        frame,
        "AN",
        "Aa_20yr",
        ["Elevation", "LST"],
        "ols",
    )

    assert result["status"] == "ok"
    assert result["coefficient"] > 0
    assert result["direction"] == "positive"
