from __future__ import annotations

import numpy as np
import pandas as pd


def _paired_frame(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(19)
    x = np.tile(np.linspace(0.1, 1.4, n // 2), 2)
    group = np.repeat(["low_env", "high_env"], n // 2)
    slopes = np.where(group == "low_env", -1.5, 1.2)
    noise = rng.normal(0, 0.03, n)
    high_erosion = np.expm1(x + 0.2)
    low_erosion = np.expm1(np.full(n, 0.2))
    delta = slopes * x + noise
    return pd.DataFrame(
        {
            "pair_id": [f"p{i}" for i in range(n)],
            "method": "M4",
            "exposure": "Aa_20yr",
            "spatial_block_id": [f"b{i % 20}" for i in range(n)],
            "high_erosion_cell_id": [f"h{i}" for i in range(n)],
            "low_erosion_cell_id": [f"l{i}" for i in range(n)],
            "erosion_high": high_erosion,
            "erosion_low": low_erosion,
            "high_AN": 10 + delta,
            "low_AN": 10.0,
            "test_group": group,
        }
    )


def test_pixel_candidate_scope_excludes_water_soil_from_primary_rankings():
    from experiments.oracle_exploration.phase_g_completion import mark_pixel_candidate_scope

    candidates = pd.DataFrame(
        {
            "candidate_id": [
                "subgroup_Aa_20yr_pH_soil_85",
                "subgroup_Aa_20yr_OM_soil_27",
                "full_Aa_20yr_SHI_candidate_V5",
            ],
            "analysis_unit": ["pixel_level"] * 3,
        }
    )

    scoped = mark_pixel_candidate_scope(candidates, primary_soils=[26, 27, 42])

    assert scoped["in_primary_cropland_scope"].tolist() == [False, True, True]
    assert scoped.loc[0, "scope_status"] == "excluded_soil_85_water"


def test_pair_model_search_records_full_sample_metrics_and_direction():
    from experiments.oracle_exploration.phase_g_completion import search_paired_response_models

    pairs = _paired_frame(40)
    pairs["high_AN"] = pairs["low_AN"] - 2 * (
        np.log1p(pairs["erosion_high"]) - np.log1p(pairs["erosion_low"])
    )

    results = search_paired_response_models(
        pairs,
        outcomes=["AN"],
        model_families=["ols", "quadratic"],
        seed=23,
    )

    assert set(results["model_family"]) == {"ols", "quadratic"}
    assert results["fit_scope"].eq("full_matched_pair_sample").all()
    assert results["analysis_unit"].eq("matched_pair_difference").all()
    assert results["n_pairs"].eq(40).all()
    assert results["direction"].eq("negative").all()
    assert results["r_squared"].between(0.99, 1.0).all()


def test_group_slope_interaction_detects_different_directions():
    from experiments.oracle_exploration.phase_g_completion import compare_group_slopes

    pairs = _paired_frame(80)
    tests, slopes = compare_group_slopes(
        pairs,
        outcomes=["AN"],
        group_columns=["test_group"],
        min_pairs=20,
        min_spatial_blocks=5,
    )

    assert len(tests) == 1
    assert tests.loc[0, "status"] == "ok"
    assert tests.loc[0, "interaction_p_value"] < 0.05
    estimates = slopes.set_index("group_label")["slope_delta_y_per_delta_log1p_erosion"]
    assert estimates["low_env"] < 0
    assert estimates["high_env"] > 0


def test_spatial_pair_validation_tests_only_validation_selected_models():
    from experiments.oracle_exploration.phase_g_completion import validate_pair_models_spatial

    pairs = _paired_frame(160)
    trials, selected = validate_pair_models_spatial(
        pairs,
        outcomes=["AN"],
        model_families=["ols", "quadratic"],
        seeds=[31, 32],
        block_size_m=5000,
        min_train_pairs=10,
    )

    assert set(trials["model_family"]) == {"ols", "quadratic"}
    assert trials["test_r_squared"].isna().all()
    assert set(selected["selection_type"]) <= {
        "highest_validation_r2",
        "highest_negative_validation_r2",
    }
    assert selected["test_r_squared"].notna().all()


def test_wald_probability_underflow_keeps_a_positive_probability_and_log_value():
    from experiments.oracle_exploration.phase_g_completion import _probability_from_log

    probability = _probability_from_log(-1000.0)

    assert probability == np.nextafter(0.0, 1.0)


def test_extreme_wald_tail_keeps_a_finite_log_probability():
    from scipy.special import gammaln

    from experiments.oracle_exploration.phase_g_completion import _chi2_logsf

    statistic = 2000.0
    degrees = 20
    log_probability = _chi2_logsf(statistic, degrees)
    x = statistic / 2
    shape = degrees / 2
    tail_series = sum(
        np.prod([shape - offset for offset in range(1, order + 1)]) / x**order
        for order in range(int(shape))
    )
    expected = (shape - 1) * np.log(x) - x - gammaln(shape) + np.log(tail_series)

    assert np.isfinite(log_probability)
    assert np.isclose(log_probability, expected, rtol=0, atol=1e-12)
