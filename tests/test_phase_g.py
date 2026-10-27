from __future__ import annotations

import numpy as np
import pandas as pd


def test_pair_associations_report_direction_and_block_bootstrap_interval():
    from experiments.oracle_exploration.phase_g import summarize_pair_associations

    frame = pd.DataFrame(
        {
            "method": ["M4"] * 8,
            "exposure": ["Aa_20yr"] * 8,
            "spatial_block_id": ["a", "a", "b", "b", "c", "c", "d", "d"],
            "erosion_high": [4, 5, 8, 12, 16, 22, 30, 40],
            "erosion_low": [0, 1, 2, 3, 4, 5, 6, 7],
            "high_AN": [10, 9, 8, 7, 6, 5, 4, 3],
            "low_AN": [11] * 8,
        }
    )

    result = summarize_pair_associations(
        frame, outcomes=["AN"], bootstrap_replicates=40, seed=7
    )

    overall = result.loc[result["subset"].eq("all")].iloc[0]
    assert overall["n_pairs"] == 8
    assert overall["mean_delta_y"] < 0
    assert overall["spearman_rho"] < 0
    assert overall["kendall_tau"] < 0
    assert overall["negative_delta_y_fraction"] == 1.0
    assert overall["bootstrap_mean_ci_low"] <= overall["mean_delta_y"]
    assert overall["mean_delta_y"] <= overall["bootstrap_mean_ci_high"]
    assert overall["spatial_blocks"] == 4


def test_pair_association_strata_separate_zero_and_positive_low_erosion():
    from experiments.oracle_exploration.phase_g import summarize_pair_associations

    frame = pd.DataFrame(
        {
            "method": ["M4"] * 6,
            "exposure": ["Aa_20yr"] * 6,
            "spatial_block_id": ["a", "a", "b", "b", "c", "c"],
            "erosion_high": [3, 5, 8, 9, 6, 7],
            "erosion_low": [0, 1, 0, 2, 0, 4],
            "high_AN": [1, 1, 2, 3, 4, 4],
            "low_AN": [2, 2, 3, 4, 5, 5],
        }
    )

    result = summarize_pair_associations(
        frame, outcomes=["AN"], bootstrap_replicates=20, seed=13
    )

    assert {"all", "low_zero", "both_positive"}.issubset(set(result["subset"]))
    low_zero = result.loc[result["subset"].eq("low_zero")].iloc[0]
    both_positive = result.loc[result["subset"].eq("both_positive")].iloc[0]
    assert low_zero["n_pairs"] == 3
    assert both_positive["n_pairs"] == 3


def test_phase_g_model_boards_keep_apparent_negative_and_spatial_selection_separate():
    from experiments.oracle_exploration.phase_g import rank_phase_g_models

    candidates = pd.DataFrame(
        {
            "candidate_id": ["pos", "neg_high_r2", "neg_strong", "spatial_a", "spatial_b"],
            "outcome": ["AN"] * 5,
            "exposure": ["Aa_20yr"] * 5,
            "evaluation_set": ["apparent", "apparent", "apparent", "test", "test"],
            "selection_mode": ["oracle"] * 3 + ["validation_selected"] * 2,
            "direction": ["positive", "negative", "negative", "negative", "positive"],
            "r_squared": [0.9, 0.7, 0.2, 0.3, 0.8],
            "validation_r2": [np.nan, np.nan, np.nan, 0.55, 0.4],
            "standardized_effect": [0.4, -0.2, -0.9, -0.2, 0.1],
            "spearman_rho": [0.5, -0.3, -0.8, -0.2, 0.2],
            "negative_delta_y_fraction": [0.2, 0.7, 0.9, 0.7, 0.3],
            "n": [1000, 900, 500, 300, 300],
            "max_abs_matched_smd": [0.05, 0.05, 0.08, 0.06, 0.06],
        }
    )

    boards = rank_phase_g_models(candidates)

    assert boards["highest_apparent_r2"].iloc[0]["candidate_id"] == "pos"
    assert boards["highest_negative_apparent_r2"].iloc[0]["candidate_id"] == "neg_high_r2"
    assert boards["strongest_negative_association"].iloc[0]["candidate_id"] == "neg_strong"
    assert boards["validation_selected_spatial"].iloc[0]["candidate_id"] == "spatial_a"
    assert boards["validation_selected_spatial"].iloc[0]["r_squared"] == 0.3


def test_hierarchical_groups_are_deterministic_and_stratified_by_soil():
    from experiments.oracle_exploration.phase_g import fit_hierarchical_group_labels

    frame = pd.DataFrame(
        {
            "SoilType": [26] * 6 + [27] * 6,
            "Elevation": [1, 2, 3, 8, 9, 10] * 2,
            "LST": [1, 2, 3, 8, 9, 10] * 2,
        }
    )
    labels = fit_hierarchical_group_labels(
        frame, ["Elevation", "LST"], n_clusters=2, max_fit_rows=12, seed=11
    )
    repeated = fit_hierarchical_group_labels(
        frame, ["Elevation", "LST"], n_clusters=2, max_fit_rows=12, seed=11
    )

    pd.testing.assert_series_equal(labels, repeated)
    assert labels.notna().all()
    assert labels.nunique() == 4
    assert labels.str.startswith(("26:", "27:")).all()


def test_repeated_pair_spatial_selection_keeps_only_m3_m4_base_p1_rows():
    from experiments.oracle_exploration.phase_g import select_pair_spatial_p1_rows

    spatial = pd.DataFrame(
        [
            {"method": "M0", "evaluation_set": "test", "validation_model": "P1_intercept_ols", "environment_group": np.nan, "exposure": "Aa_20yr"},
            {"method": "M3", "evaluation_set": "test", "validation_model": "P1_intercept_ols", "environment_group": np.nan, "exposure": "Aa_20yr"},
            {"method": "M3", "evaluation_set": "test", "validation_model": "P5_natural_spline_diagnostic", "environment_group": np.nan, "exposure": "Aa_20yr"},
            {"method": "M3", "evaluation_set": "test", "validation_model": "P1_environment_group_stratum", "environment_group": 2600, "exposure": "Aa_20yr"},
            {"method": "M4", "evaluation_set": "validation", "validation_model": "P1_intercept_ols", "environment_group": np.nan, "exposure": "Aa_2020"},
            {"method": "M4", "evaluation_set": "train", "validation_model": "P1_intercept_ols", "environment_group": np.nan, "exposure": "Aa_20yr"},
        ]
    )

    selected = select_pair_spatial_p1_rows(spatial)

    assert len(selected) == 2
    assert set(zip(selected["method"], selected["exposure"], selected["evaluation_set"], strict=True)) == {
        ("M3", "Aa_20yr", "test"),
        ("M4", "Aa_2020", "validation"),
    }


def test_pair_spatial_summary_uses_training_slope_direction_and_test_scores():
    from experiments.oracle_exploration.phase_g import summarize_pair_spatial_validation

    spatial = pd.DataFrame(
        [
            {"method": "M3", "exposure": "Aa_20yr", "outcome": "AN", "evaluation_set": "test", "seed": 1, "r_squared": 0.2, "beta_log1p_erosion_train": -0.5, "direction_train": "negative"},
            {"method": "M3", "exposure": "Aa_20yr", "outcome": "AN", "evaluation_set": "test", "seed": 2, "r_squared": 0.4, "beta_log1p_erosion_train": -0.3, "direction_train": "negative"},
            {"method": "M4", "exposure": "Aa_20yr", "outcome": "AN", "evaluation_set": "test", "seed": 1, "r_squared": 0.1, "beta_log1p_erosion_train": 0.4, "direction_train": "positive"},
            {"method": "M4", "exposure": "Aa_20yr", "outcome": "AN", "evaluation_set": "test", "seed": 2, "r_squared": 0.3, "beta_log1p_erosion_train": -0.2, "direction_train": "negative"},
        ]
    )

    summary = summarize_pair_spatial_validation(spatial)
    m3 = summary.loc[summary["method"].eq("M3")].iloc[0]
    m4 = summary.loc[summary["method"].eq("M4")].iloc[0]

    assert np.isclose(m3["median_spatial_r2"], 0.3)
    assert np.isclose(m3["median_training_slope"], -0.4)
    assert m3["training_negative_direction_fraction"] == 1.0
    assert m3["spatial_seeds"] == 2
    assert m4["training_negative_direction_fraction"] == 0.5
