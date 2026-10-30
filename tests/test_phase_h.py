from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest


def _config():
    return {
        "phase": "H",
        "mode": "EXPLORATORY_ORACLE",
        "outcome_guided_selection": True,
        "input_run_id": "aligned_run",
        "phase_f_run_id": "phase_f_run",
        "phase_g_run_id": "phase_g_run",
        "output_root": "outputs/exploratory_oracle",
        "run_id": "phase_h_test",
        "seed": 20261028,
        "exposures": ["Aa_20yr", "Aa_2020"],
        "outcomes": ["AN", "OM", "CEC", "pH", "SW", "BD", "MBC", "RESP", "SHI_candidate"],
        "primary_soil_codes": [26, 27, 42],
    }


def _write_phase_h_inputs(root):
    aligned = root / "outputs" / "aligned_run" / "databases"
    aligned.mkdir(parents=True)
    environment = pd.DataFrame(
        {
            "cell_id": [10, 11], "row": [1, 1], "col": [1, 2],
            "x": [100.0, 130.0], "y": [200.0, 200.0],
            "Aa_20yr": [1.0, 2.0], "Aa_2020": [0.5, 1.5], "valid_cell": [True, True],
        }
    )
    health = pd.DataFrame(
        {
            "cell_id": [10, 11], "row": [1, 1], "col": [1, 2],
            "x": [100.0, 130.0], "y": [200.0, 200.0],
            "AN": [1.0, 2.0], "OM": [3.0, 4.0], "CEC": [5.0, 6.0],
            "pH": [6.0, 7.0], "SW": [0.2, 0.3], "BD": [1.1, 1.2],
            "MBC": [9.0, 10.0], "RESP": [2.0, 3.0],
            "SHI_candidate": [0.4, 0.5], "valid_cell": [True, True],
        }
    )
    environment.to_parquet(aligned / "environment_erosion.parquet", index=False)
    health.to_parquet(aligned / "health.parquet", index=False)

    pair_dir = root / "outputs" / "exploratory_oracle" / "phase_f_run" / "pairs"
    label_dir = root / "outputs" / "exploratory_oracle" / "phase_g_run" / "tables"
    pair_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    pairs = pd.DataFrame(
        {
            "pair_id": ["p1", "p2"], "high_erosion_cell_id": [10, 10],
            "low_erosion_cell_id": [11, 11], "method": ["M3", "M4"],
            "exposure": ["Aa_20yr", "Aa_2020"], "soil_type": [26, 26],
        }
    )
    labels = pd.DataFrame(
        {
            "pair_id": ["p1", "p2"], "high_erosion_cell_id": [10, 10],
            "kmeans_terrain_k4": ["s26_c1", "s26_c1"],
        }
    )
    pairs.to_parquet(pair_dir / "full_sample_pairs.parquet", index=False)
    labels.to_parquet(label_dir / "environment_group_labels.parquet", index=False)


def test_merge_phase_h_labels_is_one_to_one_and_keeps_repeated_high_pixel_memberships():
    from experiments.oracle_exploration.phase_h import merge_phase_h_pair_labels

    pairs = pd.DataFrame(
        {
            "pair_id": ["p1", "p2", "p3"],
            "high_erosion_cell_id": [10, 10, 12],
            "method": ["M3", "M4", "M4"],
        }
    )
    labels = pd.DataFrame(
        {
            "pair_id": ["p1", "p2", "p3"],
            "high_erosion_cell_id": [10, 10, 12],
            "kmeans_terrain_k4": ["s26_c1", "s26_c2", "s27_c1"],
        }
    )

    joined = merge_phase_h_pair_labels(pairs, labels)

    assert len(joined) == 3
    assert joined["pair_id"].tolist() == ["p1", "p2", "p3"]
    assert joined["high_erosion_cell_id"].tolist() == [10, 10, 12]
    assert joined["kmeans_terrain_k4"].tolist() == ["s26_c1", "s26_c2", "s27_c1"]


def test_merge_phase_h_labels_rejects_mismatched_pair_membership():
    from experiments.oracle_exploration.phase_h import merge_phase_h_pair_labels

    pairs = pd.DataFrame({"pair_id": ["p1"], "high_erosion_cell_id": [10]})
    labels = pd.DataFrame({"pair_id": ["p1"], "high_erosion_cell_id": [11], "group": ["a"]})

    with pytest.raises(ValueError, match="high_erosion_cell_id"):
        merge_phase_h_pair_labels(pairs, labels)


def test_phase_h_config_requires_oracle_mode_and_safe_output_root():
    from experiments.oracle_exploration.phase_h import validate_phase_h_config

    config = validate_phase_h_config(_config())
    assert config["outcome_guided_selection"] is True
    assert config["primary_soil_codes"] == [26, 27, 42]

    unsafe = _config()
    unsafe["output_root"] = "outputs/exploratory_oracle/../../formal"
    with pytest.raises(ValueError, match="output_root"):
        validate_phase_h_config(unsafe)


def test_load_phase_h_inputs_joins_pairs_labels_without_deduplicating_memberships(tmp_path):
    from experiments.oracle_exploration.phase_h import load_phase_h_inputs

    _write_phase_h_inputs(tmp_path)
    loaded = load_phase_h_inputs(tmp_path, _config())

    assert len(loaded.cells) == 2
    assert len(loaded.pairs) == 2
    assert loaded.pairs["high_erosion_cell_id"].tolist() == [10, 10]
    assert loaded.pairs["kmeans_terrain_k4"].tolist() == ["s26_c1", "s26_c1"]
    assert np.isclose(loaded.cells.loc[0, "Aa_20yr"], 1.0)


def test_prepare_phase_h_run_records_input_inventory_and_preserves_existing_run(tmp_path):
    from experiments.oracle_exploration.phase_h import prepare_phase_h_run

    _write_phase_h_inputs(tmp_path)
    run_dir, _ = prepare_phase_h_run(tmp_path, _config())

    inventory = pd.read_csv(run_dir / "tables" / "input_inventory.csv")
    provenance = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
    assert set(inventory["role"]) == {
        "aligned_environment", "aligned_health", "phase_f_pairs", "phase_g_labels"
    }
    assert set(inventory["row_count"]) == {2}
    assert provenance["mode"] == "EXPLORATORY_ORACLE"
    assert (run_dir / "checkpoints").is_dir()

    changed = _config()
    changed["seed"] = 10
    with pytest.raises(FileExistsError, match="already exists"):
        prepare_phase_h_run(tmp_path, changed)


def test_interaction_estimability_rejects_singular_covariance_and_suppresses_p_value():
    from experiments.oracle_exploration.phase_h import assess_interaction_estimability

    result = assess_interaction_estimability(
        np.array([3.0, 0.1]),
        np.diag([1.0, 1e-19]),
        design_rank=6,
        design_columns=6,
        spatial_blocks=33,
        group_block_counts={"a": 12, "b": 13, "c": 11},
    )

    assert result["status"] == "unreliable_covariance_rank_deficient"
    assert result["interaction_df"] == 2
    assert result["covariance_rank"] == 1
    assert np.isnan(result["interaction_p_value"])
    assert result["tolerance_sensitivity_ratio"] > 10


def test_interaction_estimability_allows_stable_low_df_test():
    from experiments.oracle_exploration.phase_h import assess_interaction_estimability

    result = assess_interaction_estimability(
        np.array([0.2, 0.5]),
        np.diag([1.0, 1.0]),
        design_rank=6,
        design_columns=6,
        spatial_blocks=33,
        group_block_counts={"a": 12, "b": 13, "c": 11},
    )

    assert result["status"] == "reliable"
    assert result["interaction_p_value"] >= 0
    assert result["interaction_denominator_df"] == 32


def test_interaction_estimability_rejects_ill_conditioned_design():
    from experiments.oracle_exploration.phase_h import assess_interaction_estimability

    result = assess_interaction_estimability(
        np.array([0.2, 0.5]),
        np.eye(2),
        design_rank=6,
        design_columns=6,
        design_condition_number=1e12,
        spatial_blocks=33,
        group_block_counts={"a": 12, "b": 13, "c": 11},
    )

    assert result["status"] == "unreliable_design_condition"
    assert np.isnan(result["interaction_p_value"])


def test_group_interaction_reports_spatial_block_bootstrap_slopes():
    from experiments.oracle_exploration.phase_h import fit_group_interaction

    rng = np.random.default_rng(82)
    rows = []
    for block in range(20):
        intercept = rng.normal(0, 0.2)
        for group, slope in (("A", -1.0), ("B", 1.0)):
            x = np.linspace(0.0, 1.0, 8)
            y = intercept + slope * x + rng.normal(0, 0.03, len(x))
            rows.extend(
                {"x": x_i, "y": y_i, "group": group, "block": f"b{block}"}
                for x_i, y_i in zip(x, y, strict=True)
            )
    data = pd.DataFrame(rows)

    test, slopes = fit_group_interaction(
        data,
        x_column="x",
        y_column="y",
        group_column="group",
        block_column="block",
        bootstrap_replicates=99,
        seed=83,
    )

    assert test["status"] == "reliable"
    estimates = slopes.set_index("group_label")
    assert estimates.loc["A", "slope"] < 0 < estimates.loc["B", "slope"]
    assert estimates["block_bootstrap_ci_low"].notna().all()
    assert estimates["spatial_blocks"].eq(20).all()


def test_h1_revision_tables_compare_old_results_to_reliable_inference():
    from experiments.oracle_exploration.phase_h import build_h1_inference_tables

    rng = np.random.default_rng(91)
    rows = []
    for block in range(20):
        for group, slope in (("A", -1.0), ("B", 0.8)):
            for x in np.linspace(0.1, 1.0, 8):
                delta = slope * x + rng.normal(0, 0.05)
                rows.append(
                    {
                        "method": "M4", "exposure": "Aa_20yr", "pair_id": f"{block}_{group}_{x}",
                        "spatial_block_id": f"b{block}", "erosion_high": np.expm1(x + 0.3),
                        "erosion_low": np.expm1(0.3), "high_AN": 10 + delta, "low_AN": 10.0,
                        "test_group": group,
                    }
                )
    pairs = pd.DataFrame(rows)
    legacy = pd.DataFrame(
        {
            "method": ["M4"], "exposure": ["Aa_20yr"], "outcome": ["AN"],
            "group_column": ["test_group"], "interaction_p_value": [1e-20],
            "interaction_q_value_global": [1e-18], "interaction_df": [1],
        }
    )

    diagnostics, revised, slopes, comparison = build_h1_inference_tables(
        pairs,
        legacy,
        outcomes=["AN"],
        group_columns=["test_group"],
        bootstrap_group_columns=["test_group"],
        bootstrap_replicates=99,
        seed=92,
    )

    assert len(diagnostics) == len(revised) == len(comparison) == 1
    assert revised.loc[0, "status"] == "reliable"
    assert comparison.loc[0, "legacy_interaction_p_value"] == 1e-20
    assert set(slopes["group_label"]) == {"A", "B"}
    assert slopes["block_bootstrap_ci_low"].notna().all()


def test_phase_h_candidate_design_is_bounded_and_covers_configured_match_domains():
    from experiments.oracle_exploration.phase_h_search import build_phase_h_candidate_designs

    config = _config()
    config["matching_search"] = {
        "source_methods": ["M3", "M4"],
        "environment_vectors": {"V1": ["Elevation", "Cur", "TWI", "LST"]},
        "environment_calipers": [0.20, 0.35, 0.50, 0.75],
        "geographic_radii_m": [300, 500, 1000, 1500],
        "erosion_rank_gaps": [0.20, 0.30, 0.40, 0.50],
        "stage1_max_candidates_per_exposure": 24,
    }

    first = build_phase_h_candidate_designs(config, "Aa_20yr", seed=33)
    repeated = build_phase_h_candidate_designs(config, "Aa_20yr", seed=33)

    assert len(first) == 24
    assert first == repeated
    assert {item["reference_method"] for item in first} == {"M3", "M4"}
    assert {item["cluster_method"] for item in first} == {"none", "kmeans", "agglomerative"}
    assert {item["cluster_count"] for item in first if item["cluster_count"]} == {4, 6, 8, 10, 12}
    assert {item["environment_caliper"] for item in first} == {0.20, 0.35, 0.50, 0.75}
    assert {item["geographic_radius_m"] for item in first} == {300, 500, 1000, 1500}
    assert {item["erosion_rank_gap"] for item in first} == {0.20, 0.30, 0.40, 0.50}
    signature_columns = (
        "vector_name", "reference_method", "distance_method", "cluster_method", "cluster_count",
        "environment_caliper", "geographic_radius_m", "erosion_rank_gap", "exposure_rule",
    )
    assert len({tuple(item[column] for column in signature_columns) for item in first}) == len(first)


def test_phase_h_subsets_keep_environment_retention_and_oracle_residual_separate():
    from experiments.oracle_exploration.phase_h_search import build_phase_h_pair_subsets

    pairs = pd.DataFrame(
        {
            "pair_id": [f"p{i}" for i in range(10)],
            "environmental_distance": np.arange(10, dtype=float),
            "delta_log1p_Aa": np.arange(1, 11, dtype=float),
            "delta_AN": np.arange(-5, 5, dtype=float),
            "both_positive": [True, False] * 5,
            "zero_positive": [False, True] * 5,
            "high_erosion_gap_q4": [False] * 7 + [True] * 3,
            "gradient_TWI": ["low"] * 5 + ["high"] * 5,
        }
    )

    subsets = build_phase_h_pair_subsets(
        pairs,
        outcome="AN",
        retention_fractions=[1.0, 0.5, 0.2],
        include_oracle_residual=True,
    )

    assert len(subsets["environment_distance_low_50pct"]) == 5
    assert subsets["environment_distance_low_50pct"]["environmental_distance"].max() == 4
    assert "oracle_extreme_residual_bottom_20pct" in subsets
    assert subsets["oracle_extreme_residual_bottom_20pct"].attrs["outcome_guided_selection"] is True
    assert subsets["all"].attrs["outcome_guided_selection"] is False


def test_phase_h_leaderboards_separate_pixel_pair_estimands_and_negative_directions():
    from experiments.oracle_exploration.phase_h_search import build_phase_h_leaderboards

    trials = pd.DataFrame(
        [
            {"estimand": "matched_pixel", "exposure": "Aa_20yr", "outcome": "AN", "r_squared": 0.80, "direction": "positive", "spearman_rho": 0.80, "candidate_id": "p1", "model_family": "random_forest", "n": 100},
            {"estimand": "matched_pixel", "exposure": "Aa_20yr", "outcome": "AN", "r_squared": 0.61, "direction": "negative", "spearman_rho": -0.61, "candidate_id": "p2", "model_family": "natural_spline", "n": 95},
            {"estimand": "matched_pair", "exposure": "Aa_20yr", "outcome": "AN", "r_squared": 0.52, "direction": "negative", "spearman_rho": -0.70, "candidate_id": "q1", "model_family": "ols", "n": 60},
        ]
    )

    tables = build_phase_h_leaderboards(trials)

    best_apparent = tables["leaderboard_1_top_apparent_r2"]
    best_negative = tables["leaderboard_2_highest_negative_direction_r2"]
    assert set(best_apparent["estimand"]) == {"matched_pixel", "matched_pair"}
    pixel_negative = best_negative.loc[best_negative["estimand"].eq("matched_pixel")].iloc[0]
    assert pixel_negative["candidate_id"] == "p2"
    assert pixel_negative["direction"] == "negative"
    assert tables["leaderboard_4_highest_matched_pixel_r2"]["estimand"].eq("matched_pixel").all()


def test_phase_h_model_trial_uses_centered_r2_and_exposure_support_direction():
    from experiments.oracle_exploration.phase_h_search import _model_trial

    exposure = np.linspace(0.0, 8.0, 40)
    frame = pd.DataFrame({"Aa_20yr": exposure, "AN": 12.0 - 1.7 * exposure, "Elevation": exposure / 2})
    config = {
        "seed": 71,
        "models": {"random_seed": 72},
        "response_curves": {"curve_points": 41},
    }
    candidate = {
        "candidate_id": "candidate_a", "exposure": "Aa_20yr", "reference_method": "M4",
        "environment_variables": ["Elevation"], "vector_name": "V1", "distance_method": "direct_nearest_neighbor",
        "cluster_method": "none", "cluster_count": 0, "environment_caliper": 0.35,
        "geographic_radius_m": 1000, "erosion_rank_gap": 0.3, "exposure_rule": "within_group_rank",
    }

    trial = _model_trial(
        frame,
        config=config,
        candidate=candidate,
        estimand="matched_pixel",
        outcome="AN",
        subset_name="all",
        model_family="ols",
        fit_scope="exposure_only",
        stage="stage2_full_sample",
    )

    assert trial["status"] == "ok"
    assert trial["r_squared"] > 0.99
    assert trial["direction"] == "negative"
    assert trial["n"] == trial["n_evaluation"] == len(frame)
    assert trial["r2_definition"] == "1 - SSE / sum((y - mean(y))^2)"


def test_phase_h_leaderboards_exclude_unmatched_historical_oracle_rows():
    from experiments.oracle_exploration.phase_h_search import build_phase_h_leaderboards

    matched = {
        "estimand": "matched_pixel", "exposure": "Aa_20yr", "outcome": "AN", "r_squared": 0.3,
        "direction": "negative", "spearman_rho": -0.4, "candidate_id": "matched", "model_family": "ols", "n": 20,
    }
    historical = {**matched, "estimand": "unmatched_pixel", "r_squared": 0.99, "candidate_id": "historical_oracle"}

    leaderboard = build_phase_h_leaderboards(pd.DataFrame([matched, historical]))

    assert "historical_oracle" not in set(leaderboard["leaderboard_1_top_apparent_r2"]["candidate_id"])
    assert leaderboard["leaderboard_7_all_candidate_ranked_audit"]["analysis_population"].eq("phase_h_matched_only").all()


def test_phase_h_stage1_candidate_selection_uses_best_response_family_per_unit():
    from experiments.oracle_exploration.phase_h_search import _stage1_summary

    trials = pd.DataFrame(
        [
            {"candidate_id": "c1", "outcome": "AN", "estimand": "matched_pixel", "model_family": "ols", "r_squared": 0.2, "direction": "negative", "n": 30, "status": "ok"},
            {"candidate_id": "c1", "outcome": "AN", "estimand": "matched_pixel", "model_family": "natural_spline", "r_squared": 0.7, "direction": "positive", "n": 30, "status": "ok"},
            {"candidate_id": "c1", "outcome": "AN", "estimand": "matched_pair", "model_family": "ols", "r_squared": 0.4, "direction": "negative", "n": 15, "status": "ok"},
            {"candidate_id": "c1", "outcome": "AN", "estimand": "matched_pair", "model_family": "quadratic", "r_squared": 0.6, "direction": "mixed", "n": 15, "status": "ok"},
        ]
    )

    summary = _stage1_summary(
        [trials],
        [{"candidate_id": "c1", "exposure": "Aa_20yr", "reference_method": "M4", "n_matched_pairs": 15}],
    )

    assert summary.loc[0, "stage1_mean_best_r2"] == pytest.approx(0.65)
    assert summary.loc[0, "stage1_negative_response_fraction"] == 0.0


def test_phase_h_stage2_model_trials_checkpoint_each_outcome_and_resume(tmp_path):
    from experiments.oracle_exploration.phase_h_search import _candidate_model_trials

    n = 30
    high_ids = np.arange(n)
    low_ids = np.arange(n, 2 * n)
    exposure_high = np.linspace(2.0, 10.0, n)
    exposure_low = np.linspace(0.2, 1.5, n)
    pixel_exposure = np.concatenate([exposure_high, exposure_low])
    health = 12.0 - 0.8 * pixel_exposure + 0.01 * pixel_exposure**2
    cells = pd.DataFrame(
        {
            "cell_id": np.arange(2 * n), "Aa_20yr": pixel_exposure,
            "AN": health, "Elevation": np.linspace(0, 20, 2 * n),
            "x": np.arange(2 * n, dtype=float), "y": np.zeros(2 * n),
        }
    )
    cells = pd.concat(
        [
            cells,
            pd.DataFrame(
                {
                    "cell_id": np.arange(60, 66),
                    "SoilType": [85] * 6,
                    "Elevation": np.linspace(80, 85, 6),
                    "LST": np.linspace(30, 35, 6),
                    "TWI": np.linspace(-1, 1, 6),
                    "AN": np.zeros(6),
                }
            ),
        ],
        ignore_index=True,
    )
    pairs = pd.DataFrame(
        {
            "pair_id": [f"p{i}" for i in range(n)],
            "high_erosion_cell_id": high_ids, "low_erosion_cell_id": low_ids,
            "erosion_high": exposure_high, "erosion_low": exposure_low,
            "environmental_distance": np.linspace(0.01, 0.3, n),
            "geographic_distance": np.full(n, 100.0),
            "environment_group": ["g1"] * (n // 2) + ["g2"] * (n - n // 2),
            "high_AN": health[:n], "low_AN": health[n:],
            "Elevation_high": cells.loc[high_ids, "Elevation"].to_numpy(),
            "Elevation_low": cells.loc[low_ids, "Elevation"].to_numpy(),
        }
    )
    candidate = {
        "candidate_id": "stage2_test", "exposure": "Aa_20yr", "reference_method": "M4",
        "environment_variables": ["Elevation"], "vector_name": "V1", "distance_method": "direct_nearest_neighbor",
        "cluster_method": "kmeans", "cluster_count": 4, "environment_caliper": 0.35,
        "geographic_radius_m": 500, "erosion_rank_gap": 0.3, "exposure_rule": "within_group_rank",
        "stage1_rank": 1,
    }
    config = {
        "seed": 77, "outcomes": ["AN"], "models": {"families": ["ols"], "random_seed": 78},
        "subsets": {"retention_fractions": [1.0, 0.5], "record_extreme_oracle_residual_subset": True},
        "response_curves": {"curve_points": 20},
    }

    first = _candidate_model_trials(cells, pairs, candidate, config, progress_dir=tmp_path / "progress")
    resumed = _candidate_model_trials(cells, pairs, candidate, config, progress_dir=tmp_path / "progress")

    assert len(first) == len(resumed) > 0
    assert all(row["estimand"] in {"matched_pixel", "matched_pair"} for row in first)
    assert {row["model_schedule_version"] for row in first} == {"phase_h_multifamily_bounded_v2"}
    assert (tmp_path / "progress" / "outcome_01.parquet").is_file()
    assert any(row["subset_name"] == "both_positive" for row in first)


def test_phase_h_jonckheere_terpstra_trend_handles_ties_and_direction():
    from experiments.oracle_exploration.phase_h_post import jonckheere_terpstra_trend

    exposure = np.repeat(np.arange(5, dtype=float), 30)
    health = np.repeat(np.arange(5, dtype=float), 30) + np.tile(np.array([0, 0, 1]), 50)

    increasing = jonckheere_terpstra_trend(exposure, health, groups=5)
    decreasing = jonckheere_terpstra_trend(exposure, -health, groups=5)

    assert increasing["status"] == "ok"
    assert increasing["direction"] == "positive"
    assert decreasing["direction"] == "negative"
    assert increasing["p_value_two_sided"] < 0.01
    assert increasing["n_ordered_groups"] == 5


def test_phase_h_nonparametric_tables_keep_pixel_pair_and_paired_tests_separate():
    from experiments.oracle_exploration.phase_h_post import build_nonparametric_tables

    n = 25
    high_id = np.arange(n)
    low_id = np.arange(n, 2 * n)
    high_aa = np.linspace(2.0, 8.0, n)
    low_aa = np.linspace(0.2, 1.0, n)
    all_aa = np.concatenate([high_aa, low_aa])
    all_an = 10.0 - all_aa
    cells = pd.DataFrame({
        "cell_id": np.arange(2 * n), "Aa_20yr": all_aa, "AN": all_an,
    })
    pairs = pd.DataFrame({
        "pair_id": [f"p{i}" for i in range(n)], "candidate_id": "c1", "method": "M4",
        "exposure": "Aa_20yr", "high_erosion_cell_id": high_id, "low_erosion_cell_id": low_id,
        "erosion_high": high_aa, "erosion_low": low_aa,
        "delta_log1p_Aa": np.log1p(high_aa) - np.log1p(low_aa),
        "high_AN": all_an[:n], "low_AN": all_an[n:], "delta_AN": all_an[:n] - all_an[n:],
        "soil_type": 26, "environment_group": "g1",
    })

    tables = build_nonparametric_tables(cells, [("c1", pairs)], outcomes=["AN"], seed=7)
    response = tables["response_summary"]
    paired = tables["paired_difference_summary"].iloc[0]

    assert set(response["estimand"]) == {"matched_pixel", "matched_pair"}
    assert paired["estimand"] == "matched_pair_high_minus_low"
    assert paired["n_pairs"] == n
    assert {"high_erosion", "low_erosion", "paired_difference_high_minus_low"}.issubset(
        set(tables["boxplot_source"]["member_role"])
    )


def _synthetic_h2_tables():
    cells = pd.DataFrame(
        {
            "cell_id": [10, 11, 14, 15, 16],
            "row": [1, 1, 2, 3, 3],
            "col": [1, 2, 1, 1, 2],
            "x": [100.0, 130.0, 100.0, 100.0, 130.0],
            "y": [200.0, 200.0, 170.0, 140.0, 140.0],
            "SoilType": [26, 26, 26, 85, 85],
            "Aa_20yr": [3.0, 0.0, 1.0, 4.0, 0.0],
            "Aa_2020": [2.0, 0.0, 0.5, 4.0, 0.0],
            "Elevation": [10.0, 11.0, 12.0, 20.0, 21.0],
            "AN": [8.0, 12.0, 10.0, 3.0, 4.0],
            "OM": [3.0, 5.0, 4.0, 3.0, 4.0],
            "CEC": [5.0, 7.0, 6.0, 5.0, 7.0],
            "pH": [6.0, 7.0, 6.5, 6.0, 7.0],
            "SW": [0.2, 0.3, 0.25, 0.2, 0.3],
            "BD": [1.1, 1.2, 1.15, 1.1, 1.2],
            "MBC": [9.0, 11.0, 10.0, 9.0, 11.0],
            "RESP": [2.0, 3.0, 2.5, 2.0, 3.0],
            "SHI_candidate": [0.4, 0.5, 0.45, 0.4, 0.5],
        }
    )
    pairs = pd.DataFrame(
        {
            "pair_id": ["p1", "p2", "p3", "water"],
            "method": ["M4"] * 4,
            "exposure": ["Aa_20yr", "Aa_20yr", "Aa_2020", "Aa_20yr"],
            "seed": [1, 1, 1, 1],
            "high_erosion_cell_id": [10, 10, 10, 15],
            "low_erosion_cell_id": [11, 14, 14, 16],
            "erosion_high": [3.0, 3.0, 2.0, 4.0],
            "erosion_low": [0.0, 1.0, 0.5, 0.0],
            "soil_type": [26, 26, 26, 85],
            "environmental_distance": [0.1, 0.2, 0.1, 0.1],
            "geographic_distance": [30.0, 40.0, 30.0, 30.0],
            "spatial_block_id": ["b1", "b2", "b1", "b3"],
            "environment_group": ["g1", "g2", "g1", "water"],
            "soil_type_group": ["soil26"] * 3 + ["soil85"],
            "kmeans_terrain_k4": [1, 2, 1, 0],
            "kmeans_terrain_k6": ["s26_k1", "s26_k2", "s26_k1", "water"],
            "kmeans_terrain_k8": ["s26_k1", "s26_k2", "s26_k1", "water"],
        }
    )
    for outcome in ["SHI_candidate", "AN", "OM", "CEC", "pH", "SW", "BD", "MBC", "RESP"]:
        lookup = cells.set_index("cell_id")[outcome]
        pairs[f"high_{outcome}"] = pairs["high_erosion_cell_id"].map(lookup)
        pairs[f"low_{outcome}"] = pairs["low_erosion_cell_id"].map(lookup)
    return cells, pairs


def test_pair_difference_features_keep_high_minus_low_orientation():
    from experiments.oracle_exploration.phase_h_data import derive_pair_difference_features

    pairs = pd.DataFrame(
        {"erosion_high": [3.0], "erosion_low": [0.0], "high_AN": [8.0], "low_AN": [12.0]}
    )

    result = derive_pair_difference_features(pairs, outcomes=["AN"])

    assert result.loc[0, "Aa_high"] == 3.0
    assert result.loc[0, "Aa_low"] == 0.0
    assert np.isclose(result.loc[0, "delta_log1p_Aa"], np.log1p(3.0))
    assert result.loc[0, "delta_AN"] == -4.0


def test_matched_datasets_keep_memberships_deduplicate_pixel_model_view_and_exclude_water(tmp_path):
    from experiments.oracle_exploration.phase_h_data import build_matched_datasets, write_matched_datasets

    cells, pairs = _synthetic_h2_tables()
    result = build_matched_datasets(
        cells,
        pairs,
        outcomes=["SHI_candidate", "AN", "OM", "CEC", "pH", "SW", "BD", "MBC", "RESP"],
        group_columns=["environment_group", "soil_type_group", "kmeans_terrain_k4"],
    )

    assert len(result.pair_memberships) == 6
    assert len(result.pairs) == 3
    assert result.pairs["pair_id"].is_unique
    assert len(result.pixel_unique) == 5
    assert not result.pixel_unique.duplicated(["method", "exposure", "cell_id"]).any()
    high_20yr = result.pixel_unique.loc[
        result.pixel_unique["exposure"].eq("Aa_20yr") & result.pixel_unique["cell_id"].eq(10)
    ].iloc[0]
    assert high_20yr["membership_count"] == 2
    assert set(
        result.pair_memberships.loc[
            result.pair_memberships["exposure"].eq("Aa_20yr")
            & result.pair_memberships["cell_id"].eq(10),
            "pair_id",
        ]
    ) == {"p1", "p2"}
    assert high_20yr["environment_group_mode"] == "g1"
    assert set(result.pair_memberships["SoilType"].unique()) == {26}
    assert result.pair_subsets["both_positive"]["pair_id"].tolist() == ["p2", "p3"]
    assert result.pair_subsets["zero_positive"]["pair_id"].tolist() == ["p1"]
    assert set(result.pair_subsets["high_erosion_gap_q4"]["pair_id"]) == {"p1", "p3"}
    assert set(result.environment_group_pairs["group_column"]) == {
        "environment_group", "soil_type_group", "kmeans_terrain_k4"
    }
    assert set(result.environment_group_pairs.loc[
        result.environment_group_pairs["pair_id"].eq("p2"), "environment_group_label"
    ]) == {"g2", "soil26", "2"}
    write_matched_datasets(tmp_path, result)
    saved_group_pairs = pd.read_parquet(tmp_path / "pairs" / "matched_pairs_by_environment_group.parquet")
    assert saved_group_pairs["environment_group_label"].map(type).eq(str).all()


def test_response_curve_direction_distinguishes_monotone_mixed_flat_and_unknown():
    from experiments.oracle_exploration.phase_h_metrics import direction_from_curve

    x = np.linspace(0.0, 1.0, 101)
    assert direction_from_curve(x, -x)["direction"] == "negative"
    assert direction_from_curve(x, x)["direction"] == "positive"
    assert direction_from_curve(x, (x - 0.5) ** 2)["direction"] == "mixed"
    assert direction_from_curve(x, np.full_like(x, 4.0))["direction"] == "near_zero"
    assert direction_from_curve(np.array([1.0]), np.array([1.0]))["direction"] == "unknown"


def test_regression_r2_uses_centered_total_sum_of_squares_for_every_model():
    from experiments.oracle_exploration.phase_h_metrics import regression_metrics

    result = regression_metrics(np.array([2.0, 4.0, 6.0]), np.array([3.0, 4.0, 5.0]))

    assert np.isclose(result["r_squared"], 0.75)
    assert np.isclose(result["rmse"], np.sqrt(2.0 / 3.0))
    assert result["r2_definition"] == "1 - SSE / sum((y - mean(y))^2)"


def test_matched_pair_relation_reports_high_minus_low_health_and_exposure_association():
    from experiments.oracle_exploration.phase_h_metrics import matched_pair_relation_summary

    pairs = pd.DataFrame(
        {
            "Aa_high": [3.0, 2.0, 1.0], "Aa_low": [0.0, 0.5, 0.2],
            "high_AN": [4.0, 5.0, 6.0], "low_AN": [5.0, 7.0, 7.0],
        }
    )

    result = matched_pair_relation_summary(pairs, "AN")

    assert result["n_pairs"] == 3
    assert result["negative_delta_health_fraction"] == 1.0
    assert np.isclose(result["mean_delta_health"], -4.0 / 3.0)
    assert np.isfinite(result["spearman_rho"])


def test_spatial_validation_helpers_assign_disjoint_roles_and_enforce_buffers():
    from experiments.oracle_exploration import phase_h_search

    assign = getattr(phase_h_search, "assign_spatial_roles", None)
    enrich = getattr(phase_h_search, "enrich_pair_spatial_coordinates", None)
    filter_buffer = getattr(phase_h_search, "filter_spatial_buffer", None)
    assert callable(assign)
    assert callable(enrich)
    assert callable(filter_buffer)

    blocks = [f"b{i}" for i in range(20)]
    roles = assign(blocks, seed=123)
    assert roles == assign(blocks, seed=123)
    assert set(roles) == set(blocks)
    assert set(roles.values()) == {"train", "validation", "test"}

    cells = pd.DataFrame(
        {
            "cell_id": [1, 2, 3, 4],
            "x": [100.0, 120.0, 5100.0, 200.0],
            "y": [200.0, 220.0, 200.0, 200.0],
        }
    )
    pairs = pd.DataFrame(
        {
            "pair_id": ["same_block", "cross_block"],
            "high_erosion_cell_id": [1, 1],
            "low_erosion_cell_id": [2, 3],
        }
    )
    spatial_pairs = enrich(pairs, cells, block_size_m=5000)
    assert spatial_pairs["same_spatial_block"].tolist() == [True, False]
    assert spatial_pairs.loc[0, "spatial_block_id"] == "0_0"

    train = pd.DataFrame({"x": [0.0, 600.0, 1200.0], "y": [0.0, 0.0, 0.0]})
    protected = pd.DataFrame({"x": [0.0], "y": [0.0]})
    filtered = filter_buffer(train, protected, x_column="x", y_column="y", buffer_m=500)
    assert filtered["x"].tolist() == [600.0, 1200.0]


def test_common_period_pairs_keep_same_endpoints_and_aa_20yr_orientation():
    from experiments.oracle_exploration import phase_h_search

    build_common = getattr(phase_h_search, "common_pair_period_sample", None)
    assert callable(build_common)

    cells = pd.DataFrame(
        {
            "cell_id": [1, 2, 3, 4],
            "x": [100.0, 130.0, 160.0, 190.0],
            "y": [200.0] * 4,
            "Aa_20yr": [1.0, 5.0, 6.0, 2.0],
            "Aa_2020": [7.0, 2.0, 3.0, 1.0],
            "AN": [10.0, 8.0, 9.0, 12.0],
        }
    )
    pairs_20yr = pd.DataFrame(
        {
            "pair_id": ["p20a", "p20b", "only_20"],
            "high_erosion_cell_id": [2, 3, 4],
            "low_erosion_cell_id": [1, 4, 1],
        }
    )
    pairs_2020 = pd.DataFrame(
        {
            "pair_id": ["p20xxa", "p20xxb", "only_2020"],
            "high_erosion_cell_id": [1, 3, 4],
            "low_erosion_cell_id": [2, 4, 2],
        }
    )

    common = build_common(pairs_20yr, pairs_2020, cells, outcome="AN")

    assert len(common) == 2
    assert common["canonical_pair_key"].tolist() == ["1|2", "3|4"]
    assert common["high_erosion_cell_id"].tolist() == [2, 3]
    assert common["delta_AN"].tolist() == [-2.0, -3.0]
    assert common.loc[0, "delta_log1p_Aa_2020"] < 0
    assert common.loc[0, "delta_log1p_Aa_20yr"] > 0


def test_spatial_split_uses_whole_blocks_and_records_buffered_training_population():
    from experiments.oracle_exploration import phase_h_search

    prepare = getattr(phase_h_search, "prepare_spatial_split", None)
    assert callable(prepare)
    frame = pd.DataFrame(
        {
            "cell_id": list(range(20)),
            "spatial_block_id": [f"b{i}" for i in range(20)],
            "x": np.arange(20, dtype=float) * 5000.0,
            "y": np.zeros(20),
            "health": np.arange(20, dtype=float),
            "erosion": np.arange(20, dtype=float) / 2.0,
        }
    )

    result = prepare(
        frame,
        unit_id_column="cell_id",
        block_column="spatial_block_id",
        x_column="x",
        y_column="y",
        seed=46,
        buffer_m=300,
    )

    roles = result["role_assignment"]
    assert set(roles.values()) == {"train", "validation", "test"}
    role_blocks = {
        role: set(frame.loc[frame["spatial_block_id"].map(roles).eq(role), "spatial_block_id"])
        for role in ("train", "validation", "test")
    }
    assert role_blocks["train"].isdisjoint(role_blocks["validation"])
    assert role_blocks["train"].isdisjoint(role_blocks["test"])
    assert role_blocks["validation"].isdisjoint(role_blocks["test"])
    assert result["details"]["n_train_before_buffer"] >= len(result["train"])
    assert result["details"]["n_validation"] > 0
    assert result["details"]["n_test"] > 0


def test_spatial_split_scoring_fits_on_train_and_scores_validation_and_test():
    import importlib
    import importlib.util

    module_name = "experiments.oracle_exploration.phase_h_validation"
    assert importlib.util.find_spec(module_name) is not None
    validation = importlib.import_module(module_name)
    scorer = getattr(validation, "score_spatial_split", None)
    assert callable(scorer)

    x_train = np.linspace(0.0, 1.0, 30)
    x_validation = np.linspace(0.1, 0.9, 12)
    x_test = np.linspace(0.05, 0.95, 12)
    split = {
        "train": pd.DataFrame({"Aa": x_train, "health": np.sin(3 * x_train)}),
        "validation": pd.DataFrame({"Aa": x_validation, "health": np.sin(3 * x_validation)}),
        "test": pd.DataFrame({"Aa": x_test, "health": np.sin(3 * x_test)}),
        "details": {"n_train": len(x_train)},
    }

    scores = scorer(
        split,
        outcome_column="health",
        exposure_column="Aa",
        model_family="natural_spline",
        seed=47,
    )

    assert [row["evaluation_set"] for row in scores] == ["spatial_validation", "spatial_test"]
    assert all(row["status"] == "ok" for row in scores)
    assert [row["n_evaluation"] for row in scores] == [12, 12]
    assert all(np.isfinite(row["r_squared"]) for row in scores)


def test_phase_h_validation_writes_spatial_and_common_period_summaries(tmp_path):
    from experiments.oracle_exploration.phase_h_search import MODEL_SCHEDULE_VERSION
    from experiments.oracle_exploration.phase_h_validation import run_phase_h_validation

    run_dir = tmp_path / "phase_h_test"
    pair_dir = run_dir / "pairs" / "candidate_pairs"
    table_dir = run_dir / "tables"
    pair_dir.mkdir(parents=True)
    table_dir.mkdir(parents=True)
    cell_rows = []
    pairs_by_exposure = {}
    for block in range(20):
        x = block * 5000.0 + 1000.0
        even_id, odd_id = 2 * block, 2 * block + 1
        cell_rows.extend(
            [
                {
                    "cell_id": even_id, "x": x, "y": 1000.0,
                    "Aa_20yr": 10.0 + block, "Aa_2020": 1.0 + block * 0.2,
                    "AN": 3.0 + block * 0.1,
                },
                {
                    "cell_id": odd_id, "x": x + 30.0, "y": 1020.0,
                    "Aa_20yr": 1.0 + block * 0.1, "Aa_2020": 12.0 + block,
                    "AN": 5.0 + block * 0.2,
                },
            ]
        )
        pairs_by_exposure.setdefault("Aa_20yr", []).append(
            {
                "pair_id": f"p20_{block}", "high_erosion_cell_id": even_id,
                "low_erosion_cell_id": odd_id, "delta_log1p_Aa": 0.5 + block * 0.01,
                "delta_AN": -2.0 - block * 0.1,
            }
        )
        pairs_by_exposure.setdefault("Aa_2020", []).append(
            {
                "pair_id": f"p2020_{block}", "high_erosion_cell_id": odd_id,
                "low_erosion_cell_id": even_id, "delta_log1p_Aa": 0.7 + block * 0.01,
                "delta_AN": 2.0 + block * 0.1,
            }
        )
    cells = pd.DataFrame(cell_rows)
    trial_rows = []
    for exposure, candidate_id in (("Aa_20yr", "cand_20yr"), ("Aa_2020", "cand_2020")):
        pairs = pd.DataFrame(pairs_by_exposure[exposure])
        pairs.to_parquet(pair_dir / f"{candidate_id}.parquet", index=False)
        for estimand in ("matched_pixel", "matched_pair"):
            trial_rows.append(
                {
                    "model_schedule_version": MODEL_SCHEDULE_VERSION,
                    "stage": "stage2_full_sample",
                    "analysis_population": "phase_h_matched_only",
                    "subset_name": "all",
                    "fit_scope": "exposure_only",
                    "evaluation_set": "apparent",
                    "status": "ok",
                    "r_squared": 0.4,
                    "candidate_id": candidate_id,
                    "exposure": exposure,
                    "outcome": "AN",
                    "estimand": estimand,
                    "model_family": "natural_spline",
                    "n_matched_pairs_candidate": 20,
                    "selection_parent_candidate_id": candidate_id,
                    "stage1_rank": 1,
                }
            )
    pd.DataFrame(trial_rows).to_csv(table_dir / "phase_h_model_trials.csv", index=False)
    config = {
        "spatial_validation": {
            "block_size_m": 5000,
            "seeds": [11, 12],
            "buffer_distances_m": [300],
        },
        "models": {"tree_training_max_rows": 12000},
    }

    result = run_phase_h_validation(run_dir, cells, config)

    spatial = result["spatial_validation_by_split"]
    period = result["period_comparison_by_split"]
    period_summary = result["period_comparison_summary"]
    assert set(spatial["evaluation_set"]) == {"spatial_validation", "spatial_test"}
    assert spatial["status"].eq("ok").all()
    assert spatial["n_same_block_pairs"].eq(20).all()
    assert set(period["exposure"]) == {"Aa_20yr", "Aa_2020"}
    assert period["common_sample_n"].nunique() >= 2
    assert period["status"].eq("ok").all()
    assert period_summary["n_seed_splits"].eq(2).all()
    assert (table_dir / "phase_h_leaderboard_5_best_spatial_validation.csv").is_file()


def test_model_availability_uses_resolvable_run_scope(monkeypatch):
    from experiments.oracle_exploration import phase_h_search

    resolve = getattr(phase_h_search, "available_model_families", None)
    assert callable(resolve)
    monkeypatch.setattr(phase_h_search.importlib.util, "find_spec", lambda _: None)

    result = resolve({"models": {"families": ["ols", "natural_spline"]}})

    assert result == ["ols", "natural_spline"]


def test_nonparametric_postprocess_streams_large_candidate_pair_tables(tmp_path, monkeypatch):
    from experiments.oracle_exploration import phase_h_post

    root = tmp_path / "phase_h"
    pair_dir = root / "pairs" / "candidate_pairs"
    table_dir = root / "tables"
    pair_dir.mkdir(parents=True)
    table_dir.mkdir(parents=True)
    cells = pd.DataFrame(
        {
            "cell_id": np.arange(40),
            "Aa_20yr": np.linspace(0.0, 8.0, 40),
            "Aa_2020": np.linspace(1.0, 9.0, 40),
            "AN": np.sin(np.linspace(0.0, 2.0, 40)),
        }
    )
    trials = []
    for exposure, candidate_id in (("Aa_20yr", "stage2_Aa_20yr_01"), ("Aa_2020", "stage2_Aa_2020_01")):
        high = np.arange(0, 40, 2)
        low = high + 1
        pairs = pd.DataFrame(
            {
                "pair_id": [f"{candidate_id}_{index}" for index in range(20)],
                "high_erosion_cell_id": high,
                "low_erosion_cell_id": low,
                "high_AN": cells.set_index("cell_id").loc[high, "AN"].to_numpy(),
                "low_AN": cells.set_index("cell_id").loc[low, "AN"].to_numpy(),
                "delta_AN": cells.set_index("cell_id").loc[high, "AN"].to_numpy()
                - cells.set_index("cell_id").loc[low, "AN"].to_numpy(),
                "delta_log1p_Aa": np.linspace(0.1, 1.0, 20),
                "method": "M4",
                "exposure": exposure,
                "soil_type": 26,
                "environment_group": "g1",
            }
        )
        pairs.to_parquet(pair_dir / f"{candidate_id}.parquet", index=False)
        trials.append(
            {
                "stage": "stage2_full_sample", "estimand": "matched_pair",
                "subset_name": "all", "fit_scope": "exposure_only", "status": "ok",
                "r_squared": 0.4, "exposure": exposure, "outcome": "AN",
                "candidate_id": candidate_id,
            }
        )
    pd.DataFrame(trials).to_csv(table_dir / "phase_h_model_trials.csv", index=False)
    original = phase_h_post.build_nonparametric_tables
    call_sizes = []

    def tracked_build(cells_arg, candidate_pairs, **kwargs):
        call_sizes.append(len(candidate_pairs))
        return original(cells_arg, candidate_pairs, **kwargs)

    monkeypatch.setattr(phase_h_post, "build_nonparametric_tables", tracked_build)

    result = phase_h_post.run_phase_h_nonparametric_postprocessing(
        root,
        cells,
        {"outcomes": ["AN"], "seed": 10, "figures": {"boxplot_pair_sample_size": 10}},
    )

    assert call_sizes == [1, 1]
    assert len(result["response_summary"]) == 4
    assert (root / "checkpoints" / "h4_nonparametric_complete.json").is_file()


def test_phase_h_environment_group_labels_use_environment_only_and_are_stable():
    import importlib
    import importlib.util

    module_name = "experiments.oracle_exploration.phase_h_groups"
    assert importlib.util.find_spec(module_name) is not None
    groups = importlib.import_module(module_name)
    build_labels = getattr(groups, "build_environment_group_labels", None)
    assert callable(build_labels)

    rng = np.random.default_rng(100)
    n_per_soil = 30
    soil = np.repeat([26, 27, 85], n_per_soil)
    cells = pd.DataFrame(
        {
            "cell_id": np.arange(len(soil)),
            "SoilType": soil,
            "Elevation": np.concatenate([
                rng.normal(20, 2, n_per_soil), rng.normal(60, 3, n_per_soil), rng.normal(110, 3, n_per_soil)
            ]),
            "LST": np.concatenate([
                rng.normal(10, 1, n_per_soil), rng.normal(25, 2, n_per_soil), rng.normal(40, 1, n_per_soil)
            ]),
            "TWI": rng.normal(size=len(soil)),
            "AN": rng.normal(size=len(soil)),
        }
    )
    options = {
        "primary_soils": [26, 27],
        "kmeans_counts": [2, 3],
        "kmeans_seeds": {2: 21, 3: 22},
        "stability_replicates": 2,
        "gradient_variables": ["Elevation"],
        "gradient_bins": 3,
        "hierarchical_k": 2,
        "hierarchical_fit_rows_per_soil": 20,
        "hierarchical_seed": 23,
    }

    labels, stability = build_labels(cells, options)
    changed_outcomes = cells.assign(AN=rng.normal(100, 10, len(cells)))
    labels_after_outcome_change, _ = build_labels(changed_outcomes, options)

    assert labels["cell_id"].tolist() == cells["cell_id"].tolist()
    assert labels["kmeans_terrain_k2"].equals(labels_after_outcome_change["kmeans_terrain_k2"])
    assert labels["hierarchical_ward_v5"].equals(labels_after_outcome_change["hierarchical_ward_v5"])
    assert labels["gradient_Elevation"].equals(labels_after_outcome_change["gradient_Elevation"])
    nonprimary = labels.loc[labels["cell_id"].ge(2 * n_per_soil)].drop(columns="cell_id")
    assert nonprimary.isna().all(axis=None)
    stochastic = stability["group_method"].str.startswith("kmeans_")
    assert stability.loc[stochastic, "mean_adjusted_rand_index"].between(-1.0, 1.0).all()
    assert stability.loc[~stochastic, "stability_status"].str.startswith("deterministic").all()


def test_phase_h_group_profiles_report_nonlinear_matched_pair_summaries():
    from experiments.oracle_exploration.phase_h_groups import build_environment_group_profiles

    n_pairs = 24
    high_ids = np.arange(n_pairs) * 2
    low_ids = high_ids + 1
    cells = pd.DataFrame(
        {
            "cell_id": np.arange(n_pairs * 2),
            "x": np.tile(np.arange(n_pairs, dtype=float) * 40 + 100, 2),
            "y": np.repeat(200.0, n_pairs * 2),
            "SoilType": np.repeat(26, n_pairs * 2),
            "Elevation": np.linspace(20, 80, n_pairs * 2),
            "Cur": np.linspace(0, 1, n_pairs * 2),
            "TWI": np.linspace(1, 3, n_pairs * 2),
            "LST": np.linspace(10, 20, n_pairs * 2),
            "AN": np.linspace(2, 5, n_pairs * 2),
        }
    )
    labels = pd.DataFrame(
        {"cell_id": np.arange(n_pairs * 2), "soil_group": "soil_26"}
    )
    exposure = np.linspace(0.1, 2.4, n_pairs)
    pairs = pd.DataFrame(
        {
            "pair_id": [f"p{i}" for i in range(n_pairs)],
            "high_erosion_cell_id": high_ids,
            "low_erosion_cell_id": low_ids,
            "delta_log1p_Aa": exposure,
            "delta_AN": np.sin(exposure) - 0.2 * exposure,
            "environmental_distance": np.linspace(0.01, 0.1, n_pairs),
            "geographic_distance": np.repeat(40.0, n_pairs),
        }
    )

    profiles = build_environment_group_profiles(
        cells,
        labels,
        pairs,
        candidate_id="candidate_1",
        exposure="Aa_20yr",
        outcome="AN",
        group_columns=["soil_group"],
        min_pairs_per_group=10,
    )

    assert len(profiles) == 1
    row = profiles.iloc[0]
    assert row["group_label"] == "soil_26"
    assert row["n_pairs"] == n_pairs
    assert row["n_cells"] == n_pairs * 2
    assert np.isfinite(row["spearman_rho"])
    assert np.isfinite(row["within_group_nonlinear_r_squared"])
    assert row["outcome_guided_selection"]
