from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _api(name: str):
    module = importlib.import_module("experiments.oracle_exploration.experiment")
    function = getattr(module, name, None)
    assert callable(function), f"experiment.{name} has not been implemented"
    return function


def test_output_tree_stays_in_oracle_root_and_has_expected_directories(tmp_path):
    create_output_tree = _api("create_output_tree")

    paths = create_output_tree(tmp_path, "oracle_20261026_test")

    assert paths["root"].is_relative_to(tmp_path / "outputs" / "exploratory_oracle")
    assert (tmp_path / "outputs" / "formal").exists() is False
    for name in ("experiments", "matching", "subgroups", "models", "tables", "figures", "diagnostics", "reports"):
        assert paths[name].is_dir()
    with pytest.raises(FileExistsError):
        create_output_tree(tmp_path, "oracle_20261026_test")


def test_resume_output_tree_reopens_only_existing_oracle_runs(tmp_path):
    create_output_tree = _api("create_output_tree")
    open_output_tree = _api("_open_output_tree")
    paths = create_output_tree(tmp_path, "oracle_20261026_resume")

    reopened = open_output_tree(tmp_path, paths["root"])

    assert reopened["root"] == paths["root"]
    assert reopened["matching_stage1"].is_dir()
    with pytest.raises(ValueError, match="Oracle"):
        open_output_tree(tmp_path, tmp_path / "outputs" / "formal" / "run_1")


def test_stratified_sample_is_exact_reproducible_and_covers_strata():
    stratified_sample = _api("stratified_sample")
    frame = pd.DataFrame(
        {
            "cell_id": range(30),
            "soil": [26] * 10 + [27] * 10 + [42] * 10,
            "block": [0, 0, 1, 1, 2] * 6,
        }
    )

    first = stratified_sample(frame, 12, strata_columns=["soil", "block"], seed=10)
    second = stratified_sample(frame, 12, strata_columns=["soil", "block"], seed=10)

    assert len(first) == 12
    assert first["cell_id"].is_unique
    pd.testing.assert_frame_equal(first, second)
    assert set(first["soil"]) == {26, 27, 42}
    assert set(zip(first["soil"], first["block"])) == set(zip(frame["soil"], frame["block"]))


def test_stage_count_reconciliation_counts_trials_not_outcome_score_rows():
    reconcile_stage_counts = _api("reconcile_stage_counts")
    records = pd.DataFrame(
        {
            "experiment_type": ["stage1_matching_search"] * 4,
            "exposure": ["Aa_20yr"] * 3 + ["Aa_2020"],
            "candidate_id": ["a", "a", "b", "c"],
            "status": ["success", "success", "failed", "success"],
        }
    )

    actual = reconcile_stage_counts(records, {"Aa_20yr": 2, "Aa_2020": 1})

    assert actual == {"Aa_20yr": 2, "Aa_2020": 1}
    with pytest.raises(ValueError, match="expected.*actual|actual.*expected"):
        reconcile_stage_counts(records, {"Aa_20yr": 3, "Aa_2020": 1})


def test_saved_matching_stages_can_be_reloaded_without_repeating_trials(tmp_path):
    load_matching = _api("_load_matching_stage_outputs")
    frame = pd.DataFrame(
        {
            "cell_id": np.arange(12),
            "SoilType": 26,
            "spatial_block": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "spatial_role": ["train"] * 4 + ["validation"] * 2 + ["test"] * 6,
        }
    )
    config = {
        "exposures": ["Aa_20yr"],
        "outcomes": ["AN"],
        "seed": 42,
        "stages": {
            "stage1": {"sample_size": 3, "candidates_per_exposure": 1},
            "stage2": {"sample_size": 4},
        },
    }
    paths = {}
    for key, relative in (
        ("matching_stage1", "matching/stage1"),
        ("matching_stage2", "matching/stage2"),
        ("matching_stage3", "matching/stage3"),
    ):
        paths[key] = tmp_path / relative
        paths[key].mkdir(parents=True)
    candidate = {
        "candidate_id": "candidate_1",
        "exposure": "Aa_20yr",
        "environment_variables": ["Elevation"],
        "vector_name": "V1",
    }
    registry = tmp_path / "registry.jsonl"
    registry.write_text(
        json.dumps({"experiment_type": "stage1_matching_search", "candidate": candidate}) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        [{"experiment_type": "stage1_matching_search", "candidate_id": "candidate_1",
          "exposure": "Aa_20yr", "status": "success", "n_pairs": 2}]
    ).to_parquet(paths["matching_stage1"] / "trial_status.parquet", index=False)
    pd.DataFrame({"candidate_id": ["candidate_1"]}).to_parquet(
        paths["matching_stage1"] / "scores.parquet", index=False
    )
    pd.DataFrame(columns=["candidate_id"]).to_parquet(
        paths["matching_stage1"] / "aggregate_ranks.parquet", index=False
    )
    stage2_trials = pd.DataFrame(
        [
            {"experiment_type": "stage2_matching_train", "candidate_id": "candidate_1",
             "exposure": "Aa_20yr", "evaluation_set": "train", "status": "success"},
            {"experiment_type": "stage2_matching_validation", "candidate_id": "candidate_1",
             "exposure": "Aa_20yr", "evaluation_set": "validation", "status": "success"},
        ]
    )
    stage2_trials.to_parquet(paths["matching_stage2"] / "trial_status.parquet", index=False)
    pd.DataFrame(
        [{"candidate_id": "candidate_1", "exposure": "Aa_20yr", "outcome": "AN",
          "score_set": "matched_cells", "evaluation_set": "validation", "r_squared": 0.7}]
    ).to_parquet(paths["matching_stage2"] / "scores.parquet", index=False)
    pd.DataFrame({"candidate_id": ["candidate_1"]}).to_parquet(
        paths["matching_stage2"] / "validation_winners.parquet", index=False
    )
    pd.DataFrame(columns=["candidate_id"]).to_parquet(
        paths["matching_stage3"] / "trial_status.parquet", index=False
    )
    pd.DataFrame(columns=["candidate_id"]).to_parquet(
        paths["matching_stage3"] / "scores.parquet", index=False
    )

    matching = load_matching(frame, config, paths, registry)

    assert matching["stage1_counts"] == {"Aa_20yr": 1}
    assert matching["validation_winners"] == {("Aa_20yr", "AN"): "candidate_1"}
    assert matching["stage1_selected"]["Aa_20yr"] == [candidate]
    assert matching["stage2_sample"]["spatial_role"].isin(["train", "validation"]).all()


def test_cli_entrypoint_is_after_all_functions_used_by_the_pipeline():
    source = Path(__file__).resolve().parents[1] / "experiments/oracle_exploration/experiment.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    main_guards = [
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
        and any(isinstance(value, ast.Constant) and value.value == "__main__" for value in node.test.comparators)
    ]
    assert len(main_guards) == 1
    function_lines = [
        node.lineno
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in {"_candidate_trial", "_run_matching_stages", "_run_regression_search", "_run_oracle_subgroups"}
    ]

    assert len(function_lines) == 4
    assert main_guards[0].lineno > max(function_lines)


def test_experiment_rows_expose_role_specific_r2_columns():
    attach_role_metrics = _api("_attach_role_metric_columns")
    records = pd.DataFrame(
        {
            "evaluation_set": ["apparent", "train", "validation", "test"],
            "r_squared": [0.4, 0.3, 0.25, 0.2],
            "validation_r2": [None, None, 0.25, 0.25],
            "test_r2": [None, None, None, 0.2],
            "outcome": ["AN"] * 4,
            "health_indicator": ["all_requested_outcomes"] * 4,
            "distance_method": ["standardized_euclidean"] * 4,
            "environment_caliper": [0.35] * 4,
            "n": [20, 12, 4, 4],
        }
    )

    result = attach_role_metrics(records)

    assert result["apparent_r2"].iloc[0] == pytest.approx(0.4)
    assert result["train_r2"].iloc[1] == pytest.approx(0.3)
    assert result["validation_r2"].iloc[2] == pytest.approx(0.25)
    assert result["test_r2"].iloc[3] == pytest.approx(0.2)
    assert {"apparent_r2", "train_r2", "validation_r2", "test_r2"}.issubset(result.columns)
    assert result["health_indicator"].eq("AN").all()
    assert result["matching_method"].eq("standardized_euclidean").all()
    assert result["environment_distance"].eq(0.35).all()
    assert result["n_cells"].tolist() == [20, 12, 4, 4]


def test_regression_search_runs_configured_models_on_stage2_sample(tmp_path):
    run_regression_search = _api("_run_regression_search")
    rng = np.random.default_rng(18)
    n = 80
    exposure = np.linspace(0.0, 20.0, n)
    frame = pd.DataFrame(
        {
            "cell_id": np.arange(n),
            "SoilType": 26,
            "Aa_20yr": exposure,
            "AN": 1.5 * np.log1p(exposure) + rng.normal(0, 0.1, n),
            "Elevation": rng.normal(size=n),
            "LST": rng.normal(size=n),
            "spatial_role": ["train"] * 48 + ["validation"] * 16 + ["test"] * 16,
        }
    )
    stage2 = frame.loc[frame["spatial_role"].isin(["train", "validation"])].copy()
    candidate = {
        "candidate_id": "candidate_1",
        "exposure": "Aa_20yr",
        "environment_variables": ["Elevation", "LST"],
        "vector_name": "V1",
    }
    config = {
        "exposures": ["Aa_20yr"],
        "outcomes": ["AN"],
        "environment": {"vectors": {"V1": ["Elevation", "LST"]}},
        "model_families": ["ols"],
        "input_run_id": "synthetic",
        "seed": 18,
    }
    paths = {name: tmp_path / name for name in ("models", "diagnostics", "tables", "matching_stage3")}
    for path in paths.values():
        path.mkdir()

    result = run_regression_search(
        frame,
        config,
        paths,
        tmp_path / "registry.jsonl",
        {
            "stage1_candidates": {"Aa_20yr": [candidate]},
            "validation_winners": {("Aa_20yr", "AN"): "candidate_1"},
            "stage2_sample": stage2,
        },
        "20261026",
        "abc123",
    )

    stage2_scores = result["stage2_model_scores"]
    assert not stage2_scores.empty
    assert set(stage2_scores["evaluation_set"]) == {"apparent", "train", "validation"}
    assert stage2_scores["sample_size_actual"].eq(len(stage2)).all()
    assert (stage2_scores["sample_size_actual"] <= 100000).all()
    assert (paths["models"] / "stage2_model_search.parquet").is_file()

    run_regression_search(
        frame,
        config,
        paths,
        tmp_path / "registry.jsonl",
        {
            "stage1_candidates": {"Aa_20yr": [candidate]},
            "validation_winners": {("Aa_20yr", "AN"): "candidate_1"},
            "stage2_sample": stage2,
        },
        "20261026",
        "abc123",
    )
    record_ids = [
        json.loads(line)["experiment_id"]
        for line in (tmp_path / "registry.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(record_ids) == len(set(record_ids))


def test_feature_regime_registry_ids_sanitize_outcome_and_model_names(tmp_path):
    run_regression_search = _api("_run_regression_search")
    rng = np.random.default_rng(41)
    n = 90
    exposure = np.linspace(0, 25, n)
    frame = pd.DataFrame(
        {
            "cell_id": np.arange(n),
            "SoilType": 26,
            "Aa_20yr": exposure,
            "SHI_candidate": 2 * np.log1p(exposure) + rng.normal(0, 0.1, n),
            "Elevation": rng.normal(size=n),
            "LST": rng.normal(size=n),
            "spatial_role": ["train"] * 54 + ["validation"] * 18 + ["test"] * 18,
        }
    )
    stage2 = frame.loc[frame["spatial_role"].isin(["train", "validation"])].copy()
    candidate = {
        "candidate_id": "candidate_1",
        "exposure": "Aa_20yr",
        "environment_variables": ["Elevation", "LST"],
        "vector_name": "V1",
    }
    config = {
        "exposures": ["Aa_20yr"],
        "outcomes": ["SHI_candidate"],
        "environment": {"vectors": {"V1": ["Elevation", "LST"]}},
        "model_families": ["ols"],
        "input_run_id": "synthetic",
        "seed": 41,
    }
    paths = {name: tmp_path / name for name in ("models", "diagnostics", "tables", "matching_stage3")}
    for path in paths.values():
        path.mkdir()
    registry = tmp_path / "registry.jsonl"

    result = run_regression_search(
        frame,
        config,
        paths,
        registry,
        {
            "stage1_candidates": {"Aa_20yr": [candidate]},
            "validation_winners": {("Aa_20yr", "SHI_candidate"): "candidate_1"},
            "stage2_sample": stage2,
        },
        "20261026",
        "abc123",
    )

    assert not result["feature_regime_scores"].empty
    records = [line for line in registry.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert any(json.loads(line)["experiment_type"] == "feature_regime_comparison" for line in records)


def test_oracle_subgroups_record_empty_complete_case_trials_and_continue(tmp_path):
    run_subgroups = _api("_run_oracle_subgroups")
    n = 24
    frame = pd.DataFrame(
        {
            "cell_id": np.arange(n),
            "SoilType": 26,
            "Aa_20yr": np.arange(n, dtype=float),
            "AN": 1.2 * np.arange(n, dtype=float) + 0.2,
            "Elevation": np.nan,
            "environment_group": 0,
        }
    )
    candidate = {
        "candidate_id": "candidate_1",
        "exposure": "Aa_20yr",
        "environment_variables": ["Elevation"],
        "vector_name": "V1",
    }
    config = {
        "exposures": ["Aa_20yr"],
        "outcomes": ["AN"],
        "input_run_id": "synthetic",
        "seed": 7,
        "minimum_subgroup_cells": 5,
        "retention_fractions": [1.0, 0.5],
        "retention_search_trials": 2,
    }
    paths = {"subgroups": tmp_path / "subgroups"}
    paths["subgroups"].mkdir()
    registry = tmp_path / "registry.jsonl"

    result = run_subgroups(
        frame,
        config,
        paths,
        registry,
        {
            "stage1_candidates": {"Aa_20yr": [candidate]},
            "validation_winners": {("Aa_20yr", "AN"): "candidate_1"},
        },
        {"group_columns": {("Elevation",): "environment_group"}},
        "20261026",
        "abc123",
    )

    records = [json.loads(line) for line in registry.read_text(encoding="utf-8").splitlines() if line.strip()]
    failed_types = {
        row["experiment_type"] for row in records if row.get("status") == "failed"
    }
    assert "oracle_retention_search" in failed_types
    assert "oracle_subgroup_search" in failed_types
    assert "subgroup_scores" in result


def test_report_formatter_renders_numpy_missing_values_as_na():
    formatter = _api("_fmt")

    assert formatter(np.float32(np.nan)) == "n/a"


def test_review_report_exposes_config_effect_and_validation_delta(tmp_path):
    write_report = _api("write_review_report")
    results = tmp_path / "oracle_run"
    for directory in ("tables", "matching/stage2", "models", "subgroups", "figures"):
        (results / directory).mkdir(parents=True, exist_ok=True)
    (results / "tables" / "experiment_summary.json").write_text(
        '{"input_run_id":"source","git_sha":"abc123","git_dirty":true,'
        '"stage1_counts":{"Aa_20yr":100,"Aa_2020":100},'
        '"stage2_config_counts":{"Aa_20yr":20,"Aa_2020":20},'
        '"stage3_matching_test_count":8,"model_trial_count":1024,'
        '"model_families":["ols","quadratic","random_forest"]}',
        encoding="utf-8",
    )
    (results / "search_config.yaml").write_text(
        "matching:\n"
        "  distance_methods: [standardized_euclidean, weighted_euclidean, mahalanobis, pca_euclidean, direct_nearest_neighbor]\n"
        "  environment_calipers: [0.15, 0.25, 0.35, 0.5, 0.75, 1.0]\n"
        "  geographic_radii_m: [100, 200, 300, 500, 750, 1000, 1500]\n"
        "  erosion_rank_gaps: [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]\n"
        "  cluster_methods: [none, minibatch_kmeans, kmeans, gaussian_mixture, agglomerative]\n"
        "  exposure_rules: [within_group_rank, positive_only, exposure_only]\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        [{"outcome": "SHI_candidate", "exposure": "Aa_20yr", "r_squared": 0.72, "n": 100,
          "model_name": "ols", "environment_vector": "V1", "coefficient": 0.5,
          "direction": "positive", "model_config": "OLS V1"}]
    ).to_parquet(results / "tables" / "leaderboard_full_sample_best.parquet", index=False)
    pd.DataFrame(
        [{"outcome": "SHI_candidate", "exposure": "Aa_20yr", "r_squared": 0.91,
          "n_selected": 50, "sample_retention_ratio": 0.5, "selection_objective": "maximize R2",
          "coefficient": 0.8, "direction": "positive", "evaluation_set": "apparent"}]
    ).to_parquet(results / "tables" / "leaderboard_oracle_apparent_best.parquet", index=False)
    pd.DataFrame(
        [{"outcome": "SHI_candidate", "exposure": "Aa_20yr", "test_r2": 0.31,
          "r_squared": 0.31, "selected_validation_r2": 0.42, "n": 20,
          "model_name": "ols", "environment_vector": "V1", "coefficient": 0.4,
          "direction": "positive",
          "model_config": '{"model_family":"ols","numeric_features":["Aa_20yr","Elevation"],"coefficients":[-0.1,0.2],"intercept":0.3}',
          "evaluation_set": "test"}]
    ).to_parquet(results / "tables" / "leaderboard_spatial_test_best.parquet", index=False)
    pd.DataFrame(
        [{"outcome": "AN", "exposure": "Aa_20yr", "delta_r2": 0.2,
          "validation_delta_r2": 0.2, "test_delta_r2": 0.05,
          "selection_basis": "validation_delta_r2", "evaluation_set": "validation"}]
    ).to_parquet(results / "tables" / "leaderboard_erosion_increment_best.parquet", index=False)
    pd.DataFrame(
        [{"outcome": "SHI_candidate", "exposure": "Aa_20yr", "r_squared": 0.42,
          "candidate_id": "match_1", "vector_name": "V1", "n_pairs": 12,
          "area_coverage": 0.3, "max_abs_matched_smd": 0.1,
          "median_erosion_difference": 8.0}]
    ).to_parquet(results / "matching" / "stage2" / "validation_winners.parquet", index=False)
    pd.DataFrame(
        [{"apparent_r2": 0.91, "predictor_set": "exposure_only",
          "exposure": "Aa_20yr", "outcome": "SHI_candidate",
          "target_retention_fraction": 0.5, "sample_retention_ratio": 0.5,
          "n_selected": 50, "n_complete_cases": 100,
          "full_sample_apparent_r2": 0.5,
          "trial_count": 20, "selection_rule": "oracle",
          "coefficient": 0.8, "direction": "positive"}]
    ).to_parquet(results / "subgroups" / "oracle_retention.parquet", index=False)
    pd.DataFrame(
        [
            {"r_squared": 0.88, "subgroup_name": "soil_26", "exposure": "Aa_20yr",
             "outcome": "AN", "subgroup_definition": '{"column":"SoilType","op":"eq","value":26}',
             "n": 30, "sample_retention_ratio": 0.3, "coefficient": 0.4,
             "direction": "positive", "scope_status": "in_scope"},
            {"r_squared": 0.99, "subgroup_name": "soil_85", "exposure": "Aa_20yr",
             "outcome": "AN", "subgroup_definition": '{"column":"SoilType","op":"eq","value":85}',
             "n": 10, "sample_retention_ratio": 0.1, "coefficient": 0.4,
             "direction": "positive", "scope_status": "excluded_soil_85_water_class"},
        ]
    ).to_parquet(results / "subgroups" / "subgroup_scores.parquet", index=False)
    (results / "figures" / "figure_index.json").write_text("[]", encoding="utf-8")

    report = write_report(results, tmp_path / "review.md").read_text(encoding="utf-8")

    assert "SHI_candidate = 0.3 - 0.1 × Aa_20yr + 0.2 × Elevation" in report
    assert "validation=0.2000; test=0.0500" in report
    assert "5 distance methods" in report
    assert "200 Stage 1 candidate parameter configurations" in report
    assert "3 regression model families across 1,024 model trials" in report
    assert "baseline full-sample R²=0.5000 on n=100 complete cases" in report
    assert "no SoilType or erosion-interval restriction" in report
    assert "definition={\"column\":\"SoilType\"" in report
    assert "Highest explicit subgroup apparent model: soil_26" in report
    assert "soil_85" not in report
    assert "trials=20" in report
    assert "natural spline fitted to the fixed stratified plotting sample" in report


def test_time_scale_comparison_uses_identical_rows_and_ols_specification():
    compare_common = _api("_compare_exposures_on_common_rows")
    rng = np.random.default_rng(27)
    n = 40
    aa_20yr = np.linspace(0.0, 12.0, n)
    aa_2020 = aa_20yr * 1.4 + rng.normal(0, 0.05, n)
    frame = pd.DataFrame(
        {
            "cell_id": np.arange(n),
            "Aa_20yr": aa_20yr,
            "Aa_2020": aa_2020,
            "AN": 0.7 * aa_20yr + rng.normal(0, 0.1, n),
            "Elevation": rng.normal(size=n),
            "LST": rng.normal(size=n),
            "spatial_role": ["train"] * 24 + ["validation"] * 8 + ["test"] * 8,
        }
    )
    frame.loc[0, "Aa_20yr"] = np.nan
    frame.loc[1, "Aa_2020"] = np.nan

    result = compare_common(
        frame,
        outcomes=["AN"],
        exposures=["Aa_20yr", "Aa_2020"],
        environment=["Elevation", "LST"],
    )

    comparisons = result.loc[
        result["regime"].eq("environment_plus_exposure")
        & result["evaluation_set"].eq("apparent")
    ]
    assert set(comparisons["exposure"]) == {"Aa_20yr", "Aa_2020"}
    assert comparisons["common_sample_n"].eq(38).all()
    assert comparisons["n"].eq(38).all()
    assert comparisons["environment_variables"].map(tuple).nunique() == 1
    assert comparisons["selection_objective"].eq("same_sample_same_ols_for_both_exposures").all()


def test_best_full_sample_model_curves_use_apparent_winner_and_fixed_environment():
    build_curves = _api("_selected_full_sample_curves")
    x = np.linspace(0, 10, 50)
    frame = pd.DataFrame({"Aa_20yr": x, "AN": 0.5 * x**2 + 2, "Elevation": x + 20})
    scores = pd.DataFrame(
        [
            {"exposure": "Aa_20yr", "outcome": "AN", "model_name": "ols",
             "environment_variables": ["Elevation"], "environment_vector": "V5",
             "group_column": None, "evaluation_set": "apparent", "status": "ok",
             "r_squared": 0.2, "n": 50},
            {"exposure": "Aa_20yr", "outcome": "AN", "model_name": "quadratic",
             "environment_variables": ["Elevation"], "environment_vector": "V5",
             "group_column": None, "evaluation_set": "apparent", "status": "ok",
             "r_squared": 0.9, "n": 50},
            {"exposure": "Aa_2020", "outcome": "AN", "model_name": "ols",
             "environment_variables": ["Elevation"], "environment_vector": "V5",
             "group_column": None, "evaluation_set": "apparent", "status": "ok",
             "r_squared": 0.99, "n": 50},
        ]
    )

    curves = build_curves(frame, scores, exposure="Aa_20yr", curve_points=25)

    assert len(curves) == 25
    assert curves["model_name"].eq("quadratic").all()
    assert curves["selected_model_r_squared"].eq(0.9).all()
    assert curves["curve_type"].eq("full_sample_best_model_partial_dependence").all()
    assert curves["environment_at_median"].map(lambda value: value["Elevation"]).eq(25).all()


def test_existing_curve_cache_is_filled_for_missing_outcomes():
    complete_curves = _api("_complete_full_sample_curves")
    x = np.linspace(0, 10, 30)
    frame = pd.DataFrame(
        {
            "Aa_20yr": x,
            "AN": 2.0 * x + 1.0,
            "SHI_candidate": 0.4 * x + 0.2,
            "Elevation": x + 10.0,
        }
    )
    scores = pd.DataFrame(
        [
            {"exposure": "Aa_20yr", "outcome": outcome, "model_name": "ols",
             "environment_variables": ["Elevation"], "environment_vector": "V5",
             "group_column": None, "evaluation_set": "apparent", "status": "ok",
             "r_squared": 0.8, "n": len(frame)}
            for outcome in ("AN", "SHI_candidate")
        ]
    )
    cached = pd.DataFrame(
        [{"outcome": "AN", "exposure": "Aa_20yr", "curve_type": "full_sample_best_model_partial_dependence",
          "exposure_value": 1.0, "predicted_outcome": 2.0}]
    )

    curves = complete_curves(frame, scores, cached, exposure="Aa_20yr", curve_points=5)

    assert len(curves.loc[curves["outcome"].eq("AN")]) == 1
    assert len(curves.loc[curves["outcome"].eq("SHI_candidate")]) == 5
