from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest


def _api(name: str):
    module = importlib.import_module("experiments.oracle_exploration.rank_results")
    function = getattr(module, name, None)
    assert callable(function), f"rank_results.{name} has not been implemented"
    return function


def test_leaderboards_keep_apparent_oracle_and_spatial_test_scores_separate():
    rank_leaderboards = _api("rank_leaderboards")
    registry = pd.DataFrame(
        {
            "candidate_id": ["full", "oracle", "oracle_test_trap", "spatial", "spatial_test_trap", "delta"],
            "experiment_type": ["full_sample_model", "oracle_retention", "oracle_retention", "spatial_model", "spatial_model", "feature_regime"],
            "selection_mode": ["unscreened_full_sample", "outcome_selected_oracle", "outcome_selected_oracle", "validation_selected", "validation_selected", "unscreened_full_sample"],
            "outcome_guided_selection": [False, True, True, True, True, False],
            "evaluation_set": ["apparent", "apparent", "apparent", "test", "test", "apparent"],
            "outcome": ["AN"] * 6,
            "exposure": ["Aa_20yr"] * 6,
            "n": [1000, 250, 50, 300, 300, 1000],
            "r_squared": [0.42, 0.88, 0.70, 0.61, 0.95, 0.45],
            "validation_r2": [np.nan, np.nan, np.nan, 0.8, 0.4, np.nan],
            "test_r2": [np.nan, 0.99, 0.01, 0.61, 0.95, np.nan],
            "delta_r2": [0.05, 0.3, 0.1, 0.2, 0.1, 0.4],
            "selection_objective": ["full", "apparent", "apparent", "validation", "validation", "delta"],
        }
    )

    boards = rank_leaderboards(registry)

    assert set(boards) == {"full_sample_best", "oracle_apparent_best", "spatial_test_best", "erosion_increment_best"}
    assert boards["full_sample_best"].iloc[0]["candidate_id"] == "full"
    assert boards["oracle_apparent_best"].iloc[0]["candidate_id"] == "oracle"
    assert boards["oracle_apparent_best"]["evaluation_set"].eq("apparent").all()
    assert boards["spatial_test_best"].iloc[0]["candidate_id"] == "spatial"
    assert boards["spatial_test_best"].iloc[0]["test_r2"] == 0.61
    assert boards["spatial_test_best"]["evaluation_set"].eq("test").all()
    assert boards["erosion_increment_best"].iloc[0]["candidate_id"] == "delta"
    assert boards["oracle_apparent_best"].iloc[0]["r_squared"] == 0.88


def test_jsonl_registry_can_be_flattened_for_ranking(tmp_path):
    rank_leaderboards = _api("rank_leaderboards")
    path = tmp_path / "experiments.jsonl"
    path.write_text(
        '{"experiment_id":"e1","experiment_type":"full_sample_model","selection_mode":"unscreened_full_sample","outcome_guided_selection":false,"exposure":"Aa_20yr","scores":[{"outcome":"AN","evaluation_set":"apparent","score_type":"apparent","r_squared":0.5,"delta_r2":0.2,"n":50}]}'
        "\n",
        encoding="utf-8",
    )

    boards = rank_leaderboards(path)

    assert boards["full_sample_best"].iloc[0]["experiment_id"] == "e1"
    assert boards["full_sample_best"].iloc[0]["outcome"] == "AN"


def test_registry_flattening_restores_outcome_from_health_indicator(tmp_path):
    load_registry = _api("_load_registry")
    path = tmp_path / "experiments.jsonl"
    path.write_text(
        '{"experiment_id":"e1","experiment_type":"full_sample_model",'
        '"health_indicator":"AN","selection_mode":"unscreened_full_sample",'
        '"outcome_guided_selection":false,"exposure":"Aa_20yr",'
        '"scores":[{"evaluation_set":"apparent","model_name":"ols","r_squared":0.5}]}',
        encoding="utf-8",
    )

    flattened = load_registry(path)

    assert flattened.loc[0, "outcome"] == "AN"


def test_spatial_model_ranking_ignores_parent_ids_from_matching_records():
    rank_leaderboards = _api("rank_leaderboards")
    registry = pd.DataFrame(
        {
            "candidate_id": ["match_1", "match_1", "spatial_A", "spatial_A"],
            "parent_candidate_id": ["match_1", "match_1", None, None],
            "experiment_type": [
                "stage2_matching_validation",
                "stage3_matching_test",
                "spatial_model_search",
                "spatial_model_test",
            ],
            "selection_mode": ["outcome_guided_oracle"] * 4,
            "outcome_guided_selection": [True] * 4,
            "evaluation_set": ["validation", "test", "validation", "test"],
            "outcome": ["OM", "OM", "AN", "AN"],
            "exposure": ["Aa_20yr"] * 4,
            "model_name": [None, None, "ols", "ols"],
            "r_squared": [0.9, 0.4, 0.8, 0.7],
            "validation_r2": [np.nan, 0.9, np.nan, 0.8],
            "test_r2": [np.nan, 0.4, np.nan, 0.7],
            "delta_r2": [np.nan] * 4,
        }
    )

    board = rank_leaderboards(registry)["spatial_test_best"]

    assert board.loc[board["outcome"].eq("AN"), "candidate_id"].tolist() == ["spatial_A"]


def test_spatial_test_board_does_not_mix_matching_test_scores_with_model_scores():
    rank_leaderboards = _api("rank_leaderboards")
    registry = pd.DataFrame(
        {
            "candidate_id": ["match_1", "match_1", "spatial_A", "spatial_A"],
            "parent_candidate_id": ["match_1", "match_1", None, None],
            "experiment_type": [
                "stage3_matching_validation",
                "stage3_matching_test",
                "spatial_model_search",
                "spatial_model_test",
            ],
            "selection_mode": ["validation_selected"] * 4,
            "outcome_guided_selection": [True] * 4,
            "evaluation_set": ["validation", "test", "validation", "test"],
            "outcome": ["AN"] * 4,
            "exposure": ["Aa_20yr"] * 4,
            "model_name": [None, None, "ols", "ols"],
            "r_squared": [0.99, 0.95, 0.8, 0.7],
            "validation_r2": [np.nan, 0.99, np.nan, 0.8],
            "test_r2": [np.nan, 0.95, np.nan, 0.7],
            "delta_r2": [np.nan] * 4,
        }
    )

    board = rank_leaderboards(registry)["spatial_test_best"]

    assert board["candidate_id"].tolist() == ["spatial_A"]
    assert board.iloc[0]["model_name"] == "ols"
    assert board.iloc[0]["test_r2"] == pytest.approx(0.7)


def test_oracle_leaderboard_omits_scope_excluded_subgroups():
    rank_leaderboards = _api("rank_leaderboards")
    registry = pd.DataFrame(
        {
            "candidate_id": ["subgroup_Aa_20yr_AN_soil_85", "retention_Aa_20yr_AN_05"],
            "experiment_id": ["excluded", "in_scope"],
            "experiment_type": ["oracle_subgroup_search", "oracle_retention_search"],
            "selection_mode": ["outcome_selected_oracle"] * 2,
            "outcome_guided_selection": [True, True],
            "scope_status": ["excluded_soil_85_water_class", "in_scope"],
            "evaluation_set": ["apparent", "apparent"],
            "outcome": ["AN", "AN"],
            "exposure": ["Aa_20yr", "Aa_20yr"],
            "r_squared": [0.99, 0.8],
            "test_r2": [np.nan, np.nan],
        }
    )

    board = rank_leaderboards(registry)["oracle_apparent_best"]

    assert board["candidate_id"].tolist() == ["retention_Aa_20yr_AN_05"]


def test_spatial_test_ranking_handles_nested_registry_metadata():
    rank_leaderboards = _api("rank_leaderboards")
    registry = pd.DataFrame(
        {
            "experiment_id": ["spatial_ols", "spatial_ols"],
            "candidate_id": ["spatial_A", "spatial_A"],
            "experiment_type": ["spatial_model_search", "spatial_model_search"],
            "selection_mode": ["validation_selected"] * 2,
            "outcome_guided_selection": [True] * 2,
            "evaluation_set": ["validation", "test"],
            "outcome": ["AN", "AN"],
            "exposure": ["Aa_20yr", "Aa_20yr"],
            "model_name": ["ols", "ols"],
            "r_squared": [0.8, 0.7],
            "validation_r2": [0.8, 0.8],
            "test_r2": [np.nan, 0.7],
            "delta_r2": [np.nan, np.nan],
            "matching_diagnostics": [{"smd": {"Elevation": 0.1}}] * 2,
            "features": [["Aa_20yr", "Elevation"]] * 2,
        }
    )

    board = rank_leaderboards(registry)["spatial_test_best"]

    assert board["candidate_id"].tolist() == ["spatial_A"]
    assert board.iloc[0]["r_squared"] == pytest.approx(0.7)
    assert board.iloc[0]["selected_validation_r2"] == pytest.approx(0.8)


def test_erosion_increment_leaderboard_selects_by_validation_delta_not_test_delta():
    rank_leaderboards = _api("rank_leaderboards")
    registry = pd.DataFrame(
        {
            "candidate_id": ["validation_winner", "validation_winner", "test_trap", "test_trap"],
            "experiment_type": ["feature_regime_comparison"] * 4,
            "selection_mode": ["validation_selected"] * 4,
            "outcome_guided_selection": [True] * 4,
            "evaluation_set": ["validation", "test", "validation", "test"],
            "regime": ["environment_plus_exposure"] * 4,
            "model_name": ["ols"] * 4,
            "outcome": ["AN"] * 4,
            "exposure": ["Aa_20yr"] * 4,
            "r_squared": [0.6, 0.4, 0.5, 0.98],
            "validation_r2": [0.6, 0.6, 0.5, 0.5],
            "test_r2": [np.nan, 0.4, np.nan, 0.98],
            "delta_r2": [0.3, 0.02, 0.25, 0.99],
        }
    )

    board = rank_leaderboards(registry)["erosion_increment_best"]

    assert board.iloc[0]["candidate_id"] == "validation_winner"
    assert board.iloc[0]["evaluation_set"] == "validation"
    assert board.iloc[0]["delta_r2"] == pytest.approx(0.3)
    assert board.iloc[0]["test_delta_r2"] == pytest.approx(0.02)
