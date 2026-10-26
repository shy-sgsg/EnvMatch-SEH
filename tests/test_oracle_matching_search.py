from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _api(name: str):
    try:
        module = importlib.import_module("experiments.oracle_exploration.matching_search")
    except ModuleNotFoundError:
        pytest.fail("experiments.oracle_exploration.matching_search has not been implemented")
    function = getattr(module, name, None)
    assert callable(function), f"matching_search.{name} has not been implemented"
    return function


def _config():
    from experiments.oracle_exploration.registry import load_search_config

    path = Path(__file__).resolve().parents[1] / "experiments/oracle_exploration/config/search.yaml"
    return load_search_config(path)


def _candidate():
    return {
        "candidate_id": "Aa_20yr_001",
        "experiment_id": "oracle_20261026_0001_Aa20yr001",
        "exposure": "Aa_20yr",
        "environment_variables": ["Elevation", "LST"],
        "vector_name": "V2",
        "distance_method": "standardized_euclidean",
        "cluster_method": "none",
        "cluster_count": 0,
        "environment_weights": None,
        "environment_caliper": 2.0,
        "geographic_radius_m": 1000,
        "erosion_rank_gap": 0.20,
        "exposure_rule": "within_group_rank",
        "outcome_guided_selection": True,
        "selection_outcomes": ["AN", "SHI_candidate"],
        "selection_objective": "evaluate_all_outcomes_r2_and_association",
        "input_run_id": "run_synthetic",
        "git_sha": "abc1234",
        "random_seed": 20261026,
    }


def _matching_frame():
    exposure = np.array([10.0, 9.0, 8.0, 7.0, 2.0, 1.0, 0.5, 0.0])
    log_exposure = np.log1p(exposure)
    return pd.DataFrame(
        {
            "cell_id": np.arange(1, 9),
            "x": np.arange(8, dtype=float) * 10.0,
            "y": np.zeros(8),
            "SoilType": [26] * 8,
            "Elevation": np.linspace(0.0, 0.07, 8),
            "LST": np.linspace(12.0, 12.2, 8),
            "Aa_20yr": exposure,
            "Aa_2020": exposure * 1.2,
            "AN": 3.0 * log_exposure + np.array([0.0, 0.1, -0.1, 0.0, 0.1, -0.1, 0.0, 0.05]),
            "SHI_candidate": -0.4 * log_exposure + np.array([0.01, -0.01, 0.0, 0.02, -0.02, 0.0, 0.01, -0.01]),
            "OM": np.linspace(1.0, 2.0, 8),
            "CEC": np.linspace(1.0, 3.0, 8),
            "pH": np.linspace(6.2, 7.0, 8),
            "SW": np.linspace(0.1, 0.3, 8),
            "BD": np.linspace(1.0, 1.3, 8),
            "MBC": np.linspace(20.0, 30.0, 8),
            "RESP": np.linspace(0.2, 0.8, 8),
        }
    )


@pytest.mark.parametrize("exposure", ["Aa_20yr", "Aa_2020"])
def test_draw_matching_candidates_is_deterministic_and_uses_exact_domains(exposure):
    draw_matching_candidates = _api("draw_matching_candidates")
    config = _config()

    first = draw_matching_candidates(config, exposure, seed=20261026)
    second = draw_matching_candidates(config, exposure, seed=20261026)
    candidate_ids = [candidate["candidate_id"] for candidate in first]

    assert len(first) == len(second) == 100
    assert first == second
    assert len(set(candidate_ids)) == 100
    assert {candidate["exposure"] for candidate in first} == {exposure}
    assert {candidate["environment_caliper"] for candidate in first} <= set(
        config["matching"]["environment_calipers"]
    )
    assert {candidate["geographic_radius_m"] for candidate in first} <= set(
        config["matching"]["geographic_radii_m"]
    )
    assert {candidate["erosion_rank_gap"] for candidate in first} <= set(
        config["matching"]["erosion_rank_gaps"]
    )
    assert {candidate["distance_method"] for candidate in first} <= set(
        config["matching"]["distance_methods"]
    )
    assert all(candidate["outcome_guided_selection"] is True for candidate in first)
    assert all(candidate["selection_outcomes"] == config["outcomes"] for candidate in first)
    assert all(candidate["cluster_count"] == 0 for candidate in first if candidate["cluster_method"] == "none")
    assert all(candidate["cluster_method"] == "none" for candidate in first if candidate["distance_method"] == "direct_nearest_neighbor")


def test_matching_candidate_scores_fixed_pairs_and_records_actual_exposures(tmp_path):
    run_matching_candidate = _api("run_matching_candidate")
    frame = _matching_frame()
    candidate = _candidate()
    registry = tmp_path / "experiments.jsonl"

    result = run_matching_candidate(frame, candidate, ["AN", "SHI_candidate"], registry)

    assert result.status == "success"
    assert not result.pairs.empty
    pair_cells = result.pairs["high_erosion_cell_id"].tolist() + result.pairs["low_erosion_cell_id"].tolist()
    assert len(pair_cells) == len(set(pair_cells))
    assert (result.pairs["erosion_high"] >= result.pairs["erosion_low"]).all()
    assert (result.pairs["exposure_percentile_gap"] >= candidate["erosion_rank_gap"]).all()
    assert (result.pairs["environmental_distance"] <= candidate["environment_caliper"]).all()
    assert (result.pairs["geographic_distance"] <= candidate["geographic_radius_m"]).all()
    assert {"Aa_20yr_high", "Aa_20yr_low", "AN_high", "AN_low"}.issubset(result.pairs.columns)
    assert {"Elevation_high", "Elevation_low", "LST_high", "LST_low"}.issubset(result.pairs.columns)

    cells = result.pairs["high_erosion_cell_id"].tolist() + result.pairs["low_erosion_cell_id"].tolist()
    matched = frame.set_index("cell_id").loc[cells]
    expected_r2 = np.corrcoef(np.log1p(matched["Aa_20yr"]), matched["AN"])[0, 1] ** 2
    score = result.scores.loc[
        result.scores["outcome"].eq("AN") & result.scores["score_set"].eq("matched_cells"), "r_squared"
    ].iloc[0]
    assert score == pytest.approx(expected_r2)
    assert set(result.scores["outcome"]) == {"AN", "SHI_candidate"}

    records = [json.loads(line) for line in registry.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["outcome_guided_selection"] is True
    assert records[0]["selection_outcomes"] == ["AN", "SHI_candidate"]
    assert records[0]["selection_objective"] == candidate["selection_objective"]
    assert records[0]["status"] == "success"
    assert records[0]["n_cells"] == 8
    assert records[0]["n_matched_cells"] == len(set(pair_cells))
    assert records[0]["area_coverage"] == pytest.approx(len(set(pair_cells)) / 8)
    assert records[0]["median_erosion_difference"] == pytest.approx(
        result.pairs["erosion_difference"].median()
    )
    for variable in ("Elevation", "LST"):
        high = result.pairs[f"{variable}_high"].to_numpy(dtype=float)
        low = result.pairs[f"{variable}_low"].to_numpy(dtype=float)
        pooled_sd = np.sqrt((np.var(high, ddof=1) + np.var(low, ddof=1)) / 2)
        expected_smd = (high.mean() - low.mean()) / pooled_sd
        assert records[0]["matched_smd"][variable] == pytest.approx(expected_smd)


def test_failed_matching_candidate_is_preserved_with_cause(tmp_path):
    run_matching_candidate = _api("run_matching_candidate")
    candidate = _candidate()
    candidate["environment_variables"] = ["not_an_environment_variable"]
    registry = tmp_path / "experiments.jsonl"

    result = run_matching_candidate(_matching_frame(), candidate, ["AN"], registry)

    assert result.status == "failed"
    assert result.error
    record = json.loads(registry.read_text().splitlines()[0])
    assert record["status"] == "failed"
    assert record["error"]
    assert record["experiment_id"] == candidate["experiment_id"]


def test_stage1_ranking_is_separate_by_exposure_outcome_and_score_set():
    rank_stage1_candidates = _api("rank_stage1_candidates")
    scores = pd.DataFrame(
        {
            "candidate_id": ["a", "b", "c", "d"],
            "experiment_id": ["ea", "eb", "ec", "ed"],
            "exposure": ["Aa_20yr", "Aa_20yr", "Aa_20yr", "Aa_2020"],
            "outcome": ["AN", "AN", "OM", "AN"],
            "score_set": ["matched_cells"] * 4,
            "r_squared": [0.4, 0.8, 0.6, 0.7],
            "test_r2": [0.99, 0.01, 0.1, 0.2],
        }
    )

    ranked = rank_stage1_candidates(scores)

    assert ranked.loc[ranked["outcome"].eq("AN") & ranked["exposure"].eq("Aa_20yr"), "candidate_id"].tolist() == ["b", "a"]
    assert dict(zip(ranked["candidate_id"], ranked["rank"], strict=True)) == {
        "a": 2,
        "b": 1,
        "c": 1,
        "d": 1,
    }
    assert set(ranked["selection_mode"]) == {"outcome_guided_oracle_apparent"}
