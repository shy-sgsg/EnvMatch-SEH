from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
import yaml


def _api(module_name: str, name: str):
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        pytest.fail(f"{module_name} has not been implemented")
    function = getattr(module, name, None)
    assert callable(function), f"{module_name}.{name} has not been implemented"
    return function


def _record(experiment_id: str, status: str):
    return {
        "experiment_id": experiment_id,
        "git_sha": "abc1234",
        "input_run_id": "run_20261025T114741_470772Z",
        "exposure": "Aa_20yr",
        "health_indicator": "AN",
        "outcome_guided_selection": True,
        "selection_outcomes": ["AN"],
        "selection_objective": "maximize_validation_r2",
        "random_seed": 20261026,
        "status": status,
        "n_cells": 40000,
        "error": "candidate failed" if status == "failed" else None,
    }


def test_experiment_id_is_stable_and_sequence_padded():
    make_experiment_id = _api("experiments.oracle_exploration.registry", "make_experiment_id")

    assert make_experiment_id("20261026", 7, "v1") == "oracle_20261026_0007_v1"
    assert make_experiment_id("20261026", 7, "v1") == make_experiment_id("20261026", 7, "v1")


def test_registry_appends_success_and_failure_records(tmp_path):
    record_experiment = _api("experiments.oracle_exploration.registry", "record_experiment")
    registry = tmp_path / "experiments.jsonl"

    record_experiment(registry, _record("oracle_20261026_0001_v1", "success"))
    record_experiment(registry, _record("oracle_20261026_0002_v2", "failed"))

    records = [json.loads(line) for line in registry.read_text().splitlines()]
    assert [record["status"] for record in records] == ["success", "failed"]
    assert all(record["outcome_guided_selection"] is True for record in records)
    assert all(record["selection_objective"] for record in records)
    assert all(record["input_run_id"] == "run_20261025T114741_470772Z" for record in records)
    assert all(record["git_sha"] == "abc1234" for record in records)
    assert records[1]["error"] == "candidate failed"


def test_registry_rejects_incomplete_record_without_appending(tmp_path):
    record_experiment = _api("experiments.oracle_exploration.registry", "record_experiment")
    registry = tmp_path / "experiments.jsonl"
    incomplete = _record("oracle_20261026_0001_v1", "success")
    incomplete.pop("selection_objective")

    with pytest.raises(ValueError, match="selection_objective"):
        record_experiment(registry, incomplete)

    assert not registry.exists()


def test_registry_can_label_an_unscreened_baseline_as_not_outcome_guided(tmp_path):
    record_experiment = _api("experiments.oracle_exploration.registry", "record_experiment")
    registry = tmp_path / "experiments.jsonl"
    baseline = _record("oracle_20261026_0003_v3", "success")
    baseline["outcome_guided_selection"] = False
    baseline["selection_objective"] = "describe_unscreened_full_sample_baseline"

    record_experiment(registry, baseline)

    saved = json.loads(registry.read_text().splitlines()[0])
    assert saved["outcome_guided_selection"] is False


def test_default_search_config_has_requested_sample_and_candidate_counts():
    load_search_config = _api("experiments.oracle_exploration.registry", "load_search_config")
    config_path = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "oracle_exploration"
        / "config"
        / "search.yaml"
    )

    config = load_search_config(config_path)

    assert config["mode"] == "EXPLORATORY_ORACLE"
    assert config["outcome_guided_selection"] is True
    assert config["seed"] == 20261026
    assert config["stages"]["stage1"]["sample_size"] == 40000
    assert config["stages"]["stage1"]["candidates_per_exposure"] == 100
    assert config["stages"]["stage2"]["max_configs"] == 20
    assert config["stages"]["stage2"]["sample_size"] == 100000
    assert config["stages"]["stage3"]["block_size_m"] == 5000
    assert config["matching"]["environment_calipers"] == [0.15, 0.25, 0.35, 0.5, 0.75, 1.0]
    assert config["matching"]["geographic_radii_m"] == [100, 200, 300, 500, 750, 1000, 1500]
    assert config["matching"]["erosion_rank_gaps"] == [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
