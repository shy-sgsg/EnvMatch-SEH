from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest


def _api(name: str):
    try:
        module = importlib.import_module("experiments.oracle_exploration.data")
    except ModuleNotFoundError:
        pytest.fail("experiments.oracle_exploration.data has not been implemented")
    function = getattr(module, name, None)
    assert callable(function), f"data.{name} has not been implemented"
    return function


def _write_run(tmp_path, env: pd.DataFrame, health: pd.DataFrame):
    databases = tmp_path / "databases"
    databases.mkdir()
    env.to_parquet(databases / "environment_erosion.parquet", index=False)
    health.to_parquet(databases / "health.parquet", index=False)
    return tmp_path


def _source_tables():
    env = pd.DataFrame(
        {
            "cell_id": [10, 11, 12],
            "row": [1, 1, 2],
            "col": [1, 2, 1],
            "x": [100.0, 130.0, 100.0],
            "y": [200.0, 200.0, 170.0],
            "Elevation": [10.0, 11.0, 12.0],
            "SoilType": [26, 26, 27],
            "Aa_20yr": [0.0, 1.0, 2.0],
            "Aa_2020": [0.0, 2.0, 3.0],
            "valid_cell": [True, True, True],
        }
    )
    health = pd.DataFrame(
        {
            "cell_id": [10, 11, 12],
            "row": [1, 1, 2],
            "col": [1, 2, 1],
            "x": [100.0, 130.0, 100.0],
            "y": [200.0, 200.0, 170.0],
            "AN": [2.0, np.nan, 4.0],
            "OM": [4.0, 5.0, 6.0],
            "CEC": [2.0, 3.0, 4.0],
            "pH": [6.0, 7.0, 8.0],
            "SW": [0.2, 0.3, 0.4],
            "BD": [1.1, 1.2, 1.3],
            "MBC": [10.0, 11.0, 12.0],
            "RESP": [1.0, 2.0, 3.0],
            "SHI_candidate": [0.4, 0.5, np.nan],
            "valid_cell": [True, False, False],
        }
    )
    return env, health


def test_load_oracle_data_joins_once_and_keeps_outcome_missingness(tmp_path):
    load_oracle_data = _api("load_oracle_data")
    env, health = _source_tables()
    run_dir = _write_run(tmp_path, env, health)

    bundle = load_oracle_data(run_dir)

    assert len(bundle.frame) == 3
    assert bundle.frame["cell_id"].tolist() == [10, 11, 12]
    assert bundle.frame["SHI_candidate"].isna().tolist() == [False, False, True]
    assert np.isnan(bundle.frame.loc[1, "AN"])
    assert "SHI" not in bundle.frame
    assert "SHI_candidate" in bundle.outcomes
    assert bundle.frame["environment_valid_cell"].all()
    assert bundle.frame["health_valid_cell"].tolist() == [True, False, False]
    assert bundle.input_run_id == tmp_path.name


def test_load_oracle_data_rejects_duplicate_cell_ids(tmp_path):
    load_oracle_data = _api("load_oracle_data")
    env, health = _source_tables()
    env = pd.concat([env, env.iloc[[0]]], ignore_index=True)
    run_dir = _write_run(tmp_path, env, health)

    with pytest.raises(ValueError, match="unique.*cell_id|duplicate.*cell_id"):
        load_oracle_data(run_dir)


def test_load_oracle_data_rejects_nonmatching_cell_id_sets(tmp_path):
    load_oracle_data = _api("load_oracle_data")
    env, health = _source_tables()
    health.loc[2, "cell_id"] = 13
    run_dir = _write_run(tmp_path, env, health)

    with pytest.raises(ValueError, match="cell_id.*match|matching.*cell_id"):
        load_oracle_data(run_dir)


def test_load_oracle_data_rejects_misaligned_coordinates(tmp_path):
    load_oracle_data = _api("load_oracle_data")
    env, health = _source_tables()
    health.loc[1, "x"] += 1.0
    run_dir = _write_run(tmp_path, env, health)

    with pytest.raises(ValueError, match="Spatial alignment|coordinate"):
        load_oracle_data(run_dir)
