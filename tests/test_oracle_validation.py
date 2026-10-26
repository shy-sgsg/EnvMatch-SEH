from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest


def _api(name: str):
    try:
        module = importlib.import_module("experiments.oracle_exploration.validation")
    except ModuleNotFoundError:
        pytest.fail("experiments.oracle_exploration.validation has not been implemented")
    function = getattr(module, name, None)
    assert callable(function), f"validation.{name} has not been implemented"
    return function


def test_spatial_roles_are_reproducible_and_split_whole_blocks_approximately_60_20_20():
    from soilmatch.spatial_statistics import make_spatial_blocks

    spatial_roles = _api("spatial_roles")
    x, y = np.meshgrid(np.arange(5) * 5000.0 + 10, np.arange(4) * 5000.0 + 10)
    frame = pd.DataFrame({"x": x.ravel(), "y": y.ravel()})

    roles = spatial_roles(frame, block_size_m=5000, seed=20261026)
    repeated = spatial_roles(frame, block_size_m=5000, seed=20261026)
    blocks = make_spatial_blocks(frame["x"], frame["y"], 5000)

    assert roles.equals(repeated)
    assert set(roles) == {"train", "validation", "test"}
    block_roles = pd.DataFrame({"block": blocks, "role": roles}).groupby("block")["role"].nunique()
    assert block_roles.eq(1).all()
    role_block_counts = pd.DataFrame({"block": blocks, "role": roles}).drop_duplicates().groupby("role").size()
    assert role_block_counts.to_dict() == {"test": 4, "train": 12, "validation": 4}


def test_spatial_roles_require_projected_finite_coordinates_and_enough_blocks():
    spatial_roles = _api("spatial_roles")
    with pytest.raises(ValueError, match="finite|coordinate"):
        spatial_roles(pd.DataFrame({"x": [0.0, np.nan], "y": [0.0, 10.0]}), seed=1)
    with pytest.raises(ValueError, match="three|3|blocks"):
        spatial_roles(pd.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0]}), seed=1)


def test_candidate_selection_uses_validation_score_and_rejects_test_score_selection():
    select_validation_candidates = _api("select_validation_candidates")
    candidates = pd.DataFrame(
        {
            "candidate_id": ["a", "b", "c"],
            "validation_r2": [0.8, 0.4, 0.6],
            "test_r2": [0.1, 0.99, 0.5],
        }
    )

    selected = select_validation_candidates(candidates, max_configs=2)

    assert selected["candidate_id"].tolist() == ["a", "c"]
    candidates["test_r2"] = [0.0, 1.0, 0.5]
    repeated = select_validation_candidates(candidates, max_configs=2)
    assert repeated["candidate_id"].tolist() == ["a", "c"]
    with pytest.raises(ValueError, match="train_r2|validation_r2"):
        select_validation_candidates(candidates, max_configs=2, score_column="test_r2")
