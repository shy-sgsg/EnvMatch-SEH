from __future__ import annotations

import importlib
import json

import numpy as np
import pandas as pd


def _api(module_name: str, name: str):
    module = importlib.import_module(f"experiments.oracle_exploration.{module_name}")
    function = getattr(module, name, None)
    assert callable(function), f"{module_name}.{name} has not been implemented"
    return function


def test_subgroup_masks_support_predicates_and_nested_combinations():
    build_subgroup_masks = _api("subgroup_search", "build_subgroup_masks")
    frame = pd.DataFrame(
        {
            "SoilType": [26, 26, 27, 42, 85],
            "Elevation": [5.0, 15.0, 25.0, 35.0, 45.0],
            "Aa_20yr": [0.0, 1.0, 2.0, 0.0, 4.0],
            "environment_group": [0, 1, 1, 2, 3],
        }
    )
    definitions = {
        "soil_26": {"column": "SoilType", "op": "eq", "value": 26},
        "selected_soils": {"column": "SoilType", "op": "in", "values": [26, 42]},
        "elevation_band": {"column": "Elevation", "op": "between", "lower": 10, "upper": 35},
        "positive_high_env": {
            "all": [
                {"column": "Aa_20yr", "op": "gt", "value": 1},
                {"column": "environment_group", "op": "ge", "value": 1},
            ]
        },
    }

    masks = build_subgroup_masks(frame, definitions)

    assert masks["soil_26"].tolist() == [True, True, False, False, False]
    assert masks["selected_soils"].tolist() == [True, True, False, True, False]
    assert masks["elevation_band"].tolist() == [False, True, True, True, False]
    assert masks["positive_high_env"].tolist() == [False, False, True, False, True]
    assert all(mask.dtype == bool and len(mask) == len(frame) for mask in masks.values())


def test_oracle_retention_search_records_exact_sizes_and_selection_rules():
    oracle_retention_search = _api("subgroup_search", "oracle_retention_search")
    rng = np.random.default_rng(123)
    predictor = np.linspace(0.0, 10.0, 100)
    frame = pd.DataFrame(
        {
            "cell_id": np.arange(100),
            "Aa_20yr": predictor,
            "AN": 1.7 * predictor + rng.normal(0.0, 1.0, 100),
        }
    )
    frame.loc[3, "AN"] = np.nan

    result = oracle_retention_search(
        frame,
        "AN",
        ["Aa_20yr"],
        fractions=[1.0, 0.5, 0.25],
        trials=20,
        seed=9,
    )

    assert result["target_retention_fraction"].tolist() == [1.0, 0.5, 0.25]
    assert result["n_complete_cases"].eq(99).all()
    assert result["n_selected"].tolist() == [99, 50, 25]
    assert result["outcome_guided_selection"].eq(True).all()
    assert result["selection_mode"].eq("outcome_selected_oracle").all()
    assert result["trial_count"].ge(1).all()
    assert result["coefficient"].gt(0).all()
    assert result["direction"].eq("positive").all()
    for value in result["selected_cell_ids_json"]:
        assert len(json.loads(value)) in {99, 50, 25}


def test_unknown_subgroup_predicates_fail_clearly():
    build_subgroup_masks = _api("subgroup_search", "build_subgroup_masks")
    frame = pd.DataFrame({"x": [1, 2]})
    try:
        build_subgroup_masks(frame, {"bad": {"column": "x", "op": "approx", "value": 1}})
    except ValueError as error:
        assert "operator" in str(error).lower() or "op" in str(error).lower()
    else:
        raise AssertionError("unsupported subgroup operators must be rejected")


def test_experiment_subgroup_search_builds_environment_group_pairs_and_triplets():
    build_definitions = _api("experiment", "_subgroup_definitions")
    frame = pd.DataFrame(
        {
            "SoilType": [26] * 12 + [27] * 12 + [85] * 12,
            "Aa_20yr": np.tile(np.arange(1, 13, dtype=float), 3),
            "environment_group": np.repeat([0, 1, 2, 3, 4, 5], 6),
        }
    )

    definitions = build_definitions(
        frame,
        "Aa_20yr",
        "environment_group",
        minimum_group_size=2,
    )

    assert definitions["environment_group_pair_0_1"] == {
        "column": "environment_group",
        "op": "in",
        "values": [0, 1],
    }
    assert definitions["environment_group_triple_0_1_2"]["values"] == [0, 1, 2]
    assert "soil_85" not in definitions
    assert not any(name.startswith("soil_85_") for name in definitions)
    assert definitions["soil_26_environment_group_pair_0_1"]["all"] == [
        {"column": "SoilType", "op": "eq", "value": 26},
        {"column": "environment_group", "op": "in", "values": [0, 1]},
    ]
