from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest


def _api(name: str):
    try:
        module = importlib.import_module("soilmatch.pcnm_comparison")
    except ModuleNotFoundError:
        pytest.fail("soilmatch.pcnm_comparison has not been implemented")
    function = getattr(module, name, None)
    assert callable(function), f"pcnm_comparison.{name} has not been implemented"
    return function


def test_pcnm_comparison_preserves_missing_fields_and_compares_direction():
    compare_pcnm = _api("compare_pcnm")
    pcnm = pd.DataFrame(
        {"indicator": ["SHI"], "scenario": ["actual"], "beta": [-0.2]}
    )
    matched = pd.DataFrame(
        {
            "outcome": ["SHI"],
            "erosion_scenario": ["actual"],
            "method": ["matched_pairs"],
            "coefficient": [-0.4],
            "standardized_coefficient": [-0.3],
        }
    )

    result = compare_pcnm(
        pcnm,
        matched,
        {"outcome": "indicator", "erosion_scenario": "scenario", "coefficient": "beta"},
    ).iloc[0]

    assert result["coefficient_pcnm"] == pytest.approx(-0.2)
    assert result["coefficient_environmental"] == pytest.approx(-0.4)
    assert result["standardized_coefficient_pcnm"] != result["standardized_coefficient_pcnm"]
    assert result["direction_consistent"]


def test_missing_pcnm_table_does_not_block_matched_results():
    compare_pcnm = _api("compare_pcnm")
    matched = pd.DataFrame(
        {"outcome": ["AN"], "erosion_scenario": ["actual"], "coefficient": [0.1]}
    )

    result = compare_pcnm(pd.DataFrame(), matched)

    assert len(result) == 1
    assert result.iloc[0]["outcome"] == "AN"
    assert result.iloc[0]["status"] == "pcnm_missing"
    assert np.isnan(result.iloc[0]["coefficient_pcnm"])


def test_pcnm_reader_accepts_csv_and_excel(tmp_path):
    read_pcnm_results = _api("read_pcnm_results")
    source = pd.DataFrame({"indicator": ["SHI"], "beta": [0.2]})
    csv_path = tmp_path / "pcnm.csv"
    xlsx_path = tmp_path / "pcnm.xlsx"
    source.to_csv(csv_path, index=False)
    source.to_excel(xlsx_path, index=False)

    pd.testing.assert_frame_equal(read_pcnm_results(csv_path), source)
    pd.testing.assert_frame_equal(read_pcnm_results(xlsx_path), source)


def test_indicator_mapping_preserves_ph_case():
    compare_pcnm = _api("compare_pcnm")
    pcnm = pd.DataFrame(
        {"outcome": ["pH"], "erosion_scenario": ["Aa"], "coefficient": [0.1]}
    )
    matched = pd.DataFrame(
        {"outcome": ["pH"], "erosion_scenario": ["actual"], "coefficient": [0.2]}
    )

    result = compare_pcnm(pcnm, matched).iloc[0]

    assert result["outcome"] == "pH"
    assert result["erosion_scenario"] == "actual"
