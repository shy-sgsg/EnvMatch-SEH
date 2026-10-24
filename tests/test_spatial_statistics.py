from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest


def _api(name: str):
    try:
        module = importlib.import_module("soilmatch.spatial_statistics")
    except ModuleNotFoundError:
        pytest.fail("soilmatch.spatial_statistics has not been implemented")
    function = getattr(module, name, None)
    assert callable(function), f"spatial_statistics.{name} has not been implemented"
    return function


def test_spatial_blocks_are_deterministic_and_keep_same_block_cells_together():
    make_spatial_blocks = _api("make_spatial_blocks")

    labels = make_spatial_blocks([5, 25, 25, 25], [5, 25, 5, 25], block_size=20)

    np.testing.assert_array_equal(labels, [0, 2, 1, 2])


def test_block_bootstrap_resamples_whole_blocks_and_reports_insufficiency():
    block_bootstrap = _api("block_bootstrap")
    frame = pd.DataFrame(
        {"block": [0, 0, 1, 1, 2, 2], "value": [1, 2, 3, 4, 5, 6]}
    )

    results = block_bootstrap(
        frame,
        "block",
        lambda sample: {
            f"count_{block}": int((sample["block"] == block).sum())
            for block in (0, 1, 2)
        },
        n_boot=5,
        random_state=3,
    )
    insufficient = block_bootstrap(
        frame.loc[frame["block"] == 0],
        "block",
        lambda sample: {
            f"count_{block}": int((sample["block"] == block).sum())
            for block in (0, 1, 2)
        },
        n_boot=5,
    )

    assert len(results) == 5
    assert (results["status"] == "ok").all()
    assert results["n_blocks"].eq(3).all()
    for column in ("count_0", "count_1", "count_2"):
        assert results[column].isin([0, 2, 4, 6]).all()
    assert insufficient.iloc[0]["status"] == "insufficient_blocks"
    assert "count_0" not in insufficient


def test_spatial_blocks_reject_invalid_coordinates_or_size():
    make_spatial_blocks = _api("make_spatial_blocks")
    with pytest.raises(ValueError, match="finite"):
        make_spatial_blocks([0, np.nan], [0, 1], block_size=10)
    with pytest.raises(ValueError, match="block_size"):
        make_spatial_blocks([0], [0], block_size=0)


def test_bootstrap_summary_calculates_percentile_confidence_interval():
    summarize_bootstrap = _api("summarize_bootstrap")
    results = pd.DataFrame(
        {"status": ["ok"] * 4 + ["failed"], "coefficient": [1.0, 2.0, 3.0, 4.0, 100.0]}
    )

    summary = summarize_bootstrap(results, ["coefficient"], alpha=0.5).iloc[0]

    assert summary["n_bootstrap"] == 4
    assert summary["estimate_mean"] == pytest.approx(2.5)
    assert summary["ci_lower"] == pytest.approx(1.75)
    assert summary["ci_upper"] == pytest.approx(3.25)
