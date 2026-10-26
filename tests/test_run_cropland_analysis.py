from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin


def _runner():
    module = importlib.import_module("scripts.run_cropland_analysis")
    function = getattr(module, "_run_feasibility", None)
    assert callable(function), "run_cropland_analysis._run_feasibility has not been implemented"
    return function


def test_g4_feasibility_writes_candidate_diagnostics_without_health_inputs(tmp_path):
    run_dir = tmp_path / "run_g4"
    for folder in ("tables", "figures", "maps/aligned", "matched_pairs", "diagnostics", "databases"):
        (run_dir / folder).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(20261025)
    rows = []
    for soil_type in (26, 27):
        for index in range(40):
            cell_id = len(rows)
            rows.append(
                {
                    "cell_id": cell_id,
                    "row": cell_id // 10,
                    "col": cell_id % 10,
                    "x": 500_015 + (cell_id % 10) * 30,
                    "y": 4_000_000 - (cell_id // 10) * 30,
                    "SoilType": soil_type,
                    "Elevation": soil_type * 10 + index + rng.normal(0, 0.1),
                    "Cur": rng.normal(0, 1),
                    "TWI": rng.normal(5, 1),
                    "LST": rng.normal(20, 2),
                    "Aa_20yr": float(index * 3 + rng.uniform(0, 1)),
                    "Aa_2020": float((39 - index) * 2 + rng.uniform(0, 1)),
                    "Aa_20yr_nearest": float(index * 3 + rng.uniform(0, 1)),
                    "Aa_2020_nearest": float((39 - index) * 2 + rng.uniform(0, 1)),
                }
            )
    pd.DataFrame(rows).to_parquet(run_dir / "databases" / "environment_erosion.parquet", index=False)

    mask_path = run_dir / "maps" / "aligned" / "cropland_mask_binary.tif"
    profile = {
        "driver": "GTiff",
        "height": 8,
        "width": 10,
        "count": 1,
        "dtype": "uint8",
        "crs": "EPSG:32649",
        "transform": from_origin(500_000, 4_000_000, 30, 30),
        "nodata": 255,
    }
    with rasterio.open(mask_path, "w", **profile) as dataset:
        dataset.write(np.ones((8, 10), dtype=np.uint8), 1)

    _runner()(run_dir, {"analysis": {}})

    assert (run_dir / "g4_report.md").is_file()
    candidate_summary = pd.read_csv(run_dir / "tables" / "g4_candidate_summary.csv")
    assert set(candidate_summary["exposure_resampling"]) == {"area_average", "nearest"}
    assert set(candidate_summary["candidate_k_per_soil_type"].unique()) == {6, 8, 10, 12}
    assert len(candidate_summary) == 18
    baseline = candidate_summary.loc[candidate_summary["setting_type"] == "baseline"]
    assert set(baseline["radius_m"].unique()) == {300.0, 500.0, 1000.0}
    assert (run_dir / "tables" / "environment_groups.csv").is_file()
    assert len(list((run_dir / "matched_pairs").glob("*.parquet"))) > 0
    assert len(list((run_dir / "maps").glob("environment_groups_k*.tif"))) == 4

