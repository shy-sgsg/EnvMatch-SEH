from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin


def _api(name: str):
    try:
        module = importlib.import_module("soilmatch.grid_database")
    except ModuleNotFoundError:
        pytest.fail("soilmatch.grid_database has not been implemented")
    function = getattr(module, name, None)
    assert callable(function), f"grid_database.{name} has not been implemented"
    return function


def _write_raster(path, data, *, nodata=-9999):
    array = np.asarray(data)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=array.dtype,
        crs="EPSG:32649",
        transform=from_origin(400000, 3900000, 30, 30),
        nodata=nodata,
    ) as dst:
        dst.write(array, 1)


def test_database_preserves_cell_identity_mask_and_nodata(tmp_path):
    build_grid_database = _api("build_grid_database")
    erosion = tmp_path / "Aa.tif"
    environment = tmp_path / "Elevation.tif"
    mask = tmp_path / "mask.tif"
    _write_raster(erosion, np.array([[10, 20], [30, 40]], dtype="int16"))
    _write_raster(environment, np.array([[100, 200], [300, -9999]], dtype="int16"))
    _write_raster(mask, np.array([[1, 0], [1, 1]], dtype="uint8"), nodata=0)
    output = tmp_path / "grid.parquet"

    result_path = build_grid_database(
        {"Aa": erosion, "Elevation": environment},
        output,
        mask=mask,
        batch_size=2,
    )
    frame = pd.read_parquet(result_path).sort_values("cell_id").reset_index(drop=True)

    assert frame["cell_id"].tolist() == [0, 2, 3]
    assert frame[["row", "col"]].values.tolist() == [[0, 0], [1, 0], [1, 1]]
    assert frame["Aa"].tolist() == [10, 30, 40]
    assert frame["valid_cell"].tolist() == [True, True, False]
    assert frame.loc[2, "Elevation"] != frame.loc[2, "Elevation"]
    assert frame.loc[0, "x"] == pytest.approx(400015)
    assert frame.loc[0, "y"] == pytest.approx(3899985)


def test_database_refuses_to_overwrite_existing_output(tmp_path):
    build_grid_database = _api("build_grid_database")
    raster = tmp_path / "Aa.tif"
    mask = tmp_path / "mask.tif"
    _write_raster(raster, np.ones((1, 1), dtype="int16"))
    _write_raster(mask, np.ones((1, 1), dtype="uint8"), nodata=0)
    output = tmp_path / "grid.parquet"
    output.write_text("existing")

    with pytest.raises(FileExistsError, match="overwrite"):
        build_grid_database({"Aa": raster}, output, mask=mask)
