from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin


def _api(name: str):
    try:
        module = importlib.import_module("soilmatch.raster_validation")
    except ModuleNotFoundError:
        pytest.fail("soilmatch.raster_validation has not been implemented")
    function = getattr(module, name, None)
    assert callable(function), f"raster_validation.{name} has not been implemented"
    return function


def _write_raster(path, data, *, transform=None, crs="EPSG:32649", nodata=-9999):
    array = np.asarray(data)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=array.dtype,
        crs=crs,
        transform=transform or from_origin(400000, 3900000, 30, 30),
        nodata=nodata,
    ) as dst:
        dst.write(array, 1)


def test_alignment_accepts_same_grid_and_rejects_shifted_grid(tmp_path):
    assert_aligned = _api("assert_aligned")
    reference = tmp_path / "reference.tif"
    same = tmp_path / "same.tif"
    shifted = tmp_path / "shifted.tif"
    _write_raster(reference, np.ones((2, 2), dtype="float32"))
    _write_raster(same, np.ones((2, 2), dtype="float32"))
    _write_raster(
        shifted,
        np.ones((2, 2), dtype="float32"),
        transform=from_origin(400001, 3900000, 30, 30),
    )

    with rasterio.open(reference) as ref, rasterio.open(same) as candidate:
        assert_aligned(ref, candidate) is None
    with rasterio.open(reference) as ref, rasterio.open(shifted) as candidate:
        with pytest.raises(ValueError, match="transform|grid"):
            assert_aligned(ref, candidate)


def test_alignment_rejects_missing_or_different_crs_and_grid_shape(tmp_path):
    assert_aligned = _api("assert_aligned")
    reference = tmp_path / "reference-crs.tif"
    no_crs = tmp_path / "no-crs.tif"
    no_crs_reference = tmp_path / "no-crs-reference.tif"
    different_crs = tmp_path / "different-crs.tif"
    different_shape = tmp_path / "different-shape.tif"
    _write_raster(reference, np.ones((2, 2), dtype="float32"))
    _write_raster(no_crs, np.ones((2, 2), dtype="float32"), crs=None)
    _write_raster(no_crs_reference, np.ones((2, 2), dtype="float32"), crs=None)
    _write_raster(different_crs, np.ones((2, 2), dtype="float32"), crs="EPSG:4326")
    _write_raster(different_shape, np.ones((1, 2), dtype="float32"))

    with rasterio.open(reference) as ref, rasterio.open(no_crs) as candidate:
        with pytest.raises(ValueError, match="CRS"):
            assert_aligned(ref, candidate)
    with rasterio.open(no_crs_reference) as ref, rasterio.open(no_crs) as candidate:
        with pytest.raises(ValueError, match="CRS"):
            assert_aligned(ref, candidate)
    with rasterio.open(reference) as ref, rasterio.open(different_crs) as candidate:
        with pytest.raises(ValueError, match="CRS"):
            assert_aligned(ref, candidate)
    with rasterio.open(reference) as ref, rasterio.open(different_shape) as candidate:
        with pytest.raises(ValueError, match="shape"):
            assert_aligned(ref, candidate)


def test_metric_crs_requires_projected_coordinates_in_metres(tmp_path):
    assert_metric_crs = _api("assert_metric_crs")
    projected = tmp_path / "metric.tif"
    geographic = tmp_path / "geographic.tif"
    _write_raster(projected, np.ones((1, 1), dtype="float32"), crs="EPSG:32649")
    _write_raster(geographic, np.ones((1, 1), dtype="float32"), crs="EPSG:4326")

    with rasterio.open(projected) as dataset:
        assert_metric_crs(dataset) is None
    with rasterio.open(geographic) as dataset:
        with pytest.raises(ValueError, match="projected|metre"):
            assert_metric_crs(dataset)


def test_inspection_counts_nodata_and_nonfinite_values(tmp_path):
    inspect_rasters = _api("inspect_rasters")
    raster = tmp_path / "quality.tif"
    _write_raster(
        raster,
        np.array([[1.0, np.nan, -5.0, -9999.0]], dtype="float32"),
    )

    report = inspect_rasters({"Aa": raster}, {"Aa": {"unit": "t/ha/year"}})

    row = report.set_index("logical_name").loc["Aa"]
    assert row["nodata_count"] == 1
    assert row["nonfinite_count"] == 1
    assert row["valid_count"] == 2
    assert row["valid_min"] == -5.0
    assert row["valid_max"] == 1.0
    assert row["unit"] == "t/ha/year"


def test_inspection_reports_zero_erosion_fraction(tmp_path):
    inspect_rasters = _api("inspect_rasters")
    raster = tmp_path / "erosion-zeroes.tif"
    _write_raster(raster, np.array([[0.0, 0.0, 2.0, -9999.0]], dtype="float32"))

    row = inspect_rasters({"Aa": raster}, {}).iloc[0]

    assert row["zero_count"] == 2
    assert row["zero_fraction"] == pytest.approx(2 / 3)


def test_rusle_consistency_reports_product_error():
    check_rusle_consistency = _api("check_rusle_consistency")
    result = check_rusle_consistency(
        np.array([1.5, 6.0]),
        {"R": [2.0, 3.0], "K": [1.0, 1.0], "LS": [1.0, 1.0], "C": [0.5, 2.0], "P": [1.0, 1.0]},
    )

    assert result["compared_count"] == 2
    assert result["within_tolerance_count"] == 1
    assert result["within_tolerance_fraction"] == pytest.approx(0.5)


def test_resolve_path_preserves_absolute_local_data_path(tmp_path):
    from soilmatch.config import resolve_path

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_path = config_dir / "default.yaml"
    absolute_path = (tmp_path / "local" / "Aa.tif").resolve()

    assert resolve_path(absolute_path, config_path) == absolute_path

