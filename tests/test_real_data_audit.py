from __future__ import annotations

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin


def test_duplicate_groups_only_include_byte_identical_files(tmp_path):
    from scripts.audit_real_data import find_duplicate_groups

    first = tmp_path / "first.bin"
    duplicate = tmp_path / "duplicate.bin"
    same_size_different = tmp_path / "different.bin"
    first.write_bytes(b"same contents")
    duplicate.write_bytes(b"same contents")
    same_size_different.write_bytes(b"other content")

    groups = find_duplicate_groups([first, duplicate, same_size_different])

    assert groups[first] == groups[duplicate]
    assert same_size_different not in groups


def test_raster_profile_counts_valid_zero_nodata_and_distribution(tmp_path):
    from scripts.audit_real_data import profile_raster

    path = tmp_path / "sample.tif"
    values = np.array([[0.0, 2.0, 4.0, -9999.0]], dtype="float32")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=1,
        width=4,
        count=1,
        dtype="float32",
        crs="EPSG:32649",
        transform=from_origin(400000, 3900000, 30, 30),
        nodata=-9999,
    ) as destination:
        destination.write(values, 1)

    result = profile_raster(path)

    assert result["valid_count"] == 3
    assert result["nodata_count"] == 1
    assert result["zero_count"] == 1
    assert result["valid_mean"] == pytest.approx(2.0)
    assert result["q50"] == pytest.approx(2.0)


def test_inventory_records_logical_inference_and_confirmation_state(tmp_path):
    from scripts.audit_real_data import build_inventory

    data_root = tmp_path / "data"
    actual_erosion = data_root / "侵蚀（及五因子）" / "实际侵蚀_wx.tif"
    actual_erosion.parent.mkdir(parents=True)
    actual_erosion.write_bytes(b"raster placeholder")

    inventory = build_inventory(data_root).set_index("relative_path")
    row = inventory.loc["侵蚀（及五因子）/实际侵蚀_wx.tif"]

    assert "actual erosion" in row["logical_inference"]
    assert row["confirmation_status"] == "user-selected"

