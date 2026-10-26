from __future__ import annotations

import struct

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin


def _write_raster(path, values, *, nodata=-128):
    array = np.asarray(values, dtype=np.int16)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype="int16",
        crs="EPSG:32649",
        transform=from_origin(500000, 3900000, 30, 30),
        nodata=nodata,
    ) as destination:
        destination.write(array, 1)


def _write_polygon_shapefile(path, rings):
    points = [point for ring in rings for point in ring]
    parts = []
    cursor = 0
    for ring in rings:
        parts.append(cursor)
        cursor += len(ring)
    coordinates = np.asarray(points, dtype="<f8")
    bbox = (
        float(coordinates[:, 0].min()),
        float(coordinates[:, 1].min()),
        float(coordinates[:, 0].max()),
        float(coordinates[:, 1].max()),
    )
    content = struct.pack("<i4d2i", 5, *bbox, len(parts), len(points))
    content += struct.pack("<" + "i" * len(parts), *parts)
    content += coordinates.tobytes()
    record = struct.pack(">2i", 1, len(content) // 2) + content
    header = struct.pack(">i5i", 9994, 0, 0, 0, 0, 0)
    header += struct.pack(">i", (100 + len(record)) // 2)
    header += struct.pack("<2i4d", 1000, 5, *bbox)
    header += struct.pack("<4d", 0.0, 0.0, 0.0, 0.0)
    path.write_bytes(header + record)


def test_binary_mask_uses_only_cropland_code_and_preserves_source_nodata(tmp_path):
    from soilmatch.cropland import create_binary_cropland_mask

    source = tmp_path / "landuse.tif"
    output = tmp_path / "cropland.tif"
    _write_raster(source, [[12, 21], [-128, 12]])

    result = create_binary_cropland_mask(source, output, cropland_codes={12})

    with rasterio.open(output) as dataset:
        values = dataset.read(1, masked=True)
        assert values.filled(0).tolist() == [[1, 0], [0, 1]]
        assert np.ma.getmaskarray(values).tolist() == [[False, False], [True, False]]
    assert result["cropland_count"] == 2
    assert result["excluded_valid_count"] == 1
    assert result["source_nodata_count"] == 1


def test_polygon_shapefile_area_handles_hole_rings(tmp_path):
    from soilmatch.cropland import inspect_polygon_shapefile

    path = tmp_path / "polygon.shp"
    outer = [(0, 0), (0, 4), (4, 4), (4, 0), (0, 0)]
    hole = [(1, 1), (3, 1), (3, 3), (1, 3), (1, 1)]
    _write_polygon_shapefile(path, [outer, hole])

    result = inspect_polygon_shapefile(path)

    assert result["feature_count"] == 1
    assert result["ring_count"] == 2
    assert result["area_coordinate_units_squared"] == pytest.approx(12.0)


def test_polygon_shapefile_area_rejects_truncated_records(tmp_path):
    from soilmatch.cropland import inspect_polygon_shapefile

    path = tmp_path / "truncated.shp"
    path.write_bytes(b"short")

    with pytest.raises(ValueError, match="header"):
        inspect_polygon_shapefile(path)
