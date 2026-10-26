from __future__ import annotations

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin


def _write_raster(path, values, *, transform, nodata=-9999):
    array = np.asarray(values, dtype=np.float32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:32649",
        transform=transform,
        nodata=nodata,
    ) as destination:
        destination.write(array, 1)


def _align():
    from soilmatch.raster_alignment import align_raster_to_template

    return align_raster_to_template


def test_alignment_preserves_categorical_codes_with_nearest_neighbor(tmp_path):
    source = tmp_path / "soil.tif"
    template = tmp_path / "template.tif"
    output = tmp_path / "soil_aligned.tif"
    transform = from_origin(500000, 3900000, 30, 30)
    _write_raster(source, [[12, 21], [12, 21]], transform=transform)
    _write_raster(template, [[1, 1], [1, 1]], transform=transform)

    _align()(source, template, output, resampling="nearest")

    with rasterio.open(output) as dataset:
        values = dataset.read(1, masked=True).compressed()
        assert set(values.tolist()) == {12.0, 21.0}
        assert dataset.transform == transform
        assert dataset.crs.to_epsg() == 32649


def test_alignment_keeps_nodata_missing_instead_of_turning_it_into_zero(tmp_path):
    source = tmp_path / "source.tif"
    template = tmp_path / "template.tif"
    output = tmp_path / "aligned.tif"
    transform = from_origin(500000, 3900000, 30, 30)
    _write_raster(source, [[3, -9999], [7, 9]], transform=transform)
    _write_raster(template, [[1, 1], [1, 1]], transform=transform)

    _align()(source, template, output, resampling="nearest")

    with rasterio.open(output) as dataset:
        values = dataset.read(1, masked=True)
        assert np.ma.getmaskarray(values).sum() == 1
        assert 0 not in values.compressed()


def test_average_resampling_aggregates_source_cell_values(tmp_path):
    source = tmp_path / "source.tif"
    template = tmp_path / "template.tif"
    output = tmp_path / "aligned.tif"
    _write_raster(source, [[1, 2], [3, 4]], transform=from_origin(500000, 3900000, 1, 1))
    _write_raster(template, [[1]], transform=from_origin(500000, 3900000, 2, 2))

    _align()(source, template, output, resampling="average")

    with rasterio.open(output) as dataset:
        assert dataset.read(1)[0, 0] == pytest.approx(2.5)


def test_alignment_does_not_overwrite_existing_output_without_permission(tmp_path):
    source = tmp_path / "source.tif"
    template = tmp_path / "template.tif"
    output = tmp_path / "aligned.tif"
    transform = from_origin(500000, 3900000, 30, 30)
    _write_raster(source, [[1]], transform=transform)
    _write_raster(template, [[1]], transform=transform)
    output.write_bytes(b"existing")

    with pytest.raises(FileExistsError):
        _align()(source, template, output, resampling="nearest")
