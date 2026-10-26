"""Explicit, read-only raster reprojection onto a reference grid."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from soilmatch.raster_validation import assert_metric_crs


_RESAMPLING = {
    "nearest": Resampling.nearest,
    "bilinear": Resampling.bilinear,
    "average": Resampling.average,
}
_SOURCE_NODATA = float(np.finfo(np.float32).min)


def align_raster_to_template(
    source: str | Path,
    template: str | Path,
    destination: str | Path,
    *,
    resampling: str,
    output_nodata: float = -9999.0,
    overwrite: bool = False,
) -> dict[str, object]:
    """Reproject one raster onto ``template`` without modifying either input.

    ``nearest`` is suitable for masks and categories, ``bilinear`` for
    continuous surfaces, and ``average`` for cell-area aggregation of
    continuous exposure rasters. NoData and nonfinite source values remain
    missing; they are never converted to zero.
    """
    source_path = Path(source)
    template_path = Path(template)
    output_path = Path(destination)
    if resampling not in _RESAMPLING:
        raise ValueError(f"Unsupported resampling method: {resampling!r}")
    if not np.isfinite(output_nodata):
        raise ValueError("output_nodata must be finite")
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    if not template_path.is_file():
        raise FileNotFoundError(template_path)
    if output_path.resolve() in {source_path.resolve(), template_path.resolve()}:
        raise ValueError("destination must not overwrite the source or template raster")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists; set overwrite=True to replace it: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(source_path) as source_dataset, rasterio.open(template_path) as template_dataset:
        if source_dataset.crs is None:
            raise ValueError(f"Source raster has no CRS: {source_path}")
        assert_metric_crs(template_dataset)

        masked = source_dataset.read(1, masked=True)
        source_values = np.asarray(masked.data, dtype=np.float32)
        source_valid = ~np.ma.getmaskarray(masked) & np.isfinite(source_values)
        source_values[~source_valid] = _SOURCE_NODATA
        destination_values = np.full(
            (template_dataset.height, template_dataset.width),
            output_nodata,
            dtype=np.float32,
        )
        reproject(
            source=source_values,
            destination=destination_values,
            src_transform=source_dataset.transform,
            src_crs=source_dataset.crs,
            src_nodata=_SOURCE_NODATA,
            dst_transform=template_dataset.transform,
            dst_crs=template_dataset.crs,
            dst_nodata=output_nodata,
            resampling=_RESAMPLING[resampling],
            init_dest_nodata=True,
        )

        profile = template_dataset.profile.copy()
        profile.update(
            driver="GTiff",
            count=1,
            dtype="float32",
            nodata=output_nodata,
            compress="deflate",
            predictor=3,
        )
        with rasterio.open(output_path, "w", **profile) as output_dataset:
            output_dataset.write(destination_values, 1)
            output_dataset.update_tags(
                source=str(source_path),
                template=str(template_path),
                resampling=resampling,
                source_nodata=str(source_dataset.nodata),
            )

    output_valid = np.isfinite(destination_values) & (destination_values != output_nodata)
    return {
        "source": str(source_path),
        "template": str(template_path),
        "destination": str(output_path),
        "resampling": resampling,
        "source_valid_count": int(source_valid.sum()),
        "destination_valid_count": int(output_valid.sum()),
        "destination_total_count": int(destination_values.size),
    }
