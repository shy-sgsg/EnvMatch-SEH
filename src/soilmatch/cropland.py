"""Cropland-mask construction and polygon geometry checks."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Iterable

import numpy as np
import rasterio


def create_binary_cropland_mask(
    source: str | Path,
    destination: str | Path,
    *,
    cropland_codes: Iterable[int],
    overwrite: bool = False,
) -> dict[str, object]:
    """Create a 1/0/NoData mask from explicitly identified source classes."""
    source_path = Path(source)
    output_path = Path(destination)
    codes = {int(value) for value in cropland_codes}
    if not codes:
        raise ValueError("cropland_codes must contain at least one class code")
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if output_path.resolve() == source_path.resolve():
        raise ValueError("destination must not overwrite the source raster")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists; set overwrite=True to replace it: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(source_path) as source_dataset:
        masked = source_dataset.read(1, masked=True)
        values = np.asarray(masked.data)
        source_valid = ~np.ma.getmaskarray(masked) & np.isfinite(values)
        selected = source_valid & np.isin(values, tuple(codes))
        output = np.zeros(values.shape, dtype=np.uint8)
        output[selected] = 1
        output[~source_valid] = 255
        profile = source_dataset.profile.copy()
        profile.update(
            driver="GTiff",
            count=1,
            dtype="uint8",
            nodata=255,
            compress="deflate",
        )
        with rasterio.open(output_path, "w", **profile) as destination_dataset:
            destination_dataset.write(output, 1)
            destination_dataset.update_tags(
                source=str(source_path),
                selected_cropland_codes=",".join(map(str, sorted(codes))),
            )

    unique, counts = np.unique(values[source_valid], return_counts=True)
    return {
        "source": str(source_path),
        "destination": str(output_path),
        "cropland_codes": sorted(codes),
        "source_valid_count": int(source_valid.sum()),
        "cropland_count": int(selected.sum()),
        "excluded_valid_count": int((source_valid & ~selected).sum()),
        "source_nodata_count": int((~source_valid).sum()),
        "source_code_counts": dict(zip(unique.tolist(), counts.astype(int).tolist(), strict=True)),
    }


def inspect_polygon_shapefile(path: str | Path) -> dict[str, object]:
    """Read polygon rings sequentially from a .shp when its .shx/.dbf are absent.

    The returned area is in squared coordinate units. This function cannot
    establish a CRS when the shapefile has no .prj file.
    """
    source = Path(path)
    data = source.read_bytes()
    if len(data) < 100:
        raise ValueError("Shapefile header is shorter than 100 bytes")
    if struct.unpack(">i", data[:4])[0] != 9994:
        raise ValueError("Invalid shapefile file code")
    declared_size = struct.unpack(">i", data[24:28])[0] * 2
    version, shape_type = struct.unpack("<2i", data[28:36])
    if version != 1000:
        raise ValueError(f"Unsupported shapefile version: {version}")
    if shape_type != 5:
        raise ValueError(f"Expected Polygon (shape type 5), got {shape_type}")
    if declared_size < 100 or declared_size > len(data):
        raise ValueError("Shapefile declared length exceeds available bytes")
    bounds = struct.unpack("<4d", data[36:68])

    offset = 100
    feature_count = 0
    ring_count = 0
    point_count = 0
    area = 0.0
    while offset < declared_size:
        if offset + 8 > declared_size:
            raise ValueError("Truncated shapefile record header")
        _, content_words = struct.unpack(">2i", data[offset : offset + 8])
        content_size = content_words * 2
        content_start = offset + 8
        content_end = content_start + content_size
        if content_words < 2 or content_end > declared_size:
            raise ValueError("Truncated shapefile record content")
        content = data[content_start:content_end]
        record_type = struct.unpack("<i", content[:4])[0]
        if record_type == 0:
            offset = content_end
            continue
        if record_type != 5:
            raise ValueError(f"Unsupported polygon record type: {record_type}")
        if len(content) < 44:
            raise ValueError("Truncated polygon record header")
        part_total, point_total = struct.unpack("<2i", content[36:44])
        parts_end = 44 + part_total * 4
        points_end = parts_end + point_total * 16
        if part_total < 1 or point_total < 4 or points_end > len(content):
            raise ValueError("Invalid or truncated polygon coordinates")
        parts = struct.unpack("<" + "i" * part_total, content[44:parts_end])
        if parts[0] != 0 or any(left >= right for left, right in zip(parts, parts[1:])):
            raise ValueError("Invalid polygon ring offsets")
        points = np.frombuffer(content, dtype="<f8", count=2 * point_total, offset=parts_end).reshape(-1, 2)
        ends = (*parts[1:], point_total)
        feature_area = 0.0
        for start, end in zip(parts, ends, strict=True):
            ring = points[start:end]
            if len(ring) < 4 or not np.array_equal(ring[0], ring[-1]):
                raise ValueError("Polygon ring is not a closed coordinate sequence")
            cross = ring[:, 0] * np.roll(ring[:, 1], -1) - np.roll(ring[:, 0], -1) * ring[:, 1]
            feature_area += 0.5 * float(cross.sum())
        area += abs(feature_area)
        feature_count += 1
        ring_count += part_total
        point_count += point_total
        offset = content_end

    if offset != declared_size:
        raise ValueError("Shapefile records do not end at the declared file length")
    return {
        "path": str(source),
        "shape_type": shape_type,
        "feature_count": feature_count,
        "ring_count": ring_count,
        "point_count": point_count,
        "bounds": bounds,
        "area_coordinate_units_squared": area,
    }
