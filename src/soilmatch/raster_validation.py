"""Raster metadata and value checks for aligned gridded inputs."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio


def assert_aligned(reference: Any, candidate: Any, *, tolerance: float = 1e-8) -> None:
    """Raise ``ValueError`` unless two open raster datasets share a grid."""
    if reference.crs is None or candidate.crs is None:
        raise ValueError("CRS is missing from one or both rasters")
    if reference.crs != candidate.crs:
        raise ValueError(f"CRS mismatch: {reference.crs!s} != {candidate.crs!s}")
    if (reference.width, reference.height) != (candidate.width, candidate.height):
        raise ValueError(
            "Grid shape mismatch: "
            f"{reference.width}x{reference.height} != "
            f"{candidate.width}x{candidate.height}"
        )
    reference_transform = np.asarray(tuple(reference.transform)[:6], dtype=float)
    candidate_transform = np.asarray(tuple(candidate.transform)[:6], dtype=float)
    if not np.allclose(reference_transform, candidate_transform, rtol=0, atol=tolerance):
        raise ValueError(
            f"Grid transform mismatch: {reference.transform!s} != {candidate.transform!s}"
        )


def assert_metric_crs(dataset: Any) -> None:
    """Require a projected raster CRS whose linear unit is metres."""
    if dataset.crs is None:
        raise ValueError("CRS is missing; a projected metre-based CRS is required")
    if not dataset.crs.is_projected:
        raise ValueError(f"CRS must be projected for metre-based distance checks: {dataset.crs!s}")
    units = (dataset.crs.linear_units or "").strip().lower()
    if units not in {"metre", "meter", "metres", "meters"}:
        raise ValueError(f"CRS linear units must be metres, got {dataset.crs.linear_units!r}")


def inspect_rasters(
    paths: Mapping[str, str | Path | None],
    metadata: Mapping[str, Mapping[str, object]] | None = None,
) -> pd.DataFrame:
    """Inspect raster grids and stream value counts/statistics by block."""
    metadata = metadata or {}
    records: list[dict[str, object]] = []
    for logical_name, raw_path in paths.items():
        path = Path(raw_path) if raw_path is not None else None
        row: dict[str, object] = {
            "logical_name": logical_name,
            "path": str(path) if path is not None else None,
            "exists": bool(path and path.is_file()),
            "unit": metadata.get(logical_name, {}).get("unit"),
            "is_rusle_factor": metadata.get(logical_name, {}).get("is_rusle_factor"),
            "used_for_matching": metadata.get(logical_name, {}).get("used_for_matching"),
        }
        if not row["exists"]:
            row["error"] = "missing path"
            records.append(row)
            continue

        try:
            with rasterio.open(path) as dataset:
                row.update(
                    crs=dataset.crs.to_string() if dataset.crs else None,
                    width=dataset.width,
                    height=dataset.height,
                    resolution_x=dataset.res[0],
                    resolution_y=dataset.res[1],
                    transform=tuple(dataset.transform)[:6],
                    nodata=dataset.nodata,
                    dtype=dataset.dtypes[0],
                    block_size=dataset.block_shapes[0],
                )
                nodata_count = 0
                nonfinite_count = 0
                valid_count = 0
                zero_count = 0
                valid_min = np.inf
                valid_max = -np.inf
                for _, window in dataset.block_windows(1):
                    values = dataset.read(1, window=window, masked=True)
                    mask = np.ma.getmaskarray(values)
                    data = np.asarray(values.data)
                    finite = np.isfinite(data)
                    valid = ~mask & finite
                    nodata_count += int(mask.sum())
                    nonfinite_count += int((~mask & ~finite).sum())
                    if valid.any():
                        valid_values = data[valid]
                        valid_count += int(valid_values.size)
                        zero_count += int(np.count_nonzero(valid_values == 0))
                        valid_min = min(valid_min, float(valid_values.min()))
                        valid_max = max(valid_max, float(valid_values.max()))
                row.update(
                    nodata_count=nodata_count,
                    nonfinite_count=nonfinite_count,
                    valid_count=valid_count,
                    zero_count=zero_count,
                    zero_fraction=zero_count / valid_count if valid_count else np.nan,
                    valid_min=float(valid_min) if valid_count else np.nan,
                    valid_max=float(valid_max) if valid_count else np.nan,
                    error=None,
                )
        except (OSError, rasterio.errors.RasterioError, ValueError) as error:
            row["error"] = str(error)
        records.append(row)
    return pd.DataFrame.from_records(records)


def export_quality_report(report: pd.DataFrame, path: str | Path) -> Path:
    """Write a raster inspection table as CSV and return its path."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(destination, index=False)
    return destination


def check_rusle_consistency(
    observed: np.ndarray,
    factors: Mapping[str, np.ndarray],
    *,
    scenario: str | None = None,
    relative_tolerance: float = 0.05,
    absolute_tolerance: float = 0.0,
) -> dict[str, object]:
    """Compare an erosion raster with the product of its supplied RUSLE factors."""
    if scenario is None:
        scenario = "actual" if "C" in factors or "P" in factors else "potential"
    if scenario not in {"actual", "potential"}:
        raise ValueError("scenario must be 'actual' or 'potential'")
    required = ("R", "K", "LS", "C", "P") if scenario == "actual" else ("R", "K", "LS")
    missing = [name for name in required if name not in factors]
    if missing:
        raise ValueError(f"Missing {scenario} RUSLE factors: {', '.join(missing)}")

    observed_array = np.asarray(observed, dtype=float)
    arrays = [np.asarray(factors[name], dtype=float) for name in required]
    if any(array.shape != observed_array.shape for array in arrays):
        raise ValueError("Observed erosion and RUSLE factors must have identical shapes")
    reconstructed = np.ones(observed_array.shape, dtype=float)
    for array in arrays:
        reconstructed *= array
    valid = np.isfinite(observed_array) & np.isfinite(reconstructed)
    valid &= np.logical_and.reduce([np.isfinite(array) for array in arrays])
    if not valid.any():
        return {
            "scenario": scenario,
            "formula": "".join(required),
            "compared_count": 0,
            "within_tolerance_count": 0,
            "within_tolerance_fraction": np.nan,
            "mean_absolute_error": np.nan,
            "max_absolute_error": np.nan,
        }

    difference = np.abs(observed_array[valid] - reconstructed[valid])
    within = np.isclose(
        observed_array[valid],
        reconstructed[valid],
        rtol=relative_tolerance,
        atol=absolute_tolerance,
    )
    return {
        "scenario": scenario,
        "formula": "".join(required),
        "compared_count": int(valid.sum()),
        "within_tolerance_count": int(within.sum()),
        "within_tolerance_fraction": float(within.mean()),
        "mean_absolute_error": float(difference.mean()),
        "max_absolute_error": float(difference.max()),
    }
