"""Spatial summaries for zero, low-positive, main and uncertain Aa regimes."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

from experiments.low_erosion_regime.distribution_diagnostics import classify_regimes


REGIME_CODES = {"zero": 0, "low_positive": 1, "main": 2, "uncertain": 3}


def regime_labels(
    values: np.ndarray, *, threshold: float | None, uncertain: np.ndarray | None = None,
) -> np.ndarray:
    labels = classify_regimes(values, threshold=threshold)
    if uncertain is not None:
        uncertainty = np.asarray(uncertain, dtype=bool)
        if len(uncertainty) != len(labels):
            raise ValueError("uncertain mask must have same length as values")
        labels[uncertainty & np.isin(labels, ["low_positive", "main"])] = "uncertain"
    return labels


def regime_grid(
    cells: pd.DataFrame, *, exposure: str, threshold: float | None,
    uncertain_cell_ids: set[int] | None = None,
) -> tuple[np.ndarray, tuple[float, float, float, float], np.ndarray]:
    """Create a masked row/column regime raster and projected-coordinate extent."""
    required = {"cell_id", "row", "col", "x", "y", "valid_cell", exposure}
    missing = sorted(required - set(cells.columns))
    if missing:
        raise ValueError(f"cells missing spatial raster columns: {missing}")
    frame = cells.copy()
    valid = frame["valid_cell"].fillna(False).astype(bool).to_numpy()
    values = pd.to_numeric(frame[exposure], errors="coerce").to_numpy(dtype=float)
    valid &= np.isfinite(values)
    uncertainty = frame["cell_id"].isin(uncertain_cell_ids or set()).to_numpy()
    labels = regime_labels(values, threshold=threshold, uncertain=uncertainty)
    row_values = pd.to_numeric(frame["row"], errors="coerce").to_numpy(dtype=int)
    col_values = pd.to_numeric(frame["col"], errors="coerce").to_numpy(dtype=int)
    rmin, rmax = int(row_values.min()), int(row_values.max())
    cmin, cmax = int(col_values.min()), int(col_values.max())
    grid = np.full((rmax - rmin + 1, cmax - cmin + 1), -1, dtype=np.int8)
    selected = np.flatnonzero(valid)
    grid[row_values[selected] - rmin, col_values[selected] - cmin] = np.fromiter(
        (REGIME_CODES[label] for label in labels[selected]), dtype=np.int8, count=len(selected)
    )
    x = pd.to_numeric(frame["x"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(frame["y"], errors="coerce").to_numpy(dtype=float)
    extent = (float(np.nanmin(x) - 15), float(np.nanmax(x) + 15),
              float(np.nanmin(y) - 15), float(np.nanmax(y) + 15))
    return grid, extent, valid


def connected_components(cells: pd.DataFrame, *, exposure: str, threshold: float) -> tuple[pd.DataFrame, np.ndarray]:
    """Label 8-connected low-positive patches on the original raster lattice."""
    frame = cells.copy()
    values = pd.to_numeric(frame[exposure], errors="coerce").to_numpy(dtype=float)
    rows = pd.to_numeric(frame["row"], errors="coerce").to_numpy(dtype=int)
    cols = pd.to_numeric(frame["col"], errors="coerce").to_numpy(dtype=int)
    valid = frame["valid_cell"].fillna(False).astype(bool).to_numpy() & np.isfinite(values)
    low = valid & (values > 0) & (values <= threshold)
    rmin, rmax = int(rows.min()), int(rows.max())
    cmin, cmax = int(cols.min()), int(cols.max())
    mask = np.zeros((rmax - rmin + 1, cmax - cmin + 1), dtype=bool)
    mask[rows[low] - rmin, cols[low] - cmin] = True
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    counts = np.bincount(labels.ravel(), minlength=count + 1)
    component_sizes = counts[1:]
    records = [{"exposure": exposure, "component_id": i + 1, "pixel_count": int(n),
                "area_m2": int(n) * 900, "area_ha": int(n) * .09}
               for i, n in enumerate(component_sizes) if n > 0]
    summary = pd.DataFrame.from_records(records)
    return summary, labels


def summarize_low_shares(
    cells: pd.DataFrame, *, exposure: str, threshold: float | None,
    group_columns: tuple[str, ...] = ("SoilType", "Elevation", "TWI", "LST", "NDWI"),
) -> pd.DataFrame:
    """Report weighted low-positive shares by soil and environmental profiles."""
    frame = cells.copy()
    values = pd.to_numeric(frame[exposure], errors="coerce")
    valid = frame["valid_cell"].fillna(False).astype(bool) & values.notna() & np.isfinite(values)
    frame = frame.loc[valid].copy()
    frame["__value"] = values.loc[valid].to_numpy(dtype=float)
    frame["__zero"] = frame["__value"].eq(0).astype(int)
    frame["__low_positive"] = (
        ((frame["__value"] > 0) & (frame["__value"] <= threshold)).astype(float)
        if threshold is not None else np.nan
    )
    records = []
    for variable in group_columns:
        if variable not in frame:
            continue
        series = frame[variable]
        profile = pd.Series(index=frame.index, dtype="object")
        soil_values = pd.to_numeric(frame["SoilType"], errors="coerce")
        if variable == "SoilType":
            profile[:] = series.map(lambda x: f"土类 {int(x)}" if pd.notna(x) else "缺失")
        else:
            for soil_code in sorted(soil_values.dropna().unique()):
                mask = soil_values.eq(soil_code)
                numeric = pd.to_numeric(series.loc[mask], errors="coerce")
                labels = pd.qcut(numeric, q=5, labels=False, duplicates="drop")
                profile.loc[mask] = labels.map(
                    lambda q: f"土类 {int(soil_code)} · Q{int(q) + 1}" if pd.notna(q) else "缺失"
                )
        grouped_input = pd.DataFrame({"profile": profile.fillna("缺失"), "low": frame["__low_positive"],
                                      "zero": frame["__zero"], "n": 1})
        grouped = grouped_input.groupby("profile", observed=True).agg(
            low_positive_n=("low", "sum"), zero_n=("zero", "sum"), valid_n=("n", "sum")
        ).reset_index()
        grouped["exposure"] = exposure
        grouped["group_variable"] = variable
        grouped["low_positive_fraction"] = grouped["low_positive_n"] / grouped["valid_n"]
        grouped["zero_fraction"] = grouped["zero_n"] / grouped["valid_n"]
        grouped["threshold_original"] = threshold
        records.extend(grouped.to_dict("records"))
    return pd.DataFrame.from_records(records)

