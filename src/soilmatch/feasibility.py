"""Helpers for scale-independent feasibility diagnostics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def add_within_group_exposure_rank(
    frame: pd.DataFrame,
    *,
    exposure_column: str,
    group_columns: Sequence[str],
    output_column: str = "exposure_percentile",
) -> pd.DataFrame:
    """Return a copy with average percentile ranks within the given groups.

    Missing exposure values stay missing. Finite erosion values must be
    nonnegative. Percentile ranks make exploratory exposure gaps independent
    of undocumented raster units.
    """
    groups = list(group_columns)
    if not groups:
        raise ValueError("At least one group column is required")
    missing_groups = [column for column in groups if column not in frame]
    if missing_groups:
        raise ValueError(f"Missing group columns: {', '.join(missing_groups)}")
    if exposure_column not in frame:
        raise ValueError(f"Missing exposure column: {exposure_column}")
    if output_column in frame:
        raise ValueError(f"Output column already exists: {output_column}")
    if frame.loc[:, groups].isna().any().any():
        raise ValueError("Group columns contain missing values")

    exposure = pd.to_numeric(frame[exposure_column], errors="coerce")
    values = exposure.to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError(f"{exposure_column} contains infinite values")
    if (values[np.isfinite(values)] < 0).any():
        raise ValueError(f"{exposure_column} must be nonnegative")

    result = frame.copy()
    rank_frame = result.loc[:, groups].copy()
    rank_frame["__exposure"] = exposure
    result[output_column] = rank_frame.groupby(groups, sort=False)["__exposure"].rank(
        method="average", pct=True
    )
    return result
