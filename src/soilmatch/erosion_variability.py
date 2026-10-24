"""Summaries of within-group erosion exposure and optional class composition."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def summarize_erosion(
    frame: pd.DataFrame,
    group_col: str,
    erosion_col: str,
    *,
    class_edges: Sequence[float] | None = None,
    class_labels: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Summarize raw and log1p erosion distributions by environment group.

    Erosion class counts are included only when explicit, study-approved edges
    are supplied; this function does not invent class thresholds.
    """
    missing = [column for column in (group_col, erosion_col) if column not in frame]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    if frame[group_col].isna().any():
        raise ValueError(f"{group_col} contains missing group identifiers")
    if (class_edges is None) != (class_labels is None):
        raise ValueError("class_edges and class_labels must be provided together")

    edges = None
    labels = None
    if class_edges is not None:
        edges = np.asarray(class_edges, dtype=float)
        labels = list(class_labels or ())
        if len(edges) != len(labels) + 1 or len(edges) < 2:
            raise ValueError("class_edges must contain exactly one more value than class_labels")
        if np.isnan(edges).any() or not np.all(np.diff(edges) > 0):
            raise ValueError("class_edges must be strictly increasing and contain no NaN")

    records: list[dict[str, object]] = []
    for group, subset in frame.groupby(group_col, sort=False, dropna=False):
        raw = pd.to_numeric(subset[erosion_col], errors="coerce").to_numpy(dtype=float)
        valid = raw[np.isfinite(raw)]
        if (valid < 0).any():
            raise ValueError(f"{erosion_col} must be nonnegative for finite values")
        logged = np.log1p(valid)
        count = len(valid)
        record: dict[str, object] = {
            group_col: group,
            "total_count": int(len(subset)),
            "valid_count": count,
            "zero_count": int(np.count_nonzero(valid == 0)),
            "zero_proportion": float(np.mean(valid == 0)) if count else np.nan,
            "mean": float(np.mean(valid)) if count else np.nan,
            "median": float(np.median(valid)) if count else np.nan,
            "std": float(np.std(valid, ddof=1)) if count > 1 else np.nan,
            "p5": float(np.quantile(valid, 0.05)) if count else np.nan,
            "p25": float(np.quantile(valid, 0.25)) if count else np.nan,
            "p75": float(np.quantile(valid, 0.75)) if count else np.nan,
            "p95": float(np.quantile(valid, 0.95)) if count else np.nan,
            "range": float(np.ptp(valid)) if count else np.nan,
            "log1p_mean": float(np.mean(logged)) if count else np.nan,
            "log1p_std": float(np.std(logged, ddof=1)) if count > 1 else np.nan,
            "log1p_p5": float(np.quantile(logged, 0.05)) if count else np.nan,
            "log1p_p25": float(np.quantile(logged, 0.25)) if count else np.nan,
            "log1p_p75": float(np.quantile(logged, 0.75)) if count else np.nan,
            "log1p_p95": float(np.quantile(logged, 0.95)) if count else np.nan,
        }
        if edges is not None and labels is not None:
            classes = pd.cut(valid, bins=edges, labels=labels, include_lowest=True)
            record["unclassified_count"] = int(classes.isna().sum())
            for label in labels:
                class_count = int(np.count_nonzero(classes == label))
                record[f"class_{label}_count"] = class_count
                record[f"class_{label}_proportion"] = class_count / count if count else np.nan
        records.append(record)
    return pd.DataFrame.from_records(records)
