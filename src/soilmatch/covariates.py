"""Outcome-independent environment covariate quality and redundancy checks."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


_RUSLE_FACTORS = {"R", "K", "LS", "C", "P"}
_HEALTH_RESULTS = {"SHI", "AN", "OM", "CEC", "PH", "SW", "BD", "MBC", "RESP"}


def screen_covariates(
    frame: pd.DataFrame,
    candidates: Sequence[str],
    *,
    max_missing_fraction: float = 0.05,
    correlation_threshold: float = 0.95,
) -> pd.DataFrame:
    """Summarize candidate environmental variables without reading outcomes.

    The screen reports missingness, variation, and environment-only Pearson
    redundancy. It does not automatically remove variables or use health
    responses to select a set.
    """
    if not 0 <= max_missing_fraction <= 1:
        raise ValueError("max_missing_fraction must be between zero and one")
    if not 0 <= correlation_threshold <= 1:
        raise ValueError("correlation_threshold must be between zero and one")

    rows: list[dict[str, object]] = []
    numeric_columns: dict[str, np.ndarray] = {}
    for name in dict.fromkeys(candidates):
        upper_name = str(name).strip().upper()
        row: dict[str, object] = {
            "logical_name": name,
            "status": "candidate",
            "n_total": len(frame),
            "n_valid": 0,
            "missing_fraction": np.nan,
            "unique_count": 0,
            "mean": np.nan,
            "std": np.nan,
            "minimum": np.nan,
            "maximum": np.nan,
            "most_correlated_with": None,
            "max_abs_correlation": np.nan,
            "high_correlation": False,
        }
        if upper_name in _RUSLE_FACTORS:
            row["status"] = "excluded_rusle_factor"
        elif upper_name in _HEALTH_RESULTS:
            row["status"] = "excluded_health_result"
        elif name not in frame:
            row["status"] = "missing_column"
        elif not pd.api.types.is_numeric_dtype(frame[name]):
            row["status"] = "non_numeric"
        else:
            numeric = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
            finite = np.isfinite(numeric)
            valid = numeric[finite]
            row["n_valid"] = int(len(valid))
            row["missing_fraction"] = 1.0 - (len(valid) / len(frame)) if len(frame) else np.nan
            row["unique_count"] = int(np.unique(valid).size)
            if not len(valid):
                row["status"] = "all_missing"
            else:
                row["mean"] = float(np.mean(valid))
                row["std"] = float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0
                row["minimum"] = float(np.min(valid))
                row["maximum"] = float(np.max(valid))
                if row["missing_fraction"] > max_missing_fraction:
                    row["status"] = "high_missingness"
                elif row["unique_count"] <= 1:
                    row["status"] = "constant"
                    row["std"] = 0.0
                else:
                    row["status"] = "candidate"
                    numeric_columns[name] = numeric
        rows.append(row)

    result = pd.DataFrame.from_records(rows)
    eligible = [name for name in numeric_columns if name in result.loc[result["status"] == "candidate", "logical_name"].tolist()]
    for name in eligible:
        correlations = []
        values = numeric_columns[name]
        for other in eligible:
            if other == name:
                continue
            other_values = numeric_columns[other]
            valid = np.isfinite(values) & np.isfinite(other_values)
            if valid.sum() < 2 or np.std(values[valid]) == 0 or np.std(other_values[valid]) == 0:
                continue
            correlations.append((other, abs(float(np.corrcoef(values[valid], other_values[valid])[0, 1]))))
        if correlations:
            other, value = max(correlations, key=lambda pair: pair[1])
            index = result.index[result["logical_name"] == name][0]
            result.loc[index, "most_correlated_with"] = other
            result.loc[index, "max_abs_correlation"] = value
            result.loc[index, "high_correlation"] = value >= correlation_threshold
    return result
