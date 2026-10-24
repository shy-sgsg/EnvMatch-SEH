"""Covariate balance diagnostics before and after environment matching."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


_BALANCE_COLUMNS = [
    "covariate",
    "unmatched_smd",
    "matched_smd",
    "mean_absolute_pair_difference",
    "n_pairs",
]


def _standardized_mean_difference(high: np.ndarray, low: np.ndarray) -> float:
    high = high[np.isfinite(high)]
    low = low[np.isfinite(low)]
    if not len(high) or not len(low):
        return np.nan
    variance_high = np.var(high, ddof=1) if len(high) > 1 else 0.0
    variance_low = np.var(low, ddof=1) if len(low) > 1 else 0.0
    pooled_scale = np.sqrt((variance_high + variance_low) / 2.0)
    difference = float(np.mean(high) - np.mean(low))
    if pooled_scale == 0:
        return 0.0 if difference == 0 else float(np.sign(difference) * np.inf)
    return difference / pooled_scale


def assess_balance(
    frame: pd.DataFrame,
    pairs: pd.DataFrame,
    covariates: Sequence[str],
    *,
    erosion_col: str | None = None,
) -> pd.DataFrame:
    """Report standardized differences in exposure strata and matched pairs.

    The unmatched comparison uses all cells at least as exposed as the least
    exposed matched high cell versus all cells no more exposed than the most
    exposed matched low cell. This rule is returned as transparent threshold
    columns so the population represented by the diagnostic is inspectable.
    """
    if pairs.empty:
        return pd.DataFrame(columns=_BALANCE_COLUMNS)
    if "cell_id" in frame:
        indexed = frame.set_index("cell_id", drop=False)
    else:
        indexed = frame.copy()
        indexed["__balance_cell_id"] = indexed.index
        indexed = indexed.set_index("__balance_cell_id", drop=False)
    required_pair_columns = {
        "high_erosion_cell_id",
        "low_erosion_cell_id",
        "erosion_high",
        "erosion_low",
    }
    missing_pairs = sorted(required_pair_columns - set(pairs.columns))
    if missing_pairs:
        raise ValueError(f"Pairs are missing balance fields: {', '.join(missing_pairs)}")
    if erosion_col is None:
        erosion_col = "Aa" if "Aa" in frame else "erosion"
    if erosion_col not in frame:
        raise ValueError(f"Missing erosion column for unmatched balance: {erosion_col}")
    missing_covariates = [name for name in covariates if name not in frame]
    if missing_covariates:
        raise ValueError(f"Missing balance covariates: {', '.join(missing_covariates)}")

    high_ids = pairs["high_erosion_cell_id"].tolist()
    low_ids = pairs["low_erosion_cell_id"].tolist()
    if not set(high_ids + low_ids).issubset(set(indexed.index)):
        raise ValueError("A matched pair cell identifier is absent from the input frame")
    matched_high = indexed.loc[high_ids]
    matched_low = indexed.loc[low_ids]
    high_cutoff = float(pd.to_numeric(pairs["erosion_high"]).min())
    low_cutoff = float(pd.to_numeric(pairs["erosion_low"]).max())
    exposure = pd.to_numeric(frame[erosion_col], errors="coerce")
    unmatched_high = frame.loc[exposure >= high_cutoff]
    unmatched_low = frame.loc[exposure <= low_cutoff]

    records = []
    for covariate in covariates:
        high_all = pd.to_numeric(unmatched_high[covariate], errors="coerce").to_numpy(dtype=float)
        low_all = pd.to_numeric(unmatched_low[covariate], errors="coerce").to_numpy(dtype=float)
        high_matched = pd.to_numeric(matched_high[covariate], errors="coerce").to_numpy(dtype=float)
        low_matched = pd.to_numeric(matched_low[covariate], errors="coerce").to_numpy(dtype=float)
        valid_pairs = np.isfinite(high_matched) & np.isfinite(low_matched)
        pair_differences = np.abs(high_matched[valid_pairs] - low_matched[valid_pairs])
        records.append(
            {
                "covariate": covariate,
                "unmatched_smd": _standardized_mean_difference(high_all, low_all),
                "matched_smd": _standardized_mean_difference(high_matched, low_matched),
                "mean_absolute_pair_difference": (
                    float(pair_differences.mean()) if len(pair_differences) else np.nan
                ),
                "n_pairs": int(valid_pairs.sum()),
                "unmatched_high_cutoff": high_cutoff,
                "unmatched_low_cutoff": low_cutoff,
            }
        )
    return pd.DataFrame.from_records(records)
