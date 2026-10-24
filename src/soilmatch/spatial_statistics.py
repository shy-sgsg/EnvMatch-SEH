"""Spatial-block identifiers and block bootstrap resampling."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd


def make_spatial_blocks(x, y, block_size: float) -> np.ndarray:
    """Map projected coordinates to deterministic square block identifiers."""
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    if x_values.ndim != 1 or y_values.ndim != 1 or x_values.shape != y_values.shape:
        raise ValueError("x and y must be one-dimensional arrays with equal length")
    if not np.isfinite(x_values).all() or not np.isfinite(y_values).all():
        raise ValueError("coordinates must be finite")
    if not np.isfinite(block_size) or block_size <= 0:
        raise ValueError("block_size must be finite and greater than zero")
    keys = np.column_stack((np.floor(x_values / block_size), np.floor(y_values / block_size)))
    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
    return inverse.astype(np.int64)


def block_bootstrap(
    frame: pd.DataFrame,
    block_col: str,
    fit_function: Callable[[pd.DataFrame], object],
    *,
    n_boot: int = 999,
    random_state: int = 0,
) -> pd.DataFrame:
    """Resample whole spatial blocks with replacement and collect estimates."""
    if block_col not in frame:
        raise ValueError(f"Missing spatial block column: {block_col}")
    if frame[block_col].isna().any():
        raise ValueError("spatial block identifiers must not be missing")
    if not isinstance(n_boot, int) or n_boot < 1:
        raise ValueError("n_boot must be a positive integer")
    blocks = list(pd.unique(frame[block_col]))
    if len(blocks) < 2:
        return pd.DataFrame(
            [{"replicate": np.nan, "status": "insufficient_blocks", "n_blocks": len(blocks)}]
        )

    rng = np.random.default_rng(random_state)
    block_frames = {block: frame.loc[frame[block_col] == block] for block in blocks}
    records: list[dict[str, object]] = []
    for replicate in range(n_boot):
        sampled = rng.choice(blocks, size=len(blocks), replace=True)
        sample = pd.concat([block_frames[block] for block in sampled], ignore_index=True)
        estimate = fit_function(sample)
        if isinstance(estimate, Mapping):
            record = dict(estimate)
        elif isinstance(estimate, pd.Series):
            record = estimate.to_dict()
        else:
            record = {"estimate": estimate}
        record.update(replicate=replicate, status=record.get("status", "ok"), n_blocks=len(blocks))
        records.append(record)
    return pd.DataFrame.from_records(records)


def summarize_bootstrap(
    results: pd.DataFrame,
    estimate_columns: list[str],
    *,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Summarize bootstrap estimate distributions with percentile intervals."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    missing = [column for column in estimate_columns if column not in results]
    if missing:
        raise ValueError(f"Bootstrap results are missing estimate columns: {', '.join(missing)}")
    valid_results = results
    if "status" in results:
        valid_results = results.loc[results["status"] == "ok"]
    records = []
    for column in estimate_columns:
        values = pd.to_numeric(valid_results[column], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if len(values):
            lower, upper = np.quantile(values, [alpha / 2, 1 - alpha / 2])
            mean = float(np.mean(values))
            standard_deviation = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
        else:
            lower = upper = mean = standard_deviation = np.nan
        records.append(
            {
                "estimate": column,
                "n_bootstrap": int(len(values)),
                "estimate_mean": mean,
                "estimate_std": standard_deviation,
                "ci_lower": float(lower),
                "ci_upper": float(upper),
                "confidence_level": 1 - alpha,
            }
        )
    return pd.DataFrame.from_records(records)
