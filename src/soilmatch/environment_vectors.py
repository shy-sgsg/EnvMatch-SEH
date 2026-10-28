"""Build standardized environment-only vectors and comparable distances."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


def build_environment_vectors(
    frame: pd.DataFrame,
    *,
    continuous: Sequence[str],
    categorical: Sequence[str] = (),
    weights: Mapping[str, float] | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Standardize continuous covariates and keep categories as strata metadata.

    Missing and nonfinite values are rejected rather than silently imputed.
    Weights multiply squared standardized distances, so the matrix is scaled by
    each weight's square root before use in Euclidean clustering.
    """
    columns = list(continuous)
    category_columns = list(categorical)
    if not columns:
        raise ValueError("At least one continuous environment variable is required")
    if len(set(columns + category_columns)) != len(columns) + len(category_columns):
        raise ValueError("Environment column names must be unique")
    missing_columns = [name for name in columns + category_columns if name not in frame]
    if missing_columns:
        raise ValueError(f"Missing environment columns: {', '.join(missing_columns)}")

    raw = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(raw).all():
        raise ValueError("Environment continuous values contain missing or nonfinite values")
    if category_columns and frame.loc[:, category_columns].isna().any().any():
        raise ValueError("Categorical environment strata contain missing values")

    unknown_weights = set(weights or {}) - set(columns)
    if unknown_weights:
        raise ValueError(f"Weights reference unknown environment columns: {sorted(unknown_weights)}")
    weight_values = np.asarray(
        [(weights or {}).get(name, 1.0) for name in columns], dtype=float
    )
    if not np.isfinite(weight_values).all() or (weight_values <= 0).any():
        raise ValueError("Environment variable weights must be finite and greater than zero")

    means = raw.mean(axis=0)
    scales = raw.std(axis=0, ddof=0)
    scales[scales == 0] = 1.0
    matrix = ((raw - means) / scales) * np.sqrt(weight_values)
    strata = frame.loc[:, category_columns].copy() if category_columns else pd.DataFrame(index=frame.index)
    metadata: dict[str, object] = {
        "columns": columns,
        "categorical_columns": category_columns,
        "means": dict(zip(columns, means.tolist(), strict=True)),
        "scales": dict(zip(columns, scales.tolist(), strict=True)),
        "weights": dict(zip(columns, weight_values.tolist(), strict=True)),
        "strata": strata.reset_index(drop=True),
    }
    return matrix, metadata


def environment_distances(
    left: np.ndarray,
    right: np.ndarray,
    *,
    weights: Sequence[float] | np.ndarray | None = None,
    metric: str = "euclidean",
    inverse_covariance: np.ndarray | None = None,
) -> np.ndarray:
    """Return row-wise Euclidean, Manhattan, or Mahalanobis distances."""
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    if left_array.ndim != 2 or left_array.shape != right_array.shape:
        raise ValueError("left and right must be equally shaped two-dimensional arrays")
    if not np.isfinite(left_array).all() or not np.isfinite(right_array).all():
        raise ValueError("Environment vectors must contain only finite values")
    delta = left_array - right_array
    if metric in {"euclidean", "manhattan"}:
        weight_values = (
            np.ones(left_array.shape[1], dtype=float)
            if weights is None
            else np.asarray(weights, dtype=float)
        )
        if weight_values.shape != (left_array.shape[1],):
            raise ValueError("weights must have one value per environment variable")
        if not np.isfinite(weight_values).all() or (weight_values < 0).any():
            raise ValueError("Distance weights must be finite and nonnegative")
        if metric == "manhattan":
            return np.sum(np.abs(delta) * weight_values, axis=1)
        return np.sqrt(np.sum(np.square(delta) * weight_values, axis=1))
    if metric == "mahalanobis":
        if inverse_covariance is None:
            raise ValueError("inverse_covariance is required for Mahalanobis distance")
        inverse = np.asarray(inverse_covariance, dtype=float)
        expected = (left_array.shape[1], left_array.shape[1])
        if inverse.shape != expected or not np.isfinite(inverse).all():
            raise ValueError(f"inverse_covariance must be a finite matrix with shape {expected}")
        squared = np.einsum("bi,ij,bj->b", delta, inverse, delta)
        return np.sqrt(np.maximum(squared, 0.0))
    raise ValueError("metric must be 'euclidean', 'manhattan', or 'mahalanobis'")
