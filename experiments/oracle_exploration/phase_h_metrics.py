"""Shared response direction and regression metrics for Phase H searches."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

R2_DEFINITION = "1 - SSE / sum((y - mean(y))^2)"


def regression_metrics(y_true: np.ndarray, y_predicted: np.ndarray) -> dict[str, Any]:
    """Score predictions with centered total sum of squares for every estimator."""

    y = np.asarray(y_true, dtype=float).reshape(-1)
    predicted = np.asarray(y_predicted, dtype=float).reshape(-1)
    if y.shape != predicted.shape:
        raise ValueError("y_true and y_predicted must have the same length")
    finite = np.isfinite(y) & np.isfinite(predicted)
    y, predicted = y[finite], predicted[finite]
    n = len(y)
    if not n:
        return {
            "n": 0,
            "r_squared": float("nan"),
            "rmse": float("nan"),
            "mae": float("nan"),
            "r2_definition": R2_DEFINITION,
        }
    residual = y - predicted
    sse = float(residual @ residual)
    centered = y - float(np.mean(y))
    total = float(centered @ centered)
    if n < 2:
        r_squared = float("nan")
    elif total > 0:
        r_squared = float(1.0 - sse / total)
    else:
        r_squared = 1.0 if sse == 0 else 0.0
    return {
        "n": int(n),
        "r_squared": r_squared,
        "rmse": float(np.sqrt(sse / n)),
        "mae": float(np.abs(residual).mean()),
        "r2_definition": R2_DEFINITION,
    }


def classify_linear_direction(coefficient: float, *, tolerance: float = 1e-10) -> str:
    """Classify a linear exposure coefficient without converting missing values to zero."""

    try:
        value = float(coefficient)
    except (TypeError, ValueError):
        return "unknown"
    if not np.isfinite(value):
        return "unknown"
    if abs(value) <= tolerance:
        return "near_zero"
    return "negative" if value < 0 else "positive"


def direction_from_curve(
    x: np.ndarray,
    predicted: np.ndarray,
    *,
    relative_tolerance: float = 0.01,
) -> dict[str, Any]:
    """Classify a fitted response curve over its observed 2nd–98th percentile support."""

    x_values = np.asarray(x, dtype=float).reshape(-1)
    y_values = np.asarray(predicted, dtype=float).reshape(-1)
    if x_values.shape != y_values.shape:
        raise ValueError("x and predicted curve values must have the same length")
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    x_values, y_values = x_values[finite], y_values[finite]
    if len(x_values) < 2:
        return {
            "direction": "unknown", "curve_change": float("nan"),
            "support_low": float("nan"), "support_high": float("nan"),
            "positive_variation": float("nan"), "negative_variation": float("nan"),
        }
    unique_x, inverse = np.unique(x_values, return_inverse=True)
    if len(unique_x) < 2:
        return {
            "direction": "unknown", "curve_change": float("nan"),
            "support_low": float(unique_x[0]), "support_high": float(unique_x[0]),
            "positive_variation": float("nan"), "negative_variation": float("nan"),
        }
    sums = np.bincount(inverse, weights=y_values)
    counts = np.bincount(inverse)
    mean_y = sums / counts
    low, high = np.quantile(unique_x, [0.02, 0.98])
    if not np.isfinite(low + high) or high <= low:
        return {
            "direction": "unknown", "curve_change": float("nan"),
            "support_low": float(low), "support_high": float(high),
            "positive_variation": float("nan"), "negative_variation": float("nan"),
        }
    grid = np.linspace(float(low), float(high), 101)
    curve = np.interp(grid, unique_x, mean_y)
    changes = np.diff(curve)
    positive_variation = float(np.maximum(changes, 0.0).sum())
    negative_variation = float(-np.minimum(changes, 0.0).sum())
    curve_change = float(curve[-1] - curve[0])
    response_scale = max(abs(float(np.median(mean_y))), float(np.ptp(mean_y)), 1e-12)
    tolerance = relative_tolerance * response_scale
    if positive_variation > 5 * tolerance and negative_variation > 5 * tolerance:
        direction = "mixed"
    elif abs(curve_change) <= tolerance:
        direction = "near_zero"
    else:
        direction = "negative" if curve_change < 0 else "positive"
    return {
        "direction": direction,
        "curve_change": curve_change,
        "support_low": float(low),
        "support_high": float(high),
        "positive_variation": positive_variation,
        "negative_variation": negative_variation,
    }


def matched_pair_relation_summary(pairs: pd.DataFrame, outcome: str) -> dict[str, Any]:
    """Summarize high-minus-low health changes against matched Aa differences."""

    high_erosion_column = "Aa_high" if "Aa_high" in pairs else "erosion_high"
    low_erosion_column = "Aa_low" if "Aa_low" in pairs else "erosion_low"
    high_health_column, low_health_column = f"high_{outcome}", f"low_{outcome}"
    required = {high_erosion_column, low_erosion_column, high_health_column, low_health_column}
    missing = sorted(required - set(pairs.columns))
    if missing:
        raise ValueError(f"pair relation data is missing columns: {missing}")
    high_erosion = pd.to_numeric(pairs[high_erosion_column], errors="coerce").to_numpy(dtype=float)
    low_erosion = pd.to_numeric(pairs[low_erosion_column], errors="coerce").to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.log1p(high_erosion) - np.log1p(low_erosion)
    y = pd.to_numeric(pairs[high_health_column], errors="coerce").to_numpy(dtype=float) - pd.to_numeric(
        pairs[low_health_column], errors="coerce"
    ).to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x, y = x[valid], y[valid]
    n = len(x)
    rho = float("nan")
    if n >= 3 and np.ptp(x) > 0 and np.ptp(y) > 0:
        rho = float(spearmanr(x, y).statistic)
    if n < 3 or not np.isfinite(rho):
        direction = "unknown"
    elif abs(rho) < 0.05:
        direction = "near_zero"
    else:
        direction = "negative" if rho < 0 else "positive"
    return {
        "n_pairs": int(n),
        "mean_delta_health": float(np.mean(y)) if n else float("nan"),
        "median_delta_health": float(np.median(y)) if n else float("nan"),
        "negative_delta_health_fraction": float(np.mean(y < 0)) if n else float("nan"),
        "positive_delta_health_fraction": float(np.mean(y > 0)) if n else float("nan"),
        "zero_delta_health_fraction": float(np.mean(y == 0)) if n else float("nan"),
        "spearman_rho": rho,
        "association_direction": direction,
        "association_unit": "matched_pair_high_minus_low",
    }
