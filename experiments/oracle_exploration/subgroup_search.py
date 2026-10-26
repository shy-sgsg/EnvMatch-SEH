"""Auditable environment subgroup and outcome-selected retention searches."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


_OPERATORS = {"eq", "ne", "in", "not_in", "gt", "ge", "lt", "le", "between", "is_null", "not_null"}


def _evaluate_definition(frame: pd.DataFrame, definition: Any) -> pd.Series:
    if not isinstance(definition, Mapping):
        raise ValueError("Each subgroup definition must be a predicate mapping")
    if "all" in definition or "any" in definition:
        if "all" in definition and "any" in definition:
            raise ValueError("A subgroup predicate cannot contain both all and any")
        key = "all" if "all" in definition else "any"
        children = definition[key]
        if not isinstance(children, Sequence) or isinstance(children, (str, bytes)) or not children:
            raise ValueError(f"Subgroup {key} must contain at least one predicate")
        masks = [_evaluate_definition(frame, child) for child in children]
        result = masks[0].copy()
        for mask in masks[1:]:
            result = result & mask if key == "all" else result | mask
        return result.fillna(False).astype(bool)

    column = definition.get("column")
    operator = definition.get("op")
    if not isinstance(column, str) or column not in frame:
        raise ValueError(f"Subgroup predicate references missing column: {column!r}")
    if operator not in _OPERATORS:
        raise ValueError(f"Unsupported subgroup operator: {operator!r}")
    values = frame[column]
    if operator == "eq":
        result = values.eq(definition.get("value"))
    elif operator == "ne":
        result = values.ne(definition.get("value"))
    elif operator == "in":
        result = values.isin(definition.get("values", ()))
    elif operator == "not_in":
        result = ~values.isin(definition.get("values", ()))
    elif operator == "gt":
        result = pd.to_numeric(values, errors="coerce").gt(float(definition["value"]))
    elif operator == "ge":
        result = pd.to_numeric(values, errors="coerce").ge(float(definition["value"]))
    elif operator == "lt":
        result = pd.to_numeric(values, errors="coerce").lt(float(definition["value"]))
    elif operator == "le":
        result = pd.to_numeric(values, errors="coerce").le(float(definition["value"]))
    elif operator == "between":
        numeric = pd.to_numeric(values, errors="coerce")
        inclusive = bool(definition.get("inclusive", True))
        result = numeric.ge(float(definition["lower"])) & numeric.le(float(definition["upper"]))
        if not inclusive:
            result = numeric.gt(float(definition["lower"])) & numeric.lt(float(definition["upper"]))
    elif operator == "is_null":
        result = values.isna()
    else:
        result = values.notna()
    return result.fillna(False).astype(bool)


def build_subgroup_masks(
    frame: pd.DataFrame, definitions: Mapping[str, Mapping[str, Any]]
) -> dict[str, np.ndarray]:
    """Build named masks from explicit, composable predicates."""

    if not isinstance(definitions, Mapping) or not definitions:
        raise ValueError("definitions must be a non-empty mapping of names to predicates")
    if frame.index.has_duplicates:
        frame = frame.reset_index(drop=True)
    return {
        str(name): _evaluate_definition(frame, definition).to_numpy(dtype=bool)
        for name, definition in definitions.items()
    }


def _linear_fit_score(x: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray, float]:
    design = np.column_stack([np.ones(len(x)), x])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    prediction = design @ coefficients
    sst = float(np.square(y - y.mean()).sum())
    sse = float(np.square(y - prediction).sum())
    r_squared = float(1.0 - sse / sst) if sst > 0 else np.nan
    return r_squared, coefficients, sse


def oracle_retention_search(
    frame: pd.DataFrame,
    outcome: str,
    predictors: Sequence[str] | str,
    fractions: Sequence[float],
    *,
    trials: int = 50,
    seed: int = 20261026,
) -> pd.DataFrame:
    """Search outcome-selected row subsets and report best apparent OLS R².

    This function intentionally uses the response to choose rows. Its output is
    an Oracle apparent-fit envelope; it contains no independent test score.
    Selected IDs and the selection rule are persisted for exact reconstruction.
    """

    predictor_columns = [predictors] if isinstance(predictors, str) else list(predictors)
    if not predictor_columns:
        raise ValueError("At least one predictor is required")
    if not fractions or any(not np.isfinite(value) or value <= 0 or value > 1 for value in fractions):
        raise ValueError("retention fractions must be finite values in (0, 1]")
    if not isinstance(trials, int) or trials < 1:
        raise ValueError("trials must be a positive integer")
    required = [outcome, *predictor_columns]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Missing Oracle retention columns: {missing}")
    numeric = frame.loc[:, required].apply(pd.to_numeric, errors="coerce")
    complete = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
    data = frame.loc[complete].copy().reset_index(drop=True)
    numeric = numeric.loc[complete].reset_index(drop=True)
    n_complete = len(data)
    if n_complete < max(3, len(predictor_columns) + 2):
        raise ValueError("Too few complete cases for Oracle retention regression")

    x_all = numeric[predictor_columns].to_numpy(dtype=float)
    y_all = numeric[outcome].to_numpy(dtype=float)
    full_r2, full_coefficients, _ = _linear_fit_score(x_all, y_all)
    full_design = np.column_stack([np.ones(n_complete), x_all])
    residuals = y_all - full_design @ full_coefficients
    cell_ids = data["cell_id"].tolist() if "cell_id" in data else data.index.tolist()
    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []

    for fraction in fractions:
        target_n = min(n_complete, max(3, int(round(float(fraction) * n_complete))))
        if target_n == n_complete:
            attempts = [(np.arange(n_complete), "all_complete_cases")]
        else:
            attempts: list[tuple[np.ndarray, str]] = []
            residual_priority = np.abs(residuals)
            attempts.append((np.argpartition(residual_priority, target_n - 1)[:target_n], "lowest_absolute_full_fit_residual"))
            for _ in range(max(0, trials - 1)):
                if rng.random() < 0.5:
                    indices = rng.choice(n_complete, size=target_n, replace=False)
                    rule = "random_subset_then_oracle_rank"
                else:
                    jitter = rng.uniform(0.0, np.nanstd(residual_priority) * 0.25 + 1e-12, size=n_complete)
                    indices = np.argpartition(residual_priority + jitter, target_n - 1)[:target_n]
                    rule = "jittered_low_residual_oracle_subset"
                attempts.append((indices, rule))

        best: dict[str, object] | None = None
        for trial_index, (indices, rule) in enumerate(attempts, start=1):
            indices = np.asarray(indices, dtype=int)
            score, coefficients, sse = _linear_fit_score(x_all[indices], y_all[indices])
            if not np.isfinite(score):
                continue
            if best is None or score > float(best["apparent_r2"]):
                best = {
                    "apparent_r2": score,
                    "coefficients": coefficients,
                    "sse": sse,
                    "indices": indices,
                    "selection_rule": rule,
                    "best_trial": trial_index,
                }
        if best is None:
            best = {"apparent_r2": np.nan, "coefficients": np.full(len(predictor_columns) + 1, np.nan), "sse": np.nan, "indices": np.arange(target_n), "selection_rule": "insufficient_variation", "best_trial": 1}
        selected_indices = np.asarray(best["indices"], dtype=int)
        coefficients = np.asarray(best["coefficients"], dtype=float)
        exposure_predictors = [
            index for index, name in enumerate(predictor_columns) if str(name).startswith("Aa_")
        ]
        coefficient_index = 1 + (exposure_predictors[0] if exposure_predictors else 0)
        exposure_coefficient = (
            float(coefficients[coefficient_index])
            if coefficient_index < len(coefficients) and np.isfinite(coefficients[coefficient_index])
            else float("nan")
        )
        direction = (
            "positive" if exposure_coefficient > 0 else "negative" if exposure_coefficient < 0 else "zero"
        ) if np.isfinite(exposure_coefficient) else "unknown"
        records.append(
            {
                "outcome": outcome,
                "predictors": predictor_columns,
                "target_retention_fraction": float(fraction),
                "sample_retention_ratio": float(len(selected_indices) / n_complete),
                "n_complete_cases": n_complete,
                "n_selected": int(len(selected_indices)),
                "apparent_r2": best["apparent_r2"],
                "full_sample_apparent_r2": full_r2,
                "coefficients": coefficients.tolist(),
                "coefficient": exposure_coefficient,
                "direction": direction,
                "selection_mode": "outcome_selected_oracle",
                "outcome_guided_selection": True,
                "selection_objective": "maximize_apparent_ols_r2_at_fixed_retention",
                "selection_rule": best["selection_rule"],
                "selection_trial": best["best_trial"],
                "trial_count": len(attempts),
                "selection_seed": int(seed),
                "selected_cell_ids_json": json.dumps([cell_ids[index] for index in selected_indices], ensure_ascii=False),
                "test_r2": np.nan,
            }
        )
    return pd.DataFrame.from_records(records)
