"""Multi-family response models with explicit apparent/spatial role metrics."""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from experiments.oracle_exploration.nonlinear_search import MODEL_FAMILIES, fit_model_family


def compute_delta_r2(full_model: float, environment_only: float) -> float:
    """Subtract R² values measured on identical rows and evaluation roles."""

    if not np.isfinite(full_model) or not np.isfinite(environment_only):
        raise ValueError("R² inputs must be finite")
    return float(full_model - environment_only)


def _complete_case_frame(
    frame: pd.DataFrame, outcome: str, exposure: str, environment: Sequence[str], group: str | None
) -> tuple[pd.DataFrame, dict[str, int]]:
    required = list(dict.fromkeys([outcome, exposure, *environment, *([group] if group else [])]))
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Missing regression columns: {missing}")
    data = frame.copy()
    numeric_columns = list(dict.fromkeys([outcome, exposure, *environment]))
    numeric = data[numeric_columns].apply(pd.to_numeric, errors="coerce")
    complete = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
    complete &= numeric[exposure].to_numpy(dtype=float) >= 0
    if group:
        complete &= data[group].notna().to_numpy()
    counts = {"input_rows": int(len(frame)), "complete_case_rows": int(complete.sum())}
    data = data.loc[complete].copy()
    for column in numeric_columns:
        data[column] = numeric.loc[complete, column].to_numpy(dtype=float)
    return data.reset_index(drop=True), counts


def _metric_values(observed: np.ndarray, predicted: np.ndarray) -> dict[str, object]:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    finite = np.isfinite(observed) & np.isfinite(predicted)
    observed, predicted = observed[finite], predicted[finite]
    n = len(observed)
    if n < 2:
        return {"status": "insufficient_evaluation_rows", "n": n, "r_squared": np.nan, "rmse": np.nan, "mae": np.nan}
    residual = observed - predicted
    sse = float(residual @ residual)
    tss = float(np.square(observed - observed.mean()).sum())
    if tss <= 0:
        return {"status": "no_outcome_variation", "n": n, "r_squared": np.nan, "rmse": float(np.sqrt(sse / n)), "mae": float(np.abs(residual).mean())}
    return {"status": "ok", "n": n, "r_squared": float(1 - sse / tss), "rmse": float(np.sqrt(sse / n)), "mae": float(np.abs(residual).mean())}


def _serialize(params: dict[str, Any]) -> str:
    return json.dumps(params, ensure_ascii=False, sort_keys=True, default=lambda value: value.item() if hasattr(value, "item") else str(value))


def _exposure_effect(
    params: dict[str, Any], family: str, exposure: str, features: Sequence[str]
) -> tuple[float, str]:
    interpretable_families = {
        "ols",
        "log_exposure",
        "quadratic",
        "piecewise",
        "huber",
        "quantile_regression",
        "environment_group_fixed_effects",
        "interactions",
    }
    coefficients = params.get("coefficients")
    if family not in interpretable_families or exposure not in features or not coefficients:
        return float("nan"), "unknown"
    if features[0] != exposure or len(coefficients) == 0:
        return float("nan"), "unknown"
    value = float(coefficients[0])
    if not np.isfinite(value):
        return float("nan"), "unknown"
    return value, "positive" if value > 0 else "negative" if value < 0 else "zero"


def _fit_family(
    data: pd.DataFrame,
    *,
    outcome: str,
    exposure: str,
    features: Sequence[str],
    group: str | None,
    family: str,
    role_col: str,
    include_group: bool = False,
) -> list[dict[str, object]]:
    has_roles = role_col in data and "train" in set(data[role_col].astype(str))
    train = data.loc[data[role_col].astype(str).eq("train")].copy() if has_roles else data
    if len(train) < 3:
        return [{"outcome": outcome, "exposure": exposure, "model_name": family, "evaluation_set": "train" if has_roles else "apparent", "score_type": "spatial_holdout" if has_roles else "apparent", "status": "insufficient_training_rows", "n": int(len(train)), "r_squared": np.nan, "rmse": np.nan, "mae": np.nan, "features": list(features), "model_config": "{}", "runtime_seconds": 0.0}]
    started = time.perf_counter()
    try:
        prediction, params = fit_model_family(train, data, outcome, exposure, features, group_col=group, model_family=family, include_group=include_group)
        coefficient, direction = _exposure_effect(params, family, exposure, features)
        runtime = time.perf_counter() - started
        records: list[dict[str, object]] = []
        role_values = data[role_col].astype(str).to_numpy() if has_roles else np.repeat("apparent", len(data))
        observed = data[outcome].to_numpy(dtype=float)
        for evaluation_set in (["train", "validation", "test"] if has_roles else ["apparent"]):
            mask = role_values == evaluation_set
            if not mask.any():
                continue
            metric = _metric_values(observed[mask], prediction[mask])
            records.append({"outcome": outcome, "exposure": exposure, "model_name": family, "evaluation_set": evaluation_set, "score_type": "training" if evaluation_set == "train" else "spatial_holdout" if has_roles else "apparent", **metric, "features": list(features), "model_config": _serialize(params), "coefficient": coefficient, "direction": direction, "runtime_seconds": runtime})

        # A separately labelled apparent score refits on every row, including test.
        if has_roles:
            apparent_started = time.perf_counter()
            apparent_prediction, apparent_params = fit_model_family(data, data, outcome, exposure, features, group_col=group, model_family=family, include_group=include_group)
            metric = _metric_values(observed, apparent_prediction)
            apparent_coefficient, apparent_direction = _exposure_effect(apparent_params, family, exposure, features)
            records.append({"outcome": outcome, "exposure": exposure, "model_name": family, "evaluation_set": "apparent", "score_type": "apparent", **metric, "features": list(features), "model_config": _serialize(apparent_params), "coefficient": apparent_coefficient, "direction": apparent_direction, "runtime_seconds": time.perf_counter() - apparent_started})
        return records
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        roles = ["apparent", "train", "validation", "test"] if has_roles else ["apparent"]
        return [{"outcome": outcome, "exposure": exposure, "model_name": family, "evaluation_set": role, "score_type": "apparent" if role == "apparent" else "spatial_holdout", "status": "failed", "error": error, "n": 0, "r_squared": np.nan, "rmse": np.nan, "mae": np.nan, "features": list(features), "model_config": "{}", "coefficient": np.nan, "direction": "unknown", "runtime_seconds": time.perf_counter() - started} for role in roles]


def fit_candidate_models(
    frame: pd.DataFrame,
    outcome: str,
    exposure: str,
    environment: Sequence[str],
    group: str | None = None,
    model_families: Sequence[str] = MODEL_FAMILIES,
    *,
    role_col: str = "spatial_role",
    outcome_guided_selection: bool = True,
) -> pd.DataFrame:
    """Fit on train blocks, score held-out blocks, and label apparent fits separately."""

    if not model_families:
        raise ValueError("At least one model family must be supplied")
    unsupported = sorted(set(model_families) - set(MODEL_FAMILIES))
    if unsupported:
        raise ValueError(f"Unsupported model families: {unsupported}")
    data, counts = _complete_case_frame(frame, outcome, exposure, environment, group)
    if data.empty:
        raise ValueError("No complete cases remain for this outcome/model")
    features = [exposure, *environment]
    records: list[dict[str, object]] = []
    for family in model_families:
        rows = _fit_family(data, outcome=outcome, exposure=exposure, features=features, group=group, family=family, role_col=role_col)
        for row in rows:
            row.update(counts, complete_case_columns=[outcome, exposure, *environment, *([group] if group else [])], outcome_guided_selection=bool(outcome_guided_selection))
        records.extend(rows)
    return pd.DataFrame.from_records(records)


def _ols_score(x: np.ndarray, y: np.ndarray) -> dict[str, object]:
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x[finite], dtype=float), np.asarray(y[finite], dtype=float)
    n = len(x)
    if n < 3:
        return {"status": "insufficient_sample", "n": n, "r_squared": np.nan, "coefficient": np.nan, "rmse": np.nan, "mae": np.nan}
    centered_x, centered_y = x - x.mean(), y - y.mean()
    sxx, syy = float(centered_x @ centered_x), float(centered_y @ centered_y)
    if sxx <= 0 or syy <= 0:
        return {"status": "no_variation", "n": n, "r_squared": np.nan, "coefficient": np.nan, "rmse": np.nan, "mae": np.nan}
    coefficient = float(centered_x @ centered_y / sxx)
    residual = centered_y - coefficient * centered_x
    sse = float(residual @ residual)
    return {"status": "ok", "n": n, "r_squared": float(1 - sse / syy), "coefficient": coefficient, "rmse": float(np.sqrt(sse / n)), "mae": float(np.abs(residual).mean())}


def _paired_score(pairs: pd.DataFrame, outcome: str) -> dict[str, object]:
    high_y = f"{outcome}_high" if f"{outcome}_high" in pairs else f"high_{outcome}"
    low_y = f"{outcome}_low" if f"{outcome}_low" in pairs else f"low_{outcome}"
    required = {"erosion_high", "erosion_low", high_y, low_y}
    if not required.issubset(pairs.columns):
        return {"status": "missing_pair_columns", "n": 0, "r_squared": np.nan, "coefficient": np.nan, "rmse": np.nan, "mae": np.nan}
    delta_x = np.log1p(pd.to_numeric(pairs["erosion_high"], errors="coerce").to_numpy(dtype=float)) - np.log1p(pd.to_numeric(pairs["erosion_low"], errors="coerce").to_numpy(dtype=float))
    delta_y = pd.to_numeric(pairs[high_y], errors="coerce").to_numpy(dtype=float) - pd.to_numeric(pairs[low_y], errors="coerce").to_numpy(dtype=float)
    return _ols_score(delta_x, delta_y)


def compare_feature_regimes(
    frame: pd.DataFrame,
    outcome: str,
    exposure: str,
    environment: Sequence[str],
    *,
    group: str | None = None,
    model_families: Sequence[str] = ("ols",),
    pairs: pd.DataFrame | None = None,
    role_col: str = "spatial_role",
) -> pd.DataFrame:
    """Compare exposure/environment feature regimes on one shared row mask."""

    data, counts = _complete_case_frame(frame, outcome, exposure, environment, group)
    if data.empty:
        raise ValueError("No complete cases remain for this outcome/regime comparison")
    regimes: dict[str, tuple[list[str], bool]] = {
        "exposure_only": ([exposure], False),
        "environment_only": (list(environment), False),
        "environment_plus_exposure": ([exposure, *environment], False),
        "group_plus_exposure": ([exposure], True),
    }
    records: list[dict[str, object]] = []
    for regime, (features, include_group) in regimes.items():
        for family in model_families:
            if family in {"environment_group_fixed_effects", "interactions"} and not include_group:
                continue
            rows = _fit_family(
                data,
                outcome=outcome,
                exposure=exposure,
                features=features,
                group=group,
                family=family,
                role_col=role_col,
                include_group=include_group,
            )
            for row in rows:
                row.update(regime=regime, complete_case_rows=counts["complete_case_rows"], trial_count=1)
            records.extend(rows)

    if pairs is not None:
        roles = ["apparent"]
        if "spatial_role" in pairs:
            pair_roles = set(pairs["spatial_role"].dropna().astype(str))
            roles.extend(role for role in ("train", "validation", "test") if role in pair_roles)
        for evaluation_set in roles:
            selected_pairs = pairs if evaluation_set == "apparent" or "spatial_role" not in pairs else pairs.loc[pairs["spatial_role"].astype(str).eq(evaluation_set)]
            metric = _paired_score(selected_pairs, outcome)
            records.append({"outcome": outcome, "exposure": exposure, "model_name": "paired_difference", "regime": "paired_difference", "evaluation_set": evaluation_set, "score_type": "apparent" if evaluation_set == "apparent" else "spatial_holdout", **metric, "complete_case_rows": metric["n"], "trial_count": 1, "selection_objective": "paired_delta_health_on_delta_log1p_exposure"})

    result = pd.DataFrame.from_records(records)
    result["delta_r2"] = np.nan
    for (model_name, evaluation_set), index in result.groupby(["model_name", "evaluation_set"], sort=False, dropna=False).groups.items():
        rows = result.loc[index]
        env = rows.loc[rows["regime"].eq("environment_only")]
        full = rows.loc[rows["regime"].eq("environment_plus_exposure")]
        if env.empty or full.empty:
            continue
        env_row, full_row = env.iloc[0], full.iloc[0]
        if int(env_row["n"]) == int(full_row["n"]) and np.isfinite(env_row["r_squared"]) and np.isfinite(full_row["r_squared"]):
            result.loc[full.index, "delta_r2"] = compute_delta_r2(float(full_row["r_squared"]), float(env_row["r_squared"]))
    result["outcome_guided_selection"] = True
    return result


def fit_selected_model_on_test(
    frame: pd.DataFrame,
    outcome: str,
    exposure: str,
    environment: Sequence[str],
    model_family: str,
    *,
    group: str | None = None,
    role_col: str = "spatial_role",
    train_roles: Sequence[str] = ("train", "validation"),
    test_role: str = "test",
) -> dict[str, object]:
    """Fit a validation-selected model on train+validation, then score test once."""

    data, counts = _complete_case_frame(frame, outcome, exposure, environment, group)
    if role_col not in data:
        raise ValueError(f"Missing spatial role column: {role_col}")
    role_values = data[role_col].astype(str)
    train = data.loc[role_values.isin(train_roles)].copy()
    test = data.loc[role_values.eq(test_role)].copy()
    result: dict[str, object] = {
        "outcome": outcome,
        "exposure": exposure,
        "model_name": model_family,
        "evaluation_set": test_role,
        "score_type": "spatial_holdout",
        "training_roles": list(train_roles),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "complete_case_rows": counts["complete_case_rows"],
        "outcome_guided_selection": True,
    }
    if len(train) < 3 or len(test) < 2:
        result.update(status="insufficient_role_rows", n=int(len(test)), r_squared=np.nan, rmse=np.nan, mae=np.nan)
        return result
    features = [exposure, *environment]
    prediction, params = fit_model_family(
        train,
        test,
        outcome,
        exposure,
        features,
        group_col=group,
        model_family=model_family,
    )
    metric = _metric_values(test[outcome].to_numpy(dtype=float), prediction)
    coefficient, direction = _exposure_effect(params, model_family, exposure, features)
    result.update(
        metric,
        model_config=_serialize(params),
        coefficient=coefficient,
        direction=direction,
    )
    return result
