"""Compact, reproducible model builders for Oracle response searches."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import HuberRegressor, LinearRegression, QuantileRegressor, Ridge
from sklearn.preprocessing import OneHotEncoder, SplineTransformer


MODEL_FAMILIES = (
    "ols",
    "log_exposure",
    "quadratic",
    "environment_group_fixed_effects",
    "interactions",
    "piecewise",
    "natural_spline",
    "statsmodels_gam",
    "huber",
    "quantile_regression",
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
)


def _dense_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=float)
    except TypeError:  # scikit-learn before sparse_output was introduced
        return OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=float)


def _base_numeric(
    train: pd.DataFrame,
    score: pd.DataFrame,
    columns: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    if not columns:
        return np.empty((len(train), 0)), np.empty((len(score), 0))
    return (
        train.loc[:, list(columns)].to_numpy(dtype=float),
        score.loc[:, list(columns)].to_numpy(dtype=float),
    )


def _group_design(
    train: pd.DataFrame,
    score: pd.DataFrame,
    group_col: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    if group_col is None:
        raise ValueError("This model family requires an environment group column")
    if group_col not in train or group_col not in score:
        raise ValueError(f"Missing environment group column: {group_col}")
    encoder = _dense_encoder()
    train_group = encoder.fit_transform(train[[group_col]])
    score_group = encoder.transform(score[[group_col]])
    return np.asarray(train_group, dtype=float), np.asarray(score_group, dtype=float)


def _design_matrices(
    train: pd.DataFrame,
    score: pd.DataFrame,
    numeric_features: Sequence[str],
    *,
    exposure: str,
    group_col: str | None,
    model_family: str,
    include_group: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    columns = list(numeric_features)
    if not columns:
        raise ValueError("At least one numeric predictor is required")
    x_train, x_score = _base_numeric(train, score, columns)
    params: dict[str, Any] = {"numeric_features": columns, "group_column": group_col}
    exposure_index = columns.index(exposure) if exposure in columns else None

    if model_family in {"log_exposure", "quadratic", "piecewise", "natural_spline", "statsmodels_gam"}:
        if exposure_index is None:
            raise ValueError(f"{model_family} requires {exposure} among the numeric predictors")
        raw_train = x_train[:, exposure_index : exposure_index + 1]
        raw_score = x_score[:, exposure_index : exposure_index + 1]
        other_indices = [index for index in range(x_train.shape[1]) if index != exposure_index]
        other_train = x_train[:, other_indices]
        other_score = x_score[:, other_indices]
        if model_family == "log_exposure":
            x_train = np.column_stack([np.log1p(raw_train[:, 0]), other_train])
            x_score = np.column_stack([np.log1p(raw_score[:, 0]), other_score])
            params["exposure_transform"] = "log1p"
        elif model_family == "quadratic":
            x_train = np.column_stack([raw_train[:, 0], np.square(raw_train[:, 0]), other_train])
            x_score = np.column_stack([raw_score[:, 0], np.square(raw_score[:, 0]), other_score])
            params["exposure_transform"] = "identity_plus_square"
        elif model_family == "piecewise":
            knots = np.unique(np.quantile(raw_train[:, 0], [0.25, 0.5, 0.75])).tolist()
            train_hinges = np.column_stack([np.maximum(raw_train[:, 0] - knot, 0.0) for knot in knots]) if knots else np.empty((len(train), 0))
            score_hinges = np.column_stack([np.maximum(raw_score[:, 0] - knot, 0.0) for knot in knots]) if knots else np.empty((len(score), 0))
            x_train = np.column_stack([raw_train[:, 0], train_hinges, other_train])
            x_score = np.column_stack([raw_score[:, 0], score_hinges, other_score])
            params.update(exposure_transform="linear_hinge", knots=knots)
        else:
            unique = np.unique(raw_train[:, 0]).size
            if unique < 2:
                raise ValueError("Spline models require at least two distinct training exposure values")
            n_knots = min(5, unique)
            degree = min(3, n_knots - 1)
            spline = SplineTransformer(
                n_knots=n_knots,
                degree=degree,
                knots="quantile",
                extrapolation="linear",
                include_bias=False,
            )
            train_spline = spline.fit_transform(raw_train)
            score_spline = spline.transform(raw_score)
            x_train = np.column_stack([train_spline, other_train])
            x_score = np.column_stack([score_spline, other_score])
            params.update(exposure_transform="cubic_spline_additive", n_knots=n_knots, degree=degree)

    add_group = include_group or model_family in {"environment_group_fixed_effects", "interactions"}
    if add_group:
        group_train, group_score = _group_design(train, score, group_col)
        if model_family == "interactions":
            if exposure_index is None:
                raise ValueError("interactions require the exposure among the numeric predictors")
            exposure_train = train[exposure].to_numpy(dtype=float)[:, None]
            exposure_score = score[exposure].to_numpy(dtype=float)[:, None]
            x_train = np.column_stack([x_train, group_train, exposure_train * group_train])
            x_score = np.column_stack([x_score, group_score, exposure_score * group_score])
            params["group_effects"] = "main_and_exposure_interaction"
        else:
            x_train = np.column_stack([x_train, group_train])
            x_score = np.column_stack([x_score, group_score])
            params["group_effects"] = "fixed_effects"
    if not np.isfinite(x_train).all() or not np.isfinite(x_score).all():
        raise ValueError("Model design contains nonfinite values")
    return x_train, x_score, params


def _estimator(model_family: str, random_state: int) -> tuple[Any, dict[str, Any]]:
    if model_family in {"ols", "log_exposure", "quadratic", "environment_group_fixed_effects", "interactions", "piecewise", "statsmodels_gam"}:
        return LinearRegression(), {"estimator": "LinearRegression", "fit_intercept": True}
    if model_family == "natural_spline":
        return Ridge(alpha=1.0), {"estimator": "Ridge", "alpha": 1.0}
    if model_family == "huber":
        return HuberRegressor(epsilon=1.35, alpha=0.0001, max_iter=200), {
            "estimator": "HuberRegressor", "epsilon": 1.35, "alpha": 0.0001, "max_iter": 200
        }
    if model_family == "quantile_regression":
        return QuantileRegressor(quantile=0.5, alpha=0.001, solver="highs"), {
            "estimator": "QuantileRegressor", "quantile": 0.5, "alpha": 0.001, "solver": "highs"
        }
    if model_family == "random_forest":
        return RandomForestRegressor(
            n_estimators=100, max_depth=16, min_samples_leaf=10, max_features=1.0,
            n_jobs=1, random_state=random_state,
        ), {"estimator": "RandomForestRegressor", "n_estimators": 100, "max_depth": 16, "min_samples_leaf": 10, "random_state": random_state}
    if model_family == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=100, max_depth=16, min_samples_leaf=10, max_features=1.0,
            n_jobs=1, random_state=random_state,
        ), {"estimator": "ExtraTreesRegressor", "n_estimators": 100, "max_depth": 16, "min_samples_leaf": 10, "random_state": random_state}
    if model_family == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            max_iter=100, max_leaf_nodes=31, min_samples_leaf=20,
            l2_regularization=1.0, early_stopping=False, random_state=random_state,
        ), {"estimator": "HistGradientBoostingRegressor", "max_iter": 100, "max_leaf_nodes": 31, "min_samples_leaf": 20, "l2_regularization": 1.0, "early_stopping": False, "random_state": random_state}
    raise ValueError(f"Unsupported model family: {model_family}")


def fit_model_family(
    train: pd.DataFrame,
    score: pd.DataFrame,
    outcome: str,
    exposure: str,
    numeric_features: Sequence[str],
    *,
    group_col: str | None = None,
    model_family: str = "ols",
    include_group: bool = False,
    random_state: int = 20261026,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit one predeclared model on train rows and predict held-out rows."""

    if model_family not in MODEL_FAMILIES:
        raise ValueError(f"Unsupported model family: {model_family}")
    if outcome not in train or outcome not in score:
        raise ValueError(f"Missing outcome column: {outcome}")
    if model_family == "statsmodels_gam":
        return _fit_statsmodels_gam(train, score, outcome, exposure, numeric_features)
    if model_family == "quantile_regression":
        return _fit_statsmodels_quantile(train, score, outcome, exposure, numeric_features)
    x_train, x_score, design_params = _design_matrices(
        train,
        score,
        numeric_features,
        exposure=exposure,
        group_col=group_col,
        model_family=model_family,
        include_group=include_group,
    )
    estimator, estimator_params = _estimator(model_family, random_state)
    y_train = pd.to_numeric(train[outcome], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(y_train).all():
        raise ValueError("Training outcome contains missing or nonfinite values")
    estimator.fit(x_train, y_train)
    prediction = np.asarray(estimator.predict(x_score), dtype=float)
    params = {**design_params, **estimator_params, "model_family": model_family}
    coefficients = getattr(estimator, "coef_", None)
    if coefficients is not None:
        params["coefficients"] = np.asarray(coefficients, dtype=float).reshape(-1).tolist()
        params["intercept"] = float(np.asarray(estimator.intercept_).reshape(-1)[0])
    importances = getattr(estimator, "feature_importances_", None)
    if importances is not None:
        params["feature_importances"] = np.asarray(importances, dtype=float).tolist()
    return prediction, params


def _fit_statsmodels_gam(
    train: pd.DataFrame,
    score: pd.DataFrame,
    outcome: str,
    exposure: str,
    numeric_features: Sequence[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit a Gaussian statsmodels GAM with a smooth exposure term."""

    import statsmodels.api as sm
    from statsmodels.gam.api import BSplines, GLMGam

    if exposure not in numeric_features:
        raise ValueError("statsmodels_gam requires the exposure among numeric predictors")
    environment = [name for name in numeric_features if name != exposure]
    x_train = train[exposure].to_numpy(dtype=float)
    x_score = score[exposure].to_numpy(dtype=float)
    distinct = np.unique(x_train).size
    if distinct < 4:
        raise ValueError("statsmodels_gam requires at least four distinct training exposure values")
    degree = min(3, distinct - 1)
    spline_df = min(6, distinct)
    lower, upper = float(np.min(x_train)), float(np.max(x_train))
    smoother = BSplines(
        x_train[:, None],
        df=[spline_df],
        degree=[degree],
        constraints="center",
        knot_kwds=[{"lower_bound": lower, "upper_bound": upper}],
    )
    if environment:
        exog_train = sm.add_constant(train[environment].to_numpy(dtype=float), has_constant="add")
        exog_score = sm.add_constant(score[environment].to_numpy(dtype=float), has_constant="add")
    else:
        exog_train = np.ones((len(train), 1), dtype=float)
        exog_score = np.ones((len(score), 1), dtype=float)
    model = GLMGam(
        pd.to_numeric(train[outcome], errors="coerce").to_numpy(dtype=float),
        exog=exog_train,
        smoother=smoother,
        family=sm.families.Gaussian(),
    ).fit()
    smooth_score = np.clip(x_score, lower, upper)[:, None]
    prediction = np.asarray(model.predict(exog=exog_score, exog_smooth=smooth_score), dtype=float)
    params = {
        "model_family": "statsmodels_gam",
        "estimator": "statsmodels.GLMGam",
        "distribution": "Gaussian",
        "smooth_term": exposure,
        "smooth_basis": "centered_cubic_B_spline",
        "spline_df": spline_df,
        "degree": degree,
        "training_exposure_bounds": [lower, upper],
        "heldout_exposure_clipped_to_training_bounds": True,
        "linear_environment_covariates": environment,
    }
    return prediction, params


def _fit_statsmodels_quantile(
    train: pd.DataFrame,
    score: pd.DataFrame,
    outcome: str,
    exposure: str,
    numeric_features: Sequence[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit median quantile regression with statsmodels' iterative solver."""

    import statsmodels.api as sm

    x_train = train[list(numeric_features)].to_numpy(dtype=float)
    x_score = score[list(numeric_features)].to_numpy(dtype=float)
    design_train = sm.add_constant(x_train, has_constant="add")
    design_score = sm.add_constant(x_score, has_constant="add")
    y_train = pd.to_numeric(train[outcome], errors="coerce").to_numpy(dtype=float)
    fitted = sm.QuantReg(y_train, design_train).fit(q=0.5, max_iter=100, p_tol=1e-5)
    params = {
        "model_family": "quantile_regression",
        "estimator": "statsmodels.QuantReg",
        "quantile": 0.5,
        "max_iter": 100,
        "p_tol": 1e-5,
        "features": list(numeric_features),
        "coefficients": np.asarray(fitted.params, dtype=float).tolist(),
        "exposure": exposure,
    }
    return np.asarray(fitted.predict(design_score), dtype=float), params
