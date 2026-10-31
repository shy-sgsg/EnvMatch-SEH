"""Phase I-1 global erosion--health response exploration.

This runner reuses Phase H's full-population Spearman statistics when the
complete-case count agrees, and adds the missing full-population effect sizes,
zero/positive contrasts, spatially blocked model comparison, quantile fits,
figures, and a report. It writes one immutable Oracle run directory.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phase_i_mplconfig")

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import statsmodels.api as sm
import yaml
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import SplineTransformer
from statsmodels.tools.sm_exceptions import ConvergenceWarning, IterationLimitWarning


ROOT = Path(__file__).resolve().parents[2]
OUTCOMES = ["SHI_candidate", "AN", "OM", "CEC", "pH", "SW", "BD", "MBC", "RESP"]
LABELS = {
    "SHI_candidate": "土壤健康指数（SHI）", "AN": "碱解氮（AN）", "OM": "有机质（OM）",
    "CEC": "阳离子交换量（CEC）", "pH": "pH", "SW": "土壤含水量（SW）",
    "BD": "土壤容重（BD）", "MBC": "微生物量碳（MBC）", "RESP": "土壤呼吸（RESP）",
}
PALETTE = {"blue": "#31688e", "orange": "#e07a3f", "dark": "#263746", "muted": "#78858f", "grid": "#d8dee3"}
try:
    font_manager.findfont("WenQuanYi Zen Hei", fallback_to_default=False)
    plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
except ValueError:
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False


def _read_inputs(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, str]]:
    input_root = ROOT / "outputs" / config["input_run_id"] / "databases"
    env_path = input_root / "environment_erosion.parquet"
    health_path = input_root / "health.parquet"
    env_columns = ["cell_id", "row", "col", "x", "y", "SoilType", "Aa_20yr", "Aa_2020"]
    health_columns = ["cell_id", "row", "col", "x", "y", *config["outcomes"]]
    env = pd.read_parquet(env_path, columns=env_columns)
    health = pd.read_parquet(health_path, columns=health_columns)
    if not env["cell_id"].is_unique or not health["cell_id"].is_unique:
        raise ValueError("Aligned input tables must have one row per cell_id.")
    aligned = env[["cell_id", "row", "col", "x", "y"]].merge(
        health[["cell_id", "row", "col", "x", "y"]], on="cell_id", how="outer", suffixes=("_env", "_health"),
        validate="one_to_one", indicator=True,
    )
    if not aligned["_merge"].eq("both").all():
        raise ValueError("Environment and health databases do not contain the same cell_id population.")
    for coordinate in ("row", "col"):
        if not aligned[f"{coordinate}_env"].equals(aligned[f"{coordinate}_health"]):
            raise ValueError(f"Raster database {coordinate} coordinates disagree by cell_id.")
    for coordinate in ("x", "y"):
        left = pd.to_numeric(aligned[f"{coordinate}_env"], errors="coerce").to_numpy(dtype=float)
        right = pd.to_numeric(aligned[f"{coordinate}_health"], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(left).all() or not np.isfinite(right).all() or not np.allclose(left, right, rtol=0, atol=1e-6):
            raise ValueError(f"Raster database {coordinate} coordinates disagree by cell_id.")
    data = env.merge(health.drop(columns=["row", "col", "x", "y"]), on="cell_id", how="inner", validate="one_to_one")
    data = data.sort_values("cell_id", kind="stable").reset_index(drop=True)
    for exposure in (config["primary_exposure"], config["supplementary_exposure"]):
        values = pd.to_numeric(data[exposure], errors="coerce")
        finite = values[np.isfinite(values)]
        if not finite.empty and float(finite.min()) < 0:
            raise ValueError(f"{exposure} contains negative values; log1p response models are not defined.")
    paths = {"environment_erosion": str(env_path.relative_to(ROOT)), "health": str(health_path.relative_to(ROOT))}
    return data, paths


def _block_ids(frame: pd.DataFrame, block_size_m: int) -> np.ndarray:
    xx = pd.to_numeric(frame["x"], errors="coerce").to_numpy(dtype=float)
    yy = pd.to_numeric(frame["y"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(xx).all() or not np.isfinite(yy).all():
        raise ValueError("Spatial block assignment requires finite projected coordinates.")
    return (np.floor(xx / block_size_m).astype(np.int64).astype(str) + "_" +
            np.floor(yy / block_size_m).astype(np.int64).astype(str))


def _block_bootstrap_correlation(
    x: np.ndarray, y: np.ndarray, blocks: np.ndarray, *, reps: int, seed: int,
) -> dict[str, float]:
    codes, _ = pd.factorize(blocks, sort=True)
    k = int(codes.max()) + 1 if len(codes) else 0
    if k < 5:
        return {"slope_ci_low": np.nan, "slope_ci_high": np.nan,
                "standardized_beta_ci_low": np.nan, "standardized_beta_ci_high": np.nan}
    weights = [np.ones(len(x)), x, y, x * x, y * y, x * y]
    sums = np.column_stack([np.bincount(codes, weights=w, minlength=k) for w in weights])
    draws = np.random.default_rng(seed).integers(0, k, size=(reps, k))
    n, sx, sy, sx2, sy2, sxy = sums[draws].sum(axis=1).T
    vx = np.maximum(sx2 - sx * sx / n, 0)
    vy = np.maximum(sy2 - sy * sy / n, 0)
    cov = sxy - sx * sy / n
    slope = np.divide(cov, vx, out=np.full(reps, np.nan), where=vx > 0)
    beta = np.divide(cov, np.sqrt(vx * vy), out=np.full(reps, np.nan), where=(vx > 0) & (vy > 0))
    slope = slope[np.isfinite(slope)]
    beta = beta[np.isfinite(beta)]
    if not len(slope) or not len(beta):
        limits = (np.nan, np.nan)
        beta_limits = (np.nan, np.nan)
    else:
        limits = tuple(float(v) for v in np.quantile(slope, [.025, .975]))
        beta_limits = tuple(float(v) for v in np.quantile(beta, [.025, .975]))
    return {"slope_ci_low": limits[0], "slope_ci_high": limits[1],
            "standardized_beta_ci_low": beta_limits[0], "standardized_beta_ci_high": beta_limits[1]}


def _block_bootstrap_mean_difference(
    y: np.ndarray, positive: np.ndarray, blocks: np.ndarray, *, reps: int, seed: int,
) -> tuple[float, float]:
    codes, _ = pd.factorize(blocks, sort=True)
    k = int(codes.max()) + 1 if len(codes) else 0
    if k < 5:
        return np.nan, np.nan
    values = []
    for mask in (positive == 0, positive == 1):
        values.extend([
            np.bincount(codes, weights=mask.astype(float), minlength=k),
            np.bincount(codes, weights=y * mask, minlength=k),
        ])
    counts0, sums0, counts1, sums1 = values
    draws = np.random.default_rng(seed).integers(0, k, size=(reps, k))
    c0 = counts0[draws].sum(axis=1)
    y0 = sums0[draws].sum(axis=1)
    c1 = counts1[draws].sum(axis=1)
    y1 = sums1[draws].sum(axis=1)
    delta = y1 / np.maximum(c1, 1) - y0 / np.maximum(c0, 1)
    delta = delta[np.isfinite(delta)]
    if not len(delta):
        return np.nan, np.nan
    return tuple(float(v) for v in np.quantile(delta, [.025, .975]))


def _load_phase_h_spearman(root: Path, run_id: str) -> pd.DataFrame:
    path = root / "outputs" / "exploratory_oracle" / run_id / "associations" / "response_associations.csv"
    if not path.exists():
        return pd.DataFrame()
    prior = pd.read_csv(path)
    return prior.loc[
        prior["analysis_unit"].eq("pixel") & prior["scheme"].eq("S0_all"),
        ["exposure", "outcome", "n", "spearman_rho", "spearman_p"],
    ].drop_duplicates(["exposure", "outcome"])


def _association_table(
    data: pd.DataFrame, config: dict[str, Any], prior: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    populations = {
        "mask_all": data,
        "mask_excluding_soil85": data.loc[~pd.to_numeric(data["SoilType"], errors="coerce").eq(85)].copy(),
    }
    reps = int(config["block_bootstrap"]["replicates"])
    seed = int(config["seed"])
    block_size = int(config["spatial_block_size_m"])
    for population, sample in populations.items():
        for exposure in (config["primary_exposure"], config["supplementary_exposure"]):
            for outcome in config["outcomes"]:
                pair = sample[[exposure, outcome, "x", "y"]].apply(pd.to_numeric, errors="coerce")
                keep = np.isfinite(pair[[exposure, outcome, "x", "y"]]).all(axis=1).to_numpy()
                x = pair.loc[keep, exposure].to_numpy(dtype=float)
                y = pair.loc[keep, outcome].to_numpy(dtype=float)
                blocks = _block_ids(pair.loc[keep].assign(x=pair.loc[keep, "x"], y=pair.loc[keep, "y"]), block_size)
                if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
                    continue
                hx = np.log1p(x)
                hrow = prior.loc[prior["exposure"].eq(exposure) & prior["outcome"].eq(outcome)] if population == "mask_all" else pd.DataFrame()
                if len(hrow) and int(hrow.iloc[0]["n"]) == len(x):
                    rho = float(hrow.iloc[0]["spearman_rho"])
                    rho_p = float(hrow.iloc[0]["spearman_p"])
                    rho_source = "Phase H S0_all; same complete-case n"
                else:
                    result = spearmanr(x, y)
                    rho, rho_p = float(result.statistic), float(result.pvalue)
                    rho_source = "recomputed on this complete-case population"
                raw_pearson = pearsonr(x, y)
                transformed_pearson = pearsonr(hx, y)
                tau = kendalltau(x, y, variant="b", method="asymptotic")
                mean_x, mean_y = float(hx.mean()), float(y.mean())
                var_x, var_y = float(np.var(hx)), float(np.var(y))
                slope = float(np.mean((hx - mean_x) * (y - mean_y)) / var_x)
                intercept = mean_y - slope * mean_x
                standardized_beta = slope * np.sqrt(var_x) / np.sqrt(var_y)
                block_ci = _block_bootstrap_correlation(hx, y, blocks, reps=reps, seed=seed + len(rows))
                rows.append({
                    "population": population, "exposure": exposure, "outcome": outcome,
                    "n_complete": len(x), "spatial_blocks_5km": int(pd.Series(blocks).nunique()),
                    "zero_fraction": float(np.mean(x == 0)),
                    "pearson_r_raw_Aa": float(raw_pearson.statistic), "pearson_p_raw_Aa_nominal": float(raw_pearson.pvalue),
                    "pearson_r_log1p_Aa": float(transformed_pearson.statistic), "pearson_p_log1p_Aa_nominal": float(transformed_pearson.pvalue),
                    "spearman_rho": rho, "spearman_p_nominal": rho_p, "spearman_source": rho_source,
                    "kendall_tau_b": float(tau.statistic), "kendall_p_nominal": float(tau.pvalue),
                    "kendall_n": len(x), "linear_slope_y_per_log1p_Aa": slope,
                    "linear_intercept": intercept, "standardized_beta_log1p_Aa": standardized_beta,
                    "r_squared_linear_log1p_Aa": float(transformed_pearson.statistic ** 2),
                    "block_bootstrap_slope_ci_low": block_ci["slope_ci_low"],
                    "block_bootstrap_slope_ci_high": block_ci["slope_ci_high"],
                    "block_bootstrap_beta_ci_low": block_ci["standardized_beta_ci_low"],
                    "block_bootstrap_beta_ci_high": block_ci["standardized_beta_ci_high"],
                    "block_bootstrap_replicates": reps, "block_size_m": block_size,
                    "outcome_guided_selection": True,
                })
    return pd.DataFrame(rows)


def _zero_positive_table(data: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    populations = {
        "mask_all": data,
        "mask_excluding_soil85": data.loc[~pd.to_numeric(data["SoilType"], errors="coerce").eq(85)].copy(),
    }
    reps = int(config["block_bootstrap"]["replicates"])
    for population, sample in populations.items():
        for exposure in (config["primary_exposure"], config["supplementary_exposure"]):
            for outcome in config["outcomes"]:
                pair = sample[[exposure, outcome, "x", "y"]].apply(pd.to_numeric, errors="coerce")
                keep = np.isfinite(pair[[exposure, outcome, "x", "y"]]).all(axis=1).to_numpy()
                x = pair.loc[keep, exposure].to_numpy(dtype=float)
                y = pair.loc[keep, outcome].to_numpy(dtype=float)
                blocks = _block_ids(pair.loc[keep], int(config["spatial_block_size_m"]))
                d = (x > 0).astype(np.int8)
                y0, y1 = y[d == 0], y[d == 1]
                if len(y0) < 2 or len(y1) < 2:
                    continue
                sd0, sd1 = float(np.std(y0, ddof=1)), float(np.std(y1, ddof=1))
                pooled = np.sqrt(((len(y0) - 1) * sd0 ** 2 + (len(y1) - 1) * sd1 ** 2) / (len(y0) + len(y1) - 2))
                cohen_d = float((y1.mean() - y0.mean()) / pooled) if pooled > 0 else np.nan
                correction = 1 - 3 / (4 * (len(y0) + len(y1)) - 9)
                ci_low, ci_high = _block_bootstrap_mean_difference(y, d, blocks, reps=reps,
                                                                    seed=int(config["seed"]) + len(rows))
                rows.append({
                    "population": population, "exposure": exposure, "outcome": outcome,
                    "n_zero": len(y0), "n_positive": len(y1), "zero_fraction": float(np.mean(d == 0)),
                    "mean_zero": float(y0.mean()), "mean_positive": float(y1.mean()),
                    "median_zero": float(np.median(y0)), "median_positive": float(np.median(y1)),
                    "mean_difference_positive_minus_zero": float(y1.mean() - y0.mean()),
                    "block_bootstrap_difference_ci_low": ci_low, "block_bootstrap_difference_ci_high": ci_high,
                    "standardized_mean_difference_hedges_g": cohen_d * correction,
                    "naive_welch_p_nominal": float(__import__("scipy.stats", fromlist=["ttest_ind"]).ttest_ind(y1, y0, equal_var=False).pvalue),
                    "block_bootstrap_replicates": reps, "block_size_m": int(config["spatial_block_size_m"]),
                    "outcome_guided_selection": True,
                })
    return pd.DataFrame(rows)


class _UnivariateModel:
    def __init__(self, family: str, *, seed: int, config: dict[str, Any], threshold: float | None = None,
                 exposure_bounds: tuple[float, float] | None = None):
        self.family = family
        self.seed = seed
        self.config = config
        self.threshold = threshold
        self.exposure_bounds = exposure_bounds
        self.spline: SplineTransformer | None = None
        self.estimator: Any = None
        self.fit_n = 0

    def _design(self, x: np.ndarray, *, fit_transformer: bool = False) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        d = (x > 0).astype(float)
        u = np.log1p(x)
        if self.family in {"linear_log1p", "random_forest", "extra_trees", "hist_gradient_boosting"}:
            return u.reshape(-1, 1)
        if self.family == "quadratic_log1p":
            return np.column_stack([u, u * u])
        if self.family == "zero_indicator":
            return d.reshape(-1, 1)
        if self.family == "two_part_linear":
            log_positive = np.where(d > 0, np.log(np.maximum(x, np.finfo(float).tiny)), 0.0)
            return np.column_stack([d, d * log_positive])
        if self.family == "segmented":
            assert self.threshold is not None
            knot = np.log1p(self.threshold)
            return np.column_stack([d, d * u, d * np.maximum(u - knot, 0.0)])
        if self.family in {"gam_style_penalized_spline", "two_part_spline"}:
            if self.spline is None:
                if not fit_transformer:
                    raise RuntimeError("Spline transformer was not fit.")
                if self.family == "gam_style_penalized_spline":
                    source = u.reshape(-1, 1)
                else:
                    positive = x[x > 0]
                    if len(positive) < 20:
                        raise ValueError("Too few positive exposure observations for two-part spline.")
                    source = np.log(positive).reshape(-1, 1)
                self.spline = SplineTransformer(n_knots=int(self.config["model_search"]["spline_knots"]),
                                                degree=3, knots="quantile", include_bias=False,
                                                extrapolation="linear")
                self.spline.fit(source)
            if self.family == "gam_style_penalized_spline":
                return self.spline.transform(u.reshape(-1, 1))
            positive_train = np.log(x[x > 0])
            safe_zero = float(np.median(positive_train)) if len(positive_train) else 0.0
            logged = np.where(d > 0, np.log(np.maximum(x, np.finfo(float).tiny)), safe_zero)
            smooth = self.spline.transform(logged.reshape(-1, 1)) * d[:, None]
            return np.column_stack([d, smooth])
        raise ValueError(f"Unknown model family: {self.family}")

    def fit(self, x: np.ndarray, y: np.ndarray) -> "_UnivariateModel":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        max_rows = int(self.config["model_search"]["tree_training_max_rows"])
        if self.family in {"random_forest", "extra_trees", "hist_gradient_boosting"} and len(x) > max_rows:
            take = np.sort(np.random.default_rng(self.seed).choice(len(x), max_rows, replace=False))
            x, y = x[take], y[take]
        self.fit_n = len(y)
        if self.family == "statsmodels_gam":
            from statsmodels.gam.api import BSplines, GLMGam
            smooth_x = np.log1p(x).reshape(-1, 1)
            lower, upper = self.exposure_bounds or (float(smooth_x.min()), float(smooth_x.max()))
            smoother = BSplines(
                smooth_x, df=[int(self.config["model_search"]["gam_spline_df"])], degree=[3],
                include_intercept=False,
                knot_kwds=[{"lower_bound": lower, "upper_bound": upper, "spacing": "quantile"}],
            )
            self.estimator = GLMGam(
                y, exog=np.ones((len(y), 1)), smoother=smoother,
                alpha=float(self.config["model_search"]["gam_alpha"]),
            ).fit(maxiter=200, tol=1e-8)
            return self
        design = self._design(x, fit_transformer=True)
        alpha = float(self.config["model_search"]["spline_ridge_alpha"])
        if self.family in {"gam_style_penalized_spline", "two_part_spline"}:
            self.estimator = Ridge(alpha=alpha)
        elif self.family == "random_forest":
            self.estimator = RandomForestRegressor(n_estimators=100, min_samples_leaf=30,
                                                   max_features=1.0, n_jobs=-1, random_state=self.seed)
        elif self.family == "extra_trees":
            self.estimator = ExtraTreesRegressor(n_estimators=100, min_samples_leaf=30,
                                                 max_features=1.0, n_jobs=-1, random_state=self.seed)
        elif self.family == "hist_gradient_boosting":
            self.estimator = HistGradientBoostingRegressor(max_iter=150, max_leaf_nodes=15,
                                                           min_samples_leaf=50, l2_regularization=.1,
                                                           random_state=self.seed)
        else:
            self.estimator = LinearRegression()
        self.estimator.fit(design, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self.family == "statsmodels_gam":
            return np.asarray(self.estimator.predict(
                exog=np.ones((len(x), 1)), exog_smooth=np.log1p(x).reshape(-1, 1),
            ), dtype=float)
        return np.asarray(self.estimator.predict(self._design(x)), dtype=float)


def _score(y: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(y, predicted))),
        "mae": float(mean_absolute_error(y, predicted)),
    }


def _spatial_split(x: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int,
                   test_fraction: float, validation_fraction: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    train_val, test = next(first.split(x, y, groups=groups))
    second = GroupShuffleSplit(n_splits=1, test_size=validation_fraction, random_state=seed + 1)
    train_rel, val_rel = next(second.split(x[train_val], y[train_val], groups=groups[train_val]))
    return train_val[train_rel], train_val[val_rel], test


def _threshold_candidates(x_train: np.ndarray, quantiles: list[float]) -> list[float]:
    positive = x_train[x_train > 0]
    values = sorted({float(np.quantile(positive, q)) for q in quantiles}) if len(positive) else []
    return [v for v in values if np.isfinite(v) and v > 0]


def _model_search(data: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_trials: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    exposure = config["primary_exposure"]
    families = ["linear_log1p", "quadratic_log1p", "statsmodels_gam", "gam_style_penalized_spline", "zero_indicator",
                "two_part_linear", "two_part_spline", "random_forest", "extra_trees", "hist_gradient_boosting"]
    split_cfg = config["spatial_split"]
    for outcome_index, outcome in enumerate(config["outcomes"]):
        columns = [exposure, outcome, "x", "y"]
        pair = data[columns].apply(pd.to_numeric, errors="coerce")
        keep = np.isfinite(pair[columns]).all(axis=1).to_numpy()
        frame = pair.loc[keep].reset_index(drop=True)
        x = frame[exposure].to_numpy(dtype=float)
        y = frame[outcome].to_numpy(dtype=float)
        groups = _block_ids(frame, int(config["spatial_block_size_m"]))
        train, validation, test = _spatial_split(
            x, y, groups, int(config["seed"]) + outcome_index,
            float(split_cfg["test_fraction"]), float(split_cfg["validation_fraction_of_train_validation"]),
        )
        candidates: list[tuple[str, float | None]] = [(family, None) for family in families]
        thresholds = _threshold_candidates(x[train], list(config["model_search"]["segmented_threshold_quantiles"]))
        candidates.extend(("segmented", threshold) for threshold in thresholds)
        fitted: list[tuple[dict[str, Any], _UnivariateModel]] = []
        for candidate_index, (family, threshold) in enumerate(candidates):
            seed = int(config["seed"]) + outcome_index * 100 + candidate_index
            try:
                model = _UnivariateModel(
                    family, seed=seed, config=config, threshold=threshold,
                    exposure_bounds=(0.0, float(np.log1p(np.max(x)))),
                ).fit(x[train], y[train])
                val_score = _score(y[validation], model.predict(x[validation]))
                test_score = _score(y[test], model.predict(x[test]))
                row = {
                    "exposure": exposure, "outcome": outcome, "model_family": family,
                    "threshold_original": threshold, "train_n": len(train), "validation_n": len(validation),
                    "test_n": len(test), "train_blocks": len(np.unique(groups[train])),
                    "validation_blocks": len(np.unique(groups[validation])), "test_blocks": len(np.unique(groups[test])),
                    "model_fit_n": model.fit_n, "validation_r2": val_score["r2"],
                    "validation_rmse": val_score["rmse"], "validation_mae": val_score["mae"],
                    "test_r2": test_score["r2"], "test_rmse": test_score["rmse"], "test_mae": test_score["mae"],
                    "selection_metric": "validation_r2", "outcome_guided_selection": True,
                    "evaluation_set": "5km spatial-block validation and held-out test",
                    "tree_training_capped": family in {"random_forest", "extra_trees", "hist_gradient_boosting"},
                    "status": "ok",
                }
                fitted.append((row, model))
            except Exception as exc:
                row = {"exposure": exposure, "outcome": outcome, "model_family": family,
                       "threshold_original": threshold, "train_n": len(train), "validation_n": len(validation),
                       "test_n": len(test), "model_fit_n": 0, "validation_r2": np.nan,
                       "validation_rmse": np.nan, "validation_mae": np.nan, "test_r2": np.nan,
                       "test_rmse": np.nan, "test_mae": np.nan, "selection_metric": "validation_r2",
                       "outcome_guided_selection": True, "evaluation_set": "5km spatial-block validation and held-out test",
                       "tree_training_capped": family in {"random_forest", "extra_trees", "hist_gradient_boosting"},
                       "status": f"fit_error:{type(exc).__name__}:{exc}"}
                all_trials.append(row)
        if not fitted:
            best_rows.append({"outcome": outcome, "status": "no_estimable_model"})
            continue
        fitted.sort(key=lambda pair_: pair_[0]["validation_r2"], reverse=True)
        best_row, best_model = fitted[0]
        for rank, (row, _) in enumerate(fitted, 1):
            row["validation_rank"] = rank
            row["selected_by_validation"] = rank == 1
            all_trials.append(row)
        best_rows.append({
            "exposure": exposure, "outcome": outcome, "selected_model_family": best_row["model_family"],
            "selected_threshold_original": best_row["threshold_original"],
            "validation_r2": best_row["validation_r2"], "validation_rmse": best_row["validation_rmse"],
            "test_r2": best_row["test_r2"], "test_rmse": best_row["test_rmse"],
            "test_mae": best_row["test_mae"], "n_complete": len(x),
            "train_n": best_row["train_n"], "validation_n": best_row["validation_n"], "test_n": best_row["test_n"],
            "train_blocks": best_row["train_blocks"], "validation_blocks": best_row["validation_blocks"],
            "test_blocks": best_row["test_blocks"],
            "selection_metric": "validation_r2", "outcome_guided_selection": True, "status": "ok",
        })
        # Refit the validation-selected form on the complete outcome-specific population for descriptive curves.
        refit = _UnivariateModel(best_row["model_family"], seed=int(config["seed"]) + outcome_index,
                                 config=config, threshold=best_row["threshold_original"],
                                 exposure_bounds=(0.0, float(np.log1p(np.max(x))))).fit(x, y)
        grid = np.linspace(0, float(np.quantile(x, .995)), 250)
        predicted = refit.predict(grid)
        for gx, gy in zip(grid, predicted):
            curve_rows.append({"exposure": exposure, "exposure_original": float(gx),
                               "log1p_exposure": float(np.log1p(gx)), "outcome": outcome,
                               "predicted_outcome": float(gy), "selected_model_family": best_row["model_family"],
                               "selected_threshold_original": best_row["threshold_original"],
                               "outcome_guided_selection": True, "curve_fit_scope": "complete-case population refit"})
    return pd.DataFrame(all_trials), pd.DataFrame(best_rows), pd.DataFrame(curve_rows)


def _quantile_regressions(data: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    exposure = config["primary_exposure"]
    cfg = config["quantile_regression"]
    split_cfg = config["spatial_split"]
    for outcome_index, outcome in enumerate(config["outcomes"]):
        pair = data[[exposure, outcome, "x", "y"]].apply(pd.to_numeric, errors="coerce")
        keep = np.isfinite(pair[[exposure, outcome, "x", "y"]]).all(axis=1).to_numpy()
        frame = pair.loc[keep].reset_index(drop=True)
        x = frame[exposure].to_numpy(dtype=float)
        y = frame[outcome].to_numpy(dtype=float)
        groups = _block_ids(frame, int(config["spatial_block_size_m"]))
        train, validation, test = _spatial_split(
            x, y, groups, int(config["seed"]) + outcome_index,
            float(split_cfg["test_fraction"]), float(split_cfg["validation_fraction_of_train_validation"]),
        )
        max_n = int(cfg["training_sample_max_rows"])
        if len(train) > max_n:
            sample_train = np.sort(np.random.default_rng(int(config["seed"]) + 500 + outcome_index)
                                   .choice(train, max_n, replace=False))
        else:
            sample_train = train
        design_train = sm.add_constant(np.log1p(x[sample_train]), has_constant="add")
        design_val = sm.add_constant(np.log1p(x[validation]), has_constant="add")
        design_test = sm.add_constant(np.log1p(x[test]), has_constant="add")
        for tau in cfg["quantiles"]:
            base = {"exposure": exposure, "outcome": outcome, "quantile": float(tau),
                    "train_sample_n": len(sample_train), "train_population_n": len(train),
                    "validation_n": len(validation), "test_n": len(test),
                    "outcome_guided_selection": True,
                    "evaluation_set": "5km spatial-block validation and held-out test"}
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    fit = sm.QuantReg(y[sample_train], design_train).fit(
                        q=float(tau), max_iter=int(cfg["max_iterations"]), p_tol=1e-6,
                    )
                val_pred = np.asarray(fit.predict(design_val))
                test_pred = np.asarray(fit.predict(design_test))
                val_resid = y[validation] - val_pred
                test_resid = y[test] - test_pred
                val_loss = np.maximum(tau * val_resid, (tau - 1) * val_resid).mean()
                test_loss = np.maximum(tau * test_resid, (tau - 1) * test_resid).mean()
                params = np.asarray(fit.params, dtype=float)
                convergence_warnings = sorted({warning.category.__name__ for warning in caught
                                               if issubclass(warning.category, (IterationLimitWarning, ConvergenceWarning))})
                fit_status = "ok" if not convergence_warnings else "not_converged:" + ",".join(convergence_warnings)
                rows.append({**base, "intercept": float(params[0]), "slope_per_log1p_Aa": float(params[1]),
                             "validation_pinball_loss": float(val_loss), "test_pinball_loss": float(test_loss),
                             "pseudo_r_squared_apparent": float(getattr(fit, "prsquared", np.nan)),
                             "iterations": int(getattr(fit, "iterations", 0)),
                             "max_iterations": int(cfg["max_iterations"]),
                             "convergence_warnings": ";".join(convergence_warnings), "status": fit_status})
            except Exception as exc:
                rows.append({**base, "intercept": np.nan, "slope_per_log1p_Aa": np.nan,
                             "validation_pinball_loss": np.nan, "test_pinball_loss": np.nan,
                             "pseudo_r_squared_apparent": np.nan, "iterations": np.nan,
                             "max_iterations": int(cfg["max_iterations"]), "convergence_warnings": "",
                             "status": f"fit_error:{type(exc).__name__}:{exc}"})
    return pd.DataFrame(rows)


def _full_sample_gam_curves(data: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit full-population Gaussian GAM curves with 5 km cluster-robust intervals."""
    from statsmodels.gam.api import BSplines, GLMGam

    curve_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    exposure = config["primary_exposure"]
    block_size = int(config["spatial_block_size_m"])
    df_smooth = int(config["model_search"]["gam_spline_df"])
    alpha = float(config["model_search"]["gam_alpha"])
    for outcome in config["outcomes"]:
        columns = [exposure, outcome, "x", "y"]
        pair = data[columns].apply(pd.to_numeric, errors="coerce")
        keep = np.isfinite(pair[columns]).all(axis=1).to_numpy()
        frame = pair.loc[keep].reset_index(drop=True)
        x = frame[exposure].to_numpy(dtype=float)
        y = frame[outcome].to_numpy(dtype=float)
        if len(x) < 100 or np.ptp(x) == 0:
            summary_rows.append({"exposure": exposure, "outcome": outcome, "n": len(x),
                                "spatial_blocks_5km": 0, "spline_df_basis": df_smooth,
                                "penalty_alpha": alpha, "effective_degrees_freedom": np.nan,
                                "local_slope_min_per_log1p_Aa": np.nan,
                                "local_slope_max_per_log1p_Aa": np.nan, "status": "non_estimable"})
            continue
        blocks = _block_ids(frame, block_size)
        smooth_x = np.log1p(x).reshape(-1, 1)
        upper = float(smooth_x.max())
        try:
            smoother = BSplines(
                smooth_x, df=[df_smooth], degree=[3], include_intercept=False,
                knot_kwds=[{"lower_bound": 0.0, "upper_bound": upper, "spacing": "quantile"}],
            )
            fit = GLMGam(
                y, exog=np.ones((len(y), 1)), smoother=smoother, alpha=alpha,
            ).fit(maxiter=200, tol=1e-8, cov_type="cluster", cov_kwds={"groups": blocks})
        except Exception as exc:
            summary_rows.append({"exposure": exposure, "outcome": outcome, "n": len(x),
                                "spatial_blocks_5km": int(pd.Series(blocks).nunique()),
                                "spline_df_basis": df_smooth, "penalty_alpha": alpha,
                                "effective_degrees_freedom": np.nan,
                                "local_slope_min_per_log1p_Aa": np.nan,
                                "local_slope_max_per_log1p_Aa": np.nan,
                                "status": f"fit_error:{type(exc).__name__}:{exc}"})
            continue
        grid = np.linspace(0, float(np.quantile(x, .995)), 250)
        grid_smooth = np.log1p(grid).reshape(-1, 1)
        predicted = np.asarray(fit.predict(exog=np.ones((len(grid), 1)), exog_smooth=grid_smooth), dtype=float)
        grid_basis = smoother.transform(grid_smooth)
        design_grid = np.column_stack([np.ones(len(grid)), grid_basis])
        covariance = np.asarray(fit.cov_params(), dtype=float)
        variance = np.einsum("ij,jk,ik->i", design_grid, covariance, design_grid)
        standard_error = np.sqrt(np.maximum(variance, 0.0))
        lower = predicted - 1.96 * standard_error
        upper_ci = predicted + 1.96 * standard_error
        local_slope = np.gradient(predicted, np.log1p(grid), edge_order=2)
        design = np.column_stack([np.ones(len(y)), smoother.basis])
        cross_product = design.T @ design
        penalty = np.zeros_like(cross_product)
        penalty[1:, 1:] = alpha * smoother.penalty_matrices[0]
        hat_trace = float(np.trace(np.linalg.solve(cross_product + penalty, cross_product)))
        n_blocks = int(pd.Series(blocks).nunique())
        summary_rows.append({
            "exposure": exposure, "outcome": outcome, "n": len(x), "spatial_blocks_5km": n_blocks,
            "spline_df_basis": df_smooth, "penalty_alpha": alpha, "effective_degrees_freedom": hat_trace,
            "local_slope_min_per_log1p_Aa": float(np.nanmin(local_slope)),
            "local_slope_max_per_log1p_Aa": float(np.nanmax(local_slope)),
            "cluster_robust_covariance": "5km spatial blocks; pointwise normal approximation",
            "status": "ok",
        })
        for gx, gy, se, lo, hi, slope in zip(grid, predicted, standard_error, lower, upper_ci, local_slope):
            curve_rows.append({
                "exposure": exposure, "exposure_original": float(gx), "log1p_exposure": float(np.log1p(gx)),
                "outcome": outcome, "predicted_outcome": float(gy), "standard_error_cluster_robust": float(se),
                "pointwise_ci95_low": float(lo), "pointwise_ci95_high": float(hi),
                "local_slope_per_log1p_Aa": float(slope), "effective_degrees_freedom": hat_trace,
                "n": len(x), "spatial_blocks_5km": n_blocks, "outcome_guided_selection": True,
            })
    return pd.DataFrame(curve_rows), pd.DataFrame(summary_rows)


def _phase_h_model_reference(root: Path, run_id: str, output: Path) -> pd.DataFrame:
    path = root / "outputs" / "exploratory_oracle" / run_id / "models" / "model_search_trials.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    selected = frame.loc[
        frame["analysis_unit"].eq("pixel") & frame["exposure"].eq("Aa_20yr") & frame["scheme"].eq("S0_all")
    ].copy()
    selected.to_csv(output, index=False)
    return selected


def _save_figure(fig: plt.Figure, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(directory / f"{stem}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _figures(run_dir: Path, associations: pd.DataFrame, zero: pd.DataFrame, curves: pd.DataFrame,
             best: pd.DataFrame, quantiles: pd.DataFrame, gam_curves: pd.DataFrame, gam_summary: pd.DataFrame,
             data: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    fig_root = run_dir / "figures"
    for name in ("forest_plots", "response_curves", "global_response", "model_comparison"):
        (fig_root / name).mkdir(parents=True, exist_ok=True)
    primary = associations.loc[(associations["population"] == "mask_all")]
    heat_fields = [
        ("pearson_r_raw_Aa", "Pearson / Aa_20yr"), ("spearman_rho", "Spearman / Aa_20yr"),
        ("standardized_beta_log1p_Aa", "標準化 β / log1p(Aa_20yr)"),
    ]
    supplemental = associations.loc[associations["population"].eq("mask_all")]
    matrix = np.full((len(config["outcomes"]), 6), np.nan)
    labels = []
    col_labels = []
    for j, exposure in enumerate((config["primary_exposure"], config["supplementary_exposure"])):
        for k, (field, title) in enumerate(heat_fields):
            col_labels.append(title if j == 0 else title.replace("20yr", "2020"))
            for i, outcome in enumerate(config["outcomes"]):
                hit = supplemental.loc[supplemental["exposure"].eq(exposure) & supplemental["outcome"].eq(outcome)]
                if len(hit):
                    matrix[i, j * 3 + k] = float(hit.iloc[0][field])
    labels = [LABELS[o] for o in config["outcomes"]]
    fig, ax = plt.subplots(figsize=(13, 6.6))
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(6), col_labels, rotation=20, ha="right")
    ax.set_yticks(np.arange(9), labels)
    for i in range(9):
        for j in range(6):
            value = matrix[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(value) > .55 else PALETTE["dark"])
    ax.set_title("侵蚀暴露与九项土壤健康结局的全掩膜关联")
    ax.set_xlabel("指标与暴露变换")
    ax.set_ylabel("健康结局（预测栅格）")
    fig.colorbar(image, ax=ax, label="相关系数 / 标准化效应（-1 至 1）", shrink=.85)
    fig.tight_layout()
    _save_figure(fig, fig_root / "global_response", "I01_global_association_heatmap")
    primary.to_csv(fig_root / "global_response" / "I01_global_association_heatmap_data.csv", index=False)

    forest = primary.loc[primary["exposure"].eq(config["primary_exposure"])].copy()
    forest = forest.set_index("outcome").reindex(config["outcomes"]).reset_index()
    yloc = np.arange(len(forest))
    estimate = forest["standardized_beta_log1p_Aa"].to_numpy(dtype=float)
    lower = forest["block_bootstrap_beta_ci_low"].to_numpy(dtype=float)
    upper = forest["block_bootstrap_beta_ci_high"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    ax.errorbar(estimate, yloc, xerr=np.vstack([estimate - lower, upper - estimate]), fmt="o",
                color=PALETTE["blue"], ecolor=PALETTE["muted"], capsize=3, linewidth=1.3)
    ax.axvline(0, color=PALETTE["dark"], linewidth=1)
    ax.set_yticks(yloc, [LABELS[o] for o in forest["outcome"]])
    ax.invert_yaxis()
    ax.set_xlabel("标准化 β（结局对 log1p(Aa_20yr) 的线性关系）")
    ax.set_title("九项指标的标准化线性效应与 5 km 空间块区间")
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=.7)
    ax.text(.01, -.14, "误差线为 399 次 5 km 空间块自助法百分位区间；不是像元独立性假设下的常规置信区间。",
            transform=ax.transAxes, fontsize=8, color=PALETTE["muted"])
    fig.tight_layout()
    _save_figure(fig, fig_root / "forest_plots", "I02_standardized_effect_forest")
    forest.to_csv(fig_root / "forest_plots" / "I02_standardized_effect_forest_data.csv", index=False)

    primary_data = data[[config["primary_exposure"], *config["outcomes"]]].apply(pd.to_numeric, errors="coerce")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for ax, outcome in zip(axes, ("SHI_candidate", "OM", "SW")):
        pair = data[[config["primary_exposure"], outcome]].apply(pd.to_numeric, errors="coerce").dropna()
        ax.hexbin(np.log1p(pair[config["primary_exposure"]]), pair[outcome], gridsize=60,
                  mincnt=1, bins="log", cmap="Blues", linewidths=0)
        ax.set_title(LABELS[outcome])
        ax.set_xlabel("log1p(Aa_20yr；原始单位未核实)")
        ax.set_ylabel("预测栅格值")
        ax.grid(color=PALETTE["grid"], linewidth=.4, alpha=.6)
    fig.suptitle("SHI、OM 与 SW 的全样本侵蚀—健康二维密度", y=1.04)
    _save_figure(fig, fig_root / "global_response", "I03_SHI_OM_SW_density")

    box_data = []
    for outcome in config["outcomes"]:
        pair = data[[config["primary_exposure"], outcome]].apply(pd.to_numeric, errors="coerce").dropna()
        values = pair[outcome].to_numpy(dtype=float)
        xvals = pair[config["primary_exposure"]].to_numpy(dtype=float)
        box_data.append((values[xvals == 0], values[xvals > 0]))
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), constrained_layout=True)
    for ax, outcome, (zero_values, positive_values) in zip(axes.flat, config["outcomes"], box_data):
        bp = ax.boxplot([zero_values, positive_values], tick_labels=["Aa=0", "Aa>0"], showfliers=False,
                        whis=(5, 95), patch_artist=True, medianprops={"color": PALETTE["dark"]})
        for patch, color in zip(bp["boxes"], ("#a8c5d8", "#e6b08b")):
            patch.set_facecolor(color)
            patch.set_alpha(.85)
        ax.set_title(LABELS[outcome])
        ax.set_ylabel("预测栅格值")
        ax.grid(axis="y", color=PALETTE["grid"], linewidth=.5)
    fig.suptitle("零侵蚀与正侵蚀像元的健康结局分布（须线为第 5–95 百分位）", y=1.02)
    _save_figure(fig, fig_root / "global_response", "I04_zero_positive_boxplots")

    curve_data = curves.loc[curves["exposure"].eq(config["primary_exposure"])]
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), constrained_layout=True)
    best_index = best.set_index("outcome")
    for ax, outcome in zip(axes.flat, config["outcomes"]):
        sample = curve_data.loc[curve_data["outcome"].eq(outcome)]
        if sample.empty:
            ax.set_visible(False)
            continue
        ax.plot(sample["log1p_exposure"], sample["predicted_outcome"], color=PALETTE["orange"], linewidth=2)
        ax.set_title(f"{LABELS[outcome]}\n{best_index.loc[outcome, 'selected_model_family']}; test R²={best_index.loc[outcome, 'test_r2']:.3f}")
        ax.set_xlabel("log1p(Aa_20yr)")
        ax.set_ylabel("拟合结局值")
        ax.grid(color=PALETTE["grid"], linewidth=.5)
    fig.suptitle("空间留出验证选定模型的全样本重拟合响应曲线（描述性；无区间带）", y=1.02)
    _save_figure(fig, fig_root / "response_curves", "I05_global_response_curves")
    curve_data.to_csv(fig_root / "response_curves" / "I05_global_response_curves_data.csv", index=False)

    z = zero.loc[(zero["population"].eq("mask_all")) & (zero["exposure"].eq(config["primary_exposure"]))]
    z = z.set_index("outcome").reindex(config["outcomes"]).reset_index()
    yloc = np.arange(len(z))
    est = z["standardized_mean_difference_hedges_g"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    colors = [PALETTE["blue"] if v < 0 else PALETTE["orange"] for v in est]
    ax.barh(yloc, est, color=colors, alpha=.82)
    ax.axvline(0, color=PALETTE["dark"], linewidth=1)
    ax.set_yticks(yloc, [LABELS[o] for o in z["outcome"]])
    ax.invert_yaxis()
    ax.set_xlabel("Hedges' g（正侵蚀 - 零侵蚀）")
    ax.set_title("零侵蚀状态与正侵蚀状态的标准化结局差异")
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=.7)
    fig.tight_layout()
    _save_figure(fig, fig_root / "forest_plots", "I06_zero_positive_effect_forest")
    z.to_csv(fig_root / "forest_plots" / "I06_zero_positive_effect_forest_data.csv", index=False)

    q = quantiles.loc[quantiles["status"].eq("ok")]
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), constrained_layout=True)
    for ax, outcome in zip(axes.flat, config["outcomes"]):
        values = q.loc[q["outcome"].eq(outcome)].sort_values("quantile")
        ax.plot(values["quantile"], values["slope_per_log1p_Aa"], marker="o", color=PALETTE["blue"])
        ax.axhline(0, color=PALETTE["dark"], linewidth=.8)
        ax.set_title(LABELS[outcome])
        ax.set_xlabel("条件分位数 τ")
        ax.set_ylabel("斜率 / log1p(Aa)")
        ax.grid(color=PALETTE["grid"], linewidth=.5)
    fig.suptitle("分位数回归：不同健康分布位置的侵蚀斜率", y=1.02)
    _save_figure(fig, fig_root / "model_comparison", "I07_quantile_slopes")
    q.to_csv(fig_root / "model_comparison" / "I07_quantile_slopes_data.csv", index=False)
    gam = gam_curves.loc[gam_curves["exposure"].eq(config["primary_exposure"])]
    gam_sum = gam_summary.set_index("outcome")
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), constrained_layout=True)
    for ax, outcome in zip(axes.flat, config["outcomes"]):
        sample = gam.loc[gam["outcome"].eq(outcome)]
        if sample.empty:
            ax.set_visible(False)
            continue
        ax.fill_between(sample["log1p_exposure"], sample["pointwise_ci95_low"],
                        sample["pointwise_ci95_high"], color="#a8c5d8", alpha=.42, linewidth=0)
        ax.plot(sample["log1p_exposure"], sample["predicted_outcome"], color=PALETTE["blue"], linewidth=1.8)
        summary = gam_sum.loc[outcome]
        ax.set_title(f"{LABELS[outcome]}\nedf={summary['effective_degrees_freedom']:.2f}")
        ax.set_xlabel("log1p(Aa_20yr)")
        ax.set_ylabel("拟合结局值")
        ax.grid(color=PALETTE["grid"], linewidth=.5)
    fig.suptitle("全样本 Gaussian GAM 响应曲线与 5 km 区块稳健点态 95% 区间", y=1.02)
    _save_figure(fig, fig_root / "response_curves", "I08_full_sample_gam_curves")
    gam.to_csv(fig_root / "response_curves" / "I08_full_sample_gam_curves_data.csv", index=False)
    return [
        "figures/global_response/I01_global_association_heatmap.svg",
        "figures/forest_plots/I02_standardized_effect_forest.svg",
        "figures/global_response/I03_SHI_OM_SW_density.svg",
        "figures/global_response/I04_zero_positive_boxplots.svg",
        "figures/response_curves/I05_global_response_curves.svg",
        "figures/forest_plots/I06_zero_positive_effect_forest.svg",
        "figures/model_comparison/I07_quantile_slopes.svg",
        "figures/response_curves/I08_full_sample_gam_curves.svg",
    ]


def _md_table(frame: pd.DataFrame, columns: list[str], *, digits: int = 3) -> str:
    selected = frame.loc[:, columns].copy()
    def format_value(value: Any) -> str:
        if pd.isna(value):
            return "—"
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{digits}f}"
        return str(value).replace("|", "\\|")
    headers = [str(col) for col in columns]
    rows = [[format_value(value) for value in row] for row in selected.itertuples(index=False, name=None)]
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] +
                     ["| " + " | ".join(row) + " |" for row in rows])


def _write_report(run_dir: Path, associations: pd.DataFrame, zero: pd.DataFrame, trials: pd.DataFrame,
                  best: pd.DataFrame, quantiles: pd.DataFrame, gam_summary: pd.DataFrame,
                  figures: list[str], config: dict[str, Any], data: pd.DataFrame,
                  input_paths: dict[str, str]) -> Path:
    output = run_dir / "reports" / "PHASE_I_GLOBAL_RESPONSE_REPORT.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    primary = associations.loc[(associations["population"].eq("mask_all")) &
                               (associations["exposure"].eq(config["primary_exposure"]))].copy()
    order = primary.sort_values("spearman_rho")
    key_columns = ["outcome", "n_complete", "pearson_r_raw_Aa", "spearman_rho", "kendall_tau_b",
                   "standardized_beta_log1p_Aa", "r_squared_linear_log1p_Aa"]
    key = order[key_columns].copy()
    key["outcome"] = key["outcome"].map(LABELS)
    z = zero.loc[(zero["population"].eq("mask_all")) & (zero["exposure"].eq(config["primary_exposure"]))].copy()
    z = z.assign(abs_g=z["standardized_mean_difference_hedges_g"].abs()).sort_values("abs_g", ascending=False).head(9)
    z_table = z[["outcome", "n_zero", "n_positive", "mean_difference_positive_minus_zero",
                 "block_bootstrap_difference_ci_low", "block_bootstrap_difference_ci_high",
                 "standardized_mean_difference_hedges_g"]].copy()
    z_table["outcome"] = z_table["outcome"].map(LABELS)
    chosen = best.copy()
    chosen["outcome"] = chosen["outcome"].map(LABELS)
    chosen_columns = ["outcome", "selected_model_family", "selected_threshold_original", "validation_r2",
                      "test_r2", "test_rmse", "test_mae", "n_complete", "train_n", "validation_n", "test_n"]
    gam_table = gam_summary.copy()
    if not gam_table.empty:
        gam_table["outcome"] = gam_table["outcome"].map(LABELS)
    gam_columns = ["outcome", "n", "spatial_blocks_5km", "effective_degrees_freedom",
                   "local_slope_min_per_log1p_Aa", "local_slope_max_per_log1p_Aa"]
    gam_table = gam_table.reindex(columns=gam_columns)
    rho_min = order.iloc[0]
    rho_max = order.iloc[-1]
    aa = pd.to_numeric(data[config["primary_exposure"]], errors="coerce")
    finite_aa = aa[np.isfinite(aa)]
    positive_counts = zero.loc[(zero["population"].eq("mask_all")) &
                               (zero["exposure"].eq(config["primary_exposure"]))]
    shi_zero = positive_counts.loc[positive_counts["outcome"].eq("SHI_candidate")].iloc[0]
    pixel_shares = best[["train_n", "validation_n", "test_n"]].div(best["n_complete"], axis=0)
    split_share_summary = {
        name: (float(pixel_shares[f"{name}_n"].min()), float(pixel_shares[f"{name}_n"].max()))
        for name in ("train", "validation", "test")
    }
    geom = data.loc[np.isfinite(pd.to_numeric(data[config["primary_exposure"]], errors="coerce")), "SoilType"]
    soil85 = int(pd.to_numeric(geom, errors="coerce").eq(85).sum())
    quant_fail = int((quantiles["status"] != "ok").sum()) if len(quantiles) else 0
    report = f"""# Phase I-1 全样本侵蚀—健康响应关系

**运行**：`{config['run_id']}`　**模式**：EXPLORATORY_ORACLE　**结果导向选择**：`true`<br>
**主暴露**：Aa_20yr　**辅助暴露**：Aa_2020　**输入数据库**：`{config['input_run_id']}`

## 主要结果

全掩膜 Aa_20yr 的有效范围为 {finite_aa.min():.6g}–{finite_aa.max():.6g}；单位及其精确时间口径尚未从现有元数据确认。Spearman 秩相关最负的是 **{LABELS[rho_min['outcome']]}**（ρ={rho_min['spearman_rho']:.3f}），最正的是 **{LABELS[rho_max['outcome']]}**（ρ={rho_max['spearman_rho']:.3f}）。这些是同一掩膜内的预测栅格关联，不表示实测样点效应或因果效应。

### 九项指标的全掩膜关联

{_md_table(key, key_columns)}

`Pearson raw Aa` 使用原始 Aa 值；线性斜率、标准化 β 与 R² 使用 `log1p(Aa)`。Phase H 的 `valid_cell=True` 先要求环境栈各变量完整，完整案例数与 I-1 不同，因此这里对各项结局的全掩膜完整案例重新计算 Spearman；Kendall τ-b 也在完整案例上计算。所有常规相关检验 p 值均按像元独立的名义值保存，不作为显著性结论，因为像元具有空间依赖。逐行来源和旧结果样本量见 `spearman_source` 字段。

![侵蚀—健康关联热图](../figures/global_response/I01_global_association_heatmap.svg)

![标准化效应森林图](../figures/forest_plots/I02_standardized_effect_forest.svg)

## 零值状态与正侵蚀强度

SHI 的 Aa_20yr 完整案例中有 {int(shi_zero['n_zero']):,} 个零值像元，零值比例为 {shi_zero['zero_fraction']:.1%}；其他结局的零值完整案例数会因结局缺失而略有不同。下表按 Hedges' g 绝对值排序，均值差定义为正侵蚀减零侵蚀；区间由 399 次 5 km 空间块自助法给出。

{_md_table(z_table, list(z_table.columns))}

![SHI、OM、SW 二维密度图](../figures/global_response/I03_SHI_OM_SW_density.svg)

![零值与正值箱线图](../figures/global_response/I04_zero_positive_boxplots.svg)

图中箱体比较零值与正值结局分布，须线为第 5–95 百分位。主分析保留掩膜内的 SoilType=85；全结果另含排除 85 的敏感性样本。该码在 Phase H 文档中标为水体，但其像元仍在输入研究掩膜数据库中。

## 非线性、分段与模型比较

按 5 km 投影坐标块分配训练、验证和测试集合（21/7/7 个空间块，整块留出）。各块面积和结局缺失率不均，像元占比并非固定 60%/20%/20%：训练 {split_share_summary['train'][0]:.1%}–{split_share_summary['train'][1]:.1%}、验证 {split_share_summary['validation'][0]:.1%}–{split_share_summary['validation'][1]:.1%}、测试 {split_share_summary['test'][0]:.1%}–{split_share_summary['test'][1]:.1%}。表中给出每项结局精确样本量。模型按验证集 R² 排序；测试集仅报告一次留出表现，不用于模型选择。分段候选拐点取训练集正 Aa 分位数 0.2–0.8，按验证集择优，因此相关阈值属于 Oracle 探索，不是独立确认的临界值。

{_md_table(chosen, chosen_columns)}

模型搜索包含 statsmodels Gaussian GAM（单一 B 样条平滑项，df={config['model_search']['gam_spline_df']}、α={config['model_search']['gam_alpha']}）及惩罚 B 样条敏感候选；树模型训练样本上限为 {config['model_search']['tree_training_max_rows']:,}，但验证与测试仍覆盖相应完整空间留出集。上一张曲线图是验证集选定模型在结局完整案例总体上的描述性重拟合，不附区间。

![全局响应曲线](../figures/response_curves/I05_global_response_curves.svg)

### 全样本 GAM 曲线、区间与局部斜率

另在每项结局的全部完整案例上拟合 Gaussian GAM。区间为按 5 km 空间块聚类稳健协方差计算的点态 95% 正态近似区间；有效自由度按惩罚样条帽子矩阵迹计算。局部斜率表示对 `log1p(Aa_20yr)` 的数值导数。当前约有 35 个空间块，区间仅作探索性不确定性描述。

{_md_table(gam_table, gam_columns)}

![全样本 GAM 响应曲线和区间](../figures/response_curves/I08_full_sample_gam_curves.svg)

![零值状态效应](../figures/forest_plots/I06_zero_positive_effect_forest.svg)

### 分位数回归

分位数线性模型在训练空间块中固定随机抽取最多 {config['quantile_regression']['training_sample_max_rows']:,} 行拟合，并在空间验证/测试块报告 pinball loss；最大迭代数为 {config['quantile_regression']['max_iterations']}。共有 {quant_fail} 个拟合未达到收敛标准或失败，图中仅展示状态为 `ok` 的估计；逐项状态见 `results/nonlinear/quantile_regression.csv`。

![分位数斜率](../figures/model_comparison/I07_quantile_slopes.svg)

## 样本范围与数据口径

- 数据库各有 {len(data):,} 个掩膜行；不再把两个栅格栈各自的 `valid_cell` 标记相交。各指标的完整案例数单独记录在 `results/global_response/global_associations.csv` 与模型表中。
- 可分析 Aa_20yr 像元为 {int(np.isfinite(aa).sum()):,}；其中 SoilType=85 为 {soil85:,} 行（全数据库内该码为 {int(pd.to_numeric(data['SoilType'], errors='coerce').eq(85).sum()):,} 行）。Aa_20yr 零值比例约为 {shi_zero['zero_fraction']:.1%}（SHI 完整案例）。
- 九项健康指标均来自土壤健康随机森林预测栅格；这些像元不等于独立实测样本。
- Phase H Spearman 行用于样本范围对照，但因 Phase H 使用更窄的 `valid_cell=True` 环境完整子集，本次 Spearman 重新计算；Phase H 旧模型族试验另存于 `results/references/phase_h_s0_model_trials.csv` 供审计。

## 解释边界

1. 暴露 Aa 的单位与时间口径不完整；本报告沿用存储数值，未换算单位，也未重建/验证 RUSLE 乘积。
2. 像元级名义 p 值受空间自相关影响。线性 β 的不确定性用 5 km 空间块重抽样描述；空间留出模型评估不能替代独立外部验证。
3. 预测栅格结果只支持栅格层面的条件关联探索。不得据此推断独立样点层面的效应或因果关系。
4. 土类码 85 同时保留在主掩膜分析，并提供排除敏感性；二者总体定义不同。
5. 所有模型和响应曲线均属于结果导向的 Oracle 探索；后续阶段需沿用该标记并报告选择过程与样本保留情况。

## 可复现文件

- 配置：`configs/phase_i.yaml`
- 全样本相关与效应量：`results/global_response/global_associations.csv`
- 零值/正值对比：`results/zero_positive/zero_positive_effects.csv`
- 空间块模型比较：`results/model_search/global_model_search_trials.csv`
- 验证集模型排行榜：`leaderboards/global_response_model_leaderboard.csv`
- 分位数回归：`results/nonlinear/quantile_regression.csv`
- 全样本 GAM 曲线、点态区间与局部斜率：`results/nonlinear/full_sample_gam_curves.csv`、`results/nonlinear/full_sample_gam_summary.csv`
- 完整运行信息：`manifests/provenance.json`
- 输入数据库：`{input_paths['environment_erosion']}`；`{input_paths['health']}`
"""
    output.write_text(report, encoding="utf-8")
    return output


def _version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                                capture_output=True, text=True)
        return result.stdout.strip()
    except Exception:
        return None


def _git_status(root: Path) -> str | None:
    try:
        result = subprocess.run(["git", "status", "--short", "--untracked-files=normal"], cwd=root,
                                check=True, capture_output=True, text=True)
        return result.stdout.strip() or "clean"
    except Exception:
        return None


def run(config_path: Path) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("mode") != "EXPLORATORY_ORACLE" or config.get("outcome_guided_selection") is not True:
        raise ValueError("Phase I runner requires explicit EXPLORATORY_ORACLE authorization in config.")
    if config["outcomes"] != OUTCOMES:
        raise ValueError("The Phase I-1 run must include SHI and all eight health indicators.")
    base = ROOT / config["output_root"]
    run_dir = base / config["run_id"]
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing Phase I run: {run_dir}")
    run_dir.mkdir(parents=True)
    directories = ["configs", "results/global_response", "results/zero_positive", "results/model_search",
                   "results/nonlinear", "results/references", "leaderboards", "figures", "manifests", "reports"]
    for name in directories:
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, run_dir / "configs" / config_path.name)
    data, input_paths = _read_inputs(config)
    prior = _load_phase_h_spearman(ROOT, config["phase_h_run_id"])
    associations = _association_table(data, config, prior)
    zero = _zero_positive_table(data, config)
    trials, best, curves = _model_search(data, config)
    quantiles = _quantile_regressions(data, config)
    gam_curves, gam_summary = _full_sample_gam_curves(data, config)
    phase_h_models = _phase_h_model_reference(ROOT, config["phase_h_run_id"],
                                               run_dir / "results/references/phase_h_s0_model_trials.csv")
    associations.to_csv(run_dir / "results/global_response/global_associations.csv", index=False)
    zero.to_csv(run_dir / "results/zero_positive/zero_positive_effects.csv", index=False)
    trials.to_csv(run_dir / "results/model_search/global_model_search_trials.csv", index=False)
    best.to_csv(run_dir / "leaderboards/global_response_model_leaderboard.csv", index=False)
    curves.to_csv(run_dir / "results/nonlinear/response_curve_points.csv", index=False)
    quantiles.to_csv(run_dir / "results/nonlinear/quantile_regression.csv", index=False)
    gam_curves.to_csv(run_dir / "results/nonlinear/full_sample_gam_curves.csv", index=False)
    gam_summary.to_csv(run_dir / "results/nonlinear/full_sample_gam_summary.csv", index=False)
    figures = _figures(run_dir, associations, zero, curves, best, quantiles, gam_curves, gam_summary, data, config)
    report = _write_report(run_dir, associations, zero, trials, best, quantiles, gam_summary,
                           figures, config, data, input_paths)
    try:
        import rasterio
        rasterio_version = rasterio.__version__
    except Exception:
        rasterio_version = None
    provenance = {
        "phase": "I-1", "mode": config["mode"], "outcome_guided_selection": True,
        "run_id": config["run_id"], "input_run_id": config["input_run_id"],
        "phase_h_run_id": config["phase_h_run_id"], "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_at_execution": _git_commit(ROOT), "git_worktree_status_at_execution": _git_status(ROOT),
        "runner": "experiments/oracle_exploration/phase_i.py",
        "config": str((run_dir / "configs" / config_path.name).relative_to(ROOT)),
        "input_tables": input_paths, "output_report": str(report.relative_to(ROOT)),
        "input_rows_per_database": len(data), "mask_population_definition": "all rows in the aligned crop-mask databases; per-analysis complete cases",
        "soiltype_85_count": int(pd.to_numeric(data["SoilType"], errors="coerce").eq(85).sum()),
        "soiltype_85_sensitivity": "reported separately; not excluded from primary mask population",
        "health_values": "prediction rasters; not independent measured samples",
        "spearman_reuse": "Phase H full-population S0_all rows reused only when complete-case n matches",
        "kendall": "recomputed on complete cases using Kendall tau-b",
        "spatial_block_size_m": int(config["spatial_block_size_m"]),
        "model_selection": "validation R2; 5 km blocks held out for validation/test; test metrics are not used for selection",
        "tree_training_max_rows": int(config["model_search"]["tree_training_max_rows"]),
        "quantile_training_sample_max_rows": int(config["quantile_regression"]["training_sample_max_rows"]),
        "quantile_max_iterations": int(config["quantile_regression"]["max_iterations"]),
        "full_sample_gam": {
            "family": "statsmodels Gaussian GAM with one B-spline smooth term",
            "spline_df": int(config["model_search"]["gam_spline_df"]),
            "penalty_alpha": float(config["model_search"]["gam_alpha"]),
            "covariance": "cluster-robust by 5 km spatial blocks",
            "effective_degrees_of_freedom": "trace of penalized-spline hat matrix",
            "interval": "pointwise 95% normal approximation",
        },
        "phase_h_model_reference_rows": len(phase_h_models), "outcomes": config["outcomes"],
        "software_versions": {
            "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "scikit-learn": _version("scikit-learn"),
            "statsmodels": sm.__version__, "matplotlib": matplotlib.__version__,
            "pyarrow": _version("pyarrow"), "pyyaml": _version("PyYAML"), "rasterio": rasterio_version,
        },
        "argv": sys.argv,
    }
    (run_dir / "manifests" / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    output_files = [str(p.relative_to(ROOT)) for p in sorted(run_dir.rglob("*")) if p.is_file()]
    (run_dir / "manifests" / "output_manifest.json").write_text(json.dumps({"run_id": config["run_id"], "files": output_files}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"run_dir={run_dir.relative_to(ROOT)}")
    print(f"report={report.relative_to(ROOT)}")
    print(f"associations={len(associations)} zero_positive={len(zero)} trials={len(trials)} best={len(best)} quantiles={len(quantiles)}")
    print(f"quantile_failures={int((quantiles['status'] != 'ok').sum()) if len(quantiles) else 0}")
    return run_dir


def render_existing(run_dir: Path, config_path: Path, *, recompute_quantiles: bool = False,
                    recompute_gam_curves: bool = False) -> Path:
    """Regenerate only the report and figure derivatives from a completed run."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = run_dir.resolve()
    if not run_dir.is_dir() or run_dir.parent != (ROOT / config["output_root"]).resolve():
        raise ValueError("The render target must be an existing direct child of the configured Phase I output root.")
    configs_dir = run_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    run_config_path = configs_dir / config_path.name
    if run_config_path.exists() and run_config_path.read_bytes() != config_path.read_bytes():
        initial_config_path = configs_dir / f"{config_path.stem}_initial{config_path.suffix}"
        if not initial_config_path.exists():
            shutil.copy2(run_config_path, initial_config_path)
    shutil.copy2(config_path, run_config_path)
    data, input_paths = _read_inputs(config)
    associations = pd.read_csv(run_dir / "results/global_response/global_associations.csv")
    zero = pd.read_csv(run_dir / "results/zero_positive/zero_positive_effects.csv")
    trials = pd.read_csv(run_dir / "results/model_search/global_model_search_trials.csv")
    best = pd.read_csv(run_dir / "leaderboards/global_response_model_leaderboard.csv")
    curves = pd.read_csv(run_dir / "results/nonlinear/response_curve_points.csv")
    quantiles = pd.read_csv(run_dir / "results/nonlinear/quantile_regression.csv")
    if recompute_quantiles:
        quantiles = _quantile_regressions(data, config)
        quantiles.to_csv(run_dir / "results/nonlinear/quantile_regression.csv", index=False)
    gam_curve_path = run_dir / "results/nonlinear/full_sample_gam_curves.csv"
    gam_summary_path = run_dir / "results/nonlinear/full_sample_gam_summary.csv"
    if recompute_gam_curves or not gam_curve_path.is_file() or not gam_summary_path.is_file():
        gam_curves, gam_summary = _full_sample_gam_curves(data, config)
        gam_curves.to_csv(gam_curve_path, index=False)
        gam_summary.to_csv(gam_summary_path, index=False)
    else:
        gam_curves = pd.read_csv(gam_curve_path)
        gam_summary = pd.read_csv(gam_summary_path)
    figures = _figures(run_dir, associations, zero, curves, best, quantiles,
                       gam_curves, gam_summary, data, config)
    report = _write_report(run_dir, associations, zero, trials, best, quantiles, gam_summary,
                           figures, config, data, input_paths)
    provenance_path = run_dir / "manifests/provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["config"] = str(run_config_path.relative_to(ROOT))
    provenance["final_config"] = str(run_config_path.relative_to(ROOT))
    initial_config_path = configs_dir / f"{config_path.stem}_initial{config_path.suffix}"
    if initial_config_path.exists():
        provenance["initial_config"] = str(initial_config_path.relative_to(ROOT))
    provenance["quantile_max_iterations"] = int(config["quantile_regression"]["max_iterations"])
    provenance["gam_settings"] = {
        "family": "statsmodels Gaussian GAM with one B-spline smooth term",
        "spline_df": int(config["model_search"]["gam_spline_df"]),
        "penalty_alpha": float(config["model_search"]["gam_alpha"]),
        "covariance": "cluster-robust by 5 km spatial blocks",
        "effective_degrees_of_freedom": "trace of penalized-spline hat matrix",
        "interval": "pointwise 95% normal approximation",
    }
    provenance["rendered_at_utc"] = datetime.now(timezone.utc).isoformat()
    provenance["rendering_revision"] = "Chinese-capable font and report wording/metrics; global association and model tables reused"
    if recompute_quantiles:
        provenance["quantile_refit_at_utc"] = datetime.now(timezone.utc).isoformat()
        provenance["quantile_refit"] = "quantile table recomputed only, with warning-based convergence status and configured iteration limit"
    if recompute_gam_curves:
        provenance["full_sample_gam_refit_at_utc"] = datetime.now(timezone.utc).isoformat()
        provenance["full_sample_gam_refit"] = "full-sample GAM curve, cluster-robust pointwise interval, effective df, and local slope tables regenerated"
    provenance["rendering_worktree_status"] = _git_status(ROOT)
    provenance_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    output_files = [str(path.relative_to(ROOT)) for path in sorted(run_dir.rglob("*"))
                    if path.is_file() and path.name != "output_manifest.json"]
    (run_dir / "manifests/output_manifest.json").write_text(
        json.dumps({"run_id": config["run_id"], "files": output_files}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"rendered_report={report.relative_to(ROOT)}")
    print(f"rendered_figures={len(figures)}; analysis_tables_reused=true")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "phase_i.yaml")
    parser.add_argument("--render-existing", type=Path, help="Regenerate figures/report from an existing run without rerunning analysis.")
    parser.add_argument("--recompute-quantiles", action="store_true", help="With --render-existing, recompute only the quantile table before rendering.")
    parser.add_argument("--recompute-gam-curves", action="store_true", help="With --render-existing, recompute full-sample GAM curves and diagnostics.")
    args = parser.parse_args()
    if args.render_existing is not None:
        render_existing(args.render_existing, args.config.resolve(), recompute_quantiles=args.recompute_quantiles,
                        recompute_gam_curves=args.recompute_gam_curves)
    else:
        if args.recompute_quantiles or args.recompute_gam_curves:
            parser.error("--recompute-quantiles and --recompute-gam-curves require --render-existing")
        run(args.config.resolve())


if __name__ == "__main__":
    main()
