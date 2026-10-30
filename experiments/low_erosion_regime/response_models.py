"""Rank-based, paired, nonlinear and spatially blocked response analyses."""

from __future__ import annotations

import warnings
from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr, wilcoxon, binomtest
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import HuberRegressor, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer

from experiments.low_erosion_regime.figures import OUTCOME_LABELS


MODEL_FAMILIES = (
    "linear", "quadratic", "natural_spline", "additive_gam", "huber",
    "random_forest", "extra_trees", "hist_gradient_boosting",
)


def matched_pixel_population(
    pairs: pd.DataFrame, cells: pd.DataFrame, *, exposure: str, method: str,
) -> pd.DataFrame:
    """Return each unique raster cell used as either endpoint of a matched pair."""
    required_pairs = {"exposure", "method", "high_erosion_cell_id", "low_erosion_cell_id"}
    required_cells = {"cell_id"}
    if not required_pairs.issubset(pairs.columns):
        raise ValueError(f"pair table is missing columns: {sorted(required_pairs - set(pairs.columns))}")
    if not required_cells.issubset(cells.columns):
        raise ValueError("cell table must include cell_id")
    selected = pairs.loc[pairs["exposure"].eq(exposure) & pairs["method"].eq(method)]
    endpoint_ids = pd.unique(pd.concat([
        selected["high_erosion_cell_id"], selected["low_erosion_cell_id"],
    ], ignore_index=True).dropna())
    matched = cells.loc[cells["cell_id"].isin(endpoint_ids)].copy()
    matched = matched.drop_duplicates("cell_id").sort_values("cell_id", kind="stable").reset_index(drop=True)
    return matched


def _finite_xy(x: Iterable[float], y: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    return x[keep], y[keep]


def _block_bootstrap_slope(
    x: np.ndarray, y: np.ndarray, blocks: np.ndarray, *, replicates: int = 399, seed: int = 20261029,
) -> tuple[float, float]:
    frame = pd.DataFrame({"block": blocks.astype(str), "n": 1.0, "x": x, "y": y, "x2": x*x, "xy": x*y})
    block = frame.groupby("block", observed=True, sort=True).sum()
    if len(block) < 5:
        return np.nan, np.nan
    sufficient = block[["n", "x", "y", "x2", "xy"]].to_numpy()
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(sufficient), size=(replicates, len(sufficient)))
    totals = sufficient[draws].sum(axis=1)
    n, sx, sy, sx2, sxy = totals.T
    denominator = sx2 - sx * sx / n
    slopes = np.divide(sxy - sx * sy / n, denominator, out=np.full(replicates, np.nan), where=denominator > 1e-12)
    slopes = slopes[np.isfinite(slopes)]
    if not len(slopes):
        return np.nan, np.nan
    return float(np.quantile(slopes, .025)), float(np.quantile(slopes, .975))


def _test_pair(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan, np.nan, np.nan, np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sr = spearmanr(x, y)
        kt = kendalltau(x, y)
    return float(sr.statistic), float(sr.pvalue), float(kt.statistic), float(kt.pvalue)


def pixel_association(
    data: pd.DataFrame, *, exposure: str, outcome: str, scheme: str, threshold: float | None = None,
    max_kendall_n: int = 50000, seed: int = 20261029,
) -> dict[str, object]:
    """Full-population Spearman and reproducibly sampled Kendall association."""
    subset = data.loc[data["valid_cell"].fillna(False).astype(bool)].copy()
    x_all = pd.to_numeric(subset[exposure], errors="coerce").to_numpy(dtype=float)
    y_all = pd.to_numeric(subset[outcome], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(x_all) & np.isfinite(y_all)
    if scheme == "S1_zero_only":
        keep &= x_all == 0
    elif scheme == "positive_only":
        keep &= x_all > 0
    elif scheme == "distribution_threshold":
        if threshold is None:
            keep &= False
        else:
            keep &= x_all > threshold
    x, y = _finite_xy(x_all[keep], y_all[keep])
    n = len(x)
    result: dict[str, object] = {
        "analysis_unit": "pixel", "exposure": exposure, "outcome": outcome, "outcome_label": OUTCOME_LABELS.get(outcome, outcome),
        "scheme": scheme, "threshold_original": threshold, "n": n, "spatial_blocks": 0,
        "outcome_guided_selection": False,
    }
    if n < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        result.update({"status": "constant_or_insufficient_exposure", "spearman_rho": np.nan, "spearman_p": np.nan,
                       "kendall_tau": np.nan, "kendall_p": np.nan, "slope": np.nan, "slope_ci_low": np.nan,
                       "slope_ci_high": np.nan})
        return result
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sr = spearmanr(x, y)
        if n > max_kendall_n:
            rng = np.random.default_rng(seed)
            sample = np.sort(rng.choice(n, max_kendall_n, replace=False))
            kt = kendalltau(x[sample], y[sample])
            kt_n = len(sample)
        else:
            kt = kendalltau(x, y)
            kt_n = n
    slope = float(np.polyfit(np.log1p(x), y, 1)[0])
    block_ids = np.floor(pd.to_numeric(subset.loc[keep, "x"], errors="coerce").to_numpy(float) / 5000).astype(int).astype(str) + "_" + \
                np.floor(pd.to_numeric(subset.loc[keep, "y"], errors="coerce").to_numpy(float) / 5000).astype(int).astype(str)
    ci_low, ci_high = _block_bootstrap_slope(np.log1p(x), y, block_ids, seed=seed) if len(block_ids) == n else (np.nan, np.nan)
    result.update({"status": "ok", "spearman_rho": float(sr.statistic), "spearman_p": float(sr.pvalue),
                   "kendall_tau": float(kt.statistic), "kendall_p": float(kt.pvalue), "kendall_n": int(kt_n),
                   "slope": slope, "slope_ci_low": ci_low, "slope_ci_high": ci_high,
                   "spatial_blocks": int(pd.Series(block_ids).nunique())})
    return result


def matched_pixel_association(
    pairs: pd.DataFrame, cells: pd.DataFrame, *, exposure: str, outcome: str, method: str,
    scheme: str, threshold: float | None = None, seed: int = 20261029,
) -> dict[str, object]:
    """Summarize each matched endpoint cell once, without pair multiplicity."""
    matched = matched_pixel_population(pairs, cells, exposure=exposure, method=method)
    result = pixel_association(matched, exposure=exposure, outcome=outcome, scheme="S0_all", seed=seed)
    result.update({
        "analysis_unit": "matched_pixel", "method": method, "scheme": scheme,
        "threshold_original": threshold,
        "n_unique_matched_endpoints": int(matched["cell_id"].nunique()),
        "source_pair_rows": int(len(pairs)),
        "selection_scope": "association_on_unique_matched_endpoints",
    })
    return result


def paired_association(
    pairs: pd.DataFrame, *, exposure: str, outcome: str, scheme: str, method: str,
    threshold: float | None = None, seed: int = 20261029,
) -> dict[str, object]:
    """Rank association, paired differences, tests and spatial block bootstrap."""
    delta_col = f"delta_{outcome}"
    if delta_col not in pairs:
        return {"analysis_unit": "pair", "exposure": exposure, "outcome": outcome, "method": method,
                "scheme": scheme, "status": "missing_outcome", "n": 0}
    frame = pairs.loc[pairs["exposure"].eq(exposure) & pairs["method"].eq(method)].copy() if "method" in pairs else pairs.copy()
    if threshold is not None:
        if "retention_scheme" in frame and not frame["retention_scheme"].eq("both_endpoints_above_threshold").all():
            # Existing and rematched paths use independent tables, so only enforce
            # the endpoint condition when Aa endpoint columns are available.
            if {"erosion_high", "erosion_low"}.issubset(frame.columns):
                frame = frame.loc[frame["erosion_high"].gt(threshold) & frame["erosion_low"].gt(threshold)]
    x_col = "delta_log1p_erosion"
    if x_col not in frame and {"erosion_high", "erosion_low"}.issubset(frame.columns):
        frame[x_col] = np.log1p(pd.to_numeric(frame["erosion_high"], errors="coerce")) - np.log1p(
            pd.to_numeric(frame["erosion_low"], errors="coerce")
        )
    if x_col not in frame:
        return {"analysis_unit": "pair", "exposure": exposure, "outcome": outcome, "method": method,
                "scheme": scheme, "status": "missing_exposure_contrast", "n": 0}
    x, y = _finite_xy(frame[x_col], frame[delta_col])
    # Align block membership to the same complete-case mask.
    block_col = "spatial_block_id" if "spatial_block_id" in frame else None
    valid_mask = np.isfinite(pd.to_numeric(frame[x_col], errors="coerce")) & np.isfinite(pd.to_numeric(frame[delta_col], errors="coerce"))
    blocks = frame.loc[valid_mask, block_col].astype(str).to_numpy() if block_col else np.arange(len(x)).astype(str)
    n = len(x)
    result: dict[str, object] = {"analysis_unit": "pair", "exposure": exposure, "outcome": outcome,
                                 "outcome_label": OUTCOME_LABELS.get(outcome, outcome), "method": method,
                                 "scheme": scheme, "threshold_original": threshold, "n": n,
                                 "spatial_blocks": int(pd.Series(blocks).nunique()), "outcome_guided_selection": False}
    if n < 3:
        result.update({"status": "insufficient_pairs", "spearman_rho": np.nan, "kendall_tau": np.nan,
                       "median_delta_health": np.nan, "mean_delta_health": np.nan, "negative_delta_fraction": np.nan,
                       "wilcoxon_p": np.nan, "slope": np.nan, "slope_ci_low": np.nan, "slope_ci_high": np.nan})
        return result
    rho, sp, tau, kp = _test_pair(x, y)
    nonzero = y[np.abs(y) > 1e-12]
    try:
        w_p = float(wilcoxon(y, alternative="two-sided", zero_method="wilcox").pvalue) if len(nonzero) else 1.0
    except ValueError:
        w_p = 1.0
    try:
        sign_p = float(binomtest(int(np.count_nonzero(y < 0)), len(nonzero), p=.5).pvalue) if len(nonzero) else 1.0
    except ValueError:
        sign_p = np.nan
    slope = float(np.polyfit(x, y, 1)[0]) if np.ptp(x) > 0 else np.nan
    ci_low, ci_high = _block_bootstrap_slope(x, y, blocks, seed=seed)
    result.update({"status": "ok", "spearman_rho": rho, "spearman_p": sp, "kendall_tau": tau, "kendall_p": kp,
                   "mean_delta_health": float(np.mean(y)), "median_delta_health": float(np.median(y)),
                   "negative_delta_fraction": float(np.mean(y < 0)), "nonzero_delta_n": int(len(nonzero)),
                   "wilcoxon_p": w_p, "sign_test_p": sign_p, "slope": slope,
                   "slope_ci_low": ci_low, "slope_ci_high": ci_high})
    return result


def _make_model(family: str, *, seed: int):
    if family == "linear":
        return LinearRegression()
    if family == "quadratic":
        return make_pipeline(PolynomialFeatures(degree=2, include_bias=False), LinearRegression())
    if family == "natural_spline":
        return make_pipeline(SplineTransformer(n_knots=5, degree=3, include_bias=False), LinearRegression())
    if family == "additive_gam":
        # A one-term Gaussian additive model represented by a penalized cubic B-spline basis.
        return make_pipeline(SplineTransformer(n_knots=7, degree=3, include_bias=False), Ridge(alpha=10.0))
    if family == "huber":
        return HuberRegressor(epsilon=1.35, alpha=0.0001, max_iter=200)
    if family == "random_forest":
        return RandomForestRegressor(n_estimators=80, min_samples_leaf=20, max_features=1.0,
                                     random_state=seed, n_jobs=1)
    if family == "extra_trees":
        return ExtraTreesRegressor(n_estimators=80, min_samples_leaf=20, max_features=1.0,
                                  random_state=seed, n_jobs=1)
    if family in {"hist_gradient_boosting", "gradient_boosting"}:
        return HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=15, l2_regularization=.1,
                                             min_samples_leaf=50, random_state=seed)
    raise ValueError(f"unsupported model family: {family}")


def model_trials(
    cells: pd.DataFrame, *, exposure: str, outcome: str, scheme: str, threshold: float | None = None,
    seed: int = 20261029, sample_size: int = 30000,
) -> pd.DataFrame:
    """Compare linear and nonlinear apparent fits on a fixed reproducible sample."""
    data = cells.loc[cells["valid_cell"].fillna(False).astype(bool)].copy()
    x = pd.to_numeric(data[exposure], errors="coerce").to_numpy(float)
    y = pd.to_numeric(data[outcome], errors="coerce").to_numpy(float)
    keep = np.isfinite(x) & np.isfinite(y)
    if scheme == "positive_only":
        keep &= x > 0
    elif scheme == "distribution_threshold":
        keep &= x > threshold if threshold is not None else False
    elif scheme == "S1_zero_only":
        keep &= x == 0
    x, y = x[keep], y[keep]
    if len(x) < 50 or np.ptp(x) == 0:
        return pd.DataFrame([{"analysis_unit": "pixel", "exposure": exposure, "outcome": outcome, "scheme": scheme,
                              "threshold_original": threshold, "model_family": name, "n": len(x),
                              "status": "non_estimable", "r_squared": np.nan,
                              "outcome_guided_selection": True} for name in MODEL_FAMILIES])
    rng = np.random.default_rng(seed)
    if len(x) > sample_size:
        take = np.sort(rng.choice(len(x), sample_size, replace=False))
        x, y = x[take], y[take]
    transformed = np.log1p(x).reshape(-1, 1)
    rows = []
    for family in MODEL_FAMILIES:
        model = _make_model(family, seed=seed)
        try:
            model.fit(transformed, y)
            pred = model.predict(transformed)
            r2 = float(r2_score(y, pred))
            mse = float(mean_squared_error(y, pred))
            # Simple endpoint direction summary is kept separate from R-squared.
            rank = spearmanr(x, y).statistic if np.ptp(y) > 0 else np.nan
            rows.append({"analysis_unit": "pixel", "exposure": exposure, "outcome": outcome, "scheme": scheme, "threshold_original": threshold,
                         "model_family": family, "n": len(x), "r_squared_apparent": r2, "mse_apparent": mse,
                         "spearman_rho": float(rank), "status": "ok", "evaluation_set": "apparent_fixed_sample",
                         "outcome_guided_selection": True, "selection_scope": "model_family_ranked_on_apparent_fit"})
        except Exception as exc:
            rows.append({"analysis_unit": "pixel", "exposure": exposure, "outcome": outcome, "scheme": scheme, "threshold_original": threshold,
                         "model_family": family, "n": len(x), "r_squared_apparent": np.nan,
                         "mse_apparent": np.nan, "spearman_rho": np.nan, "status": f"fit_error:{type(exc).__name__}",
                         "evaluation_set": "apparent_fixed_sample", "outcome_guided_selection": True,
                         "selection_scope": "model_family_ranked_on_apparent_fit"})
    return pd.DataFrame(rows)


def matched_pixel_model_trials(
    pairs: pd.DataFrame, cells: pd.DataFrame, *, exposure: str, outcome: str, method: str,
    scheme: str, threshold: float | None = None, seed: int = 20261029, sample_size: int = 30000,
) -> pd.DataFrame:
    """Compare response models on unique cells selected by a matched-pair scheme."""
    matched = matched_pixel_population(pairs, cells, exposure=exposure, method=method)
    result = model_trials(matched, exposure=exposure, outcome=outcome, scheme="S0_all",
                          seed=seed, sample_size=sample_size)
    result["analysis_unit"] = "matched_pixel"
    result["method"] = method
    result["scheme"] = scheme
    result["threshold_original"] = threshold
    result["n_unique_matched_endpoints"] = int(matched["cell_id"].nunique())
    result["source_pair_rows"] = int(pairs["exposure"].eq(exposure).mul(pairs["method"].eq(method)).sum())
    result["selection_scope"] = "model_family_ranked_on_apparent_fit_of_unique_matched_endpoints"
    return result


def paired_model_trials(
    pairs: pd.DataFrame, *, exposure: str, outcome: str, method: str, scheme: str,
    threshold: float | None = None, seed: int = 20261029, sample_size: int = 30000,
) -> pd.DataFrame:
    """Compare apparent model families on matched exposure and health differences."""
    delta = f"delta_{outcome}"
    if delta not in pairs or "delta_log1p_erosion" not in pairs:
        return pd.DataFrame()
    frame = pairs.loc[pairs["exposure"].eq(exposure) & pairs["method"].eq(method)].copy() if "method" in pairs else pairs.copy()
    x, y = _finite_xy(frame["delta_log1p_erosion"], frame[delta])
    if len(x) < 50 or np.ptp(x) == 0:
        return pd.DataFrame([{"analysis_unit": "pair", "exposure": exposure, "outcome": outcome,
                              "method": method, "scheme": scheme, "threshold_original": threshold,
                              "model_family": name, "n": len(x), "status": "non_estimable",
                              "r_squared_apparent": np.nan, "outcome_guided_selection": True}
                             for name in MODEL_FAMILIES])
    rng = np.random.default_rng(seed)
    if len(x) > sample_size:
        take = np.sort(rng.choice(len(x), sample_size, replace=False))
        x, y = x[take], y[take]
    predictor = x.reshape(-1, 1)
    rank = float(spearmanr(x, y).statistic) if np.ptp(y) > 0 else np.nan
    rows = []
    for family in MODEL_FAMILIES:
        model = _make_model(family, seed=seed)
        try:
            model.fit(predictor, y)
            prediction = model.predict(predictor)
            rows.append({"analysis_unit": "pair", "exposure": exposure, "outcome": outcome,
                         "method": method, "scheme": scheme, "threshold_original": threshold,
                         "model_family": family, "n": len(x),
                         "r_squared_apparent": float(r2_score(y, prediction)),
                         "mse_apparent": float(mean_squared_error(y, prediction)),
                         "spearman_rho": rank, "direction": "negative" if rank < 0 else "positive_or_null",
                         "status": "ok", "outcome_guided_selection": True,
                         "selection_scope": "model_family_ranked_on_apparent_fit"})
        except Exception as exc:
            rows.append({"analysis_unit": "pair", "exposure": exposure, "outcome": outcome,
                         "method": method, "scheme": scheme, "threshold_original": threshold,
                         "model_family": family, "n": len(x), "r_squared_apparent": np.nan,
                         "mse_apparent": np.nan, "spearman_rho": rank,
                         "status": f"fit_error:{type(exc).__name__}", "outcome_guided_selection": True,
                         "selection_scope": "model_family_ranked_on_apparent_fit"})
    return pd.DataFrame(rows)


def two_part_and_segmented_models(
    cells: pd.DataFrame, *, exposure: str, outcome: str, seed: int = 20261029,
    sample_size: int = 50000, threshold: float | None = None,
) -> pd.DataFrame:
    """Compare no-filter zero-indicator, positive-intensity spline and hinge models."""
    data = cells.loc[cells["valid_cell"].fillna(False).astype(bool)].copy()
    x_raw = pd.to_numeric(data[exposure], errors="coerce").to_numpy(float)
    y = pd.to_numeric(data[outcome], errors="coerce").to_numpy(float)
    keep = np.isfinite(x_raw) & np.isfinite(y)
    x_raw, y = x_raw[keep], y[keep]
    if len(x_raw) > sample_size:
        rng = np.random.default_rng(seed)
        selected = np.sort(rng.choice(len(x_raw), sample_size, replace=False))
        x_raw, y = x_raw[selected], y[selected]
    x = np.log1p(x_raw)
    zero = (x_raw == 0).astype(float)
    positive_log = np.where(x_raw > 0, np.log10(x_raw), 0.0).reshape(-1, 1)
    rows = []
    models = {
        "全样本线性 log1p(Aa)": (x.reshape(-1, 1), False),
        "零值指示 + 正值强度样条": (np.column_stack([zero, SplineTransformer(n_knots=5, degree=3, include_bias=False).fit_transform(positive_log)]), False),
    }
    if threshold is not None and threshold > 0:
        knot = float(np.log1p(threshold))
        hinge = np.maximum(x - knot, 0.0)
        models["阈值折线 + 零值指示"] = (np.column_stack([zero, x, hinge]), False)
    for label, (design, _) in models.items():
        model = LinearRegression().fit(design, y)
        predicted = model.predict(design)
        r2 = float(r2_score(y, predicted))
        sse = max(float(np.sum((y - predicted) ** 2)), 1e-12)
        p = int(design.shape[1] + 1)
        aic = float(len(y) * np.log(sse / len(y)) + 2 * p)
        rows.append({"exposure": exposure, "outcome": outcome, "model": label, "n": len(y),
                     "r_squared_apparent": r2, "aic_apparent": aic,
                     "zero_fraction": float(np.mean(zero)), "threshold_original": threshold,
                     "outcome_guided_selection": True, "evaluation_set": "apparent_fixed_sample"})
    return pd.DataFrame(rows)


def outcome_optimized_thresholds(
    cells: pd.DataFrame, *, exposure: str, outcome: str, seed: int = 20261029,
    quantiles: tuple[float, ...] = (.1, .2, .3, .4, .5, .6, .7, .8), sample_size: int = 20000,
) -> pd.DataFrame:
    """Search positive exposure quantiles for an explicitly Oracle threshold."""
    data = cells.loc[cells["valid_cell"].fillna(False).astype(bool)].copy()
    x = pd.to_numeric(data[exposure], errors="coerce").to_numpy(float)
    y = pd.to_numeric(data[outcome], errors="coerce").to_numpy(float)
    keep = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x, y = x[keep], y[keep]
    if len(x) < 100:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    if len(x) > sample_size:
        take = np.sort(rng.choice(len(x), sample_size, replace=False))
        x, y = x[take], y[take]
    rows = []
    for q in quantiles:
        threshold = float(np.quantile(x, q))
        selected = x > threshold
        if selected.sum() < max(100, int(.1 * len(x))):
            continue
        predictor = np.log1p(x[selected]).reshape(-1, 1)
        model = make_pipeline(SplineTransformer(n_knots=5, degree=3, include_bias=False), LinearRegression())
        model.fit(predictor, y[selected])
        score = float(r2_score(y[selected], model.predict(predictor)))
        rows.append({"exposure": exposure, "outcome": outcome, "threshold_quantile": q,
                     "threshold_original": threshold, "retained_fraction_positive": float(selected.mean()),
                     "model_family": "natural_spline", "r_squared_apparent": score,
                     "n": int(selected.sum()), "candidate_source": "outcome_optimized",
                     "outcome_guided_selection": True,
                     "selection_rule": "max_apparent_r_squared_across_positive_exposure_quantiles"})
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["selected_best_apparent"] = frame["r_squared_apparent"].eq(frame["r_squared_apparent"].max())
    return frame


def spatial_validation(
    cells: pd.DataFrame, *, exposure: str, outcome: str, scheme: str, threshold: float | None = None,
    seeds: tuple[int, ...] = tuple(range(20261029, 20261039)), buffer_m: int = 500,
    sample_size: int = 20000,
) -> pd.DataFrame:
    """Select model family on validation blocks and evaluate once on held-out blocks."""
    data = cells.loc[cells["valid_cell"].fillna(False).astype(bool)].copy()
    x_raw = pd.to_numeric(data[exposure], errors="coerce").to_numpy(float)
    y = pd.to_numeric(data[outcome], errors="coerce").to_numpy(float)
    x_coord = pd.to_numeric(data["x"], errors="coerce").to_numpy(float)
    y_coord = pd.to_numeric(data["y"], errors="coerce").to_numpy(float)
    keep = np.isfinite(x_raw) & np.isfinite(y) & np.isfinite(x_coord) & np.isfinite(y_coord)
    if scheme == "positive_only":
        keep &= x_raw > 0
    elif scheme == "distribution_threshold":
        keep &= x_raw > threshold if threshold is not None else False
    elif scheme == "S1_zero_only":
        keep &= x_raw == 0
    data = data.loc[keep].copy()
    x_raw, y, x_coord, y_coord = x_raw[keep], y[keep], x_coord[keep], y_coord[keep]
    blocks = np.floor(x_coord / 5000).astype(int).astype(str) + "_" + np.floor(y_coord / 5000).astype(int).astype(str)
    if len(np.unique(blocks)) < 5 or np.ptp(x_raw) == 0:
        return pd.DataFrame([{"exposure": exposure, "outcome": outcome, "scheme": scheme,
                              "threshold_original": threshold, "status": "insufficient_spatial_blocks_or_exposure",
                              "n": len(x_raw), "buffer_m": buffer_m, "outcome_guided_selection": True}])
    unique_blocks = np.unique(blocks)
    rows = []
    families = MODEL_FAMILIES
    x_all = np.log1p(x_raw).reshape(-1, 1)
    for split, seed in enumerate(seeds, 1):
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(unique_blocks)
        n_train = max(1, int(.6 * len(shuffled)))
        n_val = max(1, int(.2 * len(shuffled)))
        train_blocks = set(shuffled[:n_train])
        val_blocks = set(shuffled[n_train:n_train + n_val])
        test_blocks = set(shuffled[n_train + n_val:])
        if not test_blocks:
            test_blocks = {shuffled[-1]}
        train = np.isin(blocks, list(train_blocks))
        valid = np.isin(blocks, list(val_blocks))
        test = np.isin(blocks, list(test_blocks))
        # Remove training pixels inside the specified buffer around validation
        # and test block footprints. This preserves complete block roles.
        heldout = list(val_blocks | test_blocks)
        keep_train = np.ones(len(x_coord), dtype=bool)
        train_positions = np.flatnonzero(train)
        for block in heldout:
            block_x, block_y = map(int, block.split("_"))
            min_x, min_y = block_x * 5000, block_y * 5000
            dx = np.maximum(np.maximum(min_x - x_coord[train_positions], 0), x_coord[train_positions] - (min_x + 5000))
            dy = np.maximum(np.maximum(min_y - y_coord[train_positions], 0), y_coord[train_positions] - (min_y + 5000))
            keep_train[train_positions] &= np.hypot(dx, dy) > buffer_m
        train &= keep_train
        # Distance between distinct 5 km block interiors exceeds the requested
        # sub-km buffer; the same-block leakage path is closed by whole-block roles.
        selected = None
        best_val = -np.inf
        test_scores = {}
        for family in families:
            fit_idx = np.flatnonzero(train)
            if family in {"random_forest", "extra_trees", "hist_gradient_boosting"} and len(fit_idx) > sample_size:
                fit_idx = np.sort(rng.choice(fit_idx, sample_size, replace=False))
            model = _make_model(family, seed=seed)
            model.fit(x_all[fit_idx], y[fit_idx])
            val_r2 = float(r2_score(y[valid], model.predict(x_all[valid]))) if valid.sum() > 2 else -np.inf
            test_r2 = float(r2_score(y[test], model.predict(x_all[test]))) if test.sum() > 2 else np.nan
            test_scores[family] = test_r2
            if val_r2 > best_val:
                best_val, selected = val_r2, family
        rows.append({"exposure": exposure, "outcome": outcome, "scheme": scheme, "threshold_original": threshold,
                     "split": split, "seed": int(seed), "buffer_m": buffer_m, "n_train": int(train.sum()),
                     "n_validation": int(valid.sum()), "n_test": int(test.sum()), "train_blocks": len(train_blocks),
                     "validation_blocks": len(val_blocks), "test_blocks": len(test_blocks),
                     "selected_model_by_validation": selected, "validation_r_squared_selected": best_val,
                     "spatial_test_r_squared_selected": test_scores.get(selected, np.nan),
                     "family_test_scores_json": pd.Series(test_scores).to_json(),
                     "status": "ok", "outcome_guided_selection": True,
                     "selection_scope": "model_family_selected_on_validation_blocks"})
    return pd.DataFrame(rows)


def matched_pixel_spatial_validation(
    pairs: pd.DataFrame, cells: pd.DataFrame, *, exposure: str, outcome: str, method: str,
    scheme: str, threshold: float | None = None,
    seeds: tuple[int, ...] = tuple(range(20261029, 20261039)), buffer_m: int = 500,
    sample_size: int = 20000,
) -> pd.DataFrame:
    """Validate models on unique cells used by one matched-pair population."""
    matched = matched_pixel_population(pairs, cells, exposure=exposure, method=method)
    result = spatial_validation(
        matched, exposure=exposure, outcome=outcome, scheme="S0_all", threshold=None,
        seeds=seeds, buffer_m=buffer_m, sample_size=sample_size,
    )
    result["analysis_unit"] = "matched_pixel"
    result["method"] = method
    result["scheme"] = scheme
    result["threshold_original"] = threshold
    result["n_unique_matched_endpoints"] = int(matched["cell_id"].nunique())
    result["source_pair_rows"] = int(len(pairs))
    result["selection_scope"] = "model_family_selected_on_validation_blocks_of_unique_matched_endpoints"
    return result


def paired_spatial_validation(
    pairs: pd.DataFrame, *, exposure: str, outcome: str, method: str, scheme: str,
    threshold: float | None = None, seeds: tuple[int, ...] = tuple(range(20261029, 20261039)),
    buffer_m: int = 500, sample_size: int = 20000,
) -> pd.DataFrame:
    """Validate paired response models on whole 5 km blocks after boundary buffering."""
    delta = f"delta_{outcome}"
    needed = {"exposure", "method", "delta_log1p_erosion", delta}
    if not needed.issubset(pairs.columns):
        return pd.DataFrame()
    frame = pairs.loc[pairs["exposure"].eq(exposure) & pairs["method"].eq(method)].copy()
    if threshold is not None and {"erosion_high", "erosion_low"}.issubset(frame.columns):
        frame = frame.loc[frame["erosion_high"].gt(threshold) & frame["erosion_low"].gt(threshold)]
    if {"x_high", "x_low", "y_high", "y_low"}.issubset(frame.columns):
        high_block = np.floor(pd.to_numeric(frame["x_high"], errors="coerce") / 5000).astype("Int64").astype(str) + "_" + \
                     np.floor(pd.to_numeric(frame["y_high"], errors="coerce") / 5000).astype("Int64").astype(str)
        low_block = np.floor(pd.to_numeric(frame["x_low"], errors="coerce") / 5000).astype("Int64").astype(str) + "_" + \
                    np.floor(pd.to_numeric(frame["y_low"], errors="coerce") / 5000).astype("Int64").astype(str)
        same_block = high_block.eq(low_block)
        frame = frame.loc[same_block].copy()
    if "x_mid" in frame and "y_mid" in frame:
        x_coord = pd.to_numeric(frame["x_mid"], errors="coerce").to_numpy(float)
        y_coord = pd.to_numeric(frame["y_mid"], errors="coerce").to_numpy(float)
    elif {"x_high", "x_low", "y_high", "y_low"}.issubset(frame.columns):
        x_coord = (pd.to_numeric(frame["x_high"], errors="coerce").to_numpy(float) +
                   pd.to_numeric(frame["x_low"], errors="coerce").to_numpy(float)) / 2
        y_coord = (pd.to_numeric(frame["y_high"], errors="coerce").to_numpy(float) +
                   pd.to_numeric(frame["y_low"], errors="coerce").to_numpy(float)) / 2
    else:
        return pd.DataFrame()
    x = pd.to_numeric(frame["delta_log1p_erosion"], errors="coerce").to_numpy(float)
    y = pd.to_numeric(frame[delta], errors="coerce").to_numpy(float)
    keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(x_coord) & np.isfinite(y_coord)
    x, y, x_coord, y_coord = x[keep], y[keep], x_coord[keep], y_coord[keep]
    blocks = np.floor(x_coord / 5000).astype(int).astype(str) + "_" + np.floor(y_coord / 5000).astype(int).astype(str)
    unique_blocks = np.unique(blocks)
    if len(unique_blocks) < 5 or np.ptp(x) == 0:
        return pd.DataFrame([{"analysis_unit": "pair", "exposure": exposure, "outcome": outcome,
                              "method": method, "scheme": scheme, "threshold_original": threshold,
                              "n": len(x), "spatial_blocks": len(unique_blocks), "buffer_m": buffer_m,
                              "status": "insufficient_spatial_blocks_or_exposure",
                              "outcome_guided_selection": True}])
    rows = []
    families = MODEL_FAMILIES
    predictor = x.reshape(-1, 1)
    for split, seed in enumerate(seeds, 1):
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(unique_blocks)
        n_train = max(1, int(.6 * len(shuffled)))
        n_val = max(1, int(.2 * len(shuffled)))
        train_blocks = set(shuffled[:n_train])
        val_blocks = set(shuffled[n_train:n_train + n_val])
        test_blocks = set(shuffled[n_train + n_val:])
        if not test_blocks:
            test_blocks = {shuffled[-1]}
        train = np.isin(blocks, list(train_blocks))
        valid = np.isin(blocks, list(val_blocks))
        test = np.isin(blocks, list(test_blocks))
        train_positions = np.flatnonzero(train)
        keep_train = np.ones(len(x), dtype=bool)
        for block in val_blocks | test_blocks:
            bx, by = map(int, block.split("_"))
            min_x, min_y = bx * 5000, by * 5000
            dx = np.maximum(np.maximum(min_x - x_coord[train_positions], 0), x_coord[train_positions] - (min_x + 5000))
            dy = np.maximum(np.maximum(min_y - y_coord[train_positions], 0), y_coord[train_positions] - (min_y + 5000))
            keep_train[train_positions] &= np.hypot(dx, dy) > buffer_m
        train &= keep_train
        best_family, best_val = None, -np.inf
        family_scores = {}
        for family in families:
            fit_idx = np.flatnonzero(train)
            if family in {"random_forest", "extra_trees", "hist_gradient_boosting"} and len(fit_idx) > sample_size:
                fit_idx = np.sort(rng.choice(fit_idx, sample_size, replace=False))
            model = _make_model(family, seed=seed)
            model.fit(predictor[fit_idx], y[fit_idx])
            val_score = float(r2_score(y[valid], model.predict(predictor[valid]))) if valid.sum() > 2 else -np.inf
            test_score = float(r2_score(y[test], model.predict(predictor[test]))) if test.sum() > 2 else np.nan
            family_scores[family] = test_score
            if val_score > best_val:
                best_family, best_val = family, val_score
        rows.append({"analysis_unit": "pair", "exposure": exposure, "outcome": outcome, "method": method,
                     "scheme": scheme, "threshold_original": threshold, "split": split, "seed": int(seed),
                     "n_total_same_block_pairs": len(x), "n_train": int(train.sum()), "n_validation": int(valid.sum()),
                     "n_test": int(test.sum()), "train_blocks": len(train_blocks),
                     "validation_blocks": len(val_blocks), "test_blocks": len(test_blocks),
                     "buffer_m": buffer_m, "selected_model_by_validation": best_family,
                     "validation_r_squared_selected": best_val,
                     "spatial_test_r_squared_selected": family_scores.get(best_family, np.nan),
                     "family_test_scores_json": pd.Series(family_scores).to_json(), "status": "ok",
                     "outcome_guided_selection": True,
                     "selection_scope": "model_family_selected_on_validation_blocks"})
    return pd.DataFrame(rows)

