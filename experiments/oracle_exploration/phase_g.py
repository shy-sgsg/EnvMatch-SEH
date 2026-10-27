"""Phase G paired association summaries and outcome-aware result boards."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, norm, spearmanr, wilcoxon
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances_argmin
from sklearn.preprocessing import StandardScaler


def _finite_number(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def select_pair_spatial_p1_rows(spatial: pd.DataFrame) -> pd.DataFrame:
    """Keep the base P1 validation/test rows for the M3/M4 matched pairs."""

    return spatial.loc[
        spatial["method"].isin(["M3", "M4"])
        & spatial["evaluation_set"].isin(["validation", "test"])
        & spatial["validation_model"].eq("P1_intercept_ols")
        & spatial["environment_group"].isna()
    ].copy()


def summarize_pair_spatial_validation(spatial: pd.DataFrame) -> pd.DataFrame:
    """Summarize matched-pair P1 test scores and their training-fit slopes."""

    return spatial.groupby(
        ["method", "exposure", "outcome", "evaluation_set"], dropna=False, sort=True
    ).agg(
        median_spatial_r2=("r_squared", "median"),
        median_training_slope=("beta_log1p_erosion_train", "median"),
        training_negative_direction_fraction=(
            "direction_train",
            lambda values: float(pd.Series(values).eq("negative").mean()),
        ),
        spatial_seeds=("seed", "nunique"),
    ).reset_index()


def _block_bootstrap_intervals(
    x: np.ndarray,
    y: np.ndarray,
    blocks: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    """Bootstrap OLS slope and mean response by resampling whole spatial blocks."""

    if replicates < 1 or len(x) < 3:
        return {
            "bootstrap_mean_ci_low": float("nan"),
            "bootstrap_mean_ci_high": float("nan"),
            "bootstrap_slope_ci_low": float("nan"),
            "bootstrap_slope_ci_high": float("nan"),
        }
    grouped = pd.DataFrame(
        {
            "block": blocks.astype(str),
            "n": np.ones(len(x), dtype=float),
            "sum_x": x,
            "sum_y": y,
            "sum_x2": x * x,
            "sum_xy": x * y,
        }
    ).groupby("block", sort=True, observed=True).sum()
    values = grouped[["n", "sum_x", "sum_y", "sum_x2", "sum_xy"]].to_numpy(dtype=float)
    if len(values) < 2:
        return {
            "bootstrap_mean_ci_low": float("nan"),
            "bootstrap_mean_ci_high": float("nan"),
            "bootstrap_slope_ci_low": float("nan"),
            "bootstrap_slope_ci_high": float("nan"),
        }
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(replicates, len(values)))
    totals = values[draws].sum(axis=1)
    n = totals[:, 0]
    sum_x, sum_y, sum_x2, sum_xy = (totals[:, index] for index in range(1, 5))
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = (sum_xy - sum_x * sum_y / n) / (sum_x2 - sum_x * sum_x / n)
        mean_y = sum_y / n
    slope = slope[np.isfinite(slope)]
    mean_y = mean_y[np.isfinite(mean_y)]
    if not len(slope) or not len(mean_y):
        return {
            "bootstrap_mean_ci_low": float("nan"),
            "bootstrap_mean_ci_high": float("nan"),
            "bootstrap_slope_ci_low": float("nan"),
            "bootstrap_slope_ci_high": float("nan"),
        }
    return {
        "bootstrap_mean_ci_low": float(np.quantile(mean_y, 0.025)),
        "bootstrap_mean_ci_high": float(np.quantile(mean_y, 0.975)),
        "bootstrap_slope_ci_low": float(np.quantile(slope, 0.025)),
        "bootstrap_slope_ci_high": float(np.quantile(slope, 0.975)),
    }


def _summarize_one_pair_group(
    data: pd.DataFrame,
    *,
    exposure: str,
    outcome: str,
    subset: str,
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, object]:
    x = np.log1p(pd.to_numeric(data["erosion_high"], errors="coerce").to_numpy(dtype=float)) - np.log1p(
        pd.to_numeric(data["erosion_low"], errors="coerce").to_numpy(dtype=float)
    )
    high_col, low_col = f"high_{outcome}", f"low_{outcome}"
    y = pd.to_numeric(data[high_col], errors="coerce").to_numpy(dtype=float) - pd.to_numeric(
        data[low_col], errors="coerce"
    ).to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    data = data.iloc[np.flatnonzero(valid)]
    x, y = x[valid], y[valid]
    n = int(len(x))
    blocks = (
        data["spatial_block_id"].fillna("missing").astype(str).to_numpy()
        if "spatial_block_id" in data
        else data["pair_id"].astype(str).to_numpy()
        if "pair_id" in data
        else np.arange(n).astype(str)
    )
    n_blocks = int(pd.Series(blocks).nunique())
    if n >= 3 and np.ptp(x) > 0 and np.ptp(y) > 0:
        rho, rho_p = spearmanr(x, y)
        tau, tau_p = kendalltau(x, y, variant="b")
        slope = float(np.cov(x, y, ddof=0)[0, 1] / np.var(x))
        pearson_r = float(np.corrcoef(x, y)[0, 1])
    else:
        rho = rho_p = tau = tau_p = slope = pearson_r = float("nan")
    if n and np.ptp(y) > 0:
        try:
            w_stat, w_p = wilcoxon(y, zero_method="pratt", alternative="two-sided", method="auto")
            w_stat, w_p = float(w_stat), float(w_p)
        except ValueError:
            w_stat = w_p = float("nan")
    else:
        w_stat = w_p = float("nan")
    rho_low = rho_high = float("nan")
    if np.isfinite(rho) and n_blocks > 3 and abs(float(rho)) < 1:
        z = np.arctanh(float(rho))
        half_width = norm.ppf(0.975) / np.sqrt(n_blocks - 3)
        rho_low, rho_high = float(np.tanh(z - half_width)), float(np.tanh(z + half_width))
    bootstrap = _block_bootstrap_intervals(
        x,
        y,
        blocks,
        replicates=bootstrap_replicates,
        seed=seed,
    )
    x_sd = float(np.std(x, ddof=0)) if n else float("nan")
    y_sd = float(np.std(y, ddof=0)) if n else float("nan")
    effect_scale = x_sd / y_sd if y_sd > 0 else float("nan")
    return {
        "exposure": exposure,
        "outcome": outcome,
        "subset": subset,
        "n_pairs": n,
        "spatial_blocks": n_blocks,
        "mean_delta_log1p_erosion": float(np.mean(x)) if n else float("nan"),
        "median_delta_log1p_erosion": float(np.median(x)) if n else float("nan"),
        "mean_delta_y": float(np.mean(y)) if n else float("nan"),
        "median_delta_y": float(np.median(y)) if n else float("nan"),
        "negative_delta_y_fraction": float(np.mean(y < 0)) if n else float("nan"),
        "zero_delta_y_fraction": float(np.mean(y == 0)) if n else float("nan"),
        "positive_delta_y_fraction": float(np.mean(y > 0)) if n else float("nan"),
        "spearman_rho": float(rho),
        "spearman_p": float(rho_p),
        "spearman_ci_low_block_adjusted": rho_low,
        "spearman_ci_high_block_adjusted": rho_high,
        "kendall_tau": float(tau),
        "kendall_p": float(tau_p),
        "wilcoxon_statistic": w_stat,
        "wilcoxon_p": w_p,
        "slope_delta_y_per_delta_log1p_erosion": slope,
        "standardized_slope": slope * x_sd / y_sd if y_sd > 0 and np.isfinite(slope) else float("nan"),
        "standardized_slope_ci_low": bootstrap["bootstrap_slope_ci_low"] * effect_scale if np.isfinite(bootstrap["bootstrap_slope_ci_low"]) and np.isfinite(effect_scale) else float("nan"),
        "standardized_slope_ci_high": bootstrap["bootstrap_slope_ci_high"] * effect_scale if np.isfinite(bootstrap["bootstrap_slope_ci_high"]) and np.isfinite(effect_scale) else float("nan"),
        "pearson_r_squared": pearson_r * pearson_r if np.isfinite(pearson_r) else float("nan"),
        **bootstrap,
        "confidence_method": "5km block bootstrap for mean/slope; Fisher-z correlation interval using block count",
        "outcome_guided_selection": True,
    }


def summarize_pair_associations(
    pairs: pd.DataFrame,
    outcomes: Sequence[str],
    *,
    bootstrap_replicates: int = 300,
    seed: int = 20261027,
) -> pd.DataFrame:
    """Summarize matched-pair erosion and health differences, including zero strata."""

    required = {"exposure", "method", "erosion_high", "erosion_low"}
    missing = sorted(required - set(pairs.columns))
    if missing:
        raise ValueError(f"Pair table is missing required columns: {missing}")
    records: list[dict[str, object]] = []
    for (method, exposure), pool in pairs.groupby(["method", "exposure"], sort=True, dropna=False):
        if str(exposure) not in {"Aa_20yr", "Aa_2020"}:
            continue
        low = pd.to_numeric(pool["erosion_low"], errors="coerce")
        high = pd.to_numeric(pool["erosion_high"], errors="coerce")
        delta_e = np.log1p(high.to_numpy(dtype=float)) - np.log1p(low.to_numpy(dtype=float))
        valid_e = np.isfinite(delta_e) & (delta_e > 0)
        gap_cut = float(np.quantile(delta_e[valid_e], 0.75)) if valid_e.any() else float("nan")
        subset_masks = {
            "all": np.ones(len(pool), dtype=bool),
            "low_zero": low.le(0).to_numpy() & high.gt(0).to_numpy(),
            "both_positive": low.gt(0).to_numpy() & high.gt(0).to_numpy(),
            "high_erosion_gap_q4": delta_e >= gap_cut if np.isfinite(gap_cut) else np.zeros(len(pool), dtype=bool),
        }
        for outcome_index, outcome in enumerate(outcomes):
            if f"high_{outcome}" not in pool or f"low_{outcome}" not in pool:
                continue
            for subset_index, (subset_name, mask) in enumerate(subset_masks.items()):
                selected = pool.loc[mask].reset_index(drop=True)
                record = _summarize_one_pair_group(
                    selected,
                    exposure=str(exposure),
                    outcome=outcome,
                    subset=subset_name,
                    bootstrap_replicates=bootstrap_replicates,
                    seed=seed + outcome_index * 101 + subset_index * 17 + sum(map(ord, str(method) + str(exposure))),
                )
                record["method"] = str(method)
                record["selection_objective"] = "describe paired response to log1p actual Aa difference; report all rows and prespecified zero/gap strata"
                records.append(record)
    return pd.DataFrame.from_records(records)


def _direction_is_negative(row: pd.Series) -> bool:
    direction = str(row.get("direction", "")).strip().lower()
    if direction in {"negative", "negative_association", "decreasing"}:
        return True
    if direction in {"positive", "positive_association", "increasing"}:
        return False
    for name in ("standardized_effect", "coefficient", "slope_delta_y_per_delta_log1p_erosion"):
        value = _finite_number(row.get(name))
        if np.isfinite(value) and value != 0:
            return value < 0
    rho = _finite_number(row.get("spearman_rho"))
    return bool(np.isfinite(rho) and rho < 0)


def _strongest_negative_score(frame: pd.DataFrame) -> tuple[pd.Series, list[str]]:
    candidates = frame.loc[frame.apply(_direction_is_negative, axis=1)].copy()
    if candidates.empty:
        return pd.Series(dtype=object), []
    measures: list[tuple[str, bool]] = []
    if "standardized_effect" in candidates and pd.to_numeric(candidates["standardized_effect"], errors="coerce").notna().any():
        candidates["__effect_magnitude"] = pd.to_numeric(candidates["standardized_effect"], errors="coerce").abs()
        measures.append(("__effect_magnitude", True))
    elif "coefficient" in candidates and pd.to_numeric(candidates["coefficient"], errors="coerce").notna().any():
        candidates["__effect_magnitude"] = pd.to_numeric(candidates["coefficient"], errors="coerce").abs()
        measures.append(("__effect_magnitude", True))
    if "spearman_rho" in candidates and pd.to_numeric(candidates["spearman_rho"], errors="coerce").notna().any():
        candidates["__rho_negative_strength"] = -pd.to_numeric(candidates["spearman_rho"], errors="coerce")
        measures.append(("__rho_negative_strength", True))
    if "negative_delta_y_fraction" in candidates and pd.to_numeric(candidates["negative_delta_y_fraction"], errors="coerce").notna().any():
        candidates["__negative_fraction"] = pd.to_numeric(candidates["negative_delta_y_fraction"], errors="coerce")
        measures.append(("__negative_fraction", True))
    if "standardized_effect_ci_high" in candidates and pd.to_numeric(candidates["standardized_effect_ci_high"], errors="coerce").notna().any():
        candidates["__ci_strength"] = -pd.to_numeric(candidates["standardized_effect_ci_high"], errors="coerce")
        measures.append(("__ci_strength", True))
    if "validation_r2" in candidates and pd.to_numeric(candidates["validation_r2"], errors="coerce").notna().any():
        candidates["__validation_performance"] = pd.to_numeric(candidates["validation_r2"], errors="coerce")
        measures.append(("__validation_performance", True))
    if "max_abs_matched_smd" in candidates and pd.to_numeric(candidates["max_abs_matched_smd"], errors="coerce").notna().any():
        candidates["__balance"] = -pd.to_numeric(candidates["max_abs_matched_smd"], errors="coerce")
        measures.append(("__balance", True))
    if "n" in candidates and pd.to_numeric(candidates["n"], errors="coerce").notna().any():
        candidates["__sample_size"] = np.log1p(pd.to_numeric(candidates["n"], errors="coerce").clip(lower=0))
        measures.append(("__sample_size", True))
    score_columns: list[str] = []
    for column, _ in measures:
        score_column = f"__rank_{column}"
        values = pd.to_numeric(candidates[column], errors="coerce")
        finite = values.notna()
        candidates[score_column] = np.nan
        if finite.any():
            candidates.loc[finite, score_column] = values.loc[finite].rank(pct=True, method="average")
            score_columns.append(score_column)
    if not score_columns:
        return candidates.iloc[0], []
    candidates["negative_strength_score"] = candidates[score_columns].mean(axis=1, skipna=True)
    candidates["negative_strength_components"] = ",".join(column.replace("__rank_", "") for column in score_columns)
    sort_columns = ["negative_strength_score"]
    ascending = [False]
    if "r_squared" in candidates:
        sort_columns.append("r_squared")
        ascending.append(False)
    winner = candidates.sort_values(sort_columns, ascending=ascending, kind="stable").iloc[0]
    return winner, score_columns


def rank_phase_g_models(candidates: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return separate apparent, negative-association, and validation-selected boards."""

    required = {"outcome", "exposure", "evaluation_set"}
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(f"Model candidates are missing columns: {missing}")
    data = candidates.copy()
    if "status" in data:
        data = data.loc[data["status"].fillna("ok").astype(str).str.lower().isin({"ok", "success"})].copy()
    apparent = data.loc[data["evaluation_set"].astype(str).str.lower().isin({"apparent", "full_sample", "full_sample_descriptive_apparent"})].copy()
    group_columns = ["exposure", "outcome"]
    highest_rows: list[pd.Series] = []
    negative_r2_rows: list[pd.Series] = []
    strongest_rows: list[pd.Series] = []
    for _, group in apparent.groupby(group_columns, sort=True, dropna=False):
        score = pd.to_numeric(group.get("r_squared", group.get("apparent_r2", np.nan)), errors="coerce")
        valid = group.loc[np.isfinite(score)].copy()
        if valid.empty:
            continue
        score = pd.to_numeric(valid.get("r_squared", valid.get("apparent_r2", np.nan)), errors="coerce")
        highest_rows.append(valid.loc[score.idxmax()])
        negative = valid.loc[valid.apply(_direction_is_negative, axis=1)]
        if not negative.empty:
            neg_score = pd.to_numeric(negative.get("r_squared", negative.get("apparent_r2", np.nan)), errors="coerce")
            negative_r2_rows.append(negative.loc[neg_score.idxmax()])
            strongest, _ = _strongest_negative_score(negative)
            if not strongest.empty:
                strongest_rows.append(strongest)

    spatial = data.loc[data["evaluation_set"].astype(str).str.lower().eq("test")].copy()
    if "selection_mode" in spatial:
        spatial = spatial.loc[spatial["selection_mode"].astype(str).str.lower().str.contains("validation")].copy()
    if "validation_r2" not in spatial:
        spatial["validation_r2"] = np.nan
    spatial_rows: list[pd.Series] = []
    for _, group in spatial.groupby(group_columns, sort=True, dropna=False):
        validation = pd.to_numeric(group["validation_r2"], errors="coerce")
        valid = group.loc[np.isfinite(validation)]
        if not valid.empty:
            spatial_rows.append(valid.loc[pd.to_numeric(valid["validation_r2"], errors="coerce").idxmax()])
    return {
        "highest_apparent_r2": pd.DataFrame(highest_rows).reset_index(drop=True),
        "highest_negative_apparent_r2": pd.DataFrame(negative_r2_rows).reset_index(drop=True),
        "strongest_negative_association": pd.DataFrame(strongest_rows).reset_index(drop=True),
        "validation_selected_spatial": pd.DataFrame(spatial_rows).reset_index(drop=True),
    }


def fit_hierarchical_group_labels(
    frame: pd.DataFrame,
    variables: Sequence[str],
    *,
    soil_column: str = "SoilType",
    n_clusters: int = 6,
    max_fit_rows: int = 10000,
    seed: int = 20261027,
) -> pd.Series:
    """Fit soil-stratified Ward groups and assign other rows to nearest centroids."""

    if n_clusters < 2 or max_fit_rows < n_clusters:
        raise ValueError("n_clusters must be >=2 and max_fit_rows must cover all clusters")
    missing = [name for name in [soil_column, *variables] if name not in frame]
    if missing:
        raise ValueError(f"Missing columns required for hierarchical groups: {missing}")
    labels = pd.Series(pd.NA, index=frame.index, dtype="string", name="hierarchical_group")
    matrix = frame[list(variables)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    soils = pd.to_numeric(frame[soil_column], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(matrix).all(axis=1) & np.isfinite(soils)
    rng = np.random.default_rng(seed)
    for soil in sorted(np.unique(soils[valid]).astype(int)):
        positions = np.flatnonzero(valid & (soils == soil))
        if len(positions) < n_clusters:
            continue
        scaler = StandardScaler().fit(matrix[positions])
        standardized = scaler.transform(matrix[positions])
        if len(positions) > max_fit_rows:
            fit_indices = np.sort(rng.choice(len(positions), size=max_fit_rows, replace=False))
        else:
            fit_indices = np.arange(len(positions))
        clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
        fit_labels = clusterer.fit_predict(standardized[fit_indices])
        centers = np.vstack(
            [standardized[fit_indices][fit_labels == group].mean(axis=0) for group in range(n_clusters)]
        )
        assigned = pairwise_distances_argmin(standardized, centers, metric="euclidean")
        assigned[fit_indices] = fit_labels
        for group in range(n_clusters):
            labels.iloc[positions[assigned == group]] = f"{soil}:{group}"
    return labels
