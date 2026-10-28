"""Targeted Phase G completion analyses for matched-pair models and interactions."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import chi2, norm, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from experiments.oracle_exploration.nonlinear_search import fit_model_family

PAIR_MODEL_FAMILIES = (
    "ols",
    "huber",
    "quadratic",
    "natural_spline",
    "statsmodels_gam",
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
)


def _probability_from_log(log_probability: float) -> float:
    """Convert a log probability; the parallel log field remains authoritative."""
    value = float(log_probability)
    if np.isnan(value):
        return float("nan")
    if value == float("-inf"):
        return float(np.nextafter(0.0, 1.0))
    return max(float(np.exp(value)), float(np.nextafter(0.0, 1.0)))


def _chi2_logsf(statistic: float, degrees: int) -> float:
    """Compute log chi-square survival, including tails beyond floating range."""
    value = float(statistic)
    df = max(int(degrees), 1)
    if value <= 0:
        return 0.0
    if not np.isfinite(value):
        return float("-inf")
    direct = float(chi2.logsf(value, df))
    shape = df / 2.0
    x = value / 2.0
    deep_tail = direct < -300.0 and x > max(50.0, 5.0 * shape)
    if np.isfinite(direct) and not deep_tail:
        return direct

    # chi-square survival is Q(df/2, statistic/2). Use the upper incomplete
    # gamma asymptotic series when the direct log tail underflows or loses
    # precision in the extreme tail.
    if x <= shape:
        return direct
    term = 1.0
    series = 1.0
    for order in range(1, 1000):
        next_term = term * (shape - order) / x
        if abs(next_term) > abs(term):
            break
        term = next_term
        series += term
        if term == 0.0 or abs(term) <= np.finfo(float).eps * abs(series):
            break
    if not np.isfinite(series) or series <= 0.0:
        return direct
    return float((shape - 1.0) * np.log(x) - x - gammaln(shape) + np.log(series))


def mark_pixel_candidate_scope(
    candidates: pd.DataFrame,
    *,
    primary_soils: Sequence[int] = (26, 27, 42),
) -> pd.DataFrame:
    """Flag outcome-selected pixel candidates outside the cropland soil scope."""

    if "candidate_id" not in candidates:
        raise ValueError("Pixel candidate rows require candidate_id")
    allowed = {int(value) for value in primary_soils}
    result = candidates.copy()
    codes = result["candidate_id"].astype(str).str.extract(r"(?:^|_)soil_(\d+)(?:_|$)", expand=False)
    soil_codes = pd.to_numeric(codes, errors="coerce")
    explicit_soil = soil_codes.notna()
    in_scope = ~explicit_soil | soil_codes.isin(allowed)
    result["candidate_soil_code"] = soil_codes.astype("Int64")
    result["in_primary_cropland_scope"] = in_scope.astype(bool)
    result["scope_status"] = np.where(
        ~explicit_soil,
        "eligible_primary_cropland_candidate",
        np.where(
            in_scope,
            "eligible_primary_soil_subgroup",
            np.where(soil_codes.eq(85), "excluded_soil_85_water", "excluded_non_primary_soil"),
        ),
    )
    return result


def _paired_arrays(data: pd.DataFrame, outcome: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    required = {"erosion_high", "erosion_low", f"high_{outcome}", f"low_{outcome}"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Pair data is missing columns for {outcome}: {missing}")
    high_erosion = pd.to_numeric(data["erosion_high"], errors="coerce").to_numpy(dtype=float)
    low_erosion = pd.to_numeric(data["erosion_low"], errors="coerce").to_numpy(dtype=float)
    x = np.log1p(high_erosion) - np.log1p(low_erosion)
    high_y = pd.to_numeric(data[f"high_{outcome}"], errors="coerce").to_numpy(dtype=float)
    low_y = pd.to_numeric(data[f"low_{outcome}"], errors="coerce").to_numpy(dtype=float)
    y = high_y - low_y
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0)
    return x[valid], y[valid], valid


def _direction_from_curve(x: np.ndarray, predicted: np.ndarray) -> tuple[str, float]:
    if len(x) < 2 or np.ptp(x) <= 0 or not np.isfinite(predicted).all():
        return "unknown", float("nan")
    order = np.argsort(x, kind="stable")
    ordered_x = x[order]
    ordered_y = predicted[order]
    low_x, high_x = np.quantile(ordered_x, [0.02, 0.98])
    low_prediction = float(np.interp(low_x, ordered_x, ordered_y))
    high_prediction = float(np.interp(high_x, ordered_x, ordered_y))
    change = high_prediction - low_prediction
    tolerance = max(1.0, abs(low_prediction), abs(high_prediction)) * 1e-10
    direction = "negative" if change < -tolerance else "positive" if change > tolerance else "zero"
    slope = change / (float(high_x) - float(low_x)) if high_x > low_x else float("nan")
    return direction, slope


def _model_metrics(y: np.ndarray, predicted: np.ndarray, x: np.ndarray) -> dict[str, float | str]:
    if len(y) < 2 or len(y) != len(predicted):
        return {"r_squared": float("nan"), "rmse": float("nan"), "mae": float("nan"), "spearman_rho": float("nan")}
    rho = float(spearmanr(x, y).statistic) if np.ptp(x) > 0 and np.ptp(y) > 0 else float("nan")
    return {
        "r_squared": float(r2_score(y, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(y, predicted))),
        "mae": float(mean_absolute_error(y, predicted)),
        "spearman_rho": rho,
    }


def search_paired_response_models(
    pairs: pd.DataFrame,
    *,
    outcomes: Sequence[str],
    model_families: Sequence[str] = PAIR_MODEL_FAMILIES,
    seed: int = 20261027,
) -> pd.DataFrame:
    """Fit each requested model family to full matched-pair difference samples."""

    required = {"method", "exposure", "erosion_high", "erosion_low"}
    missing = sorted(required - set(pairs.columns))
    if missing:
        raise ValueError(f"Pair table is missing columns required for model search: {missing}")
    records: list[dict[str, Any]] = []
    for (method, exposure), pool in pairs.groupby(["method", "exposure"], sort=True, dropna=False):
        for outcome_index, outcome in enumerate(outcomes):
            x, y, valid = _paired_arrays(pool, outcome)
            n = len(x)
            base = {
                "analysis_unit": "matched_pair_difference",
                "method": str(method),
                "exposure": str(exposure),
                "outcome": str(outcome),
                "candidate_id": f"phase_f_{method}_{exposure}",
                "fit_scope": "full_matched_pair_sample",
                "evaluation_set": "apparent",
                "n_pairs": int(n),
                "n": int(n),
                "sample_fraction": float(n / len(pool)) if len(pool) else float("nan"),
                "candidate_count": int(len(model_families)),
                "random_seed": int(seed + outcome_index),
                "outcome_guided_selection": True,
                "selection_mode": "outcome_guided_oracle_model_search",
                "selection_objective": "rank apparent matched-pair Δhealth R²; report all model families and directions",
                "r2_scope": "in-sample R² for matched-pair Δhealth predicted from Δlog1p(Aa)",
            }
            if n < 8 or np.ptp(x) <= 0 or np.ptp(y) <= 0:
                for family in model_families:
                    records.append({
                        **base,
                        "model_family": str(family),
                        "model_name": str(family),
                        "status": "insufficient_or_constant_sample",
                        "error": "At least eight complete pairs with variation in both ΔAa and Δhealth are required.",
                        "direction": "unknown",
                    })
                continue
            train = pd.DataFrame({"delta_e": x, "delta_y": y})
            low, high = np.quantile(x, [0.02, 0.98])
            grid_x = np.linspace(float(low), float(high), 100)
            grid = pd.DataFrame({"delta_e": grid_x, "delta_y": float(np.median(y))})
            for family_index, family in enumerate(model_families):
                record: dict[str, Any] = {
                    **base,
                    "model_family": str(family),
                    "model_name": str(family),
                    "random_seed": int(seed + outcome_index * 101 + family_index),
                }
                try:
                    predicted, parameters = fit_model_family(
                        train,
                        train,
                        "delta_y",
                        "delta_e",
                        ["delta_e"],
                        model_family=str(family),
                        random_state=int(seed + outcome_index * 101 + family_index),
                    )
                    grid_prediction, _ = fit_model_family(
                        train,
                        grid,
                        "delta_y",
                        "delta_e",
                        ["delta_e"],
                        model_family=str(family),
                        random_state=int(seed + outcome_index * 101 + family_index),
                    )
                    direction, average_slope = _direction_from_curve(grid_x, grid_prediction)
                    metrics = _model_metrics(y, predicted, x)
                    x_sd = float(np.std(x, ddof=0))
                    y_sd = float(np.std(y, ddof=0))
                    record.update(
                        status="ok",
                        error=None,
                        **metrics,
                        direction=direction,
                        average_curve_slope=average_slope,
                        standardized_effect=average_slope * x_sd / y_sd if y_sd > 0 else float("nan"),
                        curve_low_exposure=float(low),
                        curve_high_exposure=float(high),
                        curve_change=float(grid_prediction[-1] - grid_prediction[0]),
                        model_parameters=parameters,
                    )
                except Exception as exc:  # preserve failed model families in the Oracle ledger
                    record.update(
                        status="failed",
                        error=f"{type(exc).__name__}: {exc}",
                        direction="unknown",
                        r_squared=float("nan"),
                        rmse=float("nan"),
                        mae=float("nan"),
                        spearman_rho=float("nan"),
                    )
                records.append(record)
    return pd.DataFrame.from_records(records)


def _cluster_robust_covariance(
    design: np.ndarray,
    residual: np.ndarray,
    blocks: np.ndarray,
) -> tuple[np.ndarray, int]:
    xtx_inverse = np.linalg.pinv(design.T @ design)
    block_codes, unique_blocks = pd.factorize(blocks.astype(str), sort=True)
    scores = np.zeros((len(unique_blocks), design.shape[1]), dtype=float)
    for block_code in range(len(unique_blocks)):
        mask = block_codes == block_code
        scores[block_code] = design[mask].T @ residual[mask]
    meat = scores.T @ scores
    n, p = design.shape
    g = len(unique_blocks)
    correction = (g / (g - 1)) * ((n - 1) / (n - p)) if g > 1 and n > p else 1.0
    covariance = xtx_inverse @ meat @ xtx_inverse * correction
    return covariance, g


def compare_group_slopes(
    pairs: pd.DataFrame,
    *,
    outcomes: Sequence[str],
    group_columns: Sequence[str],
    min_pairs: int = 100,
    min_spatial_blocks: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test group-specific ΔAa slopes using a block-clustered interaction Wald test."""

    required = {"method", "exposure", "erosion_high", "erosion_low"}
    missing = sorted(required - set(pairs.columns))
    if missing:
        raise ValueError(f"Pair table is missing columns required for interaction tests: {missing}")
    tests: list[dict[str, Any]] = []
    slopes: list[dict[str, Any]] = []
    for (method, exposure), pool in pairs.groupby(["method", "exposure"], sort=True, dropna=False):
        for group_column in group_columns:
            if group_column not in pool:
                continue
            for outcome in outcomes:
                base = {
                    "analysis_unit": "matched_pair_difference",
                    "method": str(method),
                    "exposure": str(exposure),
                    "outcome": str(outcome),
                    "group_column": str(group_column),
                    "group_method": str(group_column).removesuffix("_group"),
                    "outcome_guided_selection": True,
                    "selection_mode": "EXPLORATORY_ORACLE_group_heterogeneity_test",
                    "selection_objective": "test whether Δlog1p(Aa) slopes differ among environment groups",
                    "inference": "joint interaction Wald test with spatial-block cluster-robust covariance",
                }
                x, y, valid = _paired_arrays(pool, outcome)
                labels = pool.loc[valid, group_column].astype("string").to_numpy()
                valid_labels = pd.notna(labels)
                x, y, labels = x[valid_labels], y[valid_labels], labels[valid_labels]
                if "spatial_block_id" in pool:
                    blocks = pool.loc[valid, "spatial_block_id"].to_numpy()[valid_labels]
                elif "pair_id" in pool:
                    blocks = pool.loc[valid, "pair_id"].to_numpy()[valid_labels]
                else:
                    blocks = np.arange(len(x)).astype(str)
                counts = pd.Series(labels).value_counts()
                retained_groups = sorted(counts[counts.ge(min_pairs)].index.astype(str).tolist())
                keep = np.isin(labels.astype(str), retained_groups)
                x, y, labels, blocks = x[keep], y[keep], labels[keep].astype(str), blocks[keep]
                group_values = sorted(pd.unique(labels).tolist())
                n_blocks = int(pd.Series(blocks).nunique())
                if len(group_values) < 2 or n_blocks < min_spatial_blocks or len(x) <= 2 * len(group_values):
                    tests.append({
                        **base,
                        "status": "insufficient_groups_or_spatial_blocks",
                        "n_pairs": int(len(x)),
                        "n_groups": int(len(group_values)),
                        "spatial_blocks": n_blocks,
                        "interaction_df": max(0, len(group_values) - 1),
                        "interaction_wald_chi2": float("nan"),
                        "interaction_p_value": float("nan"),
                    })
                    continue
                x_centered = x - float(np.mean(x))
                dummies = np.column_stack([labels == group for group in group_values[1:]]).astype(float)
                interaction_terms = dummies * x_centered[:, None]
                design = np.column_stack(
                    [np.ones(len(x)), x_centered, dummies, interaction_terms]
                )
                beta = np.linalg.pinv(design.T @ design) @ (design.T @ y)
                residual = y - design @ beta
                covariance, n_blocks = _cluster_robust_covariance(design, residual, blocks)
                interaction_start = 2 + dummies.shape[1]
                interaction_beta = beta[interaction_start:]
                interaction_covariance = covariance[interaction_start:, interaction_start:]
                interaction_rank = int(np.linalg.matrix_rank(interaction_covariance))
                statistic = float(interaction_beta @ np.linalg.pinv(interaction_covariance) @ interaction_beta)
                degrees = max(interaction_rank, 1)
                log_p_value = _chi2_logsf(max(statistic, 0.0), degrees)
                p_value = _probability_from_log(log_p_value)
                test_record = {
                    **base,
                    "status": "ok",
                    "n_pairs": int(len(x)),
                    "n_groups": int(len(group_values)),
                    "spatial_blocks": n_blocks,
                    "interaction_df": degrees,
                    "interaction_wald_chi2": statistic,
                    "interaction_p_value": p_value,
                    "interaction_log_p_value": log_p_value,
                    "reference_group": group_values[0],
                    "groups_tested": "|".join(group_values),
                }
                tests.append(test_record)
                x_sd = float(np.std(x, ddof=0))
                y_sd = float(np.std(y, ddof=0))
                for group_index, group in enumerate(group_values):
                    if group_index == 0:
                        coefficient = float(beta[1])
                        variance = float(covariance[1, 1])
                    else:
                        interaction_index = interaction_start + group_index - 1
                        coefficient = float(beta[1] + beta[interaction_index])
                        variance = float(
                            covariance[1, 1]
                            + covariance[interaction_index, interaction_index]
                            + 2 * covariance[1, interaction_index]
                        )
                    standard_error = float(np.sqrt(max(variance, 0.0)))
                    slope_p = float(2 * norm.sf(abs(coefficient / standard_error))) if standard_error > 0 else float("nan")
                    slopes.append({
                        **base,
                        "status": "ok",
                        "group_label": group,
                        "reference_group": group_values[0],
                        "n_pairs": int(np.count_nonzero(labels == group)),
                        "spatial_blocks": int(pd.Series(blocks[labels == group]).nunique()),
                        "slope_delta_y_per_delta_log1p_erosion": coefficient,
                        "cluster_robust_standard_error": standard_error,
                        "slope_ci_low": coefficient - 1.96 * standard_error,
                        "slope_ci_high": coefficient + 1.96 * standard_error,
                        "slope_p_value": slope_p,
                        "standardized_effect": coefficient * x_sd / y_sd if y_sd > 0 else float("nan"),
                        "interaction_p_value": p_value,
                        "interaction_log_p_value": log_p_value,
                    })
    test_frame = pd.DataFrame.from_records(tests)
    slope_frame = pd.DataFrame.from_records(slopes)
    if not test_frame.empty:
        ok = test_frame["status"].eq("ok") & pd.to_numeric(test_frame["interaction_p_value"], errors="coerce").notna()
        test_frame["interaction_q_value_global"] = np.nan
        log_p_values = test_frame.loc[ok, "interaction_log_p_value"].to_numpy(dtype=float)
        if len(log_p_values):
            order = np.argsort(log_p_values)
            adjusted_log = np.empty(len(log_p_values), dtype=float)
            running_log = 0.0
            family_size_log = float(np.log(len(log_p_values)))
            for rank in range(len(log_p_values) - 1, -1, -1):
                index = order[rank]
                candidate_log = log_p_values[index] + family_size_log - float(np.log(rank + 1))
                running_log = min(running_log, candidate_log)
                adjusted_log[index] = min(running_log, 0.0)
            test_frame.loc[ok, "interaction_log_q_value_global"] = adjusted_log
            test_frame.loc[ok, "interaction_q_value_global"] = [
                _probability_from_log(value) for value in adjusted_log
            ]
    return test_frame, slope_frame


def _spatial_roles(blocks: Sequence[object], seed: int) -> dict[str, str]:
    unique = np.asarray(sorted(pd.unique(pd.Series(blocks).dropna().astype(str))))
    if len(unique) < 3:
        return {}
    shuffled = unique[np.random.default_rng(seed).permutation(len(unique))]
    n_train = max(1, int(round(0.60 * len(shuffled))))
    n_validation = max(1, int(round(0.20 * len(shuffled))))
    if n_train + n_validation >= len(shuffled):
        n_train = len(shuffled) - 2
        n_validation = 1
    roles = {block: "train" for block in shuffled[:n_train]}
    roles.update({block: "validation" for block in shuffled[n_train : n_train + n_validation]})
    roles.update({block: "test" for block in shuffled[n_train + n_validation :]})
    return roles


def validate_pair_models_spatial(
    pairs: pd.DataFrame,
    *,
    outcomes: Sequence[str],
    model_families: Sequence[str] = PAIR_MODEL_FAMILIES,
    seeds: Sequence[int] = tuple(range(10)),
    block_size_m: float = 5000.0,
    min_train_pairs: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select pair models by spatial validation score, then score selected fits on test blocks."""

    required = {"method", "exposure", "erosion_high", "erosion_low"}
    missing = sorted(required - set(pairs.columns))
    if missing:
        raise ValueError(f"Pair table is missing columns required for spatial model validation: {missing}")
    records: list[dict[str, Any]] = []
    selected_records: list[dict[str, Any]] = []
    for (method, exposure), pool in pairs.groupby(["method", "exposure"], sort=True, dropna=False):
        for outcome_index, outcome in enumerate(outcomes):
            x, y, valid = _paired_arrays(pool, outcome)
            selected_pool = pool.loc[valid].reset_index(drop=True)
            if "spatial_block_id" in selected_pool:
                blocks = selected_pool["spatial_block_id"].fillna("missing").astype(str).to_numpy()
            elif {"x_mid", "y_mid"}.issubset(selected_pool.columns):
                blocks = (
                    np.floor(pd.to_numeric(selected_pool["x_mid"], errors="coerce") / block_size_m).astype(int).astype(str)
                    + "_"
                    + np.floor(pd.to_numeric(selected_pool["y_mid"], errors="coerce") / block_size_m).astype(int).astype(str)
                ).to_numpy()
            else:
                raise ValueError("Spatial pair validation requires spatial_block_id or x_mid/y_mid")
            for seed in seeds:
                roles = _spatial_roles(blocks, int(seed))
                role_series = pd.Series(blocks).map(roles).fillna("").to_numpy()
                train_mask, validation_mask, test_mask = (
                    role_series == "train",
                    role_series == "validation",
                    role_series == "test",
                )
                if int(train_mask.sum()) < min_train_pairs or not validation_mask.any() or not test_mask.any():
                    continue
                train = pd.DataFrame({"delta_e": x[train_mask], "delta_y": y[train_mask]})
                validation = pd.DataFrame({"delta_e": x[validation_mask], "delta_y": y[validation_mask]})
                test = pd.DataFrame({"delta_e": x[test_mask], "delta_y": y[test_mask]})
                seed_trials: list[dict[str, Any]] = []
                for family_index, family in enumerate(model_families):
                    trial: dict[str, Any] = {
                        "analysis_unit": "matched_pair_difference",
                        "method": str(method),
                        "exposure": str(exposure),
                        "outcome": str(outcome),
                        "model_family": str(family),
                        "model_name": str(family),
                        "evaluation_set": "validation",
                        "fit_scope": "train_spatial_blocks_only",
                        "spatial_seed": int(seed),
                        "block_size_m": float(block_size_m),
                        "n_train_pairs": int(train_mask.sum()),
                        "n_validation_pairs": int(validation_mask.sum()),
                        "n_test_pairs": int(test_mask.sum()),
                        "n_train_blocks": int(pd.Series(blocks[train_mask]).nunique()),
                        "n_validation_blocks": int(pd.Series(blocks[validation_mask]).nunique()),
                        "n_test_blocks": int(pd.Series(blocks[test_mask]).nunique()),
                        "outcome_guided_selection": True,
                        "selection_mode": "spatial_validation_selects_model; test metrics opened after selection",
                        "selection_objective": "maximize validation R²; separately maximize validation R² among negative validation exposure-response directions",
                        "test_r_squared": float("nan"),
                    }
                    try:
                        predicted, parameters = fit_model_family(
                            train,
                            validation,
                            "delta_y",
                            "delta_e",
                            ["delta_e"],
                            model_family=str(family),
                            random_state=int(seed + outcome_index * 101 + family_index),
                        )
                        metrics = _model_metrics(
                            validation["delta_y"].to_numpy(dtype=float),
                            predicted,
                            validation["delta_e"].to_numpy(dtype=float),
                        )
                        curve_rho = float(spearmanr(validation["delta_e"], predicted).statistic) if np.ptp(predicted) > 0 else float("nan")
                        direction = "negative" if curve_rho < -1e-12 else "positive" if curve_rho > 1e-12 else "unknown"
                        trial.update(
                            status="ok",
                            validation_r_squared=metrics["r_squared"],
                            validation_rmse=metrics["rmse"],
                            validation_mae=metrics["mae"],
                            validation_spearman_x_y=metrics["spearman_rho"],
                            validation_curve_direction=direction,
                            validation_curve_spearman=curve_rho,
                            model_parameters=parameters,
                            error=None,
                        )
                    except Exception as exc:  # retain failed validation fits
                        trial.update(
                            status="failed",
                            validation_r_squared=float("nan"),
                            validation_rmse=float("nan"),
                            validation_mae=float("nan"),
                            validation_spearman_x_y=float("nan"),
                            validation_curve_direction="unknown",
                            validation_curve_spearman=float("nan"),
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    records.append(trial)
                    seed_trials.append(trial)
                valid_trials = [
                    row for row in seed_trials
                    if row["status"] == "ok" and np.isfinite(float(row["validation_r_squared"]))
                ]
                if not valid_trials:
                    continue
                choices = [
                    ("highest_validation_r2", valid_trials),
                    (
                        "highest_negative_validation_r2",
                        [row for row in valid_trials if row["validation_curve_direction"] == "negative"],
                    ),
                ]
                for selection_type, candidates in choices:
                    if not candidates:
                        selected_records.append({
                            "analysis_unit": "matched_pair_difference",
                            "method": str(method),
                            "exposure": str(exposure),
                            "outcome": str(outcome),
                            "spatial_seed": int(seed),
                            "selection_type": selection_type,
                            "status": "no_negative_validation_candidate" if "negative" in selection_type else "no_candidate",
                            "test_r_squared": float("nan"),
                        })
                        continue
                    winner = max(candidates, key=lambda row: float(row["validation_r_squared"]))
                    selected_prediction, _ = fit_model_family(
                        train,
                        test,
                        "delta_y",
                        "delta_e",
                        ["delta_e"],
                        model_family=str(winner["model_family"]),
                        random_state=int(seed + outcome_index * 101 + model_families.index(winner["model_family"])),
                    )
                    test_metrics = _model_metrics(
                        test["delta_y"].to_numpy(dtype=float),
                        selected_prediction,
                        test["delta_e"].to_numpy(dtype=float),
                    )
                    selected_records.append({
                        "analysis_unit": "matched_pair_difference",
                        "method": str(method),
                        "exposure": str(exposure),
                        "outcome": str(outcome),
                        "spatial_seed": int(seed),
                        "selection_type": selection_type,
                        "selected_model_family": str(winner["model_family"]),
                        "validation_r_squared": float(winner["validation_r_squared"]),
                        "validation_rmse": float(winner["validation_rmse"]),
                        "validation_mae": float(winner["validation_mae"]),
                        "validation_curve_direction": str(winner["validation_curve_direction"]),
                        "test_r_squared": float(test_metrics["r_squared"]),
                        "test_rmse": float(test_metrics["rmse"]),
                        "test_mae": float(test_metrics["mae"]),
                        "test_spearman_x_y": float(test_metrics["spearman_rho"]),
                        "n_train_pairs": int(winner["n_train_pairs"]),
                        "n_validation_pairs": int(winner["n_validation_pairs"]),
                        "n_test_pairs": int(winner["n_test_pairs"]),
                        "n_train_blocks": int(winner["n_train_blocks"]),
                        "n_validation_blocks": int(winner["n_validation_blocks"]),
                        "n_test_blocks": int(winner["n_test_blocks"]),
                        "status": "ok",
                        "outcome_guided_selection": True,
                        "test_opened_after_validation_selection": True,
                        "selection_mode": str(winner["selection_mode"]),
                    })
    return pd.DataFrame.from_records(records), pd.DataFrame.from_records(selected_records)
