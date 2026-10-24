"""Group-level and matched-pair regression models."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm


_RESULT_COLUMNS = [
    "outcome",
    "environment_group",
    "method",
    "status",
    "n_cells",
    "coefficient",
    "standardized_coefficient",
    "standard_error",
    "ci_lower",
    "ci_upper",
    "p_value",
    "r_squared",
    "adjusted_r_squared",
]


def _transform_erosion(values: pd.Series, transform: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if (numeric.dropna() < 0).any():
        raise ValueError("erosion exposure must be nonnegative")
    if transform == "log1p":
        return np.log1p(numeric)
    if transform == "identity":
        return numeric
    raise ValueError("transform must be 'log1p' or 'identity'")


def _empty_result(
    *, outcome: str, group: object, method: str, status: str, n: int
) -> dict[str, object]:
    return {
        "outcome": outcome,
        "environment_group": group,
        "method": method,
        "status": status,
        "n_cells": n,
        "coefficient": np.nan,
        "standardized_coefficient": np.nan,
        "standard_error": np.nan,
        "ci_lower": np.nan,
        "ci_upper": np.nan,
        "p_value": np.nan,
        "r_squared": np.nan,
        "adjusted_r_squared": np.nan,
    }


def fit_group_models(
    frame: pd.DataFrame,
    outcomes: Sequence[str],
    erosion_col: str,
    group_col: str,
    *,
    covariates: Sequence[str] = (),
    transform: str = "log1p",
) -> pd.DataFrame:
    """Fit one erosion slope per environment group and health outcome."""
    required = [erosion_col, group_col, *outcomes, *covariates]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Missing regression columns: {', '.join(missing)}")
    if not outcomes:
        raise ValueError("At least one health outcome is required")
    rows: list[dict[str, object]] = []
    for group, group_frame in frame.groupby(group_col, sort=False, dropna=False):
        exposure = _transform_erosion(group_frame[erosion_col], transform)
        for outcome in outcomes:
            columns = pd.DataFrame(
                {"erosion": exposure, "outcome": pd.to_numeric(group_frame[outcome], errors="coerce")},
                index=group_frame.index,
            )
            for covariate in covariates:
                columns[covariate] = pd.to_numeric(group_frame[covariate], errors="coerce")
            columns = columns.replace([np.inf, -np.inf], np.nan).dropna()
            n = len(columns)
            parameters = 2 + len(covariates)
            if n <= parameters:
                rows.append(
                    _empty_result(
                        outcome=outcome,
                        group=group,
                        method="within_group",
                        status="insufficient_sample",
                        n=n,
                    )
                )
                continue
            design = sm.add_constant(columns[["erosion", *covariates]], has_constant="add")
            if np.linalg.matrix_rank(design.to_numpy(dtype=float)) < design.shape[1]:
                rows.append(
                    _empty_result(
                        outcome=outcome,
                        group=group,
                        method="within_group",
                        status="singular_design",
                        n=n,
                    )
                )
                continue
            outcome_values = columns["outcome"].to_numpy(dtype=float)
            exposure_values = columns["erosion"].to_numpy(dtype=float)
            if np.std(outcome_values, ddof=1) == 0 or np.std(exposure_values, ddof=1) == 0:
                rows.append(
                    _empty_result(
                        outcome=outcome,
                        group=group,
                        method="within_group",
                        status="no_variation",
                        n=n,
                    )
                )
                continue
            model = sm.OLS(outcome_values, design).fit()
            interval = model.conf_int().loc["erosion"]
            coefficient = float(model.params["erosion"])
            rows.append(
                {
                    "outcome": outcome,
                    "environment_group": group,
                    "method": "within_group",
                    "status": "ok",
                    "n_cells": n,
                    "coefficient": coefficient,
                    "standardized_coefficient": coefficient
                    * float(np.std(exposure_values, ddof=1))
                    / float(np.std(outcome_values, ddof=1)),
                    "standard_error": float(model.bse["erosion"]),
                    "ci_lower": float(interval.iloc[0]),
                    "ci_upper": float(interval.iloc[1]),
                    "p_value": float(model.pvalues["erosion"]),
                    "r_squared": float(model.rsquared),
                    "adjusted_r_squared": float(model.rsquared_adj),
                }
            )
    return pd.DataFrame.from_records(rows, columns=_RESULT_COLUMNS)


def fit_pair_models(
    pairs: pd.DataFrame,
    outcomes: Sequence[str],
    *,
    covariate_differences: Sequence[str] = (),
) -> pd.DataFrame:
    """Regress paired health differences on paired log1p erosion differences.

    Each outcome must be present as ``high_<outcome>`` and ``low_<outcome>``.
    Optional covariates are columns named ``delta_<covariate>``.
    """
    required = ["erosion_high", "erosion_low"]
    for outcome in outcomes:
        required.extend([f"high_{outcome}", f"low_{outcome}"])
    required.extend(f"delta_{name}" for name in covariate_differences)
    missing = [column for column in required if column not in pairs]
    if missing:
        raise ValueError(f"Matched-pair table is missing columns: {', '.join(missing)}")
    if not outcomes:
        raise ValueError("At least one health outcome is required")
    high_erosion = pd.to_numeric(pairs["erosion_high"], errors="coerce")
    low_erosion = pd.to_numeric(pairs["erosion_low"], errors="coerce")
    if ((high_erosion < 0) | (low_erosion < 0)).fillna(False).any():
        raise ValueError("erosion exposure must be nonnegative")
    delta_erosion = np.log1p(high_erosion) - np.log1p(low_erosion)
    records: list[dict[str, object]] = []
    for outcome in outcomes:
        data = pd.DataFrame(
            {
                "delta_erosion": delta_erosion,
                "delta_outcome": pd.to_numeric(pairs[f"high_{outcome}"], errors="coerce")
                - pd.to_numeric(pairs[f"low_{outcome}"], errors="coerce"),
            },
            index=pairs.index,
        )
        for covariate in covariate_differences:
            data[covariate] = pd.to_numeric(pairs[f"delta_{covariate}"], errors="coerce")
        data = data.replace([np.inf, -np.inf], np.nan).dropna()
        n = len(data)
        parameters = 2 + len(covariate_differences)
        record: dict[str, object] = {
            "outcome": outcome,
            "method": "matched_pair_difference",
            "status": "insufficient_sample",
            "n_pairs": n,
            "coefficient": np.nan,
            "standardized_coefficient": np.nan,
            "standard_error": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "p_value": np.nan,
            "r_squared": np.nan,
            "adjusted_r_squared": np.nan,
        }
        if n > parameters:
            design = sm.add_constant(data[["delta_erosion", *covariate_differences]], has_constant="add")
            if np.linalg.matrix_rank(design.to_numpy(dtype=float)) == design.shape[1]:
                x = data["delta_erosion"].to_numpy(dtype=float)
                y = data["delta_outcome"].to_numpy(dtype=float)
                if np.std(x, ddof=1) > 0 and np.std(y, ddof=1) > 0:
                    model = sm.OLS(y, design).fit()
                    interval = model.conf_int().loc["delta_erosion"]
                    coefficient = float(model.params["delta_erosion"])
                    record.update(
                        status="ok",
                        coefficient=coefficient,
                        standardized_coefficient=coefficient
                        * float(np.std(x, ddof=1))
                        / float(np.std(y, ddof=1)),
                        standard_error=float(model.bse["delta_erosion"]),
                        ci_lower=float(interval.iloc[0]),
                        ci_upper=float(interval.iloc[1]),
                        p_value=float(model.pvalues["delta_erosion"]),
                        r_squared=float(model.rsquared),
                        adjusted_r_squared=float(model.rsquared_adj),
                    )
                else:
                    record["status"] = "no_variation"
            else:
                record["status"] = "singular_design"
        records.append(record)
    return pd.DataFrame.from_records(records)


def fit_group_interaction(
    frame: pd.DataFrame,
    outcome: str,
    erosion_col: str,
    group_col: str,
    *,
    covariates: Sequence[str] = (),
    transform: str = "log1p",
) -> dict[str, object]:
    """F-test whether environment groups have different erosion slopes."""
    required = [outcome, erosion_col, group_col, *covariates]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Missing interaction-model columns: {', '.join(missing)}")
    data = pd.DataFrame(
        {
            "outcome": pd.to_numeric(frame[outcome], errors="coerce"),
            "erosion": _transform_erosion(frame[erosion_col], transform),
            "group": frame[group_col],
        },
        index=frame.index,
    )
    for covariate in covariates:
        data[covariate] = pd.to_numeric(frame[covariate], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    group_values = pd.unique(data["group"])
    result: dict[str, object] = {
        "outcome": outcome,
        "status": "insufficient_sample",
        "n_cells": int(len(data)),
        "n_groups": int(len(group_values)),
        "f_statistic": np.nan,
        "p_value": np.nan,
        "df_num": np.nan,
        "df_denom": np.nan,
        "r_squared": np.nan,
    }
    if len(group_values) < 2:
        result["status"] = "insufficient_groups"
        return result

    group_dummies = pd.get_dummies(data["group"], drop_first=True, dtype=float)
    reduced = pd.DataFrame({"intercept": 1.0, "erosion": data["erosion"]}, index=data.index)
    full = reduced.copy()
    for column in group_dummies:
        reduced[f"group_{column}"] = group_dummies[column]
        full[f"group_{column}"] = group_dummies[column]
        full[f"erosion_group_{column}"] = data["erosion"] * group_dummies[column]
    for covariate in covariates:
        reduced[covariate] = data[covariate]
        full[covariate] = data[covariate]
    if len(data) <= full.shape[1] or np.linalg.matrix_rank(full.to_numpy()) < full.shape[1]:
        return result
    reduced_model = sm.OLS(data["outcome"], reduced).fit()
    full_model = sm.OLS(data["outcome"], full).fit()
    f_statistic, p_value, df_num = full_model.compare_f_test(reduced_model)
    result.update(
        status="ok",
        f_statistic=float(f_statistic),
        p_value=float(p_value),
        df_num=float(df_num),
        df_denom=float(full_model.df_resid),
        r_squared=float(full_model.rsquared),
    )
    return result
