"""Phase H input handling and immutable Oracle run initialization."""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd
import yaml
import numpy as np
from scipy.stats import f as f_distribution
from scipy.stats import t as student_t

from experiments.oracle_exploration.data import load_oracle_data

REQUIRED_EXPOSURES = {"Aa_20yr", "Aa_2020"}
REQUIRED_OUTCOMES = {"SHI_candidate", "AN", "OM", "CEC", "pH", "SW", "BD", "MBC", "RESP"}
PRIMARY_SOILS = [26, 27, 42]
REQUIRED_DIRECTORIES = (
    "tables", "pairs", "pixels", "models", "figures/data", "figures/png", "figures/svg",
    "figures/pdf", "checkpoints", "logs",
)


@dataclass(frozen=True)
class PhaseHInputs:
    """Aligned cell-level data and Phase F pairs with Phase G labels."""

    cells: pd.DataFrame
    pairs: pd.DataFrame
    input_run_id: str
    phase_f_run_id: str
    phase_g_run_id: str
    input_paths: dict[str, Path]
    outcome_counts: dict[str, int]


def assess_interaction_estimability(
    interaction_beta: np.ndarray,
    interaction_covariance: np.ndarray,
    *,
    design_rank: int,
    design_columns: int,
    design_condition_number: float = 1.0,
    spatial_blocks: int,
    group_block_counts: dict[str, int],
    tolerance_rcond: tuple[float, ...] = (0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4),
    max_covariance_condition: float = 1e10,
    min_group_spatial_blocks: int = 5,
) -> dict[str, Any]:
    """Gate a clustered slope-interaction test before returning any p-value."""

    beta = np.asarray(interaction_beta, dtype=float).reshape(-1)
    covariance = np.asarray(interaction_covariance, dtype=float)
    q = len(beta)
    if covariance.shape != (q, q):
        raise ValueError("interaction covariance must be square and match interaction coefficients")
    if q == 0:
        raise ValueError("at least one interaction coefficient is required")
    covariance = (covariance + covariance.T) / 2
    finite_covariance = np.isfinite(covariance).all()
    if finite_covariance:
        eigenvalues = np.linalg.eigvalsh(covariance)
        max_eigenvalue = float(max(np.max(eigenvalues), 0.0))
        rank_tolerance = max_eigenvalue * 1e-10
        covariance_rank = int(np.count_nonzero(eigenvalues > rank_tolerance)) if max_eigenvalue > 0 else 0
        positive = eigenvalues[eigenvalues > rank_tolerance]
        covariance_condition = float(max_eigenvalue / positive.min()) if len(positive) else float("inf")
        min_eigenvalue = float(eigenvalues.min())
    else:
        eigenvalues = np.full(q, np.nan)
        max_eigenvalue = min_eigenvalue = covariance_condition = float("nan")
        covariance_rank = 0

    wald_by_tolerance: dict[str, float] = {}
    if finite_covariance and np.isfinite(beta).all():
        for rcond in tolerance_rcond:
            inverse = np.linalg.pinv(covariance, rcond=float(rcond))
            wald_by_tolerance[str(rcond)] = float(beta @ inverse @ beta)
    finite_wald = [max(value, 0.0) for value in wald_by_tolerance.values() if np.isfinite(value)]
    sensitivity_ratio = (
        float(max(finite_wald) / max(min(finite_wald), 1e-12)) if finite_wald else float("nan")
    )
    group_blocks = [int(value) for value in group_block_counts.values()]
    min_group_blocks = min(group_blocks) if group_blocks else 0
    block_df = max(spatial_blocks - 1, 0)
    constraint_ratio = float(q / block_df) if block_df else float("inf")

    if design_rank < design_columns:
        status = "non_estimable_design_rank"
    elif not np.isfinite(design_condition_number) or design_condition_number > 1e10:
        status = "unreliable_design_condition"
    elif not finite_covariance or covariance_rank < q:
        status = "unreliable_covariance_rank_deficient"
    elif spatial_blocks < 8 or min_group_blocks < min_group_spatial_blocks:
        status = "unreliable_group_block_coverage"
    elif block_df <= 0 or q > block_df / 2:
        status = "unreliable_constraints_per_block"
    elif not np.isfinite(covariance_condition) or covariance_condition > max_covariance_condition:
        status = "unreliable_covariance_condition"
    elif not np.isfinite(sensitivity_ratio) or sensitivity_ratio > 10:
        status = "unreliable_tolerance_sensitive"
    else:
        status = "reliable"

    statistic = p_value = float("nan")
    denominator_df = block_df
    if status == "reliable":
        statistic = float(beta @ np.linalg.solve(covariance, beta))
        p_value = float(f_distribution.sf(statistic / q, q, denominator_df))
    return {
        "status": status,
        "interaction_df": q,
        "interaction_numerator_df": q,
        "interaction_denominator_df": denominator_df,
        "interaction_wald_chi2": statistic,
        "interaction_f_statistic": statistic / q if np.isfinite(statistic) else float("nan"),
        "interaction_p_value": p_value,
        "design_rank": int(design_rank),
        "design_columns": int(design_columns),
        "design_condition_number": float(design_condition_number),
        "spatial_blocks": int(spatial_blocks),
        "min_group_spatial_blocks": int(min_group_blocks),
        "group_spatial_blocks": dict(group_block_counts),
        "constraints_per_block_df": constraint_ratio,
        "covariance_rank": covariance_rank,
        "covariance_condition_number": covariance_condition,
        "covariance_min_eigenvalue": min_eigenvalue,
        "covariance_max_eigenvalue": max_eigenvalue,
        "tolerance_sensitivity_ratio": sensitivity_ratio,
        "wald_by_rcond": wald_by_tolerance,
    }


def _block_bootstrap_slopes(
    x: np.ndarray,
    y: np.ndarray,
    labels: np.ndarray,
    blocks: np.ndarray,
    groups: list[str],
    *,
    replicates: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    if replicates < 1:
        return {}
    frame = pd.DataFrame(
        {
            "block": blocks.astype(str),
            "group": labels.astype(str),
            "n": 1.0,
            "sum_x": x,
            "sum_y": y,
            "sum_x2": x * x,
            "sum_xy": x * y,
        }
    )
    summary = frame.groupby(["block", "group"], sort=True, observed=True).sum()
    block_values = sorted(pd.unique(blocks.astype(str)).tolist())
    fields = ("n", "sum_x", "sum_y", "sum_x2", "sum_xy")
    sufficient = np.stack(
        [
            summary[field].unstack("group").reindex(index=block_values, columns=groups).fillna(0.0).to_numpy()
            for field in fields
        ],
        axis=-1,
    )
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(block_values), size=(replicates, len(block_values)))
    totals = sufficient[draws].sum(axis=1)
    n, sum_x, sum_y, sum_x2, sum_xy = (totals[..., index] for index in range(5))
    with np.errstate(divide="ignore", invalid="ignore"):
        slopes = (sum_xy - sum_x * sum_y / n) / (sum_x2 - sum_x * sum_x / n)
    result: dict[str, tuple[float, float]] = {}
    for group_index, group in enumerate(groups):
        values = slopes[:, group_index]
        values = values[np.isfinite(values)]
        if len(values):
            result[group] = (float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975)))
    return result


def fit_group_interaction(
    data: pd.DataFrame,
    *,
    x_column: str,
    y_column: str,
    group_column: str,
    block_column: str,
    bootstrap_replicates: int = 999,
    seed: int = 20261028,
    min_pairs_per_group: int = 100,
    min_group_spatial_blocks: int = 5,
    tolerance_rcond: tuple[float, ...] = (0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4),
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Fit group slopes, gate an omnibus test, and bootstrap slopes by spatial block."""

    required = {x_column, y_column, group_column, block_column}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"interaction data is missing columns: {missing}")
    selected = data[list(required)].copy()
    for name in (x_column, y_column):
        selected[name] = pd.to_numeric(selected[name], errors="coerce")
    selected = selected.dropna(subset=[x_column, y_column, group_column, block_column])
    selected = selected.loc[np.isfinite(selected[x_column]) & np.isfinite(selected[y_column])].copy()
    selected[group_column] = selected[group_column].astype(str)
    selected[block_column] = selected[block_column].astype(str)
    x = selected[x_column].to_numpy(dtype=float)
    y = selected[y_column].to_numpy(dtype=float)
    labels = selected[group_column].to_numpy(dtype=str)
    blocks = selected[block_column].to_numpy(dtype=str)
    pair_counts = pd.Series(labels).value_counts()
    retained_groups = sorted(pair_counts[pair_counts.ge(min_pairs_per_group)].index.astype(str).tolist())
    retained = np.isin(labels, retained_groups)
    x, y, labels, blocks = x[retained], y[retained], labels[retained], blocks[retained]
    selected = selected.loc[retained].copy()
    groups = sorted(pd.unique(labels).tolist())
    group_block_counts = selected.groupby(group_column, observed=True)[block_column].nunique().astype(int).to_dict()
    if len(groups) < 2 or len(x) < 4:
        return {"status": "insufficient_groups", "n_pairs": len(x), "interaction_p_value": float("nan")}, pd.DataFrame()

    group_frame = pd.DataFrame({"group": labels, "block": blocks, "x": x, "y": y})
    means = group_frame.groupby("group", sort=True, observed=True)[["x", "y"]].mean().reindex(groups)
    x_means = group_frame["group"].map(means["x"]).to_numpy(dtype=float)
    y_means = group_frame["group"].map(means["y"]).to_numpy(dtype=float)
    x_centered = x - x_means
    y_centered = y - y_means
    group_codes = pd.Categorical(labels, categories=groups, ordered=True).codes
    group_sxx = np.bincount(group_codes, weights=x_centered * x_centered, minlength=len(groups))
    group_sxy = np.bincount(group_codes, weights=x_centered * y_centered, minlength=len(groups))
    group_slopes = np.divide(
        group_sxy, group_sxx, out=np.full(len(groups), np.nan, dtype=float), where=group_sxx > 0
    )
    residual = y - y_means - group_slopes[group_codes] * x_centered

    # The old reference-coded design has 2G columns. Build its Gram matrix
    # from group sufficient statistics instead of materializing N x 2G rows.
    x_global = x - float(np.mean(x))
    group_global = pd.DataFrame({"group": labels, "x": x_global}).groupby("group", sort=True, observed=True)["x"]
    counts = group_frame.groupby("group", sort=True, observed=True).size().reindex(groups).to_numpy(dtype=float)
    sums_x = group_global.sum().reindex(groups).to_numpy(dtype=float)
    sums_x2 = group_frame.assign(x2=x_global * x_global).groupby(
        "group", sort=True, observed=True
    )["x2"].sum().reindex(groups).to_numpy(dtype=float)
    group_count = len(groups)
    p = 2 * group_count
    gram = np.zeros((p, p), dtype=float)
    interaction_start = group_count + 1
    for group_index in range(group_count):
        a0 = np.zeros(p, dtype=float)
        a1 = np.zeros(p, dtype=float)
        a0[0] = 1.0
        a1[1] = 1.0
        if group_index:
            a0[1 + group_index] = 1.0
            a1[interaction_start + group_index - 1] = 1.0
        gram += counts[group_index] * np.outer(a0, a0)
        gram += sums_x[group_index] * (np.outer(a0, a1) + np.outer(a1, a0))
        gram += sums_x2[group_index] * np.outer(a1, a1)
    design_rank = int(np.linalg.matrix_rank(gram))
    design_condition = float(np.sqrt(np.linalg.cond(gram))) if design_rank == p else float("inf")

    block_score_frame = pd.DataFrame(
        {"block": blocks, "group": labels, "slope_score": x_centered * residual}
    )
    block_slope_scores = block_score_frame.groupby(
        ["block", "group"], sort=True, observed=True
    )["slope_score"].sum().unstack("group")
    unique_blocks = sorted(pd.unique(blocks).tolist())
    block_slope_scores = block_slope_scores.reindex(index=unique_blocks, columns=groups).fillna(0.0)
    score_matrix = block_slope_scores.to_numpy(dtype=float)
    n, g = len(x), len(unique_blocks)
    correction = (g / (g - 1)) * ((n - 1) / (n - p)) if g > 1 and n > p else 1.0
    inverse_sxx = np.divide(1.0, group_sxx, out=np.full(len(groups), np.nan), where=group_sxx > 0)
    with np.errstate(invalid="ignore"):
        slope_covariance = inverse_sxx[:, None] * (score_matrix.T @ score_matrix) * inverse_sxx[None, :]
    slope_covariance *= correction
    contrast = np.zeros((group_count - 1, group_count), dtype=float)
    for index in range(1, group_count):
        contrast[index - 1, 0] = -1.0
        contrast[index - 1, index] = 1.0
    interaction_beta = contrast @ group_slopes
    interaction_covariance = contrast @ slope_covariance @ contrast.T
    result = assess_interaction_estimability(
        interaction_beta,
        interaction_covariance,
        design_rank=design_rank,
        design_columns=p,
        design_condition_number=design_condition,
        spatial_blocks=g,
        group_block_counts=group_block_counts,
        tolerance_rcond=tolerance_rcond,
        min_group_spatial_blocks=min_group_spatial_blocks,
    )
    result.update(
        n_pairs=int(n),
        n_groups=len(groups),
        design_condition_number=design_condition,
        reference_group=groups[0],
        groups_tested="|".join(groups),
        inference="spatial-block cluster-robust F test with conservative estimability gates",
    )
    slopes = []
    bootstrap_ci = _block_bootstrap_slopes(
        x_centered,
        y,
        labels,
        blocks,
        groups,
        replicates=bootstrap_replicates,
        seed=seed,
    )
    for group_index, group in enumerate(groups):
        coefficient = float(group_slopes[group_index])
        variance = float(slope_covariance[group_index, group_index])
        standard_error = float(np.sqrt(max(variance, 0.0))) if np.isfinite(variance) else float("nan")
        lower, upper = bootstrap_ci.get(group, (float("nan"), float("nan")))
        slopes.append(
            {
                "group_label": group,
                "n_pairs": int(np.count_nonzero(labels == group)),
                "spatial_blocks": int(group_block_counts[group]),
                "slope": coefficient,
                "cluster_robust_standard_error": standard_error,
                "cluster_robust_ci_low": coefficient - student_t.ppf(0.975, max(g - 1, 1)) * standard_error,
                "cluster_robust_ci_high": coefficient + student_t.ppf(0.975, max(g - 1, 1)) * standard_error,
                "block_bootstrap_ci_low": lower,
                "block_bootstrap_ci_high": upper,
                "block_bootstrap_replicates": int(bootstrap_replicates),
                "standardized_effect": coefficient * float(np.std(x)) / float(np.std(y)) if np.std(y) > 0 else float("nan"),
                "inference_status": result["status"],
            }
        )
    return result, pd.DataFrame.from_records(slopes)


def _benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(p_values, errors="coerce")
    valid = numeric.notna() & np.isfinite(numeric) & numeric.between(0.0, 1.0)
    adjusted = pd.Series(np.nan, index=p_values.index, dtype=float)
    values = numeric.loc[valid].to_numpy(dtype=float)
    if not len(values):
        return adjusted
    order = np.argsort(values)
    ranks = np.arange(1, len(values) + 1, dtype=float)
    ordered = np.minimum.accumulate((values[order] * len(values) / ranks)[::-1])[::-1]
    restored = np.empty(len(values), dtype=float)
    restored[order] = np.minimum(ordered, 1.0)
    adjusted.loc[valid] = restored
    return adjusted


def build_h1_inference_tables(
    pairs: pd.DataFrame,
    legacy_tests: pd.DataFrame,
    *,
    outcomes: list[str],
    group_columns: list[str],
    bootstrap_group_columns: list[str],
    bootstrap_replicates: int = 999,
    seed: int = 20261028,
    tolerance_rcond: tuple[float, ...] = (0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Recompute Phase G group-slope tests and compare them to gated inference."""

    required = {"method", "exposure", "erosion_high", "erosion_low", "spatial_block_id"}
    missing = sorted(required - set(pairs.columns))
    if missing:
        raise ValueError(f"H1 pair data is missing required columns: {missing}")
    test_records: list[dict[str, Any]] = []
    slope_frames: list[pd.DataFrame] = []
    sequence = 0
    for (method, exposure), pool in pairs.groupby(["method", "exposure"], sort=True, dropna=False):
        high_erosion = pd.to_numeric(pool["erosion_high"], errors="coerce").to_numpy(dtype=float)
        low_erosion = pd.to_numeric(pool["erosion_low"], errors="coerce").to_numpy(dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            x = np.log1p(high_erosion) - np.log1p(low_erosion)
        for group_column in group_columns:
            if group_column not in pool:
                continue
            for outcome_index, outcome in enumerate(outcomes):
                high_name, low_name = f"high_{outcome}", f"low_{outcome}"
                if high_name not in pool or low_name not in pool:
                    continue
                y = pd.to_numeric(pool[high_name], errors="coerce").to_numpy(dtype=float) - pd.to_numeric(
                    pool[low_name], errors="coerce"
                ).to_numpy(dtype=float)
                frame = pd.DataFrame(
                    {
                        "x": x,
                        "y": y,
                        "group": pool[group_column].to_numpy(),
                        "block": pool["spatial_block_id"].to_numpy(),
                    }
                )
                bootstrap = bootstrap_replicates if (
                    str(method) == "M4" and str(exposure) == "Aa_20yr" and group_column in bootstrap_group_columns
                ) else 0
                seed_for_fit = int(seed + sequence * 31 + outcome_index)
                test, slopes = fit_group_interaction(
                    frame,
                    x_column="x",
                    y_column="y",
                    group_column="group",
                    block_column="block",
                    bootstrap_replicates=bootstrap,
                    seed=seed_for_fit,
                    tolerance_rcond=tolerance_rcond,
                )
                record = {
                    "method": str(method),
                    "exposure": str(exposure),
                    "outcome": str(outcome),
                    "group_column": str(group_column),
                    "group_method": str(group_column).removesuffix("_group"),
                    "outcome_guided_selection": True,
                    "selection_mode": "EXPLORATORY_ORACLE_revised_group_heterogeneity_test",
                    "selection_objective": "compare group-specific Δhealth slopes after Aa matching; report all groups and gate inference on design/covariance/block diagnostics",
                    "legacy_inference": "Phase G pseudo-inverse joint Wald test",
                    **test,
                }
                test_records.append(record)
                if not slopes.empty:
                    slopes = slopes.assign(
                        method=str(method),
                        exposure=str(exposure),
                        outcome=str(outcome),
                        group_column=str(group_column),
                        outcome_guided_selection=True,
                        selection_objective="estimate group-specific Δhealth response to Δlog1p(Aa); selected primary fits include spatial-block bootstrap intervals",
                    )
                    slope_frames.append(slopes)
                sequence += 1

    revised = pd.DataFrame.from_records(test_records)
    diagnostics = revised.copy()
    if not diagnostics.empty and "group_spatial_blocks" in diagnostics:
        diagnostics["group_spatial_blocks"] = diagnostics["group_spatial_blocks"].map(
            lambda value: json.dumps(value, sort_keys=True)
        )
    if not revised.empty:
        reliable = revised["status"].eq("reliable")
        revised["interaction_q_value_revised"] = np.nan
        revised.loc[reliable, "interaction_q_value_revised"] = _benjamini_hochberg(
            revised.loc[reliable, "interaction_p_value"]
        ).to_numpy()
    slopes = pd.concat(slope_frames, ignore_index=True) if slope_frames else pd.DataFrame()

    keys = ["method", "exposure", "outcome", "group_column"]
    old = legacy_tests.copy()
    old_columns = [
        name for name in (
            *keys, "status", "interaction_df", "spatial_blocks", "interaction_wald_chi2",
            "interaction_p_value", "interaction_q_value_global",
        ) if name in old
    ]
    old = old[old_columns].rename(
        columns={
            "status": "legacy_status",
            "interaction_df": "legacy_interaction_df",
            "spatial_blocks": "legacy_spatial_blocks",
            "interaction_wald_chi2": "legacy_interaction_wald_chi2",
            "interaction_p_value": "legacy_interaction_p_value",
            "interaction_q_value_global": "legacy_interaction_q_value_global",
        }
    )
    revised_columns = [
        *keys, "status", "interaction_df", "spatial_blocks", "interaction_wald_chi2",
        "interaction_p_value", "interaction_q_value_revised", "covariance_rank", "covariance_condition_number",
        "tolerance_sensitivity_ratio",
    ]
    available_revised = [name for name in revised_columns if name in revised]
    comparison = old.merge(revised[available_revised], on=keys, how="outer", validate="one_to_one")
    comparison = comparison.rename(
        columns={
            "status": "revised_status",
            "interaction_df": "revised_interaction_df",
            "spatial_blocks": "revised_spatial_blocks",
            "interaction_wald_chi2": "revised_interaction_wald_chi2",
            "interaction_p_value": "revised_interaction_p_value",
        }
    )
    comparison["legacy_q_lt_0_05"] = pd.to_numeric(
        comparison.get("legacy_interaction_q_value_global"), errors="coerce"
    ).lt(0.05)
    comparison["revised_q_lt_0_05"] = pd.to_numeric(
        comparison.get("interaction_q_value_revised"), errors="coerce"
    ).lt(0.05)
    comparison["revised_inference_reliable"] = comparison.get("revised_status", pd.Series(index=comparison.index)).eq("reliable")
    return diagnostics, revised, slopes, comparison


def _write_h1_interaction_review_note(
    run_dir: Path,
    *,
    legacy_significant: int,
    legacy_count: int,
    revised: pd.DataFrame,
) -> None:
    reliable = revised.loc[revised["status"].eq("reliable")]
    revised_significant = int(pd.to_numeric(reliable.get("interaction_q_value_revised"), errors="coerce").lt(0.05).sum())
    status_labels = {
        "reliable": "通过可靠性检查",
        "unreliable_group_block_coverage": "空间块支持不足",
        "unreliable_covariance_rank_deficient": "协方差秩不足",
        "unreliable_tolerance_sensitive": "对数值容差敏感",
        "unreliable_design_condition": "设计矩阵条件不佳",
        "unreliable_constraints_per_block": "每个空间块承担的检验约束过多",
        "unreliable_covariance_condition": "协方差矩阵条件不佳",
        "non_estimable_design_rank": "设计矩阵不可估",
        "insufficient_groups": "可用分组不足",
    }
    status_counts = revised["status"].value_counts(dropna=False)
    status_table = "\n".join(f"- {status_labels.get(str(name), str(name))}：{count}" for name, count in status_counts.items())
    note = f"""# H 阶段交互推断修订说明

## 分析范围与方法

基于 F 阶段匹配配对，重新计算了 {len(revised):,} 项 G 阶段分组斜率交互检验。旧分析使用饱和的“分组×暴露”设计和摩尔—彭若斯伪逆，没有检查设计矩阵秩、条件数、组内空间块覆盖或数值容差敏感性。只有设计与协方差可估、组别支持充分、交互约束数相对空间块数不过多、且结果对设定的广义逆容差稳定时，才报告修订后的 p 值。

修订版整体检验采用空间块聚类稳健的 F 分布参考；分子自由度等于交互约束数，分母自由度等于空间块数减一。配置中指定的主要 M4、20 年平均实际侵蚀量分组斜率区间通过整块重抽样获得。

## 主要结果

- G 阶段旧版全局 q<0.05：**{legacy_significant:,} / {legacy_count:,}**。
- 通过全部可估性与稳定性检查：**{len(reliable):,} / {len(revised):,}**。
- 通过可靠性检查且修订版全局 q<0.05：**{revised_significant:,} / {len(reliable):,}**。

修订版状态数量：

{status_table}

旧版显著性数量仅用于对照。诊断未通过的检验不报告修订 p 值，也不参与修订版 q 值计算。设计矩阵与协方差矩阵的秩、条件数、空间块覆盖、特征值、交互约束与空间块比例、不同容差下的瓦尔德统计量敏感性见 `interaction_diagnostics.csv`；逐项旧版与修订版对照见 `interaction_test_comparison.csv`。
"""
    (Path(run_dir) / "INTERACTION_INFERENCE_REVIEW.md").write_text(note, encoding="utf-8")


def run_phase_h_interaction_review(
    project_root: Path,
    run_dir: Path,
    pairs: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    """Write the revised H1 diagnostics, inference tables, comparison, and review note."""

    cfg = validate_phase_h_config(config)
    h1_checkpoint = Path(run_dir) / "checkpoints" / "h1_complete.json"
    table_dir = Path(run_dir) / "tables"
    expected_tables = {
        "diagnostics": table_dir / "interaction_diagnostics.csv",
        "revised_tests": table_dir / "interaction_tests_revised.csv",
        "group_slopes": table_dir / "group_slope_revised.csv",
        "comparison": table_dir / "interaction_test_comparison.csv",
    }
    review_note_path = Path(run_dir) / "INTERACTION_INFERENCE_REVIEW.md"
    if h1_checkpoint.is_file() and all(path.is_file() for path in expected_tables.values()):
        cached = {name: pd.read_csv(path) for name, path in expected_tables.items()}
        if not review_note_path.is_file() or "交互推断修订说明" not in review_note_path.read_text(encoding="utf-8")[:100]:
            comparison = cached["comparison"]
            legacy_significant = int(comparison["legacy_q_lt_0_05"].fillna(False).astype(bool).sum())
            _write_h1_interaction_review_note(
                run_dir,
                legacy_significant=legacy_significant,
                legacy_count=len(comparison),
                revised=cached["revised_tests"],
            )
        return cached
    legacy_path = (
        Path(project_root)
        / cfg["output_root"]
        / cfg["phase_g_run_id"]
        / "tables"
        / "group_slope_interaction_tests.csv"
    )
    if not legacy_path.is_file():
        raise FileNotFoundError(f"Missing Phase G interaction results: {legacy_path}")
    legacy = pd.read_csv(legacy_path)
    inference_config = cfg["interaction_inference"]
    diagnostics, revised, slopes, comparison = build_h1_inference_tables(
        pairs,
        legacy,
        outcomes=list(cfg["outcomes"]),
        group_columns=list(inference_config["group_columns"]),
        bootstrap_group_columns=list(inference_config["bootstrap_group_columns"]),
        bootstrap_replicates=int(inference_config["bootstrap_replicates"]),
        seed=int(cfg["seed"]),
        tolerance_rcond=tuple(float(value) for value in inference_config["tolerance_rcond"]),
    )
    output_dir = Path(run_dir) / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(output_dir / "interaction_diagnostics.csv", index=False, encoding="utf-8-sig")
    revised.to_csv(output_dir / "interaction_tests_revised.csv", index=False, encoding="utf-8-sig")
    slopes.to_csv(output_dir / "group_slope_revised.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(output_dir / "interaction_test_comparison.csv", index=False, encoding="utf-8-sig")

    old_significant = int(pd.to_numeric(legacy.get("interaction_q_value_global"), errors="coerce").lt(0.05).sum())
    _write_h1_interaction_review_note(
        run_dir,
        legacy_significant=old_significant,
        legacy_count=len(legacy),
        revised=revised,
    )
    h1_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    h1_checkpoint.write_text(
        json.dumps(
            {
                "task": "H1_interaction_inference_repair",
                "status": "complete",
                "legacy_test_count": int(len(legacy)),
                "revised_test_count": int(len(revised)),
                "reliable_test_count": int(len(reliable)),
                "reliable_q_lt_0_05_count": revised_significant,
                "outcome_guided_selection": True,
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    return {
        "diagnostics": diagnostics,
        "revised_tests": revised,
        "group_slopes": slopes,
        "comparison": comparison,
    }


def validate_phase_h_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate Phase H's Oracle scope and output boundary."""

    if not isinstance(config, dict):
        raise ValueError("Phase H config must be a mapping")
    if config.get("phase") != "H":
        raise ValueError("phase must be H")
    if config.get("mode") != "EXPLORATORY_ORACLE":
        raise ValueError("mode must be EXPLORATORY_ORACLE")
    if config.get("outcome_guided_selection") is not True:
        raise ValueError("outcome_guided_selection must be true for Phase H Oracle runs")
    for key in ("input_run_id", "phase_f_run_id", "phase_g_run_id", "run_id"):
        value = config.get(key)
        if not isinstance(value, str) or not value.strip() or Path(value).name != value:
            raise ValueError(f"{key} must be a non-empty path-safe identifier")
    output_root = PurePosixPath(str(config.get("output_root", "")))
    if output_root.parts != ("outputs", "exploratory_oracle"):
        raise ValueError("output_root must be outputs/exploratory_oracle")
    exposures = config.get("exposures")
    if not isinstance(exposures, list) or not REQUIRED_EXPOSURES.issubset(exposures):
        raise ValueError("exposures must include Aa_20yr and Aa_2020")
    outcomes = config.get("outcomes")
    if not isinstance(outcomes, list) or not REQUIRED_OUTCOMES.issubset(outcomes):
        raise ValueError("outcomes must include SHI_candidate and all eight health indicators")
    soils = [int(value) for value in config.get("primary_soil_codes", [])]
    if soils != PRIMARY_SOILS:
        raise ValueError("primary_soil_codes must be [26, 27, 42]")
    if not isinstance(config.get("seed"), int):
        raise ValueError("seed must be an integer")
    return dict(config)


def merge_phase_h_pair_labels(pairs: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Attach Phase G group labels to exact Phase F pair memberships."""

    key = "pair_id"
    membership = "high_erosion_cell_id"
    for name, frame in (("pairs", pairs), ("labels", labels)):
        missing = {key, membership} - set(frame.columns)
        if missing:
            raise ValueError(f"{name} table is missing required columns: {sorted(missing)}")
        if frame[key].isna().any() or frame[key].duplicated().any():
            raise ValueError(f"{name} table requires unique non-null pair_id values")
    pair_ids = pd.Index(pairs[key])
    label_ids = pd.Index(labels[key])
    if len(pair_ids) != len(label_ids) or not pair_ids.isin(label_ids).all() or not label_ids.isin(pair_ids).all():
        raise ValueError("Phase F pairs and Phase G labels must contain the same pair_id values")

    aligned = pairs[[key, membership]].merge(
        labels[[key, membership]],
        on=key,
        how="left",
        validate="one_to_one",
        suffixes=("_pair", "_label"),
        sort=False,
    )
    same_membership = aligned[f"{membership}_pair"].eq(aligned[f"{membership}_label"]).fillna(False)
    if not same_membership.all():
        bad_pair_id = aligned.loc[~same_membership, key].iloc[0]
        raise ValueError(f"Phase G high_erosion_cell_id does not match Phase F membership for pair_id {bad_pair_id}")

    label_columns = labels.drop(columns=[membership])
    joined = pairs.merge(label_columns, on=key, how="left", validate="one_to_one", sort=False)
    if len(joined) != len(pairs):
        raise ValueError("Pair-label join changed the number of Phase F pairs")
    return joined


def load_phase_h_inputs(project_root: Path, config: dict[str, Any]) -> PhaseHInputs:
    """Load existing aligned cells and the Phase F/G pair-label tables."""

    cfg = validate_phase_h_config(config)
    root = Path(project_root)
    aligned_dir = root / "outputs" / cfg["input_run_id"]
    phase_f_dir = root / cfg["output_root"] / cfg["phase_f_run_id"]
    phase_g_dir = root / cfg["output_root"] / cfg["phase_g_run_id"]
    input_paths = {
        "aligned_environment": aligned_dir / "databases" / "environment_erosion.parquet",
        "aligned_health": aligned_dir / "databases" / "health.parquet",
        "phase_f_pairs": phase_f_dir / "pairs" / "full_sample_pairs.parquet",
        "phase_g_labels": phase_g_dir / "tables" / "environment_group_labels.parquet",
    }
    bundle = load_oracle_data(aligned_dir)
    for role in ("phase_f_pairs", "phase_g_labels"):
        if not input_paths[role].is_file():
            raise FileNotFoundError(f"Missing {role} Parquet table: {input_paths[role]}")
    pairs = pd.read_parquet(input_paths["phase_f_pairs"])
    labels = pd.read_parquet(input_paths["phase_g_labels"])
    joined = merge_phase_h_pair_labels(pairs, labels)
    if "method" not in joined or not joined["method"].isin(["M3", "M4"]).all():
        raise ValueError("Phase F input must contain only M3/M4 matched pairs")
    if "exposure" not in joined or set(joined["exposure"].dropna().astype(str)) != REQUIRED_EXPOSURES:
        raise ValueError("Phase F input contains an unsupported Aa exposure")
    return PhaseHInputs(
        cells=bundle.frame,
        pairs=joined,
        input_run_id=bundle.input_run_id,
        phase_f_run_id=cfg["phase_f_run_id"],
        phase_g_run_id=cfg["phase_g_run_id"],
        input_paths=input_paths,
        outcome_counts=bundle.outcome_counts,
    )


def _git_metadata(project_root: Path) -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=project_root, capture_output=True, text=True, check=False
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=project_root, capture_output=True, text=True, check=False
    )
    return {
        "git_sha": revision.stdout.strip() if revision.returncode == 0 else "unknown",
        "git_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def _software_versions() -> dict[str, str]:
    packages = {
        "numpy": "numpy",
        "pandas": "pandas",
        "scipy": "scipy",
        "scikit_learn": "scikit-learn",
        "statsmodels": "statsmodels",
        "pyarrow": "pyarrow",
        "rasterio": "rasterio",
    }
    versions: dict[str, str] = {"python": platform.python_version()}
    for key, package in packages.items():
        try:
            versions[key] = version(package)
        except PackageNotFoundError:
            versions[key] = "not-installed"
    return versions


def prepare_phase_h_run(project_root: Path, config: dict[str, Any]) -> tuple[Path, PhaseHInputs]:
    """Initialize or resume one immutable Oracle run and write its input inventory."""

    cfg = validate_phase_h_config(config)
    root = Path(project_root).resolve()
    inputs = load_phase_h_inputs(root, cfg)
    run_dir = root / cfg["output_root"] / cfg["run_id"]
    if run_dir.exists():
        existing_config = run_dir / "config.yaml"
        if not existing_config.is_file():
            raise FileExistsError(f"Phase H output already exists without a resumable config: {run_dir}")
        with existing_config.open(encoding="utf-8") as stream:
            prior_config = yaml.safe_load(stream)
        if prior_config != cfg:
            raise FileExistsError(f"Phase H run already exists with a different config: {run_dir}")
        for directory in REQUIRED_DIRECTORIES:
            (run_dir / directory).mkdir(parents=True, exist_ok=True)
        return run_dir, inputs

    run_dir.mkdir(parents=True)
    for directory in REQUIRED_DIRECTORIES:
        (run_dir / directory).mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    import pyarrow.parquet as pq

    inventory_rows = []
    for role, path in inputs.input_paths.items():
        parquet = pq.ParquetFile(path)
        source_run_id = (
            inputs.input_run_id if role.startswith("aligned_")
            else inputs.phase_f_run_id if role == "phase_f_pairs"
            else inputs.phase_g_run_id
        )
        inventory_rows.append(
            {
                "role": role,
                "path": str(path),
                "row_count": parquet.metadata.num_rows,
                "column_count": len(parquet.schema_arrow.names),
                "source_run_id": source_run_id,
            }
        )
    inventory = pd.DataFrame.from_records(inventory_rows)
    inventory.to_csv(run_dir / "tables" / "input_inventory.csv", index=False, encoding="utf-8-sig")
    provenance = {
        "phase": "H",
        "run_id": cfg["run_id"],
        "mode": cfg["mode"],
        "outcome_guided_selection": cfg["outcome_guided_selection"],
        "selection_outcomes": cfg["outcomes"],
        "seed": cfg["seed"],
        "input_run_id": inputs.input_run_id,
        "phase_f_run_id": inputs.phase_f_run_id,
        "phase_g_run_id": inputs.phase_g_run_id,
        "cell_rows": len(inputs.cells),
        "pair_rows": len(inputs.pairs),
        "outcome_nonmissing_counts": inputs.outcome_counts,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "software_versions": _software_versions(),
        **_git_metadata(root),
    }
    (run_dir / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return run_dir, inputs
