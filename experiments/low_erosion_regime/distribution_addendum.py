"""Reproducible addendum for distribution detail and GMM stability checks."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

from experiments.low_erosion_regime.distribution_diagnostics import (
    fit_gaussian_mixtures,
    summarize_values,
)


THRESHOLD_COLUMNS = [
    "exposure", "threshold_id", "method", "method_parameters", "threshold_value", "threshold_unit",
    "zero_fraction", "near_zero_fraction", "total_low_value_fraction", "retained_cells",
    "retained_fraction", "retained_fraction_positive", "component_count", "component_weights",
    "threshold_stability", "bootstrap_ci_low", "bootstrap_ci_high", "separation_quality",
    "model_fit_quality", "status", "failure_reason", "candidate_source", "outcome_guided_selection",
]


def fit_gmm_seed_sensitivity(
    values: Iterable[float], *, exposure: str, seeds: Iterable[int], sample_size: int,
    rows: Iterable[int], cols: Iterable[int],
) -> pd.DataFrame:
    """Refit K=1..4 on spatially stratified samples across fixed random seeds."""
    values = np.asarray(values, dtype=float)
    rows = np.asarray(rows)
    cols = np.asarray(cols)
    frames = []
    for seed in seeds:
        models, _ = fit_gaussian_mixtures(
            values, exposure=exposure, seed=int(seed), sample_size=int(sample_size), rows=rows, cols=cols,
        )
        if models.empty:
            continue
        selected_k = int(models.loc[models["bic"].idxmin(), "n_components"])
        models = models.copy()
        models["seed"] = int(seed)
        models["selected_k_bic"] = selected_k
        models["selected_by_bic"] = models["n_components"].eq(selected_k)
        frames.append(models)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _split_list(value: Any) -> list[float]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, str):
        tokens = value.split("|")
    elif isinstance(value, (list, tuple, np.ndarray)):
        tokens = list(value)
    else:
        tokens = [value]
    result = []
    for token in tokens:
        try:
            number = float(token)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            result.append(number)
    return result


def summarize_gmm_seed_sensitivity(trials: pd.DataFrame) -> pd.DataFrame:
    """Summarize BIC-selected K agreement and the first low-component boundary."""
    if trials.empty:
        return pd.DataFrame()
    summaries = []
    for exposure, part in trials.groupby("exposure", sort=False):
        selected = part.loc[part["selected_by_bic"].fillna(False)].copy()
        if selected.empty:
            continue
        selected_k = pd.to_numeric(selected["selected_k_bic"], errors="coerce").dropna().astype(int)
        frequencies = selected_k.value_counts().sort_index()
        modes = frequencies.loc[frequencies.eq(frequencies.max())].index.astype(int).tolist()
        selected_k_mode = min(modes)
        selected_boundaries = []
        for row in selected.to_dict("records"):
            if int(row["selected_k_bic"]) < 2:
                continue
            boundaries = _split_list(row.get("adjacent_boundaries_log10"))
            if boundaries:
                selected_boundaries.append(boundaries[0])
        original = np.power(10.0, selected_boundaries) if selected_boundaries else np.array([], dtype=float)
        selected_converged = selected["converged"].fillna(False).astype(bool)
        summaries.append({
            "exposure": exposure,
            "seed_count": int(selected["seed"].nunique()),
            "selected_k_counts": json.dumps({str(int(k)): int(v) for k, v in frequencies.items()}),
            "selected_k_mode": int(selected_k_mode),
            "selected_k_agreement_fraction": float(frequencies.max() / max(frequencies.sum(), 1)),
            "selected_convergence_fraction": float(selected_converged.mean()),
            "first_boundary_seed_count": int(len(original)),
            "first_boundary_median_log10": float(np.median(selected_boundaries)) if selected_boundaries else np.nan,
            "first_boundary_median_original": float(np.median(original)) if len(original) else np.nan,
            "first_boundary_cv": float(np.std(original, ddof=1) / np.mean(original))
            if len(original) > 1 and np.mean(original) else np.nan,
            "first_boundary_ci_low": float(np.quantile(original, .025)) if len(original) else np.nan,
            "first_boundary_ci_high": float(np.quantile(original, .975)) if len(original) else np.nan,
            "candidate_source": "distribution_derived_gmm_seed_sensitivity",
            "outcome_guided_selection": False,
        })
    return pd.DataFrame.from_records(summaries)


def summarize_posterior_cutoffs(
    assignments: pd.DataFrame, *, n_valid: int, cutoffs: Iterable[float] = (.80, .90, .95),
) -> pd.DataFrame:
    """Compare confident low-component, other-component and uncertain counts."""
    if assignments.empty:
        return pd.DataFrame()
    max_probability = pd.to_numeric(assignments["posterior_max"], errors="coerce").to_numpy(float)
    component = pd.to_numeric(assignments["posterior_component"], errors="coerce").to_numpy(float)
    selected_k_values = pd.to_numeric(
        assignments.get("selected_n_components_bic", pd.Series(index=assignments.index, dtype=float)),
        errors="coerce",
    ).dropna()
    selected_k = int(selected_k_values.mode().min()) if len(selected_k_values) else 1
    low_identifiable = selected_k > 1
    positive_n = len(assignments)
    rows = []
    for cutoff in cutoffs:
        confident = np.isfinite(max_probability) & (max_probability >= float(cutoff))
        uncertain = np.isfinite(max_probability) & ~confident
        low = confident & (component == 1) if low_identifiable else np.zeros(len(component), dtype=bool)
        other = confident & (component > 1) if low_identifiable else confident
        n_low = int(low.sum())
        n_other = int(other.sum())
        n_uncertain = int(uncertain.sum())
        rows.append({
            "exposure": assignments["exposure"].iloc[0] if "exposure" in assignments else None,
            "selected_k_bic": selected_k,
            "low_component_identifiable": low_identifiable,
            "posterior_cutoff": float(cutoff),
            "n_positive": positive_n,
            "n_valid_including_zero": int(n_valid),
            "n_low_component_assigned": n_low,
            "n_other_component_assigned": n_other,
            "n_uncertain_positive": n_uncertain,
            "low_component_fraction_positive": n_low / positive_n if positive_n else np.nan,
            "low_component_fraction_all_valid": n_low / n_valid if n_valid else np.nan,
            "uncertain_fraction_positive": n_uncertain / positive_n if positive_n else np.nan,
            "candidate_source": "distribution_derived_gmm_posterior",
            "outcome_guided_selection": False,
        })
    return pd.DataFrame.from_records(rows)


def build_threshold_candidate_table(
    *, exposure: str, values: Iterable[float], kde: pd.DataFrame, gmm: pd.DataFrame,
    cdf: pd.DataFrame, otsu: pd.DataFrame, kde_bootstrap: pd.DataFrame,
    cdf_bootstrap: pd.DataFrame, gmm_seed_trials: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Combine every distribution threshold attempt and its support into one table."""
    raw = np.asarray(values, dtype=float).reshape(-1)
    finite_values = raw[np.isfinite(raw)]
    positives = finite_values[finite_values > 0]
    zero_fraction = float(np.mean(finite_values == 0)) if len(finite_values) else np.nan
    rows: list[dict[str, Any]] = []

    def method_frame(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty or "exposure" not in frame:
            return frame.iloc[0:0]
        return frame.loc[frame["exposure"].eq(exposure)].copy()

    def add(
        *, method: str, parameters: dict[str, Any], threshold: float | None,
        status: str, failure_reason: str | None = None, component_count: int | None = None,
        component_weights: Any = None, stability: dict[str, Any] | None = None,
        ci_low: float | None = None, ci_high: float | None = None,
        separation: dict[str, Any] | None = None, fit_quality: dict[str, Any] | None = None,
        retained_fraction_positive: float | None = None,
    ) -> None:
        valid_threshold = threshold is not None and np.isfinite(threshold) and threshold >= 0
        if valid_threshold and len(finite_values):
            low_count = int(np.count_nonzero(finite_values <= threshold))
            retained_count = int(np.count_nonzero(finite_values > threshold))
            retained_fraction = retained_count / len(finite_values)
            near_fraction = float(np.mean(positives <= threshold)) if len(positives) else np.nan
            if retained_fraction_positive is None and len(positives):
                retained_fraction_positive = float(np.mean(positives > threshold))
        else:
            low_count = retained_count = None
            retained_fraction = near_fraction = np.nan
        rows.append({
            "exposure": exposure, "method": method, "method_parameters": json.dumps(parameters, ensure_ascii=False, default=str),
            "threshold_value": float(threshold) if valid_threshold else np.nan,
            "threshold_unit": "Aa 原始单位；来源元数据未提供单位",
            "zero_fraction": zero_fraction,
            "near_zero_fraction": near_fraction,
            "total_low_value_fraction": low_count / len(finite_values) if low_count is not None and len(finite_values) else np.nan,
            "retained_cells": retained_count,
            "retained_fraction": retained_fraction,
            "retained_fraction_positive": retained_fraction_positive,
            "component_count": component_count,
            "component_weights": "|".join(map(str, _split_list(component_weights))) if component_weights is not None else None,
            "threshold_stability": json.dumps(stability or {}, ensure_ascii=False, default=str),
            "bootstrap_ci_low": ci_low,
            "bootstrap_ci_high": ci_high,
            "separation_quality": json.dumps(separation or {}, ensure_ascii=False, default=str),
            "model_fit_quality": json.dumps(fit_quality or {}, ensure_ascii=False, default=str),
            "status": status,
            "failure_reason": failure_reason,
            "candidate_source": "distribution_derived",
            "outcome_guided_selection": False,
        })

    kde_exp = method_frame(kde)
    kde_boot_exp = method_frame(kde_bootstrap)
    for item in kde_exp.to_dict("records"):
        base = str(item.get("bandwidth_base", "unknown"))
        multiplier = float(item.get("bandwidth_multiplier", np.nan))
        stability: dict[str, Any] = {"assessment": "spatial_block_bootstrap" if base == "scott" and multiplier == 1.0 else "bandwidth_sensitivity_only"}
        ci_low = ci_high = None
        if base == "scott" and multiplier == 1.0 and not kde_boot_exp.empty:
            valid_boot = pd.to_numeric(
                kde_boot_exp.loc[kde_boot_exp["status"].eq("valid_valley"), "threshold_original"], errors="coerce",
            ).dropna().to_numpy(float)
            stability.update({
                "replicate_count": int(len(kde_boot_exp)),
                "valid_fraction": float(len(valid_boot) / len(kde_boot_exp)),
                "valid_threshold_median": float(np.median(valid_boot)) if len(valid_boot) else None,
            })
            if len(valid_boot):
                ci_low, ci_high = map(float, np.quantile(valid_boot, [.025, .975]))
        threshold = item.get("threshold_original")
        add(
            method=f"KDE_{base}_{multiplier:g}x",
            parameters={"bandwidth_base": base, "bandwidth_multiplier": multiplier,
                        "fit_sample_n": item.get("fit_sample_n"), "sample_design": item.get("sample_design")},
            threshold=threshold, status=str(item.get("status", "missing_status")),
            failure_reason="未达到双峰分离判据，未识别可靠密度谷" if pd.isna(threshold) else None,
            stability=stability, ci_low=ci_low, ci_high=ci_high,
            separation={"low_peak_log10": item.get("low_peak_log10"),
                        "next_peak_log10": item.get("next_peak_log10"),
                        "valley_density_ratio": item.get("valley_density_ratio")},
            retained_fraction_positive=item.get("retained_fraction_positive"),
        )

    gmm_exp = method_frame(gmm)
    seed_exp = method_frame(gmm_seed_trials if gmm_seed_trials is not None else pd.DataFrame())
    if not gmm_exp.empty:
        best = gmm_exp.loc[pd.to_numeric(gmm_exp["bic"], errors="coerce").idxmin()]
        selected_k = int(best["n_components"])
        boundaries = _split_list(best.get("adjacent_boundaries_log10"))
        overlaps = _split_list(best.get("adjacent_overlap_index"))
        weights = best.get("component_weights")
        seed_selected = seed_exp.loc[seed_exp.get("selected_by_bic", pd.Series(False, index=seed_exp.index)).fillna(False)] if not seed_exp.empty else pd.DataFrame()
        k_frequency = float(seed_selected["selected_k_bic"].eq(selected_k).mean()) if not seed_selected.empty else np.nan
        if selected_k < 2 or not boundaries:
            add(
                method=f"GMM_BIC_K{selected_k}_no_boundary", parameters={"selected_by": "BIC", "K": selected_k},
                threshold=None, status="single_component_no_boundary",
                failure_reason="BIC 选择单成分模型，没有低值/高值分界。", component_count=selected_k,
                component_weights=weights, stability={"seed_selected_k_agreement_fraction": k_frequency},
                fit_quality={"AIC": best.get("aic"), "BIC": best.get("bic"), "converged": best.get("converged")},
            )
        else:
            for index, boundary_log in enumerate(boundaries):
                overlap = overlaps[index] if index < len(overlaps) else np.nan
                threshold = float(10 ** boundary_log)
                seed_thresholds = []
                if not seed_selected.empty:
                    same_k = seed_selected.loc[pd.to_numeric(seed_selected["selected_k_bic"], errors="coerce").eq(selected_k)]
                    for boundary_value in same_k["adjacent_boundaries_log10"]:
                        values_for_seed = _split_list(boundary_value)
                        if len(values_for_seed) > index:
                            seed_thresholds.append(float(10 ** values_for_seed[index]))
                seed_thresholds_array = np.asarray(seed_thresholds, dtype=float)
                stability = {
                    "seed_count": int(seed_selected["seed"].nunique()) if not seed_selected.empty else 0,
                    "selected_k_agreement_fraction": k_frequency,
                    "same_k_boundary_seed_count": int(len(seed_thresholds_array)),
                    "same_k_boundary_cv": float(np.std(seed_thresholds_array, ddof=1) / np.mean(seed_thresholds_array))
                    if len(seed_thresholds_array) > 1 and np.mean(seed_thresholds_array) else None,
                    "same_k_boundary_median": float(np.median(seed_thresholds_array)) if len(seed_thresholds_array) else None,
                    "same_k_boundary_min": float(seed_thresholds_array.min()) if len(seed_thresholds_array) else None,
                    "same_k_boundary_max": float(seed_thresholds_array.max()) if len(seed_thresholds_array) else None,
                }
                status = "candidate_distribution_boundary" if np.isfinite(overlap) and overlap <= .35 else "rejected_component_overlap"
                add(
                    method=f"GMM_BIC_K{selected_k}_boundary_{index + 1}",
                    parameters={"selected_by": "BIC", "K": selected_k, "boundary_index": index + 1,
                                "seed_list": sorted(seed_selected["seed"].dropna().astype(int).unique().tolist()) if not seed_selected.empty else []},
                    threshold=threshold, status=status,
                    failure_reason=None if status == "candidate_distribution_boundary" else "相邻混合成分重叠超过预设可分性界限 0.35。",
                    component_count=selected_k, component_weights=weights, stability=stability,
                    separation={"adjacent_overlap_index": overlap},
                    fit_quality={"AIC": best.get("aic"), "BIC": best.get("bic"), "converged": best.get("converged")},
                )
    else:
        add(method="GMM", parameters={"components": [1, 2, 3, 4]}, threshold=None,
            status="fit_unavailable", failure_reason="没有可用的 GMM 拟合结果。")

    cdf_exp = method_frame(cdf)
    cdf_boot_exp = method_frame(cdf_bootstrap)
    for item in cdf_exp.to_dict("records"):
        boot_values = pd.to_numeric(
            cdf_boot_exp.loc[cdf_boot_exp["status"].eq("valid_cdf_breakpoint"), "threshold_original"], errors="coerce",
        ).dropna().to_numpy(float) if not cdf_boot_exp.empty else np.array([], dtype=float)
        stability = {
            "replicate_count": int(len(cdf_boot_exp)),
            "valid_fraction": float(len(boot_values) / len(cdf_boot_exp)) if len(cdf_boot_exp) else 0.0,
            "threshold_cv": float(np.std(boot_values, ddof=1) / np.mean(boot_values))
            if len(boot_values) > 1 and np.mean(boot_values) else None,
        }
        ci_low, ci_high = map(float, np.quantile(boot_values, [.025, .975])) if len(boot_values) else (None, None)
        threshold = item.get("threshold_original")
        add(
            method="empirical_CDF_breakpoint",
            parameters={"x": "log10(Aa_positive)", "y": "empirical_cdf", "fit_n": item.get("fit_n"),
                        "delta_aic_one_break_vs_linear": item.get("delta_aic_one_break_vs_linear")},
            threshold=threshold, status=str(item.get("status", "missing_status")),
            failure_reason=None if pd.notna(threshold) else "分段CDF模型未支持可靠单折点。",
            stability=stability, ci_low=ci_low, ci_high=ci_high,
            separation={"delta_aic_one_break_vs_linear": item.get("delta_aic_one_break_vs_linear")},
            fit_quality={"delta_aic_one_break_vs_linear": item.get("delta_aic_one_break_vs_linear")},
        )

    otsu_exp = method_frame(otsu)
    otsu_threshold_values = pd.to_numeric(otsu_exp.get("threshold_original", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(float)
    otsu_stability = {
        "assessment": "histogram_bin_sensitivity",
        "bin_count": int(len(otsu_threshold_values)),
        "threshold_min": float(otsu_threshold_values.min()) if len(otsu_threshold_values) else None,
        "threshold_max": float(otsu_threshold_values.max()) if len(otsu_threshold_values) else None,
    }
    for item in otsu_exp.to_dict("records"):
        add(
            method="Otsu",
            parameters={"histogram_bins": item.get("histogram_bins"), "log10_positive_scale": True},
            threshold=item.get("threshold_original"), status=str(item.get("status", "comparison_only")),
            failure_reason="仅作直方图分箱敏感性对照，不用于主阈值选择。",
            stability=otsu_stability,
            separation={"threshold_range_across_bins": [otsu_stability["threshold_min"], otsu_stability["threshold_max"]]},
            fit_quality={"criterion": "Otsu between-class variance"},
        )

    result = pd.DataFrame.from_records(rows)
    if result.empty:
        return pd.DataFrame(columns=THRESHOLD_COLUMNS)
    result.insert(1, "threshold_id", [f"D{i:03d}" for i in range(1, len(result) + 1)])
    return result.reindex(columns=THRESHOLD_COLUMNS)


def _write_table(frame: pd.DataFrame, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_csv(directory / f"{stem}.csv", index=False, encoding="utf-8-sig")
    frame.to_parquet(directory / f"{stem}.parquet", index=False)


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "（无可用记录）"
    columns = [str(value) for value in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for record in frame.to_dict("records"):
        values = []
        for column in frame.columns:
            value = record[column]
            if pd.isna(value):
                value = "—"
            elif isinstance(value, (float, np.floating)):
                if any(token in str(column) for token in ("比例", "一致率", "收敛率", "变异系数", "CV")):
                    value = f"{value:.1%}"
                elif "后验概率阈值" in str(column):
                    value = f"{value:.2f}"
                else:
                    value = f"{value:.4f}"
            elif isinstance(value, (int, np.integer)):
                value = f"{value:,}"
            values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _period_label(value: object) -> str:
    return {"Aa_20yr": "20 年平均实际侵蚀", "Aa_2020": "2020 年实际侵蚀"}.get(str(value), str(value))


def _k_frequency_label(value: object) -> str:
    try:
        counts = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return str(value)
    return "；".join(f"K={key}：{count} 次" for key, count in sorted(counts.items(), key=lambda item: int(item[0])))


def build_addendum_report_text(
    config: dict[str, Any], summary: pd.DataFrame, seed_summary: pd.DataFrame,
    posterior_summary: pd.DataFrame,
) -> str:
    """Render a Chinese narrative and Chinese-labeled evidence tables."""
    summary_view = summary.copy()
    summary_view["侵蚀时期"] = summary_view["exposure"].map(_period_label)
    summary_columns = {
        "n_finite": "有效侵蚀像元", "n_zero": "精确零值像元",
        "positive_q01": "正值 P1", "positive_q05": "正值 P5", "positive_q10": "正值 P10",
        "positive_q25": "正值 P25", "positive_q50": "正值 P50", "positive_q75": "正值 P75",
        "positive_q90": "正值 P90", "positive_q95": "正值 P95", "positive_q99": "正值 P99",
    }
    summary_view = summary_view[["侵蚀时期", *summary_columns]].rename(columns=summary_columns)

    seed_view = seed_summary.copy()
    seed_view["侵蚀时期"] = seed_view["exposure"].map(_period_label)
    seed_view["BIC 选中 K 频次"] = seed_view["selected_k_counts"].map(_k_frequency_label)
    seed_view["首个分界范围（Aa 原值）"] = seed_view.apply(
        lambda row: f"{row['first_boundary_ci_low']:.4f}–{row['first_boundary_ci_high']:.4f}"
        if pd.notna(row.get("first_boundary_ci_low")) and pd.notna(row.get("first_boundary_ci_high")) else "—",
        axis=1,
    )
    seed_columns = {
        "seed_count": "随机种子数", "selected_k_mode": "选中 K 众数",
        "selected_k_agreement_fraction": "众数 K 一致率", "selected_convergence_fraction": "收敛率",
        "first_boundary_seed_count": "可用分界数", "first_boundary_median_original": "首个分界中位数（Aa 原值）",
        "first_boundary_cv": "首个分界变异系数", "first_boundary_ci_low": "低端 2.5%分位",
        "first_boundary_ci_high": "高端 97.5%分位",
    }
    seed_view = seed_view[["侵蚀时期", "seed_count", "BIC 选中 K 频次", "selected_k_mode",
                           "selected_k_agreement_fraction", "selected_convergence_fraction",
                           "first_boundary_seed_count", "first_boundary_median_original",
                           "first_boundary_cv", "首个分界范围（Aa 原值）"]].rename(columns=seed_columns)

    posterior_view = posterior_summary.copy()
    posterior_view["侵蚀时期"] = posterior_view["exposure"].map(_period_label)
    posterior_columns = {
        "selected_k_bic": "BIC 选中成分数", "posterior_cutoff": "后验概率阈值",
        "n_positive": "正值像元数", "n_low_component_assigned": "高置信归入低值成分",
        "n_other_component_assigned": "高置信归入其他成分", "n_uncertain_positive": "归属不确定像元",
        "low_component_fraction_positive": "低值成分占正值比例",
        "uncertain_fraction_positive": "不确定像元占正值比例",
    }
    posterior_view = posterior_view[["侵蚀时期", *posterior_columns]].rename(columns=posterior_columns)

    seed_interpretations = []
    for row in seed_summary.to_dict("records"):
        period = _period_label(row["exposure"])
        counts = _k_frequency_label(row["selected_k_counts"])
        cv = row.get("first_boundary_cv")
        median = row.get("first_boundary_median_original")
        interval = row.get("first_boundary_ci_low"), row.get("first_boundary_ci_high")
        interval_text = f"{interval[0]:.4f}–{interval[1]:.4f}" if all(pd.notna(v) for v in interval) else "区间不可用"
        if pd.notna(cv) and pd.notna(median):
            seed_interpretations.append(
                f"- **{period}：** BIC 选中频次为{counts}；首个相邻成分分界中位数为 {median:.4f}，"
                f"种子间变异系数为 {cv:.1%}，经验分位范围为 {interval_text}。"
            )
        else:
            seed_interpretations.append(f"- **{period}：** BIC 选中频次为{counts}；没有稳定的多成分分界可汇总。")

    seed_list = tuple(int(seed) for seed in config["gmm_stability_seeds"])
    summary_md = _markdown_table(summary_view)
    seed_md = _markdown_table(seed_view)
    posterior_md = _markdown_table(posterior_view)
    return f"""# Phase H 侵蚀分布诊断补充

运行编号：`{config['run_id']}`；模式：`EXPLORATORY_ORACLE`。本补充只读取既有 Aa 与分布诊断，没有读取健康结局；`outcome_guided_selection=true` 是 Oracle 运行标记，`outcome_guided_selection_used=false` 表示本补充没有按健康结果筛选阈值。所有结果写入独立运行目录，没有覆盖原运行。

## 正侵蚀值分位数

分母只包括有效耕地范围内有限的正 Aa；零值和 NoData 单独统计。表中阈值沿用 Aa 原值单位，但输入元数据没有提供单位说明。

{summary_md}

重复值比例按正 Aa 分别四舍五入至 0、1、2、3、6、9、12 位小数后计算；完整精度表和各 log10 数量级频率见 [`exposure_distribution_detail.csv`](../diagnostics/exposure_distribution_detail.csv)。

## GMM 随机种子稳定性

每个侵蚀时期使用 {len(seed_list)} 个固定随机种子（`{', '.join(map(str, seed_list))}`），每个种子在按 5 km 空间块分层抽取的 {config['gmm_sample_size']:,} 个正值样本上拟合 K=1–4。分界范围是种子间经验分位数，不是置信区间。

{seed_md}

{chr(10).join(seed_interpretations)}

这项敏感性同时包含空间分层样本变化和 GMM 初始化变化。20 年平均实际侵蚀的 BIC 选中成分数在 K=2 与 K=3 之间切换；2020 年虽然各次均选中 K=4，首个分界的种子间相对波动仍较大。因此，K 的稳定不等于分界位置稳定，混合成分也不能直接解释为彼此独立的自然侵蚀类别。

逐种子、逐 K 的 AIC/BIC、组分参数及收敛状态见 [`gmm_seed_stability_by_seed.csv`](../diagnostics/gmm_seed_stability_by_seed.csv)。

## GMM 后验分类阈值敏感性

依据原始 BIC 最优模型的后验概率，分别以 0.80、0.90、0.95 区分高置信分类与不确定像元。低值成分比例只在 K≥2 时计算。

{posterior_md}

各阈值下的像元归属变化见[后验分类图](../figures/png/HADD02_gmm_posterior_sensitivity.png)及[`gmm_posterior_cutoff_sensitivity.csv`](../diagnostics/gmm_posterior_cutoff_sensitivity.csv)。

## 统一阈值候选表

[`low_erosion_threshold_candidates.csv`](../thresholds/low_erosion_threshold_candidates.csv) 将所有 KDE 带宽、BIC 最优 GMM 分界、经验 CDF 折点和 Otsu 分箱候选放在同一表中，记录阈值单位说明、零值与低值比例、保留量、稳定性、分离度、拟合信息和失败原因。高重叠 GMM 分界与未通过判据的 KDE 结果作为失败候选保留；Otsu 仍只作敏感性对照。

候选范围与空间重抽样结果见[阈值比较图](../figures/png/HADD03_threshold_candidate_stability.png)。总体上，两期 KDE 均未形成可靠密度谷；CDF 折点是可重复的分布转折候选；GMM 的种子敏感性、分量重叠和后验不确定性不支持将其硬化为唯一筛选线。CDF 候选不证明存在独立侵蚀类别或数据错误边界。

"""


def write_distribution_addendum_report(run_dir: Path, config: dict[str, Any] | None = None) -> Path:
    """Rebuild the Chinese report from persisted tables without refitting models."""
    run_dir = Path(run_dir)
    if config is None:
        config = yaml.safe_load((run_dir / "config/config.yaml").read_text(encoding="utf-8"))
    summary = pd.read_csv(run_dir / "diagnostics/exposure_distribution_detail.csv")
    seed_summary = pd.read_csv(run_dir / "diagnostics/gmm_seed_stability_summary.csv")
    posterior_summary = pd.read_csv(run_dir / "diagnostics/gmm_posterior_cutoff_sensitivity.csv")
    path = run_dir / "reports/LOW_EROSION_DISTRIBUTION_ADDENDUM.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_addendum_report_text(config, summary, seed_summary, posterior_summary), encoding="utf-8")
    return path


def run_distribution_addendum(project_root: Path, config_path: Path) -> Path:
    """Build additive diagnostics in a new Oracle run without altering source runs."""
    project_root, config_path = Path(project_root), Path(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("mode") != "EXPLORATORY_ORACLE" or config.get("outcome_guided_selection") is not True:
        raise ValueError("distribution addendum must use EXPLORATORY_ORACLE and outcome_guided_selection=true")
    output_root = Path(config["output_root"])
    if output_root.is_absolute() or ".." in output_root.parts or output_root.parts[:2] != ("outputs", "exploratory_oracle"):
        raise ValueError("output_root must remain within outputs/exploratory_oracle")
    run_dir = project_root / output_root / str(config["run_id"])
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")
    source_run = project_root / output_root / str(config["source_run_id"])
    input_path = project_root / "outputs" / str(config["input_run_id"]) / "databases" / "environment_erosion.parquet"
    if not source_run.is_dir() or not input_path.is_file():
        raise FileNotFoundError(f"missing source run or aligned database: {source_run}, {input_path}")

    cells = pd.read_parquet(input_path)
    valid = cells["valid_cell"].fillna(False).astype(bool).to_numpy()
    rows = pd.to_numeric(cells["row"], errors="coerce").to_numpy(float)
    cols = pd.to_numeric(cells["col"], errors="coerce").to_numpy(float)
    exposures = list(config["exposures"])
    summary_rows = []
    seed_frames = []
    posterior_frames = []
    candidate_frames = []

    source_tables = {
        "kde": pd.read_csv(source_run / "diagnostics/kde_bandwidth_diagnostics.csv"),
        "gmm": pd.read_csv(source_run / "diagnostics/gaussian_mixture_diagnostics.csv"),
        "cdf": pd.read_csv(source_run / "diagnostics/empirical_cdf_breakpoints.csv"),
        "otsu": pd.read_csv(source_run / "diagnostics/otsu_comparison.csv"),
        "kde_bootstrap": pd.read_csv(source_run / "diagnostics/spatial_kde_bootstrap.csv"),
        "cdf_bootstrap": pd.read_csv(source_run / "diagnostics/spatial_cdf_bootstrap.csv"),
    }
    seed_list = tuple(int(seed) for seed in config["gmm_stability_seeds"])
    for exposure in exposures:
        values = pd.to_numeric(cells[exposure], errors="coerce").where(valid).to_numpy(float)
        summary_rows.append({"exposure": exposure, "analysis_scope": "valid_cell=True; zero and NoData separated", **summarize_values(values)})
        seeds = fit_gmm_seed_sensitivity(
            values, exposure=exposure, seeds=seed_list,
            sample_size=int(config["gmm_sample_size"]), rows=rows, cols=cols,
        )
        seed_frames.append(seeds)
        assignment_path = source_run / f"diagnostics/gmm_posterior_assignments_{exposure}.parquet"
        assignments = pd.read_parquet(assignment_path)
        posterior_frames.append(summarize_posterior_cutoffs(
            assignments, n_valid=int(np.isfinite(values).sum()), cutoffs=config["posterior_cutoffs"],
        ))
        candidate_frames.append(build_threshold_candidate_table(
            exposure=exposure, values=values,
            kde=source_tables["kde"], gmm=source_tables["gmm"], cdf=source_tables["cdf"],
            otsu=source_tables["otsu"], kde_bootstrap=source_tables["kde_bootstrap"],
            cdf_bootstrap=source_tables["cdf_bootstrap"], gmm_seed_trials=seeds,
        ))

    summaries = pd.DataFrame.from_records(summary_rows)
    seed_trials = pd.concat(seed_frames, ignore_index=True) if seed_frames else pd.DataFrame()
    seed_summary = summarize_gmm_seed_sensitivity(seed_trials)
    posterior_summary = pd.concat(posterior_frames, ignore_index=True) if posterior_frames else pd.DataFrame()
    candidates = pd.concat(candidate_frames, ignore_index=True) if candidate_frames else pd.DataFrame(columns=THRESHOLD_COLUMNS)

    (run_dir / "config").mkdir(parents=True)
    for directory in ("diagnostics", "thresholds", "reports"):
        (run_dir / directory).mkdir(parents=True, exist_ok=True)
    snapshot = yaml.safe_dump(config, allow_unicode=True, sort_keys=False)
    (run_dir / "config/config.yaml").write_text(snapshot, encoding="utf-8")
    (run_dir / "config.yaml").write_text(snapshot, encoding="utf-8")
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=project_root,
                                capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        commit = None
    provenance = {
        "phase": "Phase H distribution diagnostic addendum", "mode": "EXPLORATORY_ORACLE",
        "outcome_guided_selection": True, "outcome_guided_selection_used": False,
        "run_id": config["run_id"], "source_run_id": config["source_run_id"],
        "input_run_id": config["input_run_id"], "git_commit_at_run": commit,
        "started_at_utc": datetime.now(UTC).isoformat(), "python": platform.python_version(),
        "health_outcomes_read": False,
        "gmm_stability_seeds": list(seed_list), "gmm_sample_size": int(config["gmm_sample_size"]),
        "posterior_cutoffs": list(config["posterior_cutoffs"]),
        "threshold_unit_note": "Source Aa unit metadata is absent; thresholds remain in original Aa value units.",
        "rusle_policy": "Use existing Aa only; no factor-product reconstruction or validation.",
        "pcnm_policy": "Existing PCNM not rerun.",
    }
    (run_dir / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_table(summaries, run_dir / "diagnostics", "exposure_distribution_detail")
    _write_table(seed_trials, run_dir / "diagnostics", "gmm_seed_stability_by_seed")
    _write_table(seed_summary, run_dir / "diagnostics", "gmm_seed_stability_summary")
    _write_table(posterior_summary, run_dir / "diagnostics", "gmm_posterior_cutoff_sensitivity")
    _write_table(candidates, run_dir / "thresholds", "low_erosion_threshold_candidates")
    from experiments.low_erosion_regime.distribution_addendum_figures import generate_distribution_addendum_figures

    generate_distribution_addendum_figures(run_dir)
    write_distribution_addendum_report(run_dir, config=config)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(run_distribution_addendum(args.project_root, args.config))


if __name__ == "__main__":
    main()
