"""Phase I-2: summarize existing M4 matched-pixel and matched-pair evidence."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import yaml
from scipy.stats import kendalltau, pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[2]
OUTCOMES_DEFAULT = ["SHI_candidate", "AN", "OM", "CEC", "pH", "SW", "BD", "MBC", "RESP"]
POPULATIONS = ["all_erosion", "zero_positive", "both_positive"]
POPULATION_LABELS = {
    "all_erosion": "全侵蚀范围",
    "zero_positive": "零值—正值配对",
    "both_positive": "双正值配对",
}
OUTCOME_LABELS = {
    "SHI_candidate": "SHI",
    "AN": "AN",
    "OM": "OM",
    "CEC": "CEC",
    "pH": "pH",
    "SW": "SW",
    "BD": "BD",
    "MBC": "MBC",
    "RESP": "RESP",
}
plt.rcParams["font.family"] = ["WenQuanYi Zen Hei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


def _git(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except Exception:
        return None


def _version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _numeric(values: pd.Series) -> np.ndarray:
    return pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)


def _population_masks(frame: pd.DataFrame, high: np.ndarray, low: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "all_erosion": np.ones(len(frame), dtype=bool),
        "zero_positive": ((high == 0) ^ (low == 0)),
        "both_positive": (high > 0) & (low > 0),
    }


def _association_row(
    *, exposure: str, candidate_id: str, population: str, outcome: str, analysis_unit: str,
    x: np.ndarray, y: np.ndarray, n_pairs: int | None = None,
) -> dict[str, Any]:
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x[keep], dtype=float), np.asarray(y[keep], dtype=float)
    n = len(x)
    row: dict[str, Any] = {
        "exposure": exposure, "candidate_id": candidate_id, "population": population,
        "outcome": outcome, "analysis_unit": analysis_unit, "n_complete": n,
        "n_pairs": n_pairs,
        "exposure_scale": "delta_log1p_Aa" if analysis_unit == "matched_pair" else "log1p_Aa",
    }
    if n < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        row.update({"spearman_rho": np.nan, "kendall_tau_b": np.nan, "pearson_r": np.nan,
                    "r_squared_ols": np.nan, "slope_y_per_x": np.nan,
                    "standardized_slope_beta": np.nan})
    else:
        pearson = pearsonr(x, y)
        rho = spearmanr(x, y)
        tau = kendalltau(x, y, variant="b", method="asymptotic")
        slope = float(np.mean((x - x.mean()) * (y - y.mean())) / np.var(x))
        row.update({
            "spearman_rho": float(rho.statistic), "kendall_tau_b": float(tau.statistic),
            "pearson_r": float(pearson.statistic), "r_squared_ols": float(pearson.statistic**2),
            "slope_y_per_x": slope,
            "standardized_slope_beta": float(slope * np.std(x) / np.std(y)),
        })
    if analysis_unit == "matched_pair":
        row["mean_delta_health"] = float(np.mean(y)) if n else np.nan
        row["median_delta_health"] = float(np.median(y)) if n else np.nan
        row["negative_pair_fraction"] = float(np.mean(y < 0)) if n else np.nan
        row["positive_pair_fraction"] = float(np.mean(y > 0)) if n else np.nan
        row["zero_pair_fraction"] = float(np.mean(y == 0)) if n else np.nan
    return row


def _load_candidates(config: dict[str, Any]) -> tuple[dict[tuple[str, str], pd.DataFrame], pd.DataFrame]:
    pair_dir = ROOT / config["candidate_pair_dir"]
    candidates: dict[tuple[str, str], pd.DataFrame] = {}
    supports: list[dict[str, Any]] = []
    for exposure, candidate_ids in config["candidate_pairs"].items():
        for candidate_id in candidate_ids:
            path = pair_dir / f"{candidate_id}.parquet"
            if not path.is_file():
                raise FileNotFoundError(path)
            frame = pd.read_parquet(path)
            frame = frame.loc[frame["method"].eq(config["matching_method"])].copy()
            if frame.empty:
                raise ValueError(f"No {config['matching_method']} records in {path}.")
            if "candidate_id" in frame and not frame["candidate_id"].astype(str).eq(candidate_id).all():
                raise ValueError(f"Candidate identifier mismatch in {path}.")
            high, low = _numeric(frame[f"{exposure}_high"]), _numeric(frame[f"{exposure}_low"])
            masks = _population_masks(frame, high, low)
            if np.any((high < low) & np.isfinite(high) & np.isfinite(low)):
                raise ValueError(f"M4 pairs are not oriented from lower to higher {exposure} in {path}.")
            if not frame["pair_id"].is_unique:
                raise ValueError(f"Duplicate pair_id in {path}.")
            candidates[(exposure, candidate_id)] = frame
            for population, mask in masks.items():
                subset = frame.loc[mask]
                endpoints = pd.unique(pd.concat([subset["high_erosion_cell_id"], subset["low_erosion_cell_id"]], ignore_index=True))
                supports.append({
                    "exposure": exposure, "candidate_id": candidate_id, "population": population,
                    "method": config["matching_method"], "n_pairs": int(len(subset)),
                    "n_unique_endpoints": int(len(endpoints)),
                    "median_geographic_distance_m": float(pd.to_numeric(subset["geographic_distance"], errors="coerce").median()),
                    "median_environmental_distance": float(pd.to_numeric(subset["environmental_distance"], errors="coerce").median()),
                    "source_file": str(path.relative_to(ROOT)),
                    "outcome_guided_selection": True,
                })
    return candidates, pd.DataFrame(supports)


def _build_associations(
    candidates: dict[tuple[str, str], pd.DataFrame], config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pair_rows: list[dict[str, Any]] = []
    pixel_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    for (exposure, candidate_id), frame in candidates.items():
        high_exposure, low_exposure = _numeric(frame[f"{exposure}_high"]), _numeric(frame[f"{exposure}_low"])
        delta_exposure = np.log1p(high_exposure) - np.log1p(low_exposure)
        masks = _population_masks(frame, high_exposure, low_exposure)
        for outcome in config["outcomes"]:
            delta_col = f"delta_{outcome}"
            high_col, low_col = f"high_{outcome}", f"low_{outcome}"
            if delta_col not in frame:
                raise KeyError(f"Missing paired outcome column {delta_col} in {candidate_id}.")
            delta_y_all = _numeric(frame[delta_col])
            high_y_all, low_y_all = _numeric(frame[high_col]), _numeric(frame[low_col])
            for population, mask in masks.items():
                pair_subset = frame.loc[mask]
                pair_x = delta_exposure[mask]
                pair_y = delta_y_all[mask]
                pair_rows.append(_association_row(
                    exposure=exposure, candidate_id=candidate_id, population=population,
                    outcome=outcome, analysis_unit="matched_pair", x=pair_x, y=pair_y,
                    n_pairs=int(len(pair_subset)),
                ))

                endpoint_frame = pd.concat([
                    pd.DataFrame({"cell_id": pair_subset["high_erosion_cell_id"].to_numpy(),
                                  "Aa": high_exposure[mask], "health": high_y_all[mask]}),
                    pd.DataFrame({"cell_id": pair_subset["low_erosion_cell_id"].to_numpy(),
                                  "Aa": low_exposure[mask], "health": low_y_all[mask]}),
                ], ignore_index=True)
                endpoint_frame = endpoint_frame.loc[np.isfinite(endpoint_frame["Aa"]) & np.isfinite(endpoint_frame["health"])]
                endpoint_frame = endpoint_frame.drop_duplicates("cell_id", keep="first")
                px = np.log1p(endpoint_frame["Aa"].to_numpy(dtype=float))
                py = endpoint_frame["health"].to_numpy(dtype=float)
                pixel_rows.append(_association_row(
                    exposure=exposure, candidate_id=candidate_id, population=population,
                    outcome=outcome, analysis_unit="matched_pixel", x=px, y=py,
                    n_pairs=int(len(pair_subset)),
                ))

                for unit, x, y in (("matched_pair", pair_x, pair_y), ("matched_pixel", px, py)):
                    keep = np.isfinite(x) & np.isfinite(y)
                    xv, yv = np.asarray(x[keep], dtype=float), np.asarray(y[keep], dtype=float)
                    if not len(xv):
                        continue
                    order = np.argsort(xv, kind="stable")
                    number_of_bins = min(int(config["curve_bins"]), len(order))
                    for bin_index, indices in enumerate(np.array_split(order, number_of_bins), start=1):
                        if not len(indices):
                            continue
                        curve_rows.append({
                            "exposure": exposure, "candidate_id": candidate_id,
                            "population": population, "outcome": outcome,
                            "analysis_unit": unit, "bin": bin_index,
                            "n_in_bin": int(len(indices)),
                            "x_median": float(np.median(xv[indices])),
                            "y_median": float(np.median(yv[indices])),
                            "y_mean": float(np.mean(yv[indices])),
                        })
    return pd.DataFrame(pair_rows), pd.DataFrame(pixel_rows), pd.DataFrame(curve_rows)


def _reuse_model_trials(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trials_path = ROOT / config["model_trials"]
    trials = pd.read_csv(trials_path)
    candidate_ids = {candidate for ids in config["candidate_pairs"].values() for candidate in ids}
    trials = trials.loc[
        trials["reference_method"].eq(config["matching_method"])
        & trials["analysis_population"].eq("phase_h_matched_only")
        & trials["candidate_id"].astype(str).isin(candidate_ids)
        & trials["estimand"].isin(["matched_pixel", "matched_pair"])
        & trials["subset_name"].isin(["all", "zero_positive", "both_positive"])
        & trials["evaluation_set"].eq("apparent")
    ].copy()
    trials["r_squared"] = pd.to_numeric(trials["r_squared"], errors="coerce")
    trials["n"] = pd.to_numeric(trials["n"], errors="coerce")
    good = trials.loc[trials["status"].eq("ok") & trials["r_squared"].notna()].copy()
    group_columns = ["exposure", "subset_name", "estimand", "outcome"]
    best = (good.sort_values(["r_squared", "n"], ascending=[False, False])
            .drop_duplicates(group_columns, keep="first"))
    design_columns = [
        "candidate_id", "exposure", "reference_method", "vector_name", "environment_variables_json",
        "distance_method", "cluster_method", "cluster_count", "environment_caliper",
        "geographic_radius_m", "erosion_rank_gap", "exposure_rule", "n_matched_pairs_candidate",
        "n_matched_pixels_candidate", "outcome_guided_selection", "subset_outcome_guided_selection",
    ]
    designs = trials[design_columns].drop_duplicates("candidate_id", keep="first").sort_values("candidate_id")
    return trials, best, designs


def _global_match_comparison(config: dict[str, Any], pair: pd.DataFrame, pixel: pd.DataFrame) -> pd.DataFrame:
    global_path = ROOT / config["global_reference_table"]
    global_results = pd.read_csv(global_path)
    global_results = global_results.loc[
        global_results["population"].eq("mask_all")
        & global_results["exposure"].eq(config["primary_exposure"]),
        ["outcome", "n_complete", "spearman_rho"],
    ].rename(columns={"n_complete": "global_n", "spearman_rho": "global_spearman_rho"})
    candidate = config["curve_candidates"][config["primary_exposure"]]
    matched_pixel = pixel.loc[
        pixel["exposure"].eq(config["primary_exposure"])
        & pixel["candidate_id"].eq(candidate) & pixel["population"].eq("all_erosion"),
        ["outcome", "n_complete", "spearman_rho"],
    ].rename(columns={"n_complete": "matched_pixel_n", "spearman_rho": "matched_pixel_spearman_rho"})
    matched_pair = pair.loc[
        pair["exposure"].eq(config["primary_exposure"])
        & pair["candidate_id"].eq(candidate) & pair["population"].eq("all_erosion"),
        ["outcome", "n_complete", "spearman_rho"],
    ].rename(columns={"n_complete": "matched_pair_n", "spearman_rho": "matched_pair_spearman_rho"})
    comparison = global_results.merge(matched_pixel, on="outcome", validate="one_to_one").merge(
        matched_pair, on="outcome", validate="one_to_one",
    )
    comparison["matched_pixel_minus_global_rho"] = comparison["matched_pixel_spearman_rho"] - comparison["global_spearman_rho"]
    comparison["candidate_id"] = candidate
    return comparison


def _save_figure(fig: plt.Figure, base: Path, dpi: int) -> list[str]:
    base.parent.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for suffix in ("png", "svg", "pdf"):
        path = base.with_suffix(f".{suffix}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(str(path.relative_to(ROOT)))
    plt.close(fig)
    return paths


def _figures(run_dir: Path, pairs: pd.DataFrame, curves: pd.DataFrame, best: pd.DataFrame,
             config: dict[str, Any]) -> list[str]:
    figure_paths: list[str] = []
    focus = ["SHI_candidate", "OM", "SW"]
    population_order = ["all_erosion", "zero_positive", "both_positive"]
    for exposure, candidate_id in config["curve_candidates"].items():
        for unit in ("matched_pair", "matched_pixel"):
            fig, axes = plt.subplots(3, 3, figsize=(13.2, 10.2), constrained_layout=True)
            subset = curves.loc[
                curves["exposure"].eq(exposure) & curves["candidate_id"].eq(candidate_id)
                & curves["analysis_unit"].eq(unit) & curves["outcome"].isin(focus)
            ]
            for row_idx, population in enumerate(population_order):
                for col_idx, outcome in enumerate(focus):
                    ax = axes[row_idx, col_idx]
                    d = subset.loc[subset["population"].eq(population) & subset["outcome"].eq(outcome)]
                    if len(d):
                        ax.plot(d["x_median"], d["y_median"], color="#277DA1", linewidth=2.1)
                        ax.scatter(d["x_median"], d["y_median"], s=14, color="#277DA1", alpha=0.75)
                    else:
                        ax.text(0.5, 0.5, "该候选无此配对", ha="center", va="center", transform=ax.transAxes)
                    ax.set_title(f"{POPULATION_LABELS[population]} · {OUTCOME_LABELS[outcome]}")
                    ax.grid(axis="y", color="#e5e7eb", linewidth=0.7)
                    if row_idx == 2:
                        ax.set_xlabel("Δlog1p(Aa)" if unit == "matched_pair" else "log1p(Aa)")
                    if col_idx == 0:
                        ax.set_ylabel("配对Δ健康值" if unit == "matched_pair" else "匹配像元健康值")
            title_unit = "配对差值" if unit == "matched_pair" else "匹配端点像元"
            fig.suptitle(f"M4 {title_unit}响应曲线｜{exposure}｜{candidate_id}", fontsize=15, y=1.02)
            fig.text(0.5, -0.015, "点为等样本量分箱中位数；图用于描述，点数不代表独立空间单元。", ha="center", fontsize=9)
            base = run_dir / "figures" / "response_curves" / f"I09_{exposure}_{unit}"
            figure_paths.extend(_save_figure(fig, base, int(config["figure_dpi"])))
            source = subset.loc[subset["outcome"].isin(focus)]
            source_path = base.with_name(base.name + "_data.csv")
            source.to_csv(source_path, index=False)
            figure_paths.append(str(source_path.relative_to(ROOT)))

    r2 = best.loc[best["exposure"].eq(config["primary_exposure"]) & best["subset_name"].eq("all")]
    pivot = r2.pivot(index="outcome", columns="estimand", values="r_squared").reindex(config["outcomes"])
    fig, ax = plt.subplots(figsize=(10.5, 5.7), constrained_layout=True)
    x = np.arange(len(pivot))
    width = 0.36
    for offset, unit, label, color in [(-width / 2, "matched_pixel", "Matched-Pixel", "#277DA1"),
                                       (width / 2, "matched_pair", "Matched-Pair", "#F4A261")]:
        values = pivot[unit].to_numpy(dtype=float) if unit in pivot else np.full(len(pivot), np.nan)
        ax.bar(x + offset, values, width, label=label, color=color)
    ax.set_xticks(x, [OUTCOME_LABELS[o] for o in pivot.index])
    ax.set_ylabel("最高表观 R²（Phase H Oracle 候选）")
    ax.set_title("M4 匹配像元与配对差值模型｜Aa_20yr 全范围")
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.7)
    ax.legend(frameon=False, ncols=2)
    figure_paths.extend(_save_figure(fig, run_dir / "figures" / "model_comparison" / "I10_m4_best_apparent_r2", int(config["figure_dpi"])))
    plot_data = r2[["exposure", "candidate_id", "subset_name", "estimand", "outcome", "model_family", "r_squared", "rmse", "mae", "n"]]
    data_path = run_dir / "figures" / "model_comparison" / "I10_m4_best_apparent_r2_data.csv"
    plot_data.to_csv(data_path, index=False)
    figure_paths.append(str(data_path.relative_to(ROOT)))
    return figure_paths


def _md_table(frame: pd.DataFrame, columns: list[str], labels: dict[str, str] | None = None, limit: int | None = None) -> str:
    d = frame[columns].copy()
    if limit is not None:
        d = d.head(limit)
    if labels:
        d = d.rename(columns=labels)
    return d.to_markdown(index=False, floatfmt=".3f")


def _write_report(run_dir: Path, config: dict[str, Any], support: pd.DataFrame,
                  pair_associations: pd.DataFrame, pixel_associations: pd.DataFrame,
                  model_best: pd.DataFrame, designs: pd.DataFrame,
                  global_comparison: pd.DataFrame, figures: list[str]) -> Path:
    primary = config["primary_exposure"]
    main_candidate = config["curve_candidates"][primary]
    main_pairs = pair_associations.loc[pair_associations["exposure"].eq(primary) & pair_associations["candidate_id"].eq(main_candidate)]
    main_pixels = pixel_associations.loc[pixel_associations["exposure"].eq(primary) & pixel_associations["candidate_id"].eq(main_candidate)]
    all_pairs = main_pairs.loc[main_pairs["population"].eq("all_erosion")].sort_values("spearman_rho")
    all_model = model_best.loc[model_best["exposure"].eq(primary) & model_best["subset_name"].eq("all")]
    figure_links = "\n".join(
        f"- `{path}`" for path in figures if path.endswith((".png", ".svg", ".pdf"))
    )
    support_table = _md_table(
        support, ["exposure", "candidate_id", "population", "n_pairs", "n_unique_endpoints",
                  "median_geographic_distance_m", "median_environmental_distance"],
        {"exposure": "暴露", "candidate_id": "M4候选", "population": "配对总体", "n_pairs": "配对数",
         "n_unique_endpoints": "唯一端点数", "median_geographic_distance_m": "中位地理距离(m)",
         "median_environmental_distance": "中位环境距离"},
    )
    pair_table = _md_table(
        all_pairs, ["outcome", "n_complete", "spearman_rho", "kendall_tau_b", "mean_delta_health",
                    "median_delta_health", "standardized_slope_beta", "negative_pair_fraction", "r_squared_ols"],
        {"outcome": "指标", "n_complete": "配对完整案例", "spearman_rho": "Spearman ρ",
         "kendall_tau_b": "Kendall τ-b", "mean_delta_health": "平均Δ健康", "median_delta_health": "中位Δ健康",
         "standardized_slope_beta": "标准化斜率β", "negative_pair_fraction": "负向配对比例",
         "r_squared_ols": "配对OLS R²"},
    )
    pixel_table = _md_table(
        main_pixels.loc[main_pixels["population"].eq("all_erosion")].sort_values("spearman_rho"),
        ["outcome", "n_complete", "spearman_rho", "kendall_tau_b", "standardized_slope_beta", "r_squared_ols"],
        {"outcome": "指标", "n_complete": "唯一端点完整案例", "spearman_rho": "Spearman ρ",
         "kendall_tau_b": "Kendall τ-b", "standardized_slope_beta": "标准化斜率β", "r_squared_ols": "像元OLS R²"},
    )
    model_table = _md_table(
        all_model.sort_values(["estimand", "outcome"]),
        ["outcome", "estimand", "candidate_id", "model_family", "n", "r_squared", "rmse", "mae"],
        {"outcome": "指标", "estimand": "统计单位", "candidate_id": "候选", "model_family": "模型",
         "n": "n", "r_squared": "表观R²", "rmse": "RMSE", "mae": "MAE"},
    )
    comparison_table = _md_table(
        global_comparison.sort_values("global_spearman_rho"),
        ["outcome", "global_n", "global_spearman_rho", "matched_pixel_n", "matched_pixel_spearman_rho",
         "matched_pixel_minus_global_rho", "matched_pair_n", "matched_pair_spearman_rho"],
        {"outcome": "指标", "global_n": "全样本n", "global_spearman_rho": "全样本ρ",
         "matched_pixel_n": "匹配端点n", "matched_pixel_spearman_rho": "匹配端点ρ",
         "matched_pixel_minus_global_rho": "匹配端点−全样本Δρ", "matched_pair_n": "配对n",
         "matched_pair_spearman_rho": "配对差值ρ"},
    )
    main_negative = all_pairs.iloc[0]
    best_pixel = all_model.loc[all_model["estimand"].eq("matched_pixel")].sort_values("r_squared", ascending=False).iloc[0]
    best_pair = all_model.loc[all_model["estimand"].eq("matched_pair")].sort_values("r_squared", ascending=False).iloc[0]
    xgb = False
    report = f"""# Phase I-2：环境匹配后的条件关联

## 研究范围

本阶段沿用 Phase H 的 M4 配对候选及其模型试验，不重新匹配、不覆盖 Phase F–H。主暴露为 Aa_20yr，Aa_2020 作补充。配对端点按高/低侵蚀方向保留；零值—正值和双正值子总体按所用 Aa 期别分别定义。完整配对表和 H2 模型试验都来自 `EXPLORATORY_ORACLE`，匹配候选与响应模型曾依健康结果择优；因此报告作为 Oracle 条件关联，不作为独立确认。

统计单位分开呈现：Matched-Pixel 使用配对端点 cell_id 去重后的唯一栅格；Matched-Pair 保留每一对端点的高减低差值。像元值来自健康随机森林预测栅格，不能把像元量解释成独立实测样本量。Aa 沿用预计算值，不重算或验证 RUSLE 因子乘积。

## M4 支持量

{support_table}

Aa_2020 的 H2 结果包含三个既有 M4 候选（`stage2_Aa_2020_01/03/06`）。零—正值配对仅在候选 06 中有支持；候选 01/03 全部为双正值配对。因此零—正值结果与候选 06 对应，不能与候选 01/03 当作同一匹配样本的子集。Aa_20yr 使用既有候选 `stage2_Aa_20yr_01`。

## Aa_20yr：主要条件关联

### 全侵蚀范围配对

{pair_table}

该候选中配对 Spearman 最负的是 {OUTCOME_LABELS[str(main_negative['outcome'])]}（ρ={main_negative['spearman_rho']:.3f}，n={int(main_negative['n_complete']):,}）；负向配对比例为 {main_negative['negative_pair_fraction']:.1%}。这描述每对高侵蚀端与低侵蚀端的健康差值同侵蚀差距的排序关系，不是因果效应。均值差、中央値差、标准化斜率与零值/双正值分层详见输出 CSV。

### 唯一匹配端点像元

{pixel_table}

这组相关将每个 M4 候选内的端点按 cell_id 去重后计算。端点依赖同一匹配设计与空间表面，不能按独立像元解释常规显著性。

### 与 I-1 全样本对照

下表在同一 Aa_20yr 暴露下，把 I-1 全掩膜像元相关与当前 M4 唯一端点相关及配对差值相关并列。匹配样本端点和全样本的估计总体不同；差值仅用于描述关系变化，不作为匹配的因果效应。

{comparison_table}

SHI、OM、SW 的全样本与匹配端点 Spearman 接近（例如 OM 约 −0.415 与 −0.416），而配对差值相关较弱；这显示端点层面的空间梯度与每对高低端差值回答不同问题。M4 匹配后不应只用单一配对系数概括全部侵蚀—健康关系。

## 模型表现

下表为 Phase H 保留的全侵蚀范围候选模型中，按表观 R² 汇总的 Aa_20yr 结果。Matched-Pixel 最佳表观 R²为 {OUTCOME_LABELS[str(best_pixel['outcome'])]} / {best_pixel['model_family']}（R²={best_pixel['r_squared']:.3f}）；Matched-Pair 最佳表观 R²为 {OUTCOME_LABELS[str(best_pair['outcome'])]} / {best_pair['model_family']}（R²={best_pair['r_squared']:.3f}）。

{model_table}

零—正与双正值模型试验仅保留 OLS 子组模型；候选覆盖及全部原始模型记录见 `tables/model_trials_reused.csv`。全范围模型家族来自 Phase H 已有试验（OLS、二次项、自然样条、statsmodels GAM、Huber、Random Forest、Extra Trees、HistGradientBoosting 等），不在 Phase I 重跑。XGBoost 未运行：当前环境未安装 `xgboost`，本阶段未新增依赖。

所有上述 R²是同样本表观拟合，不表示空间外推。空间分块验证应读取 Phase H `phase_h_spatial_validation_summary.csv` 与 Phase I-1 的全样本空间留出结果；后续 Phase I-4 将开展配对差值空间模型。

## 配对响应曲线与模型比较图

曲线图展示选定 M4 候选中每个总体的等样本量分箱中位数；源数据保存在 `figures/response_curves/`。Aa_2020 图使用有零—正值支持的候选 06，环境向量与候选 01/03 不同，图中明确标注。

{figure_links}

## Oracle 候选定义与复现

|暴露|M4候选|环境变量|距离方法|地理卡尺(m)|环境卡尺|侵蚀秩差|
|---|---|---|---|---:|---:|---:|
""" + "\n".join(
        f"|{r.exposure}|{r.candidate_id}|{r.environment_variables_json}|{r.distance_method}|{r.geographic_radius_m}|{r.environment_caliper}|{r.erosion_rank_gap}|"
        for r in designs.itertuples(index=False)
    ) + f"""

结果选择沿用 Phase H 的 Oracle 试验登记；所有已保存试验及筛选字段均复制到本运行目录。报告不把匹配后的关联解释成处理因果效应，且不改变正式研究设计或既有 PCNM 结果。

## 文件索引

- 配对关联：`tables/matched_pair_associations.csv`
- 唯一端点关联：`tables/matched_pixel_associations.csv`
- M4 支持量：`tables/match_population_support.csv`
- 全样本—匹配对照：`tables/global_vs_matched_associations.csv`
- 最佳模型表：`leaderboards/matched_model_leaderboard.csv`
- Phase H 原始模型试验副本：`tables/model_trials_reused.csv`
- 候选匹配规则：`tables/m4_candidate_designs.csv`
- 分箱响应曲线：`results/matching/response_curve_points.csv`
- 运行配置与来源：`configs/phase_i_matching.yaml`、`manifests/provenance.json`
"""
    path = run_dir / "reports" / "PHASE_I_ENVIRONMENT_MATCHING_REPORT.md"
    path.write_text(report, encoding="utf-8")
    return path


def run(config_path: Path) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("mode") != "EXPLORATORY_ORACLE" or config.get("outcome_guided_selection") is not True:
        raise ValueError("Phase I-2 requires explicit EXPLORATORY_ORACLE authorization in its config.")
    run_dir = ROOT / config["output_root"] / config["run_id"]
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing Phase I-2 run: {run_dir}")
    for name in ("configs", "tables", "leaderboards", "results/matching", "figures/response_curves",
                 "figures/model_comparison", "reports", "manifests"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, run_dir / "configs" / config_path.name)

    candidates, support = _load_candidates(config)
    pair_associations, pixel_associations, curves = _build_associations(candidates, config)
    model_trials, model_best, designs = _reuse_model_trials(config)
    global_comparison = _global_match_comparison(config, pair_associations, pixel_associations)

    pair_associations.to_csv(run_dir / "tables/matched_pair_associations.csv", index=False)
    pixel_associations.to_csv(run_dir / "tables/matched_pixel_associations.csv", index=False)
    support.to_csv(run_dir / "tables/match_population_support.csv", index=False)
    model_trials.to_csv(run_dir / "tables/model_trials_reused.csv", index=False)
    model_best.to_csv(run_dir / "leaderboards/matched_model_leaderboard.csv", index=False)
    designs.to_csv(run_dir / "tables/m4_candidate_designs.csv", index=False)
    global_comparison.to_csv(run_dir / "tables/global_vs_matched_associations.csv", index=False)
    curves.to_csv(run_dir / "results/matching/response_curve_points.csv", index=False)
    figures = _figures(run_dir, pair_associations, curves, model_best, config)
    report = _write_report(run_dir, config, support, pair_associations, pixel_associations,
                           model_best, designs, global_comparison, figures)

    provenance = {
        "phase": "I-2", "mode": config["mode"], "outcome_guided_selection": True,
        "run_id": config["run_id"], "input_run_id": config["input_run_id"],
        "phase_h_run_id": config["phase_h_run_id"], "matching_method": config["matching_method"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_at_execution": _git(["git", "rev-parse", "HEAD"]),
        "git_worktree_status_at_execution": _git(["git", "status", "--short"]),
        "runner": "experiments/oracle_exploration/phase_i_matching.py",
        "config": str((run_dir / "configs" / config_path.name).relative_to(ROOT)),
        "source_candidate_pair_dir": config["candidate_pair_dir"],
        "source_model_trials": config["model_trials"],
        "global_reference_table": config["global_reference_table"],
        "candidate_ids": config["candidate_pairs"],
        "curve_candidates": config["curve_candidates"],
        "population_definitions": {
            "all_erosion": "all saved M4 candidate pairs",
            "zero_positive": "exactly one endpoint has Aa=0 and the other Aa>0",
            "both_positive": "both endpoints have Aa>0",
        },
        "analysis_units": {
            "matched_pair": "oriented high-minus-low outcome differences; one row per pair",
            "matched_pixel": "unique endpoints deduplicated by cell_id within each candidate and population",
        },
        "model_search": "Phase H apparent model trials reused; no models refitted; subgroup trial coverage is OLS only",
        "xgboost_available": False,
        "health_values": "random-forest prediction rasters; not independent measured samples",
        "software_versions": {
            "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "matplotlib": _version("matplotlib"),
            "scikit_learn": _version("scikit-learn"), "pyarrow": _version("pyarrow"), "pyyaml": _version("PyYAML"),
        },
    }
    (run_dir / "manifests/provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    files = [str(p.relative_to(ROOT)) for p in sorted(run_dir.rglob("*")) if p.is_file() and p.name != "output_manifest.json"]
    (run_dir / "manifests/output_manifest.json").write_text(
        json.dumps({"run_id": config["run_id"], "files": files}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"run_dir={run_dir.relative_to(ROOT)}")
    print(f"report={report.relative_to(ROOT)}")
    print(f"pair_associations={len(pair_associations)} pixel_associations={len(pixel_associations)} model_trials_reused={len(model_trials)}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "phase_i_matching.yaml")
    args = parser.parse_args()
    run(args.config.resolve())


if __name__ == "__main__":
    main()
