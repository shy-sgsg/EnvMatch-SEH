"""Phase I-3: consolidate previously completed Phase G/H environment heterogeneity."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
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

ROOT = Path(__file__).resolve().parents[2]
KEY_OUTCOMES = ["SHI_candidate", "OM", "SW"]
OUTCOME_LABELS = {"SHI_candidate": "SHI", "OM": "OM", "SW": "SW"}
GROUP_LABELS = {
    "soil_type_group": "土壤类型",
    "kmeans_v5": "Phase F V5 k=6",
    "hierarchical_ward_v5": "土类内 Ward k=6",
    "gradient_Elevation": "Elevation 五分位",
    "gradient_TWI": "TWI 五分位",
    "gradient_LST": "LST 五分位",
    "gradient_NDWI": "NDWI 五分位",
    "gradient_NDVI": "NDVI 五分位",
    "gradient_Precipitation": "降水五分位",
    "kmeans_terrain_k4": "土类内 K-means k=4",
    "kmeans_terrain_k6": "土类内 K-means k=6",
    "kmeans_terrain_k8": "土类内 K-means k=8",
    "kmeans_terrain_k10": "土类内 K-means k=10",
    "kmeans_terrain_k12": "土类内 K-means k=12",
}
SCHEME_LABELS = {
    "S0_all": "原始全配对",
    "S1_both_positive": "双端正值",
    "T01_fixed_original": "CDF阈值后保留原配对",
    "T01_empirical_CDF_breakpoint": "经验CDF折点",
    "T01_refit_k6_filtered_population": "过滤后重拟合k=6",
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


def _read_sources(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    phase_g = ROOT / config["phase_g_run"] / "tables"
    phase_h = ROOT / config["phase_h_heterogeneity_run"] / "tables"
    slopes = pd.read_csv(phase_g / "group_slope_estimates.csv")
    interactions = pd.read_csv(phase_g / "group_slope_interaction_tests.csv")
    profiles = pd.read_csv(phase_g / "environment_group_profiles.csv")
    paired_models = pd.read_csv(phase_g / "paired_model_leaderboard.csv")
    h_response = pd.read_csv(phase_h / "group_response_comparison.csv")
    h_support = pd.read_csv(phase_h / "group_population_support.csv")
    h_bootstrap = pd.read_csv(phase_h / "group_pair_slope_spatial_bootstrap.csv")
    h_spatial = pd.read_csv(phase_h / "group_spatial_summary.csv")
    valid_methods = {"M4"}
    exposures = {config["primary_exposure"], config["supplementary_exposure"]}
    slopes = slopes.loc[
        slopes["analysis_unit"].eq("matched_pair_difference")
        & slopes["method"].isin(valid_methods) & slopes["exposure"].isin(exposures)
    ].copy()
    interactions = interactions.loc[
        interactions["analysis_unit"].eq("matched_pair_difference")
        & interactions["method"].isin(valid_methods) & interactions["exposure"].isin(exposures)
    ].copy()
    profiles = profiles.loc[profiles["method"].isin(valid_methods) & profiles["exposure"].isin(exposures)].copy()
    paired_models = paired_models.loc[
        paired_models["analysis_unit"].eq("matched_pair_difference")
        & paired_models["method"].isin(valid_methods) & paired_models["exposure"].isin(exposures)
    ].copy()
    h_response = h_response.loc[
        h_response["method"].isin(valid_methods) & h_response["exposure"].isin(exposures)
        & h_response["scheme"].isin(SCHEME_LABELS)
    ].copy()
    h_support = h_support.loc[
        h_support["method"].isin(valid_methods) & h_support["exposure"].isin(exposures)
        & h_support["scheme"].isin(SCHEME_LABELS)
    ].copy()
    h_bootstrap = h_bootstrap.loc[
        h_bootstrap["method"].isin(valid_methods) & h_bootstrap["exposure"].isin(exposures)
        & h_bootstrap["scheme"].isin(SCHEME_LABELS)
    ].copy()
    h_spatial = h_spatial.loc[
        h_spatial["method"].isin(valid_methods) & h_spatial["exposure"].isin(exposures)
        & h_spatial["scheme"].isin(SCHEME_LABELS)
    ].copy()
    return {
        "slopes": slopes, "interactions": interactions, "profiles": profiles,
        "paired_models": paired_models, "h_response": h_response, "h_support": h_support,
        "h_bootstrap": h_bootstrap, "h_spatial": h_spatial,
    }


def _supported_best_groups(slopes: pd.DataFrame, interactions: pd.DataFrame, min_blocks: int) -> pd.DataFrame:
    merged = slopes.merge(
        interactions[["method", "exposure", "outcome", "group_column", "interaction_p_value",
                     "interaction_log_p_value", "interaction_q_value_global", "interaction_log_q_value_global"]],
        on=["method", "exposure", "outcome", "group_column"], how="left", validate="many_to_one",
    )
    merged = merged.loc[
        merged["status"].eq("ok") & merged["spatial_blocks"].ge(min_blocks)
        & merged["standardized_effect"].notna()
    ].copy()
    merged["group_column_label"] = merged["group_column"].map(GROUP_LABELS)
    merged["outcome_label"] = merged["outcome"].map(OUTCOME_LABELS).fillna(merged["outcome"])
    return (merged.sort_values("standardized_effect")
            .drop_duplicates(["exposure", "outcome", "group_column"], keep="first"))


def _save_fig(fig: plt.Figure, base: Path, dpi: int) -> list[str]:
    base.parent.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for suffix in ("png", "svg", "pdf"):
        path = base.with_suffix(f".{suffix}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(str(path.relative_to(ROOT)))
    plt.close(fig)
    return paths


def _figures(run_dir: Path, slopes: pd.DataFrame, interactions: pd.DataFrame,
             h_response: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    exposure = config["primary_exposure"]
    key_slopes = slopes.loc[
        slopes["exposure"].eq(exposure) & slopes["group_column"].eq("gradient_TWI")
        & slopes["outcome"].isin(KEY_OUTCOMES) & slopes["spatial_blocks"].ge(5)
    ].copy()
    colors = {"26": "#277DA1", "27": "#E07A5F", "42": "#70A288"}
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 7.5), constrained_layout=True, sharey=True)
    order = [f"{soil}:q{q}" for soil in (26, 27, 42) for q in range(1, 6)]
    for ax, outcome in zip(axes, KEY_OUTCOMES, strict=True):
        d = key_slopes.loc[key_slopes["outcome"].eq(outcome)].copy()
        d["group_order"] = d["group_label"].map({name: i for i, name in enumerate(order)})
        d = d.sort_values("group_order", ascending=False)
        for row in d.itertuples(index=False):
            soil = str(row.group_label).split(":")[0]
            y = order.index(str(row.group_label))
            ax.errorbar(row.slope_delta_y_per_delta_log1p_erosion, y,
                        xerr=[[row.slope_delta_y_per_delta_log1p_erosion - row.slope_ci_low],
                              [row.slope_ci_high - row.slope_delta_y_per_delta_log1p_erosion]],
                        fmt="o", color=colors.get(soil, "#555555"), capsize=2, markersize=4.5)
        ax.axvline(0, color="#333333", linewidth=0.9)
        ax.set_title(OUTCOME_LABELS[outcome])
        ax.grid(axis="x", color="#e5e7eb", linewidth=0.7)
        ax.set_xlabel("配对差值斜率 / Δlog1p(Aa)")
    axes[0].set_yticks(range(len(order)), order)
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[s], label=f"土类 {s}", markersize=8)
               for s in ("26", "27", "42")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.04), ncols=3, frameon=False)
    fig.suptitle("M4 Aa_20yr：TWI 土类内五分位的配对响应斜率", fontsize=15, y=1.10)
    fig.text(0.5, -0.015, "点为5 km聚类稳健斜率，线为95%区间；分组按土类内TWI五分位定义。", ha="center", fontsize=9)
    paths.extend(_save_fig(fig, run_dir / "figures/environment_heterogeneity/I11_m4_twi_group_slopes", int(config["figure_dpi"])))
    key_slopes.to_csv(run_dir / "figures/environment_heterogeneity/I11_m4_twi_group_slopes_data.csv", index=False)
    paths.append(str((run_dir / "figures/environment_heterogeneity/I11_m4_twi_group_slopes_data.csv").relative_to(ROOT)))

    order_groups = list(GROUP_LABELS)
    d = interactions.loc[interactions["exposure"].eq(exposure)].copy()
    d["negative_log10_q"] = -pd.to_numeric(d["interaction_log_q_value_global"], errors="coerce") / np.log(10)
    d["negative_log10_q"] = d["negative_log10_q"].clip(lower=0, upper=12)
    matrix = (d.pivot(index="group_column", columns="outcome", values="negative_log10_q")
              .reindex(index=order_groups, columns=config["outcomes"]))
    fig, ax = plt.subplots(figsize=(12, 7.4), constrained_layout=True)
    im = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", cmap="YlOrRd", vmin=0, vmax=12)
    ax.set_xticks(range(len(config["outcomes"])), [OUTCOME_LABELS.get(o, o) for o in config["outcomes"]])
    ax.set_yticks(range(len(order_groups)), [GROUP_LABELS[g] for g in order_groups])
    ax.set_title("M4 Aa_20yr：环境分组斜率交互的全局 q 值")
    ax.set_xlabel("健康结局")
    ax.set_ylabel("环境分组定义")
    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label("−log₁₀(q)，显示上限 12")
    for yi in range(matrix.shape[0]):
        for xi in range(matrix.shape[1]):
            val = matrix.iloc[yi, xi]
            if pd.notna(val):
                ax.text(xi, yi, f"{int(np.rint(max(float(val), 0.0)))}", ha="center", va="center", fontsize=7,
                        color="white" if val >= 8 else "#222222")
    paths.extend(_save_fig(fig, run_dir / "figures/environment_heterogeneity/I12_m4_interaction_q_heatmap", int(config["figure_dpi"])))
    heatmap_data = d[["exposure", "outcome", "group_column", "n_pairs", "n_groups", "interaction_p_value",
                      "interaction_log_p_value", "interaction_q_value_global", "interaction_log_q_value_global", "negative_log10_q"]]
    heatmap_path = run_dir / "figures/environment_heterogeneity/I12_m4_interaction_q_heatmap_data.csv"
    heatmap_data.to_csv(heatmap_path, index=False)
    paths.append(str(heatmap_path.relative_to(ROOT)))

    low = h_response.loc[
        h_response["exposure"].eq(exposure) & h_response["analysis_unit"].eq("pair")
        & h_response["group_column"].eq("gradient_TWI") & h_response["group_label"].eq("26:q1")
        & h_response["outcome"].isin(KEY_OUTCOMES)
    ].copy()
    scheme_order = ["S0_all", "T01_fixed_original", "T01_empirical_CDF_breakpoint", "T01_refit_k6_filtered_population"]
    low = low.loc[low["scheme"].isin(scheme_order)]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.3), constrained_layout=True, sharey=True)
    for ax, outcome in zip(axes, KEY_OUTCOMES, strict=True):
        z = low.loc[low["outcome"].eq(outcome)].copy()
        z["order"] = z["scheme"].map({v: i for i, v in enumerate(scheme_order)})
        z = z.sort_values("order")
        ax.plot(z["spearman_rho"], z["order"], color="#277DA1", marker="o", linewidth=1.8)
        for row in z.itertuples(index=False):
            ax.annotate(f"n={int(row.n_pairs):,}", (row.spearman_rho, row.order), xytext=(5, 3),
                        textcoords="offset points", fontsize=7)
        ax.axvline(0, color="#333333", linewidth=0.8)
        ax.set_title(OUTCOME_LABELS[outcome])
        ax.grid(axis="x", color="#e5e7eb", linewidth=0.7)
        ax.set_xlabel("配对差值 Spearman ρ")
    axes[0].set_yticks(range(len(scheme_order)), [SCHEME_LABELS[s] for s in scheme_order])
    fig.suptitle("M4 Aa_20yr：26 冲积土低 TWI 候选在不同筛选总体下的关联", fontsize=14, y=1.04)
    fig.text(0.5, -0.01, "各方案的配对总体与匹配支持量不同，ρ变化不作为模型改进或因果效应。", ha="center", fontsize=9)
    paths.extend(_save_fig(fig, run_dir / "figures/environment_heterogeneity/I13_m4_soil26_twi_q1_sensitivity", int(config["figure_dpi"])))
    low_source = run_dir / "figures/environment_heterogeneity/I13_m4_soil26_twi_q1_sensitivity_data.csv"
    low.to_csv(low_source, index=False)
    paths.append(str(low_source.relative_to(ROOT)))
    return paths


def _md_table(frame: pd.DataFrame, columns: list[str], labels: dict[str, str] | None = None, limit: int | None = None) -> str:
    d = frame[columns].copy()
    if limit is not None:
        d = d.head(limit)
    if labels:
        d = d.rename(columns=labels)
    return d.to_markdown(index=False, floatfmt=".3f")


def _write_report(run_dir: Path, config: dict[str, Any], data: dict[str, pd.DataFrame], best: pd.DataFrame,
                  figures: list[str]) -> Path:
    exp = config["primary_exposure"]
    slopes = data["slopes"].loc[data["slopes"]["exposure"].eq(exp)].merge(
        data["interactions"][["method", "exposure", "outcome", "group_column", "interaction_q_value_global",
                              "interaction_log_q_value_global"]],
        on=["method", "exposure", "outcome", "group_column"], how="left", validate="many_to_one",
    )
    focus = slopes.loc[slopes["outcome"].isin(KEY_OUTCOMES)].copy()
    group_column_order = ["soil_type_group", "gradient_TWI", "gradient_NDVI", "gradient_NDWI", "gradient_LST",
                          "kmeans_terrain_k4", "kmeans_terrain_k6", "kmeans_terrain_k8"]
    winners = best.loc[best["exposure"].eq(exp) & best["group_column"].isin(group_column_order)
                       & best["outcome"].isin(KEY_OUTCOMES)].copy()
    winner_table = _md_table(
        winners.sort_values(["group_column", "outcome"]),
        ["outcome_label", "group_column_label", "group_label", "n_pairs", "spatial_blocks",
         "standardized_effect", "slope_ci_low", "slope_ci_high", "interaction_q_value_global"],
        {"outcome_label": "指标", "group_column_label": "分组定义", "group_label": "最负向组",
         "n_pairs": "配对数", "spatial_blocks": "5km块数", "standardized_effect": "标准化效应",
         "slope_ci_low": "斜率CI下限", "slope_ci_high": "斜率CI上限", "interaction_q_value_global": "全局q"},
    )
    soil = focus.loc[focus["group_column"].eq("soil_type_group")].sort_values(["outcome", "group_label"])
    soil_table = _md_table(
        soil, ["outcome", "group_label", "n_pairs", "spatial_blocks", "standardized_effect",
               "slope_delta_y_per_delta_log1p_erosion", "slope_ci_low", "slope_ci_high"],
        {"outcome": "指标", "group_label": "土类", "n_pairs": "配对数", "spatial_blocks": "5km块数",
         "standardized_effect": "标准化效应", "slope_delta_y_per_delta_log1p_erosion": "原始斜率",
         "slope_ci_low": "CI下限", "slope_ci_high": "CI上限"},
    )
    twi = focus.loc[focus["group_column"].eq("gradient_TWI") & focus["group_label"].isin(["27:q2", "26:q1", "42:q1"])]
    twi_table = _md_table(
        twi.sort_values(["outcome", "group_label"]),
        ["outcome", "group_label", "n_pairs", "spatial_blocks", "standardized_effect",
         "slope_delta_y_per_delta_log1p_erosion", "slope_ci_low", "slope_ci_high"],
        {"outcome": "指标", "group_label": "土类内TWI组", "n_pairs": "配对数", "spatial_blocks": "5km块数",
         "standardized_effect": "标准化效应", "slope_delta_y_per_delta_log1p_erosion": "原始斜率",
         "slope_ci_low": "CI下限", "slope_ci_high": "CI上限"},
    )
    interaction = data["interactions"].loc[data["interactions"]["exposure"].eq(exp)]
    q_count = int(interaction["interaction_q_value_global"].lt(0.05).sum())
    q_total = len(interaction)
    overall_interactions = pd.read_csv(ROOT / config["phase_g_run"] / "tables/group_slope_interaction_tests.csv")
    overall_interactions = overall_interactions.loc[overall_interactions["analysis_unit"].eq("matched_pair_difference")]
    all_q_count = int(overall_interactions["interaction_q_value_global"].lt(0.05).sum())
    all_q_total = len(overall_interactions)
    low_pair = data["h_response"].loc[
        data["h_response"]["exposure"].eq(exp) & data["h_response"]["method"].eq("M4")
        & data["h_response"]["analysis_unit"].eq("pair")
        & data["h_response"]["group_column"].eq("gradient_TWI")
        & data["h_response"]["group_label"].eq("26:q1")
        & data["h_response"]["outcome"].isin(KEY_OUTCOMES)
    ].copy()
    low_pair = low_pair.loc[low_pair["scheme"].isin(SCHEME_LABELS)]
    low_table = _md_table(
        low_pair, ["scheme", "outcome", "n_pairs", "spearman_rho", "slope", "mean_delta_health",
                   "negative_delta_fraction"],
        {"scheme": "Phase H方案", "outcome": "指标", "n_pairs": "配对数", "spearman_rho": "Spearman ρ",
         "slope": "斜率", "mean_delta_health": "平均Δ健康", "negative_delta_fraction": "负向比例"},
    )
    figures_md = "\n".join(f"- `{p}`" for p in figures if p.endswith((".png", ".svg", ".pdf")))
    report = f"""# Phase I-3：环境异质性与敏感环境候选

## 数据和复用范围

本阶段整合 Phase G 全配对 M4 的环境组斜率/交互表，以及 Phase H 低值筛选后匹配环境组结果；不重做匹配、不重跑聚类，也不改既有分组。主分析 Aa_20yr，Aa_2020 作为附表敏感情景。Phase G 的组内斜率为配对差值模型，按 5 km 空间块聚类计算区间；Phase H 表还按零/正值筛选重设了匹配支持，不能与全配对基线合并。

分组定义沿用已有输出：土类标签使用高侵蚀端土类；Phase F V5 与土类内 Ward 使用 Elevation/LST 相关环境聚类；Elevation、TWI、LST、NDWI、NDVI、降水在各土类内分五分位；另有土类内环境聚类 K=4/6/8/10/12。原分组标签以环境变量建立，健康结果用于后续 Oracle 关联与候选筛选。R、K、LS、C、P 和 Aa 不进入环境距离向量。

## 土类及连续环境差异：M4 Aa_20yr

### 土壤类型

在三个主要土类中，27 潮土的 SHI、OM、SW 标准化负斜率绝对值均最大，支持量也最大。分组样本是匹配对，不表示独立实测点。

{soil_table}

### TWI 梯度

TWI 在每个土类内分五分位。全配对 M4 中，27:q2（潮土内第二个 TWI 五分位）在 SHI、OM、SW 三项的标准化效应最负；这不等价于所有土类中的低 TWI 一律更敏感。

{twi_table}

### NDVI、NDWI 与 LST

在同一全配对 M4 Aa_20yr 表中，SHI/OM/SW 最负向的 NDVI 组集中于 27:q5，NDWI 组集中于 27:q1（OM、SW）和 26:q5（SHI），LST 组集中于 27:q3。不同指标的组别不完全一致；具体估计、支持量和全部其他组均保存在 `tables/environment_group_slopes_m4.csv` 及 `leaderboards/best_supported_environment_group.csv`。

## 联合环境组和斜率异质性

在 Aa_20yr 的 126 项 M4 组间交互检验中，{q_count}/{q_total} 项全局 BH q<0.05；跨两期、M3/M4 的 504 项检验中 {all_q_count}/{all_q_total} 项 q<0.05。联合 K-means 分组对重点指标的负斜率候选多数位于 27 潮土的 Elevation/LST 聚类；例如 K=6 的 cluster 2 在 SHI、OM、SW 中均较负，K=8 的 cluster 5 在 OM、SW 较负。热图对 `−log10(q)` 显示上限 12，精确 log(q) 和估计支持保留在 CSV。

交互检验和组内结果来自同一探索流程，q 值校正范围为已登记的 504 项，但不能覆盖历史上所有 Oracle 筛选和空间依赖的不确定性。将其视作异质性线索，不直接解释成预注册确认。

## 26 冲积土低 TWI 候选：Phase H 低值筛选对照

Phase H 在多种分组、指标和筛选方案中识别出 26:q1（冲积土内最低 TWI 五分位）作为值得后续复核的 Oracle 候选。以下为配对差值单位结果；不同方案的配对总体、样本支持和匹配设计并不相同。

{low_table}

在 `T01_fixed_original` 下，SHI/OM/SW 的配对 Spearman 分别约 −0.166/−0.245/−0.253，候选组各有 601 对（SHI 完整案例 593 对）；S0 全配对中分别约 −0.047/−0.022/−0.212。按配对中点限制为同 5 km 空间块后，空间 bootstrap 分别保留 540/547/547 对、16 个空间块；SHI、OM、SW 斜率的 95% 区间分别为 [−0.0276, −0.0021]、[−4.119, −1.499]、[−6.198, −1.951]。该增强不是全指标一致，也不是独立验证；不同筛选方案改变配对总体和匹配支持量。

## 图件

下图分别展示全配对 M4 TWI 分组原始斜率与区间、跨分组的交互 q 值，以及 26:q1 候选在不同 Phase H 样本定义中的 Spearman 变化。

{figures_md}

## 解释边界

- 所有分组关联均来自健康预测栅格和匹配配对，不解释为独立实测效应或因果效应。
- 分组样本量、空间块数和匹配支持差异明显。局部较强的效应不能仅凭 R²或相关系数判为最优环境。
- `best_supported_environment_group.csv` 只在至少 5 个 5 km 空间块的既有组内估计中按标准化效应做探索性排序；所有条目仍是 Oracle 结果。
- 26:q1 是阈值筛选及多分组搜索后出现的候选，需与全配对结果、其他分组、空间块敏感性共同解读，不能直接称为确定的高风险环境。

## 文件索引

- 全配对组内斜率：`tables/environment_group_slopes_m4.csv`
- 交互检验：`tables/environment_interactions_m4.csv`
- 组环境剖面：`tables/environment_group_profiles_m4.csv`
- 配对模型排行榜：`tables/environment_group_models_m4.csv`
- Phase H 分组响应与支持：`tables/low_value_group_response_m4.csv`、`tables/low_value_group_support_m4.csv`
- Phase H 5 km bootstrap/空间表现：`tables/low_value_group_spatial_bootstrap_m4.csv`、`tables/low_value_group_spatial_summary_m4.csv`
- 分组定义：`tables/environment_group_definitions.json`
- 最负向、且至少 5 个空间块的分组摘要：`leaderboards/best_supported_environment_group.csv`
- 来源与筛选记录：`manifests/provenance.json`
"""
    path = run_dir / "reports" / "PHASE_I_ENVIRONMENT_HETEROGENEITY_REPORT.md"
    path.write_text(report, encoding="utf-8")
    return path


def run(config_path: Path) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("mode") != "EXPLORATORY_ORACLE" or config.get("outcome_guided_selection") is not True:
        raise ValueError("Phase I-3 requires explicit EXPLORATORY_ORACLE authorization in its config.")
    run_dir = ROOT / config["output_root"] / config["run_id"]
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing Phase I-3 run: {run_dir}")
    for name in ("configs", "tables", "leaderboards", "figures/environment_heterogeneity", "reports", "manifests"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, run_dir / "configs" / config_path.name)
    data = _read_sources(config)
    best = _supported_best_groups(data["slopes"], data["interactions"], int(config["spatial_block_min_for_supported_ranking"]))
    data["slopes"].to_csv(run_dir / "tables/environment_group_slopes_m4.csv", index=False)
    data["interactions"].to_csv(run_dir / "tables/environment_interactions_m4.csv", index=False)
    data["profiles"].to_csv(run_dir / "tables/environment_group_profiles_m4.csv", index=False)
    data["paired_models"].to_csv(run_dir / "tables/environment_group_models_m4.csv", index=False)
    data["h_response"].to_csv(run_dir / "tables/low_value_group_response_m4.csv", index=False)
    data["h_support"].to_csv(run_dir / "tables/low_value_group_support_m4.csv", index=False)
    data["h_bootstrap"].to_csv(run_dir / "tables/low_value_group_spatial_bootstrap_m4.csv", index=False)
    data["h_spatial"].to_csv(run_dir / "tables/low_value_group_spatial_summary_m4.csv", index=False)
    best.to_csv(run_dir / "leaderboards/best_supported_environment_group.csv", index=False)
    definitions = json.loads((ROOT / config["phase_g_run"] / "tables/environment_group_definitions.json").read_text(encoding="utf-8"))
    (run_dir / "tables/environment_group_definitions.json").write_text(json.dumps(definitions, ensure_ascii=False, indent=2), encoding="utf-8")
    figure_paths = _figures(run_dir, data["slopes"], data["interactions"], data["h_response"], config)
    report = _write_report(run_dir, config, data, best, figure_paths)
    provenance = {
        "phase": "I-3", "mode": config["mode"], "outcome_guided_selection": True,
        "run_id": config["run_id"], "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_at_execution": _git(["git", "rev-parse", "HEAD"]),
        "git_worktree_status_at_execution": _git(["git", "status", "--short"]),
        "runner": "experiments/oracle_exploration/phase_i_heterogeneity.py",
        "config": str((run_dir / "configs" / config_path.name).relative_to(ROOT)),
        "source_runs": {"phase_g": config["phase_g_run"], "phase_h_heterogeneity": config["phase_h_heterogeneity_run"]},
        "new_fits": False,
        "reused_outputs": ["Phase G group slopes/interactions/profiles/model leaderboard", "Phase H filtered group response/support/5km sensitivity"],
        "group_labels_health_blind": "Source group definitions use soil/environment variables; health outcomes guide Oracle association comparisons and candidate ranking.",
        "supported_group_ranking_min_spatial_blocks": int(config["spatial_block_min_for_supported_ranking"]),
        "health_values": "random-forest prediction rasters; not independent measured samples",
        "software_versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                              "scipy": scipy.__version__, "matplotlib": _version("matplotlib"), "pyyaml": _version("PyYAML")},
    }
    (run_dir / "manifests/provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    files = [str(path.relative_to(ROOT)) for path in sorted(run_dir.rglob("*")) if path.is_file() and path.name != "output_manifest.json"]
    (run_dir / "manifests/output_manifest.json").write_text(json.dumps({"run_id": config["run_id"], "files": files}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"run_dir={run_dir.relative_to(ROOT)}")
    print(f"report={report.relative_to(ROOT)}")
    print(f"phase_g_slopes={len(data['slopes'])} interactions={len(data['interactions'])} phase_h_comparisons={len(data['h_response'])}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "phase_i_heterogeneity.yaml")
    args = parser.parse_args()
    run(args.config.resolve())


if __name__ == "__main__":
    main()
