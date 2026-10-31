"""Phase I-7 synthesis across health dimensions, PLS-SEM, and existing PCNM."""

from __future__ import annotations

import argparse
import json
import platform
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
import yaml

ROOT = Path(__file__).resolve().parents[2]
OUTCOMES = ["SHI_candidate", "SW", "BD", "OM", "AN", "CEC", "pH", "MBC", "RESP"]
DIMENSIONS = {"SHI_candidate": "SHI综合", "SW": "物理", "BD": "物理",
              "OM": "化学", "AN": "化学", "CEC": "化学", "pH": "化学",
              "MBC": "生物", "RESP": "生物"}
DIMENSION_ORDER = ["SHI综合", "物理", "化学", "生物"]
LABELS = {"SHI_candidate": "SHI", "SW": "SW", "BD": "BD", "OM": "OM", "AN": "AN",
          "CEC": "CEC", "pH": "pH", "MBC": "MBC", "RESP": "RESP"}
COLORS = {"I-1 全掩膜": "#4477AA", "I-2 M4 配对": "#E07A5F", "I-4 环境 GWR": "#4D908E"}
try:
    from matplotlib import font_manager
    font_manager.findfont("WenQuanYi Zen Hei", fallback_to_default=False)
    plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
except (ImportError, ValueError):
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False


def _version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True,
                              text=True).stdout.strip()
    except Exception:
        return None


def _read(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)


def _merge_crosswalk(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    runs = {name: ROOT / path for name, path in config["source_runs"].items()}
    i1_assoc = _read(runs["i1"] / "results/global_response/global_associations.csv")
    i1_assoc = i1_assoc[(i1_assoc.exposure == "Aa_20yr") & (i1_assoc.population == "mask_all")]
    i1_assoc = i1_assoc[["outcome", "n_complete", "spearman_rho", "standardized_beta_log1p_Aa",
                         "block_bootstrap_beta_ci_low", "block_bootstrap_beta_ci_high",
                         "r_squared_linear_log1p_Aa", "linear_slope_y_per_log1p_Aa"]].rename(columns={
                             "n_complete": "global_n", "spearman_rho": "global_spearman_rho",
                             "r_squared_linear_log1p_Aa": "global_linear_r2",
                             "linear_slope_y_per_log1p_Aa": "global_linear_slope"})
    i1_models = _read(runs["i1"] / "leaderboards/global_response_model_leaderboard.csv")
    i1_models = i1_models[i1_models.exposure.eq("Aa_20yr")]
    i1_models = i1_models[["outcome", "selected_model_family", "selected_threshold_original", "validation_r2",
                           "test_r2", "test_rmse", "n_complete", "status"]].rename(
                               columns={"n_complete": "i1_model_n", "status": "i1_model_status"})
    i1_gam = _read(runs["i1"] / "results/nonlinear/full_sample_gam_summary.csv")
    i1_gam = i1_gam[i1_gam.exposure.eq("Aa_20yr")]
    i1_gam = i1_gam[["outcome", "effective_degrees_freedom", "local_slope_min_per_log1p_Aa",
                     "local_slope_max_per_log1p_Aa", "status"]].rename(columns={"status": "i1_gam_status"})
    i1_qr = _read(runs["i1"] / "results/nonlinear/quantile_regression.csv")
    i1_qr = i1_qr[(i1_qr.exposure == "Aa_20yr") & (i1_qr.status == "ok")]
    i1_qr_summary = i1_qr.groupby("outcome", as_index=False).agg(
        quantile_slope_min=("slope_per_log1p_Aa", "min"), quantile_slope_max=("slope_per_log1p_Aa", "max"),
        quantile_tau_count=("quantile", "nunique"))

    i2_pairs = _read(runs["i2"] / "tables/matched_pair_associations.csv")
    i2_pairs = i2_pairs[(i2_pairs.exposure.eq("Aa_20yr"))
                        & (i2_pairs.candidate_id.eq("stage2_Aa_20yr_01"))
                        & (i2_pairs.population.eq("all_erosion"))
                        & (i2_pairs.analysis_unit.eq("matched_pair"))]
    i2_pairs = i2_pairs[["outcome", "n_complete", "n_pairs", "spearman_rho", "slope_y_per_x",
                         "standardized_slope_beta", "r_squared_ols", "negative_pair_fraction"]].rename(columns={
                             "n_complete": "matched_pair_n_complete", "n_pairs": "matched_pair_n",
                             "spearman_rho": "matched_pair_spearman_rho", "slope_y_per_x": "matched_pair_slope_y_per_x",
                             "r_squared_ols": "matched_pair_r2"})
    i2_pixels = _read(runs["i2"] / "tables/matched_pixel_associations.csv")
    i2_pixels = i2_pixels[(i2_pixels.exposure.eq("Aa_20yr"))
                          & (i2_pixels.candidate_id.eq("stage2_Aa_20yr_01"))
                          & (i2_pixels.population.eq("all_erosion"))]
    i2_pixels = i2_pixels[["outcome", "n_complete", "spearman_rho"]].rename(columns={
        "n_complete": "matched_pixel_n", "spearman_rho": "matched_pixel_spearman_rho"})

    i3 = _read(runs["i3"] / "tables/environment_group_slopes_m4.csv")
    i3 = i3[(i3.exposure.eq("Aa_20yr")) & (i3.method.eq("M4"))
            & (i3.analysis_unit.eq("matched_pair_difference"))
            & (i3.group_column.eq("soil_type_group"))
            & i3.group_label.isin(["soil_26", "soil_27", "soil_42"])]
    i3_best = (i3.sort_values("standardized_effect")
               .groupby("outcome", as_index=False).first()[["outcome", "group_label", "n_pairs",
                                                            "standardized_effect", "slope_ci_low", "slope_ci_high"]]
               .rename(columns={"group_label": "i3_best_soil_label", "n_pairs": "i3_best_soil_n_pairs",
                                "standardized_effect": "i3_best_soil_standardized_effect",
                                "slope_ci_low": "i3_best_soil_slope_ci_low", "slope_ci_high": "i3_best_soil_slope_ci_high"}))

    i4 = _read(runs["i4"] / "tables/gwr_spatial_model_summary.csv")
    i4_adj = i4[i4.analysis_population.eq("environment_adjusted")]
    i4_keep = ["outcome", "n_spatial_units", "local_beta_standardized_mean", "local_beta_standardized_median",
               "local_beta_p05", "local_beta_p95", "negative_effect_ratio", "local_r2_weighted_mean"]
    i4_gwr = i4_adj[i4_adj.model_family.eq("GWR")][i4_keep]
    i4_lcr = i4_adj[i4_adj.model_family.eq("GWR-LCR")][i4_keep]
    i4_gwr = i4_gwr.add_prefix("i4_gwr_").rename(columns={"i4_gwr_outcome": "outcome"})
    i4_lcr = i4_lcr.add_prefix("i4_lcr_").rename(columns={"i4_lcr_outcome": "outcome"})

    i5_gam = _read(runs["i5"] / "tables/svc_gam_model_comparison.csv")
    i5_global = i5_gam[i5_gam.model_family.eq("global_GAM")][["outcome", "test_r2_spatial_block"]].rename(
        columns={"test_r2_spatial_block": "i5_global_gam_test_r2"})
    i5_svc = i5_gam[i5_gam.model_family.eq("svc_gam")][["outcome", "test_r2_spatial_block"]].rename(
        columns={"test_r2_spatial_block": "i5_svc_gam_test_r2"})
    i5_effects = _read(runs["i5"] / "tables/svc_gam_local_marginal_effects.parquet")
    i5_effects = i5_effects[i5_effects.exposure_quantile.eq(.5)].groupby("outcome", as_index=False).agg(
        i5_svc_marginal_effect_mean=("marginal_effect_standardized", "mean"),
        i5_svc_marginal_effect_median=("marginal_effect_standardized", "median"),
        i5_svc_marginal_effect_negative_ratio=("marginal_effect_standardized", lambda x: float((x < 0).mean())))

    i6_local = _read(runs["i6"] / "tables/local_spearman_summary.csv")
    i6_local = i6_local[["outcome", "rho_median", "rho_p10", "rho_p90", "negative_ratio",
                         "median_neighborhood_radius_m"]]
    i6_lisa = _read(runs["i6"] / "tables/bivariate_local_moran_cluster_counts.csv")
    i6_hl = i6_lisa[i6_lisa.lisa_class.eq("HL")][["outcome", "cell_count"]].rename(
        columns={"cell_count": "i6_lisa_fdr_HL_count"})
    i6_same = _read(runs["i6"] / "tables/same_site_colocation_summary.csv")
    i6_same = i6_same[["outcome", "high_Aa_low_health_count", "high_Aa_low_health_ratio"]]
    i6_gi = _read(runs["i6"] / "tables/local_negative_response_gistar_summary.csv")
    i6_gi = i6_gi[i6_gi.model_family.eq("GWR")][["outcome", "hotspot_count"]].rename(
        columns={"hotspot_count": "i6_gistar_GWR_hotspot_count"})

    frames = [i1_assoc, i1_models, i1_gam, i1_qr_summary, i2_pairs, i2_pixels, i3_best,
              i4_gwr, i4_lcr, i5_global, i5_svc, i5_effects, i6_local, i6_hl, i6_same, i6_gi]
    merged = pd.DataFrame({"outcome": OUTCOMES})
    for frame in frames:
        if frame.outcome.duplicated().any():
            raise ValueError("Expected one authoritative row per outcome in the I-7 source table.")
        merged = merged.merge(frame, on="outcome", how="left", validate="one_to_one")
    merged.insert(1, "health_dimension", merged.outcome.map(DIMENSIONS))
    merged["indicator_direction"] = "preserved_as_recorded; no sign reversal"
    merged["pcnm_status"] = "source_not_provided"
    merged["pls_sem_status"] = "source_not_provided"
    merged["outcome_guided_selection"] = True
    return merged, {"i1_assoc": i1_assoc, "i1_models": i1_models, "i1_gam": i1_gam,
                    "i1_qr": i1_qr, "i2_pairs": i2_pairs, "i3_best": i3_best,
                    "i4_gwr": i4_gwr, "i4_lcr": i4_lcr, "i5_gam": i5_gam,
                    "i5_effects": i5_effects, "i6_local": i6_local, "i6_lisa": i6_lisa,
                    "i6_same": i6_same, "i6_gi": i6_gi}


def _dimension_summary(crosswalk: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dimension in ["物理", "化学", "生物"]:
        d = crosswalk[crosswalk.health_dimension.eq(dimension)]
        global_beta = pd.to_numeric(d.standardized_beta_log1p_Aa, errors="coerce")
        matched_beta = pd.to_numeric(d.standardized_slope_beta, errors="coerce")
        gwr = pd.to_numeric(d.i4_gwr_local_beta_standardized_mean, errors="coerce")
        slope_min = pd.to_numeric(d.local_slope_min_per_log1p_Aa, errors="coerce")
        slope_max = pd.to_numeric(d.local_slope_max_per_log1p_Aa, errors="coerce")
        rows.append({"health_dimension": dimension, "n_indicators": len(d),
                     "global_beta_negative_indicators": int((global_beta < 0).sum()),
                     "global_beta_positive_indicators": int((global_beta > 0).sum()),
                     "global_beta_median": float(global_beta.median()),
                     "matched_pair_beta_negative_indicators": int((matched_beta < 0).sum()),
                     "matched_pair_beta_positive_indicators": int((matched_beta > 0).sum()),
                     "matched_pair_beta_median": float(matched_beta.median()),
                     "environment_adjusted_GWR_beta_median": float(gwr.median()),
                     "local_spearman_median_of_indicators": float(pd.to_numeric(d.rho_median, errors="coerce").median()),
                     "global_GAM_slope_sign_change_indicators": int(((slope_min < 0) & (slope_max > 0)).sum()),
                     "spatial_test_R2_median": float(pd.to_numeric(d.test_r2, errors="coerce").median()),
                     "composite_dimension_index_constructed": False,
                     "note": "indicator signs retained; summary is descriptive across indicators"})
    shi = crosswalk[crosswalk.outcome.eq("SHI_candidate")].iloc[0]
    rows.insert(0, {"health_dimension": "SHI综合", "n_indicators": 1,
                    "global_beta_negative_indicators": int(shi.standardized_beta_log1p_Aa < 0),
                    "global_beta_positive_indicators": int(shi.standardized_beta_log1p_Aa > 0),
                    "global_beta_median": float(shi.standardized_beta_log1p_Aa),
                    "matched_pair_beta_negative_indicators": int(shi.standardized_slope_beta < 0),
                    "matched_pair_beta_positive_indicators": int(shi.standardized_slope_beta > 0),
                    "matched_pair_beta_median": float(shi.standardized_slope_beta),
                    "environment_adjusted_GWR_beta_median": float(shi.i4_gwr_local_beta_standardized_mean),
                    "local_spearman_median_of_indicators": float(shi.rho_median),
                    "global_GAM_slope_sign_change_indicators": int(shi.local_slope_min_per_log1p_Aa < 0 < shi.local_slope_max_per_log1p_Aa),
                    "spatial_test_R2_median": float(shi.test_r2),
                    "composite_dimension_index_constructed": False,
                    "note": "SHI is retained as the existing composite response; no reweighting"})
    return pd.DataFrame(rows)


def _pcnm_comparison(crosswalk: pd.DataFrame, inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    i1 = inputs["i1_assoc"].set_index("outcome")
    i2 = inputs["i2_pairs"].set_index("outcome")
    i4 = inputs["i4_gwr"].set_index("outcome")
    records = []
    for outcome in OUTCOMES:
        env_specs = [
            ("I-1 full-mask global", "full_mask_pixel", i1.loc[outcome, "global_linear_slope"],
             i1.loc[outcome, "standardized_beta_log1p_Aa"], i1.loc[outcome, "block_bootstrap_beta_ci_low"],
             i1.loc[outcome, "block_bootstrap_beta_ci_high"], i1.loc[outcome, "global_n"],
             i1.loc[outcome, "global_linear_r2"]),
            ("I-2 M4 pair difference", "matched_pair", i2.loc[outcome, "matched_pair_slope_y_per_x"],
             i2.loc[outcome, "standardized_slope_beta"], np.nan, np.nan, i2.loc[outcome, "matched_pair_n"],
             i2.loc[outcome, "matched_pair_r2"]),
            ("I-4 adjusted GWR local mean", "240m_environment_adjusted_local_units", np.nan,
             i4.loc[outcome, "i4_gwr_local_beta_standardized_mean"],
             i4.loc[outcome, "i4_gwr_local_beta_p05"], i4.loc[outcome, "i4_gwr_local_beta_p95"],
             i4.loc[outcome, "i4_gwr_n_spatial_units"], i4.loc[outcome, "i4_gwr_local_r2_weighted_mean"]),
        ]
        for method, unit, coef, std, lo, hi, n, r2 in env_specs:
            records.append({"outcome": outcome, "health_dimension": DIMENSIONS[outcome],
                            "erosion_scenario": "Aa_20yr", "environmental_method": method,
                            "environmental_analysis_unit": unit, "coefficient_environmental": coef,
                            "standardized_coefficient_environmental": std,
                            "environmental_interval_low": lo, "environmental_interval_high": hi,
                            "environmental_interval_type": "5km_block_bootstrap_95CI" if method.startswith("I-1") else
                            "local_unit_P05_P95_distribution" if method.startswith("I-4") else "not_available",
                            "environmental_n": n, "environmental_r_squared": r2,
                            "coefficient_pcnm": np.nan, "standardized_coefficient_pcnm": np.nan,
                            "standard_error_pcnm": np.nan, "ci_lower_pcnm": np.nan, "ci_upper_pcnm": np.nan,
                            "p_value_pcnm": np.nan, "r_squared_pcnm": np.nan, "n_pcnm": np.nan,
                            "analysis_unit_pcnm": np.nan, "controls_pcnm": np.nan,
                            "pcnm_source_status": "not_provided_in_config_or_workspace",
                            "direction_consistent": pd.NA, "direct_coefficient_comparison_allowed": False,
                            "comparison_status": "awaiting_existing_PCNM_table",
                            "outcome_guided_selection": True})
    return pd.DataFrame(records)


def _pls_crosswalk() -> pd.DataFrame:
    paths = [
        ("Erosion → Physical", "SW; BD", "water content; bulk density"),
        ("Erosion → Chemical", "OM; AN; CEC; pH", "organic matter; available nitrogen; cation exchange capacity; pH"),
        ("Erosion → Biological", "MBC; RESP", "microbial biomass carbon; soil respiration"),
        ("Physical → SHI", "SW; BD → SHI", "existing PLS-SEM structural path"),
        ("Chemical → SHI", "OM; AN; CEC; pH → SHI", "existing PLS-SEM structural path"),
        ("Biological → SHI", "MBC; RESP → SHI", "existing PLS-SEM structural path"),
    ]
    return pd.DataFrame([{"path": path, "indicator_or_construct_mapping": mapping,
                          "envmatch_proxy": proxy, "pls_sem_coefficient": np.nan,
                          "pls_sem_standardized_coefficient": np.nan, "standard_error": np.nan,
                          "ci_lower": np.nan, "ci_upper": np.nan, "p_value": np.nan,
                          "r_squared": np.nan, "sample_size": np.nan, "controls": np.nan,
                          "analysis_unit": np.nan, "measurement_weights_and_signs": np.nan,
                          "source_status": "PLS_SEM_results_not_provided", "outcome_guided_selection": True}
                         for path, mapping, proxy in paths])


def _save_forest(crosswalk: pd.DataFrame, run_dir: Path, dpi: int) -> list[str]:
    order = ["SHI_candidate", "SW", "BD", "OM", "AN", "CEC", "pH", "MBC", "RESP"]
    y = {outcome: len(order) - 1 - index for index, outcome in enumerate(order)}
    fig, ax = plt.subplots(figsize=(11, 8.5), constrained_layout=True)
    dimension_blocks = [("SHI综合", ["SHI_candidate"]), ("物理", ["SW", "BD"]),
                        ("化学", ["OM", "AN", "CEC", "pH"]), ("生物", ["MBC", "RESP"])]
    for dimension, outcomes in dimension_blocks:
        ys = [y[o] for o in outcomes]
        ax.axhspan(min(ys) - .43, max(ys) + .43, color="#f1f4f6", zorder=0)
        # Dimension membership is included in the y tick label to avoid
        # overlapping the indicator labels in the left margin.
    offsets = {"I-1 全掩膜": .20, "I-2 M4 配对": 0.0, "I-4 环境 GWR": -.20}
    row = crosswalk.set_index("outcome")
    for outcome in order:
        r = row.loc[outcome]
        base_y = y[outcome]
        global_x = float(r.standardized_beta_log1p_Aa)
        ax.errorbar(global_x, base_y + offsets["I-1 全掩膜"],
                    xerr=[[global_x - float(r.block_bootstrap_beta_ci_low)],
                          [float(r.block_bootstrap_beta_ci_high) - global_x]],
                    fmt="o", color=COLORS["I-1 全掩膜"], capsize=2.5, markersize=5,
                    label="I-1 全掩膜 β（5 km block 95% CI）" if outcome == order[0] else None, zorder=3)
        matched_x = float(r.standardized_slope_beta)
        ax.plot(matched_x, base_y + offsets["I-2 M4 配对"], marker="s", linestyle="", color=COLORS["I-2 M4 配对"],
                markersize=5, label="I-2 M4 配对差值 β（无区间）" if outcome == order[0] else None, zorder=3)
        gwr_x = float(r.i4_gwr_local_beta_standardized_median)
        ax.errorbar(gwr_x, base_y + offsets["I-4 环境 GWR"],
                    xerr=[[gwr_x - float(r.i4_gwr_local_beta_p05)],
                          [float(r.i4_gwr_local_beta_p95) - gwr_x]],
                    fmt="^", color=COLORS["I-4 环境 GWR"], capsize=2.5, markersize=5,
                    label="I-4 环境 GWR 局部中位数（P05–P95）" if outcome == order[0] else None, zorder=3)
    ax.axvline(0, color="#303c46", lw=.8)
    ax.set_yticks([y[o] for o in order], [f"{LABELS[o]} ({DIMENSIONS[o]})" for o in order])
    ax.set_xlabel("标准化侵蚀系数（方向按源结局保留）")
    ax.set_title("EnvMatch 各分析单位响应系数对照；PCNM 结果未提供")
    ax.grid(axis="x", color="#d8dee3", linewidth=.6)
    ax.legend(loc="lower right", frameon=True, fontsize=8)
    ax.text(.01, -.10, "横线类型：I-1 为 5 km 空间块自助 95% CI；I-4 为局部单元系数 P05–P95 分布，不是置信区间；I-2 未提供区间。三种分析单位不同，不直接比较绝对值。",
            transform=ax.transAxes, fontsize=8, va="top", wrap=True)
    paths = []
    base = run_dir / "figures" / "health_dimension_comparison_forest"
    for ext in ("png", "svg", "pdf"):
        path = base.with_suffix(f".{ext}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(str(path.relative_to(ROOT)))
    plt.close(fig)
    return paths


def _report(run_dir: Path, crosswalk: pd.DataFrame, dimensions: pd.DataFrame,
            pcnm: pd.DataFrame, pls: pd.DataFrame, source_status: dict[str, str]) -> str:
    lines = ["# Phase I-7：健康维度、PLS-SEM 与 PCNM 衔接", "",
             f"权威运行：`{run_dir.relative_to(ROOT)}`。模式：`EXPLORATORY_ORACLE`，`outcome_guided_selection=true`。", "",
             "## 数据可用性与综合边界", "",
             f"- 现有 I-1 至 I-6 的 EnvMatch 结果均可复用。PCNM 源状态：{source_status['pcnm']}；项目配置 `config/data_paths.yaml` 中 `pcnm.results=null`，工作区和 Oracle 输出目录内未找到 PCNM 结果表。PLS-SEM 源状态：{source_status['pls_sem']}；未找到可核对的结构路径/载荷结果表。",
             "- 本阶段不重跑 PCNM，不从文献、栅格或 EnvMatch 回归反推 PLS-SEM 路径。比较 CSV 保留 PCNM 相关字段为空，并记录等待现有结果输入的状态。",
             "- SW/BD、OM/AN/CEC/pH、MBC/RESP 按任务规定归为物理、化学、生物维度。现有指标的方向和符号保持不变；因为缺少经核实的指标隶属度方向、构念载荷和维度权重，未构造新的综合维度指数。以下维度表是各指标结果的描述性汇总，不是潜变量路径效应。", "",
             "## 三个健康维度与 SHI", "",
             "| 维度 | 指标数 | I-1 全掩膜 β 负/正 | I-1 β 中位数 | I-2 匹配配对 β 中位数 | I-4 环境调整 GWR β 中位数 | I-6 局部 ρ 中位数 | I-1 GAM 斜率跨零指标数 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in dimensions.itertuples(index=False):
        lines.append(f"| {r.health_dimension} | {r.n_indicators} | {r.global_beta_negative_indicators}/{r.global_beta_positive_indicators} | {r.global_beta_median:.3f} | {r.matched_pair_beta_median:.3f} | {r.environment_adjusted_GWR_beta_median:.3f} | {r.local_spearman_median_of_indicators:.3f} | {r.global_GAM_slope_sign_change_indicators} |")
    lines.extend(["", "以上中位数汇总每项指标各自的标准化系数，保留原始结局符号。物理维度中 SW 与 BD 的全局响应均为负，但绝对幅度差异明显；化学维度方向混合，CEC 为正而 OM、AN、pH 为负；生物维度中 RESP 为负向较强、MBC 全局线性关联较弱。维度指标数量分别为 2、4、2，不能把中位系数称作经验证的潜变量效应。", "",
                  "## 指标级响应规律", "",
                  "| 维度 | 指标 | I-1 β（95% block CI） | I-1 最佳空间测试模型 / R² | GAM edf；局部斜率范围 | I-2 M4 配对 β / Spearman | I-4 环境 GWR β / LCR β | I-3 最负向土类候选 |", "|---|---|---:|---|---|---:|---:|---|"])
    for r in crosswalk.itertuples(index=False):
        threshold = f"阈值={r.selected_threshold_original:.3f}" if pd.notna(r.selected_threshold_original) else "未选中分段模型"
        lines.append(f"| {r.health_dimension} | {LABELS[r.outcome]} | {r.standardized_beta_log1p_Aa:.3f} ({r.block_bootstrap_beta_ci_low:.3f}, {r.block_bootstrap_beta_ci_high:.3f}) | {r.selected_model_family} / {r.test_r2:.3f} | edf {r.effective_degrees_freedom:.2f}; {r.local_slope_min_per_log1p_Aa:.3f}–{r.local_slope_max_per_log1p_Aa:.3f}; {threshold} | {r.standardized_slope_beta:.3f} / {r.matched_pair_spearman_rho:.3f} | {r.i4_gwr_local_beta_standardized_mean:.3f} / {r.i4_lcr_local_beta_standardized_mean:.3f} | {r.i3_best_soil_label} ({r.i3_best_soil_standardized_effect:.3f}) |")
    lines.extend(["", "- I-1 的分段拐点候选从训练 Aa 分位数中按验证表现选择；当前只有 MBC 的选定模型为 segmented（Aa 原始存储值约 4.617）。这是 Oracle 探索候选，不是经独立数据验证的阈值。",
                  "- I-1 GAM 的有效自由度约 7.4；SHI 局部斜率范围维持负向，其余多项指标的全局斜率范围跨零。空间 SVC-GAM 对 SHI/OM/SW 的 5 km 测试增益相对全局 GAM 为 +0.014/+0.007/+0.005，属于小幅变化。",
                  "- 匹配配对估计、全掩膜像元估计及 240 m 局部系数使用不同分析单位和支持量。I-2 配对 β不是 I-1 全样本 β的复现；I-4 局部均值也不是独立样本总体效应。", "",
                  "## 与 PLS-SEM 现有路径的衔接", "",
                  "PLS-SEM 结果表缺失，路径交叉表已列出待接入路径以及每条路径的 EnvMatch 指标代理。当前不得判断 Erosion→Physical/Chemical/Biological 或三个维度→SHI 的路径方向与显著性，也不计算间接效应。",
                  "接入现有结果至少需要：路径名称与方向；标准化路径系数；标准误或 bootstrap 置信区间、p/t 值；R²；样本量与分析单位；控制变量；构念/观测指标映射和各指标正负方向；直接、间接、总效应；估计方法与 bootstrap 次数。缺失字段应继续留空。", "",
                  "## 与既有 PCNM 结果的比较", "",
                  f"PCNM 结果表当前不可读（{source_status['pcnm']}），因此 `EnvMatch_PCNM_comparison.csv` 中 PCNM 系数、区间、P 值、R²、N、分析单位和控制变量全部留空，`direction_consistent` 不作判断。PCNM 不重跑。",
                  "当前 EnvMatch 侧给出全掩膜 I-1、M4 配对差值 I-2 和环境调整 GWR I-4 的系数供后续逐项核对；森林图仅绘制可用 EnvMatch 估计，不能视为 EnvMatch-PCNM 一致性图。比较绝对系数前必须先协调 PCNM/EnvMatch 的样本单位、暴露定义、协变量、模型阶段及标准化口径。", "",
                  "## EnvMatch 系数对照图", "",
                  "图中 I-1 误差线为 5 km 空间块 bootstrap 95% CI，I-4 误差线为局部系数 P05–P95 分布，I-2 配对系数没有区间。三者点位方向可并列看，绝对值不可直接排名。", "",
                  "## 解释限制", "",
                  "- 健康结局来自随机森林预测栅格，不等同独立实测样本；SHI 来源仍为项目暂定候选层。",
                  "- 结果来自此前 Oracle 搜索筛选；表内方向和效应不作为正式独立确认或因果路径。",
                  "- 本阶段未更改指标符号、SHI 权重或维度载荷；未运行 PCNM、未重算/验证 RUSLE 因子乘积、未修改研究总纲。", "",
                  "表格和绘图数据位于 `tables/`；输入运行、数据缺口及配置记录见 `manifests/provenance.json`。", ""])
    path = run_dir / "reports" / "PHASE_I_HEALTH_MECHANISM_REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path.relative_to(ROOT))


def run(config_path: Path) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("mode") != "EXPLORATORY_ORACLE" or config.get("outcome_guided_selection") is not True:
        raise ValueError("Phase I-7 must be EXPLORATORY_ORACLE with outcome_guided_selection=true.")
    run_dir = ROOT / config["output_root"] / config["run_id"]
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    for sub in ("configs", "tables", "figures", "reports", "manifests"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    (run_dir / "configs" / config_path.name).write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    crosswalk, inputs = _merge_crosswalk(config)
    dimensions = _dimension_summary(crosswalk)
    pcnm = _pcnm_comparison(crosswalk, inputs)
    pls = _pls_crosswalk()
    crosswalk.to_csv(run_dir / "tables/indicator_response_crosswalk.csv", index=False)
    dimensions.to_csv(run_dir / "tables/health_dimension_summary.csv", index=False)
    pcnm.to_csv(run_dir / "tables/EnvMatch_PCNM_comparison.csv", index=False)
    pls.to_csv(run_dir / "tables/pls_sem_path_crosswalk.csv", index=False)

    figure_paths = _save_forest(crosswalk, run_dir, int(config["map_dpi"]))
    pcnm_source = config.get("pcnm_source_path")
    pls_source = config.get("pls_sem_source_path")
    pcnm_ok = bool(pcnm_source and (ROOT / pcnm_source).is_file())
    pls_ok = bool(pls_source and (ROOT / pls_source).is_file())
    source_status = {"pcnm": "provided" if pcnm_ok else "not provided; config/data_paths.yaml has pcnm.results=null",
                     "pls_sem": "provided" if pls_ok else "not provided; no path coefficient table was found"}
    report = _report(run_dir, crosswalk, dimensions, pcnm, pls, source_status)
    source_paths = {name: str((ROOT / value).relative_to(ROOT)) for name, value in config["source_runs"].items()}
    provenance = {"phase": "I-7", "mode": config["mode"], "outcome_guided_selection": True,
                  "run_id": config["run_id"], "created_utc": datetime.now(timezone.utc).isoformat(),
                  "git_commit": _git("rev-parse", "HEAD"), "git_status_porcelain": _git("status", "--short"),
                  "inputs": {"source_runs": source_paths, "pcnm_source": pcnm_source, "pls_sem_source": pls_source},
                  "configuration": str((run_dir / "configs" / config_path.name).relative_to(ROOT)),
                  "rows": {"indicator_crosswalk": len(crosswalk), "health_dimension_summary": len(dimensions),
                           "pcnm_comparison_candidates": len(pcnm), "pls_sem_expected_paths": len(pls)},
                  "health_dimensions": {"physical": ["SW", "BD"], "chemical": ["OM", "AN", "CEC", "pH"],
                                        "biological": ["MBC", "RESP"], "composite": ["SHI_candidate"]},
                  "source_status": source_status,
                  "analysis_limits": ["indicator signs preserved", "no new dimension index constructed",
                                       "no PCNM rerun", "no PLS-SEM paths inferred"],
                  "software": {"python": platform.python_version(), "pandas": pd.__version__,
                               "numpy": np.__version__, "matplotlib": _version("matplotlib"), "pyyaml": _version("PyYAML")},
                  "outputs": [str((run_dir / p).relative_to(ROOT)) for p in [
                      Path("tables/indicator_response_crosswalk.csv"), Path("tables/health_dimension_summary.csv"),
                      Path("tables/EnvMatch_PCNM_comparison.csv"), Path("tables/pls_sem_path_crosswalk.csv"),
                      Path("reports/PHASE_I_HEALTH_MECHANISM_REPORT.md")]] + figure_paths,
                  "report": report}
    (run_dir / "manifests/provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "RUN_STATUS.md").write_text(
        "# Run status\n\nStatus: authoritative exploratory synthesis\n\n- Phase: I-7\n- `outcome_guided_selection=true`\n- PCNM and PLS-SEM source tables were not provided; missing fields are preserved.\n- PCNM was not rerun; no PLS-SEM path was inferred.\n",
        encoding="utf-8")
    files = sorted(p.relative_to(run_dir).as_posix() for p in run_dir.rglob("*") if p.is_file() and p.name != "output_manifest.json")
    (run_dir / "manifests/output_manifest.json").write_text(json.dumps(files, ensure_ascii=False, indent=2), encoding="utf-8")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "phase_i_health_mechanism.yaml")
    args = parser.parse_args()
    run_dir = run(args.config)
    print(run_dir.relative_to(ROOT))
    print((run_dir / "reports/PHASE_I_HEALTH_MECHANISM_REPORT.md").relative_to(ROOT))


if __name__ == "__main__":
    main()
