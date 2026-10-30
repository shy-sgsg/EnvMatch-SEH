"""Matched-pixel and matched-pair response supplement for low-erosion regimes."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from statsmodels.stats.multitest import multipletests

from experiments.low_erosion_regime.figures import (
    EXPOSURES, EXPOSURE_LABELS, OUTCOMES, OUTCOME_LABELS, _save,
)
from experiments.low_erosion_regime.response_models import (
    MODEL_FAMILIES, matched_pixel_association, matched_pixel_model_trials,
    matched_pixel_population, matched_pixel_spatial_validation,
    paired_association, paired_model_trials, paired_spatial_validation,
)
from experiments.low_erosion_regime.distribution_diagnostics import summarize_values
from experiments.low_erosion_regime.run import _load_inputs


SOIL_TYPE_LABELS = {26: "冲积土", 27: "潮土", 42: "脱潮土"}
SCHEME_LABELS = {
    "S0_all": "全部原始匹配对",
    "S1_both_positive": "两端均为正侵蚀",
    "T01_empirical_CDF_breakpoint": "CDF 阈值筛选原配对",
    "T01_fixed_original": "阈值后固定原组重配",
    "T01_refit_k6_filtered_population": "阈值后重拟合 k=6 组",
}
FAMILY_LABELS = {
    "linear": "线性回归", "quadratic": "二次回归", "natural_spline": "自然样条",
    "additive_gam": "加性样条（GAM 近似）", "huber": "Huber 稳健回归",
    "random_forest": "随机森林", "extra_trees": "极端随机树",
    "hist_gradient_boosting": "直方图梯度提升",
}
STYLE = {
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.edgecolor": "#D5DEE4", "axes.labelcolor": "#253542", "text.color": "#253542",
    "xtick.color": "#52636E", "ytick.color": "#52636E", "grid.color": "#E7EDF0",
    "font.family": "Noto Sans CJK JP", "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9, "legend.frameon": False, "svg.fonttype": "none",
}


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _read_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def _write_table(frame: pd.DataFrame, root: Path, stem: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(root / f"{stem}.csv", index=False, encoding="utf-8-sig")
    frame.to_parquet(root / f"{stem}.parquet", index=False)


def _source_populations(
    low_run: Path, method: str, exposure: str,
) -> dict[str, Path]:
    return {
        "S0_all": low_run / "filtered_pairs" / f"existing_{method}_{exposure}_S0_all.parquet",
        "S1_both_positive": low_run / "filtered_pairs" / f"existing_{method}_{exposure}_S1_both_positive.parquet",
        "T01_empirical_CDF_breakpoint": low_run / "filtered_pairs" / f"existing_{method}_{exposure}_T01.parquet",
        "T01_fixed_original": low_run / "rematched_pairs" / f"rematched_{method}_{exposure}_T01_fixed_original.parquet",
        "T01_refit_k6_filtered_population": low_run / "rematched_pairs" / f"rematched_{method}_{exposure}_T01_refit_k6_filtered_population.parquet",
    }


def _population_summary(
    pairs: pd.DataFrame, *, method: str, exposure: str, scheme: str, threshold: float | None,
    baseline_pairs: int, baseline_unique_endpoints: int,
) -> dict[str, Any]:
    high = set(pairs.get("high_erosion_cell_id", pd.Series(dtype=object)).dropna().tolist())
    low = set(pairs.get("low_erosion_cell_id", pd.Series(dtype=object)).dropna().tolist())
    unique = high | low
    n_pairs = int(len(pairs))
    return {
        "analysis_unit": "matched_population", "method": method, "exposure": exposure,
        "scheme": scheme, "scheme_label": SCHEME_LABELS[scheme], "threshold_original": threshold,
        "n_pairs": n_pairs, "n_unique_matched_endpoints": len(unique),
        "n_high_endpoint_cells": len(high), "n_low_endpoint_cells": len(low),
        "n_cells_in_both_endpoint_roles": len(high & low),
        "endpoint_rows_before_deduplication": n_pairs * 2,
        "pair_retention_vs_S0": n_pairs / baseline_pairs if baseline_pairs else np.nan,
        "unique_endpoint_retention_vs_S0": len(unique) / baseline_unique_endpoints if baseline_unique_endpoints else np.nan,
        "outcome_guided_selection": True,
    }


def _positive_distribution_audit(cells: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for exposure in EXPOSURES:
        summary = summarize_values(pd.to_numeric(cells[exposure], errors="coerce").to_numpy(float))
        rows.append({"exposure": exposure, "analysis_scope": "valid_cell=True; zero and NoData separated", **summary})
    return pd.DataFrame.from_records(rows)


def _add_adjusted_p_values(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    group_columns = [name for name in ("analysis_unit", "exposure", "method", "scheme") if name in result]
    for p_column in ("spearman_p", "wilcoxon_p", "sign_test_p"):
        if p_column not in result:
            continue
        q_column = f"{p_column}_q"
        result[q_column] = np.nan
        for _, index in result.groupby(group_columns, dropna=False, sort=False).groups.items():
            values = pd.to_numeric(result.loc[index, p_column], errors="coerce")
            valid = values.notna() & np.isfinite(values)
            if valid.any():
                result.loc[values.index[valid], q_column] = multipletests(values.loc[valid], method="fdr_bh")[1]
    return result


def _comparison_table(
    models: pd.DataFrame, associations: pd.DataFrame, spatial: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["analysis_unit", "exposure", "method", "scheme", "outcome"]
    model_rows: list[dict[str, Any]] = []
    for key, part in models.groupby(keys, dropna=False, sort=False):
        key_values = dict(zip(keys, key, strict=True))
        successful = part.loc[part["status"].eq("ok") & pd.to_numeric(part["r_squared_apparent"], errors="coerce").notna()]
        negative = successful.loc[pd.to_numeric(successful["spearman_rho"], errors="coerce").lt(0)]
        best = successful.loc[pd.to_numeric(successful["r_squared_apparent"], errors="coerce").idxmax()] if not successful.empty else None
        best_negative = negative.loc[pd.to_numeric(negative["r_squared_apparent"], errors="coerce").idxmax()] if not negative.empty else None
        row = {
            **key_values,
            "scheme_label": SCHEME_LABELS.get(str(key_values["scheme"]), str(key_values["scheme"])),
            "n_unique_matched_endpoints": int(pd.to_numeric(part.get("n_unique_matched_endpoints", pd.Series(dtype=float)), errors="coerce").dropna().max()) if part.get("n_unique_matched_endpoints") is not None and pd.to_numeric(part.get("n_unique_matched_endpoints"), errors="coerce").notna().any() else np.nan,
            "n_pairs": int(pd.to_numeric(part.loc[part["analysis_unit"].eq("pair"), "n"], errors="coerce").max()) if key_values["analysis_unit"] == "pair" and not part.empty else np.nan,
            "best_apparent_model_family": str(best["model_family"]) if best is not None else None,
            "best_apparent_r_squared": float(best["r_squared_apparent"]) if best is not None else np.nan,
            "best_negative_model_family": str(best_negative["model_family"]) if best_negative is not None else None,
            "best_negative_r_squared": float(best_negative["r_squared_apparent"]) if best_negative is not None else np.nan,
            "model_families_attempted": int(part["model_family"].nunique()),
            "successful_model_families": int(successful["model_family"].nunique()),
            "outcome_guided_selection": True,
            "r_squared_scope": "apparent_same_sample_oracle_search",
        }
        model_rows.append(row)
    result = pd.DataFrame(model_rows)
    association_columns = [*keys, "n", "spearman_rho", "spearman_p_q", "mean_delta_health", "median_delta_health", "negative_delta_fraction", "wilcoxon_p_q", "slope", "slope_ci_low", "slope_ci_high"]
    existing_association_columns = [name for name in association_columns if name in associations]
    result = result.merge(associations[existing_association_columns], on=keys, how="left", validate="one_to_one")
    if not spatial.empty:
        spatial_summary = spatial.loc[spatial["status"].eq("ok")].groupby(keys, as_index=False).agg(
            n_spatial_splits=("split", "nunique"),
            spatial_test_r2_median=("spatial_test_r_squared_selected", "median"),
            spatial_test_r2_q25=("spatial_test_r_squared_selected", lambda values: values.quantile(.25)),
            spatial_test_r2_q75=("spatial_test_r_squared_selected", lambda values: values.quantile(.75)),
            best_spatial_model_family=("selected_model_by_validation", lambda values: values.mode().iloc[0] if not values.mode().empty else None),
            negative_spatial_r2_fraction=("spatial_test_r_squared_selected", lambda values: float((values < 0).mean())),
        )
        result = result.merge(spatial_summary, on=keys, how="left", validate="one_to_one")
    baseline_keys = ["analysis_unit", "exposure", "method", "outcome"]
    baseline = result.loc[result["scheme"].eq("S0_all"), baseline_keys + ["best_apparent_r_squared", "spearman_rho"]].rename(
        columns={"best_apparent_r_squared": "baseline_best_apparent_r_squared", "spearman_rho": "baseline_spearman_rho"}
    )
    result = result.merge(baseline, on=baseline_keys, how="left", validate="many_to_one")
    result["apparent_r_squared_change_vs_S0"] = result["best_apparent_r_squared"] - result["baseline_best_apparent_r_squared"]
    result["spearman_change_vs_S0"] = result["spearman_rho"] - result["baseline_spearman_rho"]
    result["comparison_note"] = "群体筛选或重配对会改变分析样本与估计对象；R²差异不是单独的模型改进效应。"
    return result


def _make_figures(
    figure_root: Path, cells: pd.DataFrame, pairs_by_key: dict[tuple[str, str, str], pd.DataFrame],
    populations: pd.DataFrame, associations: pd.DataFrame, models: pd.DataFrame,
    spatial: pd.DataFrame, threshold_by_exposure: dict[str, float],
) -> list[dict[str, str]]:
    plt.rcParams.update(STYLE)
    sns.set_theme(style="whitegrid", rc=STYLE)
    manifest: list[dict[str, str]] = []

    # Apparent nonlinear model comparison for the 20-year CDF threshold.
    selected = models.loc[
        models["exposure"].eq("Aa_20yr")
        & models["scheme"].eq("T01_empirical_CDF_breakpoint")
        & models["method"].isin(["M3", "M4"])
        & models["analysis_unit"].isin(["matched_pixel", "pair"])
    ].copy()
    if not selected.empty:
        fig, axes = plt.subplots(2, 2, figsize=(20, 11), constrained_layout=True, sharex=True, sharey=True)
        outcome_order = list(OUTCOMES)
        family_order = list(MODEL_FAMILIES)
        for row, unit in enumerate(("matched_pixel", "pair")):
            for col, method in enumerate(("M3", "M4")):
                ax = axes[row, col]
                part = selected.loc[selected["analysis_unit"].eq(unit) & selected["method"].eq(method)]
                matrix = part.pivot_table(index="outcome", columns="model_family", values="r_squared_apparent", aggfunc="median").reindex(index=outcome_order, columns=family_order)
                sns.heatmap(matrix, ax=ax, cmap="vlag", center=0, vmin=-.15, vmax=.45, annot=True, fmt=".2f",
                            linewidths=.45, linecolor="white", cbar=(row == 0 and col == 1),
                            cbar_kws={"label": "同样本表观 R²（仅探索）"} if row == 0 and col == 1 else None)
                ax.set_xticklabels([FAMILY_LABELS.get(str(value), str(value)) for value in matrix.columns], rotation=35, ha="right")
                ax.set_yticklabels([OUTCOME_LABELS.get(str(value), str(value)) for value in matrix.index], rotation=0)
                ax.set_xlabel("响应模型形式")
                ax.set_ylabel("健康指标" if col == 0 else "")
                ax.set_title(f"{'匹配像元' if unit == 'matched_pixel' else '配对差值'} · {method}", loc="left", fontweight="bold")
        threshold = threshold_by_exposure.get("Aa_20yr", np.nan)
        fig.suptitle(f"20 年侵蚀 CDF 阈值群体：八类模型响应比较（Aa 阈值={threshold:.3f}）", x=.01, y=1.025, ha="left", fontsize=15, fontweight="bold")
        manifest.append(_save(fig, figure_root, "M03_matched_unit_model_family_comparison", selected,
            "每格为对应健康指标、分析单位和匹配方法下八类模型的同样本表观 R²。匹配像元每个端点按 cell_id 去重；配对面板以高低侵蚀端健康差为响应。群体与模型均在 Oracle 探索中比较，表观 R²不视为空间泛化或因果效应。",
            scientific_question="低侵蚀阈值后，环境匹配像元与配对差值呈现何种响应形态？",
            experiment="阈值筛选后八类模型搜索", figure_type="模型族比较热图"))

    # Endpoint association heatmaps, with each raster cell counted once.
    matched_assoc = associations.loc[associations["analysis_unit"].eq("matched_pixel")].copy()
    if not matched_assoc.empty:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True, sharex=True, sharey=True)
        scheme_order = list(SCHEME_LABELS)
        for row, exposure in enumerate(EXPOSURES):
            for col, method in enumerate(("M3", "M4")):
                ax = axes[row, col]
                part = matched_assoc.loc[matched_assoc["exposure"].eq(exposure) & matched_assoc["method"].eq(method)]
                matrix = part.pivot_table(index="outcome", columns="scheme", values="spearman_rho", aggfunc="first").reindex(index=list(OUTCOMES), columns=scheme_order)
                sns.heatmap(matrix, ax=ax, cmap="vlag", center=0, vmin=-1, vmax=1, annot=True, fmt=".2f",
                            linewidths=.5, linecolor="white", cbar=(row == 0 and col == 1),
                            cbar_kws={"label": "Spearman 秩相关系数 ρ"} if row == 0 and col == 1 else None)
                ax.set_xticklabels([SCHEME_LABELS.get(str(value), str(value)) for value in matrix.columns], rotation=30, ha="right")
                ax.set_yticklabels([OUTCOME_LABELS.get(str(value), str(value)) for value in matrix.index], rotation=0)
                ax.set_xlabel("匹配群体方案")
                ax.set_ylabel("健康指标" if col == 0 else "")
                ax.set_title(f"{EXPOSURE_LABELS[exposure]} · {method} · 每像元一次", loc="left", fontweight="bold")
        fig.suptitle("匹配端点像元的侵蚀—健康秩关联", x=.01, y=1.025, ha="left", fontsize=15, fontweight="bold")
        manifest.append(_save(fig, figure_root, "R04_matched_pixel_rank_associations", matched_assoc,
            "每个匹配端点按 cell_id 只计一次，不因一个像元参与多个配对而重复加权。对照全部原配对、两端正值、CDF 阈值筛选及两种阈值后重配方案。高低配对差值的秩关联另见配套表和森林图。",
            scientific_question="低值筛选与重配后，匹配端点像元的秩关联如何变化？",
            experiment="匹配像元秩关联", figure_type="相关性热图"))

    # Spatial test scores for matched pixels and pairs under the primary M4 rule.
    spatial_primary = spatial.loc[
        spatial["method"].eq("M4")
        & spatial["scheme"].isin(["S1_both_positive", "T01_empirical_CDF_breakpoint"])
        & spatial["status"].eq("ok")
    ].copy() if not spatial.empty else pd.DataFrame()
    if not spatial_primary.empty:
        fig, axes = plt.subplots(2, 2, figsize=(17, 10), constrained_layout=True, sharey=True)
        scheme_colors = {"S1_both_positive": "#4B8C87", "T01_empirical_CDF_breakpoint": "#D18A45"}
        for row, unit in enumerate(("matched_pixel", "pair")):
            for col, exposure in enumerate(EXPOSURES):
                ax = axes[row, col]
                part = spatial_primary.loc[spatial_primary["analysis_unit"].eq(unit) & spatial_primary["exposure"].eq(exposure)]
                positions = np.arange(len(OUTCOMES))
                offsets = {"S1_both_positive": -.17, "T01_empirical_CDF_breakpoint": .17}
                for scheme, group in part.groupby("scheme", sort=False):
                    values = [pd.to_numeric(group.loc[group["outcome"].eq(outcome), "spatial_test_r_squared_selected"], errors="coerce").dropna().to_numpy() for outcome in OUTCOMES]
                    chosen = [(index, value) for index, value in enumerate(values) if len(value)]
                    if not chosen:
                        continue
                    ax.boxplot([value for _, value in chosen], positions=[positions[index] + offsets[scheme] for index, _ in chosen],
                               widths=.28, showfliers=False, patch_artist=True,
                               boxprops={"facecolor": scheme_colors[scheme], "alpha": .74, "edgecolor": "white"},
                               medianprops={"color": "#253542", "linewidth": 1.1},
                               whiskerprops={"color": scheme_colors[scheme]}, capprops={"color": scheme_colors[scheme]})
                ax.axhline(0, color="#52636E", linewidth=.8, linestyle="--")
                ax.set_xticks(positions, [OUTCOME_LABELS[name] for name in OUTCOMES], rotation=35, ha="right")
                ax.set_ylabel("5 km 空间测试 R²" if col == 0 else "")
                ax.set_title(f"{'匹配像元' if unit == 'matched_pixel' else '配对差值'} · {EXPOSURE_LABELS[exposure]}", loc="left", fontweight="bold")
                ax.grid(axis="y", linewidth=.5)
        handles = [Line2D([0], [0], color=color, linewidth=7, alpha=.74, label=SCHEME_LABELS[name]) for name, color in scheme_colors.items()]
        fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(.5, -.015), frameon=False)
        fig.suptitle("M4 的 5 km 整块空间留出表现（十次划分）", x=.01, y=1.025, ha="left", fontsize=15, fontweight="bold")
        manifest.append(_save(fig, figure_root, "V02_matched_unit_spatial_validation", spatial_primary,
            "每个箱线图汇总十次 5 km 空间整块划分的测试 R²。模型族仅依据空间验证块选择；训练像元按 500 m 缓冲剔除邻近留出边界。匹配像元端点按 cell_id 去重，配对端点须位于同一 5 km 块。",
            scientific_question="低值筛选后，非线性响应能否在留出空间块保持？",
            experiment="M4 空间分块验证", figure_type="空间验证箱线图"))

    # Spatial footprint of a fixed-original M4 rematch.
    key = ("M4", "Aa_20yr", "T01_fixed_original")
    map_pairs = pairs_by_key.get(key, pd.DataFrame())
    if not map_pairs.empty:
        high = set(map_pairs["high_erosion_cell_id"].dropna().tolist())
        low = set(map_pairs["low_erosion_cell_id"].dropna().tolist())
        endpoint_ids = high | low
        endpoint = cells.loc[cells["cell_id"].isin(endpoint_ids), ["cell_id", "x", "y", "SoilType"]].drop_duplicates("cell_id").copy()
        endpoint["端点角色"] = np.select(
            [endpoint["cell_id"].isin(high & low), endpoint["cell_id"].isin(high), endpoint["cell_id"].isin(low)],
            ["同时作为高低端", "高侵蚀端", "低侵蚀端"], default="未识别",
        )
        eligible = cells.loc[cells["valid_cell"].fillna(False).astype(bool) & cells["SoilType"].isin(SOIL_TYPE_LABELS), ["x", "y"]]
        if len(eligible) > 20000:
            eligible = eligible.sample(n=20000, random_state=20261029)
        fig, ax = plt.subplots(figsize=(10.5, 8.2), constrained_layout=True)
        ax.scatter(eligible["x"] / 1000, eligible["y"] / 1000, s=.3, color="#CDD5DA", alpha=.28, linewidths=0, rasterized=True, label="主土类耕地展示样本")
        palette = {"高侵蚀端": "#C85D4E", "低侵蚀端": "#3977A8", "同时作为高低端": "#8A6BA8"}
        for role, marker in (("高侵蚀端", "^"), ("低侵蚀端", "o"), ("同时作为高低端", "s")):
            part = endpoint.loc[endpoint["端点角色"].eq(role)]
            if not part.empty:
                ax.scatter(part["x"] / 1000, part["y"] / 1000, s=5, marker=marker, color=palette[role], alpha=.78,
                           linewidths=0, rasterized=True, label=f"{role}（{len(part):,} 个唯一像元）")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("投影坐标东向（千米）")
        ax.set_ylabel("投影坐标北向（千米）")
        ax.set_title("20 年侵蚀 CDF 阈值后 M4 固定原组重配端点", loc="left", fontweight="bold")
        ax.grid(linewidth=.45, alpha=.6)
        ax.legend(loc="upper left", frameon=True, framealpha=.94, facecolor="white")
        x_span = (endpoint["x"].max() - endpoint["x"].min()) / 1000
        x0 = endpoint["x"].min() / 1000 + x_span * .06
        y0 = endpoint["y"].min() / 1000 + (endpoint["y"].max() - endpoint["y"].min()) / 1000 * .06
        ax.plot([x0, x0 + 5], [y0, y0], color="white", linewidth=4, zorder=5)
        ax.plot([x0, x0 + 5], [y0, y0], color="#253542", linewidth=1.8, zorder=6)
        ax.text(x0 + 2.5, y0 + .3, "5 千米", ha="center", va="bottom", fontsize=8,
                bbox={"boxstyle": "round,pad=.15", "facecolor": "white", "edgecolor": "none", "alpha": .9})
        ax.annotate("N", xy=(.94, .92), xytext=(.94, .80), xycoords="axes fraction", ha="center", va="center",
                    fontsize=10, fontweight="bold", arrowprops={"arrowstyle": "-|>", "color": "#253542", "lw": 1.1})
        fig.text(.02, -.015, "土壤类型：26 冲积土、27 潮土、42 脱潮土。背景点为固定展示抽样，不表示连续侵蚀面。", fontsize=8, color="#52636E")
        export = endpoint.merge(cells[["cell_id", "Aa_20yr"]], on="cell_id", how="left", validate="one_to_one")
        export["scheme"] = "T01_fixed_original"
        export["method"] = "M4"
        manifest.append(_save(fig, figure_root, "S05_matched_endpoint_spatial_footprint", export,
            f"地图展示 M4 在两端 Aa_20yr 均高于 CDF 分布阈值 {threshold_by_exposure.get('Aa_20yr', float('nan')):.3f} 后、固定原环境组重新匹配的唯一端点像元。点按 cell_id 去重；符号标记端点在配对中的高低侵蚀角色。背景为主土类有效耕地的固定展示样本。",
            scientific_question="阈值后重新匹配的高低侵蚀端点在研究区如何分布？",
            experiment="M4 固定原组阈值后重配", figure_type="匹配端点空间图"))

    # Counts contrast pair multiplicity and unique raster-cell support.
    if not populations.empty:
        fig, axes = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True, sharey="row")
        scheme_order = list(SCHEME_LABELS)
        x = np.arange(len(scheme_order))
        for row, exposure in enumerate(EXPOSURES):
            for col, method in enumerate(("M3", "M4")):
                ax = axes[row, col]
                part = populations.loc[populations["exposure"].eq(exposure) & populations["method"].eq(method)].set_index("scheme").reindex(scheme_order)
                ax.bar(x, part["n_pairs"], color="#DDE6EA", edgecolor="white", label="匹配对数")
                ax.plot(x, part["n_unique_matched_endpoints"], color="#3977A8", marker="o", linewidth=1.8, label="去重端点像元数")
                ax.set_xticks(x, [SCHEME_LABELS[name] for name in scheme_order], rotation=28, ha="right")
                ax.set_title(f"{EXPOSURE_LABELS[exposure]} · {method}", loc="left", fontweight="bold")
                ax.set_ylabel("数量（个）")
                ax.grid(axis="y", linewidth=.5)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(.5, -.015), frameon=False)
        fig.suptitle("匹配对数与唯一匹配端点像元数", x=.01, y=1.025, ha="left", fontsize=15, fontweight="bold")
        manifest.append(_save(fig, figure_root, "Q03_pair_and_unique_endpoint_counts", populations,
            "柱形表示匹配对数，折线表示高、低侵蚀端点按 cell_id 合并后的唯一像元数。一个像元可能参与多个匹配对，因此两种数量是不同分析单位；不得把配对行数当作独立像元数。",
            scientific_question="各筛选与重配方案保留了多少配对与独立栅格端点？",
            experiment="匹配样本支持量比较", figure_type="样本量比较图"))
    return manifest


def _localized_comparison_table(frame: pd.DataFrame) -> str:
    labels = {
        "analysis_unit": {"matched_pixel": "唯一匹配像元", "pair": "匹配配对差值"},
        "exposure": EXPOSURE_LABELS,
        "method": {"M3": "M3 匹配方案", "M4": "M4 匹配方案"},
        "outcome": OUTCOME_LABELS,
        "best_apparent_model_family": FAMILY_LABELS,
    }
    headers = {
        "analysis_unit": "分析单位", "exposure": "侵蚀时期", "method": "匹配方案",
        "outcome": "健康指标", "n": "有效样本量", "n_unique_matched_endpoints": "唯一端点像元数",
        "best_apparent_model_family": "最高表观拟合模型",
        "best_apparent_r_squared": "最高同样本表观 R²", "spearman_rho": "Spearman 秩相关系数",
        "spatial_test_r2_median": "空间测试 R² 中位数", "n_spatial_splits": "空间划分次数",
    }
    result = frame.copy()
    for column, mapping in labels.items():
        if column in result:
            result[column] = result[column].map(lambda value: mapping.get(value, value))
    result = result.rename(columns={column: headers[column] for column in result.columns if column in headers})
    result = result.fillna("—")
    return result.to_markdown(index=False, floatfmt=".3f")


def _write_report(
    run_dir: Path, run_id: str, thresholds: dict[str, float], populations: pd.DataFrame,
    models: pd.DataFrame, associations: pd.DataFrame, spatial: pd.DataFrame,
    comparisons: pd.DataFrame, figures: list[dict[str, str]], distribution_audit: pd.DataFrame,
) -> None:
    best = comparisons.loc[comparisons["scheme"].eq("T01_empirical_CDF_breakpoint")].copy()
    if not best.empty:
        top = best.sort_values("best_apparent_r_squared", ascending=False).groupby(
            ["analysis_unit", "exposure", "method", "outcome"], sort=False, dropna=False
        ).head(1)
        columns = [name for name in ("analysis_unit", "exposure", "method", "outcome", "n", "n_unique_matched_endpoints", "best_apparent_model_family", "best_apparent_r_squared", "spearman_rho", "spatial_test_r2_median", "n_spatial_splits") if name in top]
        results_table = _localized_comparison_table(top[columns])
    else:
        results_table = "（无可估计结果）"
    # Keep a compact method-separated population summary in the report body.
    counts = populations[["exposure", "method", "scheme_label", "n_pairs", "n_unique_matched_endpoints"]].copy()
    report = [
        "# 低侵蚀阈值后的匹配像元与配对响应补充分析",
        "",
        "## 研究范围",
        "",
        f"本补充运行编号：`{run_id}`；模式为 `EXPLORATORY_ORACLE`，`outcome_guided_selection=true`。沿用已计算的 Aa_20yr/Aa_2020 和 Phase F M3/M4 配对；未重算或验证 RUSLE 因子乘积，也未重跑 PCNM。健康数据为预测栅格值，像元数不等于独立实测样本数。",
        "",
        "分布阈值来自既有低侵蚀运行中的经验 CDF 折点：20 年平均 Aa 阈值为 **{:.6f}**，2020 年 Aa 阈值为 **{:.6f}**。配对样本要求两端均高于各自时期阈值。结果分别按匹配像元（端点 cell_id 去重）和匹配配对（每一配对差值一行）统计。".format(thresholds["Aa_20yr"], thresholds["Aa_2020"]),
        "",
        "## 群体规模",
        "",
        "|时期|方法|群体方案|匹配对数|唯一端点像元数|",
        "|---|---|---|---:|---:|",
    ]
    for row in counts.itertuples(index=False):
        report.append(f"|{EXPOSURE_LABELS.get(row.exposure,row.exposure)}|{row.method}|{row.scheme_label}|{int(row.n_pairs):,}|{int(row.n_unique_matched_endpoints):,}|")
    report.extend([
        "",
        "配对行数可因同一 cell_id 多次参与而高于唯一端点数。本补充的 matched-pixel 分析先把所有高低端点合并，再按 cell_id 去重；没有把重复匹配参与次数当成独立像元。",
        "",
        "## 正侵蚀值分布精度摘要",
        "",
        "以下分位数仅在正侵蚀值中计算；零值单独计数，不混入正值分布。重复比例表示在指定小数位四舍五入后、属于至少一个重复值组的正值像元比例。",
        "",
        "|时期|零值数|正值数|正值最小值|P1|P5|P10|P25|P50|P75|P90|P95|P99|正值最大值|重复比例 0 位|3 位|6 位|9 位|12 位|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in distribution_audit.itertuples(index=False):
        report.append(
            f"|{EXPOSURE_LABELS.get(row.exposure,row.exposure)}|{int(row.n_zero):,}|{int(row.n_positive):,}|"
            f"{row.positive_min:.6f}|{row.positive_q01:.6f}|{row.positive_q05:.6f}|{row.positive_q10:.6f}|"
            f"{row.positive_q25:.6f}|{row.positive_q50:.6f}|{row.positive_q75:.6f}|{row.positive_q90:.6f}|"
            f"{row.positive_q95:.6f}|{row.positive_q99:.6f}|{row.positive_max:.6f}|"
            f"{row.fraction_repeated_0dp:.2%}|{row.fraction_repeated_3dp:.2%}|{row.fraction_repeated_6dp:.2%}|"
            f"{row.fraction_repeated_9dp:.2%}|{row.fraction_repeated_12dp:.2%}|"
        )
    report.extend([
        "",
        "完整精度字段见 `tables/positive_exposure_distribution_audit.csv`。",
        "",
        "## 八类响应模型",
        "",
        "对 SHI 与八项健康指标分别比较线性、二次、自然样条、加性样条（GAM 近似）、Huber、随机森林、极端随机树和直方图梯度提升。匹配像元模型为 `Y = f(log1p(Aa))`；配对模型为 `ΔY = f(Δlog1p(Aa))`。模型族先在同样本上比较表观 R²，因此属于 Oracle 探索；按表观 R²挑中的曲线不能解释为空间外推表现。",
        "",
        "### CDF 阈值方案的最高表观 R²与空间测试摘要",
        "",
        results_table,
        "",
        "### 阈值前后比较表",
        "",
        "完整逐指标结果、最高表观 R²变化、Spearman 方向及空间测试摘要见 `tables/threshold_model_comparison.csv`。跨方案比较改变了样本与估计对象，R²变化不等价于模型本身改善。",
        "",
        "## 空间验证",
        "",
        f"对 M4 的两端正值和 CDF 阈值筛选方案，分别在匹配像元及匹配配对上进行了 {int(spatial['split'].nunique()) if not spatial.empty and 'split' in spatial else 0} 个固定 5 km 整块空间划分；训练样本按 500 m 缓冲剔除邻近留出块，八类模型在验证块选择，并在测试块评分。配对分析要求两端位于同一 5 km 块。本补充空间重跑仅覆盖 M4；M3 的对应单元不作本轮空间测试解释。分布阈值由全区侵蚀分布确定，故空间测试仅检验其下游响应建模，不构成完整阈值发现流程的独立验证。",
        "",
        "## 主要图表",
        "",
    ])
    for item in figures:
        report.append(f"- [{item['title']}](../figures/png/{item['id']}.png)；源数据：`figures/data/{item['id']}_source.csv`。")
    report.extend([
        "",
        "## 解释边界",
        "",
        "- 连续栅格像元存在空间依赖；空间块测试比随机拆分更保守，但仍是内部空间留出，不是外部独立验证。",
        "- 匹配像元、匹配配对与全域像元是不同统计单位，必须分别呈现。",
        "- 表观最高 R²与最佳空间测试 R²分开保存；Oracle 结果用于探索，不构成预先注册的正式结论。",
        "- 土壤类型代码在地图中标注为：26 冲积土、27 潮土、42 脱潮土。",
        "",
    ])
    (run_dir / "reports" / "MATCHED_UNIT_SUPPLEMENT_REPORT.md").write_text("\n".join(report), encoding="utf-8")


def run_matched_unit_supplement(project_root: Path, config_path: Path) -> Path:
    root = Path(project_root).resolve()
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = root / config_path
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("mode") != "EXPLORATORY_ORACLE" or config.get("outcome_guided_selection") is not True:
        raise ValueError("matched-unit supplement requires EXPLORATORY_ORACLE and outcome_guided_selection=true")
    if set(config.get("model_families", [])) != set(MODEL_FAMILIES):
        raise ValueError("configured model families must match the eight implemented response families")
    run_id = str(config["run_id"])
    output_root = Path(config["output_root"])
    if output_root.is_absolute() or ".." in output_root.parts or output_root.parts[:2] != ("outputs", "exploratory_oracle"):
        raise ValueError("output_root must remain within outputs/exploratory_oracle")
    run_dir = root / output_root / run_id
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing run: {run_dir}")
    for name in ("config", "tables", "associations", "models", "spatial_validation", "figures/png", "figures/svg", "figures/pdf", "figures/data", "figures/captions", "reports"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    snapshot = yaml.safe_dump(config, allow_unicode=True, sort_keys=False)
    (run_dir / "config.yaml").write_text(snapshot, encoding="utf-8")
    (run_dir / "config/config.yaml").write_text(snapshot, encoding="utf-8")
    try:
        baseline_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        baseline_sha = None

    low_run = root / output_root / str(config["low_erosion_run_id"])
    low_config = yaml.safe_load((low_run / "config.yaml").read_text(encoding="utf-8"))
    cells, _, input_paths = _load_inputs(root, low_config)
    thresholds_frame = pd.read_csv(low_run / "thresholds/distribution_threshold_candidates.csv")
    thresholds = {}
    for exposure in EXPOSURES:
        primary = thresholds_frame.loc[thresholds_frame["exposure"].eq(exposure) & thresholds_frame["is_primary"].fillna(False)]
        if primary.empty:
            raise ValueError(f"no primary distribution-derived threshold for {exposure}")
        thresholds[exposure] = float(primary.iloc[0]["threshold_original"])
    distribution_audit = _positive_distribution_audit(cells)

    model_rows: list[pd.DataFrame] = []
    association_rows: list[dict[str, Any]] = []
    population_rows: list[dict[str, Any]] = []
    pair_spatial_rows: list[pd.DataFrame] = []
    pixel_spatial_rows: list[pd.DataFrame] = []
    pairs_for_figures: dict[tuple[str, str, str], pd.DataFrame] = {}
    sample_size = int(config["models"]["fixed_sample_size"])
    seeds = tuple(int(value) for value in config["models"]["spatial_validation_seeds"])
    buffer_m = int(config["models"]["spatial_validation_buffer_m"])

    for exposure in EXPOSURES:
        threshold = thresholds[exposure]
        for method in config["matching_methods"]:
            paths = _source_populations(low_run, method, exposure)
            populations_for_exposure = {scheme: _read_table(path) for scheme, path in paths.items()}
            baseline = populations_for_exposure["S0_all"]
            baseline_n = len(baseline)
            baseline_unique_n = int(pd.unique(pd.concat([baseline["high_erosion_cell_id"], baseline["low_erosion_cell_id"]], ignore_index=True).dropna()).size)
            for scheme, population in populations_for_exposure.items():
                population_rows.append(_population_summary(
                    population, method=method, exposure=exposure, scheme=scheme,
                    threshold=threshold if scheme.startswith("T01_") else None,
                    baseline_pairs=baseline_n, baseline_unique_endpoints=baseline_unique_n,
                ))
                pairs_for_figures[(method, exposure, scheme)] = population
                for outcome in OUTCOMES:
                    matched_assoc = matched_pixel_association(
                        population, cells, exposure=exposure, outcome=outcome, method=method,
                        scheme=scheme, threshold=threshold if scheme.startswith("T01_") else None,
                        seed=int(config["seed"]),
                    )
                    matched_assoc["outcome_guided_selection"] = True
                    association_rows.append(matched_assoc)
                    pair_assoc = paired_association(
                        population, exposure=exposure, outcome=outcome, method=method,
                        scheme=scheme, threshold=threshold if scheme.startswith("T01_") else None,
                        seed=int(config["seed"]),
                    )
                    pair_assoc["outcome_guided_selection"] = True
                    association_rows.append(pair_assoc)
                    model_rows.append(matched_pixel_model_trials(
                        population, cells, exposure=exposure, outcome=outcome, method=method,
                        scheme=scheme, threshold=threshold if scheme.startswith("T01_") else None,
                        seed=int(config["seed"]), sample_size=sample_size,
                    ))
                    model_rows.append(paired_model_trials(
                        population, exposure=exposure, outcome=outcome, method=method,
                        scheme=scheme, threshold=threshold if scheme.startswith("T01_") else None,
                        seed=int(config["seed"]), sample_size=sample_size,
                    ))

            # Validate the primary M4 endpoint and pair populations using the same 5 km split seeds.
            if method in config["models"]["spatial_methods"]:
                for scheme in config["models"]["spatial_schemes"]:
                    population = populations_for_exposure[scheme]
                    threshold_arg = threshold if scheme.startswith("T01_") else None
                    for outcome in OUTCOMES:
                        pixel_result = matched_pixel_spatial_validation(
                            population, cells, exposure=exposure, outcome=outcome, method=method,
                            scheme=scheme, threshold=threshold_arg, seeds=seeds,
                            buffer_m=buffer_m, sample_size=sample_size,
                        )
                        pair_result = paired_spatial_validation(
                            population, exposure=exposure, outcome=outcome, method=method,
                            scheme=scheme, threshold=threshold_arg, seeds=seeds,
                            buffer_m=buffer_m, sample_size=sample_size,
                        )
                        if not pixel_result.empty:
                            pixel_result["n_unique_matched_endpoints"] = int(matched_pixel_population(
                                population, cells, exposure=exposure, method=method)["cell_id"].nunique())
                            pixel_spatial_rows.append(pixel_result)
                        if not pair_result.empty:
                            pair_result["analysis_unit"] = "pair"
                            pair_spatial_rows.append(pair_result)
            del populations_for_exposure

    model_trials = pd.concat(model_rows, ignore_index=True, sort=False) if model_rows else pd.DataFrame()
    associations = _add_adjusted_p_values(pd.DataFrame.from_records(association_rows))
    populations = pd.DataFrame.from_records(population_rows)
    pixel_spatial = pd.concat(pixel_spatial_rows, ignore_index=True, sort=False) if pixel_spatial_rows else pd.DataFrame()
    pair_spatial = pd.concat(pair_spatial_rows, ignore_index=True, sort=False) if pair_spatial_rows else pd.DataFrame()
    spatial = pd.concat([pixel_spatial, pair_spatial], ignore_index=True, sort=False)
    comparisons = _comparison_table(model_trials, associations, spatial)

    _write_table(populations, run_dir / "tables", "matched_population_summary")
    _write_table(distribution_audit, run_dir / "tables", "positive_exposure_distribution_audit")
    _write_table(associations, run_dir / "associations", "matched_unit_associations")
    _write_table(model_trials, run_dir / "models", "matched_unit_model_trials")
    _write_table(comparisons, run_dir / "models", "threshold_model_comparison")
    _write_table(pixel_spatial, run_dir / "spatial_validation", "matched_pixel_spatial_validation")
    _write_table(pair_spatial, run_dir / "spatial_validation", "matched_pair_spatial_validation")
    _write_table(spatial, run_dir / "spatial_validation", "matched_unit_spatial_validation")

    figures = _make_figures(run_dir / "figures", cells, pairs_for_figures, populations,
                            associations, model_trials, spatial, thresholds)
    manifest = {
        "run_id": run_id, "mode": "EXPLORATORY_ORACLE", "outcome_guided_selection": True,
        "figure_count": len(figures), "formats": ["png", "svg", "pdf"], "figures": figures,
    }
    (run_dir / "figures/figure_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_report(run_dir, run_id, thresholds, populations, model_trials, associations, spatial, comparisons, figures, distribution_audit)
    provenance = {
        "phase": "H-matched-unit-supplement", "run_id": run_id,
        "mode": "EXPLORATORY_ORACLE", "outcome_guided_selection": True,
        "input_run_id": config["input_run_id"], "phase_f_run_id": config["phase_f_run_id"],
        "phase_g_run_id": config["phase_g_run_id"], "low_erosion_run_id": config["low_erosion_run_id"],
        "git_commit_at_run": baseline_sha, "started_or_finished_at_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "packages": {name: _package_version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "statsmodels", "pyarrow")},
        "thresholds_original": thresholds,
        "analysis_units": ["matched_pixel_unique_cell_id", "matched_pair_difference"],
        "matched_pixel_rule": "unique union of high_erosion_cell_id and low_erosion_cell_id within each method, exposure, and selected pair population",
        "spatial_validation": {"block_size_m": int(config["models"]["spatial_validation_block_size_m"]), "buffer_m": buffer_m, "seeds": list(seeds), "methods": list(config["models"]["spatial_methods"]), "schemes": list(config["models"]["spatial_schemes"]), "model_families": list(MODEL_FAMILIES)},
        "figure_count": len(figures), "model_trial_rows": len(model_trials),
        "association_rows": len(associations), "spatial_validation_rows": len(spatial),
        "input_paths": {name: str(path) for name, path in input_paths.items()},
    }
    (run_dir / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (run_dir / "checkpoints.json").write_text(json.dumps({
        "matched_population_summary_complete": True, "matched_pixel_associations_complete": True,
        "eight_family_model_search_complete": True, "matched_unit_spatial_validation_complete": True,
        "figure_generation_complete": True,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return run_dir


def main(argv: list[str] | None = None) -> int:
    root = Path.cwd()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--config", type=Path, default=Path("config/low_erosion_matched_units.yaml"))
    args = parser.parse_args(argv)
    result = run_matched_unit_supplement(args.project_root, args.config)
    print(result.relative_to(args.project_root.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
