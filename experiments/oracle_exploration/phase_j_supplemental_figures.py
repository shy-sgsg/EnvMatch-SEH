"""Render additional Phase J figures from saved model and synthesis tables.

This script does not fit models. It reads the saved Phase J r10 result tables,
creates figures J-S, and places the already-rendered direction-score sensitivity
figure from its separate run into the Phase J figure set as figure T.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "outputs/exploratory_oracle/phase_j/phase_j_20261102_seed20261102_r10"
FIG_DIR = RUN / "figures/supplementary"
DATA_DIR = FIG_DIR / "data"
DIRECTION_RUN = ROOT / "outputs/exploratory_oracle/phase_j/phase_j_20261103_seed20261103_direction_r1"
CORRECTED_NINE_OUTCOME_GWR = ROOT / (
    "outputs/exploratory_oracle/phase_i/"
    "phase_i_20261101_corrected_cec_nine_outcome_refresh_r2/"
    "tables/full_sample_local_coefficients.parquet"
)
GWR_MGWR_DIFFERENCES = ROOT / (
    "outputs/exploratory_oracle/phase_i/"
    "phase_i_20261031_seed20261031_closure_r1/"
    "tables/gwr_mgwr_local_coefficient_differences.csv"
)

BLUE = "#326B8C"
TEAL = "#3B8C88"
GOLD = "#D38A32"
RED = "#B4564A"
PURPLE = "#7866A6"
GREY = "#667782"
METHOD_ORDER = ["piecewise_sem", "CB_SEM_lavaan_ML", "PLS_PM_ModeB_formative"]
METHOD_LABELS = {
    "piecewise_sem": "分段回归（5 km空间块区间）",
    "CB_SEM_lavaan_ML": "协方差型SEM（常规区间）",
    "PLS_PM_ModeB_formative": "形成性PLS-PM（未提供区间）",
}
OUTCOME_LABELS = {
    "SHI_candidate": "SHI",
    "z_SHI": "SHI",
    "physical": "物理维度",
    "chemical": "化学维度",
    "biological_activity": "生物活动维度",
    "Physical": "物理维度",
    "Chemical": "化学维度",
    "Biological": "生物活动维度",
    "SHI": "SHI",
    "OM": "OM",
    "SW": "SW",
}
MODEL_LABELS = {
    "GWR-LCR（九项结局）": "局部共线性收缩GWR（240 m单元）",
    "同样本标准GWR（770空间单元）": "标准GWR（770个空间单元）",
    "标准MGWR（770空间单元）": "标准MGWR（770个空间单元）",
}
ENV_LABELS = {
    "Elevation": "高程",
    "TWI": "地形湿度",
    "NDVI": "植被指数",
    "NDWI": "水分指数",
    "LST": "地表温度",
}
DATASET_LABELS = {
    "D1_common_env_case": "全域环境完整像元",
    "D2_240m_blocks": "240 m空间汇总单元",
    "D3_M4_unique_pixels": "环境匹配端点像元",
    "D4_M4_pair_differences": "环境匹配配对差值",
}
OUTCOME_ORDER = ["physical", "chemical", "biological_activity", "z_SHI"]
OUTCOME_COLORS = {
    "physical": BLUE,
    "chemical": GOLD,
    "biological_activity": TEAL,
    "z_SHI": PURPLE,
}
NINE_OUTCOME_ORDER = ["SHI_candidate", "AN", "OM", "CEC", "pH", "SW", "BD", "MBC", "RESP"]
NINE_OUTCOME_LABELS = {
    "SHI_candidate": "土壤健康指数（SHI）",
    "AN": "碱解氮（AN）",
    "OM": "有机质（OM）",
    "CEC": "阳离子交换量（CEC）",
    "pH": "酸碱度（pH）",
    "SW": "土壤含水量（SW）",
    "BD": "土壤容重（BD）",
    "MBC": "微生物量碳（MBC）",
    "RESP": "土壤呼吸（RESP）",
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Noto Sans CJK SC",
            "WenQuanYi Micro Hei",
            "Microsoft YaHei",
            "DejaVu Sans",
        ],
        "axes.unicode_minus": False,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 120,
        "savefig.dpi": 320,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.23,
        "grid.linewidth": 0.6,
    }
)


def read_csv(relative: str) -> pd.DataFrame:
    return pd.read_csv(RUN / relative)


def write_figure(
    letter: str,
    stem: str,
    title: str,
    explanation: str,
    fig: plt.Figure,
    source: pd.DataFrame,
) -> dict[str, str]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    png = FIG_DIR / f"{stem}.png"
    fig.savefig(png, bbox_inches="tight", facecolor="white")
    fig.savefig(png.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(png.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    source_path = DATA_DIR / f"{stem}_source.csv"
    source.to_csv(source_path, index=False, encoding="utf-8-sig")
    return {
        "letter": letter,
        "id": stem,
        "title": title,
        "explanation": explanation,
        "png": png.relative_to(RUN).as_posix(),
        "svg": png.with_suffix(".svg").relative_to(RUN).as_posix(),
        "pdf": png.with_suffix(".pdf").relative_to(RUN).as_posix(),
        "source": source_path.relative_to(RUN).as_posix(),
    }


def errorbar_x(ax: plt.Axes, row: pd.Series, y: float, color: str, **kwargs) -> None:
    beta = float(row["beta_standardized"])
    low = row.get("ci_low", np.nan)
    high = row.get("ci_high", np.nan)
    if pd.notna(low) and pd.notna(high):
        ax.errorbar(
            beta,
            y,
            xerr=[[beta - float(low)], [float(high) - beta]],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=2.6,
            markersize=5,
            linewidth=1.4,
            **kwargs,
        )
    else:
        ax.plot(beta, y, "o", color=color, markersize=5, **kwargs)


def figure_j() -> dict[str, str]:
    d = read_csv("results/SEM_PATH_COEFFICIENTS.csv")
    d = d.loc[
        d["method"].eq("piecewise_sem")
        & d["structure"].eq("S2_direct_plus_indirect_SHI")
        & d["dataset"].isin(DATASET_LABELS)
        & d["control"].isin(["M0", "M1", "M2", "M3"])
        & d["from"].eq("z_E")
        & d["to"].eq("z_SHI")
    ].copy()
    controls = ["M0", "M1", "M2", "M3"]
    control_labels = [
        "M0 不控制环境",
        "M1 核心环境变量",
        "M2 环境主成分",
        "M3 环境变量+空间趋势",
    ]
    colors = [BLUE, TEAL, GOLD, PURPLE]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2), sharex=True)
    for ax, dataset in zip(axes.flat, DATASET_LABELS):
        sub = d.loc[d["dataset"].eq(dataset)].copy()
        sub["control"] = pd.Categorical(sub["control"], controls, ordered=True)
        sub = sub.sort_values("control")
        ax.axvline(0, color="#4a5560", linestyle="--", linewidth=1)
        ys = np.arange(len(controls))
        for i, (_, row) in enumerate(sub.iterrows()):
            errorbar_x(ax, row, ys[i], colors[i])
        if len(sub):
            ax.plot(sub["beta_standardized"], ys, color="#8295a0", linewidth=1, zorder=0)
        ax.set_yticks(ys, control_labels)
        ax.invert_yaxis()
        n = int(sub["n"].median()) if len(sub) else 0
        ax.set_title(f"{DATASET_LABELS[dataset]}\n样本量 n={n:,}")
        ax.grid(axis="y", visible=False)
    fig.supxlabel("侵蚀至 SHI 的标准化直接路径 β（95% 空间块区间）")
    fig.suptitle("不同样本总体和控制层级下的 SHI 直接路径", y=1.02, fontsize=15)
    fig.tight_layout()
    return write_figure(
        "J",
        "J_sem_control_sensitivity",
        "不同控制条件下 SHI 直接路径的跨总体比较",
        "四个面板分别表示全域环境完整像元、240 m汇总单元、环境匹配端点和匹配对差值。M0至M3逐步改变环境/空间控制。横线为5 km空间块聚类区间；该图用于显示模型条件敏感性，不作因果解释。",
        fig,
        d,
    )


def figure_k() -> dict[str, str]:
    d = read_csv("results/SEM_PATH_COEFFICIENTS.csv")
    rows: list[pd.Series] = []
    for method in METHOD_ORDER:
        for outcome, target in [
            ("物理维度", "physical"),
            ("化学维度", "chemical"),
            ("生物活动维度", "biological_activity"),
            ("SHI", "z_SHI"),
        ]:
            structure = "S2_direct_plus_indirect_SHI" if outcome == "SHI" else "S1_parallel_dimensions"
            if method == "PLS_PM_ModeB_formative":
                source_name = "Erosion"
                target_name = {
                    "physical": "Physical",
                    "chemical": "Chemical",
                    "biological_activity": "Biological",
                    "z_SHI": "SHI",
                }[target]
            else:
                source_name = "z_E"
                target_name = target
            sub = d.loc[
                d["dataset"].eq("D1_common_env_case")
                & d["method"].eq(method)
                & d["structure"].eq(structure)
                & d["control"].eq("M0")
                & d["from"].eq(source_name)
                & d["to"].eq(target_name)
            ]
            if not sub.empty:
                item = sub.iloc[0].copy()
                item["outcome_label"] = outcome
                item["method_label"] = METHOD_LABELS[method]
                rows.append(item)
    plot = pd.DataFrame(rows)
    outcome_order = ["物理维度", "化学维度", "生物活动维度", "SHI"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    method_colors = {
        "piecewise_sem": BLUE,
        "CB_SEM_lavaan_ML": GOLD,
        "PLS_PM_ModeB_formative": TEAL,
    }
    for ax, outcome in zip(axes.flat, outcome_order):
        sub = plot.loc[plot["outcome_label"].eq(outcome)]
        ax.axvline(0, color="#4a5560", linestyle="--", linewidth=1)
        positions = []
        labels = []
        for method_i, method in enumerate(METHOD_ORDER):
            found = sub.loc[sub["method"].eq(method)]
            if found.empty:
                continue
            row = found.iloc[0]
            y = method_i
            errorbar_x(ax, row, y, method_colors[method])
            positions.append(y)
            labels.append(METHOD_LABELS[method])
        ax.set_yticks(positions, labels)
        ax.invert_yaxis()
        ax.set_title(outcome)
        ax.grid(axis="y", visible=False)
    fig.supxlabel("侵蚀至结局的标准化路径 β（有区间时显示95%区间）")
    fig.suptitle("同一全域样本与基础模型下的三种方法对照", y=1.02, fontsize=15)
    fig.tight_layout()
    return write_figure(
        "K",
        "K_sem_method_comparison",
        "三种结构模型方法的路径系数对照",
        "比较分段路径模型、协方差型SEM与形成性PLS-PM在同一全域样本上的侵蚀路径。分段模型区间按5 km空间块聚类；协方差型SEM区间按常规独立同分布假设计算；PLS-PM未提供区间，故只显示点估计。各模型的构念定义和估计假设不同，系数仅用于方向和量级的探索性对照。",
        fig,
        plot,
    )


def figure_l() -> dict[str, str]:
    fits = read_csv("results/SEM_MODEL_FIT_COMPARISON.csv")
    fits = fits.loc[
        fits["method"].eq("piecewise_sem")
        & fits["structure"].eq("S2_direct_plus_indirect_SHI")
        & fits["dataset"].isin(DATASET_LABELS)
        & fits["control"].isin(["M0", "M1", "M2", "M3"])
        & fits["endogenous"].eq("z_SHI")
    ].copy()
    delta = read_csv("results/SEM_EROSION_INCREMENTAL_CONTRIBUTION.csv")
    delta = delta.loc[
        delta["method"].eq("piecewise_sem")
        & delta["structure"].eq("S2_direct_plus_indirect_SHI")
        & delta["dataset"].isin(DATASET_LABELS)
        & delta["control"].isin(["M0", "M1", "M2", "M3"])
        & delta["outcome"].eq("z_SHI")
    ].copy()
    datasets = list(DATASET_LABELS)
    controls = ["M0", "M1", "M2", "M3"]
    fit_matrix = fits.pivot(index="dataset", columns="control", values="r_squared").reindex(index=datasets, columns=controls)
    delta_matrix = delta.pivot(index="dataset", columns="control", values="delta_r2_erosion_conditional").reindex(index=datasets, columns=controls)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), gridspec_kw={"width_ratios": [1, 1.12]})
    for ax, matrix, title, cmap, fmt in [
        (axes[0], fit_matrix, "SHI 模型 R²", "YlGnBu", ".3f"),
        (axes[1], delta_matrix, "加入侵蚀后的条件 ΔR²", "YlOrRd", ".4f"),
    ]:
        values = matrix.to_numpy(dtype=float)
        im = ax.imshow(values, aspect="auto", cmap=cmap)
        ax.set_xticks(range(4), ["M0", "M1", "M2", "M3"])
        ax.set_yticks(range(4), [DATASET_LABELS[x] for x in datasets])
        ax.set_title(title)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                if np.isfinite(values[i, j]):
                    ax.text(j, i, format(values[i, j], fmt), ha="center", va="center", fontsize=8, color="#1f2933")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle("不同样本总体下的模型拟合与侵蚀增量并不等价", fontsize=15, y=1.02)
    fig.tight_layout()
    plotted = fits[["dataset", "control", "n", "r_squared"]].merge(
        delta[["dataset", "control", "delta_r2_erosion_conditional"]], on=["dataset", "control"], how="outer"
    )
    plotted["dataset_label"] = plotted["dataset"].map(DATASET_LABELS)
    return write_figure(
        "L",
        "L_sem_fit_increment_sensitivity",
        "不同总体下 SHI 模型拟合度与侵蚀增量",
        "左图为SHI模型内R²，右图为在同一控制条件上加入侵蚀变量后得到的条件ΔR²。M0不控制环境；M1加入高程、地形湿度、植被、地表温度和土类；M2使用环境主成分；M3在核心环境变量上增加空间位置趋势。更高的总拟合度不等于侵蚀具有更大的独立增量；所有数值均为表观模型内拟合。",
        fig,
        plotted,
    )


def figure_m() -> dict[str, str]:
    d = read_csv("spatial_synthesis/local_coefficient_distribution_summary.csv")
    d = d.loc[d["outcome"].isin(["SHI_candidate", "OM", "SW"])].copy()
    outcomes = ["SHI_candidate", "OM", "SW"]
    models = list(MODEL_LABELS)
    colors = [BLUE, GOLD, TEAL]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 5.2), sharex=True)
    for ax, outcome in zip(axes, outcomes):
        sub = d.loc[d["outcome"].eq(outcome)].set_index("model_family").reindex(models).reset_index()
        for pos, (_, row) in enumerate(sub.iterrows()):
            ax.hlines(pos, row["p5_beta"], row["p95_beta"], color=colors[pos], linewidth=1.3, alpha=0.9)
            ax.hlines(pos, row["p25_beta"], row["p75_beta"], color=colors[pos], linewidth=5, alpha=0.65)
            ax.plot(row["median_beta"], pos, "o", color=colors[pos], markersize=6)
        ax.axvline(0, color="#4a5560", linestyle="--", linewidth=1)
        ax.set_yticks(range(len(models)), ["局部岭收缩GWR\n（240 m）", "标准GWR\n（770单元）", "标准MGWR\n（770单元）"])
        ax.invert_yaxis()
        ax.set_title(OUTCOME_LABELS[outcome])
        ax.grid(axis="y", visible=False)
    fig.supxlabel("局部标准化系数 β（外线 P5–P95；粗线 P25–P75；圆点为中位数）")
    fig.suptitle("SHI、OM、SW 在三种局部模型中的系数空间分布", y=1.02, fontsize=15)
    fig.tight_layout()
    return write_figure(
        "M",
        "M_local_coefficient_distributions",
        "三项重点结局的局部系数空间分布",
        "比较SHI、OM和SW在局部共线性收缩GWR、同样本标准GWR及标准MGWR中的局部系数分布。线段表示空间单元间的系数分位数范围，不是置信区间；三类模型使用的空间单元数和模型设定不同。",
        fig,
        d,
    )


def figure_n() -> dict[str, str]:
    d = read_csv("spatial_synthesis/CROSS_INDICATOR_SPATIAL_OVERLAP.csv")
    model_order = ["GWR-LCR", "GWR", "MGWR"]
    pair_order = [("SHI_candidate", "OM"), ("SHI_candidate", "SW"), ("OM", "SW")]
    pair_labels = ["SHI—OM", "SHI—SW", "OM—SW"]
    matrix = np.full((len(pair_order), len(model_order)), np.nan)
    for i, (a, b) in enumerate(pair_order):
        for j, model in enumerate(model_order):
            row = d.loc[
                d["model_family"].eq(model)
                & d["outcome_a"].eq(a)
                & d["outcome_b"].eq(b)
            ]
            if row.empty:
                row = d.loc[
                    d["model_family"].eq(model)
                    & d["outcome_a"].eq(b)
                    & d["outcome_b"].eq(a)
                ]
            if not row.empty:
                matrix[i, j] = float(row.iloc[0]["decile_overlap_jaccard"])
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    im = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(3), ["局部岭收缩GWR", "标准GWR", "标准MGWR"])
    ax.set_yticks(range(3), pair_labels)
    ax.set_title("局部系数最负的十分位区域有多少空间重合")
    for i in range(3):
        for j in range(3):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", color="#172b3a", fontsize=10)
    colorbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    colorbar.set_label("Jaccard 重合率（0–1）")
    ax.set_xlabel("局部模型")
    ax.set_ylabel("结局组合")
    fig.tight_layout()
    return write_figure(
        "N",
        "N_local_response_overlap",
        "三项结局的强负向局部区域重合度",
        "每格比较两个结局各自局部系数最低十分位的空间单元集合。Jaccard值越高，两个低系数区域的空间重合越多；它不表示两种结局的系数相等，也不构成显著性检验。",
        fig,
        d,
    )


def figure_o() -> dict[str, str]:
    d = read_csv("spatial_synthesis/LOCAL_BETA_ENVIRONMENT_ASSOCIATION.csv")
    env_order = ["Elevation", "TWI", "NDVI", "NDWI", "LST"]
    model_order = ["GWR-LCR", "GWR", "MGWR"]
    outcome_order = ["SHI_candidate", "OM", "SW"]
    d = d.loc[d["environment_variable"].isin(env_order)].copy()
    row_order = [(m, o) for m in model_order for o in outcome_order]
    row_labels = [f"{m}\n{OUTCOME_LABELS[o]}" for m, o in row_order]
    matrix = np.full((len(row_order), len(env_order)), np.nan)
    for i, (model, outcome) in enumerate(row_order):
        for j, env in enumerate(env_order):
            row = d.loc[
                d["model_family"].eq(model)
                & d["outcome"].eq(outcome)
                & d["environment_variable"].eq(env)
            ]
            if not row.empty:
                matrix[i, j] = float(row.iloc[0]["spearman_local_beta_environment_rho"])
    fig, ax = plt.subplots(figsize=(9.3, 7.0))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(env_order)), [ENV_LABELS[x] for x in env_order])
    ax.set_yticks(range(len(row_order)), row_labels)
    ax.set_title("局部侵蚀系数与连续环境变量的 Spearman 相关")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:+.2f}", ha="center", va="center", fontsize=8, color="#172b3a")
    for y in [2.5, 5.5]:
        ax.axhline(y, color="white", linewidth=2)
    colorbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    colorbar.set_label("Spearman ρ")
    ax.set_xlabel("环境变量（连续变量）")
    ax.set_ylabel("局部模型与结局")
    fig.tight_layout()
    return write_figure(
        "O",
        "O_local_beta_environment_association",
        "局部系数与连续环境变量的秩相关",
        "热图展示局部侵蚀系数与高程、地形湿度、植被、水分指数和地表温度的Spearman相关。正负号表示空间高低排序的共同方向；相关不等于环境变量造成了局部系数变化。",
        fig,
        d,
    )


def figure_p() -> dict[str, str]:
    d = read_csv("spatial_synthesis/LOCAL_BETA_ENVIRONMENT_CONTRASTS.csv")
    d = d.loc[
        d["model_family"].eq("GWR-LCR")
        & d["environment_variable"].isin(ENV_LABELS)
    ].copy()
    env_order = ["Elevation", "TWI", "NDVI", "NDWI", "LST"]
    outcomes = ["SHI_candidate", "OM", "SW"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.2), sharex=True)
    colors = [BLUE, GOLD, TEAL]
    for ax, outcome, color in zip(axes, outcomes, colors):
        sub = d.loc[d["outcome"].eq(outcome)].set_index("environment_variable").reindex(env_order).reset_index()
        ys = np.arange(len(env_order))
        vals = sub["stronger_minus_weaker_beta"].astype(float)
        ax.axvline(0, color="#4a5560", linestyle="--", linewidth=1)
        ax.scatter(vals, ys, color=color, s=34, zorder=3)
        for y, value in zip(ys, vals):
            if pd.notna(value):
                ax.plot([0, value], [y, y], color=color, alpha=0.45, linewidth=2)
        ax.set_yticks(ys, [ENV_LABELS[x] for x in env_order])
        ax.invert_yaxis()
        ax.set_title(OUTCOME_LABELS[outcome])
        ax.grid(axis="y", visible=False)
    fig.supxlabel("强负向环境组中位 β − 弱负向环境组中位 β")
    fig.suptitle("环境分组两端的局部系数中位数差（局部收缩GWR）", y=1.02, fontsize=15)
    fig.tight_layout()
    return write_figure(
        "P",
        "P_local_beta_environment_contrasts",
        "环境强弱组的局部系数描述性差异",
        "每个点是按各环境变量分组后，局部系数最负组与最弱负组的中位数之差。越负表示两个已挑出的组之间差异越大；分组选择参考了结果本身，因此只能作为探索性定位，不是环境因果效应或正式组间显著性检验。",
        fig,
        d,
    )


def figure_q() -> dict[str, str]:
    d = read_csv("results/SEM_ENVIRONMENT_GROUP_COMPARISON.csv")
    d = d.loc[
        d["dataset"].str.startswith("D7_soil")
        & d["method"].eq("piecewise_sem")
        & d["structure"].eq("S1_parallel_dimensions")
        & d["control"].eq("M1")
        & d["from"].eq("z_E")
        & d["to"].isin(OUTCOME_ORDER)
    ].copy()
    soil_order = ["26 冲积土", "27 潮土", "42 脱潮土"]
    twi_order = ["q1", "q2", "q3", "q4"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 6), sharex=True)
    for ax, soil in zip(axes, soil_order):
        sub = d.loc[d["soil_label"].eq(soil)]
        n_min, n_max = int(sub["n"].min()), int(sub["n"].max())
        ax.axvline(0, color="#4a5560", linestyle="--", linewidth=1)
        for q_i, twi in enumerate(twi_order):
            row_group = sub.loc[sub["twi_group"].eq(twi)]
            for outcome_i, outcome in enumerate(OUTCOME_ORDER[:3]):
                row = row_group.loc[row_group["to"].eq(outcome)]
                if row.empty:
                    continue
                y = q_i + (outcome_i - 1) * 0.18
                errorbar_x(ax, row.iloc[0], y, OUTCOME_COLORS[outcome])
        ax.set_yticks(range(4), ["最低湿度组", "较低湿度组", "较高湿度组", "最高湿度组"])
        ax.invert_yaxis()
        ax.set_title(f"{soil}\n每组样本量 {n_min:,}–{n_max:,}")
        ax.grid(axis="y", visible=False)
    handles = [
        plt.Line2D([0], [0], marker="o", color=OUTCOME_COLORS[key], linestyle="", label=label)
        for key, label in [
            ("physical", "物理维度"),
            ("chemical", "化学维度"),
            ("biological_activity", "生物活动维度"),
        ]
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.supxlabel("侵蚀至健康维度的标准化路径 β（95%空间块区间）")
    fig.suptitle("不同土类和地形湿度组中的路径估计", y=1.02, fontsize=15)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return write_figure(
        "Q",
        "Q_soil_twi_dimension_paths",
        "不同土类和地形湿度组中的健康维度路径",
        "每个面板对应一个土类，纵向按该土类内部的地形湿度四分位从低到高排列，颜色区分物理、化学和生物活动维度。区间按5 km空间块聚类。结果来自结果导向挑选的环境组，组间差异尚未通过正式的多组系数等值检验。",
        fig,
        d,
    )


def figure_r() -> dict[str, str]:
    d = read_csv("results/SEM_DIRECT_INDIRECT_EFFECTS.csv")
    paths = {
        "E→Physical→SHI": "经物理维度的关联乘积",
        "E→Chemical→SHI": "经化学维度的关联乘积",
        "E→Biological→SHI": "经生物活动维度的关联乘积",
    }
    datasets = list(DATASET_LABELS)
    d = d.loc[
        d["method"].eq("piecewise_sem")
        & d["structure"].eq("S4_chained_health_dimensions")
        & d["control"].eq("M1")
        & d["dataset"].isin(datasets)
        & d["path"].isin(paths)
    ].copy()
    colors = [BLUE, GOLD, TEAL]
    fig, ax = plt.subplots(figsize=(10, 5.8))
    for (path, label), color in zip(paths.items(), colors):
        sub = d.loc[d["path"].eq(path)].set_index("dataset").reindex(datasets).reset_index()
        ax.plot(
            range(len(datasets)),
            sub["indirect_effect"].astype(float),
            marker="o",
            linewidth=2,
            color=color,
            label=label,
        )
    ax.axhline(0, color="#4a5560", linestyle="--", linewidth=1)
    ax.set_xticks(range(len(datasets)), [DATASET_LABELS[x] for x in datasets], rotation=13, ha="right")
    ax.set_ylabel("标准化条件关联系数乘积")
    ax.set_title("三条维度路径乘积随分析总体变化")
    ax.legend(frameon=False, loc="best")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return write_figure(
        "R",
        "R_chained_path_products",
        "链式维度路径的标准化关联乘积",
        "图中为侵蚀到单个健康维度、再到SHI的标准化条件关联系数乘积，并比较四类分析总体。它不是经因果识别的中介效应；当前没有为这些乘积计算置信区间，图线只连接不同总体的点估计。",
        fig,
        d,
    )


def figure_s() -> dict[str, str]:
    d = read_csv("results/SEM_PATH_COEFFICIENTS.csv")
    d = d.loc[
        d["dataset"].eq("D1_common_env_case")
        & d["method"].eq("piecewise_sem")
        & d["structure"].eq("S6_two_part_erosion")
        & d["control"].eq("M1")
        & d["from"].isin(["erosion_occurrence", "positive_intensity"])
        & d["to"].isin(OUTCOME_ORDER)
    ].copy()
    exposure_order = ["erosion_occurrence", "positive_intensity"]
    exposure_labels = {
        "erosion_occurrence": "是否出现侵蚀（全域）",
        "positive_intensity": "正侵蚀强度（仅正值）",
    }
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.4), sharex=True)
    for ax, outcome in zip(axes.flat, OUTCOME_ORDER):
        sub = d.loc[d["to"].eq(outcome)]
        ax.axvline(0, color="#4a5560", linestyle="--", linewidth=1)
        positions = []
        labels = []
        for i, exposure in enumerate(exposure_order):
            row = sub.loc[sub["from"].eq(exposure)]
            if row.empty:
                continue
            row = row.iloc[0]
            errorbar_x(ax, row, i, BLUE if i == 0 else GOLD)
            positions.append(i)
            labels.append(exposure_labels[exposure])
        ax.set_yticks(positions, labels)
        ax.invert_yaxis()
        n_vals = sub.groupby("from")["n"].first().to_dict()
        occurrence_n = int(n_vals["erosion_occurrence"]) if "erosion_occurrence" in n_vals else 0
        intensity_n = int(n_vals["positive_intensity"]) if "positive_intensity" in n_vals else 0
        n_text = f"发生状态 n={occurrence_n:,}；正侵蚀样本 n={intensity_n:,}"
        ax.set_title(f"{OUTCOME_LABELS[outcome]}\n{n_text}")
        ax.grid(axis="y", visible=False)
    fig.supxlabel("标准化路径 β（95%空间块区间）")
    fig.suptitle("侵蚀发生状态与正侵蚀强度的分开比较", y=1.02, fontsize=15)
    fig.tight_layout()
    return write_figure(
        "S",
        "S_two_part_erosion_paths",
        "侵蚀发生状态与正侵蚀强度的路径对照",
        "左侧路径表示全域像元从零侵蚀到正侵蚀的状态关联；右侧路径只在正侵蚀像元中表示侵蚀强度关联。二者的样本数与暴露定义不同，不应相减或解释为因果分解。图示SHI和三类健康维度，区间按5 km空间块聚类。",
        fig,
        d,
    )


def figure_t() -> dict[str, str]:
    stem = "T_health_direction_sensitivity"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    source_map = {
        ".png": DIRECTION_RUN / "figures/J9_health_direction_comparison.png",
        ".svg": DIRECTION_RUN / "figures/J9_health_direction_comparison.svg",
        ".pdf": DIRECTION_RUN / "figures/J9_health_direction_comparison.pdf",
    }
    for suffix, source in source_map.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, FIG_DIR / f"{stem}{suffix}")
    source_csv = DIRECTION_RUN / "results/HEALTH_DIRECTION_ASSOCIATIONS.csv"
    if not source_csv.is_file():
        raise FileNotFoundError(source_csv)
    d = pd.read_csv(source_csv)
    target_source = DATA_DIR / f"{stem}_source.csv"
    d.to_csv(target_source, index=False, encoding="utf-8-sig")
    return {
        "letter": "T",
        "id": stem,
        "title": "原始指标与健康方向统一评分的敏感性对照",
        "explanation": "对比八项指标原始值、方向统一评分、健康维度和SHI的条件关联。CEC标签仍写CEC，但因恢复公式含侵蚀项，其关联不作为独立侵蚀效应；方向转换后的SHI未精确复现既有SHI栅格，所以只作敏感性比较。",
        "png": (FIG_DIR / f"{stem}.png").relative_to(RUN).as_posix(),
        "svg": (FIG_DIR / f"{stem}.svg").relative_to(RUN).as_posix(),
        "pdf": (FIG_DIR / f"{stem}.pdf").relative_to(RUN).as_posix(),
        "source": target_source.relative_to(RUN).as_posix(),
    }


def figure_u() -> dict[str, str]:
    """Map saved GWR-LCR local coefficients for SHI and all eight indicators."""
    if not CORRECTED_NINE_OUTCOME_GWR.is_file():
        raise FileNotFoundError(CORRECTED_NINE_OUTCOME_GWR)
    d = pd.read_parquet(CORRECTED_NINE_OUTCOME_GWR)
    d = d.loc[
        d["model_family"].eq("GWR-LCR") & d["outcome"].isin(NINE_OUTCOME_ORDER)
    ].copy()
    if d.empty:
        raise ValueError("No saved GWR-LCR local coefficients found for the nine health outcomes")

    xmin, xmax = d["x"].min() / 1000, d["x"].max() / 1000
    ymin, ymax = d["y"].min() / 1000, d["y"].max() / 1000
    xpad, ypad = (xmax - xmin) * 0.035, (ymax - ymin) * 0.035
    ext = (xmin - xpad, xmax + xpad, ymin - ypad, ymax + ypad)
    fig, axes = plt.subplots(3, 3, figsize=(17, 14), constrained_layout=True, sharex=True, sharey=True)
    for ax, outcome in zip(axes.flat, NINE_OUTCOME_ORDER):
        sub = d.loc[d["outcome"].eq(outcome)]
        beta = pd.to_numeric(sub["local_beta_standardized"], errors="coerce")
        finite = beta[np.isfinite(beta)]
        limit = max(float(finite.abs().quantile(0.98)), 0.04)
        norm = mcolors.TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
        points = ax.scatter(
            sub["x"] / 1000,
            sub["y"] / 1000,
            c=beta,
            s=3.0,
            cmap="RdBu_r",
            norm=norm,
            linewidths=0,
            rasterized=True,
        )
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.12, linewidth=0.45)
        ax.set_title(f"{NINE_OUTCOME_LABELS[outcome]}\n局部单元 {len(sub):,} 个；色阶 ±{limit:.2f}", fontsize=10)
        cb = fig.colorbar(points, ax=ax, fraction=0.043, pad=0.025)
        cb.set_ticks([-limit, 0, limit])
        cb.ax.tick_params(labelsize=7)
        cb.set_label("标准化局部 β", fontsize=7)
    for ax in axes[-1, :]:
        ax.set_xlabel("东向坐标（km）")
    for ax in axes[:, 0]:
        ax.set_ylabel("北向坐标（km）")
    # One shared scale cue and north arrow, using the real projected coordinates.
    ax0 = axes[0, 0]
    x0 = ext[0] + 0.06 * (ext[1] - ext[0])
    y0 = ext[2] + 0.07 * (ext[3] - ext[2])
    ax0.plot([x0, x0 + 5], [y0, y0], color="#263238", linewidth=2.0, solid_capstyle="butt")
    ax0.text(x0 + 2.5, y0 + 0.35, "5 km", ha="center", va="bottom", fontsize=7, color="#263238")
    axes[0, 2].annotate(
        "N", xy=(0.93, 0.89), xytext=(0.93, 0.75), xycoords="axes fraction",
        textcoords="axes fraction", ha="center", va="center", fontsize=9,
        arrowprops=dict(arrowstyle="-|>", color="#263238", lw=1.1),
    )
    fig.suptitle("SHI 与八项土壤指标的 240 m 局部侵蚀系数（GWR-LCR）", fontsize=16)
    fig.text(
        0.5,
        -0.012,
        "数据来自当前CEC公式下保存的九结局局部回归；各面板采用各自98%绝对分位对称色阶，颜色只比较单项指标内部的空间形态。CEC含Aa公式项，不能据此视为独立侵蚀效应；地图不表示局部显著性。",
        ha="center",
        fontsize=8.5,
    )
    keep = [
        "x", "y", "tile_row", "tile_col", "outcome", "local_beta_standardized",
        "local_ci_low_model_based", "local_ci_high_model_based", "local_se_standardized_model_based",
        "local_r2", "n_pixels", "local_support_pixels", "model_family", "outcome_guided_selection",
    ]
    source = d[[col for col in keep if col in d.columns]].copy()
    item = write_figure(
        "U",
        "U_nine_outcome_local_beta_maps",
        "SHI与八项土壤指标的240米局部侵蚀系数地图",
        "九个小图使用当前CEC数据版本下保存的GWR-LCR局部标准化系数和真实投影坐标。每个指标采用自己的98%绝对分位对称色阶，适合查看各指标内部的空间形态，不宜直接比较不同面板的色彩深浅。局部置信区间是模型内区间，没有用于显著性热点判定；CEC的恢复公式含侵蚀项，存在代数耦合。",
        fig,
        source,
    )
    item["input_table"] = CORRECTED_NINE_OUTCOME_GWR.relative_to(ROOT).as_posix()
    item["input_cec_run"] = "phase_i_20261101_corrected_cec_nine_outcome_refresh_r2"
    return item


def figure_v() -> dict[str, str]:
    """Map where saved same-location GWR and MGWR estimates differ."""
    if not GWR_MGWR_DIFFERENCES.is_file():
        raise FileNotFoundError(GWR_MGWR_DIFFERENCES)
    d = pd.read_csv(GWR_MGWR_DIFFERENCES)
    outcomes = ["SHI_candidate", "OM", "SW"]
    d = d.loc[d["outcome"].isin(outcomes)].copy()
    if d.empty:
        raise ValueError("No saved same-location GWR/MGWR local coefficient differences found")
    diff = pd.to_numeric(d["local_beta_difference_mgwr_minus_gwr"], errors="coerce")
    limit = max(float(diff.abs().max()), 0.01)
    norm = mcolors.TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    xmin, xmax = d["x"].min() / 1000, d["x"].max() / 1000
    ymin, ymax = d["y"].min() / 1000, d["y"].max() / 1000
    xpad, ypad = (xmax - xmin) * 0.04, (ymax - ymin) * 0.04
    ext = (xmin - xpad, xmax + xpad, ymin - ypad, ymax + ypad)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.5), constrained_layout=True, sharex=True, sharey=True)
    plotted = None
    labels = {"SHI_candidate": "土壤健康指数（SHI）", "OM": "有机质（OM）", "SW": "土壤含水量（SW）"}
    for ax, outcome in zip(axes, outcomes):
        sub = d.loc[d["outcome"].eq(outcome)]
        plotted = ax.scatter(
            sub["x"] / 1000,
            sub["y"] / 1000,
            c=sub["local_beta_difference_mgwr_minus_gwr"],
            s=8,
            cmap="RdBu_r",
            norm=norm,
            linewidths=0,
            rasterized=True,
        )
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.13, linewidth=0.5)
        ax.set_title(f"{labels[outcome]}\n共同位置 n={len(sub):,}", fontsize=10)
        ax.set_xlabel("东向坐标（km）")
    axes[0].set_ylabel("北向坐标（km）")
    cb = fig.colorbar(plotted, ax=axes, shrink=0.86, pad=0.03)
    cb.set_label("局部系数差值（MGWR − GWR）")
    cb.set_ticks([-limit, 0, limit])
    fig.suptitle("标准MGWR相对标准GWR的局部系数差值", fontsize=15)
    fig.text(
        0.5,
        -0.015,
        "三项指标使用共同的770个空间均衡位置和同一颜色范围。正值表示MGWR系数相对GWR更高（通常更不负），负值表示更低；差值仅为模型间描述，不是显著性检验。",
        ha="center",
        fontsize=8.5,
    )
    source = d[[
        "x", "y", "tile_row", "tile_col", "outcome",
        "local_beta_log1p_Aa_standardized_gwr",
        "local_beta_log1p_Aa_standardized_mgwr",
        "local_beta_difference_mgwr_minus_gwr",
        "n_pixels_gwr", "n_pixels_mgwr",
    ]].copy()
    return write_figure(
        "V",
        "V_gwr_mgwr_difference_maps",
        "标准MGWR相对标准GWR的局部系数差值地图",
        "三幅地图显示770个共同空间位置上MGWR减去标准GWR的局部标准化系数差值，使用共同颜色范围。正值表示MGWR估计相对更高，负值表示相对更低；该差值是模型对照，不代表显著区域或因果差异。",
        fig,
        source,
    )


def sem_diagram_rows(structure: str) -> pd.DataFrame:
    d = read_csv("results/SEM_PATH_COEFFICIENTS.csv")
    return d.loc[
        d["dataset"].eq("D1_common_env_case")
        & d["method"].eq("piecewise_sem")
        & d["structure"].eq(structure)
        & d["control"].eq("M1")
    ].copy()


def draw_node(
    ax: plt.Axes,
    center: tuple[float, float],
    label: str,
    r2: float | None = None,
    width: float = 0.18,
    height: float = 0.10,
    fill: str = "#edf4f7",
) -> None:
    x, y = center
    box = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        facecolor=fill,
        edgecolor="#536a75",
        linewidth=1.0,
        zorder=3,
    )
    ax.add_patch(box)
    text = label if r2 is None or not np.isfinite(r2) else f"{label}\nR²={r2:.3f}"
    ax.text(x, y, text, ha="center", va="center", fontsize=8.3, color="#20343e", zorder=4)


def draw_path(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    beta: float,
    label_pos: tuple[float, float],
    label: str | None = None,
    rad: float = 0.0,
) -> None:
    color = BLUE if beta < 0 else GOLD
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.25 + min(abs(beta), 1.0) * 1.6,
        color=color,
        linestyle="solid" if beta < 0 else "dashed",
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=22,
        shrinkB=22,
        alpha=0.90,
        zorder=1,
    )
    ax.add_patch(patch)
    if label is None:
        label = f"β={beta:+.2f}"
    ax.text(
        label_pos[0],
        label_pos[1],
        label,
        ha="center",
        va="center",
        fontsize=7.5 if "\n" in label else 8,
        color="#23343c",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#d6dee2", alpha=0.93),
        zorder=2,
    )


def init_path_axis(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def figure_w() -> dict[str, str]:
    d = sem_diagram_rows("S1_parallel_dimensions")
    outcomes = ["physical", "chemical", "biological_activity"]
    d = d.loc[d["from"].eq("z_E") & d["to"].isin(outcomes)].copy()
    r2_by = d.set_index("to")["r_squared"].to_dict()
    fig, ax = plt.subplots(figsize=(12.5, 6.4))
    init_path_axis(ax)
    source = (0.14, 0.50)
    positions = {"physical": (0.78, 0.80), "chemical": (0.78, 0.50), "biological_activity": (0.78, 0.20)}
    draw_node(ax, source, "标准化侵蚀暴露\nlog1p(Aa)", width=0.20, height=0.13, fill="#e8f1f5")
    label_names = {"physical": "物理维度", "chemical": "化学维度", "biological_activity": "生物活动维度"}
    for outcome in outcomes:
        row = d.loc[d["to"].eq(outcome)].iloc[0]
        target = positions[outcome]
        draw_node(ax, target, label_names[outcome], float(r2_by[outcome]), width=0.22, height=0.13, fill="#f3f5f5")
        interval = f"\n[{row.ci_low:+.2f}, {row.ci_high:+.2f}]"
        mid = ((source[0] + target[0]) / 2, (source[1] + target[1]) / 2)
        draw_path(
            ax, source, target, float(row.beta_standardized),
            (mid[0], mid[1] + (0.035 if outcome == "physical" else 0.0 if outcome == "chemical" else -0.035)),
            label=f"β={row.beta_standardized:+.2f}{interval}",
        )
    ax.text(0.5, 0.96, "平行结局结构 S1｜核心环境变量控制 M1", ha="center", va="top", fontsize=12, color="#334a56")
    fig.suptitle("侵蚀与三类土壤健康维度的平行路径图", fontsize=15, y=0.99)
    fig.text(0.5, 0.035, "全域共同环境完整像元 n=293,145；区间为5 km空间块聚类95%区间。模型还控制高程、地形湿度、植被指数、地表温度和土类；图中只展开侵蚀路径。",
             ha="center", fontsize=8.5)
    return write_figure(
        "W", "W_sem_parallel_dimensions_diagram", "侵蚀与三类土壤健康维度的平行路径图",
        "按已保存的S1分段路径模型绘图；箭头标签给出标准化β和5 km空间块聚类95%区间，结局框内为R²。模型控制高程、地形湿度、植被指数、地表温度和土类（M1）；图中仅画侵蚀到三类健康维度的焦点路径。箭头表示条件关联，不是已验证的因果关系。",
        fig, d,
    )


def figure_x() -> dict[str, str]:
    d = sem_diagram_rows("S4_chained_health_dimensions")
    pairs = [
        ("z_E", "physical"), ("z_E", "chemical"), ("z_E", "biological_activity"), ("z_E", "z_SHI"),
        ("physical", "chemical"), ("physical", "biological_activity"), ("chemical", "biological_activity"),
        ("physical", "z_SHI"), ("chemical", "z_SHI"), ("biological_activity", "z_SHI"),
    ]
    graph = {(row["from"], row["to"]): row for _, row in d.iterrows()}
    missing = [pair for pair in pairs if pair not in graph]
    if missing:
        raise ValueError(f"Saved S4 path table is missing paths: {missing}")
    r2_by = d.groupby("to")["r_squared"].first().to_dict()
    outcome_names = {
        "physical": "物理维度",
        "chemical": "化学维度",
        "biological_activity": "生物活动维度",
        "z_SHI": "SHI",
    }
    fig, axes = plt.subplots(1, 3, figsize=(17, 7.3))

    # Split the same fitted S4 structure into three readable path groups.
    ax = axes[0]
    init_path_axis(ax)
    exposure = (0.15, 0.50)
    draw_node(ax, exposure, "标准化侵蚀\nlog1p(Aa)", width=0.24, height=0.14, fill="#e8f1f5")
    target_y = {"physical": 0.86, "chemical": 0.62, "biological_activity": 0.38, "z_SHI": 0.14}
    for outcome, y in target_y.items():
        row = graph[("z_E", outcome)]
        center = (0.82, y)
        draw_node(ax, center, outcome_names[outcome], r2_by[outcome], width=0.25, height=0.12, fill="#f1f4f4")
        draw_path(
            ax, exposure, center, float(row["beta_standardized"]),
            (0.49, (exposure[1] + y) / 2),
        )
    ax.set_title("A｜侵蚀到各结局", fontsize=10)

    ax = axes[1]
    init_path_axis(ax)
    dimension_pos = {"physical": (0.16, 0.50), "chemical": (0.52, 0.82), "biological_activity": (0.84, 0.42)}
    for outcome, center in dimension_pos.items():
        draw_node(ax, center, outcome_names[outcome], r2_by[outcome], width=0.25, height=0.13, fill="#f1f4f4")
    dimension_edges = [
        (("physical", "chemical"), (0.34, 0.70), 0.0),
        (("physical", "biological_activity"), (0.48, 0.39), -0.04),
        (("chemical", "biological_activity"), (0.69, 0.66), 0.0),
    ]
    for pair, label_pos, rad in dimension_edges:
        row = graph[pair]
        draw_path(ax, dimension_pos[pair[0]], dimension_pos[pair[1]], float(row["beta_standardized"]), label_pos, rad=rad)
    ax.set_title("B｜健康维度之间", fontsize=10)

    ax = axes[2]
    init_path_axis(ax)
    dimension_left = {"physical": (0.16, 0.82), "chemical": (0.16, 0.50), "biological_activity": (0.16, 0.18)}
    shi = (0.83, 0.50)
    for outcome, center in dimension_left.items():
        draw_node(ax, center, outcome_names[outcome], r2_by[outcome], width=0.26, height=0.13, fill="#f1f4f4")
    draw_node(ax, shi, outcome_names["z_SHI"], r2_by["z_SHI"], width=0.21, height=0.12, fill="#f1f4f4")
    for outcome, center in dimension_left.items():
        row = graph[(outcome, "z_SHI")]
        draw_path(ax, center, shi, float(row["beta_standardized"]), ((center[0] + shi[0]) / 2, (center[1] + shi[1]) / 2))
    ax.set_title("C｜健康维度到 SHI", fontsize=10)

    fig.suptitle("S4 链式结构：侵蚀、健康维度和 SHI 的路径图", fontsize=15, y=0.98)
    fig.text(0.5, 0.035, "全域共同环境完整像元 n=293,145；框内为结局模型R²，箭头为标准化β点估计，完整95%空间块区间见图源表。模型控制高程、地形湿度、植被指数、地表温度和土类。",
             ha="center", fontsize=8.5)
    return write_figure(
        "X", "X_sem_chained_dimensions_diagram", "侵蚀与健康维度链式路径图",
        "将S4链式结构拆成三组便于阅读的路径：侵蚀到健康结局、健康维度之间、健康维度到SHI。框内为各结局模型R²，箭头为标准化β点估计；完整95%空间块区间见图源表。模型控制核心环境变量和土类，图中箭头是条件关联，不是已证实的中介因果链。",
        fig, pd.DataFrame([graph[pair] for pair in pairs]),
    )


def figure_y() -> dict[str, str]:
    d = sem_diagram_rows("S5_indicator_responses")
    targets = [
        ("z_SHI_candidate", "土壤健康指数（SHI）"), ("z_AN", "碱解氮（AN）"),
        ("z_OM", "有机质（OM）"), ("z_CEC", "阳离子交换量（CEC）"),
        ("z_pH", "酸碱度（pH）"), ("z_SW", "土壤含水量（SW）"),
        ("z_BD", "土壤容重（BD）"), ("z_MBC", "微生物量碳（MBC）"),
        ("z_RESP", "土壤呼吸（RESP）"),
    ]
    d = d.loc[d["from"].eq("z_E") & d["to"].isin([x[0] for x in targets])].copy()
    d = d.set_index("to").reindex([x[0] for x in targets]).reset_index()
    fig, ax = plt.subplots(figsize=(12, 11))
    init_path_axis(ax)
    source = (0.13, 0.52)
    draw_node(ax, source, "标准化侵蚀暴露\nlog1p(Aa)", width=0.20, height=0.13, fill="#e8f1f5")
    ys = np.linspace(0.92, 0.12, len(targets))
    for (target, label), y, (_, row) in zip(targets, ys, d.iterrows()):
        center = (0.84, float(y))
        draw_node(ax, center, label, float(row["r_squared"]), width=0.25, height=0.075, fill="#f2f5f5")
        draw_path(
            ax, source, center, float(row["beta_standardized"]),
            (0.55, float(y) + 0.015),
        )
    ax.text(0.5, 0.98, "九项健康结局结构 S5｜核心环境变量控制 M1", ha="center", va="top", fontsize=12, color="#334a56")
    fig.suptitle("侵蚀与 SHI 及八项土壤指标的单结局路径图", fontsize=15, y=0.995)
    fig.text(
        0.5, 0.025,
        "全域共同环境完整像元 n=293,145；箭头标签是标准化β点估计，结局框内为各自R²。模型控制高程、地形湿度、植被指数、地表温度和土类；CEC公式包含Aa项，CEC路径存在代数耦合。完整区间见图源表。",
        ha="center", fontsize=8.2,
    )
    return write_figure(
        "Y", "Y_sem_nine_outcome_diagram", "侵蚀与SHI及八项土壤指标的单结局路径图",
        "展示S5结构下侵蚀暴露到SHI和八项土壤指标的九条标准化条件路径；结局框中的R²来自各自单结局模型。M1控制高程、地形湿度、植被指数、地表温度和土类。CEC标签仅写CEC，但恢复公式含侵蚀项，因此该路径存在代数耦合。箭头不是因果证据，区间见源表。",
        fig, d,
    )


def figure_z() -> dict[str, str]:
    d = sem_diagram_rows("S6_two_part_erosion")
    exposures = [
        ("erosion_occurrence", "是否出现侵蚀", "全域像元"),
        ("positive_intensity", "正侵蚀强度", "仅正侵蚀像元"),
    ]
    outcomes = [
        ("z_SHI", "SHI"), ("physical", "物理维度"),
        ("chemical", "化学维度"), ("biological_activity", "生物活动维度"),
    ]
    d = d.loc[d["from"].isin([x[0] for x in exposures]) & d["to"].isin([x[0] for x in outcomes])].copy()
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))
    for ax, (exposure, exposure_label, scope) in zip(axes, exposures):
        init_path_axis(ax)
        sub = d.loc[d["from"].eq(exposure)].set_index("to")
        n = int(sub["n"].iloc[0])
        source = (0.16, 0.52)
        draw_node(ax, source, f"{exposure_label}\n n={n:,}", width=0.25, height=0.14, fill="#e8f1f5")
        ys = {"z_SHI": 0.86, "physical": 0.63, "chemical": 0.39, "biological_activity": 0.16}
        for outcome, label in outcomes:
            row = sub.loc[outcome]
            target = (0.80, ys[outcome])
            draw_node(ax, target, label, float(row["r_squared"]), width=0.24, height=0.10, fill="#f2f5f5")
            draw_path(ax, source, target, float(row["beta_standardized"]), (0.49, ys[outcome] + 0.03))
        ax.set_title(f"{scope}\nS6 两部分侵蚀模型｜M1", fontsize=10)
    fig.suptitle("侵蚀发生状态与正侵蚀强度的两组路径图", fontsize=15, y=0.98)
    fig.text(
        0.5, 0.035,
        "发生状态样本 n=293,145；正侵蚀强度样本 n=195,013。箭头标签为标准化β点估计，框内R²按各自模型给出；完整95%空间块区间见源表。两种暴露总体不同，不能把两组路径相减解释为因果分解。",
        ha="center", fontsize=8.3,
    )
    return write_figure(
        "Z", "Z_sem_two_part_path_diagram", "侵蚀发生与正侵蚀强度的两组结构路径图",
        "左右两图分别展示S6中侵蚀发生状态和正侵蚀强度到SHI及三类健康维度的标准化条件路径；左侧使用全域样本，右侧只使用正侵蚀像元。框内是各自模型R²，箭头点估计的95%空间块区间见图源表。两种暴露定义和样本不同，不能相减或视为因果分解。",
        fig, d,
    )


def main() -> None:
    builders = [figure_j, figure_k, figure_l, figure_m, figure_n, figure_o, figure_p, figure_q, figure_r, figure_s]
    figures = [builder() for builder in builders]
    figures.append(figure_t())
    figures.extend([figure_u(), figure_v(), figure_w(), figure_x(), figure_y(), figure_z()])
    manifest = {
        "run_id": RUN.name,
        "outcome_guided_selection": True,
        "model_refit": False,
        "description": "Figures J-S and W-Z were rendered from saved Phase J r10 tables. Figure T reuses the saved direction-score sensitivity figure. Figures U-V use saved, corrected-CEC local GWR tables from Phase I. No model refits were run.",
        "figures": figures,
    }
    (FIG_DIR / "figure_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Rendered {len(figures)} supplementary figures into {FIG_DIR}")
    for item in figures:
        print(f"{item['letter']} {item['title']}: {item['png']}")


if __name__ == "__main__":
    main()
