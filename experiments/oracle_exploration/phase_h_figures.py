"""Build Phase H publication-style figures and inspectable chart sources."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phase_h_matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import binned_statistic

from experiments.oracle_exploration.nonlinear_search import fit_model_family


OUTCOMES = ["SHI_candidate", "AN", "OM", "CEC", "pH", "SW", "BD", "MBC", "RESP"]
EXPOSURES = ["Aa_20yr", "Aa_2020"]
EXPOSURE_LABELS = {"Aa_20yr": "20 年平均实际侵蚀", "Aa_2020": "2020 年实际侵蚀"}
EXPOSURE_COLORS = {"Aa_20yr": "#3977a8", "Aa_2020": "#df8554"}
OUTCOME_LABELS = {
    "SHI_candidate": "土壤健康指数（SHI）",
    "AN": "碱解氮（AN）",
    "OM": "有机质（OM）",
    "CEC": "阳离子交换量（CEC）",
    "pH": "土壤酸碱度（pH）",
    "SW": "土壤含水量（SW）",
    "BD": "土壤容重（BD）",
    "MBC": "微生物量碳（MBC）",
    "RESP": "土壤呼吸（RESP）",
}
SOIL_TYPE_LABELS = {26: "冲积土", 27: "潮土", 42: "脱潮土"}
STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#d8dee8",
    "axes.labelcolor": "#263746",
    "text.color": "#263746",
    "xtick.color": "#51606e",
    "ytick.color": "#51606e",
    "grid.color": "#e8edf2",
    "font.family": "Noto Sans CJK JP",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.frameon": False,
    "svg.fonttype": "none",
}


def _group_method_label(value: Any) -> str:
    text = str(value)
    if text == "soil_type_group":
        return "土壤类型分层"
    match = re.fullmatch(r"kmeans_terrain_k(\d+)", text)
    if match:
        return f"高程与地表温度 K 均值聚类（k={match.group(1)}）"
    if text == "hierarchical_ward_v5":
        return "分土类沃德聚类"
    if text.startswith("gradient_"):
        variable = text.removeprefix("gradient_")
        variable_labels = {"Elevation": "高程", "TWI": "地形湿度指数", "LST": "地表温度", "NDWI": "水体指数", "NDVI": "植被指数", "Precipitation": "降水量"}
        return f"{variable_labels.get(variable, variable)}梯度"
    return text


def _group_value_label(value: Any) -> str:
    text = str(value)
    match = re.fullmatch(r"soil_(\d+)_cluster_(\d+)", text)
    if match:
        code = int(match.group(1))
        return f"{code} {SOIL_TYPE_LABELS.get(code, '未知土类')} · 第 {int(match.group(2)) + 1} 组"
    match = re.fullmatch(r"(\d+):q(\d+)", text)
    if match:
        code = int(match.group(1))
        return f"{code} {SOIL_TYPE_LABELS.get(code, '未知土类')} · 第 {match.group(2)} 分位组"
    match = re.fullmatch(r"soil_(\d+)", text)
    if match:
        code = int(match.group(1))
        return f"{code} {SOIL_TYPE_LABELS.get(code, '未知土类')}"
    return text


def _status_label(value: Any) -> str:
    labels = {
        "reliable": "通过可靠性检查",
        "unreliable_covariance_rank_deficient": "协方差秩不足",
        "unreliable_group_block_coverage": "空间块支持不足",
        "unreliable_tolerance_sensitive": "对数值容差敏感",
        "unreliable_design_condition": "设计矩阵条件不佳",
        "non_estimable_design_rank": "设计矩阵不可估",
    }
    return labels.get(str(value), str(value))


def _read_csv(table_dir: Path, name: str) -> pd.DataFrame:
    path = table_dir / name
    return pd.read_csv(path, low_memory=False) if path.is_file() else pd.DataFrame()


def _save_figure(fig: plt.Figure, figure_dir: Path, stem: str, source: pd.DataFrame, caption: str) -> None:
    for fmt, subdir in (("png", "png"), ("svg", "svg"), ("pdf", "pdf")):
        path = figure_dir / subdir / f"{stem}.{fmt}"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    source_dir = figure_dir / "data"
    source_dir.mkdir(parents=True, exist_ok=True)
    source.to_csv(source_dir / f"{stem}_source.csv", index=False, encoding="utf-8-sig")
    caption_dir = figure_dir / "captions"
    caption_dir.mkdir(parents=True, exist_ok=True)
    (caption_dir / f"{stem}.md").write_text(caption.rstrip() + "\n", encoding="utf-8")
    plt.close(fig)


def _first_available_frame(*frames: pd.DataFrame) -> pd.DataFrame:
    for frame in frames:
        if not frame.empty:
            return frame
    return pd.DataFrame()


def _selected_trials(run_dir: Path) -> pd.DataFrame:
    trials = _read_csv(run_dir / "tables", "phase_h_model_trials.csv")
    if trials.empty:
        return trials
    trials = trials.loc[
        trials["stage"].eq("stage2_full_sample")
        & trials["analysis_population"].eq("phase_h_matched_only")
        & trials["subset_name"].eq("all")
        & trials["fit_scope"].eq("exposure_only")
        & trials["evaluation_set"].eq("apparent")
        & trials["status"].eq("ok")
    ].copy()
    trials["r_squared"] = pd.to_numeric(trials["r_squared"], errors="coerce")
    trials = trials.loc[np.isfinite(trials["r_squared"])].copy()
    winners = trials.sort_values(
        ["exposure", "outcome", "estimand", "r_squared", "candidate_id", "model_family"],
        ascending=[True, True, True, False, True, True],
        kind="stable",
    ).drop_duplicates(["exposure", "outcome", "estimand"], keep="first")
    return winners


def _spline_curve(data: pd.DataFrame, x: str, y: str, *, label_values: dict[str, Any], points: int = 180) -> pd.DataFrame:
    selected = data[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
    selected = selected.loc[np.isfinite(selected.to_numpy(dtype=float)).all(axis=1)]
    if len(selected) < 25 or selected[x].nunique() < 5 or selected[y].nunique() < 2:
        return pd.DataFrame()
    low, high = np.quantile(selected[x], [0.005, 0.995])
    if not np.isfinite([low, high]).all() or high <= low:
        return pd.DataFrame()
    grid = np.linspace(float(low), float(high), points)
    score = pd.DataFrame({x: grid, y: np.zeros_like(grid)})
    try:
        prediction, parameters = fit_model_family(
            selected.assign(**{y: selected[y]}),
            score,
            outcome=y,
            exposure=x,
            numeric_features=[x],
            model_family="natural_spline",
            random_state=20261028,
        )
    except Exception:
        return pd.DataFrame()
    result = pd.DataFrame({"x_value": grid, "predicted_outcome": prediction})
    for key, value in label_values.items():
        result[key] = value
    result["n_fit"] = int(len(selected))
    result["model_family"] = "natural_spline"
    result["model_parameters_json"] = json.dumps(parameters, ensure_ascii=False, sort_keys=True, default=str)
    return result


def _axis_grid(axes: np.ndarray, count: int) -> None:
    for ax in axes.flat[:count]:
        ax.grid(axis="y", linewidth=0.6, alpha=0.7)
        ax.set_axisbelow(True)
    for ax in axes.flat[count:]:
        ax.set_visible(False)


def _figure_1(run_dir: Path, figure_dir: Path) -> None:
    data = _read_csv(run_dir / "tables", "phase_h_nonparametric_response_summary.csv")
    if data.empty:
        return
    order = [value for value in OUTCOMES if value in set(data["outcome"])]
    selected = data.loc[data["outcome"].isin(order)].copy()
    summary = selected.groupby(["exposure", "outcome", "estimand"], as_index=False).agg(
        median_spearman_rho=("spearman_rho", "median"),
        min_spearman_rho=("spearman_rho", "min"),
        max_spearman_rho=("spearman_rho", "max"),
        median_kendall_tau=("kendall_tau", "median"),
        median_jt_tau_b=("jt_tau_b", "median"),
        candidate_count=("candidate_id", "nunique"),
        median_n=("n", "median"),
    ).copy()
    fig, axes = plt.subplots(1, 3, figsize=(17, 7.5), gridspec_kw={"width_ratios": [1.2, 1.2, 0.9]})
    metrics = [
        ("median_spearman_rho", "斯皮尔曼秩相关系数 ρ", "vlag", -1, 1),
        ("median_jt_tau_b", "JT 有序趋势统计量 τ-b", "vlag", -1, 1),
    ]
    for ax, (metric, title, cmap, vmin, vmax) in zip(axes[:2], metrics, strict=True):
        matrix = summary.pivot(index="outcome", columns=["exposure", "estimand"], values=metric).reindex(order)
        matrix.columns = [f"{'20 年平均' if e == 'Aa_20yr' else '2020 年'}\n{'配对' if estimand == 'matched_pair' else '像元'}" for e, estimand in matrix.columns]
        matrix.index = [OUTCOME_LABELS.get(value, value) for value in matrix.index]
        sns.heatmap(
            matrix, ax=ax, cmap=cmap, center=0, vmin=vmin, vmax=vmax,
            annot=True, fmt=".2f", linewidths=0.6, linecolor="white",
            cbar_kws={"shrink": 0.72, "label": "秩关联强度"},
        )
        ax.set_title(title, loc="left", fontweight="bold", pad=12)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=0, labelsize=8)
        ax.tick_params(axis="y", rotation=0)
    paired = _read_csv(run_dir / "tables", "phase_h_paired_difference_summary.csv")
    if not paired.empty:
        paired = paired.groupby(["exposure", "outcome"], as_index=False).agg(
            negative_delta_fraction=("negative_delta_fraction", "median"),
            median_delta_health=("median_delta_health", "median"),
            candidate_count=("candidate_id", "nunique"),
        )
        fraction = paired.pivot(index="outcome", columns="exposure", values="negative_delta_fraction").reindex(order)
        fraction = fraction.reindex(columns=[e for e in EXPOSURES if e in fraction.columns])
        fraction = fraction.rename(columns=EXPOSURE_LABELS)
        fraction.plot(kind="barh", ax=axes[2], color=[EXPOSURE_COLORS[e] for e in EXPOSURES if e in set(paired["exposure"])], width=0.72)
        axes[2].axvline(0.5, color="#65717e", linestyle="--", linewidth=1)
        axes[2].set_xlim(0, 1)
        axes[2].set_yticklabels([OUTCOME_LABELS.get(value, value) for value in fraction.index])
        axes[2].set_xlabel("健康差值为负的配对比例")
        axes[2].set_ylabel("")
        axes[2].set_title("配对差值：高侵蚀端减低侵蚀端\n健康差值为负的比例", loc="left", fontweight="bold", pad=12)
        axes[2].legend(frameon=False, fontsize=8, loc="lower right")
        axes[2].grid(axis="x", linewidth=0.6)
    fig.suptitle("环境匹配后侵蚀—健康关联方向总览", x=0.02, y=1.02, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.02, -0.02, "热图为第二阶段六个探索候选的中位数；匹配像元与匹配配对分开显示。配对健康差值按高侵蚀端减低侵蚀端计算。", fontsize=8, color="#5c6b79")
    fig.tight_layout()
    source = summary.merge(
        paired if not paired.empty else pd.DataFrame(columns=["exposure", "outcome", "negative_delta_fraction", "median_delta_health", "candidate_count"]),
        on=["exposure", "outcome"], how="left", suffixes=("_rank", "_pair"),
    )
    _save_figure(fig, figure_dir, "01_matched_response_direction", source,
        "斯皮尔曼秩相关与 JT 有序趋势统计量为第二阶段候选中位数，匹配像元与匹配配对分开呈现。右侧显示高侵蚀端健康值低于低侵蚀端的配对比例。候选集由健康结果导向筛选，图中结果均为描述性探索。")


def _figure_2(run_dir: Path, figure_dir: Path) -> None:
    data = _read_csv(run_dir / "tables", "phase_h_boxplot_source.csv")
    if data.empty:
        return
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), squeeze=False)
    _axis_grid(axes, 9)
    legend_handles = [
        Patch(facecolor=EXPOSURE_COLORS["Aa_20yr"], alpha=0.75, label=EXPOSURE_LABELS["Aa_20yr"]),
        Patch(facecolor=EXPOSURE_COLORS["Aa_2020"], alpha=0.75, label=EXPOSURE_LABELS["Aa_2020"]),
    ]
    for ax, outcome in zip(axes.flat, OUTCOMES, strict=True):
        subset = data.loc[data["outcome"].eq(outcome)]
        positions: list[float] = []
        values: list[np.ndarray] = []
        colors: list[str] = []
        ticks: list[float] = []
        for exp_index, exposure in enumerate(EXPOSURES):
            for role_index, role in enumerate(("low_erosion", "high_erosion")):
                rows = subset.loc[subset["exposure"].eq(exposure) & subset["member_role"].eq(role), "health_value"]
                rows = pd.to_numeric(rows, errors="coerce").dropna()
                pos = exp_index * 3.0 + role_index * 0.9
                positions.append(pos)
                values.append(rows.to_numpy(dtype=float))
                colors.append(EXPOSURE_COLORS[exposure])
                ticks.append(pos)
        artists = ax.boxplot(values, positions=positions, widths=0.65, patch_artist=True, showfliers=False,
                             medianprops={"color": "#293744", "linewidth": 1.2},
                             whiskerprops={"color": "#7d8b97"}, capprops={"color": "#7d8b97"})
        for patch, color in zip(artists["boxes"], colors, strict=True):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
            patch.set_edgecolor("white")
        ax.set_title(OUTCOME_LABELS[outcome], loc="left", fontweight="bold")
        ax.set_xticks(ticks, ["20 年\n低侵蚀", "20 年\n高侵蚀", "2020 年\n低侵蚀", "2020 年\n高侵蚀"], fontsize=8)
        ax.set_ylabel("健康指标值")
        ax.grid(axis="y", linewidth=0.6)
    fig.suptitle("环境匹配后高、低侵蚀配对端点的健康指标分布", x=0.02, y=1.02, ha="left", fontsize=15, fontweight="bold")
    fig.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(0.99, 1.01), ncol=2)
    fig.text(0.02, -0.015, "每对样本各包含一个高 Aa 端点和一个低 Aa 端点。箱线图使用已保存的确定性绘图样本；重复像元成员关系予以保留。", fontsize=8, color="#5c6b79")
    fig.tight_layout()
    _save_figure(fig, figure_dir, "02_matched_endpoint_boxplots", data,
        "箱线图分别比较 20 年平均实际侵蚀和 2020 年实际侵蚀下，高侵蚀端与低侵蚀端的健康指标值，并覆盖全部结局。图中为匹配关系的确定性抽样；同一像元重复参与配对时仍按成员关系保留，因此这些记录不等于相互独立的观测。")


def _model_trial_summary(run_dir: Path) -> pd.DataFrame:
    trials = _read_csv(run_dir / "tables", "phase_h_model_trials.csv")
    if trials.empty:
        return trials
    selected = trials.loc[
        trials["stage"].eq("stage2_full_sample")
        & trials["analysis_population"].eq("phase_h_matched_only")
        & trials["subset_name"].eq("all")
        & trials["fit_scope"].eq("exposure_only")
        & trials["evaluation_set"].eq("apparent")
        & trials["status"].eq("ok")
    ].copy()
    selected["r_squared"] = pd.to_numeric(selected["r_squared"], errors="coerce")
    selected = selected.loc[np.isfinite(selected["r_squared"])].copy()
    return selected.sort_values("r_squared", ascending=False).drop_duplicates(
        ["candidate_id", "exposure", "outcome", "estimand"], keep="first"
    )


def _figure_3(run_dir: Path, figure_dir: Path) -> pd.DataFrame:
    data = _model_trial_summary(run_dir)
    if data.empty:
        return data
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), sharey=True)
    for ax, estimand in zip(axes, ("matched_pixel", "matched_pair"), strict=True):
        subset = data.loc[data["estimand"].eq(estimand) & data["outcome"].isin(OUTCOMES)]
        x_positions = {outcome: index for index, outcome in enumerate(OUTCOMES)}
        for exposure in EXPOSURES:
            chosen = subset.loc[subset["exposure"].eq(exposure)]
            color = EXPOSURE_COLORS[exposure]
            for outcome in OUTCOMES:
                values = chosen.loc[chosen["outcome"].eq(outcome), "r_squared"].dropna().to_numpy(dtype=float)
                if not len(values):
                    continue
                x = x_positions[outcome] + (-0.15 if exposure == "Aa_20yr" else 0.15)
                median = float(np.median(values))
                low, high = np.quantile(values, [0.1, 0.9])
                ax.vlines(x, low, high, color=color, linewidth=3, alpha=0.58)
                ax.scatter([x], [median], color=color, s=42, edgecolor="white", linewidth=0.8, zorder=4)
        ax.axhline(0, color="#87939e", linewidth=0.8)
        ax.set_xticks(range(len(OUTCOMES)), [OUTCOME_LABELS[o] for o in OUTCOMES], rotation=35, ha="right")
        ax.set_ylabel("表观拟合度 R²")
        ax.set_title("匹配像元" if estimand == "matched_pixel" else "匹配配对", loc="left", fontweight="bold")
        ax.grid(axis="y", linewidth=0.6)
    handles = [Line2D([0], [0], marker="o", linestyle="", color=EXPOSURE_COLORS[e], label=EXPOSURE_LABELS[e]) for e in EXPOSURES]
    axes[1].legend(handles=handles, frameon=False, loc="upper right")
    fig.suptitle("第二阶段入选候选的表观模型拟合度", x=0.02, y=1.02, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.02, -0.04, "点为各候选最佳模型 R² 的中位数，竖线为候选分布的第 10 至第 90 百分位。像元与配对的样本单位和 R² 始终分开。", fontsize=8, color="#5c6b79")
    fig.tight_layout()
    _save_figure(fig, figure_dir, "03_apparent_r2_pixel_vs_pair", data,
        "对每项健康指标和侵蚀时段，取每个第二阶段候选中暴露单变量模型的最高表观 R²。点为候选中位数，竖线为第 10 至第 90 百分位。像元健康值模型和配对差值模型的响应单位及抽样结构不同，因此分面展示。候选由健康结果导向筛选，属于表观拟合。")
    return data


def _figure_4(run_dir: Path, figure_dir: Path, models: pd.DataFrame) -> pd.DataFrame:
    rank = _read_csv(run_dir / "tables", "phase_h_nonparametric_response_summary.csv")
    if rank.empty or models.empty:
        return pd.DataFrame()
    rank = rank.loc[rank["outcome"].isin(OUTCOMES)].copy()
    merged = rank.merge(
        models[["candidate_id", "exposure", "outcome", "estimand", "r_squared", "model_family"]],
        on=["candidate_id", "exposure", "outcome", "estimand"], how="inner", validate="many_to_one",
    )
    if merged.empty:
        return merged
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), sharex=True, sharey=True)
    palette = dict(zip(OUTCOMES, sns.color_palette("husl", n_colors=len(OUTCOMES)), strict=True))
    for ax, estimand in zip(axes, ("matched_pixel", "matched_pair"), strict=True):
        subset = merged.loc[merged["estimand"].eq(estimand)]
        for outcome in OUTCOMES:
            rows = subset.loc[subset["outcome"].eq(outcome)]
            for exposure, marker in (("Aa_20yr", "o"), ("Aa_2020", "s")):
                points = rows.loc[rows["exposure"].eq(exposure)]
                ax.scatter(points["spearman_rho"], points["r_squared"], s=38, marker=marker,
                           color=palette[outcome], edgecolor="white", linewidth=0.4, alpha=0.82)
        ax.axvline(0, color="#a1acb6", linewidth=0.8)
        ax.set_xlabel("斯皮尔曼秩相关系数 ρ")
        ax.set_title("匹配像元" if estimand == "matched_pixel" else "匹配配对", loc="left", fontweight="bold")
        ax.grid(linewidth=0.5)
    axes[0].set_ylabel("各候选最佳模型的表观拟合度 R²")
    outcome_handles = [Line2D([0], [0], marker="o", linestyle="", color=palette[o], label=OUTCOME_LABELS[o]) for o in OUTCOMES]
    exp_handles = [Line2D([0], [0], marker=m, linestyle="", color="#65717e", label=EXPOSURE_LABELS[e]) for e, m in (("Aa_20yr", "o"), ("Aa_2020", "s"))]
    axes[1].legend(handles=outcome_handles + exp_handles, frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.suptitle("秩关联与非线性模型拟合反映不同响应特征", x=0.02, y=1.02, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.02, -0.04, "每个点代表一个第二阶段候选。斯皮尔曼秩相关描述单调关联；模型 R²还可能反映弯曲或其他非线性拟合。", fontsize=8, color="#5c6b79")
    fig.tight_layout()
    _save_figure(fig, figure_dir, "04_rank_association_vs_r2", merged,
        "比较每个候选、健康指标、侵蚀时段和分析单位对应的斯皮尔曼秩相关与最高表观模型 R²。匹配像元与匹配配对分开展示。非线性 R²较高不必然代表单调秩关联强；候选集本身经过健康结局导向筛选。")
    return merged


def _curve_panel(figure_dir: Path, stem: str, curve_data: pd.DataFrame, *, title: str, x_label: str, caption: str, fixed_sample: pd.DataFrame | None = None) -> None:
    if curve_data.empty:
        return
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), squeeze=False)
    _axis_grid(axes, 9)
    for ax, outcome in zip(axes.flat, OUTCOMES, strict=True):
        subset = curve_data.loc[curve_data["outcome"].eq(outcome)]
        if fixed_sample is not None and not fixed_sample.empty:
            scatter = fixed_sample.loc[fixed_sample["outcome"].eq(outcome)]
            if len(scatter) > 900:
                scatter = scatter.sample(n=900, random_state=20261028)
            for exposure in EXPOSURES:
                rows = scatter.loc[scatter["exposure"].eq(exposure)]
                if not rows.empty:
                    ax.scatter(rows["x_value"], rows["y_value"], s=3, alpha=0.09,
                               color=EXPOSURE_COLORS[exposure], rasterized=True)
        for exposure in EXPOSURES:
            period_subset = subset.loc[subset["exposure"].eq(exposure)]
            if "legacy" in period_subset:
                period_subset = period_subset.loc[~period_subset["legacy"].fillna(False)]
            curve = period_subset.sort_values("x_value")
            if not curve.empty:
                ax.plot(curve["x_value"], curve["predicted_outcome"], color=EXPOSURE_COLORS[exposure], linewidth=1.8, label=EXPOSURE_LABELS[exposure])
        if "legacy" in subset and subset["legacy"].fillna(False).any():
            old = subset.loc[subset["legacy"].fillna(False)].sort_values("x_value")
            ax.plot(old["x_value"], old["predicted_outcome"], color="#596b5b", linewidth=1.8, linestyle="--", label="F 阶段历史探索性择优表观曲线")
        ax.set_title(OUTCOME_LABELS[outcome], loc="left", fontweight="bold")
        ax.set_xlabel(x_label)
        ax.set_ylabel("配对健康差值" if subset.get("estimand", pd.Series(dtype=str)).eq("matched_pair").any() else "健康指标值")
        ax.grid(axis="y", linewidth=0.5)
        if ax is axes.flat[0]:
            ax.legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle(title, x=0.02, y=1.02, ha="left", fontsize=15, fontweight="bold")
    fig.tight_layout()
    _save_figure(fig, figure_dir, stem, curve_data, caption)
    if fixed_sample is not None and not fixed_sample.empty:
        fixed_sample.to_csv(figure_dir / "data" / f"{stem}_scatter_source.csv", index=False, encoding="utf-8-sig")


def _figure_5(run_dir: Path, figure_dir: Path, cells: pd.DataFrame, config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    outputs: dict[str, pd.DataFrame] = {}
    fig_conf = config.get("response_curves", {})
    wanted = int(fig_conf.get("fixed_plot_sample_size", 20000))
    require = ["cell_id", *EXPOSURES, *OUTCOMES]
    available = [column for column in require if column in cells]
    eligible = cells[available].copy()
    eligible[EXPOSURES + OUTCOMES] = eligible[EXPOSURES + OUTCOMES].apply(pd.to_numeric, errors="coerce")
    eligible = eligible.loc[np.isfinite(eligible[EXPOSURES + OUTCOMES].to_numpy(dtype=float)).all(axis=1)].copy()
    n_fixed = min(wanted, len(eligible))
    if n_fixed:
        fixed = eligible.sample(n=n_fixed, random_state=int(config.get("seed", 20261028))).sort_values("cell_id")
        fixed_rows: list[pd.DataFrame] = []
        curve_rows: list[pd.DataFrame] = []
        for outcome in OUTCOMES:
            for exposure in EXPOSURES:
                data = fixed[[exposure, outcome]].copy()
                data["x_value"] = np.log1p(pd.to_numeric(data[exposure], errors="coerce"))
                data["y_value"] = pd.to_numeric(data[outcome], errors="coerce")
                data["exposure"] = exposure
                data["outcome"] = outcome
                data["sample_scope"] = f"fixed_random_{n_fixed}_cells"
                fixed_rows.append(data[["x_value", "y_value", "exposure", "outcome", "sample_scope"]])
                curve = _spline_curve(
                    data.rename(columns={"x_value": "__x", "y_value": "__y"}), "__x", "__y",
                    label_values={"exposure": exposure, "outcome": outcome, "sample_scope": f"fixed_random_{n_fixed}_cells"},
                    points=int(fig_conf.get("curve_points", 250)),
                )
                curve_rows.append(curve)
        fixed_source = pd.concat(fixed_rows, ignore_index=True)
        fixed_curves = pd.concat(curve_rows, ignore_index=True)
        outputs["fixed_20k"] = fixed_curves
        _curve_panel(
            figure_dir, "05a_fixed_20k_natural_spline", fixed_curves,
            title=f"固定随机样本：自然样条响应曲线（共同有效像元 {n_fixed:,} 个）",
            x_label="log(1 + 实际侵蚀量 Aa)",
            caption=f"对同一组确定性随机抽取的 {n_fixed:,} 个对齐像元拟合自然样条；这些像元在两个侵蚀时期和九项健康指标上均有完整值。本图是全域描述性样本，不是匹配像元拟合，也不是独立验证。",
            fixed_sample=fixed_source,
        )

    legacy_path = Path("outputs/exploratory_oracle/phase_f_20261026_seed20261026/figures/data/08_health_response_curves.csv")
    legacy = pd.read_csv(legacy_path) if (Path.cwd() / legacy_path).is_file() else pd.DataFrame()
    if not legacy.empty:
        legacy = legacy.loc[legacy["curve_type"].eq("selected_full_sample_apparent_model_partial_dependence")].copy()
        legacy = legacy.rename(columns={"exposure_value": "x_value", "predicted_outcome": "predicted_outcome"})
        legacy["exposure"] = "Aa_20yr"
        legacy["legacy"] = True
        legacy["estimand"] = "legacy_full_sample"
        outputs["legacy_full_sample"] = legacy
        _curve_panel(
            figure_dir, "05b_historical_full_sample_apparent", legacy,
            title="历史全样本模型择优表观曲线", x_label="20 年平均实际侵蚀量 Aa（沿用历史横轴尺度）",
            caption="继承自 F 阶段结果导向探索中 20 年平均实际侵蚀量的模型曲线。历史来源记录了 13 类模型的 1,024 次试验，并按全样本表观 R²选择模型。曲线使用原始侵蚀量横轴，且曾按健康结局择优；仅供历史参照，不与固定样本样条或本轮匹配曲线直接比较。",
        )

    winners = _selected_trials(run_dir)
    candidate_dir = run_dir / "pairs" / "candidate_pairs"
    for estimand, stem, ylabel in (
        ("matched_pixel", "05c_matched_pixel_natural_spline", "匹配像元健康响应"),
        ("matched_pair", "05d_matched_pair_natural_spline", "高侵蚀端减低侵蚀端的健康差值"),
    ):
        selected = winners.loc[winners["estimand"].eq(estimand)]
        curve_parts: list[pd.DataFrame] = []
        for candidate_id, candidate_rows in selected.groupby("candidate_id", sort=True):
            candidate_id = str(candidate_id)
            pairs_path = candidate_dir / f"{candidate_id}.parquet"
            if not pairs_path.is_file():
                continue
            pairs = pd.read_parquet(pairs_path)
            if estimand == "matched_pixel":
                endpoint_ids = pd.Index(pd.unique(pd.concat(
                    [pairs["high_erosion_cell_id"], pairs["low_erosion_cell_id"]], ignore_index=True
                )))
            for _, row in candidate_rows.iterrows():
                exposure, outcome = str(row["exposure"]), str(row["outcome"])
                if estimand == "matched_pixel":
                    data = cells.loc[cells["cell_id"].isin(endpoint_ids), ["cell_id", exposure, outcome]].drop_duplicates("cell_id").copy()
                    data["x_value"] = np.log1p(pd.to_numeric(data[exposure], errors="coerce"))
                    data["y_value"] = pd.to_numeric(data[outcome], errors="coerce")
                else:
                    data = pairs[["delta_log1p_Aa", f"delta_{outcome}"]].copy()
                    data["x_value"] = pd.to_numeric(data["delta_log1p_Aa"], errors="coerce")
                    data["y_value"] = pd.to_numeric(data[f"delta_{outcome}"], errors="coerce")
                curve = _spline_curve(
                    data[["x_value", "y_value"]].rename(columns={"x_value": "__x", "y_value": "__y"}),
                    "__x", "__y",
                    label_values={
                        "exposure": exposure, "outcome": outcome, "estimand": estimand,
                        "candidate_id": candidate_id, "selected_model_family": str(row["model_family"]),
                        "apparent_r_squared": float(row["r_squared"]),
                    },
                    points=int(fig_conf.get("curve_points", 250)),
                )
                curve_parts.append(curve)
                del data
            del pairs
        current = pd.concat(curve_parts, ignore_index=True) if curve_parts else pd.DataFrame()
        outputs[estimand] = current
        _curve_panel(
            figure_dir, stem, current,
            title=f"环境匹配样本中的非线性响应：{ylabel}",
            x_label="log(1 + 实际侵蚀量 Aa)" if estimand == "matched_pixel" else "Δlog(1 + Aa)，高侵蚀端 − 低侵蚀端",
            caption=f"每条曲线均为描述性自然样条，数据来自对应侵蚀时期、健康指标和 {'匹配像元' if estimand == 'matched_pixel' else '匹配配对'}分析中第二阶段表观模型 R²最高的候选。候选及模型类型先按表观拟合择优，图中曲线再单独拟合自然样条。样本量和候选编号见图源 CSV。配对差值按各时期自身的高侵蚀端减低侵蚀端计算。",
        )
    return outputs


def _figure_6(run_dir: Path, figure_dir: Path) -> None:
    data = _read_csv(run_dir / "tables", "group_slope_revised.csv")
    if data.empty:
        return
    subset = data.loc[
        data["exposure"].eq("Aa_20yr")
        & data["method"].eq("M3")
        & data["group_column"].eq("soil_type_group")
        & data["outcome"].isin(OUTCOMES)
        & data["inference_status"].eq("reliable")
    ].copy()
    subset["lower"] = pd.to_numeric(subset["cluster_robust_ci_low"], errors="coerce")
    subset["upper"] = pd.to_numeric(subset["cluster_robust_ci_high"], errors="coerce")
    subset["slope"] = pd.to_numeric(subset["slope"], errors="coerce")
    subset = subset.loc[np.isfinite(subset[["slope", "lower", "upper"]].to_numpy(dtype=float))].copy()
    fig, axes = plt.subplots(3, 3, figsize=(14, 9.5), squeeze=False)
    _axis_grid(axes, 9)
    group_colors = {"soil_26": "#3977a8", "soil_27": "#df8554", "soil_42": "#54a38b"}
    for ax, outcome in zip(axes.flat, OUTCOMES, strict=True):
        rows = subset.loc[subset["outcome"].eq(outcome)].sort_values("group_label")
        yloc = np.arange(len(rows))
        if len(rows):
            for y, row in zip(yloc, rows.itertuples(index=False), strict=True):
                ax.errorbar(row.slope, y, xerr=[[row.slope - row.lower], [row.upper - row.slope]],
                            fmt="o", color=group_colors.get(row.group_label, "#697783"),
                            markersize=5, capsize=2, linewidth=1.2)
            ax.set_yticks(yloc, rows["group_label"].map(_group_value_label))
        else:
            ax.set_yticks([])
            ax.text(0.5, 0.5, "无通过可靠性检查的区间", transform=ax.transAxes, ha="center", color="#7c8792")
        ax.axvline(0, color="#8995a0", linewidth=0.8)
        ax.set_title(OUTCOME_LABELS[outcome], loc="left", fontweight="bold")
        ax.set_xlabel("线性斜率：健康差值 / 侵蚀差值")
        ax.grid(axis="x", linewidth=0.5)
    fig.suptitle("土类分组斜率及通过可靠性检查的空间区间", x=0.02, y=1.02, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.02, -0.03, "20 年平均实际侵蚀 Aa，匹配方法 M3；只绘制通过可靠性检查的结果。线性斜率是辅助诊断，与秩相关和样条分析互补，并非唯一响应模型。", fontsize=8, color="#5c6b79")
    fig.tight_layout()
    _save_figure(fig, figure_dir, "06_group_effect_forest", subset,
        "森林图展示按土壤类型分层的配对线性斜率，响应为高侵蚀端减低侵蚀端的健康差值，暴露为 20 年平均侵蚀量对数差。只显示通过 H 阶段可靠性门槛的结果，区间按空间块聚类计算。这些斜率是辅助交互诊断；报告还提供秩相关与非线性分析。")


def _figure_7(run_dir: Path, figure_dir: Path) -> None:
    data = _read_csv(run_dir / "tables", "phase_h_environment_group_method_comparison.csv")
    if data.empty:
        return
    data = data.sort_values("median_spearman_rho", na_position="last").copy()
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5), gridspec_kw={"width_ratios": [1.25, 1]})
    ypos = np.arange(len(data))
    rho = pd.to_numeric(data["median_spearman_rho"], errors="coerce")
    stability = pd.to_numeric(data["mean_adjusted_rand_index"], errors="coerce")
    sizes = 40 + 90 * stability.fillna(0).clip(lower=0, upper=1)
    finite_stability = stability.notna()
    axes[0].scatter(rho[~finite_stability], ypos[~finite_stability], s=60, c="#9da8b2", edgecolor="white", linewidth=0.7, label="稳定性不适用")
    points = axes[0].scatter(rho[finite_stability], ypos[finite_stability], s=sizes[finite_stability], c=stability[finite_stability], cmap="viridis", vmin=0, vmax=1, edgecolor="white", linewidth=0.7)
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].axvline(0, color="#87939e", linewidth=0.8)
    axes[0].set_yticks(ypos, data["group_method"].map(_group_method_label))
    axes[0].set_xlabel("组内斯皮尔曼秩相关 ρ 中位数")
    axes[0].set_title("响应方向与 K 均值聚类稳定性", loc="left", fontweight="bold")
    colorbar = fig.colorbar(points, ax=axes[0], fraction=0.035, pad=0.02)
    colorbar.set_label("不同随机种子间平均调整兰德指数（ARI）\n（非 K 均值聚类方法不适用）")
    axes[0].grid(axis="x", linewidth=0.5)
    x = pd.to_numeric(data["median_environmental_distance"], errors="coerce")
    y = pd.to_numeric(data["median_within_group_nonlinear_r_squared"], errors="coerce")
    size = 35 + 0.12 * pd.to_numeric(data["median_pairs_per_group"], errors="coerce").fillna(0).clip(0, 2000)
    axes[1].scatter(x, y, s=size, c="#3977a8", alpha=0.82, edgecolor="white", linewidth=0.6)
    for row in data.itertuples(index=False):
        if np.isfinite(getattr(row, "median_environmental_distance", np.nan)) and np.isfinite(getattr(row, "median_within_group_nonlinear_r_squared", np.nan)):
            axes[1].annotate(_group_method_label(row.group_method),
                             (row.median_environmental_distance, row.median_within_group_nonlinear_r_squared),
                             xytext=(4, 3), textcoords="offset points", fontsize=7, color="#4c5c69")
    axes[1].set_xlabel("匹配配对的环境距离中位数")
    axes[1].set_ylabel("组内自然样条表观 R² 中位数")
    axes[1].set_title("匹配质量与描述性拟合", loc="left", fontweight="bold")
    axes[1].grid(linewidth=0.5)
    fig.suptitle("环境分组方法比较：健康响应、匹配重叠与稳定性", x=0.02, y=1.02, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.02, -0.02, "分组标签只依据环境数据生成。组内画像使用经健康结果筛选的候选；右侧点的大小随组内配对数中位数变化。", fontsize=8, color="#5c6b79")
    fig.tight_layout()
    _save_figure(fig, figure_dir, "07_environment_group_method_comparison", data,
        "每行代表一种环境分组方法，指标为该方法所有健康结局导向配对画像的中位数。左图显示组内斯皮尔曼秩相关方向；颜色表示 K 均值聚类更换随机种子后的调整兰德指数（ARI）稳定性，确定性类别/梯度不计算该指标。右图比较匹配后的环境距离与组内自然样条表观 R²，点大小对应组内配对数中位数。这些均为探索性描述，不构成确认性证据。")


def _figure_8(run_dir: Path, figure_dir: Path, cells: pd.DataFrame) -> None:
    profiles = _read_csv(run_dir / "tables", "phase_h_environment_group_profiles.csv")
    labels_path = run_dir / "tables" / "phase_h_environment_group_cell_labels.parquet"
    if profiles.empty or not labels_path.is_file():
        return
    target_method = "kmeans_terrain_k8"
    selected = profiles.loc[profiles["group_column"].eq(target_method)].copy()
    group_rank = selected.groupby("group_label", as_index=False).agg(
        median_spearman_rho=("spearman_rho", "median"),
        negative_profile_fraction=("spearman_rho", lambda values: float((pd.to_numeric(values, errors="coerce") < 0).mean())),
        median_pairs=("n_pairs", "median"),
        outcomes_profiled=("outcome", "nunique"),
        candidate_profiles=("candidate_id", "nunique"),
    ).sort_values(["median_spearman_rho", "median_pairs"], ascending=[True, False], kind="stable")
    if group_rank.empty:
        return
    chosen = str(group_rank.iloc[0]["group_label"])
    labels = pd.read_parquet(labels_path, columns=["cell_id", target_method])
    map_data = cells[["cell_id", "x", "y", "SoilType"]].merge(labels, on="cell_id", how="left", validate="one_to_one")
    selected_cells = map_data.loc[map_data[target_method].astype("string").eq(chosen)]
    background = map_data.loc[map_data[target_method].notna() & ~map_data[target_method].astype("string").eq(chosen)]
    if len(background) > 50000:
        background = background.sample(n=50000, random_state=20261028)
    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    ax.scatter(background["x"], background["y"], s=0.25, c="#cbd4dc", alpha=0.42, linewidths=0, rasterized=True)
    ax.scatter(selected_cells["x"], selected_cells["y"], s=1.2, c="#d95f43", alpha=0.86, linewidths=0, rasterized=True, label=f"组内斯皮尔曼秩相关中位数最负组：{_group_value_label(chosen)}")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("投影坐标东向（米）")
    ax.set_ylabel("投影坐标北向（米）")
    ax.set_title("耕地像元上的环境分组空间位置", loc="left", fontweight="bold", pad=12)
    ax.legend(frameon=False, loc="upper right")
    ax.grid(linewidth=0.4, alpha=0.6)
    fig.suptitle("按健康结果筛选的环境组空间分布（K 均值聚类，k=8）", x=0.02, y=1.01, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.02, -0.02, "高亮组依据全部健康指标、侵蚀时期和第二阶段候选中的组内斯皮尔曼秩相关中位数筛选。灰色背景为固定地图展示样本；红色显示该组全部像元。", fontsize=8, color="#5c6b79")
    fig.tight_layout()
    map_display = pd.concat(
        [background.assign(map_role="background_display_sample"), selected_cells.assign(map_role="selected_group_all_cells")],
        ignore_index=True,
    )
    export = pd.concat([group_rank.assign(map_role="group_selection_summary"), map_display], ignore_index=True, sort=False)
    _save_figure(fig, figure_dir, "08_selected_group_map", export,
        f"图中使用真实投影坐标。红色高亮组（{_group_value_label(chosen)}）来自 K 均值聚类 k=8 画像，并按跨健康指标、侵蚀时期及候选的组内斯皮尔曼秩相关中位数最负确定。这是结果导向探索中的展示选择，不代表经过独立验证的热点，也不是因果风险面。灰色背景为确定性地图展示样本；红色组显示全部像元。")


def _figure_9(run_dir: Path, figure_dir: Path) -> None:
    data = _read_csv(run_dir / "tables", "phase_h_spatial_validation_summary.csv")
    if data.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.4), sharex=True, sharey=True)
    palette = dict(zip(OUTCOMES, sns.color_palette("husl", n_colors=len(OUTCOMES)), strict=True))
    marker = {"Aa_20yr": "o", "Aa_2020": "s"}
    buffers = sorted(pd.to_numeric(data["buffer_m"], errors="coerce").dropna().unique())
    for ax, estimand in zip(axes, ("matched_pixel", "matched_pair"), strict=True):
        subset = data.loc[data["estimand"].eq(estimand)]
        for buffer_m, buffer_group in subset.groupby("buffer_m", sort=True):
            for row in buffer_group.itertuples(index=False):
                ax.scatter(row.mean_apparent_r_squared, row.mean_spatial_test_r_squared,
                           color=palette.get(row.outcome, "#637381"), marker=marker.get(row.exposure, "o"),
                           s=35 + float(buffer_m) / 1000.0 * 35, alpha=0.72, edgecolor="white", linewidth=0.4)
        ax.axline((0, 0), slope=1, color="#929da7", linewidth=0.8, linestyle="--")
        ax.axhline(0, color="#929da7", linewidth=0.6)
        ax.set_xlabel("全匹配样本表观拟合度 R²")
        ax.set_title("匹配像元" if estimand == "matched_pixel" else "匹配配对", loc="left", fontweight="bold")
        ax.grid(linewidth=0.5)
    axes[0].set_ylabel("10 次空间分块测试的平均 R²")
    outcome_handles = [Line2D([0], [0], marker="o", linestyle="", color=palette[o], label=OUTCOME_LABELS[o]) for o in OUTCOMES]
    period_handles = [Line2D([0], [0], marker=marker[e], linestyle="", color="#637381", label=EXPOSURE_LABELS[e]) for e in EXPOSURES]
    buffer_handles = [Line2D([0], [0], marker="o", linestyle="", color="#637381", markersize=4 + b / 300, label=f"{int(b)} 米缓冲") for b in buffers]
    axes[1].legend(handles=outcome_handles + period_handles + buffer_handles, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, frameon=False)
    fig.suptitle("全样本表观拟合与空间留出测试表现", x=0.02, y=1.02, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.02, -0.04, "按完整 5 km 网格块分配角色；训练样本中距验证/测试样本小于设定缓冲距离的记录会被剔除。模型族在全样本上已择优，因此空间分数是迁移诊断，不是完全未接触数据的模型选择测试。", fontsize=8, color="#5c6b79")
    fig.tight_layout()
    _save_figure(fig, figure_dir, "09_apparent_vs_spatial_test", data,
        "比较全匹配样本表观 R² 与 10 次整块 5 km 空间划分的测试 R²，匹配像元与配对分析分开显示。空间评分前，候选和模型族已按表观表现选择；因此空间测试用于诊断已选结果的地理迁移能力，不是独立的模型选择程序。点的大小代表缓冲距离。")


def _figure_10(run_dir: Path, figure_dir: Path) -> None:
    data = _read_csv(run_dir / "tables", "interaction_test_comparison.csv")
    if data.empty:
        return
    data["legacy_q"] = pd.to_numeric(data["legacy_interaction_q_value_global"], errors="coerce").clip(lower=1e-300)
    data["revised_q"] = pd.to_numeric(data["interaction_q_value_revised"], errors="coerce").clip(lower=1e-300)
    data["legacy_neglog10_q"] = -np.log10(data["legacy_q"])
    data["revised_neglog10_q"] = -np.log10(data["revised_q"])
    reliable = data["revised_inference_reliable"].fillna(False).astype(bool)
    plotted = data.loc[reliable & np.isfinite(data[["legacy_q", "revised_q"]].to_numpy(dtype=float)).all(axis=1)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.5), gridspec_kw={"width_ratios": [1.1, 0.9]})
    ax = axes[0]
    upper = max(
        float(plotted["legacy_neglog10_q"].max()) if not plotted.empty else 0.0,
        float(plotted["revised_neglog10_q"].max()) if not plotted.empty else 0.0,
        2.0,
    )
    ax.scatter(plotted["legacy_neglog10_q"], plotted["revised_neglog10_q"],
               c="#3977a8", s=28, alpha=0.75, edgecolor="white", linewidth=0.3,
               label="修订后通过可靠性检查")
    ax.plot([0, upper], [0, upper], color="#7f8a94", linewidth=0.9, linestyle="--")
    threshold = -np.log10(0.05)
    ax.axvline(threshold, color="#aab4bd", linewidth=0.8, linestyle=":")
    ax.axhline(threshold, color="#aab4bd", linewidth=0.8, linestyle=":")
    ax.set_xlim(left=0, right=upper * 1.03)
    ax.set_ylim(bottom=0, top=upper * 1.03)
    ax.set_xlabel("旧版全局 q 值的 −log10")
    ax.set_ylabel("修订版全局 q 值的 −log10")
    ax.set_title("通过可靠性检查的检验：旧版与修订版 q 值", loc="left", fontweight="bold", pad=12)
    ax.grid(linewidth=0.5)
    ax.legend(frameon=False, loc="upper left")
    status_counts = data.groupby("revised_status", dropna=False).size().sort_values()
    status_colors = ["#9da8b2" if str(value).startswith("unreliable") else "#54a38b" for value in status_counts.index]
    axes[1].barh([_status_label(value) for value in status_counts.index], status_counts.values, color=status_colors, height=0.62)
    axes[1].set_xlabel("交互检验数量")
    axes[1].set_title("可靠性检查结果", loc="left", fontweight="bold", pad=12)
    axes[1].grid(axis="x", linewidth=0.5)
    for y, count in enumerate(status_counts.values):
        axes[1].text(count + max(status_counts.values) * 0.015, y, f"{count}", va="center", fontsize=8)
    fig.suptitle("H1 交互分析：修订前后推断诊断", x=0.02, y=1.01, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.02, -0.02, "虚线表示 q=0.05。未通过可靠性检查的检验仍计入状态数量，但不报告修订后的 p/q 值。", fontsize=8, color="#5c6b79")
    fig.tight_layout()
    _save_figure(fig, figure_dir, "10_h1_interaction_inference_repair", data,
        "左图比较通过修订版可靠性门槛的检验，其虚线为 q=0.05，斜线为两版 q 值相等。右图按修订版诊断状态统计全部检验，包括未报告 q 值的检验。若设计矩阵秩、条件数、空间块支持或数值容差敏感性不合格，则修订流程会抑制 p/q 值；这不等于没有交互关系。")


def _figure_11(run_dir: Path, figure_dir: Path) -> None:
    raw = _read_csv(run_dir / "tables", "phase_h_period_comparison_by_split.csv")
    if raw.empty:
        return
    test = raw.loc[raw["evaluation_set"].eq("spatial_test") & raw["status"].eq("ok")].copy()
    if test.empty:
        return
    summary = test.groupby(["exposure", "outcome", "estimand"], as_index=False).agg(
        median_spatial_test_r_squared=("r_squared", "median"),
        q25_spatial_test_r_squared=("r_squared", lambda values: float(pd.Series(values).quantile(0.25))),
        q75_spatial_test_r_squared=("r_squared", lambda values: float(pd.Series(values).quantile(0.75))),
        common_sample_n=("common_sample_n", "first"),
        n_seed_splits=("seed", "nunique"),
        median_spearman_rho=("spearman_rho", "median"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.3), sharey=True)
    for ax, estimand in zip(axes, ("matched_pixel", "matched_pair"), strict=True):
        subset = summary.loc[summary["estimand"].eq(estimand)]
        for exp_index, exposure in enumerate(EXPOSURES):
            rows = subset.loc[subset["exposure"].eq(exposure)].set_index("outcome").reindex(OUTCOMES).dropna(subset=["median_spatial_test_r_squared"])
            x = np.array([OUTCOMES.index(str(outcome)) for outcome in rows.index]) + (-0.14 if exposure == "Aa_20yr" else 0.14)
            mid = rows["median_spatial_test_r_squared"].to_numpy(dtype=float)
            low = rows["q25_spatial_test_r_squared"].to_numpy(dtype=float)
            high = rows["q75_spatial_test_r_squared"].to_numpy(dtype=float)
            ax.errorbar(x, mid, yerr=[mid - low, high - mid], fmt="o", color=EXPOSURE_COLORS[exposure],
                        capsize=2, linewidth=1.2, markersize=5, label=EXPOSURE_LABELS[exposure])
        ax.axhline(0, color="#87939e", linewidth=0.8)
        ax.set_xticks(range(len(OUTCOMES)), [OUTCOME_LABELS[o] for o in OUTCOMES], rotation=35, ha="right")
        ax.set_title("共同匹配像元" if estimand == "matched_pixel" else "共同匹配配对", loc="left", fontweight="bold")
        ax.set_xlabel("健康指标")
        ax.grid(axis="y", linewidth=0.5)
    axes[0].set_ylabel("空间测试 R² 中位数（10 次共同划分的四分位距）")
    axes[1].legend(frameon=False, loc="best")
    fig.suptitle("两个侵蚀时期在共同样本与相同空间划分上的比较", x=0.02, y=1.02, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.02, -0.04, "两个时期的配对方向均按 20 年平均实际侵蚀量固定。曲线使用同一自然样条模型、相同 5 km 空间块和 500 米缓冲；该结果是描述性比较，不是时期的因果效应。", fontsize=8, color="#5c6b79")
    fig.tight_layout()
    _save_figure(fig, figure_dir, "11_common_sample_period_comparison", raw,
        "对每项健康指标和分析单位，20 年平均实际侵蚀与 2020 年实际侵蚀在完全相同的像元编号或规范化匹配配对上评分，并使用相同的 5 km 空间块、随机种子和 500 米缓冲。两个时期的配对方向均按 20 年平均实际侵蚀量固定。点为空间测试 R²中位数，区间为共同划分下的四分位距。这是预测模式的描述性对照，不估计时期的因果效应。")


def generate_phase_h_figures(run_dir: Path, cells: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """Create all Phase H figure exports and their CSV data/captions."""

    figure_dir = Path(run_dir) / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(STYLE)
    sns.set_theme(style="whitegrid", rc=STYLE)
    _figure_1(run_dir, figure_dir)
    _figure_2(run_dir, figure_dir)
    models = _figure_3(run_dir, figure_dir)
    _figure_4(run_dir, figure_dir, models)
    curves = _figure_5(run_dir, figure_dir, cells, config)
    _figure_6(run_dir, figure_dir)
    _figure_7(run_dir, figure_dir)
    _figure_8(run_dir, figure_dir, cells)
    _figure_9(run_dir, figure_dir)
    _figure_10(run_dir, figure_dir)
    _figure_11(run_dir, figure_dir)
    figures = sorted(path.stem for path in (figure_dir / "png").glob("*.png"))
    manifest = {
        "figure_count": len(figures),
        "figures": figures,
        "formats": ["png", "svg", "pdf"],
        "fixed_plot_sample_size_requested": int(config.get("response_curves", {}).get("fixed_plot_sample_size", 20000)),
        "outcome_guided_selection": True,
    }
    (figure_dir / "figure_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"manifest": manifest, "curve_sources": curves}
