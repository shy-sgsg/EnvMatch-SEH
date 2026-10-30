"""Chinese publication figures for the distribution-diagnostic addendum."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.low_erosion_regime.figures import (
    COLORS,
    EXPOSURE_LABELS,
    _save,
    _setup,
)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


EXPOSURES = ("Aa_20yr", "Aa_2020")
MARKERS_BY_K = {1: "o", 2: "o", 3: "s", 4: "^"}
PLOT_COLORS = {"Aa_20yr": COLORS["blue"], "Aa_2020": COLORS["coral"]}


def _first_boundary(value: object) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    try:
        for token in str(value).split("|"):
            boundary = float(token)
            if np.isfinite(boundary):
                return float(10 ** boundary)
        return np.nan
    except (TypeError, ValueError, OverflowError):
        return np.nan


def _record_figure(
    fig: plt.Figure,
    figure_root: Path,
    *,
    stem: str,
    source: pd.DataFrame,
    caption: str,
    question: str,
    chart_type: str,
) -> dict[str, str]:
    return _save(
        fig,
        figure_root,
        stem,
        source,
        caption,
        scientific_question=question,
        experiment="低侵蚀分布诊断补充",
        figure_type=chart_type,
    )


def _plot_gmm_seed_stability(run_dir: Path, figure_root: Path) -> dict[str, str]:
    trials = pd.read_csv(run_dir / "diagnostics/gmm_seed_stability_by_seed.csv")
    selected = trials.loc[trials["selected_by_bic"].fillna(False)].copy()
    selected["selected_k_bic"] = pd.to_numeric(selected["selected_k_bic"], errors="coerce")
    selected["boundary_first_original"] = selected["adjacent_boundaries_log10"].map(_first_boundary)
    selected = selected.loc[selected["boundary_first_original"].gt(0)].copy()

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2), sharey=False)
    source_rows = []
    for ax, exposure in zip(axes, EXPOSURES, strict=True):
        part = selected.loc[selected["exposure"].eq(exposure)].sort_values("seed").copy()
        if part.empty:
            ax.text(.5, .5, "沒有可用的多成分边界", transform=ax.transAxes,
                    ha="center", va="center", color=COLORS["slate"])
            ax.set_axis_off()
            continue
        counts = part["selected_k_bic"].astype(int).value_counts().sort_index()
        count_label = "；".join(f"K={k}：{count}/{part['seed'].nunique()}" for k, count in counts.items())
        x = np.arange(len(part))
        values = part["boundary_first_original"].to_numpy(float)
        q_low, median, q_high = np.quantile(values, [.025, .5, .975])
        color = PLOT_COLORS[exposure]
        ax.axhspan(q_low, q_high, color=color, alpha=.12, zorder=0)
        ax.axhline(median, color=COLORS["ink"], lw=1.35, ls="--", zorder=1)
        for component_count, group in part.groupby("selected_k_bic", sort=True):
            locations = group.index.map(part.index.get_loc).to_numpy(dtype=float)
            ax.scatter(
                locations,
                group["boundary_first_original"],
                s=54,
                marker=MARKERS_BY_K.get(int(component_count), "o"),
                color=color,
                edgecolor="white",
                linewidth=.65,
                zorder=3,
                label=f"BIC 选 K={int(component_count)}",
            )
        ax.set_yscale("log")
        ax.set_xticks(x, [str(i + 1) for i in range(len(part))])
        ax.set_xlabel("固定随机种子序号")
        ax.set_ylabel("首个相邻成分分界（Aa 原值，对数轴）")
        ax.set_title(f"{EXPOSURE_LABELS[exposure]}\n{count_label}", loc="left", fontweight="bold")
        ax.grid(axis="y", color="#E7EDF0", lw=.75, which="both")
        ax.grid(axis="x", visible=False)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(
            .98, .04,
            f"中位数 {median:.3g}  ·  CV {np.std(values, ddof=1) / np.mean(values):.1%}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
            color=COLORS["ink"],
        )
        for _, record in part.iterrows():
            source_rows.append({
                "exposure": exposure,
                "seed": int(record["seed"]),
                "seed_order": int(part.index.get_loc(record.name) + 1),
                "bic_selected_components": int(record["selected_k_bic"]),
                "first_adjacent_boundary_Aa_original": float(record["boundary_first_original"]),
                "converged": bool(record["converged"]),
            })

    legend_handles = [
        Line2D([0], [0], marker=MARKERS_BY_K[k], linestyle="", color=COLORS["ink"],
               markerfacecolor=COLORS["blue"], markeredgecolor="white", label=f"BIC 选中 K={k}")
        for k in (2, 3, 4)
    ]
    legend_handles.extend([
        Line2D([0], [0], color=COLORS["ink"], lw=1.35, ls="--", label="跨种子中位数"),
        Patch(facecolor=COLORS["blue"], edgecolor="none", alpha=.16, label="2.5%–97.5%经验分位范围"),
    ])
    fig.suptitle("GMM 选中成分数与首处分界的随机种子敏感性", x=.03, ha="left", y=1.02,
                 fontsize=13, fontweight="bold")
    fig.legend(handles=legend_handles, loc="lower center", ncol=5, bbox_to_anchor=(.5, -.035),
               frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, .075, 1, .96))
    caption = (
        "每个点对应一个固定随机种子下由 BIC 选中的 GMM 模型；点形表示成分数，纵轴为首个相邻成分边界的 Aa 原值。"
        "虚线为跨种子中位数，浅色带为种子经验分位范围（2.5%–97.5%），不是参数置信区间。"
        "20 年平均实际侵蚀的选中 K 在 2 与 3 之间变化；2020 年选中 K 虽一致，首处分界仍需结合较大的种子间波动解读。"
    )
    return _record_figure(
        fig, figure_root, stem="HADD01_gmm_seed_stability", source=pd.DataFrame(source_rows),
        caption=caption, question="GMM 的成分数和分界是否受随机种子影响？", chart_type="随机种子稳定性图",
    )


def _plot_posterior_sensitivity(run_dir: Path, figure_root: Path) -> dict[str, str]:
    data = pd.read_csv(run_dir / "diagnostics/gmm_posterior_cutoff_sensitivity.csv")
    components = [
        ("n_low_component_assigned", "低值成分", COLORS["ochre"]),
        ("n_other_component_assigned", "其他成分", COLORS["blue"]),
        ("n_uncertain_positive", "归属不确定", COLORS["slate"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.9), sharey=True)
    for ax, exposure in zip(axes, EXPOSURES, strict=True):
        part = data.loc[data["exposure"].eq(exposure)].sort_values("posterior_cutoff")
        if part.empty:
            ax.text(.5, .5, "没有可用的后验分类", transform=ax.transAxes, ha="center", va="center")
            ax.set_axis_off()
            continue
        x = np.arange(len(part))
        denominators = pd.to_numeric(part["n_positive"], errors="coerce").to_numpy(float)
        bottoms = np.zeros(len(part), dtype=float)
        for column, label, color in components:
            counts = pd.to_numeric(part[column], errors="coerce").fillna(0).to_numpy(float)
            fractions = np.divide(counts, denominators, out=np.zeros_like(counts), where=denominators > 0)
            ax.bar(x, fractions, bottom=bottoms, color=color, width=.58, label=label,
                   edgecolor="white", linewidth=.65)
            for position, fraction, bottom in zip(x, fractions, bottoms, strict=True):
                if fraction >= .12:
                    ax.text(position, bottom + fraction / 2, f"{fraction:.0%}", ha="center", va="center",
                            fontsize=8, color="white" if label != "归属不确定" else COLORS["ink"], fontweight="bold")
            bottoms += fractions
        uncertain_fraction = pd.to_numeric(part["uncertain_fraction_positive"], errors="coerce").to_numpy(float)
        for position, fraction in zip(x, uncertain_fraction, strict=True):
            ax.text(position, 1.025, f"不确定 {fraction:.0%}", ha="center", va="bottom", fontsize=7.8,
                    color=COLORS["ink"], clip_on=False)
        n_positive = int(part["n_positive"].iloc[0])
        ax.set_xticks(x, [f"{cutoff:.2f}" for cutoff in part["posterior_cutoff"]])
        ax.set_ylim(0, 1.18)
        ax.set_xlabel("最大后验概率阈值")
        ax.set_title(f"{EXPOSURE_LABELS[exposure]}\n正值像元 n={n_positive:,}", loc="left", fontweight="bold")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
        ax.grid(axis="y", color="#E7EDF0", lw=.75)
        ax.grid(axis="x", visible=False)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("正侵蚀像元占比")
    fig.suptitle("GMM 后验阈值提高时的像元归属变化", x=.03, ha="left", y=1.02,
                 fontsize=13, fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(.5, -.035), frameon=False)
    fig.tight_layout(rect=(0, .075, 1, .94))
    caption = (
        "每个柱形以正 Aa 像元为分母，分成低值成分、高值成分和最大后验概率低于指定阈值的归属不确定像元。"
        "柱顶直接标出不确定比例，柱内标注较大的组分比例。阈值越高，分类更保守，但不确定像元比例上升；"
        "2020 年 GMM 的大部分正值像元在三个阈值下均不能高置信归属。"
    )
    return _record_figure(
        fig, figure_root, stem="HADD02_gmm_posterior_sensitivity", source=data,
        caption=caption, question="GMM 后验分类的不确定像元比例如何随阈值改变？", chart_type="后验分类敏感性图",
    )


def _method_label(method: str) -> tuple[str, str]:
    if method.startswith("KDE"):
        return "KDE", "kde"
    if method.startswith("GMM"):
        match = re.search(r"boundary_(\d+)", method)
        suffix = match.group(1) if match else "1"
        return f"GMM 分界 {suffix}", "gmm"
    if method == "empirical_CDF_breakpoint":
        return "经验 CDF 单折点", "cdf"
    if method == "Otsu":
        return "Otsu 分箱对照", "otsu"
    return method, "other"


def _plot_threshold_candidates(run_dir: Path, figure_root: Path) -> dict[str, str]:
    candidates = pd.read_csv(run_dir / "thresholds/low_erosion_threshold_candidates.csv")
    candidates["threshold_value"] = pd.to_numeric(candidates["threshold_value"], errors="coerce")
    candidates["bootstrap_ci_low"] = pd.to_numeric(candidates["bootstrap_ci_low"], errors="coerce")
    candidates["bootstrap_ci_high"] = pd.to_numeric(candidates["bootstrap_ci_high"], errors="coerce")
    candidates["display_label"] = candidates["method"].map(lambda value: _method_label(str(value))[0])
    candidates["method_kind"] = candidates["method"].map(lambda value: _method_label(str(value))[1])
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.1), sharey=True)
    y_map = {"KDE": 0, "GMM 分界 1": 1, "GMM 分界 2": 2, "GMM 分界 3": 3,
             "经验 CDF 单折点": 4, "Otsu 分箱对照": 5}
    source = candidates.copy()

    for ax, exposure in zip(axes, EXPOSURES, strict=True):
        part = candidates.loc[candidates["exposure"].eq(exposure)].copy()
        kde = part.loc[part["method_kind"].eq("kde")]
        kde_values = kde.loc[kde["threshold_value"].gt(0)]
        if kde_values.empty:
            ax.text(.98, y_map["KDE"] + .22, "未识别可靠谷值", transform=ax.get_yaxis_transform(),
                    ha="right", va="bottom", fontsize=8.5, color=COLORS["slate"])
        else:
            ax.scatter(kde_values["threshold_value"], [y_map["KDE"]] * len(kde_values),
                       marker="D", color=COLORS["blue"], s=40, zorder=3)

        gmm = part.loc[part["method_kind"].eq("gmm") & part["threshold_value"].gt(0)]
        for _, row in gmm.iterrows():
            label = row["display_label"]
            y = y_map.get(label, y_map["GMM 分界 3"])
            rejected = str(row.get("status", "")).startswith("rejected")
            ax.scatter(
                row["threshold_value"], y, marker="o", s=55,
                facecolors="white" if rejected else COLORS["lavender"],
                edgecolors=COLORS["lavender"], linewidths=1.4, zorder=4,
            )

        cdf = part.loc[part["method_kind"].eq("cdf") & part["threshold_value"].gt(0)]
        for _, row in cdf.iterrows():
            threshold = float(row["threshold_value"])
            low, high = row["bootstrap_ci_low"], row["bootstrap_ci_high"]
            if np.isfinite(low) and np.isfinite(high) and low <= threshold <= high:
                ax.errorbar(threshold, y_map["经验 CDF 单折点"],
                            xerr=[[threshold - low], [high - threshold]], fmt="o",
                            color=COLORS["navy"], ecolor=COLORS["navy"], elinewidth=1.4,
                            capsize=3, markersize=6.5, zorder=5)
            else:
                ax.scatter(threshold, y_map["经验 CDF 单折点"], marker="o", color=COLORS["navy"], s=45)

        otsu = part.loc[part["method_kind"].eq("otsu") & part["threshold_value"].gt(0)].sort_values("threshold_value")
        if not otsu.empty:
            values = otsu["threshold_value"].to_numpy(float)
            ax.hlines(y_map["Otsu 分箱对照"], values.min(), values.max(), color=COLORS["ochre"], lw=2.2, alpha=.65)
            jitter = np.linspace(-.11, .11, len(values)) if len(values) > 1 else np.array([0.0])
            ax.scatter(values, y_map["Otsu 分箱对照"] + jitter, marker="x", color=COLORS["ochre"],
                       s=48, linewidth=1.6, zorder=4)

        values = part["threshold_value"].dropna().to_numpy(float)
        if len(values):
            ax.set_xscale("log")
            ax.set_xlim(max(values.min() * .62, np.finfo(float).tiny), values.max() * 1.65)
        else:
            ax.set_xscale("log")
            ax.set_xlim(.001, 10)
        ax.set_yticks(list(y_map.values()), list(y_map.keys()))
        ax.invert_yaxis()
        ax.set_xlabel("阈值（Aa 原值，对数轴）")
        ax.set_title(EXPOSURE_LABELS[exposure], loc="left", fontweight="bold")
        ax.grid(axis="x", color="#E7EDF0", lw=.75, which="both")
        ax.grid(axis="y", visible=False)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("阈值识别方法")
    handles = [
        Line2D([0], [0], marker="o", color=COLORS["navy"], linestyle="", label="CDF 折点；线段为 95%空间重抽样分位区间"),
        Line2D([0], [0], marker="o", markerfacecolor="white", markeredgecolor=COLORS["lavender"],
               color=COLORS["lavender"], linestyle="", label="高重叠 GMM 分界（不作为可靠阈值）"),
        Line2D([0], [0], marker="x", color=COLORS["ochre"], linestyle="", label="Otsu 分箱候选；短线表示范围"),
    ]
    fig.suptitle("不同分布方法给出的 Aa 阈值与稳定性", x=.03, ha="left", y=1.02,
                 fontsize=13, fontweight="bold")
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(.5, -.005),
               frameon=False, fontsize=8.2)
    fig.tight_layout(rect=(0, .08, 1, .95))
    caption = (
        "横轴为 Aa 原值阈值并使用对数刻度。CDF 点两侧线段表示 100 次 5 km 空间分块重抽样的 2.5%–97.5%经验分位区间；"
        "Otsu 的短线表示不同直方图分箱得到的候选范围；空心 GMM 点表示成分重叠超出预设可分性界限；"
        "KDE 未达到双峰分离判据时明确标记为未识别可靠谷值。阈值单位沿用 Aa 原值，输入元数据未提供单位说明。"
    )
    return _record_figure(
        fig, figure_root, stem="HADD03_threshold_candidate_stability", source=source,
        caption=caption, question="各分布方法的阈值是否收敛到相似范围？", chart_type="阈值比较图",
    )


def generate_distribution_addendum_figures(run_dir: Path) -> Path:
    """Render three source-linked Chinese figures from saved addendum tables."""
    run_dir = Path(run_dir)
    _setup()
    figure_root = run_dir / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    manifest = [
        _plot_gmm_seed_stability(run_dir, figure_root),
        _plot_posterior_sensitivity(run_dir, figure_root),
        _plot_threshold_candidates(run_dir, figure_root),
    ]
    manifest_path = figure_root / "figure_manifest.json"
    manifest_path.write_text(json.dumps({"figures": manifest}, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path
