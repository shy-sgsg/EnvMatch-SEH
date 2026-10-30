"""Publication-style Chinese figures for the low-erosion regime analysis."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/low_erosion_regime_matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from scipy.stats import gaussian_kde, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer

from experiments.low_erosion_regime.spatial_analysis import REGIME_CODES, regime_grid


EXPOSURES = ("Aa_20yr", "Aa_2020")
EXPOSURE_LABELS = {"Aa_20yr": "20 年平均实际侵蚀", "Aa_2020": "2020 年实际侵蚀"}
OUTCOMES = ("SHI_candidate", "AN", "OM", "CEC", "pH", "SW", "BD", "MBC", "RESP")
OUTCOME_LABELS = {
    "SHI_candidate": "土壤健康指数（SHI）", "AN": "碱解氮（AN）", "OM": "有机质（OM）",
    "CEC": "阳离子交换量（CEC）", "pH": "土壤酸碱度（pH）", "SW": "土壤含水量（SW）",
    "BD": "土壤容重（BD）", "MBC": "微生物量碳（MBC）", "RESP": "土壤呼吸（RESP）",
}
SOIL_TYPE_LABELS = {26: "冲积土", 27: "潮土", 42: "脱潮土", 85: "江及河"}
COLORS = {"navy": "#25455C", "blue": "#3A7C92", "teal": "#69A99A", "ochre": "#D7A548",
          "coral": "#CE7663", "lavender": "#8A7BAE", "slate": "#9BAAB2", "ink": "#253542"}
REGIME_COLORS = ["#D5DEE3", "#3B8190", "#D8AB55", "#9B86B7"]
REGIME_LABELS = ["零侵蚀", "低值正侵蚀", "主体正侵蚀", "分类不确定"]

STYLE = {
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.edgecolor": "#D5DEE4", "axes.labelcolor": COLORS["ink"], "text.color": COLORS["ink"],
    "xtick.color": "#52636E", "ytick.color": "#52636E", "grid.color": "#E7EDF0",
    "font.family": "Noto Sans CJK JP", "font.size": 9, "axes.titlesize": 11,
    "axes.labelsize": 9, "legend.frameon": False, "svg.fonttype": "none",
}


def _setup():
    plt.rcParams.update(STYLE)
    sns.set_theme(style="whitegrid", rc=STYLE)


def _soil_profile_label(group_variable: str, profile: object) -> str:
    text = str(profile)
    if group_variable != "SoilType":
        return text
    match = re.fullmatch(r"(?:土类\s*)?(\d+)", text)
    if match:
        code = int(match.group(1))
        return f"{code} {SOIL_TYPE_LABELS.get(code, '其他类别')}"
    try:
        code = int(float(text))
    except (TypeError, ValueError):
        return text
    return f"{code} {SOIL_TYPE_LABELS.get(code, '其他类别')}"


def _save(fig: plt.Figure, root: Path, stem: str, source: pd.DataFrame, caption: str,
          *, scientific_question: str, experiment: str, figure_type: str) -> dict[str, str]:
    root = Path(root)
    for fmt in ("png", "svg", "pdf"):
        target = root / fmt / f"{stem}.{fmt}"
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=320, bbox_inches="tight", facecolor="white")
    source_path = root / "data" / f"{stem}_source.csv"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source.to_csv(source_path, index=False, encoding="utf-8-sig")
    caption_path = root / "captions" / f"{stem}.md"
    caption_path.parent.mkdir(parents=True, exist_ok=True)
    caption_path.write_text(caption.rstrip() + "\n", encoding="utf-8")
    plt.close(fig)
    return {"id": stem, "title": caption.split("。", 1)[0], "scientific_question": scientific_question,
            "experiment": experiment, "figure_type": figure_type, "png": f"png/{stem}.png",
            "svg": f"svg/{stem}.svg", "pdf": f"pdf/{stem}.pdf",
            "source": f"data/{stem}_source.csv", "caption": f"captions/{stem}.md"}


def _fit_kde(values: np.ndarray, *, base: str = "scott", multiplier: float = 1.0):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) > 30000:
        rng = np.random.default_rng(20261029)
        values = values[rng.choice(len(values), 30000, replace=False)]
    if len(values) < 20 or np.ptp(values) < 1e-8:
        return None
    pad = max(np.ptp(values) * .04, .08)
    grid = np.linspace(values.min() - pad, values.max() + pad, 400)
    estimator = gaussian_kde(values, bw_method=base)
    estimator.set_bandwidth(bw_method=estimator.factor * multiplier)
    return grid, estimator(grid)


def create_distribution_figures(
    figure_root: Path, cells: pd.DataFrame, summaries: pd.DataFrame, kde_candidates: pd.DataFrame,
    gmm_models: pd.DataFrame, gmm_assignments: dict[str, pd.DataFrame],
    cdf_candidates: pd.DataFrame, otsu_candidates: pd.DataFrame, bootstrap: pd.DataFrame,
    profile_summaries: pd.DataFrame, threshold_candidates: pd.DataFrame,
) -> list[dict[str, str]]:
    _setup()
    manifest = []
    source_rows = []
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.6), constrained_layout=True)
    for row, exposure in enumerate(EXPOSURES):
        values = pd.to_numeric(cells[exposure], errors="coerce").to_numpy(dtype=float)
        valid = cells["valid_cell"].fillna(False).astype(bool).to_numpy() & np.isfinite(values)
        finite = values[valid]
        positive = finite[finite > 0]
        zero_fraction = float(np.mean(finite == 0)) if len(finite) else np.nan
        axes[row, 0].barh([0], [zero_fraction], color=COLORS["navy"], height=.48, label="零值")
        axes[row, 0].barh([0], [1 - zero_fraction], left=[zero_fraction], color="#DDE5E9", height=.48, label="正值")
        axes[row, 0].text(zero_fraction / 2 if zero_fraction else .02, 0, f"零值 {zero_fraction:.1%}",
                          ha="center", va="center", color="white" if zero_fraction > .12 else COLORS["ink"], fontweight="bold")
        axes[row, 0].text(zero_fraction + (1 - zero_fraction) / 2, 0, f"正值 {1-zero_fraction:.1%}",
                          ha="center", va="center", color=COLORS["ink"], fontweight="bold")
        axes[row, 0].set_xlim(0, 1)
        axes[row, 0].set_yticks([])
        axes[row, 0].set_xlabel("有效像元比例")
        axes[row, 0].set_title(f"{EXPOSURE_LABELS[exposure]}：零值质量与正值占比", loc="left", fontweight="bold")
        axes[row, 0].grid(axis="x", linewidth=.5)
        if len(positive):
            logs = np.log10(positive)
            sample = logs if len(logs) <= 50000 else logs[np.random.default_rng(20261029 + row).choice(len(logs), 50000, replace=False)]
            sns.histplot(sample, stat="density", bins=50, color="#E7EEF1", edgecolor="white", ax=axes[row, 1])
            kde = _fit_kde(sample)
            if kde:
                axes[row, 1].plot(*kde, color=COLORS["blue"], lw=2.1, label="Scott KDE（描述性）")
            axes[row, 1].axvline(np.median(logs), color=COLORS["ochre"], lw=1.3, ls="--", label="中位数")
            axes[row, 1].set_xlabel("正侵蚀值 log10(Aa)")
            axes[row, 1].set_ylabel("密度")
            axes[row, 1].legend(loc="upper left", fontsize=8)
            axes[row, 1].set_title(f"{EXPOSURE_LABELS[exposure]}：正值分布（不含零值）", loc="left", fontweight="bold")
            for q, value in zip((.1, .25, .5, .75, .9), np.quantile(positive, [.1, .25, .5, .75, .9]), strict=True):
                source_rows.append({"exposure": exposure, "zero_fraction": zero_fraction,
                                    "quantile": q, "positive_quantile_original": float(value),
                                    "positive_quantile_log10": float(np.log10(value))})
    manifest.append(_save(fig, figure_root, "D01_zero_and_positive_distributions", pd.DataFrame(source_rows),
                          "零侵蚀与正侵蚀的分布分开呈现。零值是有效观测中的独立质量点；右侧密度图只使用正侵蚀值并在 log10 尺度展示。",
                          scientific_question="低值与零值是否构成独立侵蚀状态？", experiment="分布诊断", figure_type="分布统计图"))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    kde_source = []
    multipliers = (.5, 1.0, 1.5, 2.0)
    line_colors = [COLORS["lavender"], COLORS["blue"], COLORS["teal"], COLORS["ochre"]]
    for ax, exposure in zip(axes, EXPOSURES, strict=True):
        values = pd.to_numeric(cells.loc[cells["valid_cell"].fillna(False), exposure], errors="coerce").to_numpy(float)
        positive = values[np.isfinite(values) & (values > 0)]
        logs = np.log10(positive)
        sample = logs if len(logs) <= 30000 else logs[np.random.default_rng(20261029).choice(len(logs), 30000, replace=False)]
        for multiplier, color in zip(multipliers, line_colors, strict=True):
            curve = _fit_kde(sample, base="scott", multiplier=multiplier)
            if curve:
                x, y = curve
                ax.plot(x, y, color=color, lw=1.7, label=f"带宽 × {multiplier:g}")
                kde_source.extend({"exposure": exposure, "bandwidth_multiplier": multiplier,
                                   "log10_Aa": float(xv), "density": float(yv)} for xv, yv in zip(x, y, strict=True))
        selected = kde_candidates.loc[kde_candidates["exposure"].eq(exposure) & kde_candidates["bandwidth_base"].eq("scott")]
        for rec in selected.loc[selected["status"].eq("valid_valley")].to_dict("records"):
            ax.axvline(rec["threshold_log10"], color=COLORS["coral"], ls="--", lw=1.15)
        ax.set_title(f"{EXPOSURE_LABELS[exposure]}：带宽敏感性", loc="left", fontweight="bold")
        ax.set_xlabel("正侵蚀值 log10(Aa)")
        ax.set_ylabel("核密度")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(axis="y", lw=.55)
    manifest.append(_save(fig, figure_root, "D02_kde_bandwidth_sensitivity", pd.DataFrame(kde_source),
                          "对 Scott 带宽分别使用 0.5、1、1.5、2 倍平滑强度。虚线只标出满足预设峰谷规则的候选分界；未出现可靠谷值时不强行画阈值。",
                          scientific_question="低值分界是否对平滑参数稳定？", experiment="KDE 带宽敏感性", figure_type="密度曲线图"))

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.7), constrained_layout=True)
    gmm_source = []
    for i, exposure in enumerate(EXPOSURES):
        ax = axes[i]
        g = gmm_models.loc[gmm_models["exposure"].eq(exposure)].copy()
        if not g.empty:
            best = float(g["bic"].min())
            for metric, marker, color in (("aic", "o", COLORS["blue"]), ("bic", "D", COLORS["ochre"])):
                ax.plot(g["n_components"], g[metric] - g[metric].min(), marker=marker, lw=1.8,
                        color=color, label=f"Δ{metric.upper()}（相对该指标最小值）")
            ax.set_xticks(g["n_components"])
            ax.set_xlabel("高斯混合成分数 K")
            ax.set_ylabel("相对信息准则（越低越好）")
            ax.legend(fontsize=8)
            gmm_source.extend(g.to_dict("records"))
        ax.set_title(f"{EXPOSURE_LABELS[exposure]}：混合模型比较", loc="left", fontweight="bold")
        ax.grid(axis="y", lw=.55)
    manifest.append(_save(fig, figure_root, "D03_gaussian_mixture_diagnostics", pd.DataFrame(gmm_source),
                          "高斯混合模型在 log10 正值尺度比较 K=1–4。BIC 较低并不自动代表存在自然低值分界；分量重叠和后验不确定性须结合判断。",
                          scientific_question="正侵蚀值是否呈现可区分的多个群体？", experiment="高斯混合诊断", figure_type="模型诊断图"))

    candidates = pd.concat([kde_candidates, cdf_candidates, otsu_candidates], ignore_index=True, sort=False)
    candidates = candidates.loc[candidates["threshold_original"].notna()].copy()
    gmm_boundary_records = []
    for rec in gmm_models.loc[gmm_models["n_components"].gt(1)].to_dict("records"):
        means = str(rec.get("component_means_log10", "")).split("|")
        bounds = str(rec.get("adjacent_boundaries_log10", "")).split("|")
        overlaps = str(rec.get("adjacent_overlap_index", "")).split("|")
        for j, bound in enumerate(bounds):
            try:
                log_bound = float(bound)
                overlap = float(overlaps[j])
            except (ValueError, IndexError):
                continue
            if not np.isfinite(log_bound):
                continue
            gmm_boundary_records.append({"exposure": rec["exposure"], "threshold_original": 10 ** log_bound,
                                         "threshold_log10": log_bound, "candidate_source": "GMM boundary",
                                         "status": "comparison_only_overlap" if overlap > .35 else "valid_gmm_boundary",
                                         "overlap": overlap})
    candidate_plot = pd.concat([candidates, pd.DataFrame(gmm_boundary_records)], ignore_index=True, sort=False)
    if not candidate_plot.empty:
        candidate_plot["method_label"] = candidate_plot["candidate_source"].map({
            "kde": "KDE（满足峰谷判据）", "empirical_cdf": "经验 CDF 断点",
            "otsu_comparison": "Otsu（敏感性参考）", "gmm": "GMM（边界参考）",
            "valid_gmm_boundary": "GMM（低重叠边界）", "GMM boundary": "GMM（边界参考）",
        }).fillna(candidate_plot["candidate_source"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    threshold_source = []
    method_order = ["KDE（满足峰谷判据）", "经验 CDF 断点", "GMM（低重叠边界）", "GMM（边界参考）", "Otsu（敏感性参考）"]
    for ax, exposure in zip(axes, EXPOSURES, strict=True):
        part = candidate_plot.loc[candidate_plot["exposure"].eq(exposure)].copy()
        if not part.empty:
            part["method_label"] = part["method_label"].fillna(part["candidate_source"])
            ymap = {v: i for i, v in enumerate(method_order)}
            part["y"] = part["method_label"].map(ymap).fillna(len(ymap))
            colors = [COLORS["blue"] if "KDE" in s else COLORS["lavender"] if "CDF" in s else COLORS["ochre"] for s in part["method_label"].astype(str)]
            ax.scatter(part["threshold_original"], part["y"], s=45, c=colors, edgecolors="white", linewidths=.7, zorder=3)
            for rec in part.to_dict("records"):
                threshold_source.append(rec)
        ax.set_xscale("log")
        ax.set_yticks(range(len(method_order)), method_order)
        ax.invert_yaxis()
        ax.set_xlabel("候选阈值 Aa（原始单位，对数轴）")
        ax.set_title(EXPOSURE_LABELS[exposure], loc="left", fontweight="bold")
        ax.grid(axis="x", lw=.55)
    manifest.append(_save(fig, figure_root, "D04_threshold_candidate_comparison", pd.DataFrame(threshold_source),
                          "比较所有通过判据的分布候选、CDF 断点、GMM 边界与 Otsu 参考值。Otsu 与高重叠 GMM 只用于敏感性对照，不作为主筛选阈值。",
                          scientific_question="不同分布方法给出的低值界点是否一致？", experiment="多方法阈值对照", figure_type="阈值比较图"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    bootstrap_source = bootstrap.copy()
    for ax, exposure in zip(axes, EXPOSURES, strict=True):
        part = bootstrap_source.loc[bootstrap_source["exposure"].eq(exposure)]
        methods = ["KDE", "empirical_CDF"]
        labels = ["KDE 峰谷", "经验 CDF 折点"]
        rates = []
        for method in methods:
            method_rows = part.loc[part.get("method", pd.Series(index=part.index, dtype=str)).eq(method)]
            valid_status = "valid_valley" if method == "KDE" else "valid_cdf_breakpoint"
            rates.append(float(method_rows["status"].eq(valid_status).mean()) if len(method_rows) else 0.0)
        ax.bar(labels, rates, color=[COLORS["blue"], COLORS["ochre"]], width=.58)
        for j, value in enumerate(rates):
            ax.text(j, value + .025, f"稳定识别 {value:.0%}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("100 次空间块重抽样中的识别比例")
        ax.set_title(EXPOSURE_LABELS[exposure], loc="left", fontweight="bold")
        ax.grid(axis="y", lw=.55)
    manifest.append(_save(fig, figure_root, "D05_spatial_bootstrap_stability", bootstrap_source,
                          "以 5 km 网格为空间块进行重抽样，检查 Scott KDE 是否重复识别出峰谷。若多数重抽样未识别峰谷，说明该正值分界缺少空间稳定性。",
                          scientific_question="候选低值界点在空间重抽样下是否稳定？", experiment="空间块自助法", figure_type="稳健性图"))

    threshold_lookup = {}
    for exposure in EXPOSURES:
        selected = threshold_candidates.loc[threshold_candidates["exposure"].eq(exposure)
                                            & threshold_candidates["is_primary"].fillna(False)] if not threshold_candidates.empty else pd.DataFrame()
        threshold_lookup[exposure] = float(selected.iloc[0]["threshold_original"]) if not selected.empty else None
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 7.3), constrained_layout=True)
    map_rows = []
    cmap = ListedColormap(REGIME_COLORS)
    cmap.set_bad((1, 1, 1, 0))
    norm = BoundaryNorm([-.5, .5, 1.5, 2.5, 3.5], cmap.N)
    for ax, exposure in zip(axes, EXPOSURES, strict=True):
        assignments = gmm_assignments.get(exposure, pd.DataFrame())
        uncertain_ids = set(assignments.loc[assignments["classification_status"].eq("uncertain"), "cell_id"].dropna().astype(int)) if not assignments.empty else set()
        grid, extent, valid = regime_grid(cells, exposure=exposure, threshold=threshold_lookup[exposure],
                                         uncertain_cell_ids=uncertain_ids)
        masked = np.ma.masked_where(grid < 0, grid)
        ax.imshow(masked, origin="upper", extent=extent, interpolation="nearest", cmap=cmap, norm=norm,
                  aspect="equal", rasterized=True)
        xmin, xmax, ymin, ymax = extent
        x_ticks = np.arange(np.ceil(xmin / 5000) * 5000, xmax, 5000)
        y_ticks = np.arange(np.ceil(ymin / 5000) * 5000, ymax, 5000)
        for x in x_ticks:
            ax.axvline(x, color="white", alpha=.55, lw=.55, zorder=2)
        for y in y_ticks:
            ax.axhline(y, color="white", alpha=.55, lw=.55, zorder=2)
        ax.annotate("N", xy=(.93, .94), xytext=(.93, .82), xycoords="axes fraction", textcoords="axes fraction",
                    ha="center", va="center", arrowprops={"arrowstyle": "-|>", "color": COLORS["ink"], "lw": 1.2},
                    fontsize=9, fontweight="bold")
        # 5 km scale bar in projected coordinate units.
        bar_x, bar_y = xmin + .07 * (xmax - xmin), ymin + .06 * (ymax - ymin)
        ax.plot([bar_x, bar_x + 5000], [bar_y, bar_y], color=COLORS["ink"], lw=2.2, solid_capstyle="butt")
        ax.plot([bar_x, bar_x], [bar_y - 200, bar_y + 200], color=COLORS["ink"], lw=1)
        ax.plot([bar_x + 5000, bar_x + 5000], [bar_y - 200, bar_y + 200], color=COLORS["ink"], lw=1)
        ax.text(bar_x + 2500, bar_y + 450, "5 km", ha="center", va="bottom", fontsize=8)
        ax.set_title(EXPOSURE_LABELS[exposure], loc="left", fontweight="bold", pad=9)
        ax.set_xlabel("东向坐标（m）")
        ax.set_ylabel("北向坐标（m）")
        ax.ticklabel_format(style="plain", axis="both", useOffset=False)
        ax.tick_params(axis="both", labelsize=7)
        counts = pd.Series(grid[grid >= 0].ravel()).value_counts().to_dict()
        for code, label in enumerate(REGIME_LABELS):
            map_rows.append({"exposure": exposure, "regime": label, "cell_count": int(counts.get(code, 0)),
                             "valid_cell_count": int(valid.sum()), "threshold_original": threshold_lookup[exposure],
                             "row_count": int(grid.shape[0]), "col_count": int(grid.shape[1]),
                             "x_min": xmin, "x_max": xmax, "y_min": ymin, "y_max": ymax})
    legend = [Line2D([0], [0], marker="s", color="none", markerfacecolor=color, markeredgecolor="white",
                     markersize=9, label=label) for color, label in zip(REGIME_COLORS, REGIME_LABELS, strict=True)]
    fig.legend(handles=legend, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(.5, -.02))
    manifest.append(_save(fig, figure_root, "S01_low_erosion_regime_maps", pd.DataFrame(map_rows),
                          "地图基于有效耕地掩膜和原始 30 m 行列网格。零值独立显示；只有存在有效分布阈值时才显示低值正侵蚀类别。正值分界未通过时，地图不会把正值人为拆成低值和主体。",
                          scientific_question="零值与可识别低值侵蚀在空间上如何分布？", experiment="有效阈值空间映射", figure_type="空间分类图"))

    if not profile_summaries.empty:
        source = profile_summaries.copy()
        variable_labels = {"SoilType": "土壤类型", "Elevation": "高程", "TWI": "地形湿度指数",
                           "LST": "地表温度", "NDWI": "水体指数"}
        source["group_label"] = source["group_variable"].map(variable_labels).fillna(source["group_variable"])
        source["profile"] = [
            _soil_profile_label(variable, profile)
            for variable, profile in zip(source["group_variable"], source["profile"], strict=True)
        ]
        metric = "low_positive_fraction"
        title = "低值正侵蚀在土壤类型与环境分位组中的占比"
        colorbar = "低值正侵蚀占有效像元比例"
        valid_by_period = source.groupby("exposure")["low_positive_fraction"].apply(lambda s: s.notna().any()).reindex(EXPOSURES).fillna(False)
        if not valid_by_period.all():
            metric = "zero_fraction"
            title = "零侵蚀像元在土壤类型与环境分位组中的占比"
            colorbar = "零侵蚀占有效像元比例"
        data = source.pivot_table(index=["group_label", "profile"], columns="exposure", values=metric, aggfunc="first")
        data = data.reindex(columns=list(EXPOSURES))
        if not data.empty:
            fig, ax = plt.subplots(figsize=(9.8, max(5.2, .27 * len(data))), constrained_layout=True)
            upper = float(data.max().max()) if np.isfinite(data.to_numpy(float)).any() else 1.0
            sns.heatmap(data, cmap=sns.color_palette("crest", as_cmap=True), vmin=0, vmax=max(.05, upper),
                        annot=True, fmt=".1%", linewidths=.55, linecolor="white", cbar_kws={"label": colorbar}, ax=ax)
            ax.set_xlabel("侵蚀时期")
            ax.set_ylabel("环境画像（变量 / 分组）")
            ax.set_xticklabels(["20 年平均", "2020 年"], rotation=0)
            ax.set_title(title, loc="left", fontweight="bold", pad=12)
            manifest.append(_save(fig, figure_root, "S02_low_regime_by_environment_profile", source,
                                  "仅在有效分布阈值存在时计算阈值以下正值比例；每个环境分位组按该组完整有效像元作分母。Otsu 参考值不用于本图的主分组。",
                                  scientific_question="低值侵蚀状态是否集中于特定土类或环境背景？", experiment="环境分层画像", figure_type="空间环境分层图"))

    return manifest


def create_component_figure(figure_root: Path, components: pd.DataFrame) -> dict[str, str] | None:
    if components.empty:
        return None
    _setup()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for ax, exposure in zip(axes, EXPOSURES, strict=True):
        part = components.loc[components["exposure"].eq(exposure)].sort_values("pixel_count", ascending=False).copy()
        part = part.head(20)
        if part.empty:
            ax.text(.5, .5, "无有效低值分界", ha="center", va="center", transform=ax.transAxes, color="#73838C")
            ax.set_axis_off()
            continue
        part["rank"] = np.arange(1, len(part) + 1)
        ax.bar(part["rank"], part["area_ha"], color=COLORS["blue"], width=.8)
        ax.set_yscale("log")
        ax.set_xlabel("低值斑块面积排名")
        ax.set_ylabel("连通斑块面积（公顷，对数轴）")
        ax.set_title(EXPOSURE_LABELS[exposure], loc="left", fontweight="bold")
        ax.grid(axis="y", lw=.55)
    return _save(fig, figure_root, "S03_low_regime_connected_components", components,
                 "低值正侵蚀斑块在原始 30 m 格网上按八邻域连通定义，面积按每像元 900 m² 计算。图展示面积最大的前 20 个斑块。",
                 scientific_question="低值侵蚀斑块是零散分布还是空间聚集？", experiment="连通斑块分析", figure_type="斑块结构图")


def create_block_composition_figure(
    figure_root: Path, cells: pd.DataFrame, thresholds: pd.DataFrame,
    gmm_assignments: dict[str, pd.DataFrame],
) -> dict[str, str]:
    """Map zero, candidate low-positive and GMM uncertainty shares by 5 km block."""
    _setup()
    rows = []
    grouped_by_exposure = {}
    for exposure in EXPOSURES:
        values = pd.to_numeric(cells[exposure], errors="coerce")
        valid = cells["valid_cell"].fillna(False).astype(bool) & values.notna() & np.isfinite(values)
        primary = thresholds.loc[thresholds["exposure"].eq(exposure)
                                 & thresholds["is_primary"].fillna(False)] if not thresholds.empty else pd.DataFrame()
        threshold = float(primary.iloc[0]["threshold_original"]) if not primary.empty else None
        uncertain_ids = set()
        assignment = gmm_assignments.get(exposure, pd.DataFrame())
        if not assignment.empty:
            uncertain_ids = set(assignment.loc[assignment["classification_status"].eq("uncertain"), "cell_id"].dropna().astype(int))
        frame = cells.loc[valid, ["cell_id", "x", "y"]].copy()
        frame["value"] = values.loc[valid].to_numpy(float)
        frame["uncertain"] = frame["cell_id"].isin(uncertain_ids)
        frame["block_x"] = np.floor(pd.to_numeric(frame["x"], errors="coerce") / 5000).astype(int)
        frame["block_y"] = np.floor(pd.to_numeric(frame["y"], errors="coerce") / 5000).astype(int)
        frame["zero"] = frame["value"].eq(0).astype(int)
        frame["low_positive"] = (
            ((frame["value"] > 0) & (frame["value"] <= threshold)).astype(int)
            if threshold is not None else 0
        )
        frame["uncertain_share_n"] = frame["uncertain"].astype(int)
        blocks = frame.groupby(["block_x", "block_y"], as_index=False).agg(
            valid_n=("cell_id", "size"), zero_n=("zero", "sum"),
            low_positive_n=("low_positive", "sum"), uncertain_n=("uncertain_share_n", "sum"),
        )
        blocks["zero_fraction"] = blocks["zero_n"] / blocks["valid_n"]
        blocks["low_positive_fraction"] = blocks["low_positive_n"] / blocks["valid_n"] if threshold is not None else np.nan
        blocks["uncertain_fraction"] = blocks["uncertain_n"] / blocks["valid_n"]
        blocks["center_x"] = (blocks["block_x"] + .5) * 5000
        blocks["center_y"] = (blocks["block_y"] + .5) * 5000
        blocks["exposure"] = exposure
        blocks["threshold_original"] = threshold
        grouped_by_exposure[exposure] = blocks
        rows.extend(blocks.to_dict("records"))
    source = pd.DataFrame(rows)
    metrics = [
        ("zero_fraction", "零侵蚀像元占比", "Blues"),
        ("low_positive_fraction", "低值正侵蚀占比", "crest"),
        ("uncertain_fraction", "GMM 分类不确定占比", "magma"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.4), constrained_layout=True)
    metric_maxima = {}
    for metric, _, _ in metrics:
        maxima = [pd.to_numeric(blocks[metric], errors="coerce").max()
                  for blocks in grouped_by_exposure.values() if blocks[metric].notna().any()]
        metric_maxima[metric] = max(.05, float(np.nanmax(maxima))) if maxima else .05
    colorbar_mappables: list[PatchCollection | None] = [None, None, None]
    for r, exposure in enumerate(EXPOSURES):
        blocks = grouped_by_exposure[exposure]
        for c, (metric, label, cmap_name) in enumerate(metrics):
            ax = axes[r, c]
            values = pd.to_numeric(blocks[metric], errors="coerce")
            if values.notna().any():
                patches = [Rectangle((int(item.block_x) * 5000, int(item.block_y) * 5000), 5000, 5000)
                           for item in blocks.itertuples()]
                collection = PatchCollection(patches, cmap=cmap_name, edgecolor="white", linewidth=.65,
                                             alpha=.92, zorder=2)
                collection.set_array(values.to_numpy(float))
                collection.set_clim(0, metric_maxima[metric])
                ax.add_collection(collection)
                ax.autoscale_view()
                if colorbar_mappables[c] is None:
                    colorbar_mappables[c] = collection
                ax.ticklabel_format(style="plain", axis="both", useOffset=False)
                ax.tick_params(axis="both", labelsize=7)
                ax.set_xlabel("东向坐标（m）")
                if c == 0:
                    ax.set_ylabel("北向坐标（m）")
                else:
                    ax.set_ylabel("")
                    ax.tick_params(axis="y", labelleft=False)
            else:
                ax.text(.5, .5, "当前时期未识别出有效分布阈值", ha="center", va="center",
                        transform=ax.transAxes, color="#6D7F89", fontsize=8)
                ax.set_axis_off()
            ax.set_title(f"{EXPOSURE_LABELS[exposure]} · {label}", loc="left", fontsize=9, fontweight="bold")
    for c, ((_, label, _), mappable) in enumerate(zip(metrics, colorbar_mappables, strict=True)):
        if mappable is not None:
            colorbar = fig.colorbar(mappable, ax=axes[:, c].tolist(), fraction=.045, pad=.025)
            colorbar.set_label(label, fontsize=8)
    fig.suptitle("零值、低值正侵蚀与分类不确定性的空间块组成", x=.06, ha="left", fontsize=13, fontweight="bold")
    return _save(fig, figure_root, "S04_spatial_block_composition_maps", source,
                 "每格为空间块级汇总，格内统计实际有效耕地像元占比；同一指标的两个时期使用相同色阶，三个指标各自设定色阶。零值比例和 GMM 后验不确定比例在两个时期均可绘制；只有通过空间稳定性检查的分布阈值才显示低值正侵蚀比例。空间块图用于识别聚集，不替代 30 m 像元地图。",
                 scientific_question="侵蚀零值、低值与不确定分类是否呈现空间聚集？",
                 experiment="5 km 空间块组成", figure_type="空间分块图")


def _response_scheme_label(value: object) -> str:
    text = str(value)
    if text == "S0_all":
        return "全部配对"
    if text in {"S1_both_positive", "positive_only"}:
        return "正值群体"
    if "fixed_original" in text:
        return "阈值后固定原组重配"
    if "refit_k6_filtered_population" in text:
        return "阈值后重拟合分组"
    if text.startswith("T"):
        return "阈值筛选原配对"
    return text


def create_response_figures(
    figure_root: Path, cells: pd.DataFrame, associations: pd.DataFrame, model_trials: pd.DataFrame,
    segmented_models: pd.DataFrame, pixel_spatial: pd.DataFrame, pair_spatial: pd.DataFrame,
    pair_quality: pd.DataFrame, *, seed: int = 20261029, sample_size: int = 20000,
) -> list[dict[str, str]]:
    """Create all-outcome response, matching quality and held-out test figures."""
    _setup()
    manifest = []

    # Flexible exposure-response curves: binned medians and spline descriptions.
    for exposure in EXPOSURES:
        fig, axes = plt.subplots(3, 3, figsize=(14.2, 11.1), constrained_layout=False)
        fig.subplots_adjust(left=.075, right=.99, bottom=.145, top=.97, wspace=.22, hspace=.34)
        source_rows = []
        data = cells.loc[cells["valid_cell"].fillna(False).astype(bool)].copy()
        x_all = pd.to_numeric(data[exposure], errors="coerce").to_numpy(float)
        finite_x = np.isfinite(x_all)
        positive_n = int(np.count_nonzero(finite_x & (x_all > 0)))
        zero_n = int(np.count_nonzero(finite_x & (x_all == 0)))
        for ax, outcome in zip(axes.flat, OUTCOMES, strict=True):
            y_all = pd.to_numeric(data[outcome], errors="coerce").to_numpy(float)
            keep = finite_x & np.isfinite(y_all)
            x, y = x_all[keep], y_all[keep]
            rng = np.random.default_rng(seed)
            if len(x) > sample_size:
                selected = np.sort(rng.choice(len(x), sample_size, replace=False))
                x, y = x[selected], y[selected]
            if len(x) < 20 or np.ptp(x) == 0 or np.ptp(y) == 0:
                ax.text(.5, .5, "有效变化不足", ha="center", va="center", transform=ax.transAxes, color="#6D7F89")
                ax.set_axis_off()
                continue
            scale = np.subtract(*np.quantile(y, [.75, .25]))
            scale = float(scale) if np.isfinite(scale) and scale > 0 else float(np.std(y))
            y_std = (y - np.median(y)) / scale if scale > 0 else y - np.mean(y)
            xx = np.log1p(x)
            ax.hexbin(xx, y_std, gridsize=34, mincnt=1, cmap="Blues", bins="log", linewidths=0, alpha=.78)
            bins = pd.qcut(pd.Series(xx), q=min(24, max(5, len(xx) // 100)), duplicates="drop")
            grouped = pd.DataFrame({"x": xx, "y": y_std, "bin": bins}).groupby("bin", observed=True).agg(
                x_median=("x", "median"), y_median=("y", "median"),
                y_q25=("y", lambda v: v.quantile(.25)), y_q75=("y", lambda v: v.quantile(.75)),
                n=("y", "size"),
            ).reset_index(drop=True)
            if len(grouped):
                yerr = np.vstack([grouped["y_median"] - grouped["y_q25"], grouped["y_q75"] - grouped["y_median"]])
                ax.errorbar(grouped["x_median"], grouped["y_median"], yerr=yerr, fmt="o", ms=3.2,
                            color=COLORS["ochre"], ecolor="#B99A62", elinewidth=.7, capsize=1.5,
                            label="分位箱中位数与四分位范围", zorder=4)
            predictor = xx.reshape(-1, 1)
            spline = make_pipeline(SplineTransformer(n_knots=5, degree=3, include_bias=False), LinearRegression())
            spline.fit(predictor, y_std)
            grid = np.linspace(np.quantile(xx, .005), np.quantile(xx, .995), 150)
            pred = spline.predict(grid.reshape(-1, 1))
            ax.plot(grid, pred, color=COLORS["coral"], lw=1.9, label="自然样条描述曲线", zorder=5)
            ax.axhline(0, color="#8B999F", lw=.7, ls=(0, (3, 3)))
            rho = float(spearmanr(x, y).statistic)
            ax.text(.98, .96, f"n={len(x):,}  ρ={rho:.2f}", ha="right", va="top", transform=ax.transAxes,
                    fontsize=7.5, color="#52636E")
            ax.set_title(OUTCOME_LABELS[outcome], loc="left", fontsize=9, fontweight="bold")
            ax.set_xlabel("log1p(实际侵蚀 Aa)")
            ax.set_ylabel("健康指标稳健标准化值")
            ax.grid(axis="y", lw=.5)
            for xv, yv, lo, hi, count in zip(grouped["x_median"], grouped["y_median"],
                                               grouped["y_q25"], grouped["y_q75"], grouped["n"], strict=True):
                source_rows.append({"exposure": exposure, "outcome": outcome, "curve_type": "binned_median",
                                    "x_log1p_Aa": xv, "y_standardized": yv, "y_q25": lo, "y_q75": hi,
                                    "bin_n": int(count), "n_zero_population": zero_n,
                                    "n_positive_population": positive_n})
            source_rows.extend({"exposure": exposure, "outcome": outcome, "curve_type": "natural_spline",
                                "x_log1p_Aa": float(xv), "y_standardized": float(yv),
                                "bin_n": len(x), "n_zero_population": zero_n,
                                "n_positive_population": positive_n}
                               for xv, yv in zip(grid, pred, strict=True))
        handles = [Line2D([0], [0], marker="o", color=COLORS["ochre"], lw=0, label="分位箱中位数与四分位范围"),
                   Line2D([0], [0], color=COLORS["coral"], lw=2, label="自然样条描述曲线")]
        fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(.5, .018))
        manifest.append(_save(fig, figure_root, f"R01_nonlinear_response_curves_{exposure}", pd.DataFrame(source_rows),
                              "九项健康指标使用同一固定抽样。曲线为自然样条描述关系，分位箱显示中位数和四分位范围；标准化仅用于图内尺度比较，不代表独立实测效应或因果效应。",
                              scientific_question="环境匹配后，侵蚀对各健康指标的响应形态如何？",
                              experiment="九指标非线性响应", figure_type="非线性响应图"))

    # Rank association heatmap, explicitly separating pixels from paired units.
    if not associations.empty:
        assoc = associations.loc[associations["status"].eq("ok")].copy()
        assoc["panel"] = np.where(assoc["analysis_unit"].eq("pixel"),
                                  assoc["exposure"].map({"Aa_20yr": "20 年像元", "Aa_2020": "2020 年像元"}),
                                  assoc["exposure"].map({"Aa_20yr": "20 年配对", "Aa_2020": "2020 年配对"}) + " · " + assoc["method"].fillna("重配对"))
        assoc = assoc.loc[assoc["scheme"].isin(["S0_all", "S1_both_positive", "positive_only"])]
        summary = assoc.groupby(["panel", "outcome"], as_index=False).agg(
            median_spearman_rho=("spearman_rho", "median"), n_median=("n", "median"))
        matrix = summary.pivot(index="outcome", columns="panel", values="median_spearman_rho").reindex(OUTCOMES)
        matrix.index = [OUTCOME_LABELS.get(value, value) for value in matrix.index]
        fig, ax = plt.subplots(figsize=(13.4, 7.0), constrained_layout=True)
        sns.heatmap(matrix, ax=ax, cmap="vlag", center=0, vmin=-1, vmax=1, annot=True, fmt=".2f",
                    linewidths=.6, linecolor="white", cbar_kws={"label": "斯皮尔曼秩相关系数 ρ"})
        ax.set_xlabel("侵蚀时期与分析单位")
        ax.set_ylabel("")
        ax.set_title("侵蚀—健康指标秩关联总览", loc="left", fontweight="bold", pad=12)
        ax.tick_params(axis="x", labelrotation=20)
        manifest.append(_save(fig, figure_root, "R02_rank_association_heatmap", summary,
                              "颜色表示 Spearman 秩相关系数。像元分析与配对分析使用不同统计单位；每个格子汇总指定基线与正值群体方案，不把相关方向解释成因果效应。",
                              scientific_question="侵蚀关联在九项健康指标之间是否一致？", experiment="非参数关联总览", figure_type="相关热图"))

    # Pair slopes with spatial-block bootstrap intervals.
    pair_assoc = associations.loc[associations["analysis_unit"].eq("pair")].copy() if not associations.empty else pd.DataFrame()
    if not pair_assoc.empty:
        pair_assoc["scheme_label"] = pair_assoc["scheme"].map(_response_scheme_label)
        allowed = ["全部配对", "正值群体", "阈值筛选原配对", "阈值后固定原组重配", "阈值后重拟合分组"]
        pair_assoc = pair_assoc.loc[pair_assoc["scheme_label"].isin(allowed) & pair_assoc["slope"].notna()].copy()
        grouped = pair_assoc.groupby(["exposure", "outcome", "scheme_label"], as_index=False).agg(
            slope=("slope", "median"), ci_low=("slope_ci_low", "min"), ci_high=("slope_ci_high", "max"),
            n=("n", "sum"))
        fig, axes = plt.subplots(1, 2, figsize=(13, 9.8), sharey=True, constrained_layout=False)
        fig.subplots_adjust(left=.12, right=.985, bottom=.17, top=.97, wspace=.12)
        colors = {name: color for name, color in zip(allowed, (COLORS["navy"], COLORS["teal"], COLORS["ochre"], COLORS["blue"], COLORS["lavender"]), strict=True)}
        for ax, exposure in zip(axes, EXPOSURES, strict=True):
            part = grouped.loc[grouped["exposure"].eq(exposure)]
            for yi, outcome in enumerate(OUTCOMES):
                outcome_part = part.loc[part["outcome"].eq(outcome)]
                for j, scheme in enumerate(allowed):
                    row = outcome_part.loc[outcome_part["scheme_label"].eq(scheme)]
                    if row.empty:
                        continue
                    item = row.iloc[0]
                    y_pos = yi + (j - (len(allowed)-1)/2) * .105
                    lo, hi = item["ci_low"], item["ci_high"]
                    if not np.isfinite([lo, hi]).all():
                        lo = hi = item["slope"]
                    ax.plot([lo, hi], [y_pos, y_pos], color=colors[scheme], lw=1.35, alpha=.9)
                    ax.scatter(item["slope"], y_pos, color=colors[scheme], s=25, edgecolor="white", linewidth=.6, zorder=3)
                    source_rows.append({"exposure": exposure, "outcome": outcome, "scheme": scheme,
                                        "slope": item["slope"], "ci_low": lo, "ci_high": hi, "pair_n_sum": item["n"]})
            ax.axvline(0, color="#87969F", lw=.8, ls=(0, (3, 3)))
            ax.set_yticks(np.arange(len(OUTCOMES)), [OUTCOME_LABELS[o] for o in OUTCOMES])
            ax.invert_yaxis()
            ax.set_xlabel("Δ健康 / Δlog1p(Aa) 斜率")
            ax.set_title(EXPOSURE_LABELS[exposure], loc="left", fontweight="bold")
            ax.grid(axis="x", lw=.55)
        handles = [Line2D([0], [0], marker="o", color=color, lw=1.4, label=scheme)
                   for scheme, color in colors.items()]
        fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(.5, .02))
        manifest.append(_save(fig, figure_root, "R03_paired_slope_forest", pd.DataFrame(source_rows),
                              "点为配对差值回归斜率，线段为 5 km 空间块自助法 95% 区间。全配对、两端均正值、阈值筛选原配对及两类重配对分别展示；按侵蚀强弱方向定义的 Δ 健康保持一致。",
                              scientific_question="环境匹配后侵蚀差值与健康差值的方向和强度如何？", experiment="配对差值响应", figure_type="森林图"))

    # Apparent nonlinear family comparison across populations.
    pixel_trials = model_trials.loc[model_trials.get("analysis_unit", pd.Series(index=model_trials.index, dtype=str)).eq("pixel")].copy() if not model_trials.empty else pd.DataFrame()
    if not pixel_trials.empty:
        scheme_specs = [("S0_all", "全部有效像元"), ("positive_only", "全部正值像元")]
        threshold_present = pixel_trials["scheme"].eq("distribution_threshold").any()
        if threshold_present:
            scheme_specs.append(("distribution_threshold", "分布阈值筛选"))
        fig, axes = plt.subplots(2, len(scheme_specs), figsize=(4.4 * len(scheme_specs), 9.2), constrained_layout=True)
        source_rows = []
        for r, exposure in enumerate(EXPOSURES):
            for c, (scheme, scheme_label) in enumerate(scheme_specs):
                ax = axes[r, c] if len(scheme_specs) > 1 else axes[r]
                part = pixel_trials.loc[pixel_trials["exposure"].eq(exposure) & pixel_trials["scheme"].eq(scheme)]
                if part.empty:
                    ax.text(.5, .5, "无有效分布阈值", ha="center", va="center", transform=ax.transAxes, color="#6D7F89")
                    ax.set_axis_off()
                    continue
                matrix = part.pivot_table(index="outcome", columns="model_family", values="r_squared_apparent", aggfunc="median").reindex(OUTCOMES)
                matrix = matrix.rename(columns={"linear": "线性", "quadratic": "二次项",
                                                "natural_spline": "自然样条", "gradient_boosting": "梯度提升"})
                matrix.index = [OUTCOME_LABELS.get(value, value) for value in matrix.index]
                sns.heatmap(matrix, ax=ax, cmap="mako", vmin=0, vmax=max(.15, min(.9, float(np.nanmax(matrix.to_numpy())))),
                            annot=True, fmt=".2f", linewidths=.5, linecolor="white", cbar=(c == len(scheme_specs)-1),
                            cbar_kws={"label": "表观 R²"} if c == len(scheme_specs)-1 else None)
                ax.set_xlabel("模型形式")
                ax.set_ylabel("")
                ax.set_title(f"{EXPOSURE_LABELS[exposure]} · {scheme_label}", loc="left", fontsize=9, fontweight="bold")
                ax.tick_params(axis="x", rotation=30, labelsize=7)
                source_rows.extend(part.to_dict("records"))
        manifest.append(_save(fig, figure_root, "M01_nonlinear_model_family_comparison", pd.DataFrame(source_rows),
                              "固定抽样上比较线性、二次、多项式样条和梯度提升的表观 R²。模型按同一批样本比较；表观拟合仅用于 Oracle 探索，独立空间测试见后续图。",
                              scientific_question="过滤侵蚀低值后，健康响应是否更适合非线性模型？", experiment="模型形式比较", figure_type="模型比较热图"))

    # No-filter zero-indicator and segmented/spline models.
    if not segmented_models.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12.2, 6.4), constrained_layout=True)
        source_rows = []
        for ax, exposure in zip(axes, EXPOSURES, strict=True):
            part = segmented_models.loc[segmented_models["exposure"].eq(exposure)]
            matrix = part.pivot_table(index="outcome", columns="model", values="r_squared_apparent", aggfunc="median").reindex(OUTCOMES)
            matrix.index = [OUTCOME_LABELS.get(value, value) for value in matrix.index]
            if matrix.empty:
                ax.set_axis_off()
                continue
            sns.heatmap(matrix, ax=ax, cmap="mako", vmin=0, vmax=max(.12, min(.75, float(np.nanmax(matrix.to_numpy())))),
                        annot=True, fmt=".2f", linewidths=.55, linecolor="white", cbar=True,
                        cbar_kws={"label": "表观 R²"})
            ax.set_xlabel("响应函数")
            ax.set_ylabel("")
            ax.set_title(EXPOSURE_LABELS[exposure], loc="left", fontweight="bold")
            ax.tick_params(axis="x", rotation=25, labelsize=8)
            source_rows.extend(part.to_dict("records"))
        manifest.append(_save(fig, figure_root, "M02_zero_indicator_and_segmented_models", pd.DataFrame(source_rows),
                              "不筛除零值，对比 log1p(Aa) 线性项、零值指示加正值强度样条，以及存在合格阈值时的折线模型。结果为固定样本表观拟合，不能替代空间测试。",
                              scientific_question="零侵蚀是否需要与正值强度分开建模？", experiment="零值指示与分段响应", figure_type="分段模型图"))

    # Held-out whole-block spatial test distributions.
    spatial_parts = []
    for unit, frame in (("像元", pixel_spatial), ("配对", pair_spatial)):
        if frame.empty or "spatial_test_r_squared_selected" not in frame:
            continue
        part = frame.loc[frame["status"].eq("ok")].copy()
        if part.empty:
            continue
        part["analysis_unit_label"] = unit
        spatial_parts.append(part)
    if spatial_parts:
        spatial_data = pd.concat(spatial_parts, ignore_index=True, sort=False)
        spatial_data["outcome_label"] = spatial_data["outcome"].map(OUTCOME_LABELS)
        spatial_data["scheme_label"] = spatial_data["scheme"].map({
            "positive_only": "全部正值", "distribution_threshold": "分布阈值筛选",
            "S1_both_positive": "两端正值", "S0_all": "全部配对",
        }).fillna(spatial_data["scheme"])
        fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
        for row, unit in enumerate(("像元", "配对")):
            for col, exposure in enumerate(EXPOSURES):
                ax = axes[row, col]
                part = spatial_data.loc[spatial_data["analysis_unit_label"].eq(unit)
                                         & spatial_data["exposure"].eq(exposure)]
                if part.empty:
                    ax.text(.5, .5, "该空间测试群体样本不足", ha="center", va="center", transform=ax.transAxes, color="#6D7F89")
                    ax.set_axis_off()
                    continue
                scheme_values = list(spatial_data["scheme_label"].dropna().unique())
                scheme_colors = sns.color_palette("colorblind", n_colors=max(1, len(scheme_values)))
                palette = dict(zip(scheme_values, scheme_colors, strict=True))
                sns.boxplot(data=part, x="outcome_label", y="spatial_test_r_squared_selected", hue="scheme_label",
                            palette=palette, fliersize=1.5,
                            width=.72, ax=ax)
                ax.axhline(0, color="#87969F", lw=.7, ls=(0, (3, 3)))
                ax.set_xlabel("")
                ax.set_ylabel("留出空间块 R²")
                ax.tick_params(axis="x", rotation=28, labelsize=7)
                ax.set_title(f"{unit} · {EXPOSURE_LABELS[exposure]}", loc="left", fontweight="bold")
                ax.legend(title="分析群体", fontsize=7, title_fontsize=7, frameon=False)
        manifest.append(_save(fig, figure_root, "V01_spatial_block_test_r2", spatial_data,
                              "空间块按 5 km 整块分配；响应模型先在训练块拟合，再在验证块选模型，最终只在测试块报告 R²。训练样本已按 500 m 缓冲从验证/测试块边界剔除。",
                              scientific_question="表观关系在独立空间块中能否复现？", experiment="空间块外推验证", figure_type="空间验证统计图"))

    # Pair count and quality diagnostics in one inspectable table/figure.
    if not pair_quality.empty:
        quality = pair_quality.copy()
        quality["scheme_label"] = quality["scheme"].map(_response_scheme_label)
        quality["source_label"] = quality["source"].astype(str).str.replace("Phase F", "阶段 F", regex=False)
        quality["source_label"] = quality["source_label"].str.replace("fixed_original", "固定原组", regex=False)
        quality["source_label"] = quality["source_label"].str.replace("refit_k6_filtered_population", "重拟合分组", regex=False)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)
        for ax, exposure in zip(axes, EXPOSURES, strict=True):
            part = quality.loc[quality["exposure"].eq(exposure)].copy()
            if part.empty:
                ax.set_axis_off()
                continue
            sources = list(part["source_label"].dropna().unique())
            source_colors = sns.color_palette("colorblind", n_colors=max(1, len(sources)))
            source_palette = dict(zip(sources, source_colors, strict=True))
            sns.barplot(data=part, x="scheme_label", y="pair_count", hue="source_label", ax=ax,
                        palette=source_palette)
            ax.set_yscale("log")
            ax.set_xlabel("")
            ax.set_ylabel("配对数（对数轴）")
            ax.tick_params(axis="x", rotation=25, labelsize=7)
            ax.set_title(EXPOSURE_LABELS[exposure], loc="left", fontweight="bold")
            ax.legend(title="配对来源", fontsize=7, title_fontsize=7, frameon=False)
        manifest.append(_save(fig, figure_root, "Q01_pair_retention_counts", quality,
                              "柱高为匹配对数量（对数轴），比较原有 M3/M4 配对、端点均为正值以及阈值后重配对。筛选改变了目标群体和样本量，R²差异不解释为单纯模型改善。",
                              scientific_question="筛除低值侵蚀会保留多少配对样本？", experiment="配对筛选与重配对", figure_type="匹配样本量图"))

        fig, ax = plt.subplots(figsize=(9.5, 6.5), constrained_layout=True)
        plot = quality.loc[quality["pair_count"].gt(0) & quality["environmental_distance_median"].notna()
                           & quality["geographic_distance_median"].notna()].copy()
        if not plot.empty:
            sc = ax.scatter(plot["environmental_distance_median"], plot["geographic_distance_median"],
                            s=35 + 120 * np.log1p(plot["pair_count"]) / max(np.log1p(plot["pair_count"]).max(), 1),
                            c=plot["mean_absolute_environment_smd"].fillna(0), cmap="crest_r", edgecolors="white", linewidths=.7)
            fig.colorbar(sc, ax=ax, label="环境变量平均绝对 SMD")
            for item in plot.to_dict("records"):
                ax.annotate(f"{item['source']}\\n{item['scheme_label']}",
                            (item["environmental_distance_median"], item["geographic_distance_median"]),
                            xytext=(4, 4), textcoords="offset points", fontsize=6.5, color="#475963")
            source = plot.copy()
            ax.set_xlabel("环境距离中位数（标准化尺度）")
            ax.set_ylabel("地理距离中位数（m）")
            ax.set_title("环境接近程度、空间距离与平衡诊断", loc="left", fontweight="bold")
            ax.grid(lw=.55)
            manifest.append(_save(fig, figure_root, "Q02_matching_quality_bubble", source,
                                  "每个点代表一个配对群体；点大小表示配对数，颜色表示环境变量平均绝对标准化差异。图同时检查环境接近、地理距离与协变量平衡。",
                                  scientific_question="低值筛选和重配对是否改善环境可比性？", experiment="匹配质量", figure_type="匹配诊断图"))
        else:
            plt.close(fig)

    return manifest

