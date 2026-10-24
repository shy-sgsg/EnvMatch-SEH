"""Consistent, publication-oriented plots for project diagnostics and results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager


def _chinese_font_family() -> str:
    """Register an installed CJK font that Matplotlib's cache may not know."""
    system_fonts = font_manager.findSystemFonts()
    for token in ("notosanscjk", "ukai", "uming", "droidsansfallback", "wqy"):
        candidates = sorted(
            path for path in system_fonts if token in Path(path).name.lower()
        )
        for path in candidates:
            try:
                font_manager.fontManager.addfont(path)
                family = font_manager.FontProperties(fname=path).get_name()
            except (OSError, RuntimeError, ValueError):
                continue
            if family:
                return family
    return "DejaVu Sans"


_FONT = [_chinese_font_family(), "DejaVu Sans"]


def _finish(figure, output_path: str | Path) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return destination


def plot_environment_groups(
    frame: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str,
    output_path: str | Path,
    *,
    title: str = "环境组空间分布",
) -> Path:
    """Plot georeferenced environment-group labels for valid cells."""
    required = [x_col, y_col, group_col]
    if any(column not in frame for column in required):
        raise ValueError("frame is missing a coordinate or environment-group column")
    data = frame.loc[:, required].dropna()
    if data.empty:
        raise ValueError("No valid environment-group coordinates to plot")
    categories = pd.unique(data[group_col])
    codes = pd.Categorical(data[group_col], categories=categories).codes
    with plt.rc_context({"font.sans-serif": _FONT, "axes.unicode_minus": False}):
        figure, axis = plt.subplots(figsize=(7.2, 6.2))
        points = axis.scatter(data[x_col], data[y_col], c=codes, cmap="tab20", s=7, linewidths=0)
        colorbar = figure.colorbar(points, ax=axis, ticks=np.arange(len(categories)))
        colorbar.ax.set_yticklabels([str(value) for value in categories])
        colorbar.set_label("环境组")
        axis.set(title=title, xlabel="X 坐标（m）", ylabel="Y 坐标（m）")
        axis.set_aspect("equal", adjustable="datalim")
        return _finish(figure, output_path)


def plot_environment_profile(
    centers: pd.DataFrame,
    output_path: str | Path,
    *,
    value_columns: list[str] | None = None,
) -> Path:
    """Plot standardized cluster-center environment profiles as a heatmap."""
    if value_columns is None:
        value_columns = [column for column in centers if column.startswith("center_")]
    if not value_columns or not set(value_columns).issubset(centers.columns):
        raise ValueError("No environment center columns were supplied")
    values = centers[value_columns].to_numpy(dtype=float)
    group_labels = (
        centers["environment_group"].astype(str).tolist()
        if "environment_group" in centers
        else [str(index) for index in centers.index]
    )
    with plt.rc_context({"font.sans-serif": _FONT, "axes.unicode_minus": False}):
        figure, axis = plt.subplots(figsize=(max(6, len(value_columns) * 0.8), 5))
        image = axis.imshow(values, aspect="auto", cmap="RdBu_r")
        axis.set_xticks(np.arange(len(value_columns)), labels=value_columns, rotation=35, ha="right")
        axis.set_yticks(np.arange(len(group_labels)), labels=group_labels)
        axis.set(title="环境组特征", xlabel="环境变量", ylabel="环境组")
        figure.colorbar(image, ax=axis, label="标准化环境特征")
        return _finish(figure, output_path)


def plot_erosion_distribution(
    frame: pd.DataFrame,
    group_col: str,
    erosion_col: str,
    output_path: str | Path,
    *,
    log_scale: bool = True,
) -> Path:
    """Plot group-level erosion distributions, optionally on log1p scale."""
    data = frame[[group_col, erosion_col]].copy()
    data[erosion_col] = pd.to_numeric(data[erosion_col], errors="coerce")
    data = data.dropna()
    if (data[erosion_col] < 0).any():
        raise ValueError("erosion must be nonnegative")
    if log_scale:
        data[erosion_col] = np.log1p(data[erosion_col])
        ylabel = "log1p(土壤侵蚀模数)"
    else:
        ylabel = "土壤侵蚀模数"
    groups = list(pd.unique(data[group_col]))
    values = [data.loc[data[group_col] == group, erosion_col].to_numpy() for group in groups]
    with plt.rc_context({"font.sans-serif": _FONT, "axes.unicode_minus": False}):
        figure, axis = plt.subplots(figsize=(max(6, len(groups) * 0.65), 5))
        axis.boxplot(values, showfliers=False)
        axis.set_xticks(np.arange(1, len(groups) + 1), labels=[str(group) for group in groups])
        axis.set(title="环境组内侵蚀分布", xlabel="环境组", ylabel=ylabel)
        return _finish(figure, output_path)


def plot_matching_balance(balance: pd.DataFrame, output_path: str | Path) -> Path:
    """Plot covariate standardized mean differences before and after matching."""
    required = {"covariate", "unmatched_smd", "matched_smd"}
    if not required.issubset(balance.columns) or balance.empty:
        raise ValueError("balance table must contain covariate and before/after SMD columns")
    values = balance.set_index("covariate")[["unmatched_smd", "matched_smd"]]
    with plt.rc_context({"font.sans-serif": _FONT, "axes.unicode_minus": False}):
        figure, axis = plt.subplots(figsize=(7.5, max(3.5, len(values) * 0.38)))
        values.plot.barh(ax=axis, color=["#8a9ba8", "#3d7a72"], width=0.75)
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set(title="匹配前后环境变量平衡", xlabel="标准化均值差", ylabel="环境变量")
        axis.legend(["匹配前", "匹配后"], frameon=False)
        return _finish(figure, output_path)


def plot_group_regression(
    frame: pd.DataFrame,
    results: pd.DataFrame,
    outcome: str,
    erosion_col: str,
    group_col: str,
    output_path: str | Path,
) -> Path:
    """Plot group-level outcome versus log1p erosion with fitted slopes."""
    required = {outcome, erosion_col, group_col}
    if not required.issubset(frame.columns):
        raise ValueError("frame is missing regression columns")
    if not {"outcome", "environment_group", "coefficient", "status"}.issubset(results.columns):
        raise ValueError("results are missing fitted group coefficients")
    data = frame.loc[:, [outcome, erosion_col, group_col]].copy()
    data[outcome] = pd.to_numeric(data[outcome], errors="coerce")
    data[erosion_col] = pd.to_numeric(data[erosion_col], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    data["__exposure"] = np.log1p(data[erosion_col])
    selected_results = results.loc[(results["outcome"] == outcome) & (results["status"] == "ok")]
    with plt.rc_context({"font.sans-serif": _FONT, "axes.unicode_minus": False}):
        figure, axis = plt.subplots(figsize=(7.2, 5.5))
        for index, (group, subset) in enumerate(data.groupby(group_col, sort=False)):
            axis.scatter(
                subset["__exposure"], subset[outcome], s=18, alpha=0.62,
                color=plt.get_cmap("tab10")(index % 10), label=f"组 {group}",
            )
            row = selected_results.loc[selected_results["environment_group"] == group]
            if row.empty:
                continue
            slope = float(row.iloc[0]["coefficient"])
            x = subset["__exposure"].to_numpy(dtype=float)
            y = subset[outcome].to_numpy(dtype=float)
            line_x = np.linspace(x.min(), x.max(), 50)
            line_y = y.mean() + slope * (line_x - x.mean())
            axis.plot(line_x, line_y, color=plt.get_cmap("tab10")(index % 10))
        axis.set(title=f"{outcome} 与侵蚀暴露的环境组内关系", xlabel="log1p(土壤侵蚀模数)", ylabel=outcome)
        axis.legend(frameon=False)
        return _finish(figure, output_path)


def plot_pair_differences(pairs: pd.DataFrame, outcome: str, output_path: str | Path) -> Path:
    """Plot matched-pair health differences against log1p erosion differences."""
    required = ["erosion_high", "erosion_low", f"high_{outcome}", f"low_{outcome}"]
    if any(column not in pairs for column in required):
        raise ValueError(f"matched pairs require columns: {', '.join(required)}")
    delta_erosion = np.log1p(pd.to_numeric(pairs["erosion_high"], errors="coerce")) - np.log1p(
        pd.to_numeric(pairs["erosion_low"], errors="coerce")
    )
    delta_health = pd.to_numeric(pairs[f"high_{outcome}"], errors="coerce") - pd.to_numeric(
        pairs[f"low_{outcome}"], errors="coerce"
    )
    valid = np.isfinite(delta_erosion) & np.isfinite(delta_health)
    with plt.rc_context({"font.sans-serif": _FONT, "axes.unicode_minus": False}):
        figure, axis = plt.subplots(figsize=(6.5, 5.5))
        axis.scatter(delta_erosion[valid], delta_health[valid], color="#346b75", alpha=0.72, s=24)
        axis.axhline(0, color="gray", linewidth=0.8)
        axis.axvline(0, color="gray", linewidth=0.8)
        axis.set(title=f"匹配对 {outcome} 差值", xlabel="高低侵蚀差：Δlog1p(Aa)", ylabel=f"高低健康差：Δ{outcome}")
        return _finish(figure, output_path)


def plot_regression_forest(results: pd.DataFrame, output_path: str | Path) -> Path:
    """Plot coefficient estimates and confidence intervals for successful models."""
    required = {"outcome", "coefficient", "ci_lower", "ci_upper", "status"}
    if not required.issubset(results.columns):
        raise ValueError("regression results must include coefficient and confidence interval fields")
    data = results.loc[results["status"] == "ok"].copy().reset_index(drop=True)
    if data.empty:
        raise ValueError("No successful regression estimates to plot")
    labels = data["outcome"].astype(str)
    if "environment_group" in data:
        labels = labels + " | 组 " + data["environment_group"].astype(str)
    order = np.arange(len(data))
    estimate = data["coefficient"].to_numpy(dtype=float)
    lower = data["ci_lower"].to_numpy(dtype=float)
    upper = data["ci_upper"].to_numpy(dtype=float)
    with plt.rc_context({"font.sans-serif": _FONT, "axes.unicode_minus": False}):
        figure, axis = plt.subplots(figsize=(7.4, max(3.5, len(data) * 0.35)))
        axis.errorbar(estimate, order, xerr=[estimate - lower, upper - estimate], fmt="o", color="#2d6b69", capsize=3)
        axis.axvline(0, color="gray", linewidth=0.8)
        axis.set_yticks(order, labels=labels)
        axis.invert_yaxis()
        axis.set(title="侵蚀效应及置信区间", xlabel="回归系数", ylabel="指标与环境组")
        return _finish(figure, output_path)


def plot_indicator_response_heatmap(results: pd.DataFrame, output_path: str | Path) -> Path:
    """Plot standardized effects across health indicators and environment groups."""
    required = {"outcome", "standardized_coefficient"}
    if not required.issubset(results.columns):
        raise ValueError("results require outcome and standardized_coefficient columns")
    group_col = "environment_group" if "environment_group" in results else "method"
    if group_col not in results:
        raise ValueError("results require environment_group or method")
    data = results.loc[results.get("status", pd.Series("ok", index=results.index)) == "ok"]
    table = data.pivot_table(
        index="outcome", columns=group_col, values="standardized_coefficient", aggfunc="mean"
    )
    if table.empty:
        raise ValueError("No standardized effects to plot")
    with plt.rc_context({"font.sans-serif": _FONT, "axes.unicode_minus": False}):
        figure, axis = plt.subplots(figsize=(max(6, table.shape[1] * 0.8), max(4, table.shape[0] * 0.42)))
        image = axis.imshow(table.to_numpy(dtype=float), aspect="auto", cmap="RdBu_r")
        axis.set_xticks(np.arange(table.shape[1]), labels=[str(value) for value in table.columns], rotation=35, ha="right")
        axis.set_yticks(np.arange(table.shape[0]), labels=[str(value) for value in table.index])
        axis.set(title="健康指标标准化响应", xlabel=group_col, ylabel="健康指标")
        figure.colorbar(image, ax=axis, label="标准化回归系数")
        return _finish(figure, output_path)


def plot_pcnm_comparison(comparison: pd.DataFrame, output_path: str | Path) -> Path:
    """Compare environmental and PCNM coefficients on a one-to-one plot."""
    required = {"coefficient_pcnm", "coefficient_environmental"}
    if not required.issubset(comparison.columns):
        raise ValueError("comparison requires PCNM and environmental coefficient columns")
    x = pd.to_numeric(comparison["coefficient_pcnm"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(comparison["coefficient_environmental"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if not valid.any():
        raise ValueError("No paired PCNM and environmental coefficients to plot")
    low = float(min(x[valid].min(), y[valid].min()))
    high = float(max(x[valid].max(), y[valid].max()))
    with plt.rc_context({"font.sans-serif": _FONT, "axes.unicode_minus": False}):
        figure, axis = plt.subplots(figsize=(6, 6))
        axis.scatter(x[valid], y[valid], color="#6b777e", alpha=0.8, s=28)
        axis.plot([low, high], [low, high], linestyle="--", color="#a05b45", linewidth=1)
        axis.axhline(0, color="gray", linewidth=0.6)
        axis.axvline(0, color="gray", linewidth=0.6)
        axis.set(title="PCNM 与环境匹配效应比较", xlabel="PCNM 回归系数", ylabel="环境匹配回归系数")
        return _finish(figure, output_path)
