"""Close Phase I figure gaps using saved outputs and run standard MGWR."""

from __future__ import annotations

import argparse
import json
import os
import platform
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
import yaml
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUTCOME_ORDER = ["SHI_candidate", "AN", "OM", "CEC", "pH", "SW", "BD", "MBC", "RESP"]
HEALTH_INDICATORS = ["AN", "OM", "CEC", "pH", "SW", "BD", "MBC", "RESP"]
KEY_OUTCOMES = ["SHI_candidate", "OM", "SW"]
OUTCOME_LABELS = {"SHI_candidate": "SHI", "AN": "AN", "OM": "OM", "CEC": "CEC",
                  "pH": "pH", "SW": "SW", "BD": "BD", "MBC": "MBC", "RESP": "RESP"}
SOIL_CODES = [26, 27, 42]
SOIL_COLORS = {26: "#2878a5", 27: "#dc7658", 42: "#61a082"}
ENV_GROUPS = ["soil_type_group", "gradient_TWI", "gradient_NDVI"]
plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
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


def _save_figure(fig: plt.Figure, base: Path, dpi: int) -> list[str]:
    base.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in ("png", "svg", "pdf"):
        path = base.with_suffix(f".{extension}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(path.relative_to(ROOT).as_posix())
    plt.close(fig)
    return paths


def _run_dir(config: dict[str, Any]) -> Path:
    return ROOT / config["output_root"] / config["run_id"]


def _source(config: dict[str, Any], key: str) -> Path:
    return ROOT / config[key]


def _make_run(config: dict[str, Any], config_path: Path) -> Path:
    if config.get("mode") != "EXPLORATORY_ORACLE" or config.get("outcome_guided_selection") is not True:
        raise ValueError("Phase I closure requires explicit Oracle mode markers.")
    run_dir = _run_dir(config)
    if run_dir.exists() and (run_dir / "RUN_STATUS.md").exists():
        raise FileExistsError(f"Run directory already exists and will not be overwritten: {run_dir}")
    for name in ("configs", "tables", "figures", "reports", "manifests"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    (run_dir / "configs" / config_path.name).write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    return run_dir


def _curve_bins(pairs: pd.DataFrame, group_column: str, outcomes: list[str],
                group_order: list[str], minimum_pairs: int = 80) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for outcome in outcomes:
        ycol = f"delta_{outcome}"
        if ycol not in pairs:
            continue
        for group in group_order:
            subset = pairs.loc[pairs[group_column].eq(group), ["delta_log1p_Aa", ycol]].copy()
            values = subset.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            subset = subset.loc[np.isfinite(values).all(axis=1)].copy()
            if len(subset) < minimum_pairs:
                continue
            try:
                subset["_bin"] = pd.qcut(subset["delta_log1p_Aa"], q=8, labels=False, duplicates="drop")
            except ValueError:
                continue
            grouped = subset.groupby("_bin", observed=True).agg(
                mean_delta_log1p_Aa=("delta_log1p_Aa", "mean"),
                mean_delta_health=(ycol, "mean"), n_pairs=(ycol, "size"),
            ).reset_index()
            grouped = grouped.loc[grouped["n_pairs"].ge(10)]
            for row in grouped.itertuples(index=False):
                records.append({"outcome": outcome, "group": group,
                                "mean_delta_log1p_Aa": row.mean_delta_log1p_Aa,
                                "mean_delta_health": row.mean_delta_health,
                                "n_pairs": int(row.n_pairs)})
    return pd.DataFrame.from_records(records)


def _pair_group_labels(config: dict[str, Any]) -> tuple[pd.DataFrame, Path, Path]:
    pair_path = _source(config, "pair_run_id") / "pairs" / "candidate_pairs" / f"{config['pair_candidate']}.parquet"
    input_root = ROOT / "outputs" / config["input_run_id"] / "databases"
    env_path = input_root / "environment_erosion.parquet"
    pairs = pd.read_parquet(pair_path)
    env = pd.read_parquet(env_path, columns=["cell_id", "SoilType", "TWI", "NDVI"])
    env["SoilType"] = pd.to_numeric(env["SoilType"], errors="coerce")
    for variable in ("TWI", "NDVI"):
        env[variable] = pd.to_numeric(env[variable], errors="coerce")
        label_column = f"gradient_{variable}"
        env[label_column] = pd.Series(pd.NA, index=env.index, dtype="string")
        for soil in SOIL_CODES:
            mask = env["SoilType"].eq(soil) & np.isfinite(env[variable])
            if int(mask.sum()) < 5:
                continue
            q = pd.qcut(env.loc[mask, variable], q=5, labels=False, duplicates="drop")
            env.loc[mask, label_column] = q.astype("Int64").map(
                lambda value: f"{soil}:q{int(value) + 1}" if pd.notna(value) else pd.NA
            ).astype("string")
    index = env.set_index("cell_id")
    hi = pairs["high_erosion_cell_id"]
    lo = pairs["low_erosion_cell_id"]
    pairs["soil_high"] = index["SoilType"].reindex(hi).to_numpy()
    pairs["soil_low"] = index["SoilType"].reindex(lo).to_numpy()
    pairs["soil_group"] = pd.Series(pd.NA, index=pairs.index, dtype="string")
    same_soil = pairs["soil_high"].eq(pairs["soil_low"]) & pairs["soil_high"].isin(SOIL_CODES)
    pairs.loc[same_soil, "soil_group"] = pairs.loc[same_soil, "soil_high"].astype(int).map(lambda value: f"soil_{value}")
    for variable in ("TWI", "NDVI"):
        group_column = f"gradient_{variable}"
        high_group = index[group_column].reindex(hi).astype("string").to_numpy()
        low_group = index[group_column].reindex(lo).astype("string").to_numpy()
        pairs[group_column] = np.where((high_group == low_group) & (high_group != "<NA>"), high_group, None)
        pairs[f"{variable}_quintile"] = pd.Series(pairs[group_column], index=pairs.index).str.extract(r":q(\d+)$", expand=False)
    return pairs, pair_path, env_path


def _plot_i1_core(config: dict[str, Any], run_dir: Path, dpi: int) -> list[str]:
    source = _source(config, "i1_run_id")
    paths: list[str] = []
    trials = pd.read_csv(source / "results/model_search/global_model_search_trials.csv")
    trials = trials.loc[trials["exposure"].eq("Aa_20yr") & trials["status"].eq("ok")].copy()
    trials["validation_r2"] = pd.to_numeric(trials["validation_r2"], errors="coerce")
    trials["test_r2"] = pd.to_numeric(trials["test_r2"], errors="coerce")
    best = trials.sort_values("validation_r2", ascending=False).drop_duplicates(["outcome", "model_family"])
    families = list(dict.fromkeys(best["model_family"].tolist()))
    outcomes = [o for o in OUTCOME_ORDER if o in set(best["outcome"])]
    matrices = []
    for metric in ("validation_r2", "test_r2"):
        matrix = best.pivot(index="outcome", columns="model_family", values=metric).reindex(index=outcomes, columns=families)
        matrices.append(matrix)
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.2), constrained_layout=True, sharey=True)
    for ax, matrix, title in zip(axes, matrices, ["空间块验证集 R²（模型选择指标）", "空间块留出测试 R²（报告，不参与选择）"], strict=True):
        data = matrix.to_numpy(dtype=float)
        im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=-0.5, vmax=0.5)
        ax.set_xticks(range(len(families)), [name.replace("_", "\n") for name in families], rotation=0, fontsize=8)
        ax.set_yticks(range(len(outcomes)), [OUTCOME_LABELS[o] for o in outcomes])
        ax.set_title(title)
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                value = data[row, col]
                if np.isfinite(value):
                    ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if abs(value) > .30 else "#1d2933")
        fig.colorbar(im, ax=ax, shrink=.78, label="R²")
    fig.suptitle("CORE 04 · 全样本模型族表现：每个模型族保留验证集最佳候选", fontsize=14)
    fig.text(.5, -.01, "按验证集 R²选取每个结局×模型族的最佳候选；留出测试结果只作一次性报告。分段阈值候选已包含在模型搜索中。",
             ha="center", fontsize=8)
    paths.extend(_save_figure(fig, run_dir / "figures" / "core_figures/C04_global_best_model_r2", dpi))
    best.to_csv(run_dir / "tables/C04_global_best_model_r2_source.csv", index=False)

    curve = pd.read_csv(source / "results/nonlinear/response_curve_points.csv")
    selected = pd.read_csv(source / "leaderboards/global_response_model_leaderboard.csv")
    threshold_rows = selected.loc[
        selected["exposure"].eq("Aa_20yr") & selected["selected_model_family"].eq("segmented")
        & pd.to_numeric(selected["selected_threshold_original"], errors="coerce").notna()
    ]
    threshold = float(threshold_rows.loc[threshold_rows["outcome"].eq("MBC"), "selected_threshold_original"].iloc[0]) if threshold_rows["outcome"].eq("MBC").any() else np.nan
    mbc = curve.loc[curve["exposure"].eq("Aa_20yr") & curve["outcome"].eq("MBC")].copy()
    mbc = mbc.sort_values("exposure_original")
    fig, ax = plt.subplots(figsize=(9.5, 5.8), constrained_layout=True)
    ax.plot(mbc["exposure_original"], mbc["predicted_outcome"], color="#326f87", linewidth=2.2,
            label="验证集选出的全样本 refit 响应")
    if np.isfinite(threshold):
        ax.axvline(threshold, color="#c44e52", linewidth=1.6, linestyle="--",
                   label=f"分段模型选中候选 Aa={threshold:.3f}")
        ax.text(threshold, ax.get_ylim()[0], f"  Aa={threshold:.3f}", rotation=90, va="bottom", color="#a43d41")
    ax.set_xlabel("Aa_20yr（原始源值单位尚未确认）")
    ax.set_ylabel("MBC 预测值")
    ax.set_title("CORE 05 · MBC 的 Oracle 分段候选阈值")
    ax.grid(color="#e5e7eb", linewidth=.7)
    ax.legend(frameon=False, loc="best")
    fig.text(.5, -.02, "仅 MBC 的验证集最佳模型族为 segmented；Aa≈4.617 是Oracle候选，不是独立确认的生态阈值。",
             ha="center", fontsize=8)
    paths.extend(_save_figure(fig, run_dir / "figures" / "core_figures/C05_MBC_segmented_threshold_candidate", dpi))
    mbc[["exposure_original", "log1p_exposure", "predicted_outcome", "selected_threshold_original",
         "selected_model_family", "outcome_guided_selection"]].to_csv(
        run_dir / "tables/C05_MBC_segmented_threshold_source.csv", index=False)
    return paths


def _plot_environment_figures(config: dict[str, Any], run_dir: Path, dpi: int) -> tuple[list[str], Path, Path]:
    pairs, pair_path, env_path = _pair_group_labels(config)
    pairs = pairs.loc[pairs["exposure"].eq("Aa_20yr")].copy()
    paths: list[str] = []

    soil_order = [f"soil_{soil}" for soil in SOIL_CODES]
    soil_curves = _curve_bins(pairs.loc[pairs["soil_group"].notna()], "soil_group", OUTCOME_ORDER, soil_order)
    colors = {f"soil_{soil}": SOIL_COLORS[soil] for soil in SOIL_CODES}
    fig, axes = plt.subplots(3, 3, figsize=(14.5, 11), squeeze=False, constrained_layout=True)
    for ax, outcome in zip(axes.flat, OUTCOME_ORDER, strict=True):
        data = soil_curves.loc[soil_curves["outcome"].eq(outcome)]
        for group in soil_order:
            d = data.loc[data["group"].eq(group)].sort_values("mean_delta_log1p_Aa")
            if len(d):
                ax.plot(d["mean_delta_log1p_Aa"], d["mean_delta_health"], marker="o", markersize=3,
                        linewidth=1.4, color=colors[group], label=group.replace("soil_", "土类 "))
        ax.axhline(0, color="#777777", linewidth=.65)
        ax.set_title(OUTCOME_LABELS[outcome], loc="left", fontweight="bold")
        ax.set_xlabel("Δlog1p(Aa)，高侵蚀端−低侵蚀端")
        ax.set_ylabel("配对健康差值 ΔY")
        ax.grid(color="#e5e7eb", linewidth=.55)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("CORE 07 · M4 配对响应曲线按相同土类分层", fontsize=15)
    fig.text(.5, -.012, "点为各组内侵蚀差值八分位的配对均值；仅保留高、低端同土类的匹配对。描述性曲线，无区间推断。",
             ha="center", fontsize=8)
    paths.extend(_save_figure(fig, run_dir / "figures/core_figures/C07_soil_type_matched_response_curves", dpi))
    soil_curves.to_csv(run_dir / "tables/C07_soil_type_matched_response_curve_data.csv", index=False)

    twi_curves = _curve_bins(pairs.loc[pairs["TWI_quintile"].notna()], "TWI_quintile", OUTCOME_ORDER,
                             [str(q) for q in range(1, 6)])
    twi_curves["group"] = "q" + twi_curves["group"].astype(str)
    fig, axes = plt.subplots(3, 3, figsize=(14.5, 11), squeeze=False, constrained_layout=True)
    q_colors = plt.get_cmap("viridis")(np.linspace(.12, .88, 5))
    for ax, outcome in zip(axes.flat, OUTCOME_ORDER, strict=True):
        data = twi_curves.loc[twi_curves["outcome"].eq(outcome)]
        for q in range(1, 6):
            group = f"q{q}"
            d = data.loc[data["group"].eq(group)].sort_values("mean_delta_log1p_Aa")
            if len(d):
                ax.plot(d["mean_delta_log1p_Aa"], d["mean_delta_health"], marker="o", markersize=3,
                        linewidth=1.35, color=q_colors[q - 1], label=f"TWI组内五分位 q{q}")
        ax.axhline(0, color="#777777", linewidth=.65)
        ax.set_title(OUTCOME_LABELS[outcome], loc="left", fontweight="bold")
        ax.set_xlabel("Δlog1p(Aa)，高侵蚀端−低侵蚀端")
        ax.set_ylabel("配对健康差值 ΔY")
        ax.grid(color="#e5e7eb", linewidth=.55)
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle("CORE 08 · M4 配对响应曲线按土类内 TWI 五分位分层", fontsize=15)
    fig.text(.5, -.012, "仅保留高、低端处于同一土类内TWI五分位的匹配对；不同土类按五分位等级汇总。描述性曲线，无区间推断。",
             ha="center", fontsize=8)
    paths.extend(_save_figure(fig, run_dir / "figures/core_figures/C08_twi_group_matched_response_curves", dpi))
    twi_curves.to_csv(run_dir / "tables/C08_twi_group_matched_response_curve_data.csv", index=False)

    i3 = _source(config, "i3_run_id") / "tables"
    slopes = pd.read_csv(i3 / "environment_group_slopes_m4.csv")
    slopes = slopes.loc[slopes["exposure"].eq("Aa_20yr") & slopes["method"].eq("M4")].copy()
    slopes["soil"] = pd.to_numeric(slopes["group_label"].astype(str).str.extract(r"^(\d+):q")[0], errors="coerce")
    slopes["quintile"] = pd.to_numeric(slopes["group_label"].astype(str).str.extract(r":q(\d+)$")[0], errors="coerce")
    for variable, label, core_number in [("TWI", "TWI", "C09"), ("NDVI", "NDVI", "C10")]:
        data = slopes.loc[slopes["group_column"].eq(f"gradient_{variable}") &
                          slopes["outcome"].isin(KEY_OUTCOMES) & slopes["quintile"].notna() &
                          pd.to_numeric(slopes["slope_ci_low"], errors="coerce").notna() &
                          pd.to_numeric(slopes["slope_ci_high"], errors="coerce").notna()].copy()
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=False, constrained_layout=True)
        for ax, outcome in zip(axes, KEY_OUTCOMES, strict=True):
            sub = data.loc[data["outcome"].eq(outcome)]
            for soil in SOIL_CODES:
                d = sub.loc[sub["soil"].eq(soil)].sort_values("quintile")
                if d.empty:
                    continue
                y = d["slope_delta_y_per_delta_log1p_erosion"].to_numpy(dtype=float)
                lower = y - pd.to_numeric(d["slope_ci_low"], errors="coerce").to_numpy(dtype=float)
                upper = pd.to_numeric(d["slope_ci_high"], errors="coerce").to_numpy(dtype=float) - y
                ax.errorbar(d["quintile"], y, yerr=np.vstack([lower, upper]), marker="o", capsize=2,
                            linewidth=1.3, color=SOIL_COLORS[soil], label=f"土类 {soil}")
            ax.axhline(0, color="#505b63", linewidth=.8)
            ax.set_xticks(range(1, 6), [f"q{x}" for x in range(1, 6)])
            ax.set_title(OUTCOME_LABELS[outcome], loc="left", fontweight="bold")
            ax.set_xlabel(f"土类内 {label} 五分位")
            ax.set_ylabel("ΔY / Δlog1p(Aa)")
            ax.grid(axis="y", color="#e5e7eb", linewidth=.6)
        axes[0].legend(frameon=False, fontsize=8)
        fig.suptitle(f"{core_number} · Aa×{label} 组内配对斜率及聚类稳健区间", fontsize=14)
        fig.text(.5, -.02, "按土类内环境五分位分别估计；点为M4配对差值斜率，线为5 km聚类稳健95%区间。Oracle探索结果。",
                 ha="center", fontsize=8)
        paths.extend(_save_figure(fig, run_dir / f"figures/core_figures/{core_number}_{label.lower()}_interaction_slopes", dpi))
        data.to_csv(run_dir / f"tables/{core_number}_{label.lower()}_interaction_slopes.csv", index=False)

    forest = slopes.loc[slopes["group_column"].isin(ENV_GROUPS) & slopes["outcome"].isin(KEY_OUTCOMES) &
                        pd.to_numeric(slopes["spatial_blocks"], errors="coerce").ge(5)].copy()
    forest["display_group"] = forest["group_column"].replace({"soil_type_group": "土类"}) + " · " + forest["group_label"].astype(str)
    order = sorted(forest["display_group"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(17, 13), sharey=True, constrained_layout=True)
    group_colors = {"soil_type_group": "#327a8b", "gradient_TWI": "#d17c34", "gradient_NDVI": "#55865e"}
    for ax, outcome in zip(axes, KEY_OUTCOMES, strict=True):
        d = forest.loc[forest["outcome"].eq(outcome)].copy()
        d["ord"] = d["display_group"].map({name: i for i, name in enumerate(order)})
        d = d.sort_values("ord", ascending=False)
        for row in d.itertuples(index=False):
            ax.errorbar(row.slope_delta_y_per_delta_log1p_erosion, row.ord,
                        xerr=[[row.slope_delta_y_per_delta_log1p_erosion - row.slope_ci_low],
                              [row.slope_ci_high - row.slope_delta_y_per_delta_log1p_erosion]],
                        fmt="o", markersize=3.5, capsize=1.5,
                        color=group_colors.get(row.group_column, "#66717a"), linewidth=.8)
        ax.axvline(0, color="#313a40", linewidth=.8)
        ax.set_title(OUTCOME_LABELS[outcome], loc="left", fontweight="bold")
        ax.set_xlabel("ΔY / Δlog1p(Aa)")
        ax.grid(axis="x", color="#e5e7eb", linewidth=.55)
    axes[0].set_yticks(range(len(order)), order, fontsize=6)
    handles = [plt.Line2D([0], [0], marker="o", color=group_colors[key], label=label, linestyle="")
               for key, label in [("soil_type_group", "土类"), ("gradient_TWI", "TWI五分位"), ("gradient_NDVI", "NDVI五分位")]]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(.995, 1.0), ncols=3, frameon=False)
    fig.suptitle("CORE 11 · 土类与环境梯度组内斜率森林图", x=.01, y=1.04, ha="left", fontsize=15)
    fig.text(.5, -.015, "仅展示SHI、OM、SW且至少5个空间块的既有M4估计；区间按5 km空间块聚类。图中不按健康结果重新排序筛选。",
             ha="center", fontsize=8)
    paths.extend(_save_figure(fig, run_dir / "figures/core_figures/C11_environment_group_slope_forest", dpi))
    forest.to_csv(run_dir / "tables/C11_environment_group_slope_forest.csv", index=False)

    response = pd.read_csv(i3 / "low_value_group_response_m4.csv")
    response = response.loc[response["exposure"].eq("Aa_20yr") & response["scheme"].eq("S0_all") &
                            response["analysis_unit"].eq("matched_pixel") & response["group_column"].isin(ENV_GROUPS) &
                            response["outcome"].isin(OUTCOME_ORDER)].copy()
    response["display_group"] = response["group_column"].replace({"soil_type_group": "土类"}) + " · " + response["group_label"].astype(str)
    matrix = response.pivot(index="display_group", columns="outcome", values="r_squared_apparent").reindex(columns=OUTCOME_ORDER)
    fig, ax = plt.subplots(figsize=(13, max(7, .30 * len(matrix))), constrained_layout=True)
    image = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", cmap="YlGnBu", vmin=0, vmax=.5)
    ax.set_xticks(range(len(OUTCOME_ORDER)), [OUTCOME_LABELS[o] for o in OUTCOME_ORDER])
    ax.set_yticks(range(len(matrix.index)), matrix.index, fontsize=7)
    ax.set_title("CORE 12 · 环境组内配对像元 apparent R² 热图（已保存的Phase H字段）")
    ax.set_xlabel("健康结局")
    ax.set_ylabel("环境分组")
    fig.colorbar(image, ax=ax, shrink=.82, label="组内表观 R²（上限显示为0.5）")
    fig.text(.5, -.015, "源表提供的是r_squared_apparent；现有文件未提供可审计的每组模型族最优选择记录，因此不将其称为最佳模型R²。Oracle探索，非空间外推性能。",
             ha="center", fontsize=8)
    paths.extend(_save_figure(fig, run_dir / "figures/core_figures/C12_environment_group_apparent_r2", dpi))
    response.to_csv(run_dir / "tables/C12_environment_group_apparent_r2_source.csv", index=False)
    return paths, pair_path, env_path


def _plot_gwr_composite(config: dict[str, Any], run_dir: Path, dpi: int) -> list[str]:
    from experiments.oracle_exploration.phase_i_spatial import _add_map_decor, _mask_arrays, _read_pixels

    i4 = _source(config, "i4_run_id")
    local = pd.read_parquet(i4 / "tables/full_sample_local_coefficients.parquet")
    local = local.loc[local["analysis_population"].eq("environment_adjusted") & local["model_family"].eq("GWR") &
                      local["outcome"].isin(HEALTH_INDICATORS)].copy()
    pixel_config = yaml.safe_load((ROOT / "config/phase_i_spatial.yaml").read_text(encoding="utf-8"))
    pixels, _, _ = _read_pixels(pixel_config)
    mask_info = _mask_arrays(pixels)
    values = pd.to_numeric(local["local_beta_standardized"], errors="coerce").to_numpy(dtype=float)
    limit = max(float(np.nanquantile(np.abs(values), .97)), .05)
    fig, axes = plt.subplots(3, 3, figsize=(15.5, 13.2), squeeze=False, constrained_layout=True)
    for ax, outcome in zip(axes.flat, HEALTH_INDICATORS, strict=False):
        data = local.loc[local["outcome"].eq(outcome)]
        finite = np.isfinite(data[["x", "y", "local_beta_standardized"]].to_numpy(dtype=float)).all(axis=1)
        data = data.loc[finite]
        scatter = ax.scatter(data["x"] / 1000, data["y"] / 1000, c=data["local_beta_standardized"],
                             s=5, marker="s", cmap="RdBu_r", vmin=-limit, vmax=limit, linewidths=0,
                             rasterized=True)
        _add_map_decor(ax, *mask_info)
        ax.set_title(OUTCOME_LABELS[outcome], loc="left", fontweight="bold")
        ax.set_xlabel("Easting (km, EPSG:32649)")
        ax.set_ylabel("Northing (km)")
        ax.set_aspect("equal", adjustable="datalim")
    axes.flat[-1].axis("off")
    colorbar = fig.colorbar(ScalarMappable(norm=Normalize(-limit, limit), cmap="RdBu_r"),
                            ax=list(axes.flat[:-1]), orientation="horizontal", fraction=.035, pad=.03)
    colorbar.set_label("环境调整 GWR 局部标准化 log1p(Aa) 系数，所有八指标共用色标")
    fig.suptitle("CORE 16 · 八项健康指标环境调整 GWR 局部侵蚀系数", fontsize=15)
    fig.text(.5, -.008, "每个点为约240m汇总单元；同一色标以0为中心。系数为探索性局部关联，不作局部因果或显著性判定。",
             ha="center", fontsize=8)
    return _save_figure(fig, run_dir / "figures/core_figures/C16_eight_indicator_gwr_composite", dpi)


def _plot_source_gaps(run_dir: Path, dpi: int) -> list[str]:
    paths: list[str] = []
    fig, ax = plt.subplots(figsize=(10, 5.3), constrained_layout=True)
    ax.axis("off")
    ax.text(.5, .86, "CORE 30 · EnvMatch 与 PCNM 对照森林图", transform=ax.transAxes,
            ha="center", fontsize=16, fontweight="bold")
    ax.text(.5, .62, "未提供可读取的既有 PCNM 结果表", transform=ax.transAxes,
            ha="center", fontsize=13, color="#a44343", fontweight="bold")
    ax.text(.5, .43, "系数、标准误/区间、P值、R²、样本量、控制变量及分析单位均无法从本工作区核实。\n"
            "I-7交叉表对应字段保持为空；未重跑PCNM，也未绘制虚构的森林点或区间。",
            transform=ax.transAxes, ha="center", va="center", fontsize=11, linespacing=1.7)
    ax.text(.5, .12, "状态：数据源缺失，科学比较图待源表接入后生成", transform=ax.transAxes,
            ha="center", fontsize=10, color="#4c5b66")
    paths.extend(_save_figure(fig, run_dir / "figures/core_figures/C30_PCNM_comparison_NOT_AVAILABLE", dpi))

    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(.5, .9, "CORE 31 · 综合胁迫机制路径图", ha="center", fontsize=16, fontweight="bold")
    boxes = [(.05, .43, .34, .29, "EnvMatch Phase I\n\n预测栅格关联与空间异质性\n已完成探索性估计", "#e7f0f3"),
             (.61, .43, .34, .29, "既有 PLS-SEM\n\n路径系数、载荷/符号、控制项\n源结果未提供", "#f8e9e9")]
    for x, y, w, h, label, color in boxes:
        rect = plt.Rectangle((x, y), w, h, transform=ax.transAxes, facecolor=color,
                             edgecolor="#63727c", linewidth=1.2, joinstyle="round")
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, transform=ax.transAxes, ha="center", va="center", fontsize=11)
    ax.plot([.40, .60], [.575, .575], transform=ax.transAxes, color="#a44343", linestyle="--", linewidth=1.5)
    ax.text(.50, .63, "不可估计", transform=ax.transAxes, ha="center", color="#a44343", fontsize=10, fontweight="bold")
    ax.text(.5, .22, "此状态图仅标明证据缺口；不表示因果路径，不补造方向、系数或机制。\n"
            "待接入实际 PLS-SEM 表及测量模型定义后再完成科学机制图。",
            transform=ax.transAxes, ha="center", va="center", fontsize=10, linespacing=1.6)
    paths.extend(_save_figure(fig, run_dir / "figures/core_figures/C31_mechanism_path_NOT_AVAILABLE", dpi))
    return paths


def _build_mgwr_sample(config: dict[str, Any], run_dir: Path) -> tuple[Path, Path]:
    input_root = ROOT / "outputs" / config["input_run_id"] / "databases"
    env_path = input_root / "environment_erosion.parquet"
    health_path = input_root / "health.parquet"
    env_cols = ["cell_id", "row", "col", "x", "y", config["primary_exposure"], *config["environment_covariates"]]
    env = pd.read_parquet(env_path, columns=env_cols)
    health = pd.read_parquet(health_path, columns=["cell_id", *KEY_OUTCOMES])
    data = env.merge(health, on="cell_id", how="inner", validate="one_to_one")
    data["log1p_Aa"] = np.log1p(pd.to_numeric(data[config["primary_exposure"]], errors="coerce"))
    features = ["log1p_Aa", *config["environment_covariates"]]
    required = ["x", "y", *features, *KEY_OUTCOMES]
    values = data[required].apply(pd.to_numeric, errors="coerce")
    data = data.loc[np.isfinite(values.to_numpy(dtype=float)).all(axis=1)].copy()
    tile_size = int(config["tile_cells"])
    data["tile_row"] = np.floor(data["row"].to_numpy(dtype=float) / tile_size).astype(np.int32)
    data["tile_col"] = np.floor(data["col"].to_numpy(dtype=float) / tile_size).astype(np.int32)
    agg = {name: "mean" for name in ["x", "y", *features, *KEY_OUTCOMES]}
    units = data.groupby(["tile_row", "tile_col"], sort=True).agg(agg)
    units["n_pixels"] = data.groupby(["tile_row", "tile_col"], sort=True).size()
    units = units.reset_index()
    units["unit_id"] = units["tile_row"].astype(str) + "_" + units["tile_col"].astype(str)
    grid = int(config["mgwr_sampling_grid_m"])
    units["sample_grid_x"] = np.floor(units["x"] / grid).astype(np.int32)
    units["sample_grid_y"] = np.floor(units["y"] / grid).astype(np.int32)
    selected = units.groupby(["sample_grid_x", "sample_grid_y"], sort=True, group_keys=False).sample(
        n=1, random_state=int(config["mgwr_sample_seed"])
    ).sort_values(["sample_grid_y", "sample_grid_x", "tile_row", "tile_col"]).reset_index(drop=True)
    sample_path = run_dir / "tables/mgwr_fit_input_spatially_balanced.csv"
    selected.to_csv(sample_path, index=False)
    summary = {
        "sampling_design": "one 240m block-mean unit sampled per occupied fixed-grid cell",
        "grid_cell_m": grid, "seed": int(config["mgwr_sample_seed"]),
        "source_complete_240m_units": int(len(units)), "sample_units": int(len(selected)),
        "source_pixel_count": int(selected["n_pixels"].sum()), "tile_cells": tile_size,
        "coordinate_reference": "EPSG:32649", "analysis_unit": "240m block mean; unweighted across selected units",
        "sample_path": sample_path.relative_to(ROOT).as_posix(),
        "outcomes": KEY_OUTCOMES, "predictors": features,
    }
    summary_path = run_dir / "tables/mgwr_sampling_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return sample_path, summary_path


def _source_gap_report(run_dir: Path, paths: list[str], pair_path: Path, env_path: Path) -> Path:
    report = run_dir / "reports/PHASE_I_SUPPLEMENTARY_FIGURES_REPORT.md"
    rel = lambda p: Path(p).as_posix()
    lines = [
        "# Phase I 图件闭合与标准 MGWR 补充",
        "",
        "phase: I-closure",
        "mode: EXPLORATORY_ORACLE",
        "outcome_guided_selection: true",
        "",
        "本运行只整理既有 Phase I/Phase H 表格与数据库、补制缺失核心图件，并准备标准 MGWR 的空间均衡样本。不会覆盖 I-1 至 I-9 既有运行，不重建/验证 RUSLE 因子乘积，不重跑 PCNM。",
        "",
        "## 环境异质性补图",
        "",
        f"配对响应曲线沿用既有 Aa_20yr M4 候选 `{pair_path.relative_to(ROOT)}`，只按端点土类、以及土类内环境五分位分层后汇总八分位配对均值。曲线是描述性汇总，不是新增模型拟合。环境分组使用对齐数据库 `{env_path.relative_to(ROOT)}` 中的 SoilType、TWI、NDVI。",
        "",
        "核心图 12 的可读源表只提供 `r_squared_apparent` 字段；没有每个环境组的可审计最佳模型族搜索/筛选记录。因此图中按表字段诚实标为组内 apparent R²，不称为环境组最佳模型R²，也不作空间外推解释。",
        "",
        "## GWR 与标准 MGWR 图件",
        "",
        "八指标 GWR 组合图复用 I-4 环境调整全样本 GWR 局部标准化侵蚀系数，所有面板共用以零为中心的色标。标准 MGWR 使用与标准 GWR 相同的空间均衡 240m 汇总单元样本、相同的 Aa_20yr 和九项环境协变量、adaptive Gaussian kernel；每个抽样网格单元等权。全样本 I-4 是像元支持加权的 240m 单元，与本补充的空间均衡样本估计对象不同。",
        "",
        "标准 MGWR 只拟合 SHI、OM、SW 三个关键结局。样本规模、算法收敛、拟合指标和变量邻域尺度应查看 `tables/` 与最终阶段报告；不将 MGWR 局部系数解释为因果效应。",
        "",
        "## PCNM/PLS-SEM 源数据缺口",
        "",
        "PCNM 源结果表缺失，PLS-SEM 路径/载荷/控制项表也未提供。核心图 30、31 的文件是标明缺口的状态图，不是对照森林图或机制图；I-7交叉字段仍保持空值，不补造结果。收到实际源表后才可完成科学对照。",
        "",
        "## 已生成图件",
        "",
        *[f"- `{rel(p)}`" for p in paths],
        "",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def _write_core_figure_audit(run_dir: Path, include_mgwr: bool) -> Path:
    closure = run_dir.relative_to(ROOT).as_posix()
    i1 = "outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_r2"
    i2 = "outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_i2_r2"
    i3 = "outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_i3_r2"
    i4 = "outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_i4_r3"
    i5 = "outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_i5_r1"
    i6 = "outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_i6_r1"
    i7 = "outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_i7_r1"
    i8 = "outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_i8_r2"
    rows = [
        (1, "九项指标全样本响应曲线", "available", f"{i1}/figures/response_curves/I05_global_response_curves.png", "全样本验证选中模型 refit 曲线"),
        (2, "零侵蚀与正侵蚀状态差异", "available", f"{i1}/figures/global_response/I04_zero_positive_boxplots.png", ""),
        (3, "九项指标相关性森林图", "available", f"{i1}/figures/forest_plots/I02_standardized_effect_forest.png", ""),
        (4, "最佳模型R²比较", "available", f"{closure}/figures/core_figures/C04_global_best_model_r2.png", "验证R²用于模型族比较，测试R²仅报告"),
        (5, "非线性阈值图", "partial", f"{closure}/figures/core_figures/C05_MBC_segmented_threshold_candidate.png", "仅MBC的验证集最佳模型为segmented；Aa≈4.617为Oracle候选"),
        (6, "分位数回归系数图", "available", f"{i1}/figures/model_comparison/I07_quantile_slopes.png", ""),
        (7, "土壤类型响应曲线", "available", f"{closure}/figures/core_figures/C07_soil_type_matched_response_curves.png", "9项结局，匹配对描述曲线"),
        (8, "TWI分组响应曲线", "available", f"{closure}/figures/core_figures/C08_twi_group_matched_response_curves.png", "9项结局，土类内TWI五分位"),
        (9, "Aa×TWI交互图", "partial", f"{closure}/figures/core_figures/C09_twi_interaction_slopes.png", "SHI、OM、SW组内斜率和区间"),
        (10, "Aa×NDVI交互图", "partial", f"{closure}/figures/core_figures/C10_ndvi_interaction_slopes.png", "SHI、OM、SW组内斜率和区间"),
        (11, "环境组斜率森林图", "partial", f"{closure}/figures/core_figures/C11_environment_group_slope_forest.png", "SHI、OM、SW；土类/TWI/NDVI组"),
        (12, "环境组最佳模型R²热图", "partial", f"{closure}/figures/core_figures/C12_environment_group_apparent_r2.png", "源字段为组内apparent R²；未声称最佳模型选择"),
        (13, "SHI的GWR局部系数图", "available", f"{i4}/figures/gwr_SHI_candidate_template.png", ""),
        (14, "OM的GWR局部系数图", "available", f"{i4}/figures/gwr_OM_template.png", ""),
        (15, "SW的GWR局部系数图", "available", f"{i4}/figures/gwr_SW_template.png", ""),
        (16, "八项指标GWR组合图", "available", f"{closure}/figures/core_figures/C16_eight_indicator_gwr_composite.png", "八项非SHI健康指标，共用色标"),
        (17, "MGWR局部系数比较图", "available" if include_mgwr else "pending", f"{closure}/figures/core_figures/C17_MGWR_local_coefficients.png", "仅SHI、OM、SW空间均衡样本"),
        (18, "MGWR变量作用尺度图", "available" if include_mgwr else "pending", f"{closure}/figures/core_figures/C18_MGWR_variable_scales.png", "自适应邻居数与中位数距离"),
        (19, "GWR与MGWR空间结果对照", "available" if include_mgwr else "pending", f"{closure}/figures/core_figures/C19_GWR_MGWR_difference_map.png", "同一空间均衡样本、同一协变量集"),
        (20, "GRF局部R²图", "available", f"{i5}/figures/grf_local_r2.png", ""),
        (21, "GRF局部侵蚀响应图", "available", f"{i5}/figures/grf_local_Aa_marginal_effect.png", ""),
        (22, "SHAP侵蚀贡献空间图", "available", f"{i5}/figures/spatial_shap_Aa.png", ""),
        (23, "空间ALE交互效应图", "available", f"{i5}/figures/local_ale_by_soil_type.png", ""),
        (24, "SVC-GAM局部响应曲线", "available", f"{i5}/figures/svc_gam_local_curves.png", ""),
        (25, "Local Spearman空间图", "available", f"{i6}/figures/local_spearman_240m.png", ""),
        (26, "Bivariate LISA聚类图", "available", f"{i6}/figures/bivariate_local_moran_clusters.png", ""),
        (27, "Getis-Ord Gi*热点图", "available", f"{i6}/figures/gistar_negative_response_gwr.png", ""),
        (28, "综合胁迫分区图", "available", f"{i8}/figures/stress_zoning_map.png", ""),
        (29, "物理/化学/生物维度对照", "available", f"{i7}/figures/health_dimension_comparison_forest.png", ""),
        (30, "EnvMatch与PCNM对照森林图", "deferred_source_missing", f"{closure}/figures/core_figures/C30_PCNM_comparison_NOT_AVAILABLE.png", "只有数据缺口状态图；PCNM源表缺失"),
        (31, "综合胁迫机制路径图", "deferred_source_missing", f"{closure}/figures/core_figures/C31_mechanism_path_NOT_AVAILABLE.png", "只有数据缺口状态图；PLS-SEM源表缺失"),
    ]
    frame = pd.DataFrame(rows, columns=["figure_number", "required_figure", "status", "artifact_path", "evidence_note"])
    path = run_dir / "tables/core_figure_completeness.csv"
    frame.to_csv(path, index=False)
    return path


def _plot_mgwr_figures(config: dict[str, Any], run_dir: Path, dpi: int) -> tuple[list[str], bool]:
    table_dir = run_dir / "tables"
    status_path = table_dir / "mgwr_fit_status.csv"
    coef_path = table_dir / "mgwr_local_coefficients.csv"
    if not status_path.is_file() or not coef_path.is_file():
        return [], False
    statuses = pd.read_csv(status_path)
    coefs = pd.read_csv(coef_path)
    mgwr_ok = set(statuses.loc[statuses["status"].eq("ok"), "outcome"])
    required = set(KEY_OUTCOMES)
    if not required.issubset(mgwr_ok):
        return [], False
    from experiments.oracle_exploration.phase_i_spatial import _add_map_decor, _mask_arrays, _read_pixels

    pixel_config = yaml.safe_load((ROOT / "config/phase_i_spatial.yaml").read_text(encoding="utf-8"))
    pixels, _, _ = _read_pixels(pixel_config)
    mask_info = _mask_arrays(pixels)
    dpi = max(int(dpi), 600)
    paths: list[str] = []
    map_data = coefs.loc[coefs["outcome"].isin(KEY_OUTCOMES) &
                         coefs["model_family"].isin(["standard_GWR_same_sample", "standard_MGWR_spatially_balanced_sample"])].copy()
    vals = pd.to_numeric(map_data["local_beta_log1p_Aa_standardized"], errors="coerce").to_numpy(dtype=float)
    limit = max(float(np.nanquantile(np.abs(vals), .97)), .05)
    fig, axes = plt.subplots(3, 2, figsize=(13, 13), squeeze=False, constrained_layout=True)
    for row, outcome in enumerate(KEY_OUTCOMES):
        for col, model_family in enumerate(["standard_GWR_same_sample", "standard_MGWR_spatially_balanced_sample"]):
            ax = axes[row, col]
            d = map_data.loc[map_data["outcome"].eq(outcome) & map_data["model_family"].eq(model_family)]
            finite = np.isfinite(d[["x", "y", "local_beta_log1p_Aa_standardized"]].to_numpy(dtype=float)).all(axis=1)
            d = d.loc[finite]
            ax.scatter(d["x"] / 1000, d["y"] / 1000, c=d["local_beta_log1p_Aa_standardized"], s=12,
                       marker="s", cmap="RdBu_r", vmin=-limit, vmax=limit, linewidths=0, rasterized=True)
            _add_map_decor(ax, *mask_info)
            ax.set_title(f"{OUTCOME_LABELS[outcome]} · {'GWR' if col == 0 else 'MGWR'}", loc="left", fontweight="bold")
            ax.set_xlabel("Easting (km, EPSG:32649)")
            ax.set_ylabel("Northing (km)")
            ax.set_aspect("equal", adjustable="datalim")
    cbar = fig.colorbar(ScalarMappable(norm=Normalize(-limit, limit), cmap="RdBu_r"),
                        ax=list(axes.flat), orientation="horizontal", fraction=.035, pad=.03)
    cbar.set_label("局部标准化 log1p(Aa) 系数；同一结局GWR/MGWR共用色标")
    fig.suptitle("CORE 17 · 标准 GWR 与 MGWR 局部侵蚀系数（相同空间均衡样本）", fontsize=15)
    fig.text(.5, -.008, "仅SHI、OM、SW；适应性Gaussian核。每个抽样240m单元等权。局部系数描述条件关联。",
             ha="center", fontsize=8)
    paths.extend(_save_figure(fig, run_dir / "figures/core_figures/C17_MGWR_local_coefficients", dpi))

    bandwidth_path = table_dir / "mgwr_variable_bandwidths.csv"
    if bandwidth_path.is_file():
        bandwidths = pd.read_csv(bandwidth_path)
        variables = ["Intercept", "log1p_Aa", "Elevation", "Cur", "TWI", "LST", "NDWI", "NDBSI", "EVI", "NDVI", "Precipitation"]
        labels = {"Intercept": "截距", "log1p_Aa": "log1p(Aa)", "Elevation": "Elevation",
                  "Cur": "Cur", "TWI": "TWI", "LST": "LST", "NDWI": "NDWI",
                  "NDBSI": "NDBSI", "EVI": "EVI", "NDVI": "NDVI", "Precipitation": "降水"}
        fig, axes = plt.subplots(1, 2, figsize=(13, 8), squeeze=False, constrained_layout=True)
        for ax, metric, title, cmap in [
            (axes[0, 0], "bandwidth_neighbors", "自适应邻居数", "Blues"),
            (axes[0, 1], "median_neighbor_distance_m", "中位邻域距离（km）", "YlGnBu"),
        ]:
            matrix = bandwidths.pivot(index="variable", columns="outcome", values=metric).reindex(index=variables, columns=KEY_OUTCOMES)
            display = matrix.to_numpy(dtype=float)
            if metric == "median_neighbor_distance_m":
                display = display / 1000
            im = ax.imshow(display, aspect="auto", cmap=cmap)
            ax.set_xticks(range(len(KEY_OUTCOMES)), [OUTCOME_LABELS[o] for o in KEY_OUTCOMES])
            ax.set_yticks(range(len(variables)), [labels[v] for v in variables])
            ax.set_title(title)
            for yi in range(display.shape[0]):
                for xi in range(display.shape[1]):
                    value = display[yi, xi]
                    if np.isfinite(value):
                        red, green, blue, _ = im.cmap(im.norm(value))
                        luminance = .2126 * red + .7152 * green + .0722 * blue
                        ax.text(xi, yi, f"{value:.0f}" if metric == "bandwidth_neighbors" else f"{value:.1f}",
                                ha="center", va="center", fontsize=8,
                                color="white" if luminance < .52 else "#16232b")
            fig.colorbar(im, ax=ax, shrink=.78)
        fig.suptitle("CORE 18 · 标准 MGWR 的变量作用尺度", fontsize=15)
        fig.text(.5, -.012, "作用尺度由MGWR自适应邻居数及相应中位距离表示；边界值和局部共线性须结合原始结果表审阅。",
                 ha="center", fontsize=8)
        paths.extend(_save_figure(fig, run_dir / "figures/core_figures/C18_MGWR_variable_scales", dpi))

    difference_records = []
    for outcome in KEY_OUTCOMES:
        gwr = map_data.loc[map_data["outcome"].eq(outcome) & map_data["model_family"].eq("standard_GWR_same_sample")]
        mgwr = map_data.loc[map_data["outcome"].eq(outcome) & map_data["model_family"].eq("standard_MGWR_spatially_balanced_sample")]
        merged = gwr.merge(mgwr, on=["x", "y", "tile_row", "tile_col", "outcome"],
                           suffixes=("_gwr", "_mgwr"), validate="one_to_one")
        merged["local_beta_difference_mgwr_minus_gwr"] = (
            merged["local_beta_log1p_Aa_standardized_mgwr"] - merged["local_beta_log1p_Aa_standardized_gwr"]
        )
        difference_records.append(merged)
    differences = pd.concat(difference_records, ignore_index=True) if difference_records else pd.DataFrame()
    if not differences.empty:
        difference_limit = max(float(np.nanquantile(np.abs(differences["local_beta_difference_mgwr_minus_gwr"]), .97)), .03)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.7), squeeze=False, constrained_layout=True)
        for ax, outcome in zip(axes[0], KEY_OUTCOMES, strict=True):
            d = differences.loc[differences["outcome"].eq(outcome)]
            ax.scatter(d["x"] / 1000, d["y"] / 1000, c=d["local_beta_difference_mgwr_minus_gwr"], s=13,
                       marker="s", cmap="RdBu_r", vmin=-difference_limit, vmax=difference_limit,
                       linewidths=0, rasterized=True)
            _add_map_decor(ax, *mask_info)
            ax.set_title(OUTCOME_LABELS[outcome], loc="left", fontweight="bold")
            ax.set_xlabel("Easting (km)"); ax.set_ylabel("Northing (km)")
            ax.set_aspect("equal", adjustable="datalim")
        cbar = fig.colorbar(ScalarMappable(norm=Normalize(-difference_limit, difference_limit), cmap="RdBu_r"),
                            ax=list(axes.flat), orientation="horizontal", fraction=.07, pad=.05)
        cbar.set_label("MGWR局部系数 − 同样本GWR局部系数")
        fig.suptitle("CORE 19 · 同一空间单元的 GWR–MGWR 系数差异", fontsize=15)
        fig.text(.5, -.008, "只比较相同抽样单元和变量标准化口径；差异反映模型尺度设定敏感性，不代表模型优劣。",
                 ha="center", fontsize=8)
        paths.extend(_save_figure(fig, run_dir / "figures/core_figures/C19_GWR_MGWR_difference_map", dpi))
        differences.to_csv(table_dir / "gwr_mgwr_local_coefficient_differences.csv", index=False)
    return paths, True


def finalize(config_path: Path) -> Path:
    config = yaml.safe_load(config_path.resolve().read_text(encoding="utf-8"))
    run_dir = _run_dir(config)
    if not run_dir.exists():
        raise FileNotFoundError(f"Prepare the closure run first: {run_dir}")
    dpi = int(config["figure_dpi"])
    mgwr_paths, include_mgwr = _plot_mgwr_figures(config, run_dir, dpi)
    _write_core_figure_audit(run_dir, include_mgwr=include_mgwr)
    status_path = run_dir / "tables/mgwr_fit_status.csv"
    statuses = pd.read_csv(status_path) if status_path.is_file() else pd.DataFrame()
    summary_path = run_dir / "tables/mgwr_model_summary.csv"
    summaries = pd.read_csv(summary_path) if summary_path.is_file() else pd.DataFrame()
    bandwidth_path = run_dir / "tables/mgwr_variable_bandwidths.csv"
    bandwidths = pd.read_csv(bandwidth_path) if bandwidth_path.is_file() else pd.DataFrame()
    sampling_summary_path = run_dir / "tables/mgwr_sampling_summary.json"
    sampling = json.loads(sampling_summary_path.read_text(encoding="utf-8")) if sampling_summary_path.is_file() else {}
    rows = []
    for outcome in KEY_OUTCOMES:
        one = summaries.loc[summaries.get("outcome", pd.Series(dtype=str)).eq(outcome)] if not summaries.empty else pd.DataFrame()
        gwr = one.loc[one["model_family"].eq("standard_GWR_same_sample")] if not one.empty else pd.DataFrame()
        mgwr = one.loc[one["model_family"].eq("standard_MGWR_spatially_balanced_sample")] if not one.empty else pd.DataFrame()
        rows.append(f"| {OUTCOME_LABELS[outcome]} | {_format(gwr, 'r_squared')} | {_format(mgwr, 'r_squared')} | {_format(gwr, 'aicc')} | {_format(mgwr, 'aicc')} |")
    fit_summary = "\n".join(rows) if rows else "| — | — | — | — | — |"
    mgwr_ready = include_mgwr
    lines = [
        "# Phase I 图件闭合与标准 MGWR 补充报告",
        "",
        "phase: I-closure",
        "mode: EXPLORATORY_ORACLE",
        "outcome_guided_selection: true",
        "",
        "本运行创建核心图件缺口审计并补制可由已保存数据支持的图件。所有旧运行均保留；图件基于既有 M4 配对、I-1 模型搜索、I-3/I-4/I-7表格及对齐数据库。没有重建/验证 RUSLE 因子乘积，也没有重跑 PCNM。",
        "",
        "## 标准 MGWR",
        "",
        f"- 240m完整案例空间单元：{sampling.get('source_complete_240m_units', '—'):,}；按{sampling.get('grid_cell_m', '—')}m固定网格每格抽取1个单元，实际样本 n={sampling.get('sample_units', '—'):,}；随机种子 {sampling.get('seed', '—')}。抽样只依空间位置，不按健康响应取样。",
        "- 模型使用九项环境协变量及 log1p(Aa_20yr)，Gaussian kernel，自适应邻居数，AICc带宽选择；每个空间均衡单元等权，不使用源像元频数权重。标准GWR与标准MGWR使用同一空间样本和设计矩阵。",
        f"- 标准MGWR fit status：{'三个关键结局均成功' if mgwr_ready else '存在失败或未完成结局'}；包版本、参数、迭代状态及错误信息见 `tables/standard_mgwr_metadata.json` 与 `tables/mgwr_fit_status.csv`。",
        "- `mgwr==2.2.1` 尚未实现多带宽 MGWR 的局部 R²；`mgwr_local_coefficients.csv` 中 MGWR 行的 `local_r2` 留空，不能按零解释。模型级 R²、AICc及局部系数仍按包可提供的字段记录。",
        "",
        "| 指标 | 同样本标准GWR R² | 标准MGWR R² | GWR AICc | MGWR AICc |",
        "|---|---:|---:|---:|---:|",
        fit_summary,
        "",
        "作用尺度表 `mgwr_variable_bandwidths.csv` 同时保留每个变量的自适应邻居数和对应中位邻域距离。局部系数是预测栅格上的条件关联，样本空间依赖、变量局部共线性与 Oracle 选择仍限制解释；本拟合不替代I-4全样本像元支持加权GWR。",
        "",
        "## 核心图件状态",
        "",
        "逐项证据见 `tables/core_figure_completeness.csv`。核心图 5 仅有 MBC 的最佳验证模型族选中 segmented，因此该图只标出 MBC 的候选阈值；其他结局未选中 segmented，不填造阈值。核心图 12 的源数据只有 `r_squared_apparent`，图中保留该字段名，不伪称最佳模型R²。",
        "",
        "核心图 30、31 已创建显式的来源缺口状态图，但这两张不能代替科学对照森林图/机制路径图。I-7的PCNM比较值、PLS路径系数、测量载荷、方向、控制项均继续为空；需在接入源表后完成。",
        "",
        "## 图件与数据",
        "",
        "| 图件 | PNG | SVG | PDF |",
        "|---|---|---|---|",
    ]
    audit = pd.read_csv(run_dir / "tables/core_figure_completeness.csv")
    report = run_dir / "reports/PHASE_I_CLOSURE_REPORT.md"
    for row in audit.itertuples(index=False):
        path = ROOT / row.artifact_path
        if path.suffix == ".png":
            stem = path.with_suffix("")
            links = []
            for ext in ("png", "svg", "pdf"):
                artifact = stem.with_suffix(f".{ext}")
                links.append(f"[{ext.upper()}]({Path(os.path.relpath(artifact, start=report.parent)).as_posix()})" if artifact.exists() else "—")
            lines.append(f"| {row.figure_number}. {row.required_figure} ({row.status}) | {links[0]} | {links[1]} | {links[2]} |")
    lines.extend(["", "## 源文件和方法状态", "",
                  "全部输入数据库及图件配置/来源版本见 `manifests/provenance.json`；图件数据汇总表与局部系数明细位于 `tables/`。最终综合报告及图件索引在独立 I-9 r3 运行生成。", ""])
    report.write_text("\n".join(lines), encoding="utf-8")
    report_file = run_dir / "reports/PHASE_I_SUPPLEMENTARY_FIGURES_REPORT.md"
    if report_file.is_file():
        current = report_file.read_text(encoding="utf-8").split("\n\n## 标准 MGWR 拟合完成状态", 1)[0].rstrip()
        current += "\n\n## 标准 MGWR 拟合完成状态\n\n"
        current += f"标准 MGWR 包 `mgwr` 已安装于隔离临时环境；当前拟合状态：{'三个关键结局成功' if mgwr_ready else '部分结局失败/未完成'}。详细样本、模型、带宽和结果见 [标准 MGWR 补充报告](PHASE_I_CLOSURE_REPORT.md)。\n"
        report_file.write_text(current, encoding="utf-8")
    provenance_path = run_dir / "manifests/provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["mgwr_status"] = "standard MGWR completed" if mgwr_ready else "standard MGWR partial or failed"
    provenance["mgwr_fit_status"] = statuses.to_dict(orient="records") if not statuses.empty else []
    provenance["standard_mgwr_report"] = report.relative_to(ROOT).as_posix()
    provenance["figure_outputs"] = list(dict.fromkeys(provenance.get("figure_outputs", []) + mgwr_paths))
    provenance["software"]["mgwr"] = "2.2.1" if mgwr_ready else None
    provenance["software"]["mgwr_runtime"] = "/tmp/envmatch-mgwr-venv (temporary isolated environment)"
    provenance["core_figure_status_counts"] = audit["status"].value_counts().to_dict()
    (run_dir / "manifests/provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_output_manifest(run_dir)
    (run_dir / "RUN_STATUS.md").write_text(
        f"# {config['run_id']}\n\nStatus: {'analysis and feasible figure closure complete' if mgwr_ready else 'partial: standard MGWR not fully fitted'}\n\n"
        "Mode: EXPLORATORY_ORACLE\n\noutcome_guided_selection: true\n\n"
        "PCNM/PLS source comparison remains deferred because source tables are absent; see core_figure_completeness.csv.\n",
        encoding="utf-8")
    return report


def _format(frame: pd.DataFrame, column: str) -> str:
    if frame.empty or column not in frame:
        return "—"
    value = pd.to_numeric(frame.iloc[0][column], errors="coerce")
    return "—" if not np.isfinite(value) else f"{value:.3f}"


def prepare(config_path: Path) -> Path:
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = _make_run(config, config_path)
    dpi = int(config["figure_dpi"])
    paths = _plot_i1_core(config, run_dir, dpi)
    env_paths, pair_path, env_path = _plot_environment_figures(config, run_dir, dpi)
    paths.extend(env_paths)
    paths.extend(_plot_gwr_composite(config, run_dir, dpi))
    paths.extend(_plot_source_gaps(run_dir, dpi))
    sample_path, sample_summary = _build_mgwr_sample(config, run_dir)
    report_path = _source_gap_report(run_dir, paths, pair_path, env_path)
    _write_core_figure_audit(run_dir, include_mgwr=False)
    provenance = {
        "run_id": config["run_id"], "phase": config["phase"], "mode": "EXPLORATORY_ORACLE",
        "outcome_guided_selection": True, "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git("rev-parse", "HEAD"), "git_status_porcelain": _git("status", "--short"),
        "inputs": {"paired_candidate": pair_path.relative_to(ROOT).as_posix(),
                   "aligned_environment": env_path.relative_to(ROOT).as_posix(),
                   "I1_run": config["i1_run_id"], "I3_run": config["i3_run_id"],
                   "I4_run": config["i4_run_id"], "I7_run": config["i7_run_id"]},
        "mgwr_sample": sample_path.relative_to(ROOT).as_posix(), "mgwr_sample_summary": sample_summary.relative_to(ROOT).as_posix(),
        "mgwr_status": "sample prepared; standard MGWR fit pending", "figure_outputs": paths,
        "report": report_path.relative_to(ROOT).as_posix(),
        "software": {"python": platform.python_version(), "pandas": pd.__version__, "matplotlib": _version("matplotlib"), "numpy": np.__version__},
    }
    (run_dir / "manifests/provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_output_manifest(run_dir)
    (run_dir / "RUN_STATUS.md").write_text(
        f"# {config['run_id']}\n\nStatus: standard MGWR fit pending\n\n"
        "Mode: EXPLORATORY_ORACLE\n\noutcome_guided_selection: true\n\n"
        "I1/I3/I4/I7 supplements and the explicit PCNM/PLS source-gap artifacts are prepared.\n",
        encoding="utf-8")
    return run_dir


def _write_output_manifest(run_dir: Path) -> Path:
    path = run_dir / "manifests/output_manifest.json"
    files = sorted(p for p in run_dir.rglob("*") if p.is_file() and p != path)
    path.write_text(json.dumps([{"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size}
                                for p in files], ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "finalize"])
    parser.add_argument("--config", type=Path, default=ROOT / "config/phase_i_closure.yaml")
    args = parser.parse_args()
    if args.action == "prepare":
        print(prepare(args.config))
    else:
        print(finalize(args.config))


if __name__ == "__main__":
    main()
