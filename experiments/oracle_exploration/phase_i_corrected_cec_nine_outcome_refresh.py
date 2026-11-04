"""Consolidate nine-outcome Phase I spatial results with corrected CEC."""

from __future__ import annotations

import json
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
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "oracle_exploration"))
import phase_i_spatial as spatial  # noqa: E402

RUN_ID = "phase_i_20261101_corrected_cec_nine_outcome_refresh_r2"
RUN_DIR = ROOT / "outputs" / "exploratory_oracle" / "phase_i" / RUN_ID
OUTCOMES = ["SHI_candidate", "AN", "OM", "CEC", "pH", "SW", "BD", "MBC", "RESP"]
FOCUS = ["SHI_candidate", "SW", "OM"]
LABELS = {
    "SHI_candidate": "综合土壤健康指数（SHI）",
    "AN": "碱解氮（AN）",
    "OM": "有机质（OM）",
    "CEC": "阳离子交换量（CEC）",
    "pH": "土壤酸碱度（pH）",
    "SW": "土壤含水量（SW）",
    "BD": "土壤容重（BD）",
    "MBC": "微生物量碳（MBC）",
    "RESP": "土壤呼吸（RESP）",
}
OLD_I4 = "outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_i4_r3"
NEW_I4 = "outputs/exploratory_oracle/phase_i/phase_i_cec_corrected_20_2p9_plus18_20261031_i4"
OLD_I6 = "outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_i6_r1"
NEW_I6 = "outputs/exploratory_oracle/phase_i/phase_i_cec_corrected_20_2p9_plus18_20261031_i6"
I1_CORRECTED = "outputs/exploratory_oracle/phase_i/phase_i_cec_corrected_20_2p9_plus18_20261031_i1"
I2_CORRECTED = "outputs/exploratory_oracle/phase_i/phase_i_cec_corrected_20_2p9_plus18_20261031_i2"
CEC_FORMULA = "CEC_corrected = (CEC - 20 × Erosion) / 2.9 + 18"
FONT = "WenQuanYi Zen Hei"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [FONT, "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True,
                              text=True).stdout.strip()
    except Exception:
        return None


def _version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _merge(old_path: Path, corrected_path: Path, *, keys: list[str], parquet: bool) -> pd.DataFrame:
    reader = pd.read_parquet if parquet else pd.read_csv
    old = reader(old_path)
    corrected = reader(corrected_path)
    if "outcome" not in old or "outcome" not in corrected:
        raise ValueError(f"Missing outcome column in {old_path} or {corrected_path}.")
    old = old.loc[old["outcome"].ne("CEC")].copy()
    corrected = corrected.loc[corrected["outcome"].eq("CEC")].copy()
    old["outcome_guided_selection"] = True
    corrected["outcome_guided_selection"] = True
    combined = pd.concat([old, corrected], ignore_index=True, sort=False)
    absent = [key for key in keys if key not in combined]
    if absent:
        raise ValueError(f"Missing merge keys {absent}: {old_path}, {corrected_path}.")
    duplicate = combined.duplicated(keys, keep=False)
    if duplicate.any():
        raise ValueError(f"Duplicate rows after explicit CEC merge on {keys}: {int(duplicate.sum())}.")
    present = set(combined["outcome"].astype(str).unique())
    if present != set(OUTCOMES):
        raise ValueError(f"Merged outcomes differ from expected nine: {sorted(present)}")
    return combined.sort_values(["outcome", *[k for k in keys if k != "outcome"]],
                                kind="stable").reset_index(drop=True)


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    y = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    w = pd.to_numeric(weights, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(y) & np.isfinite(w) & (w > 0)
    return float(np.average(y[valid], weights=w[valid]))


def _weighted_negative_ratio(values: pd.Series, weights: pd.Series) -> float:
    y = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    w = pd.to_numeric(weights, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(y) & np.isfinite(w) & (w > 0)
    return float(w[valid & (y < 0)].sum() / w[valid].sum())


def _summary_tables(i4: pd.DataFrame, spearman: pd.DataFrame, lisa: pd.DataFrame,
                    same_site: pd.DataFrame, gistar: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    env = i4.loc[(i4.analysis_population == "environment_adjusted")
                 & i4.model_family.isin(["GWR", "GWR-LCR"])].copy()
    rows: list[dict[str, Any]] = []
    for outcome in OUTCOMES:
        for model in ("GWR", "GWR-LCR"):
            d = env.loc[(env.outcome == outcome) & (env.model_family == model)]
            if d.empty:
                raise ValueError(f"No I4 local coefficients for {outcome}/{model}.")
            rows.append({
                "outcome": outcome,
                "model_family": model,
                "n_units": len(d),
                "n_supported_pixels": int(pd.to_numeric(d.n_pixels).sum()),
                "pixel_weighted_mean_local_standardized_beta": _weighted_mean(
                    d.local_beta_standardized, d.n_pixels),
                "pixel_weighted_negative_beta_fraction": _weighted_negative_ratio(
                    d.local_beta_standardized, d.n_pixels),
                "outcome_guided_selection": True,
            })
    gwr_summary = pd.DataFrame(rows)
    primary = gwr_summary.loc[gwr_summary.model_family.eq("GWR")].copy()
    primary = primary.sort_values("pixel_weighted_mean_local_standardized_beta", kind="stable")
    primary["negative_response_rank"] = np.arange(1, len(primary) + 1)
    rank = primary[["outcome", "negative_response_rank", "n_units", "n_supported_pixels",
                    "pixel_weighted_mean_local_standardized_beta",
                    "pixel_weighted_negative_beta_fraction"]].rename(columns={
                        "pixel_weighted_mean_local_standardized_beta": "GWR_weighted_mean_beta",
                        "pixel_weighted_negative_beta_fraction": "GWR_negative_pixel_support_fraction",
                    })
    lsum = spearman.groupby("outcome").agg(
        local_spearman_units=("local_spearman", "size"),
        local_spearman_median=("local_spearman", "median"),
        local_spearman_p10=("local_spearman", lambda x: x.quantile(.10)),
        local_spearman_p90=("local_spearman", lambda x: x.quantile(.90)),
        local_spearman_negative_fraction=("local_spearman", lambda x: float((x < 0).mean())),
    ).reset_index()
    hl = lisa.assign(is_HL=lisa.lisa_class.eq("HL")).groupby("outcome").agg(
        lisa_hl_fdr_count=("is_HL", "sum"), lisa_units=("is_HL", "size")
    ).reset_index()
    site = same_site.assign(is_high_low=same_site.same_site_class.eq("High Aa—Low health")).groupby("outcome").agg(
        same_site_high_aa_low_health_count=("is_high_low", "sum"), same_site_units=("is_high_low", "size")
    ).reset_index()
    site["same_site_high_aa_low_health_fraction"] = (
        site.same_site_high_aa_low_health_count / site.same_site_units
    )
    gi = gistar.groupby(["outcome", "model_family", "gi_star_class"]).size().unstack(fill_value=0)
    gi = gi.reset_index().rename(columns={"Hot spot": "gistar_hotspot_count", "Cold spot": "gistar_coldspot_count"})
    for col in ("gistar_hotspot_count", "gistar_coldspot_count"):
        if col not in gi:
            gi[col] = 0
    gi_wide = gi.pivot(index="outcome", columns="model_family",
                       values=["gistar_hotspot_count", "gistar_coldspot_count"])
    gi_wide.columns = [f"{metric}_{model.replace('-', '_')}" for metric, model in gi_wide.columns]
    gi_wide = gi_wide.reset_index()
    summaries = rank.merge(lsum, on="outcome", validate="one_to_one").merge(
        hl, on="outcome", validate="one_to_one").merge(site, on="outcome", validate="one_to_one").merge(
        gi_wide, on="outcome", validate="one_to_one")
    lcr = gwr_summary.loc[gwr_summary.model_family.eq("GWR-LCR"), [
        "outcome", "pixel_weighted_mean_local_standardized_beta",
        "pixel_weighted_negative_beta_fraction",
    ]].rename(columns={
        "pixel_weighted_mean_local_standardized_beta": "GWR_LCR_weighted_mean_beta",
        "pixel_weighted_negative_beta_fraction": "GWR_LCR_negative_pixel_support_fraction",
    })
    summaries = summaries.merge(lcr, on="outcome", validate="one_to_one")
    return gwr_summary, summaries


def _setup_axis(ax: plt.Axes, *, xcoords: np.ndarray | None = None,
                ycoords: np.ndarray | None = None, mask: np.ndarray | None = None) -> None:
    ax.set_xlabel("东向坐标（km，UTM 49N）", fontsize=8)
    ax.set_ylabel("北向坐标（km）", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(color="#dce4e9", linewidth=.35, alpha=.65)
    ax.set_aspect("equal", adjustable="datalim")
    if xcoords is not None and ycoords is not None and mask is not None:
        ax.contour(xcoords / 1000, ycoords / 1000, mask.astype(float), levels=[.5],
                   colors="#263746", linewidths=.4, zorder=5)


def _save_grid(name: str, frames: list[tuple[str, pd.DataFrame]], *, kind: str,
               mask_info: tuple[np.ndarray, np.ndarray, np.ndarray],
               gwr_means: dict[str, tuple[float, float]] | None = None) -> list[str]:
    mask, xcoords, ycoords = mask_info
    fig, axs = plt.subplots(3, 3, figsize=(18, 15), squeeze=False, constrained_layout=True)
    axs_flat = axs.ravel()
    scatter = None
    category_styles: dict[str, tuple[str, str]] = {}
    gwr_scale = None
    if kind == "gwr":
        values = np.concatenate([
            pd.to_numeric(frame.local_beta_standardized, errors="coerce").to_numpy(dtype=float)
            for _, frame in frames
        ])
        finite = values[np.isfinite(values)]
        gwr_scale = max(float(np.quantile(np.abs(finite), .98)), .05)
    if kind == "lisa":
        category_styles = {"Not significant": ("未达 FDR 显著", "#d9d9d9"),
                           "HH": ("本地高 Aa—邻域高健康", "#D73027"),
                           "LL": ("本地低 Aa—邻域低健康", "#4575B4"),
                           "HL": ("本地高 Aa—邻域低健康", "#F46D43"),
                           "LH": ("本地低 Aa—邻域高健康", "#74ADD1")}
    elif kind == "colocation":
        category_styles = {"Near mean": ("接近各自均值", "#d9d9d9"),
                           "High Aa—Low health": ("同位高侵蚀—低健康", "#D73027"),
                           "High Aa—High health": ("同位高侵蚀—高健康", "#F46D43"),
                           "Low Aa—Low health": ("同位低侵蚀—低健康", "#4575B4"),
                           "Low Aa—High health": ("同位低侵蚀—高健康", "#74ADD1")}
    else:
        category_styles = {"Not significant": ("未达 FDR 显著", "#d9d9d9"),
                           "Hot spot": ("负向系数强度热点", "#D73027"),
                           "Cold spot": ("负向系数强度冷点", "#4575B4")}

    for index, (outcome, frame) in enumerate(frames):
        ax = axs_flat[index]
        if kind == "gwr":
            value = pd.to_numeric(frame.local_beta_standardized, errors="coerce").to_numpy(dtype=float)
            scatter = ax.scatter(frame.x / 1000, frame.y / 1000, c=value, s=5, marker="s",
                                 linewidths=0, cmap="RdBu_r", vmin=-gwr_scale, vmax=gwr_scale,
                                 rasterized=True)
            if gwr_means and outcome in gwr_means:
                mean_beta, negative_support = gwr_means[outcome]
                ax.text(.025, .03, f"面积加权平均 β={mean_beta:+.3f}\n负系数像元支持={negative_support:.1%}",
                        transform=ax.transAxes, fontsize=8, color="#263746",
                        bbox={"facecolor": "white", "edgecolor": "#d5dde2", "alpha": .92, "pad": 3})
        elif kind == "spearman":
            values = pd.to_numeric(frame.local_spearman, errors="coerce").to_numpy(dtype=float)
            scatter = ax.scatter(frame.x / 1000, frame.y / 1000, c=values, s=5, marker="s",
                                 linewidths=0, cmap="RdBu_r", vmin=-1, vmax=1, rasterized=True)
        else:
            column = "lisa_class" if kind == "lisa" else "same_site_class" if kind == "colocation" else "gi_star_class"
            for category, (_, color) in category_styles.items():
                use = frame[column].eq(category)
                ax.scatter(frame.loc[use, "x"] / 1000, frame.loc[use, "y"] / 1000,
                           c=color, s=5, marker="s", linewidths=0, rasterized=True)
        _setup_axis(ax, xcoords=xcoords, ycoords=ycoords, mask=mask)
        ax.set_title(LABELS[outcome], fontsize=11, pad=6)

    if kind == "gistar":
        model_label = "GWR-LCR" if "gwr_lcr" in name else "GWR"
        title = f"九项健康结局的局部负向响应强度热点（{model_label}）"
    else:
        title = {"gwr": "九项健康结局的环境调整 GWR 局部系数",
                  "spearman": "九项健康结局与侵蚀的局部 Spearman 秩相关（100 近邻）",
                  "lisa": "九项健康结局的双变量局部空间关联（LISA）",
                  "colocation": "九项健康结局与侵蚀的同位高低共现"}[kind]
    fig.suptitle(title, fontsize=16)
    if kind in {"gwr", "spearman"}:
        fig.colorbar(scatter, ax=list(axs_flat), shrink=.76, pad=.015,
                     label="局部标准化系数 β（蓝负红正）" if kind == "gwr" else "局部 Spearman 秩相关 ρ（蓝负红正）")
    else:
        handles = [Line2D([0], [0], marker="s", linestyle="", color=color,
                          markersize=7, label=label)
                   for label, color in category_styles.values()]
        fig.legend(handles=handles, loc="outside lower center", ncol=min(5, len(handles)),
                   frameon=False, fontsize=9)
    (RUN_DIR / "figures").mkdir(parents=True, exist_ok=True)
    target = RUN_DIR / "figures" / name
    paths = []
    for ext in ("png", "svg", "pdf"):
        out = target.with_suffix(f".{ext}")
        fig.savefig(out, dpi=240 if ext == "png" else None, bbox_inches="tight")
        paths.append(out.relative_to(ROOT).as_posix())
    plt.close(fig)
    return paths


def _focus_report(summaries: pd.DataFrame, i1: pd.DataFrame, i2: pd.DataFrame) -> str:
    i1_focus = i1.loc[(i1.population == "mask_all") & (i1.exposure == "Aa_20yr")
                      & i1.outcome.isin(FOCUS)].set_index("outcome")
    i2_focus = i2.loc[(i2.candidate_id == "stage2_Aa_20yr_01")
                      & (i2.population == "all_erosion") & i2.outcome.isin(FOCUS)].set_index("outcome")
    gwr = summaries.set_index("outcome")
    lines = [
        "# Phase I 九项指标空间结果整合与重点指标分析", "",
        "本次整合保留 I4/I6 既有分析设定，只把 CEC 结局替换为公式校正后的结果；其余八项健康结局沿用原分析。Oracle 模式，`outcome_guided_selection=true`。",
        "",
        f"CEC 校正式：`{CEC_FORMULA}`；Erosion 为 Aa_20yr 与 Aa_2020 分别除以两张完整对齐栅格的全域最大值后求和。该变换把负向侵蚀项直接写入 CEC_corrected，因此 CEC 的负向系数存在公式机械关联，不能作为独立验证。",
        "",
        "## 九项指标的 GWR 负向强度排序", "",
        "排序按环境调整标准 GWR 局部标准化系数的 30 m 像元支持加权平均，由最负到最弱；这是一种探索性空间关联强度排序，不表示因果。SHI、SW、OM 是原方案预定重点，和数值前三的排序分开展示。",
        "",
        "| 排名 | 指标 | 加权平均局部 β | 负系数像元支持比例 | 约240 m单元 | 来源像元数 |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in summaries.sort_values("negative_response_rank").itertuples(index=False):
        outcome = row.outcome
        lines.append(f"| {row.negative_response_rank} | {LABELS[outcome]} | {row.GWR_weighted_mean_beta:+.3f} | {row.GWR_negative_pixel_support_fraction:.1%} | {row.n_units:,} | {row.n_supported_pixels:,} |")
    lines.extend(["", "数值前三为 SW、OM、BD；SHI 排名另列为预定综合指标重点。强度排序只描述所拟合模型的标准化系数，不等同于土壤健康价值或治理优先级。", "",
                  "## 预定重点：SHI、SW、OM", "",
                  "| 指标 | 全掩膜 Aa_20yr Spearman ρ | M4 匹配差值 Spearman ρ | M4 标准化配对斜率 | GWR 加权平均 β | GWR-LCR 加权平均 β | GWR 负向像元支持 | 局部ρ中位数（P10–P90） | LISA 显著 HL | 同位高侵蚀—低健康 | GWR / GWR-LCR 负向强度热点 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for outcome in FOCUS:
        a = i1_focus.loc[outcome]
        m = i2_focus.loc[outcome]
        row = gwr.loc[outcome]
        same_n = row.same_site_high_aa_low_health_count
        same_p = row.same_site_high_aa_low_health_fraction
        lines.append(
            f"| {LABELS[outcome]} | {a.spearman_rho:+.3f} | {m.spearman_rho:+.3f} | {m.standardized_slope_beta:+.3f} | "
            f"{row.GWR_weighted_mean_beta:+.3f} | {row.GWR_LCR_weighted_mean_beta:+.3f} | "
            f"{row.GWR_negative_pixel_support_fraction:.1%} | {row.local_spearman_median:+.3f}（{row.local_spearman_p10:+.3f}–{row.local_spearman_p90:+.3f}） | "
            f"{int(row.lisa_hl_fdr_count):,} | {int(same_n):,}（{same_p:.1%}） | "
            f"{int(row.gistar_hotspot_count_GWR):,} / {int(row.gistar_hotspot_count_GWR_LCR):,} |"
        )
    lines.extend([
        "",
        "### 怎么理解三项重点", "",
        "- **SHI** 是综合多项土壤健康指标形成的总体指数。把全区相关、匹配差值、局部 GWR、局部秩相关和空间聚集一起看，可以判断总体关系是否在不同统计视角下保持方向；GWR 系数接近 0 的区域也应与局部相关图对照。",
        "- **SW** 是土壤含水量。若全局、匹配和局部 GWR 均为负向且大多数像元支持也为负，说明空间预测面中这一方向相对一致；但空间重叠、共同环境结构和预测栅格属性仍限制独立推断。",
        "- **OM** 是有机质。与 SHI、SW 一样，分别比较全局和环境匹配后的配对差值，再看局部系数与高侵蚀—低健康空间位置是否重合；匹配后效应减弱时，应解释为条件关联变弱，而不是分析失败。",
        "- 三种空间图回答的问题不同：Local Spearman 是局部邻域内的秩关系；LISA 的 HL 是本地侵蚀高、邻域健康偏低且经 FDR 标记的空间关联；同位图比较同一单元；Gi* 检查 GWR 负向系数强度是否空间聚集。不可互相代替。",
        "",
        "## 方法和限制", "",
        "- I4 图为环境调整 GWR；色标以零为中心，所有九项共用尺度。负系数像元支持比例按每个 240 m 汇总单元包含的 30 m 来源像元数加权。GWR-LCR 是局部岭收缩敏感性分析，系数向零收缩。",
        "- I6 Local Spearman 使用 100 个最近空间单元的重叠邻域；只作描述，不给局部 p 值。双变量 LISA 使用行标准化 8 近邻、999 次随机置换和每个结局内 BH-FDR。Gi* 对 max(−β,0) 使用 1 km 邻域，正态近似 p 并按结局和模型做 BH-FDR。",
        "- 局部分析单元约 240 m；预测栅格像元和汇总单元不是独立实测样本。Oracle 结果为探索性描述，不能当作独立验证或因果效应。",
        "- 校正 CEC 的负向关联含有公式写入的侵蚀项。不得用它声称侵蚀降低了真实实测 CEC，也不能由模型结果反向验证校正式。",
        "",
        "完整表格在 `tables/`，制图数据来自整合后的表；原始运行与旧 CEC 输出保留在原目录，仅作为历史记录。", "",
    ])
    return "\n".join(lines)


def run() -> Path:
    if RUN_DIR.exists():
        raise FileExistsError(f"Run directory already exists: {RUN_DIR}")
    for sub in ("tables", "figures", "reports", "configs", "manifests"):
        (RUN_DIR / sub).mkdir(parents=True, exist_ok=True)

    i4 = _merge(ROOT / OLD_I4 / "tables/full_sample_local_coefficients.parquet",
                ROOT / NEW_I4 / "tables/full_sample_local_coefficients.parquet",
                keys=["outcome", "analysis_population", "model_family", "tile_row", "tile_col"], parquet=True)
    i6_spec = _merge(ROOT / OLD_I6 / "tables/local_spearman.parquet",
                     ROOT / NEW_I6 / "tables/local_spearman.parquet",
                     keys=["outcome", "tile_row", "tile_col"], parquet=True)
    i6_lisa = _merge(ROOT / OLD_I6 / "tables/bivariate_local_moran.parquet",
                     ROOT / NEW_I6 / "tables/bivariate_local_moran.parquet",
                     keys=["outcome", "tile_row", "tile_col"], parquet=True)
    i6_site = _merge(ROOT / OLD_I6 / "tables/same_site_erosion_health_colocation.csv",
                     ROOT / NEW_I6 / "tables/same_site_erosion_health_colocation.csv",
                     keys=["outcome", "tile_row", "tile_col"], parquet=False)
    i6_gi = _merge(ROOT / OLD_I6 / "tables/local_negative_response_gistar.parquet",
                   ROOT / NEW_I6 / "tables/local_negative_response_gistar.parquet",
                   keys=["outcome", "model_family", "analysis_population", "x", "y"], parquet=True)

    i4.to_parquet(RUN_DIR / "tables/full_sample_local_coefficients.parquet", index=False)
    i6_spec.to_parquet(RUN_DIR / "tables/local_spearman.parquet", index=False)
    i6_lisa.to_parquet(RUN_DIR / "tables/bivariate_local_moran.parquet", index=False)
    i6_site.to_csv(RUN_DIR / "tables/same_site_erosion_health_colocation.csv", index=False)
    i6_gi.to_parquet(RUN_DIR / "tables/local_negative_response_gistar.parquet", index=False)

    env = i4.loc[(i4.analysis_population == "environment_adjusted")
                 & i4.model_family.isin(["GWR", "GWR-LCR"])].copy()
    gwr_summary, summary = _summary_tables(i4, i6_spec, i6_lisa, i6_site, i6_gi)
    gwr_summary.to_csv(RUN_DIR / "tables/gwr_weighted_effect_summary.csv", index=False)
    summary.to_csv(RUN_DIR / "tables/nine_outcome_spatial_summary_and_rank.csv", index=False)

    # The map boundary comes from the aligned cropland table, using coordinates only.
    env_path = ROOT / "outputs/run_20261025T114741_470772Z/databases/environment_erosion.parquet"
    geometry = pd.read_parquet(env_path, columns=["row", "col", "x", "y"])
    mask_info = spatial._mask_arrays(geometry)
    means = {}
    for outcome in OUTCOMES:
        d = env.loc[(env.outcome == outcome) & env.model_family.eq("GWR")]
        means[outcome] = (_weighted_mean(d.local_beta_standardized, d.n_pixels),
                          _weighted_negative_ratio(d.local_beta_standardized, d.n_pixels))
    maps: list[str] = []
    gwr_frames = [(outcome, env.loc[(env.outcome == outcome) & env.model_family.eq("GWR")])
                  for outcome in OUTCOMES]
    maps += _save_grid("nine_outcome_gwr_coefficients", gwr_frames, kind="gwr", mask_info=mask_info, gwr_means=means)
    maps += _save_grid("nine_outcome_local_spearman", [(o, i6_spec.loc[i6_spec.outcome.eq(o)]) for o in OUTCOMES],
                       kind="spearman", mask_info=mask_info)
    maps += _save_grid("nine_outcome_bivariate_lisa", [(o, i6_lisa.loc[i6_lisa.outcome.eq(o)]) for o in OUTCOMES],
                       kind="lisa", mask_info=mask_info)
    maps += _save_grid("nine_outcome_same_site_colocation", [(o, i6_site.loc[i6_site.outcome.eq(o)]) for o in OUTCOMES],
                       kind="colocation", mask_info=mask_info)
    for model in ("GWR", "GWR-LCR"):
        maps += _save_grid(f"nine_outcome_gistar_{model.lower().replace('-', '_')}",
                           [(o, i6_gi.loc[(i6_gi.outcome.eq(o)) & i6_gi.model_family.eq(model)]) for o in OUTCOMES],
                           kind="gistar", mask_info=mask_info)

    i1 = pd.read_csv(ROOT / I1_CORRECTED / "results/global_response/global_associations.csv")
    i2 = pd.read_csv(ROOT / I2_CORRECTED / "tables/matched_pair_associations.csv")
    i1.to_csv(RUN_DIR / "tables/corrected_i1_global_associations.csv", index=False)
    i2.to_csv(RUN_DIR / "tables/corrected_i2_matched_pair_associations.csv", index=False)
    (RUN_DIR / "reports/PHASE_I_NINE_OUTCOME_SPATIAL_REPORT.md").write_text(
        _focus_report(summary, i1, i2), encoding="utf-8")

    config = {
        "phase": "Phase I-4/I-6 corrected nine-outcome consolidation",
        "mode": "EXPLORATORY_ORACLE",
        "outcome_guided_selection": True,
        "run_id": RUN_ID,
        "outcomes": OUTCOMES,
        "predesignated_focus_outcomes": FOCUS,
        "focus_reason": "Existing Phase I configuration designated SHI, OM, SW before this refresh; numeric ranking reported separately.",
        "cec_formula": CEC_FORMULA,
        "cec_erosion_definition": "Aa_20yr/max_full_valid_Aa_20yr + Aa_2020/max_full_valid_Aa_2020",
        "sources": {"i4_non_cec": OLD_I4, "i4_corrected_cec": NEW_I4,
                    "i6_non_cec": OLD_I6, "i6_corrected_cec": NEW_I6,
                    "i1_global_corrected": I1_CORRECTED, "i2_matching_corrected": I2_CORRECTED},
        "merge_keys": {"i4": ["outcome", "analysis_population", "model_family", "tile_row", "tile_col"],
                       "i6_local": ["outcome", "tile_row", "tile_col"],
                       "i6_gistar": ["outcome", "model_family", "analysis_population", "x", "y"]},
        "i6_parameters": {"local_spearman_neighbors": 100, "lisa_neighbors": 8,
                           "lisa_permutations": 999, "gi_star_radius_m": 1000},
        "rank_definition": "30 m pixel-support-weighted mean of environment-adjusted standard GWR local standardized beta; more negative ranks higher.",
        "map_figures": maps,
    }
    (RUN_DIR / "configs/refresh_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    provenance = {
        **config,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_status_porcelain": _git("status", "--short"),
        "input_row_counts": {"i4": len(i4), "local_spearman": len(i6_spec), "lisa": len(i6_lisa),
                             "same_site": len(i6_site), "gistar": len(i6_gi)},
        "outcome_counts": {"i4_environment_adjusted_gwr": {
                               f"{outcome}/{model}": int(count)
                               for (outcome, model), count in env.groupby(["outcome", "model_family"]).size().items()},
                           "local_spearman": i6_spec.groupby("outcome").size().to_dict(),
                           "lisa": i6_lisa.groupby("outcome").size().to_dict(),
                           "same_site": i6_site.groupby("outcome").size().to_dict(),
                           "gistar": {f"{outcome}/{model}": int(count)
                                      for (outcome, model), count in i6_gi.groupby(["outcome", "model_family"]).size().items()}},
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "matplotlib": _version("matplotlib"), "scipy": _version("scipy")},
        "note": "Existing non-CEC output rows are reused; corrected CEC rows replace old CEC using explicit outcome/tile keys. No model refitting is performed in this consolidation.",
    }
    (RUN_DIR / "manifests/provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (RUN_DIR / "RUN_STATUS.md").write_text(
        "# Run status\n\nStatus: consolidated corrected nine-outcome exploratory result.\n\n"
        "- Mode: `EXPLORATORY_ORACLE`\n- `outcome_guided_selection=true`\n"
        "- Existing SHI/SW/OM focus retained; the numeric nine-outcome ranking is reported separately.\n"
        "- CEC values in this run use the formula recorded in `configs/refresh_config.json`.\n"
        "- Source analyses were merged by explicit outcome and spatial-unit keys; no GWR, LISA, or Gi* model was refit.\n",
        encoding="utf-8")
    return RUN_DIR


if __name__ == "__main__":
    print(run().relative_to(ROOT))
