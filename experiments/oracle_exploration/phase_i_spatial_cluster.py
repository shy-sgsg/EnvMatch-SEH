"""Phase I-6 local association and spatial hotspot exploration."""

from __future__ import annotations

import argparse
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
import scipy
import yaml
from matplotlib.lines import Line2D
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from scipy.stats import norm, rankdata

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase_i_spatial as spatial  # noqa: E402

OUTCOMES = ["SHI_candidate", "AN", "OM", "CEC", "pH", "SW", "BD", "MBC", "RESP"]
KEY_OUTCOMES = ["SHI_candidate", "OM", "SW"]
LABELS = {"SHI_candidate": "SHI", "AN": "AN", "OM": "OM", "CEC": "CEC", "pH": "pH",
          "SW": "SW", "BD": "BD", "MBC": "MBC", "RESP": "RESP"}
LISA_COLORS = {"Not significant": "#d9d9d9", "HH": "#D73027", "LL": "#4575B4",
               "HL": "#F46D43", "LH": "#74ADD1"}
COLOCATION_COLORS = {"Near mean": "#d9d9d9", "High Aa—Low health": "#D73027",
                      "High Aa—High health": "#F46D43", "Low Aa—Low health": "#4575B4",
                      "Low Aa—High health": "#74ADD1"}
HOTSPOT_COLORS = {"Not significant": "#d9d9d9", "Hot spot": "#D73027", "Cold spot": "#4575B4"}
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


def _bh_qvalues(pvalues: np.ndarray) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    q = np.full(p.shape, np.nan, dtype=float)
    valid = np.isfinite(p)
    if not valid.any():
        return q
    vals = p[valid]
    order = np.argsort(vals)
    ranked = vals[order] * len(vals) / np.arange(1, len(vals) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    restored = np.empty_like(ranked)
    restored[order] = np.clip(ranked, 0.0, 1.0)
    q[valid] = restored
    return q


def _knn_weights(coords: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, csr_matrix]:
    n = len(coords)
    if n <= k:
        raise ValueError(f"Need more than {k} spatial units; found {n}.")
    distances, neighbors = cKDTree(coords).query(coords, k=k + 1)
    distances, neighbors = distances[:, 1:], neighbors[:, 1:]
    rows = np.repeat(np.arange(n), k)
    weights = np.full(n * k, 1.0 / k, dtype=float)
    matrix = csr_matrix((weights, (rows, neighbors.reshape(-1))), shape=(n, n))
    return distances[:, -1], neighbors, matrix


def _local_spearman(x: np.ndarray, y: np.ndarray, neighbors: np.ndarray) -> np.ndarray:
    xr = rankdata(x[neighbors], axis=1)
    yr = rankdata(y[neighbors], axis=1)
    xr -= xr.mean(axis=1, keepdims=True)
    yr -= yr.mean(axis=1, keepdims=True)
    numerator = np.einsum("ij,ij->i", xr, yr)
    denominator = np.sqrt(np.einsum("ij,ij->i", xr, xr) * np.einsum("ij,ij->i", yr, yr))
    rho = np.full(len(x), np.nan, dtype=float)
    np.divide(numerator, denominator, out=rho, where=denominator > 0)
    return np.clip(rho, -1, 1)


def _same_site_classes(zx: np.ndarray, zy: np.ndarray) -> np.ndarray:
    result = np.full(len(zx), "Near mean", dtype=object)
    high_x, high_y = zx > 0, zy > 0
    result[high_x & ~high_y] = "High Aa—Low health"
    result[high_x & high_y] = "High Aa—High health"
    result[~high_x & ~high_y] = "Low Aa—Low health"
    result[~high_x & high_y] = "Low Aa—High health"
    return result


def _bivariate_lisa(x: np.ndarray, y: np.ndarray, weights: csr_matrix,
                    permutations: int, seed: int, alpha: float) -> pd.DataFrame:
    zx = (x - x.mean()) / max(x.std(ddof=0), 1e-12)
    zy = (y - y.mean()) / max(y.std(ddof=0), 1e-12)
    lag_y = np.asarray(weights @ zy).ravel()
    local_i = zx * lag_y
    rng = np.random.default_rng(seed)
    n = len(y)
    permuted_indices = np.column_stack([rng.permutation(n) for _ in range(permutations)])
    lag_permuted = weights @ zy[permuted_indices]
    i_permuted = zx[:, None] * lag_permuted
    pvalues = (1 + np.count_nonzero(np.abs(i_permuted) >= np.abs(local_i)[:, None], axis=1)) / (permutations + 1)
    qvalues = _bh_qvalues(pvalues)
    significant = qvalues < alpha
    classes = np.full(n, "Not significant", dtype=object)
    classes[significant & (zx > 0) & (lag_y > 0)] = "HH"
    classes[significant & (zx < 0) & (lag_y < 0)] = "LL"
    classes[significant & (zx > 0) & (lag_y < 0)] = "HL"
    classes[significant & (zx < 0) & (lag_y > 0)] = "LH"
    return pd.DataFrame({"z_Aa": zx, "lag_z_health": lag_y, "bivariate_local_moran_i": local_i,
                         "p_global_permutation_two_sided": pvalues, "q_bh_within_outcome": qvalues,
                         "lisa_class": classes, "same_site_class": _same_site_classes(zx, zy),
                         "outcome_guided_selection": True})


def _gi_star(values: np.ndarray, coords: np.ndarray, radius_m: float,
             alpha: float) -> tuple[pd.DataFrame, np.ndarray]:
    n = len(values)
    tree = cKDTree(coords)
    neighborhood = tree.query_ball_point(coords, r=radius_m)
    row_sizes = np.fromiter((len(row) for row in neighborhood), dtype=int, count=n)
    rows = np.repeat(np.arange(n), row_sizes)
    cols = np.concatenate([np.asarray(row, dtype=int) for row in neighborhood])
    weights = csr_matrix((np.ones(len(rows), dtype=float), (rows, cols)), shape=(n, n))
    sum_w = np.asarray(weights.sum(axis=1)).ravel()
    sum_w2 = np.asarray(weights.multiply(weights).sum(axis=1)).ravel()
    mean = float(np.mean(values))
    sd = float(np.sqrt(np.mean(values ** 2) - mean ** 2))
    numerator = np.asarray(weights @ values).ravel() - mean * sum_w
    variance_factor = (n * sum_w2 - sum_w ** 2) / max(n - 1, 1)
    denominator = sd * np.sqrt(np.maximum(variance_factor, 0.0))
    z = np.zeros(n, dtype=float)
    np.divide(numerator, denominator, out=z, where=denominator > 0)
    p = np.minimum(1.0, 2.0 * norm.sf(np.abs(z)))
    q = _bh_qvalues(p)
    status = np.full(n, "Not significant", dtype=object)
    status[(z > 0) & (q < alpha)] = "Hot spot"
    status[(z < 0) & (q < alpha)] = "Cold spot"
    result = pd.DataFrame({"local_Gi_star_z": z, "p_normal_approx_two_sided": p,
                           "q_bh_within_outcome_and_model": q, "gi_star_class": status,
                           "neighbors_including_self_within_radius": row_sizes,
                           "outcome_guided_selection": True})
    return result, row_sizes


def _save_map_grid(run_dir: Path, name: str, frames: list[tuple[str, pd.DataFrame, str]],
                   mask_info: tuple[np.ndarray, np.ndarray, np.ndarray], dpi: int,
                   value_kind: str, title: str) -> list[str]:
    mask, xcoords, ycoords = mask_info
    fig, axes = plt.subplots(3, 3, figsize=(17, 13.5), constrained_layout=True)
    mappable = None
    palette = LISA_COLORS if value_kind == "lisa" else COLOCATION_COLORS if value_kind == "colocation" else HOTSPOT_COLORS
    for panel, (ax, (outcome, frame, value_col)) in enumerate(zip(axes.ravel(), frames, strict=True)):
        if value_kind == "continuous":
            values = pd.to_numeric(frame[value_col], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(values)
            limit = 1.0 if value_col == "local_spearman" else 4.0
            mappable = ax.scatter(frame.loc[valid, "x"] / 1000, frame.loc[valid, "y"] / 1000,
                                  c=values[valid], s=7, marker="s", linewidths=0,
                                  cmap="RdBu_r", vmin=-limit, vmax=limit, rasterized=True)
        else:
            for category, color in palette.items():
                use = frame[value_col].eq(category)
                ax.scatter(frame.loc[use, "x"] / 1000, frame.loc[use, "y"] / 1000,
                           c=color, s=7, marker="s", linewidths=0, rasterized=True)
        spatial._add_map_decor(ax, mask, xcoords, ycoords)
        ax.set_title(f"({chr(97 + panel)}) {LABELS[outcome]}", fontsize=11)
        ax.set_xlabel("Easting (km, EPSG:32649)")
        ax.set_ylabel("Northing (km)")
        ax.set_aspect("equal", adjustable="datalim")
    fig.suptitle(title, fontsize=15)
    if value_kind == "continuous":
        fig.colorbar(mappable, ax=list(axes.ravel()), shrink=.70,
                     label="Local Spearman ρ" if frames[0][2] == "local_spearman" else "Getis–Ord Gi* z")
    else:
        handles = [Line2D([0], [0], marker="s", linestyle="", color=color, markersize=7, label=label)
                   for label, color in palette.items()]
        fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False)
    paths: list[str] = []
    base = run_dir / "figures" / name
    base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg", "pdf"):
        out = base.with_suffix(f".{ext}")
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        paths.append(str(out.relative_to(ROOT)))
    plt.close(fig)
    return paths


def _report(run_dir: Path, local_summary: pd.DataFrame, lisa_summary: pd.DataFrame,
            same_site_summary: pd.DataFrame, gi_summary: pd.DataFrame,
            n_env_pixels: int, local_k: int, local_radius_median: float,
            n_units_by_outcome: dict[str, int], config: dict[str, Any]) -> str:
    lines = ["# Phase I-6：局部空间关联与负向响应热点", "",
             f"权威运行：`{run_dir.relative_to(ROOT)}`。运行模式为 `EXPLORATORY_ORACLE`，`outcome_guided_selection=true`。", "",
             "## 总体与空间尺度", "",
             f"- 输入为对齐库 392,527 个耕地掩膜像元；Aa 与九项环境协变量共同完整的像元为 {n_env_pixels:,}。主暴露为预计算 Aa_20yr 的 log1p，未重建/验证 RUSLE 因子乘积。",
             f"- 每项结局分别按 8×8 个 30 m 像元汇总为约 240 m 空间单元；每项局部分析单元数为 {min(n_units_by_outcome.values()):,}–{max(n_units_by_outcome.values()):,}。局部 Spearman 使用 {local_k} 个最近空间单元，距离中位数 {local_radius_median/1000:.2f} km；窗口相互重叠，因此只作描述性局部关联，不报告局部名义 p 值。",
             f"- Bivariate Local Moran 使用行标准化 {int(config['bivariate_lisa_neighbors'])} 近邻，健康值在邻居间空间滞后；按全局随机置换 {int(config['bivariate_lisa_permutations'])} 次计算双侧伪 p，并在每个结局内做 BH FDR。",
             f"- Gi* 基于 I-4 已保存的环境调整 GWR 局部标准化 Aa 系数，定义负向响应强度为 max(−β, 0)，对非负系数置零而不反向计为胁迫；以二元 1 km 距离邻域并包含焦点单元计算 Gi*，正态近似双侧 p 后按结局/模型分别做 BH FDR。GWR-LCR 作为敏感性层一并汇总。", "",
             "## Local Spearman", "",
             "| 指标 | 单元数 | 局部 ρ 中位数 | P10–P90 | 负相关单元占比 |", "|---|---:|---:|---:|---:|"]
    for r in local_summary.itertuples(index=False):
        lines.append(f"| {LABELS[r.outcome]} | {r.n_units:,} | {r.rho_median:.3f} | {r.rho_p10:.3f}–{r.rho_p90:.3f} | {r.negative_ratio:.1%} |")
    lines.extend(["", "## Bivariate Local Moran's I", "",
                  "分类中的 HH/LL/HL/LH 依次表示焦点位置 Aa 高/低与邻域健康空间滞后高/低；HL 表示高侵蚀位置邻域健康偏低。该分类来自空间滞后关联，不表示焦点像元自身健康低。", "",
                  "| 指标 | FDR 显著 HL 单元 | FDR 显著 HH | FDR 显著 LL | FDR 显著 LH |", "|---|---:|---:|---:|---:|"])
    for outcome in OUTCOMES:
        d = lisa_summary[lisa_summary.outcome.eq(outcome)].set_index("lisa_class").cell_count
        lines.append(f"| {LABELS[outcome]} | {int(d.get('HL',0)):,} | {int(d.get('HH',0)):,} | {int(d.get('LL',0)):,} | {int(d.get('LH',0)):,} |")
    lines.extend(["", "## 同一位置的高 Aa—低健康共现", "",
                  "下表按同一空间单元的 Aa 和健康标准化值正负分类，属于共现描述，不是 Bivariate Local Moran 的空间滞后结果，也不表示局部显著性。", "",
                  "| 指标 | 同位高 Aa—低健康单元 | 占分析单元比例 |", "|---|---:|---:|"])
    for r in same_site_summary.itertuples(index=False):
        lines.append(f"| {LABELS[r.outcome]} | {r.high_Aa_low_health_count:,} | {r.high_Aa_low_health_ratio:.1%} |")
    lines.extend(["", "## 局部负向响应 Gi* 热点", "",
                  "| 指标 | 局部模型 | 高值热点（FDR q<0.05） | 低值冷点（FDR q<0.05） |", "|---|---|---:|---:|"])
    for r in gi_summary.itertuples(index=False):
        lines.append(f"| {LABELS[r.outcome]} | {r.model_family} | {r.hotspot_count:,} | {r.coldspot_count:,} |")
    gi_lookup = gi_summary.set_index(["outcome", "model_family"])
    lines.append("")
    for outcome in KEY_OUTCOMES:
        gwr = int(gi_lookup.loc[(outcome, "GWR"), "hotspot_count"])
        lcr = int(gi_lookup.loc[(outcome, "GWR-LCR"), "hotspot_count"])
        lines.append(f"- **{LABELS[outcome]}** 的高值热点在 GWR/GWR-LCR 分别有 {gwr:,}/{lcr:,} 个单元；该差异显示热点范围对局部共线性收缩方案敏感，不能当作稳定的区域分类。")
    lines.extend(["", "## 解释限制", "",
                  "- Local Spearman、LISA 和 Gi* 都基于约 240 m 汇总单元；地图显示局部空间统计量，不能解释为 30 m 独立结果。健康指标为随机森林预测面，像元数不等于独立实测样本数。",
                  "- LISA 伪 p 使用全局随机置换，未保留健康变量的局部空间结构；Gi* p 值使用正态近似。BH 控制也不消除空间依赖或 Oracle 筛选造成的不确定性，显著标签仅作为探索性定位线索。",
                  "- 同位高 Aa—低健康分类、Bivariate LISA 的高 Aa—低邻域健康类别及 Gi* 负向响应强度热点是三个不同统计对象；不得互换解释。环境匹配与既有 PCNM 结果另见 I-2/I-3 和后续 I-7 整合。",
                  "- 本轮复用 I-4 GWR 系数，不重拟合空间响应模型；继续使用预计算 Aa，不重算或验证 RUSLE 因子乘积、不重跑 PCNM，也未修改研究总纲。",
                  "", "结果表见 `tables/`，图件见 `figures/`，输入和方法记录见 `manifests/provenance.json`。", ""])
    path = run_dir / "reports" / "PHASE_I_SPATIAL_CLUSTER_REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path.relative_to(ROOT))


def run(config_path: Path) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("mode") != "EXPLORATORY_ORACLE" or config.get("outcome_guided_selection") is not True:
        raise ValueError("Phase I-6 must be explicitly marked EXPLORATORY_ORACLE with outcome_guided_selection=true.")
    run_dir = ROOT / config["output_root"] / config["run_id"]
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    for sub in ("configs", "tables", "figures", "reports", "manifests"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    (run_dir / "configs" / config_path.name).write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")

    pixels, env_path, health_path = spatial._read_pixels({
        "input_run_id": config["input_run_id"], "primary_exposure": config["primary_exposure"],
        "environment_covariates": config["environment_covariates"], "outcomes": config["outcomes"]})
    env_columns = ["log1p_Aa", *config["environment_covariates"]]
    env_complete = np.isfinite(pixels[env_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)).all(axis=1)
    env_pixels = pixels.loc[env_complete].copy()
    mask_info = spatial._mask_arrays(pixels)
    k_spearman = int(config["local_spearman_neighbors"])
    k_lisa = int(config["bivariate_lisa_neighbors"])
    n_permutations = int(config["bivariate_lisa_permutations"])
    alpha = float(config["fdr_alpha"])
    local_rows: list[pd.DataFrame] = []
    lisa_rows: list[pd.DataFrame] = []
    same_site_rows: list[pd.DataFrame] = []
    local_summary_rows: list[dict[str, Any]] = []
    lisa_summary_rows: list[dict[str, Any]] = []
    same_site_summary_rows: list[dict[str, Any]] = []
    unit_counts: dict[str, int] = {}
    radius_values: list[float] = []
    map_local: list[tuple[str, pd.DataFrame, str]] = []
    map_lisa: list[tuple[str, pd.DataFrame, str]] = []
    map_colocation: list[tuple[str, pd.DataFrame, str]] = []

    for index, outcome in enumerate(config["outcomes"]):
        units = spatial._tile_pixels(env_pixels, outcome, ["log1p_Aa"], int(config["tile_cells"]))
        n = len(units)
        unit_counts[outcome] = n
        coords = units[["x", "y"]].to_numpy(dtype=float)
        x = units["log1p_Aa"].to_numpy(dtype=float)
        y = units[outcome].to_numpy(dtype=float)
        tree = cKDTree(coords)
        distances, neighbors_spearman = tree.query(coords, k=min(k_spearman, n))
        if neighbors_spearman.ndim == 1:
            neighbors_spearman = neighbors_spearman[:, None]
            distances = distances[:, None]
        rho = _local_spearman(x, y, neighbors_spearman)
        local = units[["tile_row", "tile_col", "x", "y", "n_pixels"]].copy()
        local["outcome"] = outcome
        local["local_spearman"] = rho
        local["n_neighbors"] = neighbors_spearman.shape[1]
        local["radius_to_farthest_neighbor_m"] = distances[:, -1]
        local["outcome_guided_selection"] = True
        local_rows.append(local)
        map_local.append((outcome, local, "local_spearman"))
        radius_values.extend(distances[:, -1].tolist())
        local_summary_rows.append({"outcome": outcome, "n_units": n, "rho_median": float(np.nanmedian(rho)),
                                   "rho_p10": float(np.nanquantile(rho, .1)), "rho_p90": float(np.nanquantile(rho, .9)),
                                   "negative_ratio": float(np.nanmean(rho < 0)),
                                   "median_neighborhood_radius_m": float(np.nanmedian(distances[:, -1])),
                                   "outcome_guided_selection": True})

        radius_knn, lisa_neighbors, weights = _knn_weights(coords, k_lisa)
        lisa = _bivariate_lisa(x, y, weights, n_permutations,
                               int(config["bivariate_lisa_seed"]) + index, alpha)
        lisa.insert(0, "tile_col", units["tile_col"].to_numpy())
        lisa.insert(0, "tile_row", units["tile_row"].to_numpy())
        lisa.insert(0, "n_pixels", units["n_pixels"].to_numpy())
        lisa.insert(0, "y", units["y"].to_numpy())
        lisa.insert(0, "x", units["x"].to_numpy())
        lisa.insert(0, "outcome", outcome)
        lisa["n_neighbors"] = k_lisa
        lisa["radius_to_farthest_neighbor_m"] = radius_knn
        lisa_rows.append(lisa)
        map_lisa.append((outcome, lisa, "lisa_class"))
        same_site_rows.append(lisa[["outcome", "x", "y", "tile_row", "tile_col", "same_site_class",
                                    "z_Aa", "outcome_guided_selection"]].copy())
        map_colocation.append((outcome, lisa, "same_site_class"))
        vc = lisa["lisa_class"].value_counts()
        for category in ["HH", "LL", "HL", "LH", "Not significant"]:
            lisa_summary_rows.append({"outcome": outcome, "lisa_class": category,
                                      "cell_count": int(vc.get(category, 0)), "n_units": n,
                                      "outcome_guided_selection": True})
        high_low = int(lisa["same_site_class"].eq("High Aa—Low health").sum())
        same_site_summary_rows.append({"outcome": outcome, "high_Aa_low_health_count": high_low,
                                       "high_Aa_low_health_ratio": high_low / n,
                                       "n_units": n, "outcome_guided_selection": True})
        print(f"I-6 Local Spearman and Bivariate LISA {outcome}: {n:,} units", flush=True)

    local_df = pd.concat(local_rows, ignore_index=True)
    lisa_df = pd.concat(lisa_rows, ignore_index=True)
    same_site_df = pd.concat(same_site_rows, ignore_index=True)
    local_summary = pd.DataFrame(local_summary_rows)
    lisa_summary = pd.DataFrame(lisa_summary_rows)
    same_site_summary = pd.DataFrame(same_site_summary_rows)
    local_df.to_parquet(run_dir / "tables" / "local_spearman.parquet", index=False)
    lisa_df.to_parquet(run_dir / "tables" / "bivariate_local_moran.parquet", index=False)
    same_site_df.to_csv(run_dir / "tables" / "same_site_erosion_health_colocation.csv", index=False)
    local_summary.to_csv(run_dir / "tables" / "local_spearman_summary.csv", index=False)
    lisa_summary.to_csv(run_dir / "tables" / "bivariate_local_moran_cluster_counts.csv", index=False)
    same_site_summary.to_csv(run_dir / "tables" / "same_site_colocation_summary.csv", index=False)

    i4_path = ROOT / config["i4_run_id"] / "tables" / "full_sample_local_coefficients.parquet"
    i4 = pd.read_parquet(i4_path)
    gi_rows: list[pd.DataFrame] = []
    gi_summary_rows: list[dict[str, Any]] = []
    map_gi_by_model: dict[str, list[tuple[str, pd.DataFrame, str]]] = {"GWR": [], "GWR-LCR": []}
    for oi, outcome in enumerate(config["outcomes"]):
        for model_family in ("GWR", "GWR-LCR"):
            d = i4.loc[i4["outcome"].eq(outcome)
                       & i4["analysis_population"].eq("environment_adjusted")
                       & i4["model_family"].eq(model_family)].copy()
            if d.empty:
                continue
            beta = pd.to_numeric(d["local_beta_standardized"], errors="coerce").to_numpy(dtype=float)
            finite = np.isfinite(beta)
            d = d.loc[finite].copy()
            beta = beta[finite]
            coords = d[["x", "y"]].to_numpy(dtype=float)
            stress = np.where(beta < 0, -beta, 0.0)
            gi, neighbors = _gi_star(stress, coords, float(config["gi_star_radius_m"]), alpha)
            gi.insert(0, "analysis_population", "environment_adjusted")
            gi.insert(0, "model_family", model_family)
            gi.insert(0, "outcome", outcome)
            gi.insert(0, "n_pixels", d["n_pixels"].to_numpy())
            gi.insert(0, "y", d["y"].to_numpy())
            gi.insert(0, "x", d["x"].to_numpy())
            gi["local_beta_standardized"] = beta
            gi["negative_response_strength"] = stress
            gi["outcome_guided_selection"] = True
            gi_rows.append(gi)
            map_gi_by_model[model_family].append((outcome, gi, "gi_star_class"))
            counts = gi["gi_star_class"].value_counts()
            gi_summary_rows.append({"outcome": outcome, "model_family": model_family, "n_units": len(gi),
                                   "hotspot_count": int(counts.get("Hot spot", 0)),
                                   "coldspot_count": int(counts.get("Cold spot", 0)),
                                   "nonnegative_beta_count": int((beta >= 0).sum()),
                                   "outcome_guided_selection": True})
        print(f"I-6 Gi* negative-response hotspots {outcome}: complete", flush=True)
    gi_df = pd.concat(gi_rows, ignore_index=True)
    gi_summary = pd.DataFrame(gi_summary_rows)
    gi_df.to_parquet(run_dir / "tables" / "local_negative_response_gistar.parquet", index=False)
    gi_summary.to_csv(run_dir / "tables" / "local_negative_response_gistar_summary.csv", index=False)

    dpi = int(config["map_dpi"])
    figure_paths: list[str] = []
    figure_paths += _save_map_grid(run_dir, "local_spearman_240m", map_local, mask_info, dpi,
                                   "continuous", "Local Spearman：log1p(Aa) 与健康的局部秩关联")
    figure_paths += _save_map_grid(run_dir, "bivariate_local_moran_clusters", map_lisa, mask_info, dpi,
                                   "lisa", "Bivariate Local Moran：Aa 与邻域健康空间滞后")
    figure_paths += _save_map_grid(run_dir, "same_site_erosion_health_colocation", map_colocation, mask_info, dpi,
                                   "colocation", "同位 Aa—健康标准化值共现（非 LISA）")
    for model_family, frames in map_gi_by_model.items():
        figure_paths += _save_map_grid(run_dir, f"gistar_negative_response_{model_family.lower()}", frames,
                                       mask_info, dpi, "hotspot", f"{model_family} 局部负向响应强度 Gi* 聚类")

    report = _report(run_dir, local_summary, lisa_summary, same_site_summary, gi_summary,
                     int(env_complete.sum()), k_spearman,
                     float(local_summary.median_neighborhood_radius_m.median()), unit_counts, config)
    provenance = {"phase": "I-6", "mode": config["mode"], "outcome_guided_selection": True,
                  "run_id": config["run_id"], "created_utc": datetime.now(timezone.utc).isoformat(),
                  "git_commit": _git("rev-parse", "HEAD"), "git_status_porcelain": _git("status", "--short"),
                  "inputs": {"environment_erosion": str(env_path.relative_to(ROOT)),
                             "health": str(health_path.relative_to(ROOT)),
                             "i4_local_coefficients": str(i4_path.relative_to(ROOT))},
                  "configuration": str((run_dir / "configs" / config_path.name).relative_to(ROOT)),
                  "rows": {"mask_pixels": len(pixels), "environment_and_exposure_complete_pixels": int(env_complete.sum()),
                           "analysis_units_by_outcome": unit_counts},
                  "analysis_unit": "8x8 pixels; approximately 240 m summary units",
                  "coordinate_reference": "EPSG:32649",
                  "methods": {"local_spearman": {"neighbors": k_spearman,
                                 "distance_median_m": float(local_summary.median_neighborhood_radius_m.median()),
                                 "p_values": "not computed; overlapping windows are descriptive"},
                              "bivariate_local_moran": {"neighbors": k_lisa, "weights": "row-standardized KNN; focal exposure × neighbor-lagged health",
                                 "permutations": n_permutations, "seed": int(config["bivariate_lisa_seed"]),
                                 "p_values": "two-sided global randomization pseudo-p; BH within outcome"},
                              "gistar": {"source": "Phase I-4 environment-adjusted GWR and GWR-LCR", "radius_m": float(config["gi_star_radius_m"]),
                                 "weights": "binary distance band including focal unit", "strength": "max(-standardized_beta, 0)",
                                 "p_values": "two-sided normal approximation; BH within outcome and model"}},
                  "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                               "scipy": scipy.__version__, "matplotlib": _version("matplotlib"), "pyyaml": _version("PyYAML")},
                  "outputs": [str((run_dir / p).relative_to(ROOT)) for p in [
                      Path("tables/local_spearman.parquet"), Path("tables/bivariate_local_moran.parquet"),
                      Path("tables/same_site_erosion_health_colocation.csv"), Path("tables/local_negative_response_gistar.parquet"),
                      Path("reports/PHASE_I_SPATIAL_CLUSTER_REPORT.md")]] + figure_paths,
                  "report": report}
    (run_dir / "manifests" / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "RUN_STATUS.md").write_text(
        "# Run status\n\nStatus: authoritative exploratory result\n\n- Phase: I-6\n- `outcome_guided_selection=true`\n- Local Spearman uses overlapping descriptive neighborhoods without local p-values.\n- LISA global-randomization pseudo-p and Gi* normal-approximation p-values are exploratory.\n- Same-site high Aa/low health classes are separate from bivariate LISA neighbor-lag classes.\n",
        encoding="utf-8")
    files = sorted(p.relative_to(run_dir).as_posix() for p in run_dir.rglob("*") if p.is_file() and p.name != "output_manifest.json")
    (run_dir / "manifests" / "output_manifest.json").write_text(json.dumps(files, ensure_ascii=False, indent=2), encoding="utf-8")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "phase_i_spatial_cluster.yaml")
    args = parser.parse_args()
    run_dir = run(args.config)
    print(run_dir.relative_to(ROOT))
    print((run_dir / "reports" / "PHASE_I_SPATIAL_CLUSTER_REPORT.md").relative_to(ROOT))


if __name__ == "__main__":
    main()
