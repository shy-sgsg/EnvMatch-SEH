"""Exploratory GWR maps for the current CEC data."""

from __future__ import annotations

import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phase_i_cec_corrected_gwr_mplconfig")

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import yaml

from experiments.oracle_exploration.phase_i_spatial import (
    _add_map_decor,
    _fit_gwr,
    _mask_arrays,
    _tile_pixels,
)

ROOT = Path(__file__).resolve().parents[2]
INPUT_RUN_ID = "run_20261025T114741_470772Z"
EROSION_MULTIPLIER = float(os.environ.get("CEC_GWR_EROSION_MULTIPLIER", "20"))
RESPONSE_DIVISOR = float(os.environ.get("CEC_GWR_RESPONSE_DIVISOR", "2.9"))
RESPONSE_OFFSET = float(os.environ.get("CEC_GWR_RESPONSE_OFFSET", "18"))
RUN_ID = os.environ.get(
    "CEC_GWR_RUN_ID", "phase_i_cec_corrected_20_2p9_plus18_20261031_i4_gwr"
)
RUN_DIR = ROOT / "outputs" / "exploratory_oracle" / "phase_i" / RUN_ID
ALIGNED_DIR = ROOT / "outputs" / INPUT_RUN_ID / "maps" / "aligned"
DB_DIR = ROOT / "outputs" / INPUT_RUN_ID / "databases"
EXPOSURES = ["Aa_20yr", "Aa_2020"]
COLORS = {"blue": "#4477AA", "gold": "#CC9933", "ink": "#263746"}


def _outcome_formula() -> str:
    return (
        f"(CEC - {EROSION_MULTIPLIER:g} * "
        "(Aa_20yr/max_Aa_20yr + Aa_2020/max_Aa_2020))/"
        f"{RESPONSE_DIVISOR:g} "
        f"+ {RESPONSE_OFFSET:g}"
    )

try:
    font_manager.findfont("WenQuanYi Zen Hei", fallback_to_default=False)
    plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
except ValueError:
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
    except Exception:
        return None


def _read_full_grid_maxima() -> tuple[dict[str, float], dict[str, Any]]:
    maxima: dict[str, float] = {}
    grid: dict[str, Any] | None = None
    for exposure in EXPOSURES:
        path = ALIGNED_DIR / f"{exposure}.tif"
        with rasterio.open(path) as dataset:
            current_grid = {
                "shape": [dataset.height, dataset.width],
                "crs": dataset.crs.to_string() if dataset.crs else None,
                "transform": [float(value) for value in tuple(dataset.transform)[:6]],
            }
            if grid is None:
                grid = current_grid
            elif current_grid != grid:
                raise ValueError("Aligned Aa rasters do not share one grid.")
            values = dataset.read(1, masked=True).compressed()
            finite = values[np.isfinite(values)]
            if not finite.size:
                raise ValueError(f"No valid values in {path}.")
            maxima[exposure] = float(finite.max())
    assert grid is not None
    return maxima, grid


def _read_pixels(maxima: dict[str, float]) -> tuple[pd.DataFrame, dict[str, str]]:
    env_path = DB_DIR / "environment_erosion.parquet"
    health_path = DB_DIR / "health.parquet"
    env = pd.read_parquet(
        env_path,
        columns=["cell_id", "row", "col", "x", "y", *EXPOSURES],
    )
    health = pd.read_parquet(health_path, columns=["cell_id", "CEC"])
    data = env.merge(health, on="cell_id", how="inner", validate="one_to_one")
    if len(data) != len(env) or not data["cell_id"].is_unique:
        raise ValueError("Pixel databases must align one-to-one by cell_id.")

    required = ["x", "y", "CEC", *EXPOSURES]
    numeric = data[required].apply(pd.to_numeric, errors="coerce")
    valid = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
    data = data.loc[valid].copy()
    for column in required:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    for exposure in EXPOSURES:
        data[f"{exposure}_normalized"] = data[exposure] / maxima[exposure]
    data["Erosion_normalized_sum"] = sum(
        data[f"{exposure}_normalized"] for exposure in EXPOSURES
    )
    data["CEC_corrected"] = (
        data["CEC"] - EROSION_MULTIPLIER * data["Erosion_normalized_sum"]
    ) / RESPONSE_DIVISOR + RESPONSE_OFFSET
    return data, {"environment_erosion": str(env_path), "health": str(health_path)}


def _fit_models(
    pixels: pd.DataFrame, maxima: dict[str, float]
) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    config = {
        "bandwidth_neighbors": [30, 40, 50, 70, 100, 140, 200, 300, 440, 600],
        "cv_target_units": 400,
        "cv_seed": 20261034,
    }
    fits: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    cv_tables: list[pd.DataFrame] = []
    for exposure in EXPOSURES:
        feature = f"{exposure}_normalized"
        units = _tile_pixels(pixels, "CEC_corrected", [feature], tile_cells=8)
        units.attrs["analysis_population"] = "cropland_common_complete_case"
        local, summary, cv = _fit_gwr(
            units,
            "CEC_corrected",
            [feature],
            config,
            seed_offset=0,
        )
        summary["exposure"] = exposure
        summary["predictor"] = feature
        summary["predictor_normalization_max"] = maxima[exposure]
        summary["outcome_formula"] = _outcome_formula()
        cv["exposure"] = exposure
        cv["predictor"] = feature
        fits.append({"exposure": exposure, "units": units, "local": local, "cv": cv})
        summaries.append(summary)
        cv_tables.append(cv)
    return fits, pd.DataFrame(summaries), pd.concat(cv_tables, ignore_index=True)


def _pixel_ols(pixels: pd.DataFrame, maxima: dict[str, float]) -> pd.DataFrame:
    y = pixels["CEC_corrected"].to_numpy(dtype=float)
    y_mean = float(y.mean())
    y_var = float(np.mean((y - y_mean) ** 2))
    rows = []
    for exposure in EXPOSURES:
        x = pixels[f"{exposure}_normalized"].to_numpy(dtype=float)
        x_mean = float(x.mean())
        x_var = float(np.mean((x - x_mean) ** 2))
        covariance = float(np.mean((x - x_mean) * (y - y_mean)))
        slope = covariance / x_var
        r = covariance / np.sqrt(x_var * y_var)
        rows.append(
            {
                "exposure": exposure,
                "n_pixels": len(pixels),
                "pearson_r": r,
                "ols_slope_normalized_units": slope,
                "ols_slope_raw_Aa_units": slope / maxima[exposure],
                "ols_intercept": y_mean - slope * x_mean,
                "r_squared": r * r,
            }
        )
    return pd.DataFrame(rows)


def _save_figure(
    pixels: pd.DataFrame, fits: list[dict[str, Any]], run_dir: Path, dpi: int
) -> tuple[Path, float]:
    mask_info = _mask_arrays(pixels)
    all_beta = np.concatenate(
        [fit["local"]["local_beta_standardized"].to_numpy(dtype=float) for fit in fits]
    )
    finite = all_beta[np.isfinite(all_beta)]
    scale = max(float(np.quantile(np.abs(finite), 0.97)), 0.05)

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.5), constrained_layout=True)
    scatter = None
    for ax, fit in zip(axes, fits, strict=True):
        exposure = fit["exposure"]
        local = fit["local"]
        beta = local["local_beta_standardized"].to_numpy(dtype=float)
        scatter = ax.scatter(
            local["x"] / 1000,
            local["y"] / 1000,
            c=beta,
            s=9,
            marker="s",
            linewidths=0,
            cmap="RdBu_r",
            vmin=-scale,
            vmax=scale,
            rasterized=True,
        )
        _add_map_decor(ax, *mask_info)
        summary = next(
            item for item in fit["summary_rows"] if item["exposure"] == exposure
        )
        exposure_label = {
            "Aa_20yr": "20 年平均实际侵蚀",
            "Aa_2020": "2020 年实际侵蚀",
        }[exposure]
        ax.set_title(
            f"{exposure_label}  |  邻近单元数 k={int(summary['bandwidth_neighbors'])}，"
            f"中位带宽={summary['bandwidth_median_distance_m']/1000:.2f} km",
            fontsize=11,
        )
        ax.set_xlabel("东向坐标（km，UTM 49N）")
        ax.set_ylabel("北向坐标（km）")
        ax.set_aspect("equal", adjustable="datalim")
        ax.text(
            0.02,
            0.02,
            f"像元加权平均 β={summary['local_beta_standardized_mean']:+.3f}\n"
            f"负向系数占比={summary['negative_effect_ratio']:.1%}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            color=COLORS["ink"],
            bbox={"facecolor": "white", "edgecolor": "#d5dde2", "alpha": 0.92, "pad": 4},
        )

    assert scatter is not None
    scatter.set_clim(-scale, scale)
    fig.suptitle(
        "CEC 的局部 GWR 系数",
        fontsize=15,
        color=COLORS["ink"],
    )
    fig.colorbar(
        scatter,
        ax=axes,
        shrink=0.80,
        pad=0.02,
        label="局部标准化系数 β（红色正向，蓝色负向）",
    )
    fig.text(
        0.5,
        -0.015,
        "每幅图单独拟合一个归一化侵蚀项；按 8×8 像元汇总，并按像元数加权。",
        ha="center",
        va="top",
        fontsize=9,
        color="#4b5b66",
    )
    output = run_dir / "figures" / "cec_corrected_gwr_coefficients.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output, scale


def _write_run(
    pixels: pd.DataFrame,
    fits: list[dict[str, Any]],
    summary: pd.DataFrame,
    cv: pd.DataFrame,
    sources: dict[str, str],
    maxima: dict[str, float],
    grid: dict[str, Any],
    scale: float,
    figure_path: Path,
) -> None:
    run_dir = RUN_DIR
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "tables").mkdir(exist_ok=True)
    (run_dir / "configs").mkdir(exist_ok=True)
    (run_dir / "manifests").mkdir(exist_ok=True)

    summary.to_csv(run_dir / "tables" / "gwr_summary.csv", index=False)
    cv.to_csv(run_dir / "tables" / "bandwidth_cv.csv", index=False)
    ols = _pixel_ols(pixels, maxima)
    ols.to_csv(run_dir / "tables" / "global_ols.csv", index=False)
    local_frames = []
    for fit in fits:
        local = fit["local"].copy()
        local["exposure"] = fit["exposure"]
        local_frames.append(local)
    pd.concat(local_frames, ignore_index=True).to_csv(
        run_dir / "tables" / "local_coefficients.csv", index=False
    )

    config = {
        "phase": "I-CEC-GWR",
        "mode": "EXPLORATORY_ORACLE",
        "outcome_guided_selection": True,
        "run_id": RUN_ID,
        "input_run_id": INPUT_RUN_ID,
        "outcome_formula": _outcome_formula(),
        "normalization_scope": "each full aligned Aa raster valid cells; maxima recorded in provenance",
        "exposures": EXPOSURES,
        "tile_cells": 8,
        "cell_size_m": 30,
        "bandwidth_neighbors": [30, 40, 50, 70, 100, 140, 200, 300, 440, 600],
        "cv_target_units": 400,
        "cv_seed": 20261034,
        "kernel": "adaptive Gaussian over k nearest 240m units",
        "bandwidth_selection": "sampled leave-one-unit-out RMSE; exploratory tuning",
        "local_beta": "support-weighted standardized coefficient",
        "output_root": str(RUN_DIR.relative_to(ROOT)),
    }
    (run_dir / "configs" / "run.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    cv_selected = summary[
        [
            "exposure",
            "n_spatial_units",
            "n_source_pixels_or_pairs",
            "bandwidth_neighbors",
            "bandwidth_median_distance_m",
            "local_beta_standardized_mean",
            "local_beta_standardized_median",
            "local_beta_p05",
            "local_beta_p95",
            "negative_effect_ratio",
            "local_r2_weighted_mean",
        ]
    ]
    lines = [
        "# Formula-derived CEC local GWR",
        "",
        f"Run: `{RUN_ID}`; mode: `EXPLORATORY_ORACLE`; `outcome_guided_selection=true`.",
        "",
        "## Definition",
        "",
        "`Erosion = Aa_20yr/max(Aa_20yr) + Aa_2020/max(Aa_2020)`, where each maximum is taken over valid cells in its full aligned raster.",
        f"`CEC_corrected = (CEC - {EROSION_MULTIPLIER:g} * Erosion)/{RESPONSE_DIVISOR:g} + {RESPONSE_OFFSET:g}`.",
        "The response is formula-derived from both erosion rasters. Each local model below uses one normalized exposure at a time, so its coefficient reflects both the response construction and the selected predictor.",
        "",
        "## GWR design",
        "",
        f"- Common complete pixels: {len(pixels):,}; CRS: EPSG:32649; aligned 30 m grid.",
        f"- Formula-derived CEC_corrected range on these pixels: {pixels['CEC_corrected'].min():.4f} to {pixels['CEC_corrected'].max():.4f}.",
        "- Aggregate 8×8 pixel tiles to approximately 240 m units; weight local regressions by contributing pixel count.",
        "- Adaptive Gaussian kernel; candidate bandwidths 30–600 units; selected by sampled leave-one-unit-out RMSE on 400 units.",
        "- Coefficients are standardized local associations. Local model intervals are not spatial-block corrected.",
        "",
        "## Results",
        "",
        "| Exposure | Units | Pixels | k | Median bandwidth (km) | Weighted mean β | Negative support | Weighted local R² |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.exposure} | {int(row.n_spatial_units):,} | {int(row.n_source_pixels_or_pairs):,} | "
            f"{int(row.bandwidth_neighbors)} | {row.bandwidth_median_distance_m/1000:.2f} | "
            f"{row.local_beta_standardized_mean:+.3f} | {row.negative_effect_ratio:.1%} | "
            f"{row.local_r2_weighted_mean:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Pixel-level simple OLS",
            "",
            "Each model uses one normalized Aa predictor and an intercept; no pixel-level standard errors are used because raster pixels are spatially dependent.",
            "",
            "| Exposure | Pearson r | OLS slope per normalized unit | R² |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in ols.itertuples(index=False):
        lines.append(
            f"| {row.exposure} | {row.pearson_r:+.4f} | "
            f"{row.ols_slope_normalized_units:+.4f} | {row.r_squared:.4f} |"
        )
    lines.extend(
        [
            "",
            "The corrected response is not an independently observed CEC layer. Because the supplied formula directly subtracts normalized erosion, these GWR patterns do not independently validate the alleged original CEC values.",
            "",
            f"Figure: `{figure_path.relative_to(ROOT)}` (shared color scale; symmetric limits ±{scale:.4f}; pooled absolute 97th percentile).",
        ]
    )
    (run_dir / "reports").mkdir(exist_ok=True)
    (run_dir / "reports" / "gwr_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    software = {
        "python": platform.python_version(),
        "numpy": _package_version("numpy"),
        "scipy": _package_version("scipy"),
        "pandas": _package_version("pandas"),
        "rasterio": _package_version("rasterio"),
        "matplotlib": _package_version("matplotlib"),
    }
    provenance = {
        "phase": "I-CEC-GWR",
        "mode": "EXPLORATORY_ORACLE",
        "outcome_guided_selection": True,
        "run_id": RUN_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "outcome_formula": _outcome_formula(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_status_porcelain": _git("status", "--porcelain"),
        "analysis_script": str(Path(__file__).relative_to(ROOT)),
        "inputs": sources,
        "source_data_mapping": "config/data_paths.yaml",
        "raw_CEC_source": yaml.safe_load(
            (ROOT / "config" / "data_paths.yaml").read_text(encoding="utf-8")
        )["health"]["CEC"],
        "aligned_CEC_raster": str((ALIGNED_DIR / "CEC.tif").relative_to(ROOT)),
        "aligned_erosion_rasters": {
            name: str((ALIGNED_DIR / f"{name}.tif").relative_to(ROOT)) for name in EXPOSURES
        },
        "full_grid_maxima": maxima,
        "grid": grid,
        "sample_pixels": len(pixels),
        "analysis_unit": "240 m (8x8 30 m pixels), pixel-support-weighted",
        "coordinate_reference": "EPSG:32649",
        "software": software,
        "figure_color_scale_abs_p97": scale,
        "outputs": [
            str(figure_path.relative_to(ROOT)),
            str((run_dir / "tables" / "gwr_summary.csv").relative_to(ROOT)),
            str((run_dir / "tables" / "global_ols.csv").relative_to(ROOT)),
            str((run_dir / "tables" / "local_coefficients.csv").relative_to(ROOT)),
            str((run_dir / "tables" / "bandwidth_cv.csv").relative_to(ROOT)),
            str((run_dir / "reports" / "gwr_report.md").relative_to(ROOT)),
        ],
    }
    (run_dir / "manifests" / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run() -> Path:
    if RUN_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {RUN_DIR}")
    maxima, grid = _read_full_grid_maxima()
    pixels, sources = _read_pixels(maxima)
    fits, summary, cv = _fit_models(pixels, maxima)
    for fit, row in zip(fits, summary.to_dict(orient="records"), strict=True):
        fit["summary_rows"] = [row]
    figure_path, scale = _save_figure(pixels, fits, RUN_DIR, dpi=300)
    _write_run(pixels, fits, summary, cv, sources, maxima, grid, scale, figure_path)
    print(f"run_id={RUN_ID}")
    print(f"sample_pixels={len(pixels)}")
    print(f"figure={figure_path}")
    print(summary.to_string(index=False))
    return figure_path


if __name__ == "__main__":
    run()
