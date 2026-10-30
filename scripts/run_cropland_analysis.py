"""Prepare the cropland grid, dual erosion exposures, and G2-G4 diagnostics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/envmatch-matplotlib-cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from sklearn.metrics import adjusted_rand_score, silhouette_score

from soilmatch.balance import assess_balance
from soilmatch.clustering import cluster_environments
from soilmatch.config import load_config, load_data_mapping, project_root, resolve_path
from soilmatch.covariates import screen_covariates
from soilmatch.cropland import create_binary_cropland_mask, inspect_polygon_shapefile
from soilmatch.erosion_variability import summarize_erosion
from soilmatch.feasibility import add_within_group_exposure_rank
from soilmatch.environment_vectors import build_environment_vectors
from soilmatch.grid_database import build_grid_database
from soilmatch.matching import match_cells
from soilmatch.raster_alignment import align_raster_to_template
from soilmatch.reporting import create_run_directory
from soilmatch.raster_validation import assert_aligned


CORE_ENVIRONMENT = ("Elevation", "Cur", "TWI", "LST")
OPTIONAL_ENVIRONMENT = ("NDWI", "NDBSI", "EVI", "NDVI", "Precipitation")
HEALTH_INDICATORS = ("AN", "OM", "CEC", "pH", "SW", "BD", "MBC", "RESP")
SOIL_CODES = (26, 27, 42)
SOIL_TYPE_LABELS = {26: "冲积土", 27: "潮土", 42: "脱潮土"}
MASK_VALUE = 12
ENVIRONMENT_DISTANCE = 0.5
EXPOSURE_RANK_GAP = 0.25
RADIUS_CANDIDATES_M = (300.0, 500.0, 1000.0)
CLUSTER_COUNTS = (6, 8, 10, 12)
RANDOM_STATE = 20261024
EXPOSURE_FILES = {
    "Aa_20yr": "Aa_20yr_average.tif",
    "Aa_2020": "Aa_2020_average.tif",
    "Aa_20yr_nearest": "Aa_20yr_nearest.tif",
    "Aa_2020_nearest": "Aa_2020_nearest.tif",
}

FIGURE_FONT = "Noto Sans CJK JP"
FIGURE_COLORS = {
    "navy": "#28566E",
    "teal": "#4E8983",
    "ochre": "#C08A3E",
    "coral": "#B85E52",
    "ink": "#263A46",
    "muted": "#687983",
    "grid": "#DDE5E8",
    "paper": "#F7F9FA",
}
GROUP_COLORS = [
    "#315D78", "#4E8A86", "#C57645", "#A7A64C", "#6A6A9B", "#8C6D58",
    "#4A7896", "#9A5870", "#8C9D73", "#6F889C", "#BD9451", "#58745D",
]


def _configure_figure_style() -> None:
    plt.rcParams.update({
        "font.family": FIGURE_FONT,
        "axes.edgecolor": "#AAB8BE",
        "axes.linewidth": 0.8,
        "axes.labelcolor": FIGURE_COLORS["ink"],
        "axes.titlecolor": FIGURE_COLORS["ink"],
        "xtick.color": FIGURE_COLORS["muted"],
        "ytick.color": FIGURE_COLORS["muted"],
        "grid.color": FIGURE_COLORS["grid"],
        "grid.linewidth": 0.65,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def _nested(mapping: dict[str, Any], dotted_name: str) -> Any:
    value: Any = mapping
    for part in dotted_name.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _mapped_path(mapping: dict[str, Any], config_path: Path, dotted_name: str) -> Path | None:
    value = _nested(mapping, dotted_name)
    if value is None:
        return None
    return resolve_path(value, config_path)


def _source_grid(path: Path) -> dict[str, object]:
    with rasterio.open(path) as dataset:
        return {
            "source_crs": dataset.crs.to_string() if dataset.crs else None,
            "source_width": dataset.width,
            "source_height": dataset.height,
            "source_resolution_x": float(dataset.res[0]),
            "source_resolution_y": float(dataset.res[1]),
            "source_transform": ",".join(f"{value:.9f}" for value in tuple(dataset.transform)[:6]),
            "source_nodata": dataset.nodata,
            "source_dtype": dataset.dtypes[0],
        }


def _read_raster(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(path) as dataset:
        masked = dataset.read(1, masked=True)
        values = np.asarray(masked.data, dtype=np.float64)
        valid = ~np.ma.getmaskarray(masked) & np.isfinite(values)
    values[~valid] = np.nan
    return values, valid


def _layer_statistics(
    name: str,
    path: Path,
    source_path: Path,
    crop_mask: np.ndarray,
    resampling: str,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    values, valid = _read_raster(path)
    crop_valid = crop_mask & valid
    selected = values[crop_valid]
    stats: dict[str, object] = {
        "logical_name": name,
        "source_path": str(source_path),
        "aligned_path": str(path),
        **_source_grid(source_path),
        "template_width": int(values.shape[1]),
        "template_height": int(values.shape[0]),
        "resampling": resampling,
        "cropland_cells": int(crop_mask.sum()),
        "valid_cropland_cells": int(crop_valid.sum()),
        "missing_cropland_cells": int(crop_mask.sum() - crop_valid.sum()),
        "cropland_coverage": float(crop_valid.sum() / crop_mask.sum()) if crop_mask.any() else np.nan,
        "zero_count": int(np.count_nonzero(selected == 0)),
        "zero_fraction": float(np.mean(selected == 0)) if selected.size else np.nan,
        "negative_count": int(np.count_nonzero(selected < 0)),
        "minimum": float(np.min(selected)) if selected.size else np.nan,
        "p05": float(np.quantile(selected, 0.05)) if selected.size else np.nan,
        "p25": float(np.quantile(selected, 0.25)) if selected.size else np.nan,
        "median": float(np.median(selected)) if selected.size else np.nan,
        "p75": float(np.quantile(selected, 0.75)) if selected.size else np.nan,
        "p95": float(np.quantile(selected, 0.95)) if selected.size else np.nan,
        "maximum": float(np.max(selected)) if selected.size else np.nan,
        "mean": float(np.mean(selected)) if selected.size else np.nan,
        "standard_deviation": float(np.std(selected, ddof=1)) if selected.size > 1 else np.nan,
    }
    if name == "SoilType" and selected.size:
        codes, counts = np.unique(selected.astype(int), return_counts=True)
        stats["category_counts"] = json.dumps(
            dict(zip(codes.tolist(), counts.astype(int).tolist(), strict=True)), ensure_ascii=False
        )
    return stats, values, valid


def _write_area_figure(
    output: Path,
    raster_area_km2: float,
    polygon_area_km2: float | None,
    charter_area_km2: float,
) -> None:
    _configure_figure_style()
    labels = ["耕地掩膜栅格", "边界矢量几何（单位暂按米）", "总纲记录面积"]
    values = [raster_area_km2, polygon_area_km2, charter_area_km2]
    shown_labels = [label for label, value in zip(labels, values, strict=True) if value is not None]
    shown_values = [value for value in values if value is not None]
    figure, axis = plt.subplots(figsize=(9.2, 4.8), constrained_layout=True)
    bars = axis.barh(shown_labels, shown_values,
                     color=[FIGURE_COLORS["navy"], FIGURE_COLORS["teal"], FIGURE_COLORS["ochre"]][:len(shown_values)],
                     height=0.56, edgecolor="white", linewidth=1.0, zorder=3)
    axis.set_xlabel("面积（平方千米）")
    axis.set_title("耕地面积估算对照", loc="left", fontsize=15, fontweight="semibold", pad=14)
    axis.set_xlim(0, max(shown_values, default=1) * 1.19)
    axis.grid(axis="x", zorder=0)
    axis.set_axisbelow(True)
    for bar, value in zip(bars, shown_values, strict=True):
        axis.text(value + max(shown_values, default=1) * 0.012, bar.get_y() + bar.get_height() / 2,
                  f"{value:.3f}", ha="left", va="center", fontsize=10, color=FIGURE_COLORS["ink"])
    if polygon_area_km2 is not None:
        figure.text(0.01, -0.015, "注：边界矢量缺少可确认的投影信息，其面积按米制假设计算，仅供对照。",
                    ha="left", va="top", fontsize=8, color=FIGURE_COLORS["muted"])
    figure.savefig(output, dpi=320, bbox_inches="tight")
    plt.close(figure)


def _write_correlation_figure(correlation: pd.DataFrame, output: Path) -> None:
    _configure_figure_style()
    labels = {
        "Elevation": "高程", "Cur": "地表曲率", "TWI": "地形湿度指数", "LST": "地表温度",
        "NDWI": "水体指数", "NDBSI": "裸地与建筑指数", "EVI": "增强植被指数",
        "NDVI": "归一化植被指数", "Precipitation": "降水",
    }
    figure, axis = plt.subplots(figsize=(10.2, 8.4), constrained_layout=True)
    image = axis.imshow(correlation.to_numpy(), vmin=-1, vmax=1, cmap="RdBu_r", interpolation="nearest")
    axis.set_xticks(range(len(correlation.columns)), [labels.get(str(c), str(c)) for c in correlation.columns], rotation=38, ha="right")
    axis.set_yticks(range(len(correlation.index)), [labels.get(str(c), str(c)) for c in correlation.index])
    axis.tick_params(length=0, labelsize=9)
    axis.set_xticks(np.arange(-0.5, len(correlation.columns), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(correlation.index), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=1.2)
    axis.tick_params(which="minor", bottom=False, left=False)
    for row in range(len(correlation.index)):
        for column in range(len(correlation.columns)):
            value = correlation.iloc[row, column]
            if np.isfinite(value):
                text_color = "white" if abs(value) >= 0.62 else FIGURE_COLORS["ink"]
                axis.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=8, color=text_color)
    axis.set_title("耕地范围内环境变量相关性", loc="left", fontsize=15, fontweight="semibold", pad=14)
    figure.colorbar(image, ax=axis, label="Pearson 相关系数", fraction=0.045, pad=0.04, ticks=np.linspace(-1, 1, 5))
    figure.savefig(output, dpi=320, bbox_inches="tight")
    plt.close(figure)


def _write_exposure_figure(exposure_frame: pd.DataFrame, output: Path) -> None:
    _configure_figure_style()
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.3), constrained_layout=True, sharey=True)
    exposure_labels = {"Aa_20yr": "20 年平均实际侵蚀", "Aa_2020": "2020 年实际侵蚀"}
    method_labels = {"area average": "面积平均", "nearest": "最近邻"}
    colors = {"area average": FIGURE_COLORS["navy"], "nearest": FIGURE_COLORS["coral"]}
    for axis, exposure in zip(axes, ("Aa_20yr", "Aa_2020"), strict=True):
        zero_rates = []
        for method, suffix, color in (
            ("area average", "", colors["area average"]),
            ("nearest", "_nearest", colors["nearest"]),
        ):
            column = f"{exposure}{suffix}"
            values = pd.to_numeric(exposure_frame[column], errors="coerce").to_numpy(dtype=float)
            values = values[np.isfinite(values) & (values >= 0)]
            if values.size:
                logged = np.log1p(values)
                axis.hist(logged, bins=64, density=True, histtype="step", linewidth=2.0,
                          label=method_labels[method], color=color)
                zero_rates.append((method_labels[method], float(np.mean(values == 0))))
        axis.set_title(exposure_labels[exposure], loc="left", fontsize=12, fontweight="semibold", pad=9)
        axis.set_xlabel("log(1 + 栅格记录值；原始单位未编码)")
        axis.grid(axis="y", alpha=0.7)
        annotation = "\n".join(f"{label}零值率：{rate:.1%}" for label, rate in zero_rates)
        axis.text(0.98, 0.96, annotation, transform=axis.transAxes, ha="right", va="top", fontsize=8,
                  color=FIGURE_COLORS["muted"], bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#DDE5E8", "alpha": 0.92})
    axes[0].set_ylabel("密度")
    axes[1].legend(title="重采样方式", frameon=False, loc="upper left")
    figure.suptitle("实际侵蚀值的重采样敏感性（既有 Aa）", x=0.02, y=1.04, ha="left", fontsize=15, fontweight="semibold")
    figure.savefig(output, dpi=320, bbox_inches="tight")
    plt.close(figure)


def _markdown_table(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    selected = frame.loc[:, columns] if columns else frame
    if selected.empty:
        return "(no rows)"
    headers = [str(column) for column in selected.columns]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in selected.itertuples(index=False, name=None):
        cells = ["" if pd.isna(value) else str(value) for value in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _prepare_run(
    project_root: Path,
    config_path: Path,
    config: dict[str, Any],
    mapping: dict[str, Any],
    *,
    run_id: str | None,
) -> Path:
    output_root = resolve_path(config.get("project", {}).get("output_root", "outputs"), config_path)
    if output_root is None:
        raise ValueError("project.output_root could not be resolved")

    mask_source = _mapped_path(mapping, config_path, "spatial.cropland_mask")
    if mask_source is None:
        raise ValueError("spatial.cropland_mask is required")
    template_source = _mapped_path(mapping, config_path, "spatial.template") or mask_source
    if template_source.resolve() != mask_source.resolve():
        raise ValueError("This run uses the confirmed cropland-mask grid as its analysis template")

    source_paths: dict[str, Path] = {}
    required = [
        "erosion.Aa_20yr",
        "erosion.Aa_2020",
        *(f"environment.{name}" for name in CORE_ENVIRONMENT),
        "environment.SoilType",
        *(f"health.{name}" for name in HEALTH_INDICATORS),
        "health.SHI",
    ]
    for logical in required:
        source = _mapped_path(mapping, config_path, logical)
        if source is None or not source.exists():
            raise FileNotFoundError(f"Required mapped input is missing: {logical}: {source}")
        source_paths[logical] = source
    for name in OPTIONAL_ENVIRONMENT:
        source = _mapped_path(mapping, config_path, f"environment.{name}")
        if source is not None and source.exists():
            source_paths[f"environment.{name}"] = source

    path_metadata = {
        logical: str(path.relative_to(project_root)) if path.is_relative_to(project_root) else str(path)
        for logical, path in {"spatial.cropland_mask": mask_source, **source_paths}.items()
    }
    run_config = {
        "study_scope": "Wenxian cropland",
        "template": path_metadata["spatial.cropland_mask"],
        "cropland_codes": [MASK_VALUE],
        "environment_core": list(CORE_ENVIRONMENT),
        "environment_optional_diagnostics_only": list(OPTIONAL_ENVIRONMENT),
        "soil_codes_in_primary_groups": list(SOIL_CODES),
        "erosion_exposures": {"Aa_20yr": "20-year mean (user-defined)", "Aa_2020": "single year 2020 (user-defined)"},
        "resampling": {
            "categorical": "nearest",
            "continuous_environment": "bilinear",
            "erosion_primary": "average",
            "erosion_sensitivity": "nearest",
            "health_surfaces": "bilinear",
        },
        "g4_candidates": {
            "clusters_per_soil_type": list(CLUSTER_COUNTS),
            "environment_distance_z_euclidean": ENVIRONMENT_DISTANCE,
            "within_group_exposure_rank_gap": EXPOSURE_RANK_GAP,
            "radii_m": list(RADIUS_CANDIDATES_M),
            "allow_reference_reuse": False,
            "random_state": RANDOM_STATE,
        },
        "inputs": path_metadata,
    }
    run_dir = create_run_directory(
        output_root,
        run_id=run_id,
        config=run_config,
        inputs=path_metadata,
    )
    (run_dir / "databases").mkdir()
    aligned_dir = run_dir / "maps" / "aligned"
    aligned_dir.mkdir()

    mask_output = aligned_dir / "cropland_mask_binary.tif"
    mask_result = create_binary_cropland_mask(
        mask_source,
        mask_output,
        cropland_codes=config.get("analysis", {}).get("cropland_codes", [MASK_VALUE]),
    )
    with rasterio.open(mask_output) as mask_dataset:
        mask_masked = mask_dataset.read(1, masked=True)
        mask_values = np.asarray(mask_masked.data)
        crop_mask = (~np.ma.getmaskarray(mask_masked)) & (mask_values == 1)
        template_info = {
            "crs": mask_dataset.crs.to_string() if mask_dataset.crs else None,
            "width": mask_dataset.width,
            "height": mask_dataset.height,
            "resolution_x": float(mask_dataset.res[0]),
            "resolution_y": float(mask_dataset.res[1]),
            "transform": tuple(mask_dataset.transform)[:6],
            "bounds": tuple(mask_dataset.bounds),
        }

    layer_specs: dict[str, tuple[Path, str]] = {}
    for exposure in ("Aa_20yr", "Aa_2020"):
        layer_specs[f"{exposure}"] = (source_paths[f"erosion.{exposure}"], "average")
        layer_specs[f"{exposure}_nearest"] = (source_paths[f"erosion.{exposure}"], "nearest")
    for name in CORE_ENVIRONMENT:
        layer_specs[name] = (source_paths[f"environment.{name}"], "bilinear")
    layer_specs["SoilType"] = (source_paths["environment.SoilType"], "nearest")
    for name in OPTIONAL_ENVIRONMENT:
        logical = f"environment.{name}"
        if logical in source_paths:
            layer_specs[name] = (source_paths[logical], "bilinear")
    for name in HEALTH_INDICATORS:
        layer_specs[name] = (source_paths[f"health.{name}"], "bilinear")
    layer_specs["SHI_candidate"] = (source_paths["health.SHI"], "bilinear")

    alignment_rows: list[dict[str, object]] = []
    aligned_paths: dict[str, Path] = {}
    layer_stats: list[dict[str, object]] = []
    layer_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, (source, method) in layer_specs.items():
        output = aligned_dir / f"{name}.tif"
        result = align_raster_to_template(
            source,
            mask_source,
            output,
            resampling=method,
        )
        aligned_paths[name] = output
        row = {"logical_name": name, **_source_grid(source), **result}
        alignment_rows.append(row)
        stats, values, valid = _layer_statistics(name, output, source, crop_mask, method)
        layer_stats.append(stats)
        layer_arrays[name] = (values, valid)
        print(f"Aligned {name}: {int(valid.sum())} valid cells on target grid ({method})")

    alignment_table = pd.DataFrame(alignment_rows)
    alignment_table.to_csv(run_dir / "tables" / "raster_alignment.csv", index=False)
    statistics = pd.DataFrame(layer_stats)
    statistics.to_csv(run_dir / "tables" / "raster_statistics_cropland.csv", index=False)

    noncrop_path = _mapped_path(mapping, config_path, "spatial.noncropland_mask")
    mask_validation: dict[str, object] = {
        **mask_result,
        "cell_area_m2": float(abs(template_info["resolution_x"] * template_info["resolution_y"])),
        "cropland_area_km2": float(crop_mask.sum() * abs(template_info["resolution_x"] * template_info["resolution_y"]) / 1e6),
    }
    if noncrop_path is not None and noncrop_path.exists():
        with rasterio.open(mask_source) as crop_ds, rasterio.open(noncrop_path) as noncrop_ds:
            assert_aligned(crop_ds, noncrop_ds)
            crop_values = crop_ds.read(1, masked=True)
            noncrop_values = noncrop_ds.read(1, masked=True)
            crop_valid_source = ~np.ma.getmaskarray(crop_values)
            noncrop_valid_source = ~np.ma.getmaskarray(noncrop_values)
            mask_validation.update(
                noncrop_valid_cells=int(noncrop_valid_source.sum()),
                crop_noncrop_overlap=int((crop_valid_source & noncrop_valid_source).sum()),
                crop_noncrop_union=int((crop_valid_source | noncrop_valid_source).sum()),
                noncrop_area_km2=float(
                    noncrop_valid_source.sum()
                    * abs(template_info["resolution_x"] * template_info["resolution_y"])
                    / 1e6
                ),
            )

    study_area_path = _mapped_path(mapping, config_path, "spatial.study_area")
    polygon_area_km2: float | None = None
    polygon_info: dict[str, object] | None = None
    if study_area_path is not None and study_area_path.is_file():
        polygon_info = inspect_polygon_shapefile(study_area_path)
        polygon_area_km2 = float(polygon_info["area_coordinate_units_squared"]) / 1e6
        polygon_info.update(
            prj_exists=study_area_path.with_suffix(".prj").is_file(),
            shx_exists=study_area_path.with_suffix(".shx").is_file(),
            dbf_exists=study_area_path.with_suffix(".dbf").is_file(),
            area_interpretation="coordinate units squared; m² interpretation is provisional without .prj",
        )
        mask_validation["study_area_geometry"] = polygon_info
        mask_validation["study_area_geometry_area_km2_assuming_metre_coordinates"] = polygon_area_km2

    reference_area = float(config.get("analysis", {}).get("reference_cropland_area_km2", 347.877))
    raster_area = float(mask_validation["cropland_area_km2"])
    mask_validation["prior_charter_area_km2"] = reference_area
    mask_validation["difference_from_prior_km2"] = raster_area - reference_area
    mask_validation["difference_from_prior_percent"] = 100 * (raster_area - reference_area) / reference_area
    (run_dir / "diagnostics" / "cropland_mask.json").write_text(
        json.dumps(mask_validation, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    counts: list[dict[str, object]] = []

    def add_count(label: str, valid_mask: np.ndarray) -> None:
        count = int((crop_mask & valid_mask).sum())
        counts.append(
            {
                "analysis_subset": label,
                "valid_cells": count,
                "cropland_cells": int(crop_mask.sum()),
                "coverage_of_cropland": count / int(crop_mask.sum()) if crop_mask.any() else np.nan,
            }
        )

    valid_layers = {name: values[1] for name, values in layer_arrays.items()}
    add_count("cropland_mask", np.ones(crop_mask.shape, dtype=bool))
    core_valid = np.logical_and.reduce([valid_layers[name] for name in CORE_ENVIRONMENT])
    soil_values = layer_arrays["SoilType"][0]
    soil_valid = valid_layers["SoilType"] & np.isin(soil_values, SOIL_CODES)
    core_soil_valid = core_valid & soil_valid
    add_count("core_environment_and_valid_soil_type", core_soil_valid)
    for exposure in ("Aa_20yr", "Aa_2020"):
        exposure_valid = valid_layers[exposure] & (layer_arrays[exposure][0] >= 0)
        add_count(f"core_environment_soil_{exposure}", core_soil_valid & exposure_valid)
    both_exposures = (
        valid_layers["Aa_20yr"]
        & valid_layers["Aa_2020"]
        & (layer_arrays["Aa_20yr"][0] >= 0)
        & (layer_arrays["Aa_2020"][0] >= 0)
    )
    add_count("core_environment_soil_both_average_exposures", core_soil_valid & both_exposures)
    for name in (*HEALTH_INDICATORS, "SHI_candidate"):
        add_count(f"cropland_{name}", valid_layers[name])
        add_count(f"core_soil_Aa20yr_{name}", core_soil_valid & valid_layers["Aa_20yr"] & valid_layers[name])
    all_health_valid = np.logical_and.reduce([valid_layers[name] for name in HEALTH_INDICATORS])
    add_count("all_8_health_indicators", all_health_valid)
    add_count("core_soil_both_exposures_and_all_8_health", core_soil_valid & both_exposures & all_health_valid)
    common_cells = pd.DataFrame(counts)
    common_cells.to_csv(run_dir / "tables" / "common_valid_cells.csv", index=False)

    exposure_rows: list[dict[str, object]] = []
    for exposure in ("Aa_20yr", "Aa_2020"):
        avg, avg_valid = layer_arrays[exposure]
        nearest, nearest_valid = layer_arrays[f"{exposure}_nearest"]
        valid = crop_mask & avg_valid & nearest_valid
        left = avg[valid]
        right = nearest[valid]
        exposure_rows.append(
            {
                "exposure": exposure,
                "common_cells": int(valid.sum()),
                "area_average_zero_fraction": float(np.mean(left == 0)) if left.size else np.nan,
                "nearest_zero_fraction": float(np.mean(right == 0)) if right.size else np.nan,
                "area_average_mean": float(np.mean(left)) if left.size else np.nan,
                "nearest_mean": float(np.mean(right)) if right.size else np.nan,
                "area_average_p95": float(np.quantile(left, 0.95)) if left.size else np.nan,
                "nearest_p95": float(np.quantile(right, 0.95)) if right.size else np.nan,
                "area_average_max": float(np.max(left)) if left.size else np.nan,
                "nearest_max": float(np.max(right)) if right.size else np.nan,
                "pearson_average_nearest": float(np.corrcoef(left, right)[0, 1]) if left.size > 1 else np.nan,
                "mean_absolute_difference": float(np.mean(np.abs(left - right))) if left.size else np.nan,
            }
        )
    exposure_frame = pd.DataFrame(
        {
            name: layer_arrays[name][0][crop_mask]
            for name in ("Aa_20yr", "Aa_2020", "Aa_20yr_nearest", "Aa_2020_nearest")
        }
    )
    valid_both = np.isfinite(exposure_frame["Aa_20yr"]) & np.isfinite(exposure_frame["Aa_2020"])
    values20 = exposure_frame.loc[valid_both, "Aa_20yr"].to_numpy(dtype=float)
    values2020 = exposure_frame.loc[valid_both, "Aa_2020"].to_numpy(dtype=float)
    zero_20 = values20 == 0
    zero_2020 = values2020 == 0
    exposure_comparison = {
        "common_cells": int(valid_both.sum()),
        "Aa_20yr_zero_fraction": float(zero_20.mean()) if zero_20.size else np.nan,
        "Aa_2020_zero_fraction": float(zero_2020.mean()) if zero_2020.size else np.nan,
        "zero_both_count": int((zero_20 & zero_2020).sum()),
        "Aa_20yr_zero_Aa_2020_positive_count": int((zero_20 & ~zero_2020).sum()),
        "Aa_20yr_positive_Aa_2020_zero_count": int((~zero_20 & zero_2020).sum()),
        "pearson_common_average": float(np.corrcoef(values20, values2020)[0, 1]) if values20.size > 1 else np.nan,
        "spearman_common_average": float(
            pd.Series(values20).corr(pd.Series(values2020), method="spearman")
        ) if values20.size > 1 else np.nan,
        "note": "The zero-set implication is diagnostic only; the 20-year period and averaging method are not documented.",
    }
    pd.DataFrame(exposure_rows).to_csv(run_dir / "tables" / "erosion_resampling_sensitivity.csv", index=False)
    (run_dir / "diagnostics" / "erosion_time_scale_comparison.json").write_text(
        json.dumps(exposure_comparison, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    dictionary_rows = [
        {"logical_name": "Aa_20yr", "role": "primary exposure", "time": "20-year mean (user-defined); exact years/method unverified", "unit": "not encoded", "source": path_metadata["erosion.Aa_20yr"], "resampling": "area average (primary), nearest (sensitivity)", "status": "user-defined identity; metadata incomplete"},
        {"logical_name": "Aa_2020", "role": "single-year exposure comparison", "time": "2020 (user-defined; ArcGrid lineage references SE_Henan_2020)", "unit": "not encoded", "source": path_metadata["erosion.Aa_2020"], "resampling": "area average (primary), nearest (sensitivity)", "status": "user-defined identity; metadata incomplete"},
        {"logical_name": "cropland_mask", "role": "analysis mask and grid template", "time": "not encoded", "unit": "30 m cells", "source": path_metadata["spatial.cropland_mask"], "resampling": "source grid retained", "status": "code 12 selected; legend inferred from filename and paired geometry"},
        {"logical_name": "SHI_candidate", "role": "health outcome candidate", "time": "not encoded", "unit": "not encoded", "source": path_metadata["health.SHI"], "resampling": "bilinear", "status": "provisional identity; formula/source missing"},
    ]
    health_units = {
        "AN": "mg/kg",
        "OM": "g/kg",
        "CEC": "cmol/kg",
        "pH": "dimensionless",
        "SW": "%",
        "BD": "g/cm³",
        "MBC": "mg/kg",
        "RESP": "µg/kg/h",
    }
    for name in HEALTH_INDICATORS:
        dictionary_rows.append(
            {
                "logical_name": name,
                "role": "health outcome; held out from G3/G4",
                "time": "map/measurement year not encoded",
                "unit": health_units[name],
                "source": path_metadata[f"health.{name}"],
                "resampling": "bilinear",
                "status": "lineage points to final indicator maps; prediction/model provenance not fully supplied",
            }
        )
    data_dictionary = pd.DataFrame(dictionary_rows)
    data_dictionary.to_csv(run_dir / "tables" / "data_dictionary.csv", index=False)

    continuous_candidates = [name for name in (*CORE_ENVIRONMENT, *OPTIONAL_ENVIRONMENT) if name in layer_arrays]
    environment_frame = pd.DataFrame(
        {name: layer_arrays[name][0][crop_mask] for name in continuous_candidates}
    )
    screened = screen_covariates(environment_frame, continuous_candidates, max_missing_fraction=0.05)
    screened.to_csv(run_dir / "tables" / "environment_screen.csv", index=False)
    correlation = environment_frame.corr(min_periods=2)
    correlation.to_csv(run_dir / "tables" / "environment_correlation.csv")
    _write_correlation_figure(correlation, run_dir / "figures" / "environment_correlation.png")
    _write_exposure_figure(exposure_frame, run_dir / "figures" / "erosion_resampling_sensitivity.png")
    _write_area_figure(
        run_dir / "figures" / "cropland_area_comparison.png",
        raster_area,
        polygon_area_km2,
        reference_area,
    )

    environment_layers = {
        name: aligned_paths[name]
        for name in (*CORE_ENVIRONMENT, "SoilType", *OPTIONAL_ENVIRONMENT)
        if name in aligned_paths
    }
    environment_layers.update({name: aligned_paths[name] for name in EXPOSURE_FILES})
    build_grid_database(
        environment_layers,
        run_dir / "databases" / "environment_erosion.parquet",
        mask=mask_output,
        overwrite=False,
    )
    health_layers = {name: aligned_paths[name] for name in HEALTH_INDICATORS}
    health_layers["SHI_candidate"] = aligned_paths["SHI_candidate"]
    build_grid_database(
        health_layers,
        run_dir / "databases" / "health.parquet",
        mask=mask_output,
        overwrite=False,
    )

    crop_area_difference = raster_area - reference_area
    common_table = _markdown_table(
        common_cells,
        ["analysis_subset", "valid_cells", "coverage_of_cropland"],
    )
    mask_report = f"""# Cropland mask validation

- Source: `{mask_source}`; values observed in the valid raster: `{mask_result['source_code_counts']}`.
- Operational cropland code: {MASK_VALUE}; selected cells: {int(crop_mask.sum()):,}.
- Target grid: {template_info['crs']}, {template_info['width']}×{template_info['height']} at {template_info['resolution_x']} m; transform `{template_info['transform']}`.
- Raster cell area: {mask_validation['cell_area_m2']:.1f} m²; selected raster area: {raster_area:.3f} km².
- Previous charter area: {reference_area:.3f} km²; mask raster exceeds it by {crop_area_difference:.3f} km² ({mask_validation['difference_from_prior_percent']:.2f}%). The delivered crop polygon geometry is {polygon_area_km2:.3f} coordinate km² if read as metres; its `.prj` is missing, so that interpretation is provisional. Raster and polygon areas are close; the difference from the prior area remains unresolved and is not forced away.
- Paired non-cropland raster valid cells: {mask_validation.get('noncrop_valid_cells', 'not available')}; crop/non-crop valid overlap: {mask_validation.get('crop_noncrop_overlap', 'not available')}.
- The shapefile bundle lacks `.shx`, `.dbf`, and `.prj`; the geometry-only area calculation does not recover attributes or establish its CRS.

The raster named `温县耕地.tif` is used as the user-designated cropland mask. Its only valid code is 12. Source NoData is preserved as NoData in the binary mask; non-cropland valid source codes, if any, are encoded as zero.
"""
    (run_dir / "cropland_mask_validation.md").write_text(mask_report, encoding="utf-8")

    grid_report = f"""# Analysis grid specification

- Scope: Wenxian cropland, using the user-designated `温县耕地.tif` raster as both mask source and target grid.
- CRS: {template_info['crs']}; projected metres; cell size {template_info['resolution_x']}×{template_info['resolution_y']} m; shape {template_info['width']} columns × {template_info['height']} rows.
- Affine transform: `{template_info['transform']}`; bounds: `{template_info['bounds']}`.
- Mask rule: source value 12 is selected; source NoData remains NoData. The original raster is read-only.
- Resampling: categorical soil/mask—nearest; continuous environmental and health prediction surfaces—bilinear; each continuous erosion exposure—area average for primary alignment and nearest for a sensitivity comparison.
- Main paired analysis uses the area-average versions of `Aa_20yr` and `Aa_2020`; nearest versions are retained as sensitivity layers. All aligned outputs are new files under this run directory.
- Aa_20yr exact start/end years, calculation definition, and both exposure units are absent from source metadata. The user has identified their time roles; this report does not infer missing units or calculation lineage.
"""
    (run_dir / "analysis_grid_specification.md").write_text(grid_report, encoding="utf-8")

    alignment_report = f"""# Raster alignment report

All source layers were explicitly reprojected to the cropland-mask grid. Original rasters were not changed. Continuous variables were warped with bilinear interpolation, categorical SoilType with nearest neighbour, and both erosion exposures with area average plus nearest-neighbour sensitivity. NoData was propagated as missing rather than filled with zero.

{_markdown_table(alignment_table, ['logical_name', 'source_crs', 'source_width', 'source_height', 'source_resolution_x', 'source_resolution_y', 'resampling', 'destination_valid_count'])}

Per-layer valid coverage, distributions, and NoData counts within the cropland mask are in `tables/raster_statistics_cropland.csv`. Source and target transforms are in `tables/raster_alignment.csv`.
"""
    (run_dir / "raster_alignment_report.md").write_text(alignment_report, encoding="utf-8")

    env_counts = common_cells.loc[
        common_cells["analysis_subset"].isin(
            ["cropland_mask", "core_environment_and_valid_soil_type", "core_environment_soil_both_average_exposures", "all_8_health_indicators", "core_soil_both_exposures_and_all_8_health"]
        )
    ]
    report = f"""# G2-G3 cropland data preparation

## G2

The cropland mask provides an explicit common target grid, allowing the previous grid-mismatch stop to be addressed without modifying inputs. The binary mask contains {int(crop_mask.sum()):,} crop cells ({raster_area:.3f} km² by {mask_validation['cell_area_m2']:.0f} m² raster-cell area). Unit and exact averaging lineage remain unverified for the two precomputed erosion rasters. RUSLE factor-product validation was not performed, per the user's prior instruction. G2 status: **WARNING for exploratory G3/G4 work; physical-unit interpretation and formal G5 inference remain limited**.

## Common-cell counts

{_markdown_table(env_counts, ['analysis_subset', 'valid_cells', 'coverage_of_cropland'])}

Full per-variable counts are in `tables/common_valid_cells.csv`.

## Exposure comparison

- Common cropland cells with both area-average exposures: {exposure_comparison['common_cells']:,}.
- Zero fraction on that common set: Aa_20yr={exposure_comparison['Aa_20yr_zero_fraction']:.2%}; Aa_2020={exposure_comparison['Aa_2020_zero_fraction']:.2%}.
- Pearson r={exposure_comparison['pearson_common_average']:.4f}; Spearman rho={exposure_comparison['spearman_common_average']:.4f}.
- Cells where Aa_20yr is zero and Aa_2020 positive: {exposure_comparison['Aa_20yr_zero_Aa_2020_positive_count']:,}. This is a conditional consistency diagnostic only; exact 20-year dates and averaging construction are unknown.

## G3

The core candidate vector remains Elevation, Cur, TWI, and 2020 mean LST, with SoilType strata. Optional layers are reported in `tables/environment_screen.csv`; they are not added to the primary vector because source-scale/time provenance is incomplete and the vegetation indices may be redundant. G3 status: **WARNING; the core vector is testable on the common grid, while metadata and shared RF/RUSLE predictor lineage remain incomplete**.

## Databases

- `databases/environment_erosion.parquet` contains cell IDs, coordinates, core/optional environmental layers, and both erosion exposures under area-average and nearest-neighbour resampling.
- `databases/health.parquet` contains the eight health prediction surfaces and a provisional `SHI_candidate`. It is separate from the table read by G4 clustering and matching.

No health outcome was used for covariate selection, clustering, or matching. G4 diagnostics remain preliminary; candidate settings are not a locked design.
"""
    (run_dir / "g2_g3_report.md").write_text(report, encoding="utf-8")

    print(f"Prepared cropland analysis run: {run_dir}")
    return run_dir


def _write_cluster_map(run_dir: Path, frame: pd.DataFrame, labels: np.ndarray, cluster_count: int) -> None:
    template_path = run_dir / "maps" / "aligned" / "cropland_mask_binary.tif"
    output_path = run_dir / "maps" / f"environment_groups_k{cluster_count}.tif"
    with rasterio.open(template_path) as template:
        profile = template.profile.copy()
        bounds = template.bounds
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)
        profile.update(dtype="int32", nodata=0, compress="deflate", count=1)
        raster = np.zeros((template.height, template.width), dtype=np.int32)
    cell_ids = frame["cell_id"].to_numpy(dtype=np.int64)
    if cell_ids.size and (cell_ids.min() < 0 or cell_ids.max() >= raster.size):
        raise ValueError("Cell IDs fall outside the analysis grid")
    raster.ravel()[cell_ids] = labels.astype(np.int32) + 1
    with rasterio.open(output_path, "w", **profile) as destination:
        destination.write(raster, 1)

    _plot_cluster_map(run_dir, cluster_count)


def _plot_cluster_map(run_dir: Path, cluster_count: int) -> None:
    """Render stored group IDs as soil-stratified small-multiple maps."""
    _configure_figure_style()
    map_path = run_dir / "maps" / f"environment_groups_k{cluster_count}.tif"
    soil_path = run_dir / "maps" / "aligned" / "SoilType.tif"
    centers_path = run_dir / "tables" / f"environment_centers_k{cluster_count}.csv"
    with rasterio.open(map_path) as dataset:
        group_values = dataset.read(1, masked=True)
        bounds = dataset.bounds
        res_x, res_y = dataset.res
    if not soil_path.is_file():
        figure, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
        image = axis.imshow(np.ma.masked_equal(group_values, 0), cmap="tab20", interpolation="nearest")
        axis.set_title(f"环境组空间格局（未提供土壤类型栅格，k = {cluster_count}）")
        axis.set_axis_off()
        figure.colorbar(image, ax=axis, label="环境组编号")
        figure.savefig(run_dir / "figures" / f"environment_groups_k{cluster_count}.png", dpi=320)
        plt.close(figure)
        return
    with rasterio.open(soil_path) as dataset:
        soil_values = dataset.read(1, masked=True)
    centers = pd.read_csv(centers_path)
    cmap = ListedColormap(GROUP_COLORS[:cluster_count], name=f"envmatch_groups_{cluster_count}")
    norm = BoundaryNorm(np.arange(0.5, cluster_count + 1.5), cmap.N)
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 5.6), constrained_layout=True)
    figure.suptitle(f"土壤类型分层的环境分组空间格局（每类 k = {cluster_count}）",
                    x=0.02, y=1.055, ha="left", fontsize=16, fontweight="semibold", color=FIGURE_COLORS["ink"])
    figure.text(0.02, 1.005, "分面为 26 冲积土、27 潮土、42 脱潮土；各面板独立聚类，组号不跨土类解释。",
                ha="left", va="top", fontsize=9, color=FIGURE_COLORS["muted"])
    map_images = []
    for axis, soil_code in zip(axes, (26, 27, 42), strict=True):
        soil_mask = (soil_values.data == soil_code) & ~np.ma.getmaskarray(soil_values)
        crop_mask = soil_mask & (group_values.data > 0) & ~np.ma.getmaskarray(group_values)
        rows, columns = np.where(crop_mask)
        if not rows.size:
            axis.text(0.5, 0.5, "没有有效像元", ha="center", va="center", transform=axis.transAxes)
            axis.set_axis_off()
            continue
        r0, r1 = max(int(rows.min()) - 2, 0), min(int(rows.max()) + 3, group_values.shape[0])
        c0, c1 = max(int(columns.min()) - 2, 0), min(int(columns.max()) + 3, group_values.shape[1])
        stored_ids = group_values.data[r0:r1, c0:c1]
        selected_soil = soil_mask[r0:r1, c0:c1]
        local_groups = np.zeros(stored_ids.shape, dtype=np.int16)
        strata = centers.loc[pd.to_numeric(centers["soil_type"], errors="coerce").eq(soil_code)]
        for record in strata.itertuples(index=False):
            local_groups[(stored_ids == int(record.environment_group)) & selected_soil] = int(record.local_cluster) + 1
        shown = np.ma.masked_equal(local_groups, 0)
        west = (bounds.left + c0 * res_x) / 1000
        east = (bounds.left + c1 * res_x) / 1000
        north = (bounds.top - r0 * res_y) / 1000
        south = (bounds.top - r1 * res_y) / 1000
        extent = (west, east, south, north)
        image = axis.imshow(shown, cmap=cmap, norm=norm, interpolation="nearest", extent=extent,
                            origin="upper", rasterized=True)
        map_images.append(image)
        axis.set_title(f"{soil_code}｜{SOIL_TYPE_LABELS[soil_code]}", loc="left", fontsize=11,
                       fontweight="semibold", pad=8)
        axis.set_aspect("equal")
        axis.set_facecolor(FIGURE_COLORS["paper"])
        axis.set_xlabel("东向坐标（千米）", fontsize=8)
        axis.set_ylabel("北向坐标（千米）", fontsize=8)
        axis.tick_params(labelsize=7, length=0)
        axis.grid(color="white", linewidth=0.6, alpha=0.9, zorder=2)
        axis.set_axisbelow(True)

        # A local scale bar and north arrow keep each cropped soil panel interpretable.
        span_x, span_y = east - west, north - south
        candidates = [1, 2, 5, 10, 20, 50]
        scale_km = min(candidates, key=lambda value: abs(value - span_x * 0.2))
        scale_x, scale_y = west + span_x * 0.06, south + span_y * 0.07
        axis.plot([scale_x, scale_x + scale_km], [scale_y, scale_y], color="white", linewidth=4.5, zorder=5)
        axis.plot([scale_x, scale_x + scale_km], [scale_y, scale_y], color=FIGURE_COLORS["ink"], linewidth=2.2, zorder=6)
        axis.text(scale_x + scale_km / 2, scale_y + span_y * 0.025, f"{scale_km:g} 千米",
                  ha="center", va="bottom", fontsize=7, color=FIGURE_COLORS["ink"], zorder=6,
                  bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.88})
        axis.annotate("N", xy=(0.93, 0.91), xytext=(0.93, 0.78), xycoords="axes fraction",
                      ha="center", va="center", fontsize=9, fontweight="bold", color=FIGURE_COLORS["ink"],
                      arrowprops={"arrowstyle": "-|>", "color": FIGURE_COLORS["ink"], "lw": 1.2}, zorder=7)

    if map_images:
        colorbar = figure.colorbar(map_images[0], ax=axes.tolist(), orientation="horizontal", pad=0.13, fraction=0.065,
                                   ticks=np.arange(1, cluster_count + 1), boundaries=np.arange(0.5, cluster_count + 1.5))
        colorbar.set_label("土类内环境组编号（离散类别）", labelpad=5)
        colorbar.outline.set_edgecolor("#C7D1D5")
    figure.savefig(run_dir / "figures" / f"environment_groups_k{cluster_count}.png", dpi=320, bbox_inches="tight")
    plt.close(figure)


def _run_feasibility(run_dir: Path, config: dict[str, Any]) -> None:
    """Run environment-only G4 feasibility candidates and save diagnostics."""
    source = pd.read_parquet(run_dir / "databases" / "environment_erosion.parquet")
    required = [
        "cell_id", "row", "col", "x", "y", "SoilType", *CORE_ENVIRONMENT,
        "Aa_20yr", "Aa_2020", "Aa_20yr_nearest", "Aa_2020_nearest",
    ]
    missing = [column for column in required if column not in source]
    if missing:
        raise ValueError(f"Environment/erosion database is missing G4 fields: {', '.join(missing)}")

    numeric = source.loc[:, [*CORE_ENVIRONMENT, "Aa_20yr", "Aa_2020"]].apply(
        pd.to_numeric, errors="coerce"
    )
    finite = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
    soil = pd.to_numeric(source["SoilType"], errors="coerce")
    valid_soil = soil.isin(SOIL_CODES).to_numpy()
    valid_exposure = (numeric[["Aa_20yr", "Aa_2020"]].to_numpy(dtype=float) >= 0).all(axis=1)
    included = finite & valid_soil & valid_exposure
    analysis_frame = source.loc[included, required].copy().reset_index(drop=True)
    if analysis_frame.empty:
        raise ValueError("No complete cropland cells remain for G4 after core environment, soil, and exposure filters")
    analysis_frame["SoilType"] = soil.loc[included].to_numpy()

    analysis_config = config.get("analysis", {})
    matching_config = config.get("matching", {})
    cluster_counts = [
        int(value) for value in analysis_config.get("cluster_counts_per_soil_type", CLUSTER_COUNTS)
    ]
    radii = [
        float(value) for value in analysis_config.get("candidate_geographic_radii_m", RADIUS_CANDIDATES_M)
    ]
    base_environment_limit = float(
        analysis_config.get("candidate_environment_distance_z_euclidean", ENVIRONMENT_DISTANCE)
    )
    base_rank_gap = float(analysis_config.get("candidate_exposure_rank_gap", EXPOSURE_RANK_GAP))
    if not cluster_counts or any(value < 1 for value in cluster_counts):
        raise ValueError("G4 cluster candidates must be positive integers")
    if not radii or any(value <= 0 or not np.isfinite(value) for value in radii):
        raise ValueError("G4 geographic radius candidates must be finite and positive")

    matrix, metadata = build_environment_vectors(
        analysis_frame,
        continuous=CORE_ENVIRONMENT,
        categorical=["SoilType"],
        weights=matching_config.get("variable_weights", {}),
    )
    environment_columns = [f"__match_env_{index}" for index in range(matrix.shape[1])]
    for index, column in enumerate(environment_columns):
        analysis_frame[column] = matrix[:, index]
    soil_strata = metadata["strata"]["SoilType"].to_numpy()

    quality_rows: list[dict[str, object]] = []
    group_rows: list[pd.DataFrame] = []
    clustered_frames: dict[int, pd.DataFrame] = {}
    rng = np.random.default_rng(RANDOM_STATE)
    for cluster_count in cluster_counts:
        labels, centers = cluster_environments(
            matrix, soil_strata, n_clusters=cluster_count, random_state=RANDOM_STATE
        )
        stable_labels, _ = cluster_environments(
            matrix, soil_strata, n_clusters=cluster_count, random_state=RANDOM_STATE + 1
        )
        sample_size = min(2000, len(matrix))
        sample_indices = rng.choice(len(matrix), size=sample_size, replace=False)
        sampled_labels = labels[sample_indices]
        silhouette: float | None = None
        if 1 < np.unique(sampled_labels).size < sample_size:
            try:
                silhouette = float(silhouette_score(matrix[sample_indices], sampled_labels))
            except ValueError:
                silhouette = None

        cluster_frame = analysis_frame.copy()
        cluster_frame["environment_group"] = labels
        clustered_frames[cluster_count] = cluster_frame
        group_counts = cluster_frame.groupby("environment_group", sort=True).size()
        realized_per_soil = (
            cluster_frame.groupby("SoilType", sort=True)["environment_group"].nunique().to_dict()
        )
        quality_rows.append(
            {
                "candidate_k_per_soil_type": cluster_count,
                "total_environment_groups": int(group_counts.size),
                "valid_cells": int(len(cluster_frame)),
                "min_group_cells": int(group_counts.min()),
                "median_group_cells": float(group_counts.median()),
                "max_group_cells": int(group_counts.max()),
                "sampled_silhouette": silhouette,
                "seed_stability_ARI": float(adjusted_rand_score(labels, stable_labels)),
                "realized_clusters_by_soil": json.dumps(realized_per_soil, ensure_ascii=False),
            }
        )

        group_environment = (
            cluster_frame.groupby(["environment_group", "SoilType"], sort=True)
            .agg(
                n_cells=("cell_id", "size"),
                **{f"mean_{name}": (name, "mean") for name in CORE_ENVIRONMENT},
                **{f"sd_{name}": (name, "std") for name in CORE_ENVIRONMENT},
            )
            .reset_index()
        )
        for exposure_column in ("Aa_20yr", "Aa_2020", "Aa_20yr_nearest", "Aa_2020_nearest"):
            exposure_summary = summarize_erosion(
                cluster_frame,
                "environment_group",
                exposure_column,
            )
            exposure_summary["exposure"] = exposure_column
            group_rows.append(
                group_environment.merge(exposure_summary, on="environment_group", how="left").assign(
                    candidate_k_per_soil_type=cluster_count
                )
            )
        centers = centers.rename(
            columns={f"center_{index}": f"center_{name}" for index, name in enumerate(CORE_ENVIRONMENT)}
        )
        centers["candidate_k_per_soil_type"] = cluster_count
        centers.to_csv(run_dir / "tables" / f"environment_centers_k{cluster_count}.csv", index=False)
        _write_cluster_map(run_dir, cluster_frame, labels, cluster_count)

    quality = pd.DataFrame(quality_rows)
    quality.to_csv(run_dir / "tables" / "g4_cluster_quality.csv", index=False)
    pd.concat(group_rows, ignore_index=True).to_csv(
        run_dir / "tables" / "environment_groups.csv", index=False
    )

    match_settings: list[dict[str, object]] = []
    baseline_k = 8 if 8 in cluster_counts else cluster_counts[0]
    baseline_radius = 500.0 if 500.0 in radii else radii[len(radii) // 2]
    for cluster_count in cluster_counts:
        for exposure in ("Aa_20yr", "Aa_2020"):
            match_settings.append(
                {
                    "candidate_k_per_soil_type": cluster_count,
                    "radius_m": baseline_radius,
                    "exposure": exposure,
                    "exposure_column": exposure,
                    "environment_limit": base_environment_limit,
                    "rank_gap": base_rank_gap,
                    "exposure_resampling": "area_average",
                    "setting_type": "baseline",
                }
            )
    for radius in radii:
        if radius == baseline_radius:
            continue
        for exposure in ("Aa_20yr", "Aa_2020"):
            match_settings.append(
                {
                    "candidate_k_per_soil_type": baseline_k,
                    "radius_m": radius,
                    "exposure": exposure,
                    "exposure_column": exposure,
                    "environment_limit": base_environment_limit,
                    "rank_gap": base_rank_gap,
                    "exposure_resampling": "area_average",
                    "setting_type": "baseline",
                }
            )
    sensitivity_settings = (
        ("environment_caliper_sensitivity", min(1.0, base_environment_limit * 1.5), base_rank_gap),
        ("rank_gap_sensitivity", base_environment_limit, max(0.05, base_rank_gap * 0.8)),
    )
    for setting_type, environment_limit, rank_gap in sensitivity_settings:
        for exposure in ("Aa_20yr", "Aa_2020"):
            match_settings.append(
                {
                    "candidate_k_per_soil_type": baseline_k,
                    "radius_m": baseline_radius,
                    "exposure": exposure,
                    "exposure_column": exposure,
                    "environment_limit": environment_limit,
                    "rank_gap": rank_gap,
                    "exposure_resampling": "area_average",
                    "setting_type": setting_type,
                }
            )
    for exposure in ("Aa_20yr", "Aa_2020"):
        match_settings.append(
            {
                "candidate_k_per_soil_type": baseline_k,
                "radius_m": baseline_radius,
                "exposure": exposure,
                "exposure_column": f"{exposure}_nearest",
                "environment_limit": base_environment_limit,
                "rank_gap": base_rank_gap,
                "exposure_resampling": "nearest",
                "setting_type": "nearest_resampling_sensitivity",
            }
        )

    candidate_rows: list[dict[str, object]] = []
    balance_frames: list[pd.DataFrame] = []
    diagnostics_rows: list[dict[str, object]] = []
    for position, settings in enumerate(match_settings, start=1):
        cluster_count = int(settings["candidate_k_per_soil_type"])
        cluster_frame = clustered_frames[cluster_count].copy()
        exposure_column = str(settings["exposure_column"])
        ranked = add_within_group_exposure_rank(
            cluster_frame,
            exposure_column=exposure_column,
            group_columns=["environment_group", "SoilType"],
            output_column="__exposure_rank",
        )
        print(
            f"G4 candidate {position}/{len(match_settings)}: k={cluster_count}, "
            f"radius={settings['radius_m']:.0f} m, exposure={settings['exposure']} ({settings['exposure_resampling']})"
        )
        pairs, diagnostics = match_cells(
            ranked,
            environment_columns=environment_columns,
            group_col="environment_group",
            erosion_col="__exposure_rank",
            soil_type_col="SoilType",
            x_col="x",
            y_col="y",
            distance_limit=float(settings["radius_m"]),
            environment_distance_limit=float(settings["environment_limit"]),
            erosion_gap=float(settings["rank_gap"]),
            allow_reuse=False,
        )
        identifier = (
            f"k{cluster_count}_r{int(float(settings['radius_m']))}_"
            f"{settings['setting_type']}_{settings['exposure']}"
        )
        actual_values = ranked.set_index("cell_id")[exposure_column]
        pairs["exposure_rank_high"] = pairs.pop("erosion_high")
        pairs["exposure_rank_low"] = pairs.pop("erosion_low")
        pairs["exposure_high_actual"] = pairs["high_erosion_cell_id"].map(actual_values)
        pairs["exposure_low_actual"] = pairs["low_erosion_cell_id"].map(actual_values)
        pairs["exposure_difference_actual"] = pairs["exposure_high_actual"] - pairs["exposure_low_actual"]
        for name, value in settings.items():
            pairs[name] = value
        pair_path = run_dir / "matched_pairs" / f"{identifier}.parquet"
        pairs.to_parquet(pair_path, index=False)

        balance_input = pairs.rename(
            columns={"exposure_rank_high": "erosion_high", "exposure_rank_low": "erosion_low"}
        )
        balance = assess_balance(
            ranked,
            balance_input,
            CORE_ENVIRONMENT,
            erosion_col="__exposure_rank",
        )
        for name in ("candidate_k_per_soil_type", "radius_m", "exposure", "exposure_resampling", "setting_type"):
            balance[name] = settings[name] if name in settings else settings.get("exposure_resampling")
        balance.to_csv(run_dir / "tables" / f"balance_{identifier}.csv", index=False)
        if not balance.empty:
            balance_frames.append(balance)

        failures = diagnostics["failure_counts"]
        row: dict[str, object] = {
            **settings,
            **{key: value for key, value in diagnostics.items() if key != "failure_counts"},
            "mean_environmental_distance": float(pairs["environmental_distance"].mean()) if not pairs.empty else np.nan,
            "mean_geographic_distance_m": float(pairs["geographic_distance"].mean()) if not pairs.empty else np.nan,
            "median_actual_exposure_difference": float(pairs["exposure_difference_actual"].median()) if not pairs.empty else np.nan,
            "max_abs_matched_smd": float(balance["matched_smd"].abs().max()) if not balance.empty else np.nan,
            **{f"failure_{key}": value for key, value in failures.items()},
            "pairs_file": str(pair_path.relative_to(run_dir)),
        }
        candidate_rows.append(row)
        diagnostics_rows.append({"candidate_id": identifier, **row})

    candidate_summary = pd.DataFrame(candidate_rows)
    candidate_summary["matched_high_fraction"] = np.where(
        candidate_summary["potential_high_cells"] > 0,
        candidate_summary["matched_pairs"] / candidate_summary["potential_high_cells"],
        np.nan,
    )
    candidate_summary.to_csv(run_dir / "tables" / "g4_candidate_summary.csv", index=False)
    if balance_frames:
        pd.concat(balance_frames, ignore_index=True).to_csv(
            run_dir / "tables" / "g4_balance_summary.csv", index=False
        )
    (run_dir / "diagnostics" / "g4_matching.json").write_text(
        json.dumps(diagnostics_rows, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    base_results = candidate_summary.loc[candidate_summary["setting_type"] == "baseline"]
    quality_table = _markdown_table(
        quality,
        [
            "candidate_k_per_soil_type", "total_environment_groups", "valid_cells", "min_group_cells",
            "median_group_cells", "max_group_cells", "sampled_silhouette", "seed_stability_ARI",
        ],
    )
    candidate_table = _markdown_table(
        base_results,
        [
            "candidate_k_per_soil_type", "radius_m", "exposure", "matched_pairs", "potential_high_cells",
            "matched_high_fraction", "mean_environmental_distance", "mean_geographic_distance_m",
            "median_actual_exposure_difference", "max_abs_matched_smd",
        ],
    )
    report = f"""# G4 matching feasibility (exploratory)

## Data and method

G4 used only `databases/environment_erosion.parquet`. Health data were not read. The complete-case analysis contains {len(analysis_frame):,} cropland cells with Elevation, Cur, TWI, LST, valid SoilType (26, 27, or 42), and both area-average erosion rasters. SoilType 85 and other codes were excluded from primary grouping. Erosion was ranked within each environment group and soil type; a percentile-rank gap makes the candidate contrast independent of unknown physical units.

The environment vector is the z-standardized four-variable core. Candidate cluster counts are per soil type. The baseline matching grid uses a {base_environment_limit:g} standardized Euclidean caliper, rank gap {base_rank_gap:g}, one-to-one matching without reuse, and geographic radii {radii} m. Nearest-neighbour resampling and alternate caliper/rank-gap settings are sensitivity runs. None of these settings is locked.

## Environment group quality

{quality_table}

Group summaries for both exposures under both resampling methods are in `tables/environment_groups.csv`. Cluster rasters and figures are in `maps/` and `figures/`; label 0 is outside the G4 complete-case set.

## Baseline matching candidates

{candidate_table}

All candidate combinations, failure counts, sensitivity settings, and pair file paths are in `tables/g4_candidate_summary.csv` and `diagnostics/g4_matching.json`. Each pair file records actual raster values and rank contrasts. Per-candidate covariate balance is in `tables/balance_*.csv`; the combined table is `tables/g4_balance_summary.csv`.

## Gate status and review needed

G4 is **exploratory / not locked**. G2/G3 remain warnings because the erosion units, exact 20-year dates, averaging construction, and shared predictor lineage are undocumented. Candidate settings have not been reviewed. The health maps are prediction surfaces with incomplete model provenance, and `SHI_candidate` is not confirmed. No G5 health regression or PCNM comparison was run. Review the candidate thresholds, group counts, matched coverage, balance, and area-average versus nearest sensitivity before locking a design or starting G5.
"""
    (run_dir / "g4_report.md").write_text(report, encoding="utf-8")
    print(f"G4 feasibility diagnostics saved: {run_dir}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare Wenxian cropland G2-G4 diagnostics")
    parser.add_argument("--config", default="config/default.yaml", help="Project configuration YAML")
    parser.add_argument("--run-id", default=None, help="Optional output run identifier")
    args = parser.parse_args(argv)

    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    mapping, _ = load_data_mapping(config, config_path)
    run_dir = _prepare_run(
        project_root(config_path), config_path, config, mapping, run_id=args.run_id
    )
    _run_feasibility(run_dir, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
