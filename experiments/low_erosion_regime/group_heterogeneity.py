"""Run Phase H environmental-group response and spatial diagnostics."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from experiments.low_erosion_regime.environment_heterogeneity import (
    compare_group_responses_to_baseline,
    summarize_group_responses,
    validate_group_spatial_performance,
)
from experiments.low_erosion_regime.figures import (
    EXPOSURES,
    EXPOSURE_LABELS,
    OUTCOMES,
    OUTCOME_LABELS,
    _save,
)
from experiments.low_erosion_regime.matched_units import SCHEME_LABELS, _source_populations
from experiments.low_erosion_regime.response_models import _block_bootstrap_slope
from experiments.low_erosion_regime.run import _load_inputs


SOIL_TYPE_LABELS = {26: "冲积土", 27: "潮土", 42: "脱潮土", 85: "江及河"}
GROUP_NAMES = {
    "soil_type_group": ("土壤类型", "土类"),
    "kmeans_terrain_k4": ("地形 K-means（k=4）", "K-means k=4"),
    "kmeans_terrain_k6": ("地形 K-means（k=6）", "K-means k=6"),
    "kmeans_terrain_k8": ("地形 K-means（k=8）", "K-means k=8"),
    "hierarchical_ward_v5": ("Ward 层次分组（k=5）", "Ward"),
    "gradient_Elevation": ("高程梯度分组", "高程"),
    "gradient_TWI": ("地形湿度指数梯度分组", "TWI"),
    "gradient_LST": ("地表温度梯度分组", "LST"),
    "gradient_NDWI": ("NDWI 梯度分组", "NDWI"),
    "gradient_NDVI": ("NDVI 梯度分组", "NDVI"),
    "gradient_Precipitation": ("降水梯度分组", "降水"),
}
CORE_GROUP_COLUMNS = (
    "soil_type_group", "kmeans_terrain_k6", "hierarchical_ward_v5",
    "gradient_TWI", "gradient_LST",
)
ENVIRONMENT_VARIABLES = (
    "Elevation", "Cur", "TWI", "LST", "NDWI", "NDBSI", "EVI", "NDVI", "Precipitation",
)
SCHEME_ORDER = tuple(SCHEME_LABELS)
FILTER_SCHEMES = tuple(scheme for scheme in SCHEME_ORDER if scheme != "S0_all")
PLOT_STYLE = {
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.edgecolor": "#D6E0E5", "axes.labelcolor": "#263B49", "text.color": "#263B49",
    "xtick.color": "#526875", "ytick.color": "#526875", "grid.color": "#E7EDF0",
    "font.family": "Noto Sans CJK JP", "font.size": 8.5, "axes.titlesize": 10,
    "axes.labelsize": 8.5, "legend.frameon": False, "svg.fonttype": "none",
}


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else np.nan


def _soil_name(code: int) -> str:
    return SOIL_TYPE_LABELS.get(code, "其他土类")


def _short_group_label(group_column: str, value: object) -> str:
    text = str(value)
    if group_column == "soil_type_group":
        if text.startswith("soil_") and text[5:].isdigit():
            code = int(text[5:])
            return f"{code} { _soil_name(code) }"
        return text
    if text.startswith("soil_") and "_cluster_" in text:
        prefix, cluster = text.split("_cluster_", 1)
        code = int(prefix[5:]) if prefix[5:].isdigit() else -1
        return f"{code}{_soil_name(code)}·组{cluster}"
    if ":" in text:
        prefix, subgroup = text.split(":", 1)
        if prefix.isdigit():
            code = int(prefix)
            return f"{code}{_soil_name(code)}·{subgroup}"
    return text


def _display_group_column(group_column: str) -> str:
    return GROUP_NAMES.get(group_column, (group_column, group_column))[1]


def _valid_cells(cells: pd.DataFrame) -> pd.DataFrame:
    if "valid_cell" not in cells:
        return cells.copy()
    return cells.loc[cells["valid_cell"].fillna(False).astype(bool)].copy()


def summarize_cell_group_retention(
    cells: pd.DataFrame,
    group_labels: pd.DataFrame,
    thresholds: dict[str, float],
    *,
    group_columns: tuple[str, ...] | list[str],
) -> pd.DataFrame:
    """Count valid and above-threshold cells within every fixed environment group."""
    if cells["cell_id"].duplicated().any() or group_labels["cell_id"].duplicated().any():
        raise ValueError("cell and group-label tables must have one row per cell_id")
    missing = sorted(set(group_columns) - set(group_labels.columns))
    if missing:
        raise ValueError(f"environment labels are missing group columns: {missing}")
    frame = _valid_cells(cells).merge(
        group_labels[["cell_id", *group_columns]], on="cell_id", how="left", validate="one_to_one"
    )
    rows: list[dict[str, Any]] = []
    for exposure, threshold in thresholds.items():
        if exposure not in frame:
            raise ValueError(f"cells are missing exposure column {exposure}")
        finite = pd.to_numeric(frame[exposure], errors="coerce")
        selected = frame.loc[np.isfinite(finite)].copy()
        selected["_retained"] = finite.loc[selected.index].gt(float(threshold)).to_numpy()
        for group_column in group_columns:
            selected["_group_label"] = selected[group_column].astype("string").fillna("未分组")
            for group_label, part in selected.groupby("_group_label", sort=True, observed=True):
                n_valid = len(part)
                n_retained = int(part["_retained"].sum())
                rows.append({
                    "exposure": exposure, "threshold_original": float(threshold),
                    "group_column": group_column, "group_label": str(group_label),
                    "group_column_label": GROUP_NAMES.get(group_column, (group_column, group_column))[0],
                    "group_label_display": _short_group_label(group_column, group_label),
                    "n_valid_cells": n_valid, "n_retained_cells": n_retained,
                    "n_low_value_cells": n_valid - n_retained,
                    "retained_fraction": _safe_ratio(n_retained, n_valid),
                    "low_value_excluded_fraction": _safe_ratio(n_valid - n_retained, n_valid),
                    "outcome_guided_selection": True,
                })
    return pd.DataFrame.from_records(rows)


def _pair_group_support(
    pairs: pd.DataFrame,
    group_labels: pd.DataFrame,
    cell_retention: pd.DataFrame,
    cells: pd.DataFrame,
    *,
    method: str,
    exposure: str,
    scheme: str,
    group_columns: tuple[str, ...] | list[str],
) -> pd.DataFrame:
    label_index = group_labels.set_index("cell_id")
    high_ids = pd.to_numeric(pairs["high_erosion_cell_id"], errors="coerce")
    low_ids = pd.to_numeric(pairs["low_erosion_cell_id"], errors="coerce")
    endpoint_ids = pd.Index(pd.concat([high_ids, low_ids], ignore_index=True).dropna().unique())
    endpoint_labels = label_index.reindex(endpoint_ids)
    valid = _valid_cells(cells)
    environment_sd = {
        name: float(pd.to_numeric(valid[name], errors="coerce").std(ddof=0))
        for name in ENVIRONMENT_VARIABLES if name in valid
    }
    standardized_names = [name for name, scale in environment_sd.items() if scale > 0]
    if standardized_names:
        environment_frame = valid.set_index("cell_id")[standardized_names]
        high_environment = environment_frame.reindex(pd.Index(high_ids.to_numpy()))
        low_environment = environment_frame.reindex(pd.Index(low_ids.to_numpy()))
        high_environment.index = pairs.index
        low_environment.index = pairs.index
        scales = pd.Series({name: environment_sd[name] for name in standardized_names})
        standardized = high_environment.subtract(low_environment).abs().divide(scales, axis="columns")
    else:
        standardized = pd.DataFrame(index=pairs.index)
    if not standardized.empty:
        pairs_work = pairs.copy()
        pairs_work["_mean_abs_standardized_env_diff"] = standardized.mean(axis=1)
        pairs_work["_max_abs_standardized_env_diff"] = standardized.max(axis=1)
    else:
        pairs_work = pairs.copy()
        pairs_work["_mean_abs_standardized_env_diff"] = np.nan
        pairs_work["_max_abs_standardized_env_diff"] = np.nan

    rows: list[dict[str, Any]] = []
    for group_column in group_columns:
        high_label = high_ids.map(label_index[group_column])
        low_label = low_ids.map(label_index[group_column])
        eligible = high_label.notna() & low_label.notna()
        same_group = eligible & high_label.eq(low_label)
        endpoint_count = endpoint_labels[group_column].astype("string").fillna("未分组").value_counts()
        cell_part = cell_retention.loc[
            cell_retention["exposure"].eq(exposure) & cell_retention["group_column"].eq(group_column)
        ].set_index("group_label")
        labels_present = set(cell_part.index.astype(str)) | set(endpoint_count.index.astype(str))
        labels_present |= set(high_label.dropna().astype(str).unique()) | set(low_label.dropna().astype(str).unique())
        for group_label in sorted(labels_present):
            same = same_group & high_label.astype("string").eq(group_label)
            group_pairs = pairs_work.loc[same]
            touching = high_label.astype("string").eq(group_label) | low_label.astype("string").eq(group_label)
            if group_label in cell_part.index:
                cell_row = cell_part.loc[group_label]
            else:
                cell_row = pd.Series({
                    "n_valid_cells": 0, "n_retained_cells": 0, "n_low_value_cells": 0,
                    "retained_fraction": np.nan, "low_value_excluded_fraction": np.nan,
                    "threshold_original": np.nan, "group_label_display": _short_group_label(group_column, group_label),
                })
            rows.append({
                "analysis_unit": "group_population", "exposure": exposure, "method": method,
                "scheme": scheme, "scheme_label": SCHEME_LABELS.get(scheme, scheme),
                "group_column": group_column,
                "group_column_label": GROUP_NAMES.get(group_column, (group_column, group_column))[0],
                "group_label": group_label,
                "group_label_display": cell_row.get("group_label_display", _short_group_label(group_column, group_label)),
                "threshold_original": cell_row.get("threshold_original", np.nan),
                "n_valid_cells": int(cell_row.get("n_valid_cells", 0)),
                "n_retained_cells": int(cell_row.get("n_retained_cells", 0)),
                "n_low_value_cells": int(cell_row.get("n_low_value_cells", 0)),
                "retained_fraction": cell_row.get("retained_fraction", np.nan),
                "low_value_excluded_fraction": cell_row.get("low_value_excluded_fraction", np.nan),
                "n_pairs_touching_group": int(touching.sum()),
                "n_pairs_same_group": int(same.sum()),
                "same_group_pair_fraction": _safe_ratio(int(same.sum()), int(eligible.sum())),
                "n_unique_matched_endpoints": int(endpoint_count.get(group_label, 0)),
                "n_unique_same_group_endpoints": int(pd.concat([
                    group_pairs.get("high_erosion_cell_id", pd.Series(dtype=float)),
                    group_pairs.get("low_erosion_cell_id", pd.Series(dtype=float)),
                ], ignore_index=True).dropna().nunique()),
                "mean_abs_standardized_environment_difference": pd.to_numeric(
                    group_pairs["_mean_abs_standardized_env_diff"], errors="coerce"
                ).mean(),
                "median_abs_standardized_environment_difference": pd.to_numeric(
                    group_pairs["_mean_abs_standardized_env_diff"], errors="coerce"
                ).median(),
                "max_abs_standardized_environment_difference": pd.to_numeric(
                    group_pairs["_max_abs_standardized_env_diff"], errors="coerce"
                ).mean(),
                "mean_environmental_distance": pd.to_numeric(group_pairs.get("environmental_distance"), errors="coerce").mean(),
                "median_environmental_distance": pd.to_numeric(group_pairs.get("environmental_distance"), errors="coerce").median(),
                "mean_geographic_distance_m": pd.to_numeric(group_pairs.get("geographic_distance"), errors="coerce").mean(),
                "median_geographic_distance_m": pd.to_numeric(group_pairs.get("geographic_distance"), errors="coerce").median(),
                "mean_delta_log1p_erosion": pd.to_numeric(group_pairs.get("delta_log1p_erosion"), errors="coerce").mean(),
                "median_delta_log1p_erosion": pd.to_numeric(group_pairs.get("delta_log1p_erosion"), errors="coerce").median(),
                "n_spatial_blocks": int(group_pairs["spatial_block_id"].nunique()) if "spatial_block_id" in group_pairs else np.nan,
                "outcome_guided_selection": True,
            })
    return pd.DataFrame.from_records(rows)


def summarize_group_population_support(
    cells: pd.DataFrame,
    group_labels: pd.DataFrame,
    populations: dict[tuple[str, str, str], pd.DataFrame],
    *,
    thresholds: dict[str, float],
    group_columns: tuple[str, ...] | list[str],
) -> pd.DataFrame:
    """Combine full-population low-value retention with group-specific pair support."""
    cell_retention = summarize_cell_group_retention(
        cells, group_labels, thresholds, group_columns=group_columns
    )
    rows = []
    for (method, exposure, scheme), pairs in populations.items():
        if pairs.empty:
            continue
        rows.append(_pair_group_support(
            pairs, group_labels, cell_retention, cells, method=method, exposure=exposure,
            scheme=scheme, group_columns=group_columns,
        ))
    result = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return _merge_group_population_baseline(result)


def _merge_group_population_baseline(result: pd.DataFrame) -> pd.DataFrame:
    if result.empty:
        return result
    key = ["exposure", "method", "group_column", "group_label"]
    metric_names = [
        "n_pairs_same_group", "n_unique_matched_endpoints", "n_unique_same_group_endpoints",
        "mean_abs_standardized_environment_difference", "mean_environmental_distance", "mean_geographic_distance_m",
    ]
    baseline = result.loc[result["scheme"].eq("S0_all"), key + metric_names].rename(
        columns={column: f"baseline_{column}" for column in metric_names}
    )
    result = result.merge(baseline, on=key, how="left", validate="many_to_one")
    for metric, baseline_metric, ratio_name in (
        ("n_pairs_same_group", "baseline_n_pairs_same_group", "pair_retention_vs_S0"),
        ("n_unique_matched_endpoints", "baseline_n_unique_matched_endpoints", "endpoint_retention_vs_S0"),
        ("n_unique_same_group_endpoints", "baseline_n_unique_same_group_endpoints", "same_group_endpoint_retention_vs_S0"),
    ):
        denominator = pd.to_numeric(result[baseline_metric], errors="coerce").to_numpy(float)
        numerator = pd.to_numeric(result[metric], errors="coerce").to_numpy(float)
        result[ratio_name] = np.divide(
            numerator, denominator, out=np.full(len(result), np.nan), where=denominator > 0
        )
    return result


def bootstrap_group_pair_slopes(
    pairs: pd.DataFrame,
    group_labels: pd.DataFrame,
    *,
    group_columns: tuple[str, ...] | list[str],
    outcomes: tuple[str, ...] | list[str],
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    """Estimate within-group pair slopes with 5 km spatial-block bootstrap intervals."""
    from scipy.stats import spearmanr

    label_index = group_labels.set_index("cell_id")
    high_x = pd.to_numeric(pairs["x_high"], errors="coerce").to_numpy(float)
    high_y = pd.to_numeric(pairs["y_high"], errors="coerce").to_numpy(float)
    low_x = pd.to_numeric(pairs["x_low"], errors="coerce").to_numpy(float)
    low_y = pd.to_numeric(pairs["y_low"], errors="coerce").to_numpy(float)
    same_block = (
        np.isfinite(high_x) & np.isfinite(high_y) & np.isfinite(low_x) & np.isfinite(low_y)
        & (np.floor(high_x / 5000) == np.floor(low_x / 5000))
        & (np.floor(high_y / 5000) == np.floor(low_y / 5000))
    )
    selected = pairs.loc[same_block].copy()
    if selected.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    high_label = selected["high_erosion_cell_id"].map(label_index[group_columns[0]])
    low_label = selected["low_erosion_cell_id"].map(label_index[group_columns[0]])
    seed_offset = 0
    for group_column in group_columns:
        high_label = selected["high_erosion_cell_id"].map(label_index[group_column])
        low_label = selected["low_erosion_cell_id"].map(label_index[group_column])
        same = high_label.notna() & low_label.notna() & high_label.eq(low_label)
        work = selected.loc[same].copy()
        work["_group_label"] = high_label.loc[same].astype(str).to_numpy()
        for group_label, group in work.groupby("_group_label", sort=True):
            blocks = group.get("spatial_block_id", pd.Series(index=group.index, dtype=object)).astype("string")
            x = pd.to_numeric(group["delta_log1p_erosion"], errors="coerce")
            for outcome in outcomes:
                y_column = f"delta_{outcome}"
                if y_column not in group:
                    continue
                y = pd.to_numeric(group[y_column], errors="coerce")
                values = pd.DataFrame({"x": x, "y": y, "block": blocks}).replace([np.inf, -np.inf], np.nan).dropna()
                x_values, y_values = values["x"].to_numpy(float), values["y"].to_numpy(float)
                slope = float(np.polyfit(x_values, y_values, 1)[0]) if len(values) >= 3 and np.ptp(x_values) > 0 else np.nan
                rho = float(spearmanr(x_values, y_values).statistic) if len(values) >= 3 and np.ptp(x_values) > 0 and np.ptp(y_values) > 0 else np.nan
                blocks_values = values["block"].astype(str).to_numpy()
                ci_low, ci_high = _block_bootstrap_slope(
                    x_values, y_values, blocks_values, replicates=replicates,
                    seed=int(seed) + seed_offset,
                ) if len(values) else (np.nan, np.nan)
                rows.append({
                    "analysis_unit": "pair", "exposure": selected["exposure"].iloc[0] if "exposure" in selected else None,
                    "method": selected["method"].iloc[0] if "method" in selected else None,
                    "scheme": selected["scheme"].iloc[0] if "scheme" in selected else None,
                    "outcome": outcome, "group_column": group_column, "group_label": group_label,
                    "n_pairs_same_group_same_block": int(len(values)),
                    "spatial_blocks": int(values["block"].nunique()) if not values.empty else 0,
                    "spearman_rho_same_block": rho, "slope_same_block": slope,
                    "slope_block_bootstrap_ci_low": ci_low, "slope_block_bootstrap_ci_high": ci_high,
                    "block_bootstrap_replicates": int(replicates),
                    "outcome_guided_selection": True,
                })
                seed_offset += 1
    return pd.DataFrame.from_records(rows)


def summarize_group_spatial_validation(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    keys = ["exposure", "method", "scheme", "outcome", "group_column", "group_label"]
    rows = []
    for key, part in frame.groupby(keys, dropna=False, sort=False):
        ok = part.loc[part["status"].eq("ok")]
        r2 = pd.to_numeric(ok["spatial_test_r_squared"], errors="coerce").dropna()
        slopes = pd.to_numeric(ok["slope"], errors="coerce").dropna()
        rows.append({
            **dict(zip(keys, key, strict=True)),
            "n_spatial_splits": int(part["split"].nunique()),
            "n_successful_splits": int(ok["split"].nunique()),
            "n_pairs_same_group_same_block": int(pd.to_numeric(part["n_pairs_same_group_same_block"], errors="coerce").max()),
            "spatial_blocks": int(pd.to_numeric(part["spatial_blocks"], errors="coerce").max()),
            "spatial_test_r2_median": float(r2.median()) if not r2.empty else np.nan,
            "spatial_test_r2_q25": float(r2.quantile(.25)) if not r2.empty else np.nan,
            "spatial_test_r2_q75": float(r2.quantile(.75)) if not r2.empty else np.nan,
            "negative_spatial_r2_fraction": float((r2 < 0).mean()) if not r2.empty else np.nan,
            "negative_slope_fraction": float((slopes < 0).mean()) if not slopes.empty else np.nan,
            "model_family": "预先固定线性回归",
            "outcome_guided_selection": True,
        })
    result = pd.DataFrame.from_records(rows)
    baseline_keys = ["exposure", "method", "outcome", "group_column", "group_label"]
    metrics = ["n_successful_splits", "spatial_test_r2_median", "negative_slope_fraction"]
    baseline = result.loc[result["scheme"].eq("S0_all"), baseline_keys + metrics].rename(
        columns={name: f"baseline_{name}" for name in metrics}
    )
    result = result.merge(baseline, on=baseline_keys, how="left", validate="many_to_one")
    result["change_spatial_test_r2_median_vs_S0"] = result["spatial_test_r2_median"] - result["baseline_spatial_test_r2_median"]
    result["change_negative_slope_fraction_vs_S0"] = result["negative_slope_fraction"] - result["baseline_negative_slope_fraction"]
    return result


def _table_write(frame: pd.DataFrame, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_csv(directory / f"{stem}.csv", index=False, encoding="utf-8-sig")
    frame.to_parquet(directory / f"{stem}.parquet", index=False)


def _group_row_order(frame: pd.DataFrame, group_columns: tuple[str, ...] | list[str]) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    group_order = {name: index for index, name in enumerate(group_columns)}
    result = frame.loc[frame["group_column"].isin(group_columns)].drop_duplicates(
        ["group_column", "group_label"]
    ).copy()
    result["_group_order"] = result["group_column"].map(group_order).fillna(len(group_order))
    result = result.sort_values(["_group_order", "group_label"], kind="stable")
    result["row_label"] = result.apply(
        lambda row: f"{_display_group_column(row['group_column'])} · {_short_group_label(row['group_column'], row['group_label'])}",
        axis=1,
    )
    return result


def _build_figures(
    figure_root: Path,
    cell_retention: pd.DataFrame,
    population_support: pd.DataFrame,
    response_comparison: pd.DataFrame,
    spatial_summary: pd.DataFrame,
    slope_bootstrap: pd.DataFrame,
    *,
    core_group_columns: tuple[str, ...],
) -> list[dict[str, str]]:
    plt.rcParams.update(PLOT_STYLE)
    sns.set_theme(style="whitegrid", rc=PLOT_STYLE)
    figures: list[dict[str, str]] = []

    # Whole-population retention by environment group.
    plot_groups = [name for name in core_group_columns if name in cell_retention["group_column"].unique()]
    fig, axes = plt.subplots(2, 3, figsize=(19, 17), constrained_layout=True, squeeze=False)
    exposure_colors = {"Aa_20yr": "#3F788B", "Aa_2020": "#C28A43"}
    for column, group_column in enumerate(plot_groups):
        ax = axes.flat[column]
        by_exposure = {}
        group_names = set()
        for exposure in EXPOSURES:
            part = cell_retention.loc[
                cell_retention["exposure"].eq(exposure) & cell_retention["group_column"].eq(group_column)
            ].copy()
            by_exposure[exposure] = part.set_index("group_label_display")
            group_names.update(part["group_label_display"].astype(str))
        ordered = sorted(group_names)
        y_positions = np.arange(len(ordered))
        for exposure_index, exposure in enumerate(EXPOSURES):
            part = by_exposure[exposure]
            retained = np.array([part.loc[name, "retained_fraction"] if name in part.index else np.nan for name in ordered], dtype=float)
            offsets = (exposure_index - .5) * .32
            ax.barh(y_positions + offsets, retained, color=exposure_colors[exposure], height=.29,
                    edgecolor="white", linewidth=.45, label=EXPOSURE_LABELS[exposure])
        ax.set_yticks(y_positions, ordered, fontsize=7.1)
        ax.set_xlim(0, 1.08)
        ax.axvline(.5, color="#71838C", linewidth=.7, linestyle=(0, (3, 2)))
        ax.set_xlabel("阈值以上保留像元比例")
        ax.set_title(_display_group_column(group_column), loc="left", fontweight="bold")
        ax.grid(axis="x", linewidth=.5)
    for index in range(len(plot_groups), len(axes.flat)):
        axes.flat[index].set_axis_off()
    handles = [Line2D([0], [0], color=exposure_colors[name], linewidth=6, label=EXPOSURE_LABELS[name]) for name in EXPOSURES]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(.5, 1.015))
    fig.suptitle("两期侵蚀分布阈值下各环境组的像元保留比例", y=1.045, fontsize=14, fontweight="bold")
    figures.append(_save(
        fig, figure_root, "HET01_group_low_value_retention", cell_retention,
        "固定环境组内超过分布阈值的有效像元比例；颜色深浅表示该组被阈值排除的比例。分组只使用既有环境变量标签，阈值由侵蚀分布确定。",
        scientific_question="低侵蚀阈值在不同土类和环境组中各保留多少有效像元？",
        experiment="环境组低值保留与覆盖诊断", figure_type="环境组保留率图",
    ))

    # Within-group rank association change for every health outcome, exposure and filter path.
    labels = _group_row_order(response_comparison, core_group_columns)
    row_labels = labels.set_index(["group_column", "group_label"])["row_label"]
    for exposure in EXPOSURES:
        for scheme in FILTER_SCHEMES:
            selected = response_comparison.loc[
                response_comparison["method"].eq("M4")
                & response_comparison["exposure"].eq(exposure)
                & response_comparison["scheme"].eq(scheme)
                & response_comparison["group_column"].isin(core_group_columns)
            ].copy()
            if selected.empty:
                continue
            group_index = list(labels.set_index(["group_column", "group_label"]).index)
            y_labels = [row_labels.loc[key] for key in group_index]
            height = max(13.0, .205 * len(group_index) + 2.8)
            fig = plt.figure(figsize=(22, height), constrained_layout=False)
            grid = fig.add_gridspec(1, 3, width_ratios=(1, 1, .035), wspace=.16)
            axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
            cax = fig.add_subplot(grid[0, 2])
            support_floor = 30
            for ax_index, unit in enumerate(("matched_pixel", "pair")):
                ax = axes[ax_index]
                part = selected.loc[selected["analysis_unit"].eq(unit)].copy()
                part["_support_ok"] = (
                    pd.to_numeric(part["n"], errors="coerce").ge(support_floor)
                    & pd.to_numeric(part["baseline_n"], errors="coerce").ge(support_floor)
                )
                part["_display_row"] = list(zip(part["group_column"], part["group_label"], strict=True))
                matrix = part.pivot_table(index="_display_row", columns="outcome", values="change_spearman_rho_vs_S0", aggfunc="first")
                support = part.pivot_table(index="_display_row", columns="outcome", values="_support_ok", aggfunc="first")
                matrix = matrix.reindex(index=group_index, columns=list(OUTCOMES))
                support = support.reindex(index=group_index, columns=list(OUTCOMES)).astype("boolean").fillna(False)
                matrix = matrix.where(support)
                sns.heatmap(
                    matrix, ax=ax, cmap="vlag", center=0, vmin=-.65, vmax=.65,
                    mask=matrix.isna(), linewidths=.16, linecolor="#FFFFFF",
                    cbar=ax_index == 1, cbar_ax=cax if ax_index == 1 else None,
                    cbar_kws={"label": "过滤/重配后 Spearman ρ − 基线 ρ"} if ax_index == 1 else None,
                )
                ax.set_yticks(np.arange(len(y_labels)) + .5)
                if ax_index == 0:
                    ax.set_yticklabels(y_labels, fontsize=6.8)
                else:
                    ax.set_yticklabels([])
                ax.set_xticklabels([OUTCOME_LABELS.get(value, value) for value in matrix.columns], rotation=35, ha="right", fontsize=7.3)
                ax.set_xlabel("健康指标")
                ax.set_ylabel("环境组（每格需基线与当前方案均至少 30 个观测）" if ax_index == 0 else "")
                ax.set_title("唯一匹配端点" if unit == "matched_pixel" else "同组配对差值", loc="left", fontweight="bold")
                previous = None
                for position, key in enumerate(group_index):
                    current = key[0]
                    if previous is not None and current != previous:
                        ax.axhline(position, color="#526875", linewidth=.8)
                    previous = current
            title = f"{EXPOSURE_LABELS[exposure]} · M4 · {SCHEME_LABELS[scheme]}"
            fig.suptitle(f"环境组内关联相对基线的变化\n{title}", x=.04, y=.995, ha="left", fontsize=14, fontweight="bold")
            fig.subplots_adjust(left=.24, right=.96, bottom=.08, top=.94)
            source = selected[[
                "analysis_unit", "exposure", "method", "scheme", "outcome", "group_column", "group_label",
                "n", "baseline_n", "spearman_rho", "baseline_spearman_rho", "change_spearman_rho_vs_S0",
                "same_group_pair_fraction", "pair_population_share",
            ]].copy()
            figures.append(_save(
                fig, figure_root, f"HET02_group_rho_change_{exposure.removeprefix('Aa_')}_{scheme}", source,
                f"颜色表示该方案组内 Spearman ρ 减去 S0 全部原配对的组内 ρ；每像元按唯一 cell_id 计数，配对仅保留两端属于同一环境组的配对。空白表示两种样本的有效量至少一方不足 {support_floor}。筛选和重配改变了估计总体，变化不等于因果效应或纯模型改进。",
                scientific_question="低值筛选与重配后，不同环境组内九项健康响应的秩相关如何改变？",
                experiment=f"{EXPOSURE_LABELS[exposure]} {SCHEME_LABELS[scheme]} 组内秩相关比较",
                figure_type="环境组响应变化热图",
            ))

    # Group-specific spatially held-out changes from the all-pairs baseline.
    spatial_scheme = "T01_fixed_original"
    spatial_rows = spatial_summary.loc[
        spatial_summary["method"].eq("M4")
        & spatial_summary["scheme"].eq(spatial_scheme)
        & spatial_summary["group_column"].isin(core_group_columns)
    ].copy()
    if not spatial_rows.empty:
        labels_sp = _group_row_order(spatial_rows, core_group_columns)
        row_index = list(labels_sp.set_index(["group_column", "group_label"]).index)
        row_label_map = labels_sp.set_index(["group_column", "group_label"])["row_label"]
        row_text = [row_label_map.loc[key] for key in row_index]
        fig = plt.figure(figsize=(22, max(13, .205 * len(row_index) + 3)))
        grid = fig.add_gridspec(1, 3, width_ratios=(1, 1, .035), wspace=.16)
        axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
        cax = fig.add_subplot(grid[0, 2])
        for ax_index, exposure in enumerate(EXPOSURES):
            ax = axes[ax_index]
            part = spatial_rows.loc[spatial_rows["exposure"].eq(exposure)].copy()
            part["_row"] = list(zip(part["group_column"], part["group_label"], strict=True))
            matrix = part.pivot_table(index="_row", columns="outcome", values="change_spatial_test_r2_median_vs_S0", aggfunc="first")
            support = part.pivot_table(index="_row", columns="outcome", values="n_successful_splits", aggfunc="first")
            baseline_support = part.pivot_table(index="_row", columns="outcome", values="baseline_n_successful_splits", aggfunc="first")
            matrix = matrix.reindex(index=row_index, columns=list(OUTCOMES))
            enough = support.reindex(index=row_index, columns=list(OUTCOMES)).ge(5) & baseline_support.reindex(index=row_index, columns=list(OUTCOMES)).ge(5)
            matrix = matrix.where(enough)
            sns.heatmap(
                matrix, ax=ax, cmap="vlag", center=0, vmin=-.35, vmax=.35,
                mask=matrix.isna(), linewidths=.16, linecolor="#FFFFFF",
                cbar=ax_index == 1, cbar_ax=cax if ax_index == 1 else None,
                cbar_kws={"label": "中位空间测试 R²变化"} if ax_index == 1 else None,
            )
            ax.set_yticks(np.arange(len(row_text)) + .5)
            if ax_index == 0:
                ax.set_yticklabels(row_text, fontsize=6.8)
            else:
                ax.set_yticklabels([])
            ax.set_xticklabels([OUTCOME_LABELS.get(value, value) for value in OUTCOMES], rotation=35, ha="right", fontsize=7.3)
            ax.set_xlabel("健康指标")
            ax.set_ylabel("环境组（基线与过滤方案均至少 5 次有效空间划分）" if ax_index == 0 else "")
            ax.set_title(EXPOSURE_LABELS[exposure], loc="left", fontweight="bold")
            previous = None
            for position, key in enumerate(row_index):
                current = key[0]
                if previous is not None and current != previous:
                    ax.axhline(position, color="#526875", linewidth=.8)
                previous = current
        fig.suptitle("阈值后固定原环境组重配：组内空间测试表现相对基线的变化\nM4 · 5 km 整块留出 · 500 m 训练缓冲 · 固定线性模型",
                     x=.04, y=.995, ha="left", fontsize=14, fontweight="bold")
        fig.subplots_adjust(left=.24, right=.96, bottom=.08, top=.94)
        figures.append(_save(
            fig, figure_root, "HET03_group_spatial_test_change", spatial_rows,
            "颜色表示 T01_fixed_original 组内空间测试 R²中位数减去 S0 基线。配对两端须位于同一 5 km 块；训练块按留出块 500 m 缓冲剔除邻近样本；模型固定为线性基线，不按测试分数选组或选模型。空白表示基线或当前方案不足 5 次有效划分。",
            scientific_question="低值过滤并在固定环境组内重配后，组内响应能否在留出空间块保持？",
            experiment="M4 环境组空间留出比较", figure_type="空间块稳健热图",
        ))

    # Soil-stratified paired slopes with spatial-block bootstrap intervals.
    soil_slopes = slope_bootstrap.loc[
        slope_bootstrap["group_column"].eq("soil_type_group")
        & slope_bootstrap["exposure"].eq("Aa_20yr")
        & slope_bootstrap["outcome"].isin(("SHI_candidate", "OM", "SW"))
        & slope_bootstrap["scheme"].isin(("S0_all", "T01_fixed_original"))
    ].copy()
    if not soil_slopes.empty:
        fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True, constrained_layout=False)
        scheme_colors = {"S0_all": "#567687", "T01_fixed_original": "#D18B45"}
        soil_order = sorted(soil_slopes["group_label"].unique())
        for axis_index, outcome in enumerate(("SHI_candidate", "OM", "SW")):
            ax = axes[axis_index]
            part = soil_slopes.loc[soil_slopes["outcome"].eq(outcome)]
            for scheme_index, scheme in enumerate(("S0_all", "T01_fixed_original")):
                color = scheme_colors[scheme]
                group_part = part.loc[part["scheme"].eq(scheme)]
                for y_index, group_label in enumerate(soil_order):
                    row = group_part.loc[group_part["group_label"].eq(group_label)]
                    if row.empty:
                        continue
                    item = row.iloc[0]
                    y = y_index + (scheme_index - .5) * .18
                    estimate = float(item["slope_same_block"])
                    lower = float(item["slope_block_bootstrap_ci_low"])
                    upper = float(item["slope_block_bootstrap_ci_high"])
                    if np.isfinite([lower, upper]).all():
                        ax.plot([lower, upper], [y, y], color=color, linewidth=1.7, alpha=.95)
                    ax.scatter(estimate, y, s=37, color=color, edgecolor="white", linewidth=.75, zorder=3,
                               marker="o" if scheme_index == 0 else "D")
            ax.axvline(0, color="#788B94", linewidth=.8, linestyle=(0, (3, 2)))
            ax.set_yticks(np.arange(len(soil_order)), [
                f"{int(value.split('_')[1])} {_soil_name(int(value.split('_')[1]))}" if value.startswith("soil_") else value
                for value in soil_order
            ])
            ax.invert_yaxis()
            ax.set_xlabel("Δ 健康 / Δlog(1+Aa) 斜率")
            ax.set_title(OUTCOME_LABELS[outcome], loc="left", fontweight="bold")
            ax.grid(axis="x", linewidth=.5)
        fig.suptitle("主要土类中的配对健康响应斜率及空间块区间\nM4 · 20 年平均实际侵蚀",
                     y=.98, fontsize=14, fontweight="bold")
        handles = [
            Line2D([0], [0], color=color, marker="o" if scheme == "S0_all" else "D", linewidth=1.5,
                   label=SCHEME_LABELS[scheme])
            for scheme, color in scheme_colors.items()
        ]
        fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(.5, .02))
        fig.subplots_adjust(left=.10, right=.99, bottom=.20, top=.82, wspace=.34)
        figures.append(_save(
            fig, figure_root, "HET04_soil_group_paired_slopes", soil_slopes,
            "点为同土类、同 5 km 块配对上的固定线性斜率，横线为按 5 km 空间块重抽样得到的 95% 区间。比较全部原配对与阈值后固定原环境组重配；26 为冲积土、27 为潮土、42 为脱潮土。",
            scientific_question="低值筛选前后，三类主要土壤中的 SHI、OM 和 SW 配对斜率如何变化？",
            experiment="土壤类型分层的空间块斜率对照", figure_type="空间块稳健森林图",
        ))

    # Bubble plot: group-level matched-pair support and environmental balance.
    bubble = population_support.loc[
        population_support["method"].eq("M4")
        & population_support["exposure"].eq("Aa_20yr")
        & population_support["group_column"].isin(core_group_columns)
    ].copy()
    if not bubble.empty:
        groups = _group_row_order(bubble, core_group_columns)
        row_index = list(groups.set_index(["group_column", "group_label"]).index)
        row_map = groups.set_index(["group_column", "group_label"])["row_label"]
        y_lookup = {key: index for index, key in enumerate(row_index)}
        fig, ax = plt.subplots(figsize=(11, max(13, .205 * len(row_index) + 2.5)), constrained_layout=False)
        scheme_x = {scheme: index for index, scheme in enumerate(SCHEME_ORDER)}
        pair_counts = pd.to_numeric(bubble["n_pairs_same_group"], errors="coerce").fillna(0).to_numpy(float)
        max_count = max(float(pair_counts.max()), 1.0)
        sizes = 16 + 155 * np.sqrt(pair_counts / max_count)
        scatter = ax.scatter(
            bubble["scheme"].map(scheme_x),
            [y_lookup[(row.group_column, row.group_label)] for row in bubble.itertuples(index=False)],
            s=sizes, c=pd.to_numeric(bubble["mean_abs_standardized_environment_difference"], errors="coerce"),
            cmap="crest_r", vmin=0, vmax=max(1.0, float(pd.to_numeric(bubble["mean_abs_standardized_environment_difference"], errors="coerce").quantile(.95))),
            alpha=.88, edgecolors="white", linewidths=.5,
        )
        ax.set_xticks(list(scheme_x.values()), [SCHEME_LABELS[name] for name in scheme_x], rotation=25, ha="right")
        ax.set_yticks(np.arange(len(row_index)), [row_map.loc[key] for key in row_index], fontsize=6.8)
        ax.invert_yaxis()
        ax.set_ylabel("环境组")
        ax.set_xlabel("配对群体方案")
        ax.set_title("M4 · 20 年平均实际侵蚀", loc="left", fontweight="bold")
        fig.suptitle("环境组层面的匹配支持量与环境差异", fontsize=14, fontweight="bold")
        ax.grid(axis="x", linewidth=.5)
        cbar = fig.colorbar(scatter, ax=ax, fraction=.03, pad=.02)
        cbar.set_label("组内平均绝对标准化环境差异（越低越接近）")
        for size, label in ((25, "较少配对"), (80, "中等配对"), (140, "较多配对")):
            ax.scatter([], [], s=size, c="#6F8995", alpha=.75, edgecolors="white", label=label)
        ax.legend(title="气泡大小：组内配对数", loc="lower center", bbox_to_anchor=(.5, 1.005),
                  ncol=3, frameon=False, fontsize=7, title_fontsize=7)
        fig.subplots_adjust(left=.25, right=.93, bottom=.055, top=.91)
        figures.append(_save(
            fig, figure_root, "HET05_group_matching_support_balance", bubble,
            "每个气泡代表一个环境组与一个匹配方案；气泡大小按该组同组配对数缩放，颜色表示从配对端点环境值重算的连续环境差平均绝对标准化幅度，各变量以全体有效耕地像元标准差统一量纲。来源匹配表的标量环境距离另列于源数据与报告。重配方案改变匹配对象，气泡不能视作同一固定样本的单纯改善。",
            scientific_question="阈值过滤及重新匹配后，各环境组保留多少配对、环境相似度如何？",
            experiment="M4 分组匹配支持量与环境差异", figure_type="分组匹配诊断气泡图",
        ))
    return figures


def _md_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "（无可估计结果）"
    return frame.fillna("—").to_markdown(index=False, floatfmt=".3f")


def _write_report(
    report_path: Path,
    *,
    run_id: str,
    low_run_id: str,
    group_label_path: Path,
    thresholds: dict[str, float],
    cell_retention: pd.DataFrame,
    population_support: pd.DataFrame,
    response_comparison: pd.DataFrame,
    spatial_summary: pd.DataFrame,
    slope_bootstrap: pd.DataFrame,
    figures: list[dict[str, str]],
    core_group_columns: tuple[str, ...],
) -> None:
    primary_retention = cell_retention.loc[cell_retention["group_column"].isin(core_group_columns)].copy()
    highlights = primary_retention.sort_values(
        ["exposure", "group_column", "low_value_excluded_fraction"], ascending=[True, True, False]
    ).groupby(["exposure", "group_column"], sort=False, dropna=False).head(2)
    highlight_table = highlights[[
        "exposure", "group_column_label", "group_label_display", "n_valid_cells",
        "n_low_value_cells", "low_value_excluded_fraction",
    ]].rename(columns={
        "exposure": "侵蚀时期", "group_column_label": "分组方式", "group_label_display": "环境组",
        "n_valid_cells": "有效像元数", "n_low_value_cells": "阈值以下像元数", "low_value_excluded_fraction": "阈值排除比例",
    })
    primary_population = population_support.loc[
        population_support["method"].eq("M4")
        & population_support["exposure"].eq("Aa_20yr")
        & population_support["group_column"].eq("soil_type_group")
        & population_support["scheme"].isin(("S0_all", "T01_fixed_original", "T01_refit_k6_filtered_population"))
    ].copy()
    population_table = primary_population[[
        "group_label_display", "scheme_label", "n_valid_cells", "n_retained_cells",
        "n_unique_matched_endpoints", "n_pairs_same_group", "pair_retention_vs_S0",
        "mean_abs_standardized_environment_difference", "mean_environmental_distance", "mean_geographic_distance_m",
    ]].rename(columns={
        "group_label_display": "土壤类型", "scheme_label": "群体方案", "n_valid_cells": "有效像元数",
        "n_retained_cells": "阈值以上像元数", "n_unique_matched_endpoints": "唯一匹配端点像元数",
        "n_pairs_same_group": "同组配对数", "pair_retention_vs_S0": "配对数相对基线",
        "mean_abs_standardized_environment_difference": "平均绝对标准化环境差异",
        "mean_environmental_distance": "配对表环境距离均值",
        "mean_geographic_distance_m": "平均地理距离（米）",
    })

    fixed_pair = response_comparison.loc[
        response_comparison["analysis_unit"].eq("pair")
        & response_comparison["method"].eq("M4")
        & response_comparison["exposure"].eq("Aa_20yr")
        & response_comparison["scheme"].eq("T01_fixed_original")
        & response_comparison["group_column"].isin(core_group_columns)
    ].copy()
    direction_rows = []
    for group_column in core_group_columns:
        for outcome in OUTCOMES:
            part = fixed_pair.loc[fixed_pair["group_column"].eq(group_column) & fixed_pair["outcome"].eq(outcome)]
            supported = part.loc[
                pd.to_numeric(part["n"], errors="coerce").ge(30)
                & pd.to_numeric(part["baseline_n"], errors="coerce").ge(30)
            ]
            baseline_rho = pd.to_numeric(supported["baseline_spearman_rho"], errors="coerce")
            filtered_rho = pd.to_numeric(supported["spearman_rho"], errors="coerce")
            valid = baseline_rho.notna() & filtered_rho.notna()
            baseline_rho, filtered_rho = baseline_rho.loc[valid], filtered_rho.loc[valid]
            direction_rows.append({
                "环境分组": GROUP_NAMES.get(group_column, (group_column, group_column))[0],
                "健康指标": OUTCOME_LABELS[outcome],
                "满足样本量门槛的组数": int(valid.sum()),
                "负向关联组数（基线）": int(baseline_rho.lt(0).sum()),
                "负向关联组数（过滤重配）": int(filtered_rho.lt(0).sum()),
                "正转负组数": int((baseline_rho.ge(0) & filtered_rho.lt(0)).sum()),
                "负转正组数": int((baseline_rho.lt(0) & filtered_rho.ge(0)).sum()),
                "ρ下降至少 0.1 的组数": int((filtered_rho - baseline_rho).le(-.1).sum()),
            })

    spatial_counts = spatial_summary.groupby(["exposure", "scheme"], dropna=False).agg(
        分组类型数=("group_column", "nunique"),
        有效空间测试单元数=("n_successful_splits", lambda value: int(pd.to_numeric(value, errors="coerce").gt(0).sum())),
        成功空间划分总数=("n_successful_splits", "sum"),
    ).reset_index() if not spatial_summary.empty else pd.DataFrame()
    spatial_counts = spatial_counts.rename(columns={"exposure": "侵蚀时期", "scheme": "匹配方案"})
    spatial_counts["侵蚀时期"] = spatial_counts.get("侵蚀时期", pd.Series(dtype=str)).map(EXPOSURE_LABELS).fillna(spatial_counts.get("侵蚀时期"))
    spatial_counts["匹配方案"] = spatial_counts.get("匹配方案", pd.Series(dtype=str)).map(SCHEME_LABELS).fillna(spatial_counts.get("匹配方案"))

    lines = [
        "# 低侵蚀筛选前后的环境组响应与空间表现",
        "",
        f"运行编号：`{run_id}`；模式：`EXPLORATORY_ORACLE`，`outcome_guided_selection=true`。沿用低侵蚀主分析 `{low_run_id}` 的 Aa 和配对群体，以及既有环境变量分组标签 `{group_label_path}`；没有用健康结果重新构造环境组。未重算 RUSLE 因子乘积，未重跑 PCNM。",
        "",
        "## 分析口径",
        "",
        "- 匹配端点像元：高、低侵蚀端点先合并，再按 cell_id 去重；暴露使用 log(1+Aa)，同一像元不因重复出现在多对中而重复加权。",
        "- 配对差值：仅在一对两端具有同一个环境组标签时计入组内分析；响应为高侵蚀端减低侵蚀端，暴露差为 Δlog(1+Aa)。跨组配对不强行归入任一组，另以每个方案的同组配对占比记录。",
        "- 组别：土类、K-means、Ward 层次分组和环境梯度分组均来自既有环境标签；组别定义不由健康指标决定。完整 11 种分组列均保存结果表，空间验证重点覆盖土类、K-means k=6、Ward、TWI 与 LST 五类。",
        "- 相关统计：Spearman 与斜率、表观 R²均按可用组内样本计算；为避免大像元组的高代价排序检验，Kendall τ仅在有效量不超过 5,000 时精确计算，状态见 `kendall_tau_status`。",
        "- 空间测试：M4 配对两端必须处于同一 5 km 块；整块分入训练/验证/测试，训练配对剔除距留出块 500 m 内的样本。使用预先固定线性基线，10 个种子重复；测试分数不用于选组或选模型。",
        "- 低值阈值：20 年平均 Aa={:.6f}，2020 年 Aa={:.6f}；阈值来自侵蚀分布。筛选和重配改变研究对象，因此组间或方案间的拟合变化不是因果效应，也不能单独解释为模型改善。".format(thresholds["Aa_20yr"], thresholds["Aa_2020"]),
        "",
        "## 各环境组低值排除比例",
        "",
        "下表显示核心环境分组中，分布阈值以下的像元比例最高的组。完整像元数、保留数及 11 种环境分组结果见 `tables/group_cell_retention.csv`。",
        "",
        _md_table(highlight_table),
        "",
        "## 土类内匹配支持量与平衡",
        "",
        "下表以 M4、20 年平均 Aa 为例，比较全部原配对、过滤后沿用原分组重配和过滤后重拟合 k=6 组。图中着色的标准化环境差由配对端点的连续环境值重算：各变量绝对差除以全体有效耕地像元标准差，再对变量取均值；“配对表环境距离”则单独保留原匹配表中的距离字段，两者口径不同。完整 11 种组别的像元数、配对数、唯一端点、空间块、地理距离和环境距离见 `tables/group_population_support.csv`。",
        "",
        _md_table(population_table),
        "",
        "## 九项健康指标的组内秩关联变化",
        "",
        "以下对 M4、20 年平均 Aa、阈值后固定原环境组重配作支持量总结：基线和过滤方案中的组内有效观测均不少于 30 才计入。正转负/负转正是描述性方向变化；Oracle 搜索结果需要后续独立或预先锁定设计复核。每个组、每项指标、两种分析单位和所有匹配方案的原始数值见 `tables/group_response_comparison.csv`。",
        "",
        _md_table(pd.DataFrame(direction_rows)),
        "",
        "## 空间留出表现",
        "",
        "组内线性测试的完整 10 次划分、训练/测试样本量、5 km 组数、缓冲后状态与测试 R²见 `spatial_validation/group_spatial_validation.csv`；按组汇总的 R²中位数、四分位数和负斜率比例见 `tables/group_spatial_summary.csv`。",
        "",
        _md_table(spatial_counts),
        "",
        "## 空间块斜率区间",
        "",
        "`tables/group_pair_slope_spatial_bootstrap.csv` 为同环境组且两端位于同一 5 km 块的配对斜率，按空间块有放回抽样计算 95% 区间。区间少于 5 个空间块时留空；图中重点展示 26 冲积土、27 潮土、42 脱潮土的 SHI、OM 和 SW。",
        "",
        "## 图件",
        "",
    ]
    for item in figures:
        lines.append(f"- [{item['title']}](../figures/png/{item['id']}.png)；源数据 `figures/data/{item['id']}_source.csv`。")
    lines.extend([
        "",
        "## 解释边界与后续",
        "",
        "1. 本次分组响应是 Oracle 探索性描述，组别标签来自环境数据，但健康指标用于结果评估；多组、多指标比较中出现的最强关系不代表独立确认。",
        "2. 预测栅格的像元数量不等于独立土样数量；像元分析、配对分析和空间块分析分别报告，不能把它们互换为独立样本量。",
        "3. 组内配对要求两端环境组一致；这会排除跨组配对，并可能使低支持组无法估计。缺失的图格或空间分数表示支持不足，不表示关联为零。",
        "4. 空间验证仅针对 M4 的固定线性诊断模型；其目的是检查组内方向/拟合的空间稳定性，不证明模型可外推至独立地区。",
        "5. 推荐后续先锁定本次候选组定义和分析单位，再在独立空间块或独立实测点开展确认；不要仅依据某个组的表观 ρ、R²或区间决定删组。",
        "",
    ])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def _write_final_group_artifacts(
    *,
    root: Path,
    run_dir: Path,
    config: dict[str, Any],
    thresholds: dict[str, float],
    cell_retention: pd.DataFrame,
    population_support: pd.DataFrame,
    response_summary: pd.DataFrame,
    response_comparison: pd.DataFrame,
    spatial: pd.DataFrame,
    spatial_summary: pd.DataFrame,
    slopes: pd.DataFrame,
    core_group_columns: tuple[str, ...],
    input_paths: dict[str, str],
    baseline_sha: str | None,
    resumed_from_saved_tables: bool,
) -> list[dict[str, str]]:
    figures = _build_figures(
        run_dir / "figures", cell_retention, population_support, response_comparison,
        spatial_summary, slopes, core_group_columns=core_group_columns,
    )
    figure_manifest = {
        "run_id": config["run_id"], "mode": "EXPLORATORY_ORACLE", "outcome_guided_selection": True,
        "figure_count": len(figures), "formats": ["png", "svg", "pdf"], "figures": figures,
    }
    (run_dir / "figures/figure_manifest.json").write_text(
        json.dumps(figure_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_report(
        run_dir / "reports/ENVIRONMENT_GROUP_HETEROGENEITY_REPORT.md",
        run_id=str(config["run_id"]), low_run_id=str(config["low_erosion_run_id"]),
        group_label_path=Path(config["group_labels_path"]), thresholds=thresholds,
        cell_retention=cell_retention, population_support=population_support,
        response_comparison=response_comparison, spatial_summary=spatial_summary,
        slope_bootstrap=slopes, figures=figures, core_group_columns=core_group_columns,
    )
    provenance = {
        "phase": "H-environment-group-heterogeneity", "run_id": config["run_id"],
        "mode": "EXPLORATORY_ORACLE", "outcome_guided_selection": True,
        "low_erosion_run_id": config["low_erosion_run_id"],
        "group_label_path": str(root / Path(config["group_labels_path"])),
        "git_commit_at_run": baseline_sha,
        "finalized_at_utc": datetime.now(UTC).isoformat(),
        "analysis_tables_loaded_from_saved_outputs": resumed_from_saved_tables,
        "python": platform.python_version(),
        "packages": {name: _package_version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "seaborn", "pyarrow")},
        "thresholds_original": thresholds, "group_columns": list(config["group_columns"]),
        "core_spatial_group_columns": list(core_group_columns),
        "spatial_validation": {
            "method": config["spatial_validation"]["method"],
            "schemes": list(config["spatial_validation"]["schemes"]), "block_size_m": 5000,
            "buffer_m": int(config["spatial_validation"]["buffer_m"]),
            "seeds": [int(value) for value in config["spatial_validation"]["seeds"]],
            "model_family": "fixed_linear_baseline",
            "pair_rule": "both endpoints must lie within the same 5 km block",
        },
        "slope_bootstrap": {
            "schemes": list(config["slope_bootstrap"]["schemes"]),
            "replicates": int(config["slope_bootstrap"]["replicates"]), "block_size_m": 5000,
        },
        "figure_count": len(figures), "group_response_rows": len(response_summary),
        "group_population_rows": len(population_support), "spatial_validation_rows": len(spatial),
        "input_paths": input_paths,
    }
    (run_dir / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "checkpoints.json").write_text(json.dumps({
        "environment_group_cell_retention_complete": True,
        "group_matched_population_support_complete": True,
        "matched_pixel_and_within_group_pair_responses_complete": True,
        "group_spatial_validation_complete": True,
        "spatial_block_slope_bootstrap_complete": True,
        "figures_and_report_complete": True,
        "resumed_from_saved_analysis_tables": resumed_from_saved_tables,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return figures


def finalize_saved_group_heterogeneity_run(project_root: Path, config_path: Path) -> Path:
    """Render figures and reports from group-analysis tables already written to disk."""
    root = Path(project_root).resolve()
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = root / config_path
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_root = Path(config["output_root"])
    run_dir = root / output_root / str(config["run_id"])
    required = {
        "cell_retention": run_dir / "tables/group_cell_retention.parquet",
        "population_support": run_dir / "tables/group_population_support.parquet",
        "response_summary": run_dir / "associations/group_response_summary.parquet",
        "response_comparison": run_dir / "tables/group_response_comparison.parquet",
        "spatial": run_dir / "spatial_validation/group_spatial_validation.parquet",
        "spatial_summary": run_dir / "tables/group_spatial_summary.parquet",
        "slopes": run_dir / "tables/group_pair_slope_spatial_bootstrap.parquet",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"saved group-analysis tables are missing: {missing}")
    tables = {name: pd.read_parquet(path) for name, path in required.items()}
    # Older filtered-pair sources omit the scheme column. The all-pairs baseline
    # therefore reached the saved validation/bootstrap tables with a null key.
    # Repair that run metadata from the fixed population order and rebuild the
    # derived comparisons from the persisted, otherwise unchanged estimates.
    for frame_name in ("spatial", "slopes"):
        frame = tables[frame_name]
        if "scheme" in frame and frame["scheme"].isna().any():
            frame["scheme"] = frame["scheme"].fillna("S0_all")
    tables["spatial_summary"] = summarize_group_spatial_validation(tables["spatial"])
    tables["response_comparison"] = compare_group_responses_to_baseline(tables["response_summary"])
    if not tables["slopes"].empty:
        keys = ["exposure", "method", "scheme", "outcome", "group_column", "group_label"]
        tables["response_comparison"] = tables["response_comparison"].merge(
            tables["slopes"], on=keys, how="left", suffixes=("", "_bootstrap"), validate="many_to_one"
        )
    _table_write(tables["spatial"], run_dir / "spatial_validation", "group_spatial_validation")
    _table_write(tables["spatial_summary"], run_dir / "tables", "group_spatial_summary")
    _table_write(tables["slopes"], run_dir / "tables", "group_pair_slope_spatial_bootstrap")
    _table_write(tables["response_comparison"], run_dir / "tables", "group_response_comparison")
    low_run = root / output_root / str(config["low_erosion_run_id"])
    low_config = yaml.safe_load((low_run / "config.yaml").read_text(encoding="utf-8"))
    cells, _, _ = _load_inputs(root, low_config)
    group_labels = pd.read_parquet(root / Path(config["group_labels_path"]))
    candidate_table = pd.read_csv(low_run / "thresholds/distribution_threshold_candidates.csv")
    thresholds = {}
    for exposure in EXPOSURES:
        row = candidate_table.loc[
            candidate_table["exposure"].eq(exposure)
            & candidate_table["is_primary"].fillna(False).astype(bool)
        ]
        if row.empty:
            raise ValueError(f"missing distribution-derived primary threshold for {exposure}")
        thresholds[exposure] = float(row.iloc[0]["threshold_original"])
    # Rebuild support metrics from persisted pair endpoints. Some rematched
    # sources contain only a scalar matching distance, so standardized
    # environment differences are reconstructed from cell-level values.
    support_rows: list[pd.DataFrame] = []
    for exposure in EXPOSURES:
        for method in config["matching_methods"]:
            for scheme, pair_path in _source_populations(low_run, method, exposure).items():
                if not pair_path.is_file():
                    raise FileNotFoundError(pair_path)
                pairs = pd.read_parquet(pair_path)
                if pairs.empty:
                    continue
                support_rows.append(_pair_group_support(
                    pairs, group_labels, tables["cell_retention"], cells,
                    method=method, exposure=exposure, scheme=scheme,
                    group_columns=tuple(config["group_columns"]),
                ))
                del pairs
    population_support = pd.concat(support_rows, ignore_index=True, sort=False) if support_rows else pd.DataFrame()
    population_support = _merge_group_population_baseline(population_support)
    if not population_support.empty:
        for metric in ("n_pairs_same_group", "n_unique_matched_endpoints", "n_unique_same_group_endpoints"):
            population_support[f"{metric}_change_vs_S0"] = (
                pd.to_numeric(population_support[metric], errors="coerce")
                - pd.to_numeric(population_support[f"baseline_{metric}"], errors="coerce")
            )
    tables["population_support"] = population_support
    _table_write(population_support, run_dir / "tables", "group_population_support")
    low_provenance_path = low_run / "provenance.json"
    low_provenance = json.loads(low_provenance_path.read_text(encoding="utf-8")) if low_provenance_path.is_file() else {}
    baseline_sha = low_provenance.get("git_commit_at_run")
    _write_final_group_artifacts(
        root=root, run_dir=run_dir, config=config, thresholds=thresholds,
        cell_retention=tables["cell_retention"], population_support=tables["population_support"],
        response_summary=tables["response_summary"], response_comparison=tables["response_comparison"],
        spatial=tables["spatial"], spatial_summary=tables["spatial_summary"], slopes=tables["slopes"],
        core_group_columns=tuple(config["core_group_columns"]),
        input_paths=low_provenance.get("input_paths", {}), baseline_sha=baseline_sha,
        resumed_from_saved_tables=True,
    )
    return run_dir


def run_group_heterogeneity(project_root: Path, config_path: Path) -> Path:
    root = Path(project_root).resolve()
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = root / config_path
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("mode") != "EXPLORATORY_ORACLE" or config.get("outcome_guided_selection") is not True:
        raise ValueError("group heterogeneity requires EXPLORATORY_ORACLE and outcome_guided_selection=true")
    output_root = Path(config["output_root"])
    if output_root.is_absolute() or ".." in output_root.parts or output_root.parts[:2] != ("outputs", "exploratory_oracle"):
        raise ValueError("output_root must remain inside outputs/exploratory_oracle")
    run_id = str(config["run_id"])
    run_dir = root / output_root / run_id
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing run: {run_dir}")
    for directory in (
        "config", "tables", "associations", "spatial_validation",
        "figures/png", "figures/svg", "figures/pdf", "figures/data", "figures/captions", "reports",
    ):
        (run_dir / directory).mkdir(parents=True, exist_ok=True)
    snapshot = yaml.safe_dump(config, allow_unicode=True, sort_keys=False)
    (run_dir / "config.yaml").write_text(snapshot, encoding="utf-8")
    (run_dir / "config/config.yaml").write_text(snapshot, encoding="utf-8")
    try:
        baseline_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        baseline_sha = None

    low_run_id = str(config["low_erosion_run_id"])
    low_run = root / output_root / low_run_id
    low_config = yaml.safe_load((low_run / "config.yaml").read_text(encoding="utf-8"))
    cells, _, input_paths = _load_inputs(root, low_config)
    label_path = root / Path(config["group_labels_path"])
    if not label_path.is_file():
        raise FileNotFoundError(label_path)
    group_labels = pd.read_parquet(label_path)
    group_columns = tuple(config.get("group_columns", [column for column in group_labels.columns if column != "cell_id"]))
    core_group_columns = tuple(config.get("core_group_columns", CORE_GROUP_COLUMNS))
    if not set(group_columns).issubset(group_labels.columns) or not set(core_group_columns).issubset(group_columns):
        raise ValueError("configured environment group columns must exist in the Phase H label table")
    threshold_candidates = pd.read_csv(low_run / "thresholds/distribution_threshold_candidates.csv")
    thresholds = {}
    for exposure in EXPOSURES:
        row = threshold_candidates.loc[
            threshold_candidates["exposure"].eq(exposure)
            & threshold_candidates["is_primary"].fillna(False).astype(bool)
        ]
        if row.empty:
            raise ValueError(f"missing distribution-derived primary threshold for {exposure}")
        thresholds[exposure] = float(row.iloc[0]["threshold_original"])

    cell_retention = summarize_cell_group_retention(
        cells, group_labels, thresholds, group_columns=group_columns
    )
    print(f"已统计 {len(group_columns)} 类环境标签下的全域像元保留率。", flush=True)
    response_rows: list[pd.DataFrame] = []
    support_rows: list[pd.DataFrame] = []
    spatial_rows: list[pd.DataFrame] = []
    slope_rows: list[pd.DataFrame] = []
    seeds = tuple(int(value) for value in config["spatial_validation"]["seeds"])
    spatial_schemes = tuple(config["spatial_validation"]["schemes"])
    for exposure in EXPOSURES:
        print(f"开始处理：{EXPOSURE_LABELS[exposure]}。", flush=True)
        for method in config["matching_methods"]:
            print(f"  {method} 匹配方案", flush=True)
            for scheme, pair_path in _source_populations(low_run, method, exposure).items():
                pairs = pd.read_parquet(pair_path)
                if pairs.empty:
                    continue
                # Existing filtered-pair parquet files omit their scheme key.
                # Attach loop metadata before grouped spatial/bootstrap summaries.
                pairs["method"] = method
                pairs["exposure"] = exposure
                pairs["scheme"] = scheme
                print(f"    {SCHEME_LABELS[scheme]}：{len(pairs):,} 对", flush=True)
                population_key = (method, exposure, scheme)
                response_rows.append(summarize_group_responses(
                    cells, group_labels, {population_key: pairs},
                    group_columns=group_columns, outcomes=tuple(low_config["outcomes"]),
                ))
                support_rows.append(_pair_group_support(
                    pairs, group_labels, cell_retention, cells,
                    method=method, exposure=exposure, scheme=scheme, group_columns=group_columns,
                ))
                if method == config["spatial_validation"]["method"] and scheme in spatial_schemes:
                    spatial_rows.append(validate_group_spatial_performance(
                        pairs, group_labels, group_columns=core_group_columns,
                        outcomes=tuple(low_config["outcomes"]), seeds=seeds,
                        buffer_m=int(config["spatial_validation"]["buffer_m"]),
                    ))
                    print(f"      已完成 {len(core_group_columns)} 类分组的空间留出。", flush=True)
                if method == config["spatial_validation"]["method"] and scheme in config["slope_bootstrap"]["schemes"]:
                    slope_rows.append(bootstrap_group_pair_slopes(
                        pairs, group_labels,
                        group_columns=core_group_columns,
                        outcomes=tuple(low_config["outcomes"]),
                        replicates=int(config["slope_bootstrap"]["replicates"]),
                        seed=int(config["seed"]),
                    ))
                del pairs

    response_summary = pd.concat(response_rows, ignore_index=True, sort=False) if response_rows else pd.DataFrame()
    response_comparison = compare_group_responses_to_baseline(response_summary)
    support = pd.concat(support_rows, ignore_index=True, sort=False) if support_rows else pd.DataFrame()
    support = _merge_group_population_baseline(support)
    if not support.empty:
        for metric in ("n_pairs_same_group", "n_unique_matched_endpoints", "n_unique_same_group_endpoints"):
            support[f"{metric}_change_vs_S0"] = (
                pd.to_numeric(support[metric], errors="coerce")
                - pd.to_numeric(support[f"baseline_{metric}"], errors="coerce")
            )

    spatial = pd.concat(spatial_rows, ignore_index=True, sort=False) if spatial_rows else pd.DataFrame()
    spatial_summary = summarize_group_spatial_validation(spatial)
    slopes = pd.concat(slope_rows, ignore_index=True, sort=False) if slope_rows else pd.DataFrame()
    if not slopes.empty:
        keys = ["exposure", "method", "scheme", "outcome", "group_column", "group_label"]
        response_comparison = response_comparison.merge(slopes, on=keys, how="left", suffixes=("", "_bootstrap"), validate="many_to_one")

    _table_write(cell_retention, run_dir / "tables", "group_cell_retention")
    _table_write(support, run_dir / "tables", "group_population_support")
    _table_write(response_summary, run_dir / "associations", "group_response_summary")
    _table_write(response_comparison, run_dir / "tables", "group_response_comparison")
    _table_write(spatial, run_dir / "spatial_validation", "group_spatial_validation")
    _table_write(spatial_summary, run_dir / "tables", "group_spatial_summary")
    _table_write(slopes, run_dir / "tables", "group_pair_slope_spatial_bootstrap")
    _write_final_group_artifacts(
        root=root, run_dir=run_dir, config=config, thresholds=thresholds,
        cell_retention=cell_retention, population_support=support,
        response_summary=response_summary, response_comparison=response_comparison,
        spatial=spatial, spatial_summary=spatial_summary, slopes=slopes,
        core_group_columns=core_group_columns,
        input_paths={name: str(path) for name, path in input_paths.items()},
        baseline_sha=baseline_sha, resumed_from_saved_tables=False,
    )
    return run_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path("config/low_erosion_group_heterogeneity.yaml"))
    parser.add_argument("--finalize-existing", action="store_true", help="Build charts and report from already saved analysis tables")
    args = parser.parse_args(argv)
    result = (
        finalize_saved_group_heterogeneity_run(args.project_root, args.config)
        if args.finalize_existing else run_group_heterogeneity(args.project_root, args.config)
    )
    print(result.relative_to(args.project_root.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
