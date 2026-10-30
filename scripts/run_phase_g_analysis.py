#!/usr/bin/env python3
"""Build the Phase G Oracle report from existing aligned and matched data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/phase-g-matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import r2_score

from experiments.oracle_exploration.data import load_oracle_data
from experiments.oracle_exploration.nonlinear_search import fit_model_family
from experiments.oracle_exploration.phase_g import (
    _strongest_negative_score,
    _summarize_one_pair_group,
    fit_hierarchical_group_labels,
    rank_phase_g_models,
    select_pair_spatial_p1_rows,
    summarize_pair_associations,
    summarize_pair_spatial_validation,
)
from experiments.oracle_exploration.validation import spatial_roles

OUTCOMES = ["SHI_candidate", "AN", "OM", "CEC", "pH", "SW", "BD", "MBC", "RESP"]
SOIL_TYPE_LABELS = {26: "冲积土", 27: "潮土", 42: "脱潮土"}
ENVIRONMENT = ["Elevation", "Cur", "TWI", "LST", "NDWI", "NDBSI", "EVI", "NDVI", "Precipitation"]
FIGURE_TITLES = {
    "01_matching_quality": "匹配覆盖率与环境平衡",
    "02_paired_health_differences": "高低侵蚀配对的健康差值分布",
    "03_pair_association_heatmap": "配对侵蚀差值与健康差值的秩相关",
    "04_model_r2_comparison": "像元级模型拟合度比较",
    "05_response_curves": "自然样条与全样本择优响应曲线",
    "06_environment_group_forest": "不同环境组的负向关联",
    "07_grouping_method_comparison": "不同环境分组下的关联分布",
    "08_key_group_maps": "结果导向筛选环境组的空间位置",
    "09_spatial_validation": "空间留出表现与重复配对拟合",
}


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value):
        return None
    return str(value)


def _display_group_label(value: Any) -> str:
    text = str(value)
    match = re.fullmatch(r"soil_(\d+)", text)
    if match:
        code = int(match.group(1))
        return f"{code} {SOIL_TYPE_LABELS.get(code, '未知土类')}"
    match = re.fullmatch(r"(\d+):q(\d+)", text)
    if match:
        code, quantile = map(int, match.groups())
        return f"{code} {SOIL_TYPE_LABELS.get(code, '未知土类')} · 第 {quantile} 分位组"
    match = re.fullmatch(r"soil_(\d+)_cluster_(\d+)", text)
    if match:
        code, cluster = map(int, match.groups())
        return f"{code} {SOIL_TYPE_LABELS.get(code, '未知土类')} · 第 {cluster + 1} 组"
    return text


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _packages() -> dict[str, str]:
    names = ["numpy", "pandas", "scipy", "scikit-learn", "statsmodels", "pyarrow", "PyYAML", "matplotlib"]
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _progress(message: str) -> None:
    print(f"[Phase G] {message}", flush=True)


def _validate_pairs(pairs: pd.DataFrame, outcomes: list[str]) -> dict[str, object]:
    required = {"pair_id", "method", "exposure", "high_erosion_cell_id", "low_erosion_cell_id", "erosion_high", "erosion_low", "x_mid", "y_mid"}
    missing = sorted(required - set(pairs.columns))
    if missing:
        raise ValueError(f"Phase F pair table is missing columns required by Phase G: {missing}")
    if pairs["pair_id"].isna().any() or pairs["pair_id"].duplicated().any():
        raise ValueError("Phase F pair IDs must be present and unique")
    if (pd.to_numeric(pairs["erosion_high"], errors="coerce") < pd.to_numeric(pairs["erosion_low"], errors="coerce")).any():
        raise ValueError("A high-erosion member has a lower precomputed Aa value than its paired low member")
    if pairs[["x_mid", "y_mid"]].apply(pd.to_numeric, errors="coerce").isna().any().any():
        raise ValueError("Pair midpoint coordinates contain missing values")
    for (method, exposure), group in pairs.groupby(["method", "exposure"], sort=True):
        used = pd.concat([group["high_erosion_cell_id"], group["low_erosion_cell_id"]], ignore_index=True)
        if used.duplicated().any():
            raise ValueError(f"The no-replacement pair set reuses a cell: {method}/{exposure}")
    missing_outcomes = [outcome for outcome in outcomes if f"high_{outcome}" not in pairs or f"low_{outcome}" not in pairs]
    if missing_outcomes:
        raise ValueError(f"Phase F pair table is missing health outcomes: {missing_outcomes}")
    return {
        "rows": int(len(pairs)),
        "unique_pairs": int(pairs["pair_id"].nunique()),
        "pair_sets": pairs.groupby(["method", "exposure"], sort=True).size().rename("n_pairs").reset_index().to_dict(orient="records"),
        "negative_actual_Aa_pairs": int((pd.to_numeric(pairs["erosion_high"], errors="coerce") < pd.to_numeric(pairs["erosion_low"], errors="coerce")).sum()),
        "reused_member_count": 0,
        "health_variables_present": outcomes,
        "spatial_block_count": int(pairs["spatial_block_id"].nunique()) if "spatial_block_id" in pairs else None,
    }


def _gradient_group_labels(frame: pd.DataFrame, variables: list[str], bins: int, soils: list[int]) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    for variable in variables:
        column = f"phase_g_gradient_{variable}"
        labels = pd.Series(pd.NA, index=frame.index, dtype="string")
        values = pd.to_numeric(frame[variable], errors="coerce")
        soil_values = pd.to_numeric(frame["SoilType"], errors="coerce")
        for soil in soils:
            mask = soil_values.eq(soil) & np.isfinite(values)
            if int(mask.sum()) < bins:
                continue
            cut = pd.qcut(values.loc[mask], q=bins, labels=False, duplicates="drop")
            labels.loc[mask] = cut.astype("Int64").map(lambda value: f"{soil}:q{int(value) + 1}" if pd.notna(value) else pd.NA)
        records.append(pd.DataFrame({"cell_id": frame["cell_id"].to_numpy(), column: labels.to_numpy()}))
    result = pd.DataFrame({"cell_id": frame["cell_id"].to_numpy()})
    for part in records:
        result = result.merge(part, on="cell_id", how="left", validate="one_to_one")
    return result


def _group_pair_data(
    frame: pd.DataFrame,
    pairs: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, object]]:
    group_cfg = config["environment_groups"]
    soils = [int(value) for value in group_cfg["primary_soils"]]
    source = frame.loc[pd.to_numeric(frame["SoilType"], errors="coerce").isin(soils)].copy()
    variables = list(group_cfg["cluster_environment_variables"])
    hierarchy = fit_hierarchical_group_labels(
        source,
        variables,
        n_clusters=int(group_cfg["cluster_count"]),
        max_fit_rows=int(group_cfg["hierarchy_fit_rows_per_soil"]),
        seed=int(group_cfg["hierarchy_seed"]),
    )
    cell_labels = pd.DataFrame({"cell_id": source["cell_id"].to_numpy(), "hierarchical_ward_v5": hierarchy.to_numpy()})
    gradient_labels = _gradient_group_labels(
        source,
        list(group_cfg["gradient_variables"]),
        int(group_cfg["gradient_bins"]),
        soils,
    )
    cell_labels = cell_labels.merge(gradient_labels, on="cell_id", how="left", validate="one_to_one")
    by_cell = cell_labels.set_index("cell_id")
    result = pairs.copy()
    result["soil_type_group"] = pd.to_numeric(result["soil_type"], errors="coerce").map(
        lambda value: f"soil_{int(value)}" if pd.notna(value) else pd.NA
    ).astype("string")
    kmeans_source = result.get("descriptive_environment_group", result.get("environment_group"))
    result["kmeans_v5"] = pd.to_numeric(kmeans_source, errors="coerce").map(
        lambda value: f"{int(value)}" if pd.notna(value) else pd.NA
    ).astype("string")
    high_ids = result["high_erosion_cell_id"]
    result["hierarchical_ward_v5"] = high_ids.map(by_cell["hierarchical_ward_v5"]).astype("string")
    gradient_columns: list[str] = []
    for variable in group_cfg["gradient_variables"]:
        source_column = f"phase_g_gradient_{variable}"
        target_column = f"gradient_{variable}"
        result[target_column] = high_ids.map(by_cell[source_column]).astype("string")
        gradient_columns.append(target_column)
    label_columns = ["soil_type_group", "kmeans_v5", "hierarchical_ward_v5", *gradient_columns]
    meta = {
        "group_columns": label_columns,
        "group_definition": {
            "soil_type_group": "high-erosion member SoilType; primary soil codes only",
            "kmeans_v5": "Phase F V5 (Elevation, LST) soil-specific k=6 label carried by high-erosion member",
            "hierarchical_ward_v5": f"Ward agglomerative k={group_cfg['cluster_count']} within each primary soil; standardized on all eligible environment rows; fit sample capped at {group_cfg['hierarchy_fit_rows_per_soil']} rows per soil; nearest standardized centroid assigns other rows",
            "gradient_groups": "within-soil empirical quintile bins for each named environmental variable, assigned by the high-erosion member",
        },
        "gradient_variables": list(group_cfg["gradient_variables"]),
        "hierarchical_training_rows_per_soil": {
            str(soil): int(min(group_cfg["hierarchy_fit_rows_per_soil"], int((pd.to_numeric(source["SoilType"], errors="coerce") == soil).sum())))
            for soil in soils
        },
    }
    return result, meta


def _group_tables(
    pairs: pd.DataFrame,
    group_columns: list[str],
    outcomes: list[str],
    *,
    bootstrap_replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    env_vars = [name for name in ENVIRONMENT if f"mean_{name}" in pairs]
    stat_rows: list[dict[str, object]] = []
    assoc_rows: list[dict[str, object]] = []
    for group_column in group_columns:
        group_method = group_column.replace("_group", "")
        for (method, exposure, group_label), selected in pairs.groupby(
            ["method", "exposure", group_column], sort=True, dropna=True, observed=True
        ):
            if pd.isna(group_label):
                continue
            stats: dict[str, object] = {
                "group_method": group_method,
                "group_column": group_column,
                "group_label": str(group_label),
                "method": str(method),
                "exposure": str(exposure),
                "n_pairs": int(len(selected)),
                "spatial_blocks": int(selected["spatial_block_id"].nunique()) if "spatial_block_id" in selected else None,
                "selection_metric": "paired health difference association; all group candidates retained",
                "outcome_guided_selection": True,
            }
            for variable in env_vars:
                values = pd.to_numeric(selected[f"mean_{variable}"], errors="coerce")
                stats[f"mean_{variable}"] = float(values.mean()) if values.notna().any() else float("nan")
                stats[f"median_{variable}"] = float(values.median()) if values.notna().any() else float("nan")
            if "soil_type" in selected:
                stats["soil_type_counts"] = json.dumps(selected["soil_type"].value_counts(dropna=False).to_dict(), ensure_ascii=False)
            stat_rows.append(stats)
            for outcome_index, outcome in enumerate(outcomes):
                record = _summarize_one_pair_group(
                    selected.reset_index(drop=True),
                    exposure=str(exposure),
                    outcome=outcome,
                    subset="group_all",
                    bootstrap_replicates=bootstrap_replicates,
                    seed=seed + outcome_index + int(pd.util.hash_pandas_object(pd.Series([group_method, group_label])).iloc[0] % 100_003),
                )
                record.update(
                    group_method=group_method,
                    group_column=group_column,
                    group_label=str(group_label),
                    method=str(method),
                    selection_objective="screen all outcome-specific environmental groups; report association direction and strength",
                )
                record["direction"] = "negative" if record["standardized_slope"] < 0 else "positive" if record["standardized_slope"] > 0 else "unknown"
                record["standardized_effect"] = record["standardized_slope"]
                record["standardized_effect_ci_low"] = record["standardized_slope_ci_low"]
                record["standardized_effect_ci_high"] = record["standardized_slope_ci_high"]
                assoc_rows.append(record)
    stats_table = pd.DataFrame.from_records(stat_rows)
    associations = pd.DataFrame.from_records(assoc_rows)
    leaderboard_rows: list[dict[str, object]] = []
    if not associations.empty:
        for (group_method, method, exposure, outcome), subset in associations.groupby(
            ["group_method", "method", "exposure", "outcome"], sort=True
        ):
            eligible = subset.loc[subset["n_pairs"].ge(20)].copy()
            if eligible.empty:
                continue
            r2_row = eligible.loc[pd.to_numeric(eligible["pearson_r_squared"], errors="coerce").idxmax()]
            leaderboard_rows.append({**r2_row.to_dict(), "ranking_type": "highest_group_r_squared", "candidate_count": int(len(eligible))})
            negatives = eligible.loc[eligible["direction"].eq("negative") & pd.to_numeric(eligible["spearman_rho"], errors="coerce").lt(0)]
            if not negatives.empty:
                winner, _ = _strongest_negative_score(negatives)
                if not winner.empty:
                    leaderboard_rows.append({**winner.to_dict(), "ranking_type": "strongest_negative_group", "candidate_count": int(len(eligible))})
        stability = associations.loc[associations["n_pairs"].ge(20)].groupby(
            ["group_method", "group_label", "outcome"], sort=True
        ).agg(
            conditions=("spearman_rho", "count"),
            negative_fraction=("direction", lambda values: float(pd.Series(values).eq("negative").mean())),
            median_abs_spearman=("spearman_rho", lambda values: float(pd.to_numeric(values, errors="coerce").abs().median())),
            median_pairs=("n_pairs", "median"),
        ).reset_index()
        stability["stability_score"] = stability["negative_fraction"] * stability["median_abs_spearman"]
        for outcome, subset in stability.groupby("outcome", sort=True):
            winner = subset.sort_values(["stability_score", "conditions", "median_pairs"], ascending=False, kind="stable").iloc[0]
            leaderboard_rows.append({
                "group_method": winner["group_method"],
                "group_label": winner["group_label"],
                "outcome": outcome,
                "ranking_type": "most_stable_negative_direction_across_matching_methods_and_Aa_periods",
                "candidate_count": int(len(subset)),
                "stability_score": winner["stability_score"],
                "negative_fraction": winner["negative_fraction"],
                "median_abs_spearman": winner["median_abs_spearman"],
                "conditions": winner["conditions"],
                "median_pairs": winner["median_pairs"],
                "outcome_guided_selection": True,
            })
    return stats_table, associations, pd.DataFrame.from_records(leaderboard_rows)


def _simple_pair_metrics(data: pd.DataFrame, outcome: str) -> dict[str, float | int | str]:
    x = np.log1p(pd.to_numeric(data["erosion_high"], errors="coerce").to_numpy(dtype=float)) - np.log1p(pd.to_numeric(data["erosion_low"], errors="coerce").to_numpy(dtype=float))
    y = pd.to_numeric(data[f"high_{outcome}"], errors="coerce").to_numpy(dtype=float) - pd.to_numeric(data[f"low_{outcome}"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return {"n": int(n), "slope": float("nan"), "standardized_slope": float("nan"), "spearman_rho": float("nan"), "r_squared": float("nan"), "negative_delta_y_fraction": float("nan"), "direction": "unknown"}
    slope = float(np.cov(x, y, ddof=0)[0, 1] / np.var(x))
    rho = float(pd.Series(x).corr(pd.Series(y), method="spearman"))
    sy = float(np.std(y, ddof=0))
    standardized = slope * float(np.std(x, ddof=0)) / sy if sy > 0 else float("nan")
    pearson = float(np.corrcoef(x, y)[0, 1])
    return {"n": int(n), "slope": slope, "standardized_slope": standardized, "spearman_rho": rho, "r_squared": pearson * pearson, "negative_delta_y_fraction": float(np.mean(y < 0)), "direction": "negative" if slope < 0 else "positive" if slope > 0 else "unknown"}


def _group_spatial_selection(
    pairs: pd.DataFrame,
    group_columns: list[str],
    outcomes: list[str],
    seeds: list[int],
    block_size_m: float,
    min_pairs: int,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    methods = [column.replace("_group", "") for column in group_columns]
    for (method, exposure), source in pairs.groupby(["method", "exposure"], sort=True):
        if exposure != "Aa_20yr":
            continue
        for seed in seeds:
            roles = spatial_roles(source[["x_mid", "y_mid"]].rename(columns={"x_mid": "x", "y_mid": "y"}), block_size_m=block_size_m, seed=seed)
            role_data = source.copy()
            role_data["spatial_role"] = roles.to_numpy()
            for group_column, group_method in zip(group_columns, methods, strict=True):
                for outcome in outcomes:
                    validation_candidates: list[dict[str, object]] = []
                    for label, group in role_data.groupby(group_column, sort=True, dropna=True, observed=True):
                        if pd.isna(label):
                            continue
                        val_metrics = _simple_pair_metrics(group.loc[group["spatial_role"].eq("validation")], outcome)
                        if val_metrics["n"] < min_pairs or val_metrics["direction"] != "negative" or not np.isfinite(val_metrics["spearman_rho"]) or val_metrics["spearman_rho"] >= 0:
                            continue
                        validation_candidates.append({"group_label": str(label), **val_metrics})
                    base = {
                        "method": str(method),
                        "exposure": str(exposure),
                        "outcome": outcome,
                        "group_method": group_method,
                        "seed": int(seed),
                        "block_size_m": block_size_m,
                        "candidate_group_count": int(role_data[group_column].nunique(dropna=True)),
                        "candidate_selection": "validation-only composite of standardized negative slope, negative Spearman, negative-difference fraction, and n",
                        "outcome_guided_selection": True,
                    }
                    if not validation_candidates:
                        records.append({**base, "status": "no_negative_validation_group", "group_label": None})
                        continue
                    candidates = pd.DataFrame(validation_candidates)
                    for metric, ascending in [("standardized_slope", True), ("spearman_rho", True), ("negative_delta_y_fraction", False), ("n", False)]:
                        candidates[f"rank_{metric}"] = candidates[metric].rank(pct=True, ascending=ascending, method="average")
                    candidates["selection_score"] = candidates[["rank_standardized_slope", "rank_spearman_rho", "rank_negative_delta_y_fraction", "rank_n"]].mean(axis=1)
                    selected = candidates.sort_values(["selection_score", "n"], ascending=[False, False], kind="stable").iloc[0]
                    selected_test = role_data.loc[role_data[group_column].astype("string").eq(str(selected["group_label"])) & role_data["spatial_role"].eq("test")]
                    test_metrics = _simple_pair_metrics(selected_test, outcome)
                    records.append({
                        **base,
                        "status": "ok",
                        "group_label": selected["group_label"],
                        "validation_n": int(selected["n"]),
                        "validation_slope": float(selected["slope"]),
                        "validation_standardized_slope": float(selected["standardized_slope"]),
                        "validation_spearman_rho": float(selected["spearman_rho"]),
                        "validation_negative_delta_y_fraction": float(selected["negative_delta_y_fraction"]),
                        "validation_r_squared": float(selected["r_squared"]),
                        "validation_selection_score": float(selected["selection_score"]),
                        "test_n": int(test_metrics["n"]),
                        "test_slope": float(test_metrics["slope"]),
                        "test_standardized_slope": float(test_metrics["standardized_slope"]),
                        "test_spearman_rho": float(test_metrics["spearman_rho"]),
                        "test_negative_delta_y_fraction": float(test_metrics["negative_delta_y_fraction"]),
                        "test_r_squared": float(test_metrics["r_squared"]),
                        "test_direction": test_metrics["direction"],
                    })
    return pd.DataFrame.from_records(records)


def _pair_balance_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    variables = [name for name in ENVIRONMENT if f"{name}_high" in pairs and f"{name}_low" in pairs]
    rows: list[dict[str, object]] = []
    for (method, exposure), group in pairs.groupby(["method", "exposure"], sort=True):
        balance: dict[str, float] = {}
        for variable in variables:
            high = pd.to_numeric(group[f"{variable}_high"], errors="coerce").to_numpy(dtype=float)
            low = pd.to_numeric(group[f"{variable}_low"], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(high) & np.isfinite(low)
            high, low = high[valid], low[valid]
            pooled = np.sqrt((np.var(high, ddof=1) + np.var(low, ddof=1)) / 2) if len(high) > 1 else np.nan
            balance[variable] = float((np.mean(high) - np.mean(low)) / pooled) if pooled and np.isfinite(pooled) else float("nan")
        rows.append({
            "method": str(method),
            "exposure": str(exposure),
            "n_pairs": int(len(group)),
            "max_abs_environment_smd_high_vs_low": float(np.nanmax(np.abs(list(balance.values())))) if balance else float("nan"),
            "environment_smd": json.dumps(balance, ensure_ascii=False, sort_keys=True),
            "balance_scope": "matched high-erosion versus low-erosion member covariates; not pre/post treatment causal balance",
        })
    return pd.DataFrame.from_records(rows)


def _curve_directions(curves: pd.DataFrame) -> pd.DataFrame:
    if curves.empty:
        return pd.DataFrame(columns=["outcome", "exposure", "model_name", "environment_vector", "curve_change_10_90", "curve_direction"])
    records: list[dict[str, object]] = []
    required = {"outcome", "exposure", "model_name", "exposure_value", "predicted_outcome"}
    if not required.issubset(curves.columns):
        return pd.DataFrame(columns=["outcome", "exposure", "model_name", "environment_vector", "curve_change_10_90", "curve_direction"])
    for keys, group in curves.groupby(["outcome", "exposure", "model_name", "environment_vector"], dropna=False, sort=True):
        values = group.copy()
        values["exposure_value"] = pd.to_numeric(values["exposure_value"], errors="coerce")
        values["predicted_outcome"] = pd.to_numeric(values["predicted_outcome"], errors="coerce")
        values = values.loc[np.isfinite(values["exposure_value"]) & np.isfinite(values["predicted_outcome"])].sort_values("exposure_value")
        if values.empty:
            continue
        x = values["exposure_value"].to_numpy(dtype=float)
        y = values["predicted_outcome"].to_numpy(dtype=float)
        y_low = float(np.interp(np.quantile(x, 0.1), x, y))
        y_high = float(np.interp(np.quantile(x, 0.9), x, y))
        change = y_high - y_low
        scale = max(float(np.std(y)), 1e-12)
        direction = "negative" if change < -0.01 * scale else "positive" if change > 0.01 * scale else "flat"
        records.append({
            "outcome": keys[0], "exposure": keys[1], "model_name": keys[2], "environment_vector": keys[3],
            "curve_change_10_90": change, "curve_direction": direction,
        })
    return pd.DataFrame.from_records(records)


def _prepare_model_results(
    oracle_dir: Path,
    config: dict[str, Any],
    pairs: pd.DataFrame,
    pair_associations: pd.DataFrame,
    pair_balance: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source_registry_path = oracle_dir / "tables" / "experiment_results.parquet"
    registry = pd.read_parquet(source_registry_path)
    if registry.empty or "experiment_id" not in registry:
        raise ValueError(f"Existing Oracle experiment registry is empty or invalid: {source_registry_path}")
    model_results = registry.loc[registry.get("model_name", pd.Series(index=registry.index, dtype=object)).notna()].copy()
    model_results = model_results.loc[~model_results.get("regime", pd.Series(index=model_results.index, dtype=object)).fillna("").astype(str).eq("paired_difference")].copy()
    model_results["analysis_unit"] = "pixel_level"
    model_results["source_run_id"] = str(config["oracle_source_run_id"])
    curves_path = oracle_dir / str(config["response_curves"]["best_model_curve_source"])
    curves = pd.read_parquet(curves_path)
    curve_direction = _curve_directions(curves.loc[curves["exposure"].eq(config["primary_exposure"])].copy())
    if not curve_direction.empty and "environment_vector" in model_results:
        model_results = model_results.merge(
            curve_direction,
            on=["outcome", "exposure", "model_name", "environment_vector"],
            how="left",
            validate="many_to_one",
        )
        unknown = model_results["direction"].fillna("unknown").astype(str).eq("unknown") & model_results["curve_direction"].notna()
        model_results.loc[unknown, "direction"] = model_results.loc[unknown, "curve_direction"]
    for column in ["spearman_rho", "standardized_effect", "max_abs_matched_smd"]:
        if column not in model_results:
            model_results[column] = np.nan
    model_results["model_spearman_status"] = "not stored in source model trial; matched-pair Spearman is reported separately"
    if "outcome" in model_results and "exposure" in model_results:
        by_pair = pair_associations.loc[pair_associations["subset"].eq("all") & pair_associations["method"].eq("M4"), ["outcome", "exposure", "spearman_rho"]].drop_duplicates(["outcome", "exposure"])
        model_results = model_results.merge(by_pair.rename(columns={"spearman_rho": "matched_pair_spearman_rho"}), on=["outcome", "exposure"], how="left", validate="many_to_one")

    matching_types = set(config["model_search"]["apparent_model_types"])
    matched_apparent = model_results.loc[
        model_results["experiment_type"].isin(matching_types)
        & model_results["evaluation_set"].fillna("").astype(str).str.lower().eq("apparent")
    ].copy()
    matched_boards = rank_phase_g_models(matched_apparent)
    matched_count = matched_apparent.groupby(["exposure", "outcome"], dropna=False).size().to_dict()
    board_rows: list[dict[str, object]] = []
    for rank_type, table in matched_boards.items():
        for _, row in table.iterrows():
            record = row.to_dict()
            record.update(
                ranking_type=rank_type,
                analysis_unit="pixel_level_matched_candidate_or_outcome_selected_subset",
                candidate_count=int(matched_count.get((row.get("exposure"), row.get("outcome")), 0)),
                r2_scope="apparent fit to matched or outcome-selected pixel subset; not paired-difference R2",
            )
            board_rows.append(record)

    full_models = model_results.loc[
        model_results["experiment_type"].eq(config["model_search"]["full_sample_model_type"])
        & model_results["evaluation_set"].fillna("").astype(str).str.lower().eq("apparent")
    ].copy()
    full_boards = rank_phase_g_models(full_models)
    for _, row in full_boards["highest_apparent_r2"].iterrows():
        record = row.to_dict()
        record.update(
            ranking_type="highest_full_sample_apparent_r2",
            analysis_unit="pixel_level_all_eligible_cropland",
            candidate_count=int(len(full_models.loc[full_models["exposure"].eq(row["exposure"]) & full_models["outcome"].eq(row["outcome"])])),
            r2_scope="apparent prediction fit on all eligible complete-case pixels; includes environmental predictors",
        )
        board_rows.append(record)

    spatial_path = oracle_dir / "models" / "spatial_test_selected_models.parquet"
    spatial_test = pd.read_parquet(spatial_path).copy()
    spatial_test["analysis_unit"] = "pixel_level_spatial_test"
    spatial_test["source_run_id"] = str(config["oracle_source_run_id"])
    spatial_boards = rank_phase_g_models(spatial_test)
    for _, row in spatial_boards["validation_selected_spatial"].iterrows():
        record = row.to_dict()
        record.update(
            ranking_type="validation_selected_spatial_test",
            analysis_unit="pixel_level_spatial_test",
            candidate_count=1,
            r2_scope="spatial test score of model/configuration chosen by validation score",
        )
        board_rows.append(record)

    phase_f_spatial_path = ROOT / "outputs" / "exploratory_oracle" / str(config["phase_f_run_id"]) / "tables" / "repeated_spatial_validation.csv"
    phase_f_spatial = select_pair_spatial_p1_rows(pd.read_csv(phase_f_spatial_path))
    pair_spatial = summarize_pair_spatial_validation(phase_f_spatial)
    validation = pair_spatial.loc[pair_spatial["evaluation_set"].eq("validation"), ["method", "exposure", "outcome", "median_spatial_r2", "training_negative_direction_fraction", "spatial_seeds"]].rename(columns={"median_spatial_r2": "validation_r2", "training_negative_direction_fraction": "validation_training_negative_fraction"})
    test = pair_spatial.loc[pair_spatial["evaluation_set"].eq("test"), ["method", "exposure", "outcome", "median_spatial_r2", "training_negative_direction_fraction", "spatial_seeds"]].rename(columns={"median_spatial_r2": "median_test_r2", "training_negative_direction_fraction": "test_training_negative_fraction", "spatial_seeds": "test_spatial_seeds"})
    paired_candidates = pair_associations.loc[pair_associations["subset"].eq("all")].merge(
        validation, on=["method", "exposure", "outcome"], how="left", validate="many_to_one"
    ).merge(test, on=["method", "exposure", "outcome"], how="left", validate="many_to_one")
    paired_candidates = paired_candidates.merge(pair_balance[["method", "exposure", "max_abs_environment_smd_high_vs_low"]], on=["method", "exposure"], how="left", validate="many_to_one")
    paired_candidates = paired_candidates.rename(columns={
        "standardized_slope": "standardized_effect",
        "slope_delta_y_per_delta_log1p_erosion": "coefficient",
        "spearman_rho": "spearman_rho",
        "negative_delta_y_fraction": "negative_delta_y_fraction",
        "max_abs_environment_smd_high_vs_low": "max_abs_matched_smd",
    })
    paired_candidates["standardized_effect_ci_high"] = paired_candidates["standardized_slope_ci_high"]
    paired_candidates["direction"] = np.where(paired_candidates["coefficient"] < 0, "negative", np.where(paired_candidates["coefficient"] > 0, "positive", "unknown"))
    paired_candidates["candidate_id"] = "phase_f_" + paired_candidates["method"].astype(str) + "_" + paired_candidates["exposure"].astype(str)
    paired_candidates["evaluation_set"] = "apparent"
    paired_candidates["r_squared"] = paired_candidates["pearson_r_squared"]
    paired_candidates["analysis_unit"] = "matched_pair_difference"
    paired_candidates["status"] = "ok"
    paired_candidates["selection_mode"] = "outcome_guided_composite_after_full_summary"
    pair_boards = rank_phase_g_models(paired_candidates)
    for _, row in pair_boards["strongest_negative_association"].iterrows():
        record = row.to_dict()
        record.update(
            ranking_type="strongest_negative_association",
            analysis_unit="matched_pair_difference",
            candidate_count=int(len(paired_candidates.loc[paired_candidates["outcome"].eq(row["outcome"]) & paired_candidates["exposure"].eq(row["exposure"])])),
            r2_scope="paired difference association only; separate from pixel-level model R2",
            outcome_guided_selection=True,
        )
        board_rows.append(record)
    for rank_type in ["highest_apparent_r2", "highest_negative_apparent_r2"]:
        for _, row in pair_boards[rank_type].iterrows():
            record = row.to_dict()
            record.update(ranking_type=f"pair_{rank_type}", analysis_unit="matched_pair_difference", r2_scope="paired difference OLS association R2")
            board_rows.append(record)

    leaderboard = pd.DataFrame.from_records(board_rows)
    final_candidates = leaderboard.loc[leaderboard["ranking_type"].isin([
        "highest_apparent_r2", "highest_negative_apparent_r2", "strongest_negative_association",
        "highest_full_sample_apparent_r2", "validation_selected_spatial_test", "pair_highest_apparent_r2", "pair_highest_negative_apparent_r2",
    ])].copy()
    return model_results, leaderboard, final_candidates, curves, pair_spatial, phase_f_spatial


def _build_response_curve_data(
    frame: pd.DataFrame,
    selected_curves: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sample_size = int(config["response_curves"]["fixed_sample_size"])
    display_points = int(config["response_curves"]["display_points"])
    primary_soils = [int(value) for value in config["environment_groups"]["primary_soils"]]
    needed = ["Aa_20yr", *OUTCOMES, "SoilType"]
    numeric = frame[needed].apply(pd.to_numeric, errors="coerce")
    complete = frame.loc[
        frame["SoilType"].isin(primary_soils)
        & np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
    ].copy()
    n_sample = min(sample_size, len(complete))
    sample = complete.sample(n=n_sample, random_state=int(config["seed"])).sort_values("cell_id").reset_index(drop=True)
    points = sample.sample(n=min(display_points, len(sample)), random_state=int(config["seed"]) + 1)
    curve_rows: list[pd.DataFrame] = []
    point_rows: list[pd.DataFrame] = []
    for outcome in OUTCOMES:
        x = pd.to_numeric(sample["Aa_20yr"], errors="coerce").to_numpy(dtype=float)
        low, high = np.quantile(x, [0.02, 0.98])
        grid_x = np.linspace(float(low), float(high), 100)
        train = sample[["Aa_20yr", outcome]].copy()
        score = pd.DataFrame({"Aa_20yr": grid_x, outcome: float(pd.to_numeric(sample[outcome], errors="coerce").median())})
        prediction, model_config = fit_model_family(
            train,
            score,
            outcome,
            "Aa_20yr",
            ["Aa_20yr"],
            model_family="natural_spline",
            random_state=int(config["seed"]),
        )
        curve_rows.append(pd.DataFrame({
            "outcome": outcome,
            "exposure": "Aa_20yr",
            "curve_type": "fixed_common_sample_natural_spline",
            "exposure_value": grid_x,
            "predicted_outcome": prediction,
            "n_fit": int(len(sample)),
            "model_name": "natural_spline",
            "model_config": json.dumps(model_config, sort_keys=True),
            "selection_metric": "none; fixed model family used descriptively",
        }))
        selected = selected_curves.loc[
            selected_curves["outcome"].eq(outcome)
            & selected_curves["exposure"].eq("Aa_20yr")
            & selected_curves["curve_type"].eq("full_sample_best_model_partial_dependence")
        ].copy()
        if not selected.empty:
            selected["exposure_value"] = pd.to_numeric(selected["exposure_value"], errors="coerce")
            selected["predicted_outcome"] = pd.to_numeric(selected["predicted_outcome"], errors="coerce")
            selected = selected.loc[
                np.isfinite(selected["exposure_value"]) & np.isfinite(selected["predicted_outcome"])
            ].copy()
            if not selected.empty:
                curve_rows.append(pd.DataFrame({
                    "outcome": outcome,
                    "exposure": "Aa_20yr",
                    "curve_type": "oracle_best_full_sample_apparent_model",
                    "exposure_value": selected["exposure_value"].to_numpy(dtype=float),
                    "predicted_outcome": selected["predicted_outcome"].to_numpy(dtype=float),
                    "n_fit": int(pd.to_numeric(selected["n_complete_cases"], errors="coerce").iloc[0]),
                    "model_name": str(selected["model_name"].iloc[0]),
                    "model_config": selected["environment_at_median"].iloc[0] if "environment_at_median" in selected else "",
                    "apparent_r_squared": float(pd.to_numeric(selected["selected_model_r_squared"], errors="coerce").iloc[0]),
                    "selection_metric": "full-sample apparent R2; selected from 1,024 prior trials across 13 model families",
                }))
        sample_points = points[["cell_id", "Aa_20yr", outcome]].copy()
        sample_points = sample_points.rename(columns={"Aa_20yr": "exposure_value", outcome: "observed_outcome"})
        sample_points["outcome"] = outcome
        sample_points["curve_type"] = "fixed_common_sample_scatter"
        sample_points["n_fit"] = int(len(sample))
        point_rows.append(sample_points)
    curves = pd.concat(curve_rows, ignore_index=True, sort=False) if curve_rows else pd.DataFrame()
    scatter = pd.concat(point_rows, ignore_index=True, sort=False) if point_rows else pd.DataFrame()
    sample_profile = pd.DataFrame([{
        "sample_size_requested": sample_size,
        "sample_size_actual": int(len(sample)),
        "display_points_per_outcome": int(min(display_points, len(sample))),
        "seed": int(config["seed"]),
        "scope": "same complete-case cell sample across all nine health predictions; primary soils 26/27/42; Aa_20yr",
        "selection_metric": "no selection for natural spline; separate full-sample best model curve was selected by apparent R2 in prior Oracle run",
    }])
    return curves, scatter, sample_profile


def _save_figure(fig: plt.Figure, name: str, data: pd.DataFrame, caption: str, paths: dict[str, Path]) -> dict[str, object]:
    data_path = paths["review_dir"] / "figures" / "data" / f"{name}.csv"
    _write_csv(data_path, data)
    caption_path = paths["review_dir"] / "figures" / "captions" / f"{name}.md"
    caption_path.parent.mkdir(parents=True, exist_ok=True)
    caption_path.write_text(caption.strip() + "\n", encoding="utf-8")
    files: dict[str, str] = {}
    for extension in ("png", "svg", "pdf"):
        target = paths["output_figures"] / f"{name}.{extension}"
        fig.savefig(target, dpi=320, bbox_inches="tight")
        review_target = paths["review_dir"] / "figures" / f"{name}.{extension}"
        review_target.parent.mkdir(parents=True, exist_ok=True)
        review_target.write_bytes(target.read_bytes())
        files[extension] = str(review_target.relative_to(paths["review_dir"]))
    plt.close(fig)
    return {
        "name": name,
        "title": FIGURE_TITLES[name],
        "files": files,
        "data": str(data_path.relative_to(paths["review_dir"])),
        "caption": str(caption_path.relative_to(paths["review_dir"])),
        "rows": int(len(data)),
    }


def _generate_figures(
    *,
    paths: dict[str, Path],
    config: dict[str, Any],
    pairs_grouped: pd.DataFrame,
    association_results: pd.DataFrame,
    matching_configurations: pd.DataFrame,
    matching_balance: pd.DataFrame,
    leaderboard: pd.DataFrame,
    group_associations: pd.DataFrame,
    group_leaderboard: pd.DataFrame,
    spatial_selected: pd.DataFrame,
    phase_f_spatial: pd.DataFrame,
    response_curves: pd.DataFrame,
    response_scatter: pd.DataFrame,
) -> list[dict[str, object]]:
    plt.rcParams.update({"font.family": "Noto Sans CJK JP", "axes.titlesize": 11, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "axes.edgecolor": "#4B5563", "axes.linewidth": 0.8, "grid.color": "#D9DEE7", "grid.linewidth": 0.55, "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"})
    figures: list[dict[str, object]] = []
    outcomes = list(config["outcomes"])
    outcome_labels = {"SHI_candidate": "土壤健康指数（SHI）", "AN": "碱解氮（AN）", "OM": "有机质（OM）", "CEC": "阳离子交换量（CEC）", "pH": "土壤酸碱度（pH）", "SW": "土壤含水量（SW）", "BD": "土壤容重（BD）", "MBC": "微生物量碳（MBC）", "RESP": "土壤呼吸（RESP）"}
    group_labels = {"soil_type": "土壤类型", "kmeans_v5": "V5 土类内 K 均值", "hierarchical_ward_v5": "V5 土类内 Ward 聚类", "gradient_Elevation": "高程分位组", "gradient_TWI": "地形湿度指数分位组", "gradient_LST": "地表温度分位组", "gradient_NDWI": "NDWI 分位组", "gradient_NDVI": "NDVI 分位组", "gradient_Precipitation": "降水分位组"}
    model_labels = {"random_forest": "随机森林", "extra_trees": "极端随机树", "hist_gradient_boosting": "直方图梯度提升", "ols": "普通最小二乘", "piecewise": "分段回归", "log_exposure": "对数暴露回归", "environment_group_fixed_effects": "环境组固定效应"}
    palette = {"blue": "#3D6F91", "orange": "#D99139", "red": "#B4494B", "teal": "#3B8C83", "gray": "#7A8084", "light": "#D9DEE2"}

    # Figure 1: outcome-blind matching search coverage and balance.
    balance = matching_balance.loc[matching_balance.get("diagnostic_type", pd.Series(index=matching_balance.index, dtype=object)).eq("local_grid_variable_balance")].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    if not matching_configurations.empty:
        x = pd.to_numeric(matching_configurations.get("max_abs_matched_smd"), errors="coerce")
        y = pd.to_numeric(matching_configurations.get("area_coverage"), errors="coerce")
        color = pd.to_numeric(matching_configurations.get("median_abs_log1p_Aa_difference"), errors="coerce")
        valid = np.isfinite(x) & np.isfinite(y)
        scatter = axes[0].scatter(x[valid], y[valid], c=color[valid], cmap="cividis", s=34, alpha=0.78, edgecolors="white", linewidths=.35)
        axes[0].axvline(0.1, color=palette["red"], linestyle="--", linewidth=0.9, label="SMD = 0.10 参考线")
        axes[0].set(xlabel="匹配后最大绝对 SMD", ylabel="匹配覆盖率", title="Phase F 的 243 组局地候选")
        axes[0].legend(loc="best", frameon=False)
        fig.colorbar(scatter, ax=axes[0], label="|Δlog1p(Aa)| 中位数")
    else:
        axes[0].text(0.5, 0.5, "没有可用的匹配摘要", ha="center", va="center", transform=axes[0].transAxes)
    if not balance.empty and {"smd_before", "smd_after"}.issubset(balance.columns):
        before = pd.to_numeric(balance["smd_before"], errors="coerce").abs().dropna().to_numpy()
        after = pd.to_numeric(balance["smd_after"], errors="coerce").abs().dropna().to_numpy()
        axes[1].boxplot([before, after], labels=["匹配前", "匹配后"], showfliers=False, whis=(10, 90), patch_artist=True, boxprops={"facecolor": "#AFC6D5"}, medianprops={"color": palette["red"], "linewidth": 1.5})
        axes[1].set(ylabel="绝对标准化均值差异（SMD）", title="环境协变量平衡")
        axes[1].grid(axis="y", alpha=0.2)
    else:
        axes[1].text(0.5, 0.5, "没有可用的平衡诊断记录", ha="center", va="center", transform=axes[1].transAxes)
    figures.append(_save_figure(fig, "01_matching_quality", matching_configurations, "候选覆盖率与平衡诊断来自既有 Phase F 结果盲局地搜索。SMD 参考线仅作描述，不单独用于判定最终匹配方案。", paths))

    # Figure 2: native-unit paired health differences for primary M4/Aa_20yr.
    source = pairs_grouped.loc[pairs_grouped["method"].eq("M4") & pairs_grouped["exposure"].eq("Aa_20yr")]
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), constrained_layout=True)
    rows: list[dict[str, object]] = []
    for ax, outcome in zip(axes.flat, outcomes, strict=True):
        y = pd.to_numeric(source[f"high_{outcome}"], errors="coerce") - pd.to_numeric(source[f"low_{outcome}"], errors="coerce")
        y = y[np.isfinite(y)]
        if len(y):
            ax.boxplot([y], vert=True, showfliers=False, whis=(10, 90), widths=0.45, patch_artist=True, boxprops={"facecolor": "#AFC6D5"}, medianprops={"color": palette["red"], "linewidth": 1.5})
            ax.axhline(0, color=palette["gray"], linestyle="--", linewidth=0.7)
            ax.scatter([1], [float(y.mean())], marker="D", color=palette["orange"], s=24, zorder=3, label="均值")
            ax.set_xticks([])
            ax.set_title(f"{outcome_labels.get(outcome,outcome)}  （配对数={len(y):,}）", loc="left", fontweight="semibold")
            rows.append({"outcome": outcome, "n_pairs": len(y), "mean_delta_y": float(y.mean()), "median_delta_y": float(y.median()), "p10_delta_y": float(y.quantile(0.1)), "p90_delta_y": float(y.quantile(0.9)), "negative_fraction": float((y < 0).mean())})
        ax.set_ylabel("高侵蚀端 − 低侵蚀端\n（预测栅格值）")
    axes.flat[0].legend(loc="best", frameon=False)
    fig.suptitle("M4 土类内最近邻配对｜20 年平均实际侵蚀｜健康预测差值", x=.02, y=1.02, ha="left", fontsize=15, fontweight="bold")
    figures.append(_save_figure(fig, "02_paired_health_differences", pd.DataFrame(rows), "每个箱线图汇总复用 Phase F M4 配对中高侵蚀端减低侵蚀端的健康预测值。须线表示第 10–90 百分位，菱形表示均值。单位沿用源栅格；部分指标单位元数据仍未核实。", paths))

    # Figure 3: paired-difference Spearman estimates for both Aa periods.
    heat = association_results.loc[association_results["method"].eq("M4") & association_results["subset"].eq("all")].copy()
    pivot = heat.pivot(index="outcome", columns="exposure", values="spearman_rho").reindex(outcomes)
    fig, ax = plt.subplots(figsize=(7.5, 6.2), constrained_layout=True)
    matrix = pivot.reindex(columns=["Aa_20yr", "Aa_2020"]).to_numpy(dtype=float)
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(2), ["20 年平均侵蚀", "2020 年侵蚀"])
    ax.set_yticks(range(len(outcomes)), [outcome_labels.get(o,o) for o in outcomes])
    for row_index, outcome in enumerate(outcomes):
        for col_index, exposure in enumerate(["Aa_20yr", "Aa_2020"]):
            item = heat.loc[heat["outcome"].eq(outcome) & heat["exposure"].eq(exposure)]
            if item.empty:
                label = "NA"
            else:
                r = float(item["spearman_rho"].iloc[0])
                n = int(item["n_pairs"].iloc[0])
                label = f"ρ={r:.2f}\nn={n:,}"
            ax.text(col_index, row_index, label, ha="center", va="center", fontsize=7, color="#17232B")
    ax.set_title("配对差值的 Spearman 秩相关：Δ健康与 Δlog1p(Aa)", loc="left", fontweight="bold")
    fig.colorbar(image, ax=ax, label="Spearman 秩相关系数 ρ")
    figures.append(_save_figure(fig, "03_pair_association_heatmap", heat, "Spearman 相关以配对高减低侵蚀差和高减低健康差计算。配套表中的 95% 相关区间使用 Fisher-z 近似，并以 5 km 配对中点空间块数作为有效样本量；在空间依赖存在时，P 值仍仅作描述性参考。", paths))

    # Figure 4: keep pixel-model R2 levels together; pair R2 is excluded.
    r2_types = ["highest_full_sample_apparent_r2", "highest_apparent_r2", "highest_negative_apparent_r2", "validation_selected_spatial_test"]
    r2_labels = ["全体合格像元\n最高表观拟合", "匹配子集\n最高表观拟合", "匹配子集\n最高负向拟合", "验证集选模后\n空间测试"]
    r2_rows: list[dict[str, object]] = []
    for rank_type, label in zip(r2_types, r2_labels, strict=True):
        selected = leaderboard.loc[leaderboard["ranking_type"].eq(rank_type) & leaderboard["exposure"].eq("Aa_20yr")]
        for _, row in selected.iterrows():
            r2_rows.append({"outcome": row.get("outcome"), "ranking_type": rank_type, "label": label, "r_squared": _as_float(row.get("r_squared")), "candidate_id": row.get("candidate_id"), "model_name": row.get("model_name"), "n": row.get("n"), "direction": row.get("direction")})
    r2_data = pd.DataFrame.from_records(r2_rows)
    fig, ax = plt.subplots(figsize=(13, 5.5), constrained_layout=True)
    x = np.arange(len(outcomes))
    width = 0.19
    colors = [palette["gray"], palette["blue"], palette["orange"], palette["teal"]]
    for idx, (rank_type, label, color) in enumerate(zip(r2_types, r2_labels, colors, strict=True)):
        values = r2_data.loc[r2_data["ranking_type"].eq(rank_type)].set_index("outcome")["r_squared"].reindex(outcomes)
        ax.bar(x + (idx - 1.5) * width, values, width=width, color=color, label=label.replace("\n", " "))
    ax.axhline(0, color="#222222", linewidth=0.7)
    ax.set_xticks(x, [outcome_labels.get(o,o) for o in outcomes], rotation=25, ha="right")
    ax.set_ylabel("R²（各行按对应测试集定义）")
    ax.set_title("20 年平均实际侵蚀下的像元级健康预测拟合")
    ax.legend(frameon=False, ncol=2, loc="best")
    ax.text(0.01, -0.27, "全像元与匹配子集模型均包含环境变量，R²不代表侵蚀的独立贡献；配对差值模型另行报告。", transform=ax.transAxes, fontsize=8, color="#4B5563")
    figures.append(_save_figure(fig, "04_model_r2_comparison", r2_data, "图中 R²均为像元级健康栅格预测拟合。全体合格像元、结果导向筛选的匹配子集和空间测试块具有不同样本范围；匹配配对差值模型的 R²不在此图混排。", paths))

    # Figure 5: natural spline and previously selected full-sample best model.
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), constrained_layout=True)
    for position, (ax, outcome) in enumerate(zip(axes.flat, outcomes, strict=True)):
        scatter = response_scatter.loc[response_scatter["outcome"].eq(outcome)]
        ax.scatter(scatter["exposure_value"], scatter["observed_outcome"], s=3, color="#87939B", alpha=0.11, rasterized=True, label="固定抽样像元")
        natural = response_curves.loc[response_curves["outcome"].eq(outcome) & response_curves["curve_type"].eq("fixed_common_sample_natural_spline")].sort_values("exposure_value")
        if not natural.empty:
            ax.plot(natural["exposure_value"], natural["predicted_outcome"], color=palette["orange"], linestyle="--", linewidth=1.7, label="固定样本自然样条")
        best = response_curves.loc[response_curves["outcome"].eq(outcome) & response_curves["curve_type"].eq("oracle_best_full_sample_apparent_model")].sort_values("exposure_value")
        title = outcome
        if not best.empty:
            ax.plot(best["exposure_value"], best["predicted_outcome"], color=palette["red"], linewidth=1.7, label="历史全样本择优表观模型")
            best_row = best.iloc[0]
            title = f"{outcome_labels.get(outcome,outcome)} · {model_labels.get(str(best_row['model_name']),str(best_row['model_name']))} · R²={_as_float(best_row.get('apparent_r_squared')):.3f}"
        ax.set_title(title, fontsize=9, loc="left", fontweight="semibold")
        ax.set_xlabel("20 年平均实际侵蚀 Aa（源栅格值）")
        ax.set_ylabel("健康指标预测值")
        if position == 0:
            ax.legend(frameon=False, fontsize=6)
    fig.suptitle("两类描述性响应曲线｜固定 20,000 像元样本与历史 Oracle 择优模型", x=.02, y=1.02, ha="left", fontsize=15, fontweight="bold")
    figures.append(_save_figure(fig, "05_response_curves", response_curves, "虚线自然样条拟合于固定的共同 20,000 个完整像元样本（土类 26/27/42），未按拟合优度挑选。红色实线为此前 13 类模型、1,024 次全样本表观 R²搜索中入选的模型曲线；环境变量固定在记录的中位数或参考值。两者均描述预测栅格，不是独立验证。", paths))

    # Figure 6: standardized negative slopes in the largest-support candidate groups.
    group_data = group_associations.loc[
        group_associations["method"].eq("M4")
        & group_associations["exposure"].eq("Aa_20yr")
        & group_associations["n_pairs"].ge(100)
        & group_associations["direction"].eq("negative")
        & pd.to_numeric(group_associations["spearman_rho"], errors="coerce").lt(0)
    ].copy()
    forest_rows: list[dict[str, object]] = []
    for outcome in outcomes:
        candidates = group_data.loc[group_data["outcome"].eq(outcome)].sort_values(["standardized_slope", "n_pairs"], ascending=[True, False]).head(2)
        forest_rows.extend(candidates.to_dict(orient="records"))
    forest = pd.DataFrame.from_records(forest_rows)
    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
    if not forest.empty:
        forest = forest.sort_values(["outcome", "standardized_slope"], ascending=[True, True]).reset_index(drop=True)
        y_positions = np.arange(len(forest))
        low = pd.to_numeric(forest["standardized_slope_ci_low"], errors="coerce").to_numpy(dtype=float)
        mid = pd.to_numeric(forest["standardized_slope"], errors="coerce").to_numpy(dtype=float)
        high = pd.to_numeric(forest["standardized_slope_ci_high"], errors="coerce").to_numpy(dtype=float)
        valid_ci = np.isfinite(low) & np.isfinite(high) & np.isfinite(mid)
        ax.errorbar(mid[valid_ci], y_positions[valid_ci], xerr=np.vstack([mid[valid_ci] - low[valid_ci], high[valid_ci] - mid[valid_ci]]), fmt="o", color=palette["blue"], ecolor="#87939B", capsize=2, markersize=4)
        labels = [f"{outcome_labels.get(str(row.outcome), str(row.outcome))} · {group_labels.get(str(row.group_method), str(row.group_method))}：{_display_group_label(row.group_label)}（n={int(row.n_pairs):,}）" for row in forest.itertuples()]
        ax.set_yticks(y_positions, labels)
        ax.axvline(0, color=palette["red"], linestyle="--", linewidth=0.8)
        ax.set_xlabel("标准化斜率：Δ健康 / Δlog1p(Aa)（空间块自助 95% 区间）")
        ax.set_title("各指标负向关联最强的两个环境组候选（M4，20 年平均侵蚀）")
        ax.grid(axis="x", alpha=0.2)
    else:
        ax.text(0.5, 0.5, "没有环境组同时达到样本量与负向关联条件", ha="center", va="center", transform=ax.transAxes)
    figures.append(_save_figure(fig, "06_environment_group_forest", forest, "入图候选要求至少 100 个配对，且斜率与 Spearman 方向均为负。经 Oracle 授权后，按健康结局对环境组排序。区间按 5 km 配对中点空间块重抽样；候选选择已使用全研究区，不构成独立验证。", paths))

    # Figure 7: distribution of group-specific rank correlations across methods.
    group_methods = list(dict.fromkeys(group_associations["group_method"].astype(str))) if not group_associations.empty else []
    method_order = [name for name in ["soil_type", "kmeans_v5", "hierarchical_ward_v5", "gradient_Elevation", "gradient_TWI", "gradient_LST", "gradient_NDWI", "gradient_NDVI", "gradient_Precipitation"] if name in group_methods]
    method_data = group_associations.loc[
        group_associations["method"].eq("M4") & group_associations["exposure"].eq("Aa_20yr") & group_associations["n_pairs"].ge(100)
    ].copy()
    fig, ax = plt.subplots(figsize=(12, 5.6), constrained_layout=True)
    box_data = [pd.to_numeric(method_data.loc[method_data["group_method"].eq(name), "spearman_rho"], errors="coerce").dropna().to_numpy() for name in method_order]
    if any(len(values) for values in box_data):
        ax.boxplot(box_data, labels=[f"{group_labels.get(name,name)}\n（组数={len(values)}）" for name, values in zip(method_order, box_data, strict=True)], showfliers=False, whis=(10, 90), patch_artist=True, boxprops={"facecolor": "#C7D8E3"}, medianprops={"color": palette["blue"], "linewidth": 1.5})
        ax.axhline(0, color=palette["red"], linestyle="--", linewidth=0.8)
        ax.set_ylabel("组内 Spearman 秩相关系数 ρ")
        ax.set_title("结果盲环境定义下的组内关联分布")
        ax.tick_params(axis="x", rotation=20)
    else:
        ax.text(0.5, 0.5, "没有可估计的环境组", ha="center", va="center", transform=ax.transAxes)
    figures.append(_save_figure(fig, "07_grouping_method_comparison", method_data, "每个分布汇总九项结局中各环境组的 Spearman 相关；每组至少保留 100 个配对。土类、K 均值、Ward 聚类和单变量分位组均由环境/土壤值定义，不使用健康值；高亮组的选择则使用了健康结局。", paths))

    # Figure 8: project key outcome-selected groups onto the matched pair map.
    map_source = pairs_grouped.loc[pairs_grouped["method"].eq("M4") & pairs_grouped["exposure"].eq("Aa_20yr")].copy()
    best_group_rows = group_leaderboard.loc[
        group_leaderboard["ranking_type"].eq("strongest_negative_group")
        & group_leaderboard["method"].eq("M4")
        & group_leaderboard["exposure"].eq("Aa_20yr")
        & group_leaderboard["outcome"].isin(["SHI_candidate", "OM", "SW"])
    ].copy()
    chosen_groups: list[dict[str, object]] = []
    for outcome in ["SHI_candidate", "OM", "SW"]:
        candidates = best_group_rows.loc[best_group_rows["outcome"].eq(outcome)].sort_values("negative_strength_score", ascending=False)
        if not candidates.empty:
            chosen_groups.append(candidates.iloc[0].to_dict())
    map_rows: list[pd.DataFrame] = []
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 5.5), constrained_layout=False)
    for ax, outcome in zip(axes.flat, ["SHI_candidate", "OM", "SW"], strict=True):
        selected = next((row for row in chosen_groups if row.get("outcome") == outcome), None)
        if selected is None:
            ax.text(0.5, 0.5, "没有符合条件的负向环境组", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(outcome_labels.get(outcome,outcome))
            continue
        group_column = str(selected["group_column"])
        group_label = str(selected["group_label"])
        points = map_source.sample(n=min(15000, len(map_source)), random_state=int(config["seed"]) + len(outcome))
        is_selected = points[group_column].astype("string").eq(group_label)
        ax.scatter(points.loc[~is_selected, "x_mid"], points.loc[~is_selected, "y_mid"], s=2, color="#C9CFD3", alpha=0.3, rasterized=True, label="其他匹配配对")
        ax.scatter(points.loc[is_selected, "x_mid"], points.loc[is_selected, "y_mid"], s=5, color=palette["red"], alpha=0.75, rasterized=True, label="结果导向入选组")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("投影坐标东向（米）")
        ax.set_ylabel("投影坐标北向（米）")
        display_label = _display_group_label(group_label)
        ax.set_title(f"{outcome_labels.get(outcome,outcome)}\n{group_labels.get(str(selected['group_method']),str(selected['group_method']))} · {display_label}", loc="left", fontweight="semibold")
        map_rows.append(pd.DataFrame({"outcome": outcome, "group_method": selected["group_method"], "group_label": group_label,
                                      "display_group_label": display_label, "candidate_id": selected.get("candidate_id"),
                                      "n_map_points": len(points), "n_selected_group_points": int(is_selected.sum())}, index=[0]))
        if outcome == "SHI_candidate":
            ax.legend(frameon=False, loc="best")
    fig.suptitle("M4 20 年平均侵蚀配对中点｜健康结局筛选的环境组位置", x=.025, y=.975, ha="left", fontsize=15, fontweight="bold")
    fig.subplots_adjust(left=.065, right=.99, bottom=.16, top=.79, wspace=.27)
    map_data = pd.concat(map_rows, ignore_index=True) if map_rows else pd.DataFrame()
    figures.append(_save_figure(fig, "08_key_group_maps", map_data, "每个面板展示固定随机抽取的 15,000 个 M4、20 年平均侵蚀配对中点。红色为该指标按复合负向关联指标选中的环境组，灰色为其他样本。坐标为投影坐标，单位米。这些点图表示配对位置，不是连续栅格面或田间实测点。", paths))

    # Figure 9: validation-selected pixel model and repeated paired holdouts.
    selected_test = spatial_selected.loc[spatial_selected["exposure"].eq("Aa_20yr")].copy()
    repeated = phase_f_spatial.loc[
        phase_f_spatial["method"].isin(["M3", "M4"])
        & phase_f_spatial["exposure"].eq("Aa_20yr")
        & phase_f_spatial["outcome"].isin(outcomes)
        & phase_f_spatial["evaluation_set"].eq("test")
        & phase_f_spatial["validation_model"].eq("P1_intercept_ols")
    ].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    x = np.arange(len(outcomes))
    width = 0.35
    validation = selected_test.set_index("outcome")["validation_r2"].reindex(outcomes) if not selected_test.empty else pd.Series(index=outcomes, dtype=float)
    test_values = selected_test.set_index("outcome")["r_squared"].reindex(outcomes) if not selected_test.empty else pd.Series(index=outcomes, dtype=float)
    axes[0].bar(x - width / 2, validation, width=width, color=palette["blue"], label="验证集选模得分")
    axes[0].bar(x + width / 2, test_values, width=width, color=palette["orange"], label="空间测试得分")
    axes[0].axhline(0, color="#222222", linewidth=0.7)
    axes[0].set_xticks(x, [outcome_labels.get(o,o) for o in outcomes], rotation=30, ha="right")
    axes[0].set_ylabel("像元级 R²")
    axes[0].set_title("验证集选出的历史 Oracle 候选")
    axes[0].legend(frameon=False)
    box_data: list[np.ndarray] = []
    box_labels: list[str] = []
    for outcome in outcomes:
        for method in ["M3", "M4"]:
            values = pd.to_numeric(repeated.loc[repeated["outcome"].eq(outcome) & repeated["method"].eq(method), "r_squared"], errors="coerce").dropna().to_numpy()
            if len(values):
                box_data.append(values)
                box_labels.append(f"{outcome_labels.get(outcome,outcome)}\n{method}")
    if box_data:
        positions = np.arange(len(box_data))
        axes[1].boxplot(box_data, positions=positions, showfliers=False, patch_artist=True, boxprops={"facecolor": "#C7D8E3"}, medianprops={"color": palette["red"]})
        axes[1].set_xticks(positions, box_labels, rotation=55, ha="right")
        axes[1].axhline(0, color="#222222", linewidth=0.7)
        axes[1].set_ylabel("匹配配对 Δ健康模型的测试 R²")
        axes[1].set_title("M3/M4 在 10 次历史 5 km 空间随机划分中的表现")
    else:
        axes[1].text(0.5, 0.5, "没有可用的重复空间配对记录", ha="center", va="center", transform=axes[1].transAxes)
    fig.suptitle("历史 Oracle 探索之后的内部空间留出诊断", x=.02, y=1.02, ha="left", fontsize=15, fontweight="bold")
    spatial_fig_data = pd.concat([
        selected_test.assign(result_type="validation_selected_pixel_test"),
        repeated.assign(result_type="repeated_pair_test"),
    ], ignore_index=True, sort=False)
    figures.append(_save_figure(fig, "09_spatial_validation", spatial_fig_data, "左图为历史像元模型经验证集筛选后，在空间测试块上的得分；右图为 Phase F M3/M4 配对 OLS 在 10 次 5 km 空间随机划分中的测试 R²。由于此前 Oracle 搜索已使用全研究区，这些划分是内部诊断，不是未触碰的外部验证；像元和配对指标分别呈现。", paths))

    index_path = paths["review_dir"] / "figures" / "figure_index.json"
    _write_json(index_path, figures)
    _write_json(paths["output_figures"] / "figure_index.json", figures)
    return figures


def _as_float(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def _group_stability_summary(spatial: pd.DataFrame) -> pd.DataFrame:
    if spatial.empty:
        return pd.DataFrame()
    valid = spatial.loc[spatial["status"].eq("ok")].copy()
    if valid.empty:
        return pd.DataFrame()
    return valid.groupby(["method", "exposure", "outcome", "group_method"], dropna=False).agg(
        repeats=("seed", "nunique"),
        successful_validation_selections=("group_label", "count"),
        validation_selected_group_count=("group_label", "nunique"),
        selected_group=("group_label", lambda values: values.value_counts().index[0]),
        modal_group_selection_fraction=("group_label", lambda values: float(values.value_counts(normalize=True).iloc[0])),
        test_negative_direction_fraction=("test_direction", lambda values: float(pd.Series(values).eq("negative").mean())),
        median_test_spearman=("test_spearman_rho", "median"),
        median_test_r2=("test_r_squared", "median"),
        median_test_pairs=("test_n", "median"),
    ).reset_index()


def _new_registry_rows(
    pair_associations: pd.DataFrame,
    group_associations: pd.DataFrame,
    leaderboard: pd.DataFrame,
    group_spatial: pd.DataFrame,
    *,
    run_id: str,
    baseline_sha: str,
    input_run_id: str,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    sequence = 1
    groups = [
        ("pair_association", pair_associations, "matched_pair_difference"),
        ("environment_group_association", group_associations, "matched_pair_difference_by_environment_group"),
        ("model_or_association_ranking", leaderboard, "ranked_candidate"),
        ("group_validation_selection", group_spatial, "validation_selected_group_spatial_test"),
    ]
    for experiment_type, table, analysis_unit in groups:
        for _, result in table.iterrows():
            outcome = str(result.get("outcome", "unknown"))
            exposure = str(result.get("exposure", "unknown"))
            row_n = _as_float(result.get("n_pairs"))
            if not np.isfinite(row_n):
                row_n = _as_float(result.get("n"))
            if not np.isfinite(row_n):
                row_n = _as_float(result.get("test_n"))
            status = "success" if str(result.get("status", "ok")).lower() in {"ok", "success", "nan"} and np.isfinite(row_n) and row_n > 0 else "failed"
            rows.append({
                "experiment_id": f"{run_id}_{sequence:06d}_{experiment_type}",
                "run_id": run_id,
                "experiment_type": experiment_type,
                "git_sha": baseline_sha,
                "input_run_id": input_run_id,
                "source_run_id": result.get("source_run_id", "phase_f_and_prior_oracle"),
                "exposure": exposure,
                "health_indicator": outcome,
                "outcome": outcome,
                "outcome_guided_selection": True,
                "selection_outcomes": json.dumps([outcome], ensure_ascii=False),
                "selection_objective": result.get("selection_objective", result.get("ranking_type", "summarize and rank exploratory results")),
                "selection_metric": result.get("ranking_type", result.get("selection_metric", "descriptive association")),
                "analysis_unit": analysis_unit,
                "candidate_id": result.get("candidate_id", result.get("group_label", result.get("method"))),
                "group_method": result.get("group_method"),
                "group_label": result.get("group_label"),
                "evaluation_set": result.get("evaluation_set", "full_sample_descriptive"),
                "random_seed": result.get("seed", seed),
                "n": row_n if np.isfinite(row_n) else None,
                "n_pairs": result.get("n_pairs") if np.isfinite(_as_float(result.get("n_pairs"))) else result.get("test_n"),
                "n_cells": result.get("n_cells"),
                "sample_retention_ratio": result.get("sample_retention_fraction", result.get("sample_retention_ratio")),
                "candidate_trial_count": result.get("candidate_count", result.get("candidate_group_count", result.get("candidate_trial_count"))),
                "r_squared": result.get("r_squared", result.get("pearson_r_squared")),
                "rmse": result.get("rmse"),
                "mae": result.get("mae"),
                "coefficient": result.get("coefficient", result.get("slope_delta_y_per_delta_log1p_erosion")),
                "direction": result.get("direction", result.get("test_direction")),
                "spearman_rho": result.get("spearman_rho", result.get("test_spearman_rho")),
                "status": status,
                "error": (result.get("error") or "No candidate met the minimum validation support and negative-direction rule") if status == "failed" else None,
            })
            sequence += 1
    return pd.DataFrame.from_records(rows)


def _md(value: object, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "—"
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return "—"
        return f"{float(value):.{digits}f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _markdown_table(frame: pd.DataFrame, columns: list[tuple[str, str]], digits: int = 3) -> str:
    if frame.empty:
        return "（无可用记录）"
    headers = [label for _, label in columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in frame.iterrows():
        cells = []
        for name, _ in columns:
            value = row.get(name)
            if name.endswith("_p") and pd.notna(value) and np.isfinite(_as_float(value)) and 0 <= _as_float(value) < 0.001:
                cells.append("<0.001")
            else:
                cells.append(_md(value, digits))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _display_candidate_table(leaderboard: pd.DataFrame, ranking_types: list[str], exposure: str) -> pd.DataFrame:
    selected = leaderboard.loc[leaderboard["ranking_type"].isin(ranking_types) & leaderboard["exposure"].eq(exposure)].copy()
    preferred = ["outcome", "ranking_type", "candidate_id", "model_name", "r_squared", "direction", "n", "group_method", "group_label", "spearman_rho", "standardized_effect", "negative_delta_y_fraction", "validation_r2", "median_test_r2"]
    columns = [name for name in preferred if name in selected]
    return selected[columns].sort_values(["outcome", "ranking_type"], kind="stable") if not selected.empty else selected


def _write_report(
    *,
    report_path: Path,
    config: dict[str, Any],
    source_profile: dict[str, object],
    pair_profile: dict[str, object],
    pair_associations: pd.DataFrame,
    leaderboard: pd.DataFrame,
    group_leaderboard: pd.DataFrame,
    group_stability: pd.DataFrame,
    spatial_selected: pd.DataFrame,
    figure_index: list[dict[str, object]],
    source_sha: str,
    oracle_source_sha: str,
    software: dict[str, str],
    run_id: str,
    output_dir: Path,
) -> None:
    primary = pair_associations.loc[
        pair_associations["method"].eq("M4") & pair_associations["exposure"].eq("Aa_20yr") & pair_associations["subset"].eq("all")
    ].copy().set_index("outcome").reindex(OUTCOMES).reset_index()
    secondary = pair_associations.loc[
        pair_associations["method"].eq("M4") & pair_associations["exposure"].eq("Aa_2020") & pair_associations["subset"].eq("all")
    ].copy().set_index("outcome").reindex(OUTCOMES).reset_index()
    primary_model_rows = _display_candidate_table(
        leaderboard,
        ["highest_apparent_r2", "highest_negative_apparent_r2", "highest_full_sample_apparent_r2", "validation_selected_spatial_test", "strongest_negative_association"],
        "Aa_20yr",
    )
    top_groups = group_leaderboard.loc[
        group_leaderboard["exposure"].eq("Aa_20yr")
        & group_leaderboard["method"].eq("M4")
        & group_leaderboard["ranking_type"].eq("strongest_negative_group")
    ].sort_values("negative_strength_score", ascending=False).head(15) if not group_leaderboard.empty else pd.DataFrame()
    stable_groups = group_leaderboard.loc[
        group_leaderboard["ranking_type"].eq("most_stable_negative_direction_across_matching_methods_and_Aa_periods")
    ].head(15) if not group_leaderboard.empty else pd.DataFrame()
    spatial_main = spatial_selected.loc[spatial_selected["exposure"].eq("Aa_20yr")].copy()
    output_rel = Path("../../../../") / output_dir.relative_to(ROOT)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    figure_lines = "\n".join(
        f"- **{item['name']}**: [PNG](figures/{item['name']}.png) · [SVG](figures/{item['name']}.svg) · [PDF](figures/{item['name']}.pdf) · [数据](figures/data/{item['name']}.csv) · [图注](figures/captions/{item['name']}.md)"
        for item in figure_index
    )
    report = f"""# Phase G 探索性研究报告：环境匹配后的侵蚀—土壤健康关联与环境异质性

> **分析模式：** `EXPLORATORY_ORACLE`，`outcome_guided_selection=true`。本报告回答的是现有 30 m 健康预测栅格中的条件关联与空间异质性，不是正式因果估计，也不是独立实测验证。

## 1. 研究问题与本轮结论

本轮推进研究总纲 Q1–Q3：环境匹配后侵蚀与健康指标是否相关、不同环境背景下关联是否不同、九项指标是否呈现不同响应。主要结论来自预计算的实际侵蚀 `Aa_20yr`，`Aa_2020` 为时间对照。两期与九项结局均单独记录。

Phase F 的 M4 直接最近邻配对中，Aa_20yr 的 Δ健康方向如下表；Δ定义为高侵蚀成员减低侵蚀成员，侵蚀差使用 `log1p(Aa_high)-log1p(Aa_low)`。Spearman 和 Kendall 反映单调关系，Wilcoxon 检验仅作为描述性配对检验；区间和显著性不能消除像元空间依赖。

{_markdown_table(primary, [("outcome", "指标"), ("n_pairs", "配对数"), ("mean_delta_y", "平均Δ健康"), ("median_delta_y", "中位Δ健康"), ("negative_delta_y_fraction", "Δ健康<0比例"), ("spearman_rho", "Spearman ρ"), ("spearman_ci_low_block_adjusted", "ρ CI下限"), ("spearman_ci_high_block_adjusted", "ρ CI上限"), ("kendall_tau", "Kendall τ"), ("wilcoxon_p", "Wilcoxon P")])}

2020 年 Aa 的对应结果为：

{_markdown_table(secondary, [("outcome", "指标"), ("n_pairs", "配对数"), ("mean_delta_y", "平均Δ健康"), ("median_delta_y", "中位Δ健康"), ("negative_delta_y_fraction", "Δ健康<0比例"), ("spearman_rho", "Spearman ρ"), ("spearman_ci_low_block_adjusted", "ρ CI下限"), ("spearman_ci_high_block_adjusted", "ρ CI上限"), ("kendall_tau", "Kendall τ"), ("wilcoxon_p", "Wilcoxon P")])}

九项指标方向并不一致；应以逐项结果为准，不能概括成“侵蚀普遍降低土壤健康”。要判断侵蚀差值是否与健康差值同步变化，优先看 Spearman/Kendall、标准化斜率及其区间，而不能只看高低成员均值差或 P 值。

## 2. 项目定位与研究治理

- 当前 Git 基线：`{source_sha}`；总纲为 `docs/RESEARCH_CHARTER.md` V1.1。分析脚本、配置和本报告在生成时尚未提交。
- 历史模型筛选来源：Oracle run `{config['oracle_source_run_id']}`，其来源代码提交为 `{oracle_source_sha}`。Phase G 将其记录作为候选搜索证据并重新排名；不会把该历史提交冒充为本轮代码版本。
- 本轮只写入 `outputs/exploratory_oracle/{run_id}/` 和 `docs/reviews/2026-10-27/phase-g/`，不修改总纲、不重算/验证 RUSLE 因子乘积、不重跑 PCNM。
- 本阶段是探索性研究输出，不表示正式 G5 或设计锁定完成。依据 `CHATGPT_REVIEW_PROTOCOL.md`，空间测试不得用于参数或模型选择；已有候选按验证集分数选定后再读取其测试分数。

## 3. 数据、空间支持与限制

- 输入 run：`{config['input_run_id']}`；环境表和健康表各含 392,527 行，以 `cell_id` 一对一连接。加载时核查唯一 ID、完整 ID 集合和行列/投影坐标一致；没有应用全层共同有效掩膜。
- Aa_20yr 有效值范围 0–143.005，零值 133,220；Aa_2020 范围 0–20.029，零值 5,491。实际侵蚀的物理单位和精确时间构造仍按既有登记标记为未核实；本轮不推断。
- 主分析土类为 26、27、42。`SHI_candidate` 与八项指标均为随机森林栅格预测结果；这些像元数不是独立实测样本数。
- Phase F 配对输入共 {pair_profile['unique_pairs']:,} 对：M3 为土类内 V5（Elevation、LST）K-means k=6，M4 为直接最近邻；两者使用 0.35 环境卡尺、1,000 m 最大地理距离、组内侵蚀百分位秩差 0.50、无重复使用。各暴露/方法的配对数见 `tables/pair_input_profile.json`。

## 4. 方法与选择过程

### 4.1 配对差值与关联

高低成员按既有匹配结果确定，使用独立配对单元。全样本及预先定义的 `low_zero`、`both_positive`、`high_erosion_gap_q4` 子集分别报告；差值回归以 Δlog1p(Aa) 解释 Δ健康。平均差和斜率的 95% 区间使用 5 km 配对中点空间块 bootstrap；Spearman 区间使用 Fisher-z 变换并以空间块数为有效样本量近似。Wilcoxon、Spearman/Kendall 的原始 P 值不视为独立像元推断。

### 4.2 模型搜索与模型区分

Phase G 复用上游 Oracle 的 10,616 条实验记录：100 个 Aa_20yr 与 100 个 Aa_2020 候选匹配初筛，后续匹配/模型/环境子集搜索以及已存空间分块结果。源搜索覆盖 OLS、对数/二次/分段/自然样条/GAM 类回归、Huber、中位数回归、随机森林、Extra Trees 和 HistGradientBoosting 等已安装模型。所有 model leaderboard 均分别标记像元级预测模型与配对差值模型；二者 R² 不混排。

对每项结局分别保留最高表观 R²、负向模型中最高 R²，以及按标准化效应、负 Spearman、Δ健康负值比例、验证表现、匹配平衡与样本量形成的最强负向配对关联。对非线性模型无法从单一系数确定方向时，以存档的全样本偏依赖曲线在共同 Aa 范围内的变化补充方向；仍不可判定的结果保留为 `unknown`。

### 4.3 环境分组与空间验证

分组包括土类、Phase F V5 土类内 K-means k=6、土类内 Ward 层次聚类 k=6，以及高程、TWI、LST、NDWI、NDVI、降水的土类内分位组。层次聚类仅用环境变量形成分组，每个土类最多抽取 1,500 行拟合后按最近标准化中心赋给其他行。组别关联使用高侵蚀成员定义的环境组，并提供组样本数、环境统计量和所有候选结果。

组内空间候选以验证空间块选择，之后仅对选定组报告测试块。另沿用 Phase F 十个 5 km 空间种子的 M3/M4 测试结果。由于先前 Oracle 已反复使用全研究区数据，这些留出结果是内部空间诊断，不是未触碰的外部验证。

## 5. 模型排名与最佳候选

下表把全体像元模型的表观上限、匹配样本表观模型、负向匹配模型、验证集选定后的测试模型以及配对差值最强负向候选分开。模型表观 R²可能主要来自健康栅格本身的环境空间结构，不能解释为 Aa 的独立解释贡献。

{_markdown_table(primary_model_rows.head(45), [("outcome", "指标"), ("ranking_type", "排名"), ("candidate_id", "候选"), ("model_name", "模型"), ("r_squared", "R²"), ("direction", "方向"), ("n", "样本数"), ("group_method", "分组法"), ("group_label", "组别"), ("spearman_rho", "配对Spearman"), ("standardized_effect", "标准化效应"), ("validation_r2", "验证R²"), ("median_test_r2", "十种子测试R²中位数")])}

最佳像元模型的完整参数、筛选来源和样本量见 `tables/model_leaderboard.csv` 与 `tables/model_search_results.csv`。最强负向配对关联的复合排名会同时显示系数、Spearman、Δ健康负值比例、bootstrap 上界、验证 R²、SMD 和样本量；其 composite score 只是 Oracle 排序指标，不能被解释为正式效应量。

## 6. 环境组异质性

负向关联组的完整清单、关联区间与组内环境特征分别见 `tables/environment_group_associations.csv`、`tables/environment_group_statistics.csv` 和 `tables/environment_group_leaderboard.csv`。下表为 M4/Aa_20yr 下复合负向排名靠前的组：

{_markdown_table(top_groups.head(12), [("outcome", "指标"), ("group_method", "分组法"), ("group_label", "组别"), ("n_pairs", "配对数"), ("standardized_slope", "标准化斜率"), ("spearman_rho", "Spearman ρ"), ("negative_delta_y_fraction", "Δ健康<0比例"), ("negative_strength_score", "探索性综合分")])}

跨匹配方法与两种 Aa 时段的方向稳定组另列于 `most_stable_negative_direction_across_matching_methods_and_Aa_periods`。分组结果属于结果导向筛选，且单变量分位组可能残留其他环境差异；没有把某组标签解释为治理因果机制。

## 7. 空间验证

验证选定模型的空间测试结果如下，选择仅由验证集完成：

{_markdown_table(spatial_main, [("outcome", "指标"), ("model_name", "模型"), ("validation_r2", "验证R²"), ("r_squared", "空间测试R²"), ("rmse", "测试RMSE"), ("mae", "测试MAE"), ("direction", "方向"), ("n", "测试像元数"), ("environment_vector", "环境向量")])}

图 9 还展示了 Phase F 十个空间种子的配对 OLS 测试分布；本报告保留每一种子记录。预测面的模型测试只能检验空间区块间预测稳定性，不能替代独立采样验证。

## 8. 图表与复核路径

{figure_lines}

结果表、完整来源实验登记、运行配置、输入画像和产物校验清单位于 `{output_rel.as_posix()}/`。主要文件包括：

- `tables/experiment_registry.csv`
- `tables/matching_configurations.csv` 与 `tables/matching_balance_results.csv`
- `tables/matched_pair_statistics.csv` 与 `tables/association_results.csv`
- `tables/model_search_results.csv`、`tables/model_leaderboard.csv`、`tables/final_candidate_results.csv`
- `tables/environment_group_statistics.csv`、`tables/environment_group_associations.csv`
- `tables/spatial_validation_results.csv`

## 9. 研究判断

当前结果支持继续研究“环境条件相近时侵蚀差异与土壤健康预测差值的关系”，但不支持把关联概括成全指标一致的负效应。匹配与分组可以揭示异质性候选，但部分高拟合子集仅保留少量像元；跨空间测试表现低于表观上限，且搜索重复使用同一研究区，估计有选择乐观偏差。曲线图同时给出固定共同样本自然样条和先前 R² 选出的全样本模型曲线，两者样本范围与选择机制不同。

Phase G 不更改总纲，也不进入正式设计锁定。下一步应先由 ChatGPT 和用户审查每项指标的方向、单位/时间元信息、空间分块是否充分、预测面依赖及高 R² 子集的保留率，再决定哪些少数候选进入预先锁定的 FORMAL 设计。独立土壤实测点和其抽样/验证方案是正式验证的关键输入；不应因当前表观关联高而省略。

## 10. 执行状态与范围外事项

- Phase G Oracle 代码、CSV、九张图及报告已在本地生成，具体试验计数写入 `tables/experiment_summary.json` 与 `tables/experiment_registry.csv`。
- 本轮未运行原始 RUSLE、未重跑 PCNM、未修改总纲正文。
- 所有健康结局均是预测栅格值；结果不能写成独立实测点结论或因果效应。
- `Aa_20yr` 与 `Aa_2020` 单位及时间期定义缺失仍是报告限制；数据表的物理单位不予推定。
- ChatGPT 审查请求已生成，未向外发送。用户提交首个 Phase G 成果提交后，应把该提交完整 SHA 写入审查请求，再单独提交审查请求。
"""
    report_path.write_text(report, encoding="utf-8")


def _write_review_request(
    path: Path,
    *,
    config: dict[str, Any],
    report_path: Path,
    output_dir: Path,
    summary: dict[str, object],
    primary_associations: pd.DataFrame,
    leaderboard: pd.DataFrame,
    group_leaderboard: pd.DataFrame,
    source_sha: str,
    oracle_source_sha: str,
) -> None:
    primary = primary_associations.loc[
        primary_associations["method"].eq("M4") & primary_associations["exposure"].eq("Aa_20yr") & primary_associations["subset"].eq("all")
    ].copy().set_index("outcome").reindex(OUTCOMES).reset_index()
    ranking_summary = _display_candidate_table(leaderboard, ["highest_apparent_r2", "highest_negative_apparent_r2", "highest_full_sample_apparent_r2", "validation_selected_spatial_test", "strongest_negative_association"], "Aa_20yr")
    env_summary = group_leaderboard.loc[group_leaderboard["ranking_type"].isin(["strongest_negative_group", "most_stable_negative_direction_across_matching_methods_and_Aa_periods"])].copy() if not group_leaderboard.empty else pd.DataFrame()
    text = f"""# ChatGPT 阶段科研审查请求：Phase G

请首先读取本次Git提交对应版本的 `docs/RESEARCH_CHARTER.md`，完整理解EnvMatch-SEH的研究主题、总体研究目标、核心科学问题、主要研究内容及长期技术路线，然后结合这些内容审查本阶段的代码、实验、数据和研究结论。

## 1. 项目与版本

- 仓库：`https://github.com/shy-sgsg/EnvMatch-SEH`
- 审查提交 SHA：`<请在 Phase G 代码、配置、结果表、图表和研究报告的首个本地提交后，填写该提交的完整 SHA；随后单独提交本审查请求>`
- 本轮运行时的基线 HEAD：`{source_sha}`（不包含尚未提交的 Phase G 文件）
- 研究总纲：`docs/RESEARCH_CHARTER.md`，V1.1；对应本地基线提交：`{source_sha}`
- 阶段：Phase G `EXPLORATORY_ORACLE`；不是 FORMAL G5 完成，也不代表设计锁定
- 研究报告：`{report_path.relative_to(ROOT)}`
- 审查材料：`docs/reviews/2026-10-27/phase-g/`
- 输出根目录：`{output_dir.relative_to(ROOT)}/`

## 2. 总体研究定位与科学问题

本阶段对应 Q1–Q3：在环境匹配后，侵蚀与 SHI/八项健康指标是否有关联；不同环境分组下关系是否异质；物理、化学、生物指标的关联是否不同。PCNM 结果继续保留并暂不重跑。未修改总纲。

本阶段设定 `outcome_guided_selection=true`。本轮结果只用于发现候选及研究方法审查，不作为独立验证、因果结论或正式证据。

## 3. 本阶段工作

- 目标：复用 Phase F 匹配对与既有 Oracle 模型搜索，补齐全部九指标差值/秩相关、像元候选模型排名、环境组比较、空间块诊断和复核图表。
- 输入：`outputs/{config['input_run_id']}/databases/environment_erosion.parquet`、`health.parquet`；Phase F `outputs/exploratory_oracle/{config['phase_f_run_id']}/pairs/full_sample_pairs.parquet`；历史 Oracle `{config['oracle_source_run_id']}`。
- 暴露：预计算 `Aa_20yr` 为主、`Aa_2020` 为补充。未重算或验证 RUSLE 因子乘积。
- 配对：M3 土类内 V5 (Elevation/LST) K-means k=6；M4 直接最近邻；0.35 卡尺、1,000 m、组内秩差0.50、无重复使用。
- 环境组：SoilType、K-means、Ward 层次聚类以及 Elevation/TWI/LST/NDWI/NDVI/降水分位组。
- 历史模型筛选代码/记录：Oracle 源提交 `{oracle_source_sha}`；当前 Phase G 新增脚本/模块由待提交 SHA 覆盖。
- 新增主要代码/配置：`scripts/run_phase_g_analysis.py`、`experiments/oracle_exploration/phase_g.py`、`config/phase_g.yaml`、`tests/test_phase_g.py`。

## 4. 数据与方法质量

- 对齐数据库各 392,527 行；以 `cell_id` 一对一合并，坐标一致性通过；有效土类 26/27/42 共 {summary.get('primary_soil_rows', 'NA'):,} 行。
- Phase F 配对总数 {summary.get('phase_f_pair_rows', 'NA'):,}，具体按 M3/M4 与 Aa 暴露的数量在 `tables/pair_input_profile.json`。
- 历史 Oracle 候选记录数 {summary.get('source_oracle_trial_rows', 'NA'):,}；模型种类、失败项、样本数、匹配参数和选择规则可在 `tables/experiment_registry.csv` 与 `tables/model_search_results.csv` 查核。
- Aa 单位/精确时间构造与健康预测面的独立实测来源仍未核实；这些限制记录在研究报告和 `tables/data_profile.json`。
- 配对区间按 5 km pair-midpoint blocks bootstrap；相关系数区间使用 block-count-adjusted Fisher-z 近似。检验 P 值需结合空间依赖审查。
- 模型 R²分别标记像元预测、配对差值和空间测试；测试候选由验证分数选出。先前 Oracle 已复用全研究区，留出属于内部空间诊断。

### M4 / Aa_20yr 配对关联方向与效应

{_markdown_table(primary, [("outcome", "指标"), ("n_pairs", "配对数"), ("mean_delta_y", "平均Δ健康"), ("negative_delta_y_fraction", "Δ健康<0比例"), ("spearman_rho", "Spearman ρ"), ("kendall_tau", "Kendall τ"), ("standardized_slope", "标准化斜率"), ("bootstrap_slope_ci_low", "斜率区间下限"), ("bootstrap_slope_ci_high", "斜率区间上限")])}

## 5. 结果与研究价值

### 模型排名（Aa_20yr）

{_markdown_table(ranking_summary.head(60), [("outcome", "指标"), ("ranking_type", "排名"), ("candidate_id", "候选"), ("model_name", "模型"), ("r_squared", "R²"), ("direction", "方向"), ("n", "样本数"), ("group_method", "分组法"), ("group_label", "组别"), ("spearman_rho", "配对Spearman"), ("validation_r2", "验证R²"), ("median_test_r2", "测试R²中位数")])}

### 环境组

环境组所有候选及其环境特征见 `tables/environment_group_statistics.csv`；九指标关联见 `tables/environment_group_associations.csv`；按关联强度、负向效应、跨匹配/时段方向一致性分别排名的结果见 `tables/environment_group_leaderboard.csv`。空间验证入选与测试结果见 `tables/group_spatial_validation.csv`。

## 6. 空间验证与风险

空间验证结果表：`tables/spatial_validation_results.csv`。空间图 09 及源数据：`figures/09_spatial_validation.*` 与 `figures/data/09_spatial_validation.csv`。请特别审查先前结果选择已重复访问全域数据之后，是否仍可称为内部留出而非独立验证。

高表观 R²模型可能依赖健康预测栅格共享的环境协变量；匹配阈值未消除所有环境混杂；部分 Oracle 子组样本保留比例很低；模型结果的 direction 对非线性预测曲线定义而非单一系数。高拟合度不代表侵蚀独立贡献或因果效应。

## 7. 关键图表、实验表与代码

九张图均提供 PNG/SVG/PDF，并带源数据和图注：

{chr(10).join(f"- {item['name']}: `figures/{item['name']}.png`, `.svg`, `.pdf`; `figures/data/{item['name']}.csv`" for item in summary.get('figures', []))}

实验和结果表：

- `tables/experiment_registry.csv`
- `tables/matching_configurations.csv`
- `tables/matching_balance_results.csv`
- `tables/matched_pair_statistics.csv`
- `tables/association_results.csv`
- `tables/model_search_results.csv`
- `tables/model_leaderboard.csv`
- `tables/environment_group_statistics.csv`
- `tables/environment_group_associations.csv`
- `tables/spatial_validation_results.csv`
- `tables/final_candidate_results.csv`
- `tables/experiment_summary.json`、`tables/data_profile.json`、`artifact_manifest.json`
- 核心代码：`scripts/run_phase_g_analysis.py`、`experiments/oracle_exploration/phase_g.py`、`config/phase_g.yaml`

## 8. 对整体路线的影响与建议

建议先审查关联方向、置信区间方法、环境组定义、配对成员空间块分配、模型搜索选择偏差及高 R² 的环境解释。Phase G 不替代既有 PCNM 控制，也不建议仅依据表观最佳结果锁定正式方案。请提出下一阶段可预注册、可由独立实测数据验证的最小方案。

## 9. 总纲审查

`NO CHANGE`。本阶段未修改总纲正文。

审查材料生成后未发送给外部。请在固定到实际代码/结果提交 SHA 后审查上述路径。
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run_analysis(config_path: Path) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("mode") != "EXPLORATORY_ORACLE" or config.get("outcome_guided_selection") is not True:
        raise ValueError("Phase G config must explicitly enable EXPLORATORY_ORACLE outcome-guided selection")
    output_dir = (ROOT / config["output_dir"]).resolve()
    review_dir = (ROOT / config["review_dir"]).resolve()
    if not output_dir.is_relative_to(ROOT / "outputs" / "exploratory_oracle"):
        raise ValueError("Phase G Oracle outputs must remain under outputs/exploratory_oracle")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite an existing Phase G output directory: {output_dir}")
    if review_dir.exists() and any(review_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing review files: {review_dir}")
    output_dir.mkdir(parents=True)
    review_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "root": output_dir,
        "tables": output_dir / "tables",
        "diagnostics": output_dir / "diagnostics",
        "output_figures": output_dir / "figures",
        "reports": output_dir / "reports",
        "config": output_dir / "config",
        "review_dir": review_dir,
    }
    for name in ("tables", "diagnostics", "output_figures", "reports", "config"):
        paths[name].mkdir(parents=True, exist_ok=True)
    (paths["config"] / config_path.name).write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    (output_dir / "config.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")

    baseline_sha = _git("rev-parse", "HEAD")
    source_run_dir = ROOT / "outputs" / str(config["input_run_id"])
    phase_f_dir = ROOT / "outputs" / "exploratory_oracle" / str(config["phase_f_run_id"])
    oracle_dir = ROOT / "outputs" / "exploratory_oracle" / str(config["oracle_source_run_id"])
    _progress("loading aligned environment and health tables")
    bundle = load_oracle_data(source_run_dir)
    frame = bundle.frame
    environment_path = source_run_dir / "databases" / "environment_erosion.parquet"
    health_path = source_run_dir / "databases" / "health.parquet"
    pairs_path = phase_f_dir / "pairs" / "full_sample_pairs.parquet"
    pairs = pd.read_parquet(pairs_path)
    pair_profile = _validate_pairs(pairs, list(config["outcomes"]))
    _progress(f"loaded {len(frame):,} aligned cells and {len(pairs):,} Phase F pairs")
    soil_codes = [int(value) for value in config["environment_groups"]["primary_soils"]]
    numeric_profile: dict[str, object] = {
        "input_run_id": config["input_run_id"],
        "joined_rows": int(len(frame)),
        "unique_cell_ids": int(frame["cell_id"].nunique()),
        "primary_soil_rows": int(pd.to_numeric(frame["SoilType"], errors="coerce").isin(soil_codes).sum()),
        "soil_counts": pd.to_numeric(frame["SoilType"], errors="coerce").value_counts(dropna=False).sort_index().to_dict(),
        "outcome_nonmissing": {name: int(pd.to_numeric(frame[name], errors="coerce").notna().sum()) for name in config["outcomes"]},
        "exposure_ranges_and_zeros": {},
        "health_surface_type": "spatial prediction raster values; not independent field measurements",
        "erosion_source": "precomputed Aa_20yr and Aa_2020; no RUSLE factor-product recomputation or validation",
        "unresolved_metadata": ["Aa physical unit", "Aa_20yr exact time range and averaging construction", "health prediction model metadata for all eight indicators and SHI_candidate"],
    }
    for exposure in config["exposures"]:
        values = pd.to_numeric(frame[exposure], errors="coerce")
        finite = values[np.isfinite(values)]
        numeric_profile["exposure_ranges_and_zeros"][exposure] = {
            "nonmissing": int(len(finite)),
            "min": float(finite.min()) if len(finite) else None,
            "median": float(finite.median()) if len(finite) else None,
            "max": float(finite.max()) if len(finite) else None,
            "zero_count": int(finite.eq(0).sum()),
            "negative_count": int(finite.lt(0).sum()),
        }
    input_manifest = {
        "environment_erosion": {"path": str(environment_path.relative_to(ROOT)), "bytes": environment_path.stat().st_size, "sha256": _sha256(environment_path)},
        "health": {"path": str(health_path.relative_to(ROOT)), "bytes": health_path.stat().st_size, "sha256": _sha256(health_path)},
        "phase_f_pairs": {"path": str(pairs_path.relative_to(ROOT)), "bytes": pairs_path.stat().st_size, "sha256": _sha256(pairs_path)},
        "prior_oracle_registry": {"path": str((oracle_dir / "tables" / "experiment_results.parquet").relative_to(ROOT)), "bytes": (oracle_dir / "tables" / "experiment_results.parquet").stat().st_size, "sha256": _sha256(oracle_dir / "tables" / "experiment_results.parquet")},
    }
    pair_associations = summarize_pair_associations(
        pairs,
        list(config["outcomes"]),
        bootstrap_replicates=int(config["pair_analysis"]["bootstrap_replicates"]),
        seed=int(config["seed"]),
    )
    _progress(f"summarized {len(pair_associations):,} matched-pair association rows")
    pair_statistics = pair_associations.copy()
    pair_statistics["statistic_type"] = "matched_pair_difference_summary"

    # Reuse Phase F matching records; preserve every candidate and its balance diagnostics.
    local_config_path = phase_f_dir / "tables" / "local_matching_grid.csv"
    local_balance_path = phase_f_dir / "tables" / "matching_balance_diagnostics.csv"
    local_configs = pd.read_csv(local_config_path, encoding="utf-8-sig")
    local_configs["source"] = "Phase F outcome-blind local matching grid"
    local_configs["outcome_guided_selection"] = False
    phase_f_config = yaml.safe_load((phase_f_dir / "config.yaml").read_text(encoding="utf-8"))
    reference_rows: list[dict[str, object]] = []
    for exposure in config["exposures"]:
        for method in config["matched_methods"]:
            reference = phase_f_config["matching"]["reference"] if method == "M4" else phase_f_config["matching"]["kmeans_comparison"]
            reference_rows.append({
                "candidate_id": f"phase_f_reference_{method}_{exposure}",
                "exposure": exposure,
                "method": method,
                "vector": reference["vector"],
                "cluster_method": "kmeans" if method == "M3" else "none",
                "cluster_count": int(phase_f_config["primary_soil_codes"][0] * 0 + (6 if method == "M3" else 0)),
                "environment_caliper": float(reference["environment_caliper"]),
                "geographic_radius_m": float(reference["geographic_radius_m"]),
                "erosion_rank_gap": float(reference["erosion_rank_gap"]),
                "allow_replacement": False,
                "source": "Phase F full-sample reference configuration",
                "outcome_guided_selection": False,
            })
    source_registry = pd.read_parquet(oracle_dir / "tables" / "experiment_results.parquet")
    match_mask = source_registry["experiment_type"].astype(str).str.contains("matching", case=False, na=False)
    old_match_configs = source_registry.loc[match_mask].drop_duplicates(["candidate_id", "exposure"], keep="first").copy()
    old_match_configs["source"] = "prior Oracle outcome-guided matching search"
    old_match_configs["outcome_guided_selection"] = True
    matching_configs = pd.concat([local_configs, pd.DataFrame(reference_rows), old_match_configs], ignore_index=True, sort=False)
    matching_balance = pd.read_csv(local_balance_path, encoding="utf-8-sig")
    matching_balance["source"] = "Phase F local matching quality diagnostics"
    oracle_stage2_scores_path = oracle_dir / "matching" / "stage2" / "scores.parquet"
    if oracle_stage2_scores_path.is_file():
        oracle_balance = pd.read_parquet(oracle_stage2_scores_path)
        oracle_balance["source"] = "prior Oracle stage-2 matching quality diagnostics"
        matching_balance = pd.concat([matching_balance, oracle_balance], ignore_index=True, sort=False)

    _progress("assigning soil, cluster, and environmental-gradient groups")
    pairs_grouped, group_meta = _group_pair_data(frame, pairs, config)
    _progress(f"summarizing associations across {len(group_meta['group_columns'])} group definitions")
    group_stats, group_associations, group_leaderboard = _group_tables(
        pairs_grouped,
        group_meta["group_columns"],
        list(config["outcomes"]),
        bootstrap_replicates=max(80, int(config["pair_analysis"]["bootstrap_replicates"]) // 2),
        seed=int(config["seed"]),
    )
    _progress(f"summarized {len(group_associations):,} environment-group associations")
    group_spatial = _group_spatial_selection(
        pairs_grouped,
        group_meta["group_columns"],
        list(config["outcomes"]),
        [int(value) for value in config["spatial_validation"]["seeds"]],
        float(config["spatial_validation"]["block_size_m"]),
        int(config["environment_groups"]["min_spatial_test_pairs"]),
    )
    _progress(f"completed {len(group_spatial):,} validation-selected group spatial checks")
    group_spatial_stability = _group_stability_summary(group_spatial)
    if not group_spatial_stability.empty:
        group_spatial_stability["ranking_type"] = "spatial_repeat_stability_diagnostic_not_used_for_selection"
        group_spatial_stability["outcome_guided_selection"] = True
    group_spatial_leaders = group_spatial_stability.copy()
    group_leaderboard = pd.concat([group_leaderboard, group_spatial_leaders], ignore_index=True, sort=False)
    pair_balance = _pair_balance_summary(pairs)

    model_results, leaderboard, final_candidates, selected_curves, pair_spatial, phase_f_spatial = _prepare_model_results(
        oracle_dir,
        config,
        pairs,
        pair_associations,
        pair_balance,
    )
    _progress(f"reconciled {len(model_results):,} model records into {len(leaderboard):,} leaderboard rows")
    spatial_selected = pd.read_parquet(oracle_dir / "models" / "spatial_test_selected_models.parquet")
    response_curves, response_scatter, curve_sample_profile = _build_response_curve_data(frame, selected_curves, config)
    _progress(f"prepared {len(response_curves):,} response-curve points from both curve methods")
    if not group_leaderboard.empty:
        group_best = group_leaderboard.loc[
            group_leaderboard["ranking_type"].eq("strongest_negative_group")
            & group_leaderboard["exposure"].eq("Aa_20yr")
            & group_leaderboard["method"].eq("M4")
        ].sort_values("negative_strength_score", ascending=False).groupby("outcome", sort=True).head(1)
        stable = group_leaderboard.loc[group_leaderboard["ranking_type"].eq("most_stable_negative_direction_across_matching_methods_and_Aa_periods")]
        final_candidates = pd.concat([
            final_candidates,
            group_best.assign(analysis_unit="matched_pair_environment_group", final_candidate_type="strongest_negative_environment_group"),
            stable.assign(analysis_unit="matched_pair_environment_group", final_candidate_type="most_stable_group_direction"),
        ], ignore_index=True, sort=False)

    # Preserve pixel, paired, and group spatial results as explicitly labeled rows.
    spatial_selected = spatial_selected.assign(result_type="validation_selected_pixel_model_test")
    phase_f_spatial = phase_f_spatial.assign(result_type="Phase_F_repeated_pair_validation")
    group_spatial = group_spatial.assign(result_type="validation_selected_group_pair_spatial_test")
    spatial_validation = pd.concat([spatial_selected, phase_f_spatial, group_spatial], ignore_index=True, sort=False)

    # Source-level outcome-guided records are preserved; append all Phase G summaries and selection decisions.
    new_registry = _new_registry_rows(
        pair_associations,
        group_associations,
        leaderboard,
        group_spatial,
        run_id=str(config["run_id"]),
        baseline_sha=baseline_sha,
        input_run_id=str(config["input_run_id"]),
        seed=int(config["seed"]),
    )
    experiment_registry = pd.concat([source_registry, new_registry], ignore_index=True, sort=False)

    tables: dict[str, pd.DataFrame] = {
        "experiment_registry": experiment_registry,
        "matching_configurations": matching_configs,
        "matching_balance_results": matching_balance,
        "matched_pair_statistics": pair_statistics,
        "association_results": pair_associations,
        "model_search_results": model_results,
        "model_leaderboard": leaderboard,
        "environment_group_statistics": group_stats,
        "environment_group_associations": group_associations,
        "environment_group_leaderboard": group_leaderboard,
        "group_spatial_validation": group_spatial,
        "group_spatial_stability": group_spatial_stability,
        "spatial_validation_results": spatial_validation,
        "final_candidate_results": final_candidates,
        "pair_balance_summary": pair_balance,
        "response_curve_data": response_curves,
        "response_curve_sample_points": response_scatter,
        "pixel_model_r2_summary": leaderboard.loc[leaderboard["analysis_unit"].astype(str).str.startswith("pixel_level")].copy(),
        "paired_spatial_validation_summary": pair_spatial,
    }
    for name, table in tables.items():
        _write_csv(paths["tables"] / f"{name}.csv", table)
    _progress("wrote analysis tables; generating review figures")
    _write_json(paths["tables"] / "pair_input_profile.json", pair_profile)
    _write_json(paths["tables"] / "data_profile.json", numeric_profile)
    _write_json(paths["tables"] / "environment_group_definitions.json", group_meta)
    _write_csv(paths["tables"] / "response_curve_sample_profile.csv", curve_sample_profile)

    source_sha_values = source_registry["git_sha"].dropna().astype(str)
    oracle_source_sha = source_sha_values.mode().iloc[0] if not source_sha_values.empty else "unavailable"
    software = _packages()
    figures = _generate_figures(
        paths=paths,
        config=config,
        pairs_grouped=pairs_grouped,
        association_results=pair_associations,
        matching_configurations=local_configs,
        matching_balance=matching_balance,
        leaderboard=leaderboard,
        group_associations=group_associations,
        group_leaderboard=group_leaderboard,
        spatial_selected=spatial_selected,
        phase_f_spatial=phase_f_spatial,
        response_curves=response_curves,
        response_scatter=response_scatter,
    )
    _progress(f"generated {len(figures)} figures in the configured formats")
    # The report package carries the same CSV sources as the ignored run tree so a reviewer can inspect a fixed commit.
    review_tables = review_dir / "tables"
    review_tables.mkdir(parents=True, exist_ok=True)
    for path in paths["tables"].glob("*"):
        if path.is_file():
            (review_tables / path.name).write_bytes(path.read_bytes())
    input_profile = {**numeric_profile, "input_files": input_manifest, "coordinate_checks": "load_oracle_data required one-to-one cell_id sets and matched row/column/x/y coordinates"}
    _write_json(paths["diagnostics"] / "input_profile.json", input_profile)
    provenance = {
        "phase": "G",
        "mode": "EXPLORATORY_ORACLE",
        "outcome_guided_selection": True,
        "run_id": config["run_id"],
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "baseline_commit_sha": baseline_sha,
        "working_tree_dirty_at_run": bool(_git("status", "--porcelain")),
        "research_charter_version": "V1.1",
        "research_charter_commit": baseline_sha,
        "source_oracle_run_id": config["oracle_source_run_id"],
        "source_oracle_code_commit": oracle_source_sha,
        "phase_f_run_id": config["phase_f_run_id"],
        "input_run_id": config["input_run_id"],
        "input_files": input_manifest,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": software,
        "config_path": str(config_path.relative_to(ROOT)),
        "config_sha256": _sha256(config_path),
        "selection_policy": {
            "highest_apparent": "rank apparent matched-pixel and full-sample rows separately; retain outcome-guided subset rows with sample retention",
            "highest_negative_r2": "only rows whose recorded/inferred direction is negative",
            "strongest_negative_association": "pair-level composite over standardized effect, Spearman, negative delta fraction, block-bootstrap effect interval, validation score, match SMD, and n; test scores excluded from ranking",
            "spatial_test": "reuse prior validation-selected candidates; report test metric without reranking on test",
            "group_spatial": "select group on validation blocks for each seed, then score selected group on test blocks",
        },
        "no_rusle_factor_product_validation": True,
        "no_pcnm_rerun": True,
        "health_data_are_predictions": True,
    }
    _write_json(paths["diagnostics"] / "provenance.json", provenance)
    source_trial_counts = source_registry["experiment_type"].value_counts(dropna=False).to_dict()
    summary = {
        "run_id": config["run_id"],
        "mode": config["mode"],
        "outcome_guided_selection": True,
        "input_rows": int(len(frame)),
        "primary_soil_rows": numeric_profile["primary_soil_rows"],
        "phase_f_pair_rows": int(len(pairs)),
        "phase_f_pair_set_counts": pair_profile["pair_sets"],
        "phase_g_pair_association_rows": int(len(pair_associations)),
        "environment_group_association_rows": int(len(group_associations)),
        "group_spatial_selection_rows": int(len(group_spatial)),
        "source_oracle_trial_rows": int(len(source_registry)),
        "source_oracle_experiment_types": {str(key): int(value) for key, value in source_trial_counts.items()},
        "model_search_candidate_rows": int(len(model_results)),
        "model_leaderboard_rows": int(len(leaderboard)),
        "spatial_test_candidate_rows": int(len(spatial_selected)),
        "phase_f_pair_spatial_p1_rows": int(len(phase_f_spatial)),
        "paired_spatial_validation_summary_rows": int(len(pair_spatial)),
        "spatial_validation_result_rows": int(len(spatial_validation)),
        "figure_count": len(figures),
        "figures": figures,
        "baseline_commit_sha": baseline_sha,
        "oracle_source_commit_sha": oracle_source_sha,
        "report_path": str((review_dir / "PHASE_G_RESEARCH_REPORT.md").relative_to(ROOT)),
        "review_request_path": str((review_dir / "CHATGPT_REVIEW_REQUEST.md").relative_to(ROOT)),
        "output_path": str(output_dir.relative_to(ROOT)),
        "selection_policy": provenance["selection_policy"],
        "limitations": [
            "Aa physical units and exact Aa_20yr averaging period remain unverified",
            "health layers are prediction surfaces rather than independent measurements",
            "prior Oracle exploration used the study area before spatial holdouts; tests are internal diagnostics",
            "model-specific Spearman values were not stored by the prior Oracle code; observed matched-pair Spearman is reported separately",
        ],
    }
    _write_json(paths["tables"] / "experiment_summary.json", summary)
    report_path = review_dir / "PHASE_G_RESEARCH_REPORT.md"
    _write_report(
        report_path=report_path,
        config=config,
        source_profile=numeric_profile,
        pair_profile=pair_profile,
        pair_associations=pair_associations,
        leaderboard=leaderboard,
        group_leaderboard=group_leaderboard,
        group_stability=group_spatial_stability,
        spatial_selected=spatial_selected,
        figure_index=figures,
        source_sha=baseline_sha,
        oracle_source_sha=oracle_source_sha,
        software=software,
        run_id=str(config["run_id"]),
        output_dir=output_dir,
    )
    review_request_path = review_dir / "CHATGPT_REVIEW_REQUEST.md"
    _write_review_request(
        review_request_path,
        config=config,
        report_path=report_path,
        output_dir=output_dir,
        summary=summary,
        primary_associations=pair_associations,
        leaderboard=leaderboard,
        group_leaderboard=group_leaderboard,
        source_sha=baseline_sha,
        oracle_source_sha=oracle_source_sha,
    )
    (paths["reports"] / "PHASE_G_RESEARCH_REPORT.md").write_bytes(report_path.read_bytes())
    _write_json(review_dir / "artifact_manifest.json", {
        "run_id": config["run_id"],
        "baseline_commit_sha": baseline_sha,
        "mode": "EXPLORATORY_ORACLE",
        "files": [
            {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size}
            for path in sorted([*output_dir.rglob("*"), *review_dir.rglob("*")])
            if path.is_file() and path.name != "artifact_manifest.json"
        ],
    })
    _write_json(output_dir / "artifact_manifest.json", {
        "run_id": config["run_id"],
        "baseline_commit_sha": baseline_sha,
        "input_manifest": input_manifest,
        "figure_count": len(figures),
        "table_count": len(list(paths["tables"].glob("*.csv"))),
        "review_dir": str(review_dir.relative_to(ROOT)),
    })
    print(f"Phase G analysis complete: {output_dir}")
    print(f"Review report: {report_path}")
    print(f"ChatGPT review request: {review_request_path}")
    print(f"Pair association rows: {len(pair_associations):,}; model results: {len(model_results):,}; group rows: {len(group_associations):,}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Phase G exploratory matched-pair and environment analysis")
    parser.add_argument("--config", type=Path, default=Path("config/phase_g.yaml"))
    args = parser.parse_args()
    path = args.config if args.config.is_absolute() else ROOT / args.config
    run_analysis(path)


if __name__ == "__main__":
    main()




