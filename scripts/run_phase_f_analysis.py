#!/usr/bin/env python3
"""Run the reproducible, internal Phase F matched-pair analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import warnings
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ["MPLCONFIGDIR"] = "/tmp/phase-f-matplotlib-cache"
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import statsmodels.api as sm
import yaml
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from soilmatch.matching import match_cells


CONFIG_PATH = ROOT / "config" / "phase_f.yaml"
ENVIRONMENT_COLUMNS = [
    "Elevation", "Cur", "TWI", "LST", "NDWI", "NDBSI", "EVI", "NDVI", "Precipitation"
]
PRIMARY_OUTCOMES = ["SHI_candidate", "OM", "SW"]
EXPOSURE_COLUMNS = ["Aa_20yr", "Aa_2020"]
MODEL_NAMES = {
    "M0": "Direct OLS",
    "M1": "Environment-adjusted OLS",
    "M2": "Environment-group slopes",
    "M3": "K-means matched pairs",
    "M4": "Direct nearest-neighbor pairs",
}
SOIL_TYPE_LABELS = {26: "冲积土", 27: "潮土", 42: "脱潮土"}


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (Path,)):
        return str(value)
    if pd.isna(value):
        return None
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default) + "\n")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def write_frame_cache(path: Path, frame: pd.DataFrame) -> None:
    write_json(path, {"columns": list(frame.columns), "records": frame.to_dict(orient="records")})


def read_frame_cache(path: Path) -> pd.DataFrame:
    cached = json.loads(path.read_text())
    return pd.DataFrame.from_records(cached["records"], columns=cached["columns"])


def same_run_signature(
    previous: dict[str, Any] | None,
    current: dict[str, Any],
    previous_config: dict[str, Any] | None = None,
    current_config: dict[str, Any] | None = None,
) -> bool:
    if not previous:
        return False
    fields = ["baseline_commit_sha", "input_run_id", "input_files"]
    if not all(previous.get(field) == current.get(field) for field in fields):
        return False
    if previous.get("config_sha256") == current.get("config_sha256"):
        return True
    if previous_config is None or current_config is None:
        return False
    previous_analysis_config = dict(previous_config)
    current_analysis_config = dict(current_config)
    previous_analysis_config.pop("output_root", None)
    current_analysis_config.pop("output_root", None)
    return previous_analysis_config == current_analysis_config


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(*args: str, fallback: str = "unavailable") -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else fallback


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text())


def load_input(input_run_id: str) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Path]]:
    database_dir = ROOT / "outputs" / input_run_id / "databases"
    environment_path = database_dir / "environment_erosion.parquet"
    health_path = database_dir / "health.parquet"
    environment = pd.read_parquet(environment_path)
    health = pd.read_parquet(health_path)
    if environment["cell_id"].isna().any() or environment["cell_id"].duplicated().any():
        raise ValueError("Environment-erosion database cell_id must be non-null and unique")
    if health["cell_id"].isna().any() or health["cell_id"].duplicated().any():
        raise ValueError("Health database cell_id must be non-null and unique")
    frame = environment.merge(
        health, on="cell_id", how="inner", validate="one_to_one", suffixes=("_env", "_health")
    )
    # Both aligned inputs store the same grid coordinates; keep the environment-side
    # coordinates under the unambiguous names expected by matching and spatial folds.
    frame = frame.rename(columns={"x_env": "x", "y_env": "y"})
    frame = frame.drop(columns=[name for name in ["x_health", "y_health"] if name in frame])
    if "valid_cell_env" in frame:
        frame = frame.rename(columns={"valid_cell_env": "environment_valid_cell"})
    if "valid_cell_health" in frame:
        frame = frame.rename(columns={"valid_cell_health": "health_valid_cell"})

    profile: dict[str, Any] = {
        "input_run_id": input_run_id,
        "environment_rows": int(len(environment)),
        "health_rows": int(len(health)),
        "joined_rows": int(len(frame)),
        "environment_columns": list(environment.columns),
        "health_columns": list(health.columns),
        "unique_environment_cells": int(environment["cell_id"].nunique()),
        "unique_health_cells": int(health["cell_id"].nunique()),
        "primary_soil_rows": int(frame["SoilType"].isin([26, 27, 42]).sum()),
        "outcome_nonmissing": {
            name: int(pd.to_numeric(frame[name], errors="coerce").notna().sum())
            for name in ["SHI_candidate", "AN", "OM", "CEC", "pH", "SW", "BD", "MBC", "RESP"]
        },
        "exposure_nonmissing": {
            name: int(pd.to_numeric(frame[name], errors="coerce").notna().sum())
            for name in EXPOSURE_COLUMNS
        },
        "exposure_ranges_and_zeros": {},
    }
    for exposure in EXPOSURE_COLUMNS:
        values = pd.to_numeric(frame[exposure], errors="coerce")
        finite = values[np.isfinite(values)]
        profile["exposure_ranges_and_zeros"][exposure] = {
            "min": float(finite.min()) if len(finite) else None,
            "median": float(finite.median()) if len(finite) else None,
            "max": float(finite.max()) if len(finite) else None,
            "zero_count": int(finite.eq(0).sum()),
            "negative_count": int(finite.lt(0).sum()),
        }
    paths = {"environment": environment_path, "health": health_path}
    return frame, profile, paths


def provenance(config: dict[str, Any], paths: dict[str, Path], profile: dict[str, Any]) -> dict[str, Any]:
    packages = {}
    for name in ["numpy", "pandas", "scipy", "scikit-learn", "statsmodels", "pyarrow", "PyYAML", "matplotlib"]:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    return {
        "phase": "F",
        "mode": "EXPLORATORY_ORACLE",
        "outcome_guided_selection": config.get("mode") == "EXPLORATORY_ORACLE",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "baseline_commit_sha": git_value("rev-parse", "HEAD"),
        "working_tree_dirty_at_run": bool(git_value("status", "--porcelain", fallback="").strip()),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "config_path": str(CONFIG_PATH.relative_to(ROOT)),
        "config_sha256": sha256_file(CONFIG_PATH),
        "input_run_id": config["input_run_id"],
        "input_files": {
            key: {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for key, path in paths.items()
        },
        "source_profile": profile,
        "spatial_seed_list": config["spatial_validation"]["seeds"],
        "primary_soils": config["primary_soil_codes"],
        "erosion_values": "Precomputed Aa_20yr and Aa_2020; no RUSLE recomputation or factor validation.",
    }


def complete_mask(frame: pd.DataFrame, columns: list[str], primary_soils: list[int]) -> pd.Series:
    numeric = frame[columns].apply(pd.to_numeric, errors="coerce")
    return frame["SoilType"].isin(primary_soils) & np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)


def fit_environment_model(
    train: pd.DataFrame, variables: list[str], *, clusters: int, seed: int, primary_soils: list[int]
) -> dict[str, Any]:
    """Fit train-only global distance scaling and train-only per-soil K-means."""
    eligible = train.loc[complete_mask(train, variables, primary_soils)]
    if eligible.empty:
        raise ValueError("No complete training rows for the environment transform")
    scaler = StandardScaler().fit(eligible[variables].to_numpy(dtype=float))
    cluster_models: dict[int, tuple[StandardScaler, KMeans]] = {}
    for soil in primary_soils:
        subset = eligible.loc[eligible["SoilType"].eq(soil), variables]
        if clusters <= 0:
            continue
        if len(subset) < max(clusters, 2):
            continue
        local_scaler = StandardScaler().fit(subset.to_numpy(dtype=float))
        matrix = local_scaler.transform(subset.to_numpy(dtype=float))
        model = KMeans(n_clusters=clusters, n_init=10, random_state=seed + soil).fit(matrix)
        cluster_models[int(soil)] = (local_scaler, model)
    return {"variables": variables, "scaler": scaler, "clusters": clusters, "cluster_models": cluster_models}


def apply_environment_model(
    model: dict[str, Any], frame: pd.DataFrame, primary_soils: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    variables = model["variables"]
    z = np.full((len(frame), len(variables)), np.nan, dtype=float)
    groups = np.full(len(frame), -1, dtype=np.int32)
    mask = complete_mask(frame, variables, primary_soils)
    if mask.any():
        values = frame.loc[mask, variables].to_numpy(dtype=float)
        z[mask.to_numpy()] = model["scaler"].transform(values)
        row_indices = np.flatnonzero(mask.to_numpy())
        soils = pd.to_numeric(frame.loc[mask, "SoilType"], errors="coerce").to_numpy(dtype=int)
        for soil in primary_soils:
            local = soils == soil
            if not local.any():
                continue
            bundle = model["cluster_models"].get(int(soil))
            if bundle is None:
                groups[row_indices[local]] = int(soil) * 100
                continue
            local_scaler, cluster_model = bundle
            labels = cluster_model.predict(local_scaler.transform(values[local]))
            groups[row_indices[local]] = int(soil) * 100 + labels.astype(np.int32)
    return z, groups


def build_pair_table(
    frame: pd.DataFrame,
    *,
    exposure: str,
    variables: list[str],
    env_model: dict[str, Any],
    method: str,
    seed: int,
    role: str,
    config: dict[str, Any],
    primary_soils: list[int],
    block_size_m: float,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Match without accessing outcomes, then join pair-level measurements."""
    z, groups = apply_environment_model(env_model, frame, primary_soils)
    exposure_values = pd.to_numeric(frame[exposure], errors="coerce").to_numpy(dtype=float)
    x = pd.to_numeric(frame["x"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(frame["y"], errors="coerce").to_numpy(dtype=float)
    soil = pd.to_numeric(frame["SoilType"], errors="coerce").to_numpy(dtype=float)
    eligible = np.isfinite(z).all(axis=1) & np.isfinite(exposure_values) & np.isfinite(x) & np.isfinite(y)
    eligible &= np.isin(soil, primary_soils) & (exposure_values >= 0) & (groups >= 0)
    source = frame.loc[eligible].copy().reset_index(drop=True)
    z = z[eligible]
    exposure_values = exposure_values[eligible]
    groups = groups[eligible]
    soil = soil[eligible].astype(int)
    descriptive_groups = groups.copy()
    ranks_frame = pd.DataFrame({"environment_group": groups, "soil_type": soil, "exposure": exposure_values})
    ranks = ranks_frame.groupby(["environment_group", "soil_type"], sort=False)["exposure"].rank(pct=True).to_numpy()
    source["environment_group"] = groups
    source["descriptive_environment_group"] = descriptive_groups
    source["exposure_rank"] = ranks
    if method == "M4":
        groups = np.zeros(len(source), dtype=np.int32)
        source["environment_group"] = groups
        ranks_frame = pd.DataFrame({"environment_group": groups, "soil_type": soil, "exposure": exposure_values})
        ranks = ranks_frame.groupby(["environment_group", "soil_type"], sort=False)["exposure"].rank(pct=True).to_numpy()
        source["exposure_rank"] = ranks
    matching_frame = pd.DataFrame(
        {
            "cell_id": source["cell_id"].to_numpy(),
            "x": pd.to_numeric(source["x"], errors="coerce").to_numpy(dtype=float),
            "y": pd.to_numeric(source["y"], errors="coerce").to_numpy(dtype=float),
            "SoilType": soil,
            "environment_group": groups,
            "exposure_rank": ranks,
        }
    )
    distance_columns = []
    for index in range(z.shape[1]):
        column = f"z_env_{index:02d}"
        matching_frame[column] = z[:, index]
        distance_columns.append(column)

    selected = config["matching"]["reference"] if method == "M4" else config["matching"]["kmeans_comparison"]
    if method not in {"M3", "M4"}:
        raise ValueError(f"Unsupported matched-pair method: {method}")
    raw_pairs, matching_diagnostics = match_cells(
        matching_frame,
        environment_columns=distance_columns,
        group_col="environment_group",
        erosion_col="exposure_rank",
        soil_type_col="SoilType",
        x_col="x",
        y_col="y",
        distance_limit=float(selected["geographic_radius_m"]),
        environment_distance_limit=float(selected["environment_caliper"]),
        erosion_gap=float(selected["erosion_rank_gap"]),
        allow_reuse=False,
    )
    if raw_pairs.empty:
        diagnostics = {
            **matching_diagnostics,
            "method": method,
            "exposure": exposure,
            "seed": seed,
            "role": role,
            "status": "no_pairs",
            "area_coverage": 0.0,
            "max_abs_matched_smd": np.nan,
            "failure_counts": matching_diagnostics.get("failure_counts", {}),
        }
        return pd.DataFrame(), diagnostics, pd.DataFrame()

    by_id = source.set_index("cell_id", drop=False)
    high = by_id.loc[raw_pairs["high_erosion_cell_id"].tolist()].reset_index(drop=True)
    low = by_id.loc[raw_pairs["low_erosion_cell_id"].tolist()].reset_index(drop=True)
    pairs = raw_pairs.copy()
    pairs["pair_id"] = [f"seed{seed}_{method}_{exposure}_{role}_{i:06d}" for i in range(1, len(pairs) + 1)]
    pairs["method"] = method
    pairs["exposure"] = exposure
    pairs["seed"] = seed
    pairs["spatial_role"] = role
    pairs["exposure_rank_high"] = raw_pairs["erosion_high"].to_numpy(dtype=float)
    pairs["exposure_rank_low"] = raw_pairs["erosion_low"].to_numpy(dtype=float)
    pairs["exposure_percentile_gap"] = raw_pairs["erosion_difference"].to_numpy(dtype=float)
    pairs["erosion_high"] = pd.to_numeric(high[exposure], errors="coerce").to_numpy(dtype=float)
    pairs["erosion_low"] = pd.to_numeric(low[exposure], errors="coerce").to_numpy(dtype=float)
    pairs["erosion_difference"] = pairs["erosion_high"] - pairs["erosion_low"]
    pairs["delta_log1p_erosion"] = np.log1p(pairs["erosion_high"]) - np.log1p(pairs["erosion_low"])
    pairs["x_high"] = pd.to_numeric(high["x"], errors="coerce").to_numpy(dtype=float)
    pairs["y_high"] = pd.to_numeric(high["y"], errors="coerce").to_numpy(dtype=float)
    pairs["x_low"] = pd.to_numeric(low["x"], errors="coerce").to_numpy(dtype=float)
    pairs["y_low"] = pd.to_numeric(low["y"], errors="coerce").to_numpy(dtype=float)
    pairs["x_mid"] = (pairs["x_high"] + pairs["x_low"]) / 2
    pairs["y_mid"] = (pairs["y_high"] + pairs["y_low"]) / 2
    pairs["spatial_block_id"] = (
        np.floor(pairs["x_mid"] / block_size_m).astype(int).astype(str)
        + "_"
        + np.floor(pairs["y_mid"] / block_size_m).astype(int).astype(str)
    )
    pairs["soil_type"] = pd.to_numeric(high["SoilType"], errors="coerce").to_numpy(dtype=float)
    pairs["descriptive_environment_group"] = pd.to_numeric(
        high["descriptive_environment_group"], errors="coerce"
    ).to_numpy(dtype=float)
    for variable in ENVIRONMENT_COLUMNS:
        if variable not in high or variable not in low:
            continue
        pairs[f"{variable}_high"] = pd.to_numeric(high[variable], errors="coerce").to_numpy(dtype=float)
        pairs[f"{variable}_low"] = pd.to_numeric(low[variable], errors="coerce").to_numpy(dtype=float)
        pairs[f"delta_{variable}"] = pairs[f"{variable}_high"] - pairs[f"{variable}_low"]
        pairs[f"mean_{variable}"] = (pairs[f"{variable}_high"] + pairs[f"{variable}_low"]) / 2
    for outcome in config["outcomes"]:
        if outcome not in high or outcome not in low:
            continue
        pairs[f"high_{outcome}"] = pd.to_numeric(high[outcome], errors="coerce").to_numpy(dtype=float)
        pairs[f"low_{outcome}"] = pd.to_numeric(low[outcome], errors="coerce").to_numpy(dtype=float)
        pairs[f"delta_{outcome}"] = pairs[f"high_{outcome}"] - pairs[f"low_{outcome}"]
    if (pairs["erosion_difference"] < -1e-12).any():
        raise ValueError("High-exposure member has lower actual Aa than its low member")
    diagnostics, balance_rows = match_quality_diagnostics(
        source=source,
        pairs=pairs,
        variables=variables,
        config=selected,
        method=method,
        exposure=exposure,
        seed=seed,
        role=role,
        matching_diagnostics=matching_diagnostics,
    )
    return pairs, diagnostics, balance_rows


def standardized_mean_difference(high: np.ndarray, low: np.ndarray) -> float:
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    high = high[np.isfinite(high)]
    low = low[np.isfinite(low)]
    if len(high) < 2 or len(low) < 2:
        return np.nan
    pooled = math.sqrt((np.var(high, ddof=1) + np.var(low, ddof=1)) / 2)
    difference = float(high.mean() - low.mean())
    return difference / pooled if pooled > 0 else (0.0 if difference == 0 else np.nan)


def match_quality_diagnostics(
    *,
    source: pd.DataFrame,
    pairs: pd.DataFrame,
    variables: list[str],
    config: dict[str, Any],
    method: str,
    exposure: str,
    seed: int,
    role: str,
    matching_diagnostics: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    pair_n = int(len(pairs))
    matched_cells = int(pd.unique(pairs[["high_erosion_cell_id", "low_erosion_cell_id"]].to_numpy().ravel()).size)
    eligible_n = int(matching_diagnostics.get("eligible_cells", len(source)))
    abs_log_gap = pd.to_numeric(pairs["delta_log1p_erosion"], errors="coerce").abs()
    abs_aa_gap = pd.to_numeric(pairs["erosion_difference"], errors="coerce").abs()
    diagnostics = {
        **matching_diagnostics,
        "method": method,
        "exposure": exposure,
        "seed": seed,
        "role": role,
        "status": "ok",
        "eligible_cells": eligible_n,
        "n_pairs": pair_n,
        "n_matched_cells": matched_cells,
        "area_coverage": matched_cells / eligible_n if eligible_n else np.nan,
        "environment_distance_mean": float(pairs["environmental_distance"].mean()),
        "environment_distance_median": float(pairs["environmental_distance"].median()),
        "environment_distance_p95": float(pairs["environmental_distance"].quantile(0.95)),
        "geographic_distance_mean_m": float(pairs["geographic_distance"].mean()),
        "geographic_distance_median_m": float(pairs["geographic_distance"].median()),
        "geographic_distance_p95_m": float(pairs["geographic_distance"].quantile(0.95)),
        "mean_abs_Aa_difference": float(abs_aa_gap.mean()),
        "median_abs_Aa_difference": float(abs_aa_gap.median()),
        "mean_abs_log1p_Aa_difference": float(abs_log_gap.mean()),
        "median_abs_log1p_Aa_difference": float(abs_log_gap.median()),
        "zero_positive_pair_fraction": float(
            ((pairs["erosion_low"].eq(0) & pairs["erosion_high"].gt(0))
             | (pairs["erosion_high"].eq(0) & pairs["erosion_low"].gt(0))).mean()
        ),
        "failure_counts": matching_diagnostics.get("failure_counts", {}),
        "environment_variables": variables,
        **config,
    }
    rows: list[dict[str, Any]] = [{
        **diagnostics,
        "diagnostic_type": "summary",
        "variable": None,
        "stratum_type": "overall",
        "stratum": "all",
    }]
    for variable in variables:
        high_all = pd.to_numeric(pairs[f"{variable}_high"], errors="coerce").to_numpy(dtype=float)
        low_all = pd.to_numeric(pairs[f"{variable}_low"], errors="coerce").to_numpy(dtype=float)
        finite_pair = np.isfinite(high_all) & np.isfinite(low_all)
        rows.append({
            **diagnostics,
            "diagnostic_type": "variable_balance",
            "variable": variable,
            "stratum_type": "overall",
            "stratum": "all",
            "smd_before": smd_before_matching(source, variable, exposure, config, variables),
            "smd_after": standardized_mean_difference(high_all[finite_pair], low_all[finite_pair]),
            "mean_signed_difference": float(np.nanmean(high_all - low_all)) if finite_pair.any() else np.nan,
            "mean_absolute_difference": float(np.nanmean(np.abs(high_all - low_all))) if finite_pair.any() else np.nan,
            "p95_absolute_difference": float(np.nanquantile(np.abs(high_all - low_all), 0.95)) if finite_pair.any() else np.nan,
            "n_balance_pairs": int(finite_pair.sum()),
        })
        for soil_value, idx in pairs.groupby("soil_type", sort=True).groups.items():
            positions = np.asarray(list(idx), dtype=int)
            ok = np.isfinite(high_all[positions]) & np.isfinite(low_all[positions])
            rows.append({
                **diagnostics,
                "diagnostic_type": "variable_balance",
                "variable": variable,
                "stratum_type": "soil_type",
                "stratum": soil_value,
                "smd_after": standardized_mean_difference(high_all[positions][ok], low_all[positions][ok]),
                "mean_absolute_difference": float(np.nanmean(np.abs(high_all[positions] - low_all[positions]))) if ok.any() else np.nan,
                "n_balance_pairs": int(ok.sum()),
            })
        for group_value, idx in pairs.groupby("environment_group", sort=True).groups.items():
            positions = np.asarray(list(idx), dtype=int)
            ok = np.isfinite(high_all[positions]) & np.isfinite(low_all[positions])
            rows.append({
                **diagnostics,
                "diagnostic_type": "variable_balance",
                "variable": variable,
                "stratum_type": "environment_group",
                "stratum": group_value,
                "smd_after": standardized_mean_difference(high_all[positions][ok], low_all[positions][ok]),
                "mean_absolute_difference": float(np.nanmean(np.abs(high_all[positions] - low_all[positions]))) if ok.any() else np.nan,
                "n_balance_pairs": int(ok.sum()),
            })
    return diagnostics, pd.DataFrame.from_records(rows)


def smd_before_matching(source: pd.DataFrame, variable: str, exposure: str, selected: dict[str, Any], variables: list[str]) -> float:
    """Unmatched SMD between within-stratum top and bottom exposure ranks."""
    del selected, variables
    rank = pd.to_numeric(source["exposure_rank"], errors="coerce")
    values = pd.to_numeric(source[variable], errors="coerce")
    valid = rank.notna() & values.notna()
    high = values.loc[valid & rank.ge(0.5)].to_numpy(dtype=float)
    low = values.loc[valid & rank.lt(0.5)].to_numpy(dtype=float)
    return standardized_mean_difference(high, low)


def run_local_matching_search(
    frame: pd.DataFrame, config: dict[str, Any], output_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Outcome-blind local matching grid; the prior V5 config stays prespecified."""
    matching = config["matching"]
    local = matching["local_search"]
    variables_union = list(dict.fromkeys(v for names in config["environment_vectors"].values() for v in names))
    required = [*variables_union, "x", "y", "SoilType", *EXPOSURE_COLUMNS]
    mask = complete_mask(frame, required, config["primary_soil_codes"])
    pool = frame.loc[mask].copy()
    n_sample = min(int(local["sample_rows"]), len(pool))
    pool = pool.sample(n=n_sample, random_state=int(config["seed"])).sort_values("cell_id").reset_index(drop=True)
    grid_rows: list[dict[str, Any]] = []
    balance_rows: list[pd.DataFrame] = []
    grid_started = time.perf_counter()
    vector_models: dict[tuple[str, int], dict[str, Any]] = {}
    for vector_name, variables in config["environment_vectors"].items():
        for cluster_name, cluster_count in local["cluster_options"].items():
            key = (vector_name, int(cluster_count))
            model = fit_environment_model(
                pool, variables, clusters=int(cluster_count), seed=int(config["seed"]),
                primary_soils=config["primary_soil_codes"],
            )
            _, groups = apply_environment_model(model, pool, config["primary_soil_codes"])
            vector_models[key] = {"model": model, "groups": groups, "cluster_name": cluster_name}

    candidate_number = 0
    for vector_name, variables in config["environment_vectors"].items():
        for cluster_name, cluster_count in local["cluster_options"].items():
            bundle = vector_models[(vector_name, int(cluster_count))]
            _, groups = apply_environment_model(bundle["model"], pool, config["primary_soil_codes"])
            exposure = str(local["exposure"])
            exposure_values = pd.to_numeric(pool[exposure], errors="coerce").to_numpy(dtype=float)
            ranks = pd.DataFrame({"g": groups, "s": pool["SoilType"].to_numpy(), "e": exposure_values}).groupby(
                ["g", "s"], sort=False
            )["e"].rank(pct=True).to_numpy()
            valid = np.isfinite(np.asarray(bundle["model"]["scaler"].transform(pool[variables]), dtype=float)).all(axis=1)
            valid &= np.isfinite(ranks) & pool["SoilType"].isin(config["primary_soil_codes"]).to_numpy()
            groups_valid = groups[valid]
            exposure_valid = exposure_values[valid]
            soil_valid = pd.to_numeric(pool.loc[valid, "SoilType"], errors="coerce").to_numpy(dtype=int)
            ranks_valid = pd.DataFrame({"g": groups_valid, "s": soil_valid, "e": exposure_valid}).groupby(
                ["g", "s"], sort=False
            )["e"].rank(pct=True).to_numpy()
            # Keep the matching distance in the globally standardized training/search frame.
            z_valid = bundle["model"]["scaler"].transform(pool.loc[valid, variables].to_numpy(dtype=float))
            for caliper in local["environment_calipers"]:
                for radius in local["geographic_radii_m"]:
                    for rank_gap in local["erosion_rank_gaps"]:
                        candidate_number += 1
                        candidate_id = f"local_{candidate_number:03d}_{vector_name}_{cluster_name}_c{caliper}_r{radius}_g{rank_gap}"
                        match_input = pd.DataFrame({
                            "cell_id": pool.loc[valid, "cell_id"].to_numpy(),
                            "x": pd.to_numeric(pool.loc[valid, "x"], errors="coerce").to_numpy(dtype=float),
                            "y": pd.to_numeric(pool.loc[valid, "y"], errors="coerce").to_numpy(dtype=float),
                            "SoilType": soil_valid,
                            "environment_group": groups_valid,
                            "exposure_rank": ranks_valid,
                        })
                        distance_columns = []
                        for index in range(z_valid.shape[1]):
                            col = f"z{index}"
                            match_input[col] = z_valid[:, index]
                            distance_columns.append(col)
                        started = time.perf_counter()
                        try:
                            pairs, diag = match_cells(
                                match_input,
                                environment_columns=distance_columns,
                                group_col="environment_group",
                                erosion_col="exposure_rank",
                                soil_type_col="SoilType",
                                x_col="x",
                                y_col="y",
                                distance_limit=float(radius),
                                environment_distance_limit=float(caliper),
                                erosion_gap=float(rank_gap),
                                allow_reuse=False,
                            )
                            by_id = pool.set_index("cell_id", drop=False)
                            high = by_id.loc[pairs["high_erosion_cell_id"].tolist()] if len(pairs) else pd.DataFrame()
                            low = by_id.loc[pairs["low_erosion_cell_id"].tolist()] if len(pairs) else pd.DataFrame()
                            abs_smd = {}
                            signed_smd = {}
                            abs_diffs = {}
                            for var in variables:
                                if len(pairs):
                                    h = pd.to_numeric(high[var], errors="coerce").to_numpy(dtype=float)
                                    l = pd.to_numeric(low[var], errors="coerce").to_numpy(dtype=float)
                                    signed_smd[var] = standardized_mean_difference(h, l)
                                    abs_smd[var] = abs(signed_smd[var])
                                    abs_diffs[var] = float(np.nanmean(np.abs(h - l)))
                            actual_gap = (
                                pd.to_numeric(high[exposure], errors="coerce").to_numpy(dtype=float)
                                - pd.to_numeric(low[exposure], errors="coerce").to_numpy(dtype=float)
                            ) if len(pairs) else np.array([], dtype=float)
                            abs_log1p_gap = np.abs(
                                np.log1p(pd.to_numeric(high[exposure], errors="coerce").to_numpy(dtype=float))
                                - np.log1p(pd.to_numeric(low[exposure], errors="coerce").to_numpy(dtype=float))
                            ) if len(pairs) else np.array([], dtype=float)
                            record = {
                                "candidate_id": candidate_id,
                                "exposure": exposure,
                                "environment_vector": vector_name,
                                "environment_variables": json.dumps(variables, ensure_ascii=False),
                                "distance_method": "direct_nearest_neighbor",
                                "cluster_method": "none" if int(cluster_count) == 0 else "kmeans",
                                "cluster_count_per_soil": int(cluster_count),
                                "environment_caliper": float(caliper),
                                "geographic_radius_m": int(radius),
                                "erosion_rank_gap": float(rank_gap),
                                "exposure_rule": "within_group_rank",
                                "eligible_cells": int(diag["eligible_cells"]),
                                "n_pairs": int(len(pairs)),
                                "matched_cells": int(2 * len(pairs)),
                                "area_coverage": float(2 * len(pairs) / diag["eligible_cells"]) if diag["eligible_cells"] else np.nan,
                                "environment_distance_mean": float(pairs["environmental_distance"].mean()) if len(pairs) else np.nan,
                                "environment_distance_median": float(pairs["environmental_distance"].median()) if len(pairs) else np.nan,
                                "environment_distance_p95": float(pairs["environmental_distance"].quantile(.95)) if len(pairs) else np.nan,
                                "geographic_distance_mean_m": float(pairs["geographic_distance"].mean()) if len(pairs) else np.nan,
                                "geographic_distance_median_m": float(pairs["geographic_distance"].median()) if len(pairs) else np.nan,
                                "geographic_distance_p95_m": float(pairs["geographic_distance"].quantile(.95)) if len(pairs) else np.nan,
                                "mean_abs_Aa_difference": float(np.mean(np.abs(actual_gap))) if len(actual_gap) else np.nan,
                                "median_abs_Aa_difference": float(np.median(np.abs(actual_gap))) if len(actual_gap) else np.nan,
                                "mean_abs_log1p_Aa_difference": float(np.mean(abs_log1p_gap)) if len(abs_log1p_gap) else np.nan,
                                "median_abs_log1p_Aa_difference": float(np.median(abs_log1p_gap)) if len(abs_log1p_gap) else np.nan,
                                "max_abs_matched_smd": max(abs_smd.values(), default=np.nan),
                                "environment_variable_abs_smd_json": json.dumps(abs_smd, ensure_ascii=False),
                                "environment_variable_smd_json": json.dumps(signed_smd, ensure_ascii=False),
                                "environment_variable_abs_difference_json": json.dumps(abs_diffs, ensure_ascii=False),
                                "zero_positive_pair_fraction": float(((high[exposure].eq(0).to_numpy() & low[exposure].gt(0).to_numpy()) | (low[exposure].eq(0).to_numpy() & high[exposure].gt(0).to_numpy())).mean()) if len(pairs) else np.nan,
                                "failure_counts_json": json.dumps(diag.get("failure_counts", {}), ensure_ascii=False),
                                "runtime_seconds": time.perf_counter() - started,
                                "outcome_guided_selection": False,
                                "status": "ok" if len(pairs) else "no_pairs",
                            }
                            grid_rows.append(record)
                            detail = {
                                "candidate_id": candidate_id,
                                "exposure": exposure,
                                "method": record["cluster_method"],
                                "vector": vector_name,
                                "environment_caliper": caliper,
                                "geographic_radius_m": radius,
                                "erosion_rank_gap": rank_gap,
                                "diagnostic_type": "local_grid_summary",
                                "stratum_type": "overall",
                                "stratum": "all",
                                "n_pairs": len(pairs),
                                "area_coverage": record["area_coverage"],
                                "max_abs_matched_smd": record["max_abs_matched_smd"],
                                "environment_distance_mean": record["environment_distance_mean"],
                                "geographic_distance_mean_m": record["geographic_distance_mean_m"],
                                "mean_abs_Aa_difference": record["mean_abs_Aa_difference"],
                                "failure_counts_json": record["failure_counts_json"],
                            }
                            balance_rows.append(pd.DataFrame([detail]))
                            before_rank_frame = pd.DataFrame({
                                "group": groups_valid, "soil": soil_valid, "exposure": exposure_valid
                            })
                            before_rank = before_rank_frame.groupby(["group", "soil"], sort=False)["exposure"].rank(pct=True)
                            for var in variables:
                                high_values = pd.to_numeric(high[var], errors="coerce").to_numpy(dtype=float) if len(pairs) else np.array([])
                                low_values = pd.to_numeric(low[var], errors="coerce").to_numpy(dtype=float) if len(pairs) else np.array([])
                                raw_values = pd.to_numeric(pool.loc[valid, var], errors="coerce").to_numpy(dtype=float)
                                before_smd = standardized_mean_difference(
                                    raw_values[before_rank.to_numpy() >= .5], raw_values[before_rank.to_numpy() < .5]
                                )
                                balance_rows.append(pd.DataFrame([{
                                    **detail,
                                    "diagnostic_type": "local_grid_variable_balance",
                                    "variable": var,
                                    "stratum_type": "overall",
                                    "stratum": "all",
                                    "smd_before": before_smd,
                                    "smd_after": signed_smd.get(var),
                                    "mean_signed_difference": float(np.nanmean(high_values - low_values)) if len(high_values) else np.nan,
                                    "mean_absolute_difference": abs_diffs.get(var),
                                    "n_balance_pairs": len(pairs),
                                }]))
                        except Exception as exc:
                            grid_rows.append({
                                "candidate_id": candidate_id,
                                "exposure": exposure,
                                "environment_vector": vector_name,
                                "environment_variables": json.dumps(variables, ensure_ascii=False),
                                "distance_method": "direct_nearest_neighbor",
                                "cluster_method": "none" if int(cluster_count) == 0 else "kmeans",
                                "cluster_count_per_soil": int(cluster_count),
                                "environment_caliper": float(caliper),
                                "geographic_radius_m": int(radius),
                                "erosion_rank_gap": float(rank_gap),
                                "status": "failed",
                                "error": f"{type(exc).__name__}: {exc}",
                                "runtime_seconds": time.perf_counter() - started,
                                "outcome_guided_selection": False,
                            })
    result = pd.DataFrame.from_records(grid_rows)
    valid = result.loc[result["status"].eq("ok")].copy()
    if not valid.empty:
        valid["balance_rank"] = valid["max_abs_matched_smd"].rank(method="average", ascending=True, pct=True)
        valid["coverage_rank"] = valid["area_coverage"].rank(method="average", ascending=True, pct=True)
        valid["distance_rank"] = valid["environment_distance_mean"].rank(method="average", ascending=True, pct=True)
        valid["erosion_gap_rank"] = valid["median_abs_log1p_Aa_difference"].rank(method="average", ascending=True, pct=True)
        valid["design_score_balance_coverage_distance_exposure"] = (
            (1 - valid["balance_rank"]) + valid["coverage_rank"] + (1 - valid["distance_rank"]) + valid["erosion_gap_rank"]
        )
        valid["outcome_blind_rank"] = valid["design_score_balance_coverage_distance_exposure"].rank(
            method="min", ascending=False
        ).astype(int)
        valid["pareto_candidate"] = False
        values = valid[["area_coverage", "max_abs_matched_smd", "environment_distance_mean", "median_abs_log1p_Aa_difference"]].to_numpy(dtype=float)
        for i in range(len(values)):
            no_worse = (values[:, 0] >= values[i, 0]) & (values[:, 1] <= values[i, 1]) & (values[:, 2] <= values[i, 2]) & (values[:, 3] >= values[i, 3])
            strictly_better = (values[:, 0] > values[i, 0]) | (values[:, 1] < values[i, 1]) | (values[:, 2] < values[i, 2]) | (values[:, 3] > values[i, 3])
            if not np.any(no_worse & strictly_better):
                valid.iloc[i, valid.columns.get_loc("pareto_candidate")] = True
        result = result.merge(
            valid[["candidate_id", "design_score_balance_coverage_distance_exposure", "outcome_blind_rank", "pareto_candidate"]],
            on="candidate_id", how="left",
        )
    write_csv(output_dir / "local_matching_grid.csv", result)
    balance = pd.concat(balance_rows, ignore_index=True) if balance_rows else pd.DataFrame()
    write_csv(output_dir / "matching_balance_diagnostics.csv", balance)
    print(f"Local matching grid: {len(result)} configurations, {time.perf_counter() - grid_started:.1f}s")
    return result, balance


def assign_spatial_roles(frame: pd.DataFrame, block_size_m: float, seed: int) -> pd.Series:
    x_block = np.floor(pd.to_numeric(frame["x"], errors="coerce") / block_size_m).astype(int)
    y_block = np.floor(pd.to_numeric(frame["y"], errors="coerce") / block_size_m).astype(int)
    blocks = x_block.astype(str) + "_" + y_block.astype(str)
    unique = np.asarray(sorted(blocks.unique()), dtype=object)
    if len(unique) < 3:
        raise ValueError(f"Need at least three spatial blocks; found {len(unique)}")
    permutation = np.random.default_rng(seed).permutation(unique)
    n_train = max(1, int(round(0.60 * len(unique))))
    n_validation = max(1, int(round(0.20 * len(unique))))
    if n_train + n_validation >= len(unique):
        n_train = len(unique) - 2
        n_validation = 1
    train_blocks = set(permutation[:n_train])
    validation_blocks = set(permutation[n_train : n_train + n_validation])
    role_by_block = {
        **{str(block): "train" for block in train_blocks},
        **{str(block): "validation" for block in validation_blocks},
        **{str(block): "test" for block in permutation[n_train + n_validation :]},
    }
    return pd.Series([role_by_block[value] for value in blocks], index=frame.index, name="spatial_role")


def safe_score(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    finite = np.isfinite(y_true) & np.isfinite(prediction)
    y_true, prediction = y_true[finite], prediction[finite]
    if len(y_true) < 2:
        return {"r_squared": np.nan, "rmse": np.nan, "mae": np.nan}
    return {
        "r_squared": float(r2_score(y_true, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, prediction))),
        "mae": float(mean_absolute_error(y_true, prediction)),
    }


def ols_design_result(
    train: pd.DataFrame,
    score: pd.DataFrame,
    outcome: str,
    exposure: str,
    model_code: str,
    env_model: dict[str, Any],
    primary_soils: list[int],
) -> dict[str, Any]:
    """Fit one M0/M1/M2 model on train cells and evaluate same-fold cells."""
    variables = env_model["variables"]
    needs = [outcome, exposure, *variables]
    train_mask = complete_mask(train, needs, primary_soils)
    score_mask = complete_mask(score, needs, primary_soils)
    tr = train.loc[train_mask].copy()
    sc = score.loc[score_mask].copy()
    if len(tr) < 5 or len(sc) < 2:
        return {
            "status": "insufficient_sample", "n_train": len(tr), "n_score": len(sc),
            "r_squared": np.nan, "rmse": np.nan, "mae": np.nan, "beta_log1p_erosion": np.nan,
        }
    z_train, group_train = apply_environment_model(env_model, tr, primary_soils)
    z_score, group_score = apply_environment_model(env_model, sc, primary_soils)
    x_train = np.log1p(pd.to_numeric(tr[exposure], errors="coerce").to_numpy(dtype=float))
    x_score = np.log1p(pd.to_numeric(sc[exposure], errors="coerce").to_numpy(dtype=float))
    y_train = pd.to_numeric(tr[outcome], errors="coerce").to_numpy(dtype=float)
    y_score = pd.to_numeric(sc[outcome], errors="coerce").to_numpy(dtype=float)
    if model_code == "M0":
        design_train = np.column_stack([np.ones(len(tr)), x_train])
        design_score = np.column_stack([np.ones(len(sc)), x_score])
        names = ["const", "log1p_erosion"]
    elif model_code == "M1":
        design_train = np.column_stack([np.ones(len(tr)), x_train, z_train])
        design_score = np.column_stack([np.ones(len(sc)), x_score, z_score])
        names = ["const", "log1p_erosion", *[f"z_{name}" for name in variables]]
    elif model_code == "M2":
        categories = np.unique(group_train[group_train >= 0])
        if len(categories) < 2:
            return {
                "status": "insufficient_environment_groups", "n_train": len(tr), "n_score": len(sc),
                "r_squared": np.nan, "rmse": np.nan, "mae": np.nan, "beta_log1p_erosion": np.nan,
            }
        baseline = int(categories[0])
        train_dummies = np.column_stack([group_train == category for category in categories[1:]]).astype(float)
        score_dummies = np.column_stack([group_score == category for category in categories[1:]]).astype(float)
        design_train = np.column_stack([np.ones(len(tr)), x_train, train_dummies, train_dummies * x_train[:, None]])
        design_score = np.column_stack([np.ones(len(sc)), x_score, score_dummies, score_dummies * x_score[:, None]])
        names = ["const", "log1p_erosion", *[f"group_{c}" for c in categories[1:]], *[f"interaction_{c}" for c in categories[1:]]]
    else:
        raise ValueError(f"Unknown cell model: {model_code}")
    if np.linalg.matrix_rank(design_train) < design_train.shape[1]:
        return {
            "status": "singular_design", "n_train": len(tr), "n_score": len(sc),
            "r_squared": np.nan, "rmse": np.nan, "mae": np.nan, "beta_log1p_erosion": np.nan,
        }
    fitted = sm.OLS(y_train, design_train).fit()
    prediction = np.asarray(fitted.predict(design_score), dtype=float)
    metrics = safe_score(y_score, prediction)
    coefficient = float(fitted.params[1])
    if model_code == "M2":
        slopes = {baseline: coefficient}
        for index, category in enumerate(categories[1:]):
            slopes[int(category)] = coefficient + float(fitted.params[1 + len(categories) + index])
        counts = pd.Series(group_train).value_counts().to_dict()
        total = sum(counts.values()) or 1
        coefficient = float(sum(slopes.get(int(group), 0.0) * count for group, count in counts.items()) / total)
    return {
        "status": "ok",
        "n_train": int(len(tr)),
        "n_score": int(len(sc)),
        "r_squared": metrics["r_squared"],
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "beta_log1p_erosion": coefficient,
        "direction": "positive" if coefficient > 0 else "negative" if coefficient < 0 else "zero",
        "train_r_squared": float(fitted.rsquared),
        "model_parameters": json.dumps(dict(zip(names, np.asarray(fitted.params).tolist(), strict=True))),
    }


def fit_paired_model(pairs: pd.DataFrame, outcome: str, model_name: str) -> dict[str, Any]:
    required = ["delta_log1p_erosion", f"delta_{outcome}"]
    covariates = ["delta_Elevation", "delta_LST"] if model_name == "P3_environment_difference_adjusted" else []
    required.extend(covariates)
    missing = [name for name in required if name not in pairs]
    if missing:
        return {"status": "missing_columns", "error": ",".join(missing), "n_pairs": 0}
    data = pairs[required].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    n = len(data)
    if n < 5:
        return {"status": "insufficient_sample", "n_pairs": n}
    x = data["delta_log1p_erosion"].to_numpy(dtype=float)
    y = data[f"delta_{outcome}"].to_numpy(dtype=float)
    if np.std(x, ddof=1) == 0 or np.std(y, ddof=1) == 0:
        return {"status": "no_variation", "n_pairs": n}
    if model_name == "P2_no_intercept_ols":
        design = data[["delta_log1p_erosion"]].to_numpy(dtype=float)
        fitted = sm.OLS(y, design).fit()
        coefficient = float(fitted.params[0])
        ci = np.asarray(fitted.conf_int(alpha=0.05))[0]
        prediction = np.asarray(fitted.predict(design), dtype=float)
        r_squared = float(fitted.rsquared)
        adjusted_r_squared = float(fitted.rsquared_adj)
        standard_error = float(fitted.bse[0])
        p_value = float(fitted.pvalues[0])
        intercept = 0.0
    else:
        design_frame = data[["delta_log1p_erosion", *covariates]]
        design = sm.add_constant(design_frame, has_constant="add")
        if np.linalg.matrix_rank(design.to_numpy(dtype=float)) < design.shape[1]:
            return {"status": "singular_design", "n_pairs": n}
        if model_name == "P4_huber":
            fitted = sm.RLM(y, design, M=sm.robust.norms.HuberT()).fit()
            ci_values = np.asarray(fitted.conf_int(alpha=0.05), dtype=float)
            ci = ci_values[1]
            r_squared = float(1 - np.sum((y - fitted.fittedvalues) ** 2) / np.sum((y - y.mean()) ** 2))
            adjusted_r_squared = np.nan
            standard_error = float(fitted.bse.iloc[1])
            p_value = float(fitted.pvalues.iloc[1])
            coefficient = float(fitted.params.iloc[1])
            intercept = float(fitted.params.iloc[0])
            prediction = np.asarray(fitted.predict(design), dtype=float)
        else:
            fitted = sm.OLS(y, design).fit()
            ci = np.asarray(fitted.conf_int(alpha=0.05).loc["delta_log1p_erosion"], dtype=float)
            r_squared = float(fitted.rsquared)
            adjusted_r_squared = float(fitted.rsquared_adj)
            standard_error = float(fitted.bse["delta_log1p_erosion"])
            p_value = float(fitted.pvalues["delta_log1p_erosion"])
            coefficient = float(fitted.params["delta_log1p_erosion"])
            intercept = float(fitted.params["const"])
            prediction = np.asarray(fitted.predict(design), dtype=float)
    return {
        "status": "ok",
        "n_pairs": int(n),
        "coefficient": coefficient,
        "standardized_coefficient": coefficient * float(np.std(x, ddof=1)) / float(np.std(y, ddof=1)),
        "intercept": float(intercept),
        "standard_error": standard_error,
        "ci_lower": float(ci[0]),
        "ci_upper": float(ci[1]),
        "p_value": p_value,
        "r_squared": r_squared,
        "adjusted_r_squared": adjusted_r_squared,
        "rmse": float(np.sqrt(np.mean((y - prediction) ** 2))),
        "mae": float(np.mean(np.abs(y - prediction))),
        "mean_delta_outcome": float(np.mean(y)),
        "median_delta_outcome": float(np.median(y)),
        "positive_delta_outcome_fraction": float(np.mean(y > 0)),
        "mean_delta_log1p_erosion": float(np.mean(x)),
        "median_delta_log1p_erosion": float(np.median(x)),
        "mean_delta_Aa": float(np.mean(pd.to_numeric(pairs.loc[data.index, "erosion_difference"], errors="coerce"))),
        "zero_positive_pair_fraction": float(
            ((pd.to_numeric(pairs.loc[data.index, "erosion_low"], errors="coerce").eq(0) & pd.to_numeric(pairs.loc[data.index, "erosion_high"], errors="coerce").gt(0))
             | (pd.to_numeric(pairs.loc[data.index, "erosion_high"], errors="coerce").eq(0) & pd.to_numeric(pairs.loc[data.index, "erosion_low"], errors="coerce").gt(0))).mean()
        ),
        "direction": "positive" if coefficient > 0 else "negative" if coefficient < 0 else "zero",
    }


def predict_pair_score(train_pairs: pd.DataFrame, score_pairs: pd.DataFrame, outcome: str, model_name: str) -> dict[str, Any]:
    train_need = ["delta_log1p_erosion", f"delta_{outcome}"]
    score_need = train_need.copy()
    tr = train_pairs[train_need].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    sc = score_pairs[score_need].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(tr) < 5 or len(sc) < 2:
        return {"status": "insufficient_sample", "n_train_pairs": len(tr), "n_score_pairs": len(sc)}
    x_train = tr["delta_log1p_erosion"].to_numpy(dtype=float)[:, None]
    x_score = sc["delta_log1p_erosion"].to_numpy(dtype=float)[:, None]
    y_train = tr[f"delta_{outcome}"].to_numpy(dtype=float)
    y_score = sc[f"delta_{outcome}"].to_numpy(dtype=float)
    if model_name == "P5_natural_spline":
        if np.unique(x_train).size < 4:
            return {"status": "insufficient_exposure_levels", "n_train_pairs": len(tr), "n_score_pairs": len(sc)}
        design_train, design_score = natural_spline_design(
            x_train[:, 0], x_score[:, 0], df=4, clip_score_to_training=True
        )
        fitted = sm.OLS(y_train, sm.add_constant(design_train, has_constant="add")).fit()
        prediction = np.asarray(fitted.predict(sm.add_constant(design_score, has_constant="add")), dtype=float)
    else:
        design_train = sm.add_constant(x_train, has_constant="add")
        design_score = sm.add_constant(x_score, has_constant="add")
        fitted = sm.OLS(y_train, design_train).fit()
        prediction = np.asarray(fitted.predict(design_score), dtype=float)
    metrics = safe_score(y_score, prediction)
    return {"status": "ok", "n_train_pairs": len(tr), "n_score_pairs": len(sc), **metrics}


def natural_spline_design(
    train_x: np.ndarray, score_x: np.ndarray, *, df: int, clip_score_to_training: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Build natural cubic spline bases with knots learned from training x only."""
    from patsy import build_design_matrices, dmatrix

    train_x = np.asarray(train_x, dtype=float).reshape(-1)
    score_x = np.asarray(score_x, dtype=float).reshape(-1)
    degrees_freedom = max(3, min(int(df), np.unique(train_x).size - 1))
    basis_train = dmatrix(f"cr(x, df={degrees_freedom}) - 1", {"x": train_x}, return_type="dataframe")
    score_values = np.clip(score_x, float(train_x.min()), float(train_x.max())) if clip_score_to_training else score_x
    basis_score = build_design_matrices([basis_train.design_info], {"x": score_values})[0]
    return np.asarray(basis_train, dtype=float), np.asarray(basis_score, dtype=float)


def adjusted_p_values(values: pd.Series) -> np.ndarray:
    from statsmodels.stats.multitest import multipletests

    result = np.full(len(values), np.nan, dtype=float)
    valid = np.isfinite(pd.to_numeric(values, errors="coerce").to_numpy(dtype=float))
    if valid.any():
        result[valid] = multipletests(values.to_numpy(dtype=float)[valid], method="fdr_bh")[1]
    return result


def run_spatial_validation(
    frame: pd.DataFrame, config: dict[str, Any], run_dir: Path, provenance_record: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[pd.DataFrame], dict[str, Any]]:
    validation_config = config["spatial_validation"]
    block_size = float(validation_config["block_size_m"])
    soils = config["primary_soil_codes"]
    variables = config["environment_vectors"]["V5"]
    main = frame.loc[frame["SoilType"].isin(soils) & frame[["x", "y"]].notna().all(axis=1)].copy()
    cell_rows: list[dict[str, Any]] = []
    pair_score_rows: list[dict[str, Any]] = []
    increment_rows: list[dict[str, Any]] = []
    pair_balance: list[pd.DataFrame] = []
    pair_data_files: list[dict[str, Any]] = []
    fold_metadata: list[dict[str, Any]] = []
    pair_time = time.perf_counter()
    checkpoint_root = run_dir / "spatial_validation" / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    for seed in validation_config["seeds"]:
        seed_dir = checkpoint_root / f"seed_{int(seed)}"
        checkpoint_manifest_path = seed_dir / "checkpoint.json"
        checkpoint_paths = {
            "cell": seed_dir / "cell_rows.json",
            "pairs": seed_dir / "pair_score_rows.json",
            "increments": seed_dir / "increment_rows.json",
            "balance": seed_dir / "pair_balance.json",
            "fold": seed_dir / "fold_metadata.json",
        }
        pair_data_parquet = seed_dir / "pair_data.parquet"
        pair_data_json = seed_dir / "pair_data.json"
        checkpoint_is_complete = False
        if checkpoint_manifest_path.exists() and all(path.exists() for path in checkpoint_paths.values()):
            checkpoint_manifest = json.loads(checkpoint_manifest_path.read_text())
            checkpoint_is_complete = (
                checkpoint_manifest.get("status") == "complete"
                and checkpoint_manifest.get("seed") == int(seed)
                and same_run_signature(checkpoint_manifest.get("signature"), provenance_record)
                and (pair_data_parquet.exists() or pair_data_json.exists())
            )
        if checkpoint_is_complete:
            cell_cache = read_frame_cache(checkpoint_paths["cell"])
            pair_cache = read_frame_cache(checkpoint_paths["pairs"])
            increment_cache = read_frame_cache(checkpoint_paths["increments"])
            balance_cache = read_frame_cache(checkpoint_paths["balance"])
            cell_rows.extend(cell_cache.to_dict(orient="records"))
            pair_score_rows.extend(pair_cache.to_dict(orient="records"))
            increment_rows.extend(increment_cache.to_dict(orient="records"))
            if not balance_cache.empty:
                pair_balance.append(balance_cache)
            pair_data_path = pair_data_parquet if pair_data_parquet.exists() else pair_data_json
            pair_data_files.append({
                "seed": int(seed), "path": str(pair_data_path.relative_to(ROOT)),
                "rows": int(checkpoint_manifest.get("pair_data_rows", 0)),
            })
            fold_metadata.append(json.loads(checkpoint_paths["fold"].read_text()))
            print(f"Spatial seed {seed}: reused completed checkpoint")
            continue

        seed_cell_start = len(cell_rows)
        seed_pair_score_start = len(pair_score_rows)
        seed_increment_start = len(increment_rows)
        seed_balance_start = len(pair_balance)
        seed_pair_frames: list[pd.DataFrame] = []
        seed_dir.mkdir(parents=True, exist_ok=True)
        roles = assign_spatial_roles(main, block_size, int(seed))
        fold = main.copy()
        fold["spatial_role"] = roles
        role_counts = fold.groupby("spatial_role")["cell_id"].agg(["count", "nunique"]).to_dict(orient="index")
        fold_metadata.append({
            "seed": int(seed),
            "block_size_m": block_size,
            "n_spatial_blocks": int(np.floor(fold["x"] / block_size).astype(int).astype(str).add("_").add(np.floor(fold["y"] / block_size).astype(int).astype(str)).nunique()),
            "role_counts": role_counts,
        })
        train_all = fold.loc[fold["spatial_role"].eq("train")]
        env_model = fit_environment_model(train_all, variables, clusters=6, seed=int(seed), primary_soils=soils)
        z_train, group_train = apply_environment_model(env_model, train_all, soils)

        for outcome in config["outcomes"]:
            common_cols = [*variables, *EXPOSURE_COLUMNS, outcome]
            common_mask = complete_mask(fold, common_cols, soils)
            common = fold.loc[common_mask].copy()
            common_roles = common["spatial_role"]
            for exposure in EXPOSURE_COLUMNS:
                for role_name in ["validation", "test"]:
                    train = common.loc[common_roles.eq("train")]
                    score = common.loc[common_roles.eq(role_name)]
                    for model_code in ["M0", "M1", "M2"]:
                        result = ols_design_result(train, score, outcome, exposure, model_code, env_model, soils)
                        cell_rows.append({
                            "seed": int(seed), "block_size_m": block_size, "outcome": outcome,
                            "exposure": exposure, "method": model_code, "model_name": MODEL_NAMES[model_code],
                            "evaluation_set": role_name, "common_sample_for_exposures": True,
                            "status": result.get("status"), **result,
                        })
                    # Same rows, OLS family, split and environmental covariates for the nested ΔR² comparison.
                    train_mask = complete_mask(train, [outcome, exposure, *variables], soils)
                    score_mask = complete_mask(score, [outcome, exposure, *variables], soils)
                    tr = train.loc[train_mask]
                    sc = score.loc[score_mask]
                    z_tr, _ = apply_environment_model(env_model, tr, soils)
                    z_sc, _ = apply_environment_model(env_model, sc, soils)
                    if len(tr) >= 10 and len(sc) >= 2:
                        x_tr = np.log1p(pd.to_numeric(tr[exposure], errors="coerce").to_numpy(dtype=float))
                        x_sc = np.log1p(pd.to_numeric(sc[exposure], errors="coerce").to_numpy(dtype=float))
                        y_tr = pd.to_numeric(tr[outcome], errors="coerce").to_numpy(dtype=float)
                        y_sc = pd.to_numeric(sc[outcome], errors="coerce").to_numpy(dtype=float)
                        env_train = sm.add_constant(z_tr, has_constant="add")
                        env_score = sm.add_constant(z_sc, has_constant="add")
                        both_train = np.column_stack([env_train, x_tr])
                        both_score = np.column_stack([env_score, x_sc])
                        fit_env = sm.OLS(y_tr, env_train).fit()
                        fit_both = sm.OLS(y_tr, both_train).fit()
                        metrics_env = safe_score(y_sc, fit_env.predict(env_score))
                        metrics_both = safe_score(y_sc, fit_both.predict(both_score))
                        increment_rows.append({
                            "analysis_type": "cell_level_nested_ols",
                            "seed": int(seed), "block_size_m": block_size, "outcome": outcome,
                            "exposure": exposure, "evaluation_set": role_name,
                            "model_1": "environment_only_ols",
                            "model_2": "environment_plus_log1p_Aa_ols",
                            "same_sample_n": int(len(sc)), "training_n": int(len(tr)),
                            "r2_environment_only": metrics_env["r_squared"],
                            "r2_environment_plus_Aa": metrics_both["r_squared"],
                            "delta_r2": metrics_both["r_squared"] - metrics_env["r_squared"],
                            "beta_log1p_Aa": float(fit_both.params[-1]),
                            "direction": "positive" if fit_both.params[-1] > 0 else "negative" if fit_both.params[-1] < 0 else "zero",
                            "status": "ok",
                        })

        for exposure in EXPOSURE_COLUMNS:
            for method in ["M3", "M4"]:
                pair_by_role = {}
                for role_name in ["train", "validation", "test"]:
                    role_frame = fold.loc[fold["spatial_role"].eq(role_name)].copy()
                    pairs, diagnostics, balance = build_pair_table(
                        role_frame,
                        exposure=exposure,
                        variables=variables,
                        env_model=env_model,
                        method=method,
                        seed=int(seed),
                        role=role_name,
                        config=config,
                        primary_soils=soils,
                        block_size_m=block_size,
                    )
                    pair_by_role[role_name] = pairs
                    pair_balance.append(balance)
                    if not pairs.empty:
                        seed_pair_frames.append(pairs)
                        endpoints = pd.concat([pairs["high_erosion_cell_id"], pairs["low_erosion_cell_id"]], ignore_index=True)
                        if endpoints.duplicated().any():
                            raise ValueError(f"Cells were reused within {method}/{exposure}/{role_name}/{seed}")
                        expected = fold.set_index("cell_id").loc[endpoints, "spatial_role"].to_numpy()
                        if not np.all(expected == role_name):
                            raise ValueError(f"Pair endpoint crossed spatial role in {method}/{exposure}/{seed}")
                train_pairs = pair_by_role["train"]
                for outcome in config["outcomes"]:
                    if train_pairs.empty:
                        continue
                    train_fit = fit_paired_model(train_pairs, outcome, "P1_intercept_ols")
                    if train_fit.get("status") != "ok":
                        continue
                    for role_name in ["validation", "test"]:
                        score_pairs = pair_by_role[role_name]
                        linear = predict_pair_score(train_pairs, score_pairs, outcome, "P1_intercept_ols")
                        pair_score_rows.append({
                            "seed": int(seed), "block_size_m": block_size, "method": method,
                            "model_code": method, "model_name": MODEL_NAMES[method], "outcome": outcome,
                            "exposure": exposure, "evaluation_set": role_name,
                            "validation_model": "P1_intercept_ols",
                            "status": linear.get("status"), "n_train_pairs": linear.get("n_train_pairs"),
                            "n_score_pairs": linear.get("n_score_pairs"),
                            "r_squared": linear.get("r_squared"), "rmse": linear.get("rmse"),
                            "mae": linear.get("mae"),
                            "beta_log1p_erosion_train": train_fit.get("coefficient"),
                            "direction_train": train_fit.get("direction"),
                            "train_pair_count": train_fit.get("n_pairs"),
                        })
                        if outcome in PRIMARY_OUTCOMES:
                            spline = predict_pair_score(train_pairs, score_pairs, outcome, "P5_natural_spline")
                            pair_score_rows.append({
                                "seed": int(seed), "block_size_m": block_size, "method": method,
                                "model_code": method, "model_name": MODEL_NAMES[method], "outcome": outcome,
                                "exposure": exposure, "evaluation_set": role_name,
                                "validation_model": "P5_natural_spline_diagnostic",
                                "status": spline.get("status"), "n_train_pairs": spline.get("n_train_pairs"),
                                "n_score_pairs": spline.get("n_score_pairs"),
                                "r_squared": spline.get("r_squared"), "rmse": spline.get("rmse"),
                                "mae": spline.get("mae"),
                                "beta_log1p_erosion_train": np.nan,
                                "direction_train": "not_applicable",
                                "train_pair_count": len(train_pairs),
                            })
                        if role_name == "test" and not score_pairs.empty:
                            delta_need = ["delta_log1p_erosion", f"delta_{outcome}", "delta_Elevation", "delta_LST"]
                            tr = train_pairs[delta_need].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                            sc = score_pairs[delta_need].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                            if len(tr) > 10 and len(sc) > 2:
                                y_tr = tr[f"delta_{outcome}"].to_numpy(dtype=float)
                                y_sc = sc[f"delta_{outcome}"].to_numpy(dtype=float)
                                env_tr = sm.add_constant(tr[["delta_Elevation", "delta_LST"]], has_constant="add")
                                env_sc = sm.add_constant(sc[["delta_Elevation", "delta_LST"]], has_constant="add")
                                plus_tr = sm.add_constant(tr[["delta_log1p_erosion", "delta_Elevation", "delta_LST"]], has_constant="add")
                                plus_sc = sm.add_constant(sc[["delta_log1p_erosion", "delta_Elevation", "delta_LST"]], has_constant="add")
                                base_fit = sm.OLS(y_tr, env_tr).fit()
                                plus_fit = sm.OLS(y_tr, plus_tr).fit()
                                base_metrics = safe_score(y_sc, base_fit.predict(env_sc))
                                plus_metrics = safe_score(y_sc, plus_fit.predict(plus_sc))
                                increment_rows.append({
                                    "analysis_type": "matched_pair_nested_ols",
                                    "seed": int(seed), "block_size_m": block_size, "outcome": outcome,
                                    "exposure": exposure, "method": method, "evaluation_set": "test",
                                    "model_1": "environment_difference_only_ols",
                                    "model_2": "environment_difference_plus_delta_log1p_Aa_ols",
                                    "same_sample_n": int(len(sc)), "training_n": int(len(tr)),
                                    "r2_environment_only": base_metrics["r_squared"],
                                    "r2_environment_plus_Aa": plus_metrics["r_squared"],
                                    "delta_r2": plus_metrics["r_squared"] - base_metrics["r_squared"],
                                    "beta_log1p_Aa": float(plus_fit.params.iloc[1]),
                                    "direction": "positive" if plus_fit.params.iloc[1] > 0 else "negative" if plus_fit.params.iloc[1] < 0 else "zero",
                                    "status": "ok",
                                })
                    for role_name in ["validation", "test"]:
                        score_pairs = pair_by_role[role_name]
                        if score_pairs.empty or "descriptive_environment_group" not in score_pairs:
                            continue
                        for group_value, subgroup_pairs in score_pairs.groupby("descriptive_environment_group", dropna=True):
                            if len(subgroup_pairs) < 20 or f"delta_{outcome}" not in subgroup_pairs:
                                continue
                            subgroup_fit = fit_paired_model(subgroup_pairs, outcome, "P1_intercept_ols")
                            if subgroup_fit.get("status") != "ok":
                                continue
                            pair_score_rows.append({
                                "seed": int(seed), "block_size_m": block_size, "method": method,
                                "model_code": method, "model_name": MODEL_NAMES[method],
                                "outcome": outcome, "exposure": exposure,
                                "evaluation_set": role_name,
                                "validation_model": "P1_environment_group_stratum",
                                "environment_group": int(group_value),
                                "status": subgroup_fit.get("status"),
                                "n_score_pairs": subgroup_fit.get("n_pairs"),
                                "r_squared": subgroup_fit.get("r_squared"),
                                "rmse": subgroup_fit.get("rmse"), "mae": subgroup_fit.get("mae"),
                                "coefficient": subgroup_fit.get("coefficient"),
                                "ci_lower": subgroup_fit.get("ci_lower"),
                                "ci_upper": subgroup_fit.get("ci_upper"),
                                "direction": subgroup_fit.get("direction"),
                            })
        seed_cell_cache = pd.DataFrame.from_records(cell_rows[seed_cell_start:])
        seed_pair_cache = pd.DataFrame.from_records(pair_score_rows[seed_pair_score_start:])
        seed_increment_cache = pd.DataFrame.from_records(increment_rows[seed_increment_start:])
        seed_balance_parts = pair_balance[seed_balance_start:]
        seed_balance_cache = pd.concat(seed_balance_parts, ignore_index=True, sort=False) if seed_balance_parts else pd.DataFrame()
        seed_pair_data_cache = pd.concat(seed_pair_frames, ignore_index=True, sort=False) if seed_pair_frames else pd.DataFrame()
        write_frame_cache(checkpoint_paths["cell"], seed_cell_cache)
        write_frame_cache(checkpoint_paths["pairs"], seed_pair_cache)
        write_frame_cache(checkpoint_paths["increments"], seed_increment_cache)
        write_frame_cache(checkpoint_paths["balance"], seed_balance_cache)
        write_json(checkpoint_paths["fold"], fold_metadata[-1])
        if not seed_pair_data_cache.empty:
            seed_pair_data_cache.to_parquet(pair_data_parquet, index=False, compression="zstd")
            if pair_data_json.exists():
                pair_data_json.unlink()
        else:
            write_frame_cache(pair_data_json, seed_pair_data_cache)
            if pair_data_parquet.exists():
                pair_data_parquet.unlink()
        write_json(checkpoint_manifest_path, {
            "status": "complete", "seed": int(seed), "signature": provenance_record,
            "cell_rows": int(len(seed_cell_cache)), "pair_score_rows": int(len(seed_pair_cache)),
            "increment_rows": int(len(seed_increment_cache)), "pair_data_rows": int(len(seed_pair_data_cache)),
        })
        pair_data_path = pair_data_parquet if pair_data_parquet.exists() else pair_data_json
        pair_data_files.append({
            "seed": int(seed), "path": str(pair_data_path.relative_to(ROOT)),
            "rows": int(len(seed_pair_data_cache)),
        })
        print(f"Spatial seed {seed}: train/validation/test blocks assigned; paired roles built and checkpointed")

    pair_cv = pd.DataFrame.from_records(pair_score_rows)
    cell_cv = pd.DataFrame.from_records(cell_rows)
    repeated = pd.concat([cell_cv, pair_cv], ignore_index=True, sort=False)
    increments = pd.DataFrame.from_records(increment_rows)
    balance = pd.concat(pair_balance, ignore_index=True, sort=True) if pair_balance else pd.DataFrame()
    folds = {"folds": fold_metadata}
    print(f"Repeated spatial validation complete in {time.perf_counter() - pair_time:.1f}s")
    return repeated, increments, balance, [], {
        "pair_data_files": pair_data_files,
        "pair_data_rows": int(sum(item["rows"] for item in pair_data_files)),
        "folds": folds,
    }


def run_full_sample_pairs(frame: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    variables = config["environment_vectors"]["V5"]
    soils = config["primary_soil_codes"]
    full_model = fit_environment_model(frame, variables, clusters=6, seed=int(config["seed"]), primary_soils=soils)
    full_pairs: list[pd.DataFrame] = []
    diagnostic_rows: list[pd.DataFrame] = []
    for exposure in EXPOSURE_COLUMNS:
        for method in ["M3", "M4"]:
            pairs, diagnostics, balance = build_pair_table(
                frame,
                exposure=exposure,
                variables=variables,
                env_model=full_model,
                method=method,
                seed=int(config["seed"]),
                role="full_sample_descriptive",
                config=config,
                primary_soils=soils,
                block_size_m=float(config["spatial_validation"]["block_size_m"]),
            )
            if not pairs.empty:
                full_pairs.append(pairs)
            diagnostic_rows.append(balance)
            print(f"Full-sample pairs: {method} / {exposure}: {len(pairs):,}")
    pair_table = pd.concat(full_pairs, ignore_index=True, sort=False) if full_pairs else pd.DataFrame()
    balance_table = pd.concat(diagnostic_rows, ignore_index=True, sort=True) if diagnostic_rows else pd.DataFrame()
    return pair_table, balance_table


def run_full_paired_models(pair_table: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (method, exposure), group in pair_table.groupby(["method", "exposure"], sort=True):
        for outcome in config["outcomes"]:
            for model_name in config["paired_models"]:
                result = fit_paired_model(group, outcome, model_name)
                rows.append({
                    "method": method,
                    "method_name": MODEL_NAMES[method],
                    "exposure": exposure,
                    "outcome": outcome,
                    "model": model_name,
                    "evaluation_set": "full_sample_descriptive_apparent",
                    "selection_scope": "prespecified_matching_configuration",
                    "outcome_guided_selection": False,
                    **result,
                })
    return pd.DataFrame.from_records(rows)


def paired_subgroup_results(
    pair_table: pd.DataFrame, frame: pd.DataFrame, config: dict[str, Any], repeated: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate subgroup slopes and predeclared interaction tests on M4 pairs."""
    rows: list[dict[str, Any]] = []
    high_rows: list[dict[str, Any]] = []
    v5 = config["environment_vectors"]["V5"]
    full_env_model = fit_environment_model(frame, v5, clusters=6, seed=int(config["seed"]), primary_soils=config["primary_soil_codes"])
    _, labels = apply_environment_model(full_env_model, frame, config["primary_soil_codes"])
    env_group = pd.Series(labels, index=frame["cell_id"])
    for exposure in EXPOSURE_COLUMNS:
        exposure_cells = frame.loc[frame["SoilType"].isin(config["primary_soil_codes"])].copy()
        exposure_cells["descriptive_environment_group"] = env_group.reindex(exposure_cells["cell_id"]).to_numpy()
        for outcome in config["outcomes"]:
            complete = complete_mask(exposure_cells, [exposure, outcome, *v5], config["primary_soil_codes"])
            cell_values = exposure_cells.loc[complete].copy()
            groups = cell_values.groupby("descriptive_environment_group", sort=True, dropna=False)
            for group_value, cells in groups:
                y = pd.to_numeric(cells[outcome], errors="coerce").to_numpy(dtype=float)
                a = pd.to_numeric(cells[exposure], errors="coerce").to_numpy(dtype=float)
                log_a = np.log1p(a)
                slope = np.nan
                r_squared = np.nan
                ci_lower = np.nan
                ci_upper = np.nan
                p_value = np.nan
                if len(cells) >= 5 and np.std(log_a, ddof=1) > 0 and np.std(y, ddof=1) > 0:
                    model = sm.OLS(y, sm.add_constant(log_a)).fit()
                    slope = float(model.params[1])
                    ci_lower, ci_upper = [float(value) for value in model.conf_int()[1]]
                    p_value = float(model.pvalues[1])
                    r_squared = float(model.rsquared)
                high_rows.append({
                    "exposure": exposure,
                    "outcome": outcome,
                    "environment_group": group_value,
                    "environment_group_definition": "soil-specific V5 K-means k=6; fitted unsupervised on full sample for descriptive grouping",
                    "n_cells": int(len(cells)),
                    "mean_outcome": float(np.mean(y)) if len(y) else np.nan,
                    "median_outcome": float(np.median(y)) if len(y) else np.nan,
                    "mean_Aa": float(np.mean(a)) if len(a) else np.nan,
                    "median_Aa": float(np.median(a)) if len(a) else np.nan,
                    "cell_level_beta_log1p_Aa": slope,
                    "cell_level_r_squared_apparent": r_squared,
                    "cell_level_ci_lower": ci_lower,
                    "cell_level_ci_upper": ci_upper,
                    "cell_level_p_value": p_value,
                    "outcome_guided_selection": True,
                    "outcome_guided_group_ranking": True,
                    "selection_metric": "full_sample_mean_outcome",
                    "selection_objective": "rank groups by descending mean outcome; retain the top three per exposure and outcome",
                    "selection_rule": "rank prespecified environmental groups by full-sample mean outcome",
                })

            pair_group = pair_table.loc[
                pair_table["method"].eq("M4") & pair_table["exposure"].eq(exposure)
            ].copy()
            if pair_group.empty or f"delta_{outcome}" not in pair_group:
                continue
            pair_group["descriptive_environment_group"] = pd.to_numeric(
                pair_group["descriptive_environment_group"], errors="coerce"
            )
            specifications: list[tuple[str, pd.Series, str]] = [
                ("soil_type", pair_group["soil_type"], "Exact soil type"),
                ("V5_kmeans_group", pair_group["descriptive_environment_group"], "Soil-specific V5 K-means k=6"),
            ]
            for variable in ["Elevation", "TWI", "LST", "NDWI"]:
                context_col = f"mean_{variable}"
                if context_col not in pair_group:
                    continue
                context = pd.to_numeric(pair_group[context_col], errors="coerce")
                if context.notna().sum() < 30 or context.nunique() < 3:
                    continue
                try:
                    bins = pd.qcut(context, q=3, duplicates="drop")
                except ValueError:
                    continue
                specifications.append((f"{variable}_tertile", bins.astype(str), f"Pair midpoint {variable} tertile"))
            for subgroup_name, subgroup_values, definition in specifications:
                data = pd.DataFrame({
                    "dx": pd.to_numeric(pair_group["delta_log1p_erosion"], errors="coerce"),
                    "dy": pd.to_numeric(pair_group[f"delta_{outcome}"], errors="coerce"),
                    "group": subgroup_values,
                }, index=pair_group.index).replace([np.inf, -np.inf], np.nan).dropna()
                for group_value, subset in data.groupby("group", sort=True):
                    if len(subset) < 20 or subset["dx"].std(ddof=1) == 0 or subset["dy"].std(ddof=1) == 0:
                        continue
                    fit = sm.OLS(subset["dy"], sm.add_constant(subset["dx"])).fit()
                    ci = fit.conf_int().iloc[1]
                    rows.append({
                        "method": "M4", "exposure": exposure, "outcome": outcome,
                        "evaluation_set": "full_sample_descriptive_apparent",
                        "subgroup_variable": subgroup_name, "subgroup_definition": definition,
                        "subgroup": str(group_value), "n_pairs": int(len(subset)),
                        "mean_delta_log1p_Aa": float(subset["dx"].mean()),
                        "mean_delta_outcome": float(subset["dy"].mean()),
                        "coefficient": float(fit.params.iloc[1]),
                        "standardized_coefficient": float(fit.params.iloc[1] * subset["dx"].std(ddof=1) / subset["dy"].std(ddof=1)),
                        "ci_lower": float(ci.iloc[0]), "ci_upper": float(ci.iloc[1]),
                        "p_value": float(fit.pvalues.iloc[1]), "r_squared_apparent": float(fit.rsquared),
                        "positive_delta_outcome_fraction": float((subset["dy"] > 0).mean()),
                    })
                groups_present = data["group"].nunique()
                interaction_p = np.nan
                if groups_present >= 2 and len(data) > 2 * groups_present + 5:
                    dummy = pd.get_dummies(data["group"].astype(str), drop_first=True, dtype=float)
                    reduced = sm.add_constant(pd.concat([data[["dx"]], dummy], axis=1), has_constant="add")
                    full = reduced.copy()
                    for col in dummy:
                        full[f"dx_x_{col}"] = data["dx"] * dummy[col]
                    if np.linalg.matrix_rank(full.to_numpy(dtype=float)) == full.shape[1]:
                        fit_reduced = sm.OLS(data["dy"], reduced).fit()
                        fit_full = sm.OLS(data["dy"], full).fit()
                        interaction_p = float(fit_full.compare_f_test(fit_reduced)[1])
                rows.append({
                    "method": "M4", "exposure": exposure, "outcome": outcome,
                    "evaluation_set": "full_sample_descriptive_apparent",
                    "subgroup_variable": subgroup_name, "subgroup_definition": definition,
                    "subgroup": "__interaction_test__", "n_pairs": int(len(data)),
                    "mean_delta_log1p_Aa": float(data["dx"].mean()) if len(data) else np.nan,
                    "mean_delta_outcome": float(data["dy"].mean()) if len(data) else np.nan,
                    "interaction_p_value": interaction_p,
                    "n_groups": int(groups_present),
                    "positive_delta_outcome_fraction": float((data["dy"] > 0).mean()) if len(data) else np.nan,
                })
    if not repeated.empty and "validation_model" in repeated:
        group_cv = repeated.loc[
            repeated["validation_model"].eq("P1_environment_group_stratum")
            & repeated["evaluation_set"].eq("test")
        ].copy()
        if not group_cv.empty:
            for (method, exposure, outcome, group_value), subset in group_cv.groupby(
                ["method", "exposure", "outcome", "environment_group"], sort=True
            ):
                betas = pd.to_numeric(subset["coefficient"], errors="coerce").dropna()
                r2 = pd.to_numeric(subset["r_squared"], errors="coerce").dropna()
                if betas.empty:
                    continue
                rows.append({
                    "method": method, "exposure": exposure, "outcome": outcome,
                    "evaluation_set": "repeated_spatial_test_10_seed_summary",
                    "subgroup_variable": "V5_kmeans_environment_group",
                    "subgroup_definition": "Training-fitted soil-specific V5 K-means k=6; group-wise test-pair P1 estimate",
                    "subgroup": int(group_value), "n_pairs": int(pd.to_numeric(subset["n_score_pairs"], errors="coerce").median()),
                    "coefficient": float(betas.median()),
                    "ci_lower": float(pd.to_numeric(subset["ci_lower"], errors="coerce").median()),
                    "ci_upper": float(pd.to_numeric(subset["ci_upper"], errors="coerce").median()),
                    "r_squared_apparent": float(r2.median()) if len(r2) else np.nan,
                    "n_spatial_seeds": int(subset["seed"].nunique()),
                    "positive_direction_fraction": float((betas > 0).mean()),
                    "negative_direction_fraction": float((betas < 0).mean()),
                    "direction_stability": max(float((betas > 0).mean()), float((betas < 0).mean())),
                })
    heterogeneity = pd.DataFrame.from_records(rows)
    if "interaction_p_value" in heterogeneity:
        mask = heterogeneity["subgroup"].eq("__interaction_test__")
        heterogeneity.loc[mask, "interaction_p_value_fdr"] = adjusted_p_values(heterogeneity.loc[mask, "interaction_p_value"])
    high = pd.DataFrame.from_records(high_rows)
    if not high.empty:
        high["soil_type"] = (pd.to_numeric(high["environment_group"], errors="coerce") // 100).astype("Int64")
        high["cluster_within_soil"] = (pd.to_numeric(high["environment_group"], errors="coerce") % 100).astype("Int64")
        high["environment_group_label"] = "soil_" + high["soil_type"].astype(str) + "_k6_" + high["cluster_within_soil"].astype(str)
        high["mean_outcome_rank"] = high.groupby(["exposure", "outcome"])["mean_outcome"].rank(method="min", ascending=False)
        high["top_three_high_response_group"] = high["mean_outcome_rank"].le(3)
        selection_keys = ["exposure", "outcome"]
        high["selection_trial_count"] = high.groupby(selection_keys)["environment_group"].transform("size")
        high["selection_selected_group_count"] = high.groupby(selection_keys)["top_three_high_response_group"].transform("sum")
        high["selection_source_complete_case_cells"] = high.groupby(selection_keys)["n_cells"].transform("sum")
        selected_cells = high["n_cells"].where(high["top_three_high_response_group"], 0)
        high["selection_retained_cells"] = selected_cells.groupby([high[key] for key in selection_keys]).transform("sum")
        high["group_cell_retention_fraction"] = high["n_cells"] / high["selection_source_complete_case_cells"]
        high["selection_retained_cell_fraction"] = high["selection_retained_cells"] / high["selection_source_complete_case_cells"]
    return heterogeneity, high


def full_sample_exposure_comparison(frame: pd.DataFrame, full_pairs: pd.DataFrame, paired_results: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows = []
    variables = config["environment_vectors"]["V5"]
    soils = config["primary_soil_codes"]
    for outcome in config["outcomes"]:
        common_cols = [*variables, *EXPOSURE_COLUMNS, outcome]
        common = frame.loc[complete_mask(frame, common_cols, soils)].copy()
        if len(common) >= 10:
            y = pd.to_numeric(common[outcome], errors="coerce").to_numpy(dtype=float)
            z = StandardScaler().fit_transform(common[variables].to_numpy(dtype=float))
            env_design = sm.add_constant(z, has_constant="add")
            env_fit = sm.OLS(y, env_design).fit()
            env_r2 = float(env_fit.rsquared)
            for exposure in EXPOSURE_COLUMNS:
                x = np.log1p(pd.to_numeric(common[exposure], errors="coerce").to_numpy(dtype=float))
                fit = sm.OLS(y, np.column_stack([env_design, x])).fit()
                rows.append({
                    "comparison_id": f"common_{outcome}_V5_both_exposures",
                    "analysis_type": "cell_level_common_sample_apparent",
                    "same_sample_n": int(len(common)), "outcome": outcome, "exposure": exposure,
                    "method": "M1", "model": "environment_plus_log1p_Aa_ols",
                    "r2_environment_only": env_r2,
                    "r2_environment_plus_Aa": float(fit.rsquared),
                    "delta_r2": float(fit.rsquared - env_r2),
                    "beta_log1p_Aa": float(fit.params[-1]),
                    "direction": "positive" if fit.params[-1] > 0 else "negative" if fit.params[-1] < 0 else "zero",
                    "evaluation_set": "apparent_full_sample",
                    "exposure_periods_comparable": True,
                })
    if not full_pairs.empty:
        p1 = paired_results.loc[paired_results["model"].eq("P1_intercept_ols")]
        for _, result in p1.iterrows():
            rows.append({
                "comparison_id": f"matched_M4_{result['outcome']}_V5",
                "analysis_type": "matched_pair_full_sample_apparent",
                "same_sample_n": result.get("n_pairs"),
                "outcome": result["outcome"], "exposure": result["exposure"],
                "method": "M4", "model": "P1_intercept_ols",
                "r2_environment_only": np.nan,
                "r2_environment_plus_Aa": result.get("r_squared"),
                "delta_r2": np.nan,
                "beta_log1p_Aa": result.get("coefficient"),
                "direction": result.get("direction"),
                "evaluation_set": "apparent_full_sample",
                "exposure_periods_comparable": False,
            })
    return pd.DataFrame.from_records(rows)


def robustness_sensitivities(
    full_pairs: pd.DataFrame, repeated: pd.DataFrame, local_grid: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    primary = full_pairs.loc[full_pairs["method"].eq("M4")].copy()
    rng = np.random.default_rng(int(config["seed"]))
    for (method, exposure), pairs_by_model in primary.groupby(["method", "exposure"], sort=True):
        for outcome in config["outcomes"]:
            if f"delta_{outcome}" not in pairs_by_model:
                continue
            base = pairs_by_model[["delta_log1p_erosion", f"delta_{outcome}"]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if len(base) < 10:
                continue
            order = rng.permutation(len(base))
            for fraction in config["robustness"]["retention_fractions"]:
                subset = base.iloc[order[: max(5, int(round(len(base) * fraction)))]]
                fit = sm.OLS(subset[f"delta_{outcome}"], sm.add_constant(subset["delta_log1p_erosion"])).fit()
                rows.append({
                    "sensitivity": "nested_random_pair_retention", "method": method,
                    "exposure": exposure, "outcome": outcome, "fraction": fraction,
                    "n_pairs": len(subset), "coefficient": float(fit.params.iloc[1]),
                    "r_squared": float(fit.rsquared), "direction": "positive" if fit.params.iloc[1] > 0 else "negative" if fit.params.iloc[1] < 0 else "zero",
                    "selection_scope": "outcome_blind_random_nested_sample",
                })
            raw = pairs_by_model[["erosion_difference", f"delta_{outcome}"]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if len(raw) >= 10:
                fit_raw = sm.OLS(raw[f"delta_{outcome}"], sm.add_constant(raw["erosion_difference"])).fit()
                rows.append({
                    "sensitivity": "raw_Aa_difference", "method": method,
                    "exposure": exposure, "outcome": outcome, "fraction": 1.0,
                    "n_pairs": len(raw), "coefficient": float(fit_raw.params.iloc[1]),
                    "r_squared": float(fit_raw.rsquared), "direction": "positive" if fit_raw.params.iloc[1] > 0 else "negative" if fit_raw.params.iloc[1] < 0 else "zero",
                    "selection_scope": "full_sample_descriptive",
                })
            low_zero = pd.to_numeric(pairs_by_model["erosion_low"], errors="coerce").eq(0)
            high_zero = pd.to_numeric(pairs_by_model["erosion_high"], errors="coerce").eq(0)
            for label, mask in [
                ("zero_positive_pairs", low_zero ^ high_zero),
                ("both_positive_pairs", (~low_zero) & (~high_zero)),
            ]:
                subset = pairs_by_model.loc[mask, ["delta_log1p_erosion", f"delta_{outcome}"]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                if len(subset) < 10:
                    continue
                fit = sm.OLS(subset[f"delta_{outcome}"], sm.add_constant(subset["delta_log1p_erosion"])).fit()
                rows.append({
                    "sensitivity": label, "method": method,
                    "exposure": exposure, "outcome": outcome, "fraction": len(subset) / len(pairs_by_model),
                    "n_pairs": len(subset), "coefficient": float(fit.params.iloc[1]),
                    "r_squared": float(fit.rsquared), "direction": "positive" if fit.params.iloc[1] > 0 else "negative" if fit.params.iloc[1] < 0 else "zero",
                    "selection_scope": "full_sample_descriptive",
                })
            if outcome in PRIMARY_OUTCOMES:
                block_data = pairs_by_model[["spatial_block_id", "delta_log1p_erosion", f"delta_{outcome}"]].copy()
                block_data = block_data.apply(lambda col: pd.to_numeric(col, errors="coerce") if col.name != "spatial_block_id" else col)
                block_data = block_data.replace([np.inf, -np.inf], np.nan).dropna()
                block_means = block_data.groupby("spatial_block_id", sort=True)[["delta_log1p_erosion", f"delta_{outcome}"]].mean()
                if len(block_means) >= 8:
                    observed = np.polyfit(block_means["delta_log1p_erosion"], block_means[f"delta_{outcome}"], 1)[0]
                    null = []
                    for _ in range(int(config["robustness"]["block_permutation_replicates"])):
                        permuted = rng.permutation(block_means["delta_log1p_erosion"].to_numpy(dtype=float))
                        null.append(float(np.polyfit(permuted, block_means[f"delta_{outcome}"], 1)[0]))
                    p_value = (1 + sum(abs(value) >= abs(observed) for value in null)) / (1 + len(null))
                    rows.append({
                        "sensitivity": "spatial_block_mean_permutation",
                        "method": method, "exposure": exposure, "outcome": outcome,
                        "fraction": 1.0, "n_pairs": int(len(block_data)),
                        "n_spatial_blocks": int(len(block_means)), "coefficient": float(observed),
                        "p_value_two_sided": float(p_value),
                        "null_abs_coefficient_p95": float(np.quantile(np.abs(null), .95)),
                        "selection_scope": f"block-level permutation; {len(null)} permutations",
                    })

    # Existing RF search is used only for a predictor-set overlap diagnostic.
    old_models = ROOT / "outputs" / "exploratory_oracle" / "oracle_20261026T041521459615Z_20261026" / "models" / "full_sample_model_search.parquet"
    if old_models.exists():
        models = pd.read_parquet(old_models)
        models = models.loc[models["model_name"].eq("random_forest")]
        for outcome in config["outcomes"]:
            selected = []
            for exposure in EXPOSURE_COLUMNS:
                subset = models.loc[models["exposure"].eq(exposure) & models["outcome"].eq(outcome)].copy()
                subset["r_squared"] = pd.to_numeric(subset["r_squared"], errors="coerce")
                if subset.empty:
                    selected.append(set())
                    continue
                best = subset.sort_values("r_squared", ascending=False).iloc[0]
                value = best.get("model_config")
                if isinstance(value, str):
                    try:
                        value = json.loads(value)
                    except json.JSONDecodeError:
                        value = {}
                features = set(value.get("numeric_features", [])) if isinstance(value, dict) else set()
                selected.append(features - set(EXPOSURE_COLUMNS))
            union = selected[0] | selected[1]
            jaccard = len(selected[0] & selected[1]) / len(union) if union else np.nan
            rows.append({
                "sensitivity": "prior_RF_environment_predictor_overlap",
                "method": "full_sample_best_random_forest",
                "outcome": outcome, "exposure": "Aa_20yr_vs_Aa_2020",
                "n_pairs": np.nan, "jaccard_environment_features": jaccard,
                "Aa_20yr_environment_features": json.dumps(sorted(selected[0])),
                "Aa_2020_environment_features": json.dumps(sorted(selected[1])),
                "selection_scope": "prior full-sample apparent model search; descriptive and outcome-guided",
            })
    return pd.DataFrame.from_records(rows)


def add_overall_max_smd_to_balance(balance: pd.DataFrame) -> pd.DataFrame:
    if balance.empty or not {"diagnostic_type", "stratum_type", "smd_after"}.issubset(balance.columns):
        return balance
    keys = [name for name in ["exposure", "method", "role", "seed"] if name in balance.columns]
    overall = balance.loc[
        balance["diagnostic_type"].eq("variable_balance")
        & balance["stratum_type"].eq("overall")
    ].copy()
    if overall.empty or not keys:
        return balance
    overall["abs_smd_after"] = pd.to_numeric(overall["smd_after"], errors="coerce").abs()
    maximum = overall.groupby(keys, dropna=False)["abs_smd_after"].max()
    summary_mask = balance["diagnostic_type"].eq("summary")
    if not summary_mask.any():
        return balance
    result = balance.copy()
    if "max_abs_matched_smd" not in result:
        result["max_abs_matched_smd"] = np.nan
    summary_index = pd.MultiIndex.from_frame(result.loc[summary_mask, keys])
    derived = maximum.reindex(summary_index).to_numpy(dtype=float)
    existing = pd.to_numeric(result.loc[summary_mask, "max_abs_matched_smd"], errors="coerce").to_numpy(dtype=float)
    result.loc[summary_mask, "max_abs_matched_smd"] = np.where(np.isfinite(derived), derived, existing)
    return result


def summarize_validation(repeated: pd.DataFrame, balance: pd.DataFrame) -> pd.DataFrame:
    if repeated.empty:
        return pd.DataFrame()
    balance = add_overall_max_smd_to_balance(balance)
    eligible = repeated.loc[
        repeated["evaluation_set"].eq("test")
        & repeated["method"].isin(list(MODEL_NAMES))
        & repeated["r_squared"].notna()
    ].copy()
    if "validation_model" in eligible:
        eligible = eligible.loc[eligible["validation_model"].isna() | eligible["validation_model"].eq("P1_intercept_ols")]
    beta_col = "beta_log1p_erosion" if "beta_log1p_erosion" in eligible else "beta_log1p_erosion_train"
    if "beta_log1p_erosion_train" in eligible:
        eligible["beta_for_summary"] = pd.to_numeric(eligible.get("beta_log1p_erosion"), errors="coerce")
        if "beta_log1p_erosion_train" in eligible:
            eligible["beta_for_summary"] = eligible["beta_for_summary"].fillna(
                pd.to_numeric(eligible["beta_log1p_erosion_train"], errors="coerce")
            )
    elif beta_col in eligible:
        eligible["beta_for_summary"] = pd.to_numeric(eligible[beta_col], errors="coerce")
    grouped = eligible.groupby(["exposure", "outcome", "method", "model_name"], dropna=False)
    summary = grouped.agg(
        validation_r2_median=("r_squared", "median"),
        validation_r2_mean=("r_squared", "mean"),
        validation_r2_q25=("r_squared", lambda values: values.quantile(.25)),
        validation_r2_q75=("r_squared", lambda values: values.quantile(.75)),
        rmse_median=("rmse", "median"),
        mae_median=("mae", "median"),
        n_seed_folds=("seed", "nunique"),
        n_score_median=("n_score", "median"),
    ).reset_index()
    if "beta_for_summary" in eligible:
        signs = eligible.assign(
            positive=eligible["beta_for_summary"].gt(0), negative=eligible["beta_for_summary"].lt(0)
        ).groupby(["exposure", "outcome", "method"], dropna=False).agg(
            positive_direction_fraction=("positive", "mean"),
            negative_direction_fraction=("negative", "mean"),
        ).reset_index()
        summary = summary.merge(signs, on=["exposure", "outcome", "method"], how="left")
    if "n_score_pairs" in eligible:
        summary_counts = eligible.assign(
            score_count=pd.to_numeric(eligible.get("n_score"), errors="coerce").fillna(
                pd.to_numeric(eligible["n_score_pairs"], errors="coerce")
            )
        ).groupby(["exposure", "outcome", "method", "model_name"], dropna=False)["score_count"].median()
        summary["n_score_median"] = summary.set_index(
            ["exposure", "outcome", "method", "model_name"]
        ).index.map(summary_counts)
    if not balance.empty:
        summaries = balance.loc[
            balance["diagnostic_type"].eq("summary") & balance["role"].eq("test")
        ].copy()
        if not summaries.empty:
            matching = summaries.groupby(["exposure", "method"], dropna=False).agg(
                median_test_coverage=("area_coverage", "median"),
                median_test_max_abs_smd=("max_abs_matched_smd", "median"),
                median_test_environment_distance=("environment_distance_median", "median"),
                median_test_geographic_distance_m=("geographic_distance_median_m", "median"),
                median_test_abs_log1p_Aa_gap=("median_abs_log1p_Aa_difference", "median"),
                n_test_matched_pairs=("n_pairs", "median"),
            ).reset_index()
            summary = summary.merge(matching, on=["exposure", "method"], how="left")
    return summary


def build_matching_method_comparison(
    final_models: pd.DataFrame, local_grid: pd.DataFrame, full_pairs: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    if not final_models.empty:
        for _, record in final_models.iterrows():
            rows.append({
                "analysis_type": "10_seed_spatial_test_model_comparison",
                "exposure": record.get("exposure"), "outcome": record.get("outcome"),
                "method": record.get("method"), "method_name": record.get("model_name"),
                "median_test_r2": record.get("validation_r2_median"),
                "test_r2_q25": record.get("validation_r2_q25"),
                "test_r2_q75": record.get("validation_r2_q75"),
                "test_rmse_median": record.get("rmse_median"),
                "direction_consistency": max(
                    [v for v in [record.get("positive_direction_fraction"), record.get("negative_direction_fraction")] if pd.notna(v)],
                    default=np.nan,
                ),
                "matched_pair_coverage_median": record.get("median_test_coverage"),
                "max_abs_smd_median": record.get("median_test_max_abs_smd"),
                "median_environment_distance": record.get("median_test_environment_distance"),
                "median_geographic_distance_m": record.get("median_test_geographic_distance_m"),
                "median_abs_log1p_Aa_gap": record.get("median_test_abs_log1p_Aa_gap"),
                "selection_basis": "prespecified split; summarize all 10 internal spatial seeds",
            })
    if not local_grid.empty:
        for _, record in local_grid.iterrows():
            rows.append({
                "analysis_type": "outcome_blind_local_matching_grid",
                "exposure": record.get("exposure"), "outcome": None,
                "method": "M4" if record.get("cluster_method") == "none" else "M3",
                "method_name": "Direct nearest-neighbor" if record.get("cluster_method") == "none" else f"K-means k={record.get('cluster_count_per_soil')}",
                "candidate_id": record.get("candidate_id"),
                "environment_vector": record.get("environment_vector"),
                "environment_caliper": record.get("environment_caliper"),
                "geographic_radius_m": record.get("geographic_radius_m"),
                "erosion_rank_gap": record.get("erosion_rank_gap"),
                "matched_pair_coverage_median": record.get("area_coverage"),
                "max_abs_smd_median": record.get("max_abs_matched_smd"),
                "median_environment_distance": record.get("environment_distance_median"),
                "median_geographic_distance_m": record.get("geographic_distance_median_m"),
                "median_abs_log1p_Aa_gap": record.get("median_abs_log1p_Aa_difference"),
                "outcome_blind_rank": record.get("outcome_blind_rank"),
                "pareto_candidate": record.get("pareto_candidate"),
                "selection_basis": "outcome-blind rank of coverage, balance, distance, and exposure separation",
            })
    return pd.DataFrame.from_records(rows)


def build_exposure_comparison(
    full_comparison: pd.DataFrame,
    repeated: pd.DataFrame,
    increments: pd.DataFrame,
    paired_results: pd.DataFrame,
) -> pd.DataFrame:
    rows = full_comparison.to_dict(orient="records") if not full_comparison.empty else []
    if not increments.empty:
        cell_test = increments.loc[
            increments["analysis_type"].eq("cell_level_nested_ols") & increments["evaluation_set"].eq("test")
        ]
        for (exposure, outcome), group in cell_test.groupby(["exposure", "outcome"], sort=True):
            values = pd.to_numeric(group["delta_r2"], errors="coerce").dropna()
            betas = pd.to_numeric(group["beta_log1p_Aa"], errors="coerce").dropna()
            rows.append({
                "comparison_id": f"common_{outcome}_V5_10seed_test",
                "analysis_type": "cell_level_common_sample_10_seed_summary",
                "same_sample_n": int(group["same_sample_n"].median()) if len(group) else np.nan,
                "outcome": outcome, "exposure": exposure, "method": "M1",
                "model": "environment_plus_log1p_Aa_ols",
                "delta_r2_median": float(values.median()) if len(values) else np.nan,
                "delta_r2_q25": float(values.quantile(.25)) if len(values) else np.nan,
                "delta_r2_q75": float(values.quantile(.75)) if len(values) else np.nan,
                "delta_r2_positive_fraction": float((values > 0).mean()) if len(values) else np.nan,
                "beta_log1p_Aa_median": float(betas.median()) if len(betas) else np.nan,
                "beta_direction_consistency": max(float((betas > 0).mean()), float((betas < 0).mean())) if len(betas) else np.nan,
                "n_spatial_seeds": int(group["seed"].nunique()),
                "evaluation_set": "repeated_5km_spatial_test",
                "exposure_periods_comparable": True,
            })
        pair_test = repeated.loc[
            repeated["method"].isin(["M3", "M4"])
            & repeated["evaluation_set"].eq("test")
            & repeated.get("validation_model", pd.Series(index=repeated.index, dtype=object)).eq("P1_intercept_ols")
        ] if not repeated.empty else pd.DataFrame()
        for (method, exposure, outcome), group in pair_test.groupby(["method", "exposure", "outcome"], sort=True):
            values = pd.to_numeric(group["r_squared"], errors="coerce").dropna()
            betas = pd.to_numeric(group["beta_log1p_erosion_train"], errors="coerce").dropna()
            rows.append({
                "comparison_id": f"{method}_{outcome}_paired_10seed_test",
                "analysis_type": "matched_pair_10_seed_test_summary",
                "same_sample_n": int(group["n_score_pairs"].median()) if len(group) else np.nan,
                "outcome": outcome, "exposure": exposure, "method": method,
                "model": "P1_intercept_ols",
                "delta_r2_median": float(values.median()) if len(values) else np.nan,
                "delta_r2_q25": float(values.quantile(.25)) if len(values) else np.nan,
                "delta_r2_q75": float(values.quantile(.75)) if len(values) else np.nan,
                "beta_log1p_Aa_median": float(betas.median()) if len(betas) else np.nan,
                "beta_direction_consistency": max(float((betas > 0).mean()), float((betas < 0).mean())) if len(betas) else np.nan,
                "n_spatial_seeds": int(group["seed"].nunique()),
                "evaluation_set": "repeated_5km_spatial_test",
                "exposure_periods_comparable": False,
            })
    return pd.DataFrame.from_records(rows)


def curve_data() -> pd.DataFrame:
    old_root = ROOT / "outputs" / "exploratory_oracle" / "oracle_20261026T041521459615Z_20261026"
    plot_path = old_root / "diagnostics" / "plot_data.parquet"
    full_curve_path = old_root / "models" / "selected_full_sample_curves.parquet"
    if not plot_path.exists() or not full_curve_path.exists():
        return pd.DataFrame()
    plot_sample = pd.read_parquet(plot_path)
    model_curves = pd.read_parquet(full_curve_path)
    results: list[pd.DataFrame] = []
    outcomes = ["SHI_candidate", "AN", "OM", "CEC", "pH", "SW", "BD", "MBC", "RESP"]
    exposure = "Aa_20yr"
    for outcome in outcomes:
        if exposure not in plot_sample or outcome not in plot_sample:
            continue
        data = plot_sample[[exposure, outcome]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(data) < 10:
            continue
        data = data.rename(columns={exposure: "exposure_value", outcome: "predicted_outcome"})
        data["outcome"] = outcome
        data["curve_type"] = "fixed_plot_sample_scatter"
        results.append(data.copy())
        grid = np.unique(np.quantile(data["exposure_value"], np.linspace(.01, .99, 100)))
        n_unique = np.unique(data["exposure_value"]).size
        if len(grid) > 1 and n_unique >= 4:
            design, grid_design = natural_spline_design(data["exposure_value"].to_numpy(dtype=float), grid, df=5)
            fit = sm.OLS(data["predicted_outcome"].to_numpy(dtype=float), sm.add_constant(design, has_constant="add")).fit()
            pred = fit.predict(sm.add_constant(grid_design, has_constant="add"))
            results.append(pd.DataFrame({
                "outcome": outcome, "exposure_value": grid, "predicted_outcome": pred,
                "curve_type": "natural_spline_fixed_plot_sample",
            }))
        curves = model_curves.loc[model_curves["outcome"].eq(outcome)].copy()
        if not curves.empty:
            results.append(pd.DataFrame({
                "outcome": outcome,
                "exposure_value": pd.to_numeric(curves["exposure_value"], errors="coerce"),
                "predicted_outcome": pd.to_numeric(curves["predicted_outcome"], errors="coerce"),
                "curve_type": "selected_full_sample_apparent_model_partial_dependence",
            }))
    return pd.concat(results, ignore_index=True, sort=False) if results else pd.DataFrame()


def spline_necessity(repeated: pd.DataFrame) -> pd.DataFrame:
    if repeated.empty or "validation_model" not in repeated:
        return pd.DataFrame()
    test = repeated.loc[repeated["evaluation_set"].eq("test") & repeated["outcome"].isin(PRIMARY_OUTCOMES)].copy()
    linear = test.loc[test["validation_model"].eq("P1_intercept_ols"), ["seed", "method", "exposure", "outcome", "r_squared"]].rename(columns={"r_squared": "linear_test_r2"})
    spline = test.loc[test["validation_model"].eq("P5_natural_spline_diagnostic"), ["seed", "method", "exposure", "outcome", "r_squared"]].rename(columns={"r_squared": "spline_test_r2"})
    merged = linear.merge(spline, on=["seed", "method", "exposure", "outcome"], how="inner")
    if merged.empty:
        return pd.DataFrame()
    merged["spline_minus_linear_test_r2"] = merged["spline_test_r2"] - merged["linear_test_r2"]
    summary = merged.groupby(["method", "exposure", "outcome"], as_index=False).agg(
        median_spline_minus_linear_r2=("spline_minus_linear_test_r2", "median"),
        positive_improvement_fraction=("spline_minus_linear_test_r2", lambda values: float((values > 0).mean())),
        n_seeds=("seed", "nunique"),
    )
    summary["p5_indicated"] = (
        summary["median_spline_minus_linear_r2"].ge(0.02)
        & summary["positive_improvement_fraction"].ge(0.70)
        & summary["n_seeds"].ge(8)
    )
    return summary


def fit_full_paired_spline(pairs: pd.DataFrame, outcome: str) -> dict[str, Any]:
    data = pairs[["delta_log1p_erosion", f"delta_{outcome}"]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 10 or data["delta_log1p_erosion"].nunique() < 4:
        return {"status": "insufficient_sample", "n_pairs": len(data)}
    x = data["delta_log1p_erosion"].to_numpy(dtype=float)[:, None]
    y = data[f"delta_{outcome}"].to_numpy(dtype=float)
    design, _ = natural_spline_design(x[:, 0], x[:, 0], df=4)
    fit = sm.OLS(y, sm.add_constant(design, has_constant="add")).fit()
    return {
        "status": "ok", "n_pairs": len(data), "coefficient": np.nan,
        "standardized_coefficient": np.nan, "intercept": np.nan,
        "ci_lower": np.nan, "ci_upper": np.nan, "p_value": np.nan,
        "r_squared": float(fit.rsquared), "adjusted_r_squared": float(fit.rsquared_adj),
        "rmse": float(np.sqrt(np.mean(fit.resid ** 2))), "mae": float(np.mean(np.abs(fit.resid))),
        "direction": "nonlinear_curve", "model_parameters": {
            "basis": "natural cubic regression spline (Patsy cr basis)", "degrees_freedom": 4,
            "degree": 3, "exposure": "delta_log1p_erosion",
        },
    }


def full_sample_increment_rows(full_comparison: pd.DataFrame) -> pd.DataFrame:
    data = full_comparison.loc[full_comparison["analysis_type"].eq("cell_level_common_sample_apparent")].copy() if not full_comparison.empty else pd.DataFrame()
    if data.empty:
        return pd.DataFrame()
    return pd.DataFrame({
        "analysis_type": "cell_level_nested_ols_apparent",
        "seed": np.nan,
        "block_size_m": np.nan,
        "outcome": data["outcome"],
        "exposure": data["exposure"],
        "evaluation_set": "apparent_full_sample",
        "model_1": "environment_only_ols",
        "model_2": "environment_plus_log1p_Aa_ols",
        "same_sample_n": data["same_sample_n"],
        "training_n": data["same_sample_n"],
        "r2_environment_only": data["r2_environment_only"],
        "r2_environment_plus_Aa": data["r2_environment_plus_Aa"],
        "delta_r2": data["delta_r2"],
        "beta_log1p_Aa": data["beta_log1p_Aa"],
        "direction": data["direction"],
        "status": "ok",
    })


def summarize_increment_r2(increments: pd.DataFrame) -> pd.DataFrame:
    """Summarize repeated test-fold ΔR² without mixing apparent/full-sample rows."""
    if increments.empty:
        return pd.DataFrame()
    test = increments.loc[
        increments["evaluation_set"].eq("test")
        & increments["status"].eq("ok")
        & increments["seed"].notna()
    ].copy()
    if test.empty:
        return pd.DataFrame()
    group_columns = [
        column for column in ["analysis_type", "exposure", "outcome", "method", "model_1", "model_2"]
        if column in test
    ]
    rows = []
    for keys, group in test.groupby(group_columns, dropna=False, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_columns, keys, strict=True))
        delta = pd.to_numeric(group["delta_r2"], errors="coerce").dropna()
        beta = pd.to_numeric(group.get("beta_log1p_Aa"), errors="coerce").dropna()
        row.update({
            "evaluation_set": "repeated_5km_spatial_test_summary",
            "n_spatial_seeds": int(group["seed"].nunique()),
            "median_same_sample_n": float(pd.to_numeric(group["same_sample_n"], errors="coerce").median()),
            "median_delta_r2": float(delta.median()) if len(delta) else np.nan,
            "q25_delta_r2": float(delta.quantile(.25)) if len(delta) else np.nan,
            "q75_delta_r2": float(delta.quantile(.75)) if len(delta) else np.nan,
            "positive_delta_r2_fraction": float((delta > 0).mean()) if len(delta) else np.nan,
            "median_beta_log1p_Aa": float(beta.median()) if len(beta) else np.nan,
            "beta_direction_consistency": max(float((beta > 0).mean()), float((beta < 0).mean())) if len(beta) else np.nan,
        })
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def report_text(
    *,
    run_id: str,
    baseline_sha: str,
    provenance_record: dict[str, Any],
    local_grid: pd.DataFrame,
    paired_results: pd.DataFrame,
    repeated: pd.DataFrame,
    increments: pd.DataFrame,
    heterogeneity: pd.DataFrame,
    high_groups: pd.DataFrame,
    final_models: pd.DataFrame,
    matching_balance: pd.DataFrame,
    p5_summary: pd.DataFrame,
    figure_index: list[dict[str, Any]],
) -> str:
    def fmt(value: Any, digits: int = 3) -> str:
        try:
            if pd.isna(value):
                return "NA"
            return f"{float(value):.{digits}f}"
        except (TypeError, ValueError):
            return str(value)

    def top_rows(table: pd.DataFrame, columns: list[str], n: int = 8) -> str:
        if table.empty:
            return "No estimable rows."
        usable = [name for name in columns if name in table]
        subset = table[usable].head(n)
        headers = [str(name).replace("|", "\\|") for name in usable]
        lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
        for _, row in subset.iterrows():
            values = [fmt(value).replace("|", "\\|") for value in row.tolist()]
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    main_p1 = paired_results.loc[
        paired_results["model"].eq("P1_intercept_ols")
        & paired_results["method"].eq("M4")
        & paired_results["exposure"].eq("Aa_20yr")
        & paired_results["outcome"].isin(PRIMARY_OUTCOMES)
    ].copy()
    main_validation = final_models.loc[
        final_models["exposure"].eq("Aa_20yr") & final_models["outcome"].isin(PRIMARY_OUTCOMES)
    ].copy() if not final_models.empty else pd.DataFrame()
    delta_summary = summarize_increment_r2(increments)
    test_bal = matching_balance.loc[
        matching_balance.get("diagnostic_type", pd.Series(index=matching_balance.index, dtype=object)).eq("summary")
        & matching_balance.get("role", pd.Series(index=matching_balance.index, dtype=object)).eq("test")
    ] if not matching_balance.empty else pd.DataFrame()
    local_best = local_grid.sort_values("outcome_blind_rank").head(5) if not local_grid.empty and "outcome_blind_rank" in local_grid else pd.DataFrame()
    hetero_interactions = heterogeneity.loc[heterogeneity.get("subgroup", pd.Series(index=heterogeneity.index, dtype=object)).eq("__interaction_test__")].copy() if not heterogeneity.empty else pd.DataFrame()
    sig_hetero = int(pd.to_numeric(hetero_interactions.get("interaction_p_value_fdr"), errors="coerce").lt(.05).sum()) if not hetero_interactions.empty else 0
    top_groups = high_groups.loc[high_groups.get("top_three_high_response_group", pd.Series(index=high_groups.index, dtype=bool)).fillna(False)] if not high_groups.empty else pd.DataFrame()
    high_selection_audit = provenance_record.get("outcome_guided_selection_audit", {}).get("high_response_group_ranking", {})
    inherited_curve_audit = provenance_record.get("outcome_guided_selection_audit", {}).get("inherited_full_sample_model_curves", {})

    method_summary = {}
    for method in ["M3", "M4"]:
        subset = main_validation.loc[main_validation["method"].eq(method)] if not main_validation.empty else pd.DataFrame()
        method_summary[method] = {
            "median_test_r2": float(pd.to_numeric(subset.get("validation_r2_median"), errors="coerce").median()) if not subset.empty else np.nan,
            "median_coverage": float(pd.to_numeric(subset.get("median_test_coverage"), errors="coerce").median()) if not subset.empty else np.nan,
            "median_abs_smd": float(pd.to_numeric(subset.get("median_test_max_abs_smd"), errors="coerce").median()) if not subset.empty else np.nan,
        }
    if (
        np.isfinite(method_summary["M4"]["median_abs_smd"])
        and method_summary["M4"]["median_abs_smd"] <= .10
        and method_summary["M4"]["median_coverage"] >= .8 * method_summary["M3"]["median_coverage"]
        and method_summary["M4"]["median_test_r2"] >= method_summary["M3"]["median_test_r2"]
    ):
        method_decision = "M4 is the stronger provisional candidate under the joint balance, coverage, and spatial-test rule; it still requires external review and cannot be called a formal optimum."
    else:
        method_decision = "The evidence does not meet the joint rule for dropping either matcher; retain M3 and M4 as parallel exploratory candidates."

    figures_md = "\n".join(
        f"- [{item['name']} PNG](phase-f/{item['formats'][0]}) · [SVG](phase-f/{item['formats'][1]}) · [PDF](phase-f/{item['formats'][2]}) — {item['caption']}"
        for item in figure_index
    )
    return f"""# Phase F research report

**Run:** `{run_id}`  
**Mode:** `EXPLORATORY_ORACLE`  
**Baseline commit:** `{baseline_sha}` (Phase F files remain uncommitted in this workspace)  
**Input run:** `{provenance_record['input_run_id']}`  
**Charter:** `docs/RESEARCH_CHARTER.md` v1.1; unchanged.  

## 1. Research objectives

Phase F compares direct regression, environmental adjustment, environment-group slopes, K-means matching, and direct nearest-neighbor matching. It corrects the previous pooled matched-cell regression by fitting high-minus-low pair differences, repeats whole-block spatial evaluation across ten seeds, estimates erosion's incremental predictive value, checks environmental heterogeneity, and compares the two precomputed exposure periods.

All results remain exploratory. The input outcomes are spatial prediction surfaces; no causal effect or independent field-observation validation is claimed.

## 2. Inherited work and input boundary

The analysis reuses the aligned 392,527-row databases from `{provenance_record['input_run_id']}`. `Aa_20yr` and `Aa_2020` are used as already-computed values. RUSLE was not recomputed or factor-validated, PCNM was not run, and the charter was not changed. Input SHA-256 digests, schemas, versions, and row counts are in `phase-f/tables/provenance.json` and the run directory.

The previous Oracle run searched outcome-guided model and subset choices. Its selected curves and subgroups are kept separate and explicitly marked apparent/outcome-guided. Phase F's repeated spatial estimates are internal repeated block holdouts, not external validation.

## 3. Environment matching design

The M4 reference is the prior V5 design (Elevation and LST, direct nearest neighbor, no K-means, caliper 0.35, 1,000m radius, within-stratum exposure percentile gap 0.50, no replacement). M3 uses the same vector and thresholds with soil-specific K-means k=6. The local Aa_20yr search evaluates 243 outcome-blind combinations across V1/V5/V7, calipers 0.25/0.35/0.50, radii 300/500/1,000m, rank gaps 0.30/0.40/0.50, and no clustering/k=6/k=8. Search results rank coverage, balance, distance, and exposure separation; no health outcome enters matching selection.

The local grid uses `{int(local_grid.get('eligible_cells', pd.Series(dtype=float)).max()) if not local_grid.empty else 0:,}` matched-eligible rows per sampled search pool where reported. Final repeated validation fits the environmental scaler and K-means labels on each training role, then constructs train, validation and test pairs independently under frozen thresholds.

## 4. Matching before and after diagnostics

The local search contains {len(local_grid):,} candidate configurations; {int(local_grid.get('pareto_candidate', pd.Series(dtype=bool)).fillna(False).sum()) if not local_grid.empty else 0} are Pareto candidates under the four recorded design criteria. The CV balance table contains {len(test_bal):,} test-role configuration summaries. Pair-level environmental distance, geographic distance, overall and stratum SMD, per-variable absolute differences, actual Aa gaps, zero-positive pairs, and failure reasons are available in `phase-f/tables/matching_balance_diagnostics.csv`.

Outcome-blind local leaders:

{top_rows(local_best, ['candidate_id','environment_vector','cluster_method','cluster_count_per_soil','environment_caliper','geographic_radius_m','erosion_rank_gap','area_coverage','max_abs_matched_smd','environment_distance_median','median_abs_log1p_Aa_difference','outcome_blind_rank'], 5)}

## 5. Matched-pair difference models

For each fixed full-sample M3/M4 configuration, the primary exposure contrast is `Δlog1p(Aa) = log1p(Aa_high) − log1p(Aa_low)` and the response is `ΔY = Y_high − Y_low`. P1 uses an intercept; P2 omits the intercept as an assumption sensitivity; P3 adjusts for paired Elevation and LST differences; P4 uses Huber robust regression. Tables include coefficient, standardized coefficient, 95% interval, p-value, paired R², RMSE/MAE, difference distribution, zero-positive pairing, and n.

M4 Aa_20yr primary outcomes (full-sample descriptive estimates):

{top_rows(main_p1, ['outcome','coefficient','ci_lower','ci_upper','standardized_coefficient','r_squared','n_pairs','mean_delta_outcome','positive_delta_outcome_fraction'], 9)}

These full-sample confidence intervals are model-based and do not account for residual spatial dependence. Repeated spatial test results are reported separately below.

## 6. SHI, OM, and SW

The three priority indicators are compared using the same P1 definition across M3 and M4, both Aa periods, and the same repeated 5km spatial seeds. The table below reports test R² medians and interquartile ranges; no indicator is selected solely by its largest R².

{top_rows(main_validation, ['outcome','method','validation_r2_median','validation_r2_q25','validation_r2_q75','positive_direction_fraction','negative_direction_fraction','median_test_coverage','median_test_max_abs_smd','n_seed_folds'], 18)}

## 7. Other six health indicators

AN, CEC, pH, BD, MBC, and RESP are included in all cell-level and matched-pair analyses. Inspect `paired_difference_results.csv`, `repeated_spatial_validation.csv`, and `final_model_comparison.csv` for per-indicator coefficients, uncertainty, sample sizes, and spatial test metrics. Because outcomes are prediction surfaces, these results describe associations among aligned spatial predictions.

## 8. Repeated spatial validation

There are {len(config_seed_list(provenance_record))} deterministic 5km whole-block seeds. Existing grid coordinates define the folds; M3/M4 matching is performed separately within each role, and pair endpoints are checked against the role assignment. M0–M2 use the same complete-case pixels for both Aa periods; M3/M4 use role-specific pairs. Validation and test are reported independently. These are repeated internal spatial holdouts after prior data exploration, not external validation.

## 9. Incremental R² of erosion exposure

The cell-level comparison uses the same V5 covariates, training/test pixels, OLS family, and spatial split for environment-only versus environment-plus-log1p(Aa). Matched-pair comparisons similarly add `Δlog1p(Aa)` to paired environment differences on the same pairs. Test-fold summaries below keep cell-level and matched-pair estimands separate and report medians, interquartile ranges, positive-fold fractions, coefficient-direction consistency, and seed counts:

{top_rows(delta_summary, ['analysis_type','method','exposure','outcome','median_delta_r2','q25_delta_r2','q75_delta_r2','positive_delta_r2_fraction','beta_direction_consistency','n_spatial_seeds'], 36)}

The seed-level records are in `erosion_increment_r2.csv`; these summaries are also in `erosion_increment_r2_summary.csv`. Positive ΔR² is predictive increment under this design; it does not establish an independent causal contribution.

## 10. Environmental heterogeneity

Pair slopes are summarized by exact soil type, soil-specific V5 K-means groups, and pair-midpoint Elevation/TWI/LST/NDWI tertiles where sample support permits. Interaction tests compare group-specific slopes; Benjamini–Hochberg adjusted p-values are reported for the interaction-test family. {sig_hetero} interaction tests have FDR-adjusted p<0.05 among estimable comparisons. Environmental groups use unsupervised full-sample fitting only for descriptive subgroup summaries; fold models use training-only transforms.

## 11. High-response environmental groups

This Oracle run is explicitly marked `outcome_guided_selection=true`. The group screen uses full-sample mean outcome and ranks groups from highest to lowest, retaining the top three in each exposure-by-outcome family. It considered {high_selection_audit.get('candidate_trial_count', 0)} group candidates across {high_selection_audit.get('selection_family_count', 0)} families and retained {high_selection_audit.get('selected_group_count', 0)} groups. The selected groups cover {fmt(high_selection_audit.get('cell_retention_fraction_min'), 3)}–{fmt(high_selection_audit.get('cell_retention_fraction_max'), 3)} of eligible complete-case cells per family (median {fmt(high_selection_audit.get('cell_retention_fraction_median'), 3)}). `high_response_environment_groups.csv` records the selection metric, objective, candidate count, group and selected-set cell retention, and complete-case denominator on each row. These are outcome-ranked descriptive groups; matched-pair subgroup estimates remain in `environmental_heterogeneity.csv`.

{top_rows(top_groups, ['exposure','outcome','environment_group','n_cells','mean_outcome','median_outcome','mean_Aa','cell_level_beta_log1p_Aa','cell_level_r_squared_apparent'], 12)}

## 12. Aa_2020 comparison

Aa_2020 is the secondary exposure period. The cell-level comparison requires both exposure values, V5, and the outcome on the same pixels, with identical model forms and spatial seeds. M3/M4 pairs are separately generated for each exposure and therefore are not treated as identical matched observations across periods. Units and exact temporal construction remain as documented in the prior audit; they were not inferred here.

## 13. Main method and data limitations

- The health indicators are spatial prediction surfaces rather than independent field observations.
- The full-sample matched-pair fits are descriptive apparent estimates; their OLS intervals do not address spatial dependence.
- Ten 5km splits are internal repeated spatial holdouts following earlier Oracle exploration, not an untouched external test set.
- The local parameter grid is searched for Aa_20yr; the same V5 reference thresholds are applied to Aa_2020 without claiming they are optimal for that exposure.
- V5 comprises Elevation and LST; residual environmental confounding can remain. P3 adjusts only for paired differences in those two variables.
- The inherited Aa units/time metadata limitations remain. This phase does not validate or recalculate RUSLE.
- Figure 8 compares fixed plotting-sample splines with prior outcome-selected full-sample model curves. The inherited green curves were selected by full-sample apparent R² from {inherited_curve_audit.get('selection_trial_count', 'unknown')} model trials across {inherited_curve_audit.get('model_family_count', 'unknown')} model families; outcome-specific complete-case counts and selected models are in `phase-f/figures/data/08_health_response_curves.csv`. Both curve families are descriptive and use different sample scopes.
- Maps plot sampled points and do not imply continuous observed surfaces.

## 14. Provisional method recommendation

{method_decision}

Current median 10-seed M3/M4 test summaries for Aa_20yr priority outcomes:

{top_rows(main_validation.loc[main_validation['method'].isin(['M3','M4'])] if not main_validation.empty else main_validation, ['outcome','method','validation_r2_median','median_test_coverage','median_test_max_abs_smd','median_test_environment_distance','median_test_geographic_distance_m','median_test_abs_log1p_Aa_gap'], 6)}

Continue to prioritize SHI, OM, and SW only if their paired difference direction, held-out performance, ΔR² stability, balance, and subgroup evidence agree. Do not promote a model based only on apparent R².

## 15. Next research actions

1. Obtain ChatGPT and researcher review of the matching and paired-difference estimands at the exact commit that contains the Phase F artifacts.
2. Reconcile reviewer comments with the charter and decide whether to lock a matcher; preserve M3/M4 if no method dominates on balance, coverage, exposure separation, and repeated spatial performance.
3. Trace source, units, and temporal definitions for Aa and the health prediction surfaces before formal inference.
4. If an indicator remains stable, seek independent measured validation or a genuinely new spatial/temporal assessment set before making stronger claims.
5. Do not advance the next formal phase until the review request is answered and the user approves the resulting research decision.

## Required data tables

{chr(10).join(f'- [`{name}`](phase-f/tables/{name})' for name in ['matching_method_comparison.csv','matching_balance_diagnostics.csv','paired_difference_results.csv','repeated_spatial_validation.csv','erosion_increment_r2.csv','environmental_heterogeneity.csv','high_response_environment_groups.csv','exposure_period_comparison.csv','final_model_comparison.csv'])}

## Figures

{figures_md}

Run-level provenance, exact dependencies, input digests, fold metadata, pair-level parquet, local-grid rows, and robustness sensitivities are stored under `outputs/exploratory_oracle/{run_id}/` (Git-ignored). The review directory contains the nine CSVs, figure files, figure data and captions.
"""


def config_seed_list(provenance_record: dict[str, Any]) -> list[int]:
    return [int(seed) for seed in provenance_record.get("spatial_seed_list", [])]


def chatgpt_review_request(
    run_id: str, baseline_sha: str, provenance_record: dict[str, Any], review_dir: Path,
    figure_index: list[dict[str, Any]],
) -> str:
    selection_audit = provenance_record.get("outcome_guided_selection_audit", {})
    high_selection = selection_audit.get("high_response_group_ranking", {})
    curve_selection = selection_audit.get("inherited_full_sample_model_curves", {})
    tables = [
        "matching_method_comparison.csv", "matching_balance_diagnostics.csv", "paired_difference_results.csv",
        "repeated_spatial_validation.csv", "erosion_increment_r2.csv", "environmental_heterogeneity.csv",
        "high_response_environment_groups.csv", "exposure_period_comparison.csv", "final_model_comparison.csv",
    ]
    figure_links = "\n".join(
        f"- `{item['name']}`: PNG `phase-f/{item['formats'][0]}`, SVG `phase-f/{item['formats'][1]}`, PDF `phase-f/{item['formats'][2]}`; data `phase-f/{item['data_path']}`; caption `phase-f/{item['caption_path']}`"
        for item in figure_index
    )
    return f"""# ChatGPT scientific review request — Phase F

请先在基线提交 `{baseline_sha}` 阅读 `docs/RESEARCH_CHARTER.md` 的完整 v1.1 版本，确认 Q1–Q5、正式研究边界和 Oracle 授权；随后再审阅 Phase F 改动和结果。Phase F 当前尚未提交，因为本执行环境的 `.git` 只读；本请求记录基线 SHA `{baseline_sha}`，正式 Phase F review commit SHA 待用户本地提交后由 `git rev-parse HEAD` 补入。请把本报告与提交后的代码/配置一起审阅，不要把基线 SHA 误当作 Phase F 提交 SHA。

## Review manifest

- Phase F run ID: `{run_id}`
- Input run: `{provenance_record['input_run_id']}`
- Mode: `EXPLORATORY_ORACLE`
- `outcome_guided_selection`: `{str(provenance_record.get('outcome_guided_selection', False)).lower()}`
- Charter version: v1.1, unchanged
- Current baseline commit SHA: `{baseline_sha}`
- Phase F commit SHA: pending user commit
- Research report: [`PHASE_F_RESEARCH_REPORT.md`](PHASE_F_RESEARCH_REPORT.md)
- Runner/config: [`scripts/run_phase_f_analysis.py`](../../../scripts/run_phase_f_analysis.py), [`config/phase_f.yaml`](../../../config/phase_f.yaml)
- Run provenance and table schemas: `outputs/exploratory_oracle/{run_id}/provenance.json`, `experiment_summary.json`
- Pair-level data manifest: `outputs/exploratory_oracle/{run_id}/pairs/paired_differences_manifest.json`; full-sample pairs and each spatial seed are separate Parquet files (not intended for Git)
- Model parameters: `outputs/exploratory_oracle/{run_id}/tables/selected_model_parameters.json`
- Robustness: `outputs/exploratory_oracle/{run_id}/tables/robustness_sensitivity.csv`
- Exact input checksums and software versions: `outputs/exploratory_oracle/{run_id}/provenance.json`
- Automated software test suite: not added or run; the task is an empirical analysis and developer instructions prohibit tests unless requested. The requested analysis and artifact checks are the execution evidence.

## Required CSV tables

{chr(10).join(f'- `phase-f/tables/{name}`' for name in tables)}

## Figures

{figure_links}

## Outcome-guided selection audit

- The local matching grid is outcome-blind. Separately, the high-response screen ranks soil-specific V5 k=6 groups by `{high_selection.get('selection_metric', 'unknown')}` and retains the top three per exposure-by-outcome family. It considered {high_selection.get('candidate_trial_count', 0)} group candidates in {high_selection.get('selection_family_count', 0)} families and retained {high_selection.get('selected_group_count', 0)}; selected groups cover {high_selection.get('cell_retention_fraction_min', 'unknown')}–{high_selection.get('cell_retention_fraction_max', 'unknown')} of eligible complete-case cells per family (median {high_selection.get('cell_retention_fraction_median', 'unknown')}). Per-group counts and retention fractions are in `phase-f/tables/high_response_environment_groups.csv`.
- Figure 8's inherited full-sample curves come from `{curve_selection.get('source_run_id', 'unknown')}`. They were selected by `{curve_selection.get('selection_metric', 'unknown')}` from {curve_selection.get('selection_trial_count', 'unknown')} model trials across {curve_selection.get('model_family_count', 'unknown')} model families; each selected model and complete-case sample size are in the Figure 8 data CSV. The natural spline uses the fixed plotting sample and is descriptive.
- Detailed curve source metadata is in `phase-f/tables/phase_f_figure_source_summary.json`; run metadata records the same outcome-guided selection audit in `outputs/exploratory_oracle/{run_id}/provenance.json`.

Figure 8 contains two different curve estimands: natural spline on the fixed plot sample and the inherited full-sample best-model partial-dependence curve. The full-sample curve is outcome-selected/apparent. The chart and caption label both scopes.

## Review questions

1. Are M0–M4 defined and compared fairly, especially the distinction between regression adjustment, environment-group slopes, K-means matching, and direct nearest-neighbor matching?
2. Does the percentile-rank matching contrast, V5 distance, 0.35 environment caliper, 1,000m radius, 0.50 rank gap, and no-replacement rule define a defensible paired design? Are the local-grid selection criteria adequate and outcome-blind?
3. Do the recorded pre/post SMDs, variable-level absolute differences, soil/group diagnostics, distance quantiles, zero-positive pairing and unmatched reasons support the balance claims?
4. Are P1–P4 estimands and interpretations correct? Is the no-intercept P2 appropriately treated as a sensitivity check, and is P3's limited V5 adjustment clear?
5. Are all pair endpoints kept within one spatial role, are transformations and K-means learned from training only, and are ten 5km seeds sufficient for internal stability assessment?
6. Is the ΔR² comparison genuinely nested on identical pixels/pairs, identical model families and identical splits? Are the median, IQR, positive fraction and coefficient-direction summaries interpreted appropriately?
7. Are environmental heterogeneity definitions, interaction tests, and FDR correction adequate? Are outcome-ranked high-response groups labeled as exploratory selection?
8. Is Aa_2020 compared on the same common pixels/model/splits at the cell level, with exposure-specific matched pairs clearly distinguished?
9. Are full-sample apparent results, prior Oracle choices, raster prediction-surface limitations, spatial dependence, and unit/time metadata limitations described without causal or independent-observation claims?
10. Based on balance, coverage, exposure separation, repeated spatial performance, ΔR² and subgroup stability together, should M3 or M4 remain the main next-stage candidate, and which indicators merit further work?

## Requested response format

Please identify (a) critical design or calculation issues that would change the scientific conclusion, (b) important repairs needed before the next phase, (c) minor improvements, (d) any claims that exceed the evidence, and (e) whether the proposed next research actions align with the charter. Cite exact file/table/figure and row/column where possible. Do not assume values or sources absent from the manifest; flag them as unknown.

## Review boundaries and next gate

`Aa_20yr` and `Aa_2020` are the already-computed erosion values. No RUSLE reconstruction or factor-product validation was performed. PCNM remains deferred. This phase does not edit the charter or make causal claims. The next formal research phase remains on hold until the user receives this review and decides how to proceed.
"""


def update_project_log(run_id: str, baseline_sha: str, output_path: Path) -> None:
    log_path = ROOT / "docs" / "DEVELOPMENT_LOG.md"
    if run_id in log_path.read_text():
        return
    summary_path = output_path / "experiment_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    selection_audit = summary.get("outcome_guided_selection_audit", {})
    high_selection = selection_audit.get("high_response_group_ranking", {})
    curve_selection = selection_audit.get("inherited_full_sample_model_curves", {})
    retention_text = "unknown"
    if high_selection.get("cell_retention_fraction_min") is not None:
        retention_text = (
            f"{high_selection['cell_retention_fraction_min']:.3f}–"
            f"{high_selection['cell_retention_fraction_max']:.3f} "
            f"(median {high_selection['cell_retention_fraction_median']:.3f})"
        )
    log_path.write_text(log_path.read_text() + f"""

## 2026-10-26：Phase F 配对差值与重复空间验证

- 探索模式：`EXPLORATORY_ORACLE`；基线 SHA `{baseline_sha}`。Phase F 运行 `{run_id}`，输入 `{output_path.relative_to(ROOT)}`；当前改动尚未提交，`.git` 只读。
- 使用已有对齐 Parquet 中的 `Aa_20yr` 和 `Aa_2020`；未重算/校验 RUSLE，未运行 PCNM，未修改研究总纲。新增 outcome-blind 局部匹配搜索、M0–M4 比较、成对差值 P1–P4、10 个 5km 空间种子、嵌套 ΔR²、环境异质性、两期对照和敏感性分析。
- 运行标记 `outcome_guided_selection={str(summary.get('outcome_guided_selection', False)).lower()}`。高响应组按 `{high_selection.get('selection_metric', 'unknown')}` 比较 {high_selection.get('candidate_trial_count', 0)} 组候选、保留 {high_selection.get('selected_group_count', 0)} 组；各暴露—结局组合样本保留比例 {retention_text}。继承曲线按 `{curve_selection.get('selection_metric', 'unknown')}` 从 {curve_selection.get('selection_trial_count', 'unknown')} 次、{curve_selection.get('model_family_count', 'unknown')} 类模型比较中选择。
- 报告与审阅包：`docs/reviews/2026-10-26/PHASE_F_RESEARCH_REPORT.md`、`docs/reviews/2026-10-26/PHASE_F_CHATGPT_REVIEW_REQUEST.md`；图表为 PNG/SVG/PDF，表格和图注均另存。
- 证据与边界：运行清单保存输入 SHA-256、配置/依赖版本及模型设置；配对端点按空间角色独立构造。完整 pair Parquet 保留在 Git 忽略的 `outputs/exploratory_oracle/`；结果仍是内部重复空间验证和预测面关联，不作因果/外部验证结论。
- 未提交、未推送。等待用户本地提交并交 ChatGPT 审阅；Phase F 审阅前不推进正式下一阶段。
""")


def update_decisions(run_id: str, baseline_sha: str) -> None:
    path = ROOT / "docs" / "DECISIONS.md"
    if run_id in path.read_text():
        return
    path.write_text(path.read_text() + f"""

## 2026-10-26：Phase F 匹配比较与空间验证边界

1. **匹配比较。** 以 Aa_20yr 为局部参数搜索主暴露，预先固定 V5 Direct NN 参考配置（0.35环境卡尺、1,000m、组内秩差0.50、不重复使用），并与同向量K-means k=6作对照；Aa_2020使用同一参考配置，不宣称其阈值为该期最优。
2. **独立样本与比较。** 10个5km整块种子分别按训练、验证、测试角色建对；标准化和K-means仅在训练块拟合。M0–M2时期比较使用两期暴露、V5和结局共同完整像元；ΔR²取嵌套OLS在同一测试像元/配对上的差。
3. **模型解释。** P1成对截距OLS为主，P2无截距仅作敏感性，P3只调整V5的Elevation/LST配对差，P4使用Huber稳健回归；P5仅在预设重复测试增益标准达到时保留。
4. **阶段边界。** 全样本配对、历史Oracle曲线/子组为描述性表观结果；空间验证是先前探索后的内部重复空间留出，不是外部验证。健康面为预测面，结果不作因果解释。继续使用预计算Aa，不进行RUSLE重算或因子乘积验证；PCNM和章程均不变。
5. **运行状态。** Phase F run `{run_id}` 基于 `{baseline_sha}`；未提交未推送。ChatGPT/用户审阅完成前，不进入正式下一阶段。
""")


def run_analysis(args: argparse.Namespace) -> Path:
    config = load_config()
    if args.config:
        global CONFIG_PATH
        CONFIG_PATH = Path(args.config).resolve()
        config = yaml.safe_load(CONFIG_PATH.read_text())
    frame, profile, input_paths = load_input(str(config["input_run_id"]))
    run_id = args.run_id or f"phase_f_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_seed{config['seed']}"
    run_dir = ROOT / config["output_root"] / run_id
    table_dir = run_dir / "tables"
    pair_dir = run_dir / "pairs"
    table_dir.mkdir(parents=True, exist_ok=True)
    pair_dir.mkdir(parents=True, exist_ok=True)
    previous_provenance_path = run_dir / "provenance.json"
    previous_provenance = (
        json.loads(previous_provenance_path.read_text()) if previous_provenance_path.exists() else None
    )
    previous_config_path = run_dir / "config.yaml"
    previous_config = yaml.safe_load(previous_config_path.read_text()) if previous_config_path.exists() else None
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    (run_dir / "config").mkdir(parents=True, exist_ok=True)
    (run_dir / "config" / "phase_f.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    provenance_record = provenance(config, input_paths, profile)
    provenance_record["run_id"] = run_id
    write_json(run_dir / "provenance.json", provenance_record)
    write_json(run_dir / "tables" / "input_profile.json", profile)
    if args.profile_only:
        print(f"Input profile/provenance saved: {run_dir}")
        return run_dir

    can_resume = same_run_signature(previous_provenance, provenance_record, previous_config, config)
    local_grid_path = table_dir / "local_matching_grid.csv"
    local_balance_path = table_dir / "matching_balance_diagnostics.csv"
    expected_grid_rows = (
        len(config["environment_vectors"])
        * len(config["matching"]["local_search"]["cluster_options"])
        * len(config["matching"]["local_search"]["environment_calipers"])
        * len(config["matching"]["local_search"]["geographic_radii_m"])
        * len(config["matching"]["local_search"]["erosion_rank_gaps"])
    )
    cached_grid_is_complete = False
    if can_resume and local_grid_path.exists() and local_balance_path.exists():
        cached_grid = pd.read_csv(local_grid_path)
        cached_grid_is_complete = (
            len(cached_grid) == expected_grid_rows
            and cached_grid["candidate_id"].nunique() == expected_grid_rows
            and {"outcome_blind_rank", "pareto_candidate"}.issubset(cached_grid.columns)
        )
    if cached_grid_is_complete:
        local_grid = cached_grid
        local_balance = pd.read_csv(local_balance_path)
        print(f"Reused {len(local_grid)} cached outcome-blind local matching configurations")
    else:
        print(f"Phase F run {run_id}: input profile saved; starting outcome-blind local matching grid")
        local_grid, local_balance = run_local_matching_search(frame, config, table_dir)
    print("Building descriptive full-sample matched pairs")
    full_pair_cache = pair_dir / "full_sample_pairs.parquet"
    full_balance_cache = table_dir / "full_sample_matching_balance.json"
    full_pair_cache_meta = pair_dir / "full_sample_pair_cache.json"
    full_pair_cache_is_valid = False
    if can_resume and full_pair_cache.exists() and full_balance_cache.exists() and full_pair_cache_meta.exists():
        cache_meta = json.loads(full_pair_cache_meta.read_text())
        full_pair_cache_is_valid = same_run_signature(cache_meta.get("signature"), provenance_record)
    if full_pair_cache_is_valid:
        full_pairs = pd.read_parquet(full_pair_cache)
        full_balance = read_frame_cache(full_balance_cache)
        print(f"Reused {len(full_pairs):,} cached full-sample matched pairs")
    else:
        full_pairs, full_balance = run_full_sample_pairs(frame, config)
        if not full_pairs.empty:
            full_pairs.to_parquet(full_pair_cache, index=False, compression="zstd")
        write_frame_cache(full_balance_cache, full_balance)
        write_json(full_pair_cache_meta, {"signature": provenance_record})
    paired_results = run_full_paired_models(full_pairs, config)
    print("Running ten seeded, whole-block spatial validations")
    repeated, increments, cv_balance, cv_pair_tables, cv_aux = run_spatial_validation(
        frame, config, run_dir, provenance_record
    )
    pair_data_manifest_path = pair_dir / "paired_differences_manifest.json"
    write_json(pair_data_manifest_path, {
        "format": "partitioned_parquet_files",
        "full_sample_pair_file": str(full_pair_cache.relative_to(ROOT)),
        "full_sample_pair_rows": int(len(full_pairs)),
        "spatial_pair_files": cv_aux["pair_data_files"],
        "spatial_pair_rows": int(cv_aux["pair_data_rows"]),
        "total_pair_rows": int(len(full_pairs) + cv_aux["pair_data_rows"]),
        "pair_columns": list(full_pairs.columns),
    })
    write_json(run_dir / "spatial_fold_metadata.json", cv_aux["folds"])

    p5_summary = spline_necessity(repeated)
    if not p5_summary.empty:
        for _, choice in p5_summary.loc[p5_summary["p5_indicated"]].iterrows():
            subset = full_pairs.loc[
                full_pairs["method"].eq(choice["method"]) & full_pairs["exposure"].eq(choice["exposure"])
            ]
            estimate = fit_full_paired_spline(subset, choice["outcome"])
            paired_results = pd.concat([paired_results, pd.DataFrame([{
                "method": choice["method"], "method_name": MODEL_NAMES[str(choice["method"])],
                "exposure": choice["exposure"], "outcome": choice["outcome"],
                "model": "P5_natural_spline", "evaluation_set": "full_sample_descriptive_apparent",
                "selection_scope": "retained only after >=0.02 median spatial-test R2 gain in >=70% of seeds",
                "outcome_guided_selection": False, **estimate,
            }])], ignore_index=True)
    robustness = robustness_sensitivities(full_pairs, repeated, local_grid, config)
    if not p5_summary.empty:
        p5_rows = p5_summary.copy()
        p5_rows["sensitivity"] = "P5_nonlinearity_spatial_test_diagnostic"
        p5_rows["selection_scope"] = "P5 retained only if median test R2 gain >=0.02 and >=70% positive across >=8 seeds"
        robustness = pd.concat([robustness, p5_rows], ignore_index=True, sort=False)

    heterogeneity, high_groups = paired_subgroup_results(full_pairs, frame, config, repeated)
    full_comparison = full_sample_exposure_comparison(frame, full_pairs, paired_results, config)
    exposure_comparison = build_exposure_comparison(full_comparison, repeated, increments, paired_results)
    increments = pd.concat([increments, full_sample_increment_rows(full_comparison)], ignore_index=True, sort=False)
    final_models = summarize_validation(repeated, cv_balance)
    method_comparison = build_matching_method_comparison(final_models, local_grid, full_pairs)
    matching_balance = add_overall_max_smd_to_balance(
        pd.concat([local_balance, full_balance, cv_balance], ignore_index=True, sort=True)
    )

    table_frames = {
        "matching_method_comparison.csv": method_comparison,
        "matching_balance_diagnostics.csv": matching_balance,
        "paired_difference_results.csv": paired_results,
        "repeated_spatial_validation.csv": repeated,
        "erosion_increment_r2.csv": increments,
        "environmental_heterogeneity.csv": heterogeneity,
        "high_response_environment_groups.csv": high_groups,
        "exposure_period_comparison.csv": exposure_comparison,
        "final_model_comparison.csv": final_models,
    }
    if high_groups.empty:
        high_selection_audit = {
            "selection_metric": "full_sample_mean_outcome",
            "selection_objective": "rank groups by descending mean outcome; retain the top three per exposure and outcome",
            "selection_family_count": 0,
            "candidate_trial_count": 0,
            "selected_group_count": 0,
            "cell_retention_fraction_min": None,
            "cell_retention_fraction_median": None,
            "cell_retention_fraction_max": None,
            "families": [],
        }
    else:
        family_audit = high_groups.groupby(["exposure", "outcome"], as_index=False).agg(
            candidate_groups=("environment_group", "size"),
            selected_groups=("top_three_high_response_group", "sum"),
            source_complete_case_cells=("selection_source_complete_case_cells", "first"),
            retained_cell_fraction=("selection_retained_cell_fraction", "first"),
        )
        family_retention = pd.to_numeric(family_audit["retained_cell_fraction"], errors="coerce").dropna()
        high_selection_audit = {
            "selection_metric": "full_sample_mean_outcome",
            "selection_objective": "rank groups by descending mean outcome; retain the top three per exposure and outcome",
            "selection_family_count": int(len(family_audit)),
            "candidate_trial_count": int(len(high_groups)),
            "selected_group_count": int(high_groups["top_three_high_response_group"].fillna(False).sum()),
            "cell_retention_fraction_min": float(family_retention.min()) if len(family_retention) else None,
            "cell_retention_fraction_median": float(family_retention.median()) if len(family_retention) else None,
            "cell_retention_fraction_max": float(family_retention.max()) if len(family_retention) else None,
            "families": family_audit.to_dict(orient="records"),
        }
    provenance_record["outcome_guided_selection"] = True
    provenance_record["outcome_guided_selection_audit"] = {
        "high_response_group_ranking": high_selection_audit,
        "inherited_full_sample_model_curves": {
            "source_run_id": "oracle_20261026T041521459615Z_20261026",
            "selection_metric": "full_sample_apparent_r2",
            "selection_trial_count": 1024,
            "model_family_count": 13,
            "outcome_specific_complete_case_counts_are_in_curve_source_data": True,
        },
    }
    write_json(run_dir / "provenance.json", provenance_record)
    review_dir = ROOT / "docs" / "reviews" / "2026-10-26" / "phase-f"
    review_table_dir = review_dir / "tables"
    review_table_dir.mkdir(parents=True, exist_ok=True)
    write_json(review_table_dir / "provenance.json", provenance_record)
    write_json(review_table_dir / "input_profile.json", profile)
    for name, table in table_frames.items():
        write_csv(table_dir / name, table)
        write_csv(review_table_dir / name, table)
    write_csv(table_dir / "robustness_sensitivity.csv", robustness)
    write_csv(review_table_dir / "robustness_sensitivity.csv", robustness)
    write_csv(table_dir / "local_matching_grid.csv", local_grid)
    write_csv(review_table_dir / "local_matching_grid.csv", local_grid)
    write_csv(table_dir / "spline_necessity.csv", p5_summary)
    write_csv(review_table_dir / "spline_necessity.csv", p5_summary)
    write_csv(table_dir / "exposure_period_common_sample_detail.csv", full_comparison)
    write_csv(review_table_dir / "exposure_period_common_sample_detail.csv", full_comparison)
    increment_summary = summarize_increment_r2(increments)
    write_csv(table_dir / "erosion_increment_r2_summary.csv", increment_summary)
    write_csv(review_table_dir / "erosion_increment_r2_summary.csv", increment_summary)

    primary_parameters = paired_results.loc[
        paired_results["method"].eq("M4") & paired_results["exposure"].eq("Aa_20yr")
        & paired_results["outcome"].isin(PRIMARY_OUTCOMES)
    ]
    selected_config = local_grid.sort_values("outcome_blind_rank").head(10) if not local_grid.empty and "outcome_blind_rank" in local_grid else pd.DataFrame()
    write_json(table_dir / "selected_model_parameters.json", {
        "primary_full_sample_pair_models": primary_parameters.to_dict(orient="records"),
        "local_matching_top10_outcome_blind": selected_config.to_dict(orient="records"),
        "cv_model_forms": MODEL_NAMES,
        "p5_necessity": p5_summary.to_dict(orient="records"),
    })
    curve_source_root = ROOT / "outputs" / "exploratory_oracle" / "oracle_20261026T041521459615Z_20261026"
    plot_source_path = curve_source_root / "diagnostics" / "plot_data.parquet"
    selected_curve_path = curve_source_root / "models" / "selected_full_sample_curves.parquet"
    plot_source = pd.read_parquet(plot_source_path) if plot_source_path.exists() else pd.DataFrame()
    selected_curve_source = pd.read_parquet(selected_curve_path) if selected_curve_path.exists() else pd.DataFrame()
    selected_curve_metadata = []
    if not selected_curve_source.empty:
        selected_curve_metadata = (
            selected_curve_source.groupby(["outcome", "exposure"], as_index=False)
            .agg(
                model_name=("model_name", "first"),
                selected_model_r_squared=("selected_model_r_squared", "first"),
                n_complete_cases=("n_complete_cases", "first"),
                curve_rows=("predicted_outcome", "size"),
                selection_metric=("selected_by", "first"),
            )
            .to_dict(orient="records")
        )
    figure_source_summary = {
        "fixed_plot_sample_rows": int(len(plot_source)),
        "inherited_full_sample_curve_rows": int(len(selected_curve_source)),
        "inherited_curve_source_run_id": "oracle_20261026T041521459615Z_20261026",
        "inherited_curve_selection_metric": "full_sample_apparent_r2",
        "inherited_curve_model_trial_count": 1024,
        "inherited_curve_model_family_count": 13,
        "inherited_curve_selection_metadata": selected_curve_metadata,
        "new_figure8_curves": ["natural_spline_fixed_plot_sample", "selected_full_sample_apparent_model_partial_dependence"],
    }
    write_json(table_dir / "phase_f_figure_source_summary.json", figure_source_summary)
    write_json(review_table_dir / "phase_f_figure_source_summary.json", figure_source_summary)

    figure_index = generate_phase_figures(
        review_dir, local_grid, paired_results, repeated, increments, heterogeneity,
        high_groups, exposure_comparison, full_pairs, frame,
    )
    report_path = ROOT / "docs" / "reviews" / "2026-10-26" / "PHASE_F_RESEARCH_REPORT.md"
    report = report_text(
        run_id=run_id,
        baseline_sha=provenance_record["baseline_commit_sha"],
        provenance_record=provenance_record,
        local_grid=local_grid,
        paired_results=paired_results,
        repeated=repeated,
        increments=increments,
        heterogeneity=heterogeneity,
        high_groups=high_groups,
        final_models=final_models,
        matching_balance=matching_balance,
        p5_summary=p5_summary,
        figure_index=figure_index,
    )
    report_path.write_text(report)
    (run_dir / "PHASE_F_RESEARCH_REPORT.md").write_text(report)
    review_path = ROOT / "docs" / "reviews" / "2026-10-26" / "PHASE_F_CHATGPT_REVIEW_REQUEST.md"
    review_text = chatgpt_review_request(
        run_id, provenance_record["baseline_commit_sha"], provenance_record, review_dir, figure_index
    )
    review_path.write_text(review_text)
    (run_dir / "PHASE_F_CHATGPT_REVIEW_REQUEST.md").write_text(review_text)

    # Mirror deliverables into named run-level domains for review and downstream reuse.
    category_files = {
        "matching": ["matching_method_comparison.csv", "matching_balance_diagnostics.csv", "local_matching_grid.csv"],
        "paired_models": ["paired_difference_results.csv"],
        "spatial_validation": ["repeated_spatial_validation.csv"],
        "increment_r2": ["erosion_increment_r2.csv", "erosion_increment_r2_summary.csv", "exposure_period_comparison.csv", "exposure_period_common_sample_detail.csv"],
        "heterogeneity": ["environmental_heterogeneity.csv", "high_response_environment_groups.csv"],
        "robustness": ["robustness_sensitivity.csv", "spline_necessity.csv"],
    }
    for category, filenames in category_files.items():
        category_dir = run_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        for filename in filenames:
            shutil.copy2(table_dir / filename, category_dir / filename)
    shutil.copytree(review_dir / "figures", run_dir / "figures", dirs_exist_ok=True)
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(report_path, reports_dir / report_path.name)
    shutil.copy2(review_path, reports_dir / review_path.name)

    summary = {
        "run_id": run_id,
        "input_run_id": config["input_run_id"],
        "baseline_commit_sha": provenance_record["baseline_commit_sha"],
        "mode": "EXPLORATORY_ORACLE",
        "outcome_guided_selection": bool(provenance_record.get("outcome_guided_selection", False)),
        "outcome_guided_selection_audit": provenance_record.get("outcome_guided_selection_audit", {}),
        "local_matching_grid_rows": int(len(local_grid)),
        "local_matching_success_rows": int(local_grid.get("status", pd.Series(dtype=object)).eq("ok").sum()),
        "full_sample_pair_rows": int(len(full_pairs)),
        "spatial_pair_rows": int(cv_aux["pair_data_rows"]),
        "total_pair_rows": int(len(full_pairs) + cv_aux["pair_data_rows"]),
        "paired_difference_model_rows": int(len(paired_results)),
        "repeated_spatial_validation_rows": int(len(repeated)),
        "erosion_increment_rows": int(len(increments)),
        "environmental_heterogeneity_rows": int(len(heterogeneity)),
        "high_response_group_rows": int(len(high_groups)),
        "exposure_comparison_rows": int(len(exposure_comparison)),
        "final_model_comparison_rows": int(len(final_models)),
        "robustness_rows": int(len(robustness)),
        "erosion_increment_summary_rows": int(len(increment_summary)),
        "figure_count": int(len(figure_index)),
        "figure_formats": ["png", "svg", "pdf"],
        "tables": {name: {"rows": int(len(table)), "columns": list(table.columns)} for name, table in table_frames.items()},
        "erosion_increment_r2_summary": {"rows": int(len(increment_summary)), "columns": list(increment_summary.columns)},
        "report_path": str(report_path.relative_to(ROOT)),
        "review_request_path": str(review_path.relative_to(ROOT)),
        "pair_data_path": str(pair_data_manifest_path.relative_to(ROOT)),
        "software_tests": "not added or run per developer task policy; analysis execution and artifact audits used",
        "next_phase_gate": "wait for ChatGPT review and user decision",
    }
    write_json(run_dir / "experiment_summary.json", summary)
    write_json(run_dir / "artifact_manifest.json", {
        "run_id": run_id,
        "baseline_commit_sha": provenance_record["baseline_commit_sha"],
        "input_run_id": config["input_run_id"],
        "outcome_guided_selection": bool(provenance_record.get("outcome_guided_selection", False)),
        "review_files": [str(report_path.relative_to(ROOT)), str(review_path.relative_to(ROOT))],
        "tables": [str((review_table_dir / name).relative_to(ROOT)) for name in [*table_frames.keys(), "robustness_sensitivity.csv", "local_matching_grid.csv", "spline_necessity.csv", "erosion_increment_r2_summary.csv", "phase_f_figure_source_summary.json", "provenance.json", "input_profile.json"]],
        "figures": figure_index,
        "pair_data_manifest": str(pair_data_manifest_path.relative_to(ROOT)),
        "run_artifact_domains": ["config", "matching", "paired_models", "spatial_validation", "increment_r2", "heterogeneity", "robustness", "figures", "tables", "reports"],
    })
    update_project_log(run_id, provenance_record["baseline_commit_sha"], run_dir)
    update_decisions(run_id, provenance_record["baseline_commit_sha"])
    print(f"Phase F artifacts complete: {run_dir}")
    print(f"Research report: {report_path}")
    print(f"ChatGPT review request: {review_path}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None, help="YAML configuration path (defaults to config/phase_f.yaml)")
    parser.add_argument("--run-id", type=str, default=None, help="Optional stable run ID")
    parser.add_argument("--profile-only", action="store_true", help="Save input provenance/profile without running models")
    args = parser.parse_args()
    run_analysis(args)


def save_figure(fig: plt.Figure, name: str, caption: str, data: pd.DataFrame, review_dir: Path) -> dict[str, Any]:
    figure_dir = review_dir / "figures"
    data_dir = figure_dir / "data"
    caption_dir = figure_dir / "captions"
    figure_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    caption_dir.mkdir(parents=True, exist_ok=True)
    if fig.get_layout_engine() is None:
        fig.tight_layout()
    paths = []
    for extension in ["png", "svg", "pdf"]:
        path = figure_dir / f"{name}.{extension}"
        fig.savefig(path, dpi=320, bbox_inches="tight")
        paths.append(str(path.relative_to(review_dir)))
    plt.close(fig)
    csv_name = f"{name}.csv"
    write_csv(data_dir / csv_name, data)
    (caption_dir / f"{name}.md").write_text(f"# {name}\n\n{caption}\n\nFigure data: `data/{csv_name}`.\n")
    return {"name": name, "formats": paths, "data_path": str((data_dir / csv_name).relative_to(review_dir)), "caption_path": str((caption_dir / f"{name}.md").relative_to(review_dir)), "caption": caption}


def _valid_stored_map_points(map_rows: pd.DataFrame) -> pd.DataFrame:
    """Reject cached map rows without usable soil-stratified cluster IDs."""
    required = {"x", "y", "SoilType", "cluster_within_soil"}
    if map_rows.empty or not required.issubset(map_rows.columns):
        return map_rows.iloc[0:0].copy()
    result = map_rows.copy()
    for column in required:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    valid = np.isfinite(result[["x", "y", "SoilType", "cluster_within_soil"]].to_numpy(dtype=float)).all(axis=1)
    valid &= result["SoilType"].isin(SOIL_TYPE_LABELS)
    valid &= result["cluster_within_soil"].between(0, 5)
    return result.loc[valid].copy()


def generate_phase_figures(
    review_dir: Path,
    local_grid: pd.DataFrame,
    full_pair_results: pd.DataFrame,
    repeated: pd.DataFrame,
    increments: pd.DataFrame,
    heterogeneity: pd.DataFrame,
    high_groups: pd.DataFrame,
    exposure_comparison: pd.DataFrame,
    full_pairs: pd.DataFrame,
    frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    plt.rcParams.update({"font.family": "Noto Sans CJK JP", "axes.titlesize": 11, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "axes.edgecolor": "#4B5563", "axes.linewidth": 0.8, "grid.color": "#D9DEE7", "grid.linewidth": 0.55, "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"})
    figures: list[dict[str, Any]] = []
    outcome_labels = {"SHI_candidate": "土壤健康指数（SHI）", "AN": "碱解氮（AN）", "OM": "有机质（OM）", "CEC": "阳离子交换量（CEC）", "pH": "土壤酸碱度（pH）", "SW": "土壤含水量（SW）", "BD": "土壤容重（BD）", "MBC": "微生物量碳（MBC）", "RESP": "土壤呼吸（RESP）"}
    exposure_labels = {"Aa_20yr": "20 年平均实际侵蚀", "Aa_2020": "2020 年实际侵蚀"}
    method_labels = {"M0": "未调整基线（M0）", "M1": "环境回归调整（M1）", "M2": "环境组斜率（M2）", "M3": "土类内 K 均值匹配（M3）", "M4": "土类内最近邻匹配（M4）"}

    # 1. Local matching search balance and coverage.
    if not local_grid.empty:
        data = local_grid.loc[local_grid["status"].eq("ok")].copy()
        fig, ax = plt.subplots(figsize=(8, 5))
        cluster_labels = {"none": "不聚类", "kmeans": "K 均值", "ward": "Ward 层次聚类"}
        for label, group in data.groupby("cluster_method", sort=True):
            ax.scatter(group["area_coverage"], group["max_abs_matched_smd"], s=28, alpha=.72, label=cluster_labels.get(str(label), str(label)), edgecolors="white", linewidths=.35)
        ax.axhline(.1, color="#606A78", linestyle="--", linewidth=1, label="SMD = 0.10 参考线")
        ax.set(xlabel="匹配覆盖率", ylabel="环境变量最大绝对标准化差异（SMD）", title="1｜结果盲的局地匹配权衡")
        ax.grid(axis="both", alpha=.55)
        ax.legend(frameon=False)
        figures.append(save_figure(fig, "01_matching_coverage_balance", "局地搜索在 20 年平均实际侵蚀下评价 243 组匹配参数，未读取健康结局。覆盖率、SMD、环境距离和侵蚀差异均纳入设计诊断；横线仅为描述性平衡参考。", data, review_dir))
    else:
        figures.append(save_figure(plt.figure(), "01_matching_coverage_balance", "没有成功的局地匹配参数组合。", pd.DataFrame(), review_dir))

    # 2. M3 vs M4 matching diagnostics on repeated held-out spatial roles.
    match_rows = repeated.loc[repeated["method"].isin(["M3", "M4"]) & repeated["evaluation_set"].eq("test") & repeated["outcome"].isna()] if not repeated.empty else pd.DataFrame()
    # Matching summaries are taken from the balance table stored alongside pairs by the caller when available.
    data = full_pair_results.loc[full_pair_results["model"].eq("P1_intercept_ols") & full_pair_results["method"].isin(["M3", "M4"])].copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    if not data.empty:
        data["label"] = data["exposure"].map(exposure_labels).fillna(data["exposure"]) + " / " + data["method"].map(method_labels).fillna(data["method"])
        ax.bar(data["label"], data["r_squared"], color=["#326A8C" if m == "M4" else "#D28A34" for m in data["method"]], edgecolor="white", linewidth=.5)
        ax.set_ylabel("全样本配对模型表观 R²")
        ax.set_title("2｜不同匹配方法与侵蚀时期的配对拟合")
        ax.tick_params(axis="x", rotation=25)
    else:
        ax.text(.5, .5, "没有可用的配对模型记录", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    figures.append(save_figure(fig, "02_matching_method_pair_fit", "全样本配对 R² 为描述性拟合，不作为匹配规则的筛选目标。M3 为土类内 K 均值 k=6 后匹配；M4 为在相同土类内直接进行最近邻匹配。", data, review_dir))

    # 3. Primary outcome paired slopes with 95% OLS intervals.
    data = full_pair_results.loc[
        full_pair_results["model"].eq("P1_intercept_ols") & full_pair_results["outcome"].isin(PRIMARY_OUTCOMES)
    ].copy()
    fig, ax = plt.subplots(figsize=(9, 5))
    if not data.empty:
        data["label"] = data["outcome"].map(outcome_labels).fillna(data["outcome"]) + " / " + data["method"].map(method_labels).fillna(data["method"]) + " / " + data["exposure"].map(exposure_labels).fillna(data["exposure"])
        errors = np.vstack([data["coefficient"] - data["ci_lower"], data["ci_upper"] - data["coefficient"]])
        ax.errorbar(data["coefficient"], np.arange(len(data)), xerr=errors, fmt="o", capsize=2)
        ax.axvline(0, color="grey", linestyle="--", linewidth=1)
        ax.set_yticks(np.arange(len(data)), data["label"])
        ax.set(xlabel="P1 斜率：Δ健康 / Δlog1p(Aa)", title="3｜优先指标的匹配配对差值斜率")
    else:
        ax.text(.5, .5, "没有可用的优先指标配对估计", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    figures.append(save_figure(fig, "03_primary_paired_slopes", "点为全样本 P1 匹配配对斜率，横线为模型估计的 95% 置信区间。该区间未校正残余空间依赖；重复 5 km 空间留出结果另行报告。", data, review_dir))

    # 4. Repeated spatial test scores across M0-M4.
    data = repeated.loc[
        repeated["evaluation_set"].eq("test") & repeated["method"].isin(list(MODEL_NAMES))
        & repeated["outcome"].isin(PRIMARY_OUTCOMES)
        & (repeated.get("validation_model", pd.Series(index=repeated.index, dtype=object)).isna()
           | repeated.get("validation_model", pd.Series(index=repeated.index, dtype=object)).eq("P1_intercept_ols"))
    ].copy()
    fig, ax = plt.subplots(figsize=(11, 5))
    if not data.empty:
        outcome_short = {"SHI_candidate": "SHI", "AN": "AN", "OM": "OM"}
        method_short = {"M0": "M0 基线", "M1": "M1 环境调整", "M2": "M2 组内斜率", "M3": "M3 K 均值", "M4": "M4 最近邻"}
        labels = [f"{outcome_short.get(o,o)}\n{method_short.get(m,m)}" for o in PRIMARY_OUTCOMES for m in ["M0", "M1", "M2", "M3", "M4"]]
        values = [data.loc[(data["outcome"].eq(o)) & (data["method"].eq(m)), "r_squared"].dropna().to_numpy() for o in PRIMARY_OUTCOMES for m in ["M0", "M1", "M2", "M3", "M4"]]
        ax.boxplot(values, labels=labels, showfliers=False)
        ax.axhline(0, color="grey", linestyle="--", linewidth=1)
        ax.set(ylabel="5 km 空间留出测试 R²", title="4｜不同方法与优先指标的重复空间留出")
        ax.tick_params(axis="x", rotation=40)
    else:
        ax.text(.5, .5, "没有可用的重复空间测试结果", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    figures.append(save_figure(fig, "04_repeated_spatial_validation", "每个箱线图汇总 10 次固定随机种子的 5 km 整块空间测试划分。M0–M2 在两个 Aa 时期使用共同完整像元样本；M3–M4 分别构建训练、验证和测试配对，并仅用训练集拟合变换。", data, review_dir))

    # 5. Same-sample delta R2.
    data = increments.loc[
        increments["evaluation_set"].eq("test") & increments["analysis_type"].eq("cell_level_nested_ols")
    ].copy() if not increments.empty else pd.DataFrame()
    fig, ax = plt.subplots(figsize=(10, 5))
    if not data.empty:
        summary = data.groupby(["outcome", "exposure"], as_index=False)["delta_r2"].median()
        pivot = summary.pivot(index="outcome", columns="exposure", values="delta_r2")
        pivot.rename(columns=exposure_labels).plot(kind="bar", ax=ax, color=["#326A8C", "#D28A34"], edgecolor="white", linewidth=.4)
        ax.axhline(0, color="grey", linestyle="--", linewidth=1)
        ax.set(ylabel="空间留出 ΔR² 中位数", title="5｜相同像元上侵蚀信息相对环境变量的增量")
        ax.legend(title="侵蚀时期", frameon=False)
        ax.set_xticklabels([outcome_labels.get(str(t.get_text()), str(t.get_text())) for t in ax.get_xticklabels()], rotation=25, ha="right")
    else:
        ax.text(.5, .5, "没有相同样本上的 ΔR² 记录", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    figures.append(save_figure(fig, "05_erosion_increment_delta_r2", "每个随机种子和指标下，纯环境模型与环境加 log1p(Aa) 的 OLS 模型使用相同训练像元，并在相同留出像元上评分。该增量表示预测信息变化，不是因果效应或独立数据验证。", data, review_dir))

    # 6. Environmental heterogeneity slopes and formal interactions.
    data = heterogeneity.loc[heterogeneity["subgroup"].ne("__interaction_test__")].copy() if not heterogeneity.empty else pd.DataFrame()
    fig, ax = plt.subplots(figsize=(10, 6))
    if not data.empty:
        selected = data.loc[data["outcome"].isin(PRIMARY_OUTCOMES) & data["subgroup_variable"].isin(["soil_type", "V5_kmeans_group", "Elevation_tertile", "LST_tertile"])].copy()
        group_var_labels = {"soil_type": "土壤类型", "V5_kmeans_group": "V5 土类内 K 均值组", "Elevation_tertile": "高程三分位", "LST_tertile": "地表温度三分位"}
        selected["row_label"] = selected["outcome"].map(outcome_labels).fillna(selected["outcome"]) + " / " + selected["subgroup_variable"].map(group_var_labels).fillna(selected["subgroup_variable"]) + " / " + selected["subgroup"].astype(str)
        selected = selected.sort_values("coefficient").tail(35)
        if not selected.empty:
            ax.barh(selected["row_label"], selected["coefficient"], color=np.where(selected["coefficient"] >= 0, "#326A8C", "#D28A34"), edgecolor="white", linewidth=.35)
            ax.axvline(0, color="grey", linewidth=1)
            ax.set(xlabel="M4 配对斜率：Δ健康 / Δlog1p(Aa)", title="6｜不同环境背景下的描述性关联差异")
    else:
        ax.text(.5, .5, "没有可用的亚组估计", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    figures.append(save_figure(fig, "06_environmental_heterogeneity", "亚组估计使用 M4 全样本配对及预先定义的土类、V5 K 均值组和环境三分位。分组用于描述；亚组差异仍属探索结果，多重检验在表格中按 Benjamini–Hochberg 方法校正。", data, review_dir))

    # 7. Environment-group map and selected high-response strata.
    stored_map_rows = high_groups.loc[high_groups.get("map_role", pd.Series(index=high_groups.index, dtype=object)).eq("sampled cell point; no continuous surface inference")].copy() if not high_groups.empty else pd.DataFrame()
    stored_map_rows = _valid_stored_map_points(stored_map_rows)
    data = high_groups.loc[
        high_groups["top_three_high_response_group"]
        & high_groups["exposure"].eq("Aa_20yr")
        & high_groups["outcome"].isin(PRIMARY_OUTCOMES)
    ].copy() if not high_groups.empty else pd.DataFrame()
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.8), constrained_layout=True)
    if not stored_map_rows.empty:
        map_rows = stored_map_rows.copy()
    else:
        map_model = fit_environment_model(frame, ["Elevation", "LST"], clusters=6, seed=20261026, primary_soils=[26, 27, 42])
        _, map_labels = apply_environment_model(map_model, frame, [26, 27, 42])
        map_rows = frame.loc[frame["SoilType"].isin([26, 27, 42]) & frame[["x", "y", "Elevation", "LST"]].notna().all(axis=1)].copy()
        map_rows["environment_group"] = map_labels[map_rows.index.to_numpy()]
        map_rows = map_rows.loc[map_rows["environment_group"].ge(0)]
        map_rows["cluster_within_soil"] = map_rows["environment_group"] % 6
        if len(map_rows) > 12000:
            map_rows = map_rows.sample(12000, random_state=20261026)
    if not map_rows.empty:
        group_palette = ["#315D78", "#4E8A86", "#C57645", "#A7A64C", "#6A6A9B", "#9A5870"]
        group_cmap = ListedColormap(group_palette, name="soil_stratified_kmeans_groups")
        group_norm = BoundaryNorm(np.arange(-0.5, 6.5), group_cmap.N)
        markers = {26: "o", 27: "^", 42: "s"}
        soil_values = pd.to_numeric(map_rows["SoilType"], errors="coerce")
        local_groups = pd.to_numeric(map_rows["cluster_within_soil"], errors="coerce")
        map_scatter = None
        for soil_code, marker in markers.items():
            selected = soil_values.eq(soil_code) & local_groups.notna()
            if selected.any():
                map_scatter = axes[0].scatter(
                    map_rows.loc[selected, "x"] / 1000, map_rows.loc[selected, "y"] / 1000,
                    c=local_groups.loc[selected], s=3.0, alpha=.72, cmap=group_cmap, norm=group_norm,
                    marker=marker, linewidths=0, rasterized=True,
                )
        axes[0].set(xlabel="投影坐标东向（千米）", ylabel="投影坐标北向（千米）",
                    title="高程与地表温度环境组（固定抽样）")
        axes[0].grid(color="#D9E2E6", linewidth=.55, alpha=.65)
        axes[0].set_axisbelow(True)
        axes[0].tick_params(labelsize=8, length=0)
        if map_scatter is not None:
            colorbar = fig.colorbar(map_scatter, ax=axes[0], fraction=.045, pad=.03,
                                    ticks=np.arange(0, 6), boundaries=np.arange(-.5, 6.5))
            colorbar.set_ticklabels([str(value) for value in range(1, 7)])
            colorbar.set_label("土类内环境组编号", fontsize=8)
        axes[0].legend(
            handles=[Line2D([0], [0], marker=markers[code], color="none", markerfacecolor="#536C7B",
                            markeredgecolor="white", markersize=6,
                            label=f"{code} {SOIL_TYPE_LABELS[code]}") for code in markers],
            title="点形表示土类", frameon=True, framealpha=.92, facecolor="white", edgecolor="#D9E2E6",
            fontsize=7, title_fontsize=7, loc="lower left",
        )
        x_span = float(map_rows["x"].max() - map_rows["x"].min()) / 1000
        y_span = float(map_rows["y"].max() - map_rows["y"].min()) / 1000
        target_scale = x_span * .17
        scale_candidates = [1, 2, 5, 10, 20]
        scale_km = min(scale_candidates, key=lambda value: abs(value - target_scale))
        scale_x = float(map_rows["x"].min()) / 1000 + x_span * .055
        scale_y = float(map_rows["y"].min()) / 1000 + y_span * .06
        axes[0].plot([scale_x, scale_x + scale_km], [scale_y, scale_y], color="white", linewidth=4.2, zorder=5)
        axes[0].plot([scale_x, scale_x + scale_km], [scale_y, scale_y], color="#263A46", linewidth=2.0, zorder=6)
        axes[0].text(scale_x + scale_km / 2, scale_y + y_span * .02, f"{scale_km:g} 千米",
                     ha="center", va="bottom", fontsize=7, color="#263A46", zorder=6,
                     bbox={"boxstyle": "round,pad=.15", "facecolor": "white", "edgecolor": "none", "alpha": .88})
        axes[0].annotate("N", xy=(.94, .91), xytext=(.94, .79), xycoords="axes fraction",
                         ha="center", va="center", fontsize=9, fontweight="bold",
                         arrowprops={"arrowstyle": "-|>", "color": "#263A46", "lw": 1.1})
    else:
        axes[0].set_axis_off()
    if not data.empty:
        labels = [
            f"{outcome_labels.get(str(row.outcome), str(row.outcome))} / {int(row.soil_type)} {SOIL_TYPE_LABELS.get(int(row.soil_type), '未知土类')} · 组 {int(row.cluster_within_soil) + 1}"
            for row in data.itertuples()
        ]
        axes[1].barh(labels, data["mean_outcome"], color="#D28A34", edgecolor="white", linewidth=.35)
        axes[1].set(xlabel="全样本健康指标均值", title="各优先指标均值最高的三个环境组")
        axes[1].grid(axis="x", color="#D9E2E6", linewidth=.55)
        axes[1].set_axisbelow(True)
    else:
        axes[1].text(.5, .5, "没有可用的入选环境组", ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_axis_off()
    map_export = map_rows[["cell_id", "x", "y", "SoilType", "cluster_within_soil", "environment_group"]].copy() if not map_rows.empty else pd.DataFrame()
    map_export["map_role"] = "sampled cell point; no continuous surface inference"
    export = data.copy()
    if not export.empty:
        export["plot_panel"] = "high_response_environment_groups"
    map_export["outcome"] = None
    map_export["mean_outcome"] = np.nan
    export = pd.concat([export, map_export], ignore_index=True, sort=False)
    figures.append(save_figure(fig, "07_environment_groups_map_and_high_response", "左图为有效像元的固定抽样，颜色表示土类内环境组编号、点形表示土壤类型，并使用真实投影坐标绘制；组号只在土类内部解释。右图为 20 年平均实际侵蚀下各优先指标全样本均值最高的三个环境组；2020 年结果见配套数据表。高响应环境组根据健康结果排序，仅作描述性探索，不代表经独立验证的热点，也不表示连续空间面或因果机制。", export, review_dir))

    # 8. Both requested curve families.
    data = curve_data()
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), squeeze=False)
    outcomes = ["SHI_candidate", "AN", "OM", "CEC", "pH", "SW", "BD", "MBC", "RESP"]
    for ax, outcome in zip(axes.flat, outcomes, strict=True):
        subset = data.loc[data["outcome"].eq(outcome)] if not data.empty else pd.DataFrame()
        scatter = subset.loc[subset["curve_type"].eq("fixed_plot_sample_scatter")] if not subset.empty else pd.DataFrame()
        spline = subset.loc[subset["curve_type"].eq("natural_spline_fixed_plot_sample")] if not subset.empty else pd.DataFrame()
        apparent = subset.loc[subset["curve_type"].eq("selected_full_sample_apparent_model_partial_dependence")] if not subset.empty else pd.DataFrame()
        if not scatter.empty:
            ax.scatter(scatter["exposure_value"], scatter["predicted_outcome"], s=7, alpha=.3, color="#4e79a7")
        if not spline.empty:
            ax.plot(spline["exposure_value"], spline["predicted_outcome"], color="#D28A34", linestyle="--", linewidth=1.7, label="固定绘图样本自然样条")
        if not apparent.empty:
            ax.plot(apparent["exposure_value"], apparent["predicted_outcome"], color="#326A8C", linewidth=1.7, label="全样本择优表观模型")
        ax.set_title(outcome_labels.get(outcome, outcome), loc="left", fontweight="semibold")
        ax.set_xlabel("20 年平均实际侵蚀 Aa（原始栅格值）")
        ax.set_ylabel("健康预测值")
        ax.grid(axis="y", alpha=.2)
        if ax is axes.flat[0]:
            ax.legend(fontsize=7, frameon=False)
    figures.append(save_figure(fig, "08_health_response_curves", "灰蓝色散点来自固定的 10,000 行绘图样本；每项结局使用其完整观测。橙色虚线为该样本的自然样条描述曲线。蓝色实线为早先 Oracle 全样本表观模型在环境变量中位数条件下的 Aa_20yr 偏依赖曲线，曾在 13 类模型的 1,024 次试验中按表观 R²择优。模型类型与完整样本量见图源数据。两条曲线的样本范围不同，均不是独立验证曲线。", data, review_dir))

    # 9. Pair distance and robustness diagnostics.
    data = full_pairs.loc[full_pairs["method"].eq("M4")].copy() if not full_pairs.empty else pd.DataFrame()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    if not data.empty:
        axes[0].scatter(data["environmental_distance"], data["geographic_distance"], s=5, alpha=.2, c=np.where(data["erosion_low"].eq(0), "#D28A34", "#326A8C"), rasterized=True)
        axes[0].set(xlabel="V5 标准化环境距离", ylabel="配对地理距离（米）", title="配对距离诊断")
        for label, group in data.groupby("exposure", sort=True):
            axes[1].hist(group["delta_log1p_erosion"], bins=40, alpha=.52, label=exposure_labels.get(label, label), color="#326A8C" if label == "Aa_20yr" else "#D28A34")
        axes[1].set(xlabel="Δlog1p(Aa)", ylabel="配对数量", title="匹配配对的侵蚀差异")
        axes[1].legend(frameon=False, title="侵蚀时期")
    else:
        axes[0].set_axis_off()
        axes[1].set_axis_off()
    figures.append(save_figure(fig, "09_pair_distance_and_exposure_diagnostics", "每个点代表一个全样本 M4 配对；金色点表示低侵蚀端为零的配对。环境距离和地理距离用于描述匹配质量；完整空间留出表现见图 4。", data[[c for c in ["method", "exposure", "pair_id", "environmental_distance", "geographic_distance", "erosion_low", "erosion_high", "delta_log1p_erosion"] if c in data]], review_dir))

    # 10. Aa periods, same pixels for cell-level comparison.
    data = exposure_comparison.loc[
        exposure_comparison.get("analysis_type", pd.Series(index=exposure_comparison.index, dtype=object)).eq(
            "cell_level_common_sample_10_seed_summary"
        )
    ].copy() if not exposure_comparison.empty else pd.DataFrame()
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_data = data.loc[data["outcome"].isin(PRIMARY_OUTCOMES)] if not data.empty else pd.DataFrame()
    if not plot_data.empty:
        plot_data["plot_delta_r2"] = pd.to_numeric(plot_data.get("delta_r2_median"), errors="coerce")
        if plot_data["plot_delta_r2"].notna().any():
            wide = plot_data.groupby(["outcome", "exposure"], as_index=False)["plot_delta_r2"].median().pivot(index="outcome", columns="exposure", values="plot_delta_r2")
            wide.rename(columns=exposure_labels).plot(kind="bar", ax=ax, color=["#326A8C", "#D28A34"], edgecolor="white", linewidth=.4)
            ax.axhline(0, color="grey", linestyle="--", linewidth=1)
            ax.set(ylabel="相同像元下的 ΔR²", title="10｜共同像元上的两期实际侵蚀比较")
            ax.legend(title="侵蚀时期", frameon=False)
            ax.set_xticklabels([outcome_labels.get(str(t.get_text()), str(t.get_text())) for t in ax.get_xticklabels()], rotation=25, ha="right")
    else:
        ax.text(.5, .5, "没有可用的共同样本时期比较", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    figures.append(save_figure(fig, "10_exposure_period_comparison", "像元层比较要求两期使用同一批完整像元、V5 环境变量、OLS 设计和空间随机种子。全样本匹配配对按侵蚀时期分别构建，因此配套表单列展示，不视为两期完全相同的配对样本。", plot_data, review_dir))

    write_json(review_dir / "figure_index.json", figures)
    return figures


if __name__ == "__main__":
    main()








