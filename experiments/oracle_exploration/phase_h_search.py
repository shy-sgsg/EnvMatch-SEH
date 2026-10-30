"""Bounded matched-only Oracle searches for Phase H tasks H3 and H4."""

from __future__ import annotations

import ast
import importlib.util
import gc
import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from scipy.spatial import cKDTree

from experiments.oracle_exploration.matching_search import run_matching_candidate
from experiments.oracle_exploration.nonlinear_search import MODEL_FAMILIES, fit_model_family
from experiments.oracle_exploration.phase_h_metrics import direction_from_curve, regression_metrics

MODEL_SCHEDULE_VERSION = "phase_h_multifamily_bounded_v2"


def available_model_families(config: Mapping[str, Any]) -> list[str]:
    """Resolve configured families plus installed optional tree libraries."""

    families = list(config.get("models", {}).get("families", MODEL_FAMILIES))
    for optional_name, module_name in (("xgboost", "xgboost"), ("lightgbm", "lightgbm")):
        if importlib.util.find_spec(module_name) is not None and optional_name not in families:
            families.append(optional_name)
    return families


def assign_spatial_roles(
    block_ids: Sequence[Any],
    *,
    seed: int,
    fractions: tuple[float, float, float] = (0.60, 0.20, 0.20),
) -> dict[str, str]:
    """Assign each complete spatial block to one deterministic role."""

    blocks = sorted({str(value) for value in block_ids if pd.notna(value)})
    proportions = np.asarray(fractions, dtype=float)
    if len(blocks) < 3:
        raise ValueError("spatial splits require at least three non-empty blocks")
    if proportions.shape != (3,) or not np.isfinite(proportions).all() or (proportions <= 0).any():
        raise ValueError("spatial split fractions must contain three positive finite values")
    if not np.isclose(proportions.sum(), 1.0):
        raise ValueError("spatial split fractions must sum to one")
    rng = np.random.default_rng(int(seed))
    shuffled = np.asarray(blocks, dtype=object)[rng.permutation(len(blocks))]
    n_train = max(1, int(np.floor(len(blocks) * proportions[0])))
    n_validation = max(1, int(np.floor(len(blocks) * proportions[1])))
    if n_train + n_validation >= len(blocks):
        n_train = len(blocks) - 2
        n_validation = 1
    roles = np.full(len(blocks), "test", dtype=object)
    roles[:n_train] = "train"
    roles[n_train:n_train + n_validation] = "validation"
    return dict(zip(shuffled.astype(str), roles.astype(str), strict=True))


def enrich_pair_spatial_coordinates(
    pairs: pd.DataFrame,
    cells: pd.DataFrame,
    *,
    block_size_m: float = 5000.0,
) -> pd.DataFrame:
    """Attach endpoint and midpoint coordinates, plus same-block eligibility."""

    required_pairs = {"pair_id", "high_erosion_cell_id", "low_erosion_cell_id"}
    required_cells = {"cell_id", "x", "y"}
    missing = sorted(required_pairs - set(pairs.columns))
    missing_cells = sorted(required_cells - set(cells.columns))
    if missing or missing_cells:
        raise ValueError(f"missing pair columns {missing}; missing cell columns {missing_cells}")
    if not np.isfinite(float(block_size_m)) or block_size_m <= 0:
        raise ValueError("block_size_m must be positive and finite")
    if pairs["pair_id"].isna().any() or pairs["pair_id"].duplicated().any():
        raise ValueError("pair_id must be unique and non-null")
    coordinates = cells.set_index("cell_id")[["x", "y"]]
    result = pairs.copy()
    for role, key in (("high", "high_erosion_cell_id"), ("low", "low_erosion_cell_id")):
        endpoint = coordinates.reindex(result[key].to_numpy())
        values = endpoint.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"pair endpoints in {key} need finite coordinates")
        result[f"x_{role}"] = values[:, 0]
        result[f"y_{role}"] = values[:, 1]
        result[f"block_{role}"] = (
            np.floor(values[:, 0] / block_size_m).astype(np.int64).astype(str)
            + "_"
            + np.floor(values[:, 1] / block_size_m).astype(np.int64).astype(str)
        )
    result["same_spatial_block"] = result["block_high"].eq(result["block_low"])
    result["x_mid"] = (result["x_high"] + result["x_low"]) / 2.0
    result["y_mid"] = (result["y_high"] + result["y_low"]) / 2.0
    result["spatial_block_id"] = (
        np.floor(result["x_mid"] / block_size_m).astype(np.int64).astype(str)
        + "_"
        + np.floor(result["y_mid"] / block_size_m).astype(np.int64).astype(str)
    )
    return result


def filter_spatial_buffer(
    training: pd.DataFrame,
    protected: pd.DataFrame,
    *,
    x_column: str,
    y_column: str,
    buffer_m: float,
) -> pd.DataFrame:
    """Drop training rows within the requested distance of held-out rows."""

    if not np.isfinite(float(buffer_m)) or buffer_m < 0:
        raise ValueError("buffer_m must be nonnegative and finite")
    required = {x_column, y_column}
    if not required.issubset(training.columns) or not required.issubset(protected.columns):
        raise ValueError(f"training and protected frames must include {sorted(required)}")
    if training.empty or protected.empty or buffer_m == 0:
        return training.copy()
    training_coordinates = training[[x_column, y_column]].to_numpy(dtype=float)
    protected_coordinates = protected[[x_column, y_column]].to_numpy(dtype=float)
    if not np.isfinite(training_coordinates).all() or not np.isfinite(protected_coordinates).all():
        raise ValueError("spatial buffer coordinates must be finite")
    nearest_distances = cKDTree(protected_coordinates).query(training_coordinates, k=1)[0]
    return training.loc[nearest_distances > float(buffer_m)].copy()


def prepare_spatial_split(
    frame: pd.DataFrame,
    *,
    unit_id_column: str,
    block_column: str,
    x_column: str,
    y_column: str,
    seed: int,
    buffer_m: float,
    block_ids: Sequence[Any] | None = None,
    fractions: tuple[float, float, float] = (0.60, 0.20, 0.20),
) -> dict[str, Any]:
    """Create disjoint whole-block roles and remove buffered train units."""

    required = {unit_id_column, block_column, x_column, y_column}
    if not required.issubset(frame.columns):
        raise ValueError(f"spatial split is missing columns: {sorted(required - set(frame.columns))}")
    if frame[unit_id_column].isna().any() or frame[unit_id_column].duplicated().any():
        raise ValueError(f"{unit_id_column} must be unique and non-null within a spatial split")
    if frame[block_column].isna().any():
        raise ValueError(f"{block_column} cannot contain nulls")
    role_blocks = block_ids if block_ids is not None else frame[block_column].astype(str).unique()
    role_assignment = assign_spatial_roles(role_blocks, seed=seed, fractions=fractions)
    prepared = frame.copy()
    prepared[block_column] = prepared[block_column].astype(str)
    prepared["__role"] = prepared[block_column].map(role_assignment)
    if prepared["__role"].isna().any():
        raise ValueError("spatial split contains a block absent from the role assignment")
    train_before = prepared.loc[prepared["__role"].eq("train")].drop(columns="__role")
    validation = prepared.loc[prepared["__role"].eq("validation")].drop(columns="__role")
    test = prepared.loc[prepared["__role"].eq("test")].drop(columns="__role")
    protected = pd.concat([validation, test], ignore_index=True)
    train = filter_spatial_buffer(
        train_before,
        protected,
        x_column=x_column,
        y_column=y_column,
        buffer_m=buffer_m,
    )
    details = {
        "seed": int(seed),
        "buffer_m": float(buffer_m),
        "n_blocks_total": int(len(role_assignment)),
        "n_blocks_train": int(sum(role == "train" for role in role_assignment.values())),
        "n_blocks_validation": int(sum(role == "validation" for role in role_assignment.values())),
        "n_blocks_test": int(sum(role == "test" for role in role_assignment.values())),
        "n_train_before_buffer": int(len(train_before)),
        "n_train": int(len(train)),
        "n_train_buffered_out": int(len(train_before) - len(train)),
        "n_validation": int(len(validation)),
        "n_test": int(len(test)),
    }
    return {
        "train": train,
        "validation": validation,
        "test": test,
        "role_assignment": role_assignment,
        "details": details,
    }


def common_pair_period_sample(
    pairs_20yr: pd.DataFrame,
    pairs_2020: pd.DataFrame,
    cells: pd.DataFrame,
    *,
    outcome: str,
    block_size_m: float = 5000.0,
) -> pd.DataFrame:
    """Build common endpoint pairs oriented by Aa_20yr for fair period comparison."""

    pair_columns = {"pair_id", "high_erosion_cell_id", "low_erosion_cell_id"}
    missing_20yr = sorted(pair_columns - set(pairs_20yr.columns))
    missing_2020 = sorted(pair_columns - set(pairs_2020.columns))
    required_cells = {"cell_id", "x", "y", "Aa_20yr", "Aa_2020", outcome}
    missing_cells = sorted(required_cells - set(cells.columns))
    if missing_20yr or missing_2020 or missing_cells:
        raise ValueError(
            f"missing Aa_20yr pair columns {missing_20yr}; missing Aa_2020 pair columns {missing_2020}; "
            f"missing cell columns {missing_cells}"
        )
    if not np.isfinite(float(block_size_m)) or block_size_m <= 0:
        raise ValueError("block_size_m must be positive and finite")

    def keyed_pairs(frame: pd.DataFrame) -> dict[str, tuple[Any, Any, Any]]:
        if frame["pair_id"].isna().any() or frame["pair_id"].duplicated().any():
            raise ValueError("period pair_id values must be unique and non-null")
        keyed: dict[str, tuple[Any, Any, Any]] = {}
        for pair_id, high_id, low_id in frame[
            ["pair_id", "high_erosion_cell_id", "low_erosion_cell_id"]
        ].itertuples(index=False, name=None):
            endpoints = sorted((str(high_id), str(low_id)))
            keyed[f"{endpoints[0]}|{endpoints[1]}"] = (pair_id, high_id, low_id)
        return keyed

    keyed_20yr = keyed_pairs(pairs_20yr)
    keyed_2020 = keyed_pairs(pairs_2020)
    common_keys = sorted(set(keyed_20yr).intersection(keyed_2020))
    if not common_keys:
        return pd.DataFrame()
    cell_index = cells.set_index("cell_id", drop=False)
    records: list[dict[str, Any]] = []
    for key in common_keys:
        pair_id_20yr, a_id, b_id = keyed_20yr[key]
        pair_id_2020, _, _ = keyed_2020[key]
        if a_id not in cell_index.index or b_id not in cell_index.index:
            raise ValueError(f"common pair endpoints missing from cells: {key}")
        a, b = cell_index.loc[a_id], cell_index.loc[b_id]
        aa_a = float(pd.to_numeric(a["Aa_20yr"], errors="coerce"))
        aa_b = float(pd.to_numeric(b["Aa_20yr"], errors="coerce"))
        if aa_a > aa_b or (aa_a == aa_b and str(a_id) < str(b_id)):
            high_id, low_id = a_id, b_id
            high, low = a, b
        else:
            high_id, low_id = b_id, a_id
            high, low = b, a
        high_20yr = float(pd.to_numeric(high["Aa_20yr"], errors="coerce"))
        low_20yr = float(pd.to_numeric(low["Aa_20yr"], errors="coerce"))
        high_2020 = float(pd.to_numeric(high["Aa_2020"], errors="coerce"))
        low_2020 = float(pd.to_numeric(low["Aa_2020"], errors="coerce"))
        high_health = float(pd.to_numeric(high[outcome], errors="coerce"))
        low_health = float(pd.to_numeric(low[outcome], errors="coerce"))
        x_high, y_high = float(high["x"]), float(high["y"])
        x_low, y_low = float(low["x"]), float(low["y"])
        x_mid, y_mid = (x_high + x_low) / 2.0, (y_high + y_low) / 2.0
        block_high = f"{int(np.floor(x_high / block_size_m))}_{int(np.floor(y_high / block_size_m))}"
        block_low = f"{int(np.floor(x_low / block_size_m))}_{int(np.floor(y_low / block_size_m))}"
        records.append({
            "canonical_pair_key": key,
            "pair_id_20yr": pair_id_20yr,
            "pair_id_2020": pair_id_2020,
            "high_erosion_cell_id": high_id,
            "low_erosion_cell_id": low_id,
            "Aa_high_20yr": high_20yr,
            "Aa_low_20yr": low_20yr,
            "Aa_high_2020": high_2020,
            "Aa_low_2020": low_2020,
            "delta_log1p_Aa_20yr": float(np.log1p(high_20yr) - np.log1p(low_20yr)),
            "delta_log1p_Aa_2020": float(np.log1p(high_2020) - np.log1p(low_2020)),
            f"high_{outcome}": high_health,
            f"low_{outcome}": low_health,
            f"delta_{outcome}": high_health - low_health,
            "x_high": x_high,
            "y_high": y_high,
            "x_low": x_low,
            "y_low": y_low,
            "x_mid": x_mid,
            "y_mid": y_mid,
            "same_spatial_block": block_high == block_low,
            "spatial_block_id": f"{int(np.floor(x_mid / block_size_m))}_{int(np.floor(y_mid / block_size_m))}",
        })
    return pd.DataFrame.from_records(records)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def build_phase_h_candidate_designs(
    config: Mapping[str, Any], exposure: str, *, seed: int, stage: str = "stage1"
) -> list[dict[str, Any]]:
    """Draw a deterministic, bounded candidate set that covers every configured level."""

    search = config["matching_search"]
    if exposure not in config["exposures"]:
        raise ValueError(f"Unsupported Phase H exposure: {exposure}")
    count_key = "stage1_max_candidates_per_exposure" if stage == "stage1" else "stage2_max_candidates_per_exposure"
    count = int(search[count_key])
    if count < 1:
        raise ValueError(f"{count_key} must be positive")
    vectors = search.get("environment_vectors", {})
    if not vectors:
        raise ValueError("matching_search.environment_vectors must not be empty")
    vector_items = [(str(name), tuple(map(str, columns))) for name, columns in vectors.items()]
    source_methods = list(map(str, search.get("source_methods", ("M3", "M4"))))
    distance_methods = list(map(str, search.get("distance_methods", (
        "direct_nearest_neighbor", "standardized_euclidean", "mahalanobis", "weighted_euclidean"
    ))))
    cluster_methods = list(map(str, search.get("cluster_methods", ("none", "kmeans", "agglomerative"))))
    cluster_counts = [int(value) for value in search.get("cluster_counts", (4, 6, 8, 10, 12))]
    calipers = [float(value) for value in search["environment_calipers"]]
    radii = [int(value) for value in search["geographic_radii_m"]]
    rank_gaps = [float(value) for value in search["erosion_rank_gaps"]]
    exposure_rules = list(map(str, search.get("exposure_rules", ("within_group_rank",))))
    if not all((source_methods, distance_methods, cluster_methods, cluster_counts, calipers, radii, rank_gaps, exposure_rules)):
        raise ValueError("Phase H candidate domains must be non-empty")

    rng = np.random.default_rng(seed)

    def draw_batch() -> list[dict[str, Any]]:
        batch: list[dict[str, Any]] = []
        signatures: set[tuple[Any, ...]] = set()
        max_draws = max(500, count * 100)
        attempts = 0
        while len(batch) < count and attempts < max_draws:
            attempts += 1
            vector_name, variables = vector_items[int(rng.integers(0, len(vector_items)))]
            reference_method = source_methods[int(rng.integers(0, len(source_methods)))]
            distance_method = distance_methods[int(rng.integers(0, len(distance_methods)))]
            cluster_method = cluster_methods[int(rng.integers(0, len(cluster_methods)))]
            cluster_count = 0 if cluster_method == "none" else cluster_counts[int(rng.integers(0, len(cluster_counts)))]
            caliper = calipers[int(rng.integers(0, len(calipers)))]
            radius = radii[int(rng.integers(0, len(radii)))]
            rank_gap = rank_gaps[int(rng.integers(0, len(rank_gaps)))]
            exposure_rule = exposure_rules[int(rng.integers(0, len(exposure_rules)))]
            weights = None
            if distance_method == "weighted_euclidean":
                raw = rng.lognormal(mean=0.0, sigma=0.5, size=len(variables))
                raw /= raw.mean()
                weights = dict(zip(variables, raw.tolist(), strict=True))
            signature = (
                vector_name, reference_method, distance_method, cluster_method, cluster_count,
                caliper, radius, rank_gap, exposure_rule,
                tuple(sorted((weights or {}).items())),
            )
            if signature in signatures:
                continue
            signatures.add(signature)
            batch.append({
                "exposure": exposure,
                "reference_method": reference_method,
                "method": reference_method,
                "environment_variables": list(variables),
                "vector_name": vector_name,
                "distance_method": distance_method,
                "cluster_method": cluster_method,
                "cluster_count": cluster_count,
                "environment_weights": weights,
                "environment_caliper": caliper,
                "geographic_radius_m": radius,
                "erosion_rank_gap": rank_gap,
                "exposure_rule": exposure_rule,
            })
        return batch

    must_cover = count >= max(
        len(vector_items), len(source_methods), len(distance_methods), len(cluster_methods),
        len(calipers), len(radii), len(rank_gaps), len(exposure_rules),
    )
    batch: list[dict[str, Any]] = []
    for _ in range(1000):
        batch = draw_batch()
        if len(batch) != count:
            continue
        realized_cluster_counts = {row["cluster_count"] for row in batch if row["cluster_method"] != "none"}
        levels_covered = (
            {row["vector_name"] for row in batch} >= {name for name, _ in vector_items}
            and {row["reference_method"] for row in batch} >= set(source_methods)
            and {row["distance_method"] for row in batch} >= set(distance_methods)
            and {row["cluster_method"] for row in batch} >= set(cluster_methods)
            and {row["environment_caliper"] for row in batch} >= set(calipers)
            and {row["geographic_radius_m"] for row in batch} >= set(radii)
            and {row["erosion_rank_gap"] for row in batch} >= set(rank_gaps)
            and {row["exposure_rule"] for row in batch} >= set(exposure_rules)
            and (len(realized_cluster_counts) >= min(count, len(cluster_counts)) if cluster_counts else True)
        )
        if levels_covered or not must_cover:
            break
    if len(batch) != count or (must_cover and not levels_covered):
        raise ValueError("Could not draw a unique candidate set covering each configured matching level")

    designs = []
    for index, row in enumerate(batch, start=1):
        designs.append({
            **row,
            "candidate_id": f"{stage}_{exposure}_{index:03d}",
            "experiment_id": f"{config['run_id']}-{stage}-{exposure}-{index:03d}",
            "experiment_type": f"phase_h_{stage}_local_matching_search",
            "outcome_guided_selection": True,
            "selection_mode": "outcome_guided_oracle",
            "selection_outcomes": list(config["outcomes"]),
            "selection_objective": "multi_family_apparent_r2_and_negative_direction_screen_then_full_refit",
            "input_run_id": str(config["input_run_id"]),
            "random_seed": int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32)),
            "stage": stage,
        })
    return designs


def _fraction_label(fraction: float) -> str:
    return f"{int(round(float(fraction) * 100)):02d}pct"


def build_phase_h_pair_subsets(
    pairs: pd.DataFrame,
    *,
    outcome: str,
    retention_fractions: Sequence[float],
    include_oracle_residual: bool = True,
) -> dict[str, pd.DataFrame]:
    """Build explicit pair-type, environment-retention, and separately marked Oracle subsets."""

    if not 0 < len(pairs):
        return {"all": pairs.copy()}
    data = pairs.copy()
    data.attrs["outcome_guided_selection"] = False
    if "both_positive" not in data:
        high = pd.to_numeric(data.get("Aa_high", data.get("erosion_high")), errors="coerce")
        low = pd.to_numeric(data.get("Aa_low", data.get("erosion_low")), errors="coerce")
        data["both_positive"] = high.gt(0) & low.gt(0)
        data["zero_positive"] = low.le(0) & high.gt(0)
    if "high_erosion_gap_q4" not in data:
        x_name = "delta_log1p_Aa" if "delta_log1p_Aa" in data else "delta_log1p_erosion"
        if x_name in data:
            values = pd.to_numeric(data[x_name], errors="coerce")
            data["high_erosion_gap_q4"] = values.ge(values.quantile(0.75))
        else:
            data["high_erosion_gap_q4"] = False
    subsets: dict[str, pd.DataFrame] = {"all": data.copy()}
    for name, flag in (
        ("both_positive", "both_positive"), ("zero_positive", "zero_positive"),
        ("high_erosion_gap_q4", "high_erosion_gap_q4"),
    ):
        selected = data.loc[data[flag].fillna(False)].copy()
        selected.attrs["outcome_guided_selection"] = False
        selected.attrs["subset_rule"] = flag
        subsets[name] = selected

    if "environmental_distance" in data:
        distance = pd.to_numeric(data["environmental_distance"], errors="coerce")
        valid = data.loc[np.isfinite(distance)].copy()
        valid["__distance"] = pd.to_numeric(valid["environmental_distance"], errors="coerce")
        for fraction in sorted(set(float(value) for value in retention_fractions), reverse=True):
            if not 0 < fraction <= 1:
                raise ValueError("retention fractions must be in (0, 1]")
            n_keep = max(1, int(math.ceil(len(valid) * fraction))) if len(valid) else 0
            selected = valid.sort_values(["__distance", "pair_id" if "pair_id" in valid else "__distance"], kind="stable").head(n_keep).drop(columns="__distance")
            selected.attrs["outcome_guided_selection"] = False
            selected.attrs["subset_rule"] = "smallest_environmental_distance"
            selected.attrs["retention_fraction"] = fraction
            subsets[f"environment_distance_low_{_fraction_label(fraction)}"] = selected

    exposure_gap_column = "delta_log1p_Aa" if "delta_log1p_Aa" in data else "delta_log1p_erosion"
    if exposure_gap_column in data:
        gap = pd.to_numeric(data[exposure_gap_column], errors="coerce")
        valid = data.loc[np.isfinite(gap)].copy()
        valid["__gap"] = pd.to_numeric(valid[exposure_gap_column], errors="coerce")
        for fraction in sorted(set(float(value) for value in retention_fractions), reverse=True):
            n_keep = max(1, int(math.ceil(len(valid) * fraction))) if len(valid) else 0
            selected = valid.sort_values(["__gap", "pair_id" if "pair_id" in valid else "__gap"], ascending=[False, True], kind="stable").head(n_keep).drop(columns="__gap")
            selected.attrs["outcome_guided_selection"] = False
            selected.attrs["subset_rule"] = "largest_exposure_gap"
            selected.attrs["retention_fraction"] = fraction
            subsets[f"exposure_gap_high_{_fraction_label(fraction)}"] = selected

    if "environment_group" in data and data["environment_group"].dropna().astype(str).nunique() >= 2:
        groups = data.loc[data["environment_group"].notna(), "environment_group"].astype(str)
        for label in sorted(groups.unique()):
            selected = data.loc[data["environment_group"].astype(str).eq(label)].copy()
            if len(selected) >= 30:
                selected.attrs["outcome_guided_selection"] = False
                selected.attrs["subset_rule"] = "candidate_environment_group"
                subsets[f"environment_group={label}"] = selected

    delta_y = f"delta_{outcome}"
    x_name = "delta_log1p_Aa" if "delta_log1p_Aa" in data else "delta_log1p_erosion"
    if include_oracle_residual and delta_y in data and x_name in data:
        x = pd.to_numeric(data[x_name], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(data[delta_y], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        if finite.sum() >= 10 and np.ptp(x[finite]) > 0:
            coefficient = np.polyfit(x[finite], y[finite], deg=1)
            residual = np.full(len(data), np.nan)
            residual[finite] = y[finite] - np.polyval(coefficient, x[finite])
            data["__oracle_residual"] = residual
            valid = data.loc[np.isfinite(data["__oracle_residual"])].copy()
            for tail, ascending in (("bottom", True), ("top", False)):
                n_keep = max(1, int(math.ceil(len(valid) * 0.20)))
                selected = valid.sort_values("__oracle_residual", ascending=ascending, kind="stable").head(n_keep).drop(columns="__oracle_residual")
                selected.attrs["outcome_guided_selection"] = True
                selected.attrs["subset_rule"] = f"extreme_{tail}_linear_residual"
                selected.attrs["retention_fraction"] = 0.20
                subsets[f"oracle_extreme_residual_{tail}_20pct"] = selected
    for name, selected in subsets.items():
        selected.attrs.setdefault("outcome_guided_selection", False)
        selected.attrs.setdefault("subset_rule", name)
    return subsets


def _safe_corr(x: np.ndarray, y: np.ndarray, function: Any) -> float:
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x[finite], dtype=float), np.asarray(y[finite], dtype=float)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(function(x, y).statistic)


def _prediction_support_curve(
    x: np.ndarray, prediction: np.ndarray, *, max_points: int
) -> tuple[np.ndarray, np.ndarray]:
    """Summarize fitted predictions over observed exposure support without another model fit."""

    frame = pd.DataFrame({"x": np.asarray(x, dtype=float), "prediction": np.asarray(prediction, dtype=float)})
    frame = frame.loc[np.isfinite(frame).all(axis=1)].copy()
    unique = int(frame["x"].nunique())
    if unique < 2 or len(frame) < 2:
        return np.array([], dtype=float), np.array([], dtype=float)
    bins = min(int(max_points), unique, len(frame))
    frame["support_bin"] = pd.qcut(frame["x"], q=bins, labels=False, duplicates="drop")
    curve = frame.groupby("support_bin", observed=True, sort=True).agg(x=("x", "mean"), prediction=("prediction", "mean"))
    return curve["x"].to_numpy(dtype=float), curve["prediction"].to_numpy(dtype=float)


def build_phase_h_leaderboards(trials: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Create seven outcome/unit-separated audit and selection tables from matched-only trials."""

    required = {"estimand", "exposure", "outcome", "r_squared", "direction", "candidate_id", "model_family", "n"}
    missing = sorted(required - set(trials.columns))
    if missing:
        raise ValueError(f"Phase H trials are missing leaderboard columns: {missing}")
    population = trials.get("analysis_population", pd.Series("phase_h_matched_only", index=trials.index)).astype(str)
    matched = trials.loc[trials["estimand"].isin({"matched_pixel", "matched_pair"}) & population.eq("phase_h_matched_only")].copy()
    matched["analysis_population"] = "phase_h_matched_only"
    if "spearman_rho" not in matched:
        matched["spearman_rho"] = np.nan
    if "evaluation_set" not in matched:
        matched["evaluation_set"] = "apparent"
    group_keys = ["estimand", "exposure", "outcome"]
    valid = matched.loc[pd.to_numeric(matched["r_squared"], errors="coerce").notna()].copy()
    primary = valid.loc[valid["stage"].eq("stage2_full_sample")].copy() if "stage" in valid else valid

    def first_by(frame: pd.DataFrame, order_columns: list[str], ascending: list[bool]) -> pd.DataFrame:
        if frame.empty:
            return frame.copy()
        ordered = frame.sort_values([*group_keys, *order_columns, "candidate_id"], ascending=[True] * 3 + ascending + [True], kind="stable")
        return ordered.drop_duplicates(group_keys, keep="first").reset_index(drop=True)

    top_apparent = first_by(
        primary.loc[primary["evaluation_set"].eq("apparent")], ["r_squared"], [False]
    )
    negative = primary.loc[primary["direction"].eq("negative")].copy()
    negative_r2 = first_by(negative, ["r_squared"], [False])
    strongest_negative = first_by(
        primary.loc[pd.to_numeric(primary["spearman_rho"], errors="coerce").lt(0)],
        ["spearman_rho"], [True],
    )
    pixel_r2 = first_by(primary.loc[primary["estimand"].eq("matched_pixel")], ["r_squared"], [False])
    spatial = primary.loc[primary.get("evaluation_set", pd.Series(index=primary.index, dtype=object)).eq("spatial_test")].copy()
    spatial_best = first_by(spatial, ["r_squared"], [False]) if len(spatial) else pd.DataFrame(columns=matched.columns)
    environment = primary.loc[primary.get("subset_name", pd.Series(index=primary.index, dtype=object)).astype(str).str.startswith("environment_group=")].copy()
    if len(environment):
        environment = environment.sort_values([*group_keys, "r_squared"], ascending=[True, True, True, False], kind="stable")
        environment["environment_group_rank"] = environment.groupby(group_keys, sort=False).cumcount() + 1
    audit_keys = [*group_keys, *( ["stage"] if "stage" in valid else [] )]
    audit = valid.sort_values([*audit_keys, "r_squared", "candidate_id"], ascending=[True] * len(audit_keys) + [False, True], kind="stable")
    audit["apparent_r2_rank"] = audit.groupby(audit_keys, sort=False).cumcount() + 1
    return {
        "leaderboard_1_top_apparent_r2": top_apparent,
        "leaderboard_2_highest_negative_direction_r2": negative_r2,
        "leaderboard_3_strongest_negative_association": strongest_negative,
        "leaderboard_4_highest_matched_pixel_r2": pixel_r2,
        "leaderboard_5_best_spatial_validation": spatial_best,
        "leaderboard_6_environment_group_association_fit_stability": environment,
        "leaderboard_7_all_candidate_ranked_audit": audit,
    }


def build_correlation_r2_analysis(trials: pd.DataFrame) -> pd.DataFrame:
    """Expose observed association, squared correlation, and model R² side by side."""

    population = trials.get("analysis_population", pd.Series("phase_h_matched_only", index=trials.index)).astype(str)
    selected = trials.loc[trials["estimand"].isin({"matched_pixel", "matched_pair"}) & population.eq("phase_h_matched_only")].copy()
    selected["analysis_population"] = "phase_h_matched_only"
    selected["pearson_r_squared"] = pd.to_numeric(selected.get("pearson_r"), errors="coerce") ** 2
    selected["spearman_rho_squared"] = pd.to_numeric(selected.get("spearman_rho"), errors="coerce") ** 2
    selected["model_r2_minus_pearson_r2"] = pd.to_numeric(selected["r_squared"], errors="coerce") - selected["pearson_r_squared"]
    selected["model_r2_minus_spearman_rho2"] = pd.to_numeric(selected["r_squared"], errors="coerce") - selected["spearman_rho_squared"]
    return selected


def _complete_model_data(
    frame: pd.DataFrame, *, outcome: str, exposure: str, numeric_features: Sequence[str]
) -> pd.DataFrame:
    required = list(dict.fromkeys([outcome, exposure, *numeric_features]))
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"model input is missing columns: {missing}")
    selected = frame.loc[:, required].copy()
    for column in required:
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
    selected = selected.loc[np.isfinite(selected.to_numpy(dtype=float)).all(axis=1)].copy()
    return selected


def _fit_predictions(
    train: pd.DataFrame,
    score: pd.DataFrame,
    *,
    outcome: str,
    exposure: str,
    features: Sequence[str],
    family: str,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if family not in {"xgboost", "lightgbm"}:
        return fit_model_family(
            train, score, outcome, exposure, features, model_family=family, random_state=seed
        )
    if family == "xgboost":
        import xgboost

        estimator = xgboost.XGBRegressor(
            objective="reg:squarederror", n_estimators=150, max_depth=4,
            learning_rate=0.04, subsample=0.85, colsample_bytree=0.85,
            reg_lambda=1.0, n_jobs=1, random_state=seed, verbosity=0,
        )
        params = {"estimator": "XGBRegressor", "n_estimators": 150, "max_depth": 4, "learning_rate": 0.04, "random_state": seed}
    else:
        import lightgbm

        estimator = lightgbm.LGBMRegressor(
            objective="regression", n_estimators=150, num_leaves=15,
            learning_rate=0.04, subsample=0.85, colsample_bytree=0.85,
            reg_lambda=1.0, n_jobs=1, random_state=seed, verbosity=-1,
        )
        params = {"estimator": "LGBMRegressor", "n_estimators": 150, "num_leaves": 15, "learning_rate": 0.04, "random_state": seed}
    estimator.fit(train.loc[:, list(features)].to_numpy(dtype=float), train[outcome].to_numpy(dtype=float))
    return np.asarray(estimator.predict(score.loc[:, list(features)].to_numpy(dtype=float)), dtype=float), params


def _model_trial(
    frame: pd.DataFrame,
    *,
    config: Mapping[str, Any],
    candidate: Mapping[str, Any],
    estimand: str,
    outcome: str,
    subset_name: str,
    model_family: str,
    fit_scope: str,
    stage: str,
    retention_fraction: float | None = None,
) -> dict[str, Any]:
    exposure = str(candidate["exposure"])
    if estimand == "matched_pixel":
        x_column, y_column = exposure, outcome
        environment_columns = [name for name in candidate.get("environment_variables", ()) if name in frame]
    else:
        x_column, y_column = "delta_log1p_Aa", f"delta_{outcome}"
        environment_columns = [f"delta_{name}" for name in candidate.get("environment_variables", ()) if f"delta_{name}" in frame]
    features = [x_column]
    if fit_scope == "environment_adjusted":
        features.extend(environment_columns)
    base = {
        "stage": stage,
        "model_schedule_version": MODEL_SCHEDULE_VERSION,
        "analysis_population": "phase_h_matched_only",
        "estimand": estimand,
        "exposure": exposure,
        "outcome": outcome,
        "subset_name": subset_name,
        "retention_fraction": 1.0 if subset_name == "all" else retention_fraction,
        "candidate_id": candidate.get("candidate_id"),
        "selection_parent_candidate_id": candidate.get("selection_parent_candidate_id"),
        "reference_method": candidate.get("reference_method"),
        "vector_name": candidate.get("vector_name"),
        "distance_method": candidate.get("distance_method"),
        "cluster_method": candidate.get("cluster_method"),
        "cluster_count": candidate.get("cluster_count"),
        "environment_caliper": candidate.get("environment_caliper"),
        "geographic_radius_m": candidate.get("geographic_radius_m"),
        "erosion_rank_gap": candidate.get("erosion_rank_gap"),
        "exposure_rule": candidate.get("exposure_rule"),
        "environment_variables_json": _json(candidate.get("environment_variables", ())),
        "environment_weights_json": _json(candidate.get("environment_weights")),
        "stage1_rank": candidate.get("stage1_rank"),
        "stage1_selection_score": candidate.get("stage1_selection_score"),
        "n_matched_pairs_candidate": candidate.get("n_matched_pairs"),
        "n_matched_pixels_candidate": candidate.get("n_matched_pixels"),
        "area_coverage_candidate": candidate.get("area_coverage"),
        "matching_diagnostics_json": _json(candidate.get("matching_diagnostics", {})),
        "model_family": model_family,
        "fit_scope": fit_scope,
        "numeric_features": _json(features),
        "selection_objective": "maximize apparent r2; separately retain negative direction and association objectives",
        "outcome_guided_selection": True,
        "subset_outcome_guided_selection": subset_name.startswith("oracle_extreme_residual_"),
        "random_seed": int(config["seed"]),
        "random_seed_candidate": candidate.get("random_seed"),
        "subset_n_before_complete_case": int(len(frame)),
        "model_parameters_json": "{}",
        "evaluation_set": "apparent",
    }
    started = time.perf_counter()
    try:
        model_data = _complete_model_data(frame, outcome=y_column, exposure=x_column, numeric_features=features)
        n_total = len(model_data)
        if n_total < max(10, len(features) + 2):
            raise ValueError(f"insufficient complete rows: {n_total}")
        tree_families = {"random_forest", "extra_trees", "hist_gradient_boosting"}
        max_train = int(config.get("models", {}).get("tree_training_max_rows", 12000))
        if model_family in tree_families and n_total > max_train:
            train_data = model_data.sample(n=max_train, random_state=int(config["seed"])).sort_index()
        else:
            train_data = model_data
        prediction, params = _fit_predictions(
            train_data, model_data, outcome=y_column, exposure=x_column, features=features,
            family=model_family, seed=int(config["models"].get("random_seed", config["seed"])),
        )
        y = model_data[y_column].to_numpy(dtype=float)
        x = model_data[x_column].to_numpy(dtype=float)
        metric = regression_metrics(y, prediction)
        pearson = _safe_corr(x, y, pearsonr)
        rho = _safe_corr(x, y, spearmanr)
        support_x, support_prediction = _prediction_support_curve(
            x, prediction, max_points=int(config.get("response_curves", {}).get("curve_points", 250))
        )
        direction = direction_from_curve(support_x, support_prediction)
        if len(support_x) > 1:
            local_slopes = np.diff(support_prediction) / np.diff(support_x)
            direction["local_slope_median"] = float(np.median(local_slopes))
            direction["local_slope_q10"] = float(np.quantile(local_slopes, 0.10))
            direction["local_slope_q90"] = float(np.quantile(local_slopes, 0.90))
        else:
            direction.update(local_slope_median=np.nan, local_slope_q10=np.nan, local_slope_q90=np.nan)
        result = {
            **base,
            **metric,
            "n": int(metric["n"]),
            "n_train": int(len(train_data)),
            "n_evaluation": int(n_total),
            "pearson_r": pearson,
            "spearman_rho": rho,
            "correlation_direction": "unknown" if not np.isfinite(rho) else "negative" if rho < 0 else "positive" if rho > 0 else "near_zero",
            **direction,
            "model_parameters_json": _json({
                "fit": params,
                "direction_curve": "binned fitted predictions over observed exposure support",
                "environment_adjusted_direction": "predictions averaged over observed covariate distribution",
                "tree_training_cap": max_train if model_family in tree_families else None,
            }),
            "status": "ok",
            "runtime_seconds": time.perf_counter() - started,
        }
        return result
    except Exception as exc:
        return {
            **base,
            "n": int(len(frame)),
            "n_train": 0,
            "n_evaluation": 0,
            "r_squared": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "pearson_r": np.nan,
            "spearman_rho": np.nan,
            "direction": "unknown",
            "curve_change": np.nan,
            "status": f"failed: {type(exc).__name__}: {exc}",
            "runtime_seconds": time.perf_counter() - started,
        }


def _stage1_summary(score_frames: list[pd.DataFrame], candidate_rows: list[dict[str, Any]]) -> pd.DataFrame:
    scores = pd.concat(score_frames, ignore_index=True) if score_frames else pd.DataFrame()
    if scores.empty:
        return pd.DataFrame()
    scores["r_squared"] = pd.to_numeric(scores["r_squared"], errors="coerce")
    scores = scores.loc[scores.get("status", pd.Series("ok", index=scores.index)).eq("ok") & np.isfinite(scores["r_squared"])].copy()
    if scores.empty:
        return pd.DataFrame()
    keys = ["candidate_id", "outcome", "estimand"]
    best_by_response = scores.sort_values([*keys, "r_squared", "model_family"], ascending=[True, True, True, False, True], kind="stable").drop_duplicates(keys)
    by_candidate = best_by_response.groupby("candidate_id", sort=False).agg(
        stage1_mean_best_r2=("r_squared", "mean"),
        stage1_negative_response_fraction=("direction", lambda values: float(values.eq("negative").mean())),
        stage1_scored_response_units=("estimand", "size"),
        stage1_scored_outcomes=("outcome", "nunique"),
        stage1_mean_n=("n", "mean"),
    ).reset_index()
    manifest = pd.DataFrame(candidate_rows)
    if "n_matched_pairs" in manifest:
        manifest = manifest.rename(columns={"n_matched_pairs": "stage1_n_pairs"})
    summary = manifest.merge(by_candidate, on="candidate_id", how="left", validate="one_to_one")
    summary["stage1_r2_rank"] = summary["stage1_mean_best_r2"].fillna(-1.0).rank(method="average", ascending=True, pct=True)
    summary["stage1_coverage_rank"] = summary["stage1_n_pairs"].fillna(0.0).rank(method="average", ascending=True, pct=True)
    summary["stage1_negative_rank"] = summary["stage1_negative_response_fraction"].fillna(0.0).rank(method="average", ascending=True, pct=True)
    summary["stage1_design_score"] = summary[["stage1_r2_rank", "stage1_coverage_rank", "stage1_negative_rank"]].mean(axis=1)
    summary["stage1_rank"] = summary["stage1_design_score"].rank(method="min", ascending=False).astype(int)
    summary["stage1_selection_objective"] = "outcome-guided best nonlinear-family R2 per outcome and estimand + negative direction fraction + matched pair count"
    summary["outcome_guided_selection"] = True
    return summary.sort_values(["exposure", "stage1_rank", "candidate_id"], kind="stable").reset_index(drop=True)


def _choose_stage2_candidates(summary: pd.DataFrame, limit: int) -> pd.DataFrame:
    chosen: list[pd.Series] = []
    for exposure, frame in summary.groupby("exposure", sort=True):
        take = max(1, min(int(limit), len(frame)))
        # Reserve representation for each M3/M4 neighborhood before filling by Oracle screen rank.
        for method in ("M3", "M4"):
            subset = frame.loc[frame["reference_method"].eq(method)]
            if len(subset) and len(chosen) < 2 * take:
                row = subset.sort_values(["stage1_rank", "candidate_id"], kind="stable").iloc[0]
                if row["candidate_id"] not in {item["candidate_id"] for item in chosen}:
                    chosen.append(row)
        ordered = frame.sort_values(["stage1_rank", "candidate_id"], kind="stable")
        for _, row in ordered.iterrows():
            existing = [item["candidate_id"] for item in chosen if item["exposure"] == exposure]
            if len(existing) >= take:
                break
            if row["candidate_id"] not in existing:
                chosen.append(row)
    result = pd.DataFrame(chosen).drop_duplicates("candidate_id")
    return result.sort_values(["exposure", "stage1_rank", "candidate_id"], kind="stable").reset_index(drop=True)


def _add_pair_features(pairs: pd.DataFrame, candidate: Mapping[str, Any], outcomes: Sequence[str]) -> pd.DataFrame:
    result = pairs.copy()
    high_erosion = pd.to_numeric(result.get("erosion_high"), errors="coerce")
    low_erosion = pd.to_numeric(result.get("erosion_low"), errors="coerce")
    result["Aa_high"] = high_erosion
    result["Aa_low"] = low_erosion
    with np.errstate(divide="ignore", invalid="ignore"):
        result["delta_log1p_Aa"] = np.log1p(high_erosion) - np.log1p(low_erosion)
    result["both_positive"] = high_erosion.gt(0) & low_erosion.gt(0)
    result["zero_positive"] = low_erosion.le(0) & high_erosion.gt(0)
    gap = result["delta_log1p_Aa"]
    result["high_erosion_gap_q4"] = gap.ge(gap.quantile(0.75))
    for variable in candidate.get("environment_variables", ()):
        high_name, low_name = f"{variable}_high", f"{variable}_low"
        if high_name in result and low_name in result:
            result[f"delta_{variable}"] = pd.to_numeric(result[high_name], errors="coerce") - pd.to_numeric(result[low_name], errors="coerce")
    for outcome in outcomes:
        high_name, low_name = f"high_{outcome}", f"low_{outcome}"
        if high_name in result and low_name in result:
            result[f"delta_{outcome}"] = pd.to_numeric(result[high_name], errors="coerce") - pd.to_numeric(result[low_name], errors="coerce")
    result["method"] = candidate["reference_method"]
    result["candidate_id"] = candidate["candidate_id"]
    result["exposure"] = candidate["exposure"]
    return result


def _pixel_view(cells: pd.DataFrame, pairs: pd.DataFrame, candidate: Mapping[str, Any]) -> pd.DataFrame:
    ids = pd.Index(pd.unique(pd.concat([pairs.get("high_erosion_cell_id", pd.Series(dtype=object)), pairs.get("low_erosion_cell_id", pd.Series(dtype=object))], ignore_index=True)))
    selected = cells.loc[cells["cell_id"].isin(ids)].drop_duplicates("cell_id").copy()
    selected["candidate_id"] = candidate["candidate_id"]
    selected["method"] = candidate["reference_method"]
    selected["exposure"] = candidate["exposure"]
    selected["spatial_block_id"] = (
        np.floor(pd.to_numeric(selected["x"], errors="coerce") / 5000).astype("Int64").astype(str)
        + "_" + np.floor(pd.to_numeric(selected["y"], errors="coerce") / 5000).astype("Int64").astype(str)
    )
    return selected


def _build_pixel_subsets(
    cells: pd.DataFrame, pairs: pd.DataFrame, pair_subsets: Mapping[str, pd.DataFrame], candidate: Mapping[str, Any]
) -> dict[str, pd.DataFrame]:
    result = {"all": _pixel_view(cells, pairs, candidate)}
    for name, selected_pairs in pair_subsets.items():
        if name == "all" or selected_pairs.empty:
            continue
        if name.startswith("environment_group=") or name.startswith("oracle_extreme_residual_"):
            continue
        result[name] = _pixel_view(cells, selected_pairs, candidate)
    return result


def _candidate_model_trials(
    cells: pd.DataFrame,
    candidate_pairs: pd.DataFrame,
    candidate: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    progress_dir: Path | None = None,
) -> list[dict[str, Any]]:
    outcomes = list(config["outcomes"])
    matched_pairs = _add_pair_features(candidate_pairs, candidate, outcomes)
    retention = list(map(float, config.get("subsets", {}).get("retention_fractions", (1.0, 0.75, 0.5, 0.3, 0.2, 0.1, 0.05))))
    records: list[dict[str, Any]] = []
    # All finalists get the core nonlinear families. The top two per exposure get the full family sweep.
    families = available_model_families(config)
    rank = int(candidate.get("stage1_rank", 99))
    core_families = [name for name in ("ols", "log_exposure", "quadratic", "natural_spline", "random_forest") if name in families]
    for outcome_index, outcome in enumerate(outcomes, start=1):
        outcome_progress = progress_dir / f"outcome_{outcome_index:02d}.parquet" if progress_dir is not None else None
        if outcome_progress is not None and outcome_progress.is_file():
            completed = pd.read_parquet(outcome_progress)
            records.extend(completed.to_dict(orient="records"))
            print(
                f"Phase H model candidate {candidate['candidate_id']} outcome {outcome_index}/{len(outcomes)} resumed: {outcome}",
                flush=True,
            )
            continue
        outcome_records: list[dict[str, Any]] = []
        pair_subsets = build_phase_h_pair_subsets(matched_pairs, outcome=outcome, retention_fractions=retention, include_oracle_residual=bool(config.get("subsets", {}).get("record_extreme_oracle_residual_subset", True)))
        pixel_subsets = _build_pixel_subsets(cells, matched_pairs, pair_subsets, candidate)
        for estimand, subset_map in (("matched_pair", pair_subsets), ("matched_pixel", pixel_subsets)):
            denominator_n = len(subset_map.get("all", pd.DataFrame()))
            for subset_name, frame in subset_map.items():
                if frame.empty:
                    continue
                if subset_name.endswith("_100pct"):
                    # The all-row population already records retention_fraction=1.0.
                    continue
                if subset_name == "all":
                    exposure_families = families if rank <= 2 else core_families
                    adjusted_families = (
                        [name for name in ("ols", "natural_spline", "random_forest") if name in families]
                        if rank <= 2 else [name for name in ("ols", "natural_spline") if name in families]
                    )
                    family_scopes = {
                        "exposure_only": exposure_families,
                        "environment_adjusted": adjusted_families,
                    }
                elif subset_name.startswith("environment_distance_low_") or subset_name.startswith("exposure_gap_high_"):
                    chosen_families = ["ols"] if "ols" in families else families[:1]
                    family_scopes = {"exposure_only": chosen_families}
                elif subset_name.startswith("oracle_extreme_residual_"):
                    if rank > 1 or estimand != "matched_pair":
                        continue
                    family_scopes = {"exposure_only": ["ols"] if "ols" in families else families[:1]}
                elif subset_name.startswith("environment_group="):
                    if rank > 1 or estimand != "matched_pair":
                        continue
                    family_scopes = {"exposure_only": ["ols"] if "ols" in families else families[:1]}
                else:
                    chosen_families = ["ols"] if "ols" in families else families[:1]
                    family_scopes = {"exposure_only": chosen_families}
                for fit_scope, chosen_families in family_scopes.items():
                    for family in chosen_families:
                        outcome_records.append(_model_trial(
                            frame, config=config, candidate=candidate, estimand=estimand,
                            outcome=outcome, subset_name=subset_name, model_family=family,
                            fit_scope=fit_scope, stage="stage2_full_sample",
                            retention_fraction=(len(frame) / denominator_n if denominator_n else 0.0),
                        ))
        if outcome_progress is not None:
            outcome_progress.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame.from_records(outcome_records).to_parquet(outcome_progress, index=False)
        records.extend(outcome_records)
        print(
            f"Phase H model candidate {candidate['candidate_id']} outcome {outcome_index}/{len(outcomes)}: {outcome}",
            flush=True,
        )
    return records


def _stage1_nonlinear_screen(
    cells: pd.DataFrame,
    pairs: pd.DataFrame,
    candidate: Mapping[str, Any],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Screen each candidate using several response forms on the representative sample."""

    matched_pairs = _add_pair_features(pairs, candidate, config["outcomes"])
    matched_pixels = _pixel_view(cells, matched_pairs, candidate)
    configured = list(config.get("models", {}).get("families", MODEL_FAMILIES))
    family_order = ("ols", "log_exposure", "quadratic", "piecewise", "natural_spline", "statsmodels_gam")
    families = [family for family in family_order if family in configured]
    if not families:
        families = ["ols"]
    trials: list[dict[str, Any]] = []
    for estimand, frame in (("matched_pixel", matched_pixels), ("matched_pair", matched_pairs)):
        for outcome in config["outcomes"]:
            for family in families:
                trials.append(_model_trial(
                    frame,
                    config=config,
                    candidate=candidate,
                    estimand=estimand,
                    outcome=str(outcome),
                    subset_name="all",
                    model_family=family,
                    fit_scope="exposure_only",
                    stage="stage1_representative_screen",
                    retention_fraction=1.0,
                ))
    return trials


def run_phase_h_search(
    cells: pd.DataFrame, config: Mapping[str, Any], run_dir: Path
) -> dict[str, pd.DataFrame]:
    """Execute outcome-guided Stage 1 screening and bounded full-sample Stage 2 refits."""

    root = Path(run_dir)
    matching_dir, table_dir, pair_dir, checkpoint_dir = (root / "matching", root / "tables", root / "pairs" / "candidate_pairs", root / "checkpoints")
    for directory in (matching_dir, table_dir, pair_dir, checkpoint_dir):
        directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "h3_h4_complete.json"
    output_names = [
        "phase_h_model_trials.csv", "phase_h_correlation_vs_r2.csv",
        "stage1_matching_candidates.csv", "stage1_matching_scores.csv", "stage1_nonlinear_model_screen.csv",
    ]
    leaderboard_names = [f"leaderboard_{index}_{name}.csv" for index, name in (
        (1, "top_apparent_r2"), (2, "highest_negative_direction_r2"),
        (3, "strongest_negative_association"), (4, "highest_matched_pixel_r2"),
        (5, "best_spatial_validation"), (6, "environment_group_association_fit_stability"),
        (7, "all_candidate_ranked_audit"),
    )]
    if checkpoint_path.is_file() and all((table_dir / name).is_file() for name in (*output_names, *leaderboard_names)):
        trials = pd.read_csv(table_dir / output_names[0])
        return {"trials": trials, "correlation_vs_r2": pd.read_csv(table_dir / output_names[1]), **{f"leaderboard_{i}": pd.read_csv(table_dir / name) for i, name in enumerate(leaderboard_names, 1)}}

    search = config["matching_search"]
    vectors = search["environment_vectors"]
    union = list(dict.fromkeys(name for names in vectors.values() for name in names))
    required = ["cell_id", "x", "y", "SoilType", *config["exposures"], *union]
    missing = sorted(set(required) - set(cells.columns))
    if missing:
        raise ValueError(f"Phase H matched search cells are missing required columns: {missing}")
    eligible = cells.copy()
    for column in required:
        if column not in {"cell_id"}:
            eligible[column] = pd.to_numeric(eligible[column], errors="coerce")
    mask = np.isfinite(eligible[[name for name in required if name != "cell_id"]].to_numpy(dtype=float)).all(axis=1)
    mask &= eligible["SoilType"].isin(config.get("primary_soil_codes", (26, 27, 42))).to_numpy()
    mask &= eligible[list(config["exposures"])].ge(0).all(axis=1).to_numpy()
    eligible = eligible.loc[mask].copy()
    n_stage1 = min(int(search.get("stage1_representative_cells", 40000)), len(eligible))
    if n_stage1 < 10:
        raise ValueError(f"Phase H Stage 1 has too few eligible cells: {n_stage1}")
    representative = eligible.sample(n=n_stage1, random_state=int(config["seed"])).sort_values("cell_id", kind="stable").reset_index(drop=True)
    stage1_score_path = table_dir / "stage1_matching_scores.csv"
    stage1_models_path = table_dir / "stage1_nonlinear_model_screen.csv"
    stage1_summary_path = table_dir / "stage1_matching_candidates.csv"
    expected_stage1_count = len(config["exposures"]) * int(search["stage1_max_candidates_per_exposure"])
    cached_stage1 = all(path.is_file() for path in (stage1_score_path, stage1_models_path, stage1_summary_path))
    if cached_stage1:
        cached_summary = pd.read_csv(stage1_summary_path)
        cached_models = pd.read_csv(stage1_models_path)
        cached_stage1 = (
            cached_summary["candidate_id"].nunique() == expected_stage1_count
            and cached_models["candidate_id"].nunique() == expected_stage1_count
            and cached_models["stage"].eq("stage1_representative_screen").all()
        )
    if cached_stage1:
        stage1_score_frame = pd.read_csv(stage1_score_path)
        stage1_models_frame = cached_models
        stage1_summary = cached_summary
        stage1_designs = []
        for row in stage1_summary.to_dict(orient="records"):
            for field in ("environment_variables", "environment_weights"):
                value = row.get(field)
                if isinstance(value, str):
                    try:
                        row[field] = json.loads(value)
                    except json.JSONDecodeError:
                        row[field] = ast.literal_eval(value)
                elif pd.isna(value):
                    row[field] = None
            row["selection_outcomes"] = list(config["outcomes"])
            row["input_run_id"] = str(config["input_run_id"])
            row["method"] = row["reference_method"]
            row["random_seed"] = int(row["random_seed"])
            row["n_matched_pairs"] = int(row["stage1_n_pairs"])
            row["n_matched_pixels"] = int(row.get("n_matched_pixels", 0) or 0)
            row["area_coverage"] = float(row.get("area_coverage", 0.0) or 0.0)
            row["stage"] = "stage1"
            stage1_designs.append(row)
        stage1_model_trials = stage1_models_frame.to_dict(orient="records")
        print(f"Phase H Stage 1 resumed from {expected_stage1_count} saved candidates", flush=True)
    else:
        stage1_designs: list[dict[str, Any]] = []
        stage1_scores: list[pd.DataFrame] = []
        stage1_model_trials: list[dict[str, Any]] = []
        registry_stage1 = matching_dir / "phase_h_matching_trial_registry_final.jsonl"
        for exposure_index, exposure in enumerate(config["exposures"]):
            designs = build_phase_h_candidate_designs(config, str(exposure), seed=int(config["seed"]) + exposure_index)
            stage1_designs.extend(designs)
            for candidate in designs:
                result = run_matching_candidate(representative, candidate, config["outcomes"], registry_stage1)
                if result.status != "success":
                    candidate["n_matched_pairs"] = 0
                    candidate["n_matched_pixels"] = 0
                    candidate["area_coverage"] = 0.0
                    stage1_scores.append(pd.DataFrame([{
                        "candidate_id": candidate["candidate_id"], "exposure": exposure,
                        "reference_method": candidate["reference_method"], "outcome": "all",
                        "score_set": "matching_failure", "status": "failed", "error": result.error,
                        "n": 0, "n_pairs": 0, "r_squared": np.nan, "direction": "unknown",
                        "stage": "stage1_representative_screen", "outcome_guided_selection": True,
                        "selection_objective": candidate["selection_objective"],
                    }]))
                    continue
                candidate["n_matched_pairs"] = int(len(result.pairs))
                candidate["n_matched_pixels"] = int(result.diagnostics.get("n_matched_cells", 2 * len(result.pairs)))
                candidate["area_coverage"] = result.diagnostics.get("area_coverage")
                candidate["matching_diagnostics"] = result.diagnostics
                scores = result.scores.copy()
                scores["candidate_id"] = candidate["candidate_id"]
                scores["exposure"] = exposure
                scores["reference_method"] = candidate["reference_method"]
                scores["stage"] = "stage1_representative_screen"
                scores["sampled_cell_count"] = int(len(representative))
                scores["outcome_guided_selection"] = True
                scores["selection_objective"] = candidate["selection_objective"]
                stage1_scores.append(scores)
                stage1_model_trials.extend(_stage1_nonlinear_screen(representative, result.pairs, candidate, config))
                print(
                    f"Phase H Stage 1 candidate {len(stage1_scores)}/{expected_stage1_count}: "
                    f"{candidate['candidate_id']}, {len(result.pairs):,} pairs, nonlinear screen complete",
                    flush=True,
                )
        stage1_score_frame = pd.concat(stage1_scores, ignore_index=True) if stage1_scores else pd.DataFrame()
        stage1_score_frame.to_csv(stage1_score_path, index=False, encoding="utf-8-sig")
        stage1_models_frame = pd.DataFrame.from_records(stage1_model_trials)
        stage1_models_frame.to_csv(stage1_models_path, index=False, encoding="utf-8-sig")
        candidate_rows = [{**candidate, "environment_variables": _json(candidate["environment_variables"]), "environment_weights": _json(candidate["environment_weights"])} for candidate in stage1_designs]
        screen_rows: list[pd.DataFrame] = []
        for exposure in config["exposures"]:
            selected_scores = stage1_models_frame.loc[stage1_models_frame["exposure"].eq(exposure)].copy()
            candidate_subset = [candidate for candidate in candidate_rows if candidate["exposure"] == exposure]
            screen_rows.append(_stage1_summary([selected_scores], candidate_subset))
        stage1_summary = pd.concat(screen_rows, ignore_index=True) if screen_rows else pd.DataFrame()
        stage1_summary.to_csv(stage1_summary_path, index=False, encoding="utf-8-sig")
    finalist_rows = _choose_stage2_candidates(stage1_summary, int(search["stage2_max_candidates_per_exposure"]))
    full_candidates: list[dict[str, Any]] = []
    for _, row in finalist_rows.iterrows():
        candidate = next(item.copy() for item in stage1_designs if item["candidate_id"] == row["candidate_id"])
        per_exposure_number = 1 + sum(item["exposure"] == candidate["exposure"] for item in full_candidates)
        candidate["candidate_id"] = f"stage2_{candidate['exposure']}_{per_exposure_number:02d}"
        candidate["experiment_id"] = f"{config['run_id']}-stage2-{candidate['exposure']}-{len(full_candidates) + 1:03d}"
        candidate["experiment_type"] = "phase_h_stage2_full_sample_refit"
        candidate["stage"] = "stage2"
        candidate["selection_parent_candidate_id"] = row["candidate_id"]
        candidate["stage1_rank"] = int(row["stage1_rank"])
        candidate["stage1_selection_score"] = float(row["stage1_design_score"])
        candidate["selection_objective"] = "stage1_oracle_rank_then_full_sample_refit_all_outcomes"
        full_candidates.append(candidate)

    all_trials: list[dict[str, Any]] = list(stage1_model_trials)
    stage2_pair_rows = 0
    model_started = time.perf_counter()
    registry_stage2 = matching_dir / "phase_h_matching_trial_registry_stage2.jsonl"
    for candidate_index, candidate in enumerate(full_candidates, start=1):
        pair_path = pair_dir / f"{candidate['candidate_id']}.parquet"
        model_trial_path = table_dir / f"stage2_model_trials_{candidate['candidate_id']}.parquet"
        if pair_path.is_file() and model_trial_path.is_file():
            saved_trials = pd.read_parquet(model_trial_path)
            if not saved_trials.empty and saved_trials.get("model_schedule_version", pd.Series(dtype=object)).eq(MODEL_SCHEDULE_VERSION).all():
                all_trials.extend(saved_trials.to_dict(orient="records"))
                stage2_pair_rows += len(pd.read_parquet(pair_path, columns=["pair_id"]))
                print(f"Phase H Stage 2 candidate {candidate_index}/{len(full_candidates)} resumed: {candidate['candidate_id']}", flush=True)
                continue
        if pair_path.is_file():
            pairs = pd.read_parquet(pair_path)
            candidate["n_matched_pairs"] = int(len(pairs))
            candidate["n_matched_pixels"] = int(2 * len(pairs))
            candidate["area_coverage"] = float(2 * len(pairs) / max(len(eligible), 1))
            candidate["matching_diagnostics"] = {"resumed_from_candidate_pair_file": True}
            pairs = _add_pair_features(pairs, candidate, config["outcomes"])
        else:
            print(f"Phase H Stage 2 full matching {candidate_index}/{len(full_candidates)}: {candidate['candidate_id']}", flush=True)
            result = run_matching_candidate(eligible, candidate, config["outcomes"], registry_stage2)
            if result.status != "success":
                all_trials.append({
                    "stage": "stage2_full_sample", "analysis_population": "phase_h_matched_only",
                    "estimand": "matched_pair", "exposure": candidate["exposure"], "outcome": "all",
                    "subset_name": "all", "candidate_id": candidate["candidate_id"],
                    "model_family": "matching", "fit_scope": "matching", "n": 0,
                    "r_squared": np.nan, "direction": "unknown", "status": f"matching_failed: {result.error}",
                    "outcome_guided_selection": True, "selection_objective": candidate["selection_objective"],
                })
                continue
            pairs = _add_pair_features(result.pairs, candidate, config["outcomes"])
            candidate["n_matched_pairs"] = int(len(pairs))
            candidate["n_matched_pixels"] = int(result.diagnostics.get("n_matched_cells", 2 * len(pairs)))
            candidate["area_coverage"] = result.diagnostics.get("area_coverage")
            candidate["matching_diagnostics"] = result.diagnostics
            pairs.to_parquet(pair_path, index=False)
        stage2_pair_rows += len(pairs)
        model_started_candidate = time.perf_counter()
        candidate_trials = _candidate_model_trials(
            eligible, pairs, candidate, config,
            progress_dir=table_dir / "model_progress_v2" / candidate["candidate_id"],
        )
        pd.DataFrame.from_records(candidate_trials).to_parquet(model_trial_path, index=False)
        all_trials.extend(candidate_trials)
        print(f"Phase H Stage 2 candidate {candidate_index}/{len(full_candidates)}: {candidate['candidate_id']}, {len(pairs):,} pairs, model fit {time.perf_counter() - model_started_candidate:.1f}s", flush=True)
        del pairs
        gc.collect()

    trial_frame = pd.DataFrame.from_records(all_trials)
    # Include the selected design and all screened match scores in a single auditable registry.
    trial_frame["run_id"] = config["run_id"]
    trial_frame["input_run_id"] = config["input_run_id"]
    trial_frame["outcome_guided_selection"] = True
    trial_frame.to_csv(table_dir / "phase_h_model_trials.csv", index=False, encoding="utf-8-sig")
    correlation = build_correlation_r2_analysis(trial_frame)
    correlation.to_csv(table_dir / "phase_h_correlation_vs_r2.csv", index=False, encoding="utf-8-sig")
    leaderboards = build_phase_h_leaderboards(trial_frame)
    for index, name in enumerate(leaderboard_names, start=1):
        key_name = (
            "leaderboard_1_top_apparent_r2", "leaderboard_2_highest_negative_direction_r2",
            "leaderboard_3_strongest_negative_association", "leaderboard_4_highest_matched_pixel_r2",
            "leaderboard_5_best_spatial_validation", "leaderboard_6_environment_group_association_fit_stability",
            "leaderboard_7_all_candidate_ranked_audit",
        )[index - 1]
        leaderboards[key_name].to_csv(table_dir / name, index=False, encoding="utf-8-sig")
    model_availability = {
        "configured_families": available_model_families(config),
        "xgboost_installed": importlib.util.find_spec("xgboost") is not None,
        "lightgbm_installed": importlib.util.find_spec("lightgbm") is not None,
        "r2_definition": "1 - SSE / sum((y - mean(y))^2)",
    }
    (matching_dir / "model_availability.json").write_text(_json(model_availability) + "\n", encoding="utf-8")
    checkpoint = {
        "task": "H3_H4_matched_only_local_subset_model_search",
        "model_schedule_version": MODEL_SCHEDULE_VERSION,
        "status": "complete",
        "stage1_candidate_count": int(len(stage1_designs)),
        "stage1_representative_cells": int(len(representative)),
        "stage2_candidate_count": int(len(full_candidates)),
        "stage2_pair_rows": int(stage2_pair_rows),
        "model_trial_rows": int(len(trial_frame)),
        "successful_model_trials": int(trial_frame.get("status", pd.Series(dtype=object)).eq("ok").sum()),
        "outcome_guided_selection": True,
        "matched_only_primary_input": True,
        "historical_unmatched_oracle_imported": False,
        "model_availability": model_availability,
        "total_model_search_seconds": time.perf_counter() - model_started,
    }
    checkpoint_path.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Phase H H3/H4: {len(stage1_designs)} screen candidates; {len(full_candidates)} full candidates; {len(trial_frame):,} model trials", flush=True)
    return {"trials": trial_frame, "correlation_vs_r2": correlation, **{f"leaderboard_{i}": value for i, value in enumerate(leaderboards.values(), 1)}}
