"""Environment-constrained nearest-neighbor pairing for erosion exposure."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree

from soilmatch.environment_vectors import environment_distances


def match_cells(
    frame: pd.DataFrame,
    *,
    environment_columns: Sequence[str],
    group_col: str,
    erosion_col: str,
    soil_type_col: str,
    x_col: str,
    y_col: str,
    distance_limit: float,
    environment_distance_limit: float,
    erosion_gap: float,
    allow_reuse: bool = False,
    per_variable_calipers: Mapping[str, float] | None = None,
    environment_weights: Sequence[float] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Greedily pair high and low erosion cells within the same environment
    group and soil type, subject to environment and geographic calipers.

    A KDTree limits environment candidate searches to the configured radius;
    this function never constructs an all-cells distance matrix. Health columns
    are not referenced. High cells are processed from greatest exposure down;
    candidate low cells are ranked by environment distance, geographic
    distance, then input order for deterministic ties.
    """
    env_columns = list(environment_columns)
    required = [group_col, erosion_col, soil_type_col, x_col, y_col, *env_columns]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Missing matching columns: {', '.join(missing)}")
    if not env_columns:
        raise ValueError("At least one continuous environment column is required")
    for name, value in (
        ("distance_limit", distance_limit),
        ("environment_distance_limit", environment_distance_limit),
        ("erosion_gap", erosion_gap),
    ):
        if value is None or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a finite value greater than zero")
    if len(set(env_columns)) != len(env_columns):
        raise ValueError("environment_columns must be unique")

    calipers = dict(per_variable_calipers or {})
    unknown_calipers = set(calipers) - set(env_columns)
    if unknown_calipers:
        raise ValueError(f"Calipers reference unknown environment columns: {sorted(unknown_calipers)}")
    if any(not np.isfinite(value) or value < 0 for value in calipers.values()):
        raise ValueError("Per-variable calipers must be finite and nonnegative")
    weights = np.asarray(
        np.ones(len(env_columns)) if environment_weights is None else environment_weights,
        dtype=float,
    )
    if weights.shape != (len(env_columns),) or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("environment_weights must be finite, nonnegative, and match environment_columns")

    ids = frame["cell_id"].to_numpy(copy=True) if "cell_id" in frame else frame.index.to_numpy(copy=True)
    if pd.Index(ids).has_duplicates:
        raise ValueError("cell identifiers must be unique")
    numeric_columns = [erosion_col, x_col, y_col, *env_columns]
    numeric = frame.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
    valid_strata = frame[group_col].notna().to_numpy() & frame[soil_type_col].notna().to_numpy()
    eligible_mask = finite & valid_strata
    prepared = frame.loc[eligible_mask, [group_col, soil_type_col]].copy()
    prepared["__position"] = np.flatnonzero(eligible_mask)
    prepared["__cell_id"] = ids[eligible_mask]
    for column in numeric_columns:
        prepared[column] = numeric.loc[eligible_mask, column].to_numpy(dtype=float)
    if (prepared[erosion_col] < 0).any():
        raise ValueError(f"{erosion_col} must be nonnegative")

    if not prepared.empty:
        minimum_by_stratum = prepared.groupby(
            [group_col, soil_type_col], sort=False, dropna=False
        )[erosion_col].transform("min")
        prepared["__potential_high"] = prepared[erosion_col] >= minimum_by_stratum + erosion_gap
    else:
        prepared["__potential_high"] = pd.Series(dtype=bool)

    used_cells: set[object] = set()
    used_high: set[object] = set()
    used_low: set[object] = set()
    failure_counts = {
        "no_environment_candidate": 0,
        "erosion_gap": 0,
        "caliper": 0,
        "geographic_distance": 0,
        "already_used": 0,
    }
    pair_records: list[dict[str, object]] = []

    for (group_value, soil_value), stratum in prepared.groupby(
        [group_col, soil_type_col], sort=False, dropna=False
    ):
        ordered = stratum.sort_values(
            [erosion_col, "__position"], ascending=[False, True], kind="mergesort"
        )
        vectors = ordered.loc[:, env_columns].to_numpy(dtype=float)
        weighted_vectors = vectors * np.sqrt(weights)
        tree = KDTree(weighted_vectors, metric="euclidean")
        cell_ids = ordered["__cell_id"].to_numpy(dtype=object)
        exposures = ordered[erosion_col].to_numpy(dtype=float)
        coords = ordered[[x_col, y_col]].to_numpy(dtype=float)
        positions = ordered["__position"].to_numpy(dtype=int)
        potential_high = ordered["__potential_high"].to_numpy(dtype=bool)

        for high_index in np.flatnonzero(potential_high):
            high_id = cell_ids[high_index]
            if high_id in used_high or high_id in used_low:
                failure_counts["already_used"] += 1
                continue
            candidate_indices = tree.query_radius(
                weighted_vectors[high_index : high_index + 1], r=environment_distance_limit
            )[0]
            if len(candidate_indices) == 0:
                failure_counts["no_environment_candidate"] += 1
                continue
            candidate_indices = candidate_indices[exposures[candidate_indices] <= exposures[high_index] - erosion_gap]
            if len(candidate_indices) == 0:
                failure_counts["erosion_gap"] += 1
                continue

            distances = environment_distances(
                np.repeat(vectors[high_index : high_index + 1], len(candidate_indices), axis=0),
                vectors[candidate_indices],
                weights=weights,
            )
            if calipers:
                high_values = ordered.iloc[high_index][env_columns].to_numpy(dtype=float)
                differences = np.abs(ordered.iloc[candidate_indices][env_columns].to_numpy(dtype=float) - high_values)
                keep = np.ones(len(candidate_indices), dtype=bool)
                for column, limit in calipers.items():
                    keep &= differences[:, env_columns.index(column)] <= limit
                candidate_indices = candidate_indices[keep]
                distances = distances[keep]
                if len(candidate_indices) == 0:
                    failure_counts["caliper"] += 1
                    continue

            geographic = np.linalg.norm(coords[candidate_indices] - coords[high_index], axis=1)
            keep = geographic <= distance_limit
            candidate_indices = candidate_indices[keep]
            distances = distances[keep]
            geographic = geographic[keep]
            if len(candidate_indices) == 0:
                failure_counts["geographic_distance"] += 1
                continue

            candidates = sorted(
                zip(candidate_indices, distances, geographic, strict=True),
                key=lambda item: (float(item[1]), float(item[2]), int(positions[item[0]])),
            )
            chosen = None
            for low_index, environment_distance, geographic_distance in candidates:
                low_id = cell_ids[low_index]
                if low_id == high_id or low_id in used_high:
                    continue
                if not allow_reuse and (low_id in used_cells or high_id in used_cells):
                    continue
                chosen = (low_index, environment_distance, geographic_distance)
                break
            if chosen is None:
                failure_counts["already_used"] += 1
                continue

            low_index, environment_distance, geographic_distance = chosen
            low_id = cell_ids[low_index]
            high_erosion = float(exposures[high_index])
            low_erosion = float(exposures[low_index])
            pair_records.append(
                {
                    "pair_id": f"pair_{len(pair_records) + 1:06d}",
                    "high_erosion_cell_id": high_id,
                    "low_erosion_cell_id": low_id,
                    "environment_group": group_value,
                    "soil_type": soil_value,
                    "environmental_distance": float(environment_distance),
                    "geographic_distance": float(geographic_distance),
                    "erosion_high": high_erosion,
                    "erosion_low": low_erosion,
                    "erosion_difference": high_erosion - low_erosion,
                }
            )
            used_high.add(high_id)
            used_low.add(low_id)
            if not allow_reuse:
                used_cells.update((high_id, low_id))

    pairs = pd.DataFrame.from_records(
        pair_records,
        columns=[
            "pair_id",
            "high_erosion_cell_id",
            "low_erosion_cell_id",
            "environment_group",
            "soil_type",
            "environmental_distance",
            "geographic_distance",
            "erosion_high",
            "erosion_low",
            "erosion_difference",
        ],
    )
    diagnostics: dict[str, object] = {
        "input_cells": int(len(frame)),
        "eligible_cells": int(eligible_mask.sum()),
        "invalid_cells": int((~eligible_mask).sum()),
        "potential_high_cells": int(prepared["__potential_high"].sum()),
        "matched_pairs": int(len(pairs)),
        "unmatched_high_cells": int(prepared["__potential_high"].sum() - len(pairs)),
        "failure_counts": failure_counts,
        "allow_reference_reuse": bool(allow_reuse),
    }
    return pairs, diagnostics
