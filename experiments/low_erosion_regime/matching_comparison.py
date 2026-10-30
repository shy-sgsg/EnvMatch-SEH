"""Low-regime filtering and outcome-blind rematching comparisons."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

from soilmatch.matching import match_cells


ENVIRONMENT_COLUMNS = ("Elevation", "LST")


def original_group_map(pairs: pd.DataFrame, *, method: str, exposure: str) -> pd.DataFrame:
    """Recover the pre-existing Phase F group labels for this matching design."""
    selected = pairs.loc[pairs["method"].eq(method) & pairs["exposure"].eq(exposure)].copy()
    if "spatial_role" in selected:
        selected = selected.loc[selected["spatial_role"].eq("full_sample_descriptive")]
    high = selected[["high_erosion_cell_id", "environment_group"]].rename(columns={"high_erosion_cell_id": "cell_id"})
    low = selected[["low_erosion_cell_id", "environment_group"]].rename(columns={"low_erosion_cell_id": "cell_id"})
    labels = pd.concat([high, low], ignore_index=True).dropna(subset=["cell_id", "environment_group"])
    conflicts = labels.groupby("cell_id", observed=True)["environment_group"].nunique()
    if (conflicts > 1).any():
        raise ValueError(f"Phase F groups are inconsistent for {int((conflicts > 1).sum())} cells in {method}/{exposure}")
    return labels.drop_duplicates("cell_id").reset_index(drop=True)


def filter_existing_pairs(
    pairs: pd.DataFrame, cells: pd.DataFrame, *, exposure: str, threshold: float | None,
    method: str | None = None,
) -> pd.DataFrame:
    """Retain existing M3/M4 pairs only when both endpoints exceed Aa threshold."""
    selected = pairs.loc[pairs["exposure"].eq(exposure)].copy()
    if method is not None:
        selected = selected.loc[selected["method"].eq(method)].copy()
    if "spatial_role" in selected:
        selected = selected.loc[selected["spatial_role"].eq("full_sample_descriptive")].copy()
    if threshold is None:
        selected["threshold_original"] = np.nan
        selected["retention_scheme"] = "S0_all"
        return selected
    value_map = cells.set_index("cell_id")[exposure]
    high = pd.to_numeric(selected["high_erosion_cell_id"].map(value_map), errors="coerce")
    low = pd.to_numeric(selected["low_erosion_cell_id"].map(value_map), errors="coerce")
    keep = high.gt(threshold) & low.gt(threshold)
    result = selected.loc[keep].copy()
    result["threshold_original"] = float(threshold)
    result["retention_scheme"] = "both_endpoints_above_threshold"
    return result


def _refit_groups(filtered: pd.DataFrame, *, seed: int, n_clusters: int = 6) -> pd.Series:
    """Refit soil-specific Elevation/LST K-means only on filtered exposure cells."""
    labels = pd.Series(pd.NA, index=filtered.index, dtype="string")
    soil = pd.to_numeric(filtered["SoilType"], errors="coerce")
    for soil_code in sorted(soil.dropna().astype(int).unique()):
        mask = soil.eq(soil_code)
        subset = filtered.loc[mask, list(ENVIRONMENT_COLUMNS)].apply(pd.to_numeric, errors="coerce")
        finite = np.isfinite(subset.to_numpy(dtype=float)).all(axis=1)
        positions = subset.index[finite]
        if not len(positions):
            continue
        matrix = subset.loc[positions].to_numpy(dtype=float)
        scaler = StandardScaler().fit(matrix)
        scaled = scaler.transform(matrix)
        k = min(n_clusters, len(positions), np.unique(scaled, axis=0).shape[0])
        if k < 2:
            assigned = np.zeros(len(positions), dtype=int)
        else:
            model = MiniBatchKMeans(n_clusters=k, random_state=seed + soil_code, n_init=3,
                                    batch_size=min(2048, max(256, k * 64)), reassignment_ratio=0.0)
            assigned = model.fit_predict(scaled)
        labels.loc[positions] = [f"soil_{soil_code}_cluster_{int(value)}" for value in assigned]
    return labels


def rematch_filtered_cells(
    cells: pd.DataFrame, phase_f_pairs: pd.DataFrame, *, exposure: str, threshold: float,
    method: str, group_mode: str, seed: int = 20261029,
    environment_distance_limit: float = .35, distance_limit_m: float = 1000.0,
    rank_gap: float = .50, primary_soils: tuple[int, ...] = (26, 27, 42),
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Rematch above-threshold cells under fixed Phase F groups or refitted groups."""
    valid = cells["valid_cell"].fillna(False).astype(bool)
    for col in [exposure, "SoilType", "x", "y", *ENVIRONMENT_COLUMNS]:
        numeric = pd.to_numeric(cells[col], errors="coerce")
        valid &= numeric.notna() & np.isfinite(numeric)
    valid &= pd.to_numeric(cells[exposure], errors="coerce").gt(float(threshold))
    valid &= pd.to_numeric(cells["SoilType"], errors="coerce").isin(primary_soils)
    eligible_before_group = int(valid.sum())
    population = cells.loc[valid].copy()
    if group_mode == "fixed_original":
        group_map = original_group_map(phase_f_pairs, method=method, exposure=exposure)
        population = population.merge(group_map, on="cell_id", how="inner", validate="one_to_one")
        group_col = "environment_group"
    elif group_mode == "refit_k6_filtered_population":
        population["environment_group"] = _refit_groups(population, seed=seed)
        group_col = "environment_group"
        population = population.loc[population[group_col].notna()].copy()
    else:
        raise ValueError(f"unknown group_mode: {group_mode}")
    population["SoilType"] = pd.to_numeric(population["SoilType"], errors="coerce")
    # The percentile gap preserves Phase F's rank-based contrast scale.
    population["__rank"] = population.groupby([group_col, "SoilType"], observed=True)[exposure].rank(pct=True)
    matrix = population.loc[:, ENVIRONMENT_COLUMNS].apply(pd.to_numeric, errors="coerce")
    scaler = StandardScaler().fit(matrix)
    population.loc[:, ENVIRONMENT_COLUMNS] = scaler.transform(matrix)
    matched, diagnostics = match_cells(
        population,
        environment_columns=ENVIRONMENT_COLUMNS,
        group_col=group_col,
        erosion_col="__rank",
        soil_type_col="SoilType",
        x_col="x",
        y_col="y",
        distance_limit=float(distance_limit_m),
        environment_distance_limit=float(environment_distance_limit),
        erosion_gap=float(rank_gap),
        allow_reuse=False,
    )
    diagnostics.update({"method": method, "exposure": exposure, "threshold_original": float(threshold),
                        "group_mode": group_mode, "eligible_population": int(len(population)),
                        "eligible_before_group_assignment": eligible_before_group,
                        "group_label_coverage_fraction": float(len(population) / eligible_before_group) if eligible_before_group else np.nan,
                        "outcome_guided_selection": False})
    matched["method"] = method
    matched["exposure"] = exposure
    matched["group_mode"] = group_mode
    matched["threshold_original"] = float(threshold)
    if not matched.empty:
        coordinates = cells.set_index("cell_id")[["x", "y"]]
        x_high = matched["high_erosion_cell_id"].map(coordinates["x"]).to_numpy(dtype=float)
        x_low = matched["low_erosion_cell_id"].map(coordinates["x"]).to_numpy(dtype=float)
        y_high = matched["high_erosion_cell_id"].map(coordinates["y"]).to_numpy(dtype=float)
        y_low = matched["low_erosion_cell_id"].map(coordinates["y"]).to_numpy(dtype=float)
        matched["x_mid"] = (x_high + x_low) / 2
        matched["y_mid"] = (y_high + y_low) / 2
        matched["spatial_block_id"] = (
            np.floor(matched["x_mid"] / 5000).astype(int).astype(str) + "_" +
            np.floor(matched["y_mid"] / 5000).astype(int).astype(str)
        )
    return matched, diagnostics


def enrich_rematched_pairs(
    pairs: pd.DataFrame, cells: pd.DataFrame, outcomes: tuple[str, ...], *, exposure: str
) -> pd.DataFrame:
    """Attach endpoint outcomes and delta variables to newly created pairs."""
    if pairs.empty:
        return pairs
    available = [name for name in outcomes if name in cells]
    indexed = cells.set_index("cell_id")
    result = pairs.copy()
    high_ids = result["high_erosion_cell_id"]
    low_ids = result["low_erosion_cell_id"]
    for variable in (exposure, "x", "y", "Elevation", "Cur", "TWI", "LST", "NDWI"):
        result[f"{variable}_high"] = pd.to_numeric(high_ids.map(indexed[variable]), errors="coerce").to_numpy()
        result[f"{variable}_low"] = pd.to_numeric(low_ids.map(indexed[variable]), errors="coerce").to_numpy()
    result["erosion_high"] = result[f"{exposure}_high"]
    result["erosion_low"] = result[f"{exposure}_low"]
    result["erosion_difference"] = result["erosion_high"] - result["erosion_low"]
    result["x_mid"] = (result["x_high"] + result["x_low"]) / 2
    result["y_mid"] = (result["y_high"] + result["y_low"]) / 2
    result["delta_log1p_erosion"] = np.log1p(result["erosion_high"]) - np.log1p(result["erosion_low"])
    for outcome in available:
        high = pd.to_numeric(high_ids.map(indexed[outcome]), errors="coerce")
        low = pd.to_numeric(low_ids.map(indexed[outcome]), errors="coerce")
        result[f"high_{outcome}"] = high.to_numpy()
        result[f"low_{outcome}"] = low.to_numpy()
        result[f"delta_{outcome}"] = (high - low).to_numpy()
    return result

