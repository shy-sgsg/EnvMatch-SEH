"""Build matched-pixel and matched-pair analysis populations for Phase H."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from experiments.oracle_exploration.phase_h import REQUIRED_EXPOSURES, REQUIRED_OUTCOMES

MATCHED_PIXEL_METADATA = (
    "pair_id", "method", "exposure", "seed", "spatial_block_id", "environment_group",
    "descriptive_environment_group", "soil_type", "environmental_distance", "geographic_distance",
    "exposure_percentile_gap", "delta_log1p_erosion",
)
PAIR_BASE_COLUMNS = (
    "pair_id", "method", "exposure", "seed", "high_erosion_cell_id", "low_erosion_cell_id",
    "Aa_high", "Aa_low", "delta_log1p_Aa", "spatial_block_id", "soil_type",
    "environmental_distance", "geographic_distance",
)
DEFAULT_GROUP_COLUMNS = (
    "environment_group", "soil_type_group", "kmeans_v5", "hierarchical_ward_v5",
    "gradient_Elevation", "gradient_TWI", "gradient_LST", "gradient_NDWI",
    "gradient_NDVI", "gradient_Precipitation", "kmeans_terrain_k4", "kmeans_terrain_k6",
    "kmeans_terrain_k8",
)


@dataclass(frozen=True)
class MatchedDatasets:
    """Pair memberships, deduplicated pixel view, and explicit pair subsets."""

    pair_memberships: pd.DataFrame
    pixel_unique: pd.DataFrame
    pairs: pd.DataFrame
    pair_subsets: dict[str, pd.DataFrame]
    environment_group_pairs: pd.DataFrame
    summary: pd.DataFrame


def derive_pair_difference_features(pairs: pd.DataFrame, outcomes: Sequence[str]) -> pd.DataFrame:
    """Add high-minus-low Aa and health differences without changing pair direction."""

    required = {"erosion_high", "erosion_low"}
    missing = sorted(required - set(pairs.columns))
    if missing:
        raise ValueError(f"pair table is missing erosion endpoints: {missing}")
    result = pairs.copy()
    result["Aa_high"] = pd.to_numeric(result["erosion_high"], errors="coerce")
    result["Aa_low"] = pd.to_numeric(result["erosion_low"], errors="coerce")
    if (result[["Aa_high", "Aa_low"]] < 0).any(axis=None):
        raise ValueError("Aa endpoint values must be nonnegative")
    with np.errstate(divide="ignore", invalid="ignore"):
        result["delta_log1p_Aa"] = np.log1p(result["Aa_high"]) - np.log1p(result["Aa_low"])
    result["delta_log1p_erosion"] = result["delta_log1p_Aa"]
    result["erosion_order_valid"] = result["Aa_high"].ge(result["Aa_low"])
    for outcome in outcomes:
        high, low = f"high_{outcome}", f"low_{outcome}"
        if high not in result or low not in result:
            raise ValueError(f"pair table is missing {outcome} endpoints")
        result[f"delta_{outcome}"] = pd.to_numeric(result[high], errors="coerce") - pd.to_numeric(
            result[low], errors="coerce"
        )
    return result


def _group_mode(memberships: pd.DataFrame, keys: Sequence[str], column: str) -> pd.DataFrame:
    selected = memberships.loc[memberships[column].notna(), [*keys, column]].copy()
    if selected.empty:
        return pd.DataFrame(columns=[*keys, f"{column}_mode"])
    selected[column] = selected[column].astype(str)
    counts = selected.groupby([*keys, column], sort=False, observed=True).size().rename("__count").reset_index()
    counts = counts.sort_values(["__count", column], ascending=[False, True], kind="mergesort")
    modes = counts.drop_duplicates(list(keys), keep="first")[[*keys, column]]
    return modes.rename(columns={column: f"{column}_mode"})


def _pixel_memberships(
    cells: pd.DataFrame,
    pairs: pd.DataFrame,
    *,
    group_columns: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cell_index = cells.set_index("cell_id", drop=False)
    metadata = list(dict.fromkeys(
        name for name in (*MATCHED_PIXEL_METADATA, *group_columns) if name in pairs
    ))
    membership_parts = []
    for side, id_column, erosion_column in (
        ("high", "high_erosion_cell_id", "Aa_high"),
        ("low", "low_erosion_cell_id", "Aa_low"),
    ):
        membership = pairs[metadata].copy()
        membership["cell_id"] = pairs[id_column].to_numpy()
        membership["cell_role"] = side
        membership["Aa_match_member"] = pairs[erosion_column].to_numpy(dtype=float)
        membership["match_parameters_id"] = [
            f"{method}_{exposure}_seed{seed}"
            for method, exposure, seed in zip(
                membership["method"], membership["exposure"],
                membership["seed"] if "seed" in membership else pd.Series("unknown", index=membership.index),
                strict=True,
            )
        ]
        pixel_data = cell_index.reindex(membership["cell_id"].to_numpy()).reset_index(drop=True)
        pixel_data["cell_id"] = membership["cell_id"].to_numpy()
        membership = pd.concat([membership.reset_index(drop=True), pixel_data.drop(columns=["cell_id"], errors="ignore")], axis=1)
        membership["cell_erosion"] = np.select(
            [membership["exposure"].eq("Aa_20yr"), membership["exposure"].eq("Aa_2020")],
            [
                pd.to_numeric(membership.get("Aa_20yr"), errors="coerce"),
                pd.to_numeric(membership.get("Aa_2020"), errors="coerce"),
            ],
            default=np.nan,
        )
        mismatch = ~np.isclose(
            membership["cell_erosion"].to_numpy(dtype=float),
            membership["Aa_match_member"].to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-6,
            equal_nan=False,
        )
        if mismatch.any():
            bad = membership.loc[mismatch, "pair_id"].iloc[0]
            raise ValueError(f"matched-pixel Aa does not align with pair endpoint for pair_id {bad}")
        membership_parts.append(membership)
    memberships = pd.concat(membership_parts, ignore_index=True)
    key = ["method", "exposure", "cell_id"]
    memberships["membership_count"] = memberships.groupby(key, sort=False)["pair_id"].transform("size").astype(int)
    memberships["membership_weight"] = 1.0 / memberships["membership_count"]

    membership_counts = memberships.groupby(key, sort=False, dropna=False).size().rename("membership_count").reset_index()
    unique = membership_counts.merge(
        cells.drop_duplicates("cell_id"), on="cell_id", how="left", validate="many_to_one", sort=False
    )
    for column in group_columns:
        if column in memberships:
            unique = unique.merge(
                _group_mode(memberships, key, column), on=key, how="left", validate="one_to_one", sort=False
            )
    unique["pixel_model_weight"] = 1.0
    unique.attrs["pixel_model_membership_policy"] = "one row per unique cell_id within method/exposure; membership links retained separately"
    return memberships, unique


def build_matched_datasets(
    cells: pd.DataFrame,
    pairs: pd.DataFrame,
    *,
    outcomes: Sequence[str] = tuple(REQUIRED_OUTCOMES),
    primary_soils: Sequence[int] = (26, 27, 42),
    group_columns: Sequence[str] = DEFAULT_GROUP_COLUMNS,
) -> MatchedDatasets:
    """Build the Phase H matched-pixel and matched-pair populations."""

    cell_required = {"cell_id", "SoilType", *REQUIRED_EXPOSURES, *outcomes}
    missing_cells = sorted(cell_required - set(cells.columns))
    if missing_cells:
        raise ValueError(f"aligned cells are missing required columns: {missing_cells}")
    pair_required = {
        "pair_id", "method", "exposure", "soil_type", "high_erosion_cell_id", "low_erosion_cell_id",
        "erosion_high", "erosion_low", "environmental_distance", "geographic_distance",
    }
    missing_pairs = sorted(pair_required - set(pairs.columns))
    if missing_pairs:
        raise ValueError(f"matched pairs are missing required columns: {missing_pairs}")
    if cells["cell_id"].isna().any() or cells["cell_id"].duplicated().any():
        raise ValueError("aligned cells require unique non-null cell_id values")
    if pairs["pair_id"].isna().any() or pairs["pair_id"].duplicated().any():
        raise ValueError("matched pairs require unique non-null pair_id values")

    allowed = {int(code) for code in primary_soils}
    cell_index = cells.set_index("cell_id", drop=False)
    prepared = pairs.loc[pairs["method"].isin(["M3", "M4"]) & pairs["exposure"].isin(REQUIRED_EXPOSURES)].copy()
    high_ids = prepared["high_erosion_cell_id"].to_numpy()
    low_ids = prepared["low_erosion_cell_id"].to_numpy()
    missing_ids = (~pd.Index(high_ids).isin(cell_index.index)) | (~pd.Index(low_ids).isin(cell_index.index))
    if missing_ids.any():
        raise ValueError("matched pair endpoints are missing from aligned cell data")

    high_soil = pd.to_numeric(cell_index["SoilType"].reindex(high_ids), errors="coerce").to_numpy(dtype=float)
    low_soil = pd.to_numeric(cell_index["SoilType"].reindex(low_ids), errors="coerce").to_numpy(dtype=float)
    pair_soil = pd.to_numeric(prepared["soil_type"], errors="coerce").to_numpy(dtype=float)
    eligible = (
        np.isin(high_soil, list(allowed))
        & np.isin(low_soil, list(allowed))
        & (high_soil == low_soil)
        & (high_soil == pair_soil)
    )
    prepared["soil_type_raster_high"] = high_soil
    prepared["soil_type_raster_low"] = low_soil
    excluded_soil_pairs = int((~eligible).sum())
    prepared = prepared.loc[eligible].copy()

    high_from_cells = np.full(len(prepared), np.nan, dtype=float)
    low_from_cells = np.full(len(prepared), np.nan, dtype=float)
    for exposure in REQUIRED_EXPOSURES:
        selected = prepared["exposure"].eq(exposure).to_numpy()
        high_from_cells[selected] = pd.to_numeric(
            cell_index[exposure].reindex(prepared.loc[selected, "high_erosion_cell_id"]), errors="coerce"
        ).to_numpy(dtype=float)
        low_from_cells[selected] = pd.to_numeric(
            cell_index[exposure].reindex(prepared.loc[selected, "low_erosion_cell_id"]), errors="coerce"
        ).to_numpy(dtype=float)
    input_high = pd.to_numeric(prepared["erosion_high"], errors="coerce").to_numpy(dtype=float)
    input_low = pd.to_numeric(prepared["erosion_low"], errors="coerce").to_numpy(dtype=float)
    aligned = np.isclose(high_from_cells, input_high, rtol=0.0, atol=1e-6) & np.isclose(
        low_from_cells, input_low, rtol=0.0, atol=1e-6
    )
    if not aligned.all():
        bad = prepared.loc[~aligned, "pair_id"].iloc[0]
        raise ValueError(f"Phase F Aa endpoints do not align with aligned Aa rasters for pair_id {bad}")

    # Use the precomputed matched-pair endpoint values after verifying their source-cell alignment.
    prepared["erosion_high"] = input_high
    prepared["erosion_low"] = input_low
    for outcome in outcomes:
        source = cell_index[outcome]
        prepared[f"high_{outcome}"] = source.reindex(prepared["high_erosion_cell_id"]).to_numpy()
        prepared[f"low_{outcome}"] = source.reindex(prepared["low_erosion_cell_id"]).to_numpy()
    matched_pairs = derive_pair_difference_features(prepared, outcomes)
    matched_pairs["both_positive"] = matched_pairs["Aa_high"].gt(0) & matched_pairs["Aa_low"].gt(0)
    matched_pairs["zero_positive"] = matched_pairs["Aa_low"].le(0) & matched_pairs["Aa_high"].gt(0)
    gap_q75 = matched_pairs.groupby(["method", "exposure"], sort=False)["delta_log1p_Aa"].transform(
        lambda values: values.quantile(0.75)
    )
    matched_pairs["high_erosion_gap_q4"] = matched_pairs["delta_log1p_Aa"].ge(gap_q75)
    matched_pairs["analysis_unit"] = "matched_pair_difference"
    matched_pairs["outcome_guided_selection"] = True

    pair_subsets = {
        "all": matched_pairs.copy(),
        "both_positive": matched_pairs.loc[matched_pairs["both_positive"]].copy(),
        "zero_positive": matched_pairs.loc[matched_pairs["zero_positive"]].copy(),
        "high_erosion_gap_q4": matched_pairs.loc[matched_pairs["high_erosion_gap_q4"]].copy(),
    }
    group_parts = []
    for column in group_columns:
        if column not in matched_pairs:
            continue
        group_part = matched_pairs.loc[matched_pairs[column].notna(), [
            name for name in PAIR_BASE_COLUMNS if name in matched_pairs
        ] + [column]].copy()
        group_part = group_part.rename(columns={column: "environment_group_label"})
        group_part["environment_group_label"] = group_part["environment_group_label"].astype(str)
        group_part["group_column"] = column
        group_parts.append(group_part)
    group_pairs = pd.concat(group_parts, ignore_index=True) if group_parts else pd.DataFrame()

    memberships, unique_pixels = _pixel_memberships(
        cells.loc[cells["SoilType"].isin(allowed)].copy(),
        matched_pairs,
        group_columns=group_columns,
    )
    summary_records = []
    for name, frame in pair_subsets.items():
        for (method, exposure), group in frame.groupby(["method", "exposure"], sort=True, dropna=False):
            summary_records.append(
                {
                    "dataset": f"Matched-Pair/{name}",
                    "method": str(method),
                    "exposure": str(exposure),
                    "row_count": int(len(group)),
                    "unique_pair_count": int(group["pair_id"].nunique()),
                    "unique_cell_count": int(pd.unique(pd.concat([
                        group["high_erosion_cell_id"], group["low_erosion_cell_id"]
                    ])).size),
                    "excluded_soil_pair_count": excluded_soil_pairs,
                }
            )
    for (method, exposure), group in memberships.groupby(["method", "exposure"], sort=True, dropna=False):
        unique_group = unique_pixels.loc[
            unique_pixels["method"].eq(method) & unique_pixels["exposure"].eq(exposure)
        ]
        summary_records.append(
            {
                "dataset": "Matched-Pixel/memberships",
                "method": str(method), "exposure": str(exposure), "row_count": int(len(group)),
                "unique_pair_count": int(group["pair_id"].nunique()),
                "unique_cell_count": int(group["cell_id"].nunique()),
                "excluded_soil_pair_count": excluded_soil_pairs,
            }
        )
        summary_records.append(
            {
                "dataset": "Matched-Pixel/unique_cells",
                "method": str(method), "exposure": str(exposure), "row_count": int(len(unique_group)),
                "unique_pair_count": int(group["pair_id"].nunique()),
                "unique_cell_count": int(unique_group["cell_id"].nunique()),
                "excluded_soil_pair_count": excluded_soil_pairs,
            }
        )
    summary = pd.DataFrame.from_records(summary_records)
    return MatchedDatasets(
        pair_memberships=memberships,
        pixel_unique=unique_pixels,
        pairs=matched_pairs,
        pair_subsets=pair_subsets,
        environment_group_pairs=group_pairs,
        summary=summary,
    )


def write_matched_datasets(run_dir: Path, datasets: MatchedDatasets) -> None:
    """Persist separate matched-pixel and matched-pair datasets under one Phase H run."""

    root = Path(run_dir)
    pixel_dir = root / "pixels"
    pair_dir = root / "pairs"
    table_dir = root / "tables"
    pixel_dir.mkdir(parents=True, exist_ok=True)
    pair_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    datasets.pair_memberships.to_parquet(pixel_dir / "matched_pixel_memberships.parquet", index=False)
    datasets.pixel_unique.to_parquet(pixel_dir / "matched_pixel_unique.parquet", index=False)
    for name, frame in datasets.pair_subsets.items():
        frame.to_parquet(pair_dir / f"matched_pairs_{name}.parquet", index=False)
    datasets.environment_group_pairs.to_parquet(
        pair_dir / "matched_pairs_by_environment_group.parquet", index=False
    )
    datasets.summary.to_csv(table_dir / "matched_dataset_summary.csv", index=False, encoding="utf-8-sig")


def run_h2_data_prep(run_dir: Path, cells: pd.DataFrame, pairs: pd.DataFrame, config: dict) -> MatchedDatasets:
    """Build H2 analysis populations from the loaded Phase F/G inputs and checkpoint counts."""

    root = Path(run_dir)
    checkpoint_path = root / "checkpoints" / "h2_complete.json"
    expected_outputs = [
        root / "pixels" / "matched_pixel_memberships.parquet",
        root / "pixels" / "matched_pixel_unique.parquet",
        root / "pairs" / "matched_pairs_all.parquet",
        root / "pairs" / "matched_pairs_both_positive.parquet",
        root / "pairs" / "matched_pairs_zero_positive.parquet",
        root / "pairs" / "matched_pairs_high_erosion_gap_q4.parquet",
        root / "pairs" / "matched_pairs_by_environment_group.parquet",
        root / "tables" / "matched_dataset_summary.csv",
    ]
    if checkpoint_path.is_file() and all(path.is_file() for path in expected_outputs):
        subsets = {
            name: pd.read_parquet(root / "pairs" / f"matched_pairs_{name}.parquet")
            for name in ("all", "both_positive", "zero_positive", "high_erosion_gap_q4")
        }
        return MatchedDatasets(
            pair_memberships=pd.read_parquet(expected_outputs[0]),
            pixel_unique=pd.read_parquet(expected_outputs[1]),
            pairs=subsets["all"],
            pair_subsets=subsets,
            environment_group_pairs=pd.read_parquet(expected_outputs[6]),
            summary=pd.read_csv(expected_outputs[7]),
        )

    outcomes = list(config["outcomes"])
    inference_config = config.get("interaction_inference", {})
    group_columns = list(dict.fromkeys([
        "environment_group", *inference_config.get("group_columns", ()),
    ]))
    datasets = build_matched_datasets(
        cells,
        pairs,
        outcomes=outcomes,
        primary_soils=config.get("primary_soil_codes", (26, 27, 42)),
        group_columns=group_columns,
    )
    write_matched_datasets(run_dir, datasets)
    checkpoint = {
        "task": "H2_matched_analysis_populations",
        "status": "complete",
        "pixel_membership_rows": int(len(datasets.pair_memberships)),
        "unique_pixel_rows": int(len(datasets.pixel_unique)),
        "pair_rows": int(len(datasets.pairs)),
        "pair_subset_counts": {name: int(len(frame)) for name, frame in datasets.pair_subsets.items()},
        "environment_group_membership_rows": int(len(datasets.environment_group_pairs)),
        "pixel_model_policy": "one row per unique cell_id within method/exposure; keep pair membership table and inverse-membership weights",
        "outcome_guided_selection": True,
    }
    (root / "checkpoints" / "h2_complete.json").write_text(
        json.dumps(checkpoint, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return datasets
