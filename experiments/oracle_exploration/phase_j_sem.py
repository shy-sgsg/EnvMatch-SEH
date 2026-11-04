"""Phase J exploratory SEM/path analysis using corrected CEC inputs.

This script reuses Phase I corrected inputs and existing M4 memberships/pairs.
All outputs are isolated in a unique Phase J Oracle run. It deliberately does
not rebuild RUSLE surfaces, rerun GWR/MGWR, or run PCNM.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy
import scipy.linalg
import scipy.stats as st
import statsmodels.api as sm
from sklearn.decomposition import PCA


ROOT = Path(__file__).resolve().parents[2]
CORRECTED_RUN = ROOT / "outputs/exploratory_oracle/phase_i/cec_corrected_input_20_2p9_plus18_20261031"
BASE_RUN = ROOT / "outputs/run_20261025T114741_470772Z"
PHASE_I = ROOT / "outputs/exploratory_oracle/phase_i"
PHASE_H = ROOT / "outputs/exploratory_oracle/phase_h_20261028_seed20261028_h2"
OUTCOMES = ["SHI_candidate", "AN", "OM", "CEC", "pH", "SW", "BD", "MBC", "RESP"]
HEALTH = ["AN", "OM", "CEC", "pH", "SW", "BD", "MBC", "RESP"]
ENV = ["Elevation", "Cur", "TWI", "LST", "NDWI", "NDBSI", "EVI", "NDVI", "Precipitation"]
ENV_M1 = ["Elevation", "TWI", "NDVI", "LST"]
TERRAIN_PC_INPUTS = ["Elevation", "Cur", "TWI"]
VEG_PC_INPUTS = ["NDVI", "EVI", "LST"]
CORE_RESPONSES = ["physical", "chemical", "biological_activity", "SHI_candidate"]
INDICATOR_RESPONSES = ["SHI_candidate", "SW", "OM", "CEC", "pH", "AN", "BD", "MBC", "RESP"]
RUN_ID_DEFAULT = "phase_j_20261102_seed20261102_r10"


def json_write(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def numeric_finite(frame: pd.DataFrame, cols: list[str]) -> pd.Series:
    return np.isfinite(frame[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)).all(axis=1)


def audit_inputs(run_dir: Path) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    env_path = CORRECTED_RUN / "databases/environment_erosion.parquet"
    health_path = CORRECTED_RUN / "databases/health.parquet"
    original_health_path = BASE_RUN / "databases/health.parquet"
    env = pd.read_parquet(env_path)
    health = pd.read_parquet(health_path)
    original_cec = pd.read_parquet(original_health_path, columns=["cell_id", "CEC"])
    if not env.cell_id.is_unique or not health.cell_id.is_unique or not original_cec.cell_id.is_unique:
        raise ValueError("Input cell_id is not unique; one-to-one merge cannot be trusted.")
    if set(env.cell_id) != set(health.cell_id) or set(env.cell_id) != set(original_cec.cell_id):
        raise ValueError("Input Parquet cell_id sets do not match exactly.")
    health_keep = ["cell_id", "row", "col", "x", "y", *OUTCOMES, "valid_cell"]
    pixels = env.merge(health[health_keep], on="cell_id", how="inner", validate="one_to_one", suffixes=("_env", "_health"))
    coord_check = {
        key: float((pixels[f"{key}_env"] - health.set_index("cell_id").loc[pixels.cell_id, key].to_numpy()).abs().max())
        for key in ("row", "col", "x", "y")
    }
    if any(v != 0 for v in coord_check.values()):
        raise ValueError(f"Pixel grid coordinates mismatch across input tables: {coord_check}")
    pixels["valid_env"] = pixels["valid_cell_env"].fillna(False).astype(bool)
    pixels["valid_health"] = pixels["valid_cell_health"].fillna(False).astype(bool)
    pixels["row"] = pixels["row_env"].astype("int32")
    pixels["col"] = pixels["col_env"].astype("int32")
    pixels["x"] = pixels["x_env"].astype(float)
    pixels["y"] = pixels["y_env"].astype(float)
    pixels = pixels.drop(columns=["row_env", "row_health", "col_env", "col_health", "x_env", "x_health", "y_env", "y_health", "valid_cell_env", "valid_cell_health"])

    manifest = json.loads((CORRECTED_RUN / "correction_manifest.json").read_text(encoding="utf-8"))
    joined = original_cec.merge(env[["cell_id", "Aa_20yr", "Aa_2020"]], on="cell_id", validate="one_to_one")
    maxima = manifest["maxima_full_valid_extent"]
    calc = (joined["CEC"] - 20.0 * (joined["Aa_20yr"] / maxima["Aa_20yr"] + joined["Aa_2020"] / maxima["Aa_2020"])) / 2.9 + 18.0
    cec_check = pd.DataFrame({"cell_id": joined.cell_id, "calculated": calc}).merge(
        health[["cell_id", "CEC"]], on="cell_id", how="inner", validate="one_to_one"
    )
    err = (cec_check.calculated - cec_check.CEC).abs()
    cec_error = {"n_valid": int(err.notna().sum()), "max_abs_error": float(err.max()),
                 "median_abs_error": float(err.median()), "q99_abs_error": float(err.quantile(.99))}
    if cec_error["max_abs_error"] > 1e-10:
        raise ValueError(f"Stored CEC does not match the declared corrected formula: {cec_error}")

    phase_j_required = ["Aa_20yr", *OUTCOMES, "SoilType", "x", "y", *ENV]
    phase_j_finite = numeric_finite(pixels, phase_j_required)
    false_env_but_required_finite = ~pixels.valid_env & pixels.valid_health & phase_j_finite

    profile_rows = []
    for col in ["Aa_20yr", "Aa_2020", *ENV, *OUTCOMES]:
        series = pd.to_numeric(pixels[col], errors="coerce")
        profile_rows.append({"variable": col, "n_valid": int(series.notna().sum()), "n_missing": int(series.isna().sum()),
                             "minimum": float(series.min()), "p01": float(series.quantile(.01)),
                             "median": float(series.median()), "p99": float(series.quantile(.99)), "maximum": float(series.max())})
    qa = {
        "source_run": str(CORRECTED_RUN.relative_to(ROOT)),
        "environment_rows": int(len(env)), "health_rows": int(len(health)),
        "cell_id_unique": True, "cell_id_sets_equal": True, "one_to_one_join_rows": int(len(pixels)),
        "valid_env_rows": int(pixels.valid_env.sum()), "valid_health_rows": int(pixels.valid_health.sum()),
        "valid_env_health_intersection": int((pixels.valid_env & pixels.valid_health).sum()),
        "phase_j_model_required_variables_finite_rows": int(phase_j_finite.sum()),
        "valid_env_false_but_phase_j_required_variables_finite_rows": int(false_env_but_required_finite.sum()),
        "auxiliary_missing_counts_in_these_rows": {
            col: int(pixels.loc[false_env_but_required_finite, col].isna().sum())
            for col in ["Aa_20yr_nearest", "Aa_2020_nearest"] if col in pixels
        },
        "validity_note": "D1 complete-case construction uses only the variables required by each model. Seventy-one rows have valid health and finite Phase J model variables while the broader environment-stack flag is false because an auxiliary nearest-exposure field is missing; they are not missing any variable used in the Phase J model.",
        "coordinate_max_abs_differences": coord_check,
        "corrected_cec_formula_check": cec_error,
        "corrected_cec_range": [float(health.CEC.min()), float(health.CEC.max())],
        "outcome_guided_selection": True,
    }
    write_table(pd.DataFrame(profile_rows), run_dir / "data/input_profile.csv")
    json_write(run_dir / "provenance/data_quality_audit.json", qa)
    pixels.to_parquet(run_dir / "data/full_cropland_predictions.parquet", index=False)
    return pixels, qa, manifest


def build_datasets(pixels: pd.DataFrame, run_dir: Path, manifest: dict[str, Any]) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    health_complete = ["Aa_20yr", "Aa_2020", *OUTCOMES, "SoilType", "x", "y"]
    d1_all = pixels.loc[numeric_finite(pixels, [c for c in health_complete if c not in {"SoilType", "x", "y"}])].copy()
    common_cols = ["Aa_20yr", *OUTCOMES, "SoilType", "x", "y", *ENV]
    d1_common = pixels.loc[numeric_finite(pixels, [c for c in common_cols if c not in {"SoilType", "x", "y"}])].copy()
    d1_common = d1_common.dropna(subset=["SoilType", "x", "y"])

    # Reconstruct the exact Phase I-8 8x8 support (SHI, Aa_20yr, and all nine environment covariates).
    i8_required = ["Aa_20yr", "SHI_candidate", *ENV]
    i8_pixels = pixels.loc[numeric_finite(pixels, i8_required)].copy()
    i8_pixels["tile_row"] = np.floor(i8_pixels.row / 8).astype("int32")
    i8_pixels["tile_col"] = np.floor(i8_pixels.col / 8).astype("int32")
    tile_keys = i8_pixels[["tile_row", "tile_col"]].drop_duplicates()
    existing_i8 = pd.read_parquet(PHASE_I / "phase_i_20261031_seed20261031_i8_r2/tables/stress_zones_240m.parquet")
    i8_check = i8_pixels.groupby(["tile_row", "tile_col"], sort=True).agg(
        n_pixels=("cell_id", "size"), Aa_20yr_mean=("Aa_20yr", "mean"), SHI_mean=("SHI_candidate", "mean")
    ).reset_index().merge(existing_i8[["tile_row", "tile_col", "n_pixels", "Aa_20yr_mean", "SHI_mean"]],
                           on=["tile_row", "tile_col"], how="outer", suffixes=("_recreated", "_phase_i8"), indicator=True)
    if len(i8_check) != len(existing_i8) or not i8_check._merge.eq("both").all():
        raise ValueError("Could not reproduce Phase I-8's 240 m support exactly.")
    diffs = {col: float((i8_check[f"{col}_recreated"] - i8_check[f"{col}_phase_i8"]).abs().max())
             for col in ["n_pixels", "Aa_20yr_mean", "SHI_mean"]}
    if diffs["n_pixels"] != 0 or diffs["Aa_20yr_mean"] > 1e-9 or diffs["SHI_mean"] > 1e-9:
        raise ValueError(f"Phase I-8 240m support did not reproduce: {diffs}")
    # Aggregate all health indicators on that same environmental/SHI support.
    aggregate_cols = ["Aa_20yr", "Aa_2020", *OUTCOMES, *ENV, "SoilType", "x", "y"]
    g = i8_pixels.groupby(["tile_row", "tile_col"], sort=True)
    d2 = g[aggregate_cols].mean().reset_index()
    d2["n_pixels"] = g.size().to_numpy()
    d2["x"] = g.x.mean().to_numpy()
    d2["y"] = g.y.mean().to_numpy()
    # Replace the soil mean with the mode so the class code remains categorical.
    d2["SoilType"] = g["SoilType"].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan).to_numpy()
    d2.to_parquet(run_dir / "data/D2_240m_spatial_units.parquet", index=False)
    support = {}
    base_counts = g.size()
    for col in aggregate_cols:
        per_block = g[col].count()
        missing_pixels = int((base_counts - per_block).sum())
        support[col] = {"valid_pixel_support": int(per_block.sum()), "missing_pixel_support": missing_pixels,
                        "blocks_with_missing_support": int((per_block < base_counts).sum())}

    # M4 unique endpoint pixels: phase H membership file has original CEC, so only IDs are trusted.
    member_path = PHASE_H / "pixels/matched_pixel_unique.parquet"
    members = pd.read_parquet(member_path, columns=["method", "exposure", "cell_id", "membership_count"])
    members = members.loc[members.method.eq("M4") & members.exposure.eq("Aa_20yr")].copy()
    if members.cell_id.duplicated().any():
        raise ValueError("M4 unique-pixel membership contains duplicate cell_id values.")
    lookup = pixels.set_index("cell_id", drop=False)
    d3 = lookup.loc[members.cell_id.to_numpy()].reset_index(drop=True).copy()
    d3["membership_count"] = members.membership_count.to_numpy()

    # Pair differences: rejoin both endpoints to the corrected Phase I input by cell_id.
    pair_path = CORRECTED_RUN / "pairs/candidate_pairs/stage2_Aa_20yr_01.parquet"
    source_pairs = pd.read_parquet(pair_path)
    meta_cols = ["pair_id", "high_erosion_cell_id", "low_erosion_cell_id", "soil_type", "method", "candidate_id", "exposure"]
    pmeta = source_pairs[meta_cols].copy()
    value_cols = ["Aa_20yr", "Aa_2020", "SHI_candidate", *HEALTH, *ENV, "SoilType", "x", "y"]
    high = pixels[["cell_id", *value_cols]].rename(columns={"cell_id": "high_erosion_cell_id", **{c: f"{c}_high" for c in value_cols}})
    low = pixels[["cell_id", *value_cols]].rename(columns={"cell_id": "low_erosion_cell_id", **{c: f"{c}_low" for c in value_cols}})
    pairs = pmeta.merge(high, on="high_erosion_cell_id", validate="one_to_one").merge(low, on="low_erosion_cell_id", validate="one_to_one")
    if pairs.pair_id.duplicated().any():
        raise ValueError("M4 pair_id is not unique.")
    pairs["erosion"] = np.log1p(pairs.Aa_20yr_high) - np.log1p(pairs.Aa_20yr_low)
    pairs["x"] = (pairs.x_high + pairs.x_low) / 2
    pairs["y"] = (pairs.y_high + pairs.y_low) / 2
    pair_diff_cols = [*OUTCOMES, *ENV]
    for c in pair_diff_cols:
        pairs[f"d_{c}"] = pairs[f"{c}_high"] - pairs[f"{c}_low"]
    pairs["zero_positive"] = (pairs.Aa_20yr_high.gt(0) & pairs.Aa_20yr_low.eq(0))
    pairs["both_positive"] = (pairs.Aa_20yr_high.gt(0) & pairs.Aa_20yr_low.gt(0))
    pairs["n_valid_erosion"] = pairs[["Aa_20yr_high", "Aa_20yr_low"]].notna().all(axis=1)
    if pairs[["Aa_20yr_high", "Aa_20yr_low"]].isna().any().any():
        raise ValueError("Selected M4 pair endpoints are missing Aa_20yr values after corrected join.")
    if pd.concat([pairs.high_erosion_cell_id, pairs.low_erosion_cell_id]).duplicated().any():
        raise ValueError("Selected M4 pair endpoints are reused; paired sample dependence needs another design.")
    d3_ids = set(d3.cell_id)
    d4_ids = set(pd.concat([pairs.high_erosion_cell_id, pairs.low_erosion_cell_id], ignore_index=True))
    d3_d4_overlap = {
        "D3_phase_h_unique_endpoint_count": len(d3_ids),
        "D4_stage2_pair_endpoint_count": len(d4_ids),
        "shared_endpoint_count": len(d3_ids & d4_ids),
        "D3_only_endpoint_count": len(d3_ids - d4_ids),
        "D4_only_endpoint_count": len(d4_ids - d3_ids),
        "interpretation": "D3 and D4 are different historical M4 match versions and are not a single shared endpoint population",
    }

    # Positive erosion and environment groups use the existing full-raster support, not a new match.
    d5 = d1_common.loc[d1_common.Aa_20yr.gt(0)].copy()
    d6 = pairs.loc[pairs.zero_positive].copy()
    d7base = d1_common.loc[d1_common.SoilType.isin([26, 27, 42])].copy()
    d7base["twi_group"] = ""
    for soil in [26, 27, 42]:
        ix = d7base.SoilType.eq(soil)
        rank = d7base.loc[ix, "TWI"].rank(method="first", pct=True)
        d7base.loc[ix, "twi_group"] = pd.cut(rank, [0, .25, .5, .75, 1], labels=["q1", "q2", "q3", "q4"], include_lowest=True).astype(str).to_numpy()

    datasets = {"D1_full_pixels": d1_all, "D1_common_env_case": d1_common, "D2_240m_blocks": d2,
                "D3_M4_unique_pixels": d3, "D4_M4_pair_differences": pairs,
                "D5_positive_erosion": d5, "D6_zero_positive_pairs": d6, "D7_soil_TWI_groups": d7base}
    datasets["D1_full_pixels"].to_parquet(run_dir / "data/D1_full_pixels.parquet", index=False)
    datasets["D1_common_env_case"].to_parquet(run_dir / "data/D1_common_env_case.parquet", index=False)
    datasets["D3_M4_unique_pixels"].to_parquet(run_dir / "data/D3_M4_unique_pixels.parquet", index=False)
    datasets["D4_M4_pair_differences"].to_parquet(run_dir / "data/D4_M4_pair_differences.parquet", index=False)
    datasets["D5_positive_erosion"].to_parquet(run_dir / "data/D5_positive_erosion.parquet", index=False)
    datasets["D6_zero_positive_pairs"].to_parquet(run_dir / "data/D6_zero_positive_pairs.parquet", index=False)
    datasets["D7_soil_TWI_groups"].to_parquet(run_dir / "data/D7_soil_TWI_groups.parquet", index=False)

    qa = {"phase_i8_240m_reproduction": {"rows": int(len(d2)), "pixels": int(len(i8_pixels)), "max_abs_differences": diffs},
          "D2_variable_specific_support": {"base_pixels": int(len(i8_pixels)), "blocks": int(len(d2)), "variables": support,
                                           "aggregation": "each block-variable mean uses that variable's non-missing pixels"},
          "D1_full_complete_pixels": int(len(d1_all)), "D1_common_environment_complete_pixels": int(len(d1_common)),
          "D3_M4_unique_pixels": int(len(d3)), "D4_M4_pairs": int(len(pairs)),
          "D4_pair_endpoint_unique_count": int(pd.concat([pairs.high_erosion_cell_id, pairs.low_erosion_cell_id]).nunique()),
          "D4_zero_positive_pairs": int(pairs.zero_positive.sum()), "D4_both_positive_pairs": int(pairs.both_positive.sum()),
          "D5_positive_pixels": int(len(d5)), "D6_zero_positive_pairs": int(len(d6)),
          "D7_soil_twi_groups": d7base.groupby(["SoilType", "twi_group"], observed=True).size().rename("n").reset_index().to_dict("records"),
          "D4_pair_source": str(pair_path.relative_to(ROOT)), "D3_membership_source": str(member_path.relative_to(ROOT)),
          "D3_D4_endpoint_overlap": d3_d4_overlap,
          "matching_versions": {
              "D3": {"source": "Phase H matched pairs inherited from Phase F/G M4 Aa_20yr design", "environment_caliper": 0.35,
                     "geographic_radius_m": 1000, "minimum_within_group_erosion_percentile_gap": 0.50,
                     "matched_pairs": int(len(d3) // 2), "unique_endpoints": int(len(d3))},
              "D4": {"source_candidate_id": "stage2_Aa_20yr_01", "stage": "Phase H Stage 2", "environment_caliper": 0.75,
                     "geographic_radius_m": 500, "minimum_within_group_erosion_percentile_gap": 0.30,
                     "kmeans_groups_per_soil_type": 8, "matched_pairs": int(len(pairs)),
                     "unique_endpoints": len(d4_ids)},
          },
          "outcome_guided_selection": True}
    json_write(run_dir / "provenance/population_audit.json", qa)
    counts = []
    for name, data in datasets.items():
        counts.append({"dataset": name, "n_rows": len(data), "n_unique_cell_or_pair": int(data.cell_id.nunique()) if "cell_id" in data else int(data.pair_id.nunique()) if "pair_id" in data else len(data),
                       "analysis_unit": "matched pair" if "pair_id" in data else "30m raster pixel" if "cell_id" in data else "240m block" if name.startswith("D2") else "environment subgroup"})
    write_table(pd.DataFrame(counts), run_dir / "data/population_counts.csv")
    pairs.to_parquet(run_dir / "data/D4_M4_pair_differences.parquet", index=False)
    return datasets, qa


def set_scale_reference(d1: pd.DataFrame) -> dict[str, tuple[float, float]]:
    refs = {}
    for name in [*OUTCOMES, *ENV, "Aa_20yr", "Aa_2020"]:
        v = pd.to_numeric(d1[name], errors="coerce").dropna()
        refs[name] = (float(v.mean()), float(v.std(ddof=1)))
    return refs


def add_model_variables(frame: pd.DataFrame, dataset_name: str, refs: dict[str, tuple[float, float]], pca_params: dict[str, Any]) -> pd.DataFrame:
    d = frame.copy()
    paired = dataset_name in {"D4_M4_pair_differences", "D6_zero_positive_pairs"}
    for outcome in OUTCOMES:
        valcol = f"d_{outcome}" if paired else outcome
        mean, sd = refs[outcome]
        d[f"z_{outcome}"] = (pd.to_numeric(d[valcol], errors="coerce") / sd) if paired else ((pd.to_numeric(d[valcol], errors="coerce") - mean) / sd)
    for name in ENV:
        valcol = f"d_{name}" if paired else name
        mean, sd = refs[name]
        d[f"z_{name}"] = (pd.to_numeric(d[valcol], errors="coerce") / sd) if paired else ((pd.to_numeric(d[valcol], errors="coerce") - mean) / sd)
    if paired:
        d["E"] = d["erosion"]
    else:
        exposure = "Aa_20yr"
        d["E"] = np.log1p(pd.to_numeric(d[exposure], errors="coerce"))
    # Dimension scores are transparent equal-weight composites; corrected CEC is a separate sensitivity.
    d["physical"] = (d["z_SW"] - d["z_BD"]) / 2
    d["chemical"] = (d["z_OM"] + d["z_AN"]) / 2
    d["chemical_plus_CEC"] = (d["z_OM"] + d["z_AN"] + d["z_CEC"]) / 3
    d["biological_activity"] = (d["z_MBC"] + d["z_RESP"]) / 2
    d["z_SHI"] = d["z_SHI_candidate"]
    # Environmental PC scores are fit only from environment data; pair differences use the same fixed loadings.
    for label, features in [("terrain_PC", TERRAIN_PC_INPUTS), ("vegetation_thermal_PC", VEG_PC_INPUTS)]:
        params = pca_params[label]
        raw = d[[f"z_{f}" for f in features]].to_numpy(dtype=float)
        d[label] = raw @ np.asarray(params["component"], dtype=float)
    d["spatial_block_5km"] = (np.floor(pd.to_numeric(d.x, errors="coerce") / 5000).astype("Int64").astype(str)
                               + "_" + np.floor(pd.to_numeric(d.y, errors="coerce") / 5000).astype("Int64").astype(str))
    d["z_E"] = (d.E - d.E.mean()) / d.E.std(ddof=1)
    # Soil groups are limited to the three documented main soil classes plus a combined remainder.
    soil_col = "soil_type" if "soil_type" in d.columns and paired else "SoilType"
    if soil_col not in d.columns:
        d["soil_group"] = "other"
    else:
        soil = pd.to_numeric(d[soil_col], errors="coerce")
        d["soil_group"] = np.select([soil.eq(26), soil.eq(27), soil.eq(42)], ["26", "27", "42"], default="other")
    return d


def make_pca_params(d1_common: pd.DataFrame) -> dict[str, Any]:
    result = {}
    for label, features in [("terrain_PC", TERRAIN_PC_INPUTS), ("vegetation_thermal_PC", VEG_PC_INPUTS)]:
        zcols = []
        for f in features:
            m, s = float(d1_common[f].mean()), float(d1_common[f].std(ddof=1))
            zcols.append((d1_common[f].to_numpy(dtype=float) - m) / s)
        X = np.column_stack(zcols)
        pca = PCA(n_components=1, svd_solver="full").fit(X)
        comp = pca.components_[0].copy()
        anchor = "TWI" if label == "terrain_PC" else "NDVI"
        anchor_idx = features.index(anchor)
        if comp[anchor_idx] < 0:
            comp *= -1
        result[label] = {"features": features, "anchor_positive": anchor,
                         "component": comp.tolist(), "explained_variance_ratio": float(pca.explained_variance_ratio_[0]),
                         "center": [float(d1_common[f].mean()) for f in features],
                         "scale": [float(d1_common[f].std(ddof=1)) for f in features]}
    return result


def control_columns(d: pd.DataFrame, model: str, paired: bool) -> list[str]:
    # add_model_variables stores standardized pair deltas under z_<environment>
    # (the differenced source column is d_<environment>), so the model column
    # names are identical for paired and pixel-level populations.
    m1 = [f"z_{c}" for c in ENV_M1]
    # Choose a soil reference category that is present in the population; omit
    # the reference dummy to avoid exact collinearity when only main soil classes
    # are represented or when a D7 subgroup contains one soil class.
    codes = ["26", "27", "42"]
    observed = set(d.soil_group.dropna().astype(str).unique())
    present = [code for code in codes if code in observed]
    reference = None
    if "other" not in observed and len(present) > 1:
        reference = "42" if "42" in present else present[-1]
    included = [code for code in present if code != reference]
    soil_dummies = [f"soil_{code}" for code in included]
    for code in codes:
        d[f"soil_{code}"] = d.soil_group.eq(code).astype(float)
    m1 += soil_dummies
    if model == "M0":
        return []
    if model == "M1":
        return m1
    if model == "M2":
        return ["terrain_PC", "vegetation_thermal_PC", *soil_dummies]
    if model == "M3":
        xs = (d.x - d.x.mean()) / d.x.std(ddof=1)
        ys = (d.y - d.y.mean()) / d.y.std(ddof=1)
        d["space_x"] = xs
        d["space_y"] = ys
        d["space_x2"] = xs ** 2
        d["space_y2"] = ys ** 2
        d["space_xy"] = xs * ys
        return [*m1, "space_x", "space_y", "space_x2", "space_y2", "space_xy"]
    raise KeyError(model)


def fit_ols(d: pd.DataFrame, yname: str, predictor_names: list[str], model: str,
            dataset: str, structure: str, method: str = "piecewise_sem") -> tuple[list[dict[str, Any]], dict[str, Any], Any]:
    required = [yname, *predictor_names, "spatial_block_5km"]
    valid = d[required].replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) <= len(predictor_names) + 3:
        return [], {"n": len(valid), "status": "insufficient_rows"}, None
    yraw = valid[yname].to_numpy(dtype=float)
    ysd = np.std(yraw, ddof=1)
    if not np.isfinite(ysd) or ysd == 0:
        return [], {"n": len(valid), "status": "zero_variance"}, None
    y = (yraw - np.mean(yraw)) / ysd
    Xraw = valid[predictor_names].to_numpy(dtype=float) if predictor_names else np.empty((len(valid), 0))
    col_sd = np.std(Xraw, axis=0, ddof=1) if Xraw.shape[1] else np.array([])
    variable = col_sd > 1e-12
    kept = [n for n, keep in zip(predictor_names, variable) if keep]
    Xraw = Xraw[:, variable]
    xmean = np.mean(Xraw, axis=0) if Xraw.shape[1] else np.array([])
    xsd = np.std(Xraw, axis=0, ddof=1) if Xraw.shape[1] else np.array([])
    X = (Xraw - xmean) / xsd if Xraw.shape[1] else Xraw
    X = sm.add_constant(X, has_constant="add")
    groups = valid.spatial_block_5km.astype(str).to_numpy()
    n_groups = int(pd.Series(groups).nunique())
    try:
        Xdf = pd.DataFrame(X, columns=["intercept", *kept], index=valid.index)
        if n_groups >= 8:
            fit = sm.OLS(y, Xdf).fit(cov_type="cluster", cov_kwds={"groups": groups, "use_correction": True}, use_t=True)
            se_type = "5km block-cluster robust"
        else:
            fit = sm.OLS(y, Xdf).fit(cov_type="HC3")
            se_type = "HC3; fewer than 8 spatial blocks"
        params = np.asarray(fit.params, dtype=float)
        covariance = np.asarray(fit.cov_params(), dtype=float)
        variance = np.diag(covariance)
        se = np.full_like(params, np.nan)
        valid_variance = np.isfinite(variance) & (variance >= 0)
        se[valid_variance] = np.sqrt(variance[valid_variance])
        infer_df = float(getattr(fit, "df_resid_inference", fit.df_resid))
        critical = float(st.t.ppf(0.975, infer_df)) if infer_df > 0 else np.nan
        pvalues = np.full_like(params, np.nan)
        valid_se = np.isfinite(se) & (se > 0)
        pvalues[valid_se] = 2 * st.t.sf(np.abs(params[valid_se] / se[valid_se]), infer_df)
        ci_low = params - critical * se
        ci_high = params + critical * se
        names = ["intercept", *kept]
        path_rows = []
        for j, term in enumerate(names):
            if term == "intercept":
                continue
            idx = j
            path_rows.append({"dataset": dataset, "method": method, "structure": structure, "control": model,
                              "from": term, "to": yname, "path_type": "direct", "beta_standardized": float(params[idx]),
                              "se": float(se[idx]), "ci_low": float(ci_low[idx]), "ci_high": float(ci_high[idx]),
                              "p_value_nominal": float(pvalues[idx]),
                              "interval_status": "available" if np.isfinite(se[idx]) else "covariance variance non-positive/non-finite",
                              "r_squared": float(fit.rsquared),
                              "adjusted_r_squared": float(fit.rsquared_adj), "n": int(fit.nobs),
                              "spatial_blocks_5km": n_groups, "se_type": se_type,
                              "design_condition_number": float(np.linalg.cond(X)), "outcome_guided_selection": True})
        fitrow = {"dataset": dataset, "method": method, "structure": structure, "control": model,
                  "endogenous": yname, "n": int(fit.nobs), "spatial_blocks_5km": n_groups,
                  "r_squared": float(fit.rsquared), "adjusted_r_squared": float(fit.rsquared_adj),
                  "aic": float(fit.aic), "bic": float(fit.bic), "design_condition_number": float(np.linalg.cond(X)),
                  "se_type": se_type, "invalid_covariance_variance_count": int((~valid_variance).sum()),
                  "status": "ok", "outcome_guided_selection": True}
        return path_rows, fitrow, fit
    except Exception as exc:
        return [], {"dataset": dataset, "method": method, "structure": structure, "control": model,
                    "endogenous": yname, "n": int(len(valid)), "status": f"failed:{type(exc).__name__}:{exc}",
                    "outcome_guided_selection": True}, None


def assemble_piecewise(datasets: dict[str, pd.DataFrame], refs: dict[str, tuple[float, float]], pca_params: dict[str, Any], run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_paths, all_fits, increments, indirect = [], [], [], []
    primary_names = ["D1_common_env_case", "D2_240m_blocks", "D3_M4_unique_pixels", "D4_M4_pair_differences",
                     "D5_positive_erosion", "D6_zero_positive_pairs"]
    control_models = ["M0", "M1", "M2", "M3"]
    for dataset in primary_names:
        raw = datasets[dataset]
        if dataset == "D4_M4_pair_differences":
            raw = raw.loc[raw.both_positive | raw.zero_positive].copy()
        d = add_model_variables(raw, dataset, refs, pca_params)
        paired = dataset in {"D4_M4_pair_differences", "D6_zero_positive_pairs"}
        required = ["E", "physical", "chemical", "biological_activity", "z_SHI", "z_CEC", "z_pH", "z_SW", "z_OM", "z_MBC", "z_RESP", "spatial_block_5km", "x", "y"]
        envcols = [f"z_{('d_' if paired else '')}{c}" for c in ENV_M1] + ["terrain_PC", "vegetation_thermal_PC"]
        d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=required + envcols)
        for control in control_models:
            ccols = control_columns(d, control, paired)
            base_preds = ["z_E", *ccols]
            eq_fits: dict[str, Any] = {}
            # S1: erosion -> three health dimensions.
            for y in ["physical", "chemical", "biological_activity"]:
                rows, fitrow, fit = fit_ols(d, y, base_preds, control, dataset, "S1_parallel_dimensions")
                all_paths.extend(rows); all_fits.append(fitrow); eq_fits[y] = fit
                if fit is not None:
                    no_rows, no_fitrow, no_fit = fit_ols(d, y, ccols, control, dataset, "S1_without_erosion")
                    increments.append({"dataset": dataset, "method": "piecewise_sem", "structure": "S1_parallel_dimensions",
                                       "control": control, "outcome": y, "n": fitrow["n"], "r2_full": fitrow["r_squared"],
                                       "r2_without_erosion": no_fitrow.get("r_squared", np.nan),
                                       "delta_r2_erosion_conditional": fitrow["r_squared"] - no_fitrow.get("r_squared", np.nan),
                                       "interpretation": "conditional predictive increment; not causal contribution",
                                       "outcome_guided_selection": True})
            # S2 and S3: dimensions -> SHI, with / without a direct erosion path.
            for structure, predictors in [("S2_direct_plus_indirect_SHI", ["z_E", "physical", "chemical", "biological_activity", *ccols]),
                                          ("S3_indirect_only_SHI", ["physical", "chemical", "biological_activity", *ccols])]:
                rows, fitrow, fit = fit_ols(d, "z_SHI", predictors, control, dataset, structure)
                all_paths.extend(rows); all_fits.append(fitrow); eq_fits[structure] = fit
                if fit is not None and structure.startswith("S2"):
                    no_rows, no_fitrow, no_fit = fit_ols(d, "z_SHI", ["physical", "chemical", "biological_activity", *ccols], control, dataset, "S2_without_direct_erosion")
                    increments.append({"dataset": dataset, "method": "piecewise_sem", "structure": structure,
                                       "control": control, "outcome": "z_SHI", "n": fitrow["n"], "r2_full": fitrow["r_squared"],
                                       "r2_without_erosion": no_fitrow.get("r_squared", np.nan),
                                       "delta_r2_erosion_conditional": fitrow["r_squared"] - no_fitrow.get("r_squared", np.nan),
                                       "interpretation": "direct erosion increment conditional on dimension scores; indirect paths remain",
                                       "outcome_guided_selection": True})
                    # Total erosion increment in an unmediated SHI response equation is reported separately.
                    total_rows, total_fitrow, _ = fit_ols(d, "z_SHI", base_preds, control, dataset, "SHI_total_association")
                    total_no_rows, total_no_fitrow, _ = fit_ols(d, "z_SHI", ccols, control, dataset, "SHI_without_erosion_total")
                    increments.append({"dataset": dataset, "method": "piecewise_sem", "structure": "SHI_total_association",
                                       "control": control, "outcome": "z_SHI", "n": total_fitrow.get("n"),
                                       "r2_full": total_fitrow.get("r_squared"), "r2_without_erosion": total_no_fitrow.get("r_squared"),
                                       "delta_r2_erosion_conditional": total_fitrow.get("r_squared", np.nan) - total_no_fitrow.get("r_squared", np.nan),
                                       "interpretation": "total bivariate/controlled association increment, not a mediated causal effect",
                                       "outcome_guided_selection": True})
            # S4 chain: E -> physical -> chemical -> biological, retaining E as a direct input.
            for y, preds in [("physical", base_preds), ("chemical", ["z_E", "physical", *ccols]),
                             ("biological_activity", ["z_E", "physical", "chemical", *ccols]),
                             ("z_SHI", ["z_E", "physical", "chemical", "biological_activity", *ccols])]:
                rows, fitrow, fit = fit_ols(d, y, preds, control, dataset, "S4_chained_health_dimensions")
                all_paths.extend(rows); all_fits.append(fitrow)
            # S5: all individual indicators plus the provisional SHI response.
            for y in INDICATOR_RESPONSES:
                yname = f"z_{y}"
                rows, fitrow, _ = fit_ols(d, yname, base_preds, control, dataset, "S5_indicator_responses")
                all_paths.extend(rows); all_fits.append(fitrow)
            # Store path products as descriptive indirect associations; coefficient intervals are in the direct table.
            s2 = eq_fits.get("S2_direct_plus_indirect_SHI")
            if s2 is not None:
                coef = dict(zip(s2.model.exog_names, s2.params))
                indirect_sum = 0.0
                for dim in ["physical", "chemical", "biological_activity"]:
                    upstream = eq_fits.get(dim)
                    if upstream is None:
                        continue
                    b_e = float(upstream.params[upstream.model.exog_names.index("z_E")])
                    b_h = float(coef.get(dim, np.nan))
                    value = b_e * b_h
                    indirect_sum += value
                    indirect.append({"dataset": dataset, "method": "piecewise_sem", "control": control,
                                     "path": f"E→{dim}→SHI", "direct_effect": np.nan, "indirect_effect": value,
                                     "total_effect_component": value, "ci_low": np.nan, "ci_high": np.nan,
                                     "interval_status": "not computed; report point estimate only",
                                     "interpretation": "product of standardized conditional associations; not causal mediation",
                                     "outcome_guided_selection": True})
                direct = float(coef.get("z_E", 0.0))
                indirect.append({"dataset": dataset, "method": "piecewise_sem", "control": control,
                                 "path": "E→SHI total (direct + dimension-mediated association)", "direct_effect": direct,
                                 "indirect_effect": indirect_sum, "total_effect_component": direct + indirect_sum,
                                 "ci_low": np.nan, "ci_high": np.nan,
                                 "interval_status": "not computed; report point estimate only",
                                 "interpretation": "observed-path decomposition; dimensions and SHI share construction inputs",
                                 "outcome_guided_selection": True})

    paths = pd.DataFrame(all_paths)
    fits = pd.DataFrame(all_fits)
    delta = pd.DataFrame(increments)
    effects = pd.DataFrame(indirect)
    # S6 hurdle model: any erosion vs positive exposure intensity, only for full-raster pixel populations.
    d1 = add_model_variables(datasets["D1_common_env_case"], "D1_common_env_case", refs, pca_params)
    d1 = d1.replace([np.inf, -np.inf], np.nan).dropna(subset=["Aa_20yr", "spatial_block_5km", *[f"z_{c}" for c in ENV_M1]])
    d1["any_erosion"] = d1.Aa_20yr.gt(0).astype(int)
    for control in control_models:
        ccols = control_columns(d1, control, False)
        for outcome, sample, predictors, family in [
            ("erosion_occurrence", d1, ccols, "binomial_occurrence"),
            ("positive_log1p_Aa", d1.loc[d1.Aa_20yr.gt(0)].copy(), ccols, "positive_intensity_OLS"),
        ]:
            if family == "binomial_occurrence":
                valid = sample[["any_erosion", *predictors, "spatial_block_5km"]].dropna()
                X = sm.add_constant(valid[predictors].astype(float), has_constant="add")
                groups = valid.spatial_block_5km.astype(str).to_numpy()
                try:
                    fit = sm.GLM(valid.any_erosion.astype(float), X, family=sm.families.Binomial()).fit(
                        cov_type="cluster", cov_kwds={"groups": groups, "use_correction": True})
                    params = np.asarray(fit.params, dtype=float)
                    covariance = np.asarray(fit.cov_params(), dtype=float)
                    variance = np.diag(covariance)
                    se = np.full_like(params, np.nan)
                    valid_variance = np.isfinite(variance) & (variance >= 0)
                    se[valid_variance] = np.sqrt(variance[valid_variance])
                    pvalues = np.full_like(params, np.nan)
                    valid_se = np.isfinite(se) & (se > 0)
                    pvalues[valid_se] = 2 * st.norm.sf(np.abs(params[valid_se] / se[valid_se]))
                    ci_low, ci_high = params - st.norm.ppf(.975) * se, params + st.norm.ppf(.975) * se
                    for j, term in enumerate(["intercept", *predictors]):
                        if term == "intercept": continue
                        idx = j
                        all_paths.append({"dataset": "D1_common_env_case", "method": "piecewise_sem", "structure": "S6_two_part_erosion",
                                          "control": control, "from": term, "to": outcome, "path_type": "logit",
                                          "beta_standardized": float(params[idx]), "se": float(se[idx]),
                                          "ci_low": float(ci_low[idx]), "ci_high": float(ci_high[idx]),
                                          "p_value_nominal": float(pvalues[idx]),
                                          "interval_status": "available" if np.isfinite(se[idx]) else "covariance variance non-positive/non-finite",
                                          "r_squared": np.nan,
                                          "adjusted_r_squared": np.nan, "n": int(fit.nobs),
                                          "spatial_blocks_5km": int(pd.Series(groups).nunique()), "se_type": "5km block-cluster robust",
                                          "design_condition_number": float(np.linalg.cond(X)), "outcome_guided_selection": True})
                    all_fits.append({"dataset": "D1_common_env_case", "method": "piecewise_sem", "structure": "S6_two_part_erosion",
                                     "control": control, "endogenous": outcome, "n": int(fit.nobs),
                                     "spatial_blocks_5km": int(pd.Series(groups).nunique()), "r_squared": np.nan,
                                     "adjusted_r_squared": np.nan, "aic": float(fit.aic), "bic": float(fit.bic),
                                     "design_condition_number": float(np.linalg.cond(X)), "se_type": "5km block-cluster robust",
                                     "status": "ok", "outcome_guided_selection": True})
                except Exception as exc:
                    all_fits.append({"dataset": "D1_common_env_case", "method": "piecewise_sem", "structure": "S6_two_part_erosion",
                                     "control": control, "endogenous": outcome, "n": len(valid), "status": f"failed:{exc}", "outcome_guided_selection": True})
            else:
                sample["positive_log1p_Aa"] = np.log1p(sample.Aa_20yr)
                rows, fitrow, _ = fit_ols(sample, "positive_log1p_Aa", predictors, control, "D1_positive_erosion", "S6_two_part_erosion")
                all_paths.extend(rows); all_fits.append(fitrow)
    return pd.DataFrame(all_paths), pd.DataFrame(all_fits), delta, effects


def cb_covariance_fit(d: pd.DataFrame, structure: str, dataset: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Observed-variable CB path SEM with independent structural disturbances."""
    variables = ["z_E", "physical", "chemical", "biological_activity"]
    if structure != "S1_parallel_dimensions":
        variables.append("z_SHI")
    d = d[variables + ["spatial_block_5km"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 20:
        return [], {"dataset": dataset, "method": "CB_SEM_custom_observed_path", "structure": structure, "n": len(d), "status": "insufficient_rows"}
    n = len(d)
    names = variables
    # `fit_ols` returns standardized path coefficients and residuals. Build the
    # observed covariance matrix on that same sample-standardized scale.
    means = d[variables].mean(axis=0)
    scales = d[variables].std(axis=0, ddof=1)
    if (scales <= 0).any() or not np.isfinite(scales.to_numpy()).all():
        return [], {"dataset": dataset, "method": "CB_SEM_custom_observed_path", "structure": structure,
                    "n": n, "status": "zero_or_invalid_variance"}
    standardized = (d[variables] - means) / scales
    S = np.cov(standardized.to_numpy(dtype=float), rowvar=False, ddof=1)
    p = len(variables)
    B = np.zeros((p, p), dtype=float)
    psi = np.zeros((p, p), dtype=float)
    paths: list[dict[str, Any]] = []
    groups = d.spatial_block_5km.astype(str).to_numpy()
    path_specs = []
    endog = ["physical", "chemical", "biological_activity"]
    for y in endog:
        path_specs.append((y, ["z_E"]))
    if "z_SHI" in variables:
        preds = ["physical", "chemical", "biological_activity"]
        if structure == "S2_direct_plus_indirect_SHI":
            preds = ["z_E", *preds]
        path_specs.append(("z_SHI", preds))
    error_vars = {"z_E": float(S[variables.index("z_E"), variables.index("z_E")])}
    for y, preds in path_specs:
        fit_rows, _, fit = fit_ols(d, y, preds, "M0", dataset, structure, method="CB_SEM_custom_observed_path")
        if fit is None:
            return [], {"dataset": dataset, "method": "CB_SEM_custom_observed_path", "structure": structure, "n": n, "status": "equation_failed"}
        target = variables.index(y)
        for pred in preds:
            source = variables.index(pred)
            B[target, source] = float(fit.params.iloc[fit.model.exog_names.index(pred)])
        error_vars[y] = float(np.var(np.asarray(fit.resid, dtype=float), ddof=1))
        paths.extend(fit_rows)
    for var, varval in error_vars.items():
        psi[variables.index(var), variables.index(var)] = varval
    try:
        A = np.linalg.inv(np.eye(p) - B)
        implied = A @ psi @ A.T
        sign_s, logdet_s = np.linalg.slogdet(S)
        sign_m, logdet_m = np.linalg.slogdet(implied)
        if sign_s <= 0 or sign_m <= 0:
            raise np.linalg.LinAlgError("non-positive covariance determinant")
        F = float(logdet_m + np.trace(S @ np.linalg.inv(implied)) - logdet_s - p)
        F = max(F, 0.0)
        if structure == "S1_parallel_dimensions":
            k = 1 + 3 + 3  # variance of E, E -> 3 dimensions, 3 residual variances
        elif structure == "S3_indirect_only_SHI":
            k = 1 + 3 + 3 + 1 + 3
        else:
            k = 1 + 3 + 3 + 1 + 4
        moments = p * (p + 1) // 2
        df = max(moments - k, 1)
        chi2 = (n - 1) * F
        # Independence baseline for CFI/TLI.
        S0 = np.diag(np.diag(S))
        sign0, ld0 = np.linalg.slogdet(S0)
        F0 = max(float(ld0 + np.trace(S @ np.linalg.inv(S0)) - logdet_s - p), 0.0)
        df0 = moments - p
        chi0 = (n - 1) * F0
        cfi = 1.0 - max(chi2 - df, 0) / max(chi0 - df0, chi2 - df, 1e-12)
        tli = ((chi0 / max(df0, 1)) - (chi2 / df)) / max((chi0 / max(df0, 1)) - 1, 1e-12)
        rmsea = math.sqrt(max((chi2 - df) / (df * max(n - 1, 1)), 0.0))
        sd = np.sqrt(np.diag(S))
        R = (S - implied) / np.outer(sd, sd)
        upper = R[np.triu_indices(p, 0)]
        srmr = float(np.sqrt(np.mean(upper ** 2))) if len(upper) else 0.0
        fitrow = {"dataset": dataset, "method": "CB_SEM_custom_observed_path", "structure": structure, "control": "M0",
                  "n": n, "spatial_blocks_5km": int(pd.Series(groups).nunique()), "chi_square_naive": chi2, "df": df,
                  "cfi_naive": float(cfi), "tli_naive": float(tli), "rmsea_naive": float(rmsea), "srmr": srmr,
                  "aic_relative": float(n * F + 2 * k), "bic_relative": float(n * F + np.log(n) * k),
                  "status": "ok", "fit_note": "custom path approximation on a common standardized scale; SRMR uses upper-triangle standardized covariance residuals including diagonal; naive iid descriptive indices; not a mature CB-SEM package or spatially robust fit",
                  "outcome_guided_selection": True}
        return paths, fitrow
    except Exception as exc:
        return paths, {"dataset": dataset, "method": "CB_SEM_custom_observed_path", "structure": structure, "control": "M0",
                       "n": n, "status": f"failed:{type(exc).__name__}:{exc}", "outcome_guided_selection": True}


def pls_path_model(d: pd.DataFrame, dataset: str, max_iter: int = 100,
                   control: str = "M0", paired: bool = False,
                   structure: str = "S2_direct_plus_indirect_SHI"
                   ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Mode-B PLS comparisons for S1-S3 with M0-M3 controls as single-indicator constructs."""
    structures = {"S1_parallel_dimensions", "S2_direct_plus_indirect_SHI", "S3_indirect_only_SHI"}
    if structure not in structures:
        raise KeyError(structure)
    work = d.copy()
    common_controls = []
    for candidate in ["M0", "M1", "M2", "M3"]:
        common_controls.extend(control_columns(work, candidate, paired))
    common_required = ["z_E", "z_SW", "z_BD", "z_OM", "z_AN", "z_MBC", "z_RESP", "z_SHI",
                       "spatial_block_5km", *dict.fromkeys(common_controls)]
    work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=common_required).copy()
    control_cols = control_columns(work, control, paired)
    base_blocks = {"Erosion": ["z_E"], "Physical": ["z_SW", "z_BD"],
                   "Chemical": ["z_OM", "z_AN"], "Biological": ["z_MBC", "z_RESP"]}
    if structure != "S1_parallel_dimensions":
        base_blocks["SHI"] = ["z_SHI"]
    control_nodes = [f"Control_{i+1}_{col}" for i, col in enumerate(control_cols)]
    block_cols = {**base_blocks, **{node: [col] for node, col in zip(control_nodes, control_cols)}}
    required = sorted({c for cols in block_cols.values() for c in cols} | {"spatial_block_5km"})
    data = work[required].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(data) < 50:
        return [], [], [], [], {"dataset": dataset, "control": control, "status": "insufficient_rows", "n": len(data)}
    matrices = {}
    for construct, cols in block_cols.items():
        vals = data[cols].to_numpy(dtype=float).copy()
        # Lower bulk density is direction-aligned with improved physical condition.
        if construct == "Physical":
            vals[:, 1] *= -1
        # Single-indicator environment, soil and spatial controls are standardized too.
        vals = (vals - vals.mean(axis=0)) / vals.std(axis=0, ddof=1)
        matrices[construct] = vals
    dimension_targets = ["Physical", "Chemical", "Biological"]
    predictors = {target: ["Erosion", *control_nodes] for target in dimension_targets}
    if structure == "S2_direct_plus_indirect_SHI":
        predictors["SHI"] = ["Erosion", *dimension_targets, *control_nodes]
    elif structure == "S3_indirect_only_SHI":
        predictors["SHI"] = [*dimension_targets, *control_nodes]
    graph = {node: set() for node in block_cols}
    for target, preds in predictors.items():
        for source in preds:
            graph[source].add(target)
            graph[target].add(source)
    weights = {k: np.ones(v.shape[1], dtype=float) / np.sqrt(v.shape[1]) for k, v in matrices.items()}
    for key, X in matrices.items():
        if X.shape[1] == 1:
            weights[key] = np.ones(1, dtype=float)
    converged, diff = False, np.inf
    scores = {}
    for iteration in range(1, max_iter + 1):
        scores = {}
        for key, X in matrices.items():
            v = X @ weights[key]
            sd = v.std(ddof=1)
            scores[key] = (v - v.mean()) / sd if sd > 0 else np.zeros_like(v)
        new_weights = {k: v.copy() for k, v in weights.items()}
        for key, X in matrices.items():
            if X.shape[1] == 1:
                continue
            neighbor_score = np.zeros(len(data), dtype=float)
            for neighbor in sorted(graph[key]):
                corr = float(np.corrcoef(scores[key], scores[neighbor])[0, 1])
                neighbor_score += (1.0 if corr >= 0 else -1.0) * scores[neighbor]
            coef = np.linalg.pinv(X.T @ X) @ X.T @ neighbor_score
            candidate = X @ coef
            if np.corrcoef(candidate, X[:, 0])[0, 1] < 0:
                coef *= -1
            scale = np.std(X @ coef, ddof=1)
            new_weights[key] = coef / scale if scale > 1e-12 else weights[key]
        diff = max(float(np.max(np.abs(new_weights[k] - weights[k]))) for k in weights)
        weights = new_weights
        if diff < 1e-6:
            converged = True
            break
    scores = {}
    for key, X in matrices.items():
        v = X @ weights[key]
        scores[key] = (v - v.mean()) / v.std(ddof=1)
    path_rows, weight_rows, fitrows, delta_rows = [], [], [], []
    for target, preds in predictors.items():
        X = sm.add_constant(np.column_stack([scores[p] for p in preds]), has_constant="add")
        y = scores[target]
        fit = sm.OLS(y, X).fit()
        reduced_preds = [p for p in preds if p != "Erosion"]
        X0 = sm.add_constant(np.column_stack([scores[p] for p in reduced_preds]), has_constant="add") if reduced_preds else np.ones((len(data), 1))
        base_fit = sm.OLS(y, X0).fit()
        params = np.asarray(fit.params, dtype=float)
        for i, source in enumerate(preds):
            source_label = "Erosion" if source == "Erosion" else source
            path_rows.append({"dataset": dataset, "method": "PLS_PM_ModeB_formative", "structure": structure,
                              "control": control, "from": source_label, "to": target, "path_type": "PLS inner path",
                              "beta_standardized": float(params[i+1]), "se": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                              "p_value_nominal": np.nan, "r_squared": float(fit.rsquared), "adjusted_r_squared": float(fit.rsquared_adj),
                              "n": int(len(data)), "spatial_blocks_5km": int(data.spatial_block_5km.nunique()),
                              "se_type": "no bootstrap; path coefficient is descriptive", "outcome_guided_selection": True})
        fitrows.append({"dataset": dataset, "method": "PLS_PM_ModeB_formative", "structure": structure,
                        "control": control, "endogenous": target, "n": int(len(data)),
                        "spatial_blocks_5km": int(data.spatial_block_5km.nunique()), "r_squared": float(fit.rsquared),
                        "adjusted_r_squared": float(fit.rsquared_adj), "aic": np.nan, "bic": np.nan,
                        "design_condition_number": float(np.linalg.cond(X)), "status": "ok" if converged else "not_converged",
                        "converged": converged, "iterations": iteration,
                        "outer_weight_max_change": diff, "outcome_guided_selection": True})
        delta_rows.append({"dataset": dataset, "method": "PLS_PM_ModeB_formative", "structure": structure,
                           "control": control, "outcome": target, "n": int(len(data)),
                           "r2_full": float(fit.rsquared), "r2_without_erosion": float(base_fit.rsquared),
                           "delta_r2_erosion_conditional": float(fit.rsquared - base_fit.rsquared),
                           "interpretation": "conditional PLS inner-equation R2 increment from erosion; no bootstrap and not causal",
                           "outcome_guided_selection": True, "branch": "PLS inner equation"})
    for construct, cols in block_cols.items():
        X = matrices[construct]
        score = scores[construct]
        for j, col in enumerate(cols):
            weight_rows.append({"dataset": dataset, "structure": structure, "control": control,
                                "construct": construct, "indicator": col,
                                "outer_weight_mode_b": float(weights[construct][j]),
                                "loading_correlation": float(np.corrcoef(X[:, j], score)[0, 1]),
                                "n": int(len(data)), "converged": converged, "iterations": iteration,
                                "outcome_guided_selection": True})
    meta = {"dataset": dataset, "structure": structure, "control": control,
            "status": "ok" if converged else "not_converged", "n": int(len(data)),
            "spatial_blocks_5km": int(data.spatial_block_5km.nunique()),
            "converged": converged, "iterations": iteration, "max_weight_change": diff,
            "indicator_blocks": block_cols, "controls": control_cols,
            "measurement_spec": f"formative Mode B {structure}; physical=[SW,-BD], chemical=[OM,AN], biological=[MBC,RESP]; M1-M3 controls enter as fixed single-indicator exogenous constructs; CEC and pH remain separate",
            "outcome_guided_selection": True}
    return path_rows, weight_rows, fitrows, delta_rows, meta


def run_models(datasets: dict[str, pd.DataFrame], run_dir: Path, qa: dict[str, Any]) -> dict[str, Any]:
    refs = set_scale_reference(datasets["D1_common_env_case"])
    pca_params = make_pca_params(datasets["D1_common_env_case"])
    json_write(run_dir / "provenance/scale_reference_and_environment_pca.json", {"scale_reference": refs, "environment_pca": pca_params})
    paths, fits, delta, effects, pls_weights, method_status = [], [], [], [], [], []
    # All seven analysis populations are retained; paired D4/D6 are rejoined to corrected CEC above.
    # The separate D1 all-case M0 provides a full-mask baseline; common-case M0-M3 are directly comparable.
    for dataset, source in datasets.items():
        if dataset == "D1_full_pixels":
            continue
        modelled = add_model_variables(source, dataset, refs, pca_params)
        # Add scale-sensitive SHI and CEC as named paths; primary chemical composite excludes corrected CEC.
        if dataset == "D7_soil_TWI_groups":
            for (soil, twi), group in modelled.groupby(["SoilType", "twi_group"], observed=True):
                key = f"D7_soil{int(soil)}_{twi}"
                paths_i, fits_i, delta_i, effects_i = _run_population_models(group.copy(), key, False)
                paths.extend(paths_i); fits.extend(fits_i); delta.extend(delta_i); effects.extend(effects_i)
            continue
        paired = dataset in {"D4_M4_pair_differences", "D6_zero_positive_pairs"}
        if dataset == "D4_M4_pair_differences":
            modelled = modelled.loc[modelled.both_positive | modelled.zero_positive].copy()
        paths_i, fits_i, delta_i, effects_i = _run_population_models(modelled, dataset, paired)
        paths.extend(paths_i); fits.extend(fits_i); delta.extend(delta_i); effects.extend(effects_i)
        if dataset in {"D1_common_env_case", "D2_240m_blocks", "D3_M4_unique_pixels", "D4_M4_pair_differences"}:
            cb_base = modelled.replace([np.inf, -np.inf], np.nan).dropna(subset=["z_E", "physical", "chemical", "biological_activity", "z_SHI", "spatial_block_5km"])
            for structure in ["S1_parallel_dimensions", "S2_direct_plus_indirect_SHI", "S3_indirect_only_SHI"]:
                cbp, cbfit = cb_covariance_fit(cb_base, structure, dataset)
                paths.extend(cbp); fits.append(cbfit)
            for pls_structure in ["S1_parallel_dimensions", "S2_direct_plus_indirect_SHI", "S3_indirect_only_SHI"]:
                for pls_control in ["M0", "M1", "M2", "M3"]:
                    plp, wts, plfits, pldelta, plmeta = pls_path_model(
                        modelled, dataset, control=pls_control, paired=paired, structure=pls_structure
                    )
                    paths.extend(plp); pls_weights.extend(wts); fits.extend(plfits); increments.extend(pldelta)
                    method_status.append({"method": "PLS_PM_ModeB_formative", **plmeta})
    # S6: separate the occurrence of any erosion from positive-erosion intensity,
    # and estimate each exposure component's association with the same health
    # dimensions. Both branches are compared with an identical control-only fit.
    s6 = add_model_variables(datasets["D1_common_env_case"], "D1_common_env_case", refs, pca_params)
    s6 = s6.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["Aa_20yr", "spatial_block_5km", *[f"z_{c}" for c in ENV_M1]]).copy()
    s6["erosion_occurrence"] = s6.Aa_20yr.gt(0).astype(float)
    positive = s6.loc[s6.Aa_20yr.gt(0)].copy()
    positive["positive_intensity"] = np.log1p(positive.Aa_20yr)
    for control in ["M0", "M1", "M2", "M3"]:
        for outcome in ["physical", "chemical", "biological_activity", "z_SHI"]:
            for exposure_name, sample, exposure, branch in [
                ("erosion_occurrence", s6, "erosion_occurrence", "S6_occurrence"),
                ("positive_intensity", positive, "positive_intensity", "S6_positive_intensity"),
            ]:
                # Build the soil dummy columns on the actual branch sample.
                # `positive` is a subset copied before dummies are added to `s6`.
                ccols = control_columns(sample, control, False)
                full_rows, full_fit, _ = fit_ols(sample, outcome, [*ccols, exposure], control,
                                                  "D1_common_env_case", "S6_two_part_erosion")
                _, base_fit, _ = fit_ols(sample, outcome, ccols, control,
                                          "D1_common_env_case", f"{branch}_controls_only")
                paths.extend(full_rows)
                fits.append(full_fit)
                delta.append({"dataset": "D1_common_env_case", "method": "piecewise_sem",
                              "structure": "S6_two_part_erosion", "branch": branch, "control": control,
                              "outcome": outcome, "n": full_fit.get("n"),
                              "r2_full": full_fit.get("r_squared"),
                              "r2_without_erosion": base_fit.get("r_squared"),
                              "delta_r2_erosion_conditional": full_fit.get("r_squared", np.nan)-base_fit.get("r_squared", np.nan),
                              "interpretation": "conditional prediction increment for occurrence or positive intensity; not causal",
                              "outcome_guided_selection": True})
    # Add a full-mask M0 baseline (not compared with M1-M3 due different eligible samples).
    full = add_model_variables(datasets["D1_full_pixels"], "D1_full_pixels", refs, pca_params)
    full = full.replace([np.inf, -np.inf], np.nan).dropna(subset=["z_E", "physical", "chemical", "biological_activity", "z_SHI", "spatial_block_5km"])
    fp, ff, fd, fe = _run_population_models(full, "D1_all_valid_M0_only", False, controls=["M0"])
    paths.extend(fp); fits.extend(ff); delta.extend(fd); effects.extend(fe)
    if pls_weights:
        write_table(pd.DataFrame(pls_weights), run_dir / "sem/pls_sem/PLS_SEM_OUTER_WEIGHTS.csv")
    paths_df = pd.DataFrame(paths)
    write_table(paths_df, run_dir / "results/SEM_PATH_COEFFICIENTS.csv")
    write_table(paths_df.loc[paths_df.structure.eq("S6_two_part_erosion")],
                run_dir / "results/SEM_S6_TWO_PART_COMPONENTS.csv")
    write_table(pd.DataFrame(fits), run_dir / "results/SEM_MODEL_FIT_COMPARISON.csv")
    write_table(pd.DataFrame(delta), run_dir / "results/SEM_EROSION_INCREMENTAL_CONTRIBUTION.csv")
    write_table(pd.DataFrame(effects), run_dir / "results/SEM_DIRECT_INDIRECT_EFFECTS.csv")
    json_write(run_dir / "provenance/method_status.json", {"custom_pls_pm": method_status,
              "package_status": {"semopy": False, "lavaan": False, "piecewiseSEM": False, "plspm": False},
              "reason": "No mature SEM package was installed in the current environment; no dependency was added.",
              "spatial_inference": "piecewise path equations use 5 km block-cluster robust covariance when at least 8 blocks are represented; CB fit indices are naive iid descriptive indices; PLS paths have no bootstrap intervals.",
              "outcome_guided_selection": True})
    return {"paths": paths_df, "fits": pd.DataFrame(fits), "delta": pd.DataFrame(delta), "effects": pd.DataFrame(effects), "method_status": method_status}


def _run_population_models(d: pd.DataFrame, dataset: str, paired: bool, controls: list[str] | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    all_paths, all_fits, delta, effects = [], [], [], []
    control_models = controls or ["M0", "M1", "M2", "M3"]
    required = ["z_E", "physical", "chemical", "biological_activity", "z_SHI", "z_CEC", "z_pH", "spatial_block_5km", "x", "y"]
    if len(control_models) > 1:
        required += [f"z_{c}" for c in ENV_M1]
        required += ["terrain_PC", "vegetation_thermal_PC"]
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=required).copy()
    for control in control_models:
        ccols = control_columns(d, control, paired)
        base = ["z_E", *ccols]
        eq_fits = {}
        for y in ["physical", "chemical", "biological_activity"]:
            rows, fitrow, fit = fit_ols(d, y, base, control, dataset, "S1_parallel_dimensions")
            all_paths.extend(rows); all_fits.append(fitrow); eq_fits[y] = fit
            if fit is not None:
                _, no_fitrow, _ = fit_ols(d, y, ccols, control, dataset, "S1_without_erosion")
                delta.append({"dataset": dataset, "method": "piecewise_sem", "structure": "S1_parallel_dimensions", "control": control,
                              "outcome": y, "n": fitrow["n"], "r2_full": fitrow["r_squared"],
                              "r2_without_erosion": no_fitrow.get("r_squared", np.nan),
                              "delta_r2_erosion_conditional": fitrow["r_squared"] - no_fitrow.get("r_squared", np.nan),
                              "interpretation": "conditional predictive increment; not causal contribution", "outcome_guided_selection": True})
        for structure, preds in [("S2_direct_plus_indirect_SHI", ["z_E", "physical", "chemical", "biological_activity", *ccols]),
                                 ("S3_indirect_only_SHI", ["physical", "chemical", "biological_activity", *ccols])]:
            rows, fitrow, fit = fit_ols(d, "z_SHI", preds, control, dataset, structure)
            all_paths.extend(rows); all_fits.append(fitrow); eq_fits[structure] = fit
            if fit is not None and structure.startswith("S2"):
                _, no_fitrow, _ = fit_ols(d, "z_SHI", ["physical", "chemical", "biological_activity", *ccols], control, dataset, "S2_without_direct_erosion")
                delta.append({"dataset": dataset, "method": "piecewise_sem", "structure": structure, "control": control, "outcome": "z_SHI",
                              "n": fitrow["n"], "r2_full": fitrow["r_squared"], "r2_without_erosion": no_fitrow.get("r_squared", np.nan),
                              "delta_r2_erosion_conditional": fitrow["r_squared"] - no_fitrow.get("r_squared", np.nan),
                              "interpretation": "direct erosion increment conditional on dimension scores; indirect paths remain", "outcome_guided_selection": True})
                _, total_fit, _ = fit_ols(d, "z_SHI", base, control, dataset, "SHI_total_association")
                _, total_no, _ = fit_ols(d, "z_SHI", ccols, control, dataset, "SHI_without_erosion_total")
                delta.append({"dataset": dataset, "method": "piecewise_sem", "structure": "SHI_total_association", "control": control, "outcome": "z_SHI",
                              "n": total_fit.get("n"), "r2_full": total_fit.get("r_squared"), "r2_without_erosion": total_no.get("r_squared"),
                              "delta_r2_erosion_conditional": total_fit.get("r_squared", np.nan)-total_no.get("r_squared", np.nan),
                              "interpretation": "total controlled association increment, not a mediated causal effect", "outcome_guided_selection": True})
        # S4: preserve an explicit, limited soil-process chain while allowing
        # erosion to associate with each dimension and keeping a direct SHI path.
        s4_specs = [
            ("physical", ["z_E", *ccols]),
            ("chemical", ["z_E", "physical", *ccols]),
            ("biological_activity", ["z_E", "physical", "chemical", *ccols]),
            ("z_SHI", ["z_E", "physical", "chemical", "biological_activity", *ccols]),
        ]
        s4_fits: dict[str, Any] = {}
        for y, predictors in s4_specs:
            rows, fitrow, fit = fit_ols(d, y, predictors, control, dataset, "S4_chained_health_dimensions")
            all_paths.extend(rows); all_fits.append(fitrow); s4_fits[y] = fit
        if all(s4_fits.get(name) is not None for name in ["physical", "chemical", "biological_activity", "z_SHI"]):
            b = {y: dict(zip(fit.model.exog_names, np.asarray(fit.params, dtype=float))) for y, fit in s4_fits.items()}
            e_p = float(b["physical"].get("z_E", 0.0))
            e_c = float(b["chemical"].get("z_E", 0.0))
            p_c = float(b["chemical"].get("physical", 0.0))
            e_b = float(b["biological_activity"].get("z_E", 0.0))
            p_b = float(b["biological_activity"].get("physical", 0.0))
            c_b = float(b["biological_activity"].get("chemical", 0.0))
            e_h = float(b["z_SHI"].get("z_E", 0.0))
            p_h = float(b["z_SHI"].get("physical", 0.0))
            c_h = float(b["z_SHI"].get("chemical", 0.0))
            b_h = float(b["z_SHI"].get("biological_activity", 0.0))
            indirect_paths = {
                "E→Physical→SHI": e_p * p_h,
                "E→Chemical→SHI": e_c * c_h,
                "E→Biological→SHI": e_b * b_h,
                "E→Physical→Chemical→SHI": e_p * p_c * c_h,
                "E→Physical→Biological→SHI": e_p * p_b * b_h,
                "E→Chemical→Biological→SHI": e_c * c_b * b_h,
                "E→Physical→Chemical→Biological→SHI": e_p * p_c * c_b * b_h,
            }
            for path, value in indirect_paths.items():
                effects.append({"dataset": dataset, "method": "piecewise_sem", "structure": "S4_chained_health_dimensions",
                                "control": control, "path": path, "direct_effect": np.nan, "indirect_effect": value,
                                "total_effect_component": value, "ci_low": np.nan, "ci_high": np.nan,
                                "interval_status": "not computed", "interpretation": "product of standardized conditional associations; not causal mediation",
                                "outcome_guided_selection": True})
            total_indirect = float(sum(indirect_paths.values()))
            effects.append({"dataset": dataset, "method": "piecewise_sem", "structure": "S4_chained_health_dimensions",
                            "control": control, "path": "E→SHI chained total association", "direct_effect": e_h,
                            "indirect_effect": total_indirect, "total_effect_component": e_h + total_indirect,
                            "ci_low": np.nan, "ci_high": np.nan, "interval_status": "not computed",
                            "interpretation": "chain-specific standardized association decomposition; not causal mediation",
                            "outcome_guided_selection": True})
        for y in INDICATOR_RESPONSES:
            yname = f"z_{y}"
            rows, fitrow, _ = fit_ols(d, yname, base, control, dataset, "S5_indicator_responses")
            all_paths.extend(rows); all_fits.append(fitrow)
        # Indirect path products are conditional association decompositions; no causal mediation claim.
        s2 = eq_fits.get("S2_direct_plus_indirect_SHI")
        if s2 is not None:
            coef = dict(zip(s2.model.exog_names, s2.params))
            isum = 0.0
            for dim in ["physical", "chemical", "biological_activity"]:
                upstream = eq_fits.get(dim)
                if upstream is None: continue
                b_e = float(upstream.params.iloc[upstream.model.exog_names.index("z_E")])
                b_h = float(coef.get(dim, 0.0)); v = b_e * b_h; isum += v
                effects.append({"dataset": dataset, "method": "piecewise_sem", "control": control,
                                "path": f"E→{dim}→SHI", "direct_effect": np.nan, "indirect_effect": v,
                                "total_effect_component": v, "ci_low": np.nan, "ci_high": np.nan,
                                "interval_status": "not computed", "interpretation": "product of standardized conditional associations; not causal mediation",
                                "outcome_guided_selection": True})
            direct = float(coef.get("z_E", 0.0))
            effects.append({"dataset": dataset, "method": "piecewise_sem", "control": control,
                            "path": "E→SHI total (direct + dimension-mediated association)", "direct_effect": direct,
                            "indirect_effect": isum, "total_effect_component": direct+isum, "ci_low": np.nan, "ci_high": np.nan,
                            "interval_status": "not computed", "interpretation": "observed-path decomposition; SHI recipe unresolved",
                            "outcome_guided_selection": True})
    return all_paths, all_fits, delta, effects


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=RUN_ID_DEFAULT)
    args = parser.parse_args()
    run_dir = ROOT / "outputs/exploratory_oracle/phase_j" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    for sub in ["data", "sem/pls_sem", "sem/cb_sem", "sem/piecewise_sem", "results", "provenance"]:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    pixels, input_qa, manifest = audit_inputs(run_dir)
    datasets, population_qa = build_datasets(pixels, run_dir, manifest)
    model_summary = run_models(datasets, run_dir, population_qa)
    metadata = {"phase": "J", "mode": "EXPLORATORY_ORACLE", "outcome_guided_selection": True,
                "run_id": args.run_id, "created_utc": datetime.now(timezone.utc).isoformat(),
                "git_commit": None, "git_worktree": None,
                "inputs": {"corrected_database": str(CORRECTED_RUN.relative_to(ROOT)),
                           "original_cec_for_formula_audit": str((BASE_RUN / "databases/health.parquet").relative_to(ROOT)),
                           "M4_memberships": str((PHASE_H / "pixels/matched_pixel_unique.parquet").relative_to(ROOT)),
                           "M4_pairs": str((CORRECTED_RUN / "pairs/candidate_pairs/stage2_Aa_20yr_01.parquet").relative_to(ROOT)),
                           "Phase_I8_blocks": "recreated exactly from 8x8 env-complete pixel support"},
                "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                             "scipy": scipy.__version__, "statsmodels": sm.__version__, "scikit_learn": __import__("sklearn").__version__,
                             "pyarrow": pq.__version__ if hasattr(pq, "__version__") else "unknown"},
                "limitations": ["Aa physical unit and exact temporal construction are not in source metadata",
                                "health surfaces are RF predictions, not independent measurements per pixel",
                                "SHI candidate formula/source is provisional; not recomputed",
                                "CEC correction formula includes Aa algebraically; CEC associations are not independent erosion evidence",
                                "spatial block clustering does not substitute for independent field validation",
                                "no PCNM was run"],
                "input_audit": input_qa, "population_audit": population_qa,
                "path_rows": int(len(model_summary["paths"])), "fit_rows": int(len(model_summary["fits"])),
                "delta_r2_rows": int(len(model_summary["delta"])), "indirect_effect_rows": int(len(model_summary["effects"]))}
    json_write(run_dir / "provenance/run_manifest.json", metadata)
    (run_dir / "RUN_STATUS.md").write_text(
        f"# Phase J SEM run status\n\n- run_id: `{args.run_id}`\n- mode: `EXPLORATORY_ORACLE`\n- outcome_guided_selection: `true`\n- data and SEM tables generated.\n- CEC formula verified against stored corrected values; source SHI remains provisional.\n- PCNM was not run.\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir.relative_to(ROOT)), "input_qa": input_qa,
                      "population_qa": population_qa,
                      "path_rows": metadata["path_rows"], "fit_rows": metadata["fit_rows"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
