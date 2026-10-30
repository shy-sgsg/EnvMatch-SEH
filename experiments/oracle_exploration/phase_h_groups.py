"""Outcome-blind environmental grouping and response profiles for Phase H."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from experiments.oracle_exploration.phase_g import fit_hierarchical_group_labels
from experiments.oracle_exploration.phase_h_search import enrich_pair_spatial_coordinates
from experiments.oracle_exploration.nonlinear_search import fit_model_family
from experiments.oracle_exploration.phase_h_metrics import regression_metrics


ENVIRONMENT_FEATURES = ("Elevation", "LST")
PROFILE_VARIABLES = (
    "Elevation", "Cur", "TWI", "LST", "NDWI", "NDBSI", "EVI", "NDVI", "Precipitation"
)


def _soil_kmeans_labels(
    cells: pd.DataFrame,
    *,
    n_clusters: int,
    seed: int,
    primary_soils: Sequence[int],
) -> pd.Series:
    labels = pd.Series(pd.NA, index=cells.index, dtype="string")
    matrix = cells[list(ENVIRONMENT_FEATURES)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    soils = pd.to_numeric(cells["SoilType"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(matrix).all(axis=1) & np.isfinite(soils)
    for soil in sorted(map(int, primary_soils)):
        positions = np.flatnonzero(valid & (soils == soil))
        if not len(positions):
            continue
        scaler = StandardScaler().fit(matrix[positions])
        standardized = scaler.transform(matrix[positions])
        distinct = np.unique(standardized, axis=0).shape[0]
        effective_k = min(int(n_clusters), len(positions), distinct)
        if effective_k < 1:
            continue
        if effective_k == 1:
            assigned = np.zeros(len(positions), dtype=np.int32)
        else:
            model = MiniBatchKMeans(
                n_clusters=effective_k,
                random_state=int(seed),
                n_init=3,
                batch_size=min(2048, max(64, effective_k * 32)),
                reassignment_ratio=0.0,
            )
            assigned = model.fit_predict(standardized).astype(np.int32)
        labels.iloc[positions] = [f"soil_{soil}_cluster_{label}" for label in assigned]
    return labels


def _gradient_labels(
    cells: pd.DataFrame,
    *,
    variables: Sequence[str],
    bins: int,
    primary_soils: Sequence[int],
) -> dict[str, pd.Series]:
    if bins < 2:
        raise ValueError("gradient_bins must be at least 2")
    soil = pd.to_numeric(cells["SoilType"], errors="coerce")
    result: dict[str, pd.Series] = {}
    for variable in variables:
        if variable not in cells:
            raise ValueError(f"gradient variable is missing from aligned cells: {variable}")
        labels = pd.Series(pd.NA, index=cells.index, dtype="string")
        values = pd.to_numeric(cells[variable], errors="coerce")
        for soil_code in sorted(map(int, primary_soils)):
            mask = soil.eq(soil_code) & values.notna() & np.isfinite(values)
            if int(mask.sum()) < bins:
                continue
            levels = pd.qcut(values.loc[mask], q=bins, labels=False, duplicates="drop")
            labels.loc[mask] = levels.astype("Int64").map(
                lambda value: f"{soil_code}:q{int(value) + 1}" if pd.notna(value) else pd.NA
            )
        result[f"gradient_{variable}"] = labels
    return result


def build_environment_group_labels(
    cells: pd.DataFrame,
    options: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create Phase H groups from soil and environmental covariates only."""

    required = {"cell_id", "SoilType", *ENVIRONMENT_FEATURES}
    missing = sorted(required - set(cells.columns))
    if missing:
        raise ValueError(f"aligned cells are missing environmental grouping inputs: {missing}")
    if cells["cell_id"].isna().any() or cells["cell_id"].duplicated().any():
        raise ValueError("environment group labels require unique non-null cell_id values")
    primary_soils = tuple(map(int, options.get("primary_soils", (26, 27, 42))))
    counts = tuple(sorted(set(map(int, options.get("kmeans_counts", (4, 6, 8))))))
    seeds = {int(key): int(value) for key, value in options.get("kmeans_seeds", {}).items()}
    replicate_count = max(1, int(options.get("stability_replicates", 3)))
    label_frame = pd.DataFrame({"cell_id": cells["cell_id"].to_numpy()})
    soil = pd.to_numeric(cells["SoilType"], errors="coerce")
    label_frame["soil_type_group"] = soil.map(
        lambda value: f"soil_{int(value)}" if pd.notna(value) and int(value) in primary_soils else pd.NA
    ).astype("string")

    stability_rows: list[dict[str, Any]] = []
    for k in counts:
        primary_seed = seeds.get(k, int(options.get("seed", 20261028)) + k)
        primary_labels = _soil_kmeans_labels(
            cells, n_clusters=k, seed=primary_seed, primary_soils=primary_soils
        )
        column = f"kmeans_terrain_k{k}"
        label_frame[column] = primary_labels.to_numpy()
        valid = primary_labels.notna().to_numpy()
        scores = []
        for replicate in range(1, replicate_count):
            replicate_labels = _soil_kmeans_labels(
                cells,
                n_clusters=k,
                seed=primary_seed + replicate,
                primary_soils=primary_soils,
            )
            replicate_valid = valid & replicate_labels.notna().to_numpy()
            if replicate_valid.sum() >= 2:
                scores.append(
                    adjusted_rand_score(
                        primary_labels.to_numpy(dtype=str)[replicate_valid],
                        replicate_labels.to_numpy(dtype=str)[replicate_valid],
                    )
                )
        stability_rows.append(
            {
                "group_column": column,
                "group_method": column,
                "stability_status": "seed_replicate_adjusted_rand_index",
                "primary_seed": int(primary_seed),
                "replicate_count": int(replicate_count),
                "mean_adjusted_rand_index": float(np.mean(scores)) if scores else float("nan"),
                "sd_adjusted_rand_index": float(np.std(scores, ddof=1)) if len(scores) > 1 else float("nan"),
                "outcome_guided_selection": False,
            }
        )

    primary_mask = soil.isin(primary_soils)
    hierarchical = fit_hierarchical_group_labels(
        cells.loc[primary_mask],
        list(ENVIRONMENT_FEATURES),
        soil_column="SoilType",
        n_clusters=int(options.get("hierarchical_k", 6)),
        max_fit_rows=int(options.get("hierarchical_fit_rows_per_soil", 1500)),
        seed=int(options.get("hierarchical_seed", 20261027)),
    )
    hierarchical_full = pd.Series(pd.NA, index=cells.index, dtype="string")
    hierarchical_full.loc[hierarchical.index] = hierarchical
    label_frame["hierarchical_ward_v5"] = hierarchical_full.to_numpy()
    stability_rows.append(
        {
            "group_column": "hierarchical_ward_v5",
            "group_method": "hierarchical_ward_v5",
            "stability_status": "deterministic_given_seed_and_input",
            "primary_seed": int(options.get("hierarchical_seed", 20261027)),
            "replicate_count": 1,
            "mean_adjusted_rand_index": float("nan"),
            "sd_adjusted_rand_index": float("nan"),
            "outcome_guided_selection": False,
        }
    )
    gradients = _gradient_labels(
        cells,
        variables=tuple(map(str, options.get("gradient_variables", (
            "Elevation", "TWI", "LST", "NDWI", "NDVI", "Precipitation"
        )))),
        bins=int(options.get("gradient_bins", 5)),
        primary_soils=primary_soils,
    )
    for column, values in gradients.items():
        label_frame[column] = values.to_numpy()
        stability_rows.append(
            {
                "group_column": column,
                "group_method": column,
                "stability_status": "deterministic_quantile_definition",
                "primary_seed": pd.NA,
                "replicate_count": 1,
                "mean_adjusted_rand_index": float("nan"),
                "sd_adjusted_rand_index": float("nan"),
                "outcome_guided_selection": False,
            }
        )
    stability_rows.append(
        {
            "group_column": "soil_type_group",
            "group_method": "soil_type_group",
            "stability_status": "deterministic_category_definition",
            "primary_seed": pd.NA,
            "replicate_count": 1,
            "mean_adjusted_rand_index": float("nan"),
            "sd_adjusted_rand_index": float("nan"),
            "outcome_guided_selection": False,
        }
    )
    stability = pd.DataFrame.from_records(stability_rows)
    return label_frame, stability


def _association(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x[finite], dtype=float), np.asarray(y[finite], dtype=float)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return {"spearman_rho": float("nan"), "kendall_tau": float("nan")}
    return {
        "spearman_rho": float(spearmanr(x, y).statistic),
        "kendall_tau": float(kendalltau(x, y, variant="b", method="asymptotic").statistic),
    }


def _group_nonlinear_r_squared(frame: pd.DataFrame, *, min_pairs: int = 30) -> float:
    if len(frame) < min_pairs:
        return float("nan")
    x = pd.to_numeric(frame["delta_log1p_Aa"], errors="coerce")
    y = pd.to_numeric(frame["delta_health"], errors="coerce")
    valid = np.isfinite(x.to_numpy(dtype=float)) & np.isfinite(y.to_numpy(dtype=float))
    selected = frame.loc[valid].copy()
    if len(selected) < min_pairs:
        return float("nan")
    prediction, _ = fit_model_family(
        selected,
        selected,
        "delta_health",
        "delta_log1p_Aa",
        ["delta_log1p_Aa"],
        model_family="natural_spline",
        random_state=20261028,
    )
    return float(regression_metrics(y.loc[valid].to_numpy(dtype=float), prediction)["r_squared"])


def build_environment_group_profiles(
    cells: pd.DataFrame,
    labels: pd.DataFrame,
    candidate_pairs: pd.DataFrame,
    *,
    candidate_id: str,
    exposure: str,
    outcome: str,
    group_columns: Sequence[str],
    min_pairs_per_group: int = 30,
) -> pd.DataFrame:
    """Summarize matched-pair response and match quality within outcome-blind groups."""

    label_index = labels.set_index("cell_id")
    cell_index = cells.set_index("cell_id")
    pairs = enrich_pair_spatial_coordinates(candidate_pairs, cells, block_size_m=5000.0)
    pairs[f"delta_health"] = pd.to_numeric(pairs.get(f"delta_{outcome}"), errors="coerce")
    pairs["delta_log1p_Aa"] = pd.to_numeric(pairs.get("delta_log1p_Aa"), errors="coerce")
    records: list[dict[str, Any]] = []
    for column in group_columns:
        if column not in labels:
            continue
        high_labels = label_index[column].reindex(pairs["high_erosion_cell_id"]).to_numpy()
        low_labels = label_index[column].reindex(pairs["low_erosion_cell_id"]).to_numpy()
        pairs["__group_high"] = high_labels
        pairs["__group_low"] = low_labels
        group_cells = labels[["cell_id", column]].dropna(subset=[column]).merge(
            cells[["cell_id", *[name for name in PROFILE_VARIABLES if name in cells]]],
            on="cell_id",
            how="left",
            validate="one_to_one",
        )
        for group_label, cell_group in group_cells.groupby(column, sort=True, dropna=True):
            matched = pairs.loc[
                pairs["same_spatial_block"]
                & pairs["__group_high"].eq(group_label)
                & pairs["__group_low"].eq(group_label)
            ].copy()
            if len(matched) < int(min_pairs_per_group):
                continue
            x = pd.to_numeric(matched["delta_log1p_Aa"], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(matched["delta_health"], errors="coerce").to_numpy(dtype=float)
            finite = np.isfinite(x) & np.isfinite(y)
            matched = matched.loc[finite].copy()
            x, y = x[finite], y[finite]
            if len(matched) < int(min_pairs_per_group):
                continue
            assoc = _association(x, y)
            group_mean = float(np.mean(y))
            record: dict[str, Any] = {
                "candidate_id": candidate_id,
                "exposure": exposure,
                "outcome": outcome,
                "estimand": "matched_pair_high_minus_low",
                "group_column": column,
                "group_method": column,
                "group_label": str(group_label),
                "n_cells": int(cell_group["cell_id"].nunique()),
                "n_pairs": int(len(matched)),
                "spatial_blocks": int(matched["spatial_block_id"].nunique()),
                "mean_delta_health": group_mean,
                "median_delta_health": float(np.median(y)),
                "negative_delta_fraction": float(np.mean(y < 0)),
                "positive_delta_fraction": float(np.mean(y > 0)),
                "median_delta_log1p_Aa": float(np.median(x)),
                "median_environmental_distance": float(pd.to_numeric(matched["environmental_distance"], errors="coerce").median()),
                "median_geographic_distance_m": float(pd.to_numeric(matched["geographic_distance"], errors="coerce").median()),
                "same_group_pair_fraction": float(len(matched) / max(len(pairs), 1)),
                "within_group_nonlinear_r_squared": _group_nonlinear_r_squared(
                    matched.assign(delta_health=y), min_pairs=min_pairs_per_group
                ),
                "outcome_guided_selection": True,
                **assoc,
            }
            for variable in PROFILE_VARIABLES:
                if variable in cell_group:
                    values = pd.to_numeric(cell_group[variable], errors="coerce")
                    record[f"mean_{variable}"] = float(values.mean())
                    record[f"median_{variable}"] = float(values.median())
            records.append(record)
    return pd.DataFrame.from_records(records)


def build_group_method_comparison(
    profiles: pd.DataFrame,
    stability: pd.DataFrame,
) -> pd.DataFrame:
    """Compare environmental grouping methods on match, response, and stability metrics."""

    if profiles.empty:
        return pd.DataFrame()
    methods = profiles.groupby("group_method", sort=True, dropna=False).agg(
        outcome_units=("outcome", "nunique"),
        exposure_units=("exposure", "nunique"),
        groups_profiled=("group_label", "nunique"),
        median_pairs_per_group=("n_pairs", "median"),
        median_spearman_rho=("spearman_rho", "median"),
        negative_direction_fraction=("spearman_rho", lambda values: float((values < 0).mean())),
        median_within_group_nonlinear_r_squared=("within_group_nonlinear_r_squared", "median"),
        median_environmental_distance=("median_environmental_distance", "median"),
        median_geographic_distance_m=("median_geographic_distance_m", "median"),
        median_same_group_pair_fraction=("same_group_pair_fraction", "median"),
    ).reset_index()
    return methods.merge(
        stability[["group_method", "stability_status", "mean_adjusted_rand_index", "replicate_count"]],
        on="group_method",
        how="left",
        validate="one_to_one",
    )


def run_phase_h_environment_group_profiles(
    run_dir: Path,
    cells: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, pd.DataFrame]:
    """Build environment-only cell labels and outcome-guided matched group profiles."""

    root = Path(run_dir)
    table_dir = root / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    labels_path = table_dir / "phase_h_environment_group_cell_labels.parquet"
    stability_path = table_dir / "phase_h_environment_group_stability.csv"
    profiles_path = table_dir / "phase_h_environment_group_profiles.csv"
    comparison_path = table_dir / "phase_h_environment_group_method_comparison.csv"
    checkpoint_path = root / "checkpoints" / "h5_environment_group_profiles_complete.json"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "labels": labels_path,
        "stability": stability_path,
        "profiles": profiles_path,
        "method_comparison": comparison_path,
    }
    if checkpoint_path.is_file() and all(path.is_file() for path in output_paths.values()):
        return {
            "labels": pd.read_parquet(labels_path),
            "stability": pd.read_csv(stability_path),
            "profiles": pd.read_csv(profiles_path),
            "method_comparison": pd.read_csv(comparison_path),
        }

    group_options = dict(config.get("environment_groups", {}))
    if not group_options.get("kmeans_seeds"):
        definitions_path = (
            Path(config.get("phase_g_root", "outputs/exploratory_oracle/phase_g_20261027_completion"))
            / "tables"
            / "environment_group_definitions.json"
        )
        if definitions_path.is_file():
            definitions = json.loads(definitions_path.read_text(encoding="utf-8"))
            group_options["kmeans_seeds"] = {
                int(name.rsplit("k", 1)[1]): int(item["seed"])
                for name, item in definitions.get("additional_kmeans_groups", {}).items()
                if name.startswith("kmeans_terrain_k")
            }
    labels, stability = build_environment_group_labels(cells, group_options)
    labels.to_parquet(labels_path, index=False)
    stability.to_csv(stability_path, index=False, encoding="utf-8-sig")

    from experiments.oracle_exploration.phase_h_validation import _selected_apparent_trials

    selected = _selected_apparent_trials(root)
    selected = selected.loc[selected["estimand"].eq("matched_pair")].copy()
    candidate_dir = root / "pairs" / "candidate_pairs"
    profile_frames: list[pd.DataFrame] = []
    for candidate_id, candidate_rows in selected.groupby("candidate_id", sort=True):
        pair_path = candidate_dir / f"{candidate_id}.parquet"
        pairs = pd.read_parquet(pair_path)
        for _, winner in candidate_rows.iterrows():
            profiles = build_environment_group_profiles(
                cells,
                labels,
                pairs,
                candidate_id=str(candidate_id),
                exposure=str(winner["exposure"]),
                outcome=str(winner["outcome"]),
                group_columns=list(group_options.get("group_columns", [
                    "soil_type_group", "kmeans_terrain_k4", "kmeans_terrain_k6", "kmeans_terrain_k8",
                    "hierarchical_ward_v5", "gradient_Elevation", "gradient_TWI", "gradient_LST",
                    "gradient_NDWI", "gradient_NDVI", "gradient_Precipitation",
                ])),
                min_pairs_per_group=int(group_options.get("min_pairs_per_group", 30)),
            )
            profile_frames.append(profiles)
        del pairs
        print(f"Phase H environment profiles: candidate {candidate_id}", flush=True)
    profiles = pd.concat(profile_frames, ignore_index=True) if profile_frames else pd.DataFrame()
    comparison = build_group_method_comparison(profiles, stability)
    profiles.to_csv(profiles_path, index=False, encoding="utf-8-sig")
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    checkpoint_path.write_text(
        json.dumps(
            {
                "task": "H5_environment_group_profiles_and_stability",
                "status": "complete",
                "cell_label_rows": int(len(labels)),
                "group_columns": [name for name in labels if name != "cell_id"],
                "profile_rows": int(len(profiles)),
                "method_comparison_rows": int(len(comparison)),
                "outcome_guided_selection": True,
                "group_labels_outcome_blind": True,
                "within_group_response_family": "natural_spline, apparent fit within matched pairs",
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    return {
        "labels": labels,
        "stability": stability,
        "profiles": profiles,
        "method_comparison": comparison,
    }
