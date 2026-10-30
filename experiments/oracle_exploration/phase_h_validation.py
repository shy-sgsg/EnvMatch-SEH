"""Spatial holdout validation and common-period comparisons for Phase H."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from experiments.oracle_exploration.phase_h_metrics import regression_metrics
from experiments.oracle_exploration.phase_h_search import (
    MODEL_SCHEDULE_VERSION,
    _fit_predictions,
    common_pair_period_sample,
    enrich_pair_spatial_coordinates,
    prepare_spatial_split,
)


def score_spatial_split(
    split: Mapping[str, Any],
    *,
    outcome_column: str,
    exposure_column: str,
    model_family: str,
    seed: int,
    tree_training_max_rows: int = 12000,
) -> list[dict[str, Any]]:
    """Fit on train blocks and score untouched validation and test blocks."""

    train = split["train"]
    details = split.get("details", {})
    scores: list[dict[str, Any]] = []
    for evaluation_set in ("validation", "test"):
        score = split[evaluation_set]
        record: dict[str, Any] = {
            "evaluation_set": f"spatial_{evaluation_set}",
            "model_family": model_family,
            "n_train_available": int(len(train)),
            "n_train": 0,
            "n_evaluation": int(len(score)),
            "r_squared": float("nan"),
            "rmse": float("nan"),
            "mae": float("nan"),
            "spearman_rho": float("nan"),
            "direction": "unknown",
            "r2_definition": "1 - SSE / sum((y - mean(y))^2)",
            "status": "insufficient_sample",
            "model_parameters_json": "{}",
            **details,
        }
        if len(train) < 10 or len(score) < 3:
            scores.append(record)
            continue
        train_fit = train
        tree_families = {"random_forest", "extra_trees", "hist_gradient_boosting"}
        if model_family in tree_families and len(train_fit) > int(tree_training_max_rows):
            train_fit = train_fit.sample(n=int(tree_training_max_rows), random_state=int(seed)).sort_index()
        try:
            prediction, parameters = _fit_predictions(
                train_fit,
                score,
                outcome=outcome_column,
                exposure=exposure_column,
                features=[exposure_column],
                family=model_family,
                seed=int(seed),
            )
            truth = pd.to_numeric(score[outcome_column], errors="coerce").to_numpy(dtype=float)
            metrics = regression_metrics(truth, prediction)
            x = pd.to_numeric(score[exposure_column], errors="coerce").to_numpy(dtype=float)
            rho = float(spearmanr(x, truth).statistic) if np.unique(x).size > 1 and np.unique(truth).size > 1 else float("nan")
            record.update(
                metrics,
                n_train=int(len(train_fit)),
                n_evaluation=int(len(score)),
                spearman_rho=rho,
                direction=("negative" if rho < 0 else "positive" if rho > 0 else "near_zero") if np.isfinite(rho) else "unknown",
                status="ok",
                model_parameters_json=json.dumps(parameters, ensure_ascii=False, sort_keys=True, default=str),
            )
        except Exception as exc:
            record["status"] = f"failed: {type(exc).__name__}: {exc}"
        scores.append(record)
    return scores


def _finite_model_rows(frame: pd.DataFrame, *, outcome: str, exposure: str) -> pd.DataFrame:
    result = frame.copy()
    for name in (outcome, exposure):
        result[name] = pd.to_numeric(result[name], errors="coerce")
    finite = np.isfinite(result[[outcome, exposure]].to_numpy(dtype=float)).all(axis=1)
    return result.loc[finite].copy()


def _selected_apparent_trials(run_dir: Path) -> pd.DataFrame:
    trial_path = Path(run_dir) / "tables" / "phase_h_model_trials.csv"
    if not trial_path.is_file():
        raise FileNotFoundError(f"Phase H model trials are incomplete: {trial_path}")
    trials = pd.read_csv(trial_path, low_memory=False)
    primary = trials.loc[
        trials.get("model_schedule_version", pd.Series(index=trials.index, dtype=object)).eq(MODEL_SCHEDULE_VERSION)
        & trials["stage"].eq("stage2_full_sample")
        & trials["analysis_population"].eq("phase_h_matched_only")
        & trials["subset_name"].eq("all")
        & trials["fit_scope"].eq("exposure_only")
        & trials["evaluation_set"].eq("apparent")
        & trials["status"].eq("ok")
    ].copy()
    primary["r_squared"] = pd.to_numeric(primary["r_squared"], errors="coerce")
    primary = primary.loc[np.isfinite(primary["r_squared"])].copy()
    if primary.empty:
        raise ValueError("No successful Stage 2 apparent model trials are available")
    return primary.sort_values(
        ["exposure", "outcome", "estimand", "r_squared", "candidate_id", "model_family"],
        ascending=[True, True, True, False, True, True],
        kind="stable",
    ).drop_duplicates(["exposure", "outcome", "estimand"], keep="first")


def _candidate_units(
    pairs: pd.DataFrame,
    cells: pd.DataFrame,
    *,
    estimand: str,
    outcome: str,
    exposure: str,
    block_size_m: float,
) -> tuple[pd.DataFrame, int]:
    spatial_pairs = enrich_pair_spatial_coordinates(pairs, cells, block_size_m=block_size_m)
    same_block = spatial_pairs.loc[spatial_pairs["same_spatial_block"]].copy()
    if estimand == "matched_pair":
        response_column = f"delta_{outcome}"
        if "delta_log1p_Aa" not in same_block or response_column not in same_block:
            return pd.DataFrame(), len(same_block)
        units = pd.DataFrame(
            {
                "unit_id": same_block["pair_id"].to_numpy(),
                "spatial_block_id": same_block["spatial_block_id"].astype(str).to_numpy(),
                "unit_x": same_block["x_mid"].to_numpy(dtype=float),
                "unit_y": same_block["y_mid"].to_numpy(dtype=float),
                "exposure_value": pd.to_numeric(same_block["delta_log1p_Aa"], errors="coerce").to_numpy(dtype=float),
                "outcome_value": pd.to_numeric(same_block[response_column], errors="coerce").to_numpy(dtype=float),
            }
        )
        return units, len(same_block)

    if estimand != "matched_pixel":
        raise ValueError(f"unsupported Phase H spatial estimand: {estimand}")
    endpoint_ids = pd.Index(pd.unique(pd.concat(
        [same_block["high_erosion_cell_id"], same_block["low_erosion_cell_id"]], ignore_index=True
    )))
    selected = cells.loc[cells["cell_id"].isin(endpoint_ids)].drop_duplicates("cell_id").copy()
    if selected.empty:
        return pd.DataFrame(), len(same_block)
    block = (
        np.floor(pd.to_numeric(selected["x"], errors="coerce") / block_size_m).astype("Int64").astype(str)
        + "_"
        + np.floor(pd.to_numeric(selected["y"], errors="coerce") / block_size_m).astype("Int64").astype(str)
    )
    units = pd.DataFrame(
        {
            "unit_id": selected["cell_id"].to_numpy(),
            "spatial_block_id": block.to_numpy(),
            "unit_x": pd.to_numeric(selected["x"], errors="coerce").to_numpy(dtype=float),
            "unit_y": pd.to_numeric(selected["y"], errors="coerce").to_numpy(dtype=float),
            "exposure_value": pd.to_numeric(selected[exposure], errors="coerce").to_numpy(dtype=float),
            "outcome_value": pd.to_numeric(selected[outcome], errors="coerce").to_numpy(dtype=float),
        }
    )
    return units, len(same_block)


def _spatial_unit_trial(
    units: pd.DataFrame,
    *,
    candidate_id: str,
    exposure: str,
    outcome: str,
    estimand: str,
    model_family: str,
    seeds: list[int],
    buffer_distances: list[float],
    block_size_m: float,
    tree_training_max_rows: int,
    metadata: Mapping[str, Any],
) -> pd.DataFrame:
    if units.empty:
        return pd.DataFrame()
    units = _finite_model_rows(units, outcome="outcome_value", exposure="exposure_value")
    if units.empty:
        return pd.DataFrame()
    all_blocks = sorted(units["spatial_block_id"].astype(str).unique().tolist())
    records: list[dict[str, Any]] = []
    for seed in seeds:
        for buffer_m in buffer_distances:
            split = prepare_spatial_split(
                units,
                unit_id_column="unit_id",
                block_column="spatial_block_id",
                x_column="unit_x",
                y_column="unit_y",
                seed=int(seed),
                buffer_m=float(buffer_m),
                block_ids=all_blocks,
            )
            scores = score_spatial_split(
                split,
                outcome_column="outcome_value",
                exposure_column="exposure_value",
                model_family=model_family,
                seed=int(seed),
                tree_training_max_rows=tree_training_max_rows,
            )
            for score in scores:
                records.append(
                    {
                        **metadata,
                        "candidate_id": candidate_id,
                        "exposure": exposure,
                        "outcome": outcome,
                        "estimand": estimand,
                        "model_family": model_family,
                        "selection_scope": "apparent Stage 2 best candidate and family selected on all matched data; spatial scores are diagnostic",
                        **score,
                    }
                )
    return pd.DataFrame.from_records(records)


def run_phase_h_spatial_validation(
    run_dir: Path,
    cells: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate each outcome/estimand's full-sample apparent winner spatially."""

    root = Path(run_dir)
    table_dir = root / "tables"
    progress_dir = table_dir / "spatial_progress"
    progress_dir.mkdir(parents=True, exist_ok=True)
    raw_path = table_dir / "phase_h_spatial_validation_by_split.csv"
    summary_path = table_dir / "phase_h_spatial_validation_summary.csv"
    checkpoint_path = root / "checkpoints" / "h6_spatial_validation_complete.json"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_path.is_file() and raw_path.is_file() and summary_path.is_file():
        return pd.read_csv(raw_path), pd.read_csv(summary_path)

    selected = _selected_apparent_trials(root)
    validation_config = config["spatial_validation"]
    block_size = float(validation_config["block_size_m"])
    seeds = list(map(int, validation_config["seeds"]))
    buffers = list(map(float, validation_config["buffer_distances_m"]))
    max_rows = int(config.get("models", {}).get("tree_training_max_rows", 12000))
    candidate_dir = root / "pairs" / "candidate_pairs"
    progress_frames: list[pd.DataFrame] = []
    candidate_files = {
        candidate_id: candidate_dir / f"{candidate_id}.parquet"
        for candidate_id in selected["candidate_id"].astype(str).unique()
    }
    for candidate_id, candidate_rows in selected.groupby("candidate_id", sort=True):
        pair_path = candidate_files[str(candidate_id)]
        if not pair_path.is_file():
            raise FileNotFoundError(f"Selected Stage 2 pair table is missing: {pair_path}")
        candidate_pairs = pd.read_parquet(pair_path)
        pair_cache: dict[str, tuple[pd.DataFrame, int]] = {}
        for _, winner in candidate_rows.iterrows():
            exposure = str(winner["exposure"])
            outcome = str(winner["outcome"])
            estimand = str(winner["estimand"])
            progress_path = progress_dir / f"{exposure}__{outcome}__{estimand}.csv"
            if progress_path.is_file():
                progress_frames.append(pd.read_csv(progress_path))
                continue
            cache_key = f"{estimand}::{exposure}::{outcome}"
            if cache_key not in pair_cache:
                pair_cache[cache_key] = _candidate_units(
                    candidate_pairs,
                    cells,
                    estimand=estimand,
                    outcome=outcome,
                    exposure=exposure,
                    block_size_m=block_size,
                )
            units, same_block_pairs = pair_cache[cache_key]
            metadata = {
                "stage": "stage2_spatial_validation",
                "analysis_population": "phase_h_matched_only",
                "evaluation_set": "spatial_holdout",
                "selection_parent_candidate_id": winner.get("selection_parent_candidate_id"),
                "stage1_rank": winner.get("stage1_rank"),
                "apparent_r_squared": float(winner["r_squared"]),
                "n_matched_pairs_candidate": int(winner.get("n_matched_pairs_candidate", len(candidate_pairs))),
                "n_same_block_pairs": int(same_block_pairs),
                "same_block_pair_retention": float(same_block_pairs / len(candidate_pairs)) if len(candidate_pairs) else 0.0,
                "n_spatial_units": int(len(units)),
                "buffer_distances_m": json.dumps(buffers),
                "spatial_blocks_defined_by": "5 km grid cell of the raster point or pair midpoint",
                "buffer_defined_by": "Euclidean distance between unit coordinates (pixels) or matched-pair midpoints",
                "no_cross_role_leakage_rule": "whole spatial blocks assigned to one role; same-block pairs have both endpoints in that block",
                "outcome_guided_selection": True,
            }
            raw = _spatial_unit_trial(
                units,
                candidate_id=str(candidate_id),
                exposure=exposure,
                outcome=outcome,
                estimand=estimand,
                model_family=str(winner["model_family"]),
                seeds=seeds,
                buffer_distances=buffers,
                block_size_m=block_size,
                tree_training_max_rows=max_rows,
                metadata=metadata,
            )
            raw.to_csv(progress_path, index=False, encoding="utf-8-sig")
            progress_frames.append(raw)
            print(
                f"Phase H spatial validation: {exposure}/{outcome}/{estimand}, "
                f"{len(raw):,} split scores, candidate {candidate_id}",
                flush=True,
            )
        del candidate_pairs, pair_cache

    raw = pd.concat(progress_frames, ignore_index=True) if progress_frames else pd.DataFrame()
    raw.to_csv(raw_path, index=False, encoding="utf-8-sig")
    if raw.empty:
        summary = pd.DataFrame()
    else:
        test = raw.loc[raw["evaluation_set"].eq("spatial_test") & raw["status"].eq("ok")].copy()
        summary = test.groupby(
            ["exposure", "outcome", "estimand", "candidate_id", "model_family", "buffer_m"],
            sort=True,
            dropna=False,
        ).agg(
            mean_spatial_test_r_squared=("r_squared", "mean"),
            median_spatial_test_r_squared=("r_squared", "median"),
            sd_spatial_test_r_squared=("r_squared", "std"),
            mean_n_train=("n_train", "mean"),
            mean_n_train_available=("n_train_available", "mean"),
            mean_n_test=("n_evaluation", "mean"),
            n_matched_pairs_candidate=("n_matched_pairs_candidate", "first"),
            n_same_block_pairs=("n_same_block_pairs", "first"),
            n_spatial_units=("n_spatial_units", "first"),
            n_seed_splits=("seed", "nunique"),
            mean_apparent_r_squared=("apparent_r_squared", "mean"),
            mean_same_block_pair_retention=("same_block_pair_retention", "mean"),
        ).reset_index()
        summary["leaderboard_scope"] = "spatial validation of the apparent full-sample winner; not independently selected"
        summary = summary.sort_values(
            ["exposure", "outcome", "estimand", "mean_spatial_test_r_squared"],
            ascending=[True, True, True, False],
            kind="stable",
        ).reset_index(drop=True)
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    summary.to_csv(table_dir / "phase_h_leaderboard_5_best_spatial_validation.csv", index=False, encoding="utf-8-sig")
    checkpoint_path.write_text(
        json.dumps(
            {
                "task": "H6_spatial_holdout_validation",
                "status": "complete",
                "selected_apparent_units": int(len(selected)),
                "raw_spatial_scores": int(len(raw)),
                "summary_rows": int(len(summary)),
                "seeds": seeds,
                "buffer_distances_m": buffers,
                "block_size_m": block_size,
                "candidate_selection": "best Stage 2 apparent full-sample R2 by exposure, outcome, estimand",
                "outcome_guided_selection": True,
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    return raw, summary


def _common_pixel_period_sample(
    pairs_20yr: pd.DataFrame,
    pairs_2020: pd.DataFrame,
    cells: pd.DataFrame,
    *,
    outcome: str,
    block_size_m: float,
) -> pd.DataFrame:
    pairs_20yr = enrich_pair_spatial_coordinates(pairs_20yr, cells, block_size_m=block_size_m)
    pairs_2020 = enrich_pair_spatial_coordinates(pairs_2020, cells, block_size_m=block_size_m)
    pairs_20yr = pairs_20yr.loc[pairs_20yr["same_spatial_block"]]
    pairs_2020 = pairs_2020.loc[pairs_2020["same_spatial_block"]]
    ids_20yr = set(pd.concat([
        pairs_20yr["high_erosion_cell_id"], pairs_20yr["low_erosion_cell_id"]
    ], ignore_index=True))
    ids_2020 = set(pd.concat([
        pairs_2020["high_erosion_cell_id"], pairs_2020["low_erosion_cell_id"]
    ], ignore_index=True))
    common_ids = ids_20yr.intersection(ids_2020)
    selected = cells.loc[cells["cell_id"].isin(common_ids)].drop_duplicates("cell_id").copy()
    selected = selected.rename(columns={outcome: "outcome_value"})
    selected["unit_id"] = selected["cell_id"]
    selected["unit_x"] = pd.to_numeric(selected["x"], errors="coerce")
    selected["unit_y"] = pd.to_numeric(selected["y"], errors="coerce")
    selected["spatial_block_id"] = (
        np.floor(selected["unit_x"] / block_size_m).astype("Int64").astype(str)
        + "_"
        + np.floor(selected["unit_y"] / block_size_m).astype("Int64").astype(str)
    )
    for column in ("outcome_value", "Aa_20yr", "Aa_2020"):
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
    finite = np.isfinite(selected[["outcome_value", "Aa_20yr", "Aa_2020"]].to_numpy(dtype=float)).all(axis=1)
    return selected.loc[finite].copy()


def _period_pair_sample(
    pairs_20yr: pd.DataFrame,
    pairs_2020: pd.DataFrame,
    cells: pd.DataFrame,
    *,
    outcome: str,
    block_size_m: float,
) -> pd.DataFrame:
    common = common_pair_period_sample(
        pairs_20yr, pairs_2020, cells, outcome=outcome, block_size_m=block_size_m
    )
    if common.empty:
        return common
    common = common.loc[common["same_spatial_block"]].copy()
    common = common.rename(columns={f"delta_{outcome}": "outcome_value"})
    common["unit_id"] = common["canonical_pair_key"]
    common["unit_x"] = pd.to_numeric(common["x_mid"], errors="coerce")
    common["unit_y"] = pd.to_numeric(common["y_mid"], errors="coerce")
    common["spatial_block_id"] = common["spatial_block_id"].astype(str)
    for column in ("outcome_value", "delta_log1p_Aa_20yr", "delta_log1p_Aa_2020"):
        common[column] = pd.to_numeric(common[column], errors="coerce")
    finite = np.isfinite(
        common[["outcome_value", "delta_log1p_Aa_20yr", "delta_log1p_Aa_2020"]].to_numpy(dtype=float)
    ).all(axis=1)
    return common.loc[finite].copy()


def run_phase_h_period_comparison(
    run_dir: Path,
    cells: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare both Aa periods on common matched units, a shared spline, and blocks."""

    root = Path(run_dir)
    table_dir = root / "tables"
    progress_dir = table_dir / "period_progress"
    progress_dir.mkdir(parents=True, exist_ok=True)
    raw_path = table_dir / "phase_h_period_comparison_by_split.csv"
    summary_path = table_dir / "phase_h_period_comparison_summary.csv"
    checkpoint_path = root / "checkpoints" / "h7_period_comparison_complete.json"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_path.is_file() and raw_path.is_file() and summary_path.is_file():
        return pd.read_csv(raw_path), pd.read_csv(summary_path)

    selected = _selected_apparent_trials(root)
    block_size = float(config["spatial_validation"]["block_size_m"])
    seeds = list(map(int, config["spatial_validation"]["seeds"]))
    buffer_m = 500.0
    candidate_dir = root / "pairs" / "candidate_pairs"
    progress_frames: list[pd.DataFrame] = []
    selected_by_unit = {
        (str(row.exposure), str(row.outcome), str(row.estimand)): row
        for row in selected.itertuples(index=False)
    }
    outcomes = sorted(selected["outcome"].astype(str).unique())
    estimands = ("matched_pixel", "matched_pair")
    for outcome in outcomes:
        for estimand in estimands:
            winner_20 = selected_by_unit.get(("Aa_20yr", outcome, estimand))
            winner_2020 = selected_by_unit.get(("Aa_2020", outcome, estimand))
            progress_path = progress_dir / f"{outcome}__{estimand}.csv"
            if progress_path.is_file():
                progress_frames.append(pd.read_csv(progress_path))
                continue
            if winner_20 is None or winner_2020 is None:
                continue
            pairs_20yr = pd.read_parquet(candidate_dir / f"{winner_20.candidate_id}.parquet")
            pairs_2020 = pd.read_parquet(candidate_dir / f"{winner_2020.candidate_id}.parquet")
            if estimand == "matched_pixel":
                units = _common_pixel_period_sample(
                    pairs_20yr, pairs_2020, cells, outcome=outcome, block_size_m=block_size
                )
                exposure_columns = {"Aa_20yr": "Aa_20yr", "Aa_2020": "Aa_2020"}
            else:
                units = _period_pair_sample(
                    pairs_20yr, pairs_2020, cells, outcome=outcome, block_size_m=block_size
                )
                exposure_columns = {
                    "Aa_20yr": "delta_log1p_Aa_20yr",
                    "Aa_2020": "delta_log1p_Aa_2020",
                }
            if units.empty:
                progress_path.write_text("", encoding="utf-8")
                continue
            block_universe = sorted(units["spatial_block_id"].astype(str).unique().tolist())
            outcome_records: list[dict[str, Any]] = []
            for seed in seeds:
                base_split = prepare_spatial_split(
                    units,
                    unit_id_column="unit_id",
                    block_column="spatial_block_id",
                    x_column="unit_x",
                    y_column="unit_y",
                    seed=seed,
                    buffer_m=buffer_m,
                    block_ids=block_universe,
                )
                for exposure_name, exposure_column in exposure_columns.items():
                    scores = score_spatial_split(
                        base_split,
                        outcome_column="outcome_value",
                        exposure_column=exposure_column,
                        model_family="natural_spline",
                        seed=seed,
                        tree_training_max_rows=int(config.get("models", {}).get("tree_training_max_rows", 12000)),
                    )
                    for score in scores:
                        outcome_records.append(
                            {
                                "exposure": exposure_name,
                                "outcome": outcome,
                                "estimand": estimand,
                                "model_family": "natural_spline",
                                "candidate_id_20yr": winner_20.candidate_id,
                                "candidate_id_2020": winner_2020.candidate_id,
                                "apparent_r_squared_20yr": float(winner_20.r_squared),
                                "apparent_r_squared_2020": float(winner_2020.r_squared),
                                "common_sample_n": int(len(units)),
                                "common_pairs_same_block": int(units["unit_id"].nunique()) if estimand == "matched_pair" else np.nan,
                                "buffer_m": buffer_m,
                                "split_definition": "same whole 5 km blocks and same 500 m buffer in both periods",
                                "orientation": "pairs are oriented high-minus-low by Aa_20yr for both periods" if estimand == "matched_pair" else "same matched cell IDs in both periods",
                                "outcome_guided_selection": True,
                                "selection_scope": "selected matched candidates by apparent Stage 2 R2; period comparison uses the common sample and same spline family",
                                **score,
                            }
                        )
            outcome_frame = pd.DataFrame.from_records(outcome_records)
            outcome_frame.to_csv(progress_path, index=False, encoding="utf-8-sig")
            progress_frames.append(outcome_frame)
            print(
                f"Phase H period comparison: {outcome}/{estimand}, "
                f"{len(units):,} common units, natural spline, {len(seeds)} shared spatial splits",
                flush=True,
            )

    raw = pd.concat(progress_frames, ignore_index=True) if progress_frames else pd.DataFrame()
    raw.to_csv(raw_path, index=False, encoding="utf-8-sig")
    if raw.empty:
        summary = pd.DataFrame()
    else:
        test = raw.loc[raw["evaluation_set"].eq("spatial_test") & raw["status"].eq("ok")].copy()
        summary = test.groupby(
            ["exposure", "outcome", "estimand", "model_family", "buffer_m"],
            sort=True,
            dropna=False,
        ).agg(
            mean_spatial_test_r_squared=("r_squared", "mean"),
            median_spatial_test_r_squared=("r_squared", "median"),
            sd_spatial_test_r_squared=("r_squared", "std"),
            mean_n_train=("n_train", "mean"),
            mean_n_test=("n_evaluation", "mean"),
            common_sample_n=("common_sample_n", "first"),
            common_pairs_same_block=("common_pairs_same_block", "first"),
            n_seed_splits=("seed", "nunique"),
        ).reset_index()
        paired = summary.pivot_table(
            index=["outcome", "estimand", "model_family", "buffer_m"],
            columns="exposure",
            values="mean_spatial_test_r_squared",
            aggfunc="first",
        ).reset_index()
        if {"Aa_20yr", "Aa_2020"}.issubset(paired.columns):
            paired["mean_test_r2_difference_20yr_minus_2020"] = paired["Aa_20yr"] - paired["Aa_2020"]
            summary = summary.merge(
                paired[["outcome", "estimand", "model_family", "buffer_m", "mean_test_r2_difference_20yr_minus_2020"]],
                on=["outcome", "estimand", "model_family", "buffer_m"],
                how="left",
                validate="many_to_one",
            )
        summary["interpretation_scope"] = "descriptive comparison on the common matched sample; not a period effect"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    checkpoint_path.write_text(
        json.dumps(
            {
                "task": "H7_common_sample_period_comparison",
                "status": "complete",
                "summary_rows": int(len(summary)),
                "model_family": "natural_spline for both Aa periods",
                "common_sample_required": True,
                "shared_spatial_seeds": seeds,
                "buffer_m": buffer_m,
                "orientation": "paired responses fixed by Aa_20yr high-minus-low ordering",
                "interpretation_scope": "descriptive; not a period effect",
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    return raw, summary


def run_phase_h_validation(
    run_dir: Path,
    cells: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, pd.DataFrame]:
    """Run spatial holdouts and the common-period comparison."""

    spatial_raw, spatial_summary = run_phase_h_spatial_validation(run_dir, cells, config)
    period_raw, period_summary = run_phase_h_period_comparison(run_dir, cells, config)
    return {
        "spatial_validation_by_split": spatial_raw,
        "spatial_validation_summary": spatial_summary,
        "period_comparison_by_split": period_raw,
        "period_comparison_summary": period_summary,
    }
