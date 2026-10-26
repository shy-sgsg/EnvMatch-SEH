"""Outcome-guided Oracle matching candidates and transparent post-match scores."""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from experiments.oracle_exploration.data import HEALTH_OUTCOMES
from experiments.oracle_exploration.environment_search import (
    PRIMARY_SOIL_CODES,
    candidate_environment_sets,
    fit_environment_labels,
    transform_environment_features,
)
from experiments.oracle_exploration.registry import make_experiment_id, record_experiment
from soilmatch.matching import match_cells


@dataclass
class CandidateResult:
    """One matching trial, including its fixed pairs, scores, and diagnostics."""

    candidate: dict[str, Any]
    status: str
    pairs: pd.DataFrame = field(default_factory=pd.DataFrame)
    scores: pd.DataFrame = field(default_factory=pd.DataFrame)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


def _git_metadata() -> tuple[str, bool]:
    root = Path(__file__).resolve().parents[2]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=False
    )
    sha = revision.stdout.strip() if revision.returncode == 0 else "unknown"
    return sha, bool(status.stdout.strip())


def _environment_vector_candidates(config: Mapping[str, Any], seed: int) -> list[tuple[str, tuple[str, ...]]]:
    environment = config["environment"]
    available = [
        *environment.get("core_variables", ()),
        *environment.get("optional_variables", ()),
        *environment.get("supplemental_variables", ()),
    ]
    return candidate_environment_sets(
        available,
        seed=seed,
        n_random=int(environment.get("random_subset_count", 32)),
        correlated_variable_groups=environment.get("correlated_variable_groups", ()),
    )


def draw_matching_candidates(
    config: Mapping[str, Any], exposure: str, seed: int
) -> list[dict[str, object]]:
    """Draw the configured number of deterministic, outcome-blind trial designs."""

    if exposure not in config.get("exposures", ()):
        raise ValueError(f"Unsupported exposure: {exposure}")
    matching = config["matching"]
    environment_sets = _environment_vector_candidates(config, seed)
    if not environment_sets:
        raise ValueError("No environment vector candidates are available")
    methods = list(matching["distance_methods"])
    cluster_methods = list(matching["cluster_methods"])
    cluster_counts = [int(value) for value in matching["cluster_counts"] if int(value) > 0]
    if not methods or not cluster_methods or not cluster_counts:
        raise ValueError("Matching distance and clustering domains must be non-empty")

    count = int(config["stages"]["stage1"]["candidates_per_exposure"])
    rng = np.random.default_rng(seed)
    git_sha = str(config.get("git_sha") or _git_metadata()[0])
    git_dirty = bool(config.get("git_dirty", _git_metadata()[1] if "git_sha" not in config else False))
    run_date = str(config.get("run_date") or datetime.now(timezone.utc).strftime("%Y%m%d"))
    outcomes = list(config["outcomes"])
    named_vectors = [
        (name, tuple(variables))
        for name, variables in config["environment"].get("vectors", {}).items()
        if str(name).startswith("V") and set(variables).issubset(
            {feature for _, vector in environment_sets for feature in vector}
        )
    ]
    objective = "evaluate_all_outcomes_r2_and_association"
    candidates: list[dict[str, object]] = []

    for sequence in range(1, count + 1):
        if sequence <= len(named_vectors):
            vector_name, variables = named_vectors[sequence - 1]
        else:
            vector_name, variables = environment_sets[int(rng.integers(0, len(environment_sets)))]
        distance_method = str(methods[int(rng.integers(0, len(methods)))])
        cluster_method = str(cluster_methods[int(rng.integers(0, len(cluster_methods)))])
        if distance_method == "direct_nearest_neighbor":
            cluster_method = "none"
        if cluster_method == "none":
            cluster_count = 0
        else:
            cluster_count = int(cluster_counts[int(rng.integers(0, len(cluster_counts)))])

        weights: dict[str, float] | None = None
        if distance_method == "weighted_euclidean":
            raw_weights = rng.lognormal(mean=0.0, sigma=0.65, size=len(variables))
            raw_weights /= raw_weights.mean()
            weights = dict(zip(variables, raw_weights.tolist(), strict=True))

        candidate_seed = int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
        environment_caliper = matching["environment_calipers"][
            int(rng.integers(0, len(matching["environment_calipers"])))
        ]
        geographic_radius = matching["geographic_radii_m"][
            int(rng.integers(0, len(matching["geographic_radii_m"])))
        ]
        erosion_rank_gap = matching["erosion_rank_gaps"][
            int(rng.integers(0, len(matching["erosion_rank_gaps"])))
        ]
        exposure_rule = matching["exposure_rules"][
            int(rng.integers(0, len(matching["exposure_rules"])))
        ]
        candidate_id = f"{exposure}_{sequence:03d}"
        config_id = f"{exposure.replace('_', '').lower()}-{seed}-{sequence:03d}"
        candidates.append(
            {
                "candidate_id": candidate_id,
                "experiment_id": make_experiment_id(run_date, sequence, config_id),
                "experiment_type": "stage1_matching_search",
                "exposure": exposure,
                "environment_variables": list(variables),
                "vector_name": vector_name,
                "distance_method": distance_method,
                "cluster_method": cluster_method,
                "cluster_count": cluster_count,
                "environment_weights": weights,
                "environment_caliper": float(environment_caliper),
                "geographic_radius_m": int(geographic_radius),
                "erosion_rank_gap": float(erosion_rank_gap),
                "exposure_rule": str(exposure_rule),
                "outcome_guided_selection": True,
                "selection_mode": "outcome_guided_oracle",
                "selection_outcomes": outcomes,
                "selection_objective": objective,
                "input_run_id": str(config["input_run_id"]),
                "git_sha": git_sha,
                "git_dirty": git_dirty,
                "random_seed": candidate_seed,
            }
        )
    return candidates


def _ols_metrics(x: np.ndarray, y: np.ndarray) -> dict[str, object]:
    finite = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x[finite], dtype=float)
    y = np.asarray(y[finite], dtype=float)
    n = len(x)
    output: dict[str, object] = {
        "status": "insufficient_sample",
        "n": n,
        "coefficient": np.nan,
        "intercept": np.nan,
        "standard_error": np.nan,
        "p_value": np.nan,
        "r_squared": np.nan,
        "adjusted_r_squared": np.nan,
        "rmse": np.nan,
        "mae": np.nan,
        "direction": "unknown",
    }
    if n < 3:
        return output
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    sxx = float(x_centered @ x_centered)
    syy = float(y_centered @ y_centered)
    if sxx <= 0 or syy <= 0:
        output["status"] = "no_variation"
        return output
    coefficient = float((x_centered @ y_centered) / sxx)
    intercept = float(y.mean() - coefficient * x.mean())
    residual = y - (intercept + coefficient * x)
    sse = float(residual @ residual)
    r_squared = float(np.clip(1.0 - sse / syy, 0.0, 1.0))
    adjusted = float(1.0 - (1.0 - r_squared) * (n - 1) / (n - 2))
    standard_error = float(np.sqrt((sse / (n - 2)) / sxx))
    if standard_error > 0:
        t_stat = coefficient / standard_error
        p_value = float(2.0 * student_t.sf(abs(t_stat), df=n - 2))
    else:
        p_value = 0.0
    output.update(
        status="ok",
        coefficient=coefficient,
        intercept=intercept,
        standard_error=standard_error,
        p_value=p_value,
        r_squared=r_squared,
        adjusted_r_squared=adjusted,
        rmse=float(np.sqrt(sse / n)),
        mae=float(np.abs(residual).mean()),
        direction="positive" if coefficient > 0 else "negative" if coefficient < 0 else "zero",
    )
    return output


def _record_candidate(
    registry_path: Path,
    candidate: Mapping[str, object],
    *,
    status: str,
    error: str | None,
    pairs: pd.DataFrame,
    scores: pd.DataFrame,
    diagnostics: Mapping[str, object],
    runtime_seconds: float,
    outcomes: Sequence[str],
) -> None:
    selected_outcomes = candidate.get("selection_outcomes", list(outcomes))
    record: dict[str, object] = {
        "experiment_id": candidate.get("experiment_id", "oracle_unknown"),
        "candidate_id": candidate.get("candidate_id"),
        "experiment_type": candidate.get("experiment_type", "matching_search"),
        "git_sha": candidate.get("git_sha", "unknown"),
        "git_dirty": candidate.get("git_dirty", False),
        "input_run_id": candidate.get("input_run_id", "unknown"),
        "exposure": candidate.get("exposure", "unknown"),
        "health_indicator": "all_requested_outcomes",
        "outcome_guided_selection": True,
        "selection_mode": candidate.get("selection_mode", "outcome_guided_oracle"),
        "selection_outcomes": list(selected_outcomes) if isinstance(selected_outcomes, (list, tuple)) else list(outcomes),
        "selection_objective": candidate.get("selection_objective", "evaluate_all_outcomes_r2_and_association"),
        "random_seed": candidate.get("random_seed", 0),
        "status": status,
        "error": error,
        "n_cells": diagnostics.get("eligible_cells", diagnostics.get("input_cells", 0)),
        "n_pairs": int(len(pairs)),
        "n_matched_cells": int(diagnostics.get("n_matched_cells", 0)),
        "area_coverage": diagnostics.get("area_coverage"),
        "matched_smd": diagnostics.get("matched_smd", {}),
        "max_abs_matched_smd": diagnostics.get("max_abs_matched_smd"),
        "median_erosion_difference": diagnostics.get("median_erosion_difference"),
        "matching_diagnostics": dict(diagnostics),
        "scores": scores.to_dict(orient="records"),
        "runtime_seconds": float(runtime_seconds),
        "candidate": dict(candidate),
    }
    if status == "failed":
        record["error"] = error or "unknown matching failure"
    record_experiment(registry_path, record)


def _score_pairs(
    source: pd.DataFrame,
    pairs: pd.DataFrame,
    outcomes: Sequence[str],
    exposure: str,
) -> pd.DataFrame:
    by_id = source.set_index("cell_id", drop=False)
    high_ids = pairs["high_erosion_cell_id"].to_list()
    low_ids = pairs["low_erosion_cell_id"].to_list()
    cell_ids = high_ids + low_ids
    matched_cells = by_id.loc[cell_ids].copy()
    records: list[dict[str, object]] = []

    cell_x = np.log1p(pd.to_numeric(matched_cells[exposure], errors="coerce").to_numpy(dtype=float))
    for outcome in outcomes:
        if outcome not in matched_cells:
            metrics = {"status": "missing_outcome", "n": 0}
            metrics.update({key: np.nan for key in ("coefficient", "intercept", "standard_error", "p_value", "r_squared", "adjusted_r_squared", "rmse", "mae")})
            metrics["direction"] = "unknown"
        else:
            values = pd.to_numeric(matched_cells[outcome], errors="coerce").to_numpy(dtype=float)
            metrics = _ols_metrics(cell_x, values)
        records.append({"outcome": outcome, "score_set": "matched_cells", **metrics})

        high_x = np.log1p(pd.to_numeric(pairs["erosion_high"], errors="coerce").to_numpy(dtype=float))
        low_x = np.log1p(pd.to_numeric(pairs["erosion_low"], errors="coerce").to_numpy(dtype=float))
        if f"{outcome}_high" not in pairs or f"{outcome}_low" not in pairs:
            pair_metrics = {"status": "missing_outcome", "n": 0}
            pair_metrics.update({key: np.nan for key in ("coefficient", "intercept", "standard_error", "p_value", "r_squared", "adjusted_r_squared", "rmse", "mae")})
            pair_metrics["direction"] = "unknown"
        else:
            high_y = pd.to_numeric(pairs[f"{outcome}_high"], errors="coerce").to_numpy(dtype=float)
            low_y = pd.to_numeric(pairs[f"{outcome}_low"], errors="coerce").to_numpy(dtype=float)
            pair_metrics = _ols_metrics(high_x - low_x, high_y - low_y)
        records.append({"outcome": outcome, "score_set": "matched_pairs", **pair_metrics})
    return pd.DataFrame.from_records(records)


def run_matching_candidate(
    frame: pd.DataFrame,
    candidate: Mapping[str, object],
    outcomes: Sequence[str],
    registry: Path,
) -> CandidateResult:
    """Construct health-blind pairs, then join actual exposure and outcomes to score."""

    started = time.perf_counter()
    selected = dict(candidate)
    diagnostics: dict[str, Any] = {"input_cells": int(len(frame))}
    empty_pairs = pd.DataFrame()
    empty_scores = pd.DataFrame()
    try:
        exposure = str(selected["exposure"])
        variables = list(selected["environment_variables"])
        required = ["cell_id", "x", "y", "SoilType", exposure, *variables]
        missing = [column for column in required if column not in frame]
        if missing:
            raise ValueError(f"Missing matching columns: {missing}")
        if frame["cell_id"].isna().any() or frame["cell_id"].duplicated().any():
            raise ValueError("cell_id values must be non-null and unique")

        source = frame.copy()
        soil = pd.to_numeric(source["SoilType"], errors="coerce")
        numeric_columns = ["x", "y", exposure, *variables]
        numeric = source.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce")
        valid = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
        valid &= soil.isin(PRIMARY_SOIL_CODES).to_numpy()
        valid &= (numeric[exposure].to_numpy(dtype=float) >= 0)
        eligible = source.loc[valid].copy().reset_index(drop=True)
        if eligible.empty:
            raise ValueError("no cells have complete matching inputs in the primary soil strata")
        for column in numeric_columns:
            eligible[column] = numeric.loc[valid, column].to_numpy(dtype=float)
        eligible["SoilType"] = soil.loc[valid].to_numpy(dtype=int)

        transformed, transform_metadata = transform_environment_features(
            eligible,
            variables,
            method=str(selected["distance_method"]),
            weights=selected.get("environment_weights"),
        )
        cluster_method = str(selected["cluster_method"])
        cluster_count = int(selected["cluster_count"])
        if cluster_method == "none":
            groups = np.zeros(len(eligible), dtype=np.int32)
            realized_cluster_counts = {str(code): 1 for code in sorted(set(eligible["SoilType"]))}
        else:
            groups = fit_environment_labels(
                eligible,
                variables,
                method=cluster_method,
                k=cluster_count,
                by_soil=True,
                weights=selected.get("environment_weights"),
                seed=int(selected.get("random_seed", 0)),
                soil_codes=PRIMARY_SOIL_CODES,
            )
            realized_cluster_counts = {}
            for code in sorted(set(eligible["SoilType"])):
                stratum = groups[eligible["SoilType"].to_numpy() == code]
                realized_cluster_counts[str(code)] = int(len(set(stratum[stratum >= 0])))

        exposure_rule = str(selected["exposure_rule"])
        exposure_values = eligible[exposure].to_numpy(dtype=float)
        rank_frame = pd.DataFrame(
            {
                "exposure": exposure_values,
                "environment_group": groups,
                "SoilType": eligible["SoilType"].to_numpy(),
            }
        )
        if exposure_rule == "within_group_rank":
            ranks = rank_frame.groupby(["environment_group", "SoilType"], sort=False)["exposure"].rank(pct=True).to_numpy()
        elif exposure_rule == "positive_only":
            positive = exposure_values > 0
            ranks = np.full(len(eligible), np.nan, dtype=float)
            if positive.any():
                positive_frame = rank_frame.loc[positive]
                ranks[positive] = positive_frame.groupby(
                    ["environment_group", "SoilType"], sort=False
                )["exposure"].rank(pct=True).to_numpy()
        elif exposure_rule == "exposure_only":
            ranks = rank_frame["exposure"].rank(pct=True).to_numpy()
        else:
            raise ValueError(f"Unsupported exposure rule: {exposure_rule}")

        matching_frame = pd.DataFrame(
            {
                "cell_id": eligible["cell_id"].to_numpy(),
                "x": eligible["x"].to_numpy(dtype=float),
                "y": eligible["y"].to_numpy(dtype=float),
                "SoilType": eligible["SoilType"].to_numpy(),
                "__environment_group": groups,
                "__exposure_rank": ranks,
            }
        )
        distance_columns: list[str] = []
        for index in range(transformed.shape[1]):
            name = f"__environment_distance_{index:02d}"
            matching_frame[name] = transformed[:, index]
            distance_columns.append(name)

        pairs, matching_diagnostics = match_cells(
            matching_frame,
            environment_columns=distance_columns,
            group_col="__environment_group",
            erosion_col="__exposure_rank",
            soil_type_col="SoilType",
            x_col="x",
            y_col="y",
            distance_limit=float(selected["geographic_radius_m"]),
            environment_distance_limit=float(selected["environment_caliper"]),
            erosion_gap=float(selected["erosion_rank_gap"]),
            allow_reuse=False,
        )
        diagnostics.update(matching_diagnostics)
        diagnostics.update(
            environment_transform=transform_metadata,
            requested_cluster_count=cluster_count,
            realized_cluster_counts=realized_cluster_counts,
            exposure_rule=exposure_rule,
        )
        if pairs.empty:
            raise ValueError("no matched pairs under the selected matching constraints")

        # Only after pair IDs are fixed do we join observed exposure and health outcomes.
        by_id = source.set_index("cell_id", drop=False)
        high_ids = pairs["high_erosion_cell_id"].tolist()
        low_ids = pairs["low_erosion_cell_id"].tolist()
        ranks_by_id = pd.Series(ranks, index=eligible["cell_id"].to_numpy())
        high_exposure = pd.to_numeric(by_id.loc[high_ids, exposure], errors="coerce").to_numpy(dtype=float)
        low_exposure = pd.to_numeric(by_id.loc[low_ids, exposure], errors="coerce").to_numpy(dtype=float)
        rank_high = ranks_by_id.loc[high_ids].to_numpy(dtype=float)
        rank_low = ranks_by_id.loc[low_ids].to_numpy(dtype=float)
        pairs["exposure_rank_high"] = rank_high
        pairs["exposure_rank_low"] = rank_low
        pairs["exposure_percentile_gap"] = rank_high - rank_low
        pairs["erosion_high"] = high_exposure
        pairs["erosion_low"] = low_exposure
        pairs["erosion_difference"] = high_exposure - low_exposure
        evaluation_set = str(selected.get("evaluation_set", "apparent"))
        if evaluation_set in {"train", "validation", "test"}:
            pairs["spatial_role"] = evaluation_set
        pairs[f"{exposure}_high"] = high_exposure
        pairs[f"{exposure}_low"] = low_exposure
        for variable in variables:
            pairs[f"{variable}_high"] = pd.to_numeric(
                by_id.loc[high_ids, variable], errors="coerce"
            ).to_numpy(dtype=float)
            pairs[f"{variable}_low"] = pd.to_numeric(
                by_id.loc[low_ids, variable], errors="coerce"
            ).to_numpy(dtype=float)
        for outcome in outcomes:
            if outcome in by_id:
                pairs[f"{outcome}_high"] = by_id.loc[high_ids, outcome].to_numpy()
                pairs[f"{outcome}_low"] = by_id.loc[low_ids, outcome].to_numpy()
                # Existing soilmatch regression helpers use the high_<Y> convention.
                pairs[f"high_{outcome}"] = pairs[f"{outcome}_high"]
                pairs[f"low_{outcome}"] = pairs[f"{outcome}_low"]

        scores = _score_pairs(source, pairs, outcomes, exposure)
        matched_smd: dict[str, float] = {}
        for variable in variables:
            high = pairs[f"{variable}_high"].to_numpy(dtype=float)
            low = pairs[f"{variable}_low"].to_numpy(dtype=float)
            finite = np.isfinite(high) & np.isfinite(low)
            high, low = high[finite], low[finite]
            if len(high) < 2:
                matched_smd[variable] = float("nan")
                continue
            pooled_sd = float(np.sqrt((np.var(high, ddof=1) + np.var(low, ddof=1)) / 2))
            difference = float(high.mean() - low.mean())
            matched_smd[variable] = difference / pooled_sd if pooled_sd > 0 else (0.0 if difference == 0 else float("nan"))
        matched_cell_ids = set(high_ids) | set(low_ids)
        eligible_count = int(len(eligible))
        diagnostics.update(
            n_matched_cells=len(matched_cell_ids),
            area_coverage=len(matched_cell_ids) / eligible_count if eligible_count else 0.0,
            matched_smd=matched_smd,
            max_abs_matched_smd=max((abs(value) for value in matched_smd.values() if np.isfinite(value)), default=float("nan")),
            median_erosion_difference=float(pd.to_numeric(pairs["erosion_difference"], errors="coerce").abs().median()),
        )
        scores["candidate_id"] = selected.get("candidate_id")
        scores["experiment_id"] = selected.get("experiment_id")
        scores["exposure"] = exposure
        scores["evaluation_set"] = evaluation_set
        scores["score_type"] = "apparent" if evaluation_set == "apparent" else "spatial_holdout"
        scores["outcome_guided_selection"] = True
        scores["selection_objective"] = selected.get("selection_objective", "evaluate_all_outcomes_r2_and_association")
        scores["n_pairs"] = len(pairs)
        scores["n_matched_cells"] = len(matched_cell_ids)
        scores["area_coverage"] = diagnostics["area_coverage"]
        scores["matched_smd_json"] = json.dumps(
            {name: value if np.isfinite(value) else None for name, value in matched_smd.items()},
            sort_keys=True,
        )
        scores["max_abs_matched_smd"] = diagnostics["max_abs_matched_smd"]
        scores["median_erosion_difference"] = diagnostics["median_erosion_difference"]
        runtime = time.perf_counter() - started
        _record_candidate(
            Path(registry), selected, status="success", error=None, pairs=pairs, scores=scores,
            diagnostics=diagnostics, runtime_seconds=runtime, outcomes=outcomes,
        )
        return CandidateResult(selected, "success", pairs, scores, diagnostics)
    except Exception as exc:
        runtime = time.perf_counter() - started
        error = f"{type(exc).__name__}: {exc}"
        try:
            _record_candidate(
                Path(registry), selected, status="failed", error=error, pairs=empty_pairs,
                scores=empty_scores, diagnostics=diagnostics, runtime_seconds=runtime,
                outcomes=outcomes,
            )
        except Exception as registry_error:
            error = f"{error}; registry recording failed: {type(registry_error).__name__}: {registry_error}"
        return CandidateResult(selected, "failed", empty_pairs, empty_scores, diagnostics, error)


def rank_stage1_candidates(
    scores: pd.DataFrame,
    *,
    score_set: str = "matched_cells",
    score_column: str = "r_squared",
) -> pd.DataFrame:
    """Rank every exposure/outcome trial by an apparent Stage 1 metric."""

    required = {"candidate_id", "experiment_id", "exposure", "outcome", "score_set", score_column}
    missing = sorted(required - set(scores.columns))
    if missing:
        raise ValueError(f"Stage 1 scores are missing columns: {missing}")
    if score_set not in {"matched_cells", "matched_pairs"}:
        raise ValueError("score_set must be matched_cells or matched_pairs")
    ranked = scores.loc[scores["score_set"].eq(score_set)].copy()
    ranked[score_column] = pd.to_numeric(ranked[score_column], errors="coerce")
    ranked = ranked.loc[np.isfinite(ranked[score_column])].copy()
    ranked = ranked.sort_values(
        ["exposure", "outcome", score_column, "candidate_id"],
        ascending=[True, True, False, True],
        kind="stable",
    )
    ranked["rank"] = ranked.groupby(["exposure", "outcome"], sort=False).cumcount() + 1
    ranked["trial_count"] = ranked.groupby(["exposure", "outcome"], sort=False)["candidate_id"].transform("nunique")
    ranked["selection_mode"] = "outcome_guided_oracle_apparent"
    ranked["selected_score"] = score_column
    return ranked.reset_index(drop=True)
