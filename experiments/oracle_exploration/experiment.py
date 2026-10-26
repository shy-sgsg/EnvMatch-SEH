"""Staged Oracle execution, run artifacts, and review report generation."""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

from experiments.oracle_exploration.data import HEALTH_OUTCOMES, load_oracle_data
from experiments.oracle_exploration.environment_search import (
    NAMED_ENVIRONMENT_SETS,
    PRIMARY_SOIL_CODES,
    fit_environment_labels,
    soil_category_counts,
)
from experiments.oracle_exploration.matching_search import (
    draw_matching_candidates,
    rank_stage1_candidates,
    run_matching_candidate,
)
from experiments.oracle_exploration.rank_results import _load_registry, rank_leaderboards
from experiments.oracle_exploration.registry import load_search_config, make_experiment_id, record_experiment
from experiments.oracle_exploration.validation import spatial_roles
from experiments.oracle_exploration.subgroup_search import build_subgroup_masks, oracle_retention_search
from experiments.oracle_exploration.regression_search import (
    compare_feature_regimes,
    fit_candidate_models,
    fit_selected_model_on_test,
)
from experiments.oracle_exploration.nonlinear_search import MODEL_FAMILIES, fit_model_family
from soilmatch.spatial_statistics import make_spatial_blocks


REVIEW_FIGURE_NAMES = (
    "01_matching_balance.png",
    "02_environment_vector_r2.png",
    "03_cluster_count_r2.png",
    "04_matching_parameters_r2.png",
    "05_retention_upper_envelope.png",
    "06_shi_three_result_classes.png",
    "07_model_family_comparison.png",
    "08_health_response_curves.png",
    "09_erosion_increment_r2.png",
    "10_exposure_comparison.png",
)


def create_output_tree(
    project_root: Path,
    run_id: str,
    output_root: str = "outputs/exploratory_oracle",
) -> dict[str, Path]:
    """Create a fresh, mode-specific output tree under the Oracle directory."""

    relative_root = Path(output_root)
    if relative_root.is_absolute() or relative_root.parts[:2] != ("outputs", "exploratory_oracle"):
        raise ValueError("Oracle outputs must stay below outputs/exploratory_oracle")
    if not re.fullmatch(r"oracle_[A-Za-z0-9_-]+", run_id):
        raise ValueError("run_id must be a path-safe Oracle identifier")
    project_root = Path(project_root).resolve()
    root = (project_root / relative_root / run_id).resolve()
    if not root.is_relative_to(project_root / "outputs" / "exploratory_oracle"):
        raise ValueError("Oracle output path escaped the allowed output root")
    root.mkdir(parents=True, exist_ok=False)
    paths = {"root": root}
    for name in ("experiments", "matching", "subgroups", "models", "tables", "figures", "diagnostics", "reports"):
        path = root / name
        path.mkdir()
        paths[name] = path
    for name in ("stage1", "stage2", "stage3"):
        path = paths["matching"] / name
        path.mkdir()
        paths[f"matching_{name}"] = path
    return paths


def _open_output_tree(
    project_root: Path,
    run_dir: Path,
    output_root: str = "outputs/exploratory_oracle",
) -> dict[str, Path]:
    """Reopen an existing Oracle output tree for a deterministic resume."""

    relative_root = Path(output_root)
    if relative_root.is_absolute() or relative_root.parts[:2] != ("outputs", "exploratory_oracle"):
        raise ValueError("Oracle outputs must stay below outputs/exploratory_oracle")
    project_root = Path(project_root).resolve()
    supplied = Path(run_dir)
    root = (project_root / supplied).resolve() if not supplied.is_absolute() else supplied.resolve()
    allowed_root = (project_root / relative_root).resolve()
    if not root.is_relative_to(allowed_root):
        raise ValueError("Resume directory must stay below the configured Oracle output root")
    if not re.fullmatch(r"oracle_[A-Za-z0-9_-]+", root.name):
        raise ValueError("Resume directory name must be a path-safe Oracle run identifier")
    if not root.is_dir():
        raise FileNotFoundError(f"Oracle resume directory does not exist: {root}")
    paths = {"root": root}
    for name in ("experiments", "matching", "subgroups", "models", "tables", "figures", "diagnostics", "reports"):
        path = root / name
        if not path.is_dir():
            raise FileNotFoundError(f"Oracle resume output is missing its {name} directory: {path}")
        paths[name] = path
    for name in ("stage1", "stage2", "stage3"):
        path = paths["matching"] / name
        if not path.is_dir():
            raise FileNotFoundError(f"Oracle resume output is missing matching {name} directory: {path}")
        paths[f"matching_{name}"] = path
    return paths


def stratified_sample(
    frame: pd.DataFrame,
    sample_size: int,
    *,
    strata_columns: Sequence[str],
    seed: int,
) -> pd.DataFrame:
    """Draw an exact, deterministic proportional sample covering each stratum."""

    if not isinstance(sample_size, int) or sample_size < 1:
        raise ValueError("sample_size must be a positive integer")
    if not strata_columns or any(column not in frame for column in strata_columns):
        raise ValueError("Every requested sample-stratification column must exist")
    if frame.empty:
        return frame.copy()
    target = min(sample_size, len(frame))
    if target == len(frame):
        return frame.copy()

    group_positions = frame.groupby(list(strata_columns), dropna=False, sort=True).indices
    keys = list(group_positions)
    sizes = np.asarray([len(group_positions[key]) for key in keys], dtype=int)
    ideal = sizes * (target / sizes.sum())
    allocation = np.floor(ideal).astype(int)
    remainder = target - int(allocation.sum())
    fractional = np.argsort(-(ideal - allocation), kind="stable")
    for index in fractional:
        if remainder == 0:
            break
        if allocation[index] < sizes[index]:
            allocation[index] += 1
            remainder -= 1
    if remainder:
        raise RuntimeError("Could not allocate the requested sample across strata")

    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for key, count in zip(keys, allocation, strict=True):
        positions = np.asarray(group_positions[key], dtype=int)
        if count == len(positions):
            selected.extend(positions.tolist())
        elif count:
            selected.extend(rng.choice(positions, size=int(count), replace=False).tolist())
    selected.sort()
    return frame.iloc[selected].copy()


def reconcile_stage_counts(
    records: pd.DataFrame,
    expected_by_exposure: Mapping[str, int],
    *,
    experiment_type: str = "stage1_matching_search",
) -> dict[str, int]:
    """Count distinct attempted trial IDs, including failures, by exposure."""

    required = {"experiment_type", "exposure"}
    missing = sorted(required - set(records.columns))
    if missing:
        raise ValueError(f"Stage records are missing columns: {missing}")
    identifier = "candidate_id" if "candidate_id" in records else "experiment_id"
    if identifier not in records:
        raise ValueError("Stage records require candidate_id or experiment_id")
    subset = records.loc[records["experiment_type"].eq(experiment_type)]
    actual = {
        exposure: int(subset.loc[subset["exposure"].eq(exposure), identifier].nunique())
        for exposure in expected_by_exposure
    }
    mismatches = {
        exposure: {"expected": int(expected), "actual": actual[exposure]}
        for exposure, expected in expected_by_exposure.items()
        if actual[exposure] != int(expected)
    }
    if mismatches:
        raise ValueError(f"Stage result count mismatch: {mismatches}")
    return actual


def _assign_roles_and_blocks(frame: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    result = frame.copy().reset_index(drop=True)
    coords = result[["x", "y"]].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(coords.to_numpy(dtype=float)).all(axis=1)
    if int(finite.sum()) < 3:
        raise ValueError("At least three cells with projected coordinates are required")
    result["spatial_role"] = "unassigned"
    result["spatial_block"] = -1
    coordinate_frame = result.loc[finite, ["x", "y"]]
    block_size = float(config["stages"]["stage3"]["block_size_m"])
    roles = spatial_roles(
        coordinate_frame,
        block_size_m=block_size,
        seed=int(config["seed"]),
    )
    blocks = make_spatial_blocks(coordinate_frame["x"], coordinate_frame["y"], block_size)
    result.loc[finite, "spatial_role"] = roles.to_numpy()
    result.loc[finite, "spatial_block"] = blocks
    return result


def _primary_population(frame: pd.DataFrame, roles: Sequence[str] | None = None) -> pd.DataFrame:
    soil = pd.to_numeric(frame["SoilType"], errors="coerce")
    selected = soil.isin(PRIMARY_SOIL_CODES).to_numpy()
    if roles is not None:
        selected &= frame["spatial_role"].isin(roles).to_numpy()
    return frame.loc[selected].copy()


def _sample_by_role(frame: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    if len(frame) <= sample_size:
        return stratified_sample(frame, len(frame), strata_columns=["SoilType", "spatial_block"], seed=seed)
    role_blocks = frame[["spatial_role", "spatial_block"]].drop_duplicates()
    block_counts = role_blocks.groupby("spatial_role", sort=False).size()
    train_blocks = int(block_counts.get("train", 0))
    validation_blocks = int(block_counts.get("validation", 0))
    denominator = train_blocks + validation_blocks
    if denominator <= 0:
        raise ValueError("No train/validation spatial blocks are available for Stage 2")
    train_target = int(round(sample_size * train_blocks / denominator))
    validation_target = sample_size - train_target
    pieces: list[pd.DataFrame] = []
    for offset, (role, target) in enumerate((("train", train_target), ("validation", validation_target))):
        population = frame.loc[frame["spatial_role"].eq(role)]
        if population.empty or target <= 0:
            continue
        pieces.append(
            stratified_sample(
                population,
                min(target, len(population)),
                strata_columns=["SoilType", "spatial_block"],
                seed=seed + offset,
            )
        )
    sampled = pd.concat(pieces, ignore_index=True) if pieces else frame.iloc[0:0].copy()
    return sampled


def _select_stage1_configs(
    scores: pd.DataFrame,
    candidates: Sequence[Mapping[str, object]],
    max_configs: int,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    candidate_frame = pd.DataFrame([dict(candidate) for candidate in candidates])
    if scores.empty:
        return [], pd.DataFrame()
    eligible = scores.loc[scores["score_set"].eq("matched_cells")].copy()
    eligible["r_squared"] = pd.to_numeric(eligible["r_squared"], errors="coerce")
    eligible = eligible.loc[np.isfinite(eligible["r_squared"])].copy()
    if eligible.empty:
        return [], pd.DataFrame()
    ranked = rank_stage1_candidates(eligible, score_set="matched_cells")
    ranked["normalized_rank"] = np.where(
        ranked["trial_count"].gt(1),
        (ranked["rank"] - 1) / (ranked["trial_count"] - 1),
        0.0,
    )
    aggregate = (
        ranked.groupby(["exposure", "candidate_id"], sort=False)
        .agg(mean_normalized_rank=("normalized_rank", "mean"), mean_r_squared=("r_squared", "mean"), outcomes_scored=("outcome", "nunique"))
        .reset_index()
        .sort_values(["exposure", "mean_normalized_rank", "mean_r_squared", "candidate_id"], ascending=[True, True, False, True], kind="stable")
    )
    picked_by_exposure: dict[str, list[str]] = {}
    for exposure_value, group_scores in ranked.groupby("exposure", sort=False):
        exposure = str(exposure_value)
        picked = picked_by_exposure.setdefault(exposure, [])
        # Ensure each available response can nominate a Stage 2 configuration.
        for _, outcome_scores in group_scores.groupby("outcome", sort=False):
            if len(picked) >= max_configs:
                break
            candidate_id = str(outcome_scores.iloc[0]["candidate_id"])
            if candidate_id not in picked:
                picked.append(candidate_id)
        # A negative SHI slope is a separate authorized selection objective.
        shi = group_scores.loc[group_scores["outcome"].eq("SHI_candidate")]
        if len(picked) < max_configs and not shi.empty:
            negative = shi.loc[pd.to_numeric(shi["coefficient"], errors="coerce").lt(0)]
            if not negative.empty:
                candidate_id = str(negative.sort_values(["r_squared", "candidate_id"], ascending=[False, True]).iloc[0]["candidate_id"])
                if candidate_id not in picked:
                    picked.append(candidate_id)
    for exposure_value, group_aggregate in aggregate.groupby("exposure", sort=False):
        picked = picked_by_exposure.setdefault(str(exposure_value), [])
        for row in group_aggregate.itertuples(index=False):
            if row.candidate_id not in picked:
                picked.append(row.candidate_id)
            if len(picked) >= max_configs:
                break
    selected = [
        candidate
        for candidate in candidates
        if candidate["candidate_id"] in picked_by_exposure.get(str(candidate["exposure"]), [])
    ]
    return selected, aggregate


def _best_validation_configs(scores: pd.DataFrame) -> tuple[dict[tuple[str, str], str], pd.DataFrame]:
    eligible = scores.loc[
        scores["evaluation_set"].eq("validation")
        & scores["score_set"].eq("matched_cells")
    ].copy()
    eligible["r_squared"] = pd.to_numeric(eligible["r_squared"], errors="coerce")
    eligible = eligible.loc[np.isfinite(eligible["r_squared"])].copy()
    if eligible.empty:
        return {}, pd.DataFrame()
    ordered = eligible.sort_values(
        ["exposure", "outcome", "r_squared", "candidate_id"],
        ascending=[True, True, False, True],
        kind="stable",
    )
    winners = ordered.groupby(["exposure", "outcome"], sort=False).head(1).copy()
    selection = {
        (str(row.exposure), str(row.outcome)): str(row.candidate_id)
        for row in winners.itertuples(index=False)
    }
    return selection, winners.reset_index(drop=True)


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _attach_role_metric_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Expose apparent/train/validation/test R² as separate audit columns."""

    result = frame.copy()
    if "r_squared" not in result:
        result["r_squared"] = np.nan
    result["r_squared"] = pd.to_numeric(result["r_squared"], errors="coerce")
    roles = (
        ("apparent", "apparent_r2"),
        ("train", "train_r2"),
        ("validation", "validation_r2"),
        ("test", "test_r2"),
    )
    evaluation = result.get("evaluation_set", pd.Series(index=result.index, dtype=object)).astype(str).str.lower()
    for role, column in roles:
        if column not in result:
            result[column] = np.nan
        result[column] = pd.to_numeric(result[column], errors="coerce")
        fill = evaluation.eq(role) & result[column].isna() & result["r_squared"].notna()
        result.loc[fill, column] = result.loc[fill, "r_squared"]

    for canonical, source in (
        ("matching_method", "distance_method"),
        ("environment_distance", "environment_caliper"),
        ("geographic_radius", "geographic_radius_m"),
        ("erosion_gap", "erosion_rank_gap"),
        ("model_parameters", "model_config"),
        ("runtime", "runtime_seconds"),
    ):
        if source in result:
            if canonical not in result:
                result[canonical] = result[source]
            else:
                result[canonical] = result[canonical].where(result[canonical].notna(), result[source])
    if "outcome" in result:
        result["health_indicator"] = result["outcome"].where(
            result["outcome"].notna(), result.get("health_indicator")
        )
    if "n_cells" not in result:
        result["n_cells"] = pd.to_numeric(result.get("n", pd.Series(index=result.index, dtype=float)), errors="coerce")
    elif "n" in result:
        result["n_cells"] = result["n_cells"].where(
            pd.to_numeric(result["n_cells"], errors="coerce").notna(),
            pd.to_numeric(result["n"], errors="coerce"),
        )
    return result


def _attach_subgroup_scope_status(frame: pd.DataFrame) -> pd.DataFrame:
    """Mark legacy SoilType 85 subgroup trials as out of the primary scope."""

    result = frame.copy()
    result["scope_status"] = "in_scope"
    experiment_types = result.get("experiment_type", pd.Series(index=result.index, dtype=object)).astype(str)
    candidate_ids = result.get("candidate_id", pd.Series(index=result.index, dtype=object)).astype(str)
    excluded = experiment_types.eq("oracle_subgroup_search") & candidate_ids.str.contains(
        r"_soil_85(?:_|$)", regex=True
    )
    result.loc[excluded, "scope_status"] = "excluded_soil_85_water_class"
    return result


def _json_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        if result[column].dtype == object and result[column].map(lambda value: isinstance(value, (dict, list, tuple))).any():
            result[column] = result[column].map(
                lambda value: json.dumps(value, ensure_ascii=False, default=str)
                if isinstance(value, (dict, list, tuple))
                else value
            )
    return result


def _record_model_scores(
    scores: pd.DataFrame,
    registry_path: Path,
    *,
    exposure: str,
    outcome: str,
    candidate_id: str,
    experiment_type: str,
    environment_variables: Sequence[str],
    environment_vector: str,
    group_column: str | None,
    outcome_guided_selection: bool,
    selection_mode: str,
    selection_objective: str,
    input_run_id: str,
    git_sha: str,
    seed: int,
    run_date: str,
    sequence_start: int,
) -> tuple[pd.DataFrame, int]:
    """Write one registry record per model family and return identified score rows."""

    if scores.empty:
        return scores.copy(), sequence_start
    output: list[pd.DataFrame] = []
    sequence = sequence_start
    for model_name, model_scores in scores.groupby("model_name", sort=False):
        sequence += 1
        id_suffix = re.sub(r"[^A-Za-z0-9-]+", "-", f"{candidate_id}-{model_name}").strip("-")
        experiment_id = make_experiment_id(run_date, sequence, id_suffix[:80] or f"model-{sequence}")
        identified = model_scores.copy()
        identified["candidate_id"] = candidate_id
        identified["experiment_id"] = experiment_id
        identified["experiment_type"] = experiment_type
        identified["exposure"] = exposure
        identified["outcome"] = outcome
        identified["health_indicator"] = outcome
        identified["environment_variables"] = [list(environment_variables)] * len(identified)
        identified["environment_vector"] = environment_vector
        identified["group_column"] = group_column
        identified["selection_mode"] = selection_mode
        identified["selection_objective"] = selection_objective
        identified["outcome_guided_selection"] = bool(outcome_guided_selection)
        model_config = str(identified.iloc[0].get("model_config", "{}"))
        statuses = identified["status"].astype(str)
        failed = not statuses.isin(["ok"]).any()
        error_values = identified.loc[statuses.eq("failed"), "error"].dropna().astype(str).unique().tolist() if "error" in identified else []
        record: dict[str, object] = {
            "experiment_id": experiment_id,
            "candidate_id": candidate_id,
            "experiment_type": experiment_type,
            "git_sha": git_sha,
            "input_run_id": input_run_id,
            "exposure": exposure,
            "health_indicator": outcome,
            "outcome_guided_selection": bool(outcome_guided_selection),
            "selection_mode": selection_mode,
            "selection_outcomes": [outcome],
            "selection_objective": selection_objective,
            "random_seed": int(seed),
            "status": "failed" if failed else "success",
            "error": "; ".join(error_values) if error_values else None,
            "environment_variables": list(environment_variables),
            "environment_vector": environment_vector,
            "group_column": group_column,
            "model_name": str(model_name),
            "model_config": model_config,
            "scores": identified.drop(columns=["candidate_id", "experiment_id", "experiment_type", "exposure", "outcome", "health_indicator", "environment_variables", "environment_vector", "group_column", "selection_mode", "selection_objective", "outcome_guided_selection"], errors="ignore").to_dict(orient="records"),
        }
        if failed and not record["error"]:
            record["error"] = "all model evaluations failed"
        record_experiment(registry_path, record)
        output.append(identified)
    return (pd.concat(output, ignore_index=True) if output else scores.copy()), sequence


def _existing_candidate_rows(
    registry_rows: pd.DataFrame,
    experiment_type: str,
    candidate_id: str,
) -> pd.DataFrame:
    if registry_rows.empty or not {"experiment_type", "candidate_id"}.issubset(registry_rows.columns):
        return registry_rows.iloc[0:0].copy()
    mask = registry_rows["experiment_type"].astype(str).eq(experiment_type)
    mask &= registry_rows["candidate_id"].astype(str).eq(candidate_id)
    return registry_rows.loc[mask].copy()


def _training_environment_groups(
    frame: pd.DataFrame,
    variables: Sequence[str],
    *,
    seed: int,
    clusters_per_soil: int = 8,
) -> np.ndarray:
    """Fit soil-specific environment groups on train blocks and assign all roles."""

    raw = frame.loc[:, list(variables)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    complete = np.isfinite(raw).all(axis=1)
    soil = pd.to_numeric(frame["SoilType"], errors="coerce").to_numpy(dtype=float)
    roles = frame["spatial_role"].astype(str).to_numpy()
    labels = np.full(len(frame), -1, dtype=np.int32)
    next_label = 0
    for soil_code in sorted(set(soil[np.isfinite(soil)].astype(int).tolist())):
        within = np.isfinite(soil) & (soil == soil_code) & complete
        training = within & (roles == "train")
        train_values = raw[training]
        if len(train_values) < 2:
            continue
        scaler = StandardScaler().fit(train_values)
        transformed_train = scaler.transform(train_values)
        distinct = np.unique(transformed_train, axis=0).shape[0]
        k = min(int(clusters_per_soil), len(transformed_train), distinct)
        if k < 1:
            continue
        model = MiniBatchKMeans(
            n_clusters=k,
            random_state=seed + int(soil_code),
            n_init=3,
            batch_size=min(4096, max(256, k * 64)),
            reassignment_ratio=0.0,
        )
        model.fit(transformed_train)
        assigned = model.predict(scaler.transform(raw[within]))
        labels[within] = assigned.astype(np.int32) + next_label
        next_label += k
    return labels


def _candidate_environment_lookup(
    matching: Mapping[str, Any],
) -> dict[tuple[str, str], dict[str, object]]:
    lookup: dict[tuple[str, str], dict[str, object]] = {}
    for exposure, candidates in matching.get("stage1_candidates", {}).items():
        for candidate in candidates:
            lookup[(str(exposure), str(candidate["candidate_id"]))] = dict(candidate)
    return lookup


def _selected_full_sample_curves(
    frame: pd.DataFrame,
    full_scores: pd.DataFrame,
    *,
    exposure: str = "Aa_20yr",
    curve_points: int = 100,
) -> pd.DataFrame:
    """Refit each apparent full-sample winner and predict an exposure partial curve."""

    if curve_points < 2:
        raise ValueError("curve_points must be at least 2")
    if full_scores.empty or not {"exposure", "outcome", "model_name", "r_squared"}.issubset(full_scores.columns):
        return pd.DataFrame()
    scores = full_scores.loc[
        full_scores["exposure"].eq(exposure)
        & full_scores.get("evaluation_set", pd.Series(index=full_scores.index, dtype=object)).astype(str).eq("apparent")
        & full_scores.get("status", pd.Series("ok", index=full_scores.index)).astype(str).eq("ok")
    ].copy()
    scores["r_squared"] = pd.to_numeric(scores["r_squared"], errors="coerce")
    scores = scores.loc[np.isfinite(scores["r_squared"])].copy()
    if scores.empty:
        return pd.DataFrame()
    scores = scores.sort_values(
        ["outcome", "r_squared", "model_name", "environment_vector"],
        ascending=[True, False, True, True],
        kind="stable",
    )
    winners = scores.groupby("outcome", sort=False).head(1)
    records: list[dict[str, object]] = []

    for winner in winners.to_dict(orient="records"):
        outcome = str(winner["outcome"])
        model_name = str(winner["model_name"])
        if model_name not in MODEL_FAMILIES:
            continue
        environment = winner.get("environment_variables", [])
        if isinstance(environment, str):
            try:
                environment = json.loads(environment)
            except json.JSONDecodeError:
                environment = [part.strip() for part in environment.split(",") if part.strip()]
        if not isinstance(environment, (list, tuple)):
            continue
        environment = [str(name) for name in environment]
        required = [outcome, exposure, *environment]
        if any(column not in frame for column in required):
            continue
        group_column = winner.get("group_column")
        if group_column is not None and pd.isna(group_column):
            group_column = None
        if group_column is not None and str(group_column) not in frame:
            group_column = None
        if model_name in {"environment_group_fixed_effects", "interactions"} and group_column is None:
            continue

        train = frame.copy()
        numeric_columns = list(dict.fromkeys([outcome, exposure, *environment]))
        numeric = train[numeric_columns].apply(pd.to_numeric, errors="coerce")
        complete = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
        complete &= numeric[exposure].to_numpy(dtype=float) >= 0
        if group_column is not None:
            complete &= train[str(group_column)].notna().to_numpy()
        train = train.loc[complete].copy()
        for column in numeric_columns:
            train[column] = numeric.loc[complete, column].to_numpy(dtype=float)
        if len(train) < 3 or train[exposure].nunique() < 2:
            continue

        exposure_values = train[exposure].to_numpy(dtype=float)
        lower, upper = np.quantile(exposure_values, [0.01, 0.99])
        grid = np.unique(np.linspace(float(lower), float(upper), curve_points))
        if len(grid) < 2:
            continue
        prediction_frame = pd.DataFrame(
            {exposure: grid, outcome: float(train[outcome].median())}
        )
        environment_at_median = {
            variable: float(train[variable].median()) for variable in environment
        }
        for variable, value in environment_at_median.items():
            prediction_frame[variable] = value
        group_at_mode = None
        if group_column is not None:
            group_values = train[str(group_column)].mode(dropna=True)
            if group_values.empty:
                continue
            group_at_mode = group_values.iloc[0]
            prediction_frame[str(group_column)] = group_at_mode
        try:
            prediction, _ = fit_model_family(
                train,
                prediction_frame,
                outcome,
                exposure,
                [exposure, *environment],
                group_col=str(group_column) if group_column is not None else None,
                model_family=model_name,
            )
        except Exception as exc:
            print(
                f"[Figures] Could not refit full-sample curve for {exposure}/{outcome}/{model_name}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            continue
        environment_state: dict[str, object] = dict(environment_at_median)
        if group_column is not None:
            environment_state[str(group_column)] = group_at_mode
        for x_value, y_value in zip(grid, prediction, strict=True):
            records.append(
                {
                    "outcome": outcome,
                    "exposure": exposure,
                    "model_name": model_name,
                    "environment_vector": winner.get("environment_vector"),
                    "selected_model_r_squared": float(winner["r_squared"]),
                    "n_complete_cases": int(len(train)),
                    "environment_at_median": environment_state,
                    "exposure_value": float(x_value),
                    "predicted_outcome": float(y_value),
                    "curve_type": "full_sample_best_model_partial_dependence",
                    "selected_by": "full_sample_apparent_r2",
                }
            )
    return pd.DataFrame.from_records(records)


def _complete_full_sample_curves(
    frame: pd.DataFrame,
    full_scores: pd.DataFrame,
    existing_curves: pd.DataFrame,
    *,
    exposure: str = "Aa_20yr",
    curve_points: int = 100,
) -> pd.DataFrame:
    """Keep saved curves and fit curves for outcomes missing from the cache."""

    existing = existing_curves.copy()
    if existing.empty or not {"outcome", "exposure"}.issubset(existing.columns):
        existing = pd.DataFrame()
    covered = (
        set(existing.loc[existing["exposure"].eq(exposure), "outcome"].dropna().astype(str))
        if not existing.empty
        else set()
    )
    missing_scores = full_scores.loc[
        full_scores.get("exposure", pd.Series(index=full_scores.index, dtype=object)).eq(exposure)
        & ~full_scores.get("outcome", pd.Series(index=full_scores.index, dtype=object)).astype(str).isin(covered)
    ].copy()
    missing_curves = _selected_full_sample_curves(
        frame,
        missing_scores,
        exposure=exposure,
        curve_points=curve_points,
    )
    if existing.empty:
        return missing_curves
    if missing_curves.empty:
        return existing
    return pd.concat([existing, missing_curves], ignore_index=True, sort=False)


def _run_regression_search(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
    paths: Mapping[str, Path],
    registry_path: Path,
    matching: Mapping[str, Any],
    run_date: str,
    git_sha: str,
) -> dict[str, pd.DataFrame]:
    """Compare unfiltered full-sample models and validation-selected spatial models."""

    available = set(frame.columns)
    named_vectors = {
        name: tuple(variables)
        for name, variables in config["environment"].get("vectors", NAMED_ENVIRONMENT_SETS).items()
        if set(variables).issubset(available)
    }
    all_families = [family for family in config.get("model_families", MODEL_FAMILIES) if family in MODEL_FAMILIES]
    if not all_families:
        raise ValueError("No supported regression model families are available")
    train_validation = frame.loc[frame["spatial_role"].isin(["train", "validation"])].copy()
    full_sample = frame.drop(columns=["spatial_role"], errors="ignore").copy()
    stage2_sample = matching.get("stage2_sample")
    if not isinstance(stage2_sample, pd.DataFrame) or stage2_sample.empty:
        raise ValueError("Stage 2 regression search requires its sampled training/validation rows")
    stage2_sample = stage2_sample.copy()
    if len(stage2_sample) > 100000:
        raise ValueError("Stage 2 regression sample cannot exceed 100,000 cells")
    if "spatial_role" in stage2_sample and stage2_sample["spatial_role"].astype(str).eq("test").any():
        raise ValueError("Stage 2 regression sample must not include held-out test blocks")
    spatial_scores: list[pd.DataFrame] = []
    full_scores: list[pd.DataFrame] = []
    stage2_scores: list[pd.DataFrame] = []
    feature_scores: list[pd.DataFrame] = []
    candidate_context: dict[str, dict[str, object]] = {}
    group_cache: dict[tuple[str, ...], str] = {}
    registry_path = Path(registry_path)
    registry_record_count = (
        sum(1 for line in registry_path.read_text(encoding="utf-8").splitlines() if line.strip())
        if registry_path.is_file()
        else 0
    )
    registered_scores = _load_registry(registry_path) if registry_path.is_file() else pd.DataFrame()
    sequence = registry_record_count
    total = len(config["exposures"]) * len(config["outcomes"])
    completed = 0
    winner_config_map = matching.get("stage1_candidates", {})
    parent_lookup = {
        (str(exposure), str(candidate["candidate_id"])): dict(candidate)
        for exposure, values in winner_config_map.items()
        for candidate in values
    }

    for exposure in config["exposures"]:
        for outcome in config["outcomes"]:
            parent_id = matching.get("validation_winners", {}).get((exposure, outcome))
            parent = parent_lookup.get((exposure, parent_id)) if parent_id else None
            selected_variables = tuple(parent["environment_variables"]) if parent else named_vectors.get("V1", ())
            selected_vector = str(parent["vector_name"]) if parent else "V1_fallback"
            if not selected_variables or not set(selected_variables).issubset(available):
                selected_variables = named_vectors.get("V1", ())
                selected_vector = "V1_fallback"
            if not selected_variables:
                print(f"[Models] Skipping {exposure}/{outcome}: no complete environment vector.", flush=True)
                continue

            group_key = tuple(selected_variables)
            if group_key not in group_cache:
                group_column = f"oracle_environment_group_{len(group_cache) + 1:02d}"
                frame[group_column] = _training_environment_groups(
                    frame,
                    selected_variables,
                    seed=int(config["seed"]),
                    clusters_per_soil=8,
                )
                train_validation[group_column] = frame.loc[train_validation.index, group_column].to_numpy()
                full_sample[group_column] = frame[group_column].to_numpy()
                group_cache[group_key] = group_column
            group_column = group_cache[group_key]
            if group_column not in stage2_sample:
                group_by_cell = frame.set_index("cell_id")[group_column]
                stage2_sample[group_column] = stage2_sample["cell_id"].map(group_by_cell)

            stage2_candidate_id = f"stage2_{exposure}_{outcome}_{selected_vector}"
            existing_stage2 = _existing_candidate_rows(
                registered_scores, "stage2_model_search", stage2_candidate_id
            )
            existing_families = set(existing_stage2.get("model_name", pd.Series(dtype=str)).dropna().astype(str))
            missing_families = [family for family in all_families if family not in existing_families]
            new_stage2 = pd.DataFrame()
            if missing_families:
                new_stage2 = fit_candidate_models(
                    stage2_sample,
                    outcome,
                    exposure,
                    selected_variables,
                    group=group_column,
                    model_families=missing_families,
                    outcome_guided_selection=True,
                )
                new_stage2["sample_size_requested"] = int(
                    config.get("stages", {}).get("stage2", {}).get("sample_size", len(stage2_sample))
                )
                new_stage2["sample_size_actual"] = len(stage2_sample)
                new_stage2, sequence = _record_model_scores(
                    new_stage2,
                    registry_path,
                    exposure=exposure,
                    outcome=outcome,
                    candidate_id=stage2_candidate_id,
                    experiment_type="stage2_model_search",
                    environment_variables=selected_variables,
                    environment_vector=selected_vector,
                    group_column=group_column,
                    outcome_guided_selection=True,
                    selection_mode="validation_selected_stage2_model_search",
                    selection_objective="maximize_validation_r2_within_stage2_model_families",
                    input_run_id=str(config["input_run_id"]),
                    git_sha=git_sha,
                    seed=int(config["seed"]),
                    run_date=run_date,
                    sequence_start=sequence,
                )
                registered_scores = pd.concat([registered_scores, new_stage2], ignore_index=True, sort=False)
            stage2_result = pd.concat([existing_stage2, new_stage2], ignore_index=True, sort=False)
            stage2_scores.append(stage2_result)

            vectors_to_run: list[tuple[str, tuple[str, ...], bool]] = []
            seen: set[tuple[str, ...]] = set()
            for vector_name, variables in named_vectors.items():
                if tuple(variables) in seen:
                    continue
                seen.add(tuple(variables))
                vectors_to_run.append((vector_name, tuple(variables), tuple(variables) == selected_variables))
            if selected_variables not in seen:
                vectors_to_run.append((selected_vector, selected_variables, True))

            for vector_name, variables, is_selected_vector in vectors_to_run:
                model_families = all_families if is_selected_vector else ["ols"]
                group_for_vector = group_column if is_selected_vector else None
                candidate_id = f"spatial_{exposure}_{outcome}_{vector_name}"
                candidate_context[candidate_id] = {
                    "exposure": exposure,
                    "outcome": outcome,
                    "environment_variables": list(variables),
                    "environment_vector": vector_name,
                    "group_column": group_for_vector,
                    "model_families": model_families,
                }

                existing_spatial = _existing_candidate_rows(
                    registered_scores, "spatial_model_search", candidate_id
                )
                existing_spatial_families = set(
                    existing_spatial.get("model_name", pd.Series(dtype=str)).dropna().astype(str)
                )
                missing_spatial_families = [
                    family for family in model_families if family not in existing_spatial_families
                ]
                new_spatial = pd.DataFrame()
                if missing_spatial_families:
                    new_spatial = fit_candidate_models(
                        train_validation,
                        outcome,
                        exposure,
                        variables,
                        group=group_for_vector,
                        model_families=missing_spatial_families,
                    )
                    new_spatial, sequence = _record_model_scores(
                        new_spatial,
                        registry_path,
                        exposure=exposure,
                        outcome=outcome,
                        candidate_id=candidate_id,
                        experiment_type="spatial_model_search",
                        environment_variables=variables,
                        environment_vector=vector_name,
                        group_column=group_for_vector,
                        outcome_guided_selection=True,
                        selection_mode="validation_selected_model_search",
                        selection_objective="maximize_validation_r2",
                        input_run_id=str(config["input_run_id"]),
                        git_sha=git_sha,
                        seed=int(config["seed"]),
                        run_date=run_date,
                        sequence_start=sequence,
                    )
                    registered_scores = pd.concat([registered_scores, new_spatial], ignore_index=True, sort=False)
                spatial_result = pd.concat([existing_spatial, new_spatial], ignore_index=True, sort=False)
                spatial_scores.append(spatial_result)

                full_candidate_id = f"full_{exposure}_{outcome}_{vector_name}"
                existing_full = _existing_candidate_rows(
                    registered_scores, "full_sample_model", full_candidate_id
                )
                existing_full_families = set(
                    existing_full.get("model_name", pd.Series(dtype=str)).dropna().astype(str)
                )
                missing_full_families = [
                    family for family in model_families if family not in existing_full_families
                ]
                new_full = pd.DataFrame()
                if missing_full_families:
                    new_full = fit_candidate_models(
                        full_sample,
                        outcome,
                        exposure,
                        variables,
                        group=group_for_vector,
                        model_families=missing_full_families,
                        outcome_guided_selection=True,
                    )
                    new_full, sequence = _record_model_scores(
                        new_full,
                        registry_path,
                        exposure=exposure,
                        outcome=outcome,
                        candidate_id=full_candidate_id,
                        experiment_type="full_sample_model",
                        environment_variables=variables,
                        environment_vector=vector_name,
                        group_column=group_for_vector,
                        outcome_guided_selection=True,
                        selection_mode="full_sample_unfiltered_model_search",
                        selection_objective="maximize_apparent_r2_across_fixed_model_search",
                        input_run_id=str(config["input_run_id"]),
                        git_sha=git_sha,
                        seed=int(config["seed"]),
                        run_date=run_date,
                        sequence_start=sequence,
                    )
                    registered_scores = pd.concat([registered_scores, new_full], ignore_index=True, sort=False)
                full_result = pd.concat([existing_full, new_full], ignore_index=True, sort=False)
                full_scores.append(full_result)

            # All five feature regimes share the same outcome/exposure/environment rows.
            pair_path = None
            if parent is not None:
                candidate_pair_path = paths["matching_stage3"] / f"{parent_id}_S3_test_pairs.parquet"
                if candidate_pair_path.is_file():
                    pair_path = candidate_pair_path
            selected_pairs = pd.read_parquet(pair_path) if pair_path is not None else None
            regime_result = compare_feature_regimes(
                frame,
                outcome,
                exposure,
                selected_variables,
                group=group_column,
                model_families=["ols"],
                pairs=selected_pairs,
            )
            regime_result["candidate_id"] = f"feature_regime_{exposure}_{outcome}_{selected_vector}"
            regime_result["environment_variables"] = [list(selected_variables)] * len(regime_result)
            regime_result["environment_vector"] = selected_vector
            regime_result["experiment_type"] = "feature_regime_comparison"
            regime_result["selection_mode"] = "predeclared_regime_comparison"
            regime_result["outcome_guided_selection"] = False
            regime_result["selection_objective"] = "compare_exposure_environment_and_pair_regimes"
            feature_scores.append(regime_result)
            regime_candidate_id = f"feature_regime_{exposure}_{outcome}_{selected_vector}"
            existing_regimes = _existing_candidate_rows(
                registered_scores,
                "feature_regime_comparison",
                regime_candidate_id,
            )
            existing_regime_models = set(
                existing_regimes.get("model_name", pd.Series(dtype=str)).dropna().astype(str)
            )
            for model_name, model_rows in regime_result.groupby("model_name", sort=False):
                if str(model_name) in existing_regime_models:
                    continue
                sequence += 1
                config_label = re.sub(
                    r"[^A-Za-z0-9-]+",
                    "-",
                    f"regime-{exposure}-{outcome}-{selected_vector}-{model_name}",
                ).strip("-")
                experiment_id = make_experiment_id(
                    run_date,
                    sequence,
                    config_label.lower(),
                )
                record_experiment(
                    registry_path,
                    {
                        "experiment_id": experiment_id,
                        "candidate_id": regime_candidate_id,
                        "experiment_type": "feature_regime_comparison",
                        "git_sha": git_sha,
                        "input_run_id": str(config["input_run_id"]),
                        "exposure": exposure,
                        "health_indicator": outcome,
                        "outcome_guided_selection": False,
                        "selection_mode": "predeclared_regime_comparison",
                        "selection_outcomes": [outcome],
                        "selection_objective": "compare_exposure_environment_and_pair_regimes",
                        "random_seed": int(config["seed"]),
                        "status": "success" if model_rows["status"].eq("ok").any() else "failed",
                        "error": None if model_rows["status"].eq("ok").any() else "all feature regime fits failed",
                        "environment_variables": list(selected_variables),
                        "scores": model_rows.to_dict(orient="records"),
                    },
                )
            completed += 1
            if completed % 2 == 0 or completed == total:
                print(f"[Models] {completed}/{total} outcome/exposure pairs evaluated.", flush=True)

    spatial_frame = pd.concat(spatial_scores, ignore_index=True) if spatial_scores else pd.DataFrame()
    full_frame_scores = pd.concat(full_scores, ignore_index=True) if full_scores else pd.DataFrame()
    stage2_frame_scores = pd.concat(stage2_scores, ignore_index=True) if stage2_scores else pd.DataFrame()
    feature_frame = pd.concat(feature_scores, ignore_index=True) if feature_scores else pd.DataFrame()
    full_curve_exposure = "Aa_20yr" if "Aa_20yr" in config["exposures"] else str(config["exposures"][0])
    curve_path = Path(paths["models"]) / "selected_full_sample_curves.parquet"
    existing_curves = pd.read_parquet(curve_path) if curve_path.is_file() else pd.DataFrame()
    full_sample_curves = _complete_full_sample_curves(
        frame,
        full_frame_scores,
        existing_curves,
        exposure=full_curve_exposure,
    )
    _write_parquet(_json_columns(stage2_frame_scores), paths["models"] / "stage2_model_search.parquet")
    _write_parquet(_json_columns(spatial_frame), paths["models"] / "spatial_model_search.parquet")
    _write_parquet(_json_columns(full_frame_scores), paths["models"] / "full_sample_model_search.parquet")
    _write_parquet(_json_columns(full_sample_curves), paths["models"] / "selected_full_sample_curves.parquet")
    _write_parquet(_json_columns(feature_frame), paths["models"] / "feature_regime_comparison.parquet")
    if group_cache:
        group_columns = list(group_cache.values())
        _write_parquet(frame[["cell_id", "spatial_role", *group_columns]], paths["diagnostics"] / "environment_group_assignments.parquet")

    # Select model families and vectors by validation only; the test response is not opened yet.
    validation = spatial_frame.loc[
        spatial_frame["evaluation_set"].eq("validation")
        & spatial_frame["status"].eq("ok")
    ].copy()
    validation["r_squared"] = pd.to_numeric(validation["r_squared"], errors="coerce")
    validation = validation.loc[np.isfinite(validation["r_squared"])].copy()
    validation = validation.sort_values(
        ["exposure", "outcome", "r_squared", "model_name", "environment_vector"],
        ascending=[True, True, False, True, True],
        kind="stable",
    )
    validation_winners = validation.groupby(["exposure", "outcome"], sort=False).head(1).copy()
    validation_winners["selected_by"] = "validation_r2"
    _write_parquet(_json_columns(validation_winners), paths["tables"] / "selected_spatial_models.parquet")

    test_rows: list[dict[str, object]] = []
    for winner in validation_winners.to_dict(orient="records"):
        candidate_id = str(winner["candidate_id"])
        context = candidate_context[candidate_id]
        exposure, outcome = str(winner["exposure"]), str(winner["outcome"])
        family = str(winner["model_name"])
        group_for_model = context["group_column"] if family in {"environment_group_fixed_effects", "interactions"} else None
        existing_test = _existing_candidate_rows(
            registered_scores, "spatial_model_test", candidate_id
        )
        existing_test = existing_test.loc[
            existing_test.get("model_name", pd.Series(index=existing_test.index)).astype(str).eq(family)
        ] if not existing_test.empty else existing_test
        if not existing_test.empty:
            test_rows.extend(existing_test.to_dict(orient="records"))
            continue
        test_result = fit_selected_model_on_test(
            frame,
            outcome,
            exposure,
            context["environment_variables"],
            family,
            group=group_for_model,
        )
        test_result.update(
            candidate_id=candidate_id,
            experiment_type="spatial_model_test",
            environment_variables=list(context["environment_variables"]),
            environment_vector=context["environment_vector"],
            validation_r2=float(winner["r_squared"]),
            test_r2=float(test_result["r_squared"]) if np.isfinite(test_result["r_squared"]) else np.nan,
            selection_mode="validation_selected_then_test_evaluated",
            selection_objective="maximize_validation_r2_then_report_heldout_test_r2",
        )
        test_row = pd.DataFrame(
            [
                {
                    "outcome": outcome,
                    "exposure": exposure,
                    "model_name": family,
                    "evaluation_set": "test",
                    "score_type": "spatial_holdout",
                    "status": test_result["status"],
                    "n": test_result.get("n", test_result.get("n_test", 0)),
                    "r_squared": test_result.get("r_squared", np.nan),
                    "rmse": test_result.get("rmse", np.nan),
                    "mae": test_result.get("mae", np.nan),
                    "validation_r2": test_result["validation_r2"],
                    "test_r2": test_result["test_r2"],
                    "model_config": test_result.get("model_config", "{}"),
                    "coefficient": test_result.get("coefficient", np.nan),
                    "direction": test_result.get("direction", "unknown"),
                }
            ]
        )
        recorded, sequence = _record_model_scores(
            test_row,
            registry_path,
            exposure=exposure,
            outcome=outcome,
            candidate_id=candidate_id,
            experiment_type="spatial_model_test",
            environment_variables=context["environment_variables"],
            environment_vector=str(context["environment_vector"]),
            group_column=group_for_model,
            outcome_guided_selection=True,
            selection_mode="validation_selected_then_test_evaluated",
            selection_objective="maximize_validation_r2_then_report_heldout_test_r2",
            input_run_id=str(config["input_run_id"]),
            git_sha=git_sha,
            seed=int(config["seed"]),
            run_date=run_date,
            sequence_start=sequence,
        )
        recorded["validation_r2"] = test_result["validation_r2"]
        recorded["test_r2"] = test_result["test_r2"]
        test_rows.extend(recorded.to_dict(orient="records"))
        registered_scores = pd.concat([registered_scores, recorded], ignore_index=True, sort=False)
    test_frame = pd.DataFrame.from_records(test_rows)
    _write_parquet(_json_columns(test_frame), paths["models"] / "spatial_test_selected_models.parquet")

    common_environment = tuple(config["environment"].get("vectors", NAMED_ENVIRONMENT_SETS).get("V1", ()))
    common_environment = tuple(name for name in common_environment if name in frame)
    if len(config["exposures"]) >= 2 and common_environment:
        common_comparison = _compare_exposures_on_common_rows(
            frame,
            outcomes=config["outcomes"],
            exposures=config["exposures"],
            environment=common_environment,
        )
        common_records: list[pd.DataFrame] = []
        for (exposure, outcome), comparison in common_comparison.groupby(
            ["exposure", "outcome"], sort=False
        ):
            candidate_id = f"common_sample_{exposure}_{outcome}_V1"
            existing_common = _existing_candidate_rows(
                registered_scores, "common_sample_exposure_comparison", candidate_id
            )
            if not existing_common.empty:
                common_records.append(existing_common)
                continue
            identified, sequence = _record_model_scores(
                comparison,
                registry_path,
                exposure=str(exposure),
                outcome=str(outcome),
                candidate_id=candidate_id,
                experiment_type="common_sample_exposure_comparison",
                environment_variables=common_environment,
                environment_vector="V1_common",
                group_column=None,
                outcome_guided_selection=False,
                selection_mode="predeclared_common_sample_comparison",
                selection_objective="same_sample_same_ols_for_both_exposures",
                input_run_id=str(config["input_run_id"]),
                git_sha=git_sha,
                seed=int(config["seed"]),
                run_date=run_date,
                sequence_start=sequence,
            )
            common_records.append(identified)
            registered_scores = pd.concat([registered_scores, identified], ignore_index=True, sort=False)
        common_comparison = pd.concat(common_records, ignore_index=True) if common_records else common_comparison
        _write_parquet(
            _json_columns(common_comparison),
            paths["models"] / "common_sample_exposure_comparison.parquet",
        )
    else:
        common_comparison = pd.DataFrame()

    return {
        "stage2_model_scores": stage2_frame_scores,
        "spatial_scores": spatial_frame,
        "full_sample_scores": full_frame_scores,
        "selected_full_sample_curves": full_sample_curves,
        "feature_regime_scores": feature_frame,
        "common_sample_exposure_scores": common_comparison,
        "validation_model_winners": validation_winners,
        "test_model_scores": test_frame,
        "group_columns": {key: value for key, value in group_cache.items()},
    }


def _subgroup_definitions(
    frame: pd.DataFrame,
    exposure: str,
    group_column: str | None,
    *,
    minimum_group_size: int,
) -> dict[str, dict[str, Any]]:
    definitions: dict[str, dict[str, Any]] = {}
    soil = pd.to_numeric(frame["SoilType"], errors="coerce")
    for code in PRIMARY_SOIL_CODES:
        if soil.eq(code).any():
            definitions[f"soil_{code}"] = {"column": "SoilType", "op": "eq", "value": code}
    numeric_exposure = pd.to_numeric(frame[exposure], errors="coerce")
    definitions["exposure_zero"] = {"column": exposure, "op": "eq", "value": 0}
    definitions["exposure_positive"] = {"column": exposure, "op": "gt", "value": 0}
    positive = numeric_exposure.loc[np.isfinite(numeric_exposure) & numeric_exposure.gt(0)]
    quantiles = np.unique(positive.quantile([0.25, 0.5, 0.75]).to_numpy(dtype=float)) if len(positive) else np.asarray([])
    boundaries = [float(positive.min())] if len(positive) else []
    boundaries.extend(float(value) for value in quantiles)
    if len(positive):
        boundaries.append(float(positive.max()))
    boundaries = sorted(set(boundaries))
    for index in range(max(0, len(boundaries) - 1)):
        lower, upper = boundaries[index], boundaries[index + 1]
        if index == len(boundaries) - 2:
            definitions[f"positive_exposure_q{index + 1}"] = {
                "all": [
                    {"column": exposure, "op": "ge", "value": lower},
                    {"column": exposure, "op": "le", "value": upper},
                ]
            }
        else:
            definitions[f"positive_exposure_q{index + 1}"] = {
                "all": [
                    {"column": exposure, "op": "ge", "value": lower},
                    {"column": exposure, "op": "lt", "value": upper},
                ]
            }

    for variable in ("Elevation", "TWI", "LST", "NDWI", "NDBSI", "EVI", "NDVI", "Precipitation"):
        if variable not in frame:
            continue
        values = pd.to_numeric(frame[variable], errors="coerce")
        cuts = np.unique(values.loc[np.isfinite(values)].quantile([0.25, 0.5, 0.75]).to_numpy(dtype=float))
        if not len(cuts):
            continue
        definitions[f"{variable}_q1"] = {"column": variable, "op": "le", "value": float(cuts[0])}
        for index, boundary in enumerate(cuts[1:], start=2):
            lower = float(cuts[index - 2])
            definitions[f"{variable}_q{index}"] = {
                "all": [
                    {"column": variable, "op": "gt", "value": lower},
                    {"column": variable, "op": "le", "value": float(boundary)},
                ]
            }
        definitions[f"{variable}_q4"] = {"column": variable, "op": "gt", "value": float(cuts[-1])}

    if group_column and group_column in frame:
        counts = frame[group_column].value_counts(dropna=True)
        common_groups = [value for value, count in counts.items() if int(count) >= minimum_group_size]
        common_groups = sorted(common_groups, key=lambda value: (-int(counts.loc[value]), str(value)))[:20]
        for group_value in common_groups:
            label = str(group_value).replace(".", "_").replace("-", "m")
            definitions[f"environment_group_{label}"] = {
                "column": group_column,
                "op": "eq",
                "value": group_value,
            }
        for size, combination_name in ((2, "pair"), (3, "triple")):
            for group_values in combinations(common_groups[:6], size):
                labels = "_".join(str(value).replace(".", "_").replace("-", "m") for value in group_values)
                definitions[f"environment_group_{combination_name}_{labels}"] = {
                    "column": group_column,
                    "op": "in",
                    "values": list(group_values),
                }
        for soil_code in PRIMARY_SOIL_CODES:
            for group_value in common_groups[:3]:
                label = str(group_value).replace(".", "_").replace("-", "m")
                definitions[f"soil_{soil_code}_environment_group_{label}"] = {
                    "all": [
                        {"column": "SoilType", "op": "eq", "value": int(soil_code)},
                        {"column": group_column, "op": "eq", "value": group_value},
                    ]
                }
            groups_in_soil_counts = frame.loc[soil.eq(soil_code), group_column].value_counts(dropna=True)
            groups_in_soil = [value for value, count in groups_in_soil_counts.items() if int(count) >= minimum_group_size]
            groups_in_soil = sorted(
                groups_in_soil,
                key=lambda value: (-int(groups_in_soil_counts.loc[value]), str(value)),
            )[:4]
            for size, combination_name in ((2, "pair"), (3, "triple")):
                for group_values in combinations(groups_in_soil, size):
                    labels = "_".join(str(value).replace(".", "_").replace("-", "m") for value in group_values)
                    definitions[f"soil_{soil_code}_environment_group_{combination_name}_{labels}"] = {
                        "all": [
                            {"column": "SoilType", "op": "eq", "value": int(soil_code)},
                            {"column": group_column, "op": "in", "values": list(group_values)},
                        ]
                    }
    return definitions


def _record_subgroup_failure(
    registry_path: Path,
    registry_rows: pd.DataFrame,
    existing: pd.DataFrame,
    sequence: int,
    run_date: str,
    git_sha: str,
    input_run_id: str,
    exposure: str,
    outcome: str,
    candidate_id: str,
    definition: Mapping[str, Any],
    subgroup_cells: int,
    complete_cases: int,
    error: str,
    seed: int,
) -> tuple[int, str]:
    """Record a failed subgroup trial once and return its stable identifier."""

    if existing.empty:
        sequence += 1
        experiment_id = make_experiment_id(
            run_date,
            sequence,
            re.sub(r"[^A-Za-z0-9-]+", "-", candidate_id).strip("-")[:80],
        )
        record_experiment(
            registry_path,
            {
                "experiment_id": experiment_id,
                "candidate_id": candidate_id,
                "experiment_type": "oracle_subgroup_search",
                "git_sha": git_sha,
                "input_run_id": input_run_id,
                "exposure": exposure,
                "health_indicator": outcome,
                "outcome_guided_selection": True,
                "selection_mode": "outcome_selected_oracle_subgroup",
                "selection_outcomes": [outcome],
                "selection_objective": "maximize_apparent_r2_across_explicit_subgroups",
                "random_seed": seed,
                "status": "failed",
                "error": error,
                "subgroup_definition": definition,
                "n_cells": subgroup_cells,
                "n_complete_cases": complete_cases,
                "scores": [],
            },
        )
    else:
        experiment_ids = existing.get("experiment_id", pd.Series(dtype=str)).dropna().astype(str).unique()
        sequence += len(experiment_ids)
        experiment_id = experiment_ids[0] if len(experiment_ids) else make_experiment_id(
            run_date,
            sequence,
            re.sub(r"[^A-Za-z0-9-]+", "-", candidate_id).strip("-")[:80],
        )
    return sequence, experiment_id


def _failed_subgroup_frame(
    outcome: str,
    exposure: str,
    candidate_id: str,
    experiment_id: str,
    subgroup_name: str,
    definition: Mapping[str, Any],
    subgroup_cells: int,
    complete_cases: int,
    denominator: int,
    error: str,
) -> pd.DataFrame:
    """Return a reportable score row for a failed or empty subgroup trial."""

    return pd.DataFrame.from_records(
        [
            {
                "outcome": outcome,
                "exposure": exposure,
                "candidate_id": candidate_id,
                "experiment_id": experiment_id,
                "experiment_type": "oracle_subgroup_search",
                "model_name": "ols",
                "evaluation_set": "apparent",
                "status": "failed",
                "error": error,
                "n": 0,
                "n_complete_cases": complete_cases,
                "n_subgroup_cells": subgroup_cells,
                "r_squared": np.nan,
                "coefficient": np.nan,
                "direction": "unknown",
                "subgroup_name": subgroup_name,
                "subgroup_definition": definition,
                "sample_retention_ratio": 0.0 if denominator else np.nan,
            }
        ]
    )


def _run_oracle_subgroups(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
    paths: Mapping[str, Path],
    registry_path: Path,
    matching: Mapping[str, Any],
    regression: Mapping[str, Any],
    run_date: str,
    git_sha: str,
) -> dict[str, pd.DataFrame]:
    retention_records: list[pd.DataFrame] = []
    subgroup_records: list[pd.DataFrame] = []
    definitions_log: dict[str, Any] = {}
    sequence = 0
    registry_rows = _load_registry(registry_path) if Path(registry_path).is_file() else pd.DataFrame()
    candidate_lookup = {
        (str(exposure), str(candidate["candidate_id"])): dict(candidate)
        for exposure, candidates in matching.get("stage1_candidates", {}).items()
        for candidate in candidates
    }
    group_map = regression.get("group_columns", {})
    min_group_size = int(config.get("minimum_subgroup_cells", 100))
    fractions = list(config.get("retention_fractions", [1.0, 0.75, 0.5, 0.3, 0.2, 0.1, 0.05]))
    retention_trials = int(config.get("retention_search_trials", 20))

    print("[Oracle subsets] Searching explicit subgroups and sample-retention envelopes.", flush=True)
    for exposure in config["exposures"]:
        for outcome in config["outcomes"]:
            parent_id = matching.get("validation_winners", {}).get((exposure, outcome))
            parent = candidate_lookup.get((exposure, parent_id)) if parent_id else None
            variables = tuple(parent["environment_variables"]) if parent else tuple(NAMED_ENVIRONMENT_SETS["V1"])
            variables = tuple(name for name in variables if name in frame)
            if not variables:
                continue
            group_column = group_map.get(variables)
            predictor_sets: list[tuple[str, list[str]]] = [("exposure_only", [exposure])]
            combined = list(dict.fromkeys([exposure, *variables]))
            if len(combined) > 1:
                predictor_sets.append(("environment_plus_exposure", combined))

            for predictor_name, predictors in predictor_sets:
                selection_seed = int(config["seed"]) + sequence
                retention_error: str | None = None
                try:
                    retention = oracle_retention_search(
                        frame,
                        outcome,
                        predictors,
                        fractions,
                        trials=retention_trials,
                        seed=selection_seed,
                    )
                except ValueError as exc:
                    if "complete cases" not in str(exc).lower():
                        raise
                    retention_error = f"{type(exc).__name__}: {exc}"
                    required = [outcome, *predictors]
                    numeric = frame.loc[:, required].apply(pd.to_numeric, errors="coerce")
                    n_complete = int(np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1).sum())
                    retention = pd.DataFrame.from_records(
                        {
                            "outcome": outcome,
                            "predictors": predictors,
                            "target_retention_fraction": float(fraction),
                            "sample_retention_ratio": 0.0 if n_complete else np.nan,
                            "n_complete_cases": n_complete,
                            "n_selected": 0,
                            "apparent_r2": np.nan,
                            "full_sample_apparent_r2": np.nan,
                            "coefficients": [np.nan] * (len(predictors) + 1),
                            "coefficient": np.nan,
                            "direction": "unknown",
                            "selection_mode": "outcome_selected_oracle",
                            "outcome_guided_selection": True,
                            "selection_objective": "maximize_apparent_ols_r2_at_fixed_retention",
                            "selection_rule": "insufficient_complete_cases",
                            "selection_trial": 0,
                            "trial_count": 0,
                            "selection_seed": selection_seed,
                            "selected_cell_ids_json": "[]",
                            "test_r2": np.nan,
                            "error": retention_error,
                        }
                        for fraction in fractions
                    )
                retention["exposure"] = exposure
                retention["predictor_set"] = predictor_name
                retention["experiment_type"] = "oracle_retention_search"
                retention["selection_mode"] = "outcome_selected_oracle"
                retention["sample_scope"] = "all_cropland_complete_cases"
                retention["outcome_guided_selection"] = True
                for index, row in retention.iterrows():
                    sequence += 1
                    fraction = float(row["target_retention_fraction"])
                    candidate_id = f"retention_{exposure}_{outcome}_{predictor_name}_{index + 1:02d}"
                    existing = _existing_candidate_rows(
                        registry_rows, "oracle_retention_search", candidate_id
                    )
                    experiment_id = (
                        str(existing.iloc[0]["experiment_id"])
                        if not existing.empty and "experiment_id" in existing
                        else make_experiment_id(
                            run_date,
                            sequence,
                            re.sub(r"[^A-Za-z0-9-]+", "-", candidate_id).strip("-")[:80],
                        )
                    )
                    retention.loc[index, "candidate_id"] = candidate_id
                    retention.loc[index, "experiment_id"] = experiment_id
                    metrics = {
                        "evaluation_set": "apparent",
                        "score_type": "outcome_selected_apparent",
                        "n": int(row["n_selected"]),
                        "r_squared": float(row["apparent_r2"]) if np.isfinite(row["apparent_r2"]) else None,
                        "coefficient": float(row["coefficient"]) if np.isfinite(row["coefficient"]) else None,
                        "direction": str(row["direction"]),
                        "sample_retention_ratio": float(row["sample_retention_ratio"]),
                        "selection_rule": row["selection_rule"],
                        "selection_trial": int(row["selection_trial"]),
                    }
                    status = "success" if np.isfinite(row["apparent_r2"]) else "failed"
                    error = row.get("error") or (
                        None if status == "success" else "response or predictor variation was insufficient"
                    )
                    if existing.empty:
                        record_experiment(
                            registry_path,
                            {
                                "experiment_id": experiment_id,
                                "candidate_id": candidate_id,
                                "experiment_type": "oracle_retention_search",
                                "git_sha": git_sha,
                                "input_run_id": str(config["input_run_id"]),
                                "exposure": exposure,
                                "health_indicator": outcome,
                                "outcome_guided_selection": True,
                                "selection_mode": "outcome_selected_oracle",
                                "selection_outcomes": [outcome],
                                "selection_objective": "maximize_apparent_ols_r2_at_fixed_retention",
                                "random_seed": int(row["selection_seed"]),
                                "status": status,
                                "error": error,
                                "predictors": predictors,
                                "target_retention_fraction": fraction,
                                "sample_retention_ratio": float(row["sample_retention_ratio"]),
                                "n_cells": int(row["n_selected"]),
                                "n_complete_cases": int(row["n_complete_cases"]),
                                "selection_rule": row["selection_rule"],
                                "trial_count": int(row["trial_count"]),
                                "scores": [metrics],
                            },
                        )
                    else:
                        sequence += max(0, existing["experiment_id"].nunique() - 1)
                retention_records.append(retention)

            definitions = _subgroup_definitions(
                frame,
                exposure,
                group_column,
                minimum_group_size=min_group_size,
            )
            key = f"{exposure}/{outcome}"
            definitions_log[key] = definitions
            masks = build_subgroup_masks(frame, definitions)
            valid_numeric = frame[[outcome, exposure, *variables]].apply(pd.to_numeric, errors="coerce")
            denominator = int(np.isfinite(valid_numeric.to_numpy(dtype=float)).all(axis=1).sum())
            base = frame.drop(columns=["spatial_role"], errors="ignore")
            for subgroup_name, mask in masks.items():
                subgroup = base.loc[mask].copy()
                subgroup_cells = int(len(subgroup))
                candidate_id = f"subgroup_{exposure}_{outcome}_{subgroup_name}"
                existing = _existing_candidate_rows(
                    registry_rows, "oracle_subgroup_search", candidate_id
                )
                subgroup_numeric = subgroup[[outcome, exposure, *variables]].apply(pd.to_numeric, errors="coerce")
                complete_cases = int(np.isfinite(subgroup_numeric.to_numpy(dtype=float)).all(axis=1).sum())
                if subgroup_cells < min_group_size:
                    error = f"subgroup has {subgroup_cells} cells, below minimum_subgroup_cells={min_group_size}"
                    sequence, experiment_id = _record_subgroup_failure(
                        registry_path,
                        registry_rows,
                        existing,
                        sequence,
                        run_date,
                        git_sha,
                        str(config["input_run_id"]),
                        exposure,
                        outcome,
                        candidate_id,
                        definitions[subgroup_name],
                        subgroup_cells,
                        complete_cases,
                        error,
                        int(config["seed"]),
                    )
                    subgroup_records.append(
                        _failed_subgroup_frame(
                            outcome, exposure, candidate_id, experiment_id, subgroup_name,
                            definitions[subgroup_name], subgroup_cells, complete_cases,
                            denominator, error,
                        )
                    )
                    continue
                try:
                    model_scores = fit_candidate_models(
                        subgroup,
                        outcome,
                        exposure,
                        variables,
                        model_families=["ols"],
                        outcome_guided_selection=True,
                    )
                except ValueError as exc:
                    if "complete cases" not in str(exc).lower():
                        raise
                    error = f"{type(exc).__name__}: {exc}"
                    sequence, experiment_id = _record_subgroup_failure(
                        registry_path,
                        registry_rows,
                        existing,
                        sequence,
                        run_date,
                        git_sha,
                        str(config["input_run_id"]),
                        exposure,
                        outcome,
                        candidate_id,
                        definitions[subgroup_name],
                        subgroup_cells,
                        complete_cases,
                        error,
                        int(config["seed"]),
                    )
                    subgroup_records.append(
                        _failed_subgroup_frame(
                            outcome, exposure, candidate_id, experiment_id, subgroup_name,
                            definitions[subgroup_name], subgroup_cells, complete_cases,
                            denominator, error,
                        )
                    )
                    continue
                model_scores["subgroup_definition"] = [definitions[subgroup_name]] * len(model_scores)
                model_scores["subgroup_name"] = subgroup_name
                model_scores["n_subgroup_cells"] = subgroup_cells
                model_scores["sample_retention_ratio"] = (
                    pd.to_numeric(model_scores["n"], errors="coerce") / denominator if denominator else np.nan
                )
                model_scores["experiment_type"] = "oracle_subgroup_search"
                if existing.empty:
                    identified, sequence = _record_model_scores(
                        model_scores,
                        registry_path,
                        exposure=exposure,
                        outcome=outcome,
                        candidate_id=candidate_id,
                        experiment_type="oracle_subgroup_search",
                        environment_variables=variables,
                        environment_vector=str(parent["vector_name"]) if parent else "V1_fallback",
                        group_column=group_column,
                        outcome_guided_selection=True,
                        selection_mode="outcome_selected_oracle_subgroup",
                        selection_objective="maximize_apparent_r2_across_explicit_subgroups",
                        input_run_id=str(config["input_run_id"]),
                        git_sha=git_sha,
                        seed=int(config["seed"]),
                        run_date=run_date,
                        sequence_start=sequence,
                    )
                else:
                    sequence += int(existing["experiment_id"].nunique())
                    identified = model_scores.copy()
                    identifiers = existing.drop_duplicates("model_name").set_index("model_name")["experiment_id"]
                    identified["experiment_id"] = identified["model_name"].map(identifiers)
                    identified["candidate_id"] = candidate_id
                    identified["experiment_type"] = "oracle_subgroup_search"
                    identified["exposure"] = exposure
                    identified["outcome"] = outcome
                identified["subgroup_name"] = subgroup_name
                identified["subgroup_definition"] = [definitions[subgroup_name]] * len(identified)
                identified["n_subgroup_cells"] = subgroup_cells
                identified["sample_retention_ratio"] = pd.to_numeric(identified["n"], errors="coerce") / denominator if denominator else np.nan
                subgroup_records.append(identified)
            if len(subgroup_records) % 2 == 0:
                print(f"[Oracle subsets] Completed subgroup suite for {exposure}/{outcome}.", flush=True)

    retention_frame = pd.concat(retention_records, ignore_index=True) if retention_records else pd.DataFrame()
    subgroup_frame = pd.concat(subgroup_records, ignore_index=True) if subgroup_records else pd.DataFrame()
    _write_parquet(_json_columns(retention_frame), paths["subgroups"] / "oracle_retention.parquet")
    _write_parquet(_json_columns(subgroup_frame), paths["subgroups"] / "subgroup_scores.parquet")
    (paths["subgroups"] / "subgroup_definitions.json").write_text(
        json.dumps(definitions_log, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return {"oracle_retention": retention_frame, "subgroup_scores": subgroup_frame}


def _compare_exposures_on_common_rows(
    frame: pd.DataFrame,
    *,
    outcomes: Sequence[str],
    exposures: Sequence[str],
    environment: Sequence[str],
) -> pd.DataFrame:
    """Compare erosion time scales on identical cells, covariates, and OLS specs."""

    if len(exposures) < 2:
        raise ValueError("At least two exposures are required for a common-sample comparison")
    missing = sorted(set([*outcomes, *exposures, *environment]) - set(frame.columns))
    if missing:
        raise ValueError(f"Common-sample comparison is missing columns: {missing}")
    records: list[pd.DataFrame] = []
    for outcome in outcomes:
        numeric_columns = list(dict.fromkeys([outcome, *exposures, *environment]))
        numeric = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
        complete = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
        for exposure in exposures:
            complete &= numeric[exposure].to_numpy(dtype=float) >= 0
        common = frame.loc[complete].copy()
        for column in numeric_columns:
            common[column] = numeric.loc[complete, column].to_numpy(dtype=float)
        if common.empty:
            continue
        common["__common_sample_group"] = 0
        for exposure in exposures:
            comparison = compare_feature_regimes(
                common,
                outcome,
                exposure,
                environment,
                group="__common_sample_group",
                model_families=["ols"],
            )
            comparison = comparison.loc[
                comparison["regime"].isin(
                    {"exposure_only", "environment_only", "environment_plus_exposure"}
                )
            ].copy()
            comparison["common_sample_n"] = len(common)
            comparison["environment_variables"] = [list(environment)] * len(comparison)
            comparison["comparison_scope"] = "common_complete_cases_for_both_exposures"
            comparison["experiment_type"] = "common_sample_exposure_comparison"
            comparison["selection_mode"] = "predeclared_common_sample_comparison"
            comparison["selection_objective"] = "same_sample_same_ols_for_both_exposures"
            comparison["outcome_guided_selection"] = False
            records.append(comparison)
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def _fmt(value: object, digits: int = 4) -> str:
    if value is None or (isinstance(value, (float, np.floating)) and not np.isfinite(value)):
        return "n/a"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def _format_linear_equation(model_config: object, outcome: str) -> str | None:
    """Return a readable equation for a saved OLS model configuration."""

    if isinstance(model_config, Mapping):
        config = model_config
    elif isinstance(model_config, str):
        try:
            parsed = json.loads(model_config)
        except (TypeError, json.JSONDecodeError):
            return None
        if not isinstance(parsed, Mapping):
            return None
        config = parsed
    else:
        return None
    if config.get("model_family") != "ols":
        return None
    features = config.get("numeric_features", [])
    coefficients = config.get("coefficients", [])
    if not isinstance(features, list) or not isinstance(coefficients, list) or len(features) != len(coefficients):
        return None
    try:
        intercept = float(config.get("intercept", 0.0)) if config.get("fit_intercept", True) else 0.0
        terms = [(str(feature), float(coefficient)) for feature, coefficient in zip(features, coefficients, strict=True)]
    except (TypeError, ValueError):
        return None
    if not np.isfinite(intercept) or not all(np.isfinite(value) for _, value in terms):
        return None

    equation = f"{outcome} = {intercept:.6g}"
    for feature, coefficient in terms:
        operator = "+" if coefficient >= 0 else "-"
        equation += f" {operator} {abs(coefficient):.6g} × {feature}"
    return equation


def _markdown_table(columns: Sequence[str], rows: Sequence[Mapping[str, object]]) -> str:
    if not rows:
        return "_No rows available._"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(column)) for column in columns) + " |")
    return "\n".join(lines)


def _read_board(results_dir: Path, name: str) -> pd.DataFrame:
    path = results_dir / "tables" / f"leaderboard_{name}.parquet"
    return pd.read_parquet(path) if path.is_file() else pd.DataFrame()


def write_review_report(results_dir: Path, report_path: Path) -> Path:
    """Write the required review report from completed run tables and metadata."""

    results_dir, report_path = Path(results_dir), Path(report_path)
    summary_path = results_dir / "tables" / "experiment_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    config_path = results_dir / "search_config.yaml"
    search_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    search_config = search_config if isinstance(search_config, Mapping) else {}
    matching_config = search_config.get("matching", {})
    matching_config = matching_config if isinstance(matching_config, Mapping) else {}
    distance_methods = matching_config.get("distance_methods", [])
    distance_methods = distance_methods if isinstance(distance_methods, list) else []
    matching_domain_counts = {
        "environment calipers": len(matching_config.get("environment_calipers", [])),
        "geographic radii": len(matching_config.get("geographic_radii_m", [])),
        "erosion rank gaps": len(matching_config.get("erosion_rank_gaps", [])),
        "cluster methods": len(matching_config.get("cluster_methods", [])),
        "exposure rules": len(matching_config.get("exposure_rules", [])),
    }
    stage1_counts = summary.get("stage1_counts", {})
    stage1_counts = stage1_counts if isinstance(stage1_counts, Mapping) else {}
    stage2_counts = summary.get("stage2_config_counts", {})
    stage2_counts = stage2_counts if isinstance(stage2_counts, Mapping) else {}
    stage1_candidate_count = sum(int(value) for value in stage1_counts.values())
    stage2_candidate_count = sum(int(value) for value in stage2_counts.values())
    domain_text = ", ".join(f"{name}={count} levels" for name, count in matching_domain_counts.items())
    full_board = _read_board(results_dir, "full_sample_best")
    oracle_board = _read_board(results_dir, "oracle_apparent_best")
    test_board = _read_board(results_dir, "spatial_test_best")
    delta_board = _read_board(results_dir, "erosion_increment_best")
    matching_winner_path = results_dir / "matching" / "stage2" / "validation_winners.parquet"
    matching_winners = pd.read_parquet(matching_winner_path) if matching_winner_path.is_file() else pd.DataFrame()
    common_comparison_path = results_dir / "models" / "common_sample_exposure_comparison.parquet"
    common_comparison = pd.read_parquet(common_comparison_path) if common_comparison_path.is_file() else pd.DataFrame()
    retention_path = results_dir / "subgroups" / "oracle_retention.parquet"
    retention = pd.read_parquet(retention_path) if retention_path.is_file() else pd.DataFrame()
    subgroup_path = results_dir / "subgroups" / "subgroup_scores.parquet"
    subgroups = pd.read_parquet(subgroup_path) if subgroup_path.is_file() else pd.DataFrame()
    figure_index = results_dir / "figures" / "figure_index.json"
    figures = json.loads(figure_index.read_text(encoding="utf-8")) if figure_index.is_file() else []

    full_winners: list[dict[str, object]] = []
    oracle_winners: list[dict[str, object]] = []
    spatial_winners: list[dict[str, object]] = []
    for exposure in ("Aa_20yr", "Aa_2020"):
        for outcome in ("SHI_candidate", *HEALTH_OUTCOMES):
            for source, destination, metric in (
                (full_board, full_winners, "r_squared"),
                (oracle_board, oracle_winners, "r_squared"),
                (test_board, spatial_winners, "test_r2"),
            ):
                if source.empty or "outcome" not in source:
                    continue
                exposure_values = source["exposure"] if "exposure" in source else pd.Series(index=source.index)
                subset = source.loc[source["outcome"].eq(outcome) & exposure_values.eq(exposure)]
                if subset.empty:
                    continue
                score_column = metric if metric in subset and subset[metric].notna().any() else "r_squared"
                values = pd.to_numeric(subset[score_column], errors="coerce")
                if np.isfinite(values).any():
                    best = subset.loc[values.idxmax()].to_dict()
                    best["reported_metric"] = float(values.max())
                    destination.append(best)

    full_shi = full_board.loc[full_board["outcome"].eq("SHI_candidate")] if not full_board.empty and "outcome" in full_board else pd.DataFrame()
    oracle_shi = oracle_board.loc[oracle_board["outcome"].eq("SHI_candidate")] if not oracle_board.empty and "outcome" in oracle_board else pd.DataFrame()
    spatial_shi = test_board.loc[test_board["outcome"].eq("SHI_candidate")] if not test_board.empty and "outcome" in test_board else pd.DataFrame()
    best_full_shi = full_shi.sort_values("r_squared", ascending=False).iloc[0].to_dict() if not full_shi.empty else {}
    best_oracle_shi = oracle_shi.sort_values("r_squared", ascending=False).iloc[0].to_dict() if not oracle_shi.empty else {}
    spatial_metric = "test_r2" if not spatial_shi.empty and "test_r2" in spatial_shi else "r_squared"
    best_spatial_shi = spatial_shi.sort_values(spatial_metric, ascending=False).iloc[0].to_dict() if not spatial_shi.empty else {}
    spatial_shi_equation = _format_linear_equation(best_spatial_shi.get("model_config"), "SHI_candidate")

    if not retention.empty and "apparent_r2" in retention:
        retention["apparent_r2"] = pd.to_numeric(retention["apparent_r2"], errors="coerce")
        valid_retention = retention.loc[np.isfinite(retention["apparent_r2"])]
        best_retention = valid_retention.sort_values("apparent_r2", ascending=False).iloc[0].to_dict() if not valid_retention.empty else {}
    else:
        best_retention = {}
    retention_delta_r2 = (
        float(best_retention["apparent_r2"]) - float(best_retention["full_sample_apparent_r2"])
        if best_retention
        and pd.notna(best_retention.get("apparent_r2"))
        and pd.notna(best_retention.get("full_sample_apparent_r2"))
        else np.nan
    )
    if not subgroups.empty and "r_squared" in subgroups:
        subgroups["r_squared"] = pd.to_numeric(subgroups["r_squared"], errors="coerce")
        valid_subgroups = subgroups.loc[np.isfinite(subgroups["r_squared"])]
        if "scope_status" in valid_subgroups:
            valid_subgroups = valid_subgroups.loc[valid_subgroups["scope_status"].eq("in_scope")]
        best_subgroup = valid_subgroups.sort_values("r_squared", ascending=False).iloc[0].to_dict() if not valid_subgroups.empty else {}
    else:
        best_subgroup = {}
    if not delta_board.empty and "delta_r2" in delta_board:
        delta_board["delta_r2"] = pd.to_numeric(delta_board["delta_r2"], errors="coerce")
        valid_delta = delta_board.loc[np.isfinite(delta_board["delta_r2"])]
        best_delta = valid_delta.sort_values("delta_r2", ascending=False).iloc[0].to_dict() if not valid_delta.empty else {}
    else:
        best_delta = {}

    shi_matching = (
        matching_winners.loc[matching_winners["outcome"].eq("SHI_candidate")].copy()
        if not matching_winners.empty and "outcome" in matching_winners
        else pd.DataFrame()
    )
    best_match_shi = (
        shi_matching.sort_values("r_squared", ascending=False).iloc[0].to_dict()
        if not shi_matching.empty and "r_squared" in shi_matching
        else {}
    )

    full_table = _markdown_table(
        ["Exposure", "Outcome", "Model", "Vector", "R²", "n", "Evaluation"],
        [
            {
                "Exposure": row.get("exposure"),
                "Outcome": row.get("outcome"),
                "Model": row.get("model_name"),
                "Vector": row.get("environment_vector"),
                "R²": row.get("reported_metric", row.get("r_squared")),
                "n": row.get("n"),
                "Evaluation": row.get("evaluation_set"),
            }
            for row in full_winners
        ],
    )
    oracle_table = _markdown_table(
        ["Exposure", "Outcome", "Model", "Vector", "R²", "n", "Retention", "Selection"],
        [
            {
                "Exposure": row.get("exposure"),
                "Outcome": row.get("outcome"),
                "Model": row.get("model_name"),
                "Vector": row.get("vector_name", row.get("environment_vector")),
                "R²": row.get("reported_metric", row.get("r_squared")),
                "n": row.get("n", row.get("n_selected")),
                "Retention": row.get("sample_retention_ratio"),
                "Selection": row.get("selection_objective"),
            }
            for row in oracle_winners
        ],
    )
    spatial_table = _markdown_table(
        ["Exposure", "Outcome", "Model", "Vector", "Validation R²", "Test R²", "n test"],
        [
            {
                "Exposure": row.get("exposure"),
                "Outcome": row.get("outcome"),
                "Model": row.get("model_name"),
                "Vector": row.get("environment_vector"),
                "Validation R²": row.get("selected_validation_r2", row.get("validation_r2")),
                "Test R²": row.get("reported_metric", row.get("test_r2", row.get("r_squared"))),
                "n test": row.get("n"),
            }
            for row in spatial_winners
        ],
    )
    full_lookup = {(row.get("exposure"), row.get("outcome")): row for row in full_winners}
    oracle_lookup = {(row.get("exposure"), row.get("outcome")): row for row in oracle_winners}
    spatial_lookup = {(row.get("exposure"), row.get("outcome")): row for row in spatial_winners}
    indicator_rows: list[dict[str, object]] = []
    common_rows: list[dict[str, object]] = []
    for exposure in ("Aa_20yr", "Aa_2020"):
        for outcome in ("SHI_candidate", *HEALTH_OUTCOMES):
            key = (exposure, outcome)
            delta_rows = (
                delta_board.loc[
                    delta_board["exposure"].eq(exposure)
                    & delta_board["outcome"].eq(outcome)
                ]
                if not delta_board.empty and {"exposure", "outcome"}.issubset(delta_board.columns)
                else pd.DataFrame()
            )
            delta_row = delta_rows.iloc[0].to_dict() if not delta_rows.empty else {}
            full_row = full_lookup.get(key, {})
            oracle_row = oracle_lookup.get(key, {})
            spatial_row = spatial_lookup.get(key, {})
            indicator_rows.append(
                {
                    "Exposure": exposure,
                    "Outcome": outcome,
                    "Full R²": full_row.get("r_squared"),
                    "n full": full_row.get("n"),
                    "Oracle R²": oracle_row.get("r_squared"),
                    "n Oracle": oracle_row.get("n", oracle_row.get("n_selected")),
                    "Retention": oracle_row.get("sample_retention_ratio"),
                    "Validation R²": spatial_row.get("selected_validation_r2", spatial_row.get("validation_r2")),
                    "Test R²": spatial_row.get("test_r2", spatial_row.get("r_squared")),
                    "n test": spatial_row.get("n"),
                    "Validation ΔR²": delta_row.get("validation_delta_r2", delta_row.get("delta_r2")),
                    "Test ΔR²": delta_row.get("test_delta_r2"),
                }
            )

            if not common_comparison.empty:
                common = common_comparison.loc[
                    common_comparison["exposure"].eq(exposure)
                    & common_comparison["outcome"].eq(outcome)
                    & common_comparison["regime"].eq("environment_plus_exposure")
                ]
                apparent = common.loc[common["evaluation_set"].eq("apparent")]
                validation = common.loc[common["evaluation_set"].eq("validation")]
                test = common.loc[common["evaluation_set"].eq("test")]
                environment_only = common_comparison.loc[
                    common_comparison["exposure"].eq(exposure)
                    & common_comparison["outcome"].eq(outcome)
                    & common_comparison["regime"].eq("environment_only")
                    & common_comparison["evaluation_set"].eq("apparent")
                ]
                apparent_row = apparent.iloc[0] if not apparent.empty else {}
                validation_row = validation.iloc[0] if not validation.empty else {}
                test_row = test.iloc[0] if not test.empty else {}
                env_row = environment_only.iloc[0] if not environment_only.empty else {}
                common_rows.append(
                    {
                        "Exposure": exposure,
                        "Outcome": outcome,
                        "Common n": apparent_row.get("common_sample_n", apparent_row.get("n")),
                        "Environment R²": env_row.get("r_squared"),
                        "Environment + exposure R²": apparent_row.get("r_squared"),
                        "Apparent ΔR²": apparent_row.get("delta_r2"),
                        "Validation R²": validation_row.get("r_squared"),
                        "Validation ΔR²": validation_row.get("delta_r2"),
                        "Test R²": test_row.get("r_squared"),
                        "Test ΔR²": test_row.get("delta_r2"),
                        "n test": test_row.get("n"),
                    }
                )
    indicator_table = _markdown_table(
        ["Exposure", "Outcome", "Full R²", "n full", "Oracle R²", "n Oracle", "Retention", "Validation R²", "Test R²", "n test", "Validation ΔR²", "Test ΔR²"],
        indicator_rows,
    )
    common_exposure_table = _markdown_table(
        ["Exposure", "Outcome", "Common n", "Environment R²", "Environment + exposure R²", "Apparent ΔR²", "Validation R²", "Validation ΔR²", "Test R²", "Test ΔR²", "n test"],
        common_rows,
    )
    figure_lines = [
        f"- [{Path(name).name}](figures/{Path(name).name})"
        for name in figures
    ]
    embedded_figures = [
        f"![{Path(name).stem.replace('_', ' ')}](figures/{name})"
        for name in REVIEW_FIGURE_NAMES
        if (report_path.parent / "figures" / name).is_file()
    ]
    lines = [
        "# EnvMatch-SEH探索性上限实验：环境匹配与土壤健康响应的最佳表现",
        "",
        f"- Run: {results_dir.name}",
        f"- Input run: {summary.get('input_run_id', 'unknown')}",
        f"- Git SHA: {summary.get('git_sha', 'unknown')} (working tree dirty: {summary.get('git_dirty', 'unknown')})",
        "- Mode: EXPLORATORY_ORACLE; outcome-guided results are exploratory and not formal evidence.",
        "- Erosion inputs: reused existing Aa_20yr and Aa_2020 values; no RUSLE recalculation or verification was performed.",
        "- PCNM: DEFERRED.",
        "",
        "## 1. Research charter changes",
        "",
        "The charter was revised from V1.0 to V1.1. Formal analysis keeps the result-independent matching restriction and G4/G5 gates. Oracle analysis is separate, uses its own output root and identifiers, and permits outcome-guided matching, subgroup, and model selection when every trial and objective is recorded. The user explicitly authorized this internal, non-publication exploration. Other research questions were not changed.",
        "",
        "## 2. Experiment scale",
        "",
        f"- Source rows: {summary.get('source_rows', 'n/a')}; Stage 1 sample: {summary.get('stage1_sample_rows', 'n/a')}; Stage 2 sample: {summary.get('stage2_sample_rows', 'n/a')}.",
        f"- Matching search used {len(distance_methods)} distance methods ({', '.join(str(method) for method in distance_methods)}); {stage1_candidate_count:,} Stage 1 candidate parameter configurations ({stage1_counts}); {stage2_candidate_count:,} Stage 2 configurations ({stage2_counts}); and {summary.get('stage3_matching_test_count', 'n/a')} Stage 3 matching configurations scored on withheld spatial test blocks. Search domains: {domain_text}. Stage 1 sampled configurations; it did not enumerate the full Cartesian product.",
        f"- Stage 1 attempts: {summary.get('stage1_counts', {})}; failures remain in the registry.",
        f"- Stage 2 configurations per exposure: {summary.get('stage2_config_counts', {})}; role-specific attempts: {summary.get('stage2_role_trial_count', 'n/a')}.",
        f"- Stage 2 model family score rows: {summary.get('stage2_model_score_rows', 'n/a')}; common-sample time-scale comparison rows: {summary.get('common_sample_comparison_rows', 'n/a')} (SHI common-sample n={summary.get('common_sample_comparison_n', 'n/a')}; other outcomes have outcome-specific counts in Section 8).",
        f"- Selected Aa_20yr full-sample response-curve predictions: {summary.get('selected_full_sample_curve_rows', 'n/a')} rows.",
        f"- Stage 3 matching configurations evaluated on withheld test blocks: {summary.get('stage3_matching_test_count', 'n/a')}.",
        f"- Model, subgroup, and retention records: {summary.get('model_trial_count', 'n/a')}, {summary.get('subgroup_trial_count', 'n/a')}, and {summary.get('retention_trial_count', 'n/a')}; in-scope subgroup trials={summary.get('subgroup_trial_count_in_scope', 'n/a')}, with {summary.get('scope_excluded_subgroup_trial_count', 'n/a')} legacy SoilType=85 water-class trials excluded from result ranking and retained in the registry for audit.",
        f"- Regression comparison covered {len(summary.get('model_families', []))} regression model families across {_fmt(summary.get('model_trial_count'))} model trials: {', '.join(summary.get('model_families', []))}.",
        f"- Total registry records: {summary.get('registry_record_count', 'n/a')}; flattened score rows: {summary.get('experiment_result_rows', 'n/a')}.",
        "- All nine outcomes were represented when present. Missing responses use outcome-specific complete cases; no missing values were filled with zero.",
        "- Quantile Regression emitted statsmodels iteration-limit warnings (100 iterations), and Huber fits emitted scikit-learn LBFGS iteration-limit convergence warnings. These did not stop the run; the registry does not store warning flags, so treat those model-family fits cautiously.",
        "",
        "## 3. Best environment matching",
        "",
        f"Validation-selected SHI matching candidate: {best_match_shi.get('candidate_id', 'n/a')}; exposure {best_match_shi.get('exposure', 'n/a')}, validation R²={_fmt(best_match_shi.get('r_squared'))}, vector {best_match_shi.get('vector_name', 'n/a')}, distance {best_match_shi.get('distance_method', 'n/a')}, clusters {best_match_shi.get('cluster_method', 'n/a')}/{best_match_shi.get('cluster_count', 'n/a')}, environment caliper {best_match_shi.get('environment_caliper', 'n/a')}, radius {best_match_shi.get('geographic_radius_m', 'n/a')} m, rank gap {best_match_shi.get('erosion_rank_gap', 'n/a')}; pairs={_fmt(best_match_shi.get('n_pairs'))}, coverage={_fmt(best_match_shi.get('area_coverage'))}, max |SMD|={_fmt(best_match_shi.get('max_abs_matched_smd'))}, median exposure gap={_fmt(best_match_shi.get('median_erosion_difference'))}.",
        "",
        "## 4. Best SHI result",
        "",
        f"- Full-sample unfiltered model-search apparent R²: {_fmt(best_full_shi.get('r_squared'))}; n={_fmt(best_full_shi.get('n'))}; model={best_full_shi.get('model_name', 'n/a')} on {best_full_shi.get('exposure', 'n/a')} / {best_full_shi.get('environment_vector', 'n/a')}; exposure coefficient={_fmt(best_full_shi.get('coefficient'))} ({best_full_shi.get('direction', 'unknown')}); config={best_full_shi.get('model_config', 'n/a')}.",
        f"- Oracle-selected apparent R²: {_fmt(best_oracle_shi.get('r_squared'))}; n={_fmt(best_oracle_shi.get('n', best_oracle_shi.get('n_selected')))}; retention={_fmt(best_oracle_shi.get('sample_retention_ratio'))}; selection={best_oracle_shi.get('selection_objective', 'n/a')}; exposure coefficient={_fmt(best_oracle_shi.get('coefficient'))} ({best_oracle_shi.get('direction', 'unknown')}).",
        f"- Validation-selected independent spatial test R²: {_fmt(best_spatial_shi.get(spatial_metric))}; validation R²={_fmt(best_spatial_shi.get('selected_validation_r2', best_spatial_shi.get('validation_r2')))}; test n={_fmt(best_spatial_shi.get('n'))}; model={best_spatial_shi.get('model_name', 'n/a')} on {best_spatial_shi.get('exposure', 'n/a')} / {best_spatial_shi.get('environment_vector', 'n/a')}; exposure coefficient={_fmt(best_spatial_shi.get('coefficient'))} ({best_spatial_shi.get('direction', 'unknown')}); config={best_spatial_shi.get('model_config', 'n/a')}.",
        *([f"- Spatially selected OLS expression: `{spatial_shi_equation}`."] if spatial_shi_equation else []),
        f"- Largest same-row erosion increment ΔR² selected by {best_delta.get('selection_basis', 'n/a')}: validation={_fmt(best_delta.get('validation_delta_r2'))}; test={_fmt(best_delta.get('test_delta_r2'))}; exposure/outcome={best_delta.get('exposure', 'n/a')}/{best_delta.get('outcome', 'n/a')}; evaluation source={best_delta.get('evaluation_set', 'n/a')}.",
        "",
        "## 5. Best results for all health indicators",
        "",
        indicator_table,
        "",
        "## 6. Best selected data subset",
        "",
        f"Highest explicit subgroup apparent model: {best_subgroup.get('subgroup_name', 'n/a')} for {best_subgroup.get('exposure', 'n/a')}/{best_subgroup.get('outcome', 'n/a')}; definition={_fmt(best_subgroup.get('subgroup_definition'))}; R²={_fmt(best_subgroup.get('r_squared'))}, n={_fmt(best_subgroup.get('n'))}, retention={_fmt(best_subgroup.get('sample_retention_ratio'))}; coefficient={_fmt(best_subgroup.get('coefficient'))} ({best_subgroup.get('direction', 'unknown')}). This definition uses the environment group only; it does not additionally restrict SoilType or an erosion interval.",
        f"Highest fixed-fraction Oracle retention fit: {best_retention.get('exposure', 'n/a')}/{best_retention.get('outcome', 'n/a')}, predictor set={best_retention.get('predictor_set', 'n/a')}, target retention={_fmt(best_retention.get('target_retention_fraction'))}, actual retention={_fmt(best_retention.get('sample_retention_ratio'))}, n={_fmt(best_retention.get('n_selected'))} of {_fmt(best_retention.get('n_complete_cases'))} complete cases, apparent R²={_fmt(best_retention.get('apparent_r2'))}, baseline full-sample R²={_fmt(best_retention.get('full_sample_apparent_r2'))} on n={_fmt(best_retention.get('n_complete_cases'))} complete cases, apparent ΔR²={_fmt(retention_delta_r2)}, trials={_fmt(best_retention.get('trial_count'))}, selected by {best_retention.get('selection_rule', 'n/a')}, coefficient={_fmt(best_retention.get('coefficient'))} ({best_retention.get('direction', 'unknown')}). It selects lowest absolute full-fit residuals across cropland with no SoilType or erosion-interval restriction; this is a post-hoc apparent fit and the test set did not participate. The baseline is for the same predictor set and differs from the separate full-sample model-search winner in Section 4.",
        "",
        "## 7. Best full-sample models",
        "",
        full_table,
        "",
        "## 8. Aa_20yr and Aa_2020 comparison",
        "",
        "The table below compares Aa_20yr and Aa_2020 on the identical complete-case pixels for each outcome, using the same V1 environment variables and OLS specification. Separately optimized exposure configurations remain distinct and should not be read as an isolated time-scale effect.",
        "",
        common_exposure_table,
        "",
        "## 9. Upper-envelope answers",
        "",
        f"1. Unfiltered full-sample best observed SHI R²: {_fmt(best_full_shi.get('r_squared'))}. See the full-sample leaderboard for all indicators and exposures.",
        f"2. Outcome-selected Oracle apparent SHI R²: {_fmt(best_oracle_shi.get('r_squared'))}. The best fixed-retention fit retained {_fmt(best_retention.get('sample_retention_ratio'))} of complete cases ({_fmt(best_retention.get('n_selected'))} cells) at R² {_fmt(best_retention.get('apparent_r2'))}.",
        f"3. Independent spatial result: validation-selected SHI model test R² {_fmt(best_spatial_shi.get(spatial_metric))} on {_fmt(best_spatial_shi.get('n'))} test rows. Test outcomes were opened only to score selected models and did not select model families or matching configurations.",
        "",
        "Oracle apparent and spatial test results answer different questions. Outcome-selected subsets can strongly inflate apparent R² and do not measure independent predictive performance.",
        "",
        "## 10. Interpretation and next steps",
        "",
        "Health rasters are random-forest predictions. Regressing them on environmental variables used in their source prediction can reconstruct the prediction surface; that does not establish an independent erosion contribution. Use same-row ΔR² and spatial test tables to assess whether erosion adds stable information beyond environment. SHI remains labelled SHI_candidate because its source identity is provisional.",
        "",
        "For formal follow-up, prioritize configurations that retain validation-selected performance on held-out spatial blocks, report complete-case n and match coverage, and treat Oracle subgroup/R² results as internal hypotheses. No PCNM analysis was run.",
        "",
        "- Matching: carry direct nearest neighbor forward as a leading sensitivity candidate. It was the Aa_20yr/SHI validation winner (validation R²=0.1531, coverage=0.4762, max |SMD|=0.0078); keep coverage visible because fewer than half of eligible cells were matched.",
        "- Exposure and indicators: Aa_20yr has the stronger same-row spatial-test erosion increments for SHI_candidate (ΔR²=0.2993), OM (0.3591), and SW (0.3491). These are the most useful outcomes for a follow-up focused on incremental erosion information. Aa_2020 gives a higher CEC test R², but its CEC erosion increment is small (ΔR²=0.0275).",
        "- Models: the full-sample Random Forest gives the largest apparent SHI R², while the validation-selected spatial SHI OLS has validation R²=-0.0692 and test R²=0.4703. Keep the Random Forest as a descriptive surface fit and judge follow-up models with spatial validation and same-row ΔR²; the validation/test gap for SHI needs checking on new spatial splits.",
        "- Outcomes remain prediction surfaces, not independent field measurements. Reassess the promising Aa_20yr/OM/SW/SHI patterns against independent soil-health observations before making inferential claims; retain the provisional SHI_candidate label.",
        "",
        "## Spatial test scores",
        "",
        spatial_table,
        "",
        "## Figures",
        "",
        "Figure 08 overlays a natural spline fitted to the fixed stratified plotting sample and the full-sample best model refit for the partial effect, with environment variables held at their medians. The latter is selected by apparent full-sample R² and is descriptive, not an independent test result.",
        "",
        *(figure_lines or ["- No figures were generated."]),
        "",
        "## Selected figures",
        "",
        *(embedded_figures or ["_No compact review figures were copied._"]),
        "",
        "## Run artifacts",
        "",
        f"Experiment registry: outputs/exploratory_oracle/{results_dir.name}/experiments/experiment_registry.jsonl",
        f"Scope exclusions: outputs/exploratory_oracle/{results_dir.name}/subgroups/scope_exclusions.json",
        f"Flattened scores and leaderboards: outputs/exploratory_oracle/{results_dir.name}/tables/",
        f"Generated charts: outputs/exploratory_oracle/{results_dir.name}/figures/",
        "Tracked review bundle beside this report: `experiment_summary.json`, `best_model_parameters.json`, `figure_index.json`, `leaderboards/*.csv`, and `figures/*.png`.",
        "",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _git_metadata(project_root: Path) -> tuple[str, bool]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=project_root, capture_output=True, text=True, check=False
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=project_root, capture_output=True, text=True, check=False
    )
    return (revision.stdout.strip() if revision.returncode == 0 else "unknown", bool(status.stdout.strip()))


def run_exploration(
    config_path: Path,
    source_run: Path,
    *,
    resume_run: Path | None = None,
) -> Path:
    """Execute staged Oracle work, or resume a run from its saved matching stages."""

    config_path, source_run = Path(config_path), Path(source_run)
    project_root = Path(__file__).resolve().parents[2]
    config = load_search_config(config_path)
    bundle = load_oracle_data(source_run)
    if bundle.input_run_id != str(config["input_run_id"]):
        raise ValueError(
            f"Input run mismatch: config expects {config['input_run_id']}, source is {bundle.input_run_id}"
        )
    if resume_run is None:
        now = datetime.now(timezone.utc)
        run_date = now.strftime("%Y%m%d")
        run_id = now.strftime("oracle_%Y%m%dT%H%M%S%fZ_") + str(config["seed"])
        paths = create_output_tree(project_root, run_id, str(config["output_root"]))
        git_sha, git_dirty = _git_metadata(project_root)
        config_snapshot = yaml.safe_dump(config, allow_unicode=True, sort_keys=False)
        (paths["root"] / "search_config.yaml").write_text(config_snapshot, encoding="utf-8")
    else:
        paths = _open_output_tree(project_root, Path(resume_run), str(config["output_root"]))
        run_id = paths["root"].name
        match = re.match(r"oracle_(\d{8})T", run_id)
        if not match:
            raise ValueError(f"Cannot recover the Oracle run date from {run_id!r}")
        run_date = match.group(1)
        snapshot_path = paths["root"] / "search_config.yaml"
        registry_path = paths["experiments"] / "experiment_registry.jsonl"
        profile_path = paths["diagnostics"] / "input_profile.json"
        if not snapshot_path.is_file() or not registry_path.is_file() or not profile_path.is_file():
            raise FileNotFoundError("Oracle resume run is missing its saved config, registry, or input profile")
        stored_config = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
        if stored_config != config:
            raise ValueError("Resume config must exactly match the saved Oracle run config")
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        git_sha = str(profile.get("git_sha", "unknown"))
        git_dirty = bool(profile.get("git_dirty", False))
    registry_path = paths["experiments"] / "experiment_registry.jsonl"

    frame = _assign_roles_and_blocks(bundle.frame, config)
    soil_counts = soil_category_counts(frame)
    _write_parquet(soil_counts, paths["diagnostics"] / "soil_category_counts.parquet")
    role_summary = (
        frame.groupby("spatial_role", dropna=False)
        .agg(n_cells=("cell_id", "size"), n_blocks=("spatial_block", "nunique"))
        .reset_index()
    )
    _write_parquet(role_summary, paths["diagnostics"] / "spatial_role_counts.parquet")
    _write_parquet(
        frame[["cell_id", "x", "y", "SoilType", "spatial_block", "spatial_role"]],
        paths["diagnostics"] / "spatial_roles.parquet",
    )
    plot_sample = stratified_sample(
        frame,
        min(10000, len(frame)),
        strata_columns=["spatial_role", "SoilType", "spatial_block"],
        seed=int(config["seed"]) + 909,
    )
    plot_columns = [
        "cell_id", "spatial_role", "SoilType", "Aa_20yr", "Aa_2020",
        "SHI_candidate", *HEALTH_OUTCOMES,
    ]
    plot_columns = [column for column in plot_columns if column in plot_sample]
    _write_parquet(plot_sample[plot_columns], paths["diagnostics"] / "plot_data.parquet")

    profile = {
        "input_run_id": bundle.input_run_id,
        "source_run_dir": str(bundle.source_run_dir),
        "source_rows": int(len(frame)),
        "exposure_nonmissing": {
            exposure: int(pd.to_numeric(frame[exposure], errors="coerce").notna().sum())
            for exposure in bundle.exposures
        },
        "outcome_nonmissing": bundle.outcome_counts,
        "outcomes_present": list(bundle.outcomes),
        "environment_valid_cells": int(frame["environment_valid_cell"].sum())
        if "environment_valid_cell" in frame
        else None,
        "health_valid_cells": int(frame["health_valid_cell"].sum())
        if "health_valid_cell" in frame
        else None,
        "soil_rows": soil_counts.to_dict(orient="records"),
        "spatial_roles": role_summary.to_dict(orient="records"),
        "git_sha": git_sha,
        "git_dirty": git_dirty,
    }
    (paths["diagnostics"] / "input_profile.json").write_text(
        json.dumps(profile, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(
        f"[Oracle] {'Resuming' if resume_run is not None else 'Run'} {run_id}: {len(frame):,} source cells; "
        f"{len(bundle.outcomes)} outcomes; Git {git_sha[:12]}; dirty={git_dirty}.",
        flush=True,
    )

    if resume_run is None:
        matching = _run_matching_stages(frame, config, paths, registry_path, run_date)
    else:
        matching = _load_matching_stage_outputs(frame, config, paths, registry_path)
        print("[Oracle] Reused completed Stage 1–3 matching artifacts.", flush=True)
    regression = _run_regression_search(
        frame,
        config,
        paths,
        registry_path,
        matching,
        run_date,
        git_sha,
    )
    subsets = _run_oracle_subgroups(
        frame,
        config,
        paths,
        registry_path,
        matching,
        regression,
        run_date,
        git_sha,
    )

    experiment_results = _attach_subgroup_scope_status(
        _attach_role_metric_columns(_load_registry(registry_path))
    )
    if experiment_results.empty:
        raise RuntimeError("The Oracle experiment registry is empty after the run")
    experiment_results["run_id"] = run_id
    excluded_rows = experiment_results.loc[
        experiment_results["scope_status"].eq("excluded_soil_85_water_class")
    ]
    scope_exclusions = excluded_rows.loc[
        :,
        [
            column
            for column in ("experiment_id", "candidate_id", "exposure", "health_indicator")
            if column in excluded_rows
        ],
    ].drop_duplicates()
    (paths["subgroups"] / "scope_exclusions.json").write_text(
        json.dumps(
            {
                "reason": "SoilType 85 is classified as water and is excluded from primary soil-stratified candidates.",
                "excluded_trial_count": int(scope_exclusions["experiment_id"].nunique())
                if "experiment_id" in scope_exclusions
                else 0,
                "trials": scope_exclusions.to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    _write_parquet(_json_columns(experiment_results), paths["tables"] / "experiment_results.parquet")
    leaderboards = rank_leaderboards(experiment_results)
    for name, table in leaderboards.items():
        _write_parquet(_json_columns(table), paths["tables"] / f"leaderboard_{name}.parquet")

    stage2_config_counts = {
        exposure: int(
            matching["stage2_trials"].loc[
                matching["stage2_trials"]["exposure"].eq(exposure)
                & matching["stage2_trials"]["evaluation_set"].eq("validation"),
                "candidate_id",
            ].nunique()
        )
        for exposure in config["exposures"]
    }
    max_stage2 = int(config["stages"]["stage2"]["max_configs"])
    if any(count > max_stage2 for count in stage2_config_counts.values()):
        raise RuntimeError(f"Stage 2 exceeded its configured candidate cap: {stage2_config_counts}")
    registry_record_count = sum(1 for line in registry_path.read_text(encoding="utf-8").splitlines() if line.strip())
    stage3_matching_test_count = int(
        matching["stage3_matching_trials"]["candidate_id"].nunique()
    ) if not matching["stage3_matching_trials"].empty else 0
    model_trial_count = int(
        experiment_results.loc[
            experiment_results.get("experiment_type", pd.Series(index=experiment_results.index)).astype(str).str.contains("model|feature_regime", case=False),
            "experiment_id",
        ].nunique()
    )
    stage2_model_score_rows = int(len(regression["stage2_model_scores"]))
    common_sample_scores = regression.get("common_sample_exposure_scores", pd.DataFrame())
    common_sample_comparison_n = (
        int(pd.to_numeric(common_sample_scores["common_sample_n"], errors="coerce").dropna().iloc[0])
        if not common_sample_scores.empty
        and "common_sample_n" in common_sample_scores
        and pd.to_numeric(common_sample_scores["common_sample_n"], errors="coerce").notna().any()
        else None
    )
    subgroup_trial_count = int(
        experiment_results.loc[
            experiment_results.get("experiment_type", pd.Series(index=experiment_results.index)).astype(str).str.contains("subgroup", case=False),
            "experiment_id",
        ].nunique()
    )
    excluded_subgroup_trial_count = int(
        experiment_results.loc[
            experiment_results["experiment_type"].astype(str).eq("oracle_subgroup_search")
            & experiment_results["scope_status"].eq("excluded_soil_85_water_class"),
            "experiment_id",
        ].nunique()
    )
    retention_trial_count = int(
        experiment_results.loc[
            experiment_results.get("experiment_type", pd.Series(index=experiment_results.index)).astype(str).str.contains("retention", case=False),
            "experiment_id",
        ].nunique()
    )
    spatial_block_count = int(frame.loc[frame["spatial_role"].ne("unassigned"), "spatial_block"].nunique())
    summary = {
        "run_id": run_id,
        "input_run_id": bundle.input_run_id,
        "source_run_dir": str(bundle.source_run_dir),
        "source_rows": int(len(frame)),
        "stage1_sample_rows": int(len(matching["stage1_sample"])),
        "stage2_sample_rows": int(len(matching["stage2_sample"])),
        "stage1_counts": matching["stage1_counts"],
        "stage2_config_counts": stage2_config_counts,
        "stage2_role_trial_count": int(len(matching["stage2_trials"])),
        "stage3_matching_test_count": stage3_matching_test_count,
        "stage3_model_test_count": int(len(regression["test_model_scores"])),
        "stage2_model_score_rows": stage2_model_score_rows,
        "common_sample_comparison_rows": int(len(common_sample_scores)),
        "common_sample_comparison_n": common_sample_comparison_n,
        "selected_full_sample_curve_rows": int(len(regression.get("selected_full_sample_curves", pd.DataFrame()))),
        "model_trial_count": model_trial_count,
        "subgroup_trial_count": subgroup_trial_count,
        "subgroup_trial_count_in_scope": max(0, subgroup_trial_count - excluded_subgroup_trial_count),
        "scope_excluded_subgroup_trial_count": excluded_subgroup_trial_count,
        "retention_trial_count": retention_trial_count,
        "registry_record_count": registry_record_count,
        "experiment_result_rows": int(len(experiment_results)),
        "experiment_count": int(experiment_results.get("experiment_id", pd.Series(dtype=str)).nunique()),
        "status_counts": (
            experiment_results["status"].fillna("scored").value_counts(dropna=False).to_dict()
            if "status" in experiment_results
            else {}
        ),
        "spatial_block_count": spatial_block_count,
        "role_counts": role_summary.to_dict(orient="records"),
        "outcome_nonmissing": bundle.outcome_counts,
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "model_families": [family for family in config["model_families"] if family in MODEL_FAMILIES],
        "figure_count": 10,
        "selection_policy": {
            "stage1": "40k stratified train-block sample; all outcomes scored after health-blind matching",
            "stage2": "validation score ranks up to 20 configurations per exposure; no test blocks included",
            "stage3": "test blocks scored only for validation-selected matching/model configurations",
            "oracle_retention": "all-cropland post-hoc outcome-selected apparent envelope; test score intentionally absent",
        },
    }
    (paths["tables"] / "experiment_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    from experiments.oracle_exploration.generate_figures import generate_figures

    figure_paths = generate_figures(paths["root"])
    review_date = f"{run_date[:4]}-{run_date[4:6]}-{run_date[6:8]}"
    report_path = project_root / "docs" / "reviews" / review_date / "oracle-exploration-review.md"
    compact_figure_dir = report_path.parent / "figures"
    compact_figure_dir.mkdir(parents=True, exist_ok=True)
    figures_by_name = {path.name: path for path in figure_paths}
    for figure_name in REVIEW_FIGURE_NAMES:
        if figure_name in figures_by_name:
            shutil.copy2(figures_by_name[figure_name], compact_figure_dir / figure_name)
    write_review_report(paths["root"], report_path)
    run_report = paths["reports"] / "oracle-exploration-review.md"
    run_report.write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(
        f"[Oracle] Completed {registry_record_count:,} registry records, "
        f"{len(figure_paths)} figures, and report {report_path}.",
        flush=True,
    )
    return paths["root"]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the staged Oracle exploration.")
    parser.add_argument("--config", type=Path, default=Path("experiments/oracle_exploration/config/search.yaml"))
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--resume-run", type=Path, help="Resume an existing Oracle run after Stage 3 matching")
    args = parser.parse_args()
    run_dir = run_exploration(args.config, args.source_run, resume_run=args.resume_run)
    print(run_dir)


def _candidate_trial(
    source: Mapping[str, object],
    *,
    stage: str,
    sequence: int,
    role: str,
    run_date: str,
    candidate_id: str,
) -> dict[str, object]:
    trial = dict(source)
    exposure = str(source["exposure"])
    config_id = f"{stage}-{exposure.replace('_', '').lower()}-{sequence:04d}-{role}"
    trial.update(
        candidate_id=candidate_id,
        experiment_id=make_experiment_id(run_date, sequence, config_id),
        experiment_type=f"{stage}_matching_{role}",
        evaluation_set=role,
        parent_candidate_id=source.get("candidate_id"),
        selection_mode="outcome_guided_oracle",
        selection_objective=(
            "maximize_validation_r2_after_stage1_screen"
            if role == "validation"
            else "stage_screening_apparent_r2"
        ),
    )
    return trial


def _run_matching_stages(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
    paths: Mapping[str, Path],
    registry_path: Path,
    run_date: str,
) -> dict[str, Any]:
    stage1_config = config["stages"]["stage1"]
    primary_train = _primary_population(frame, ["train"])
    stage1_sample = stratified_sample(
        primary_train,
        int(stage1_config["sample_size"]),
        strata_columns=["SoilType", "spatial_block"],
        seed=int(config["seed"]),
    )
    candidates_by_exposure: dict[str, list[dict[str, object]]] = {}
    stage1_score_tables: list[pd.DataFrame] = []
    stage1_attempts: list[dict[str, object]] = []
    total_trials = len(config["exposures"]) * int(stage1_config["candidates_per_exposure"])
    completed = 0
    print(f"[Stage 1] Sampling {len(stage1_sample):,} train-block cells; target trials={total_trials}.", flush=True)
    for exposure_index, exposure in enumerate(config["exposures"]):
        candidates = draw_matching_candidates(
            config,
            exposure,
            seed=int(config["seed"]) + exposure_index * 100003,
        )
        candidates_by_exposure[exposure] = candidates
        for candidate in candidates:
            candidate["evaluation_set"] = "train"
            result = run_matching_candidate(stage1_sample, candidate, config["outcomes"], registry_path)
            stage1_attempts.append(
                {
                    "experiment_type": "stage1_matching_search",
                    "candidate_id": candidate["candidate_id"],
                    "exposure": exposure,
                    "status": result.status,
                    "error": result.error,
                    "n_pairs": len(result.pairs),
                }
            )
            if not result.scores.empty:
                scores = result.scores.copy()
                scores["experiment_type"] = "stage1_matching_search"
                scores["sample_size_requested"] = int(stage1_config["sample_size"])
                scores["sample_size_actual"] = len(stage1_sample)
                scores["selection_mode"] = "outcome_guided_oracle"
                for parameter in (
                    "vector_name", "environment_variables", "distance_method", "cluster_method",
                    "cluster_count", "environment_caliper", "geographic_radius_m", "erosion_rank_gap",
                    "exposure_rule", "environment_weights", "random_seed",
                ):
                    scores[parameter] = [candidate.get(parameter)] * len(scores)
                stage1_score_tables.append(scores)
            completed += 1
            if completed % 10 == 0 or completed == total_trials:
                print(f"[Stage 1] {completed}/{total_trials} trials complete.", flush=True)

    stage1_scores = pd.concat(stage1_score_tables, ignore_index=True) if stage1_score_tables else pd.DataFrame()
    stage1_attempt_frame = pd.DataFrame.from_records(stage1_attempts)
    expected = {exposure: int(stage1_config["candidates_per_exposure"]) for exposure in config["exposures"]}
    stage1_counts = reconcile_stage_counts(stage1_attempt_frame, expected)
    _write_parquet(stage1_attempt_frame, paths["matching_stage1"] / "trial_status.parquet")
    _write_parquet(_json_columns(stage1_scores), paths["matching_stage1"] / "scores.parquet")
    selected_by_exposure: dict[str, list[dict[str, object]]] = {}
    stage1_aggregate_tables: list[pd.DataFrame] = []
    for exposure, candidates in candidates_by_exposure.items():
        exposure_scores = stage1_scores.loc[stage1_scores["exposure"].eq(exposure)] if not stage1_scores.empty else pd.DataFrame()
        selected, aggregate = _select_stage1_configs(
            exposure_scores,
            candidates,
            int(config["stages"]["stage2"]["max_configs"]),
        )
        selected_by_exposure[exposure] = selected
        if not aggregate.empty:
            stage1_aggregate_tables.append(aggregate)
    stage1_aggregate = pd.concat(stage1_aggregate_tables, ignore_index=True) if stage1_aggregate_tables else pd.DataFrame()
    _write_parquet(stage1_aggregate, paths["matching_stage1"] / "aggregate_ranks.parquet")
    selected_log = {
        exposure: [candidate["candidate_id"] for candidate in values]
        for exposure, values in selected_by_exposure.items()
    }
    (paths["matching_stage1"] / "stage2_shortlist.json").write_text(
        json.dumps({"selection_rule": "top-by-outcome, negative-SHI, then mean normalized rank", "candidates": selected_log}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    stage2_size = int(config["stages"]["stage2"]["sample_size"])
    stage2_population = _primary_population(frame, ["train", "validation"])
    stage2_sample = _sample_by_role(stage2_population, stage2_size, int(config["seed"]) + 202)
    stage2_score_tables: list[pd.DataFrame] = []
    stage2_trials: list[dict[str, object]] = []
    sequence = 0
    planned = sum(len(candidates) for candidates in selected_by_exposure.values()) * 2
    completed = 0
    print(f"[Stage 2] Sampling {len(stage2_sample):,} train/validation cells; target role-trials={planned}.", flush=True)
    candidate_by_parent = {
        str(candidate["candidate_id"]): candidate
        for candidates in candidates_by_exposure.values()
        for candidate in candidates
    }
    for exposure, candidates in selected_by_exposure.items():
        for candidate in candidates:
            parent_id = str(candidate["candidate_id"])
            for role in ("train", "validation"):
                role_sample = stage2_sample.loc[stage2_sample["spatial_role"].eq(role)]
                if role_sample.empty:
                    continue
                sequence += 1
                trial_id = f"{parent_id}_S2_{role}"
                trial = _candidate_trial(
                    candidate,
                    stage="stage2",
                    sequence=sequence,
                    role=role,
                    run_date=run_date,
                    candidate_id=trial_id,
                )
                result = run_matching_candidate(role_sample, trial, config["outcomes"], registry_path)
                stage2_trials.append(
                    {
                        "experiment_type": "stage2_matching_validation" if role == "validation" else "stage2_matching_train",
                        "candidate_id": parent_id,
                        "candidate_trial_id": trial_id,
                        "exposure": exposure,
                        "evaluation_set": role,
                        "status": result.status,
                        "error": result.error,
                        "n_pairs": len(result.pairs),
                    }
                )
                if not result.pairs.empty:
                    pair_path = paths["matching_stage2"] / f"{trial_id}_pairs.parquet"
                    _write_parquet(result.pairs, pair_path)
                if not result.scores.empty:
                    scores = result.scores.copy()
                    scores["candidate_trial_id"] = trial_id
                    scores["candidate_id"] = parent_id
                    scores["parent_candidate_id"] = parent_id
                    scores["experiment_type"] = f"stage2_matching_{role}"
                    scores["sample_size_requested"] = stage2_size
                    scores["sample_size_actual"] = int(len(stage2_sample))
                    scores["role_sample_size"] = int(len(role_sample))
                    scores["selection_mode"] = "outcome_guided_oracle"
                    for parameter in (
                        "vector_name", "environment_variables", "distance_method", "cluster_method",
                        "cluster_count", "environment_caliper", "geographic_radius_m", "erosion_rank_gap",
                        "exposure_rule", "environment_weights", "random_seed",
                    ):
                        scores[parameter] = [candidate.get(parameter)] * len(scores)
                    stage2_score_tables.append(scores)
                completed += 1
                if completed % 10 == 0 or completed == planned:
                    print(f"[Stage 2] {completed}/{planned} role-trials complete.", flush=True)

    stage2_scores = pd.concat(stage2_score_tables, ignore_index=True) if stage2_score_tables else pd.DataFrame()
    stage2_trial_frame = pd.DataFrame.from_records(stage2_trials)
    _write_parquet(stage2_trial_frame, paths["matching_stage2"] / "trial_status.parquet")
    _write_parquet(_json_columns(stage2_scores), paths["matching_stage2"] / "scores.parquet")
    validation_winners, validation_winner_table = _best_validation_configs(stage2_scores)
    if not validation_winner_table.empty:
        validation_winner_table["experiment_type"] = "stage2_validation_winner"
    _write_parquet(_json_columns(validation_winner_table), paths["matching_stage2"] / "validation_winners.parquet")
    parent_to_candidate = candidate_by_parent
    stage3_parent_ids = sorted(set(validation_winners.values()))
    stage3_trials: list[dict[str, object]] = []
    stage3_score_tables: list[pd.DataFrame] = []
    stage3_candidates_by_exposure: dict[str, list[dict[str, object]]] = {}
    test_population = _primary_population(frame, ["test"])
    sequence = 0
    print(f"[Stage 3] Testing {len(stage3_parent_ids)} validation-selected matching configurations on withheld test blocks.", flush=True)
    for parent_id in stage3_parent_ids:
        parent = parent_to_candidate[parent_id]
        chosen_outcomes = [
            outcome
            for (exposure, outcome), candidate_id in validation_winners.items()
            if exposure == parent["exposure"] and candidate_id == parent_id
        ]
        sequence += 1
        trial_id = f"{parent_id}_S3_test"
        trial = _candidate_trial(
            parent,
            stage="stage3",
            sequence=sequence,
            role="test",
            run_date=run_date,
            candidate_id=parent_id,
        )
        trial["experiment_type"] = "stage3_matching_test"
        trial["parent_candidate_id"] = parent_id
        trial["candidate_trial_id"] = trial_id
        trial["selected_for_outcomes"] = chosen_outcomes
        trial["selection_objective"] = "validation_r2_per_outcome_then_single_test_evaluation"
        result = run_matching_candidate(test_population, trial, config["outcomes"], registry_path)
        stage3_trials.append(
            {
                "experiment_type": "stage3_matching_test",
                "candidate_id": parent_id,
                "exposure": parent["exposure"],
                "selected_for_outcomes": chosen_outcomes,
                "status": result.status,
                "error": result.error,
                "n_pairs": len(result.pairs),
            }
        )
        if not result.pairs.empty:
            _write_parquet(result.pairs, paths["matching_stage3"] / f"{trial_id}_pairs.parquet")
        if not result.scores.empty:
            scores = result.scores.copy()
            scores["candidate_id"] = parent_id
            scores["candidate_trial_id"] = trial_id
            scores["parent_candidate_id"] = parent_id
            scores["experiment_type"] = "stage3_matching_test"
            scores["selection_mode"] = "validation_selected_then_test_evaluated"
            for parameter in (
                "vector_name", "environment_variables", "distance_method", "cluster_method",
                "cluster_count", "environment_caliper", "geographic_radius_m", "erosion_rank_gap",
                "exposure_rule", "environment_weights", "random_seed",
            ):
                scores[parameter] = [parent.get(parameter)] * len(scores)
            for outcome in config["outcomes"]:
                validation_row = stage2_scores.loc[
                    stage2_scores["candidate_id"].eq(parent_id)
                    & stage2_scores["exposure"].eq(parent["exposure"])
                    & stage2_scores["outcome"].eq(outcome)
                    & stage2_scores["score_set"].eq("matched_cells")
                    & stage2_scores["evaluation_set"].eq("validation")
                ]
                if not validation_row.empty:
                    scores.loc[scores["outcome"].eq(outcome), "validation_r2"] = float(validation_row.iloc[0]["r_squared"])
            stage3_score_tables.append(scores)
        stage3_candidates_by_exposure.setdefault(str(parent["exposure"]), []).append(dict(parent))

    stage3_scores = pd.concat(stage3_score_tables, ignore_index=True) if stage3_score_tables else pd.DataFrame()
    stage3_trial_frame = pd.DataFrame.from_records(stage3_trials)
    _write_parquet(stage3_trial_frame, paths["matching_stage3"] / "trial_status.parquet")
    _write_parquet(_json_columns(stage3_scores), paths["matching_stage3"] / "scores.parquet")
    return {
        "stage1_sample": stage1_sample,
        "stage1_scores": stage1_scores,
        "stage1_attempts": stage1_attempt_frame,
        "stage1_counts": stage1_counts,
        "stage1_candidates": candidates_by_exposure,
        "stage1_selected": selected_by_exposure,
        "stage1_aggregate": stage1_aggregate,
        "stage2_sample": stage2_sample,
        "stage2_scores": stage2_scores,
        "stage2_trials": stage2_trial_frame,
        "validation_winners": validation_winners,
        "validation_winner_table": validation_winner_table,
        "stage3_matching_scores": stage3_scores,
        "stage3_matching_trials": stage3_trial_frame,
        "stage3_matching_candidates": stage3_candidates_by_exposure,
    }


def _load_matching_stage_outputs(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
    paths: Mapping[str, Path],
    registry_path: Path,
) -> dict[str, Any]:
    """Restore completed matching stages from a stopped Oracle run."""

    stage1_dir, stage2_dir, stage3_dir = (
        Path(paths[name]) for name in ("matching_stage1", "matching_stage2", "matching_stage3")
    )
    required_paths = (
        stage1_dir / "trial_status.parquet",
        stage1_dir / "scores.parquet",
        stage1_dir / "aggregate_ranks.parquet",
        stage2_dir / "trial_status.parquet",
        stage2_dir / "scores.parquet",
        stage3_dir / "trial_status.parquet",
        stage3_dir / "scores.parquet",
    )
    missing_paths = [path for path in required_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(f"Cannot resume; matching artifacts are missing: {missing_paths}")

    candidates_by_exposure: dict[str, list[dict[str, object]]] = {
        str(exposure): [] for exposure in config["exposures"]
    }
    with Path(registry_path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            candidate = record.get("candidate")
            if record.get("experiment_type") != "stage1_matching_search" or not isinstance(candidate, Mapping):
                continue
            exposure = str(candidate.get("exposure", record.get("exposure", "")))
            if exposure in candidates_by_exposure:
                candidates_by_exposure[exposure].append(dict(candidate))

    stage1_attempts = pd.read_parquet(stage1_dir / "trial_status.parquet")
    expected = {
        str(exposure): int(config["stages"]["stage1"]["candidates_per_exposure"])
        for exposure in config["exposures"]
    }
    stage1_counts = reconcile_stage_counts(stage1_attempts, expected)
    stage1_scores = pd.read_parquet(stage1_dir / "scores.parquet")
    stage1_aggregate = pd.read_parquet(stage1_dir / "aggregate_ranks.parquet")
    stage2_trials = pd.read_parquet(stage2_dir / "trial_status.parquet")
    stage2_scores = pd.read_parquet(stage2_dir / "scores.parquet")
    validation_winners, validation_winner_table = _best_validation_configs(stage2_scores)
    stage3_trials = pd.read_parquet(stage3_dir / "trial_status.parquet")
    stage3_scores = pd.read_parquet(stage3_dir / "scores.parquet")

    selected_by_exposure: dict[str, list[dict[str, object]]] = {}
    for exposure, candidates in candidates_by_exposure.items():
        selected_ids = set(
            stage2_trials.loc[
                stage2_trials.get("exposure", pd.Series(index=stage2_trials.index)).astype(str).eq(exposure),
                "candidate_id",
            ].dropna().astype(str)
        ) if "candidate_id" in stage2_trials else set()
        selected_by_exposure[exposure] = [
            candidate for candidate in candidates if str(candidate.get("candidate_id")) in selected_ids
        ]

    candidate_lookup = {
        str(candidate["candidate_id"]): candidate
        for candidates in candidates_by_exposure.values()
        for candidate in candidates
    }
    stage3_by_exposure: dict[str, list[dict[str, object]]] = {}
    if "candidate_id" in stage3_trials:
        for candidate_id in stage3_trials["candidate_id"].dropna().astype(str).unique():
            candidate = candidate_lookup.get(candidate_id)
            if candidate is not None:
                stage3_by_exposure.setdefault(str(candidate["exposure"]), []).append(candidate)

    stage1_population = _primary_population(frame, ["train"])
    stage1_sample = stratified_sample(
        stage1_population,
        int(config["stages"]["stage1"]["sample_size"]),
        strata_columns=["SoilType", "spatial_block"],
        seed=int(config["seed"]),
    )
    stage2_population = _primary_population(frame, ["train", "validation"])
    stage2_sample = _sample_by_role(
        stage2_population,
        int(config["stages"]["stage2"]["sample_size"]),
        int(config["seed"]) + 202,
    )
    return {
        "stage1_sample": stage1_sample,
        "stage1_scores": stage1_scores,
        "stage1_attempts": stage1_attempts,
        "stage1_counts": stage1_counts,
        "stage1_candidates": candidates_by_exposure,
        "stage1_selected": selected_by_exposure,
        "stage1_aggregate": stage1_aggregate,
        "stage2_sample": stage2_sample,
        "stage2_scores": stage2_scores,
        "stage2_trials": stage2_trials,
        "validation_winners": validation_winners,
        "validation_winner_table": validation_winner_table,
        "stage3_matching_scores": stage3_scores,
        "stage3_matching_trials": stage3_trials,
        "stage3_matching_candidates": stage3_by_exposure,
    }


if __name__ == "__main__":
    main()
