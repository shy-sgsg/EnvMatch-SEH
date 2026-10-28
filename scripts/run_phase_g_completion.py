#!/usr/bin/env python3
"""Complete Phase G scope, paired-model, and group-interaction analyses."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/phase-g-completion-matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from experiments.oracle_exploration.data import load_oracle_data
from experiments.oracle_exploration.environment_search import fit_environment_labels
from experiments.oracle_exploration.nonlinear_search import fit_model_family
from experiments.oracle_exploration.phase_g import rank_phase_g_models
from experiments.oracle_exploration.phase_g_completion import (
    PAIR_MODEL_FAMILIES,
    _direction_from_curve,
    _model_metrics,
    _paired_arrays,
    compare_group_slopes,
    mark_pixel_candidate_scope,
    search_paired_response_models,
    validate_pair_models_spatial,
)
from scripts.run_phase_g_analysis import _group_pair_data

ENVIRONMENT = ["Elevation", "Cur", "TWI", "LST", "NDWI", "NDBSI", "EVI", "NDVI", "Precipitation"]
MARKER = "<!-- PHASE_G_COMPLETION_ADDENDUM_START -->"


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value):
        return None
    return str(value)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _markdown_table(frame: pd.DataFrame, columns: list[tuple[str, str]], limit: int | None = None) -> str:
    selected = frame.head(limit) if limit is not None else frame
    labels = [label for _, label in columns]
    rows = ["| " + " | ".join(labels) + " |", "| " + " | ".join("---" for _ in labels) + " |"]
    if selected.empty:
        return "\n".join([*rows, "| " + " | ".join("无可报告结果" for _ in labels) + " |"])
    for _, record in selected.iterrows():
        cells = []
        for key, _ in columns:
            value = record.get(key, "")
            if pd.isna(value):
                cell = "NA"
            elif isinstance(value, (float, np.floating)):
                cell = f"{float(value):.4g}"
            else:
                cell = str(value).replace("|", "\\|").replace("\n", " ")
            cells.append(cell)
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _append_section(path: Path, content: str) -> None:
    original = path.read_text(encoding="utf-8") if path.exists() else ""
    if MARKER in original:
        original = original[: original.index(MARKER)].rstrip() + "\n\n"
    path.write_text(original + MARKER + "\n" + content.strip() + "\n", encoding="utf-8")


def _reconstruct_stage2_directions(
    scoped: pd.DataFrame,
    frame: pd.DataFrame,
    oracle_dir: Path,
    checkpoint_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recreate only ambiguous stage-2 tree candidates that could lead a board."""
    from experiments.oracle_exploration.experiment import _primary_population, _sample_by_role
    from experiments.oracle_exploration.nonlinear_search import _design_matrices, _estimator
    from sklearn.metrics import r2_score

    current = pd.read_csv(checkpoint_path) if checkpoint_path.is_file() else pd.DataFrame()
    cached = {(str(row.candidate_id), str(row.outcome), str(row.exposure), str(row.model_name)): row._asdict() for row in current.itertuples(index=False)}
    eligible = scoped.loc[scoped["in_primary_cropland_scope"]].copy()
    known_negative = eligible.loc[eligible["direction"].astype(str).eq("negative")]
    cutoffs = known_negative.groupby(["exposure", "outcome"])["r_squared"].max().to_dict()
    family_names = {"random_forest", "extra_trees", "hist_gradient_boosting"}
    possible = eligible.loc[
        eligible["experiment_type"].eq("stage2_model_search")
        & eligible["model_name"].isin(family_names)
        & eligible["status"].fillna("ok").astype(str).str.lower().isin({"ok", "success"})
    ].copy()
    possible["__current_cutoff"] = [cutoffs.get((row.exposure, row.outcome), -np.inf) for row in possible.itertuples()]
    possible = possible.loc[
        pd.to_numeric(possible["r_squared"], errors="coerce").ge(pd.to_numeric(possible["__current_cutoff"], errors="coerce"))
    ]
    if possible.empty:
        return scoped, current

    # Recreate the fixed Phase G Oracle stage-2 sample from its stored roles and seed.
    oracle_config = yaml.safe_load((oracle_dir / "search_config.yaml").read_text(encoding="utf-8"))
    roles = pd.read_parquet(oracle_dir / "diagnostics" / "spatial_roles.parquet", columns=["cell_id", "spatial_block", "spatial_role"])
    group_assignments = pd.read_parquet(oracle_dir / "diagnostics" / "environment_group_assignments.parquet")
    model_frame = frame.merge(roles, on="cell_id", how="left", validate="one_to_one")
    model_frame = model_frame.merge(group_assignments.drop(columns=["spatial_role"], errors="ignore"), on="cell_id", how="left", validate="one_to_one")
    stage2_population = _primary_population(model_frame, ["train", "validation"])
    stage2_sample = _sample_by_role(
        stage2_population,
        int(oracle_config["stages"]["stage2"]["sample_size"]),
        int(oracle_config["seed"]) + 202,
    )

    from experiments.oracle_exploration.regression_search import _complete_case_frame

    added: list[dict[str, object]] = []
    for row in possible.itertuples(index=False):
        key = (str(row.candidate_id), str(row.outcome), str(row.exposure), str(row.model_name))
        if key in cached:
            continue
        raw_features = row.features
        features = json.loads(raw_features) if isinstance(raw_features, str) else list(raw_features)
        model_config = json.loads(row.model_config) if isinstance(row.model_config, str) else dict(row.model_config or {})
        group_column = model_config.get("group_column")
        if group_column not in stage2_sample:
            group_column = None
        outcome = str(row.outcome)
        exposure = str(row.exposure)
        environment = [name for name in features if name != exposure]
        data, _ = _complete_case_frame(stage2_sample, outcome, exposure, environment, group_column)
        seed = int(model_config.get("random_state", oracle_config["seed"]))
        lower, upper = np.quantile(pd.to_numeric(data[exposure], errors="coerce"), [0.01, 0.99])
        grid_x = np.linspace(float(lower), float(upper), 100)
        score = pd.DataFrame({exposure: grid_x, outcome: float(pd.to_numeric(data[outcome], errors="coerce").median())})
        for column in environment:
            score[column] = float(pd.to_numeric(data[column], errors="coerce").median())
        if group_column is not None:
            mode = data[group_column].mode(dropna=True)
            if mode.empty:
                continue
            score[group_column] = mode.iloc[0]
        x_train, x_score, _ = _design_matrices(
            data,
            score,
            features,
            exposure=exposure,
            group_col=group_column,
            model_family=str(row.model_name),
            include_group=False,
        )
        estimator, _ = _estimator(str(row.model_name), seed)
        y = pd.to_numeric(data[outcome], errors="coerce").to_numpy(dtype=float)
        estimator.fit(x_train, y)
        reconstructed_r2 = float(r2_score(y, estimator.predict(x_train)))
        curve = np.asarray(estimator.predict(x_score), dtype=float)
        direction, slope = _direction_from_curve(grid_x, curve)
        source_r2 = float(row.r_squared)
        difference = reconstructed_r2 - source_r2
        reproduction = "exact_r_squared_reproduction" if abs(difference) <= 1e-8 else "r_squared_not_exactly_reproduced"
        record = {
            "candidate_id": key[0], "outcome": outcome, "exposure": exposure, "model_name": str(row.model_name),
            "model_family": str(row.model_name), "direction": direction, "direction_basis": reproduction + "; exact source stage-2 sample, variables, family, and seed; conditional curve at sample medians",
            "source_r_squared": source_r2, "reconstructed_r_squared": reconstructed_r2, "r_squared_difference": difference,
            "source_n": int(row.n), "reconstructed_n": int(len(data)), "n_matches_source": int(len(data)) == int(row.n),
            "average_curve_slope": slope, "curve_change_2_98": float(curve[-1] - curve[0]),
            "sample_size": int(len(stage2_sample)), "sample_seed": int(oracle_config["seed"]) + 202,
            "candidate_selection_objective": "reconstruct only stage-2 tree candidates whose stored apparent R² can meet or exceed current best verified negative candidate",
            "outcome_guided_selection": True, "status": reproduction,
        }
        added.append(record)
        cached[key] = record
        _write_csv(checkpoint_path, pd.DataFrame.from_records(list(cached.values())))
        print(f"[Phase G completion] reconstructed pixel candidate {outcome}/{exposure}/{row.model_name}: source R² {source_r2:.4f}, reproduced {reconstructed_r2:.4f}, {direction}", flush=True)

    reconstructed = pd.DataFrame.from_records(list(cached.values()))
    if not reconstructed.empty:
        for row in reconstructed.itertuples(index=False):
            mask = (
                scoped["candidate_id"].astype(str).eq(str(row.candidate_id))
                & scoped["outcome"].astype(str).eq(str(row.outcome))
                & scoped["exposure"].astype(str).eq(str(row.exposure))
                & scoped["model_name"].astype(str).eq(str(row.model_name))
            )
            scoped.loc[mask, "direction"] = str(row.direction)
            scoped.loc[mask, "direction_basis"] = str(row.direction_basis)
            scoped.loc[mask, "reconstructed_model_r_squared"] = float(row.reconstructed_r_squared)
    return scoped, reconstructed


def _candidate_scope_tables(
    oracle_dir: Path,
    outcomes: list[str],
    frame: pd.DataFrame,
    reconstruction_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    registry = pd.read_parquet(oracle_dir / "tables" / "experiment_results.parquet")
    experiment_types = {
        "stage1_matching_search",
        "stage2_model_search",
        "oracle_subgroup_search",
        "oracle_retention_search",
        "full_sample_model",
    }
    apparent = registry.loc[
        registry["experiment_type"].isin(experiment_types)
        & registry.get("model_name", pd.Series(index=registry.index, dtype=object)).notna()
        & registry["evaluation_set"].fillna("").astype(str).str.lower().eq("apparent")
        & registry["outcome"].isin(outcomes)
    ].copy()
    scoped = mark_pixel_candidate_scope(apparent)
    curve_path = oracle_dir / "models" / "selected_full_sample_curves.parquet"
    if curve_path.is_file():
        curves = pd.read_parquet(curve_path)
        curve_rows: list[dict[str, object]] = []
        for keys, curve in curves.groupby(["outcome", "exposure", "model_name", "environment_vector"], dropna=False, sort=True):
            curve = curve.sort_values("exposure_value")
            x = pd.to_numeric(curve["exposure_value"], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(curve["predicted_outcome"], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(x) & np.isfinite(y)
            x, y = x[valid], y[valid]
            if len(x) < 2 or np.ptp(x) <= 0:
                continue
            x_low, x_high = np.quantile(x, [0.1, 0.9])
            change = float(np.interp(x_high, x, y) - np.interp(x_low, x, y))
            scale = max(float(np.std(y)), 1e-12)
            direction = "flat" if abs(change) <= 0.01 * scale else "negative" if change < 0 else "positive"
            curve_rows.append({
                "outcome": keys[0], "exposure": keys[1], "model_name": keys[2],
                "environment_vector": keys[3], "matched_full_curve_direction": direction,
                "matched_full_curve_change_10_90": change,
            })
        if curve_rows:
            scoped = scoped.merge(pd.DataFrame.from_records(curve_rows), on=["outcome", "exposure", "model_name", "environment_vector"], how="left", validate="many_to_one")
    if "direction" not in scoped:
        scoped["direction"] = "unknown"
    scoped["direction_basis"] = "direction stored with candidate trial"
    curve_proxy = scoped["direction"].fillna("unknown").astype(str).eq("unknown") & scoped.get("matched_full_curve_direction", pd.Series(index=scoped.index, dtype=object)).notna()
    scoped.loc[curve_proxy, "direction"] = scoped.loc[curve_proxy, "matched_full_curve_direction"]
    scoped.loc[curve_proxy, "direction_basis"] = "same outcome/exposure/model/environment vector in stored selected full-sample response curve; curve is a proxy if candidate fit used another sample"
    if "coefficient" in scoped:
        coefficient = pd.to_numeric(scoped["coefficient"], errors="coerce")
        coefficient_proxy = scoped["direction"].fillna("unknown").astype(str).eq("unknown") & np.isfinite(coefficient)
        scoped.loc[coefficient_proxy, "direction"] = np.where(coefficient.loc[coefficient_proxy] < 0, "negative", np.where(coefficient.loc[coefficient_proxy] > 0, "positive", "flat"))
        scoped.loc[coefficient_proxy, "direction_basis"] = "stored linear exposure coefficient"
    if "analysis_unit" not in scoped:
        scoped["analysis_unit"] = "pixel_level_candidate"
    scoped, reconstructions = _reconstruct_stage2_directions(scoped, frame, oracle_dir, reconstruction_path)
    eligible = scoped.loc[scoped["in_primary_cropland_scope"]].copy()
    boards = rank_phase_g_models(eligible)
    rank_rows: list[dict[str, object]] = []
    counts = eligible.groupby(["exposure", "outcome"], dropna=False).size().to_dict()
    for key, board in boards.items():
        for _, row in board.iterrows():
            record = row.to_dict()
            record.update(
                ranking_type=f"in_scope_{key}",
                candidate_count=int(counts.get((row.get("exposure"), row.get("outcome")), 0)),
                scope_rule="exclude candidate IDs explicitly scoped to SoilType 85 water and non-primary soils; retain full-cropland candidates",
                ranking_scope="all eligible apparent pixel candidates in prior Oracle registry",
                outcome_guided_selection=True,
            )
            rank_rows.append(record)
    reconciliation = (
        scoped.groupby(["exposure", "outcome"], dropna=False)
        .agg(
            apparent_candidate_rows=("candidate_id", "size"),
            eligible_candidate_rows=("in_primary_cropland_scope", "sum"),
            excluded_candidate_rows=("in_primary_cropland_scope", lambda values: int((~values).sum())),
            apparent_unique_candidates=("candidate_id", "nunique"),
        )
        .reset_index()
    )
    reconciliation["retained_row_fraction"] = reconciliation["eligible_candidate_rows"] / reconciliation["apparent_candidate_rows"]
    reconciliation["excluded_rows_include_water_soil_85"] = True
    return scoped, pd.DataFrame.from_records(rank_rows), reconciliation, reconstructions


def _refit_discovery_winners(
    pool: pd.DataFrame,
    discovery: pd.DataFrame,
    *,
    outcomes: list[str],
    seed: int,
    full_pair_count: int,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for outcome_index, outcome in enumerate(outcomes):
        candidates = discovery.loc[
            discovery["outcome"].eq(outcome) & discovery["status"].eq("ok")
        ].copy()
        if candidates.empty:
            continue
        score = pd.to_numeric(candidates["r_squared"], errors="coerce")
        objectives: dict[str, pd.DataFrame] = {"highest_discovery_apparent_r2": candidates.loc[score.idxmax()].to_frame().T}
        negative = candidates.loc[candidates["direction"].eq("negative")]
        if not negative.empty:
            objectives["highest_negative_discovery_apparent_r2"] = negative.loc[
                pd.to_numeric(negative["r_squared"], errors="coerce").idxmax()
            ].to_frame().T
        selected_by_family: dict[str, list[str]] = {}
        for objective, winner in objectives.items():
            family = str(winner.iloc[0]["model_family"])
            selected_by_family.setdefault(family, []).append(objective)
        x, y, valid = _paired_arrays(pool, outcome)
        training = pd.DataFrame({"delta_e": x, "delta_y": y})
        for family_index, (family, selection_objectives) in enumerate(selected_by_family.items()):
            base: dict[str, object] = {
                "analysis_unit": "matched_pair_difference",
                "method": str(pool["method"].iloc[0]),
                "exposure": str(pool["exposure"].iloc[0]),
                "outcome": outcome,
                "candidate_id": f"paired_full_refit_{pool['method'].iloc[0]}_{pool['exposure'].iloc[0]}_{outcome}_{family}",
                "model_family": family,
                "model_name": family,
                "fit_scope": "full_matched_pair_sample_refit_of_discovery_winner",
                "evaluation_set": "apparent",
                "n_pairs": int(len(x)),
                "n": int(len(x)),
                "source_full_pair_count": int(full_pair_count),
                "discovery_sample_n": int(pd.to_numeric(candidates["n_pairs"], errors="coerce").max()),
                "discovery_sample_fraction": float(pd.to_numeric(candidates["n_pairs"], errors="coerce").max() / full_pair_count),
                "candidate_count": int(len(candidates)),
                "selection_objectives": "|".join(selection_objectives),
                "selection_mode": "EXPLORATORY_ORACLE: top family selected on fixed discovery sample then refit on all available pairs",
                "outcome_guided_selection": True,
                "r2_scope": "full-sample apparent matched-pair Δhealth fit; selected model family was screened on a representative subsample",
                "random_seed": int(seed + outcome_index * 101 + family_index),
            }
            try:
                prediction, parameters = fit_model_family(
                    training,
                    training,
                    "delta_y",
                    "delta_e",
                    ["delta_e"],
                    model_family=family,
                    random_state=int(seed + outcome_index * 101 + family_index),
                )
                metrics = _model_metrics(y, prediction, x)
                direction, average_slope = _direction_from_curve(x, prediction)
                x_sd, y_sd = float(np.std(x)), float(np.std(y))
                records.append({
                    **base,
                    "status": "ok",
                    "error": None,
                    **metrics,
                    "direction": direction,
                    "average_curve_slope": average_slope,
                    "standardized_effect": average_slope * x_sd / y_sd if y_sd > 0 else float("nan"),
                    "model_parameters": json.dumps(parameters, ensure_ascii=False, default=_json_default),
                })
            except Exception as exc:
                records.append({**base, "status": "failed", "error": f"{type(exc).__name__}: {exc}", "direction": "unknown", "r_squared": float("nan")})
    return pd.DataFrame.from_records(records)


def _paired_model_tables(
    pairs: pd.DataFrame,
    *,
    outcomes: list[str],
    config: dict[str, Any],
    tables_dir: Path,
    state: dict[str, object],
    state_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    settings = config["pair_model_search"]
    families = list(settings["model_families"])
    if not set(families).issubset(PAIR_MODEL_FAMILIES):
        raise ValueError("pair_model_search.model_families contains an unsupported family")
    model_path = tables_dir / "paired_response_model_trials.csv"
    model_frames = [pd.read_csv(model_path)] if model_path.is_file() else []
    spatial_path = tables_dir / "paired_spatial_model_trials.csv"
    spatial_frames = [pd.read_csv(spatial_path)] if spatial_path.is_file() else []
    selected_path = tables_dir / "paired_spatial_selected_models.csv"
    selected_frames = [pd.read_csv(selected_path)] if selected_path.is_file() else []
    baseline_path = tables_dir / "paired_full_sample_ols_baselines.csv"
    baseline_frames = [pd.read_csv(baseline_path)] if baseline_path.is_file() else []
    model_done = set(state.get("pair_model_completed", []))
    spatial_done = set(state.get("pair_spatial_completed", []))
    groups = list(pairs.groupby(["method", "exposure"], sort=True))
    for group_index, ((method, exposure), pool) in enumerate(groups):
        key = f"{method}|{exposure}"
        if key not in model_done:
            full_baseline = search_paired_response_models(pool, outcomes=outcomes, model_families=["ols"], seed=int(config["seed"]))
            full_baseline["candidate_role"] = "full_sample_ols_baseline"
            full_baseline["source_full_pair_count"] = len(pool)
            baseline_frames.append(full_baseline)

            sample_n = min(int(settings["discovery_pairs_per_method_exposure"]), len(pool))
            discovery_pool = pool.sample(n=sample_n, random_state=int(config["seed"]) + group_index).sort_values("pair_id").reset_index(drop=True)
            discovery = search_paired_response_models(discovery_pool, outcomes=outcomes, model_families=families, seed=int(config["seed"]) + group_index)
            discovery["candidate_role"] = "representative_discovery_sample"
            discovery["source_full_pair_count"] = len(pool)
            discovery["discovery_sample_size"] = sample_n
            discovery["full_pair_retention_fraction"] = sample_n / len(pool) if len(pool) else float("nan")
            refits = _refit_discovery_winners(
                pool,
                discovery,
                outcomes=outcomes,
                seed=int(config["seed"]) + group_index,
                full_pair_count=len(pool),
            )
            model_frames.extend([discovery, refits])
            model_done.add(key)
            _write_csv(model_path, pd.concat(model_frames, ignore_index=True, sort=False))
            _write_csv(baseline_path, pd.concat(baseline_frames, ignore_index=True, sort=False))
            state["pair_model_completed"] = sorted(model_done)
            _write_json(state_path, state)
            print(f"[Phase G completion] apparent pair-model search {key}: {sample_n:,}/{len(pool):,} discovery pairs", flush=True)

        if key not in spatial_done:
            spatial_n = min(int(settings["spatial_pairs_per_method_exposure"]), len(pool))
            spatial_pool = pool.sample(n=spatial_n, random_state=int(config["seed"]) + 1000 + group_index).sort_values("pair_id").reset_index(drop=True)
            trials, selected = validate_pair_models_spatial(
                spatial_pool,
                outcomes=outcomes,
                model_families=families,
                seeds=[int(value) for value in settings["spatial_seeds"]],
                block_size_m=float(settings["block_size_m"]),
                min_train_pairs=int(settings["min_train_pairs"]),
            )
            trials["source_full_pair_count"] = len(pool)
            trials["spatial_sample_size"] = spatial_n
            trials["spatial_sample_fraction"] = spatial_n / len(pool) if len(pool) else float("nan")
            selected["source_full_pair_count"] = len(pool)
            selected["spatial_sample_size"] = spatial_n
            selected["spatial_sample_fraction"] = spatial_n / len(pool) if len(pool) else float("nan")
            spatial_frames.append(trials)
            selected_frames.append(selected)
            spatial_done.add(key)
            _write_csv(spatial_path, pd.concat(spatial_frames, ignore_index=True, sort=False))
            _write_csv(selected_path, pd.concat(selected_frames, ignore_index=True, sort=False))
            state["pair_spatial_completed"] = sorted(spatial_done)
            _write_json(state_path, state)
            print(f"[Phase G completion] spatial pair-model validation {key}: {spatial_n:,}/{len(pool):,} sample pairs", flush=True)

    trials = pd.read_csv(model_path)
    discovery_rows = trials.get("candidate_role", pd.Series(index=trials.index, dtype=object)).eq("representative_discovery_sample")
    trials.loc[discovery_rows, "fit_scope"] = "representative_discovery_sample"
    _write_csv(model_path, trials)
    baseline = pd.read_csv(baseline_path)
    spatial_trials = pd.read_csv(spatial_path) if spatial_path.exists() else pd.DataFrame()
    spatial_selected = pd.read_csv(selected_path) if selected_path.exists() else pd.DataFrame()
    spatial_summary = _summarize_spatial_selected(spatial_selected)
    leaderboard = _paired_model_leaderboard(trials, baseline, spatial_summary)
    return trials, pd.concat([baseline, leaderboard], ignore_index=True, sort=False), spatial_trials, spatial_selected


def _summarize_spatial_selected(selected: pd.DataFrame) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame()
    valid = selected.loc[selected["status"].eq("ok")].copy()
    if valid.empty:
        return pd.DataFrame()
    records: list[dict[str, object]] = []
    keys = ["method", "exposure", "outcome", "selection_type"]
    for key, group in valid.groupby(keys, sort=True, dropna=False):
        validation = pd.to_numeric(group["validation_r_squared"], errors="coerce")
        test = pd.to_numeric(group["test_r_squared"], errors="coerce")
        model_counts = group["selected_model_family"].value_counts()
        records.append({
            "analysis_unit": "matched_pair_difference",
            "method": key[0],
            "exposure": key[1],
            "outcome": key[2],
            "selection_type": key[3],
            "ranking_type": "validation_selected_spatial_pair_model",
            "selected_model_families": json.dumps(model_counts.to_dict(), ensure_ascii=False),
            "median_validation_r_squared": float(validation.median()),
            "median_test_r_squared_report_only": float(test.median()),
            "test_r_squared_iqr_low": float(test.quantile(0.25)),
            "test_r_squared_iqr_high": float(test.quantile(0.75)),
            "median_test_spearman": float(pd.to_numeric(group["test_spearman_x_y"], errors="coerce").median()),
            "spatial_seed_count": int(group["spatial_seed"].nunique()),
            "selection_policy": "model selected within each seed by validation R²; test metrics are summarized without candidate re-ranking",
            "outcome_guided_selection": True,
        })
    return pd.DataFrame.from_records(records)


def _paired_model_leaderboard(trials: pd.DataFrame, baseline: pd.DataFrame, spatial: pd.DataFrame) -> pd.DataFrame:
    available = pd.concat([baseline, trials], ignore_index=True, sort=False)
    available = available.loc[available["status"].eq("ok") & available["evaluation_set"].astype(str).str.lower().eq("apparent")].copy()
    records: list[dict[str, object]] = []
    for (method, exposure, outcome), group in available.groupby(["method", "exposure", "outcome"], sort=True):
        full = group.loc[group["fit_scope"].astype(str).isin(["full_matched_pair_sample", "full_matched_pair_sample_refit_of_discovery_winner"])].copy()
        if full.empty:
            continue
        r2 = pd.to_numeric(full["r_squared"], errors="coerce")
        winner = full.loc[r2.idxmax()].to_dict()
        winner.update(ranking_type="highest_full_sample_paired_apparent_r2", candidate_count=int(len(full)), test_used_for_ranking=False)
        records.append(winner)
        negative = full.loc[full["direction"].eq("negative")]
        if not negative.empty:
            neg = negative.loc[pd.to_numeric(negative["r_squared"], errors="coerce").idxmax()].to_dict()
            neg.update(ranking_type="highest_negative_full_sample_paired_apparent_r2", candidate_count=int(len(negative)), test_used_for_ranking=False)
            records.append(neg)
        discoveries = group.loc[group["candidate_role"].eq("representative_discovery_sample")]
        if not discoveries.empty:
            discovery_winner = discoveries.loc[pd.to_numeric(discoveries["r_squared"], errors="coerce").idxmax()].to_dict()
            discovery_winner.update(ranking_type="highest_discovery_sample_paired_apparent_r2", candidate_count=int(len(discoveries)), test_used_for_ranking=False)
            records.append(discovery_winner)
            discovery_negative = discoveries.loc[discoveries["direction"].eq("negative")]
            if not discovery_negative.empty:
                negative_winner = discovery_negative.loc[pd.to_numeric(discovery_negative["r_squared"], errors="coerce").idxmax()].to_dict()
                negative_winner.update(ranking_type="highest_negative_discovery_sample_paired_apparent_r2", candidate_count=int(len(discovery_negative)), test_used_for_ranking=False)
                records.append(negative_winner)
    if not spatial.empty:
        records.extend(spatial.to_dict(orient="records"))
    return pd.DataFrame.from_records(records)


def _environment_group_assignments(
    frame: pd.DataFrame,
    pairs: pd.DataFrame,
    *,
    config: dict[str, Any],
    phase_g_config: dict[str, Any],
    labels_path: Path,
    meta_path: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if labels_path.is_file() and meta_path.is_file():
        labels = pd.read_parquet(labels_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        base_pairs, base_meta = _group_pair_data(frame, pairs, phase_g_config)
        group_columns = list(base_meta["group_columns"])
        labels = base_pairs[["pair_id", "high_erosion_cell_id", *group_columns]].drop_duplicates("pair_id").copy()
        cluster = config["group_analysis"]
        for k in [int(value) for value in cluster["cluster_counts"]]:
            fitted = fit_environment_labels(
                frame,
                [str(value) for value in cluster["cluster_variables"]],
                method=str(cluster["cluster_method"]),
                k=k,
                by_soil=True,
                seed=int(cluster["seed"]) + k,
                soil_codes=[int(value) for value in config["primary_soils"]],
            )
            values = pd.Series(fitted, index=frame.index)
            labels[f"kmeans_terrain_k{k}"] = labels["high_erosion_cell_id"].map(
                pd.Series(
                    [f"cluster_{int(value)}" if int(value) >= 0 else pd.NA for value in values.to_numpy()],
                    index=frame["cell_id"],
                )
            )
            group_columns.append(f"kmeans_terrain_k{k}")
        meta = {
            **base_meta,
            "group_columns": group_columns,
            "additional_kmeans_groups": {
                f"kmeans_terrain_k{int(k)}": {
                    "method": str(cluster["cluster_method"]),
                    "variables": [str(value) for value in cluster["cluster_variables"]],
                    "requested_clusters_per_soil": int(k),
                    "seed": int(cluster["seed"]) + int(k),
                    "health_outcomes_used_for_labels": False,
                }
                for k in cluster["cluster_counts"]
            },
        }
        labels.to_parquet(labels_path, index=False)
        _write_json(meta_path, meta)
    soil_by_cell = frame.set_index("cell_id")["SoilType"]
    high_soil = pd.to_numeric(labels["high_erosion_cell_id"].map(soil_by_cell), errors="coerce")
    for column in [name for name in meta["group_columns"] if str(name).startswith("kmeans_terrain_k")]:
        raw_ids = pd.to_numeric(labels[column].astype("string").str.extract(r"cluster_(\d+)$", expand=False), errors="coerce")
        mapping: dict[tuple[int, int], str] = {}
        pair_codes = pd.DataFrame({"soil": high_soil, "cluster": raw_ids}).dropna().astype(int)
        for soil_code, selected in pair_codes.groupby("soil", sort=True):
            for local_index, cluster_id in enumerate(sorted(selected["cluster"].unique())):
                mapping[(int(soil_code), int(cluster_id))] = f"soil_{int(soil_code)}_cluster_{local_index}"
        labels[column] = [mapping.get((int(soil), int(cluster)), pd.NA) if pd.notna(soil) and pd.notna(cluster) else pd.NA for soil, cluster in zip(high_soil, raw_ids, strict=True)]
    meta["label_schema"] = "soil_qualified_v1"
    _write_json(meta_path, meta)
    labels.to_parquet(labels_path, index=False)
    pair_groups = pairs.copy()
    assigned = labels[["pair_id", *meta["group_columns"]]]
    pair_groups = pair_groups.merge(assigned, on="pair_id", how="left", validate="one_to_one", suffixes=("", "_assigned"))
    for column in meta["group_columns"]:
        assigned_column = f"{column}_assigned"
        if assigned_column in pair_groups:
            pair_groups[column] = pair_groups[assigned_column]
            pair_groups.drop(columns=assigned_column, inplace=True)
    return pair_groups, meta


def _group_profile_table(grouped_pairs: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    env_vars = [name for name in ENVIRONMENT if f"mean_{name}" in grouped_pairs]
    rows: list[dict[str, object]] = []
    for group_column in group_columns:
        for (method, exposure, label), selected in grouped_pairs.groupby(
            ["method", "exposure", group_column], sort=True, dropna=True, observed=True
        ):
            row: dict[str, object] = {
                "group_column": group_column,
                "group_method": group_column.replace("_group", ""),
                "group_label": str(label),
                "method": str(method),
                "exposure": str(exposure),
                "n_pairs": int(len(selected)),
                "spatial_blocks": int(selected["spatial_block_id"].nunique()),
                "profile_unit": "mean of the two matched-pair members; values summarize pair members",
            }
            for variable in env_vars:
                values = pd.to_numeric(selected[f"mean_{variable}"], errors="coerce").dropna()
                row[f"mean_{variable}"] = float(values.mean()) if not values.empty else float("nan")
                row[f"median_{variable}"] = float(values.median()) if not values.empty else float("nan")
                row[f"std_{variable}"] = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
                row[f"q25_{variable}"] = float(values.quantile(0.25)) if not values.empty else float("nan")
                row[f"q75_{variable}"] = float(values.quantile(0.75)) if not values.empty else float("nan")
            rows.append(row)
    return pd.DataFrame.from_records(rows)


def _write_group_forest(slopes: pd.DataFrame, figure_dir: Path, review_figure_dir: Path) -> dict[str, object]:
    selected = slopes.loc[
        slopes["status"].eq("ok") & slopes["method"].eq("M4") & slopes["exposure"].eq("Aa_20yr")
    ].copy()
    selected = selected.loc[pd.to_numeric(selected["n_pairs"], errors="coerce").ge(100)]
    selections: list[pd.DataFrame] = []
    for outcome, group in selected.groupby("outcome", sort=False):
        group = group.sort_values("slope_delta_y_per_delta_log1p_erosion", ascending=True).head(6).copy()
        group["figure_selection_rule"] = "six most negative estimates per outcome among all tested M4/Aa_20yr environmental groups"
        selections.append(group)
    plot_data = pd.concat(selections, ignore_index=True) if selections else pd.DataFrame()
    figures = []
    if not plot_data.empty:
        fig, axes = plt.subplots(3, 3, figsize=(15, 12), constrained_layout=True)
        outcomes = ["SHI_candidate", "AN", "OM", "CEC", "pH", "SW", "BD", "MBC", "RESP"]
        for ax, outcome in zip(axes.flat, outcomes, strict=True):
            group = plot_data.loc[plot_data["outcome"].eq(outcome)].sort_values("slope_delta_y_per_delta_log1p_erosion")
            y = np.arange(len(group))
            effect = pd.to_numeric(group["slope_delta_y_per_delta_log1p_erosion"], errors="coerce").to_numpy(dtype=float)
            lower = pd.to_numeric(group["slope_ci_low"], errors="coerce").to_numpy(dtype=float)
            upper = pd.to_numeric(group["slope_ci_high"], errors="coerce").to_numpy(dtype=float)
            ax.errorbar(effect, y, xerr=np.vstack([effect - lower, upper - effect]), fmt="o", color="#285f8f", capsize=2, markersize=3)
            ax.axvline(0, color="#777777", linewidth=0.8, linestyle="--")
            ax.set_yticks(y, [f"{row.group_column.removeprefix('gradient_')}:{row.group_label} (n={int(row.n_pairs)})" for row in group.itertuples()])
            ax.set_title(outcome, fontsize=9)
            ax.set_xlabel("Δhealth / Δlog1p(Aa) (block-clustered 95% CI)", fontsize=7)
            ax.tick_params(axis="y", labelsize=6)
            ax.tick_params(axis="x", labelsize=7)
        fig.suptitle("Most negative group-specific matched-pair slopes: M4 / Aa_20yr", fontsize=13)
        for ext in ("png", "svg", "pdf"):
            target = figure_dir / f"10_environment_group_slope_forest.{ext}"
            fig.savefig(target, dpi=220 if ext == "png" else None, bbox_inches="tight")
            figures.append(target)
        plt.close(fig)
    review_figure_dir.mkdir(parents=True, exist_ok=True)
    for source in figures:
        (review_figure_dir / source.name).write_bytes(source.read_bytes())
    data_path = figure_dir / "data" / "10_environment_group_slope_forest.csv"
    _write_csv(data_path, plot_data)
    _write_csv(review_figure_dir / "data" / data_path.name, plot_data)
    caption = "Each panel shows up to six most negative group-specific slopes for that outcome in M4/Aa_20yr. Slopes use Δlog1p(Aa) to predict high-minus-low paired health differences; 95% intervals use spatial-block cluster-robust covariance. Groups were defined from environmental variables without health outcomes; selecting the most negative groups is outcome-guided Oracle screening."
    caption_path = figure_dir / "captions" / "10_environment_group_slope_forest.md"
    caption_path.parent.mkdir(parents=True, exist_ok=True)
    caption_path.write_text(caption + "\n", encoding="utf-8")
    review_caption = review_figure_dir / "captions" / caption_path.name
    review_caption.parent.mkdir(parents=True, exist_ok=True)
    review_caption.write_text(caption + "\n", encoding="utf-8")
    return {"name": "10_environment_group_slope_forest", "formats": [path.suffix[1:] for path in figures], "data": str(data_path.relative_to(ROOT)), "caption": str(caption_path.relative_to(ROOT)), "n_rows": int(len(plot_data))}


def _build_report_addendum(
    *,
    output_dir: Path,
    review_dir: Path,
    scope_leaderboard: pd.DataFrame,
    scope_reconciliation: pd.DataFrame,
    paired_leaderboard: pd.DataFrame,
    spatial_summary: pd.DataFrame,
    group_slopes: pd.DataFrame,
    group_tests: pd.DataFrame,
    group_profiles: pd.DataFrame,
    pair_associations: pd.DataFrame,
    pair_boundary_profile: dict[str, object],
    figure: dict[str, object],
    config: dict[str, Any],
    baseline_sha: str,
) -> tuple[Path, Path]:
    pixel = scope_leaderboard.loc[
        scope_leaderboard["exposure"].eq("Aa_20yr")
        & scope_leaderboard["ranking_type"].isin(["in_scope_highest_apparent_r2", "in_scope_highest_negative_apparent_r2"])
    ].copy()
    paired = paired_leaderboard.loc[
        paired_leaderboard["method"].eq("M4")
        & paired_leaderboard["exposure"].eq("Aa_20yr")
        & paired_leaderboard["outcome"].isin(config["outcomes"])
        & paired_leaderboard["ranking_type"].isin(["highest_full_sample_paired_apparent_r2", "highest_negative_full_sample_paired_apparent_r2"])
    ].copy()
    paired = paired.sort_values(["outcome", "ranking_type"])
    spatial = spatial_summary.loc[
        spatial_summary["method"].eq("M4") & spatial_summary["exposure"].eq("Aa_20yr")
    ].copy()
    strongest = group_slopes.loc[
        group_slopes["status"].eq("ok") & group_slopes["method"].eq("M4") & group_slopes["exposure"].eq("Aa_20yr")
    ].drop(columns=["interaction_p_value", "interaction_log_p_value"], errors="ignore").merge(
        group_tests[[
            "method", "exposure", "outcome", "group_column", "interaction_log_p_value",
            "interaction_log_q_value_global",
        ]].drop_duplicates(
            ["method", "exposure", "outcome", "group_column"]
        ),
        on=["method", "exposure", "outcome", "group_column"],
        how="left",
        validate="many_to_one",
    ).sort_values("slope_delta_y_per_delta_log1p_erosion").groupby("outcome", sort=False).head(1).copy()
    primary_profiles = group_profiles.loc[
        group_profiles["method"].eq("M4") & group_profiles["exposure"].eq("Aa_20yr")
    ]
    profile_columns = [
        "group_column", "group_label", "mean_Elevation", "mean_Cur", "mean_TWI", "mean_LST",
        "mean_NDWI", "mean_NDBSI", "mean_EVI", "mean_NDVI", "mean_Precipitation",
    ]
    strongest = strongest.merge(
        primary_profiles[profile_columns],
        on=["group_column", "group_label"],
        how="left",
        validate="many_to_one",
    )
    tests_primary = group_tests.loc[
        group_tests["method"].eq("M4") & group_tests["exposure"].eq("Aa_20yr")
    ].copy()
    testable_global = group_tests.loc[group_tests["status"].eq("ok")]
    global_q_below_005 = int(
        pd.to_numeric(testable_global["interaction_q_value_global"], errors="coerce").lt(0.05).sum()
    )
    primary_q_below_005 = int(
        pd.to_numeric(tests_primary.loc[tests_primary["status"].eq("ok"), "interaction_q_value_global"], errors="coerce").lt(0.05).sum()
    )
    primary_associations = pair_associations.loc[
        pair_associations["method"].eq("M4")
        & pair_associations["exposure"].eq("Aa_20yr")
        & pair_associations["subset"].eq("all")
    ].copy()
    primary_associations["association_direction"] = np.where(
        pd.to_numeric(primary_associations["spearman_rho"], errors="coerce").abs() < 0.01,
        "near zero",
        np.where(pd.to_numeric(primary_associations["spearman_rho"], errors="coerce") < 0, "negative", "positive"),
    )
    pixel_model_exact_reconstructions = int(
        scope_leaderboard.loc[
            pd.to_numeric(scope_leaderboard["reconstructed_model_r_squared"], errors="coerce").notna(),
            ["candidate_id", "outcome", "exposure", "model_name"],
        ].drop_duplicates().shape[0]
    )
    reconstructed_rows = scope_leaderboard.loc[
        pd.to_numeric(scope_leaderboard["reconstructed_model_r_squared"], errors="coerce").notna()
    ]
    reconstruction_error = pd.to_numeric(reconstructed_rows["reconstructed_model_r_squared"], errors="coerce") - pd.to_numeric(
        reconstructed_rows["r_squared"], errors="coerce"
    )
    max_reconstruction_error = float(reconstruction_error.abs().max()) if reconstruction_error.notna().any() else float("nan")
    water_count = int(scope_reconciliation["excluded_candidate_rows"].sum())
    content = f"""## 补充完成分析（2026-10-27）

本节补足首次 Phase G 报告中的候选范围校正、配对差值非线性模型搜索和组间斜率交互检验。其余主报告结果保持可追溯；本节中的像元模型排名以此处重新限定研究范围后的榜单为准。

### 三个核心问题的直接回答

M4 / Aa_20yr 配对中，Δ健康对 Δlog1p(Aa) 的 Spearman 方向如下。AN 与 MBC 的相关系数绝对值低于 0.01，应视为近零；不能据其 P 值称为有实质关联。

{_markdown_table(primary_associations, [("outcome", "指标"), ("n_pairs", "配对数"), ("mean_delta_y", "高减低平均差"), ("negative_delta_y_fraction", "Δ健康<0比例"), ("spearman_rho", "Spearman ρ"), ("association_direction", "方向"), ("standardized_slope", "标准化斜率")])}

对 20 年 Aa 的像元级表观搜索，最高 R² 候选按预测方向既有负向，也有正向（例如 CEC 为正）；完整校正后逐指标清单如下。2020 年 Aa 的同类结果另存于完整榜单中。最佳像元模型使用环境预测变量，R² 不等于侵蚀的独立解释贡献。配对差值模型的全样本最高 R²较低，M4/Aa_20yr 范围约为 0.0001–0.109；按 validation 选出的空间模型 test 中位 R²集中在 0 附近，仅部分结局在十种子摘要中略高于 0。环境组中最负斜率的指标与分组方法并不一致，见下面的组间检验和环境画像。

### 候选范围校正

重排前先从历史 Oracle 表观模型候选中排除候选 ID 明确标注为非耕地主要土类的记录；其中 SoilType 85 为水体。本次排除 {water_count} 条结果行。总体候选数和逐期、逐指标的保留比例见 `tables/completion/candidate_scope_reconciliation.csv`。更正后的 Aa_20yr 排名如下：

{_markdown_table(pixel, [("outcome", "指标"), ("ranking_type", "排名目标"), ("candidate_id", "候选"), ("model_name", "模型"), ("r_squared", "表观 R²"), ("direction", "方向"), ("direction_basis", "方向证据"), ("n", "像元数"), ("candidate_count", "范围内候选数")])}

表观排名仍是 Oracle 搜索结果，且像元健康层是预测栅格。它不代表 Aa 的独立解释贡献。原排行榜中 pH 的部分最高值来自水体 85，该结果不纳入耕地解释；请以 `tables/completion/in_scope_pixel_model_leaderboard.csv` 为准。对可能改变负向 R² 排名的 stage-2 树模型，重建了固定抽样、变量、模型族和随机种子的表观拟合；{pixel_model_exact_reconstructions} 个候选方向的源 R² 均已复现，最大绝对差为 {max_reconstruction_error:.2e}。记录见 `tables/completion/pixel_best_candidate_direction_reconstruction.csv`；未被重建或没有曲线依据的方向仍标为 `unknown`。

### 配对差值模型

分析单位为既有 Phase F 配对。预测量为 `log1p(Aa_high)-log1p(Aa_low)`，响应为 `health_high-health_low`。每个 M3/M4 × 两期 Aa × 九指标组合先对固定随机样本搜索 {len(config['pair_model_search']['model_families'])} 个模型族，再把表观 R² 最优模型及负向候选在该组合全部可用配对上重拟合；并单列全配对 OLS 基线。抽样探索、全配对重拟合和空间留出有不同 `fit_scope`，不得混合解释。

M4 / Aa_20yr 的全配对重拟合领先候选如下。方向由模型拟合曲线在观测暴露差 2%–98% 范围内的变化确定；高 R² 仍是表观拟合：

{_markdown_table(paired, [("outcome", "指标"), ("ranking_type", "排名目标"), ("model_family", "模型族"), ("r_squared", "全配对表观 R²"), ("direction", "方向"), ("spearman_rho", "实际差值 Spearman ρ"), ("n_pairs", "配对数"), ("selection_objectives", "筛选目标")])}

### 空间分块模型验证

每种配对方式与 Aa 时期从固定 5,000 对样本中按配对中点的 5 km 空间块做十种子 60/20/20 分配；模型只按各种子 validation R² 选择，然后才计算 test 指标。下表的 test 中位数和四分位距是诊断汇总，没有参与排序。约 {100*float(pair_boundary_profile['cross_5km_member_block_fraction']):.1f}% 的高、低侵蚀成员落在不同的 5 km 网格块，但配对整体按中点分块，因此边界附近仍可能带入相邻块空间依赖。加上历史 Oracle 已接触全研究区，这些是内部空间稳定性检查，不是独立外部验证。

{_markdown_table(spatial, [("outcome", "指标"), ("selection_type", "验证选择规则"), ("selected_model_families", "入选模型族计数"), ("median_validation_r_squared", "验证 R² 中位数"), ("median_test_r_squared_report_only", "测试 R² 中位数（报告）"), ("test_r_squared_iqr_low", "测试 R² Q1"), ("test_r_squared_iqr_high", "测试 R² Q3"), ("spatial_seed_count", "空间种子数")])}

### 环境组斜率差异

保留原有 SoilType、Phase F V5 K-means k=6、Ward k=6 和六种环境梯度分组；另以 Elevation 与 LST 建立土类内 MiniBatch K-means k=4、6、8、10、12。分组标签只使用环境变量和 SoilType，之后再读取配对健康差值。每组报告 5 km 空间块聚类稳健斜率区间，并对各组法的整体交互 Wald 检验做全局 BH 校正。交互检验表覆盖两种匹配方法、两期 Aa、九项结局和所有分组；低于预设 100 对或 5 个空间块的组合保留为不足状态。全局 BH 校正共 {len(testable_global)} 项，其中 q<0.05 的有 {global_q_below_005} 项；M4 / Aa_20yr 可检验的 {int(tests_primary['status'].eq('ok').sum())} 项中有 {primary_q_below_005} 项 q<0.05。报告直接展示 log(P) 和 log(q)，避免极小尾概率被浮点下限误读。

M4 / Aa_20yr 下各指标斜率最负的环境组候选如下，供探索和审查，不作为正式分区：

{_markdown_table(strongest, [("outcome", "指标"), ("group_column", "分组法"), ("group_label", "组别"), ("n_pairs", "配对数"), ("slope_delta_y_per_delta_log1p_erosion", "斜率"), ("slope_ci_low", "聚类稳健 95% CI 下限"), ("slope_ci_high", "聚类稳健 95% CI 上限"), ("interaction_log_p_value", "整体交互 log(P)"), ("interaction_log_q_value_global", "全局 BH log(q)"), ("mean_Elevation", "组内平均高程"), ("mean_TWI", "组内平均 TWI"), ("mean_LST", "组内平均 LST"), ("mean_NDVI", "组内平均 NDVI")])}

环境特征均基于高、低侵蚀配对成员均值，补充报告均值、中位数、标准差和四分位数，详见 `tables/completion/environment_group_profiles.csv`。整体组间斜率差异见 `tables/completion/group_slope_interaction_tests.csv`；其中 M4 / Aa_20yr 的可检验组法数为 {int(tests_primary['status'].eq('ok').sum())}，完整候选数为 {len(tests_primary)}。

新增图 10 是各指标最负组斜率的森林图，展示聚类稳健区间并清楚标出 Oracle 结果导向筛选。文件为 `figures/completion/10_environment_group_slope_forest.png`、`.svg`、`.pdf`，源数据和图注同目录的 `data/`、`captions/`。

### 两类响应曲线

按用户要求，原图 5 同时保留两类曲线：固定 20,000 像元共同样本拟合的自然样条，以及由全样本 1,024 次 / 13 个模型族表观 R² 搜索选出的候选曲线。两者分别标出样本范围和选择机制；后者不代表独立验证，也不用于替代新增的配对差值模型。原图文件在 `figures/05_response_curves.*`，底层曲线表在 `tables/response_curve_data.csv`。

### 补充文件与运行来源

- 运行 ID：`{config['run_id']}`；模式：`EXPLORATORY_ORACLE`；本次运行基线 HEAD：`{baseline_sha}`。
- 独立运行输出：`{output_dir.relative_to(ROOT).as_posix()}/`；审查副本：`tables/completion/`。
- 新增候选榜单、全配对模型搜索、空间验证、环境组斜率、环境剖面、实验登记和产物清单均在上述目录。
- 输入使用既有对齐数据库、Phase F 配对与 Oracle 候选登记；继续直接使用预计算 Aa，不重新验证 RUSLE 乘积，不运行 PCNM，不更改总纲。
- `CHATGPT_REVIEW_REQUEST.md` 已追加本次补充材料路径。提交 SHA 仍需用户本地提交后填写；审查材料没有发送给外部。
"""
    addendum = review_dir / "PHASE_G_COMPLETION_ADDENDUM.md"
    addendum.write_text(content, encoding="utf-8")
    main_report = review_dir / "PHASE_G_RESEARCH_REPORT.md"
    _append_section(main_report, "## 补充材料\n\n完整补充分析、校正后的排名和新增空间模型结果见 [PHASE_G_COMPLETION_ADDENDUM.md](PHASE_G_COMPLETION_ADDENDUM.md)。其中的耕地范围校正结果覆盖本报告第 5 节中受 SoilType 85 水体候选影响的表观排名。")
    request_path = review_dir / "CHATGPT_REVIEW_REQUEST.md"
    request = request_path.read_text(encoding="utf-8")
    request = request.replace("`f71aaf455e5ca9eff36dfedc0461a09125ff3d34`（不包含尚未提交的 Phase G 文件）", f"`{baseline_sha}`（本次补充运行的起始 HEAD；后续补充代码和结果尚未提交）")
    request = request.replace("`f71aaf455e5ca9eff36dfedc0461a09125ff3d34`", f"`{baseline_sha}`")
    request = request.replace("V1.1；对应本地基线提交：`f71aaf455e5ca9eff36dfedc0461a09125ff3d34`", f"V1.1；本次补充运行的本地起始提交：`{baseline_sha}`")
    completion = f"""
## 补充审查材料：Phase G completion

- 补充报告：`docs/reviews/2026-10-27/phase-g/PHASE_G_COMPLETION_ADDENDUM.md`
- 范围校正后的像元排名：`docs/reviews/2026-10-27/phase-g/tables/completion/in_scope_pixel_model_leaderboard.csv`
- 方向重建及源 R² 复核：`docs/reviews/2026-10-27/phase-g/tables/completion/pixel_best_candidate_direction_reconstruction.csv`
- 配对响应模型和空间留出：`docs/reviews/2026-10-27/phase-g/tables/completion/paired_response_model_trials.csv`、`paired_spatial_model_trials.csv`、`paired_spatial_selected_models.csv`
- 环境组斜率与交互检验：`docs/reviews/2026-10-27/phase-g/tables/completion/group_slope_estimates.csv`、`group_slope_interaction_tests.csv`、`environment_group_profiles.csv`
- 新增图：`docs/reviews/2026-10-27/phase-g/figures/completion/10_environment_group_slope_forest.*`
- 独立运行清单：`outputs/exploratory_oracle/{config['run_id']}/artifact_manifest.json`（运行目录由 Git 忽略；审查用 CSV、报告和图副本已纳入本审查目录）
- 本次补充运行起始 HEAD：`{baseline_sha}`。完整审查提交 SHA 保持待填；请在用户本地提交完成后将首个包含代码、配置和报告/表格的完整 SHA 填入第 1 节。
- 校正说明：旧 leaderboard 中 18 条 SoilType 85（水体）候选行不在主要耕地范围；补充榜单已排除它们，并覆盖旧报告中受影响的最高表观排名。
- 环境组交互：全局 BH 覆盖 {len(testable_global)} 项，其中 q<0.05 有 {global_q_below_005} 项；M4/Aa_20yr 为 {primary_q_below_005}/{int(tests_primary['status'].eq('ok').sum())}。极端 Wald 尾概率使用稳定 log(P)/log(q) 表示。
- 空间边界：{int(pair_boundary_profile['high_low_members_in_different_5km_grid_blocks']):,}/{int(pair_boundary_profile['finite_coordinate_pairs']):,} 个配对的两端跨越 5 km 网格（{100*float(pair_boundary_profile['cross_5km_member_block_fraction']):.1f}%）；验证仍按配对中点分块。
- 曲线要求：图 5 同时包含固定样本自然样条和全样本 Oracle 表观最佳模型曲线；请分别审查其样本和选择边界。
"""
    _append_section(request_path, completion)
    return addendum, request_path


def _run(config_path: Path) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("mode") != "EXPLORATORY_ORACLE" or config.get("outcome_guided_selection") is not True:
        raise ValueError("Phase G completion must explicitly use EXPLORATORY_ORACLE")
    output_dir = (ROOT / config["output_dir"]).resolve()
    if not output_dir.is_relative_to(ROOT / "outputs" / "exploratory_oracle"):
        raise ValueError("Phase G completion output must remain under outputs/exploratory_oracle")
    tables_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    report_dir = output_dir / "reports"
    state_path = output_dir / "run_state.json"
    config_sha = _sha256(config_path)
    if output_dir.exists():
        if not state_path.is_file():
            raise FileExistsError(f"Existing completion output has no resumable run state: {output_dir}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("config_sha256") != config_sha:
            raise ValueError("Cannot resume a Phase G completion run with a different configuration")
        if state.get("status") == "complete":
            raise FileExistsError(f"Phase G completion already finished: {output_dir}")
    else:
        output_dir.mkdir(parents=True)
        for directory in (tables_dir, figure_dir, report_dir):
            directory.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
        state = {
            "run_id": config["run_id"],
            "status": "running",
            "config_sha256": config_sha,
            "baseline_commit_sha": subprocess.run(
                ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
            ).stdout.strip(),
            "pair_model_completed": [],
            "pair_spatial_completed": [],
        }
        _write_json(state_path, state)
    baseline_sha = str(state["baseline_commit_sha"])
    phase_f_dir = ROOT / "outputs" / "exploratory_oracle" / str(config["phase_f_run_id"])
    source_run_dir = ROOT / "outputs" / str(config["input_run_id"])
    oracle_dir = ROOT / "outputs" / "exploratory_oracle" / str(config["oracle_source_run_id"])
    pairs_path = phase_f_dir / "pairs" / "full_sample_pairs.parquet"
    pair_columns = [
        "pair_id", "method", "exposure", "high_erosion_cell_id", "low_erosion_cell_id",
        "erosion_high", "erosion_low", "x_high", "y_high", "x_low", "y_low", "x_mid", "y_mid", "spatial_block_id", "soil_type",
        "environment_group", "descriptive_environment_group",
        *[f"mean_{name}" for name in ENVIRONMENT],
        *[f"high_{name}" for name in config["outcomes"]],
        *[f"low_{name}" for name in config["outcomes"]],
    ]
    pairs = pd.read_parquet(pairs_path, columns=pair_columns)
    bundle = load_oracle_data(source_run_dir)
    frame = bundle.frame
    high_x = pd.to_numeric(pairs["x_high"], errors="coerce").to_numpy(dtype=float)
    high_y = pd.to_numeric(pairs["y_high"], errors="coerce").to_numpy(dtype=float)
    low_x = pd.to_numeric(pairs["x_low"], errors="coerce").to_numpy(dtype=float)
    low_y = pd.to_numeric(pairs["y_low"], errors="coerce").to_numpy(dtype=float)
    coordinate_finite = np.isfinite(high_x) & np.isfinite(high_y) & np.isfinite(low_x) & np.isfinite(low_y)
    high_blocks = np.column_stack([np.floor(high_x[coordinate_finite] / 5000), np.floor(high_y[coordinate_finite] / 5000)])
    low_blocks = np.column_stack([np.floor(low_x[coordinate_finite] / 5000), np.floor(low_y[coordinate_finite] / 5000)])
    cross_member_blocks = np.any(high_blocks != low_blocks, axis=1)
    pair_boundary_profile = {
        "pairs": int(len(pairs)),
        "finite_coordinate_pairs": int(coordinate_finite.sum()),
        "high_low_members_in_different_5km_grid_blocks": int(cross_member_blocks.sum()),
        "cross_5km_member_block_fraction": float(cross_member_blocks.mean()) if len(cross_member_blocks) else float("nan"),
        "spatial_validation_pair_assignment": "pair midpoint spatial_block_id; both members are kept together as one pair",
        "block_grid": "axis-aligned 5,000 m cells using floor(x/5000), floor(y/5000)",
    }
    _write_csv(tables_dir / "pair_spatial_block_boundary.csv", pd.DataFrame([pair_boundary_profile]))
    print(f"[Phase G completion] loaded {len(frame):,} aligned cells and {len(pairs):,} Phase F pairs", flush=True)

    reconstruction_path = tables_dir / "pixel_best_candidate_direction_reconstruction.csv"
    scoped, scope_leaderboard, scope_reconciliation, reconstructions = _candidate_scope_tables(
        oracle_dir, list(config["outcomes"]), frame, reconstruction_path
    )
    _write_csv(reconstruction_path, reconstructions)
    _write_csv(tables_dir / "pixel_candidate_scope.csv", scoped)
    _write_csv(tables_dir / "candidate_scope_reconciliation.csv", scope_reconciliation)
    _write_csv(tables_dir / "in_scope_pixel_model_leaderboard.csv", scope_leaderboard)
    print(f"[Phase G completion] excluded {int(scope_reconciliation['excluded_candidate_rows'].sum())} out-of-scope apparent pixel rows", flush=True)

    model_trials, paired_leaderboard, spatial_trials, spatial_selected = _paired_model_tables(
        pairs,
        outcomes=list(config["outcomes"]),
        config=config,
        tables_dir=tables_dir,
        state=state,
        state_path=state_path,
    )
    _write_csv(tables_dir / "paired_model_leaderboard.csv", paired_leaderboard)
    spatial_summary = _summarize_spatial_selected(spatial_selected)
    _write_csv(tables_dir / "paired_spatial_selected_summary.csv", spatial_summary)
    print(f"[Phase G completion] paired response trial rows={len(model_trials):,}; spatial model trials={len(spatial_trials):,}", flush=True)

    labels_path = tables_dir / "environment_group_labels.parquet"
    meta_path = tables_dir / "environment_group_definitions.json"
    interaction_path = tables_dir / "group_slope_interaction_tests.csv"
    slope_path = tables_dir / "group_slope_estimates.csv"
    profile_path = tables_dir / "environment_group_profiles.csv"
    checkpoint_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    if (
        all(path.is_file() for path in (interaction_path, slope_path, profile_path, meta_path))
        and checkpoint_meta.get("label_schema") == "soil_qualified_v1"
        and checkpoint_meta.get("interaction_pvalue_schema") == "log_tail_v2"
    ):
        group_tests = pd.read_csv(interaction_path)
        group_slopes = pd.read_csv(slope_path)
        group_profiles = pd.read_csv(profile_path)
        group_meta = checkpoint_meta
        print("[Phase G completion] reusing checkpointed group interaction and profile tables", flush=True)
    else:
        grouped_pairs, group_meta = _environment_group_assignments(
            frame,
            pairs,
            config=config,
            phase_g_config=yaml.safe_load((ROOT / "config" / "phase_g.yaml").read_text(encoding="utf-8")),
            labels_path=labels_path,
            meta_path=meta_path,
        )
        group_columns = [str(value) for value in group_meta["group_columns"]]
        group_tests, group_slopes = compare_group_slopes(
            grouped_pairs,
            outcomes=list(config["outcomes"]),
            group_columns=group_columns,
            min_pairs=int(config["group_analysis"]["min_pairs_per_group"]),
            min_spatial_blocks=int(config["group_analysis"]["min_spatial_blocks"]),
        )
        group_profiles = _group_profile_table(grouped_pairs, group_columns)
        _write_csv(interaction_path, group_tests)
        _write_csv(slope_path, group_slopes)
        _write_csv(profile_path, group_profiles)
        group_registry = group_tests.groupby(["method", "exposure", "outcome", "group_column", "status"], dropna=False).size().rename("test_rows").reset_index()
        group_registry["run_id"] = config["run_id"]
        group_registry["mode"] = "EXPLORATORY_ORACLE"
        group_registry["outcome_guided_selection"] = True
        group_registry["selection_objective"] = "compare group-specific ΔAa slopes and identify heterogeneous environment strata"
        _write_csv(tables_dir / "group_interaction_registry.csv", group_registry)
        group_meta["interaction_pvalue_schema"] = "log_tail_v2"
        _write_json(meta_path, group_meta)
    print(f"[Phase G completion] interaction tests={len(group_tests):,}; group slope estimates={len(group_slopes):,}", flush=True)

    figure = _write_group_forest(group_slopes, figure_dir, (ROOT / config["review_dir"] / "figures" / "completion"))
    for name in [
        "pixel_candidate_scope", "candidate_scope_reconciliation", "in_scope_pixel_model_leaderboard",
        "pixel_best_candidate_direction_reconstruction",
        "pair_spatial_block_boundary",
        "paired_model_leaderboard", "paired_response_model_trials", "paired_full_sample_ols_baselines",
        "paired_spatial_model_trials", "paired_spatial_selected_models", "paired_spatial_selected_summary",
        "group_slope_interaction_tests", "group_slope_estimates", "environment_group_profiles", "group_interaction_registry",
    ]:
        source = tables_dir / f"{name}.csv"
        target = ROOT / config["review_dir"] / "tables" / "completion" / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_file():
            target.write_bytes(source.read_bytes())
    meta_source = tables_dir / "environment_group_definitions.json"
    target_meta = ROOT / config["review_dir"] / "tables" / "completion" / meta_source.name
    target_meta.parent.mkdir(parents=True, exist_ok=True)
    target_meta.write_bytes(meta_source.read_bytes())

    addendum, request_path = _build_report_addendum(
        output_dir=output_dir,
        review_dir=ROOT / config["review_dir"],
        scope_leaderboard=scope_leaderboard,
        scope_reconciliation=scope_reconciliation,
        paired_leaderboard=paired_leaderboard,
        spatial_summary=spatial_summary,
        group_slopes=group_slopes,
        group_tests=group_tests,
        group_profiles=group_profiles,
        pair_associations=pd.read_csv(ROOT / "outputs" / "exploratory_oracle" / str(config["source_phase_g_run_id"]) / "tables" / "association_results.csv"),
        pair_boundary_profile=pair_boundary_profile,
        figure=figure,
        config=config,
        baseline_sha=baseline_sha,
    )
    (report_dir / addendum.name).write_bytes(addendum.read_bytes())
    software = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": __import__("sklearn").__version__,
        "matplotlib": matplotlib.__version__,
    }
    _write_json(output_dir / "provenance.json", {
        "run_id": config["run_id"],
        "mode": "EXPLORATORY_ORACLE",
        "outcome_guided_selection": True,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "baseline_commit_sha": baseline_sha,
        "working_tree_dirty_at_run": True,
        "input_paths": {
            "aligned_environment_and_erosion": str((source_run_dir / "databases" / "environment_erosion.parquet").relative_to(ROOT)),
            "aligned_health_predictions": str((source_run_dir / "databases" / "health.parquet").relative_to(ROOT)),
            "phase_f_pairs": str(pairs_path.relative_to(ROOT)),
            "source_oracle_registry": str((oracle_dir / "tables" / "experiment_results.parquet").relative_to(ROOT)),
        },
        "input_sha256": {str(path.relative_to(ROOT)): _sha256(path) for path in [pairs_path, oracle_dir / "tables" / "experiment_results.parquet", source_run_dir / "databases" / "environment_erosion.parquet", source_run_dir / "databases" / "health.parquet"]},
        "configuration_sha256": config_sha,
        "software": software,
        "design_notes": [
            "Aa is precomputed; no RUSLE factor-product reconstruction or validation",
            "no PCNM rerun; no research charter change",
            "health layers are predicted raster surfaces, not independent field observations",
            "pixel candidates explicitly scoped to soil 85 water are excluded from the primary cropland leaderboard",
            "paired model families are discovered on a fixed sample and leaders are refit on all available pairs",
            "spatial pair models are selected on validation blocks; test metrics are report-only and internal",
        ],
    })
    state.update(
        status="complete",
        completed_at_utc=datetime.now(UTC).isoformat(),
        table_count=len(list(tables_dir.glob("*.csv"))),
        artifact_manifest=str((output_dir / "artifact_manifest.json").relative_to(ROOT)),
    )
    _write_json(state_path, state)
    manifest_files = [
        path for path in [*output_dir.rglob("*"), *(ROOT / config["review_dir"]).rglob("*")]
        if path.is_file() and path.name not in {"artifact_manifest.json", "phase_g_completion_manifest.json"}
    ]
    manifest = {
        "run_id": config["run_id"],
        "baseline_commit_sha": baseline_sha,
        "mode": "EXPLORATORY_ORACLE",
        "configuration": str(config_path.relative_to(ROOT)),
        "complete": True,
        "figure": figure,
        "files": [{"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": _sha256(path)} for path in sorted(set(manifest_files))],
    }
    _write_json(output_dir / "artifact_manifest.json", manifest)
    _write_json(ROOT / config["review_dir"] / "phase_g_completion_manifest.json", manifest)
    print(f"[Phase G completion] report: {addendum.relative_to(ROOT)}", flush=True)
    print(f"[Phase G completion] review request: {request_path.relative_to(ROOT)}", flush=True)
    print(f"[Phase G completion] output: {output_dir.relative_to(ROOT)}", flush=True)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Complete Phase G candidate scope, pair response, and group interaction analyses")
    parser.add_argument("--config", type=Path, default=Path("config/phase_g_completion.yaml"))
    args = parser.parse_args()
    path = args.config if args.config.is_absolute() else ROOT / args.config
    _run(path)


if __name__ == "__main__":
    main()
