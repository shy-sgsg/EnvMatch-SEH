"""Outcome- and exposure-aware leaderboards that preserve validation boundaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _load_registry(registry: pd.DataFrame | Path | str) -> pd.DataFrame:
    if isinstance(registry, pd.DataFrame):
        result = registry.copy()
    else:
        path = Path(registry)
        if path.suffix.lower() == ".parquet":
            result = pd.read_parquet(path)
        elif path.suffix.lower() in {".csv", ".tsv"}:
            result = pd.read_csv(path, sep="\t" if path.suffix.lower() == ".tsv" else ",")
        else:
            rows: list[dict[str, Any]] = []
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    scores = record.get("scores")
                    candidate = record.get("candidate", {})
                    base = {key: value for key, value in record.items() if key not in {"scores", "candidate"}}
                    if isinstance(scores, list) and scores:
                        rows.extend({**base, **candidate, **score} for score in scores)
                    else:
                        rows.append({**base, **candidate})
            result = pd.DataFrame.from_records(rows)
    if "health_indicator" in result:
        if "outcome" not in result:
            result["outcome"] = result["health_indicator"]
        else:
            result["outcome"] = result["outcome"].where(
                result["outcome"].notna(), result["health_indicator"]
            )
    return result


def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    if "r_squared" not in data:
        for candidate in ("apparent_r2", "test_r2", "validation_r2"):
            if candidate in data:
                data["r_squared"] = data[candidate]
                break
    if "evaluation_set" not in data:
        if "score_set" in data:
            data["evaluation_set"] = data["score_set"].map(
                lambda value: "apparent" if str(value).startswith("matched") else value
            )
        elif "score_type" in data:
            data["evaluation_set"] = data["score_type"].map(
                lambda value: "apparent" if str(value) == "apparent" else value
            )
        else:
            data["evaluation_set"] = "apparent"
    if "selection_mode" not in data:
        data["selection_mode"] = np.where(
            data.get("outcome_guided_selection", False).astype(bool)
            if isinstance(data.get("outcome_guided_selection", False), pd.Series)
            else bool(data.get("outcome_guided_selection", False)),
            "outcome_guided_oracle",
            "unscreened_full_sample",
        )
    if "outcome_guided_selection" not in data:
        data["outcome_guided_selection"] = data["selection_mode"].astype(str).str.contains("oracle|outcome", case=False)
    data["r_squared"] = pd.to_numeric(data.get("r_squared", np.nan), errors="coerce")
    if "delta_r2" in data:
        data["delta_r2"] = pd.to_numeric(data["delta_r2"], errors="coerce")
    return data


def _sort_leaderboard(frame: pd.DataFrame, metric: str, *, descending: bool = True) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    result = frame.copy()
    result[metric] = pd.to_numeric(result[metric], errors="coerce")
    result = result.loc[np.isfinite(result[metric])].copy()
    if result.empty:
        return result
    group_columns = [column for column in ("exposure", "outcome") if column in result]
    sort_columns = [*group_columns, metric]
    ascending = [True] * len(group_columns) + [not descending]
    tie_breakers = [column for column in ("candidate_id", "experiment_id", "model_name") if column in result]
    result = result.sort_values([*sort_columns, *tie_breakers], ascending=[*ascending, *([True] * len(tie_breakers))], kind="stable")
    if group_columns:
        result["rank"] = result.groupby(group_columns, sort=False).cumcount() + 1
        result["trial_count"] = result.groupby(group_columns, sort=False)[tie_breakers[0] if tie_breakers else metric].transform("nunique")
    else:
        result["rank"] = np.arange(1, len(result) + 1)
        result["trial_count"] = len(result)
    result["leaderboard_metric"] = metric
    return result.reset_index(drop=True)


def _validation_selected_test(data: pd.DataFrame) -> pd.DataFrame:
    role = data["evaluation_set"].astype(str).str.lower()
    validation_rows = data.loc[role.eq("validation")].copy()
    if "validation_r2" in data:
        explicit = pd.to_numeric(data["validation_r2"], errors="coerce")
        validation_rows = pd.concat(
            [validation_rows, data.loc[np.isfinite(explicit)].copy()],
            ignore_index=True,
        )
        stable_identity = [
            column
            for column in (
                "experiment_id",
                "candidate_id",
                "parent_candidate_id",
                "model_name",
                "exposure",
                "outcome",
                "health_indicator",
                "evaluation_set",
                "score_type",
            )
            if column in validation_rows
        ]
        if stable_identity:
            validation_rows = validation_rows.drop_duplicates(subset=stable_identity)
    if validation_rows.empty:
        return data.iloc[0:0].copy()
    validation_rows["__validation_score"] = np.where(
        validation_rows["evaluation_set"].astype(str).str.lower().eq("validation"),
        pd.to_numeric(validation_rows["r_squared"], errors="coerce"),
        pd.to_numeric(validation_rows.get("validation_r2", np.nan), errors="coerce"),
    )
    validation_rows = validation_rows.loc[np.isfinite(validation_rows["__validation_score"])].copy()
    if validation_rows.empty:
        return data.iloc[0:0].copy()

    outcome_exposure = [column for column in ("exposure", "outcome") if column in data]
    validation_rows = validation_rows.sort_values(
        [*outcome_exposure, "__validation_score"],
        ascending=[*([True] * len(outcome_exposure)), False],
        kind="stable",
    )
    selected = validation_rows.groupby(outcome_exposure, sort=False, dropna=False).head(1) if outcome_exposure else validation_rows.head(1)
    selected = selected.copy()
    selected["selected_validation_r2"] = selected["__validation_score"]

    test_rows = data.loc[role.eq("test")].copy()
    if test_rows.empty:
        return data.iloc[0:0].copy()
    output: list[pd.Series] = []
    for _, winner in selected.iterrows():
        parent_id = winner.get("parent_candidate_id")
        if pd.notna(parent_id):
            identifier = "parent_candidate_id"
        elif pd.notna(winner.get("candidate_id")):
            identifier = "candidate_id"
        else:
            identifier = "experiment_id"
        identity = [
            column
            for column in (identifier, "model_name", "exposure", "outcome")
            if column in winner.index and column in test_rows.columns and pd.notna(winner.get(column))
        ]
        match = np.ones(len(test_rows), dtype=bool)
        for column in identity:
            match &= test_rows[column].eq(winner[column]).fillna(False).to_numpy(dtype=bool)
        found = test_rows.loc[match]
        if found.empty and "test_r2" in winner and np.isfinite(pd.to_numeric(winner["test_r2"], errors="coerce")):
            chosen = winner.copy()
            chosen["r_squared"] = float(winner["test_r2"])
            chosen["evaluation_set"] = "test"
            output.append(chosen)
        elif not found.empty:
            chosen = found.iloc[0].copy()
            chosen["selected_validation_r2"] = winner["selected_validation_r2"]
            output.append(chosen)
    if not output:
        return data.iloc[0:0].copy()
    result = pd.DataFrame(output)
    result["selected_by"] = "validation_r2"
    return result.drop(columns=["__validation_score"], errors="ignore")


def _erosion_increment_board(data: pd.DataFrame) -> pd.DataFrame:
    if "delta_r2" not in data:
        return data.iloc[0:0].copy()
    delta = data.loc[np.isfinite(pd.to_numeric(data["delta_r2"], errors="coerce"))].copy()
    if delta.empty:
        return delta

    role = delta["evaluation_set"].astype(str).str.lower()
    validation = delta.loc[role.eq("validation")]
    apparent = delta.loc[role.eq("apparent")]
    group_columns = [column for column in ("exposure", "outcome") if column in delta]
    validation_groups = {
        tuple(row[column] for column in group_columns)
        for _, row in validation.iterrows()
    }
    fallback = apparent.loc[
        [tuple(row[column] for column in group_columns) not in validation_groups for _, row in apparent.iterrows()]
    ] if group_columns else apparent.iloc[0:0]
    candidates = pd.concat([validation, fallback], ignore_index=True)
    if candidates.empty:
        return candidates

    identity_columns = [
        column
        for column in ("experiment_type", "candidate_id", "model_name", "regime", "exposure", "outcome")
        if column in data
    ]
    records: list[dict[str, Any]] = []
    for _, row in candidates.iterrows():
        record = row.to_dict()
        record["selection_basis"] = "validation_delta_r2" if str(row["evaluation_set"]).lower() == "validation" else "apparent_delta_r2_fallback"
        record["validation_delta_r2"] = float(row["delta_r2"]) if str(row["evaluation_set"]).lower() == "validation" else np.nan
        record["apparent_delta_r2"] = float(row["delta_r2"]) if str(row["evaluation_set"]).lower() == "apparent" else np.nan
        record["test_delta_r2"] = np.nan
        match = data["evaluation_set"].astype(str).str.lower().eq("test")
        for column in identity_columns:
            value = row.get(column)
            if pd.notna(value):
                match &= data[column].eq(value).fillna(False)
        test = data.loc[match]
        if not test.empty and "delta_r2" in test:
            test_values = pd.to_numeric(test["delta_r2"], errors="coerce")
            if np.isfinite(test_values).any():
                record["test_delta_r2"] = float(test_values.loc[np.isfinite(test_values)].iloc[0])
        if str(row["evaluation_set"]).lower() == "validation":
            apparent_match = data["evaluation_set"].astype(str).str.lower().eq("apparent")
            for column in identity_columns:
                value = row.get(column)
                if pd.notna(value):
                    apparent_match &= data[column].eq(value).fillna(False)
            apparent_rows = data.loc[apparent_match]
            apparent_values = pd.to_numeric(apparent_rows.get("delta_r2", pd.Series(dtype=float)), errors="coerce")
            if np.isfinite(apparent_values).any():
                record["apparent_delta_r2"] = float(apparent_values.loc[np.isfinite(apparent_values)].iloc[0])
        records.append(record)
    return _sort_leaderboard(pd.DataFrame.from_records(records), "delta_r2")


def rank_leaderboards(registry: pd.DataFrame | Path | str) -> dict[str, pd.DataFrame]:
    """Build four separate boards; spatial test models are selected by validation."""

    data = _normalize(_load_registry(registry))
    if data.empty:
        empty = pd.DataFrame()
        return {"full_sample_best": empty, "oracle_apparent_best": empty, "spatial_test_best": empty, "erosion_increment_best": empty}
    if "experiment_type" not in data:
        data["experiment_type"] = ""
    experiment_type = data["experiment_type"].astype(str).str.lower()
    selection_mode = data["selection_mode"].astype(str).str.lower()
    evaluation = data["evaluation_set"].astype(str).str.lower()

    full_mask = experiment_type.str.contains("full_sample") & ~selection_mode.str.contains("oracle")
    full_mask &= evaluation.isin({"apparent", "full_sample"})
    full_sample = _sort_leaderboard(data.loc[full_mask], "r_squared")

    guided = data["outcome_guided_selection"].fillna(False).astype(bool) | selection_mode.str.contains("oracle|outcome")
    oracle_mask = guided & evaluation.isin(
        {"apparent", "train", "validation", "matched_cells", "matched_pairs", "outcome_selected_oracle"}
    )
    if "scope_status" in data:
        oracle_mask &= data["scope_status"].fillna("in_scope").astype(str).eq("in_scope")
    oracle_apparent = _sort_leaderboard(data.loc[oracle_mask], "r_squared")

    spatial_model_rows = data.loc[experiment_type.str.contains("spatial_model", regex=False)]
    spatial_test = _sort_leaderboard(_validation_selected_test(spatial_model_rows), "r_squared")

    delta = _erosion_increment_board(data)
    return {
        "full_sample_best": full_sample,
        "oracle_apparent_best": oracle_apparent,
        "spatial_test_best": spatial_test,
        "erosion_increment_best": delta,
    }
