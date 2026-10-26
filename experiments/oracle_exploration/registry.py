"""Stable run IDs, append-only experiment records, and search config validation."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from experiments.oracle_exploration.data import HEALTH_OUTCOMES

REQUIRED_RECORD_FIELDS = (
    "experiment_id",
    "git_sha",
    "input_run_id",
    "exposure",
    "health_indicator",
    "outcome_guided_selection",
    "selection_outcomes",
    "selection_objective",
    "random_seed",
    "status",
)


def make_experiment_id(date: str, sequence: int, config_id: str) -> str:
    """Build a stable, path-safe Oracle experiment identifier."""

    if not isinstance(sequence, int) or sequence < 0:
        raise ValueError("sequence must be a non-negative integer")
    for label, value in (("date", date), ("config_id", config_id)):
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9-]+", value):
            raise ValueError(f"{label} must contain only letters, digits, and hyphens")
    return f"oracle_{date}_{sequence:04d}_{config_id}"


def _json_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item") and callable(value.item):
        return _json_value(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    raise TypeError(f"Unsupported experiment record value type: {type(value).__name__}")


def record_experiment(registry_path: Path, record: Mapping[str, object]) -> None:
    """Append one complete success/failure experiment row as a JSONL record."""

    missing = [name for name in REQUIRED_RECORD_FIELDS if name not in record]
    if missing:
        raise ValueError(f"experiment record is missing required fields: {', '.join(missing)}")
    if not isinstance(record["outcome_guided_selection"], bool):
        raise ValueError("outcome_guided_selection must be a boolean so baselines can be marked explicitly")
    if not isinstance(record["selection_outcomes"], (list, tuple)) or not record["selection_outcomes"]:
        raise ValueError("selection_outcomes must be a non-empty list")
    if not isinstance(record["selection_objective"], str) or not record["selection_objective"].strip():
        raise ValueError("selection_objective must be a non-empty string")
    if record["status"] not in {"success", "failed"}:
        raise ValueError("status must be 'success' or 'failed'")
    if record["status"] == "failed" and not record.get("error"):
        raise ValueError("failed experiment records must include an error cause")

    path = Path(registry_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(_json_value(dict(record)), ensure_ascii=False, sort_keys=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
        stream.flush()


def validate_search_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the required mode boundary and staged search ranges."""

    if not isinstance(config, Mapping):
        raise ValueError("search config must be a mapping")
    if config.get("mode") != "EXPLORATORY_ORACLE":
        raise ValueError("mode must be EXPLORATORY_ORACLE")
    if config.get("outcome_guided_selection") is not True:
        raise ValueError("outcome_guided_selection must be true for Oracle runs")
    if not config.get("input_run_id"):
        raise ValueError("input_run_id is required")
    if not isinstance(config.get("seed"), int):
        raise ValueError("seed must be an integer")
    output_root = Path(str(config.get("output_root", "")))
    if output_root.is_absolute() or output_root.parts[:2] != ("outputs", "exploratory_oracle"):
        raise ValueError("output_root must stay under outputs/exploratory_oracle")

    exposures = config.get("exposures")
    if not isinstance(exposures, list) or not set(("Aa_20yr", "Aa_2020")).issubset(exposures):
        raise ValueError("exposures must include Aa_20yr and Aa_2020")
    outcomes = config.get("outcomes")
    required_outcomes = set(HEALTH_OUTCOMES)
    if not isinstance(outcomes, list) or not required_outcomes.issubset(outcomes):
        raise ValueError("outcomes must include the eight health indicators")

    stages = config.get("stages")
    if not isinstance(stages, Mapping):
        raise ValueError("stages must be a mapping")
    stage1 = stages.get("stage1", {})
    sample_size = stage1.get("sample_size")
    candidates = stage1.get("candidates_per_exposure")
    if not isinstance(sample_size, int) or not 30000 <= sample_size <= 50000:
        raise ValueError("stage1.sample_size must be between 30000 and 50000")
    if not isinstance(candidates, int) or not 100 <= candidates <= 300:
        raise ValueError("stage1.candidates_per_exposure must be between 100 and 300")

    stage2 = stages.get("stage2", {})
    max_configs = stage2.get("max_configs")
    stage2_size = stage2.get("sample_size")
    if not isinstance(max_configs, int) or not 1 <= max_configs <= 20:
        raise ValueError("stage2.max_configs must be between 1 and 20")
    if not isinstance(stage2_size, int) or not sample_size <= stage2_size <= 100000:
        raise ValueError("stage2.sample_size must be between stage1.sample_size and 100000")

    stage3 = stages.get("stage3", {})
    if stage3.get("block_size_m") != 5000:
        raise ValueError("stage3.block_size_m must be 5000")
    split = stage3.get("split")
    if not isinstance(split, Mapping) or set(split) != {"train", "validation", "test"}:
        raise ValueError("stage3.split must define train, validation, and test proportions")
    if any(not isinstance(value, (int, float)) or value <= 0 for value in split.values()):
        raise ValueError("stage3 split proportions must be positive numbers")
    if abs(sum(split.values()) - 1.0) > 1e-9:
        raise ValueError("stage3 split proportions must sum to 1")

    matching = config.get("matching")
    if not isinstance(matching, Mapping):
        raise ValueError("matching must be a mapping")
    domains = (
        ("environment_calipers", (0.15, 0.25, 0.35, 0.5, 0.75, 1.0)),
        ("geographic_radii_m", (100, 200, 300, 500, 750, 1000, 1500)),
        ("erosion_rank_gaps", (0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)),
    )
    for key, expected_values in domains:
        values = matching.get(key)
        if not isinstance(values, list) or tuple(values) != expected_values:
            raise ValueError(f"matching.{key} must match the authorized candidate domain")
    if not matching.get("distance_methods") or not matching.get("cluster_methods"):
        raise ValueError("matching distance_methods and cluster_methods must be non-empty")

    return dict(config)


def load_search_config(config_path: Path) -> dict[str, Any]:
    path = Path(config_path)
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    return validate_search_config(config)
