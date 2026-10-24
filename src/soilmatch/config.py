"""YAML configuration loading and project-relative path resolution."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a mapping-valued YAML project configuration."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Configuration file not found: {source}")
    with source.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Configuration root must be a YAML mapping: {source}")
    return config


def project_root(config_path: str | Path) -> Path:
    """Resolve the root from a config path in the conventional config folder."""
    parent = Path(config_path).resolve().parent
    return parent.parent if parent.name == "config" else parent


def resolve_path(value: str | Path | None, config_path: str | Path) -> Path | None:
    """Resolve a configured path relative to the project root."""
    if value is None:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else project_root(config_path) / path


def load_data_mapping(
    config: Mapping[str, Any], config_path: str | Path
) -> tuple[dict[str, Any], Path]:
    """Read the local logical-name-to-file mapping without hard-coded paths."""
    data_config = config.get("data", {})
    mapping_value = data_config.get("mapping_file", "config/data_paths.yaml")
    mapping_path = resolve_path(mapping_value, config_path)
    if mapping_path is None:
        raise ValueError("data.mapping_file must name a YAML file")
    if not mapping_path.is_file():
        example = mapping_path.with_name("data_paths.example.yaml")
        raise FileNotFoundError(
            f"Missing data path mapping file: {mapping_path}. "
            f"Copy {example} to {mapping_path} and fill in the available input files."
        )
    with mapping_path.open(encoding="utf-8") as stream:
        mapping = yaml.safe_load(stream) or {}
    if not isinstance(mapping, dict):
        raise ValueError(f"Data path mapping root must be a YAML mapping: {mapping_path}")
    return mapping, mapping_path
