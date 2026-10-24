"""Input mapping utilities and tabular readers."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path

import pandas as pd


_DATA_SUFFIXES = {".tif", ".tiff", ".csv", ".xlsx", ".xls", ".gpkg", ".geojson", ".shp", ".parquet"}


def flatten_path_mapping(mapping: Mapping[str, object]) -> Iterator[tuple[str, str]]:
    """Yield dotted logical names and path strings from a nested YAML mapping."""
    def visit(value: object, prefix: str) -> Iterator[tuple[str, str]]:
        if isinstance(value, Mapping):
            for key, child in value.items():
                name = f"{prefix}.{key}" if prefix else str(key)
                yield from visit(child, name)
        elif isinstance(value, (str, Path)):
            path = Path(value)
            if path.suffix.lower() in _DATA_SUFFIXES:
                yield prefix, str(value)

    yield from visit(mapping, "")


def read_table(path: str | Path, *, sheet_name: str | int | None = 0) -> pd.DataFrame:
    """Read a CSV, Excel, or Parquet table based on its suffix."""
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(source, encoding="utf-8-sig")
    if suffix in {".xlsx", ".xls"}:
        selected_sheet = 0 if sheet_name is None else sheet_name
        return pd.read_excel(source, sheet_name=selected_sheet)
    if suffix == ".parquet":
        return pd.read_parquet(source)
    raise ValueError(f"Unsupported table format: {source.suffix}")
