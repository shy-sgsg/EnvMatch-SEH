"""Load the existing aligned databases without applying a global validity mask."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_EXPOSURES = ("Aa_20yr", "Aa_2020")
HEALTH_OUTCOMES = ("AN", "OM", "CEC", "pH", "SW", "BD", "MBC", "RESP")
OPTIONAL_OUTCOMES = ("SHI_candidate",)
COORDINATE_COLUMNS = ("row", "col", "x", "y")


@dataclass(frozen=True)
class DataBundle:
    frame: pd.DataFrame
    input_run_id: str
    source_run_dir: Path
    exposures: tuple[str, ...]
    outcomes: tuple[str, ...]
    outcome_counts: dict[str, int]


def _read_unique_table(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label} Parquet table: {path}")
    frame = pd.read_parquet(path)
    if "cell_id" not in frame.columns:
        raise ValueError(f"{label} table is missing required cell_id column: {path}")
    if frame["cell_id"].isna().any():
        raise ValueError(f"{label} table contains null cell_id values: {path}")
    duplicates = frame.loc[frame["cell_id"].duplicated(keep=False), "cell_id"]
    if not duplicates.empty:
        example = duplicates.iloc[0]
        raise ValueError(f"{label} table requires unique cell_id values; duplicate cell_id {example!r}")
    return frame


def _validate_coordinates(environment: pd.DataFrame, health: pd.DataFrame) -> None:
    common = [name for name in COORDINATE_COLUMNS if name in environment and name in health]
    if common != list(COORDINATE_COLUMNS):
        raise ValueError("Both Parquet tables must include row, col, x, and y for spatial alignment")

    joined = environment[["cell_id", *common]].merge(
        health[["cell_id", *common]],
        on="cell_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_environment", "_health"),
        sort=False,
    )
    for name in common:
        left = joined[f"{name}_environment"].to_numpy()
        right = joined[f"{name}_health"].to_numpy()
        if name in ("x", "y"):
            aligned = np.isfinite(left) & np.isfinite(right) & np.isclose(
                left.astype(float), right.astype(float), rtol=0.0, atol=1e-6
            )
        else:
            aligned = pd.Series(left).eq(pd.Series(right)).fillna(False).to_numpy()
        if not aligned.all():
            cell_id = joined.loc[~aligned, "cell_id"].iloc[0]
            raise ValueError(f"Spatial alignment mismatch in {name} for cell_id {cell_id!r}")


def load_oracle_data(run_dir: Path) -> DataBundle:
    """Read the existing environment and health tables and require an exact 1:1 join.

    ``valid_cell`` flags are retained separately and never applied as a global filter;
    complete cases are determined for each outcome at analysis time.
    """

    source_run_dir = Path(run_dir)
    database_dir = source_run_dir / "databases"
    environment = _read_unique_table(database_dir / "environment_erosion.parquet", "environment_erosion")
    health = _read_unique_table(database_dir / "health.parquet", "health")

    environment_ids = pd.Index(environment["cell_id"])
    health_ids = pd.Index(health["cell_id"])
    if len(environment_ids) != len(health_ids) or not environment_ids.isin(health_ids).all():
        missing_health = environment_ids[~environment_ids.isin(health_ids)].tolist()[:3]
        missing_environment = health_ids[~health_ids.isin(environment_ids)].tolist()[:3]
        raise ValueError(
            "environment_erosion and health cell_id sets must match exactly; "
            f"missing from health={missing_health}, missing from environment={missing_environment}"
        )

    _validate_coordinates(environment, health)

    required_environment = [*EXPECTED_EXPOSURES]
    missing_exposures = [name for name in required_environment if name not in environment]
    if missing_exposures:
        raise ValueError(f"environment_erosion table is missing exposure columns: {missing_exposures}")
    missing_health = [name for name in HEALTH_OUTCOMES if name not in health]
    if missing_health:
        raise ValueError(f"health table is missing required health outcomes: {missing_health}")

    env = environment.copy()
    hlt = health.copy()
    if "valid_cell" in env:
        env = env.rename(columns={"valid_cell": "environment_valid_cell"})
    if "valid_cell" in hlt:
        hlt = hlt.rename(columns={"valid_cell": "health_valid_cell"})

    shared = (set(env.columns) & set(hlt.columns)) - {"cell_id"}
    unexpected_shared = shared - set(COORDINATE_COLUMNS)
    if unexpected_shared:
        raise ValueError(f"Unexpected overlapping data columns besides coordinates: {sorted(unexpected_shared)}")
    hlt = hlt.drop(columns=list(COORDINATE_COLUMNS), errors="ignore")
    frame = env.merge(hlt, on="cell_id", how="inner", validate="one_to_one", sort=False)

    exposures = tuple(name for name in EXPECTED_EXPOSURES if name in frame)
    outcomes = tuple(name for name in (*OPTIONAL_OUTCOMES, *HEALTH_OUTCOMES) if name in frame)
    outcome_counts = {name: int(frame[name].notna().sum()) for name in outcomes}
    return DataBundle(
        frame=frame,
        input_run_id=source_run_dir.name,
        source_run_dir=source_run_dir,
        exposures=exposures,
        outcomes=outcomes,
        outcome_counts=outcome_counts,
    )
