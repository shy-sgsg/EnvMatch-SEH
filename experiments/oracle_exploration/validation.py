"""Spatial role assignment and test-isolated candidate selection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from soilmatch.spatial_statistics import make_spatial_blocks


def spatial_roles(
    frame: pd.DataFrame,
    block_size_m: float = 5000,
    seed: int = 20261026,
) -> pd.Series:
    """Assign entire projected spatial blocks to train/validation/test roles."""

    missing = [name for name in ("x", "y") if name not in frame]
    if missing:
        raise ValueError(f"Missing projected coordinate columns: {missing}")
    block_ids = make_spatial_blocks(frame["x"], frame["y"], block_size_m)
    if len(frame) < 3:
        raise ValueError("At least three spatial blocks are required for a train/validation/test split")
    unique_blocks = np.unique(block_ids)
    if len(unique_blocks) < 3:
        raise ValueError("At least three spatial blocks are required for a train/validation/test split")

    permutation = np.random.default_rng(seed).permutation(unique_blocks)
    train_count = min(len(unique_blocks) - 2, max(1, int(round(0.60 * len(unique_blocks)))))
    validation_count = min(
        len(unique_blocks) - train_count - 1,
        max(1, int(round(0.20 * len(unique_blocks)))),
    )
    train_blocks = set(permutation[:train_count])
    validation_blocks = set(permutation[train_count : train_count + validation_count])
    role_by_block = {
        **{block: "train" for block in train_blocks},
        **{block: "validation" for block in validation_blocks},
        **{block: "test" for block in permutation[train_count + validation_count :]},
    }
    roles = pd.Series([role_by_block[block] for block in block_ids], index=frame.index, name="spatial_role")
    return roles


def select_validation_candidates(
    candidates: pd.DataFrame,
    *,
    max_configs: int,
    score_column: str = "validation_r2",
) -> pd.DataFrame:
    """Select candidate configurations by training/validation scores only."""

    if score_column not in {"train_r2", "validation_r2"}:
        raise ValueError("Candidate selection may use train_r2 or validation_r2; test scores are sealed")
    if score_column not in candidates:
        raise ValueError(f"Missing candidate selection score column: {score_column}")
    if not isinstance(max_configs, int) or max_configs < 1:
        raise ValueError("max_configs must be a positive integer")
    score_values = pd.to_numeric(candidates[score_column], errors="coerce")
    valid = candidates.loc[np.isfinite(score_values)].copy()
    valid[score_column] = score_values.loc[valid.index]
    sort_columns = [score_column]
    ascending = [False]
    if "candidate_id" in valid:
        sort_columns.append("candidate_id")
        ascending.append(True)
    return valid.sort_values(sort_columns, ascending=ascending, kind="stable").head(max_configs).reset_index(drop=True)
