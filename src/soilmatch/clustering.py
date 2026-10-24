"""Environment clustering stratified by soil type."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans


def cluster_environments(
    matrix: np.ndarray,
    soil_type: Sequence[object],
    *,
    n_clusters: int,
    random_state: int = 0,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Cluster each soil stratum independently and return globally unique IDs."""
    values = np.asarray(matrix, dtype=float)
    strata = np.asarray(soil_type, dtype=object)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("matrix must be a nonempty two-dimensional array")
    if len(strata) != values.shape[0]:
        raise ValueError("soil_type must contain one value per environment vector")
    if not np.isfinite(values).all():
        raise ValueError("matrix contains missing or nonfinite values")
    if pd.isna(strata).any():
        raise ValueError("soil_type strata contain missing values")
    if not isinstance(n_clusters, int) or n_clusters < 1:
        raise ValueError("n_clusters must be a positive integer")

    labels = np.full(values.shape[0], -1, dtype=np.int32)
    center_rows: list[dict[str, object]] = []
    next_group_id = 0
    for soil in pd.unique(strata):
        indices = np.flatnonzero(strata == soil)
        subset = values[indices]
        distinct_count = np.unique(subset, axis=0).shape[0]
        effective_clusters = min(n_clusters, len(indices), distinct_count)
        if effective_clusters == 1:
            local_labels = np.zeros(len(indices), dtype=np.int32)
        else:
            model = MiniBatchKMeans(
                n_clusters=effective_clusters,
                random_state=random_state,
                n_init=10,
                batch_size=max(32, 3 * effective_clusters),
                reassignment_ratio=0.0,
            )
            local_labels = model.fit_predict(subset).astype(np.int32)

        for local_label in np.unique(local_labels):
            member_mask = local_labels == local_label
            group_id = next_group_id
            next_group_id += 1
            labels[indices[member_mask]] = group_id
            center = subset[member_mask].mean(axis=0)
            record: dict[str, object] = {
                "environment_group": group_id,
                "soil_type": soil,
                "local_cluster": int(local_label),
                "n_cells": int(member_mask.sum()),
            }
            record.update({f"center_{column}": float(value) for column, value in enumerate(center)})
            center_rows.append(record)

    centers = pd.DataFrame.from_records(center_rows)
    return labels, centers
