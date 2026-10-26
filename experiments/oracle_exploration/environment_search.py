"""Environment-only variable, distance, and clustering candidates for Oracle runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

ENVIRONMENT_FEATURES = (
    "Elevation",
    "Cur",
    "TWI",
    "LST",
    "NDWI",
    "NDBSI",
    "EVI",
    "NDVI",
    "Precipitation",
)
PRIMARY_SOIL_CODES = (26, 27, 42)
CORRELATED_VARIABLE_GROUPS = (("NDWI", "NDBSI", "EVI", "NDVI"), ("Elevation", "Precipitation"))
NAMED_ENVIRONMENT_SETS = {
    "V1": ("Elevation", "Cur", "TWI", "LST"),
    "V2": ("Elevation", "TWI", "LST"),
    "V3": ("Elevation", "Cur", "TWI"),
    "V4": ("Cur", "TWI", "LST"),
    "V5": ("Elevation", "LST"),
    "V6": ("Elevation", "TWI"),
    "V7": ("Elevation", "Cur", "TWI", "LST", "NDWI"),
    "V8": ("Elevation", "Cur", "TWI", "LST", "EVI"),
}
MAX_GMM_FIT_ROWS = 5000
MAX_AGGLOMERATIVE_FIT_ROWS = 1000


def candidate_environment_sets(
    columns: Sequence[str] | pd.Index,
    *,
    seed: int,
    n_random: int,
    correlated_variable_groups: Sequence[Sequence[str]] = CORRELATED_VARIABLE_GROUPS,
) -> list[tuple[str, tuple[str, ...]]]:
    """Return available V1–V8 vectors and seeded, correlation-aware random sets."""

    if not isinstance(n_random, int) or n_random < 0:
        raise ValueError("n_random must be a non-negative integer")
    available_columns = set(columns)
    correlation_groups = tuple(tuple(group) for group in correlated_variable_groups)
    unknown_correlation_features = sorted(
        {name for group in correlation_groups for name in group} - set(ENVIRONMENT_FEATURES)
    )
    if unknown_correlation_features:
        raise ValueError(f"Correlation groups contain unknown environment features: {unknown_correlation_features}")
    named: list[tuple[str, tuple[str, ...]]] = [
        (name, variables)
        for name, variables in NAMED_ENVIRONMENT_SETS.items()
        if set(variables).issubset(available_columns)
    ]
    eligible = [name for name in ENVIRONMENT_FEATURES if name in available_columns]
    named_feature_sets = {variables for _, variables in named}
    random_candidates: list[tuple[str, ...]] = []
    for size in range(2, len(eligible) + 1):
        for subset in combinations(eligible, size):
            members = set(subset)
            if subset in named_feature_sets:
                continue
            if any(len(members.intersection(group)) > 1 for group in correlation_groups):
                continue
            random_candidates.append(subset)

    rng = np.random.default_rng(seed)
    if random_candidates:
        order = rng.permutation(len(random_candidates))[:n_random]
        random_sets = [random_candidates[int(index)] for index in order]
    else:
        random_sets = []
    random_named = [
        (f"R{index:03d}", variables) for index, variables in enumerate(random_sets, start=1)
    ]
    return [*named, *random_named]


def _feature_matrix(frame: pd.DataFrame, variables: Sequence[str]) -> np.ndarray:
    if not variables:
        raise ValueError("At least one environment feature is required")
    if len(set(variables)) != len(variables):
        raise ValueError("Environment feature names must be unique")
    forbidden = sorted(set(variables) - set(ENVIRONMENT_FEATURES))
    if forbidden:
        raise ValueError(f"Only allowlisted environment variables may define distance features: {forbidden}")
    missing = [name for name in variables if name not in frame]
    if missing:
        raise ValueError(f"Missing environment features: {missing}")
    return frame.loc[:, list(variables)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


def _weight_values(
    variables: Sequence[str], weights: Mapping[str, float] | Sequence[float] | np.ndarray | None
) -> np.ndarray:
    if weights is None:
        values = np.ones(len(variables), dtype=float)
    elif isinstance(weights, Mapping):
        unknown = set(weights) - set(variables)
        if unknown:
            raise ValueError(f"Weights reference unknown environment features: {sorted(unknown)}")
        values = np.asarray([weights.get(name, 1.0) for name in variables], dtype=float)
    else:
        values = np.asarray(weights, dtype=float)
    if values.shape != (len(variables),) or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("weights must be finite positive values with one value per environment feature")
    return values


def transform_environment_features(
    frame: pd.DataFrame,
    variables: Sequence[str],
    *,
    method: str,
    weights: Mapping[str, float] | Sequence[float] | np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Transform continuous environment variables for the requested distance."""

    raw = _feature_matrix(frame, variables)
    if not np.isfinite(raw).all():
        raise ValueError("Environment features contain missing or nonfinite values")
    scaler = StandardScaler()
    standardized = scaler.fit_transform(raw)
    weight_values = _weight_values(variables, weights)
    regularization = 0.0
    retained_components = standardized.shape[1]

    if method == "standardized_euclidean":
        transformed = standardized
    elif method == "weighted_euclidean":
        transformed = standardized * np.sqrt(weight_values)
    elif method == "mahalanobis":
        covariance = np.atleast_2d(np.cov(standardized, rowvar=False, ddof=1))
        scale = max(float(np.trace(covariance) / covariance.shape[0]), 1.0)
        regularization = 1e-6 * scale
        regularized = covariance + regularization * np.eye(covariance.shape[0])
        eigenvalues, eigenvectors = np.linalg.eigh(regularized)
        whitening = eigenvectors @ np.diag(1.0 / np.sqrt(np.maximum(eigenvalues, regularization)))
        whitening = whitening @ eigenvectors.T
        transformed = standardized @ whitening
    elif method == "pca_euclidean":
        pca = PCA(n_components=0.95, svd_solver="full")
        transformed = pca.fit_transform(standardized)
        retained_components = transformed.shape[1]
    elif method == "direct_nearest_neighbor":
        transformed = standardized
    else:
        raise ValueError(
            "method must be standardized_euclidean, weighted_euclidean, mahalanobis, "
            "pca_euclidean, or direct_nearest_neighbor"
        )

    metadata: dict[str, object] = {
        "method": method,
        "features": list(variables),
        "weights": dict(zip(variables, weight_values.tolist(), strict=True)),
        "means": dict(zip(variables, scaler.mean_.tolist(), strict=True)),
        "scales": dict(zip(variables, scaler.scale_.tolist(), strict=True)),
        "regularization": regularization,
        "retained_components": retained_components,
        "n_rows": len(frame),
    }
    return np.asarray(transformed, dtype=float), metadata


def soil_category_counts(frame: pd.DataFrame, *, column: str = "SoilType") -> pd.DataFrame:
    """Count all observed soil codes and mark the primary soil strata."""

    if column not in frame:
        raise ValueError(f"Missing soil category column: {column}")
    counts = frame[column].value_counts(dropna=False, sort=False)
    records = [
        {
            "soil_type": code,
            "n_cells": int(count),
            "primary_soil_stratum": bool(code in PRIMARY_SOIL_CODES),
        }
        for code, count in counts.items()
    ]
    return pd.DataFrame.from_records(records).sort_values(
        "soil_type", key=lambda values: values.astype(str), kind="stable", ignore_index=True
    )


def _bounded_sample_indices(size: int, limit: int, seed: int) -> np.ndarray:
    if size <= limit:
        return np.arange(size, dtype=int)
    return np.sort(np.random.default_rng(seed).choice(size, size=limit, replace=False))


def _cluster_one_group(matrix: np.ndarray, method: str, k: int, seed: int) -> np.ndarray:
    distinct = np.unique(matrix, axis=0).shape[0]
    effective_k = min(k, len(matrix), distinct)
    if effective_k <= 1:
        return np.zeros(len(matrix), dtype=np.int32)
    if method == "minibatch_kmeans":
        model = MiniBatchKMeans(
            n_clusters=effective_k,
            random_state=seed,
            n_init=3,
            batch_size=min(2048, max(64, effective_k * 32)),
            reassignment_ratio=0.0,
        )
        return model.fit_predict(matrix).astype(np.int32)
    if method == "kmeans":
        model = KMeans(n_clusters=effective_k, random_state=seed, n_init=3, max_iter=100)
        return model.fit_predict(matrix).astype(np.int32)
    if method == "gaussian_mixture":
        sample_indices = _bounded_sample_indices(len(matrix), MAX_GMM_FIT_ROWS, seed)
        model = GaussianMixture(
            n_components=effective_k,
            covariance_type="diag",
            reg_covar=1e-5,
            max_iter=100,
            random_state=seed,
        )
        model.fit(matrix[sample_indices])
        return model.predict(matrix).astype(np.int32)
    if method == "agglomerative":
        sample_indices = _bounded_sample_indices(len(matrix), MAX_AGGLOMERATIVE_FIT_ROWS, seed)
        sample = matrix[sample_indices]
        model = AgglomerativeClustering(n_clusters=effective_k, linkage="ward")
        sample_labels = model.fit_predict(sample).astype(np.int32)
        centers = np.vstack([sample[sample_labels == label].mean(axis=0) for label in range(effective_k)])
        distances = np.square(matrix[:, None, :] - centers[None, :, :]).sum(axis=2)
        return np.argmin(distances, axis=1).astype(np.int32)
    raise ValueError(f"Unsupported environment clustering method: {method}")


def fit_environment_labels(
    frame: pd.DataFrame,
    variables: Sequence[str],
    *,
    method: str,
    k: int,
    by_soil: bool,
    weights: Mapping[str, float] | Sequence[float] | np.ndarray | None = None,
    seed: int = 20261026,
    soil_codes: Sequence[int] = PRIMARY_SOIL_CODES,
) -> np.ndarray:
    """Fit environment groups; incomplete rows and excluded soils receive label -1.

    When stratifying by soil, ``k`` is the requested cluster count within each
    valid soil class. GMM and agglomerative fits are capped to deterministic
    samples; all remaining rows receive a label from the fitted components or
    nearest sample-cluster centroid.
    """

    if method not in {"none", "direct_nearest_neighbor", "minibatch_kmeans", "kmeans", "gaussian_mixture", "agglomerative"}:
        raise ValueError(f"Unsupported environment clustering method: {method}")
    if method in {"none", "direct_nearest_neighbor"}:
        if k != 0:
            raise ValueError("k must be 0 for none or direct_nearest_neighbor methods")
    elif not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer for clustering methods")
    raw = _feature_matrix(frame, variables)
    complete = np.isfinite(raw).all(axis=1)
    if by_soil and "SoilType" not in frame:
        raise ValueError("by_soil=True requires a SoilType column")
    if "SoilType" in frame:
        soil_values = pd.to_numeric(frame["SoilType"], errors="coerce").to_numpy()
        selected_soils = np.asarray(tuple(soil_codes), dtype=float)
        complete &= np.isfinite(soil_values) & np.isin(soil_values, selected_soils)
    else:
        soil_values = np.zeros(len(frame), dtype=float)

    labels = np.full(len(frame), -1, dtype=np.int32)
    valid_indices = np.flatnonzero(complete)
    if not len(valid_indices):
        return labels

    if method in {"none", "direct_nearest_neighbor"}:
        labels[valid_indices] = 0
        return labels

    if by_soil:
        strata = [int(code) for code in soil_codes if np.any(complete & (soil_values == code))]
    else:
        strata = [None]

    next_group_id = 0
    for stratum_index, soil_code in enumerate(strata):
        if soil_code is None:
            group_indices = valid_indices
        else:
            group_indices = np.flatnonzero(complete & (soil_values == soil_code))
        subset = frame.iloc[group_indices]
        matrix, _ = transform_environment_features(
            subset, variables, method="standardized_euclidean", weights=weights
        )
        local_labels = _cluster_one_group(matrix, method, k, seed + stratum_index)
        for local_label in np.unique(local_labels):
            members = group_indices[local_labels == local_label]
            labels[members] = next_group_id
            next_group_id += 1
    return labels
