"""Distribution-derived low-erosion threshold diagnostics.

All distributional shape diagnostics use log10 of positive Aa values. Zeros
remain a separate mass at zero and are never mixed with NoData.
"""

from __future__ import annotations

import json
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture


def _finite_nonnegative(values: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(values, dtype=float).reshape(-1)
    finite = np.isfinite(raw)
    if np.any(raw[finite] < 0):
        raise ValueError("erosion exposure must be nonnegative")
    return raw, finite


def summarize_values(values: Iterable[float]) -> dict[str, float | int]:
    """Summarize finite, zero and positive erosion observations separately."""
    raw, finite = _finite_nonnegative(values)
    selected = raw[finite]
    positive = selected[selected > 0]
    if len(selected):
        quantiles = np.quantile(selected, [0, .01, .05, .25, .5, .75, .95, .99, 1])
    else:
        quantiles = np.full(9, np.nan)
    result: dict[str, float | int] = {
        "n_total": int(len(raw)), "n_finite": int(finite.sum()),
        "n_nodata": int((~finite).sum()), "n_zero": int(np.count_nonzero(selected == 0)),
        "n_positive": int(len(positive)),
        "zero_fraction_finite": float(np.mean(selected == 0)) if len(selected) else np.nan,
        "positive_fraction_finite": float(np.mean(selected > 0)) if len(selected) else np.nan,
        "positive_min": float(positive.min()) if len(positive) else np.nan,
        "positive_log10_min": float(np.log10(positive.min())) if len(positive) else np.nan,
        "positive_max": float(positive.max()) if len(positive) else np.nan,
        "positive_median": float(np.median(positive)) if len(positive) else np.nan,
    }
    positive_quantiles = np.quantile(positive, [.01, .05, .10, .25, .50, .75, .90, .95, .99]) if len(positive) else np.full(9, np.nan)
    for name, value in zip(("q01", "q05", "q10", "q25", "q50", "q75", "q90", "q95", "q99"), positive_quantiles, strict=True):
        result[f"positive_{name}"] = float(value)
    for name, value in zip(("min", "q01", "q05", "q25", "median", "q75", "q95", "q99", "max"), quantiles, strict=True):
        result[f"{name}_finite"] = float(value)
    if len(positive):
        log_values = np.log10(positive)
        edges = np.arange(np.floor(log_values.min()), np.ceil(log_values.max()) + 1)
        counts, _ = np.histogram(log_values, bins=np.r_[edges, edges[-1] + 1])
        total = max(counts.sum(), 1)
        for order, count in zip(edges.astype(int), counts, strict=True):
            result[f"share_log10_{order}_to_{order + 1}"] = float(count / total)
        for decimals in (0, 1, 2, 3, 6, 9, 12):
            rounded = np.round(positive, decimals=decimals)
            frequencies = pd.Series(rounded).value_counts()
            result[f"fraction_repeated_{decimals}dp"] = float(frequencies[frequencies > 1].sum() / len(positive))
            result[f"unique_fraction_{decimals}dp"] = float(pd.Series(rounded).nunique() / len(positive))
    else:
        for decimals in (0, 1, 2, 3, 6, 9, 12):
            result[f"fraction_repeated_{decimals}dp"] = np.nan
            result[f"unique_fraction_{decimals}dp"] = np.nan
        result["fraction_repeated_6dp"] = np.nan
        result["unique_fraction_6dp"] = np.nan
    return result


def _kde_curve(log_values: np.ndarray, base: str, multiplier: float, grid_size: int = 512):
    if len(log_values) < 30 or np.ptp(log_values) <= 1e-10:
        return None
    pad = max(np.ptp(log_values) * .04, .08)
    grid = np.linspace(log_values.min() - pad, log_values.max() + pad, grid_size)
    bw = base.lower()
    if bw not in {"scott", "silverman"}:
        raise ValueError("base bandwidth must be scott or silverman")
    estimator = gaussian_kde(log_values, bw_method=bw)
    estimator.set_bandwidth(bw_method=estimator.factor * float(multiplier))
    density = estimator(grid)
    return grid, density


def find_kde_valleys(
    values: Iterable[float], *, exposure: str, multipliers: tuple[float, ...] = (.5, 1.0, 1.5, 2.0),
    bases: tuple[str, ...] = ("scott", "silverman"),
) -> pd.DataFrame:
    """Find supported valleys after a distinct low positive mode in log10(Aa)."""
    raw, finite = _finite_nonnegative(values)
    positive = raw[finite & (raw > 0)]
    log_values = np.log10(positive)
    rows: list[dict[str, object]] = []
    for base in bases:
        for multiplier in multipliers:
            common = {"exposure": exposure, "bandwidth_base": base, "bandwidth_multiplier": float(multiplier),
                      "candidate_source": "distribution_derived"}
            if len(log_values) < 30 or np.ptp(log_values) < .1:
                rows.append({**common, "status": "insufficient_positive_spread", "threshold_log10": np.nan,
                             "threshold_original": np.nan, "low_peak_log10": np.nan, "next_peak_log10": np.nan,
                             "valley_density_ratio": np.nan, "retained_fraction_positive": np.nan})
                continue
            curve = _kde_curve(log_values, base, multiplier)
            assert curve is not None
            grid, density = curve
            peaks, _ = find_peaks(density, prominence=max(density.max() * .015, 1e-15), distance=5)
            peaks = np.sort(peaks)
            candidate = None
            for left, right in zip(peaks[:-1], peaks[1:], strict=True):
                # Low-erosion peak is the leftmost positive mode; its adjacent trough
                # must reduce density clearly relative to both neighboring modes.
                if left != peaks[0]:
                    continue
                trough = left + int(np.argmin(density[left:right + 1]))
                ratio = float(density[trough] / min(density[left], density[right]))
                separation = float(grid[right] - grid[left])
                if ratio <= .80 and separation >= .12:
                    candidate = (left, trough, right, ratio)
                    break
            if candidate is None:
                rows.append({**common, "status": "no_reliable_valley", "threshold_log10": np.nan,
                             "threshold_original": np.nan, "low_peak_log10": np.nan, "next_peak_log10": np.nan,
                             "valley_density_ratio": np.nan, "retained_fraction_positive": np.nan})
            else:
                left, trough, right, ratio = candidate
                log_threshold = float(grid[trough])
                threshold = float(10 ** log_threshold)
                rows.append({**common, "status": "valid_valley", "threshold_log10": log_threshold,
                             "threshold_original": threshold, "low_peak_log10": float(grid[left]),
                             "next_peak_log10": float(grid[right]), "valley_density_ratio": ratio,
                             "retained_fraction_positive": float(np.mean(positive > threshold))})
    return pd.DataFrame.from_records(rows)


def empirical_cdf_breakpoint(values: Iterable[float], *, exposure: str) -> pd.DataFrame:
    """Compare a single log-positive CDF line to a continuous one-break fit."""
    raw, finite = _finite_nonnegative(values)
    positive = np.sort(raw[finite & (raw > 0)])
    full_n = len(positive)
    if full_n >= 5000:
        positions = np.unique(np.linspace(0, full_n - 1, 5000, dtype=int))
        logs = np.log10(positive[positions])
    else:
        logs = np.log10(positive)
    if len(logs) < 50 or np.ptp(logs) < .15:
        return pd.DataFrame([{"exposure": exposure, "status": "insufficient_positive_spread",
                              "threshold_log10": np.nan, "threshold_original": np.nan,
                              "delta_aic_one_break_vs_linear": np.nan,
                              "candidate_source": "distribution_derived"}])
    cdf = (np.arange(len(logs)) + .5) / len(logs)
    x = (logs - logs.mean()) / logs.std()
    linear = np.polyfit(x, cdf, 1)
    linear_sse = float(np.sum((cdf - np.polyval(linear, x)) ** 2))
    best: tuple[float, int, np.ndarray, np.ndarray] | None = None
    lo, hi = int(.1 * len(x)), int(.9 * len(x))
    candidate_knots = np.unique(np.linspace(lo, hi, min(200, hi - lo), dtype=int))
    for knot in candidate_knots:
        # Continuous hinge model y = b0 + b1*x + b2*max(x-knot, 0).
        design = np.column_stack([np.ones(len(x)), x, np.maximum(x - x[knot], 0.0)])
        coef, *_ = np.linalg.lstsq(design, cdf, rcond=None)
        sse = float(np.sum((cdf - design @ coef) ** 2))
        if best is None or sse < best[0]:
            best = (sse, knot, coef, design)
    assert best is not None
    sse, knot, coef, design = best
    n = len(x)
    aic_linear = n * np.log(max(linear_sse / n, 1e-20)) + 2 * 2
    aic_break = n * np.log(max(sse / n, 1e-20)) + 2 * 4
    improvement = float(aic_linear - aic_break)
    valid = improvement >= 2.0
    threshold_log = float(logs[knot]) if valid else np.nan
    return pd.DataFrame([{"exposure": exposure, "status": "valid_cdf_breakpoint" if valid else "no_reliable_cdf_breakpoint",
                          "threshold_log10": threshold_log,
                          "threshold_original": float(10 ** threshold_log) if valid else np.nan,
                          "delta_aic_one_break_vs_linear": improvement, "n_positive": full_n, "fit_n": n,
                          "candidate_source": "distribution_derived"}])


def fit_gaussian_mixtures(
    values: Iterable[float], *, exposure: str, seed: int = 20261029, sample_size: int = 60000,
    rows: Iterable[int] | None = None, cols: Iterable[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit K=1..4 Gaussian mixtures to positives; return ordered-component posteriors."""
    raw, finite = _finite_nonnegative(values)
    indexes = np.flatnonzero(finite & (raw > 0))
    if not len(indexes):
        return pd.DataFrame(), pd.DataFrame(columns=["source_index", "posterior_max", "classification_status"])
    log_values = np.log10(raw[indexes])
    rng = np.random.default_rng(seed)
    if len(log_values) > sample_size:
        if rows is not None and cols is not None:
            row_array = np.asarray(rows, dtype=float).reshape(-1)
            col_array = np.asarray(cols, dtype=float).reshape(-1)
            if len(row_array) != len(raw) or len(col_array) != len(raw):
                raise ValueError("rows and cols must align with values")
            block_row = np.floor(row_array[indexes] / 167).astype(int)
            block_col = np.floor(col_array[indexes] / 167).astype(int)
            block_id = np.asarray([f"{r}:{c}" for r, c in zip(block_row, block_col, strict=True)])
            _, inverse, counts = np.unique(block_id, return_inverse=True, return_counts=True)
            allocation = np.floor(sample_size * counts / counts.sum()).astype(int)
            remainder = sample_size - int(allocation.sum())
            if remainder > 0:
                allocation[np.argsort(-counts, kind="stable")[:remainder]] += 1
            selections = []
            for group_index, take in enumerate(allocation):
                candidates = np.flatnonzero(inverse == group_index)
                if take:
                    selections.extend(rng.choice(candidates, int(min(take, len(candidates))), replace=False).tolist())
            fit_positions = np.sort(np.asarray(selections, dtype=int))
        else:
            fit_positions = np.sort(rng.choice(len(log_values), sample_size, replace=False))
    else:
        fit_positions = np.arange(len(log_values))
    fit_values = log_values[fit_positions, None]
    rows = []
    fitted = {}
    for k in range(1, 5):
        if len(fit_values) < k * 10:
            continue
        model = GaussianMixture(n_components=k, covariance_type="full", random_state=seed,
                                n_init=3, reg_covar=1e-6, max_iter=300)
        model.fit(fit_values)
        means = model.means_.reshape(-1)
        order = np.argsort(means)
        means = means[order]
        stds = np.sqrt(model.covariances_.reshape(-1)[order])
        weights = model.weights_[order]
        components = model.predict_proba(fit_values)[:, order]
        overlap = []
        boundaries = []
        for left in range(k - 1):
            denom = max(stds[left] + stds[left + 1], 1e-12)
            overlap.append(float(np.exp(-.25 * ((means[left + 1] - means[left]) / denom) ** 2)))
            grid = np.linspace(means[left], means[left + 1], 500)
            left_density = weights[left] / stds[left] * np.exp(-.5 * ((grid - means[left]) / stds[left]) ** 2)
            right_density = weights[left + 1] / stds[left + 1] * np.exp(-.5 * ((grid - means[left + 1]) / stds[left + 1]) ** 2)
            crossing = np.flatnonzero(np.diff(np.sign(left_density - right_density)))
            boundaries.append(float(grid[crossing[0] + 1]) if len(crossing) else float("nan"))
        rows.append({"exposure": exposure, "n_components": k, "aic": float(model.aic(fit_values)),
                     "bic": float(model.bic(fit_values)), "converged": bool(model.converged_),
                     "iterations": int(model.n_iter_), "component_means_log10": "|".join(map(str, means)),
                     "component_sd_log10": "|".join(map(str, stds)), "component_weights": "|".join(map(str, weights)),
                     "adjacent_overlap_index": "|".join(map(str, overlap)), "fit_n": len(fit_values),
                     "adjacent_boundaries_log10": "|".join(map(str, boundaries)),
                     "candidate_source": "distribution_derived"})
        fitted[k] = model
    if not fitted:
        return pd.DataFrame(rows), pd.DataFrame(columns=["source_index", "posterior_max", "classification_status"])
    model_table = pd.DataFrame(rows)
    selected_k = int(model_table.loc[model_table["bic"].idxmin(), "n_components"])
    model = fitted[selected_k]
    order = np.argsort(model.means_.reshape(-1))
    posterior = model.predict_proba(log_values[:, None])[:, order]
    max_p = posterior.max(axis=1)
    component = posterior.argmax(axis=1) + 1
    assignments = pd.DataFrame({"source_index": indexes, "exposure": exposure,
                                "selected_n_components_bic": selected_k,
                                "posterior_component": component,
                                "posterior_max": max_p,
                                "classification_status": np.where(max_p >= .8, "assigned", "uncertain"),
                                "posterior_vector": [json.dumps(row.tolist()) for row in posterior]})
    return model_table, assignments


def otsu_thresholds(
    values: Iterable[float], *, exposure: str, bins: tuple[int, ...] = (32, 64, 128, 256)
) -> pd.DataFrame:
    """Compute comparison-only Otsu thresholds on positive log10(Aa)."""
    raw, finite = _finite_nonnegative(values)
    positive = raw[finite & (raw > 0)]
    if len(positive) < 2:
        return pd.DataFrame()
    logs = np.log10(positive)
    rows = []
    for n_bins in bins:
        counts, edges = np.histogram(logs, bins=int(n_bins))
        centers = (edges[:-1] + edges[1:]) / 2
        weight_left = np.cumsum(counts)
        weight_right = weight_left[-1] - weight_left
        mean_left = np.divide(np.cumsum(counts * centers), weight_left,
                              out=np.zeros_like(centers), where=weight_left > 0)
        reversed_sum = np.cumsum((counts * centers)[::-1])[::-1]
        mean_right = np.divide(reversed_sum - counts * centers, weight_right,
                               out=np.zeros_like(centers), where=weight_right > 0)
        variance = weight_left * weight_right * (mean_left - mean_right) ** 2
        variance[-1] = -np.inf
        threshold_log = float(centers[int(np.argmax(variance))])
        threshold = float(10 ** threshold_log)
        rows.append({"exposure": exposure, "histogram_bins": int(n_bins), "threshold_log10": threshold_log,
                     "threshold_original": threshold, "retained_fraction_positive": float(np.mean(positive > threshold)),
                     "candidate_source": "comparison_only", "status": "comparison_threshold"})
    return pd.DataFrame.from_records(rows)


def classify_regimes(values: Iterable[float], *, threshold: float | None) -> np.ndarray:
    """Classify as zero, low-positive, main, or NoData; never collapse zero into low."""
    raw, finite = _finite_nonnegative(values)
    labels = np.full(len(raw), "nodata", dtype=object)
    labels[finite & (raw == 0)] = "zero"
    if threshold is not None and np.isfinite(threshold) and threshold >= 0:
        labels[finite & (raw > 0) & (raw <= threshold)] = "low_positive"
        labels[finite & (raw > threshold)] = "main"
    else:
        labels[finite & (raw > 0)] = "main"
    return labels


def bootstrap_kde_valleys(
    values: Iterable[float], rows: Iterable[int], cols: Iterable[int], *, exposure: str,
    replicates: int = 100, seed: int = 20261029, sample_size: int = 5000,
) -> pd.DataFrame:
    """Resample 5 km lattice blocks; return one Scott KDE valley per replicate."""
    raw, finite = _finite_nonnegative(values)
    row_values = np.asarray(rows)
    col_values = np.asarray(cols)
    if len(raw) != len(row_values) or len(raw) != len(col_values):
        raise ValueError("values, rows and cols must have equal length")
    valid = finite & (raw > 0) & np.isfinite(row_values) & np.isfinite(col_values)
    # The lattice cell size is 30 m; 5 km blocks are represented by 167 cells.
    block_row = np.floor(row_values[valid].astype(float) / 167).astype(int)
    block_col = np.floor(col_values[valid].astype(float) / 167).astype(int)
    frame = pd.DataFrame({"value": raw[valid], "block": [f"{r}:{c}" for r, c in zip(block_row, block_col, strict=True)]})
    blocks = frame["block"].unique()
    if len(blocks) < 2:
        return pd.DataFrame(columns=["exposure", "replicate", "threshold_original", "status", "n_blocks"])
    rng = np.random.default_rng(seed)
    rows_out = []
    block_positions = {block: np.flatnonzero(frame["block"].to_numpy() == block) for block in blocks}
    for replicate in range(1, int(replicates) + 1):
        sampled_blocks = rng.choice(blocks, size=len(blocks), replace=True)
        pieces = [block_positions[b] for b in sampled_blocks]
        chosen = np.concatenate(pieces)
        if len(chosen) > sample_size:
            chosen = np.sort(rng.choice(chosen, sample_size, replace=False))
        sampled = frame["value"].to_numpy()[chosen]
        # Scott only keeps the requested stability analysis computationally bounded.
        result = find_kde_valleys(sampled, exposure=exposure, multipliers=(1.0,), bases=("scott",))
        scott = result.loc[result["bandwidth_base"].eq("scott")].iloc[0]
        rows_out.append({"exposure": exposure, "replicate": replicate,
                         "threshold_original": scott["threshold_original"], "status": scott["status"],
                         "n_blocks": int(len(np.unique(sampled_blocks))), "sample_n": int(len(sampled)),
                         "candidate_source": "distribution_derived_spatial_bootstrap"})
    return pd.DataFrame.from_records(rows_out)


def bootstrap_cdf_breakpoints(
    values: Iterable[float], rows: Iterable[int], cols: Iterable[int], *, exposure: str,
    replicates: int = 100, seed: int = 20261029, sample_size: int = 5000,
) -> pd.DataFrame:
    """Spatial-block bootstrap the empirical-CDF breakpoint on log-positive Aa."""
    raw, finite = _finite_nonnegative(values)
    row_values = np.asarray(rows, dtype=float)
    col_values = np.asarray(cols, dtype=float)
    if len(raw) != len(row_values) or len(raw) != len(col_values):
        raise ValueError("values, rows and cols must have equal length")
    valid = finite & (raw > 0) & np.isfinite(row_values) & np.isfinite(col_values)
    block_row = np.floor(row_values[valid] / 167).astype(int)
    block_col = np.floor(col_values[valid] / 167).astype(int)
    frame = pd.DataFrame({"value": raw[valid],
                          "block": [f"{r}:{c}" for r, c in zip(block_row, block_col, strict=True)]})
    blocks = frame["block"].unique()
    if len(blocks) < 2:
        return pd.DataFrame(columns=["exposure", "replicate", "threshold_original", "status", "n_blocks"])
    rng = np.random.default_rng(seed)
    block_positions = {block: np.flatnonzero(frame["block"].to_numpy() == block) for block in blocks}
    rows_out = []
    for replicate in range(1, int(replicates) + 1):
        sampled_blocks = rng.choice(blocks, size=len(blocks), replace=True)
        chosen = np.concatenate([block_positions[block] for block in sampled_blocks])
        if len(chosen) > sample_size:
            chosen = rng.choice(chosen, sample_size, replace=False)
        sampled = frame["value"].to_numpy()[chosen]
        result = empirical_cdf_breakpoint(sampled, exposure=exposure)
        item = result.iloc[0]
        rows_out.append({"exposure": exposure, "method": "empirical_CDF",
                         "replicate": replicate, "threshold_original": item["threshold_original"],
                         "status": item["status"], "n_blocks": int(len(np.unique(sampled_blocks))),
                         "sample_n": int(len(sampled)),
                         "candidate_source": "distribution_derived_spatial_bootstrap"})
    return pd.DataFrame.from_records(rows_out)


