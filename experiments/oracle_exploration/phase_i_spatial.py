"""Phase I-4 local spatial coefficient models on 240 m summary units."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phase_i_spatial_mplconfig")

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import yaml
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
OUTCOMES = ["SHI_candidate", "AN", "OM", "CEC", "pH", "SW", "BD", "MBC", "RESP"]
ENVIRONMENT = ["Elevation", "Cur", "TWI", "LST", "NDWI", "NDBSI", "EVI", "NDVI", "Precipitation"]
KEY_OUTCOMES = ["SHI_candidate", "OM", "SW"]
LABELS = {"SHI_candidate": "SHI", "AN": "AN", "OM": "OM", "CEC": "CEC", "pH": "pH",
          "SW": "SW", "BD": "BD", "MBC": "MBC", "RESP": "RESP"}
BLUE_RED = "#4477AA"
try:
    font_manager.findfont("WenQuanYi Zen Hei", fallback_to_default=False)
    plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
except ValueError:
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True,
                              text=True).stdout.strip()
    except Exception:
        return None


def _read_pixels(config: dict[str, Any]) -> tuple[pd.DataFrame, Path, Path]:
    db = ROOT / "outputs" / config["input_run_id"] / "databases"
    env_path, health_path = db / "environment_erosion.parquet", db / "health.parquet"
    env_cols = ["cell_id", "row", "col", "x", "y", config["primary_exposure"], *config["environment_covariates"]]
    env = pd.read_parquet(env_path, columns=env_cols)
    health = pd.read_parquet(health_path, columns=["cell_id", *config["outcomes"]])
    data = env.merge(health, on="cell_id", how="inner", validate="one_to_one")
    if len(data) != len(env) or not data["cell_id"].is_unique:
        raise ValueError("Environment and health tables must align one-to-one by cell_id.")
    data["log1p_Aa"] = np.log1p(pd.to_numeric(data[config["primary_exposure"]], errors="coerce"))
    return data, env_path, health_path


def _tile_pixels(frame: pd.DataFrame, outcome: str, features: list[str], tile_cells: int) -> pd.DataFrame:
    cols = ["row", "col", "x", "y", outcome, *features]
    values = frame[cols].apply(pd.to_numeric, errors="coerce")
    keep = np.isfinite(values.to_numpy(dtype=float)).all(axis=1)
    d = frame.loc[keep, cols].copy()
    d["tile_row"] = np.floor(d["row"].to_numpy(dtype=float) / tile_cells).astype(np.int32)
    d["tile_col"] = np.floor(d["col"].to_numpy(dtype=float) / tile_cells).astype(np.int32)
    agg = {"x": "mean", "y": "mean", outcome: "mean", **{c: "mean" for c in features}}
    units = d.groupby(["tile_row", "tile_col"], sort=True).agg(agg)
    units["n_pixels"] = d.groupby(["tile_row", "tile_col"], sort=True).size()
    units = units.reset_index()
    units["analysis_outcome"] = outcome
    return units


def _tile_pairs(frame: pd.DataFrame, outcome: str, tile_m: int) -> pd.DataFrame:
    value_col = f"delta_{outcome}"
    cols = ["mid_x", "mid_y", "delta_log1p_Aa", value_col]
    v = frame[cols].apply(pd.to_numeric, errors="coerce")
    keep = np.isfinite(v.to_numpy(dtype=float)).all(axis=1)
    d = frame.loc[keep, cols].copy()
    d["tile_x"] = np.floor(d["mid_x"].to_numpy() / tile_m).astype(np.int64)
    d["tile_y"] = np.floor(d["mid_y"].to_numpy() / tile_m).astype(np.int64)
    units = d.groupby(["tile_x", "tile_y"], sort=True).agg(
        x=("mid_x", "mean"), y=("mid_y", "mean"),
        delta_log1p_Aa=("delta_log1p_Aa", "mean"), **{value_col: (value_col, "mean")})
    units["n_pairs"] = d.groupby(["tile_x", "tile_y"], sort=True).size()
    units = units.reset_index().rename(columns={value_col: outcome})
    units["n_pixels"] = units["n_pairs"]
    return units


def _weighted_scale(a: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    total = float(weights.sum())
    mean = np.sum(a * weights[:, None], axis=0) / total
    var = np.sum(((a - mean) ** 2) * weights[:, None], axis=0) / total
    sd = np.sqrt(np.maximum(var, 1e-20))
    return mean, sd


def _prepare_design(units: pd.DataFrame, features: list[str], outcome: str) -> dict[str, Any]:
    xraw = units[features].to_numpy(dtype=float)
    yraw = units[outcome].to_numpy(dtype=float)
    freq = units["n_pixels"].to_numpy(dtype=float)
    xmean, xsd = _weighted_scale(xraw, freq)
    ymean = float(np.average(yraw, weights=freq))
    ysd = float(np.sqrt(np.average((yraw - ymean) ** 2, weights=freq)))
    if ysd <= 1e-12:
        raise ValueError(f"Outcome {outcome} has no variation after aggregation.")
    x = (xraw - xmean) / xsd
    y = (yraw - ymean) / ysd
    return {"x": np.column_stack([np.ones(len(x)), x]), "y": y, "freq": freq,
            "xmean": xmean, "xsd": xsd, "ymean": ymean, "ysd": ysd}


def _query_neighbors(tree: cKDTree, coords: np.ndarray, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    distance, index = tree.query(coords[query], k=min(k, len(coords)))
    if distance.ndim == 1:
        distance, index = distance[:, None], index[:, None]
    return distance.astype(float), index.astype(np.int64)


def _kernel_weights(distance: np.ndarray, freq: np.ndarray) -> np.ndarray:
    bandwidth = np.maximum(distance[:, -1], 1.0)
    return np.exp(-0.5 * (distance / bandwidth[:, None]) ** 2) * freq


def _local_fit(
    X: np.ndarray, y: np.ndarray, freq: np.ndarray, coords: np.ndarray,
    neighbor_idx: np.ndarray, distance: np.ndarray, *, ridge_condition: float | None = None,
    collect_local_r2: bool = True,
) -> dict[str, np.ndarray]:
    nquery, k = neighbor_idx.shape
    p = X.shape[1]
    betas = np.empty((nquery, p), dtype=float)
    ses = np.empty((nquery, p), dtype=float)
    local_r2 = np.empty(nquery, dtype=float)
    condition = np.empty(nquery, dtype=float)
    ridge_alpha = np.zeros(nquery, dtype=float)
    support_pixels = np.empty(nquery, dtype=float)
    chunk = 200
    for start in range(0, nquery, chunk):
        stop = min(start + chunk, nquery)
        ix = neighbor_idx[start:stop]
        xx = X[ix]
        yy = y[ix]
        ww = _kernel_weights(distance[start:stop], freq[ix])
        xtwx = np.einsum("nki,nk,nkj->nij", xx, ww, xx, optimize=True)
        xtwy = np.einsum("nki,nk,nk->ni", xx, ww, yy, optimize=True)
        condition[start:stop] = np.linalg.cond(xtwx)
        if ridge_condition is not None:
            eig = np.linalg.eigvalsh(xtwx)
            smallest = np.maximum(eig[:, 0], 0.0)
            largest = np.maximum(eig[:, -1], 0.0)
            alpha = np.maximum(0.0, (largest - ridge_condition * smallest) / max(ridge_condition - 1, 1))
            ridge_alpha[start:stop] = alpha
            xtwx[:, 1:, 1:] += alpha[:, None, None] * np.eye(p - 1)[None, :, :]
        try:
            beta = np.linalg.solve(xtwx, xtwy[..., None])[..., 0]
            inverse = np.linalg.solve(xtwx, np.broadcast_to(np.eye(p), xtwx.shape))
        except np.linalg.LinAlgError:
            beta = np.einsum("nij,nj->ni", np.linalg.pinv(xtwx), xtwy, optimize=True)
            inverse = np.linalg.pinv(xtwx)
        betas[start:stop] = beta
        residual = yy - np.einsum("nkp,np->nk", xx, beta, optimize=True)
        dof = np.maximum(ww.sum(axis=1) - p, 1.0)
        sigma2 = np.sum(ww * residual ** 2, axis=1) / dof
        ses[start:stop] = np.sqrt(np.maximum(np.diagonal(inverse, axis1=1, axis2=2) * sigma2[:, None], 0))
        support_pixels[start:stop] = freq[ix].sum(axis=1)
        if collect_local_r2:
            mean_y = np.sum(ww * yy, axis=1) / np.maximum(ww.sum(axis=1), 1e-12)
            ss_tot = np.sum(ww * (yy - mean_y[:, None]) ** 2, axis=1)
            ss_res = np.sum(ww * residual ** 2, axis=1)
            local_r2[start:stop] = 1 - np.divide(ss_res, ss_tot, out=np.full_like(ss_res, np.nan), where=ss_tot > 0)
        else:
            local_r2[start:stop] = np.nan
    return {"beta": betas, "se": ses, "local_r2": local_r2, "condition": condition,
            "ridge_alpha": ridge_alpha, "support_pixels": support_pixels}


def _cv_bandwidth(
    units: pd.DataFrame, design: dict[str, Any], candidates: list[int], cv_n: int, seed: int,
) -> tuple[int, pd.DataFrame]:
    n = len(units)
    viable = sorted(set(int(k) for k in candidates if 12 <= int(k) < n))
    if not viable:
        raise ValueError(f"Not enough spatial units ({n}) for configured bandwidth candidates.")
    rng = np.random.default_rng(seed)
    cv_n = min(cv_n, n)
    target = np.sort(rng.choice(n, size=cv_n, replace=False))
    tree = cKDTree(units[["x", "y"]].to_numpy(dtype=float))
    max_k = max(viable)
    dist_all, idx_all = _query_neighbors(tree, units[["x", "y"]].to_numpy(dtype=float), target, min(max_k + 1, n))
    if np.all(dist_all[:, 0] < 1e-6):
        dist_all, idx_all = dist_all[:, 1:], idx_all[:, 1:]
    rows = []
    X, y, freq = design["x"], design["y"], design["freq"]
    for k in viable:
        idx = idx_all[:, :k]
        dist = dist_all[:, :k]
        result = _local_fit(X, y, freq, units[["x", "y"]].to_numpy(dtype=float), idx, dist,
                            collect_local_r2=False)
        pred = np.sum(X[target] * result["beta"], axis=1)
        error = y[target] - pred
        rmse = float(np.sqrt(np.average(error ** 2, weights=freq[target])))
        rows.append({"neighbors": k, "cv_targets": cv_n, "cv_rmse_standardized": rmse,
                     "cv_type": "sampled leave-one-unit-out; tuning only, spatially dependent"})
    cv = pd.DataFrame(rows)
    return int(cv.sort_values("cv_rmse_standardized").iloc[0]["neighbors"]), cv


def _fit_gwr(
    units: pd.DataFrame, outcome: str, features: list[str], config: dict[str, Any],
    *, ridge_condition: float | None = None, seed_offset: int = 0,
    force_bandwidth: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    design = _prepare_design(units, features, outcome)
    if force_bandwidth is None:
        bandwidth, cv = _cv_bandwidth(units, design, config["bandwidth_neighbors"],
                                      int(config["cv_target_units"]), int(config["cv_seed"]) + seed_offset)
    else:
        bandwidth = int(force_bandwidth)
        cv = pd.DataFrame([{"neighbors": bandwidth, "cv_targets": np.nan,
                            "cv_rmse_standardized": np.nan,
                            "cv_type": "bandwidth fixed to the corresponding GWR fit"}])
    coords = units[["x", "y"]].to_numpy(dtype=float)
    tree = cKDTree(coords)
    dist, idx = _query_neighbors(tree, coords, np.arange(len(units)), bandwidth)
    local = _local_fit(design["x"], design["y"], design["freq"], coords, idx, dist,
                       ridge_condition=ridge_condition)
    exposure_index = 1
    beta = local["beta"][:, exposure_index]
    se = local["se"][:, exposure_index]
    result = units[["x", "y", "n_pixels"]].copy()
    for c in ("tile_row", "tile_col", "tile_x", "tile_y", "n_pairs"):
        if c in units:
            result[c] = units[c].to_numpy()
    result["outcome"] = outcome
    result["local_beta_standardized"] = beta
    result["local_se_standardized_model_based"] = se
    result["local_ci_low_model_based"] = beta - 1.96 * se
    result["local_ci_high_model_based"] = beta + 1.96 * se
    result["local_beta_raw"] = beta * design["ysd"] / design["xsd"][0]
    result["local_r2"] = local["local_r2"]
    result["local_condition_number"] = local["condition"]
    result["local_ridge_alpha"] = local["ridge_alpha"]
    result["local_neighbor_units"] = bandwidth
    result["local_support_pixels"] = local["support_pixels"]
    result["analysis_population"] = units.attrs.get("analysis_population", "unspecified")
    result["model_family"] = "GWR-LCR" if ridge_condition is not None else "GWR"
    result["outcome_guided_selection"] = True
    summary = {
        "outcome": outcome, "analysis_population": units.attrs.get("analysis_population", "unspecified"),
        "model_family": "GWR-LCR" if ridge_condition is not None else "GWR",
        "environment_adjustment": "+".join(features[1:]) if len(features) > 1 else "none",
        "n_spatial_units": len(units), "n_source_pixels_or_pairs": int(units["n_pixels"].sum()),
        "bandwidth_neighbors": bandwidth,
        "bandwidth_median_distance_m": float(np.median(dist[:, -1])),
        "bandwidth_p05_distance_m": float(np.quantile(dist[:, -1], .05)),
        "bandwidth_p95_distance_m": float(np.quantile(dist[:, -1], .95)),
        "cv_rmse_standardized": float(cv.loc[cv["neighbors"].eq(bandwidth), "cv_rmse_standardized"].iloc[0]),
        "local_beta_standardized_mean": float(np.average(beta, weights=units["n_pixels"])),
        "local_beta_standardized_median": float(np.median(beta)),
        "local_beta_p05": float(np.quantile(beta, .05)), "local_beta_p95": float(np.quantile(beta, .95)),
        "negative_effect_ratio": float(np.average(beta < 0, weights=units["n_pixels"])),
        "local_r2_weighted_mean": float(np.average(local["local_r2"], weights=units["n_pixels"])),
        "local_condition_p95": float(np.quantile(local["condition"], .95)),
        "ridge_alpha_positive_ratio": float(np.mean(local["ridge_alpha"] > 0)),
        "outcome_guided_selection": True,
    }
    cv["outcome"] = outcome
    cv["analysis_population"] = summary["analysis_population"]
    cv["model_family"] = summary["model_family"]
    return result, summary, cv


def _fit_multiscale_svc(units: pd.DataFrame, outcome: str, features: list[str],
                        config: dict[str, Any], seed_offset: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backfit a multiscale local-coefficient model; not a canonical MGWR fit."""
    design = _prepare_design(units, features, outcome)
    Xall = np.column_stack([np.ones(len(units)), design["x"][:, 1:]])
    y = design["y"]
    freq = design["freq"]
    coords = units[["x", "y"]].to_numpy(dtype=float)
    n, p = Xall.shape
    viable = sorted(set(int(k) for k in config["multiscale_bandwidth_neighbors"] if 12 <= int(k) < n))
    if len(viable) < 2:
        raise ValueError("Insufficient spatial units for multiscale bandwidth comparison.")
    rng = np.random.default_rng(int(config["cv_seed"]) + seed_offset)
    targets = np.sort(rng.choice(n, size=min(int(config["cv_target_units"]), n), replace=False))
    tree = cKDTree(coords)
    max_k = max(viable)
    cv_dist, cv_idx = _query_neighbors(tree, coords, targets, min(max_k + 1, n))
    cv_dist, cv_idx = cv_dist[:, 1:], cv_idx[:, 1:]
    full_dist, full_idx = _query_neighbors(tree, coords, np.arange(n), max_k)
    component_features = ["local_intercept", *features]
    p = Xall.shape[1]
    components = np.zeros((n, p), dtype=float)
    coefficients = np.zeros((n, p), dtype=float)
    bandwidths = np.full(p, viable[-1], dtype=int)
    rows: list[dict[str, Any]] = []
    for iteration in range(int(config["multiscale_iterations"])):
        for j, feature in enumerate(component_features):
            partial = y - (components.sum(axis=1) - components[:, j])
            cv_scores = []
            for k in viable:
                ix, dd = cv_idx[:, :k], cv_dist[:, :k]
                w = _kernel_weights(dd, freq[ix])
                xnb = Xall[ix, j]
                numerator = np.sum(w * xnb * partial[ix], axis=1)
                denominator = np.sum(w * xnb * xnb, axis=1) + 1e-8 * np.sum(w, axis=1)
                bcv = numerator / denominator
                error = partial[targets] - Xall[targets, j] * bcv
                cv_scores.append(float(np.sqrt(np.average(error ** 2, weights=freq[targets]))))
            kbest = viable[int(np.argmin(cv_scores))]
            bandwidths[j] = kbest
            ix, dd = full_idx[:, :kbest], full_dist[:, :kbest]
            w = _kernel_weights(dd, freq[ix])
            xnb = Xall[ix, j]
            numerator = np.sum(w * xnb * partial[ix], axis=1)
            denominator = np.sum(w * xnb * xnb, axis=1) + 1e-8 * np.sum(w, axis=1)
            coefficients[:, j] = numerator / denominator
            components[:, j] = Xall[:, j] * coefficients[:, j]
            rows.append({"outcome": outcome, "feature": feature, "iteration": iteration + 1,
                         "bandwidth_neighbors": kbest, "bandwidth_median_distance_m": float(np.median(dd[:, -1])),
                         "cv_rmse_standardized_partial": cv_scores[viable.index(kbest)],
                         "outcome_guided_selection": True})
    result = units[["x", "y", "n_pixels"]].copy()
    for c in ("tile_row", "tile_col"):
        if c in units:
            result[c] = units[c].to_numpy()
    for j, feature in enumerate(component_features):
        result[f"beta_{feature}_standardized"] = coefficients[:, j]
        result[f"component_{feature}_standardized"] = components[:, j]
        result[f"bandwidth_{feature}_neighbors"] = bandwidths[j]
    result["outcome"] = outcome
    result["analysis_population"] = units.attrs.get("analysis_population", "unspecified")
    result["model_family"] = "multiscale_SVC_backfitting_MGWR_inspired_noncanonical"
    result["outcome_guided_selection"] = True
    return result, pd.DataFrame(rows)


def _load_pairs(config: dict[str, Any], pixels: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    path = ROOT / config["matched_run_id"] / "pairs" / "candidate_pairs" / f"{config['matched_candidate']}.parquet"
    pairs = pd.read_parquet(path)
    coord = pixels[["cell_id", "x", "y"]].drop_duplicates("cell_id").set_index("cell_id")
    high = coord.reindex(pairs["high_erosion_cell_id"].to_numpy())
    low = coord.reindex(pairs["low_erosion_cell_id"].to_numpy())
    pairs["mid_x"] = (high["x"].to_numpy() + low["x"].to_numpy()) / 2
    pairs["mid_y"] = (high["y"].to_numpy() + low["y"].to_numpy()) / 2
    eligible_distance = pd.to_numeric(pairs["geographic_distance"], errors="coerce").le(
        float(config["pair_max_geographic_distance_m"]))
    coordinate_ok = np.isfinite(pairs[["mid_x", "mid_y"]].to_numpy(dtype=float)).all(axis=1)
    included = eligible_distance & coordinate_ok
    counts = {"input_pairs": len(pairs), "distance_within_limit": int(eligible_distance.sum()),
              "included_pairs": int(included.sum()), "unique_high_cells": int(pairs.loc[included, "high_erosion_cell_id"].nunique()),
              "unique_low_cells": int(pairs.loc[included, "low_erosion_cell_id"].nunique())}
    return pairs.loc[included].copy(), counts


def _mask_arrays(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = data["row"].to_numpy(dtype=int)
    cols = data["col"].to_numpy(dtype=int)
    rr0, cc0 = int(rows.min()), int(cols.min())
    nr, nc = int(rows.max() - rr0 + 1), int(cols.max() - cc0 + 1)
    mask = np.zeros((nr, nc), dtype=bool)
    mask[rows - rr0, cols - cc0] = True
    x_by_col = data.groupby("col", sort=True)["x"].median()
    y_by_row = data.groupby("row", sort=True)["y"].median()
    x_fit = np.polyfit(x_by_col.index.to_numpy(dtype=float), x_by_col.to_numpy(dtype=float), 1)
    y_fit = np.polyfit(y_by_row.index.to_numpy(dtype=float), y_by_row.to_numpy(dtype=float), 1)
    xcoord = np.polyval(x_fit, np.arange(cc0, cc0 + nc))
    ycoord = np.polyval(y_fit, np.arange(rr0, rr0 + nr))
    return mask, xcoord, ycoord


def _add_map_decor(ax: plt.Axes, mask: np.ndarray, xcoords: np.ndarray, ycoords: np.ndarray) -> None:
    if np.isfinite(xcoords).all() and np.isfinite(ycoords).all():
        ax.contour(xcoords / 1000, ycoords / 1000, mask.astype(float), levels=[0.5], colors="#1f2933", linewidths=.45)
    ax.annotate("N", xy=(.94, .91), xytext=(.94, .79), xycoords="axes fraction", ha="center", va="center",
                arrowprops={"arrowstyle": "-|>", "lw": 1.0, "color": "#263746"}, fontsize=10, color="#263746")
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    bar = 2.0
    x0 = xmin + .07 * (xmax - xmin)
    y0 = ymin + .08 * (ymax - ymin)
    ax.plot([x0, x0 + bar], [y0, y0], color="#263746", lw=2.2, solid_capstyle="butt")
    ax.text(x0 + bar / 2, y0 + .025 * (ymax - ymin), "2 km", ha="center", va="bottom", fontsize=8)
    inset = ax.inset_axes([.035, .73, .18, .20])
    inset.contourf(xcoords / 1000, ycoords / 1000, mask.astype(float), levels=[.5, 1.5], colors=["#dfe7eb"])
    inset.contour(xcoords / 1000, ycoords / 1000, mask.astype(float), levels=[.5], colors="#263746", linewidths=.45)
    inset.set_xticks([]); inset.set_yticks([]); inset.set_title("耕地掩膜范围", fontsize=7)
    for spine in inset.spines.values():
        spine.set_linewidth(.5)


def _save_map_panel(
    run_dir: Path, name: str, models: list[tuple[str, pd.DataFrame]], mask_info: tuple[np.ndarray, np.ndarray, np.ndarray], dpi: int,
) -> list[str]:
    mask, xcoords, ycoords = mask_info
    fig, axes = plt.subplots(len(models), 1, figsize=(10, 4.4 * len(models)), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, (title, d) in zip(axes, models, strict=True):
        finite = np.isfinite(d["local_beta_standardized"])
        vals = d.loc[finite, "local_beta_standardized"].to_numpy()
        scale = max(float(np.quantile(np.abs(vals), .97)), .05) if len(vals) else .1
        sc = ax.scatter(d.loc[finite, "x"] / 1000, d.loc[finite, "y"] / 1000,
                        c=vals, s=7, marker="s", linewidths=0, cmap="RdBu_r", vmin=-scale, vmax=scale,
                        rasterized=True)
        _add_map_decor(ax, mask, xcoords, ycoords)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Easting (km, EPSG:32649)")
        ax.set_ylabel("Northing (km)")
        fig.colorbar(sc, ax=ax, shrink=.82, label="局部标准化侵蚀系数（负值为负向关联）")
        ax.set_aspect("equal", adjustable="datalim")
    paths = []
    base = run_dir / "figures" / name
    base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg", "pdf"):
        path = base.with_suffix(f".{ext}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(str(path.relative_to(ROOT)))
    plt.close(fig)
    return paths


def _save_multiscale_bandwidth_figure(run_dir: Path, bandwidths: pd.DataFrame, dpi: int) -> list[str]:
    if bandwidths.empty:
        return []
    final_bws = (bandwidths.sort_values("iteration")
                 .groupby(["outcome", "feature"], as_index=False).tail(1))
    features = ["local_intercept", *ENVIRONMENT, "log1p_Aa"]
    labels = {"local_intercept": "局部截距", "log1p_Aa": "log1p(Aa)", **{x: x for x in ENVIRONMENT}}
    fig, axes = plt.subplots(1, 3, figsize=(15, 8), sharey=True, constrained_layout=True)
    for ax, outcome in zip(axes, KEY_OUTCOMES, strict=True):
        d = (final_bws.loc[final_bws["outcome"].eq(outcome)].set_index("feature")
             .reindex(features).dropna(subset=["bandwidth_median_distance_m"]))
        y = np.arange(len(d))
        ax.barh(y, d["bandwidth_median_distance_m"] / 1000, color=BLUE_RED, alpha=.88)
        ax.set_yticks(y, [labels[x] for x in d.index])
        ax.invert_yaxis()
        ax.set_title(LABELS[outcome])
        ax.set_xlabel("自适应邻域距离中位数 (km)")
        ax.grid(axis="x", color="#d8dee3", linewidth=.6)
    base = run_dir / "figures" / "multiscale_variable_bandwidths"
    paths = []
    for ext in ("png", "svg", "pdf"):
        path = base.with_suffix(f".{ext}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(str(path.relative_to(ROOT)))
    plt.close(fig)
    return paths


def _report(run_dir: Path, summary: pd.DataFrame, pair_counts: dict[str, int], n_mask: int,
            n_env: int, package_status: dict[str, Any], config: dict[str, Any],
            multiscale_bandwidths: pd.DataFrame) -> str:
    key = summary.loc[summary["outcome"].isin(KEY_OUTCOMES)].copy()
    summary_lookup = summary.set_index(["outcome", "analysis_population", "model_family"])
    lines = [
        "# Phase I-4：GWR 与空间变化系数探索", "",
        f"权威运行：`{run_dir.relative_to(ROOT)}`。运行模式为 `EXPLORATORY_ORACLE`，`outcome_guided_selection=true`。",
        "",
        "## 样本与空间单元", "",
        f"- 主耕地掩膜数据库共 {n_mask:,} 个像元行；30 m，EPSG:32649（米制平面坐标）。",
        f"- 全样本与环境调整模型按 8×8 像元汇总为约 240 m 空间单元；环境完整像元上限 {n_env:,}。每个单元以像元数加权，地图显示的是局部估计节点，不是 30 m 独立局部估计。",
        f"- 匹配模型复用 Phase H `{pair_counts['input_pairs']:,}` 个 Aa_20yr M4 配对；按 1,000 m 最大地理距离限制后纳入 {pair_counts['included_pairs']:,} 对，采用配对中点并汇总至约 240 m 空间单元。",
        "- 使用 Aa_20yr 的 log1p 变换；环境调整包含 Elevation、Cur、TWI、LST、NDWI、NDBSI、EVI、NDVI、Precipitation，不含 RUSLE 五因子与土类分类项。土类异质性见 I-3。",
        "",
        "## 全样本局部响应汇总", "",
        "表中 β̄ 是各局部标准化侵蚀系数的像元支持加权均值；空间分位数描述不同局部单元间的系数分布，不是置信区间。",
        "",
        "| 指标 | 未调整 GWR β̄（负向比） | 环境调整 GWR β̄（负向比） | 环境调整 GWR-LCR β̄（负向比） | 匹配全配对 β̄（负向比） |",
        "|---|---:|---:|---:|---:|",
    ]
    for outcome in OUTCOMES:
        cells = []
        for population, family in [
            ("mask_all_complete_case", "GWR"),
            ("environment_adjusted", "GWR"),
            ("environment_adjusted", "GWR-LCR"),
            ("matched_all", "GWR"),
        ]:
            row = summary_lookup.loc[(outcome, population, family)]
            cells.append(f"{row['local_beta_standardized_mean']:.3f} ({row['negative_effect_ratio']:.1%})")
        lines.append(f"| {LABELS[outcome]} | " + " | ".join(cells) + " |")
    gwr_rows = summary.loc[summary["model_family"].eq("GWR")]
    minimum, maximum = min(config["bandwidth_neighbors"]), max(config["bandwidth_neighbors"])
    n_at_min = int(gwr_rows["bandwidth_neighbors"].eq(minimum).sum())
    n_at_max = int(gwr_rows["bandwidth_neighbors"].eq(maximum).sum())
    lcr_penalty = summary.loc[summary["model_family"].eq("GWR-LCR"), "ridge_alpha_positive_ratio"]
    lines.extend([
        "",
        f"- 标准 GWR 的 {len(gwr_rows)} 个拟合中有 {n_at_min} 个选中邻域候选下限 k={minimum}（{n_at_max} 个选中上限 k={maximum}）。多项估计的优选局部尺度可能低于当前候选下界；不把带宽当作唯一已识别尺度。",
        f"- 环境调整 GWR 的局部条件数较高（9 项结局的条件数 P95 中位数为 {summary.loc[summary['model_family'].eq('GWR') & summary['analysis_population'].eq('environment_adjusted'), 'local_condition_p95'].dropna().median():.0f}）；GWR-LCR 在 {np.nanmean(lcr_penalty):.0%} 的局部单元施加岭惩罚，系数向零收缩，方向作为敏感性对照。",
        "",
        "## 环境调整多尺度局部系数", "",
        "下表使用逐变量带宽的多尺度 SVC backfitting 近似（见方法限制），仅对 SHI、OM、SW 拟合；尺度是适应性邻居距离中位数。",
        "",
        "| 指标 | 变量 | 邻居数 k | 带宽中位数 (km) |",
        "|---|---|---:|---:|",
    ])
    final_bws = pd.DataFrame()
    if not multiscale_bandwidths.empty:
        final_bws = (multiscale_bandwidths.sort_values("iteration")
                     .groupby(["outcome", "feature"], as_index=False).tail(1))
        for row in final_bws.itertuples(index=False):
            label = "局部截距" if row.feature == "local_intercept" else row.feature
            lines.append(f"| {LABELS[row.outcome]} | {label} | {int(row.bandwidth_neighbors)} | {row.bandwidth_median_distance_m / 1000:.2f} |")
        min_hits = int(final_bws["bandwidth_neighbors"].eq(min(config["multiscale_bandwidth_neighbors"])).sum())
        max_hits = int(final_bws["bandwidth_neighbors"].eq(max(config["multiscale_bandwidth_neighbors"])).sum())
        lines.extend(["", f"- 最后一轮 backfitting 中有 {min_hits} 个变量选中候选下限 k={min(config['multiscale_bandwidth_neighbors'])}、{max_hits} 个变量选中上限 k={max(config['multiscale_bandwidth_neighbors'])}；边界变量的作用尺度仍未充分定位。"])
    lines.extend(["", "## 匹配配对差值的局部响应", "",
                  "以下局部标准化斜率来自匹配高端减低端的 ΔY 对 Δlog1p(Aa)；全配对、零—正和双正估计使用独立空间支持。",
                  "", "| 指标 | 全配对 β̄（负向比） | 零—正 β̄（负向比） | 双正 β̄（负向比） |", "|---|---:|---:|---:|"])
    for outcome in KEY_OUTCOMES:
        cells = []
        for population in ("matched_all", "matched_zero_positive", "matched_both_positive"):
            row = summary_lookup.loc[(outcome, population, "GWR")]
            cells.append(f"{row['local_beta_standardized_mean']:.3f} ({row['negative_effect_ratio']:.1%})")
        lines.append(f"| {LABELS[outcome]} | " + " | ".join(cells) + " |")
    lines.extend(["", "## 关键结果地图", ""])
    for outcome in KEY_OUTCOMES:
        label = LABELS[outcome]
        rows = key.loc[key["outcome"].eq(outcome)]
        if rows.empty:
            continue
        parts = []
        for r in rows.itertuples(index=False):
            parts.append(f"{r.analysis_population}/{r.model_family}: β̄={r.local_beta_standardized_mean:.3f}, "
                         f"负向支持占比={r.negative_effect_ratio:.1%}, k={int(r.bandwidth_neighbors)}, "
                         f"带宽中位数={r.bandwidth_median_distance_m/1000:.2f} km")
        lines.append(f"- **{label}**：" + "；".join(parts) + ".")
    lines.extend([
        "", "## 方法状态与解释边界", "",
        "- 标准 GWR 使用自适应高斯核；候选近邻数通过抽取空间单元的 leave-one-unit-out 误差选择。该调参误差不是独立的空间泛化测试。局部 R²描述核邻域拟合，不能代表胁迫强度。",
        "- GWR-LCR 在局部加权正规方程中使用局部岭惩罚，令局部设计矩阵的条件数目标不高于 30；它用于检查局部共线性敏感性，惩罚会使局部系数向零收缩。",
        "- 多尺度部分为显式标记的 MGWR-inspired 局部系数 backfitting 近似，可为暴露与环境项选择不同邻域尺度。当前环境未安装 `mgwr`/`spglm`，因此没有运行或声称标准 mgwr 包的 MGWR 估计；近似结果不替代标准 MGWR。",
        "- 局部区间由汇总单元加权回归的模型协方差计算，未进行空间块稳健校正；像元空间依赖与 240 m 聚合使这些区间仅作探索性描述，不依据局部名义显著性划定结论。图中呈现局部系数正负方向，局部 R²与标准误保留在系数表中。",
        "- 结果来自健康预测栅格，不是独立实测样本的局部效应。未重算或验证 RUSLE 因子乘积，未重跑 PCNM，也未修改研究总纲。",
        "",
        "## 运行环境", "",
        f"- 专用包状态：`mgwr={package_status.get('mgwr')}`、`spglm={package_status.get('spglm')}`、`libpysal={package_status.get('libpysal')}`。",
        "- 表格、地图、配置副本、来源信息和状态记录见本次运行目录。",
        "",
    ])
    path = run_dir / "reports" / "PHASE_I_GWR_MGWR_REPORT.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path.relative_to(ROOT))


def run(config_path: Path) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = ROOT / config["output_root"] / config["run_id"]
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists and will not be overwritten: {run_dir}")
    run_dir.mkdir(parents=True)
    for sub in ("configs", "tables", "figures", "reports", "manifests", "leaderboards"):
        (run_dir / sub).mkdir(exist_ok=True)
    (run_dir / "configs" / config_path.name).write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    pixels, env_path, health_path = _read_pixels(config)
    env_vars = list(config["environment_covariates"])
    full_local: list[pd.DataFrame] = []
    pair_local: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    cv_rows: list[pd.DataFrame] = []
    lcr_rows: list[pd.DataFrame] = []
    multiscale_rows: list[pd.DataFrame] = []
    multiscale_cv: list[pd.DataFrame] = []
    multiscale_bandwidth_df = pd.DataFrame()
    multiscale_df = pd.DataFrame()
    fit_models: dict[tuple[str, str], pd.DataFrame] = {}
    pixel_complete_all = pixels["Aa_20yr"].notna() & np.isfinite(pixels["log1p_Aa"])
    env_complete = pixel_complete_all.copy()
    for col in env_vars:
        env_complete &= np.isfinite(pd.to_numeric(pixels[col], errors="coerce"))
    env_complete_pixels = int(env_complete.sum())
    mask_info = _mask_arrays(pixels)
    for outcome_index, outcome in enumerate(config["outcomes"]):
        outcome_finite = np.isfinite(pd.to_numeric(pixels[outcome], errors="coerce"))
        features = ["log1p_Aa"]
        specs = [
            ("mask_all_complete_case", pixels.loc[outcome_finite & pixel_complete_all], features),
            ("environment_complete_unadjusted", pixels.loc[outcome_finite & env_complete], features),
            ("environment_adjusted", pixels.loc[outcome_finite & env_complete], ["log1p_Aa", *env_vars]),
        ]
        for spec_index, (population, frame, model_features) in enumerate(specs):
            print(f"GWR {outcome} | {population} | source rows={len(frame):,}", flush=True)
            units = _tile_pixels(frame, outcome, model_features, int(config["tile_cells"]))
            units.attrs["analysis_population"] = population
            if len(units) < 50:
                continue
            local, summary, cv = _fit_gwr(units, outcome, model_features, config,
                                          seed_offset=outcome_index * 100 + spec_index)
            local["analysis_population"] = population
            local["model_family"] = "GWR"
            full_local.append(local)
            summaries.append(summary)
            cv_rows.append(cv)
            fit_models[(outcome, population)] = local
            if population == "environment_adjusted":
                lcr, lcr_summary, lcr_cv = _fit_gwr(
                    units, outcome, model_features, config,
                    ridge_condition=float(config["local_ridge_condition_threshold"]),
                    seed_offset=outcome_index * 100 + spec_index,
                    force_bandwidth=int(summary["bandwidth_neighbors"]),
                )
                lcr["analysis_population"] = population
                lcr["model_family"] = "GWR-LCR"
                full_local.append(lcr)
                summaries.append(lcr_summary)
                cv_rows.append(lcr_cv)
                lcr_rows.append(lcr_summary)
        if outcome in config["key_outcomes"]:
            print(f"Multiscale SVC {outcome} | environment-complete population", flush=True)
            units = _tile_pixels(pixels.loc[outcome_finite & env_complete], outcome, ["log1p_Aa", *env_vars],
                                 int(config["tile_cells"]))
            units.attrs["analysis_population"] = "environment_adjusted"
            svc, svc_cv = _fit_multiscale_svc(units, outcome, ["log1p_Aa", *env_vars], config,
                                              seed_offset=len(multiscale_rows) * 11 + 3)
            multiscale_rows.append(svc)
            multiscale_cv.append(svc_cv)
            exposure_row = svc_cv.loc[svc_cv["feature"].eq("log1p_Aa")].iloc[-1]
            exposure_beta = svc["beta_log1p_Aa_standardized"].to_numpy(dtype=float)
            summaries.append({
                "outcome": outcome, "analysis_population": "environment_adjusted",
                "model_family": "multiscale_SVC_backfitting_MGWR_inspired_noncanonical",
                "environment_adjustment": "+".join(env_vars),
                "n_spatial_units": len(svc), "n_source_pixels_or_pairs": int(svc["n_pixels"].sum()),
                "bandwidth_neighbors": int(exposure_row["bandwidth_neighbors"]),
                "bandwidth_median_distance_m": float(exposure_row["bandwidth_median_distance_m"]),
                "cv_rmse_standardized": float(exposure_row["cv_rmse_standardized_partial"]),
                "local_beta_standardized_mean": float(np.average(exposure_beta, weights=svc["n_pixels"])),
                "local_beta_standardized_median": float(np.nanmedian(exposure_beta)),
                "local_beta_p05": float(np.nanquantile(exposure_beta, .05)),
                "local_beta_p95": float(np.nanquantile(exposure_beta, .95)),
                "negative_effect_ratio": float(np.average(exposure_beta < 0, weights=svc["n_pixels"])),
                "local_r2_weighted_mean": np.nan, "local_condition_p95": np.nan,
                "ridge_alpha_positive_ratio": np.nan, "outcome_guided_selection": True,
            })
    # Matched-pair GWR, keeping all, zero-to-positive, and positive-to-positive populations distinct.
    pairs, pair_counts = _load_pairs(config, pixels)
    print(f"Matched pairs after geographic limit: {pair_counts['included_pairs']:,}", flush=True)
    pair_sets = [("matched_all", pairs),
                 ("matched_zero_positive", pairs.loc[pairs["zero_positive"].astype(bool)]),
                 ("matched_both_positive", pairs.loc[pairs["both_positive"].astype(bool)])]
    for population_index, (population, sample) in enumerate(pair_sets):
        for outcome_index, outcome in enumerate(config["outcomes"]):
            col = f"delta_{outcome}"
            if col not in sample:
                continue
            units = _tile_pairs(sample, outcome, int(config["tile_cells"]) * int(config["cell_size_m"]))
            if len(units) < 30:
                continue
            print(f"Matched-pair GWR {outcome} | {population} | spatial units={len(units):,}", flush=True)
            units.attrs["analysis_population"] = population
            try:
                local, summary, cv = _fit_gwr(units, outcome, ["delta_log1p_Aa"], config,
                                              seed_offset=population_index * 1000 + outcome_index * 10 + 9)
            except ValueError:
                continue
            local["analysis_population"] = population
            local["model_family"] = "matched_pair_GWR"
            pair_local.append(local)
            summaries.append(summary)
            cv_rows.append(cv)
    summary_df = pd.DataFrame(summaries)
    full_local_df = pd.concat(full_local, ignore_index=True) if full_local else pd.DataFrame()
    pair_local_df = pd.concat(pair_local, ignore_index=True) if pair_local else pd.DataFrame()
    cv_df = pd.concat(cv_rows, ignore_index=True) if cv_rows else pd.DataFrame()
    summary_df.to_csv(run_dir / "tables" / "gwr_spatial_model_summary.csv", index=False)
    full_local_df.to_parquet(run_dir / "tables" / "full_sample_local_coefficients.parquet", index=False)
    pair_local_df.to_parquet(run_dir / "tables" / "matched_pair_local_coefficients.parquet", index=False)
    cv_df.to_csv(run_dir / "tables" / "bandwidth_cv.csv", index=False)
    pd.DataFrame(lcr_rows).to_csv(run_dir / "tables" / "gwr_lcr_summary.csv", index=False)
    if multiscale_rows:
        multiscale_df = pd.concat(multiscale_rows, ignore_index=True)
        multiscale_df.to_parquet(run_dir / "tables" / "multiscale_svc_coefficients.parquet", index=False)
        multiscale_bandwidth_df = pd.concat(multiscale_cv, ignore_index=True)
        multiscale_bandwidth_df.to_csv(run_dir / "tables" / "multiscale_svc_bandwidths.csv", index=False)
    # Bandwidth leaderboard exposes comparable full-sample and matched estimates separately.
    summary_df.to_csv(run_dir / "leaderboards" / "phase_i_spatial_model_leaderboard.csv", index=False)
    # The core cartographic template uses the mask boundary and both unadjusted/adjusted SHI, OM, SW surfaces.
    map_paths: list[str] = []
    for outcome in KEY_OUTCOMES:
        map_models = []
        for population, title in [("mask_all_complete_case", "全样本未调整 GWR"),
                                  ("environment_adjusted", "全样本环境调整 GWR")]:
            d = fit_models.get((outcome, population))
            if d is not None:
                map_models.append((f"{LABELS[outcome]}：{title}", d))
        svc_outcome = multiscale_df.loc[multiscale_df["outcome"].eq(outcome)].copy() if len(multiscale_df) else pd.DataFrame()
        if len(svc_outcome):
            svc_outcome["local_beta_standardized"] = svc_outcome["beta_log1p_Aa_standardized"]
            map_models.append((f"{LABELS[outcome]}：多尺度 SVC（MGWR-inspired）", svc_outcome))
        if map_models:
            map_paths.extend(_save_map_panel(run_dir, f"gwr_{outcome}_template", map_models, mask_info,
                                             int(config["map_dpi"])))
    map_paths.extend(_save_multiscale_bandwidth_figure(run_dir, multiscale_bandwidth_df, int(config["map_dpi"])))
    package_status = {name: _package_version(name) for name in ("mgwr", "spglm", "libpysal", "numpy", "scipy", "pandas", "scikit-learn", "statsmodels")}
    report_path = _report(run_dir, summary_df, pair_counts, len(pixels), env_complete_pixels,
                          {"mgwr": package_status["mgwr"], "spglm": package_status["spglm"],
                           "libpysal": package_status["libpysal"]}, config, multiscale_bandwidth_df)
    config_copy = run_dir / "configs" / config_path.name
    provenance = {
        "phase": "I-4", "mode": "EXPLORATORY_ORACLE", "outcome_guided_selection": True,
        "run_id": config["run_id"], "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git("rev-parse", "HEAD"), "git_status_porcelain": _git("status", "--short"),
        "inputs": {"environment_erosion": str(env_path.relative_to(ROOT)), "health": str(health_path.relative_to(ROOT)),
                   "matched_pairs": str((ROOT / config["matched_run_id"] / "pairs" / "candidate_pairs" /
                                         f"{config['matched_candidate']}.parquet").relative_to(ROOT))},
        "configuration": str(config_copy.relative_to(ROOT)), "sample_rows": len(pixels),
        "env_complete_rows": env_complete_pixels, "pair_counts": pair_counts,
        "analysis_unit": "240 m (8x8 30 m pixels) block means with source pixel/pair frequency weights",
        "coordinate_reference": "EPSG:32649", "software": {"python": platform.python_version(), **package_status},
        "methods": {"standard_mgwr": "not run: mgwr/spglm not installed",
                    "multiscale_svc": "custom MGWR-inspired backfitting approximation; not canonical MGWR",
                    "gwr_lcr": "local ridge penalty targets condition number <= configured threshold"},
        "outputs": [str((run_dir / p).relative_to(ROOT)) for p in ["tables/gwr_spatial_model_summary.csv",
                    "tables/full_sample_local_coefficients.parquet", "tables/matched_pair_local_coefficients.parquet",
                    "tables/bandwidth_cv.csv", "reports/PHASE_I_GWR_MGWR_REPORT.md"]] + map_paths,
        "report": report_path,
    }
    (run_dir / "manifests" / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    outputs = [p.relative_to(run_dir).as_posix() for p in run_dir.rglob("*") if p.is_file()]
    (run_dir / "manifests" / "output_manifest.json").write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    status = ["# Run status", "", "Status: authoritative exploratory result", "", "- Phase: I-4",
              "- `outcome_guided_selection=true`", "- Standard MGWR not run because `mgwr` and `spglm` are absent; multiscale-SVC approximation is separately labeled.",
              "- Outputs are 240 m summaries and spatially dependent; local model-based intervals are exploratory.", ""]
    (run_dir / "RUN_STATUS.md").write_text("\n".join(status), encoding="utf-8")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "phase_i_spatial.yaml")
    args = parser.parse_args()
    run_dir = run(args.config)
    print(run_dir.relative_to(ROOT))
    print((run_dir / "reports" / "PHASE_I_GWR_MGWR_REPORT.md").relative_to(ROOT))


if __name__ == "__main__":
    main()
