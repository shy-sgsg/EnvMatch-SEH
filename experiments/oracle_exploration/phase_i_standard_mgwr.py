"""Fit official mgwr package GWR and MGWR on the Phase I balanced sample."""

from __future__ import annotations

import argparse
import json
import platform
import traceback
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from mgwr.gwr import GWR, MGWR
from mgwr.sel_bw import Sel_BW
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
OUTCOMES = ["SHI_candidate", "OM", "SW"]
OUTCOME_LABELS = {"SHI_candidate": "SHI", "OM": "OM", "SW": "SW"}
FEATURES = ["log1p_Aa", "Elevation", "Cur", "TWI", "LST", "NDWI", "NDBSI", "EVI", "NDVI", "Precipitation"]


def _zscore(values: np.ndarray) -> np.ndarray:
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=0))
    if not np.isfinite(sd) or sd <= 1e-12:
        raise ValueError("A predictor or response has no variation in the balanced sample.")
    return (values - mean) / sd


def _fit_summary(result: object, outcome: str, model_family: str, n: int, bandwidth: object) -> dict[str, object]:
    def value(name: str) -> float:
        try:
            return float(getattr(result, name))
        except Exception:
            return float("nan")
    return {
        "outcome": outcome, "outcome_label": OUTCOME_LABELS[outcome], "model_family": model_family,
        "n_spatial_units": n, "bandwidth_neighbors": bandwidth,
        "r_squared": value("R2"), "adjusted_r_squared": value("adj_R2"),
        "aicc": value("aicc"), "effective_parameters": value("ENP"),
        "sigma2": value("sigma2"), "outcome_guided_selection": True,
    }


def fit(sample_path: Path, output_dir: Path, min_neighbors: int = 30,
        max_iter_multi: int = 20, n_chunks: int = 8) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    sample = pd.read_csv(sample_path)
    required = {"x", "y", "n_pixels", "tile_row", "tile_col", *FEATURES, *OUTCOMES}
    missing = sorted(required - set(sample.columns))
    if missing:
        raise ValueError(f"Balanced MGWR sample is missing columns: {missing}")
    numeric = sample[["x", "y", *FEATURES, *OUTCOMES]].apply(pd.to_numeric, errors="coerce")
    valid = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
    sample = sample.loc[valid].copy().reset_index(drop=True)
    coords = sample[["x", "y"]].to_numpy(dtype=float)
    n = len(sample)
    predictors = np.column_stack([_zscore(pd.to_numeric(sample[col], errors="coerce").to_numpy(dtype=float))
                                  for col in FEATURES])
    feature_names = ["Intercept", *FEATURES]
    local_records: list[pd.DataFrame] = []
    bandwidth_records: list[dict[str, object]] = []
    summary_records: list[dict[str, object]] = []
    status_records: list[dict[str, object]] = []
    existing_coef_path = output_dir / "mgwr_local_coefficients.csv"
    existing_summary_path = output_dir / "mgwr_model_summary.csv"
    existing_coefs = pd.read_csv(existing_coef_path) if existing_coef_path.is_file() else pd.DataFrame()
    existing_summaries = pd.read_csv(existing_summary_path) if existing_summary_path.is_file() else pd.DataFrame()
    distance_tree = cKDTree(coords)
    bw_min = min(max(int(min_neighbors), len(FEATURES) + 2), n - 2)
    if n <= bw_min + 2:
        raise ValueError(f"Too few balanced units for the requested MGWR minimum bandwidth: n={n}, min={bw_min}")

    for outcome in OUTCOMES:
        print(f"Standard GWR/MGWR: {OUTCOME_LABELS[outcome]} (n={n})", flush=True)
        try:
            y = _zscore(pd.to_numeric(sample[outcome], errors="coerce").to_numpy(dtype=float)).reshape(-1, 1)
            saved_gwr = existing_coefs.loc[existing_coefs.get("outcome", pd.Series(dtype=str)).eq(outcome) &
                                           existing_coefs.get("model_family", pd.Series(dtype=str)).eq("standard_GWR_same_sample")]
            saved_gwr_summary = existing_summaries.loc[
                existing_summaries.get("outcome", pd.Series(dtype=str)).eq(outcome) &
                existing_summaries.get("model_family", pd.Series(dtype=str)).eq("standard_GWR_same_sample")
            ]
            if len(saved_gwr) == n and len(saved_gwr_summary) == 1:
                gwr_local = saved_gwr.copy()
                gwr_bw = int(float(saved_gwr_summary.iloc[0]["bandwidth_neighbors"]))
                gwr_summary = saved_gwr_summary.iloc[0].to_dict()
                print(f"Reuse completed standard GWR for {OUTCOME_LABELS[outcome]} (k={gwr_bw})", flush=True)
            else:
                gwr_selector = Sel_BW(coords, y, predictors, kernel="gaussian", fixed=False,
                                      constant=True, n_jobs=1)
                gwr_bw = int(gwr_selector.search(criterion="AICc", bw_min=bw_min, bw_max=n - 1,
                                                 tol=1e-3, max_iter=120, verbose=False))
                gwr_result = GWR(coords, y, predictors, gwr_bw, kernel="gaussian", fixed=False,
                                 constant=True, n_jobs=1).fit(max_iter=20, tol=1e-5)
                gwr_local = pd.DataFrame({
                    "x": coords[:, 0], "y": coords[:, 1], "n_pixels": sample["n_pixels"].to_numpy(),
                    "tile_row": sample["tile_row"].to_numpy(), "tile_col": sample["tile_col"].to_numpy(),
                    "outcome": outcome, "model_family": "standard_GWR_same_sample",
                    "local_beta_log1p_Aa_standardized": np.asarray(gwr_result.params)[:, 1],
                    "local_r2": np.asarray(gwr_result.localR2).reshape(-1),
                    "outcome_guided_selection": True,
                })
                gwr_summary = _fit_summary(gwr_result, outcome, "standard_GWR_same_sample", n, gwr_bw)
            local_records.append(gwr_local)
            summary_records.append(gwr_summary)

            mgwr_selector = Sel_BW(coords, y, predictors, kernel="gaussian", fixed=False,
                                   multi=True, constant=True, n_jobs=1)
            bws = mgwr_selector.search(
                criterion="AICc", bw_min=bw_min, bw_max=n - 1,
                tol=1e-3, max_iter=120, tol_multi=1e-3,
                max_iter_multi=int(max_iter_multi), multi_bw_min=[bw_min],
                multi_bw_max=[n - 1], bws_same_times=3, verbose=False,
            )
            mgwr_model = MGWR(coords, y, predictors, mgwr_selector, kernel="gaussian", fixed=False,
                              constant=True, n_jobs=1)
            mgwr_result = mgwr_model.fit(n_chunks=int(n_chunks))
            mgwr_bws = np.asarray(bws[0] if isinstance(bws, tuple) else bws, dtype=float).reshape(-1)
            mgwr_local = pd.DataFrame({
                "x": coords[:, 0], "y": coords[:, 1], "n_pixels": sample["n_pixels"].to_numpy(),
                "tile_row": sample["tile_row"].to_numpy(), "tile_col": sample["tile_col"].to_numpy(),
                "outcome": outcome, "model_family": "standard_MGWR_spatially_balanced_sample",
                "local_beta_log1p_Aa_standardized": np.asarray(mgwr_result.params)[:, 1],
                "local_r2": np.full(n, np.nan),
                "outcome_guided_selection": True,
            })
            local_records.append(mgwr_local)
            summary_records.append(_fit_summary(mgwr_result, outcome, "standard_MGWR_spatially_balanced_sample", n,
                                                np.nan))
            for index, (name, bandwidth) in enumerate(zip(feature_names, mgwr_bws, strict=True)):
                k = int(np.clip(np.rint(bandwidth), 2, n))
                distances = distance_tree.query(coords, k=k)[0]
                median_distance = float(np.median(distances[:, -1]))
                bandwidth_records.append({
                    "outcome": outcome, "outcome_label": OUTCOME_LABELS[outcome], "variable": name,
                    "bandwidth_neighbors": float(bandwidth), "median_neighbor_distance_m": median_distance,
                    "mgwr_variable_index": index, "outcome_guided_selection": True,
                })
            history = mgwr_selector.bw[1] if isinstance(mgwr_selector.bw, tuple) else []
            history_array = np.asarray(history, dtype=float)
            status_records.append({
                "outcome": outcome, "status": "ok", "n_spatial_units": n,
                "gwr_bandwidth_neighbors": gwr_bw,
                "mgwr_iterations": int(history_array.shape[0]) if history_array.ndim >= 1 else 0,
                "mgwr_converged_before_iteration_cap": bool(history_array.ndim >= 1 and history_array.shape[0] < max_iter_multi),
                "error": None, "outcome_guided_selection": True,
            })
        except Exception as exc:
            status_records.append({"outcome": outcome, "status": "error", "n_spatial_units": n,
                                   "error": f"{type(exc).__name__}: {exc}",
                                   "traceback": traceback.format_exc(), "outcome_guided_selection": True})
            print(f"Model failed for {OUTCOME_LABELS[outcome]}: {type(exc).__name__}: {exc}", flush=True)

    if local_records:
        pd.concat(local_records, ignore_index=True).to_csv(output_dir / "mgwr_local_coefficients.csv", index=False)
    pd.DataFrame(bandwidth_records).to_csv(output_dir / "mgwr_variable_bandwidths.csv", index=False)
    pd.DataFrame(summary_records).to_csv(output_dir / "mgwr_model_summary.csv", index=False)
    pd.DataFrame(status_records).to_csv(output_dir / "mgwr_fit_status.csv", index=False)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "package": "mgwr", "mgwr_version": version("mgwr"),
        "spglm_version": version("spglm"), "libpysal_version": version("libpysal"),
        "python_version": platform.python_version(), "n_spatial_units": n,
        "sample_path": sample_path.as_posix(), "predictors": FEATURES,
        "outcomes": OUTCOMES, "standardization": "unweighted z scores within common spatially balanced sample",
        "kernel": "Gaussian", "bandwidth": "adaptive nearest-neighbor count",
        "criterion": "AICc", "gwr_bandwidth_search": {"min": bw_min, "max": n - 1, "tolerance": 1e-3, "max_iterations": 120},
        "mgwr_bandwidth_search": {"min": bw_min, "max": n - 1, "tolerance": 1e-3,
                                  "max_backfitting_iterations": int(max_iter_multi), "same_bandwidth_stop_count": 3},
        "inference_chunks": int(n_chunks), "n_jobs": 1, "outcome_guided_selection": True,
    }
    (output_dir / "standard_mgwr_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output_dir / "mgwr_fit_status.csv", flush=True)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-neighbors", type=int, default=30)
    parser.add_argument("--max-iter-multi", type=int, default=20)
    parser.add_argument("--n-chunks", type=int, default=8)
    args = parser.parse_args()
    fit(args.sample.resolve(), args.output_dir.resolve(), args.min_neighbors, args.max_iter_multi, args.n_chunks)


if __name__ == "__main__":
    main()
