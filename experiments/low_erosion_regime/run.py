"""Run the low-erosion regime analysis in a separate Oracle output directory."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
import yaml

from experiments.low_erosion_regime.distribution_diagnostics import (
    bootstrap_cdf_breakpoints, bootstrap_kde_valleys, empirical_cdf_breakpoint, find_kde_valleys,
    fit_gaussian_mixtures, otsu_thresholds, summarize_values,
)
from experiments.low_erosion_regime.figures import (
    EXPOSURES, OUTCOMES, create_block_composition_figure, create_component_figure,
    create_distribution_figures, create_response_figures,
)
from experiments.low_erosion_regime.matching_comparison import (
    enrich_rematched_pairs, filter_existing_pairs, rematch_filtered_cells,
)
from experiments.low_erosion_regime.response_models import (
    model_trials, outcome_optimized_thresholds, paired_association, paired_model_trials, paired_spatial_validation,
    pixel_association, spatial_validation, two_part_and_segmented_models,
)
from experiments.low_erosion_regime.reports import write_report_package
from experiments.low_erosion_regime.spatial_analysis import (
    connected_components, summarize_low_shares,
)


DIRECTORIES = (
    "config", "diagnostics", "thresholds", "maps", "filtered_pairs", "rematched_pairs",
    "associations", "models", "spatial_validation", "tables", "reports",
    "figures/png", "figures/svg", "figures/pdf", "figures/data", "figures/captions",
)
ENVIRONMENT_PROFILE_COLUMNS = ("SoilType", "Elevation", "TWI", "LST", "NDWI")
PAIR_METHODS = ("M3", "M4")


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _read_frame(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def _write_table(frame: pd.DataFrame, root: Path, stem: str, *, parquet: bool = True) -> None:
    root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(root / f"{stem}.csv", index=False, encoding="utf-8-sig")
    if parquet:
        frame.to_parquet(root / f"{stem}.parquet", index=False)


def _stratified_positive_sample(
    cells: pd.DataFrame, exposure: str, *, sample_size: int, seed: int
) -> np.ndarray:
    values = pd.to_numeric(cells[exposure], errors="coerce").to_numpy(float)
    valid = cells["valid_cell"].fillna(False).astype(bool).to_numpy() & np.isfinite(values) & (values > 0)
    positions = np.flatnonzero(valid)
    if len(positions) <= sample_size:
        return positions
    row = pd.to_numeric(cells["row"], errors="coerce").to_numpy(float)
    col = pd.to_numeric(cells["col"], errors="coerce").to_numpy(float)
    block = np.asarray([f"{int(r // 167)}:{int(c // 167)}" for r, c in zip(row[positions], col[positions], strict=True)])
    _, inverse, counts = np.unique(block, return_inverse=True, return_counts=True)
    allocation = np.floor(sample_size * counts / counts.sum()).astype(int)
    remainder = sample_size - int(allocation.sum())
    if remainder:
        allocation[np.argsort(-counts, kind="stable")[:remainder]] += 1
    rng = np.random.default_rng(seed)
    selected = []
    for group_id, take in enumerate(allocation):
        within = np.flatnonzero(inverse == group_id)
        if take:
            selected.extend(rng.choice(within, min(int(take), len(within)), replace=False).tolist())
    return positions[np.sort(np.asarray(selected, dtype=int))]


def _prepare_run(project_root: Path, config_path: Path, config: dict[str, Any]) -> tuple[Path, dict[str, Path]]:
    if config.get("mode") != "EXPLORATORY_ORACLE" or config.get("outcome_guided_selection") is not True:
        raise ValueError("low-erosion regime exploration requires EXPLORATORY_ORACLE and outcome_guided_selection=true")
    output_root = Path(config["output_root"])
    if output_root.is_absolute() or ".." in output_root.parts or output_root.parts[:2] != ("outputs", "exploratory_oracle"):
        raise ValueError("output_root must remain within outputs/exploratory_oracle")
    run_dir = project_root / output_root / str(config["run_id"])
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    paths = {}
    for name in DIRECTORIES:
        path = run_dir / name
        path.mkdir(parents=True, exist_ok=True)
        paths[name] = path
    paths["figures"] = run_dir / "figures"
    snapshot = yaml.safe_dump(config, allow_unicode=True, sort_keys=False)
    (paths["config"] / "config.yaml").write_text(snapshot, encoding="utf-8")
    (run_dir / "config.yaml").write_text(snapshot, encoding="utf-8")
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=project_root, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        commit = None
    provenance = {
        "phase": "H-low-erosion-regime", "mode": "EXPLORATORY_ORACLE",
        "outcome_guided_selection": True, "run_id": config["run_id"],
        "input_run_id": config["input_run_id"], "phase_f_run_id": config["phase_f_run_id"],
        "phase_g_run_id": config["phase_g_run_id"], "phase_h_run_id": config.get("phase_h_run_id"),
        "git_commit_at_run": commit, "started_at_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "packages": {name: _package_version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "pyarrow")},
        "threshold_selection": "distribution-derived thresholds are outcome blind; outcome-optimized thresholds are stored separately and are exploratory Oracle results",
        "rusle_policy": "use existing Aa only; no factor-product reconstruction or validation",
        "pcnm_policy": "existing PCNM not rerun",
    }
    (run_dir / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    return run_dir, paths


def _load_inputs(project_root: Path, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Path]]:
    input_root = project_root / "outputs" / config["input_run_id"]
    phase_root = project_root / "outputs" / "exploratory_oracle"
    paths = {
        "aligned_environment": input_root / "databases" / "environment_erosion.parquet",
        "aligned_health": input_root / "databases" / "health.parquet",
        "phase_f_pairs": phase_root / config["phase_f_run_id"] / "pairs" / "full_sample_pairs.parquet",
        "phase_g_labels": phase_root / config["phase_g_run_id"] / "tables" / "environment_group_labels.parquet",
    }
    env = _read_frame(paths["aligned_environment"])
    health = _read_frame(paths["aligned_health"])
    pairs = _read_frame(paths["phase_f_pairs"])
    # Health labels are outcomes only; they do not enter regime thresholds,
    # environment grouping, or nearest-neighbour construction.
    hcols = ["cell_id", "valid_cell", *[c for c in config["outcomes"] if c in health]]
    health = health[hcols].rename(columns={"valid_cell": "health_valid_cell"})
    cells = env.merge(health, on="cell_id", how="left", validate="one_to_one")
    cells["valid_cell"] = cells["valid_cell"].fillna(False).astype(bool)
    for exposure in config["exposures"]:
        cells[exposure] = pd.to_numeric(cells[exposure], errors="coerce").where(cells["valid_cell"])
    return cells, pairs, paths


def _threshold_candidates(
    exposure: str, kde: pd.DataFrame, gmm: pd.DataFrame, cdf: pd.DataFrame,
    cdf_bootstrap: pd.DataFrame, values: np.ndarray
) -> pd.DataFrame:
    valid_values = values[np.isfinite(values) & (values > 0)]
    rows: list[dict[str, Any]] = []
    kde_selected = kde.loc[kde["exposure"].eq(exposure)
                           & kde["bandwidth_multiplier"].eq(1.0)
                           & kde["status"].eq("valid_valley")]
    for item in kde_selected.to_dict("records"):
        threshold = float(item["threshold_original"])
        retained = float(np.mean(valid_values > threshold)) if len(valid_values) else np.nan
        rows.append({"exposure": exposure, "candidate_method": f"KDE_{item['bandwidth_base']}_1x",
                     "threshold_original": threshold, "threshold_log10": float(np.log10(threshold)),
                     "retained_fraction_positive": retained, "status": "valid_distribution_candidate",
                     "candidate_source": "distribution_derived", "outcome_guided_selection": False,
                     "is_primary": item["bandwidth_base"] == "scott"})
    bootstrap = cdf_bootstrap.loc[cdf_bootstrap["exposure"].eq(exposure)]
    valid_boot = bootstrap.loc[bootstrap["status"].eq("valid_cdf_breakpoint"), "threshold_original"].dropna()
    valid_fraction = float(len(valid_boot) / len(bootstrap)) if len(bootstrap) else 0.0
    threshold_cv = float(valid_boot.std(ddof=1) / valid_boot.mean()) if len(valid_boot) > 1 and valid_boot.mean() else np.inf
    cdf_row = cdf.loc[cdf["exposure"].eq(exposure) & cdf["status"].eq("valid_cdf_breakpoint")]
    if not cdf_row.empty and valid_fraction >= .70 and threshold_cv <= .50:
        threshold = float(cdf_row.iloc[0]["threshold_original"])
        rows.append({"exposure": exposure, "candidate_method": "empirical_CDF_breakpoint",
                     "threshold_original": threshold, "threshold_log10": float(np.log10(threshold)),
                     "retained_fraction_positive": float(np.mean(valid_values > threshold)),
                     "status": "valid_distribution_candidate", "candidate_source": "distribution_derived",
                     "cdf_bootstrap_valid_fraction": valid_fraction, "cdf_bootstrap_threshold_cv": threshold_cv,
                     "outcome_guided_selection": False, "is_primary": False})
    exp_gmm = gmm.loc[gmm["exposure"].eq(exposure)]
    if not exp_gmm.empty:
        best = exp_gmm.loc[exp_gmm["bic"].idxmin()]
        try:
            overlaps = [float(x) for x in str(best["adjacent_overlap_index"]).split("|") if x and np.isfinite(float(x))]
            boundaries = [float(x) for x in str(best["adjacent_boundaries_log10"]).split("|") if x and np.isfinite(float(x))]
        except (KeyError, ValueError):
            overlaps, boundaries = [], []
        if int(best["n_components"]) > 1 and overlaps and boundaries and overlaps[0] <= .35:
            threshold = float(10 ** boundaries[0])
            rows.append({"exposure": exposure, "candidate_method": f"GMM_BIC_K{int(best['n_components'])}_low_overlap",
                         "threshold_original": threshold, "threshold_log10": float(boundaries[0]),
                         "retained_fraction_positive": float(np.mean(valid_values > threshold)),
                         "status": "valid_distribution_candidate", "candidate_source": "distribution_derived",
                         "outcome_guided_selection": False, "is_primary": False})
    result = pd.DataFrame.from_records(rows)
    if result.empty:
        return pd.DataFrame(columns=["exposure", "threshold_id", "candidate_method", "threshold_original",
                                     "threshold_log10", "retained_fraction_positive", "status",
                                     "candidate_source", "outcome_guided_selection", "is_primary"])
    # Collapse nearly identical methods for filtering while preserving every
    # method-specific value in the diagnostics table.
    result = result.sort_values(["is_primary", "candidate_method"], ascending=[False, True]).copy()
    keep = []
    accepted = []
    for index, row in result.iterrows():
        if any(abs(row["threshold_log10"] - prior) < .03 for prior in accepted):
            continue
        keep.append(index)
        accepted.append(float(row["threshold_log10"]))
    result = result.loc[keep].copy().reset_index(drop=True)
    if not result["is_primary"].fillna(False).any():
        result.loc[0, "is_primary"] = True
        result.loc[0, "primary_fallback_rule"] = "KDE Scott ×1 优先；否则取首个通过分布判据的候选"
    result["threshold_id"] = [f"T{i + 1:02d}" for i in range(len(result))]
    return result


def _add_q_values(frame: pd.DataFrame, p_column: str, group_columns: list[str]) -> pd.DataFrame:
    if p_column not in frame:
        return frame
    result = frame.copy()
    result[f"{p_column}_q"] = np.nan
    valid = result[p_column].notna() & np.isfinite(pd.to_numeric(result[p_column], errors="coerce"))
    for _, index in result.loc[valid].groupby(group_columns, dropna=False, sort=False).groups.items():
        p = pd.to_numeric(result.loc[index, p_column], errors="coerce").to_numpy(float)
        order = np.argsort(p)
        adjusted = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
        q = np.empty(len(p), dtype=float)
        q[order] = np.minimum(adjusted, 1.0)
        result.loc[index, f"{p_column}_q"] = q
    return result


def _pair_quality(pairs: pd.DataFrame, *, source: str, exposure: str, scheme: str) -> dict[str, Any]:
    record: dict[str, Any] = {
        "source": source, "exposure": exposure, "scheme": scheme, "pair_count": int(len(pairs)),
        "outcome_guided_selection": False,
    }
    for column in ("environmental_distance", "geographic_distance", "erosion_difference", "delta_log1p_erosion"):
        values = pd.to_numeric(pairs.get(column, pd.Series(dtype=float)), errors="coerce").dropna()
        record[f"{column}_median"] = float(values.median()) if len(values) else np.nan
        record[f"{column}_p95"] = float(values.quantile(.95)) if len(values) else np.nan
    smds = []
    for variable in ("Elevation", "Cur", "TWI", "LST", "NDWI"):
        high, low = f"{variable}_high", f"{variable}_low"
        if high in pairs and low in pairs:
            h = pd.to_numeric(pairs[high], errors="coerce")
            l = pd.to_numeric(pairs[low], errors="coerce")
            pooled = np.sqrt((h.var() + l.var()) / 2)
            if np.isfinite(pooled) and pooled > 0:
                smds.append(abs(float((h.mean() - l.mean()) / pooled)))
    record["mean_absolute_environment_smd"] = float(np.mean(smds)) if smds else np.nan
    record["same_5km_block_fraction"] = np.nan
    if {"x_high", "x_low", "y_high", "y_low"}.issubset(pairs.columns) and len(pairs):
        high_x = np.floor(pd.to_numeric(pairs["x_high"], errors="coerce") / 5000)
        low_x = np.floor(pd.to_numeric(pairs["x_low"], errors="coerce") / 5000)
        high_y = np.floor(pd.to_numeric(pairs["y_high"], errors="coerce") / 5000)
        low_y = np.floor(pd.to_numeric(pairs["y_low"], errors="coerce") / 5000)
        record["same_5km_block_fraction"] = float(((high_x == low_x) & (high_y == low_y)).mean())
    return record


def _make_pair_population(
    pairs: pd.DataFrame, cells: pd.DataFrame, *, exposure: str, scheme: str,
    threshold: float | None = None, method: str | None = None,
) -> pd.DataFrame:
    selected = pairs.loc[pairs["exposure"].eq(exposure)].copy()
    if method is not None and "method" in selected:
        selected = selected.loc[selected["method"].eq(method)].copy()
    values = cells.set_index("cell_id")[exposure]
    high = pd.to_numeric(selected["high_erosion_cell_id"].map(values), errors="coerce")
    low = pd.to_numeric(selected["low_erosion_cell_id"].map(values), errors="coerce")
    if scheme == "positive_only":
        selected = selected.loc[high.gt(0) & low.gt(0)].copy()
    elif scheme == "S1_zero_only":
        selected = selected.loc[high.eq(0) & low.eq(0)].copy()
    elif scheme == "distribution_threshold":
        selected = selected.loc[high.gt(threshold) & low.gt(threshold)].copy()
    selected["retention_scheme"] = scheme
    selected["threshold_original"] = threshold
    return selected


def run_low_erosion_regime(project_root: Path, config_path: Path) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir, paths = _prepare_run(project_root, config_path, config)
    cells, phase_f_pairs, input_paths = _load_inputs(project_root, config)
    inventory = []
    for role, path in input_paths.items():
        frame = _read_frame(path)
        inventory.append({"role": role, "path": str(path.relative_to(project_root)), "row_count": len(frame),
                          "column_count": len(frame.columns), "columns": "|".join(frame.columns.astype(str))})
    _write_table(pd.DataFrame(inventory), paths["diagnostics"], "input_inventory")
    config_copy = dict(config)
    config_copy["config_source"] = str(config_path.relative_to(project_root))

    # Distribution diagnostics use the complete valid cropland population;
    # non-cropland values are explicitly represented as NoData.
    summary_rows, kde_frames, gmm_frames, cdf_frames, otsu_frames, bootstrap_frames, cdf_bootstrap_frames = [], [], [], [], [], [], []
    gmm_assignment_by_exposure: dict[str, pd.DataFrame] = {}
    threshold_frames = []
    for exposure in config["exposures"]:
        values = cells[exposure].to_numpy(dtype=float)
        summary = summarize_values(values)
        summary_rows.append({"exposure": exposure, **summary, "analysis_scope": "valid_cell=True"})
        sample_positions = _stratified_positive_sample(
            cells, exposure, sample_size=int(config["distribution"]["positive_sample_size_kde"]),
            seed=int(config["seed"]),
        )
        kde = find_kde_valleys(values[sample_positions], exposure=exposure,
                               multipliers=tuple(config["distribution"]["kde_bandwidth_multipliers"]))
        kde["candidate_source"] = "kde"
        kde["fit_sample_n"] = len(sample_positions)
        kde["sample_design"] = "proportional 5 km block-stratified positive sample"
        # Retention is always recalculated on the full valid population.
        positive = values[np.isfinite(values) & (values > 0)]
        for index, row in kde.iterrows():
            if pd.notna(row["threshold_original"]):
                kde.loc[index, "retained_fraction_positive"] = float(np.mean(positive > row["threshold_original"]))
        kde_frames.append(kde)
        gmm, assignments = fit_gaussian_mixtures(
            values, exposure=exposure, seed=int(config["seed"]),
            sample_size=int(config["distribution"]["positive_sample_size_gmm"]),
            rows=cells["row"].to_numpy(), cols=cells["col"].to_numpy(),
        )
        if not gmm.empty:
            gmm["sample_design"] = "proportional 5 km block-stratified positive sample"
        gmm_frames.append(gmm)
        assignments["cell_id"] = assignments["source_index"].map(cells["cell_id"])
        gmm_assignment_by_exposure[exposure] = assignments
        cdf = empirical_cdf_breakpoint(values, exposure=exposure)
        cdf["candidate_source"] = "empirical_cdf"
        cdf_frames.append(cdf)
        otsu = otsu_thresholds(values, exposure=exposure,
                               bins=tuple(config["distribution"]["otsu_histogram_bins"]))
        otsu["candidate_source"] = "otsu_comparison"
        otsu_frames.append(otsu)
        bootstrap = bootstrap_kde_valleys(
            values, cells["row"].to_numpy(), cells["col"].to_numpy(), exposure=exposure,
            replicates=int(config["distribution"]["bootstrap_spatial_replicates"]),
            seed=int(config["seed"]), sample_size=int(config["distribution"]["bootstrap_sample_size"]),
        )
        bootstrap_frames.append(bootstrap)
        cdf_bootstrap = bootstrap_cdf_breakpoints(
            values, cells["row"].to_numpy(), cells["col"].to_numpy(), exposure=exposure,
            replicates=int(config["distribution"]["bootstrap_spatial_replicates"]),
            seed=int(config["seed"]), sample_size=max(5000, int(config["distribution"]["bootstrap_sample_size"])),
        )
        cdf_bootstrap_frames.append(cdf_bootstrap)
    summaries = pd.DataFrame(summary_rows)
    kde_all = pd.concat(kde_frames, ignore_index=True)
    gmm_all = pd.concat(gmm_frames, ignore_index=True)
    cdf_all = pd.concat(cdf_frames, ignore_index=True)
    otsu_all = pd.concat(otsu_frames, ignore_index=True)
    bootstrap_all = pd.concat(bootstrap_frames, ignore_index=True)
    cdf_bootstrap_all = pd.concat(cdf_bootstrap_frames, ignore_index=True)
    for exposure in config["exposures"]:
        candidate = _threshold_candidates(
            exposure,
            kde_all.loc[kde_all["exposure"].eq(exposure)],
            gmm_all.loc[gmm_all["exposure"].eq(exposure)],
            cdf_all.loc[cdf_all["exposure"].eq(exposure)],
            cdf_bootstrap_all.loc[cdf_bootstrap_all["exposure"].eq(exposure)],
            cells[exposure].to_numpy(float),
        )
        threshold_frames.append(candidate)
    thresholds = pd.concat(threshold_frames, ignore_index=True) if threshold_frames else pd.DataFrame()
    _write_table(summaries, paths["diagnostics"], "exposure_distribution_summary")
    _write_table(kde_all, paths["diagnostics"], "kde_bandwidth_diagnostics")
    _write_table(gmm_all, paths["diagnostics"], "gaussian_mixture_diagnostics")
    _write_table(cdf_all, paths["diagnostics"], "empirical_cdf_breakpoints")
    _write_table(otsu_all, paths["diagnostics"], "otsu_comparison")
    _write_table(bootstrap_all, paths["diagnostics"], "spatial_kde_bootstrap")
    _write_table(cdf_bootstrap_all, paths["diagnostics"], "spatial_cdf_bootstrap")
    for exposure, assignments in gmm_assignment_by_exposure.items():
        assignments.to_parquet(paths["diagnostics"] / f"gmm_posterior_assignments_{exposure}.parquet", index=False)
    _write_table(thresholds, paths["thresholds"], "distribution_threshold_candidates")

    # Spatial maps and population composition.
    profile_frames, component_frames, block_frames = [], [], []
    for exposure in config["exposures"]:
        primary = thresholds.loc[thresholds["exposure"].eq(exposure) & thresholds["is_primary"].fillna(False)]
        threshold = float(primary.iloc[0]["threshold_original"]) if not primary.empty else None
        profiles = summarize_low_shares(cells, exposure=exposure, threshold=threshold,
                                        group_columns=ENVIRONMENT_PROFILE_COLUMNS)
        profile_frames.append(profiles)
        if threshold is not None:
            components, component_labels = connected_components(cells, exposure=exposure, threshold=threshold)
            components["threshold_original"] = threshold
            component_frames.append(components)
        else:
            component_labels = None
        assignments = gmm_assignment_by_exposure[exposure]
        uncertain_ids = set(assignments.loc[assignments["classification_status"].eq("uncertain"), "cell_id"].dropna().astype(int))
        from experiments.low_erosion_regime.spatial_analysis import regime_grid
        grid, _, _ = regime_grid(cells, exposure=exposure, threshold=threshold, uncertain_cell_ids=uncertain_ids)
        rows = pd.to_numeric(cells["row"], errors="coerce").to_numpy(int)
        cols = pd.to_numeric(cells["col"], errors="coerce").to_numpy(int)
        valid = cells["valid_cell"].fillna(False).astype(bool).to_numpy() & np.isfinite(cells[exposure].to_numpy(float))
        block_rows = np.floor(rows[valid] / 167).astype(int)
        block_cols = np.floor(cols[valid] / 167).astype(int)
        regime_codes = grid[rows[valid] - rows.min(), cols[valid] - cols.min()]
        blocks = pd.DataFrame({"block_id": [f"{r}_{c}" for r, c in zip(block_rows, block_cols, strict=True)],
                               "regime_code": regime_codes})
        block_counts = blocks.groupby(["block_id", "regime_code"], observed=True).size().rename("cell_count").reset_index()
        block_counts["exposure"] = exposure
        block_counts["threshold_original"] = threshold
        block_frames.append(block_counts)
        if component_labels is not None:
            pixels = pd.DataFrame({"component_id": component_labels[rows[valid] - rows.min(), cols[valid] - cols.min()]})
            pixels = pixels.loc[pixels["component_id"].gt(0)].groupby("component_id").size().rename("pixel_count").reset_index()
            pixels["exposure"] = exposure
            pixels["area_m2"] = pixels["pixel_count"] * 900
            pixels["area_ha"] = pixels["pixel_count"] * .09
            component_frames[-1] = pixels.merge(component_frames[-1], on=["exposure", "component_id"], how="left", suffixes=("", "_summary"))
    profiles_all = pd.concat(profile_frames, ignore_index=True)
    components_all = pd.concat(component_frames, ignore_index=True) if component_frames else pd.DataFrame()
    blocks_all = pd.concat(block_frames, ignore_index=True) if block_frames else pd.DataFrame()
    _write_table(profiles_all, paths["maps"], "regime_share_by_environment_profile")
    _write_table(components_all, paths["maps"], "connected_components")
    _write_table(blocks_all, paths["maps"], "regime_composition_by_5km_block")
    figures = create_distribution_figures(
        paths["figures"], cells, summaries, kde_all, gmm_all, gmm_assignment_by_exposure,
        cdf_all, otsu_all, pd.concat([bootstrap_all.assign(method="KDE"), cdf_bootstrap_all], ignore_index=True),
        profiles_all, thresholds,
    )
    component_figure = create_component_figure(paths["figures"], components_all)
    if component_figure:
        figures.append(component_figure)
    figures.append(create_block_composition_figure(paths["figures"], cells, thresholds, gmm_assignment_by_exposure))

    # Existing-pair filtering and fixed/refitted rematching.
    filtered_pair_sets: dict[tuple[str, str], pd.DataFrame] = {}
    rematched_sets: dict[tuple[str, str, str, str], pd.DataFrame] = {}
    quality_rows, rematch_diagnostics = [], []
    for exposure in config["exposures"]:
        for method in PAIR_METHODS:
            baseline = filter_existing_pairs(phase_f_pairs, cells, exposure=exposure, threshold=None, method=method)
            filtered_pair_sets[(exposure, method, "S0_all")] = baseline
            _write_table(baseline, paths["filtered_pairs"], f"existing_{method}_{exposure}_S0_all")
            quality_rows.append(_pair_quality(baseline, source=f"阶段 F {method}", exposure=exposure, scheme="S0_all"))
            both_positive = _make_pair_population(phase_f_pairs, cells, exposure=exposure, scheme="positive_only", method=method)
            filtered_pair_sets[(exposure, method, "S1_both_positive")] = both_positive
            _write_table(both_positive, paths["filtered_pairs"], f"existing_{method}_{exposure}_S1_both_positive")
            quality_rows.append(_pair_quality(both_positive, source=f"阶段 F {method}", exposure=exposure, scheme="S1_both_positive"))
            zero_only = _make_pair_population(phase_f_pairs, cells, exposure=exposure, scheme="S1_zero_only", method=method)
            filtered_pair_sets[(exposure, method, "S1_zero_only")] = zero_only
            _write_table(zero_only, paths["filtered_pairs"], f"existing_{method}_{exposure}_S1_zero_only")
            quality_rows.append(_pair_quality(zero_only, source=f"阶段 F {method}", exposure=exposure, scheme="S1_zero_only"))
            exp_thresholds = thresholds.loc[thresholds["exposure"].eq(exposure)]
            for trow in exp_thresholds.to_dict("records"):
                t_id = trow["threshold_id"]
                threshold = float(trow["threshold_original"])
                filtered = filter_existing_pairs(phase_f_pairs, cells, exposure=exposure, threshold=threshold, method=method)
                scheme = f"{t_id}_{trow['candidate_method']}"
                filtered["scheme"] = scheme
                filtered_pair_sets[(exposure, method, scheme)] = filtered
                _write_table(filtered, paths["filtered_pairs"], f"existing_{method}_{exposure}_{t_id}")
                quality_rows.append(_pair_quality(filtered, source=f"阶段 F {method}", exposure=exposure, scheme=scheme))
                for group_mode in config["matching"]["rematch_groups"]:
                    rematched, diagnostics = rematch_filtered_cells(
                        cells, phase_f_pairs, exposure=exposure, threshold=threshold, method=method,
                        group_mode=group_mode, seed=int(config["seed"]),
                        environment_distance_limit=float(config["matching"]["environment_caliper_standardized"]),
                        distance_limit_m=float(config["matching"]["geographic_radius_m"]),
                        rank_gap=float(config["matching"]["erosion_rank_gap"]),
                    )
                    diagnostics["threshold_id"] = t_id
                    diagnostics["candidate_method"] = trow["candidate_method"]
                    rematch_diagnostics.append(diagnostics)
                    rematched = enrich_rematched_pairs(rematched, cells, tuple(config["outcomes"]), exposure=exposure)
                    rematched["scheme"] = f"{t_id}_{group_mode}"
                    rematched_sets[(exposure, method, t_id, group_mode)] = rematched
                    _write_table(rematched, paths["rematched_pairs"], f"rematched_{method}_{exposure}_{t_id}_{group_mode}")
                    quality_rows.append(_pair_quality(rematched, source=f"重配对 {method} · {group_mode}",
                                                      exposure=exposure, scheme=f"{t_id}_{group_mode}"))
    pair_quality = pd.DataFrame(quality_rows)
    _write_table(pair_quality, paths["diagnostics"], "pair_quality_comparison")
    _write_table(pd.DataFrame(rematch_diagnostics), paths["diagnostics"], "rematching_diagnostics")

    # Response associations and model searches are performed only after all
    # distribution rules and matching populations have been defined.
    associations = []
    model_frames, segmented_frames, oracle_threshold_frames = [], [], []
    spatial_frames, paired_spatial_frames = [], []
    seeds = tuple(map(int, config["models"]["spatial_validation_seeds"]))
    for exposure in config["exposures"]:
        exp_thresholds = thresholds.loc[thresholds["exposure"].eq(exposure)]
        pixel_schemes: list[tuple[str, float | None]] = [("S0_all", None), ("S1_zero_only", None), ("positive_only", None)]
        pixel_schemes.extend(("distribution_threshold", float(r["threshold_original"])) for r in exp_thresholds.to_dict("records"))
        pixel_schemes = list(dict.fromkeys(pixel_schemes))
        primary = exp_thresholds.loc[exp_thresholds["is_primary"].fillna(False)]
        primary_threshold = float(primary.iloc[0]["threshold_original"]) if not primary.empty else None
        for outcome in config["outcomes"]:
            for scheme, threshold in pixel_schemes:
                associations.append(pixel_association(cells, exposure=exposure, outcome=outcome,
                                                      scheme=scheme, threshold=threshold, seed=int(config["seed"])))
            # Outcome-optimized thresholds are segregated into Oracle-only tables.
            oracle = outcome_optimized_thresholds(cells, exposure=exposure, outcome=outcome,
                                                   seed=int(config["seed"]), sample_size=int(config["models"]["fixed_sample_size"]))
            if not oracle.empty:
                oracle_threshold_frames.append(oracle)
            model_schemes = [("S0_all", None), ("positive_only", None)]
            model_schemes.extend(("distribution_threshold", float(r["threshold_original"]))
                                 for r in exp_thresholds.to_dict("records"))
            for scheme, threshold in model_schemes:
                model_frames.append(model_trials(cells, exposure=exposure, outcome=outcome, scheme=scheme,
                                                 threshold=threshold, seed=int(config["seed"]),
                                                 sample_size=int(config["models"]["fixed_sample_size"])))
            segmented_frames.append(two_part_and_segmented_models(
                cells, exposure=exposure, outcome=outcome, seed=int(config["seed"]),
                sample_size=50000, threshold=primary_threshold,
            ))
            # Spatial validation uses fixed distribution-derived populations.
            spatial_schemes = [("positive_only", None)]
            if primary_threshold is not None:
                spatial_schemes.append(("distribution_threshold", primary_threshold))
            for scheme, threshold in spatial_schemes:
                spatial_frames.append(spatial_validation(
                    cells, exposure=exposure, outcome=outcome, scheme=scheme, threshold=threshold,
                    seeds=seeds, buffer_m=int(config["models"]["spatial_validation_buffer_m"]),
                    sample_size=int(config["models"]["fixed_sample_size"]),
                ))
            for method in PAIR_METHODS:
                populations = [
                    ("S0_all", filtered_pair_sets[(exposure, method, "S0_all")]),
                    ("S1_both_positive", filtered_pair_sets[(exposure, method, "S1_both_positive")]),
                    ("S1_zero_only", filtered_pair_sets[(exposure, method, "S1_zero_only")]),
                ]
                for key, rematched in rematched_sets.items():
                    exp, method_key, threshold_id, group_mode = key
                    if exp == exposure and method_key == method:
                        populations.append((f"{threshold_id}_{group_mode}", rematched))
                exp_thresholds_records = exp_thresholds.to_dict("records")
                for trow in exp_thresholds_records:
                    populations.append((f"{trow['threshold_id']}_{trow['candidate_method']}",
                                        filtered_pair_sets[(exposure, method, f"{trow['threshold_id']}_{trow['candidate_method']}")]))
                for scheme, population in populations:
                    associations.append(paired_association(population, exposure=exposure, outcome=outcome,
                                                           scheme=scheme, method=method,
                                                           threshold=primary_threshold if scheme.startswith("T") else None,
                                                           seed=int(config["seed"])))
                    model_frames.append(paired_model_trials(population, exposure=exposure, outcome=outcome,
                                                            method=method, scheme=scheme,
                                                            threshold=primary_threshold if scheme.startswith("T") else None,
                                                            seed=int(config["seed"]),
                                                            sample_size=int(config["models"]["fixed_sample_size"])))
                # Ten whole-block splits; test outcomes are never used for threshold or model selection.
                base_positive = filtered_pair_sets[(exposure, method, "S1_both_positive")]
                paired_spatial_frames.append(paired_spatial_validation(
                    base_positive, exposure=exposure, outcome=outcome, method=method,
                    scheme="S1_both_positive", seeds=seeds,
                    buffer_m=int(config["models"]["spatial_validation_buffer_m"]),
                ))
                for trow in exp_thresholds_records:
                    threshold = float(trow["threshold_original"])
                    scheme = f"{trow['threshold_id']}_{trow['candidate_method']}"
                    filtered = filtered_pair_sets[(exposure, method, scheme)]
                    paired_spatial_frames.append(paired_spatial_validation(
                        filtered, exposure=exposure, outcome=outcome, method=method, scheme=scheme,
                        threshold=threshold, seeds=seeds,
                        buffer_m=int(config["models"]["spatial_validation_buffer_m"]),
                    ))
    association_frame = pd.DataFrame(associations)
    if not association_frame.empty:
        for pcol in ("spearman_p", "wilcoxon_p", "sign_test_p"):
            association_frame = _add_q_values(association_frame, pcol, ["analysis_unit", "exposure", "method", "scheme"])
    _write_table(association_frame, paths["associations"], "response_associations")
    model_parts = [frame.dropna(axis=1, how="all") for frame in model_frames if not frame.empty]
    model_frame = pd.concat(model_parts, ignore_index=True, sort=False) if model_parts else pd.DataFrame()
    _write_table(model_frame, paths["models"], "model_search_trials")
    segmented_frame = pd.concat(segmented_frames, ignore_index=True) if segmented_frames else pd.DataFrame()
    _write_table(segmented_frame, paths["models"], "zero_indicator_and_segmented_models")
    oracle_thresholds = pd.concat(oracle_threshold_frames, ignore_index=True) if oracle_threshold_frames else pd.DataFrame()
    _write_table(oracle_thresholds, paths["models"], "outcome_optimized_threshold_trials")
    spatial_parts = [frame.dropna(axis=1, how="all") for frame in spatial_frames if not frame.empty]
    pair_spatial_parts = [frame.dropna(axis=1, how="all") for frame in paired_spatial_frames if not frame.empty]
    spatial_frame = pd.concat(spatial_parts, ignore_index=True, sort=False) if spatial_parts else pd.DataFrame()
    paired_spatial_frame = pd.concat(pair_spatial_parts, ignore_index=True, sort=False) if pair_spatial_parts else pd.DataFrame()
    _write_table(spatial_frame, paths["spatial_validation"], "pixel_spatial_validation")
    _write_table(paired_spatial_frame, paths["spatial_validation"], "pair_spatial_validation")
    figures.extend(create_response_figures(
        paths["figures"], cells, association_frame, model_frame, segmented_frame,
        spatial_frame, paired_spatial_frame, pair_quality,
        seed=int(config["seed"]), sample_size=int(config["models"]["fixed_sample_size"]),
    ))

    # Oracle selection record reports apparent maxima separately from spatial tests.
    if not model_frame.empty:
        successful = model_frame.loc[model_frame["status"].eq("ok")].copy()
        if "r_squared_apparent" not in successful:
            successful["r_squared_apparent"] = successful.get("r_squared_apparent", np.nan)
        apparent_key = "r_squared_apparent" if "r_squared_apparent" in successful else "r_squared"
        if apparent_key in successful:
            records = []
            group_cols = [c for c in ("analysis_unit", "exposure", "outcome", "method", "scheme") if c in successful]
            for group, part in successful.groupby(group_cols, dropna=False, sort=False):
                if not isinstance(group, tuple):
                    group = (group,)
                record = dict(zip(group_cols, group, strict=True))
                record["max_apparent_r_squared"] = pd.to_numeric(part[apparent_key], errors="coerce").max()
                negative = part.loc[pd.to_numeric(part.get("spearman_rho", np.nan), errors="coerce") < 0]
                record["max_apparent_r_squared_negative_direction"] = pd.to_numeric(
                    negative[apparent_key], errors="coerce"
                ).max() if not negative.empty else np.nan
                record["best_apparent_model"] = part.loc[pd.to_numeric(part[apparent_key], errors="coerce").idxmax(), "model_family"]
                record["strongest_negative_spearman"] = pd.to_numeric(part.get("spearman_rho", np.nan), errors="coerce").min()
                record["outcome_guided_selection"] = True
                record["selection_scope"] = "apparent_same_sample_oracle_only"
                records.append(record)
            _write_table(pd.DataFrame(records), paths["models"], "oracle_apparent_selection_summary")

    # Summary figures and manifest are produced after all inspectable source tables exist.
    manifest_path = paths["figures"] / "figure_manifest.json"
    manifest_path.write_text(json.dumps(figures, ensure_ascii=False, indent=2), encoding="utf-8")
    provenance_path = run_dir / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance.update({"completed_at_utc": datetime.now(UTC).isoformat(),
                       "cell_rows": len(cells), "valid_cropland_cells": int(cells["valid_cell"].sum()),
                       "phase_f_pair_rows": len(phase_f_pairs), "threshold_candidates": len(thresholds),
                       "figure_count": len(figures), "figure_formats": ["png", "svg", "pdf"]})
    provenance_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "checkpoints.json").write_text(json.dumps(
        {"distribution_diagnostics_complete": True, "spatial_analysis_complete": True,
         "matching_comparison_complete": True, "response_models_complete": True,
         "figure_generation_complete": True}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report_package(project_root, run_dir, config)
    return run_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="运行低侵蚀值群体诊断与响应比较")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path("config/low_erosion_regime.yaml"))
    args = parser.parse_args(argv)
    project_root = args.project_root.resolve()
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    run_dir = run_low_erosion_regime(project_root, config_path)
    print(run_dir.relative_to(project_root))
    return 0


if __name__ == "__main__":
    sys.exit(main())
