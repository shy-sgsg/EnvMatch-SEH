"""Refit corrected CEC Phase H H2/H3 results and replace affected figure assets.

The original Phase H Stage 2 candidate-pair pool is held fixed. Corrected CEC
is re-ranked within that pool, then the affected H core figures are regenerated
with the non-CEC records retained from the original exploratory run.
"""

from __future__ import annotations

import gzip
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

from experiments.oracle_exploration.phase_h_figures import (
    _curve_panel,
    _figure_3,
    _figure_4,
    _figure_7,
    _figure_8,
    _figure_9,
    _figure_11,
    _spline_curve,
)
import experiments.oracle_exploration.phase_h_figures as phase_h_figures
from experiments.oracle_exploration.phase_h_groups import (
    build_environment_group_profiles,
    build_group_method_comparison,
)
from experiments.oracle_exploration.phase_h_search import (
    MODEL_SCHEDULE_VERSION,
    _add_pair_features,
    _model_trial,
    _pixel_view,
)
from experiments.oracle_exploration.phase_h_validation import (
    run_phase_h_period_comparison,
    run_phase_h_spatial_validation,
)


WORKSPACE = Path.cwd()
OLD_RUN = WORKSPACE / "outputs/exploratory_oracle/phase_h_20261028_seed20261028_h2"
CORRECTED_INPUT = WORKSPACE / "outputs/exploratory_oracle/phase_i/cec_corrected_input_20_2p9_plus18_20261031"
FIGURE_ROOT = WORKSPACE / "docs/reviews/2026-10-29/phase-h/artifacts/phase_h_core/figures"
OUTPUT_ROOT = WORKSPACE / "outputs/exploratory_oracle/phase_h_cec_corrected_h2h3_refit_20261101"
CHART_RUN = OUTPUT_ROOT / "chart_render"
FORMULA = "CEC_corrected = (CEC - 20 * Erosion) / 2.9 + 18"
EROSION_DEFINITION = "Aa_20yr / full_valid_aligned_extent_max(Aa_20yr) + Aa_2020 / full_valid_aligned_extent_max(Aa_2020)"
FIGURE_NOTE = (
    "CEC 按 CEC_corrected = (CEC − 20 × Erosion) / 2.9 + 18 校正；Erosion 为 Aa_20yr、Aa_2020 "
    "分别除以各自完整有效对齐栅格全域最大值后的和。H2/H3 使用原有 12 个 Stage 2 候选配对集合（每个侵蚀时期 6 个），"
    "配对成员固定，并在校正 CEC 上重选候选/模型；其余健康指标沿用原值。"
)


def _read_csv(path: Path) -> pd.DataFrame:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as stream:
            return pd.read_csv(stream, low_memory=False)
    return pd.read_csv(path, low_memory=False)


def _write_gzip_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8-sig", newline="", compresslevel=6) as stream:
        frame.to_csv(stream, index=False)


def _json_value(value: Any, default: Any) -> Any:
    if value is None or (isinstance(value, float) and np.isnan(value)) or value == "":
        return default
    return json.loads(str(value))


def _load_cells() -> pd.DataFrame:
    health = pd.read_parquet(CORRECTED_INPUT / "databases/health.parquet")
    environment = pd.read_parquet(CORRECTED_INPUT / "databases/environment_erosion.parquet")
    drop = [name for name in ("row", "col", "x", "y", "valid_cell") if name in environment]
    cells = health.merge(
        environment.drop(columns=drop), on="cell_id", how="left", validate="one_to_one"
    )
    if "CEC" not in cells or not np.isfinite(pd.to_numeric(cells["CEC"], errors="coerce")).any():
        raise ValueError("Corrected aligned CEC values are unavailable")
    return cells


def _correct_pair_cec(pairs: pd.DataFrame, lookup: pd.Series) -> tuple[pd.DataFrame, int]:
    result = pairs.copy()
    high = pd.to_numeric(result["high_erosion_cell_id"].map(lookup), errors="coerce")
    low = pd.to_numeric(result["low_erosion_cell_id"].map(lookup), errors="coerce")
    valid = np.isfinite(high.to_numpy(dtype=float)) & np.isfinite(low.to_numpy(dtype=float))
    dropped = int((~valid).sum())
    result = result.loc[valid].copy()
    high = high.loc[valid]
    low = low.loc[valid]
    for name in ("high_CEC", "CEC_high"):
        result[name] = high.to_numpy(dtype=float)
    for name in ("low_CEC", "CEC_low"):
        result[name] = low.to_numpy(dtype=float)
    result["delta_CEC"] = high.to_numpy(dtype=float) - low.to_numpy(dtype=float)
    return result, dropped


def _candidate_metadata(row: pd.Series) -> dict[str, Any]:
    candidate = row.to_dict()
    candidate["environment_variables"] = _json_value(candidate.get("environment_variables_json"), [])
    candidate["environment_weights"] = _json_value(candidate.get("environment_weights_json"), None)
    candidate["matching_diagnostics"] = _json_value(candidate.get("matching_diagnostics_json"), {})
    candidate["n_matched_pairs"] = candidate.get("n_matched_pairs_candidate")
    candidate["n_matched_pixels"] = candidate.get("n_matched_pixels_candidate")
    candidate["area_coverage"] = candidate.get("area_coverage_candidate")
    return candidate


def _fit_corrected_cec_trials(
    cells: pd.DataFrame,
    old_trials: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], list[dict[str, Any]]]:
    selected = old_trials.loc[
        old_trials["stage"].eq("stage2_full_sample")
        & old_trials["analysis_population"].eq("phase_h_matched_only")
        & old_trials["outcome"].eq("CEC")
        & old_trials["subset_name"].eq("all")
        & old_trials["fit_scope"].eq("exposure_only")
        & old_trials["evaluation_set"].eq("apparent")
    ].copy()
    if selected.empty:
        raise ValueError("Original Phase H Stage 2 CEC candidate/model schedule is unavailable")
    lookup = cells.set_index("cell_id")["CEC"]
    candidates = selected.sort_values(["exposure", "stage1_rank", "candidate_id"], kind="stable").drop_duplicates("candidate_id")
    families = {
        (candidate_id, estimand): sorted(group["model_family"].astype(str).unique())
        for (candidate_id, estimand), group in selected.groupby(["candidate_id", "estimand"], sort=True)
    }
    records: list[dict[str, Any]] = []
    corrected_pairs: dict[str, pd.DataFrame] = {}
    support_ledger: list[dict[str, Any]] = []
    pair_root = OLD_RUN / "pairs/candidate_pairs"
    for candidate_index, (_, row) in enumerate(candidates.iterrows(), start=1):
        candidate = _candidate_metadata(row)
        candidate_id = str(candidate["candidate_id"])
        original_pairs = pd.read_parquet(pair_root / f"{candidate_id}.parquet")
        pairs, dropped_pairs = _correct_pair_cec(original_pairs, lookup)
        support_ledger.append({
            "candidate_id": candidate_id,
            "original_pairs": int(len(original_pairs)),
            "pairs_with_corrected_CEC": int(len(pairs)),
            "pairs_dropped_for_missing_corrected_CEC": dropped_pairs,
        })
        matched_pairs = _add_pair_features(pairs, candidate, ["CEC"])
        for estimand in ("matched_pixel", "matched_pair"):
            frame = matched_pairs if estimand == "matched_pair" else _pixel_view(cells, matched_pairs, candidate)
            for family in families.get((candidate_id, estimand), []):
                trial = _model_trial(
                    frame,
                    config=config,
                    candidate=candidate,
                    estimand=estimand,
                    outcome="CEC",
                    subset_name="all",
                    model_family=family,
                    fit_scope="exposure_only",
                    stage="stage2_full_sample",
                )
                trial.update(
                    model_schedule_version=MODEL_SCHEDULE_VERSION,
                    run_id="phase_h_cec_corrected_h2h3_refit_20261101",
                    input_run_id="corrected_CEC_full_aligned_database_20261031",
                    cec_correction_status="corrected formula; fixed historical candidate pair membership",
                )
                records.append(trial)
        corrected_pairs[candidate_id] = pairs
        print(f"CEC refit candidates {candidate_index}/{len(candidates)}: {candidate_id}", flush=True)
        del original_pairs, matched_pairs
    result = pd.DataFrame.from_records(records)
    result.to_csv(OUTPUT_ROOT / "tables/phase_h_cec_model_trials.csv", index=False, encoding="utf-8-sig")
    return result, corrected_pairs, support_ledger


def _rank_rows(
    cells: pd.DataFrame,
    trials: pd.DataFrame,
    corrected_pairs: dict[str, pd.DataFrame],
    old_trials: pd.DataFrame,
) -> pd.DataFrame:
    from experiments.oracle_exploration.phase_h_search import _add_pair_features
    from experiments.oracle_exploration.phase_h_search import _pixel_view

    candidates = old_trials.loc[
        old_trials["stage"].eq("stage2_full_sample")
        & old_trials["outcome"].eq("CEC")
        & old_trials["subset_name"].eq("all")
        & old_trials["fit_scope"].eq("exposure_only")
    ].drop_duplicates("candidate_id")
    rows: list[dict[str, Any]] = []
    candidate_map = {str(row.candidate_id): row for row in candidates.itertuples(index=False)}
    for trial in trials[["candidate_id", "exposure", "estimand"]].drop_duplicates().itertuples(index=False):
        candidate_id = str(trial.candidate_id)
        candidate = _candidate_metadata(pd.Series(candidate_map[candidate_id]._asdict()))
        pairs = _add_pair_features(corrected_pairs[candidate_id], candidate, ["CEC"])
        if trial.estimand == "matched_pixel":
            frame = _pixel_view(cells, pairs, candidate)
            x = pd.to_numeric(frame[str(trial.exposure)], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(frame["CEC"], errors="coerce").to_numpy(dtype=float)
        else:
            x = pd.to_numeric(pairs["delta_log1p_Aa"], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(pairs["delta_CEC"], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        x, y = x[finite], y[finite]
        rho = float(spearmanr(x, y).statistic) if len(x) >= 3 and np.unique(x).size > 1 and np.unique(y).size > 1 else float("nan")
        rows.append({
            "candidate_id": candidate_id,
            "exposure": str(trial.exposure),
            "outcome": "CEC",
            "estimand": str(trial.estimand),
            "spearman_rho": rho,
            "n": int(len(x)),
            "cec_correction_status": "corrected formula; fixed historical candidate pair membership",
        })
    return pd.DataFrame.from_records(rows)


def _make_corrected_curves(
    cells: pd.DataFrame,
    winners: pd.DataFrame,
    corrected_pairs: dict[str, pd.DataFrame],
    source_root: Path,
    output_figure_dir: Path,
    curve_points: int,
) -> None:
    for estimand, stem, ylabel, x_label in (
        ("matched_pixel", "05c_matched_pixel_natural_spline", "匹配像元健康响应", "log(1 + 实际侵蚀量 Aa)"),
        ("matched_pair", "05d_matched_pair_natural_spline", "高侵蚀端减低侵蚀端的健康差值", "Δlog(1 + Aa)，高侵蚀端 − 低侵蚀端"),
    ):
        old = _read_csv(source_root / "data" / f"{stem}_source.csv.gz")
        old = old.loc[old["outcome"].ne("CEC")].copy()
        parts: list[pd.DataFrame] = []
        selected = winners.loc[winners["estimand"].eq(estimand)]
        for row in selected.itertuples(index=False):
            pairs = corrected_pairs[str(row.candidate_id)]
            if estimand == "matched_pixel":
                ids = pd.Index(pd.unique(pd.concat([pairs["high_erosion_cell_id"], pairs["low_erosion_cell_id"]], ignore_index=True)))
                frame = cells.loc[cells["cell_id"].isin(ids), ["cell_id", row.exposure, "CEC"]].drop_duplicates("cell_id").copy()
                frame["x_value"] = np.log1p(pd.to_numeric(frame[row.exposure], errors="coerce"))
                frame["y_value"] = pd.to_numeric(frame["CEC"], errors="coerce")
            else:
                frame = pd.DataFrame({
                    "x_value": pd.to_numeric(pairs["delta_log1p_Aa"], errors="coerce"),
                    "y_value": pd.to_numeric(pairs["delta_CEC"], errors="coerce"),
                })
            curve = _spline_curve(
                frame[["x_value", "y_value"]].rename(columns={"x_value": "__x", "y_value": "__y"}),
                "__x", "__y",
                label_values={
                    "exposure": str(row.exposure),
                    "outcome": "CEC",
                    "estimand": estimand,
                    "candidate_id": str(row.candidate_id),
                    "selected_model_family": str(row.model_family),
                    "apparent_r_squared": float(row.r_squared),
                    "candidate_selection_status": "corrected CEC ranking within fixed Stage 2 candidate pool",
                    "cec_correction_status": "corrected formula; fixed historical candidate pair membership",
                },
                points=curve_points,
            )
            parts.append(curve)
        curve_data = pd.concat([old, *parts], ignore_index=True, sort=False)
        _curve_panel(
            output_figure_dir, stem, curve_data,
            title=f"环境匹配样本中的非线性响应：{ylabel}",
            x_label=x_label,
            caption=(
                f"每条曲线为描述性自然样条，按各时期、健康指标和{'匹配像元' if estimand == 'matched_pixel' else '匹配配对'}分别拟合。"
                f"CEC 已按校正公式重估，并在固定的 12 个历史 Stage 2 候选配对集合中按校正后的表观 R²选择候选及模型族。"
                f"其他健康指标沿用原图源。{FIGURE_NOTE}"
            ),
        )


def _replace_assets(stems: list[str], generated: Path) -> None:
    for stem in stems:
        for fmt in ("png", "svg", "pdf"):
            src = generated / fmt / f"{stem}.{fmt}"
            dest = FIGURE_ROOT / fmt / f"{stem}.{fmt}"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
        source_csv = generated / "data" / f"{stem}_source.csv"
        _write_gzip_csv(pd.read_csv(source_csv, low_memory=False), FIGURE_ROOT / "data" / f"{stem}_source.csv.gz")
        caption_src = generated / "captions" / f"{stem}.md"
        caption = caption_src.read_text(encoding="utf-8").rstrip()
        if stem in {"03_apparent_r2_pixel_vs_pair", "04_rank_association_vs_r2"}:
            caption += "\n\n" + FIGURE_NOTE
        elif stem in {"07_environment_group_method_comparison", "08_selected_group_map"}:
            caption += "\n\nCEC 组内画像已按校正值重估；环境标签和候选配对池固定，图中其他健康指标沿用原分析结果。 " + FIGURE_NOTE
        elif stem in {"09_apparent_vs_spatial_test", "11_common_sample_period_comparison"}:
            caption += "\n\n" + FIGURE_NOTE
        (FIGURE_ROOT / "captions" / f"{stem}.md").write_text(caption + "\n", encoding="utf-8")


def _apply_figure_style() -> None:
    phase_h_figures.plt.rcParams.update(phase_h_figures.STYLE)
    phase_h_figures.sns.set_theme(style="whitegrid", rc=phase_h_figures.STYLE)


def rerender_existing_outputs() -> None:
    """Re-render from the saved corrected tables without repeating model fits."""

    _apply_figure_style()
    cells = _load_cells()
    winners = pd.read_csv(OUTPUT_ROOT / "tables/cec_corrected_selected_h2h3_models.csv", low_memory=False)
    corrected_pairs = {
        candidate_id: pd.read_parquet(OUTPUT_ROOT / "pairs/candidate_pairs" / f"{candidate_id}.parquet")
        for candidate_id in winners["candidate_id"].astype(str).unique()
    }
    figure_dir = CHART_RUN / "figures"
    model_summary = _figure_3(CHART_RUN, figure_dir)
    _figure_4(CHART_RUN, figure_dir, model_summary)
    _figure_7(CHART_RUN, figure_dir)
    _figure_8(CHART_RUN, figure_dir, cells)
    _figure_9(CHART_RUN, figure_dir)
    _figure_11(CHART_RUN, figure_dir)
    config = yaml.safe_load((OUTPUT_ROOT / "config.yaml").read_text(encoding="utf-8"))
    _make_corrected_curves(
        cells, winners, corrected_pairs, FIGURE_ROOT, figure_dir,
        int(config.get("response_curves", {}).get("curve_points", 250)),
    )
    stems = [
        "03_apparent_r2_pixel_vs_pair",
        "04_rank_association_vs_r2",
        "05c_matched_pixel_natural_spline",
        "05d_matched_pair_natural_spline",
        "07_environment_group_method_comparison",
        "08_selected_group_map",
        "09_apparent_vs_spatial_test",
        "11_common_sample_period_comparison",
    ]
    _replace_assets(stems, figure_dir)
    print("Re-rendered from saved result tables with the project Phase H font style.")


def main() -> None:
    _apply_figure_style()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "tables").mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "pairs/candidate_pairs").mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((OLD_RUN / "config.yaml").read_text(encoding="utf-8"))
    config["outcomes"] = ["CEC"]
    (OUTPUT_ROOT / "config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    cells = _load_cells()
    finite_cec = pd.to_numeric(cells["CEC"], errors="coerce")
    old_trials = pd.read_csv(OLD_RUN / "tables/phase_h_model_trials.csv", low_memory=False)
    old_nonparametric = pd.read_csv(OLD_RUN / "tables/phase_h_nonparametric_response_summary.csv", low_memory=False)
    corrected_trials, candidate_pairs, support_ledger = _fit_corrected_cec_trials(cells, old_trials, config)
    valid_trials = corrected_trials.loc[corrected_trials["status"].eq("ok") & np.isfinite(pd.to_numeric(corrected_trials["r_squared"], errors="coerce"))].copy()
    if valid_trials.empty:
        raise ValueError("Corrected CEC model refit produced no usable apparent model scores")
    corrected_trials.to_csv(OUTPUT_ROOT / "tables/phase_h_model_trials.csv", index=False, encoding="utf-8-sig")
    rank_rows = _rank_rows(cells, corrected_trials, candidate_pairs, old_trials)
    rank_rows.to_csv(OUTPUT_ROOT / "tables/phase_h_nonparametric_response_summary.csv", index=False, encoding="utf-8-sig")
    winners = valid_trials.sort_values(
        ["exposure", "outcome", "estimand", "r_squared", "candidate_id", "model_family"],
        ascending=[True, True, True, False, True, True], kind="stable",
    ).drop_duplicates(["exposure", "outcome", "estimand"], keep="first")
    winners.to_csv(OUTPUT_ROOT / "tables/cec_corrected_selected_h2h3_models.csv", index=False, encoding="utf-8-sig")

    for candidate_id in winners["candidate_id"].astype(str).unique():
        candidate_pairs[candidate_id].to_parquet(OUTPUT_ROOT / "pairs/candidate_pairs" / f"{candidate_id}.parquet", index=False)

    spatial_raw, spatial_summary = run_phase_h_spatial_validation(OUTPUT_ROOT, cells, config)
    period_raw, period_summary = run_phase_h_period_comparison(OUTPUT_ROOT, cells, config)
    spatial_summary.to_csv(OUTPUT_ROOT / "tables/phase_h_spatial_validation_summary.csv", index=False, encoding="utf-8-sig")
    period_raw.to_csv(OUTPUT_ROOT / "tables/phase_h_period_comparison_by_split.csv", index=False, encoding="utf-8-sig")
    period_summary.to_csv(OUTPUT_ROOT / "tables/phase_h_period_comparison_summary.csv", index=False, encoding="utf-8-sig")

    old_profiles = pd.read_csv(OLD_RUN / "tables/phase_h_environment_group_profiles.csv", low_memory=False)
    old_stability = pd.read_csv(OLD_RUN / "tables/phase_h_environment_group_stability.csv", low_memory=False)
    labels = pd.read_parquet(OLD_RUN / "tables/phase_h_environment_group_cell_labels.parquet")
    group_columns = sorted(old_profiles["group_column"].astype(str).unique())
    profile_parts: list[pd.DataFrame] = []
    for row in winners.loc[winners["estimand"].eq("matched_pair")].itertuples(index=False):
        profile_parts.append(build_environment_group_profiles(
            cells,
            labels,
            candidate_pairs[str(row.candidate_id)],
            candidate_id=str(row.candidate_id),
            exposure=str(row.exposure),
            outcome="CEC",
            group_columns=group_columns,
            min_pairs_per_group=int(config.get("environment_groups", {}).get("min_pairs_per_group", 30)),
        ))
    corrected_profiles = pd.concat(profile_parts, ignore_index=True) if profile_parts else pd.DataFrame()
    corrected_profiles["cec_correction_status"] = "corrected formula; fixed environmental group labels and candidate pair membership"
    profiles = pd.concat([old_profiles.loc[old_profiles["outcome"].ne("CEC")], corrected_profiles], ignore_index=True, sort=False)
    group_comparison = build_group_method_comparison(profiles, old_stability)
    profiles.to_csv(OUTPUT_ROOT / "tables/phase_h_environment_group_profiles.csv", index=False, encoding="utf-8-sig")
    group_comparison.to_csv(OUTPUT_ROOT / "tables/phase_h_environment_group_method_comparison.csv", index=False, encoding="utf-8-sig")
    labels.to_parquet(OUTPUT_ROOT / "tables/phase_h_environment_group_cell_labels.parquet", index=False)

    CHART_RUN.mkdir(parents=True, exist_ok=True)
    chart_tables = CHART_RUN / "tables"
    chart_tables.mkdir(parents=True, exist_ok=True)
    combined_trials = pd.concat([old_trials.loc[old_trials["outcome"].ne("CEC")], corrected_trials], ignore_index=True, sort=False)
    combined_trials.to_csv(chart_tables / "phase_h_model_trials.csv", index=False, encoding="utf-8-sig")
    combined_rank = pd.concat([old_nonparametric.loc[old_nonparametric["outcome"].ne("CEC")], rank_rows], ignore_index=True, sort=False)
    combined_rank.to_csv(chart_tables / "phase_h_nonparametric_response_summary.csv", index=False, encoding="utf-8-sig")
    profiles.to_csv(chart_tables / "phase_h_environment_group_profiles.csv", index=False, encoding="utf-8-sig")
    group_comparison.to_csv(chart_tables / "phase_h_environment_group_method_comparison.csv", index=False, encoding="utf-8-sig")
    labels.to_parquet(chart_tables / "phase_h_environment_group_cell_labels.parquet", index=False)
    old_spatial = _read_csv(FIGURE_ROOT / "data/09_apparent_vs_spatial_test_source.csv.gz")
    combined_spatial = pd.concat([old_spatial.loc[old_spatial["outcome"].ne("CEC")], spatial_summary], ignore_index=True, sort=False)
    combined_spatial["cec_correction_status"] = np.where(combined_spatial["outcome"].eq("CEC"), "corrected H2/H3 refit", "")
    combined_spatial.to_csv(chart_tables / "phase_h_spatial_validation_summary.csv", index=False, encoding="utf-8-sig")
    old_period = _read_csv(FIGURE_ROOT / "data/11_common_sample_period_comparison_source.csv.gz")
    combined_period = pd.concat([old_period.loc[old_period["outcome"].ne("CEC")], period_raw], ignore_index=True, sort=False)
    combined_period["cec_correction_status"] = np.where(combined_period["outcome"].eq("CEC"), "corrected H2/H3 refit", "")
    combined_period.to_csv(chart_tables / "phase_h_period_comparison_by_split.csv", index=False, encoding="utf-8-sig")

    figure_dir = CHART_RUN / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    model_summary = _figure_3(CHART_RUN, figure_dir)
    _figure_4(CHART_RUN, figure_dir, model_summary)
    _figure_7(CHART_RUN, figure_dir)
    _figure_8(CHART_RUN, figure_dir, cells)
    _figure_9(CHART_RUN, figure_dir)
    _figure_11(CHART_RUN, figure_dir)

    _make_corrected_curves(cells, winners, candidate_pairs, FIGURE_ROOT, figure_dir, int(config.get("response_curves", {}).get("curve_points", 250)))
    stems = [
        "03_apparent_r2_pixel_vs_pair",
        "04_rank_association_vs_r2",
        "05c_matched_pixel_natural_spline",
        "05d_matched_pair_natural_spline",
        "07_environment_group_method_comparison",
        "08_selected_group_map",
        "09_apparent_vs_spatial_test",
        "11_common_sample_period_comparison",
    ]
    _replace_assets(stems, figure_dir)

    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=WORKSPACE, text=True).strip()
    except Exception:
        git_commit = None
    software: dict[str, str] = {"python": sys.version.split()[0]}
    for package in ("numpy", "pandas", "scipy", "matplotlib", "scikit-learn", "statsmodels", "pyyaml"):
        try:
            software[package] = version(package)
        except PackageNotFoundError:
            software[package] = "not-installed"
    provenance = {
        "run_id": "phase_h_cec_corrected_h2h3_refit_20261101",
        "mode": "EXPLORATORY_ORACLE",
        "outcome_guided_selection": True,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "formula": FORMULA,
        "erosion_definition": EROSION_DEFINITION,
        "corrected_cec_valid_cells": int(finite_cec.notna().sum()),
        "corrected_cec_min": float(finite_cec.min()),
        "corrected_cec_max": float(finite_cec.max()),
        "fixed_candidate_pool_source": str(OLD_RUN / "pairs/candidate_pairs"),
        "fixed_candidate_ids": sorted(old_trials.loc[old_trials["stage"].eq("stage2_full_sample") & old_trials["outcome"].eq("CEC"), "candidate_id"].astype(str).unique()),
        "candidate_reselection": "Corrected CEC apparent R2 re-ranked within the fixed 12 original Stage 2 candidate pair sets; no rematching.",
        "corrected_CEC_pair_support_ledger": support_ledger,
        "pair_support_rule": "Drop a fixed pair only when either endpoint has no corrected CEC value because the aligned Aa inputs required by the correction formula are missing.",
        "spatial_validation_config": config["spatial_validation"],
        "sources": [
            str(CORRECTED_INPUT / "databases/health.parquet"),
            str(CORRECTED_INPUT / "databases/environment_erosion.parquet"),
            str(OLD_RUN / "config.yaml"),
            str(OLD_RUN / "tables/phase_h_model_trials.csv"),
            str(OLD_RUN / "tables/phase_h_environment_group_cell_labels.parquet"),
            str(OLD_RUN / "tables/phase_h_environment_group_stability.csv"),
        ],
        "code": "experiments/oracle_exploration/refresh_phase_h_cec_h2_h3.py",
        "git_commit_at_run": git_commit,
        "code_state": "workspace script was uncommitted at execution; the later rendering-only change applies the existing Phase H font style and does not alter model calculations",
        "software_versions": software,
        "refit_trial_count": int(len(corrected_trials)),
        "successful_trial_count": int(corrected_trials["status"].eq("ok").sum()),
        "selected_models": winners[["exposure", "estimand", "candidate_id", "model_family", "r_squared", "n"]].to_dict(orient="records"),
        "updated_figure_stems": stems,
    }
    (OUTPUT_ROOT / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    (OUTPUT_ROOT / "CEC_H2H3_REFRESH_NOTE.md").write_text(
        "# Phase H corrected CEC H2/H3 refresh\n\n"
        f"Formula: `{FORMULA}`. `Erosion` is the sum of Aa_20yr and Aa_2020 divided by their separate full valid aligned raster maxima.\n\n"
        "The original 12 Stage 2 candidate pair sets and outcome-blind environmental group labels were held fixed. Corrected CEC model-family and candidate rankings were refit; the selected model families were then re-evaluated using the original spatial blocks, seeds, buffers, and common-period samples. Non-CEC rows retain their prior values.\n\n"
        f"Corrected CEC valid values: {int(finite_cec.notna().sum()):,}; range {float(finite_cec.min()):.8f}–{float(finite_cec.max()):.8f}.\n\n"
        f"Refit trials: {int(len(corrected_trials))}; successful: {int(corrected_trials['status'].eq('ok').sum())}.\n",
        encoding="utf-8",
    )
    print("Updated figure stems:", ", ".join(stems), flush=True)
    print("Corrected CEC selected models:")
    print(winners[["exposure", "estimand", "candidate_id", "model_family", "r_squared", "n"]].to_string(index=False))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    if args.render_only:
        rerender_existing_outputs()
    else:
        main()
