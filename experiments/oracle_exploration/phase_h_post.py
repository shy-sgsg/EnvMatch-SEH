"""Nonparametric response summaries and paired distribution data for Phase H."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.stats import binomtest, kendalltau, spearmanr, wilcoxon


def jonckheere_terpstra_trend(
    exposure: np.ndarray, response: np.ndarray, *, groups: int = 5
) -> dict[str, Any]:
    """Test an ordered exposure-bin trend with the tie-adjusted Kendall/JT asymptotic statistic.

    The exposure support is divided into ordered quantile groups. The asymptotic
    JT test with ties is equivalent to Kendall's tau-b test of response against
    ordered group labels; the returned tau-b is the signed standardized trend.
    """

    x = np.asarray(exposure, dtype=float).reshape(-1)
    y = np.asarray(response, dtype=float).reshape(-1)
    if x.shape != y.shape:
        raise ValueError("exposure and response must have the same length")
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    output: dict[str, Any] = {
        "status": "insufficient_sample",
        "n": int(len(x)),
        "n_ordered_groups": 0,
        "group_counts_json": "{}",
        "group_medians_json": "{}",
        "jt_tau_b": float("nan"),
        "p_value_two_sided": float("nan"),
        "direction": "unknown",
        "test_method": "asymptotic Jonckheere-Terpstra trend; Kendall tau-b tie adjustment",
    }
    if len(x) < 9 or np.unique(x).size < 3 or np.unique(y).size < 2:
        return output
    n_groups = min(int(groups), int(np.unique(x).size))
    if n_groups < 3:
        return output
    try:
        group = pd.qcut(pd.Series(x), q=n_groups, labels=False, duplicates="drop").to_numpy()
    except ValueError:
        return output
    group_frame = pd.DataFrame({"group": group, "response": y}).dropna()
    group_counts = group_frame.groupby("group", sort=True).size()
    if group_counts.size < 3 or group_counts.min() < 2:
        return output
    group_medians = group_frame.groupby("group", sort=True)["response"].median()
    statistic = kendalltau(
        group_frame["group"].to_numpy(dtype=float),
        group_frame["response"].to_numpy(dtype=float),
        variant="b",
        method="asymptotic",
    )
    tau = float(statistic.statistic)
    p_value = float(statistic.pvalue)
    direction = "unknown" if not np.isfinite(tau) else "negative" if tau < -0.01 else "positive" if tau > 0.01 else "near_zero"
    output.update(
        status="ok" if np.isfinite(tau) and np.isfinite(p_value) else "no_response_variation",
        n_ordered_groups=int(group_counts.size),
        group_counts_json=json.dumps({str(k): int(v) for k, v in group_counts.items()}, sort_keys=True),
        group_medians_json=json.dumps({str(k): float(v) for k, v in group_medians.items()}, sort_keys=True),
        jt_tau_b=tau,
        p_value_two_sided=p_value,
        direction=direction,
    )
    return output


def _association(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x[finite], dtype=float), np.asarray(y[finite], dtype=float)
    result = {"spearman_rho": float("nan"), "spearman_p_value": float("nan"), "kendall_tau": float("nan"), "kendall_p_value": float("nan")}
    if len(x) >= 3 and np.ptp(x) > 0 and np.ptp(y) > 0:
        rho = spearmanr(x, y)
        tau = kendalltau(x, y, variant="b", method="asymptotic")
        result.update(
            spearman_rho=float(rho.statistic),
            spearman_p_value=float(rho.pvalue),
            kendall_tau=float(tau.statistic),
            kendall_p_value=float(tau.pvalue),
        )
    return result


def _paired_tests(delta: np.ndarray) -> dict[str, float]:
    values = np.asarray(delta, dtype=float)
    values = values[np.isfinite(values)]
    nonzero = values[values != 0]
    output = {
        "n_pairs": int(len(values)),
        "mean_delta_health": float(np.mean(values)) if len(values) else float("nan"),
        "median_delta_health": float(np.median(values)) if len(values) else float("nan"),
        "negative_delta_fraction": float(np.mean(values < 0)) if len(values) else float("nan"),
        "positive_delta_fraction": float(np.mean(values > 0)) if len(values) else float("nan"),
        "zero_delta_fraction": float(np.mean(values == 0)) if len(values) else float("nan"),
        "standardized_paired_effect_dz": float(np.mean(values) / np.std(values, ddof=1)) if len(values) > 1 and np.std(values, ddof=1) > 0 else float("nan"),
        "wilcoxon_statistic": float("nan"),
        "wilcoxon_p_value": float("nan"),
        "sign_test_p_value": float("nan"),
    }
    if len(values) and np.ptp(values) > 0:
        try:
            test = wilcoxon(values, zero_method="pratt", alternative="two-sided", method="auto")
            output["wilcoxon_statistic"] = float(test.statistic)
            output["wilcoxon_p_value"] = float(test.pvalue)
        except ValueError:
            pass
    if len(nonzero):
        output["sign_test_p_value"] = float(binomtest(int(np.sum(nonzero > 0)), n=int(len(nonzero)), p=0.5, alternative="two-sided").pvalue)
    return output


def _bh_adjust(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.notna() & np.isfinite(numeric) & numeric.between(0.0, 1.0)
    result = pd.Series(np.nan, index=values.index, dtype=float)
    p = numeric.loc[valid].to_numpy(dtype=float)
    if not len(p):
        return result
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.minimum(adjusted, 1.0)
    result.loc[valid] = restored
    return result


def build_nonparametric_tables(
    cells: pd.DataFrame,
    candidate_pairs: list[tuple[str, pd.DataFrame]],
    *,
    outcomes: list[str],
    seed: int,
    boxplot_pair_sample_size: int = 5000,
    boxplot_candidate_ids: Mapping[tuple[str, str], str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Summarize paired and pixel response directions for every finalist and outcome."""

    response_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    box_rows: list[pd.DataFrame] = []
    cell_index = cells.drop_duplicates("cell_id").set_index("cell_id", drop=False)
    for candidate_index, (candidate_id, pairs) in enumerate(candidate_pairs):
        if pairs.empty:
            continue
        exposure = str(pairs["exposure"].iloc[0])
        method = str(pairs.get("method", pd.Series(["unknown"])).iloc[0])
        pixel_ids = pd.Index(pd.unique(pd.concat([
            pairs["high_erosion_cell_id"], pairs["low_erosion_cell_id"]
        ], ignore_index=True)))
        pixels = cell_index.reindex(pixel_ids).dropna(subset=["cell_id"]).copy()
        pixel_exposure = pd.to_numeric(pixels[exposure], errors="coerce").to_numpy(dtype=float)
        pixel_x = np.log1p(pixel_exposure)
        for outcome in outcomes:
            if outcome not in pixels or f"high_{outcome}" not in pairs or f"low_{outcome}" not in pairs:
                continue
            pixel_y = pd.to_numeric(pixels[outcome], errors="coerce").to_numpy(dtype=float)
            pixel_assoc = _association(pixel_x, pixel_y)
            pixel_jt = jonckheere_terpstra_trend(pixel_x, pixel_y, groups=5)
            response_rows.append({
                "candidate_id": candidate_id, "method": method, "exposure": exposure,
                "outcome": outcome, "estimand": "matched_pixel", "analysis_population": "phase_h_matched_only",
                "n": int(np.isfinite(pixel_x * pixel_y).sum()), **pixel_assoc,
                "jt_tau_b": pixel_jt["jt_tau_b"], "jt_p_value": pixel_jt["p_value_two_sided"],
                "jt_direction": pixel_jt["direction"], "jt_status": pixel_jt["status"],
                "jt_ordered_groups": pixel_jt["n_ordered_groups"],
                "jt_group_counts_json": pixel_jt["group_counts_json"],
                "jt_group_medians_json": pixel_jt["group_medians_json"],
                "test_method": pixel_jt["test_method"],
            })

            pair_x = pd.to_numeric(pairs.get("delta_log1p_Aa", pairs.get("delta_log1p_erosion")), errors="coerce").to_numpy(dtype=float)
            pair_y = pd.to_numeric(pairs[f"delta_{outcome}"], errors="coerce").to_numpy(dtype=float) if f"delta_{outcome}" in pairs else (
                pd.to_numeric(pairs[f"high_{outcome}"], errors="coerce").to_numpy(dtype=float)
                - pd.to_numeric(pairs[f"low_{outcome}"], errors="coerce").to_numpy(dtype=float)
            )
            pair_assoc = _association(pair_x, pair_y)
            pair_jt = jonckheere_terpstra_trend(pair_x, pair_y, groups=5)
            response_rows.append({
                "candidate_id": candidate_id, "method": method, "exposure": exposure,
                "outcome": outcome, "estimand": "matched_pair", "analysis_population": "phase_h_matched_only",
                "n": int(np.isfinite(pair_x * pair_y).sum()), **pair_assoc,
                "jt_tau_b": pair_jt["jt_tau_b"], "jt_p_value": pair_jt["p_value_two_sided"],
                "jt_direction": pair_jt["direction"], "jt_status": pair_jt["status"],
                "jt_ordered_groups": pair_jt["n_ordered_groups"],
                "jt_group_counts_json": pair_jt["group_counts_json"],
                "jt_group_medians_json": pair_jt["group_medians_json"],
                "test_method": pair_jt["test_method"],
            })
            paired = _paired_tests(pair_y)
            paired_rows.append({
                "candidate_id": candidate_id, "method": method, "exposure": exposure,
                "outcome": outcome, "estimand": "matched_pair_high_minus_low",
                "analysis_population": "phase_h_matched_only", **paired, **pair_assoc,
            })
            selected_box_candidate = (
                boxplot_candidate_ids is None
                or boxplot_candidate_ids.get((exposure, outcome)) == candidate_id
            )
            if not selected_box_candidate:
                continue
            finite_pair = pairs.loc[np.isfinite(pair_y)].copy()
            if len(finite_pair):
                sample_n = min(int(boxplot_pair_sample_size), len(finite_pair))
                sampled = finite_pair.sample(n=sample_n, random_state=int(seed) + candidate_index)
                box_rows.append(pd.concat([
                    pd.DataFrame({
                        "candidate_id": candidate_id, "exposure": exposure, "outcome": outcome,
                        "pair_id": sampled["pair_id"].astype(str), "member_role": "high_erosion",
                        "health_value": pd.to_numeric(sampled[f"high_{outcome}"], errors="coerce"),
                        "soil_type": sampled.get("soil_type", np.nan),
                        "environment_group": sampled.get("environment_group", np.nan),
                    }),
                    pd.DataFrame({
                        "candidate_id": candidate_id, "exposure": exposure, "outcome": outcome,
                        "pair_id": sampled["pair_id"].astype(str), "member_role": "low_erosion",
                        "health_value": pd.to_numeric(sampled[f"low_{outcome}"], errors="coerce"),
                        "soil_type": sampled.get("soil_type", np.nan),
                        "environment_group": sampled.get("environment_group", np.nan),
                    }),
                    pd.DataFrame({
                        "candidate_id": candidate_id, "exposure": exposure, "outcome": outcome,
                        "pair_id": sampled["pair_id"].astype(str), "member_role": "paired_difference_high_minus_low",
                        "health_value": pd.to_numeric(sampled[f"high_{outcome}"], errors="coerce") - pd.to_numeric(sampled[f"low_{outcome}"], errors="coerce"),
                        "soil_type": sampled.get("soil_type", np.nan),
                        "environment_group": sampled.get("environment_group", np.nan),
                    }),
                ], ignore_index=True))

    response = pd.DataFrame.from_records(response_rows)
    paired = pd.DataFrame.from_records(paired_rows)
    if not response.empty:
        response["jt_q_value_global"] = _bh_adjust(response["jt_p_value"])
        response["spearman_q_value_global"] = _bh_adjust(response["spearman_p_value"])
        response["kendall_q_value_global"] = _bh_adjust(response["kendall_p_value"])
    if not paired.empty:
        paired["wilcoxon_q_value_global"] = _bh_adjust(paired["wilcoxon_p_value"])
        paired["sign_test_q_value_global"] = _bh_adjust(paired["sign_test_p_value"])
    boxplot = pd.concat(box_rows, ignore_index=True) if box_rows else pd.DataFrame()
    return {"response_summary": response, "paired_difference_summary": paired, "boxplot_source": boxplot}


def run_phase_h_nonparametric_postprocessing(
    run_dir: Path, cells: pd.DataFrame, config: dict[str, Any]
) -> dict[str, pd.DataFrame]:
    """Create matched-pixel/pair Spearman, Kendall, JT, paired-test, and boxplot source tables."""

    root = Path(run_dir)
    table_dir = root / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    names = {
        "response_summary": table_dir / "phase_h_nonparametric_response_summary.csv",
        "paired_difference_summary": table_dir / "phase_h_paired_difference_summary.csv",
        "boxplot_source": table_dir / "phase_h_boxplot_source.csv",
    }
    checkpoint = root / "checkpoints" / "h4_nonparametric_complete.json"
    if checkpoint.is_file() and all(path.is_file() for path in names.values()):
        return {name: pd.read_csv(path) for name, path in names.items()}
    pair_files = sorted((root / "pairs" / "candidate_pairs").glob("stage2_*.parquet"))
    if not pair_files:
        raise FileNotFoundError("No Phase H Stage 2 candidate pair tables are available for H4")
    boxplot_candidate_ids: dict[tuple[str, str], str] | None = None
    trial_path = table_dir / "phase_h_model_trials.csv"
    if trial_path.is_file():
        trials = pd.read_csv(trial_path)
        selected = trials.loc[
            trials.get("stage", pd.Series(index=trials.index, dtype=object)).eq("stage2_full_sample")
            & trials["estimand"].eq("matched_pair")
            & trials.get("subset_name", pd.Series(index=trials.index, dtype=object)).eq("all")
            & trials.get("fit_scope", pd.Series(index=trials.index, dtype=object)).eq("exposure_only")
            & trials.get("status", pd.Series(index=trials.index, dtype=object)).eq("ok")
        ].copy()
        if not selected.empty:
            selected["r_squared"] = pd.to_numeric(selected["r_squared"], errors="coerce")
            selected = selected.sort_values("r_squared", ascending=False, kind="stable").drop_duplicates(["exposure", "outcome"])
            boxplot_candidate_ids = {
                (str(row.exposure), str(row.outcome)): str(row.candidate_id)
                for row in selected.itertuples(index=False)
            }
    response_frames: list[pd.DataFrame] = []
    paired_frames: list[pd.DataFrame] = []
    boxplot_frames: list[pd.DataFrame] = []
    for candidate_index, path in enumerate(pair_files):
        pairs = pd.read_parquet(path)
        part = build_nonparametric_tables(
            cells,
            [(path.stem, pairs)],
            outcomes=list(config["outcomes"]),
            seed=int(config["seed"]),
            boxplot_pair_sample_size=int(config.get("figures", {}).get("boxplot_pair_sample_size", 5000)),
            boxplot_candidate_ids=boxplot_candidate_ids,
        )
        response_frames.append(part["response_summary"])
        paired_frames.append(part["paired_difference_summary"])
        boxplot_frames.append(part["boxplot_source"])
        del pairs, part
        print(
            f"Phase H nonparametric summaries {candidate_index + 1}/{len(pair_files)}: {path.stem}",
            flush=True,
        )
    response = pd.concat(response_frames, ignore_index=True) if response_frames else pd.DataFrame()
    paired = pd.concat(paired_frames, ignore_index=True) if paired_frames else pd.DataFrame()
    boxplot = pd.concat(boxplot_frames, ignore_index=True) if boxplot_frames else pd.DataFrame()
    if not response.empty:
        response["jt_q_value_global"] = _bh_adjust(response["jt_p_value"])
        response["spearman_q_value_global"] = _bh_adjust(response["spearman_p_value"])
        response["kendall_q_value_global"] = _bh_adjust(response["kendall_p_value"])
    if not paired.empty:
        paired["wilcoxon_q_value_global"] = _bh_adjust(paired["wilcoxon_p_value"])
        paired["sign_test_q_value_global"] = _bh_adjust(paired["sign_test_p_value"])
    tables = {
        "response_summary": response,
        "paired_difference_summary": paired,
        "boxplot_source": boxplot,
    }
    for name, path in names.items():
        tables[name].to_csv(path, index=False, encoding="utf-8-sig")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text(json.dumps({
        "task": "H4_nonparametric_response_and_paired_tests",
        "status": "complete",
        "candidate_count": int(len(pair_files)),
        "response_summary_rows": int(len(tables["response_summary"])),
        "paired_difference_rows": int(len(tables["paired_difference_summary"])),
        "boxplot_source_rows": int(len(tables["boxplot_source"])),
        "outcome_guided_selection": True,
        "inference_note": "Asymptotic p-values are exploratory and do not account for spatial dependence; spatial validation is reported separately.",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return tables
