from __future__ import annotations

import numpy as np


def test_summary_separates_zero_positive_near_zero_and_missing():
    from experiments.low_erosion_regime.distribution_diagnostics import summarize_values

    result = summarize_values(np.array([0.0, 0.0, 1e-5, 0.02, 2.0, np.nan]))

    assert result["n_total"] == 6
    assert result["n_finite"] == 5
    assert result["n_zero"] == 2
    assert result["n_positive"] == 3
    assert result["zero_fraction_finite"] == 0.4
    assert result["positive_log10_min"] == -5.0


def test_summary_reports_positive_quantiles_and_precision_repetition():
    from experiments.low_erosion_regime.distribution_diagnostics import summarize_values

    result = summarize_values(np.array([0.0, 0.00123451, 0.00123456, 0.1, 1.0, 10.0]))

    assert result["positive_q01"] == np.quantile([0.00123451, 0.00123456, 0.1, 1.0, 10.0], .01)
    assert result["positive_q10"] == np.quantile([0.00123451, 0.00123456, 0.1, 1.0, 10.0], .10)
    assert result["positive_q90"] == np.quantile([0.00123451, 0.00123456, 0.1, 1.0, 10.0], .90)
    assert result["fraction_repeated_3dp"] > result["fraction_repeated_9dp"]
    assert result["unique_fraction_6dp"] < 1.0


def test_gmm_seed_sensitivity_records_each_fit_and_selected_model():
    from experiments.low_erosion_regime.distribution_addendum import (
        fit_gmm_seed_sensitivity, summarize_gmm_seed_sensitivity,
    )

    rng = np.random.default_rng(101)
    values = np.r_[10 ** rng.normal(-2.4, .13, 350), 10 ** rng.normal(-.3, .16, 450)]
    rows = np.repeat(np.arange(8) * 170, 100)
    cols = np.tile(np.arange(100), 8)
    trials = fit_gmm_seed_sensitivity(
        values, exposure="Aa_20yr", seeds=(11, 12), sample_size=500, rows=rows, cols=cols,
    )
    summary = summarize_gmm_seed_sensitivity(trials)

    assert set(trials["seed"]) == {11, 12}
    assert set(trials["n_components"]) == {1, 2, 3, 4}
    assert trials.groupby("seed")["n_components"].nunique().eq(4).all()
    assert summary.loc[0, "seed_count"] == 2
    assert summary.loc[0, "selected_k_mode"] in {1, 2, 3, 4}


def test_gmm_posterior_cutoff_sensitivity_counts_uncertain_pixels():
    import pandas as pd

    from experiments.low_erosion_regime.distribution_addendum import summarize_posterior_cutoffs

    assignments = pd.DataFrame({
        "selected_n_components_bic": [2] * 6,
        "posterior_component": [1, 1, 1, 2, 2, 2],
        "posterior_max": [.99, .85, .75, .95, .82, .70],
    })
    result = summarize_posterior_cutoffs(assignments, n_valid=8, cutoffs=(.8, .9, .95))
    at_80 = result.loc[result["posterior_cutoff"].eq(.8)].iloc[0]
    at_90 = result.loc[result["posterior_cutoff"].eq(.9)].iloc[0]

    assert at_80["n_uncertain_positive"] == 2
    assert at_80["n_low_component_assigned"] == 2
    assert at_90["n_uncertain_positive"] == 4
    assert at_90["low_component_fraction_all_valid"] == 1 / 8


def test_unified_threshold_candidate_table_has_requested_fields_and_denominators():
    import pandas as pd

    from experiments.low_erosion_regime.distribution_addendum import build_threshold_candidate_table

    values = np.array([0.0, 0.0, .01, .1, 1.0, 10.0])
    kde = pd.DataFrame([{
        "exposure": "Aa_20yr", "bandwidth_base": "scott", "bandwidth_multiplier": 1.0,
        "status": "no_reliable_valley", "threshold_original": np.nan,
        "low_peak_log10": np.nan, "next_peak_log10": np.nan, "valley_density_ratio": np.nan,
        "fit_sample_n": 6,
    }])
    gmm = pd.DataFrame([{
        "exposure": "Aa_20yr", "n_components": 2, "bic": 3.0, "aic": 2.0,
        "converged": True, "component_means_log10": "-1|0", "component_sd_log10": ".1|.1",
        "component_weights": ".2|.8", "adjacent_overlap_index": ".8",
        "adjacent_boundaries_log10": "-0.5", "fit_n": 6,
    }])
    cdf = pd.DataFrame([{
        "exposure": "Aa_20yr", "status": "valid_cdf_breakpoint", "threshold_original": .1,
        "threshold_log10": -1.0, "delta_aic_one_break_vs_linear": 8.0, "fit_n": 6,
    }])
    otsu = pd.DataFrame([{
        "exposure": "Aa_20yr", "histogram_bins": 32, "threshold_original": .2,
        "threshold_log10": np.log10(.2), "status": "comparison_threshold",
    }])
    empty_boot = pd.DataFrame(columns=["exposure", "status", "threshold_original"])

    result = build_threshold_candidate_table(
        exposure="Aa_20yr", values=values, kde=kde, gmm=gmm, cdf=cdf, otsu=otsu,
        kde_bootstrap=empty_boot, cdf_bootstrap=pd.DataFrame({
            "exposure": ["Aa_20yr", "Aa_20yr"], "status": ["valid_cdf_breakpoint"] * 2,
            "threshold_original": [.09, .11],
        }),
    )
    required = {
        "method_parameters", "threshold_value", "threshold_unit", "zero_fraction",
        "near_zero_fraction", "total_low_value_fraction", "retained_cells", "retained_fraction",
        "component_count", "component_weights", "threshold_stability", "bootstrap_ci_low",
        "bootstrap_ci_high", "separation_quality", "model_fit_quality", "status", "failure_reason",
    }
    assert required.issubset(result.columns)
    cdf_row = result.loc[result["method"].eq("empirical_CDF_breakpoint")].iloc[0]
    gmm_row = result.loc[result["method"].str.startswith("GMM")].iloc[0]
    assert cdf_row["total_low_value_fraction"] == 4 / 6
    assert cdf_row["near_zero_fraction"] == .5
    assert cdf_row["retained_cells"] == 2
    assert gmm_row["status"] == "rejected_component_overlap"


def test_distribution_addendum_report_uses_chinese_table_labels():
    import pandas as pd

    from experiments.low_erosion_regime.distribution_addendum import build_addendum_report_text

    config = {
        "run_id": "test_run", "gmm_sample_size": 20,
        "gmm_stability_seeds": [1, 2],
    }
    summary = pd.DataFrame([{
        "exposure": "Aa_20yr", "n_finite": 10, "n_zero": 2,
        **{f"positive_q{q:02d}": float(q) for q in (1, 5, 10, 25, 50, 75, 90, 95, 99)},
    }])
    seed_summary = pd.DataFrame([{
        "exposure": "Aa_20yr", "seed_count": 2, "selected_k_counts": '{"2": 2}',
        "selected_k_mode": 2, "selected_k_agreement_fraction": 1.0,
        "selected_convergence_fraction": 1.0, "first_boundary_seed_count": 2,
        "first_boundary_median_log10": -1.0, "first_boundary_median_original": .1,
        "first_boundary_cv": .05, "first_boundary_ci_low": .09, "first_boundary_ci_high": .11,
        "candidate_source": "distribution_derived", "outcome_guided_selection": False,
    }])
    posterior = pd.DataFrame([{
        "exposure": "Aa_20yr", "selected_k_bic": 2, "low_component_identifiable": True,
        "posterior_cutoff": .8, "n_positive": 10, "n_valid_including_zero": 10,
        "n_low_component_assigned": 2, "n_other_component_assigned": 6,
        "n_uncertain_positive": 2, "low_component_fraction_positive": .2,
        "low_component_fraction_all_valid": .2, "uncertain_fraction_positive": .2,
        "candidate_source": "distribution_derived", "outcome_guided_selection": False,
    }])

    report = build_addendum_report_text(config, summary, seed_summary, posterior)

    assert "| 侵蚀时期 | 有效侵蚀像元 | 精确零值像元 |" in report
    assert "| 侵蚀时期 | 随机种子数 | BIC 选中 K 频次 |" in report
    assert "| 侵蚀时期 | BIC 选中成分数 | 后验概率阈值 |" in report
    assert "selected_k_counts" not in report
    assert "n_uncertain_positive" not in report


def test_distribution_addendum_figures_write_three_chinese_linked_figure_sets(tmp_path):
    import json
    import pandas as pd

    from experiments.low_erosion_regime.distribution_addendum_figures import (
        generate_distribution_addendum_figures,
    )

    (tmp_path / "diagnostics").mkdir()
    (tmp_path / "thresholds").mkdir()
    pd.DataFrame([
        {"exposure": "Aa_20yr", "seed": seed, "selected_k_bic": 3,
         "selected_by_bic": True, "adjacent_boundaries_log10": "0.4|0.9", "converged": True}
        for seed in range(1, 6)
    ] + [
        {"exposure": "Aa_2020", "seed": seed, "selected_k_bic": 4,
         "selected_by_bic": True,
         "adjacent_boundaries_log10": "nan|-1.0|-0.4|0.1" if seed == 1 else "-1.0|-0.4|0.1",
         "converged": True}
        for seed in range(1, 6)
    ]).to_csv(tmp_path / "diagnostics/gmm_seed_stability_by_seed.csv", index=False)
    pd.DataFrame([
        {"exposure": exposure, "posterior_cutoff": cutoff,
         "n_positive": 100, "n_low_component_assigned": 10,
         "n_other_component_assigned": 60, "n_uncertain_positive": 30,
         "low_component_fraction_positive": .1, "uncertain_fraction_positive": .3}
        for exposure in ("Aa_20yr", "Aa_2020") for cutoff in (.8, .9, .95)
    ]).to_csv(tmp_path / "diagnostics/gmm_posterior_cutoff_sensitivity.csv", index=False)
    pd.DataFrame([
        {"exposure": exposure, "method": method, "threshold_value": value,
         "status": status, "bootstrap_ci_low": low, "bootstrap_ci_high": high,
         "threshold_id": f"{exposure}_{method}_{value}"}
        for exposure in ("Aa_20yr", "Aa_2020")
        for method, value, status, low, high in (
            ("KDE_scott_1x", float("nan"), "no_reliable_valley", float("nan"), float("nan")),
            ("GMM_BIC_K2_boundary_1", .2, "rejected_component_overlap", float("nan"), float("nan")),
            ("empirical_CDF_breakpoint", .5, "valid_cdf_breakpoint", .4, .6),
            ("Otsu", .3, "comparison_only", float("nan"), float("nan")),
        )
    ]).to_csv(tmp_path / "thresholds/low_erosion_threshold_candidates.csv", index=False)

    manifest_path = generate_distribution_addendum_figures(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert len(manifest["figures"]) == 3
    assert all(record["title"].isascii() is False for record in manifest["figures"])
    boundary_source = pd.read_csv(tmp_path / "figures" / manifest["figures"][0]["source"])
    assert boundary_source.groupby("exposure").size().to_dict() == {"Aa_2020": 5, "Aa_20yr": 5}
    for record in manifest["figures"]:
        for fmt in ("png", "svg", "pdf"):
            assert (tmp_path / "figures" / fmt / f"{record['id']}.{fmt}").is_file()
        assert (tmp_path / "figures" / record["source"]).is_file()
        assert (tmp_path / "figures" / record["caption"]).is_file()


def test_kde_valley_maps_back_to_original_exposure_and_allows_no_valley():
    from experiments.low_erosion_regime.distribution_diagnostics import find_kde_valleys

    rng = np.random.default_rng(17)
    low = 10 ** rng.normal(-3.0, 0.08, 500)
    high = 10 ** rng.normal(-0.1, 0.10, 700)
    candidates = find_kde_valleys(np.r_[low, high], exposure="Aa_20yr")
    valid = candidates.loc[candidates["status"].eq("valid_valley")]

    assert not valid.empty
    assert valid["threshold_original"].between(1e-3, 1.0).all()
    assert valid["candidate_source"].eq("distribution_derived").all()

    unimodal = find_kde_valleys(10 ** rng.normal(-1.0, 0.35, 500), exposure="Aa_2020")
    assert unimodal["status"].eq("no_reliable_valley").all()


def test_gaussian_mixture_components_are_ordered_and_uncertainty_is_retained():
    from experiments.low_erosion_regime.distribution_diagnostics import fit_gaussian_mixtures

    rng = np.random.default_rng(18)
    values = np.r_[10 ** rng.normal(-3, 0.15, 300), 10 ** rng.normal(-0.3, 0.2, 500)]
    models, assignments = fit_gaussian_mixtures(values, exposure="Aa_20yr", seed=41)

    two = models.loc[models["n_components"].eq(2)].iloc[0]
    means = [float(value) for value in two["component_means_log10"].split("|")]
    assert means == sorted(means)
    assert set(assignments["classification_status"]).issubset({"assigned", "uncertain"})
    assert assignments["posterior_max"].between(0, 1).all()


def test_candidate_classifier_keeps_zero_and_uncertain_separate():
    from experiments.low_erosion_regime.distribution_diagnostics import classify_regimes

    labels = classify_regimes(np.array([0.0, 0.01, 0.2, np.nan]), threshold=0.05)

    assert labels.tolist() == ["zero", "low_positive", "main", "nodata"]


def test_otsu_threshold_is_on_original_scale_and_records_histogram_bins():
    from experiments.low_erosion_regime.distribution_diagnostics import otsu_thresholds

    values = np.r_[np.zeros(100), 10 ** np.linspace(-4, 0, 300)]
    result = otsu_thresholds(values, exposure="Aa_20yr", bins=(32, 64))

    assert set(result["histogram_bins"]) == {32, 64}
    assert result["threshold_original"].between(0, 1).all()
    assert result["candidate_source"].eq("comparison_only").all()


def test_spatial_bootstrap_is_seeded_and_reports_replicate_thresholds():
    from experiments.low_erosion_regime.distribution_diagnostics import bootstrap_kde_valleys

    rng = np.random.default_rng(19)
    values = np.r_[10 ** rng.normal(-3, 0.08, 1000), 10 ** rng.normal(-0.2, 0.1, 1000)]
    rows = (np.arange(len(values)) // 100) * 300
    cols = np.arange(len(values)) % 100
    first = bootstrap_kde_valleys(values, rows, cols, exposure="Aa_2020", replicates=12, seed=77)
    second = bootstrap_kde_valleys(values, rows, cols, exposure="Aa_2020", replicates=12, seed=77)

    assert len(first) == 12
    assert first.equals(second)
    assert first["replicate"].nunique() == 12


def test_spatial_grid_and_connected_components_respect_valid_mask():
    import pandas as pd

    from experiments.low_erosion_regime.spatial_analysis import connected_components, regime_grid

    cells = pd.DataFrame({
        "cell_id": [1, 2, 3, 4], "row": [10, 10, 11, 11], "col": [20, 21, 20, 21],
        "x": [100, 130, 100, 130], "y": [200, 200, 170, 170], "valid_cell": [True, True, True, False],
        "Aa_20yr": [0.0, 0.01, 0.02, 0.01],
    })
    grid, extent, valid = regime_grid(cells, exposure="Aa_20yr", threshold=.05)
    components, labels = connected_components(cells.iloc[:3].assign(valid_cell=True), exposure="Aa_20yr", threshold=.05)

    assert grid.shape == (2, 2)
    assert grid[0, 0] == 0
    assert grid[0, 1] == 1
    assert grid[1, 1] == -1
    assert valid.sum() == 3
    assert extent[0] < extent[1] and extent[2] < extent[3]
    assert len(components) == 1
    assert int(components.iloc[0]["pixel_count"]) == 2


def test_existing_pair_filter_requires_both_endpoints_above_threshold():
    import pandas as pd

    from experiments.low_erosion_regime.matching_comparison import filter_existing_pairs

    pairs = pd.DataFrame({
        "pair_id": ["a", "b", "c"], "exposure": ["Aa_20yr"] * 3,
        "method": ["M3"] * 3, "spatial_role": ["full_sample_descriptive"] * 3,
        "high_erosion_cell_id": [1, 3, 5], "low_erosion_cell_id": [2, 4, 6],
    })
    cells = pd.DataFrame({"cell_id": [1, 2, 3, 4, 5, 6], "Aa_20yr": [2, .1, 3, 2, 0, 4]})

    result = filter_existing_pairs(pairs, cells, exposure="Aa_20yr", threshold=1.0)

    assert result["pair_id"].tolist() == ["b"]


def test_empirical_cdf_breakpoint_uses_original_scale_and_records_fit_size():
    from experiments.low_erosion_regime.distribution_diagnostics import empirical_cdf_breakpoint

    rng = np.random.default_rng(22)
    values = np.r_[10 ** rng.normal(-3.0, .12, 500), 10 ** rng.normal(-.2, .15, 700)]
    result = empirical_cdf_breakpoint(values, exposure="Aa_20yr")

    assert result.loc[0, "fit_n"] <= 5000
    assert result.loc[0, "n_positive"] == len(values)
    if result.loc[0, "status"] == "valid_cdf_breakpoint":
        assert 0 < result.loc[0, "threshold_original"] < 1


def test_pixel_association_reports_zero_only_as_non_estimable():
    import pandas as pd

    from experiments.low_erosion_regime.response_models import pixel_association

    cells = pd.DataFrame({
        "valid_cell": [True] * 8, "Aa_20yr": [0.0] * 8, "AN": np.arange(8.0),
        "x": np.arange(8.0) * 30, "y": np.zeros(8),
    })
    result = pixel_association(cells, exposure="Aa_20yr", outcome="AN", scheme="S1_zero_only")

    assert result["n"] == 8
    assert result["status"] == "constant_or_insufficient_exposure"
    assert np.isnan(result["spearman_rho"])


def test_low_erosion_runner_writes_separate_oracle_artifacts(tmp_path):
    import json
    import yaml
    import pandas as pd

    from experiments.low_erosion_regime.run import run_low_erosion_regime

    n = 120
    rng = np.random.default_rng(23)
    ids = np.arange(1000, 1000 + n)
    row = np.repeat(np.arange(6) * 300, 20)
    col = np.tile(np.arange(20), 6)
    exposure = np.r_[10 ** rng.normal(-3, .08, n // 2), 10 ** rng.normal(-.2, .10, n // 2)]
    env = pd.DataFrame({
        "cell_id": ids, "row": row, "col": col,
        "x": 670000 + col * 30.0, "y": 3880000 - row * 30.0,
        "Elevation": 100 + rng.normal(0, .2, n), "Cur": rng.normal(0, 1, n),
        "TWI": 8 + rng.normal(0, .2, n), "LST": 17 + rng.normal(0, .2, n),
        "SoilType": 27, "NDWI": rng.normal(0, .1, n),
        "Aa_20yr": exposure, "Aa_2020": exposure * .7, "valid_cell": True,
    })
    health = pd.DataFrame({"cell_id": ids, "valid_cell": True})
    outcomes = ["SHI_candidate", "AN", "OM", "CEC", "pH", "SW", "BD", "MBC", "RESP"]
    for i, name in enumerate(outcomes):
        health[name] = rng.normal(i, 1, n) + np.log1p(exposure) * .1
    aligned = tmp_path / "outputs" / "aligned" / "databases"
    aligned.mkdir(parents=True)
    env.to_parquet(aligned / "environment_erosion.parquet", index=False)
    health.to_parquet(aligned / "health.parquet", index=False)
    pairs = []
    for method in ("M3", "M4"):
        for period in ("Aa_20yr", "Aa_2020"):
            for i in range(15):
                high, low = int(ids[80 + i]), int(ids[i])
                record = {
                    "pair_id": f"{method}_{period}_{i}", "method": method, "exposure": period,
                    "spatial_role": "full_sample_descriptive", "high_erosion_cell_id": high,
                    "low_erosion_cell_id": low, "environment_group": "soil_27_cluster_0",
                    "erosion_high": float(env.loc[high - 1000, period]),
                    "erosion_low": float(env.loc[low - 1000, period]),
                    "erosion_difference": float(env.loc[high - 1000, period] - env.loc[low - 1000, period]),
                    "delta_log1p_erosion": float(np.log1p(env.loc[high - 1000, period]) - np.log1p(env.loc[low - 1000, period])),
                    "environmental_distance": .1, "geographic_distance": 300.0,
                    "x_high": float(env.loc[high - 1000, "x"]), "x_low": float(env.loc[low - 1000, "x"]),
                    "y_high": float(env.loc[high - 1000, "y"]), "y_low": float(env.loc[low - 1000, "y"]),
                    "spatial_block_id": f"{int(env.loc[high - 1000, 'x'] // 5000)}_{int(env.loc[high - 1000, 'y'] // 5000)}",
                }
                for name in outcomes:
                    record[f"high_{name}"] = float(health.loc[high - 1000, name])
                    record[f"low_{name}"] = float(health.loc[low - 1000, name])
                    record[f"delta_{name}"] = record[f"high_{name}"] - record[f"low_{name}"]
                pairs.append(record)
    pair_root = tmp_path / "outputs" / "exploratory_oracle" / "phase_f" / "pairs"
    pair_root.mkdir(parents=True)
    pd.DataFrame(pairs).to_parquet(pair_root / "full_sample_pairs.parquet", index=False)
    group_root = tmp_path / "outputs" / "exploratory_oracle" / "phase_g" / "tables"
    group_root.mkdir(parents=True)
    pd.DataFrame({"pair_id": [p["pair_id"] for p in pairs],
                  "high_erosion_cell_id": [p["high_erosion_cell_id"] for p in pairs]}).to_parquet(
                      group_root / "environment_group_labels.parquet", index=False)
    config = {
        "phase": "H-low-erosion-regime", "mode": "EXPLORATORY_ORACLE", "outcome_guided_selection": True,
        "run_id": "small_test", "seed": 23, "input_run_id": "aligned", "phase_f_run_id": "phase_f",
        "phase_g_run_id": "phase_g", "output_root": "outputs/exploratory_oracle",
        "exposures": ["Aa_20yr", "Aa_2020"], "outcomes": outcomes, "primary_soil_codes": [26, 27, 42],
        "distribution": {"positive_sample_size_kde": 80, "positive_sample_size_gmm": 80,
                         "kde_bandwidth_multipliers": [0.5, 1.0, 1.5, 2.0],
                         "otsu_histogram_bins": [16, 32], "bootstrap_spatial_replicates": 2,
                         "bootstrap_sample_size": 40},
        "matching": {"environment_caliper_standardized": 0.35, "geographic_radius_m": 1000,
                     "erosion_rank_gap": 0.5, "rematch_groups": ["fixed_original"]},
        "models": {"fixed_sample_size": 50, "spatial_validation_repeats": 2,
                   "spatial_validation_seeds": [23, 24], "spatial_validation_block_size_m": 5000,
                   "spatial_validation_buffer_m": 300},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    run_dir = run_low_erosion_regime(tmp_path, config_path)

    assert (run_dir / "diagnostics" / "exposure_distribution_summary.csv").is_file()
    assert (run_dir / "associations" / "response_associations.csv").is_file()
    assert (run_dir / "figures" / "figure_manifest.json").is_file()
    assert (run_dir / "figures" / "png" / "S01_low_erosion_regime_maps.png").is_file()
    assert json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))["mode"] == "EXPLORATORY_ORACLE"


def test_module_cli_help_loads_project_src_packages():
    import subprocess
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-m", "experiments.low_erosion_regime.run", "--help"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout


def test_matched_pixel_population_uses_unique_pair_endpoints():
    import pandas as pd

    from experiments.low_erosion_regime.response_models import matched_pixel_population

    cells = pd.DataFrame({
        "cell_id": [1, 2, 3, 4], "valid_cell": [True] * 4,
        "Aa_20yr": [2.0, 1.0, 4.0, 3.0], "AN": [10.0, 11.0, 12.0, 13.0],
    })
    pairs = pd.DataFrame({
        "pair_id": ["p1", "p2", "p3"], "exposure": ["Aa_20yr"] * 3, "method": ["M3", "M3", "M4"],
        "high_erosion_cell_id": [1, 1, 4], "low_erosion_cell_id": [2, 3, 3],
    })

    result = matched_pixel_population(pairs, cells, exposure="Aa_20yr", method="M3")

    assert result["cell_id"].tolist() == [1, 2, 3]
    assert result["cell_id"].is_unique


def test_model_trials_cover_eight_distinct_response_families():
    import pandas as pd

    from experiments.low_erosion_regime.response_models import model_trials

    rng = np.random.default_rng(24)
    x = np.linspace(0.05, 20, 180)
    cells = pd.DataFrame({"valid_cell": True, "Aa_20yr": x, "AN": np.sin(np.log1p(x)) + rng.normal(0, .05, len(x))})

    result = model_trials(cells, exposure="Aa_20yr", outcome="AN", scheme="S0_all", sample_size=180)

    assert set(result["model_family"]) == {
        "linear", "quadratic", "natural_spline", "additive_gam", "huber",
        "random_forest", "extra_trees", "hist_gradient_boosting",
    }
    assert result["status"].eq("ok").all()


def test_matched_pixel_association_uses_unique_endpoints_and_keeps_scheme_label():
    import pandas as pd

    from experiments.low_erosion_regime.response_models import matched_pixel_association

    cells = pd.DataFrame({
        "cell_id": [1, 2, 3, 4], "valid_cell": [True] * 4,
        "Aa_20yr": [0.2, 1.0, 2.0, 4.0], "AN": [3.0, 1.0, 5.0, 2.0],
        "x": [100.0, 200.0, 300.0, 400.0], "y": [100.0, 200.0, 300.0, 400.0],
    })
    pairs = pd.DataFrame({
        "pair_id": ["p1", "p2", "p3"], "exposure": ["Aa_20yr"] * 3,
        "method": ["M3"] * 3, "high_erosion_cell_id": [4, 4, 3],
        "low_erosion_cell_id": [1, 2, 2],
    })

    result = matched_pixel_association(
        pairs, cells, exposure="Aa_20yr", outcome="AN", method="M3",
        scheme="T01_fixed_original", threshold=1.0,
    )

    assert result["analysis_unit"] == "matched_pixel"
    assert result["n"] == 4
    assert result["n_unique_matched_endpoints"] == 4
    assert result["scheme"] == "T01_fixed_original"
    assert result["threshold_original"] == 1.0


def test_matched_pixel_spatial_validation_reports_population_metadata():
    import pandas as pd

    from experiments.low_erosion_regime.response_models import matched_pixel_spatial_validation

    cells = pd.DataFrame({
        "cell_id": [1, 2, 3], "valid_cell": [True] * 3,
        "Aa_20yr": [0.2, 1.0, 2.0], "AN": [3.0, 1.0, 5.0],
        "x": [100.0, 200.0, 300.0], "y": [100.0, 200.0, 300.0],
    })
    pairs = pd.DataFrame({
        "pair_id": ["p1", "p2"], "exposure": ["Aa_20yr"] * 2,
        "method": ["M4"] * 2, "high_erosion_cell_id": [3, 3], "low_erosion_cell_id": [1, 2],
    })

    result = matched_pixel_spatial_validation(
        pairs, cells, exposure="Aa_20yr", outcome="AN", method="M4",
        scheme="S0_all", seeds=(7,),
    )

    assert len(result) == 1
    assert result.loc[0, "analysis_unit"] == "matched_pixel"
    assert result.loc[0, "method"] == "M4"
    assert result.loc[0, "scheme"] == "S0_all"
    assert result.loc[0, "n_unique_matched_endpoints"] == 3


def test_matched_unit_report_localizes_analysis_unit_and_model_labels():
    import pandas as pd

    from experiments.low_erosion_regime.matched_units import _localized_comparison_table

    frame = pd.DataFrame([{
        "analysis_unit": "matched_pixel", "exposure": "Aa_20yr", "method": "M4",
        "outcome": "SHI_candidate", "n": 7306, "n_unique_matched_endpoints": 7306,
        "best_apparent_model_family": "random_forest", "best_apparent_r_squared": 0.228,
        "spearman_rho": -0.340, "spatial_test_r2_median": 0.099, "n_spatial_splits": 10,
    }])

    rendered = _localized_comparison_table(frame)

    assert "唯一匹配像元" in rendered
    assert "随机森林" in rendered
    assert "土壤健康指数（SHI）" in rendered
    assert "matched_pixel" not in rendered
    assert "random_forest" not in rendered


def test_group_response_comparison_deduplicates_pixels_and_keeps_within_group_pairs():
    import pandas as pd

    from experiments.low_erosion_regime.environment_heterogeneity import summarize_group_responses

    cells = pd.DataFrame({
        "cell_id": [1, 2, 3, 4, 5], "Aa_20yr": [0.2, 1.0, 2.0, 4.0, 6.0],
        "AN": [5.0, 4.0, 3.0, 2.0, 1.0],
    })
    labels = pd.DataFrame({
        "cell_id": [1, 2, 3, 4, 5],
        "soil_type_group": ["soil_26", "soil_27", "soil_26", "soil_27", "soil_26"],
    })
    pairs = pd.DataFrame({
        "high_erosion_cell_id": [3, 4, 5], "low_erosion_cell_id": [1, 2, 2],
        "delta_log1p_erosion": [1.0, 1.0, 2.0],
        "delta_AN": [-2.0, -2.0, -3.0],
    })

    result = summarize_group_responses(
        cells, labels, {("M4", "Aa_20yr", "S0_all"): pairs},
        group_columns=("soil_type_group",), outcomes=("AN",),
    )

    pixel = result.loc[result["analysis_unit"].eq("matched_pixel") & result["group_label"].eq("soil_26")].iloc[0]
    pair = result.loc[result["analysis_unit"].eq("pair") & result["group_label"].eq("soil_26")].iloc[0]
    assert pixel["n_unique_matched_endpoints"] == 3
    assert pair["n_pairs"] == 1
    assert pair["same_group_pair_fraction"] == 2 / 3
    assert pair["pair_population_share"] == 1 / 3


def test_group_spatial_validation_excludes_cross_block_pairs_and_reports_test_blocks():
    import pandas as pd

    from experiments.low_erosion_regime.environment_heterogeneity import validate_group_spatial_performance

    pairs = []
    labels = []
    cell_id = 1
    for block in range(5):
        center_x = (100 + block * 20) * 5000 + 2500
        center_y = 770 * 5000 + 2500
        for replicate in range(3):
            high_id, low_id = cell_id, cell_id + 1
            cell_id += 2
            x = float(block + replicate / 10)
            pairs.append({
                "high_erosion_cell_id": high_id, "low_erosion_cell_id": low_id,
                "x_high": center_x - 15, "x_low": center_x + 15,
                "y_high": center_y, "y_low": center_y,
                "delta_log1p_erosion": x, "delta_AN": 1.7 * x + (replicate - 1) * .01,
                "method": "M4", "exposure": "Aa_20yr", "scheme": "S0_all",
            })
            labels.extend([
                {"cell_id": high_id, "group": "soil_27"},
                {"cell_id": low_id, "group": "soil_27"},
            ])
    # This otherwise valid pair straddles a 5 km block edge and must be omitted.
    pairs.append({
        "high_erosion_cell_id": cell_id, "low_erosion_cell_id": cell_id + 1,
        "x_high": 105 * 5000 + 4990, "x_low": 106 * 5000 + 10,
        "y_high": 770 * 5000 + 2500, "y_low": 770 * 5000 + 2500,
        "delta_log1p_erosion": 9.0, "delta_AN": 999.0,
        "method": "M4", "exposure": "Aa_20yr", "scheme": "S0_all",
    })
    labels.extend([
        {"cell_id": cell_id, "group": "soil_27"},
        {"cell_id": cell_id + 1, "group": "soil_27"},
    ])

    result = validate_group_spatial_performance(
        pd.DataFrame(pairs), pd.DataFrame(labels), group_columns=("group",),
        outcomes=("AN",), seeds=(13,), buffer_m=500,
    )

    assert len(result) == 1
    assert result.loc[0, "n_cross_block_pairs_excluded"] == 1
    assert result.loc[0, "n_pairs_same_group_same_block"] == 15
    assert result.loc[0, "spatial_blocks"] == 5
    assert result.loc[0, "test_blocks"] == 1
    assert result.loc[0, "status"] == "ok"


def test_group_population_support_counts_unique_endpoints_and_low_value_retention():
    import pandas as pd

    from experiments.low_erosion_regime.group_heterogeneity import summarize_group_population_support

    cells = pd.DataFrame({
        "cell_id": [1, 2, 3, 4], "valid_cell": [True] * 4,
        "Aa_20yr": [0.0, 1.0, 2.0, 4.0], "Elevation": [0.0, 2.0, 4.0, 6.0],
    })
    labels = pd.DataFrame({
        "cell_id": [1, 2, 3, 4], "soil_type_group": ["soil_26", "soil_26", "soil_27", "soil_27"],
        "all_group": ["all"] * 4,
    })
    baseline = pd.DataFrame({
        "high_erosion_cell_id": [3, 4], "low_erosion_cell_id": [1, 2],
        "environmental_distance": [.1, .2], "geographic_distance": [100.0, 200.0],
        "delta_log1p_erosion": [1.0, 2.0],
    })
    filtered = pd.DataFrame({
        "high_erosion_cell_id": [4], "low_erosion_cell_id": [3],
        "environmental_distance": [.15], "geographic_distance": [150.0],
        "delta_log1p_erosion": [1.0],
    })

    result = summarize_group_population_support(
        cells, labels,
        {("M4", "Aa_20yr", "S0_all"): baseline,
         ("M4", "Aa_20yr", "T01_fixed_original"): filtered},
        thresholds={"Aa_20yr": 1.0}, group_columns=("soil_type_group", "all_group"),
    )

    soil_26_base = result.loc[
        result["group_label"].eq("soil_26") & result["scheme"].eq("S0_all")
    ].iloc[0]
    soil_27_filtered = result.loc[
        result["group_label"].eq("soil_27") & result["scheme"].eq("T01_fixed_original")
    ].iloc[0]
    assert soil_26_base["n_valid_cells"] == 2
    assert soil_26_base["n_retained_cells"] == 0
    assert soil_26_base["n_pairs_same_group"] == 0
    assert soil_26_base["n_pairs_touching_group"] == 2
    assert soil_26_base["n_unique_matched_endpoints"] == 2
    all_group_base = result.loc[
        result["group_column"].eq("all_group") & result["scheme"].eq("S0_all")
    ].iloc[0]
    assert np.isclose(all_group_base["mean_abs_standardized_environment_difference"], 4 / (5 ** .5))
    assert soil_27_filtered["n_pairs_same_group"] == 1
    assert soil_27_filtered["n_unique_matched_endpoints"] == 2
    assert np.isclose(soil_27_filtered["mean_abs_standardized_environment_difference"], 2 / (5 ** .5))
    assert np.isclose(soil_27_filtered["mean_environmental_distance"], .15)
