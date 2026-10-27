# ChatGPT scientific review request — Phase F

请先在基线提交 `0bb1b12fcab24619b3ce8e03f0871a54744be0b4` 阅读 `docs/RESEARCH_CHARTER.md` 的完整 v1.1 版本，确认 Q1–Q5、正式研究边界和 Oracle 授权；随后再审阅 Phase F 改动和结果。Phase F 当前尚未提交，因为本执行环境的 `.git` 只读；本请求记录基线 SHA `0bb1b12fcab24619b3ce8e03f0871a54744be0b4`，正式 Phase F review commit SHA 待用户本地提交后由 `git rev-parse HEAD` 补入。请把本报告与提交后的代码/配置一起审阅，不要把基线 SHA 误当作 Phase F 提交 SHA。

## Review manifest

- Phase F run ID: `phase_f_20261026_seed20261026`
- Input run: `run_20261025T114741_470772Z`
- Mode: `EXPLORATORY_ORACLE`
- `outcome_guided_selection`: `true`
- Charter version: v1.1, unchanged
- Current baseline commit SHA: `0bb1b12fcab24619b3ce8e03f0871a54744be0b4`
- Phase F commit SHA: pending user commit
- Research report: [`PHASE_F_RESEARCH_REPORT.md`](PHASE_F_RESEARCH_REPORT.md)
- Runner/config: [`scripts/run_phase_f_analysis.py`](../../../scripts/run_phase_f_analysis.py), [`config/phase_f.yaml`](../../../config/phase_f.yaml)
- Run provenance and table schemas: `outputs/exploratory_oracle/phase_f_20261026_seed20261026/provenance.json`, `experiment_summary.json`
- Pair-level data manifest: `outputs/exploratory_oracle/phase_f_20261026_seed20261026/pairs/paired_differences_manifest.json`; full-sample pairs and each spatial seed are separate Parquet files (not intended for Git)
- Model parameters: `outputs/exploratory_oracle/phase_f_20261026_seed20261026/tables/selected_model_parameters.json`
- Robustness: `outputs/exploratory_oracle/phase_f_20261026_seed20261026/tables/robustness_sensitivity.csv`
- Exact input checksums and software versions: `outputs/exploratory_oracle/phase_f_20261026_seed20261026/provenance.json`
- Automated software test suite: not added or run; the task is an empirical analysis and developer instructions prohibit tests unless requested. The requested analysis and artifact checks are the execution evidence.

## Required CSV tables

- `phase-f/tables/matching_method_comparison.csv`
- `phase-f/tables/matching_balance_diagnostics.csv`
- `phase-f/tables/paired_difference_results.csv`
- `phase-f/tables/repeated_spatial_validation.csv`
- `phase-f/tables/erosion_increment_r2.csv`
- `phase-f/tables/environmental_heterogeneity.csv`
- `phase-f/tables/high_response_environment_groups.csv`
- `phase-f/tables/exposure_period_comparison.csv`
- `phase-f/tables/final_model_comparison.csv`

## Figures

- `01_matching_coverage_balance`: PNG `phase-f/figures/01_matching_coverage_balance.png`, SVG `phase-f/figures/01_matching_coverage_balance.svg`, PDF `phase-f/figures/01_matching_coverage_balance.pdf`; data `phase-f/figures/data/01_matching_coverage_balance.csv`; caption `phase-f/figures/captions/01_matching_coverage_balance.md`
- `02_matching_method_pair_fit`: PNG `phase-f/figures/02_matching_method_pair_fit.png`, SVG `phase-f/figures/02_matching_method_pair_fit.svg`, PDF `phase-f/figures/02_matching_method_pair_fit.pdf`; data `phase-f/figures/data/02_matching_method_pair_fit.csv`; caption `phase-f/figures/captions/02_matching_method_pair_fit.md`
- `03_primary_paired_slopes`: PNG `phase-f/figures/03_primary_paired_slopes.png`, SVG `phase-f/figures/03_primary_paired_slopes.svg`, PDF `phase-f/figures/03_primary_paired_slopes.pdf`; data `phase-f/figures/data/03_primary_paired_slopes.csv`; caption `phase-f/figures/captions/03_primary_paired_slopes.md`
- `04_repeated_spatial_validation`: PNG `phase-f/figures/04_repeated_spatial_validation.png`, SVG `phase-f/figures/04_repeated_spatial_validation.svg`, PDF `phase-f/figures/04_repeated_spatial_validation.pdf`; data `phase-f/figures/data/04_repeated_spatial_validation.csv`; caption `phase-f/figures/captions/04_repeated_spatial_validation.md`
- `05_erosion_increment_delta_r2`: PNG `phase-f/figures/05_erosion_increment_delta_r2.png`, SVG `phase-f/figures/05_erosion_increment_delta_r2.svg`, PDF `phase-f/figures/05_erosion_increment_delta_r2.pdf`; data `phase-f/figures/data/05_erosion_increment_delta_r2.csv`; caption `phase-f/figures/captions/05_erosion_increment_delta_r2.md`
- `06_environmental_heterogeneity`: PNG `phase-f/figures/06_environmental_heterogeneity.png`, SVG `phase-f/figures/06_environmental_heterogeneity.svg`, PDF `phase-f/figures/06_environmental_heterogeneity.pdf`; data `phase-f/figures/data/06_environmental_heterogeneity.csv`; caption `phase-f/figures/captions/06_environmental_heterogeneity.md`
- `07_environment_groups_map_and_high_response`: PNG `phase-f/figures/07_environment_groups_map_and_high_response.png`, SVG `phase-f/figures/07_environment_groups_map_and_high_response.svg`, PDF `phase-f/figures/07_environment_groups_map_and_high_response.pdf`; data `phase-f/figures/data/07_environment_groups_map_and_high_response.csv`; caption `phase-f/figures/captions/07_environment_groups_map_and_high_response.md`
- `08_health_response_curves`: PNG `phase-f/figures/08_health_response_curves.png`, SVG `phase-f/figures/08_health_response_curves.svg`, PDF `phase-f/figures/08_health_response_curves.pdf`; data `phase-f/figures/data/08_health_response_curves.csv`; caption `phase-f/figures/captions/08_health_response_curves.md`
- `09_pair_distance_and_exposure_diagnostics`: PNG `phase-f/figures/09_pair_distance_and_exposure_diagnostics.png`, SVG `phase-f/figures/09_pair_distance_and_exposure_diagnostics.svg`, PDF `phase-f/figures/09_pair_distance_and_exposure_diagnostics.pdf`; data `phase-f/figures/data/09_pair_distance_and_exposure_diagnostics.csv`; caption `phase-f/figures/captions/09_pair_distance_and_exposure_diagnostics.md`
- `10_exposure_period_comparison`: PNG `phase-f/figures/10_exposure_period_comparison.png`, SVG `phase-f/figures/10_exposure_period_comparison.svg`, PDF `phase-f/figures/10_exposure_period_comparison.pdf`; data `phase-f/figures/data/10_exposure_period_comparison.csv`; caption `phase-f/figures/captions/10_exposure_period_comparison.md`

## Outcome-guided selection audit

- The local matching grid is outcome-blind. Separately, the high-response screen ranks soil-specific V5 k=6 groups by `full_sample_mean_outcome` and retains the top three per exposure-by-outcome family. It considered 324 group candidates in 18 families and retained 54; selected groups cover 0.06734007620622552–0.3167824857343736 of eligible complete-case cells per family (median 0.13742463709309438). Per-group counts and retention fractions are in `phase-f/tables/high_response_environment_groups.csv`.
- Figure 8's inherited full-sample curves come from `oracle_20261026T041521459615Z_20261026`. They were selected by `full_sample_apparent_r2` from 1024 model trials across 13 model families; each selected model and complete-case sample size are in the Figure 8 data CSV. The natural spline uses the fixed plotting sample and is descriptive.
- Detailed curve source metadata is in `phase-f/tables/phase_f_figure_source_summary.json`; run metadata records the same outcome-guided selection audit in `outputs/exploratory_oracle/phase_f_20261026_seed20261026/provenance.json`.

Figure 8 contains two different curve estimands: natural spline on the fixed plot sample and the inherited full-sample best-model partial-dependence curve. The full-sample curve is outcome-selected/apparent. The chart and caption label both scopes.

## Review questions

1. Are M0–M4 defined and compared fairly, especially the distinction between regression adjustment, environment-group slopes, K-means matching, and direct nearest-neighbor matching?
2. Does the percentile-rank matching contrast, V5 distance, 0.35 environment caliper, 1,000m radius, 0.50 rank gap, and no-replacement rule define a defensible paired design? Are the local-grid selection criteria adequate and outcome-blind?
3. Do the recorded pre/post SMDs, variable-level absolute differences, soil/group diagnostics, distance quantiles, zero-positive pairing and unmatched reasons support the balance claims?
4. Are P1–P4 estimands and interpretations correct? Is the no-intercept P2 appropriately treated as a sensitivity check, and is P3's limited V5 adjustment clear?
5. Are all pair endpoints kept within one spatial role, are transformations and K-means learned from training only, and are ten 5km seeds sufficient for internal stability assessment?
6. Is the ΔR² comparison genuinely nested on identical pixels/pairs, identical model families and identical splits? Are the median, IQR, positive fraction and coefficient-direction summaries interpreted appropriately?
7. Are environmental heterogeneity definitions, interaction tests, and FDR correction adequate? Are outcome-ranked high-response groups labeled as exploratory selection?
8. Is Aa_2020 compared on the same common pixels/model/splits at the cell level, with exposure-specific matched pairs clearly distinguished?
9. Are full-sample apparent results, prior Oracle choices, raster prediction-surface limitations, spatial dependence, and unit/time metadata limitations described without causal or independent-observation claims?
10. Based on balance, coverage, exposure separation, repeated spatial performance, ΔR² and subgroup stability together, should M3 or M4 remain the main next-stage candidate, and which indicators merit further work?

## Requested response format

Please identify (a) critical design or calculation issues that would change the scientific conclusion, (b) important repairs needed before the next phase, (c) minor improvements, (d) any claims that exceed the evidence, and (e) whether the proposed next research actions align with the charter. Cite exact file/table/figure and row/column where possible. Do not assume values or sources absent from the manifest; flag them as unknown.

## Review boundaries and next gate

`Aa_20yr` and `Aa_2020` are the already-computed erosion values. No RUSLE reconstruction or factor-product validation was performed. PCNM remains deferred. This phase does not edit the charter or make causal claims. The next formal research phase remains on hold until the user receives this review and decides how to proceed.
