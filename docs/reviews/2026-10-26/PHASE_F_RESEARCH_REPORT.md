# Phase F research report

**Run:** `phase_f_20261026_seed20261026`  
**Mode:** `EXPLORATORY_ORACLE`  
**Baseline commit:** `0bb1b12fcab24619b3ce8e03f0871a54744be0b4` (Phase F files remain uncommitted in this workspace)  
**Input run:** `run_20261025T114741_470772Z`  
**Charter:** `docs/RESEARCH_CHARTER.md` v1.1; unchanged.  

## 1. Research objectives

Phase F compares direct regression, environmental adjustment, environment-group slopes, K-means matching, and direct nearest-neighbor matching. It corrects the previous pooled matched-cell regression by fitting high-minus-low pair differences, repeats whole-block spatial evaluation across ten seeds, estimates erosion's incremental predictive value, checks environmental heterogeneity, and compares the two precomputed exposure periods.

All results remain exploratory. The input outcomes are spatial prediction surfaces; no causal effect or independent field-observation validation is claimed.

## 2. Inherited work and input boundary

The analysis reuses the aligned 392,527-row databases from `run_20261025T114741_470772Z`. `Aa_20yr` and `Aa_2020` are used as already-computed values. RUSLE was not recomputed or factor-validated, PCNM was not run, and the charter was not changed. Input SHA-256 digests, schemas, versions, and row counts are in `phase-f/tables/provenance.json` and the run directory.

The previous Oracle run searched outcome-guided model and subset choices. Its selected curves and subgroups are kept separate and explicitly marked apparent/outcome-guided. Phase F's repeated spatial estimates are internal repeated block holdouts, not external validation.

## 3. Environment matching design

The M4 reference is the prior V5 design (Elevation and LST, direct nearest neighbor, no K-means, caliper 0.35, 1,000m radius, within-stratum exposure percentile gap 0.50, no replacement). M3 uses the same vector and thresholds with soil-specific K-means k=6. The local Aa_20yr search evaluates 243 outcome-blind combinations across V1/V5/V7, calipers 0.25/0.35/0.50, radii 300/500/1,000m, rank gaps 0.30/0.40/0.50, and no clustering/k=6/k=8. Search results rank coverage, balance, distance, and exposure separation; no health outcome enters matching selection.

The local grid uses `40,000` matched-eligible rows per sampled search pool where reported. Final repeated validation fits the environmental scaler and K-means labels on each training role, then constructs train, validation and test pairs independently under frozen thresholds.

## 4. Matching before and after diagnostics

The local search contains 243 candidate configurations; 39 are Pareto candidates under the four recorded design criteria. The CV balance table contains 160 test-role configuration summaries. Pair-level environmental distance, geographic distance, overall and stratum SMD, per-variable absolute differences, actual Aa gaps, zero-positive pairs, and failure reasons are available in `phase-f/tables/matching_balance_diagnostics.csv`.

Outcome-blind local leaders:

| candidate_id | environment_vector | cluster_method | cluster_count_per_soil | environment_caliper | geographic_radius_m | erosion_rank_gap | area_coverage | max_abs_matched_smd | environment_distance_median | median_abs_log1p_Aa_difference | outcome_blind_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| local_090_V5_none_c0.25_r1000_g0.5 | V5 | none | 0.000 | 0.250 | 1000.000 | 0.500 | 0.442 | 0.009 | 0.059 | 2.577 | 1.000 |
| local_144_V5_kmeans_8_c0.25_r1000_g0.5 | V5 | kmeans | 8.000 | 0.250 | 1000.000 | 0.500 | 0.469 | 0.007 | 0.056 | 2.542 | 1.000 |
| local_117_V5_kmeans_6_c0.25_r1000_g0.5 | V5 | kmeans | 6.000 | 0.250 | 1000.000 | 0.500 | 0.465 | 0.009 | 0.056 | 2.548 | 3.000 |
| local_089_V5_none_c0.25_r1000_g0.4 | V5 | none | 0.000 | 0.250 | 1000.000 | 0.400 | 0.588 | 0.008 | 0.055 | 2.403 | 4.000 |
| local_143_V5_kmeans_8_c0.25_r1000_g0.4 | V5 | kmeans | 8.000 | 0.250 | 1000.000 | 0.400 | 0.618 | 0.007 | 0.053 | 2.365 | 5.000 |

## 5. Matched-pair difference models

For each fixed full-sample M3/M4 configuration, the primary exposure contrast is `Δlog1p(Aa) = log1p(Aa_high) − log1p(Aa_low)` and the response is `ΔY = Y_high − Y_low`. P1 uses an intercept; P2 omits the intercept as an assumption sensitivity; P3 adjusts for paired Elevation and LST differences; P4 uses Huber robust regression. Tables include coefficient, standardized coefficient, 95% interval, p-value, paired R², RMSE/MAE, difference distribution, zero-positive pairing, and n.

M4 Aa_20yr primary outcomes (full-sample descriptive estimates):

| outcome | coefficient | ci_lower | ci_upper | standardized_coefficient | r_squared | n_pairs | mean_delta_outcome | positive_delta_outcome_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SHI_candidate | -0.010 | -0.011 | -0.010 | -0.123 | 0.015 | 89811.000 | -0.033 | 0.233 |
| OM | -0.526 | -0.571 | -0.482 | -0.077 | 0.006 | 90867.000 | -2.854 | 0.258 |
| SW | -1.965 | -2.026 | -1.903 | -0.204 | 0.042 | 90867.000 | -6.062 | 0.142 |

These full-sample confidence intervals are model-based and do not account for residual spatial dependence. Repeated spatial test results are reported separately below.

## 6. SHI, OM, and SW

The three priority indicators are compared using the same P1 definition across M3 and M4, both Aa periods, and the same repeated 5km spatial seeds. The table below reports test R² medians and interquartile ranges; no indicator is selected solely by its largest R².

| outcome | method | validation_r2_median | validation_r2_q25 | validation_r2_q75 | positive_direction_fraction | negative_direction_fraction | median_test_coverage | median_test_max_abs_smd | n_seed_folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OM | M0 | 0.144 | 0.126 | 0.190 | 0.000 | 1.000 | NA | NA | 10.000 |
| OM | M1 | 0.122 | 0.109 | 0.169 | 0.000 | 1.000 | NA | NA | 10.000 |
| OM | M2 | 0.177 | 0.134 | 0.199 | 0.000 | 1.000 | NA | NA | 10.000 |
| OM | M3 | 0.001 | -0.006 | 0.006 | 0.000 | 1.000 | 0.505 | 0.010 | 10.000 |
| OM | M4 | -0.003 | -0.012 | 0.007 | 0.000 | 1.000 | 0.483 | 0.012 | 10.000 |
| SHI_candidate | M0 | 0.172 | 0.037 | 0.200 | 0.000 | 1.000 | NA | NA | 10.000 |
| SHI_candidate | M1 | 0.317 | 0.104 | 0.333 | 0.000 | 1.000 | NA | NA | 10.000 |
| SHI_candidate | M2 | 0.371 | 0.289 | 0.413 | 0.000 | 1.000 | NA | NA | 10.000 |
| SHI_candidate | M3 | 0.003 | -0.002 | 0.012 | 0.000 | 1.000 | 0.505 | 0.010 | 10.000 |
| SHI_candidate | M4 | 0.000 | -0.008 | 0.008 | 0.000 | 1.000 | 0.483 | 0.012 | 10.000 |
| SW | M0 | 0.335 | 0.306 | 0.352 | 0.000 | 1.000 | NA | NA | 10.000 |
| SW | M1 | 0.410 | 0.361 | 0.442 | 0.000 | 1.000 | NA | NA | 10.000 |
| SW | M2 | 0.411 | 0.354 | 0.444 | 0.000 | 1.000 | NA | NA | 10.000 |
| SW | M3 | 0.062 | 0.053 | 0.066 | 0.000 | 1.000 | 0.505 | 0.010 | 10.000 |
| SW | M4 | 0.040 | 0.027 | 0.056 | 0.000 | 1.000 | 0.483 | 0.012 | 10.000 |

## 7. Other six health indicators

AN, CEC, pH, BD, MBC, and RESP are included in all cell-level and matched-pair analyses. Inspect `paired_difference_results.csv`, `repeated_spatial_validation.csv`, and `final_model_comparison.csv` for per-indicator coefficients, uncertainty, sample sizes, and spatial test metrics. Because outcomes are prediction surfaces, these results describe associations among aligned spatial predictions.

## 8. Repeated spatial validation

There are 10 deterministic 5km whole-block seeds. Existing grid coordinates define the folds; M3/M4 matching is performed separately within each role, and pair endpoints are checked against the role assignment. M0–M2 use the same complete-case pixels for both Aa periods; M3/M4 use role-specific pairs. Validation and test are reported independently. These are repeated internal spatial holdouts after prior data exploration, not external validation.

## 9. Incremental R² of erosion exposure

The cell-level comparison uses the same V5 covariates, training/test pixels, OLS family, and spatial split for environment-only versus environment-plus-log1p(Aa). Matched-pair comparisons similarly add `Δlog1p(Aa)` to paired environment differences on the same pairs. Test-fold summaries below keep cell-level and matched-pair estimands separate and report medians, interquartile ranges, positive-fold fractions, coefficient-direction consistency, and seed counts:

| analysis_type | method | exposure | outcome | median_delta_r2 | q25_delta_r2 | q75_delta_r2 | positive_delta_r2_fraction | beta_direction_consistency | n_spatial_seeds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cell_level_nested_ols | NA | Aa_2020 | AN | 0.009 | 0.002 | 0.025 | 0.800 | 1.000 | 10.000 |
| cell_level_nested_ols | NA | Aa_2020 | BD | 0.021 | 0.018 | 0.022 | 1.000 | 1.000 | 10.000 |
| cell_level_nested_ols | NA | Aa_2020 | CEC | 0.021 | 0.012 | 0.027 | 1.000 | 1.000 | 10.000 |
| cell_level_nested_ols | NA | Aa_2020 | MBC | 0.012 | 0.010 | 0.016 | 1.000 | 1.000 | 10.000 |
| cell_level_nested_ols | NA | Aa_2020 | OM | 0.034 | 0.028 | 0.043 | 0.900 | 1.000 | 10.000 |
| cell_level_nested_ols | NA | Aa_2020 | RESP | 0.021 | 0.014 | 0.023 | 1.000 | 1.000 | 10.000 |
| cell_level_nested_ols | NA | Aa_2020 | SHI_candidate | 0.038 | 0.024 | 0.050 | 1.000 | 1.000 | 10.000 |
| cell_level_nested_ols | NA | Aa_2020 | SW | 0.044 | 0.032 | 0.051 | 1.000 | 1.000 | 10.000 |
| cell_level_nested_ols | NA | Aa_2020 | pH | 0.001 | -0.001 | 0.002 | 0.700 | 0.900 | 10.000 |
| cell_level_nested_ols | NA | Aa_20yr | AN | 0.037 | 0.020 | 0.058 | 1.000 | 1.000 | 10.000 |
| cell_level_nested_ols | NA | Aa_20yr | BD | 0.019 | 0.016 | 0.020 | 1.000 | 1.000 | 10.000 |
| cell_level_nested_ols | NA | Aa_20yr | CEC | 0.059 | 0.047 | 0.063 | 1.000 | 1.000 | 10.000 |
| cell_level_nested_ols | NA | Aa_20yr | MBC | 0.002 | -0.001 | 0.003 | 0.700 | 1.000 | 10.000 |
| cell_level_nested_ols | NA | Aa_20yr | OM | 0.170 | 0.133 | 0.197 | 1.000 | 1.000 | 10.000 |
| cell_level_nested_ols | NA | Aa_20yr | RESP | 0.041 | 0.032 | 0.046 | 1.000 | 1.000 | 10.000 |
| cell_level_nested_ols | NA | Aa_20yr | SHI_candidate | 0.172 | 0.157 | 0.178 | 1.000 | 1.000 | 10.000 |
| cell_level_nested_ols | NA | Aa_20yr | SW | 0.308 | 0.276 | 0.319 | 1.000 | 1.000 | 10.000 |
| cell_level_nested_ols | NA | Aa_20yr | pH | 0.036 | 0.016 | 0.047 | 0.900 | 1.000 | 10.000 |
| matched_pair_nested_ols | M3 | Aa_2020 | AN | 0.001 | -0.001 | 0.003 | 0.600 | 0.800 | 10.000 |
| matched_pair_nested_ols | M4 | Aa_2020 | AN | 0.001 | -0.001 | 0.002 | 0.600 | 0.800 | 10.000 |
| matched_pair_nested_ols | M3 | Aa_2020 | BD | 0.000 | 0.000 | 0.001 | 0.900 | 1.000 | 10.000 |
| matched_pair_nested_ols | M4 | Aa_2020 | BD | 0.000 | 0.000 | 0.001 | 0.900 | 1.000 | 10.000 |
| matched_pair_nested_ols | M3 | Aa_2020 | CEC | 0.002 | 0.001 | 0.005 | 0.800 | 1.000 | 10.000 |
| matched_pair_nested_ols | M4 | Aa_2020 | CEC | 0.002 | 0.000 | 0.006 | 0.800 | 1.000 | 10.000 |
| matched_pair_nested_ols | M3 | Aa_2020 | MBC | 0.000 | -0.000 | 0.000 | 0.500 | 1.000 | 10.000 |
| matched_pair_nested_ols | M4 | Aa_2020 | MBC | -0.000 | -0.000 | 0.000 | 0.500 | 1.000 | 10.000 |
| matched_pair_nested_ols | M3 | Aa_2020 | OM | 0.002 | -0.001 | 0.006 | 0.700 | 1.000 | 10.000 |
| matched_pair_nested_ols | M4 | Aa_2020 | OM | 0.002 | -0.001 | 0.007 | 0.700 | 1.000 | 10.000 |
| matched_pair_nested_ols | M3 | Aa_2020 | RESP | 0.001 | -0.001 | 0.003 | 0.700 | 1.000 | 10.000 |
| matched_pair_nested_ols | M4 | Aa_2020 | RESP | 0.000 | -0.001 | 0.003 | 0.500 | 1.000 | 10.000 |
| matched_pair_nested_ols | M3 | Aa_2020 | SHI_candidate | 0.002 | -0.002 | 0.009 | 0.700 | 1.000 | 10.000 |
| matched_pair_nested_ols | M4 | Aa_2020 | SHI_candidate | 0.003 | -0.001 | 0.010 | 0.700 | 1.000 | 10.000 |
| matched_pair_nested_ols | M3 | Aa_2020 | SW | 0.004 | 0.001 | 0.009 | 0.800 | 1.000 | 10.000 |
| matched_pair_nested_ols | M4 | Aa_2020 | SW | 0.006 | 0.001 | 0.011 | 0.800 | 1.000 | 10.000 |
| matched_pair_nested_ols | M3 | Aa_2020 | pH | 0.002 | -0.001 | 0.002 | 0.700 | 1.000 | 10.000 |
| matched_pair_nested_ols | M4 | Aa_2020 | pH | 0.002 | 0.001 | 0.003 | 0.800 | 1.000 | 10.000 |

The seed-level records are in `erosion_increment_r2.csv`; these summaries are also in `erosion_increment_r2_summary.csv`. Positive ΔR² is predictive increment under this design; it does not establish an independent causal contribution.

## 10. Environmental heterogeneity

Pair slopes are summarized by exact soil type, soil-specific V5 K-means groups, and pair-midpoint Elevation/TWI/LST/NDWI tertiles where sample support permits. Interaction tests compare group-specific slopes; Benjamini–Hochberg adjusted p-values are reported for the interaction-test family. 94 interaction tests have FDR-adjusted p<0.05 among estimable comparisons. Environmental groups use unsupervised full-sample fitting only for descriptive subgroup summaries; fold models use training-only transforms.

## 11. High-response environmental groups

This Oracle run is explicitly marked `outcome_guided_selection=true`. The group screen uses full-sample mean outcome and ranks groups from highest to lowest, retaining the top three in each exposure-by-outcome family. It considered 324 group candidates across 18 families and retained 54 groups. The selected groups cover 0.067–0.317 of eligible complete-case cells per family (median 0.137). `high_response_environment_groups.csv` records the selection metric, objective, candidate count, group and selected-set cell retention, and complete-case denominator on each row. These are outcome-ranked descriptive groups; matched-pair subgroup estimates remain in `environmental_heterogeneity.csv`.

| exposure | outcome | environment_group | n_cells | mean_outcome | median_outcome | mean_Aa | cell_level_beta_log1p_Aa | cell_level_r_squared_apparent |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Aa_20yr | SHI_candidate | 2704.000 | 29543.000 | 0.651 | 0.650 | 3.123 | -0.007 | 0.085 |
| Aa_20yr | SHI_candidate | 4200.000 | 11023.000 | 0.638 | 0.640 | 5.655 | -0.007 | 0.123 |
| Aa_20yr | SHI_candidate | 4205.000 | 11288.000 | 0.639 | 0.642 | 4.789 | -0.006 | 0.116 |
| Aa_20yr | AN | 2702.000 | 68003.000 | 99.074 | 102.615 | 5.324 | -1.011 | 0.009 |
| Aa_20yr | AN | 2704.000 | 30133.000 | 106.913 | 108.673 | 3.091 | -1.271 | 0.042 |
| Aa_20yr | AN | 4205.000 | 11364.000 | 100.179 | 101.093 | 4.793 | -0.482 | 0.003 |
| Aa_20yr | OM | 2700.000 | 79022.000 | 16.830 | 17.293 | 4.891 | -1.018 | 0.158 |
| Aa_20yr | OM | 2704.000 | 30126.000 | 16.955 | 17.745 | 3.092 | -0.668 | 0.130 |
| Aa_20yr | OM | 4200.000 | 11153.000 | 16.621 | 17.641 | 5.645 | -0.934 | 0.172 |
| Aa_20yr | CEC | 4202.000 | 10086.000 | 22.520 | 22.599 | 6.316 | 0.270 | 0.042 |
| Aa_20yr | CEC | 4204.000 | 4128.000 | 22.393 | 22.272 | 8.855 | 0.373 | 0.075 |
| Aa_20yr | CEC | 4205.000 | 11359.000 | 22.313 | 22.299 | 4.793 | 0.268 | 0.038 |

## 12. Aa_2020 comparison

Aa_2020 is the secondary exposure period. The cell-level comparison requires both exposure values, V5, and the outcome on the same pixels, with identical model forms and spatial seeds. M3/M4 pairs are separately generated for each exposure and therefore are not treated as identical matched observations across periods. Units and exact temporal construction remain as documented in the prior audit; they were not inferred here.

## 13. Main method and data limitations

- The health indicators are spatial prediction surfaces rather than independent field observations.
- The full-sample matched-pair fits are descriptive apparent estimates; their OLS intervals do not address spatial dependence.
- Ten 5km splits are internal repeated spatial holdouts following earlier Oracle exploration, not an untouched external test set.
- The local parameter grid is searched for Aa_20yr; the same V5 reference thresholds are applied to Aa_2020 without claiming they are optimal for that exposure.
- V5 comprises Elevation and LST; residual environmental confounding can remain. P3 adjusts only for paired differences in those two variables.
- The inherited Aa units/time metadata limitations remain. This phase does not validate or recalculate RUSLE.
- Figure 8 compares fixed plotting-sample splines with prior outcome-selected full-sample model curves. The inherited green curves were selected by full-sample apparent R² from 1024 model trials across 13 model families; outcome-specific complete-case counts and selected models are in `phase-f/figures/data/08_health_response_curves.csv`. Both curve families are descriptive and use different sample scopes.
- Maps plot sampled points and do not imply continuous observed surfaces.

## 14. Provisional method recommendation

The evidence does not meet the joint rule for dropping either matcher; retain M3 and M4 as parallel exploratory candidates.

Current median 10-seed M3/M4 test summaries for Aa_20yr priority outcomes:

| outcome | method | validation_r2_median | median_test_coverage | median_test_max_abs_smd | median_test_environment_distance | median_test_geographic_distance_m | median_test_abs_log1p_Aa_gap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OM | M3 | 0.001 | 0.505 | 0.010 | 0.029 | 493.314 | 2.520 |
| OM | M4 | -0.003 | 0.483 | 0.012 | 0.030 | 505.091 | 2.573 |
| SHI_candidate | M3 | 0.003 | 0.505 | 0.010 | 0.029 | 493.314 | 2.520 |
| SHI_candidate | M4 | 0.000 | 0.483 | 0.012 | 0.030 | 505.091 | 2.573 |
| SW | M3 | 0.062 | 0.505 | 0.010 | 0.029 | 493.314 | 2.520 |
| SW | M4 | 0.040 | 0.483 | 0.012 | 0.030 | 505.091 | 2.573 |

Continue to prioritize SHI, OM, and SW only if their paired difference direction, held-out performance, ΔR² stability, balance, and subgroup evidence agree. Do not promote a model based only on apparent R².

## 15. Next research actions

1. Obtain ChatGPT and researcher review of the matching and paired-difference estimands at the exact commit that contains the Phase F artifacts.
2. Reconcile reviewer comments with the charter and decide whether to lock a matcher; preserve M3/M4 if no method dominates on balance, coverage, exposure separation, and repeated spatial performance.
3. Trace source, units, and temporal definitions for Aa and the health prediction surfaces before formal inference.
4. If an indicator remains stable, seek independent measured validation or a genuinely new spatial/temporal assessment set before making stronger claims.
5. Do not advance the next formal phase until the review request is answered and the user approves the resulting research decision.

## Required data tables

- [`matching_method_comparison.csv`](phase-f/tables/matching_method_comparison.csv)
- [`matching_balance_diagnostics.csv`](phase-f/tables/matching_balance_diagnostics.csv)
- [`paired_difference_results.csv`](phase-f/tables/paired_difference_results.csv)
- [`repeated_spatial_validation.csv`](phase-f/tables/repeated_spatial_validation.csv)
- [`erosion_increment_r2.csv`](phase-f/tables/erosion_increment_r2.csv)
- [`environmental_heterogeneity.csv`](phase-f/tables/environmental_heterogeneity.csv)
- [`high_response_environment_groups.csv`](phase-f/tables/high_response_environment_groups.csv)
- [`exposure_period_comparison.csv`](phase-f/tables/exposure_period_comparison.csv)
- [`final_model_comparison.csv`](phase-f/tables/final_model_comparison.csv)

## Figures

- [01_matching_coverage_balance PNG](phase-f/figures/01_matching_coverage_balance.png) · [SVG](phase-f/figures/01_matching_coverage_balance.svg) · [PDF](phase-f/figures/01_matching_coverage_balance.pdf) — The 243-configuration search used 40,000 complete rows for Aa_20yr and did not access health outcomes. Coverage, SMD, environmental distance, and exposure separation were all retained; the horizontal line is a descriptive balance reference.
- [02_matching_method_pair_fit PNG](phase-f/figures/02_matching_method_pair_fit.png) · [SVG](phase-f/figures/02_matching_method_pair_fit.svg) · [PDF](phase-f/figures/02_matching_method_pair_fit.pdf) — Full-sample paired R² is descriptive and not used as the matching selection objective. M3 is soil-specific K-means k=6 plus matching; M4 is direct nearest-neighbor matching within exact soil strata.
- [03_primary_paired_slopes PNG](phase-f/figures/03_primary_paired_slopes.png) · [SVG](phase-f/figures/03_primary_paired_slopes.svg) · [PDF](phase-f/figures/03_primary_paired_slopes.pdf) — Points show full-sample P1 matched-pair slopes and 95% model-based confidence intervals. These intervals do not adjust for residual spatial dependence; the repeated 5km holdout results are reported separately.
- [04_repeated_spatial_validation PNG](phase-f/figures/04_repeated_spatial_validation.png) · [SVG](phase-f/figures/04_repeated_spatial_validation.svg) · [PDF](phase-f/figures/04_repeated_spatial_validation.pdf) — Each box summarizes ten seeded, whole-block 5km spatial test splits. M0–M2 use a common complete-case pixel sample across both Aa periods; M3–M4 use independently constructed train, validation and test pairs with train-fitted transformations.
- [05_erosion_increment_delta_r2 PNG](phase-f/figures/05_erosion_increment_delta_r2.png) · [SVG](phase-f/figures/05_erosion_increment_delta_r2.svg) · [PDF](phase-f/figures/05_erosion_increment_delta_r2.pdf) — For each seed and outcome, the environment-only and environment-plus-log1p(Aa) OLS models use the same training pixels and score the same held-out pixels. This is predictive increment, not causal effect or independent data validation.
- [06_environmental_heterogeneity PNG](phase-f/figures/06_environmental_heterogeneity.png) · [SVG](phase-f/figures/06_environmental_heterogeneity.svg) · [PDF](phase-f/figures/06_environmental_heterogeneity.pdf) — Subgroup estimates use M4 full-sample pairs and prespecified soil, V5 K-means and environmental tertile definitions. Groups were fitted or binned descriptively; subgroup differences remain exploratory and multiple testing is adjusted by Benjamini–Hochberg in the table.
- [07_environment_groups_map_and_high_response PNG](phase-f/figures/07_environment_groups_map_and_high_response.png) · [SVG](phase-f/figures/07_environment_groups_map_and_high_response.svg) · [PDF](phase-f/figures/07_environment_groups_map_and_high_response.pdf) — Left: a deterministic sample of valid cells colored by soil-specific V5 K-means group (Elevation and LST), shown as points on the projected grid. Right: for primary exposure Aa_20yr, the top three full-sample mean-response groups for each primary outcome. The Aa_2020 screen is available in the accompanying table. The top groups are outcome-ranked descriptive subsets, not independently validated hotspots; no continuous or causal spatial surface is inferred.
- [08_health_response_curves PNG](phase-f/figures/08_health_response_curves.png) · [SVG](phase-f/figures/08_health_response_curves.svg) · [PDF](phase-f/figures/08_health_response_curves.pdf) — Blue points use the fixed 10,000-row plotting sample; complete cases are used for each outcome. Orange dashed curves are natural-spline fits to that same plotting sample. Green curves are prior Oracle-selected full-sample apparent model partial-dependence curves for Aa_20yr at median environmental conditions, selected by apparent R² from 1,024 model trials across 13 families. Each green curve's selected model and complete-case n are in the figure data. The curves use different sample scopes; neither is an independent validation curve.
- [09_pair_distance_and_exposure_diagnostics PNG](phase-f/figures/09_pair_distance_and_exposure_diagnostics.png) · [SVG](phase-f/figures/09_pair_distance_and_exposure_diagnostics.svg) · [PDF](phase-f/figures/09_pair_distance_and_exposure_diagnostics.pdf) — Each point is a full-sample M4 pair. Red points indicate a zero-to-positive exposure pair. Distance distributions are matching diagnostics; the full spatial-validation performance is in Figure 4.
- [10_exposure_period_comparison PNG](phase-f/figures/10_exposure_period_comparison.png) · [SVG](phase-f/figures/10_exposure_period_comparison.svg) · [PDF](phase-f/figures/10_exposure_period_comparison.pdf) — Aa_20yr and Aa_2020 are compared with the same complete pixels, V5 variables, OLS design and spatial seeds for the cell-level analysis. Full-sample matched pairs are exposure-specific and are displayed in the accompanying table rather than treated as paired-period identical samples.

Run-level provenance, exact dependencies, input digests, fold metadata, pair-level parquet, local-grid rows, and robustness sensitivities are stored under `outputs/exploratory_oracle/phase_f_20261026_seed20261026/` (Git-ignored). The review directory contains the nine CSVs, figure files, figure data and captions.
