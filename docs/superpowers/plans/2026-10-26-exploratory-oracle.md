# Exploratory Oracle Implementation Plan

> **For agentic workers:** Use the native execution workflow and implement tasks in order. Steps use checkbox syntax for tracking.

**Goal:** Implement the authorized `EXPLORATORY_ORACLE` mode, run outcome-guided matching/model searches, and report full-sample, Oracle-apparent, and spatial-test results separately.

**Architecture:** Keep formal analysis rules and `outputs/run_<id>/` unchanged. Add a focused `experiments/oracle_exploration/` package that reads the existing G2/G3 Parquet databases, joins outcomes by unique `cell_id`, records every attempt in a run-scoped experiment registry, and writes only to `outputs/exploratory_oracle/`. Execute Stage 1, Stage 2, and Stage 3 in order; preserve test blocks until selection is complete.

**Tech Stack:** Python 3.13; pandas/PyArrow; NumPy/SciPy; scikit-learn; statsmodels; Matplotlib; YAML; pytest. No new package installation.

**Spec:** `docs/superpowers/specs/2026-10-26-exploratory-oracle-design.md` and user attachment `pasted-text-1.txt`.

## Global Constraints

- `FORMAL` retains health-outcome isolation from environment/matching selection.
- `EXPLORATORY_ORACLE` sets `outcome_guided_selection=true` and records the outcomes/objective that guided each selection.
- Every Oracle output is rooted below `outputs/exploratory_oracle/`; never write it to `outputs/run_<id>/`.
- Use the existing `run_20261025T114741_470772Z` database as the mapped input version; do not re-audit the raw directory or modify raster values.
- Preserve `SHI_candidate` labeling; run the eight confirmed health-indicator fields even if SHI remains provisional.
- Stage 1: 40,000 spatially/soil-covered cells, 100 configurations per exposure, seed `20261026`.
- Stage 2: at most 20 unique configurations, on up to 100,000 cells.
- Stage 3: all eligible cells; 5 km spatial blocks assigned 60/20/20 to train/validation/test; test results must not affect selection.
- Candidate environment calipers: `{0.15, 0.25, 0.35, 0.5, 0.75, 1.0}`; radii `{100, 200, 300, 500, 750, 1000, 1500}` m; percentile gaps `{0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50}`.
- Random environment vectors may also use available NDVI and Precipitation; avoid combining highly correlated NDWI/NDBSI/EVI/NDVI variables or the Elevation/Precipitation pair. V1–V8 stay exactly as specified.
- Report train, validation, test, apparent R², exposure ΔR², coefficient/direction, RMSE/MAE, cell/pair counts, balance, runtime, and random seed.
- Do not run PCNM or change the study population beyond Wenxian cropland.
- Preserve unrelated untracked `paper/` and pre-existing governance plan files.

## Review Focus

- Duplicate or mismatched `cell_id` joins must fail before model fitting; Task 2 pins one-to-one join behavior.
- Missing health data must produce outcome-specific complete-case counts, and `SHI_candidate` must remain provisional; Task 2 and Task 5 test both.
- SoilType 85 and NoData must be recorded as exclusions from soil-stratified candidates; Task 3 verifies category counts.
- Spatial test blocks must not be read by candidate ranking, Oracle retention selection, or model-family selection; Task 6 includes an isolation test.
- A failed experiment must not silently disappear from the registry; Task 2 and Task 4 verify error/status rows and deterministic IDs.

---

### Task 1: Charter version 1.1 and governance mode split (Commit 1)

**Files:**
- Modify: `docs/RESEARCH_CHARTER.md`
- Modify: `AGENTS.md`, `README.md`, `docs/METHODOLOGY.md`
- Modify: `docs/governance/CODEX_TASK_PROTOCOL.md`
- Modify: `docs/governance/CHATGPT_REVIEW_PROTOCOL.md`
- Modify: `docs/governance/RESEARCH_STAGE_GATES.md`
- Modify: `docs/DECISIONS.md`

**Interfaces:** Formal mode rules remain the existing scientific rules. Oracle mode is a separate named experiment with its own output root and selection metadata.

- [x] Change charter version to 1.1 and explicitly scope the health-result selection prohibition in section 6 to formal empirical research.
- [x] Add charter section 11 with the ten Oracle permissions/disclosures from the approved attachment; state it does not alter the research topic and defers PCNM for this run.
- [x] Add `FORMAL` / `EXPLORATORY_ORACLE` mode definitions to the listed governance and user documents.
- [x] Preserve formal G4/G5 gates; state that Oracle mode can proceed as exploratory work and cannot be represented as formal evidence.
- [x] Record the explicit user authorization in `docs/DECISIONS.md`.
- [x] Review the exact charter diff against the authorized text and confirm no other charter section changed.

### Task 2: Run identity, data joins, and trial registry (Commit 2)

**Files:**
- Create: `experiments/__init__.py`
- Create: `experiments/oracle_exploration/__init__.py`
- Create: `experiments/oracle_exploration/config/search.yaml`
- Create: `experiments/oracle_exploration/registry.py`
- Create: `experiments/oracle_exploration/data.py`
- Test: `tests/test_oracle_registry.py`
- Test: `tests/test_oracle_data.py`

**Interfaces:**
- `load_oracle_data(run_dir: Path) -> DataBundle` loads existing Parquet tables and performs a one-to-one `cell_id` join.
- `record_experiment(registry_path: Path, record: Mapping[str, object]) -> None` appends one complete success/error record.
- `make_experiment_id(date: str, sequence: int, config_id: str) -> str` returns a stable unique `oracle_<date>_<sequence>_<config_id>` ID.

- [x] Test that duplicate or non-overlapping cell IDs fail with a clear error and ordinary source tables join exactly once.
- [x] Test that missing response values stay missing and that `SHI_candidate` is never renamed to `SHI`.
- [x] Test registry persistence of success and failure rows with `outcome_guided_selection`, objective, input run ID, Git revision, and seed.
- [x] Implement the loader and append-only registry using the existing G2/G3 output run ID.
- [x] Test config validation and confirm the default candidate counts and parameter values match the design.

### Task 3: Environment and spatial-split candidates

**Files:**
- Create: `experiments/oracle_exploration/environment_search.py`
- Create: `experiments/oracle_exploration/validation.py`
- Test: `tests/test_oracle_environment_search.py`
- Test: `tests/test_oracle_validation.py`

**Interfaces:**
- `candidate_environment_sets(columns, seed, n_random) -> list[tuple[str, ...]]` returns named V1–V8 plus deterministic random subsets.
- `fit_environment_labels(frame, variables, method, k, by_soil, weights, seed) -> np.ndarray` supports no-cluster, MiniBatch K-means, and bounded sample-only GMM/agglomerative options.
- `spatial_roles(frame, block_size_m=5000, seed=20261026) -> pd.Series` assigns whole blocks to train/validation/test.

- [x] Test the named vector sets, deterministic random subsets, and exclusion of health fields from distance columns.
- [x] Test category counts and soil-stratum behavior, including code 85 exclusion and variable complete-case masks.
- [x] Test spatial role reproducibility, 60/20/20 block assignment, no block crossing roles, and test-role isolation from selection APIs.
- [x] Implement distance transforms: standardized Euclidean, weighted Euclidean, regularized Mahalanobis, PCA Euclidean, and direct nearest neighbor.
- [x] Implement bounded clustering candidates; Task 4 registry records realized group counts/sizes, runtime, and failures for each candidate.

### Task 4: Matching search and outcome feedback

**Files:**
- Create: `experiments/oracle_exploration/matching_search.py`
- Test: `tests/test_oracle_matching_search.py`

**Interfaces:**
- `draw_matching_candidates(config, exposure, seed) -> list[dict[str, object]]` returns exactly the configured count and includes vectors, metric, clustering, weights, calipers, radius, rank gap, and exposure rule.
- `run_matching_candidate(frame, candidate, outcomes, registry) -> CandidateResult` builds matches with the existing matching component, joins outcomes only after pair IDs are fixed, scores all nine outcomes, and records failures.

- [x] Test deterministic 100-candidate sampling per exposure and exact value domains for caliper/radius/rank-gap candidates.
- [x] Test `outcome_guided_selection=true`, selected outcome/objective logging, and that each candidate score matches the returned pair/outcome rows.
- [x] Test no pair-cell reuse; each matched-pair row records its actual exposure values, percentile gap, environment distance, and geographic distance.
- [x] Implement Stage 1 ranking on the fixed 40,000-cell sample without exposing test blocks.
- [x] Test error capture so a failed candidate is recorded with status and cause, not omitted.

### Task 5: Regression, nonlinear, and feature-regime search

**Files:**
- Create: `experiments/oracle_exploration/regression_search.py`
- Create: `experiments/oracle_exploration/nonlinear_search.py`
- Test: `tests/test_oracle_regression_search.py`

**Interfaces:**
- `fit_candidate_models(frame, outcome, exposure, environment, group=None, model_families=...) -> pd.DataFrame` returns apparent and role-specific metrics with explicit sample counts.
- `compare_feature_regimes(frame, outcome, exposure, ...) -> pd.DataFrame` returns exposure-only, environment-only, environment+exposure, group+exposure, and matched-pair-difference scores.
- `compute_delta_r2(full_model: float, environment_only: float) -> float` is calculated on identical rows, split, and algorithm.

- [x] Test that each metric is calculated on the same complete-case rows and that missing outcomes are not filled with zero.
- [x] Test apparent R² labels versus spatial validation/test R² and ensure test data never sets hyperparameters or candidate rank.
- [x] Implement OLS, transformed/quadratic/piecewise exposure, natural-spline/Ridge, Huber, statsmodels GAM/quantile where installed, Random Forest, Extra Trees, and HistGradientBoosting.
- [x] Implement the five feature regimes and paired-difference response using one row per non-reused match pair.
- [x] Test ΔR² against a known same-sample synthetic case and record signs, errors, coefficients, and runtime.

### Task 6: Oracle subgroup/retention search and result ranking

**Files:**
- Create: `experiments/oracle_exploration/subgroup_search.py`
- Create: `experiments/oracle_exploration/rank_results.py`
- Test: `tests/test_oracle_subgroup_search.py`
- Test: `tests/test_oracle_rank_results.py`

**Interfaces:**
- `build_subgroup_masks(frame, definitions) -> dict[str, np.ndarray]` creates named masks with explicit predicates.
- `oracle_retention_search(frame, outcome, predictors, fractions) -> pd.DataFrame` returns best apparent R² by 100/75/50/30/20/10/5 percent retention with all outcome-guided rules recorded.
- `rank_leaderboards(registry) -> dict[str, pd.DataFrame]` returns full-sample best, Oracle apparent best, spatial test best, and ΔR² leaderboards.

- [x] Test subgroup predicates and realized/target retention counts.
- [x] Test oracle rankings never label selected apparent R² as test R² and include trial counts and selection objectives.
- [x] Test leaderboard objectives rank the intended columns and preserve outcome/exposure/sample metadata.
- [x] Implement subgroup combinations for soil, environment groups, elevation/TWI/LST/vegetation bins, zero/positive exposure, and exposure quantiles.
- [x] Keep all nine outcomes and both erosion exposures represented in per-indicator outputs.

### Task 7: Stage execution, plots, and review report (Commit 3)

**Files:**
- Create: `experiments/oracle_exploration/generate_figures.py`
- Create: `experiments/oracle_exploration/experiment.py`
- Create: `docs/reviews/2026-10-26/oracle-exploration-review.md`
- Create: `docs/reviews/2026-10-26/figures/` selected summary PNGs
- Modify: `docs/DECISIONS.md`, `docs/DEVELOPMENT_LOG.md`
- Test: `tests/test_oracle_experiment.py`

**Interfaces:**
- `run_exploration(config_path: Path, source_run: Path) -> Path` creates a new `oracle_<date>_<id>` output and executes 40k screen, ≤20-design optimization, and full-sample/spatial evaluation.
- `generate_figures(results_dir: Path) -> list[Path]` writes the ten requested comparison figures.
- `write_review_report(results_dir: Path, report_path: Path) -> Path` writes the three requested answers, four leaderboards, per-indicator results, sample-selection disclosure, model configs, experiment counts, and figure index.

- [x] Test mode-specific output roots and exact stage/result count reconciliation.
- [x] Execute 100 candidates for each exposure and verify all candidates appear in `experiment_results.parquet`.
- [x] Execute the top ≤20 configurations on the larger sample; capture the chosen objective and exposure/health coverage.
- [x] Execute full-sample, Oracle apparent, and validation-selected spatial test analyses with test blocks opened only after selection.
- [x] Generate the requested balance, R²/search, retention, model-ranking, response-curve, ΔR², and exposure-comparison figures.
- [x] Produce all four leaderboards and the report answering the three core questions with exact n, pair counts, rates, metrics, and test-set role.
- [x] Run the full test suite and reconcile every registered trial with output files and summary counts.
- [ ] Stage commit 3 artifacts separately from commits 1 and 2; retain raw inputs and large run outputs under ignored paths.

## Commit grouping

1. Charter and governance mode amendment.
2. Oracle framework, config, and tests.
3. Search results, compact figures, rankings, and review report.

The current execution workspace exposes `.git` as read-only; source work will proceed in the writable tree, and these commit boundaries will be retained for local commit commands if the host continues to deny Git metadata writes.
