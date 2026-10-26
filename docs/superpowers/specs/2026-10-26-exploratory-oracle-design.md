# Exploratory Oracle Mode Design

## Intent

Add a separate internal experiment mode that estimates the best observed association and model fit attainable from the available Wenxian cropland prediction rasters. This mode may use SHI and the eight health outcomes to select matching designs, subgroups, models, and retained samples, while preserving an auditable record of every attempted configuration. Its results remain outcome-selected exploratory results and do not become formal scientific estimates without a later independently specified validation and review.

The user explicitly authorizes this mode and a version 1.1 revision of `docs/RESEARCH_CHARTER.md`. The research topic and Q1–Q5 remain unchanged; formal-mode restrictions continue to apply.

## Goals

- Keep `FORMAL` and `EXPLORATORY_ORACLE` execution and outputs distinct.
- Reuse the existing aligned 30 m cropland databases by one-to-one `cell_id` join.
- Explore environment vectors, distance methods/weights, clustering, exposure contrasts, geographic radii, subgroups, sample-retention levels, and multiple response models.
- Produce three interpretable result classes: full-sample model search, outcome-selected apparent upper envelope, and a model selected without access to held-out spatial test blocks.
- Record experiment ID, Git revision, input run ID, exposure, response, variables, matching design, selection rule, sample/pair counts, model/configuration, train/validation/test/apparent R², ΔR², coefficient/direction, errors, balance, runtime, and seed.
- Save ranked experiment tables and requested figures in `outputs/exploratory_oracle/`, with a concise review report under `docs/reviews/<date>/`.

## Non-goals

- No PCNM analysis in this run.
- No non-cropland expansion or complete raw-data re-inventory.
- No modification to any source raster values, health predictions, or erosion predictions.
- No replacement of existing formal outputs or formal-mode rules.
- No claim that model fit to prediction surfaces is independent field validation, causal effect, or a theoretical statistical upper bound.
- No installation of optional dependencies solely to increase model variety; use packages already available.

## Current evidence and constraints

- Current repository baseline is `f2a5500ec2916d255f3b996a778263fe34a1957b` on `main`; the existing run is `outputs/run_20261025T114741_470772Z/`.
- `environment_erosion.parquet` and `health.parquet` each have 392,527 rows, unique `cell_id`, and matching grid coordinates. Health values are present for 391,754–391,907 cells per indicator; `SHI_candidate` is present for 386,851 cells and retains its provisional name.
- With core environment, valid soil codes 26/27/42, and both area-average exposures, the previous G4 sample has 287,785 cells. Use variable-specific complete cases in each experiment; do not use the all-layer `valid_cell` flag as a global filter because optional raster missingness would discard otherwise usable cells.
- SoilType 85 is the mapped water category and is not a soil stratum in the primary candidates; retain counts and allow explicit oracle subgroup diagnostics to show when it was excluded.
- The G4 report records unresolved erosion units, exact Aa_20yr years/construction, cropland-area discrepancy, low clustering silhouette, and missing SHI model provenance. These facts are part of the experiment provenance, not automatic blockers for an explicitly exploratory run.
- The environment includes NumPy, pandas, SciPy, scikit-learn, statsmodels, rasterio, PyArrow, PyYAML, and Matplotlib. Optional Optuna/XGBoost/LightGBM/CatBoost/GAM packages are absent.
- Existing `match_cells`, `build_environment_vectors`, `cluster_environments`, `fit_group_models`, `fit_pair_models`, and spatial block helpers are reusable. Existing regressions cover OLS; new model families and Oracle logging need tests.

## Modes

### `FORMAL`

Formal scientific analysis retains the current charter constraints: health outcomes do not select environment variables, groups, weights, thresholds, matches, or samples. Formal outputs stay in their existing `outputs/run_<id>/` workflow.

### `EXPLORATORY_ORACLE`

Outcome-guided selection is explicitly enabled. Each run sets `outcome_guided_selection=true`, records which outcomes and objective functions guided selection, and writes only beneath `outputs/exploratory_oracle/`. All reported metrics distinguish apparent fit from spatial test performance.

## Search design

### Stage 1: rapid screen

- Draw a deterministic 40,000-cell sample with coverage across valid soil strata and 5 km spatial blocks, using seed `20261026`.
- Run 100 randomized matching/environment configurations for `Aa_20yr` and 100 for `Aa_2020`; evaluate all nine outcomes after each configuration so the matching computation is shared.
- Candidate environment subsets include V1–V8 from the task brief plus random subsets drawn from available core and optional variables. Exclude SoilType 85 from primary soil-stratified clustering.
- Candidate distances include standardized Euclidean, randomized weighted Euclidean, Mahalanobis using a regularized covariance estimate, PCA Euclidean, and direct continuous-space nearest-neighbor matching. Candidate clustering includes no clusters and MiniBatch K-means (per-soil k in 2, 3, 4, 6, 8, 10, 12, 15, 20); small-sample GMM/agglomerative candidates may be included only where runtime is bounded and label assignment is defined.
- Draw environment calipers from `{0.15, 0.25, 0.35, 0.5, 0.75, 1.0}`, geographic radii from `{100, 200, 300, 500, 750, 1000, 1500}` m, and within-group exposure percentile gaps from `{0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50}`. Also test exposure-only and positive-exposure rules as explicitly named scenarios.
- Screen with low-cost OLS and exposure-transform/quadratic forms. All nine outcomes are assessed; the ranking tables retain the objective and outcome used.

### Stage 2: focused optimization

- Select up to 20 unique configurations using validation-only rankings, with outcome and exposure coverage represented in the selected set.
- Rerun matching on a deterministic larger sample (up to 100,000 cells) and compare OLS, Huber, quadratic/piecewise, natural spline, installed statsmodels GAM and quantile regression, spline/Ridge, Random Forest, Extra Trees, and HistGradientBoosting.
- Compare feature regimes: exposure only, environment only, environment plus exposure, environment-group plus exposure, and matched-pair differences.
- Search interpretable subgroup definitions from soil type, environment group, elevation/TWI/LST/vegetation bins, zero/positive exposure, and exposure quantiles. Run the permitted outcome-guided retention sweep at 100%, 75%, 50%, 30%, 20%, 10%, and 5%. Each subset definition, retention target, realized count, and outcome-guided rule is stored.

### Stage 3: complete-sample and spatial evaluation

- Refit selected configurations on their full eligible cropland samples for the full-sample and Oracle apparent-fit leaderboards.
- For independent selection, divide projected cells into 5 km spatial blocks and assign entire blocks to 60% train, 20% validation, and 20% test partitions using a fixed seed. Matching pairs are assigned by spatial block and excluded from model splits when pair members cross partitions.
- Select configurations and subsets using training/validation results only. Open the test partition for the selected designs after selection; do not use test scores to tune the Oracle or model.
- Report that block holdout measures prediction of health raster surfaces across held-out spatial blocks. It does not validate independent soil measurements or remove the source model's spatial/predictor dependence.

## Metric definitions

- `apparent_r2`: in-sample coefficient of determination on the stated sample after fitting; Oracle-selected rows are labeled `selection_mode=outcome_selected_oracle`.
- `train_r2`, `validation_r2`, and `test_r2`: computed on disjoint spatial-block roles. Test metrics exist only for configurations selected without test access.
- `delta_r2`: R² for environment-plus-exposure minus R² for the corresponding environment-only model on the identical rows, split, and algorithm.
- Matched-pair R² uses one row per non-reused pair and predicts the pair health difference from exposure difference (plus explicitly recorded environment-difference terms).
- Coefficients, direction, RMSE, MAE, sample counts, pair counts, and matched covariate SMD accompany R². Pixel count is not described as an independent measured sample count.

## Result organization

The framework will create `experiments/oracle_exploration/` with the requested focused modules and a YAML candidate-search configuration. Each run has a distinct `oracle_<date>_<id>` experiment ID and a configuration snapshot. Results live under `outputs/exploratory_oracle/{experiments,matching,subgroups,models,tables,figures,diagnostics,reports}/`, never under formal `outputs/run_<id>/`.

The final report contains four leaderboards (full-sample best, Oracle apparent best, spatial test best, and exposure ΔR²), best matching configurations, per-indicator results, subgroup/retention details, stage counts, number of candidates/model fits, figure index, selection disclosure, and the user's requested three answers. Figures cover balance by method, R² by environment subset and group count, matching parameters versus R², retained-sample fraction versus Oracle R², full/Oracle/spatial comparison, model rankings, fitted response curves, ΔR², and the Aa_20yr/Aa_2020 comparison. Rasters remain untouched and raw data remains Git-ignored.

## Approaches considered

1. Extend the existing CLI: fewer files, but the formal CLI's health isolation and output contract would be easy to blur.
2. Use a standalone notebook: quick to explore, but hundreds of configurations, outcome-guided filters, and independent test isolation would be harder to audit and resume.
3. Add a dedicated Oracle experiment package and output root: keeps explicit mode IDs, reusable modules, complete trial records, and separate formal outputs. This matches the user's requested structure and is selected.

## Completion criteria

- Charter version 1.1 and the explicitly listed governance files describe both modes; the formal restriction remains scoped to `FORMAL`.
- Every Oracle run and outcome-selected experiment is tagged and recorded with its objective and selection rule; formal-mode regression behavior remains unchanged.
- Stage 1 performs 200 configurations (100 per exposure) on the 40,000-cell sample, unless an execution failure is logged with its cause.
- Stage 2 and Stage 3 preserve the validation/test boundary and produce all four requested leaderboards, per-indicator best rows, sample-retention tables, and requested figures.
- The review report answers the three core questions with exact samples, metrics, selection process, and validation status.
- The complete test suite passes; experiment counts and output files reconcile with the experiment registry.
