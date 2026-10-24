# Soil Erosion and Soil Health Project Initialization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Initialize a reproducible Python project for environmental-covariate matching of 30 m soil erosion and soil health data, without inventing results or replacing prior PCNM work.

**Architecture:** Keep raster validation, raster-to-table conversion, environment design, matching, statistical analysis, and reporting in focused `soilmatch` modules. A YAML-driven CLI orchestrates the stages and writes outputs to a new run directory with configuration and software metadata.

**Tech Stack:** Python 3.11+, NumPy, pandas, SciPy, scikit-learn, statsmodels, rasterio, PyArrow, PyYAML, Matplotlib, pytest, and Ruff.

**Spec:** User-provided brief at `/home/shy/.codex/attachments/63f02446-eeef-4bc2-82e8-681df59f1dca/pasted-text-1.txt`.

## Global Constraints

- Preserve the existing PCNM results and use them only for comparison; do not rerun or replace that analysis.
- Exclude R, K, LS, C, and P from the primary matching distance; do not use SHI or health indicators to choose clusters or pairs.
- Analyze actual erosion first and do not create Wenxian results without supplied research data.
- Keep predicted raster values distinct from independent field observations.
- Use log1p erosion for primary variability and regression workflows, and retain zero values safely.
- Enforce aligned grids and suitable resampling rules; never build a full all-cells pairwise distance matrix.
- Use spatial units and uncertainty methods appropriate to gridded predictions; do not call raster-cell counts independent field samples.
- Preserve raw data and existing workspace files; ignore raw rasters and generated outputs in Git.
- Test all algorithmic work with synthetic fixtures, clearly labeled as engineering fixtures.

## Review Focus

- Missing or misaligned rasters: tests must verify grid checks reject mismatched CRS, transform, or dimensions.
- NoData and zero erosion: tests must verify masks and `log1p` statistics remain finite and aligned.
- Matching leakage: tests must verify candidate pairs depend only on exposure, environment, group, soil class, and coordinates, never health columns.
- Duplicate or spatially reused cells: tests must verify one-to-one pairing and configurable reuse behavior.
- Small or singular groups: tests must verify regression and spatial resampling return explicit insufficiency statuses rather than fabricated estimates.

---

### Task 1: Project contracts, configuration, and documentation

**Files:**
- Create: `AGENTS.md`, `README.md`, `.gitignore`, `.gitattributes`, `pyproject.toml`, `requirements.txt`, `Makefile`
- Create: `config/default.yaml`, `config/data_paths.example.yaml`, `config/matching.yaml`
- Create: `docs/RESEARCH_PLAN.md`, `docs/METHODOLOGY.md`, `docs/DATA_REQUIREMENTS.md`, `docs/WORKFLOW.md`, `docs/DEVELOPMENT_LOG.md`, `docs/DECISIONS.md`
- Create directories: `data/raw/{erosion,health,environment,boundaries,pcnm}`, `data/interim`, `data/processed`, `data/fixtures`, `outputs/{tables,figures,maps,matched_pairs,diagnostics,reports}`, `logs`

**Interfaces:** Configuration defines paths by logical data role, raster metadata fields, environment variables, candidate cluster counts `[6, 8, 10, 12]`, and geographic radii `[300, 500, 1000]` metres. Documentation states the full research route and missing-data behavior.

- [ ] Create the project files and document scope, data contracts, scientific safeguards, environment, directory layout, commands, and stage outputs from the brief.
- [ ] Check that every required document is present, paths agree, and no real data paths are hard-coded.
- [ ] Record existing Python/package versions and the unavailable GitHub checkout in `docs/DEVELOPMENT_LOG.md` without claiming repository initialization or research execution.

### Task 2: Raster validation and grid database

**Files:**
- Create: `src/soilmatch/raster_validation.py`, `src/soilmatch/grid_database.py`, `src/soilmatch/io.py`
- Test: `tests/test_data_validation.py`, `tests/test_grid_database.py`

**Interfaces:** `inspect_rasters(paths: Mapping[str, Path], metadata: Mapping[str, Mapping[str, object]]) -> pandas.DataFrame`; `assert_aligned(reference, candidate) -> None`; `build_grid_database(rasters, output, mask=None, batch_size=...) -> Path`.

- [ ] Write synthetic GeoTIFF tests for CRS, pixel size, transform, shape, NoData, finite-value checks, masks, and RUSLE consistency.
- [ ] Run the focused tests and confirm expected failures before implementation.
- [ ] Implement block-window raster reads, alignment checks, quality-report export, and Parquet database creation with `cell_id`, row, column, coordinates, and valid-cell mask.
- [ ] Run focused tests and verify the database preserves cell/value correspondence and handles NoData correctly.

### Task 3: Covariates, environment groups, and erosion variability

**Files:**
- Create: `src/soilmatch/covariates.py`, `src/soilmatch/environment_vectors.py`, `src/soilmatch/clustering.py`, `src/soilmatch/erosion_variability.py`
- Test: `tests/test_environment_vectors.py`, `tests/test_clustering.py`, `tests/test_erosion_variability.py`

**Interfaces:** `build_environment_vectors(frame, continuous, categorical, weights=None) -> (matrix, metadata)`; `environment_distances(left, right, metric="euclidean")`; `cluster_environments(matrix, soil_type, n_clusters, random_state) -> labels, centers`; `summarize_erosion(frame, group_col, erosion_col) -> DataFrame`.

- [ ] Write tests for scaling across units, missing values, category strata, deterministic cluster mapping, cluster-size summaries, and erosion quantiles/zero proportions using `log1p`.
- [ ] Run focused tests and confirm expected failures before implementation.
- [ ] Implement explicit environment-only vector preparation, soil-type stratification, MiniBatch K-means, cluster summaries, raster-label remapping, and erosion summaries.
- [ ] Run focused tests and verify outputs do not inspect health columns when selecting groups.

### Task 4: Fine matching and balance diagnostics

**Files:**
- Create: `src/soilmatch/matching.py`, `src/soilmatch/balance.py`
- Test: `tests/test_matching.py`, `tests/test_balance.py`

**Interfaces:** `match_cells(frame, environment_columns, group_col, erosion_col, soil_type_col, x_col, y_col, distance_limit, erosion_gap, allow_reuse=False) -> (pairs, diagnostics)`; `assess_balance(frame, pairs, covariates) -> DataFrame`.

- [ ] Write tests for exact soil/group constraints, environmental and geographic thresholds, erosion gap, one-to-one/reuse modes, deterministic ties, failure counts, and health-column leakage.
- [ ] Run focused tests and confirm expected failures before implementation.
- [ ] Implement scalable tree-based candidate search with no all-pairs distance matrix, exposure-first high/low pair selection, pair IDs, required output fields, and balance summaries.
- [ ] Run focused tests and confirm every returned pair satisfies the configured constraints.

### Task 5: Regression, spatial resampling, and PCNM comparison

**Files:**
- Create: `src/soilmatch/regression.py`, `src/soilmatch/spatial_statistics.py`, `src/soilmatch/pcnm_comparison.py`
- Test: `tests/test_regression.py`, `tests/test_spatial_statistics.py`, `tests/test_pcnm_comparison.py`

**Interfaces:** `fit_group_models(frame, outcomes, erosion_col, group_col, covariates=()) -> DataFrame`; `fit_pair_models(pairs, outcomes) -> DataFrame`; `make_spatial_blocks(x, y, block_size) -> ndarray`; `block_bootstrap(frame, block_col, fit_function, n_boot, random_state) -> DataFrame`; `compare_pcnm(pcnm, matched, mapping) -> DataFrame`.

- [ ] Write tests for known synthetic slopes, confidence intervals, standardized coefficients, interaction contrasts, pair differences, spatial-block identity, small-sample status, and missing PCNM fields.
- [ ] Run focused tests and confirm expected failures before implementation.
- [ ] Implement group and matched-pair models for SHI plus eight indicators, spatial-block bootstrap interfaces, and schema-tolerant PCNM comparison without imputing missing fields.
- [ ] Run focused tests and distinguish prediction-grid sample size from field-observation counts in outputs.

### Task 6: CLI, visualization, run records, and end-to-end acceptance

**Files:**
- Create: `src/soilmatch/{__init__,cli,config,visualization,reporting}.py`
- Create: `tests/test_pipeline.py`, `tests/conftest.py`
- Create: `scripts/01_validate_data.py` through `scripts/12_generate_reports.py` only where thin wrappers improve the documented workflow.

**Interfaces:** CLI commands expose `--help`, `--config`, paths, logging, and clear missing-input errors. Each run writes a new `outputs/run_<id>/` with config snapshot, metadata, tables, figures, diagnostics, and report.

- [ ] Write an end-to-end synthetic raster test that exercises validation, database building, vectors, clustering, erosion summaries, matching, regression, and figure output; verify missing PCNM data does not block other stages.
- [ ] Run the test and confirm it fails for the missing pipeline behavior.
- [ ] Implement CLI orchestration, deterministic run directories, software/config metadata, and core requested plots (group map when georeferenced labels exist, environment profile, erosion distributions, balance, group regressions, pair differences, forest/indicator summaries, and PCNM comparison when supplied).
- [ ] Run focused and full test suites, Ruff, CLI help, and a missing-data command; record exact results and implementation gaps in the requested project documents.
- [ ] Review the requirements and current Git/network state; commit only project files to the named repository branch and push without force when repository access is available.
