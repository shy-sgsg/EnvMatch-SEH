# Phase J Scientific Synthesis Implementation Plan

> **For agentic workers:** Execute this plan in the current session. Keep the analysis isolated in a new Oracle run and retain existing source results.

**Goal:** Complete Phase J with traceable Phase F–I evidence, updated GWR/MGWR summaries, real multi-scenario SEM analyses, figures, and Chinese research reports.

**Architecture:** Reuse corrected Phase I input tables and saved spatial coefficients; join pixel records by `cell_id`; derive matched, spatial-block, positive-erosion, zero-positive, and environment-group populations; fit transparent path models and a mature lavaan CB-SEM using an isolated temporary R library; write all new artifacts to a unique Phase J Oracle run and review directory.

**Tech Stack:** Python 3.13, pandas/pyarrow, NumPy/SciPy, statsmodels, scikit-learn, Matplotlib, R 4.5.2, lavaan 0.7.3 installed in `/tmp/phasej_Rlib`; project dependency manifests remain unchanged.

**Spec:** `/home/shy/.codex/attachments/39814337-e44d-4750-a943-cbb8330216fb/pasted-text-1.txt`

## Global Constraints

- `EXPLORATORY_ORACLE`; every run records `outcome_guided_selection=true` and selection objectives.
- New analysis outputs go only to `outputs/exploratory_oracle/phase_j/<run_id>/`.
- Use the Phase I corrected CEC database; do not reuse old CEC columns from prior matched-pixel/pair caches.
- Reuse existing GWR/MGWR outputs; do not refit all spatial models.
- Do not run PCNM; do not modify `docs/RESEARCH_CHARTER.md`.
- Treat predictions as spatially dependent grid estimates, not independent measured samples; do not claim causation.
- Do not add or run a test suite; perform analysis-specific data, model, and artifact checks.

## Review Focus

- Corrected CEC rows require both source CEC and both Aa surfaces; compare recomputed and stored values.
- Paired data caches contain pre-correction CEC; rejoin corrected health by cell ID before analysis.
- The source/recipe for `SHI_candidate` is provisional; keep it separate from the health-dimension indicators and flag its CEC dependence as unresolved.
- pH is not a monotone health score and must not be direction-flipped as if it were.
- Model-fit indices and apparent pixel-level precision must not be presented as independent validation or causal evidence.

## Tasks

### Task 1: Register inputs and construct analysis populations

**Files:** Create Phase J run config, provenance, and population tables under `outputs/exploratory_oracle/phase_j/phase_j_20261102_seed20261102_r10/`.

- [x] Profile row grain, unique IDs, missingness, ranges, grid coordinates, and corrected-CEC formula agreement.
- [x] Build D1–D7 populations from corrected inputs and source M4 memberships/pairs; replace stale CEC values by joining corrected CEC on `cell_id`.
- [x] Record sample counts, exclusions, spatial support, and selection status for each population.

### Task 2: Synthesize Phase F–I and GWR/MGWR evidence

**Files:** Create crosswalks and spatial coefficient summaries under the r10 run's `research_index/` and `spatial_synthesis/` directories.

- [x] Link each major claim to a report, source table, figure, population, and statistic.
- [x] Summarize corrected GWR and standard MGWR coefficient distributions, model scales, sign shares, spatial agreement, and environmental associations from saved outputs.
- [x] Produce thesis-ready response, spatial, and stress-zone composite figures with PNG/SVG/PDF exports.

### Task 3: Fit and compare SEM/path scenarios

**Files:** Create a reproducible analysis script in `experiments/oracle_exploration/phase_j_sem.py`; write model tables under the run's `sem/` and `results/` directories.

- [x] Fit S1–S6 and M0–M3 on feasible D1–D7 populations using available tools; keep raw indicator directions and separately define any dimension composites.
- [x] Compare piecewise path regressions and component-based PLS estimates; label the custom covariance-path approximation as exploratory and explicitly document its limits.
- [x] Complete representative CB-SEM models with mature lavaan 0.7.3 and save standardized paths, CFI/TLI/RMSEA/SRMR, AIC/BIC, and endogenous-variable R². Fit S1–S3 for D1–D4 with M0–M3; package is isolated under `/tmp` and not added to project dependency manifests. Fit indices use conventional ML and are not spatially robust.
- [x] Fit custom Mode-B PLS across S1–S3 and M0–M3 on D1–D4 with common rows within each population; record 176 endogenous equations, control-as-single-indicator specification, and no-bootstrap/no-spatial-inference limits.
- [x] Calculate standardized/direct/indirect paths, model fit, full versus no-erosion ΔR², and 26/27/42 and TWI-group comparisons.
- [x] Save all trials, failures, criteria, and retained sample sizes; distinguish apparent model search from spatial block evaluation.

### Task 4: Make SEM figures and integrated research reports

**Files:** Create figures, `PHASE_J_SEM_RESEARCH_REPORT.md`, `PHASE_J_FINAL_TECHNICAL_REPORT.md`, academic report, abstract, presentation outline, and figure atlas in the r10 run directory.

- [x] Produce figures D–I and preserve their exact source tables.
- [x] Integrate findings and limitations into an academic narrative, three core findings, ten question answers, and a clean reader-facing report, with no causal overstatement or hidden Oracle selection.
- [x] Build results index, evidence crosswalk, core figure index, captions, and HTML atlas.

### Task 5: Archive and verify Phase J review materials

**Files:** Create `docs/reviews/2026-11-02/phase-j/` review copy and request; update `docs/DEVELOPMENT_LOG.md` and `docs/DECISIONS.md` only with completed, supported analysis facts.

- [x] Check requested outputs, report/table values, figure labels/fonts, unique figure entries, and retained historical outputs.
- [x] Record unresolved metadata, method limitations, execution status, and answers to all ten requested research questions; the mature CB-SEM acceptance gap is closed with lavaan.
- [x] Complete the earlier independent read-only review of the r10 package and fix its population-purpose wording. After that review, add the 48-model lavaan CB-SEM and update the report and figures; mark those new additions as awaiting independent review.
- [x] Add the PLS control extension to the review package and update the execution status/review request; keep this post-review addition clearly marked as not independently reviewed.
