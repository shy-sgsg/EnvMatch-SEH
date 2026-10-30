# Phase H Low Erosion Regime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether near-zero erosion values form a reproducible distributional group and compare filtering, rematching, nonlinear and segmented exposure analyses against the existing Phase H baseline.

**Architecture:** Add a standalone `experiments/low_erosion_regime` package and runner. Read only the aligned Phase H input database and existing matched pairs; write all new tables, figures, reports and checkpoints to a unique Oracle run directory. Keep distribution-derived thresholds separate from outcome-optimized thresholds.

**Tech Stack:** Python, pandas/NumPy, SciPy, scikit-learn, Matplotlib, existing `soilmatch.matching.match_cells`, Parquet and CSV.

**Spec:** `/home/shy/.codex/attachments/fb6e252d-180c-488c-9735-d0de4c5e6e8c/pasted-text-1.txt`

## Global Constraints

- Use `EXPLORATORY_ORACLE` and set `outcome_guided_selection=true` for health-guided steps.
- Do not edit the research charter, rebuild RUSLE, run PCNM, overwrite aligned input data or delete historic results.
- Label distribution-derived and health-outcome-optimized thresholds separately.
- Keep M3/M4 or baseline-matched-pair filtering separate from newly rematched samples.
- Do not interpret R² changes across altered populations as pure model improvement.

## Review Focus

- Zero values must remain distinct from NoData and from positive near-zero values.
- KDE must be allowed to return no reliable valley; GMM components must be ordered by mean and uncertain posterior assignments retained.
- Candidate thresholds must map back to original Aa values and record retained fractions.
- Pair filtering must require both endpoints above threshold; rematched pairs must not use health outcomes to define groups or nearest neighbors.
- Spatial validation must keep entire spatial blocks in one role, with test outcomes excluded from all threshold and model selection.

---

### Task 1: Distribution and threshold diagnostics

**Files:**
- Create: `experiments/low_erosion_regime/distribution_diagnostics.py`
- Create: `experiments/low_erosion_regime/run.py`
- Test: existing focused tests in `tests/test_low_erosion_regime.py`

- [ ] Summarize zero/positive values, quantiles, precision-level repetition and order-of-magnitude shares for both exposures.
- [ ] Fit KDE across Scott, Silverman and requested bandwidth multipliers; emit a threshold only for a distinct low peak followed by a density valley.
- [ ] Fit one-to-four component Gaussian mixtures on a reproducible spatially stratified positive-value sample; record AIC/BIC, convergence, weights, overlap and posterior boundaries.
- [ ] Detect an empirical-CDF breakpoint with constrained one/two-break segmented fits and compute Otsu thresholds over multiple histogram bin counts.
- [ ] Save threshold candidates, distributions, spatial bootstrap replicates and diagnostic figures.

### Task 2: Spatial regime maps and population composition

**Files:**
- Create: `experiments/low_erosion_regime/spatial_analysis.py`
- Modify: `experiments/low_erosion_regime/run.py`

- [ ] Map zero, threshold-defined low, main, and uncertain populations for both periods.
- [ ] Summarize low-value shares by soil, original environment group and environmental quintiles.
- [ ] Compute connected components and area summaries on the original row/column lattice.
- [ ] Save retained coverage and group-loss diagnostics for each valid scheme.

### Task 3: Existing-pair filtering and filtered rematching

**Files:**
- Create: `experiments/low_erosion_regime/matching_comparison.py`
- Modify: `experiments/low_erosion_regime/run.py`

- [ ] Filter existing Phase F M3/M4 pairs only when both endpoints exceed each candidate threshold.
- [ ] Re-run fixed-parameter nearest-neighbor matching after filtering with original group labels, then with groups refit on the filtered population.
- [ ] Save pair counts, overlap, environment distance, geographic distance, erosion gaps, SMD, and audit pair tables.

### Task 4: Association, model, segmented exposure and spatial comparisons

**Files:**
- Create: `experiments/low_erosion_regime/response_models.py`
- Create: `experiments/low_erosion_regime/figures.py`
- Modify: `experiments/low_erosion_regime/run.py`

- [ ] Compare nine health outcomes across baseline, zero-only, and valid distribution-derived schemes for both filter and rematch paths.
- [ ] Record Spearman, Kendall, paired tests, slopes, direction, sample sizes and apparent pixel/pair R² separately.
- [ ] Fit piecewise zero-indicator/positive-intensity models on the full population and segmented/spline models against filtered linear/nonlinear fits.
- [ ] Validate selected fixed-model candidates using shared 5 km spatial blocks and buffers; keep test splits out of selection.
- [ ] Generate Chinese PNG/SVG/PDF figures with source tables and captions.

### Task 5: Report, gallery and acceptance

**Files:**
- Create: `docs/reviews/2026-10-29/phase-h/LOW_EROSION_REGIME_RESEARCH_REPORT.md`
- Create: `docs/reviews/2026-10-29/phase-h/CHATGPT_REVIEW_REQUEST.md`
- Create: `docs/reviews/2026-10-29/phase-h/PHASE_H_RESULTS_INDEX.md`
- Modify: `docs/DEVELOPMENT_LOG.md`, `docs/DECISIONS.md`

- [ ] Answer all six threshold-regime scientific questions with full population and matched-unit denominators.
- [ ] Include a link index organized by phase, scientific question, experiment and figure type.
- [ ] Run requested tests, compile checks, diff checks and artifact/source consistency checks.
- [ ] Preserve all pre-existing untracked user files and provide local commit/push commands without running them.
