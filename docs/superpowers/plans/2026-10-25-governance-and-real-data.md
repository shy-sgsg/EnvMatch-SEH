# Research Governance and Real-Data Audit Implementation Plan

> **For agentic workers:** Follow the requested inline execution order. Steps use checkbox syntax for tracking.

**Goal:** Establish charter-first research governance, then audit the supplied Wenxian raster data and advance to environment-matching feasibility only where quality gates pass.

**Architecture:** First create the initial charter, governance rules, review templates, stage gates, and synchronized project guidance. Then keep raw inputs read-only, generate a complete inventory and local path mapping, audit raster grids/distributions for the user-selected precomputed Aa and candidate covariates, and run environment grouping/matching feasibility only for aligned, scientifically identified inputs.

**Tech Stack:** Existing Python 3.13 environment, rasterio, NumPy, pandas, PyYAML, scikit-learn, project CLI and tests; no new dependencies unless current code cannot meet a named requirement.

**Spec:** `/home/shy/.codex/attachments/2bf8b46b-f143-406b-aa3b-c546a7f090fc/pasted-text-1.txt`

## Global Constraints

- Preserve PCNM results and the approved research theme, questions, and long-term route.
- The first `docs/RESEARCH_CHARTER.md` version is authorized by this task; later body changes require explicit user approval.
- Never modify, rename, resample in place, upload, or commit files under `环境匹配原数据/`.
- Keep SHI and the eight health indicators out of environment selection, clustering, and match selection.
- Keep R/K/LS/C/P out of the primary environmental distance. Per the user's explicit instruction, use the already-calculated actual erosion raster directly and do not validate the RUSLE factor product in this task.
- Do not start health regressions before G4 review and a locked matching design.
- Keep derived outputs traceable to commit, configuration, input paths, and software versions; do not overwrite prior runs.

## Review Focus

- Ambiguous filenames and multiple candidate versions must remain provisional until metadata and raster evidence support a mapping.
- Same nominal 30 m resolution does not prove identical CRS, grid, transform, or spatial support.
- Raster alignment failures must be reported before RUSLE reconstruction; no silent resampling.
- Aa zeros, units/time metadata, and alignment affect whether the selected exposure can be used with candidate covariates.
- The available 90 m covariates may not provide independent 30 m environmental information; evaluate source resolution and resampling history.

---

### Task 1: G0 Charter and Governance

**Files:** Create `docs/RESEARCH_CHARTER.md`; create four files under `docs/governance/`; create four files under `docs/templates/`; modify `AGENTS.md`, `README.md`, and `docs/WORKFLOW.md`; update `docs/DEVELOPMENT_LOG.md` and `docs/DECISIONS.md`.

- [x] Draft the charter from the existing confirmed framework and the user specification; do not introduce a new hypothesis or route.
- [x] Add charter-first execution, ChatGPT scientific review, charter-change approval, stage gates, and review/report templates.
- [x] Check new rules against the existing methods and record the current execution exception in `DECISIONS.md` without changing the research route.
- [x] Run local Markdown link checks; no broken links found.
- [ ] Record the charter commit SHA after the user creates the first local commit.
- [ ] Commit governance as the first requested commit; this environment cannot write `.git`, so the user has a local command; never amend charter text without explicit approval.

### Task 2: G1 Inventory and Mapping

**Files:** Add a reproducible local audit utility under `scripts/`; create ignored `config/data_paths.yaml`; generate `outputs/data_audit/data_inventory.csv` and `data_mapping_report.md`; update `.gitignore` so the raw-data directory cannot be staged accidentally.

- [x] Enumerate every file and compound vector dataset, record size/type/path/category/readability/duplicate evidence and ambiguity.
- [x] Map supported inputs; keep ambiguous SHI, Ap and the official template null; mark environment candidates provisional.
- [x] Confirm absolute path support and add the missing regression test.
- [x] Keep local paths and raw data out of Git.

### Task 3: G2 Raster Quality

**Files:** Reuse `raster_validation.py` where valid; add narrowly scoped fixes and tests if real inputs expose defects; generate `raster_statistics.csv`, `raster_quality_report.md`, and principal-variable figures.

- [x] Inspect raster datasets for CRS, dimensions, transform, resolution, NoData, finite/valid counts, zero rate, distribution, and coverage; report missing units/time.
- [x] Compare every candidate against a provisional environment reference grid; report exact alignment and temporal/source-resampling limits. The official DEM/template remains unavailable.
- [x] No aligned rasters were created because the official template and approval are missing.

### Task 4: G2 Precomputed Exposure Scope

**Files:** Use the user's prepared actual erosion raster directly; generate `rusle_validation_report.md` documenting that factor reconstruction was not requested.

- [x] Map `实际侵蚀_wx.tif` as Aa per explicit user instruction; report its grid, NoData, zero fraction, distribution, and missing units/time.
- [x] Do not reconstruct the factor product or inspect factor overlap; this was explicitly waived by the user.
- [x] Mark dependent matching BLOCKED because Aa and candidate environment/mask grids do not align and the official template is absent.

### Task 5: G3 Covariate Assessment

**Files:** Reuse covariate and raster modules; generate `covariate_assessment.md`, correlation/statistics tables, and figures.

- [x] Evaluate candidate continuous covariates and SoilType for distribution, missingness, spatial variation, correlation/redundancy, time, native resolution, resampling, and shared environmental information.
- [x] Retain the charter's core candidate vector as provisional; no final vector is approved while G2 is blocked.
- [x] Keep possible optional-index changes as proposals only.

### Task 6: G4 Feasibility and Review Package

**Files:** Add tests/fixes for observed code defects; generate candidate groups, erosion variability, matching diagnostics, stage report, and ChatGPT review request.

- [x] Do not run clustering or matching while G2/G3 are blocked; preserve this stop condition in `matching_feasibility.md`.
- [x] Run focused tests for the absolute-path regression, raster audit functions, and misaligned-grid validator failure.
- [x] Run the full test suite (49 passed), inspect Git diff/status, and confirm raw data/local config/outputs remain ignored.
- [ ] Pin the review request to the second local commit SHA after the user commits the audit changes; the draft currently has a SHA placeholder.

