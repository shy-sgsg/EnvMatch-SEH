# CEC Formula Correction and Reanalysis Plan

## Goal

Use the user-confirmed formula `CEC_corrected = (CEC - 20 × Erosion) / 2.9 + 18`, where each of the two complete aligned Aa rasters is divided by its own full-valid-extent maximum and the normalized values are summed. Recompute the prior CEC-dependent evidence without overwriting source data or earlier runs.

## Scope

1. Verify the complete Aa grid alignment, full-valid-extent maxima, formula, corrected raster range, and cell/database alignment.
2. Create a separate derived CEC raster/database input under `outputs/exploratory_oracle/` with run metadata; retain the original CEC values and other seven health indicators unchanged.
3. Recompute CEC global, nonlinear, matched, group-heterogeneity, spatial/GWR, local-spatial, and synthesis results using the corrected response, carrying forward the previous analysis populations/configurations where those were outcome-guided.
4. Produce a consolidated CEC correction report and supporting tables/figures, identify superseded CEC claims in prior review documents, and update the project decision/development logs.
5. Verify provenance, counts, formulas, and generated artifacts. Keep all runs in the exploratory Oracle output tree and label their outcome-guided selection status.

## Execution

- Execute in the current worktree; no subagents are available under the session instructions.
- Reuse existing aligned Aa inputs, masks, pair IDs, and analysis code/configurations where they define prior analyses. New outputs receive unique run IDs.
- Do not edit original rasters or existing run outputs.
- Do not run tests unless explicitly requested. Validate the actual analysis artifacts and provenance directly.

## Completion evidence

- Formula-derived raster and tabular CEC agree over the shared valid grid; source files remain untouched.
- Recomputed CEC tables/reports refer to the corrected input and use a single documented formula/maxima.
- Prior stale CEC outputs are clearly superseded by the correction report, with links to corrected evidence.
- Decision and development records describe the correction and remaining analysis limits.


## Completion record

- [x] Verified full-grid maxima, common-valid-cell count, formula range and raster/database agreement.
- [x] Created a separate corrected CEC input bundle; retained source CEC and previous runs.
- [x] Recomputed CEC global/nonlinear, matched, group, GWR/spatial-ML, local-spatial and I7 synthesis analyses under distinct Oracle run IDs; updated low-erosion group sensitivity.
- [x] Added a consolidated report, supporting tables and inspected GWR figure; marked prior narrative reports as superseded for CEC-specific results.
- [x] Updated decision and development logs and checked provenance/output links.
