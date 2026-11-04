"""Expand the Phase J Mode-B PLS path comparison across S1-S3 and M0-M3."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from phase_j_sem import add_model_variables, pls_path_model


ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "phase_j_20261102_seed20261102_r10"
DATASETS = {
    "D1_common_env_case": "D1_common_env_case.parquet",
    "D2_240m_blocks": "D2_240m_spatial_units.parquet",
    "D3_M4_unique_pixels": "D3_M4_unique_pixels.parquet",
    "D4_M4_pair_differences": "D4_M4_pair_differences.parquet",
}
STRUCTURES = ["S1_parallel_dimensions", "S2_direct_plus_indirect_SHI", "S3_indirect_only_SHI"]
CONTROLS = ["M0", "M1", "M2", "M3"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=RUN_ID)
    args = parser.parse_args()
    run = ROOT / "outputs/exploratory_oracle/phase_j" / args.run_id
    refs_doc = json.loads((run / "provenance/scale_reference_and_environment_pca.json").read_text(encoding="utf-8"))
    refs = {name: tuple(values) for name, values in refs_doc["scale_reference"].items()}
    pca = refs_doc["environment_pca"]

    fit_path = run / "results/SEM_MODEL_FIT_COMPARISON.csv"
    path_path = run / "results/SEM_PATH_COEFFICIENTS.csv"
    delta_path = run / "results/SEM_EROSION_INCREMENTAL_CONTRIBUTION.csv"
    weights_path = run / "sem/pls_sem/PLS_SEM_OUTER_WEIGHTS.csv"
    fits_old = pd.read_csv(fit_path)
    paths_old = pd.read_csv(path_path)
    delta_old = pd.read_csv(delta_path)
    weights_old = pd.read_csv(weights_path)
    baseline = run / "sem/pls_sem/pre_s1_s3_expansion"
    if not baseline.exists():
        baseline.mkdir(parents=True)
        fits_old.loc[fits_old.method.eq("PLS_PM_ModeB_formative")].to_csv(baseline / "SEM_MODEL_FIT_COMPARISON.csv", index=False)
        paths_old.loc[paths_old.method.eq("PLS_PM_ModeB_formative")].to_csv(baseline / "SEM_PATH_COEFFICIENTS.csv", index=False)
        delta_old.loc[delta_old.method.eq("PLS_PM_ModeB_formative")].to_csv(baseline / "SEM_EROSION_INCREMENTAL_CONTRIBUTION.csv", index=False)
        weights_old.to_csv(baseline / "PLS_SEM_OUTER_WEIGHTS.csv", index=False)

    fits_new, paths_new, delta_new, weights_new, status_rows, support_rows = [], [], [], [], [], []
    for dataset, filename in DATASETS.items():
        raw = pd.read_parquet(run / "data" / filename)
        paired = dataset == "D4_M4_pair_differences"
        modelled = add_model_variables(raw, dataset, refs, pca)
        if dataset == "D4_M4_pair_differences":
            modelled = modelled.loc[modelled.both_positive | modelled.zero_positive].copy()
        for structure in STRUCTURES:
            for control in CONTROLS:
                path_rows, weight_rows, fit_rows, delta_rows, meta = pls_path_model(
                    modelled, dataset, control=control, paired=paired, structure=structure
                )
                paths_new.extend(path_rows)
                weights_new.extend(weight_rows)
                fits_new.extend(fit_rows)
                delta_new.extend(delta_rows)
                status_rows.append(meta)
                support_rows.append({"dataset": dataset, "structure": structure, "control": control,
                                     "n": meta.get("n"), "spatial_blocks_5km": meta.get("spatial_blocks_5km"),
                                     "status": meta.get("status"), "converged": meta.get("converged"),
                                     "outcome_guided_selection": True})

    fits_pls = pd.DataFrame(fits_new)
    expected = {"S1_parallel_dimensions": 48,
                "S2_direct_plus_indirect_SHI": 64,
                "S3_indirect_only_SHI": 64}
    if len(fits_pls) != 176 or not fits_pls.status.eq("ok").all():
        raise RuntimeError(f"Expected 176 successful PLS endogenous equations; observed {len(fits_pls)} with statuses {fits_pls.status.value_counts().to_dict()}")
    if fits_pls.groupby("structure").size().to_dict() != expected:
        raise RuntimeError(f"PLS structure coverage mismatch: {fits_pls.groupby('structure').size().to_dict()}")
    support = pd.DataFrame(support_rows)
    if support.groupby("dataset").n.nunique().max() != 1:
        raise RuntimeError("PLS S1-S3/M0-M3 models do not use a common row support within a data population.")

    fit_all = pd.concat([fits_old.loc[~fits_old.method.eq("PLS_PM_ModeB_formative")], fits_pls], ignore_index=True, sort=False)
    path_all = pd.concat([paths_old.loc[~paths_old.method.eq("PLS_PM_ModeB_formative")], pd.DataFrame(paths_new)], ignore_index=True, sort=False)
    delta_all = pd.concat([delta_old.loc[~delta_old.method.eq("PLS_PM_ModeB_formative")], pd.DataFrame(delta_new)], ignore_index=True, sort=False)
    weights_all = pd.concat([pd.DataFrame(weights_new)], ignore_index=True, sort=False)
    fit_all.to_csv(fit_path, index=False, encoding="utf-8-sig")
    path_all.to_csv(path_path, index=False, encoding="utf-8-sig")
    delta_all.to_csv(delta_path, index=False, encoding="utf-8-sig")
    weights_all.to_csv(weights_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(status_rows).to_csv(run / "sem/pls_sem/PLS_SEM_MODEL_STATUS.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(delta_new).to_csv(run / "sem/pls_sem/PLS_SEM_EROSION_INCREMENT.csv", index=False, encoding="utf-8-sig")
    support.to_csv(run / "sem/pls_sem/PLS_SEM_SAMPLE_SUPPORT.csv", index=False, encoding="utf-8-sig")

    method_path = run / "provenance/method_status.json"
    method = json.loads(method_path.read_text(encoding="utf-8"))
    method["custom_pls_pm"] = status_rows
    method["pls_structure_control_expansion"] = {
        "method": "Mode-B formative PLS outer weights with single-indicator exogenous controls in inner equations",
        "populations": list(DATASETS), "structures": STRUCTURES, "controls": CONTROLS,
        "equations": 176, "status": "ok", "common_rows_within_population": True,
        "uncertainty": "No bootstrap; path coefficients and R2 increments are descriptive and not spatially robust.",
        "outcome_guided_selection": True,
    }
    method["outcome_guided_selection"] = True
    method_path.write_text(json.dumps(method, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    config = {
        "run_id": args.run_id, "method": "Mode-B formative PLS path model",
        "populations": list(DATASETS), "structures": STRUCTURES, "controls": CONTROLS,
        "control_definition": {
            "M0": "erosion only",
            "M1": "Elevation, TWI, NDVI, LST, and population-supported soil-class dummies",
            "M2": "terrain and vegetation-thermal principal components plus soil-class dummies",
            "M3": "M1 controls plus x, y, x2, y2, and xy spatial trend terms",
        },
        "health_blocks": {"Physical": ["SW", "-BD"], "Chemical": ["OM", "AN"],
                          "Biological": ["MBC", "RESP"], "SHI": ["SHI"]},
        "cec": "Analyzed separately; excluded from the chemical composite because its formula contains erosion.",
        "path_structures": {
            "S1_parallel_dimensions": "Erosion and controls to Physical, Chemical, and Biological; no SHI equation.",
            "S2_direct_plus_indirect_SHI": "Erosion and controls to three dimensions and SHI; dimensions also point to SHI.",
            "S3_indirect_only_SHI": "Erosion and controls to three dimensions; dimensions and controls point to SHI, with no direct erosion-to-SHI path.",
        },
        "inner_model": "OLS on standardized construct scores and standardized controls; full versus no-erosion R2 computed on identical rows for each endogenous equation.",
        "uncertainty": "No bootstrap intervals; no spatially robust PLS inference.",
        "outcome_guided_selection": True,
        "sample_support": support.to_dict("records"),
    }
    (run / "provenance/pls_control_expansion.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print({"fits_total": len(fit_all), "paths_total": len(path_all), "delta_total": len(delta_all),
           "pls_fit_rows": len(fits_new), "pls_paths": len(paths_new), "pls_delta_rows": len(delta_new),
           "pls_converged_models": int(fits_pls.status.eq("ok").sum()), "samples_by_dataset": support.groupby("dataset").n.first().to_dict()})


if __name__ == "__main__":
    main()
