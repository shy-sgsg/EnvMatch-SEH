"""Run representative Phase J CB-SEM models with lavaan on saved Oracle populations."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from phase_j_sem import add_model_variables, control_columns


ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "phase_j_20261102_seed20261102_r10"
DATASETS = {
    "D1_common_env_case": "D1_common_env_case.parquet",
    "D2_240m_blocks": "D2_240m_spatial_units.parquet",
    "D3_M4_unique_pixels": "D3_M4_unique_pixels.parquet",
    "D4_M4_pair_differences": "D4_M4_pair_differences.parquet",
}
BASE_VARS = ["z_E", "physical", "chemical", "biological_activity", "z_SHI"]
ENV_CONTROL_VARS = ["z_Elevation", "z_TWI", "z_NDVI", "z_LST"]


def prepare_inputs(run: Path, out: Path) -> tuple[Path, Path]:
    refs_doc = json.loads((run / "provenance/scale_reference_and_environment_pca.json").read_text(encoding="utf-8"))
    refs = {k: tuple(v) for k, v in refs_doc["scale_reference"].items()}
    pca = refs_doc["environment_pca"]
    combined, specs = [], []
    for dataset, filename in DATASETS.items():
        raw = pd.read_parquet(run / "data" / filename)
        d = add_model_variables(raw, dataset, refs, pca)
        paired = dataset == "D4_M4_pair_differences"
        controls_by_model = {m: control_columns(d, m, paired) for m in ["M0", "M1", "M2", "M3"]}
        all_controls = list(dict.fromkeys(c for cols in controls_by_model.values() for c in cols))
        required = list(dict.fromkeys([*BASE_VARS, *all_controls]))
        d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=required).copy()
        # The same complete rows are used for M0–M3 within each population.
        dpart = d[[*required]].copy()
        dpart.insert(0, "dataset", dataset)
        dpart["outcome_guided_selection"] = True
        combined.append(dpart)
        for control, columns in controls_by_model.items():
            specs.append({"dataset": dataset, "control": control, "controls": ";".join(columns),
                          "n": len(d), "outcome_guided_selection": True})
    out.mkdir(parents=True, exist_ok=True)
    input_path = out / "lavaan_cbsem_input.csv.gz"
    spec_path = out / "lavaan_cbsem_model_spec.csv"
    pd.concat(combined, ignore_index=True, sort=False).to_csv(input_path, index=False, compression="gzip")
    pd.DataFrame(specs).to_csv(spec_path, index=False, encoding="utf-8-sig")
    return input_path, spec_path


def merge_results(run: Path, out: Path) -> dict[str, int]:
    paths = pd.read_csv(out / "lavaan_path_coefficients.csv")
    fits = pd.read_csv(out / "lavaan_model_fit.csv")
    delta = pd.read_csv(out / "lavaan_erosion_increment.csv")
    results = run / "results"
    existing_paths = pd.read_csv(results / "SEM_PATH_COEFFICIENTS.csv")
    existing_fits = pd.read_csv(results / "SEM_MODEL_FIT_COMPARISON.csv")
    existing_delta = pd.read_csv(results / "SEM_EROSION_INCREMENTAL_CONTRIBUTION.csv")
    existing_paths = existing_paths.loc[~existing_paths.method.eq("CB_SEM_lavaan_ML")]
    existing_fits = existing_fits.loc[~existing_fits.method.eq("CB_SEM_lavaan_ML")]
    existing_delta = existing_delta.loc[~existing_delta.method.eq("CB_SEM_lavaan_ML")]
    pd.concat([existing_paths, paths], ignore_index=True, sort=False).to_csv(results / "SEM_PATH_COEFFICIENTS.csv", index=False, encoding="utf-8-sig")
    pd.concat([existing_fits, fits], ignore_index=True, sort=False).to_csv(results / "SEM_MODEL_FIT_COMPARISON.csv", index=False, encoding="utf-8-sig")
    pd.concat([existing_delta, delta], ignore_index=True, sort=False).to_csv(results / "SEM_EROSION_INCREMENTAL_CONTRIBUTION.csv", index=False, encoding="utf-8-sig")
    return {"lavaan_fits": len(fits), "lavaan_paths": len(paths), "lavaan_erosion_increment_rows": len(delta)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=RUN_ID)
    ap.add_argument("--r-library", default="/tmp/phasej_Rlib")
    args = ap.parse_args()
    run = ROOT / "outputs/exploratory_oracle/phase_j" / args.run_id
    out = run / "sem/lavaan_cbsem"
    input_path, spec_path = prepare_inputs(run, out)
    env = os.environ.copy()
    env["R_LIBS_USER"] = args.r_library
    rscript = ROOT / "experiments/oracle_exploration/phase_j_lavaan_cbsem.R"
    subprocess.run(["Rscript", str(rscript), str(input_path), str(spec_path), str(out)],
                   cwd=ROOT, env=env, check=True)
    counts = merge_results(run, out)
    method = json.loads((run / "provenance/method_status.json").read_text(encoding="utf-8"))
    method["lavaan_cbsem"] = {"available": True, "package": "lavaan", "version": (out / "lavaan_version.txt").read_text(encoding="utf-8").strip(),
                                "model_count": counts["lavaan_fits"], "path_count": counts["lavaan_paths"],
                                "populations": list(DATASETS), "structures": ["S1_parallel_dimensions", "S2_direct_plus_indirect_SHI", "S3_indirect_only_SHI"],
                                "controls": ["M0", "M1", "M2", "M3"],
                                "fit_index_caveat": "Conventional iid ML fit indices; spatial dependence is handled only by the separate piecewise path analysis and must be considered in interpretation."}
    method["package_status"] = {**method.get("package_status", {}), "lavaan": True}
    method["reason"] = "A mature lavaan CB-SEM fit was completed in an isolated R library under /tmp; project dependencies were not changed."
    method["outcome_guided_selection"] = True
    (run / "provenance/method_status.json").write_text(json.dumps(method, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(counts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
