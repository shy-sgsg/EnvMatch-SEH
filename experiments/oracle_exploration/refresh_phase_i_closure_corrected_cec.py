#!/usr/bin/env python3
"""Refresh Phase I closure figures whose CEC panel used the superseded raster."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from experiments.oracle_exploration.phase_i_closure import (
    OUTCOME_LABELS,
    OUTCOME_ORDER,
    SOIL_CODES,
    SOIL_COLORS,
    _curve_bins,
    _pair_group_labels,
)

ROOT = Path(__file__).resolve().parents[2]
plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False
OUTCOME_LABELS["CEC"] = "阳离子交换量（CEC）"

RUN_ID = "phase_i_cec_corrected_closure_refresh_20261101_r2"
RUN_DIR = ROOT / "outputs/exploratory_oracle/phase_i" / RUN_ID
OLD_RUN = ROOT / "outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_closure_r1"
CORRECTED_I1 = ROOT / "outputs/exploratory_oracle/phase_i/phase_i_cec_corrected_20_2p9_plus18_20261031_i1"
CORRECTED_HEALTH = ROOT / "outputs/exploratory_oracle/phase_i/cec_corrected_input_20_2p9_plus18_20261031/databases/health.parquet"
DOC_FIGURES = ROOT / "docs/reviews/2026-10-31/phase-i/figures"
FORMULA = "CEC_corrected=(CEC−20×Erosion)/2.9+18"
EROSION = "Aa_20yr/143.0050354 + Aa_2020/48.7655754"


def _save(fig: plt.Figure, stem: str) -> list[str]:
    base = RUN_DIR / "figures" / stem
    base.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for ext in ("png", "svg", "pdf"):
        path = base.with_suffix(f".{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)
    return outputs


def _refresh_c04() -> tuple[pd.DataFrame, list[Path]]:
    source = pd.read_csv(OLD_RUN / "tables/C04_global_best_model_r2_source.csv")
    old_cec = source.loc[source["outcome"].eq("CEC")].copy()
    corrected = pd.read_csv(CORRECTED_I1 / "results/model_search/global_model_search_trials.csv")
    corrected = corrected.loc[
        corrected["outcome"].eq("CEC")
        & corrected["exposure"].eq("Aa_20yr")
        & corrected["status"].eq("ok")
    ].copy()
    corrected["validation_r2"] = pd.to_numeric(corrected["validation_r2"], errors="coerce")
    corrected = corrected.sort_values("validation_r2", ascending=False).drop_duplicates("model_family")
    if set(old_cec["model_family"]) - set(corrected["model_family"]):
        raise ValueError("Corrected CEC model families do not cover the closure figure's families.")
    replacement = corrected.set_index("model_family").reindex(old_cec["model_family"]).reset_index(drop=True)
    replacement["cec_correction_status"] = FORMULA
    source.loc[source["outcome"].eq("CEC"), replacement.columns] = replacement.to_numpy()
    source["cec_correction_status"] = source.get("cec_correction_status", pd.Series(index=source.index, dtype="object"))
    source.loc[source["outcome"].ne("CEC"), "cec_correction_status"] = "not_applicable"

    families = list(dict.fromkeys(source["model_family"].tolist()))
    outcomes = [o for o in OUTCOME_ORDER if o in set(source["outcome"])]
    matrices = []
    for metric in ("validation_r2", "test_r2"):
        matrix = source.pivot(index="outcome", columns="model_family", values=metric).reindex(index=outcomes, columns=families)
        matrices.append(matrix)
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.2), constrained_layout=True, sharey=True)
    for ax, matrix, title in zip(axes, matrices, ["空间块验证集 R²（用于模型选择）", "空间块留出测试 R²（只作报告）"], strict=True):
        data = matrix.to_numpy(dtype=float)
        image = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=-0.5, vmax=0.5)
        ax.set_xticks(range(len(families)), [name.replace("_", "\n") for name in families], rotation=0, fontsize=8)
        ax.set_yticks(range(len(outcomes)), [OUTCOME_LABELS.get(o, o) for o in outcomes])
        ax.set_title(title)
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                value = data[row, col]
                if np.isfinite(value):
                    ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if abs(value) > .30 else "#1d2933")
        fig.colorbar(image, ax=ax, shrink=.78, label="R²")
    fig.suptitle("全样本模型族表现：每个结局×模型族取验证集最佳候选", fontsize=14)
    fig.text(.5, -.01,
             "CEC 面板使用本项目当前数据版本；图内名称统一为 CEC。",
             ha="center", fontsize=8)
    outputs = _save(fig, "C04_global_best_model_r2")
    source.to_csv(RUN_DIR / "tables/C04_global_best_model_r2_source.csv", index=False)
    return source, outputs


def _correct_pair_cec() -> tuple[pd.DataFrame, dict[str, int]]:
    config = yaml.safe_load((ROOT / "config/phase_i_closure.yaml").read_text(encoding="utf-8"))
    pairs, _, _ = _pair_group_labels(config)
    health = pd.read_parquet(CORRECTED_HEALTH, columns=["cell_id", "CEC"])
    lookup = health.set_index("cell_id")["CEC"]
    pairs["high_CEC"] = lookup.reindex(pairs["high_erosion_cell_id"]).to_numpy()
    pairs["low_CEC"] = lookup.reindex(pairs["low_erosion_cell_id"]).to_numpy()
    pairs["delta_CEC"] = pairs["high_CEC"] - pairs["low_CEC"]
    counts = {
        "source_pairs": int(len(pairs)),
        "complete_corrected_CEC_pairs": int(pairs[["delta_log1p_Aa", "delta_CEC"]].apply(pd.to_numeric, errors="coerce").notna().all(axis=1).sum()),
    }
    return pairs.loc[pairs["exposure"].eq("Aa_20yr")].copy(), counts


def _refresh_curves(pairs: pd.DataFrame, stem: str) -> tuple[pd.DataFrame, list[Path]]:
    soil_order = [f"soil_{soil}" for soil in SOIL_CODES]
    q_order = [str(q) for q in range(1, 6)]
    if stem == "C07_soil_type_matched_response_curves":
        subset = pairs.loc[pairs["soil_group"].notna()]
        group_column = "soil_group"
        group_order = soil_order
    else:
        subset = pairs.loc[pairs["TWI_quintile"].notna()]
        group_column = "TWI_quintile"
        group_order = q_order
    corrected = _curve_bins(subset, group_column, ["CEC"], group_order)
    if stem == "C08_twi_group_matched_response_curves":
        corrected["group"] = "q" + corrected["group"].astype(str)
    corrected["cec_correction_status"] = FORMULA
    old_name = "C07_soil_type_matched_response_curve_data.csv" if stem.startswith("C07") else "C08_twi_group_matched_response_curve_data.csv"
    old = pd.read_csv(OLD_RUN / "tables" / old_name)
    combined = pd.concat([old.loc[~old["outcome"].eq("CEC")], corrected], ignore_index=True)
    combined["cec_correction_status"] = combined.get("cec_correction_status", pd.Series(index=combined.index, dtype="object"))
    combined.loc[combined["outcome"].ne("CEC"), "cec_correction_status"] = "not_applicable"
    combined.to_csv(RUN_DIR / "tables" / old_name, index=False)

    fig, axes = plt.subplots(3, 3, figsize=(14.5, 11), squeeze=False, constrained_layout=True)
    if stem.startswith("C07"):
        colors = {f"soil_{soil}": SOIL_COLORS[soil] for soil in SOIL_CODES}
        groups = soil_order
        title = "M4 配对响应曲线：按相同土类分层"
        footer = "点为各组内侵蚀差值八分位的配对均值；CEC 使用本项目当前数据版本。"
    else:
        colors_array = plt.get_cmap("viridis")(np.linspace(.12, .88, 5))
        colors = {f"q{q}": colors_array[q - 1] for q in range(1, 6)}
        groups = [f"q{q}" for q in range(1, 6)]
        title = "M4 配对响应曲线：按土类内 TWI 五分位分层"
        footer = "仅保留两端处于同一土类内 TWI 五分位的配对；CEC 使用本项目当前数据版本。"
    for ax, outcome in zip(axes.flat, OUTCOME_ORDER, strict=True):
        data = combined.loc[combined["outcome"].eq(outcome)]
        for group in groups:
            curve = data.loc[data["group"].eq(group)].sort_values("mean_delta_log1p_Aa")
            if len(curve):
                if stem.startswith("C07"):
                    label = group.replace("soil_", "土类 ")
                else:
                    label = f"TWI 组内五分位 {group}"
                ax.plot(curve["mean_delta_log1p_Aa"], curve["mean_delta_health"], marker="o",
                        markersize=3, linewidth=1.35, color=colors[group], label=label)
        ax.axhline(0, color="#777777", linewidth=.65)
        ax.set_title(OUTCOME_LABELS.get(outcome, outcome), loc="left", fontweight="bold")
        ax.set_xlabel("Δlog1p(Aa)，高侵蚀端减低侵蚀端")
        ax.set_ylabel("配对健康差值 ΔY")
        ax.grid(color="#e5e7eb", linewidth=.55)
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle(title, fontsize=15)
    fig.text(.5, -.012, footer, ha="center", fontsize=8)
    outputs = _save(fig, stem)
    return combined, outputs


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    (RUN_DIR / "tables").mkdir()
    (RUN_DIR / "figures").mkdir()
    c04, outputs = _refresh_c04()
    pairs, pair_counts = _correct_pair_cec()
    c07, c07_outputs = _refresh_curves(pairs, "C07_soil_type_matched_response_curves")
    c08, c08_outputs = _refresh_curves(pairs, "C08_twi_group_matched_response_curves")
    outputs.extend(c07_outputs + c08_outputs)
    fig_dir = DOC_FIGURES
    for stem in ["C04_global_best_model_r2", "C07_soil_type_matched_response_curves", "C08_twi_group_matched_response_curves"]:
        for ext in ("png", "svg", "pdf"):
            source = RUN_DIR / "figures" / f"{stem}.{ext}"
            target_stem = {"C04_global_best_model_r2": "ic_C04_global_best_model_r2",
                           "C07_soil_type_matched_response_curves": "ic_C07_soil_type_matched_response_curves",
                           "C08_twi_group_matched_response_curves": "ic_C08_twi_group_matched_response_curves"}[stem]
            shutil.copy2(source, fig_dir / f"{target_stem}.{ext}")
        shutil.copy2(RUN_DIR / "tables" / ({"C04_global_best_model_r2": "C04_global_best_model_r2_source.csv",
                                              "C07_soil_type_matched_response_curves": "C07_soil_type_matched_response_curve_data.csv",
                                              "C08_twi_group_matched_response_curves": "C08_twi_group_matched_response_curve_data.csv"}[stem]),
                     fig_dir / f"ic_{stem}_data.csv")
    provenance = {
        "run_id": RUN_ID,
        "phase": "I-closure-CEC-refresh",
        "mode": "EXPLORATORY_ORACLE",
        "outcome_guided_selection": True,
        "cec_formula": FORMULA,
        "erosion_definition": EROSION,
        "inputs": {
            "old_closure_sources": "outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_closure_r1/tables/C04,C07,C08",
            "corrected_cec_model_search": "outputs/exploratory_oracle/phase_i/phase_i_cec_corrected_20_2p9_plus18_20261031_i1/results/model_search/global_model_search_trials.csv",
            "corrected_cec_cell_values": "outputs/exploratory_oracle/phase_i/cec_corrected_input_20_2p9_plus18_20261031/databases/health.parquet",
            "fixed_pair_candidate": "outputs/exploratory_oracle/phase_h_20261028_seed20261028_h2/pairs/candidate_pairs/stage2_Aa_20yr_01.parquet",
        },
        "pair_counts": pair_counts,
        "corrected_cec_model_rows": int(c04.loc[c04["outcome"].eq("CEC")].shape[0]),
        "corrected_cec_soil_curve_rows": int(c07.loc[c07["outcome"].eq("CEC")].shape[0]),
        "corrected_cec_twi_curve_rows": int(c08.loc[c08["outcome"].eq("CEC")].shape[0]),
        "figure_outputs": [p.relative_to(ROOT).as_posix() for p in outputs],
        "active_doc_outputs": ["docs/reviews/2026-10-31/phase-i/figures/ic_C04_global_best_model_r2.*",
                               "docs/reviews/2026-10-31/phase-i/figures/ic_C07_soil_type_matched_response_curves.*",
                               "docs/reviews/2026-10-31/phase-i/figures/ic_C08_twi_group_matched_response_curves.*"],
        "interpretation": "CEC is formula-derived and includes a negative Aa component; CEC response figures are not independent evidence of an erosion effect.",
    }
    (RUN_DIR / "manifests").mkdir()
    (RUN_DIR / "manifests/provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (RUN_DIR / "RUN_STATUS.md").write_text(
        f"# {RUN_ID}\n\nStatus: corrected CEC panels refreshed; non-CEC rows retained from the fixed prior outputs.\n\n"
        "Mode: EXPLORATORY_ORACLE\n\noutcome_guided_selection: true\n\n"
        f"CEC formula: `{FORMULA}`; Erosion: `{EROSION}`.\n",
        encoding="utf-8",
    )
    print(json.dumps({"run_dir": str(RUN_DIR), "pair_counts": pair_counts,
                      "CEC_rows": {"C04": provenance["corrected_cec_model_rows"],
                                   "C07": provenance["corrected_cec_soil_curve_rows"],
                                   "C08": provenance["corrected_cec_twi_curve_rows"]}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
