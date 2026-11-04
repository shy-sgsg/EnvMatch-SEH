"""Build Phase J synthesis tables, figures, reports, and Chinese figure atlas."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PHASE_I = ROOT / "outputs/exploratory_oracle/phase_i"
RUN_ID = "phase_j_20261102_seed20261102_r10"
OUTCOME_ORDER = ["SHI_candidate", "SW", "OM", "AN", "CEC", "pH", "BD", "MBC", "RESP"]
LABELS = {"SHI_candidate": "土壤健康指数（SHI）", "AN": "碱解氮（AN）", "OM": "有机质（OM）",
          "CEC": "阳离子交换量（CEC）", "pH": "酸碱度（pH）", "SW": "土壤含水量（SW）",
          "BD": "土壤容重（BD）", "MBC": "微生物量碳（MBC）", "RESP": "土壤呼吸（RESP）"}
SOIL_LABELS = {"26": "26 冲积土", "27": "27 潮土", "42": "42 脱潮土"}
FONT_PATH = Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
font_manager.fontManager.addfont(str(FONT_PATH))
FONT = font_manager.FontProperties(fname=str(FONT_PATH)).get_name()


def csv_write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def save_figure(fig: plt.Figure, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(directory / f"{stem}.{ext}", dpi=300 if ext == "png" else None,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def source_paths() -> dict[str, Path]:
    return {
        "nine": PHASE_I / "phase_i_20261101_corrected_cec_nine_outcome_refresh_r2",
        "mgwr": PHASE_I / "phase_i_20261031_seed20261031_closure_r1",
        "i8": PHASE_I / "phase_i_20261031_seed20261031_i8_r2",
        "final_i": ROOT / "docs/reviews/2026-10-31/phase-i/PHASE_I_FINAL_RESEARCH_REPORT.md",
        "phase_f": ROOT / "docs/reviews/2026-10-26/PHASE_F_RESEARCH_REPORT.md",
        "phase_g": ROOT / "docs/reviews/2026-10-27/phase-g/PHASE_G_RESEARCH_REPORT.md",
        "phase_h": ROOT / "docs/reviews/2026-10-29/phase-h/PHASE_H_FINAL_SUMMARY.md",
    }


def build_spatial_tables(run: Path) -> dict[str, pd.DataFrame]:
    src = source_paths()
    nine = src["nine"]
    mgwr_dir = src["mgwr"] / "tables"
    weighted = pd.read_csv(nine / "tables/gwr_weighted_effect_summary.csv")
    beta = pd.read_parquet(nine / "tables/full_sample_local_coefficients.parquet")
    mgwr_local = pd.read_csv(mgwr_dir / "mgwr_local_coefficients.csv")
    model = pd.read_csv(mgwr_dir / "mgwr_model_summary.csv")
    bandwidth = pd.read_csv(mgwr_dir / "mgwr_variable_bandwidths.csv")
    differences = pd.read_csv(mgwr_dir / "gwr_mgwr_local_coefficient_differences.csv")
    spatial_rows = []
    for outcome in OUTCOME_ORDER:
        for family, frame, col, support in [
            ("GWR-LCR（九项结局）", beta.loc[beta.model_family.eq("GWR-LCR")], "local_beta_standardized", "n_pixels"),
            ("同样本标准GWR（770空间单元）", mgwr_local.loc[mgwr_local.model_family.eq("standard_GWR_same_sample")], "local_beta_log1p_Aa_standardized", "n_pixels"),
            ("标准MGWR（770空间单元）", mgwr_local.loc[mgwr_local.model_family.eq("standard_MGWR_spatially_balanced_sample")], "local_beta_log1p_Aa_standardized", "n_pixels"),
        ]:
            subset = frame.loc[frame.outcome.eq(outcome)].copy()
            if subset.empty:
                continue
            w = pd.to_numeric(subset[support], errors="coerce").fillna(1).to_numpy(dtype=float)
            v = pd.to_numeric(subset[col], errors="coerce").to_numpy(dtype=float)
            ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
            local = v[ok]
            unit_negative = local < 0
            spatial_rows.append({"outcome": outcome, "label": LABELS[outcome], "model_family": family,
                                 "n_units": int(ok.sum()), "n_supported_pixels": int(w[ok].sum()),
                                 "weighted_mean_beta": float(np.average(v[ok], weights=w[ok])),
                                 "median_beta": float(np.median(v[ok])),
                                 "min_beta": float(np.min(local)),
                                 "p5_beta": float(np.quantile(local, .05)),
                                 "p25_beta": float(np.quantile(local, .25)),
                                 "p10_beta": float(np.quantile(v[ok], .10)), "p90_beta": float(np.quantile(v[ok], .90)),
                                 "p75_beta": float(np.quantile(local, .75)),
                                 "p95_beta": float(np.quantile(local, .95)),
                                 "max_beta": float(np.max(local)),
                                 "negative_unit_fraction": float(np.mean(unit_negative)),
                                 "negative_fraction": float(np.average(v[ok] < 0, weights=w[ok])),
                                 "outcome_guided_selection": True})
    spatial_summary = pd.DataFrame(spatial_rows)
    csv_write(weighted, run / "spatial_synthesis/corrected_nine_outcome_gwr_weighted_summary.csv")
    csv_write(model, run / "spatial_synthesis/same_sample_gwr_mgwr_fit_summary.csv")
    csv_write(bandwidth, run / "spatial_synthesis/mgwr_variable_scales.csv")
    csv_write(spatial_summary, run / "spatial_synthesis/local_coefficient_distribution_summary.csv")
    csv_write(differences, run / "spatial_synthesis/gwr_mgwr_spatial_unit_differences.csv")
    gwr_lcr_agreement_rows = []
    beta_pair = beta.loc[
        beta.model_family.eq("GWR-LCR") |
        (beta.model_family.eq("GWR") & beta.analysis_population.eq("environment_adjusted"))
    ].pivot(
        index=["tile_row", "tile_col"], columns=["outcome", "model_family"], values="local_beta_standardized"
    )
    for outcome in OUTCOME_ORDER:
        try:
            gwr_values = pd.to_numeric(beta_pair[(outcome, "GWR")], errors="coerce")
            lcr_values = pd.to_numeric(beta_pair[(outcome, "GWR-LCR")], errors="coerce")
        except KeyError:
            continue
        valid = gwr_values.notna() & lcr_values.notna()
        if not valid.any():
            continue
        gwr_v, lcr_v = gwr_values[valid], lcr_values[valid]
        gwr_lcr_agreement_rows.append({
            "outcome": outcome, "label": LABELS[outcome], "n_common_units": int(valid.sum()),
            "pearson_local_beta_correlation": float(gwr_v.corr(lcr_v, method="pearson")),
            "spearman_local_beta_correlation": float(gwr_v.corr(lcr_v, method="spearman")),
            "sign_agreement_fraction": float((np.sign(gwr_v) == np.sign(lcr_v)).mean()),
            "mean_absolute_beta_difference": float((gwr_v - lcr_v).abs().mean()),
            "outcome_guided_selection": True,
        })
    gwr_lcr_agreement = pd.DataFrame(gwr_lcr_agreement_rows)
    csv_write(gwr_lcr_agreement, run / "spatial_synthesis/gwr_gwrlcr_local_agreement.csv")
    # Summarize agreement only on the common 770-unit SHI/OM/SW support.
    agreement = []
    for outcome, group in differences.groupby("outcome"):
        a = group.local_beta_log1p_Aa_standardized_gwr.to_numpy(dtype=float)
        b = group.local_beta_log1p_Aa_standardized_mgwr.to_numpy(dtype=float)
        ok = np.isfinite(a) & np.isfinite(b)
        agreement.append({"outcome": outcome, "label": LABELS.get(outcome, outcome), "n_common_units": int(ok.sum()),
                          "pearson_local_beta_correlation": float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 2 else np.nan,
                          "sign_agreement_fraction": float(np.mean(np.sign(a[ok]) == np.sign(b[ok]))) if ok.any() else np.nan,
                          "mean_absolute_beta_difference": float(np.mean(np.abs(b[ok] - a[ok]))) if ok.any() else np.nan,
                          "outcome_guided_selection": True})
    csv_write(pd.DataFrame(agreement), run / "spatial_synthesis/gwr_mgwr_local_agreement.csv")

    # Compare the strongest-negative local deciles across SHI, OM, and SW on
    # common spatial supports. This is a descriptive overlap, not a hotspot test.
    overlap_rows = []
    outcome_pair_order = [("SHI_candidate", "OM"), ("SHI_candidate", "SW"), ("OM", "SW")]
    for family, frame, col in [
        ("GWR-LCR", beta.loc[beta.model_family.eq("GWR-LCR")], "local_beta_standardized"),
        ("GWR", mgwr_local.loc[mgwr_local.model_family.eq("standard_GWR_same_sample")],
         "local_beta_log1p_Aa_standardized"),
        ("MGWR", mgwr_local.loc[mgwr_local.model_family.eq("standard_MGWR_spatially_balanced_sample")],
         "local_beta_log1p_Aa_standardized"),
    ]:
        part = frame.loc[frame.outcome.isin(["SHI_candidate", "OM", "SW"])].copy()
        part["_location"] = (part["tile_row"].astype(str) + "_" + part["tile_col"].astype(str))
        wide = part.pivot(index="_location", columns="outcome", values=col).dropna(
            subset=["SHI_candidate", "OM", "SW"]
        )
        n_common = len(wide)
        decile_n = max(1, int(np.floor(n_common * .10)))
        strongest = {
            outcome: set(wide[outcome].rank(method="first", ascending=True).nsmallest(decile_n).index)
            for outcome in ["SHI_candidate", "OM", "SW"]
        }
        for outcome_a, outcome_b in outcome_pair_order:
            intersection = strongest[outcome_a] & strongest[outcome_b]
            union = strongest[outcome_a] | strongest[outcome_b]
            overlap_rows.append({
                "model_family": family, "outcome_a": outcome_a, "label_a": LABELS[outcome_a],
                "outcome_b": outcome_b, "label_b": LABELS[outcome_b],
                "n_common_units": int(n_common), "strongest_negative_decile_units_each": int(decile_n),
                "spearman_local_beta_rho": float(wide[outcome_a].corr(wide[outcome_b], method="spearman")),
                "decile_overlap_units": int(len(intersection)),
                "decile_overlap_jaccard": float(len(intersection) / len(union)) if union else np.nan,
                "selection_definition": "各指标同一空间支持上局部β最低的10%单元；纯描述性重叠",
                "outcome_guided_selection": True,
            })
    overlap = pd.DataFrame(overlap_rows)
    csv_write(overlap, run / "spatial_synthesis/CROSS_INDICATOR_SPATIAL_OVERLAP.csv")
    # Relate saved local coefficients to the already-aggregated 240 m environment
    # units. These are descriptive comparisons; they do not refit a spatial model.
    env_columns = ["Elevation", "TWI", "NDVI", "NDWI", "LST", "SoilType"]
    env_units = pd.read_parquet(run / "data/D2_240m_spatial_units.parquet")
    env_units = env_units[["tile_row", "tile_col", "n_pixels", *env_columns]].rename(
        columns={"n_pixels": "environment_support_pixels"}
    )
    quartile_labels = ["最低四分位", "较低四分位", "较高四分位", "最高四分位"]
    for variable in ["Elevation", "TWI", "NDVI", "NDWI", "LST"]:
        rank = env_units[variable].rank(method="average", pct=True)
        env_units[f"{variable}_group"] = pd.cut(
            rank, bins=[0, .25, .5, .75, 1], labels=quartile_labels,
            include_lowest=True, right=True
        ).astype("string")
    soil_labels = {26: "26冲积土", 27: "27潮土", 42: "42脱潮土", 85: "85其他编码"}
    env_units["SoilType_group"] = pd.to_numeric(env_units.SoilType, errors="coerce").round().map(soil_labels)
    env_units = env_units.drop(columns="SoilType")

    environment_rows = []
    environment_group_rows = []
    environment_models = [
        ("GWR-LCR", beta.loc[beta.model_family.eq("GWR-LCR")], "local_beta_standardized"),
        ("GWR", mgwr_local.loc[mgwr_local.model_family.eq("standard_GWR_same_sample")],
         "local_beta_log1p_Aa_standardized"),
        ("MGWR", mgwr_local.loc[mgwr_local.model_family.eq("standard_MGWR_spatially_balanced_sample")],
         "local_beta_log1p_Aa_standardized"),
    ]
    for family, frame, coefficient_column in environment_models:
        local = frame.loc[frame.outcome.isin(["SHI_candidate", "OM", "SW"])].copy()
        local = local.merge(env_units, on=["tile_row", "tile_col"], how="inner", validate="many_to_one")
        for outcome in ["SHI_candidate", "OM", "SW"]:
            subset = local.loc[local.outcome.eq(outcome)].copy()
            beta_values = pd.to_numeric(subset[coefficient_column], errors="coerce")
            for variable in env_columns:
                if variable == "SoilType":
                    group_column = "SoilType_group"
                    rho = np.nan
                else:
                    group_column = f"{variable}_group"
                    numeric = pd.to_numeric(subset[variable], errors="coerce")
                    valid = beta_values.notna() & numeric.notna()
                    rho = float(beta_values[valid].corr(numeric[valid], method="spearman")) if valid.sum() > 2 else np.nan
                valid_group = subset.loc[beta_values.notna() & subset[group_column].notna()].copy()
                valid_group["_local_beta"] = pd.to_numeric(valid_group[coefficient_column], errors="coerce")
                for group_name, group in valid_group.groupby(group_column, observed=True):
                    environment_group_rows.append({
                        "model_family": family, "outcome": outcome, "label": LABELS[outcome],
                        "environment_variable": variable, "environment_group": str(group_name),
                        "group_definition": "土类编码" if variable == "SoilType" else "全域240 m单元环境秩四分位",
                        "n_units": int(len(group)),
                        "n_supported_pixels": int(pd.to_numeric(group.environment_support_pixels, errors="coerce").fillna(0).sum()),
                        "median_local_beta": float(group._local_beta.median()),
                        "mean_local_beta": float(group._local_beta.mean()),
                        "negative_unit_fraction": float((group._local_beta < 0).mean()),
                        "outcome_guided_selection": True,
                    })
                environment_rows.append({
                    "model_family": family, "outcome": outcome, "label": LABELS[outcome],
                    "environment_variable": variable,
                    "spearman_local_beta_environment_rho": rho,
                    "n_common_units": int(valid.sum()) if variable != "SoilType" else int(beta_values.notna().sum()),
                    "outcome_guided_selection": True,
                    "interpretation_note": "描述性空间单元相关；负向ρ表示环境值较高处局部β倾向更负，不作因果解释",
                })
    environment_associations = pd.DataFrame(environment_rows)
    environment_groups = pd.DataFrame(environment_group_rows)
    contrast_rows = []
    for (family, outcome, variable), group in environment_groups.groupby(
        ["model_family", "outcome", "environment_variable"], sort=False
    ):
        ordered = group.sort_values("median_local_beta")
        strongest = ordered.iloc[0]
        weakest = ordered.iloc[-1]
        contrast_rows.append({
            "model_family": family, "outcome": outcome, "label": LABELS[outcome],
            "environment_variable": variable,
            "strongest_negative_group": strongest.environment_group,
            "strongest_negative_median_beta": strongest.median_local_beta,
            "strongest_negative_n_units": strongest.n_units,
            "weakest_negative_group": weakest.environment_group,
            "weakest_negative_median_beta": weakest.median_local_beta,
            "weakest_negative_n_units": weakest.n_units,
            "stronger_minus_weaker_beta": strongest.median_local_beta - weakest.median_local_beta,
            "outcome_guided_selection": True,
        })
    environment_contrasts = pd.DataFrame(contrast_rows)
    csv_write(environment_associations, run / "spatial_synthesis/LOCAL_BETA_ENVIRONMENT_ASSOCIATION.csv")
    csv_write(environment_groups, run / "spatial_synthesis/LOCAL_BETA_BY_ENVIRONMENT_GROUP.csv")
    csv_write(environment_contrasts, run / "spatial_synthesis/LOCAL_BETA_ENVIRONMENT_CONTRASTS.csv")

    # Retain the locations with the largest saved GWR/MGWR coefficient differences.
    largest_differences = differences.copy()
    largest_differences = largest_differences.merge(
        env_units, on=["tile_row", "tile_col"], how="left", validate="many_to_one"
    )
    largest_differences["absolute_beta_difference"] = largest_differences[
        "local_beta_difference_mgwr_minus_gwr"
    ].abs()
    largest_differences = largest_differences.sort_values(
        ["outcome", "absolute_beta_difference"], ascending=[True, False]
    ).groupby("outcome", sort=False).head(10).reset_index(drop=True)
    largest_differences["outcome_guided_selection"] = True
    csv_write(largest_differences, run / "spatial_synthesis/GWR_MGWR_LARGEST_LOCAL_DIFFERENCES.csv")
    return {"weighted": weighted, "local_summary": spatial_summary, "fit": model,
            "bandwidth": bandwidth, "agreement": pd.DataFrame(agreement), "overlap": overlap,
            "environment_associations": environment_associations, "environment_groups": environment_groups,
            "environment_contrasts": environment_contrasts, "largest_differences": largest_differences,
            "gwr_lcr_agreement": gwr_lcr_agreement,
            "beta": beta, "mgwr_local": mgwr_local}


def build_figure_a(run: Path) -> None:
    nine = source_paths()["nine"]
    global_assoc = pd.read_csv(nine / "tables/corrected_i1_global_associations.csv")
    pair_assoc = pd.read_csv(nine / "tables/corrected_i2_matched_pair_associations.csv")
    gam_path = ROOT / "docs/reviews/2026-10-31/phase-i/figures/i1_I08_full_sample_gam_curves_data.csv"
    zero_path = ROOT / "docs/reviews/2026-10-31/phase-i/tables/i1_zero_positive_effects.csv"
    full = global_assoc.loc[global_assoc.exposure.eq("Aa_20yr") & global_assoc.population.eq("mask_all")].set_index("outcome")
    pairs = pair_assoc.loc[(pair_assoc.exposure.eq("Aa_20yr")) & pair_assoc.candidate_id.eq("stage2_Aa_20yr_01") &
                           pair_assoc.population.eq("all_erosion")].set_index("outcome")
    zero = pd.read_csv(zero_path).query("population == 'mask_all' and exposure == 'Aa_20yr'").set_index("outcome")
    gwr = pd.read_csv(nine / "tables/gwr_weighted_effect_summary.csv").query("model_family == 'GWR'").set_index("outcome")
    frame = pd.DataFrame(index=OUTCOME_ORDER)
    frame["label"] = [LABELS[x] for x in OUTCOME_ORDER]
    for col in ["spearman_rho", "standardized_beta_log1p_Aa", "block_bootstrap_beta_ci_low", "block_bootstrap_beta_ci_high", "n_complete"]:
        frame[col] = full.reindex(OUTCOME_ORDER)[col].to_numpy()
    frame["matched_pair_spearman_rho"] = pairs.reindex(OUTCOME_ORDER).spearman_rho.to_numpy()
    frame["gwr_weighted_local_beta"] = gwr.reindex(OUTCOME_ORDER).pixel_weighted_mean_local_standardized_beta.to_numpy()
    # Keep explicit aliases used by the report and crosswalk tables.
    frame["full_spearman_rho"] = frame.spearman_rho
    frame["m4_pair_spearman_rho"] = frame.matched_pair_spearman_rho
    for col in ["standardized_mean_difference_hedges_g", "mean_difference_positive_minus_zero",
                "block_bootstrap_difference_ci_low", "block_bootstrap_difference_ci_high", "n_zero", "n_positive"]:
        frame[col] = zero.reindex(OUTCOME_ORDER)[col].to_numpy()
    frame["outcome_guided_selection"] = True
    csv_write(frame.reset_index(names="outcome"), run / "figures/data/figure_A_global_zero_pair_metrics.csv")
    csv_write(frame.reset_index(names="outcome"), run / "figures/data/figure_A_multimethod_response.csv")
    gam = pd.read_csv(gam_path)
    gam = gam.loc[gam.exposure.eq("Aa_20yr") & gam.outcome.isin(["SHI_candidate", "SW", "OM"])].copy()
    csv_write(gam, run / "figures/data/figure_A_gam_curves.csv")

    fig = plt.figure(figsize=(17, 15), constrained_layout=True)
    gs = fig.add_gridspec(3, 6, height_ratios=[1.05, .9, 1.25])
    ax_global = fig.add_subplot(gs[0, :3])
    ax_zero = fig.add_subplot(gs[0, 3:])
    ax_pairs = fig.add_subplot(gs[1, :])
    curve_axes = [fig.add_subplot(gs[2, 0:2]), fig.add_subplot(gs[2, 2:4]), fig.add_subplot(gs[2, 4:6])]
    yy = np.arange(len(OUTCOME_ORDER))
    ax_global.axvline(0, color="#777", lw=.8)
    ax_global.errorbar(frame.standardized_beta_log1p_Aa, yy,
                       xerr=[frame.standardized_beta_log1p_Aa - frame.block_bootstrap_beta_ci_low,
                             frame.block_bootstrap_beta_ci_high - frame.standardized_beta_log1p_Aa],
                       fmt="D", ms=4, color="#bd6849", ecolor="#d8a28f", capsize=2,
                       label="标准化斜率 β（5 km空间块95%区间）", zorder=3)
    ax_global.scatter(frame.spearman_rho, yy, marker="o", s=24, color="#28769c", label="Spearman秩相关 ρ", zorder=4)
    ax_global.set_xlim(-.72, .45); ax_global.set_yticks(yy, frame.label, fontsize=8); ax_global.invert_yaxis()
    ax_global.set_xlabel("全域像元关联（负值表示侵蚀较高处指标较低）")
    ax_global.set_title("A｜全域关联与区间估计", loc="left", fontweight="bold")
    ax_global.legend(frameon=False, fontsize=8, loc="lower right"); ax_global.grid(axis="x", alpha=.18)
    ax_global.text(.02, .02, "共同有效像元约38.5万；预测栅格像元存在空间依赖。CEC与Aa有公式耦合。",
                   transform=ax_global.transAxes, fontsize=7.5, color="#555")

    ax_zero.axvline(0, color="#777", lw=.8)
    ax_zero.scatter(frame.standardized_mean_difference_hedges_g, yy, marker="o", s=30, color="#56876a", zorder=3)
    ax_zero.set_yticks(yy, frame.label, fontsize=8); ax_zero.invert_yaxis(); ax_zero.set_xlim(-1.3, .5)
    ax_zero.set_xlabel("正侵蚀像元均值减去零侵蚀像元均值（Hedges g，标准化）")
    ax_zero.set_title("B｜零侵蚀与正侵蚀像元差异", loc="left", fontweight="bold")
    ax_zero.grid(axis="x", alpha=.18)
    for x, y, n0, n1 in zip(frame.standardized_mean_difference_hedges_g, yy, frame.n_zero, frame.n_positive):
        if np.isfinite(x): ax_zero.annotate(f"{x:+.2f}", (x, y), xytext=(5, 0), textcoords="offset points", va="center", fontsize=7)
    ax_zero.text(.02, .02, "点为标准化均值差；原始均值差的空间块95%区间见配套CSV。", transform=ax_zero.transAxes,
                 fontsize=7.5, color="#555")

    ax_pairs.axvline(0, color="#777", lw=.8)
    pvals = frame.matched_pair_spearman_rho.to_numpy(dtype=float)
    ax_pairs.scatter(pvals, yy, s=34, color="#84679c", zorder=3)
    for x, y in zip(pvals, yy):
        if np.isfinite(x): ax_pairs.annotate(f"{x:+.2f}", (x, y), xytext=(5, 0), textcoords="offset points", va="center", fontsize=7)
    ax_pairs.set_yticks(yy, frame.label, fontsize=8); ax_pairs.invert_yaxis(); ax_pairs.set_xlim(-.42, .42)
    ax_pairs.set_xlabel("环境匹配配对差值的 Spearman ρ（111,289 对候选配对）")
    ax_pairs.set_title("C｜匹配配对中的侵蚀差值与健康差值", loc="left", fontweight="bold")
    ax_pairs.grid(axis="x", alpha=.18)

    gam_labels = {"SHI_candidate": "土壤健康指数（SHI）", "SW": "土壤含水量（SW）", "OM": "有机质（OM）"}
    for ax, outcome in zip(curve_axes, ["SHI_candidate", "SW", "OM"]):
        d = gam.loc[gam.outcome.eq(outcome)].sort_values("log1p_exposure")
        x = d.log1p_exposure.to_numpy(dtype=float); y = d.predicted_outcome.to_numpy(dtype=float)
        lo = d.pointwise_ci95_low.to_numpy(dtype=float); hi = d.pointwise_ci95_high.to_numpy(dtype=float)
        ax.fill_between(x, lo, hi, color="#75a9c3", alpha=.22, linewidth=0)
        ax.plot(x, y, color="#286f91", lw=2)
        ax.set_title(gam_labels[outcome], fontsize=10)
        ax.set_xlabel("log(1 + 20年平均实际侵蚀量)")
        ax.set_ylabel("预测指标值")
        ax.grid(alpha=.17)
    fig.suptitle("侵蚀与土壤健康：全域关联、零侵蚀差异、匹配配对和非线性响应", fontsize=16)
    fig.text(.5, -.005, "各面板使用不同统计单位和模型；曲线是全样本Gaussian GAM的平滑预测及点态95%区间。图中关系为探索性关联，不作因果解释。",
             ha="center", fontsize=9)
    save_figure(fig, run / "figures", "A_multimethod_erosion_health_response")


def build_figure_b(run: Path, spatial: dict[str, pd.DataFrame]) -> None:
    mgwr = spatial["mgwr_local"]
    outcomes = ["SHI_candidate", "OM", "SW"]
    families = ["standard_GWR_same_sample", "standard_MGWR_spatially_balanced_sample"]
    fam_labels = ["同样本标准GWR", "标准MGWR"]
    cmap = "RdBu_r"
    bandwidth = spatial["bandwidth"]
    fit_summary = spatial["fit"]
    fit_path = source_paths()["mgwr"] / "tables/mgwr_model_summary.csv"
    gwr_model = pd.read_csv(fit_path)
    fig, axes = plt.subplots(3, 2, figsize=(13, 15), constrained_layout=True, sharex=True, sharey=True)
    ext = mgwr.loc[mgwr.outcome.isin(outcomes), ["x", "y"]].agg(["min", "max"])
    xmin, xmax = float(ext.loc["min", "x"] / 1000), float(ext.loc["max", "x"] / 1000)
    ymin, ymax = float(ext.loc["min", "y"] / 1000), float(ext.loc["max", "y"] / 1000)
    xpad, ypad = (xmax - xmin) * .04, (ymax - ymin) * .04
    extent = (xmin - xpad, xmax + xpad, ymin - ypad, ymax + ypad)
    saved = []
    for i, outcome in enumerate(outcomes):
        series = mgwr.loc[mgwr.outcome.eq(outcome) & mgwr.model_family.isin(families), "local_beta_log1p_Aa_standardized"].to_numpy(dtype=float)
        finite = np.abs(series[np.isfinite(series)])
        limit = float(np.quantile(finite, .98)) if finite.size else 1.0
        limit = max(limit, .05)
        norm = mcolors.TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
        bwrow = bandwidth.loc[bandwidth.outcome.eq(outcome) & bandwidth.variable.eq("log1p_Aa")]
        mgwr_bw = float(bwrow.bandwidth_neighbors.iloc[0]) if len(bwrow) else np.nan
        mgwr_m = float(bwrow.median_neighbor_distance_m.iloc[0]) if len(bwrow) else np.nan
        gwr_row = gwr_model.loc[gwr_model.outcome.eq(outcome) & gwr_model.model_family.eq("standard_GWR_same_sample")]
        gwr_bw = float(gwr_row.bandwidth_neighbors.iloc[0]) if len(gwr_row) and "bandwidth_neighbors" in gwr_row else np.nan
        for j, family in enumerate(families):
            d = mgwr.loc[mgwr.outcome.eq(outcome) & mgwr.model_family.eq(family)]
            sc = axes[i, j].scatter(d.x / 1000, d.y / 1000, c=d.local_beta_log1p_Aa_standardized,
                                    s=10, cmap=cmap, norm=norm, linewidths=0, rasterized=True)
            saved.append((i, j, sc))
            if family == "standard_MGWR_spatially_balanced_sample":
                title = f"{LABELS[outcome]}｜{fam_labels[j]}\nAa带宽：{mgwr_bw:.0f}近邻（中位距离 {mgwr_m/1000:.1f} km）"
            else:
                title = f"{LABELS[outcome]}｜{fam_labels[j]}\nAa带宽：{gwr_bw:.0f}近邻"
            axes[i, j].set_title(title, fontsize=10)
            axes[i, j].set_xlim(extent[0], extent[1]); axes[i, j].set_ylim(extent[2], extent[3])
            axes[i, j].set_aspect("equal", adjustable="box"); axes[i, j].grid(alpha=.12)
            if i == 2: axes[i, j].set_xlabel("东向坐标（km）")
            if j == 0: axes[i, j].set_ylabel("北向坐标（km）")
            # A common projected coordinate extent makes the scale-bar length identical in every panel.
            x0, x1 = extent[0] + .05 * (extent[1] - extent[0]), extent[0] + .20 * (extent[1] - extent[0])
            y0 = extent[2] + .055 * (extent[3] - extent[2])
            axes[i, j].plot([x0, x1], [y0, y0], color="#222", lw=2.2, solid_capstyle="butt")
            axes[i, j].text((x0+x1)/2, y0 + .014*(extent[3]-extent[2]), f"{(x1-x0)*1000:.0f} m",
                            ha="center", va="bottom", fontsize=7, color="#222")
            if i == 0 and j == 1:
                axes[i, j].annotate("N", xy=(.93, .89), xytext=(.93, .75), xycoords="axes fraction",
                                    textcoords="axes fraction", ha="center", va="center", fontsize=10,
                                    arrowprops=dict(arrowstyle="-|>", color="#222", lw=1.2))
        cb = fig.colorbar(saved[-1][2], ax=axes[i, :], shrink=.85, pad=.025, fraction=.035)
        cb.set_label(f"局部标准化β（色阶±{limit:.2f}；同一行共用）", fontsize=8)
    fig.suptitle("SHI、有机质和土壤含水量的GWR与MGWR局部侵蚀系数", fontsize=16)
    fig.text(.5, -.005, "每行两图使用相同、以0为中心的颜色范围；全部面板共用投影坐标和比例尺。770个空间均衡单元，不代表30 m像元级独立检验。",
             ha="center", fontsize=9)
    save_figure(fig, run / "figures", "B_gwr_mgwr_spatial_patterns")


def build_figure_c(run: Path) -> None:
    i8 = source_paths()["i8"]
    units = pd.read_parquet(i8 / "tables/stress_zones_240m.parquet")
    summary = pd.read_csv(i8 / "tables/stress_zone_summary.csv").set_index("zone")
    zones = ["T1 高Aa—强负向响应", "T2 高Aa—弱负向响应", "T3 低Aa—低SHI", "T4 环境敏感候选", "其他/未分类"]
    zone_labels = {zones[0]: "高侵蚀—强负向关联", zones[1]: "高侵蚀—弱负向关联",
                   zones[2]: "低侵蚀—健康指数较低", zones[3]: "环境敏感候选", zones[4]: "其他/未分类"}
    zone_colors = {zones[0]: "#b4473e", zones[1]: "#e2964f", zones[2]: "#5b81ad", zones[3]: "#7c65a8", zones[4]: "#c7c9cc"}
    envs = ["Elevation", "TWI", "LST", "NDVI", "Precipitation"]
    labels = ["高程", "地形湿度指数", "地表温度", "NDVI", "降水"]
    profile_zones = zones[:4]
    soil_codes = [26, 27, 42]
    profile_rows = []
    for zone in profile_zones:
        row = units.loc[units.primary_zone.eq(zone)]
        q = summary.loc[zone]
        soilshares = {str(code): float(row.dominant_soil_type.eq(code).mean()) for code in soil_codes}
        profile_rows.append({"zone": zone, "zone_code": zone[:2], "n_240m_units": int(q.spatial_units_240m),
                             "area_km2_equivalent": float(q.area_km2_equivalent), "mean_Aa_20yr": float(q.mean_Aa_20yr),
                             "mean_SHI": float(q.mean_SHI), "median_GWR_beta": float(q.median_GWR_beta),
                             **{f"soil_{code}_dominant_share": share for code, share in soilshares.items()},
                             **{f"{var}_mean": float(row[var].mean()) for var in envs},
                             "outcome_guided_selection": True})
    profile_frame = pd.DataFrame(profile_rows)
    fig = plt.figure(figsize=(17, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.18, 1], height_ratios=[1, .9])
    axmap = fig.add_subplot(gs[:, 0])
    for zone in zones:
        d = units.loc[units.primary_zone.eq(zone)]
        if d.empty: continue
        axmap.scatter(d.x / 1000, d.y / 1000, s=2.2, color=zone_colors[zone], label=zone_labels[zone], linewidths=0, rasterized=True)
    overlay_specs = [("27潮土·TWI q2 proxy", "^", "#181818"), ("26冲积土·TWI q1 proxy", "x", "#ffdf43")]
    for value, marker, color in overlay_specs:
        d = units.loc[units.environment_overlay.eq(value)]
        if not d.empty:
            if marker == "^":
                axmap.scatter(d.x / 1000, d.y / 1000, s=11, marker=marker, color=color, edgecolors="#333",
                              linewidths=.4, label=value, rasterized=True)
            else:
                axmap.scatter(d.x / 1000, d.y / 1000, s=11, marker=marker, color="#c6a629",
                              linewidths=.65, label=value, rasterized=True)
    axmap.set_title("240 m候选压力分区与环境覆盖标记")
    axmap.set_xlabel("东向坐标（km）"); axmap.set_ylabel("北向坐标（km）")
    axmap.set_aspect("equal", adjustable="datalim"); axmap.legend(loc="best", fontsize=8, frameon=True, markerscale=3)
    heat_gs = gs[0, 1].subgridspec(1, 2, width_ratios=[5, 3], wspace=.35)
    axheat = fig.add_subplot(heat_gs[0, 0])
    axsoil = fig.add_subplot(heat_gs[0, 1], sharey=axheat)
    envmat = np.asarray([[(units.loc[units.primary_zone.eq(z), v].mean() - units[v].mean()) / units[v].std(ddof=1)
                          for v in envs] for z in profile_zones])
    soilmat = profile_frame[[f"soil_{c}_dominant_share" for c in soil_codes]].to_numpy(dtype=float)
    im = axheat.imshow(envmat, aspect="auto", cmap="RdBu_r", vmin=-1.5, vmax=1.5)
    axheat.set_xticks(range(len(envs)), labels, rotation=25, ha="right")
    axheat.set_yticks(range(4), ["T1 高Aa—强负向", "T2 高Aa—较弱负向", "T3 低Aa—低SHI", "T4 环境敏感候选"])
    axheat.set_title("T1–T4环境均值相对全区差异（标准差）")
    for ii in range(4):
        for jj in range(len(envs)):
            axheat.text(jj, ii, f"{envmat[ii,jj]:+.2f}", ha="center", va="center", fontsize=7)
    axsoil.imshow(soilmat, aspect="auto", cmap="YlGn", vmin=0, vmax=1)
    axsoil.set_xticks(range(3), ["26", "27", "42"])
    axsoil.set_title("优势土类占比")
    axsoil.tick_params(axis="y", left=False, labelleft=False)
    for ii in range(4):
        for jj in range(3):
            axsoil.text(jj, ii, f"{soilmat[ii,jj]*100:.0f}%", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=axheat, shrink=.85, label="相对全域均值的标准差")
    axbar = fig.add_subplot(gs[1, 1])
    picked = [summary.loc[z] for z in profile_zones]
    xx = np.arange(4)
    area = [v.area_km2_equivalent for v in picked]
    beta = [v.median_GWR_beta for v in picked]
    axbar.bar(xx, area, color=[zone_colors[z] for z in profile_zones], alpha=.8)
    zone_n = [int(v.spatial_units_240m) for v in picked]
    axbar.set_xticks(xx, [f"T{i+1}\n{zone_n[i]}单元" for i in range(4)])
    axbar.set_ylim(0, max(area) * 1.22)
    axbar.set_ylabel("完整案例面积等价（km²）")
    axbar.set_title("T1–T4面积及局部系数中位数")
    ax2 = axbar.twinx(); ax2.plot(xx, beta, color="#252525", marker="o", lw=1.6)
    ax2.set_ylim(min(beta) - .08, max(beta) + .08)
    ax2.set_ylabel("GWR局部β中位数")
    for i, value in enumerate(area):
        axbar.text(i, value * .13, f"{value:.1f} km²", ha="center", va="bottom", fontsize=8,
                   color="white", fontweight="semibold")
        ax2.annotate(f"β={beta[i]:.2f}", (i, beta[i]), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8)
    fig.suptitle("侵蚀压力候选区及环境画像：四类候选逐一比较", fontsize=16)
    fig.text(.5, -.005, "地图为240 m汇总单元；T1–T4分区彼此互斥，环境覆盖标记为可重叠代理条件。候选筛选参考健康结果，仅用于探索与现场核验排序，不是经验证的风险或治理分区。",
             ha="center", fontsize=9)
    save_figure(fig, run / "figures", "C_stress_sensitive_environment_zones")
    zone_csv = pd.read_csv(i8 / "tables/stress_zones_240m.csv")
    zone_csv["outcome_guided_selection"] = True
    csv_write(zone_csv, run / "figures/data/figure_C_240m_stress_zones.csv")
    csv_write(profile_frame, run / "figures/data/figure_C_environment_profiles.csv")
    overlay_rows = []
    for value, _, _ in overlay_specs:
        d = units.loc[units.environment_overlay.eq(value)]
        overlay_rows.append({"environment_overlay": value, "n_240m_units": int(len(d)),
                             "area_km2_equivalent": float(d.n_pixels.sum() * 0.0009) if "n_pixels" in d else np.nan,
                             "mean_Aa_20yr": float(d.Aa_20yr.mean()) if len(d) else np.nan,
                             "mean_SHI": float(d.SHI_candidate.mean()) if len(d) else np.nan,
                             "outcome_guided_selection": True})
    csv_write(pd.DataFrame(overlay_rows), run / "figures/data/figure_C_environment_overlays.csv")


def build_sem_products(run: Path) -> dict[str, pd.DataFrame]:
    results = run / "results"
    paths = pd.read_csv(results / "SEM_PATH_COEFFICIENTS.csv")
    fits = pd.read_csv(results / "SEM_MODEL_FIT_COMPARISON.csv")
    delta = pd.read_csv(results / "SEM_EROSION_INCREMENTAL_CONTRIBUTION.csv")
    effects = pd.read_csv(results / "SEM_DIRECT_INDIRECT_EFFECTS.csv")
    # Retain every fitted D7 path/control cell; figure I selects the same M1/S1
    # slice for a compact display, while this table preserves the full search.
    grp = paths.loc[paths.dataset.str.startswith("D7_soil") & paths.method.eq("piecewise_sem")].copy()
    parsed = grp.dataset.str.extract(r"D7_soil(?P<soil>\d+)_(?P<twi>q\d+)")
    grp["soil_code"] = parsed.soil
    grp["soil_label"] = grp.soil_code.map(SOIL_LABELS)
    grp["twi_group"] = parsed.twi
    grp["dimension"] = grp["to"].map({"physical": "物理维度", "chemical": "化学维度", "biological_activity": "生物活动", "z_SHI": "SHI"})
    grp["outcome_guided_selection"] = True
    csv_write(grp, results / "SEM_ENVIRONMENT_GROUP_COMPARISON.csv")
    # Rank the four control structures separately inside each of the four primary populations.
    # The fit, conditional ΔR², and direct path all come from the same piecewise S2 SHI model.
    primary_populations = ["D1_common_env_case", "D2_240m_blocks", "D3_M4_unique_pixels", "D4_M4_pair_differences"]
    rankings = []
    for dataset in primary_populations:
        population_fits = fits.loc[fits.dataset.eq(dataset) & fits.method.eq("piecewise_sem") &
                                   fits.structure.eq("S2_direct_plus_indirect_SHI") &
                                   fits.endogenous.eq("z_SHI") & fits.status.eq("ok")].copy()
        population_delta = delta.loc[delta.dataset.eq(dataset) & delta.method.eq("piecewise_sem") &
                                     delta.structure.eq("S2_direct_plus_indirect_SHI") & delta.outcome.eq("z_SHI")]
        population_paths = paths.loc[paths.dataset.eq(dataset) & paths.method.eq("piecewise_sem") &
                                     paths.structure.eq("S2_direct_plus_indirect_SHI") &
                                     paths["from"].eq("z_E") & paths["to"].eq("z_SHI")]
        for row in population_fits.itertuples():
            ddelta = population_delta.loc[population_delta.control.eq(row.control)]
            path = population_paths.loc[population_paths.control.eq(row.control)]
            dr = float(ddelta.delta_r2_erosion_conditional.iloc[0]) if len(ddelta) else np.nan
            direct_beta = float(path.beta_standardized.iloc[0]) if len(path) else np.nan
            rankings.append({"dataset": row.dataset, "structure": row.structure, "control": row.control,
                             "n": row.n, "SHI_R2": row.r_squared, "direct_erosion_beta_to_SHI": direct_beta,
                             "direct_erosion_delta_R2": dr,
                             "selection_basis": "within-population S2 piecewise SHI comparison; apparent, outcome-guided",
                             "outcome_guided_selection": True})
    rank = pd.DataFrame(rankings)
    if not rank.empty:
        rank["rank_R2"] = rank.groupby("dataset").SHI_R2.rank(ascending=False, method="min")
        rank["rank_delta_R2"] = rank.groupby("dataset").direct_erosion_delta_R2.rank(ascending=False, method="min")
        rank["rank_negative_beta"] = rank.groupby("dataset").direct_erosion_beta_to_SHI.rank(ascending=True, method="min")
        rank["combined_rank"] = rank[["rank_R2", "rank_delta_R2", "rank_negative_beta"]].mean(axis=1)
        rank["rank_interpretability"] = rank.control.map({"M0": 2, "M1": 1, "M2": 3, "M3": 4})
        rank["best_overall_fit"] = rank.SHI_R2.eq(rank.groupby("dataset").SHI_R2.transform("max"))
        rank["largest_erosion_increment"] = rank.direct_erosion_delta_R2.eq(rank.groupby("dataset").direct_erosion_delta_R2.transform("max"))
        rank["strongest_negative_direct_path"] = rank.direct_erosion_beta_to_SHI.eq(rank.groupby("dataset").direct_erosion_beta_to_SHI.transform("min"))
        rank["best_scientific_interpretation_candidate"] = rank.control.eq("M1")
        rank = rank.sort_values(["dataset", "combined_rank", "control"])
    csv_write(rank, results / "SEM_MODEL_RANKINGS.csv")
    status = fits.groupby(["method", "status"], dropna=False).size().rename("n_models").reset_index()
    csv_write(status, results / "SEM_EXECUTION_STATUS_COUNTS.csv")
    return {"paths": paths, "fits": fits, "delta": delta, "effects": effects, "groups": grp, "rank": rank, "status": status}


def plot_sem_d(run: Path, sem: dict[str, pd.DataFrame]) -> None:
    p, f = sem["paths"], sem["fits"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    controls = ["M0", "M1", "M2", "M3"]
    colors = {"物理维度": "#397a9d", "化学维度": "#c8873a", "生物活动": "#4a8b66", "SHI": "#6b5a92"}
    for ax, control in zip(axes.flat, controls):
        select = p.loc[p.dataset.eq("D1_common_env_case") & p.method.eq("piecewise_sem") & p.control.eq(control) &
                       ((p.structure.eq("S1_parallel_dimensions") & p["from"].eq("z_E")) |
                        (p.structure.eq("S2_direct_plus_indirect_SHI") & p["from"].eq("z_E") & p["to"].eq("z_SHI")))]
        mapping = {"physical": "物理维度", "chemical": "化学维度", "biological_activity": "生物活动", "z_SHI": "SHI"}
        for row in select.itertuples():
            key = mapping.get(row.to)
            if key: ax.barh(key, row.beta_standardized, color=colors[key], alpha=.85)
        fit = f.loc[f.dataset.eq("D1_common_env_case") & f.method.eq("piecewise_sem") &
                     f.structure.eq("S2_direct_plus_indirect_SHI") & f.control.eq(control) & f.endogenous.eq("z_SHI")]
        r2 = float(fit.r_squared.iloc[0]) if len(fit) else np.nan
        control_label = {"M0": "未加环境控制", "M1": "核心环境控制", "M2": "环境主成分控制", "M3": "核心环境＋空间趋势"}[control]
        ax.axvline(0,color="#555",lw=.8); ax.set_title(f"{control_label}｜SHI R²={r2:.3f}" if np.isfinite(r2) else control_label)
        ax.set_xlim(-0.5, 0.0)
        ax.set_xlabel("标准化侵蚀路径系数"); ax.grid(axis="x",alpha=.2)
    fig.suptitle("图D｜环境控制层级变化时的侵蚀路径", fontsize=15)
    save_figure(fig, run / "figures", "D_sem_control_hierarchy")


def plot_sem_e(run: Path, sem: dict[str, pd.DataFrame]) -> None:
    p = sem["paths"]
    groups = [("D1_common_env_case", "全域环境变量完整像元"), ("D3_M4_unique_pixels", "既有匹配方案的端点像元")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, constrained_layout=True)
    mapping = {"physical": "侵蚀→物理维度", "chemical": "侵蚀→化学维度", "biological_activity": "侵蚀→生物活动",
               "z_SHI": "侵蚀→SHI直接路径"}
    for ax,(dataset,title) in zip(axes,groups):
        q=p.loc[p.dataset.eq(dataset)&p.method.eq("piecewise_sem")&p.control.eq("M1")&
                ((p.structure.eq("S1_parallel_dimensions")&p["from"].eq("z_E"))|
                 (p.structure.eq("S2_direct_plus_indirect_SHI")&p["from"].eq("z_E")&p["to"].eq("z_SHI")))].copy()
        q["label"] = q["to"].map(mapping)
        for i,row in enumerate(q.itertuples()):
            ax.errorbar(row.beta_standardized, i, xerr=[[row.beta_standardized-row.ci_low],[row.ci_high-row.beta_standardized]] if np.isfinite(row.ci_low) and np.isfinite(row.ci_high) else None,
                        fmt="o", color="#2b6f9c", capsize=2)
        ax.set_yticks(range(len(q)),q.label); ax.axvline(0,color="#555",lw=.8); ax.grid(axis="x",alpha=.2)
        ax.set_title(title); ax.set_xlabel("标准化路径系数（核心环境控制）")
    fig.suptitle("图E｜全样本与环境匹配端点的路径对照",fontsize=15)
    save_figure(fig,run/"figures","E_full_vs_matched_sem")


def plot_sem_f(run: Path, sem: dict[str, pd.DataFrame]) -> None:
    p=sem["paths"]
    dsets=[("D1_common_env_case","全域像元"),("D2_240m_blocks","240 m汇总单元"),("D3_M4_unique_pixels","既有匹配端点像元"),("D4_M4_pair_differences","另一匹配方案的配对差值")]
    dims=[("physical","物理维度"),("chemical","化学维度"),("biological_activity","生物活动")]
    fig,axes=plt.subplots(1,3,figsize=(15,7),sharey=True,constrained_layout=True)
    colors=plt.cm.tab10(np.linspace(0,1,len(dsets)))
    for ax,(dim,label) in zip(axes,dims):
        for j,(dataset,dlabel) in enumerate(dsets):
            q=p.loc[p.dataset.eq(dataset)&p.method.eq("piecewise_sem")&p.structure.eq("S1_parallel_dimensions")&
                    p.control.eq("M1")&p["from"].eq("z_E")&p["to"].eq(dim)]
            if q.empty: continue
            row=q.iloc[0]; yy=j
            low=row.ci_low if np.isfinite(row.ci_low) else np.nan
            high=row.ci_high if np.isfinite(row.ci_high) else np.nan
            ax.errorbar(row.beta_standardized,yy,xerr=[[row.beta_standardized-low],[high-row.beta_standardized]] if np.isfinite(low) and np.isfinite(high) else None,
                        fmt="o",color=colors[j],label=dlabel,capsize=2)
        ax.axvline(0,color="#555",lw=.8);ax.set_title(label);ax.set_xlabel("标准化系数（核心环境控制）");ax.grid(axis="x",alpha=.2)
    axes[0].set_yticks(range(len(dsets)),[x[1] for x in dsets]);axes[0].legend(loc="best",fontsize=8)
    fig.suptitle("图F｜侵蚀与物理、化学、生物维度的路径比较",fontsize=15)
    save_figure(fig,run/"figures","F_erosion_dimension_path_forest")


def plot_sem_g(run: Path, sem: dict[str, pd.DataFrame]) -> None:
    p,f,d=sem["paths"],sem["fits"],sem["delta"]
    controls=["M0","M1","M2","M3"]
    rows=[]
    for c in controls:
        fit=f.loc[f.dataset.eq("D1_common_env_case")&f.method.eq("piecewise_sem")&f.structure.eq("S2_direct_plus_indirect_SHI")&f.control.eq(c)&f.endogenous.eq("z_SHI")]
        delta=d.loc[d.dataset.eq("D1_common_env_case")&d.structure.eq("S2_direct_plus_indirect_SHI")&d.control.eq(c)&d.outcome.eq("z_SHI")]
        path=p.loc[p.dataset.eq("D1_common_env_case")&p.method.eq("piecewise_sem")&p.structure.eq("S2_direct_plus_indirect_SHI")&p.control.eq(c)&p["from"].eq("z_E")&p["to"].eq("z_SHI")]
        rows.append({"control":c,"SHI_R2":float(fit.r_squared.iloc[0]) if len(fit) else np.nan,
                     "erosion_delta_R2":float(delta.delta_r2_erosion_conditional.iloc[0]) if len(delta) else np.nan,
                     "direct_erosion_beta":float(path.beta_standardized.iloc[0]) if len(path) else np.nan})
    out=pd.DataFrame(rows);out["outcome_guided_selection"]=True;csv_write(out,run/"figures/data/figure_G_sem_fit_delta.csv")
    fig,axes=plt.subplots(1,3,figsize=(13,4.8),constrained_layout=True)
    for ax,col,title,color in zip(axes,["SHI_R2","erosion_delta_R2","direct_erosion_beta"],
                                  ["SHI模型R²","侵蚀条件增量ΔR²","侵蚀→SHI直接路径β"],
                                  ["#5584a2","#d38a42","#687a4e"]):
        ax.bar(out.control,out[col],color=color);ax.axhline(0,color="#555",lw=.8);ax.set_title(title);ax.set_xlabel("环境/空间控制层级")
        ax.set_xticks(range(len(controls)), ["未加入\n环境变量", "核心环境\n变量", "环境\n主成分", "核心环境＋\n空间位置趋势"])
        for i,v in enumerate(out[col]):
            if np.isfinite(v):ax.text(i,v,f"{v:.3f}",ha="center",va="bottom" if v>=0 else "top",fontsize=8)
        ax.grid(axis="y",alpha=.2)
    fig.suptitle("图G｜整体拟合、侵蚀增量贡献与直接路径",fontsize=15)
    save_figure(fig,run/"figures","G_sem_fit_and_erosion_increment")


def plot_sem_h(run: Path, sem: dict[str, pd.DataFrame]) -> None:
    p=sem["paths"]
    dataset,control="D1_common_env_case","M1"
    links=[("z_E","physical","S1_parallel_dimensions"),("z_E","chemical","S1_parallel_dimensions"),
           ("z_E","biological_activity","S1_parallel_dimensions"),("physical","z_SHI","S2_direct_plus_indirect_SHI"),
           ("chemical","z_SHI","S2_direct_plus_indirect_SHI"),("biological_activity","z_SHI","S2_direct_plus_indirect_SHI"),
           ("z_E","z_SHI","S2_direct_plus_indirect_SHI")]
    nodes={"z_E":(.12,.52,"20年实际侵蚀量\n（对数尺度）","#b65a49"),"physical":(.43,.83,"物理维度\n含水量、容重","#4384a1"),
           "chemical":(.43,.52,"化学维度\n有机质、碱解氮","#c18745"),"biological_activity":(.43,.21,"生物活动\n微生物量碳、土壤呼吸","#568b69"),
           "z_SHI":(.83,.52,"土壤健康指数\nSHI（暂定）","#78679d")}
    fig,ax=plt.subplots(figsize=(12,6.7));ax.axis("off")
    for name,(x,y,label,color) in nodes.items():
        ax.text(x,y,label,ha="center",va="center",transform=ax.transAxes,fontsize=11,
                bbox=dict(boxstyle="round,pad=.55",fc=color,ec="white",alpha=.95),color="white")
    edge_geometry={
        ("z_E","physical"):((.20,.55),(.35,.81),0.0,(.27,.69)),
        ("z_E","chemical"):((.20,.52),(.35,.52),0.0,(.28,.56)),
        ("z_E","biological_activity"):((.20,.49),(.35,.23),0.0,(.27,.34)),
        ("physical","z_SHI"):((.51,.81),(.75,.57),0.0,(.64,.70)),
        ("chemical","z_SHI"):((.52,.52),(.75,.52),0.0,(.64,.55)),
        ("biological_activity","z_SHI"):((.51,.23),(.75,.48),0.0,(.64,.34)),
        ("z_E","z_SHI"):((.20,.56),(.75,.56),-.30,(.48,.67)),
    }
    for source,target,structure in links:
        q=p.loc[p.dataset.eq(dataset)&p.method.eq("piecewise_sem")&p.structure.eq(structure)&p.control.eq(control)&p["from"].eq(source)&p["to"].eq(target)]
        if q.empty:continue
        beta=float(q.beta_standardized.iloc[0])
        start,end,rad,label_pos=edge_geometry[(source,target)]
        color="#27805c" if beta<0 else "#ae6948"
        ax.annotate("",xy=end,xytext=start,xycoords=ax.transAxes,textcoords=ax.transAxes,
                    arrowprops=dict(arrowstyle="->",lw=1.4+min(abs(beta),.7)*3,color=color,connectionstyle=f"arc3,rad={rad}"))
        ax.text(label_pos[0],label_pos[1],f"β={beta:+.2f}",transform=ax.transAxes,
                color=color,fontsize=9,ha="center",va="center",bbox=dict(fc="white",ec="none",alpha=.8,pad=1))
    ax.text(.5,.98,"已控制高程、地形湿度指数、植被指数、地表温度和土壤类型",transform=ax.transAxes,ha="center",va="top",fontsize=10,color="#666")
    ax.text(.5,.04,"图H｜全域环境变量完整像元中的条件关联路径；箭头不代表已证实的因果作用，SHI构建公式仍待确认。",transform=ax.transAxes,ha="center",fontsize=9)
    save_figure(fig,run/"figures","H_best_sem_path_diagram")


def plot_sem_i(run: Path, sem: dict[str, pd.DataFrame]) -> None:
    g=sem["groups"]
    g=g.loc[g.control.eq("M1")&g.structure.eq("S1_parallel_dimensions")&g["from"].eq("z_E")&
            g.to.isin(["physical","chemical","biological_activity"])].copy()
    if g.empty:return
    group_labels={"q1":"湿度最低组","q2":"湿度较低组","q3":"湿度较高组","q4":"湿度最高组"}
    g["group"] = g.soil_label+"·"+g.twi_group.map(group_labels)
    pivot=g.pivot_table(index="group",columns="dimension",values="beta_standardized",aggfunc="first")
    order=[f"{SOIL_LABELS[s]}·{group_labels[f'q{q}']}" for s in ["26","27","42"] for q in range(1,5)]
    pivot=pivot.reindex(order)
    csv_write(g,run/"figures/data/figure_I_soil_twi_sem_slopes.csv")
    fig,ax=plt.subplots(figsize=(10,7),constrained_layout=True)
    vals=pivot.to_numpy(dtype=float);im=ax.imshow(vals,aspect="auto",cmap="RdBu_r",norm=mcolors.TwoSlopeNorm(vmin=-.8,vcenter=0,vmax=.8))
    ax.set_yticks(range(len(pivot.index)),pivot.index);ax.set_xticks(range(len(pivot.columns)),pivot.columns)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            if np.isfinite(vals[i,j]):ax.text(j,i,f"{vals[i,j]:+.2f}",ha="center",va="center",fontsize=8)
    ax.set_title("各土类内部按地形湿度划分四组")
    fig.colorbar(im,ax=ax,label="标准化路径系数 β")
    fig.suptitle("图I｜不同土壤类型与地形湿度背景下的路径差异",fontsize=15)
    save_figure(fig,run/"figures","I_soil_twi_sem_comparison")


def build_evidence_crosswalk(run: Path) -> pd.DataFrame:
    src = source_paths()
    nine = src["nine"] / "tables"
    global_assoc = pd.read_csv(nine / "corrected_i1_global_associations.csv")
    pair_assoc = pd.read_csv(nine / "corrected_i2_matched_pair_associations.csv")
    zero = pd.read_csv(ROOT / "docs/reviews/2026-10-31/phase-i/tables/i1_zero_positive_effects.csv")
    weighted = pd.read_csv(nine / "gwr_weighted_effect_summary.csv")
    zones = pd.read_csv(src["i8"] / "tables/stress_zone_summary.csv")
    rows = [
      {"claim":"匹配设计的空间留出评估","phase":"F","population":"10个空间留出种子","source":str(src["phase_f"].relative_to(ROOT)),"figure":"Phase F设计诊断图","statistic":"覆盖率中位数0.483；最大绝对标准化均差中位数0.012","boundary":"Oracle设计评估；覆盖质量不等同于独立实测或全部阶段总体。"},
      {"claim":"早期匹配配对结果显示指标间关联强弱不同","phase":"G","population":"历史Aa_20yr匹配候选，约90,848对","source":str(src["phase_g"].relative_to(ROOT)),"figure":"Phase G配对结果图","statistic":"SHI/OM/SW的Spearman ρ约−0.128/−0.114/−0.192；AN与MBC接近0","boundary":"历史匹配版本与后续候选版本不同；仅作阶段脉络，不与后续样本混并。"},
      {"claim":"低侵蚀筛选改变部分匹配关联，但未改善全部空间检验","phase":"H","population":"CDF筛选前后Oracle匹配候选","source":str(src["phase_h"].relative_to(ROOT)),"figure":"Phase H低侵蚀情景图","statistic":"匹配端点SHI/OM/SW ρ约−0.451/−0.446/−0.601，筛选后约−0.288/−0.318/−0.302","boundary":"结果导向阈值敏感性分析；不能替代全域基线或视为独立检验。"},
    ]
    pair = pair_assoc.loc[pair_assoc.exposure.eq("Aa_20yr") & pair_assoc.candidate_id.eq("stage2_Aa_20yr_01") & pair_assoc.population.eq("all_erosion")].set_index("outcome")
    global_i = global_assoc.loc[global_assoc.exposure.eq("Aa_20yr") & global_assoc.population.eq("mask_all")].set_index("outcome")
    for outcome in OUTCOME_ORDER:
        if outcome not in global_i.index: continue
        g = global_i.loc[outcome]; p = pair.loc[outcome] if outcome in pair.index else None
        rows.append({"claim":f"全域及匹配配对关联：{LABELS[outcome]}","phase":"I","population":f"全掩膜像元n={int(g.n_complete):,}；匹配配对n={int(p.n_pairs):,}" if p is not None else f"全掩膜像元n={int(g.n_complete):,}",
                     "source":str((nine / "corrected_i1_global_associations.csv").relative_to(ROOT))+"; "+str((nine / "corrected_i2_matched_pair_associations.csv").relative_to(ROOT)),
                     "figure":"A｜全域与配对差值关联","statistic":f"全域Spearman ρ={g.spearman_rho:+.3f}；标准化β={g.standardized_beta_log1p_Aa:+.3f}（5 km块95%CI {g.block_bootstrap_beta_ci_low:+.3f}至{g.block_bootstrap_beta_ci_high:+.3f}）"+(f"；匹配配对Spearman ρ={p.spearman_rho:+.3f}" if p is not None else ""),
                     "boundary":"全域像元和配对差值单位/估计目标不同；均为预测栅格结果。CEC的计算式含Aa，存在代数耦合。" if outcome=="CEC" else "预测栅格像元空间相关；配对数不是独立实测样本量。"})
    zero_i = zero.loc[zero.population.eq("mask_all") & zero.exposure.eq("Aa_20yr")].set_index("outcome")
    for outcome in ["SHI_candidate", "SW", "OM"]:
        z = zero_i.loc[outcome]
        rows.append({"claim":f"零侵蚀与正侵蚀像元差异：{LABELS[outcome]}","phase":"I","population":f"零侵蚀n={int(z.n_zero):,}；正侵蚀n={int(z.n_positive):,}",
                     "source":"docs/reviews/2026-10-31/phase-i/tables/i1_zero_positive_effects.csv","figure":"A｜零/正侵蚀差异",
                     "statistic":f"标准化均值差Hedges g={z.standardized_mean_difference_hedges_g:+.3f}；正侵蚀减零侵蚀均值={z.mean_difference_positive_minus_zero:+.3f}（5 km块95%CI {z.block_bootstrap_difference_ci_low:+.3f}至{z.block_bootstrap_difference_ci_high:+.3f}）",
                     "boundary":"空间块重抽样不消除全部空间依赖；该比较是全掩膜像元的状态差异。"})
    for outcome in ["SHI_candidate", "OM", "SW"]:
        q = weighted.loc[weighted.outcome.eq(outcome)].set_index("model_family")
        gwr = float(q.loc["GWR", "pixel_weighted_mean_local_standardized_beta"])
        lcr = float(q.loc["GWR-LCR", "pixel_weighted_mean_local_standardized_beta"])
        rows.append({"claim":f"局部系数分布：{LABELS[outcome]}","phase":"I","population":"GWR-LCR 7,387个240m单元；GWR加权像元支撑",
                     "source":str((nine / "gwr_weighted_effect_summary.csv").relative_to(ROOT)),"figure":"B｜GWR/MGWR局部系数；Phase I局部空间图",
                     "statistic":f"GWR像元加权标准化β={gwr:+.3f}；局部共线性岭收缩GWR-LCR β={lcr:+.3f}",
                     "boundary":"两类模型收缩程度不同；加权均值不呈现局部系数空间离散程度。"})
    rows.extend([
      {"claim":"SHI、OM、SW局部系数分布分位数","phase":"J","population":"各指标7,387–7,408个240 m GWR-LCR单元及770个标准GWR/MGWR空间位置",
       "source":"outputs/exploratory_oracle/phase_j/phase_j_20261102_seed20261102_r10/spatial_synthesis/local_coefficient_distribution_summary.csv",
       "figure":"B｜GWR/MGWR局部系数",
       "statistic":"包含最小值、P5、P25、中位数、P75、P95、最大值及负系数单元比例",
       "boundary":"负系数比例不等于显著性比例；所有结果复用既有系数，没有重估模型。"},
      {"claim":"SHI、OM、SW最强负向局部区域的空间重叠","phase":"J","population":"GWR-LCR 7,387个共同空间单元；标准GWR/MGWR 770个共同位置",
       "source":"outputs/exploratory_oracle/phase_j/phase_j_20261102_seed20261102_r10/spatial_synthesis/CROSS_INDICATOR_SPATIAL_OVERLAP.csv",
       "figure":"B｜GWR/MGWR局部系数",
       "statistic":"比较各指标局部β最低10%空间单元的重叠数、Jaccard率及局部β秩相关",
       "boundary":"纯描述性位置重叠，不是热点显著性检验。"},
      {"claim":"局部侵蚀系数与土类及环境背景的对应","phase":"J","population":"局部系数与240 m汇总环境变量共同有效的空间单元",
       "source":"outputs/exploratory_oracle/phase_j/phase_j_20261102_seed20261102_r10/spatial_synthesis/LOCAL_BETA_ENVIRONMENT_ASSOCIATION.csv; LOCAL_BETA_BY_ENVIRONMENT_GROUP.csv",
       "figure":"B/C｜局部系数与环境候选区",
       "statistic":"包含高程、地形湿度、NDVI、NDWI、地表温度、土类的组内系数摘要和连续变量Spearman相关",
       "boundary":"描述性空间单元关联；不作因果或显著性推断。"},
      {"claim":"GWR与MGWR局部估计差异最大的空间位置","phase":"J","population":"SHI、OM、SW各770个共同空间均衡位置",
       "source":"outputs/exploratory_oracle/phase_j/phase_j_20261102_seed20261102_r10/spatial_synthesis/GWR_MGWR_LARGEST_LOCAL_DIFFERENCES.csv",
       "figure":"B｜GWR/MGWR局部系数",
       "statistic":"逐指标列出两种局部β绝对差最大的10个位置及其环境背景",
       "boundary":"仅为模型间描述性差异位置，不是行政区或田块边界。"},
      {"claim":"GWR-LCR与标准GWR的局部格局一致程度","phase":"J","population":"SHI、OM、SW同一240 m空间单元",
       "source":"outputs/exploratory_oracle/phase_j/phase_j_20261102_seed20261102_r10/spatial_synthesis/gwr_gwrlcr_local_agreement.csv",
       "figure":"B｜GWR/MGWR局部系数",
       "statistic":"比较局部系数Pearson/Spearman相关、符号一致率和平均绝对差",
       "boundary":"用于评估局部岭收缩后方向及排序保留情况，不作显著性判断。"},
    ])
    for z in zones.itertuples():
        rows.append({"claim":f"空间候选分区画像：{z.zone}","phase":"I","population":f"{int(z.spatial_units_240m):,}个240m单元；面积等价{z.area_km2_equivalent:.3f}km²",
                     "source":str((src["i8"] / "tables/stress_zone_summary.csv").relative_to(ROOT)),"figure":"C｜T1–T4分区与环境画像",
                     "statistic":f"平均Aa={z.mean_Aa_20yr:.3f}；平均SHI={z.mean_SHI:.3f}；局部GWRβ中位数={z.median_GWR_beta:+.3f}；负向GWR Gi*热点占比={z.GWR_GiStar_hotspot_share:.3f}",
                     "boundary":"互斥分区来自结果导向候选规则；环境覆盖代理可以重叠；仅供探索和现场核验排序。"})
    sem_paths = pd.read_csv(run / "results/SEM_PATH_COEFFICIENTS.csv")
    sem_fit = pd.read_csv(run / "results/SEM_MODEL_FIT_COMPARISON.csv")
    for method in ["piecewise_sem", "CB_SEM_lavaan_ML"]:
        path = sem_paths.loc[sem_paths.dataset.eq("D1_common_env_case") & sem_paths.method.eq(method) & sem_paths.structure.eq("S2_direct_plus_indirect_SHI") & sem_paths.control.eq("M1") & sem_paths["from"].eq("z_E") & sem_paths.to.eq("z_SHI")]
        fit = sem_fit.loc[sem_fit.dataset.eq("D1_common_env_case") & sem_fit.method.eq(method) & sem_fit.structure.eq("S2_direct_plus_indirect_SHI") & sem_fit.control.eq("M1")]
        if len(path) and len(fit):
            r = fit.iloc[0]; p = path.iloc[0]
            rows.append({"claim":f"核心环境控制后的侵蚀—SHI路径（{method}）","phase":"J","population":f"D1共同环境完整像元n={int(r.n):,}",
                         "source":"results/SEM_PATH_COEFFICIENTS.csv; results/SEM_MODEL_FIT_COMPARISON.csv","figure":"D/G/H｜路径、拟合与侵蚀增量",
                         "statistic":f"侵蚀→SHI标准化β={p.beta_standardized:+.3f}；SHI R²={r.r_squared:.3f}；CFI={r.cfi_naive if pd.notna(r.cfi_naive) else np.nan:.3f}；RMSEA={r.rmsea_naive if pd.notna(r.rmsea_naive) else np.nan:.3f}",
                         "boundary":"控制变量调整后的关联；lavaan常规拟合指数未作空间块校正，Piecewise区间采用5km空间块协方差。"})
    rows.extend([
      {"claim":"CEC计算构造与侵蚀暴露存在确定代数联系","phase":"I/J","population":"CEC有效像元","source":"outputs/exploratory_oracle/phase_i/cec_corrected_input_20_2p9_plus18_20261031/correction_manifest.json","figure":"A / CEC单项结果仅作敏感性","statistic":"CEC=(原始CEC−20×(Aa_20yr/143.00503540039062 + Aa_2020/48.76557540893555))/2.9+18","boundary":"CEC路径/相关不能作为独立于侵蚀的证据；图表中指标统一标作CEC。"},
      {"claim":"既有PCNM空间特征向量结果保留并延期专门对照","phase":"F–J","population":"既有PCNM分析","source":"docs/reviews/2026-10-31/phase-i/PHASE_I_FINAL_RESEARCH_REPORT.md","figure":"现阶段未新增PCNM图","statistic":"后续对照既有PCNM空间控制结果；本阶段未替代或重跑PCNM","boundary":"避免把本阶段环境控制路径误称为PCNM控制。"},
    ])
    frame = pd.DataFrame(rows)
    frame["outcome_guided_selection"] = True
    csv_write(frame, run / "research_index/PHASE_F_I_EVIDENCE_CROSSWALK.csv")
    csv_write(frame, run / "research_index/RESEARCH_EVIDENCE_CROSSWALK.csv")
    return frame


def build_spatial_stress_products(run: Path, spatial: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    local = spatial["local_summary"].copy()
    family_map = {"GWR-LCR（九项结局）": "GWR-LCR", "同样本标准GWR（770空间单元）": "GWR",
                  "标准MGWR（770空间单元）": "MGWR"}
    local["estimator"] = local.model_family.map(family_map)
    fit = spatial["fit"].copy()
    fit["estimator"] = fit.model_family.map({"standard_GWR_same_sample": "GWR",
                                              "standard_MGWR_spatially_balanced_sample": "MGWR"})
    fit = fit[["outcome", "estimator", "n_spatial_units", "r_squared", "adjusted_r_squared", "bandwidth_neighbors"]]
    compare = local.merge(fit, on=["outcome", "estimator"], how="left")
    bw = spatial["bandwidth"].loc[spatial["bandwidth"].variable.eq("log1p_Aa"),
                                   ["outcome", "bandwidth_neighbors", "median_neighbor_distance_m"]].copy()
    bw = bw.rename(columns={"bandwidth_neighbors": "Aa_bandwidth_neighbors", "median_neighbor_distance_m": "Aa_median_neighbor_distance_m"})
    mgwr_lookup = bw.set_index("outcome")
    compare["Aa_bandwidth_neighbors"] = compare.outcome.map(mgwr_lookup.Aa_bandwidth_neighbors)
    compare["Aa_median_neighbor_distance_m"] = compare.outcome.map(mgwr_lookup.Aa_median_neighbor_distance_m)
    agree = spatial["agreement"].rename(columns={"pearson_local_beta_correlation": "GWR_MGWR_local_beta_r",
                                                  "sign_agreement_fraction": "GWR_MGWR_sign_agreement",
                                                  "mean_absolute_beta_difference": "GWR_MGWR_mean_abs_beta_difference"})
    compare = compare.merge(agree[["outcome", "GWR_MGWR_local_beta_r", "GWR_MGWR_sign_agreement",
                                  "GWR_MGWR_mean_abs_beta_difference"]], on="outcome", how="left")
    compare["outcome_guided_selection"] = True
    csv_write(compare, run / "spatial_synthesis/GWR_MGWR_LOCAL_COEFFICIENT_COMPARISON.csv")

    i8 = source_paths()["i8"] / "tables"
    units = pd.read_parquet(i8 / "stress_zones_240m.parquet")
    summaries = pd.read_csv(i8 / "stress_zone_summary.csv").set_index("zone")
    zones = ["T1 高Aa—强负向响应", "T2 高Aa—弱负向响应", "T3 低Aa—低SHI", "T4 环境敏感候选", "其他/未分类"]
    envs = ["Elevation", "Cur", "TWI", "LST", "NDWI", "NDBSI", "EVI", "NDVI", "Precipitation"]
    profiles = []
    for zone in zones:
        d = units.loc[units.primary_zone.eq(zone)]
        s = summaries.loc[zone]
        row = {"zone": zone, "n_240m_units": int(s.spatial_units_240m), "n_supported_30m_pixels": int(s.complete_case_pixels),
               "area_km2_equivalent": float(s.area_km2_equivalent), "mean_Aa_20yr": float(s.mean_Aa_20yr),
               "mean_SHI": float(s.mean_SHI), "median_GWR_beta": float(s.median_GWR_beta),
               "negative_GWR_share": float(s.negative_GWR_share), "GWR_GiStar_hotspot_share": float(s.GWR_GiStar_hotspot_share),
               "LISA_HL_share": float(s.LISA_HL_share), "environment_sensitive_overlay_share": float(s.environment_sensitive_overlay_share)}
        for variable in envs:
            mean = float(d[variable].mean()) if len(d) else np.nan
            sd = float(units[variable].std(ddof=1))
            row[f"{variable}_mean"] = mean
            row[f"{variable}_difference_from_overall_sd"] = (mean - float(units[variable].mean())) / sd if sd > 0 else np.nan
        for code in [26, 27, 42]:
            row[f"dominant_soil_{code}_share"] = float(d.dominant_soil_type.eq(code).mean()) if len(d) else np.nan
        row["outcome_guided_selection"] = True
        profiles.append(row)
    profile = pd.DataFrame(profiles)
    csv_write(profile, run / "spatial_synthesis/SPATIAL_STRESS_ENVIRONMENT_PROFILE.csv")
    return compare, profile


def build_cross_stage_deliverables(run: Path, sem: dict[str, pd.DataFrame], spatial: dict[str, pd.DataFrame],
                                   crosswalk: pd.DataFrame) -> None:
    paths, fits, delta = sem["paths"], sem["fits"], sem["delta"]
    compare, profile = build_spatial_stress_products(run, spatial)
    global_table = pd.read_csv(source_paths()["nine"] / "tables/corrected_i1_global_associations.csv")
    global_table = global_table.query("population == 'mask_all' and exposure == 'Aa_20yr'").set_index("outcome")
    pair_table = pd.read_csv(source_paths()["nine"] / "tables/corrected_i2_matched_pair_associations.csv")
    pair_table = pair_table.query("exposure == 'Aa_20yr' and candidate_id == 'stage2_Aa_20yr_01' and population == 'all_erosion'").set_index("outcome")
    zero_table = pd.read_csv(ROOT / "docs/reviews/2026-10-31/phase-i/tables/i1_zero_positive_effects.csv")
    zero_table = zero_table.query("population == 'mask_all' and exposure == 'Aa_20yr'").set_index("outcome")
    local_weighted = spatial["weighted"].set_index(["outcome", "model_family"])

    def beta(structure: str, control: str, source: str, target: str, method: str = "piecewise_sem") -> float:
        q = paths.loc[paths.dataset.eq("D1_common_env_case") & paths.method.eq(method) & paths.structure.eq(structure) &
                      paths.control.eq(control) & paths["from"].eq(source) & paths.to.eq(target)]
        return float(q.beta_standardized.iloc[0]) if len(q) else np.nan

    def r2(structure: str, control: str, method: str = "piecewise_sem", endogenous: str = "z_SHI") -> float:
        q = fits.loc[fits.dataset.eq("D1_common_env_case") & fits.method.eq(method) & fits.structure.eq(structure) &
                     fits.control.eq(control) & fits.endogenous.eq(endogenous)]
        return float(q.r_squared.iloc[0]) if len(q) and pd.notna(q.r_squared.iloc[0]) else np.nan

    def delta_r2(structure: str, control: str, method: str, outcome: str = "z_SHI") -> float:
        q = delta.loc[delta.dataset.eq("D1_common_env_case") & delta.method.eq(method) & delta.structure.eq(structure) &
                      delta.control.eq(control) & delta.outcome.eq(outcome)]
        return float(q.delta_r2_erosion_conditional.iloc[0]) if len(q) else np.nan

    sw, shi, om = [float(global_table.loc[o, "spearman_rho"]) for o in ["SW", "SHI_candidate", "OM"]]
    sw_pair, shi_pair, om_pair = [float(pair_table.loc[o, "spearman_rho"]) for o in ["SW", "SHI_candidate", "OM"]]
    core_text = f"""# 三条核心科学发现

## 发现一：侵蚀较高区域的SHI、土壤含水量和有机质整体偏低，配对差值也保留负向关系

**问题与方法：** 全域像元秩相关和标准化斜率、零/正侵蚀像元差异、环境匹配配对差值及全样本Gaussian GAM曲线分别回答总体排序、侵蚀状态差异、匹配端点差异和非线性形态。四类估计量的单位与统计总体不同，不能直接作“匹配削弱多少”的比较。

**定量结果：** 全域Spearman ρ为SW {sw:+.3f}、SHI {shi:+.3f}、OM {om:+.3f}；第二套环境匹配候选的配对差值ρ为SW {sw_pair:+.3f}、SHI {shi_pair:+.3f}、OM {om_pair:+.3f}。正侵蚀相对零侵蚀的标准化均值差Hedges g分别为SW {zero_table.loc['SW','standardized_mean_difference_hedges_g']:+.3f}、SHI {zero_table.loc['SHI_candidate','standardized_mean_difference_hedges_g']:+.3f}、OM {zero_table.loc['OM','standardized_mean_difference_hedges_g']:+.3f}。三项GAM曲线整体随log(1+Aa)上升而下降，但不同区间斜率不恒定。

**解释边界：** 结果来自预测栅格；栅格像元并非独立样点。配对差值关系说明所选匹配总体中侵蚀差与健康差仍有关联，不证明匹配本身造成衰减。CEC只作为单项敏感性，因为其计算式包含Aa。

**代表图与证据：** [图A](../figures/A_multimethod_erosion_health_response.png)；[全域与零/正状态源数据](../figures/data/figure_A_global_zero_pair_metrics.csv)；[GAM曲线源数据](../figures/data/figure_A_gam_curves.csv)。

## 发现二：负向局部系数占主导，但尺度和地点仍影响局部强弱

**问题与方法：** 复用九项结局的GWR-LCR全支撑局部系数，并对SHI、OM、SW复用同一770个空间均衡点的标准GWR/MGWR估计；比较局部系数相关、符号一致率、模型拟合与Aa变量带宽。

**定量结果：** GWR-LCR像元加权标准化β为SW {local_weighted.loc[('SW','GWR-LCR'),'pixel_weighted_mean_local_standardized_beta']:+.3f}、SHI {local_weighted.loc[('SHI_candidate','GWR-LCR'),'pixel_weighted_mean_local_standardized_beta']:+.3f}、OM {local_weighted.loc[('OM','GWR-LCR'),'pixel_weighted_mean_local_standardized_beta']:+.3f}。770个空间均衡点上，GWR/MGWR系数的符号一致率均为100%；局部系数Pearson相关分别为SHI {spatial['agreement'].set_index('outcome').loc['SHI_candidate','pearson_local_beta_correlation']:.3f}、OM {spatial['agreement'].set_index('outcome').loc['OM','pearson_local_beta_correlation']:.3f}、SW {spatial['agreement'].set_index('outcome').loc['SW','pearson_local_beta_correlation']:.3f}，说明符号相同但局部幅度排序的一致性并不都高。OM与SW的MGWR Aa带宽达到767近邻设定上限（中位距离约23.8 km）；不能将上限值称为精确的生态作用半径。

**空间候选：** T1含623个240m单元、面积等价18.882 km²，平均Aa=15.670、平均SHI=0.549、局部GWRβ中位数−0.567；T2为842单元、24.001 km²，β中位数−0.327；T3为60单元、2.053 km²，平均SHI=0.562；T4为961单元、38.767 km²，平均Aa=5.109、平均SHI=0.628。T1与T4分别代表高侵蚀强负向候选和环境条件候选，不是风险或治理分区。

**代表图与证据：** [图B](../figures/B_gwr_mgwr_spatial_patterns.png)、[图C](../figures/C_stress_sensitive_environment_zones.png)；[局部系数对照表](../spatial_synthesis/GWR_MGWR_LOCAL_COEFFICIENT_COMPARISON.csv)；[候选环境画像](../spatial_synthesis/SPATIAL_STRESS_ENVIRONMENT_PROFILE.csv)。

## 发现三：加入核心环境控制后，侵蚀与三个健康维度及SHI仍呈负向条件关联

**问题与方法：** 以SW、BD组合物理维度，以OM、AN组合化学维度，以MBC、RESP组合生物活动维度；比较分段路径模型和成熟lavaan CB-SEM。控制层级从不加背景变量，到核心环境、环境主成分、再到核心环境加空间趋势。CEC与pH未进入主化学综合分数。

**定量结果：** 全域主要环境完整像元、核心环境控制下，侵蚀到物理、化学、生物活动维度的标准化路径分别为{beta('S1_parallel_dimensions','M1','z_E','physical'):+.3f}、{beta('S1_parallel_dimensions','M1','z_E','chemical'):+.3f}、{beta('S1_parallel_dimensions','M1','z_E','biological_activity'):+.3f}；SHI模型中的直接路径为{beta('S2_direct_plus_indirect_SHI','M1','z_E','z_SHI'):+.3f}，R²={r2('S2_direct_plus_indirect_SHI','M1'):.3f}，条件ΔR²={delta_r2('S2_direct_plus_indirect_SHI','M1','piecewise_sem'):.4f}。48个lavaan代表模型均收敛；lavaan路径和拟合指数以常规独立观测ML计算，空间依赖推断以分段模型5 km块协方差为主要参考。

**解释边界：** SHI本身由健康指标构成，维度到SHI路径部分反映构造信息；路径模型只是条件关系分解，不是因果中介。M0不控制环境时侵蚀ΔR²较大，环境控制后该增量降低，故主结果采用环境含义清晰、仍保留负向路径且可作空间块稳健推断的M1分段模型作为解释候选，而不称其为拟合冠军。

**代表图与证据：** [图D](../figures/D_sem_control_hierarchy.png)、[图G](../figures/G_sem_fit_and_erosion_increment.png)、[图H](../figures/H_best_sem_path_diagram.png)；[统一路径表](../results/SEM_PATH_COEFFICIENTS.csv)；[拟合表](../results/SEM_MODEL_FIT_COMPARISON.csv)。
"""
    (run / "reports/PHASE_J_CORE_SCIENTIFIC_FINDINGS.md").write_text(core_text, encoding="utf-8")

    phase_text = f"""# Phase F–I跨阶段科学发现

## Phase F：先评估匹配设计是否保留可比样本

空间留出设计评估显示，M4方案的覆盖率中位数约0.483，最大绝对标准化均差中位数约0.012。环境平衡较好并不自动保证侵蚀差异足够，也不意味着剩余混杂为零；该结果是Oracle设计评估，不是最终独立验证。

## Phase G：环境匹配后各指标表现不同

历史候选匹配配对中，SHI、OM、SW的配对差值Spearman ρ约为−0.128、−0.114、−0.192，AN和MBC接近零。它说明九项指标不是统一反应；该配对版本约90,848对，必须与后续更新版本分别呈现。

## Phase H：低侵蚀样本筛选影响表观关系

对历史匹配总体施加CDF筛选后，SHI、OM、SW端点关联约由−0.451、−0.446、−0.601变为−0.288、−0.318、−0.302。部分负向关联绝对值减小，空间留出结果也未一致改善。因为阈值搜索受结果影响，此分析作为敏感性脉络，不当作预设验证。

## Phase I：纠正CEC后形成九项结局的全域和局部证据

在全掩膜像元上，Aa_20yr与SW、SHI、OM的Spearman ρ分别为{sw:+.3f}、{shi:+.3f}、{om:+.3f}；第二套匹配配对的相应差值相关分别为{sw_pair:+.3f}、{shi_pair:+.3f}、{om_pair:+.3f}。GWR-LCR像元加权标准化β分别为SW {local_weighted.loc[('SW','GWR-LCR'),'pixel_weighted_mean_local_standardized_beta']:+.3f}、SHI {local_weighted.loc[('SHI_candidate','GWR-LCR'),'pixel_weighted_mean_local_standardized_beta']:+.3f}、OM {local_weighted.loc[('OM','GWR-LCR'),'pixel_weighted_mean_local_standardized_beta']:+.3f}。全域相关、匹配配对差值和局部回归回答不同问题，不把数值变化当作匹配效果。

CEC按用户最终指定的全域最大值公式计算，报告和图表统称CEC；由于公式含Aa，CEC相关有代数耦合。九项结局均有更新后的全域及局部系数表。既有PCNM结果继续保留，留待后续与环境匹配结果专门对照。

## 可追溯源表

逐条证据、源报告、源表、样本单位和解释边界见[证据交叉索引](../research_index/RESEARCH_EVIDENCE_CROSSWALK.csv)。
"""
    (run / "reports/PHASE_F_TO_I_SCIENTIFIC_FINDINGS.md").write_text(phase_text, encoding="utf-8")

    # Densified local-coefficient and environment-zone source-backed report.
    main_fits = spatial["fit"].set_index(["outcome", "model_family"])
    agreement = spatial["agreement"].set_index("outcome")
    bws = spatial["bandwidth"].query("variable == 'log1p_Aa'").set_index("outcome")
    zone_lines = []
    for _, q in profile.iloc[:4].iterrows():
        zone_lines.append(f"| {q.zone} | {int(q.n_240m_units):,} | {q.area_km2_equivalent:.3f} | {q.mean_Aa_20yr:.3f} | {q.mean_SHI:.3f} | {q.median_GWR_beta:+.3f} | {q.dominant_soil_27_share:.1%} |")
    bw_lines = []
    for outcome in ["SHI_candidate", "OM", "SW"]:
        b = bws.loc[outcome]
        mgfit = main_fits.loc[(outcome, "standard_MGWR_spatially_balanced_sample")]
        gfit = main_fits.loc[(outcome, "standard_GWR_same_sample")]
        a = agreement.loc[outcome]
        bw_lines.append(f"| {LABELS[outcome]} | {gfit.r_squared:.3f} | {mgfit.r_squared:.3f} | {int(b.bandwidth_neighbors)} | {b.median_neighbor_distance_m/1000:.2f} | {a.pearson_local_beta_correlation:.3f} | {a.sign_agreement_fraction:.1%} |")
    family_names = {
        "GWR-LCR（九项结局）": "局部共线性收缩GWR",
        "同样本标准GWR（770空间单元）": "标准GWR",
        "标准MGWR（770空间单元）": "标准MGWR",
    }
    focus_order = ["SHI_candidate", "OM", "SW"]
    local_distribution_lines = []
    for family in family_names:
        for outcome in focus_order:
            q = spatial["local_summary"].loc[
                spatial["local_summary"].model_family.eq(family) & spatial["local_summary"].outcome.eq(outcome)
            ]
            if q.empty:
                continue
            r = q.iloc[0]
            local_distribution_lines.append(
                f"| {family_names[family]} | {LABELS[outcome]} | {int(r.n_units):,} | "
                f"{r.min_beta:+.3f} | {r.p5_beta:+.3f} | {r.p25_beta:+.3f} | {r.median_beta:+.3f} | "
                f"{r.p75_beta:+.3f} | {r.p95_beta:+.3f} | {r.max_beta:+.3f} | {r.negative_unit_fraction:.1%} |"
            )
    overlap_lines = []
    for r in spatial["overlap"].itertuples():
        overlap_lines.append(
            f"| {r.model_family} | {r.label_a}—{r.label_b} | {int(r.n_common_units):,} | "
            f"{int(r.decile_overlap_units):,}/{int(r.strongest_negative_decile_units_each):,} | "
            f"{r.decile_overlap_jaccard:.3f} | {r.spearman_local_beta_rho:.3f} |"
        )
    gwr_lcr_match_lines = []
    for outcome in focus_order:
        q = spatial["gwr_lcr_agreement"].loc[spatial["gwr_lcr_agreement"].outcome.eq(outcome)]
        if q.empty:
            continue
        r = q.iloc[0]
        gwr_lcr_match_lines.append(
            f"| {LABELS[outcome]} | {int(r.n_common_units):,} | {r.pearson_local_beta_correlation:.3f} | "
            f"{r.spearman_local_beta_correlation:.3f} | {r.sign_agreement_fraction:.1%} | "
            f"{r.mean_absolute_beta_difference:.3f} |"
        )
    environment_contrast_lines = []
    lcr_contrasts = spatial["environment_contrasts"].loc[
        spatial["environment_contrasts"].model_family.eq("GWR-LCR")
    ]
    environment_names = {"Elevation": "高程", "TWI": "地形湿度指数", "NDVI": "植被指数NDVI",
                         "NDWI": "水体/湿润指数NDWI", "LST": "地表温度", "SoilType": "土壤类型"}
    for r in lcr_contrasts.itertuples():
        environment_contrast_lines.append(
            f"| {LABELS[r.outcome]} | {environment_names[r.environment_variable]} | "
            f"{r.strongest_negative_group}（β中位数{r.strongest_negative_median_beta:+.3f}，n={int(r.strongest_negative_n_units):,}） | "
            f"{r.weakest_negative_group}（β中位数{r.weakest_negative_median_beta:+.3f}，n={int(r.weakest_negative_n_units):,}） |"
        )
    environment_rho_lines = []
    lcr_environment_rho = spatial["environment_associations"].loc[
        spatial["environment_associations"].model_family.eq("GWR-LCR") &
        spatial["environment_associations"].environment_variable.ne("SoilType")
    ]
    for r in lcr_environment_rho.itertuples():
        environment_rho_lines.append(
            f"| {LABELS[r.outcome]} | {environment_names[r.environment_variable]} | "
            f"{r.spearman_local_beta_environment_rho:+.3f} | {int(r.n_common_units):,} |"
        )
    largest_difference_lines = []
    for outcome in focus_order:
        d = spatial["largest_differences"].loc[spatial["largest_differences"].outcome.eq(outcome)].head(3)
        for r in d.itertuples():
            largest_difference_lines.append(
                f"| {LABELS[outcome]} | {r.x/1000:.3f}, {r.y/1000:.3f} | "
                f"{r.local_beta_log1p_Aa_standardized_gwr:+.3f} | {r.local_beta_log1p_Aa_standardized_mgwr:+.3f} | "
                f"{r.absolute_beta_difference:.3f} | {r.SoilType_group or '缺失'} | {r.TWI:.2f} | {r.NDVI:.2f} |"
            )
    lcr_focus = spatial["local_summary"].loc[
        spatial["local_summary"].model_family.eq("GWR-LCR（九项结局）") &
        spatial["local_summary"].outcome.isin(focus_order)
    ].set_index("outcome")
    lcr_distribution_text = "；".join(
        f"{LABELS[o]}中位β={lcr_focus.loc[o, 'median_beta']:+.3f}，范围"
        f"{lcr_focus.loc[o, 'min_beta']:+.3f}至{lcr_focus.loc[o, 'max_beta']:+.3f}，"
        f"负系数单元占{lcr_focus.loc[o, 'negative_unit_fraction']:.1%}"
        for o in focus_order
    )
    lcr_overlap = spatial["overlap"].loc[spatial["overlap"].model_family.eq("GWR-LCR")]
    strongest_overlap = lcr_overlap.sort_values("decile_overlap_jaccard", ascending=False).iloc[0]
    lcr_agreement_focus = spatial["gwr_lcr_agreement"].set_index("outcome")
    lcr_agreement_text = "；".join(
        f"{LABELS[o]}与标准GWR局部β相关r={lcr_agreement_focus.loc[o, 'pearson_local_beta_correlation']:.3f}、"
        f"符号一致率{lcr_agreement_focus.loc[o, 'sign_agreement_fraction']:.1%}"
        for o in focus_order
    )
    core_spatial_addition = (
        f"局部系数分布补充：{lcr_distribution_text}。在同一240 m空间支持上，各指标最低10%局部β的最大重叠出现在"
        f"{strongest_overlap.label_a}与{strongest_overlap.label_b}之间（{int(strongest_overlap.decile_overlap_units)}个共同单元，"
        f"Jaccard={strongest_overlap.decile_overlap_jaccard:.3f}）；{lcr_agreement_text}。"
        "这两项比较均为描述性系数对照，不是热点显著性检验。"
    )
    core_path = run / "reports/PHASE_J_CORE_SCIENTIFIC_FINDINGS.md"
    core_content = core_path.read_text(encoding="utf-8")
    core_content = core_content.replace("\n## 发现三：", f"\n\n{core_spatial_addition}\n\n## 发现三：")
    core_path.write_text(core_content, encoding="utf-8")
    spatial_text = """# 局部空间关联与环境候选分区综合报告

## 估计对象与支撑范围

九项指标的GWR-LCR以既有240 m空间单元及其30 m支撑像元作全域系数分布摘要；SHI、OM、SW的标准GWR与标准MGWR则复用同一组770个空间均衡位置。图B中每一行的GWR/MGWR共用以0为中心的对称色阶，并统一投影坐标、比例尺和指北箭头。770点图仅表示抽样空间单元的系数，不是将点插值为30 m像元，也不提供像元级独立显著性。

## GWR与MGWR的空间差异

三个指标的GWR和MGWR局部系数符号一致率均为100%，但SHI的局部系数相关为""" + f"{agreement.loc['SHI_candidate','pearson_local_beta_correlation']:.3f}，OM为{agreement.loc['OM','pearson_local_beta_correlation']:.3f}，SW为{agreement.loc['SW','pearson_local_beta_correlation']:.3f}。符号一致不代表估计幅度空间排序相同；两类局部估计的平均绝对差分别约{agreement.loc['SHI_candidate','mean_absolute_beta_difference']:.3f}、{agreement.loc['OM','mean_absolute_beta_difference']:.3f}、{agreement.loc['SW','mean_absolute_beta_difference']:.3f}。\n\n| 指标 | 同样本GWR R² | 标准MGWR R² | MGWR Aa近邻带宽 | 中位邻域距离(km) | 局部系数相关r | 符号一致率 |\n|---|---:|---:|---:|---:|---:|---:|\n""" + "\n".join(bw_lines) + "\n\n" + """SHI的Aa带宽约32近邻；OM与SW达到767近邻搜索上限，邻域中位距离约23.8 km。达到上限意味着候选空间尺度可能更宽，不应把模型上界读成准确的生态作用距离。MGWR拟合的R²是770个空间位置上的表面拟合指标，不是空间留出预测性能。

## 候选分区对比

T1–T4是互斥主分区；“27潮土TWI第二组”与“26冲积土TWI最低组”是可叠加环境代理覆盖，图C用不同标记区分，不能误认为新增互斥区域。四类候选区面积和核心响应为：

| 候选分区 | 240m单元数 | 等价面积(km²) | 平均Aa | 平均SHI | 局部GWR β中位数 | 优势27土类比例 |\n|---|---:|---:|---:|---:|---:|---:|\n""" + "\n".join(zone_lines) + "\n\n" + """T1由Aa较高且局部负向关联较强的阈值组合形成，负向Gi*热点占比高；它适合优先现场复核侵蚀量、土壤水分和剖面OM。T2同属较高侵蚀条件，但GWR系数较弱，可用于对照。T3侵蚀较低而SHI偏低，提醒健康状况并不只由侵蚀暴露排序解释。T4由环境敏感代理规则形成，样区以27类潮土为主；它用于识别环境背景相似的比较点，不能据此判定侵蚀风险更高。

## 环境画像与建议

完整环境均值、相对全研究区标准差差异、主导土类比例和区内汇总指标保存于[空间候选环境画像CSV](../spatial_synthesis/SPATIAL_STRESS_ENVIRONMENT_PROFILE.csv)。该画像可用于规划抽样时覆盖不同高程、TWI、地表温度、植被与土类组合，但当前分区由结果导向候选规则生成，需用独立土样、实测侵蚀或田块调查核查。不能将GWR热点直接当作工程治理边界；施策还需结合坡面、灌排、耕作、土类和农户管理资料。

## 表格和图件

[GWR/MGWR局部系数对照CSV](../spatial_synthesis/GWR_MGWR_LOCAL_COEFFICIENT_COMPARISON.csv)含系数分布、拟合R²、Aa带宽及两模型局部一致性；[图B](../figures/B_gwr_mgwr_spatial_patterns.png)展示三指标的同位点系数；[图C](../figures/C_stress_sensitive_environment_zones.png)展示主分区及候选环境画像。所有地图候选均标记`outcome_guided_selection=true`。
"""
    spatial_details = f"""## 局部系数分布和负向区域重叠

下表列出SHI、OM、SW在三种已保存模型中的局部标准化系数分布。GWR-LCR覆盖九项结局的240 m空间单元；标准GWR和MGWR使用相同的770个空间均衡位置。负向比例是局部估计单元比例，不代表显著性比例。

| 模型 | 指标 | 单元数 | 最小值 | P5 | P25 | 中位数 | P75 | P95 | 最大值 | 负系数比例 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(local_distribution_lines)}

GWR-LCR通过局部岭惩罚降低局部共线性影响。与同一空间单元上的标准GWR对照，三项指标的局部系数相关、符号一致率和平均绝对差为：

| 指标 | 共同单元 | Pearson相关 | Spearman相关 | 符号一致率 | 平均绝对系数差 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(gwr_lcr_match_lines)}

将相同空间支持上局部系数最低的10%定义为“最强负向十分位”，三指标两两重叠如下。该指标只是空间位置重合度，不是热点显著性检验。

| 模型 | 指标组合 | 共同单元 | 同时处于最低10%的单元 | Jaccard重叠率 | 局部β Spearman ρ |
|---|---|---:|---:|---:|---:|
{chr(10).join(overlap_lines)}

在GWR-LCR中，OM与SW的最强负向区域重叠最大；涉及SHI的重叠相对较低，说明综合指数与单项水分、有机质响应的局部强弱并非完全相同。标准GWR/MGWR同样支持三项在770个位置上的局部系数均为负，但估计幅度的空间排序一致程度因指标不同。

## 局部响应与环境背景

把局部系数与对应240 m单元的高程、地形湿度指数、NDVI、NDWI、地表温度和土壤类型连接。连续变量按全域单元秩四分位分组。下表报告局部β中位数最负与最弱的环境组；组间只是描述性比较，不作因果或显著性判断。

| 指标 | 环境因素 | 负向系数较强组 | 负向系数较弱组 |
|---|---|---|---|
{chr(10).join(environment_contrast_lines)}

连续环境与局部β的Spearman相关如下。负相关表示环境值较高时，局部β通常更负（绝对值更大）；这些数值是单变量空间共变，不表示环境因素造成关联变化。

| 指标 | 环境因素 | Spearman ρ | 单元数 |
|---|---|---:|---:|
{chr(10).join(environment_rho_lines)}

SHI和OM的局部系数在较低高程、较低NDVI及较高NDWI单元中整体更负；SW系数虽然在几乎所有单元为负，但与单个连续环境的秩相关接近零，环境组之间也没有简单一致的强弱顺序。个别土类样本较少，例如85编码仅144个单元，相关中位数只作探索提示。上述比较用于设计现场对照采样，不替代多变量环境控制。

## GWR与MGWR差异最大的共同位置

下表列出各指标两种估计绝对差最大的三个共同空间位置。坐标为投影坐标千米，点位附带的土类、TWI和NDVI用作位置背景描述，不代表行政边界或田块调查结论。

| 指标 | 东向、北向坐标（km） | GWR β | MGWR β | 系数绝对差 | 土类 | TWI | NDVI |
|---|---|---:|---:|---:|---|---:|---:|
{chr(10).join(largest_difference_lines)}

区域候选结果也显示局部强度差异：高侵蚀—强负向响应候选T1的局部GWRβ中位数为−0.567，高侵蚀—弱负向响应候选T2为−0.327。它们由结果导向规则识别，适合用于确定实地核查次序，不是经独立数据验证的风险分区。"""
    spatial_text = spatial_text.replace("\n## 候选分区对比\n", f"\n{spatial_details}\n\n## 候选分区对比\n")
    (run / "reports/PHASE_J_SPATIAL_STRESS_SYNTHESIS_REPORT.md").write_text(spatial_text, encoding="utf-8")

    # Preserve four objective-specific candidates for each primary population.
    # AIC uses one mature CB-SEM estimator; ΔR², path ranking, and the interpretation
    # candidate use the same spatial-block piecewise estimator for like-with-like comparison.
    primary_populations = ["D1_common_env_case", "D2_240m_blocks", "D3_M4_unique_pixels", "D4_M4_pair_differences"]
    population_labels = {
        "D1_common_env_case": "全域共同完整像元",
        "D2_240m_blocks": "240米空间汇总单元",
        "D3_M4_unique_pixels": "M4匹配端点像元",
        "D4_M4_pair_differences": "M4配对差值",
    }
    winner_rows = []
    for dataset in primary_populations:
        lavaan = fits.loc[fits.dataset.eq(dataset) & fits.method.eq("CB_SEM_lavaan_ML") &
                          fits.structure.isin(["S2_direct_plus_indirect_SHI", "S3_indirect_only_SHI"]) &
                          fits.status.eq("ok")].copy()
        if not lavaan.empty:
            best = lavaan.sort_values("aic").iloc[0]
            winner_rows.append({"objective":"A｜整体拟合候选","method":best.method,"dataset":dataset,
                                "population_label":population_labels[dataset],"structure":best.structure,"control":best.control,
                                "criterion":"同一总体SHI方程、S2/S3候选中的最低AIC","value":best.aic,
                                "additional_fit":f"CFI={best.cfi_naive:.3f}; TLI={best.tli_naive:.3f}; RMSEA={best.rmsea_naive:.3f}; SRMR={best.srmr:.3g}; df={best.df}; n={int(best.n)}",
                                "interpretation":"需要结合自由度判断；df=0为饱和模型，其CFI/RMSEA不能作为独立拟合质量证据。",
                                "outcome_guided_selection":True})

        population_delta = delta.loc[delta.dataset.eq(dataset) & delta.method.eq("piecewise_sem") &
                                     delta.outcome.eq("z_SHI") & delta.structure.eq("S2_direct_plus_indirect_SHI")]
        if not population_delta.empty:
            best = population_delta.sort_values("delta_r2_erosion_conditional", ascending=False).iloc[0]
            path = paths.loc[paths.dataset.eq(dataset) & paths.method.eq("piecewise_sem") &
                             paths.structure.eq("S2_direct_plus_indirect_SHI") & paths.control.eq(best.control) &
                             paths["from"].eq("z_E") & paths.to.eq("z_SHI")]
            direct_beta = float(path.beta_standardized.iloc[0]) if len(path) else np.nan
            winner_rows.append({"objective":"B｜侵蚀增量最大","method":"piecewise_sem","dataset":dataset,
                                "population_label":population_labels[dataset],"structure":best.structure,"control":best.control,
                                "criterion":"同一SHI方程中最大条件ΔR²","value":best.delta_r2_erosion_conditional,
                                "additional_fit":f"R²full={best.r2_full:.3f}; R²without={best.r2_without_erosion:.3f}; β={direct_beta:+.3f}; n={int(best.n)}",
                                "interpretation":"这是模型内侵蚀的条件预测增量；不据此将未调整模型解释为最佳科学模型。",
                                "outcome_guided_selection":True})

        population_paths = paths.loc[paths.dataset.eq(dataset) & paths.method.eq("piecewise_sem") &
                                     paths.structure.eq("S2_direct_plus_indirect_SHI") &
                                     paths["from"].eq("z_E") & paths.to.eq("z_SHI")]
        if not population_paths.empty:
            best = population_paths.sort_values("beta_standardized").iloc[0]
            winner_rows.append({"objective":"C｜负向直接路径最明显","method":best.method,"dataset":dataset,
                                "population_label":population_labels[dataset],"structure":best.structure,"control":best.control,
                                "criterion":"同一总体S2分段路径中侵蚀→SHI标准化β最小","value":best.beta_standardized,
                                "additional_fit":f"SHI R²={best.r_squared:.3f}; n={int(best.n)}",
                                "interpretation":"路径绝对值大不代表因果效应或外部预测最好。",
                                "outcome_guided_selection":True})

        fit_m1 = fits.loc[fits.dataset.eq(dataset) & fits.method.eq("piecewise_sem") &
                          fits.structure.eq("S2_direct_plus_indirect_SHI") & fits.control.eq("M1") &
                          fits.endogenous.eq("z_SHI") & fits.status.eq("ok")]
        path_m1 = population_paths.loc[population_paths.control.eq("M1")]
        delta_m1 = population_delta.loc[population_delta.control.eq("M1")]
        if not fit_m1.empty and not path_m1.empty and not delta_m1.empty:
            fit = fit_m1.iloc[0]
            direct = path_m1.iloc[0]
            increment = delta_m1.iloc[0]
            winner_rows.append({"objective":"D｜最佳科学解释候选","method":"piecewise_sem","dataset":dataset,
                                "population_label":population_labels[dataset],"structure":"S2_direct_plus_indirect_SHI","control":"M1",
                                "criterion":"核心环境控制 + 5 km空间块协方差路径区间","value":direct.beta_standardized,
                                "additional_fit":f"SHI R²={fit.r_squared:.3f}; 侵蚀ΔR²={increment.delta_r2_erosion_conditional:.4f}; n={int(fit.n)}",
                                "interpretation":"保留可解释环境控制和空间块协方差；不称为拟合冠军。",
                                "outcome_guided_selection":True})
    winners = pd.DataFrame(winner_rows)
    csv_write(winners, run / "results/SEM_MODEL_OBJECTIVE_WINNERS.csv")
    (run / "reports/SEM_MODEL_OBJECTIVE_SELECTION.md").write_text(
        "# SEM模型选择：四类总体各自报告四个目标\n\nAIC候选使用同一总体、SHI结果和成熟lavaan估计器比较S2/S3；侵蚀增量、负向路径和解释候选使用同一总体的S2分段路径模型。排名不跨总体合并。所有结果来自Oracle探索，属于表观、结果导向比较。\n\n" + "\n".join(
            f"## {row.dataset}｜{row.objective}\n\n总体：{row.population_label}。候选模型：{row.method}，结构={row.structure}，控制={row.control}。判据：{row.criterion}；数值={row.value:.4f}。\n\n{row.interpretation} {row.additional_fit}\n"
            for row in winners.itertuples()), encoding="utf-8")

    # Answer the ten required scientific questions in a concise, linked document.
    control_rows = []
    control_labels = {"M0":"仅侵蚀","M1":"核心环境控制","M2":"环境主成分控制","M3":"核心环境与空间趋势控制"}
    best_beta = beta("S2_direct_plus_indirect_SHI", "M1", "z_E", "z_SHI")
    best_delta = delta_r2("S2_direct_plus_indirect_SHI", "M1", "piecewise_sem")
    for control in ["M0","M1","M2","M3"]:
        control_rows.append((control, beta("S2_direct_plus_indirect_SHI",control,"z_E","z_SHI"),
                             r2("S2_direct_plus_indirect_SHI",control), delta_r2("S2_direct_plus_indirect_SHI",control,"piecewise_sem")))
    control_md = "| 控制层级 | 侵蚀→SHI β | SHI R² | 侵蚀条件ΔR² |\n|---|---:|---:|---:|\n" + "\n".join(
        f"| {control_labels[c]} | {b:+.3f} | {rr:.3f} | {dd:.4f} |" for c,b,rr,dd in control_rows)
    # The concise ten-question narrative retains its explicitly stated D1 baseline.
    d1_lavaan = fits.loc[fits.dataset.eq("D1_common_env_case") & fits.method.eq("CB_SEM_lavaan_ML") &
                         fits.structure.isin(["S2_direct_plus_indirect_SHI", "S3_indirect_only_SHI"]) &
                         fits.status.eq("ok")].copy()
    shi_delta = delta.loc[delta.dataset.eq("D1_common_env_case") & delta.outcome.eq("z_SHI") &
                          delta.structure.eq("S2_direct_plus_indirect_SHI") &
                          delta.method.isin(["piecewise_sem", "CB_SEM_lavaan_ML"])].copy()
    d1_lavaan_cmp = d1_lavaan.sort_values("aic") if not d1_lavaan.empty else d1_lavaan
    q7 = d1_lavaan_cmp.iloc[0] if not d1_lavaan_cmp.empty else None
    q7_structure = ({"S2_direct_plus_indirect_SHI":"包含侵蚀直达SHI路径的维度模型",
                     "S3_indirect_only_SHI":"仅通过健康维度关联SHI的模型"}.get(q7.structure, str(q7.structure)) if q7 is not None else "")
    max_delta = shi_delta.sort_values("delta_r2_erosion_conditional",ascending=False).iloc[0] if not shi_delta.empty else None
    t1 = profile.iloc[0]; t2 = profile.iloc[1]; t3 = profile.iloc[2]; t4 = profile.iloc[3]
    answers = f"""# 十个核心研究问题的结果回答

1. **侵蚀与土壤健康的主要关系是什么？** 全域Spearman ρ为SW {sw:+.3f}、SHI {shi:+.3f}、OM {om:+.3f}；在第二套匹配配对差值总体中分别为{sw_pair:+.3f}、{shi_pair:+.3f}、{om_pair:+.3f}。正侵蚀与零侵蚀像元的标准化均值差也为负：SW {zero_table.loc['SW','standardized_mean_difference_hedges_g']:+.3f}、SHI {zero_table.loc['SHI_candidate','standardized_mean_difference_hedges_g']:+.3f}、OM {zero_table.loc['OM','standardized_mean_difference_hedges_g']:+.3f}。预测栅格关系不等于独立实测证据。

2. **环境匹配后这种关系是否仍存在？** 在所选配对方案中，侵蚀差与SW、SHI、OM差值的秩相关仍为−0.272、−0.144、−0.115。这个结果说明该配对总体内仍有条件差值关联；不能直接与全域像元ρ作减法来估计匹配后的“衰减”。

3. **什么环境下负向关系更明显？** T1由高Aa与更强负向局部系数组合筛出：623个240m单元、18.882km²、平均Aa 15.670、平均SHI {t1.mean_SHI:.3f}、GWR β中位数{t1.median_GWR_beta:+.3f}。T4为环境敏感候选：961单元、38.767km²、平均Aa 5.109、27类潮土优势比例{t4.dominant_soil_27_share:.1%}；它不是高侵蚀区。T2为高侵蚀但局部系数较弱候选；T3为低侵蚀且SHI偏低候选。候选阈值参考健康结果，须独立现场核验。

4. **关系在空间上如何变化？** 770点GWR/MGWR结果中三指标的系数符号一致率为100%，但局部系数相关仅SHI {spatial['agreement'].set_index('outcome').loc['SHI_candidate','pearson_local_beta_correlation']:.3f}、OM {spatial['agreement'].set_index('outcome').loc['OM','pearson_local_beta_correlation']:.3f}、SW {spatial['agreement'].set_index('outcome').loc['SW','pearson_local_beta_correlation']:.3f}。OM和SW的Aa带宽落到767近邻上界，提示可能偏宽尺度。

5. **物理、化学、生物维度分别怎样关联？** 核心环境控制的分段路径β分别为物理{beta('S1_parallel_dimensions','M1','z_E','physical'):+.3f}、化学{beta('S1_parallel_dimensions','M1','z_E','chemical'):+.3f}、生物活动{beta('S1_parallel_dimensions','M1','z_E','biological_activity'):+.3f}。三个维度均为负向条件关联；CEC没有纳入化学合成分数。

6. **加入环境和空间因素后侵蚀路径如何变化？** 同一样本SHI模型各控制层级为：\n\n{control_md}\n\n从仅含侵蚀的基础模型，逐步加入高程、地形湿度、植被、地表温度和土类，再比较环境主成分及空间坐标趋势。β绝对值和条件ΔR²随控制结构变化，显示环境和空间背景与部分预测共变相关。

7. **哪种SEM整体拟合最好？** 在相同全域总体、相同SHI结果和成熟CB-SEM方法中，按AIC最低的候选为""" + (f"{q7_structure}、{control_labels.get(q7.control, str(q7.control))}，AIC={q7.aic:.1f}，CFI={q7.cfi_naive:.3f}，TLI={q7.tli_naive:.3f}，RMSEA={q7.rmsea_naive:.3f}，SRMR={q7.srmr:.4f}，df={q7.df:.0f}。" if q7 is not None else "暂无成熟软件可比拟合行。") + "\n\n若自由度为0，该模型为饱和/刚好识别，CFI和RMSEA不提供真正的模型检验；所有CB指标也未修正空间依赖。\n\n"
    max_delta_method = "成熟lavaan CB-SEM" if max_delta is not None and max_delta.method == "CB_SEM_lavaan_ML" else "分段路径模型"
    answers += f"8. **哪个模型的侵蚀增量贡献最大？** 同一全域SHI方程中最高条件ΔR²为{max_delta.delta_r2_erosion_conditional:.4f}（{max_delta_method}、{control_labels.get(max_delta.control, str(max_delta.control))}），对应完整R²={max_delta.r2_full:.3f}、去侵蚀R²={max_delta.r2_without_erosion:.3f}。这只是该控制条件下的预测增量，不是因果贡献。\n\n"
    answers += f"9. **哪个模型最适合解释现有成果？** 推荐用核心环境控制下的分段路径模型作主解释候选：纳入含义清晰的环境变量，SHI直接路径β={best_beta:+.3f}、SHI R²={r2('S2_direct_plus_indirect_SHI','M1'):.3f}、侵蚀ΔR²={best_delta:.4f}，且可用5 km空间块协方差形成路径区间。它并非最高拟合模型或因果模型；lavaan提供整体拟合诊断，MGWR提供局部结构补充。\n\n"
    answers += "10. **能形成什么综合结论与建议？** 当前预测面上，侵蚀与SW、OM、SHI及若干健康维度呈负向关联；匹配差值和局部空间模型提供相互补充但估计目标不同的证据。T1可优先安排独立土壤与侵蚀核查，T4可覆盖不同潮土—湿度背景作比较；不把候选图当作工程治理界线。治理方案应待现场测量后，结合坡面、灌排、耕作及土壤改良条件制定。\n\n" \
        "**限制：** 健康图层是预测栅格；CEC公式含侵蚀项；SHI源公式及权重尚待核实；候选区和模型受结果导向筛选影响；PCNM分析保留并留待后续专门对照。"
    (run / "reports/PHASE_J_RESEARCH_QUESTIONS_ANSWERS.md").write_text(answers, encoding="utf-8")

    crosswalk_path = run / "research_index/RESEARCH_EVIDENCE_CROSSWALK.csv"
    master = """# Phase F–J研究结果总索引

本索引把匹配设计、跨阶段响应、空间结果和结构路径按研究问题联接。所有新分析来自EXPLORATORY_ORACLE，`outcome_guided_selection=true`；预测栅格不能当作独立实测样点。CEC在图表中统一写作CEC，其计算公式与侵蚀暴露直接耦合；PCNM保留并后续对照。

## 三条核心发现

- [三条核心科学发现](../reports/PHASE_J_CORE_SCIENTIFIC_FINDINGS.md)
- [十个核心研究问题的回答](../reports/PHASE_J_RESEARCH_QUESTIONS_ANSWERS.md)
- [Phase F–I跨阶段发现](../reports/PHASE_F_TO_I_SCIENTIFIC_FINDINGS.md)
- [证据交叉索引CSV](RESEARCH_EVIDENCE_CROSSWALK.csv)

## 核心图件

| 图 | 主题 | PNG | 源数据/说明 |
|---|---|---|---|
"""
    titles = ["全域、匹配配对、零/正状态与GAM响应", "SHI、OM、SW的GWR/MGWR空间系数", "T1–T4环境候选分区",
              "不同控制层级的侵蚀路径", "全域与匹配端点路径", "侵蚀对三个健康维度的路径", "拟合度与侵蚀增量", "综合条件路径图", "土类×湿度组路径差异"]
    for i, stem in enumerate(["A_multimethod_erosion_health_response","B_gwr_mgwr_spatial_patterns","C_stress_sensitive_environment_zones",
                              "D_sem_control_hierarchy","E_full_vs_matched_sem","F_erosion_dimension_path_forest",
                              "G_sem_fit_and_erosion_increment","H_best_sem_path_diagram","I_soil_twi_sem_comparison"]):
        extras = {"A_multimethod_erosion_health_response":"../figures/data/figure_A_global_zero_pair_metrics.csv",
                  "B_gwr_mgwr_spatial_patterns":"../spatial_synthesis/GWR_MGWR_LOCAL_COEFFICIENT_COMPARISON.csv",
                  "C_stress_sensitive_environment_zones":"../spatial_synthesis/SPATIAL_STRESS_ENVIRONMENT_PROFILE.csv"}.get(stem, "../reports/PHASE_J_FIGURE_CAPTIONS.md")
        master += f"| 图{chr(65+i)} | {titles[i]} | [PNG](../figures/{stem}.png) | [说明/数据]({extras})；另有SVG和PDF |\n"
    master += """

## 样本总体与模型结果

- `data/population_counts.csv`记录7类像元、空间汇总、匹配端点、配对差值、正侵蚀和环境分组总体的样本单位。
- `results/SEM_MODEL_FIT_COMPARISON.csv`含piecewise、lavaan CB-SEM、自定义CB敏感性和PLS模型；`results/SEM_PATH_COEFFICIENTS.csv`、`SEM_EROSION_INCREMENTAL_CONTRIBUTION.csv`、`SEM_DIRECT_INDIRECT_EFFECTS.csv`记录路径、条件ΔR²及直接/间接效应。
- `results/SEM_MODEL_OBJECTIVE_WINNERS.csv`按D1–D4分别列出整体拟合、侵蚀增量、最强负向路径和科学解释候选；`results/SEM_MODEL_RANKINGS.csv`在每个总体内部比较四级控制的R²、侵蚀条件ΔR²和直接路径，不跨总体合并排名。
- `spatial_synthesis/GWR_MGWR_LOCAL_COEFFICIENT_COMPARISON.csv`及`SPATIAL_STRESS_ENVIRONMENT_PROFILE.csv`记录局部系数分布、带宽、模型对照和T1–T4画像。
- `local_coefficient_distribution_summary.csv`补充局部系数分位数；`gwr_gwrlcr_local_agreement.csv`比较局部岭收缩前后的格局；`CROSS_INDICATOR_SPATIAL_OVERLAP.csv`比较三个重点结局最强负向区域重叠；`LOCAL_BETA_ENVIRONMENT_ASSOCIATION.csv`、`LOCAL_BETA_BY_ENVIRONMENT_GROUP.csv`和`LOCAL_BETA_ENVIRONMENT_CONTRASTS.csv`记录局部系数与环境背景的描述性联系；`GWR_MGWR_LARGEST_LOCAL_DIFFERENCES.csv`列出两种模型估计差异较大的位置。

## 报告和交付

- [详细技术报告](../reports/PHASE_J_FINAL_TECHNICAL_REPORT.md)
- [SEM结果报告](../reports/PHASE_J_SEM_RESEARCH_REPORT.md)
- [空间压力格局报告](../reports/PHASE_J_SPATIAL_STRESS_SYNTHESIS_REPORT.md)
- [正式学术报告](../reports/ENV_MATCH_SOIL_HEALTH_ACADEMIC_REPORT.md)
- [摘要](../reports/PHASE_J_ABSTRACT.md)；[15页汇报提纲](../reports/PHASE_J_PRESENTATION_OUTLINE.md)
- [九图交互图集](../reports/PHASE_J_RESEARCH_FIGURE_ATLAS.html)

## 前序阶段证据

主要阶段报告：`docs/reviews/2026-10-26/PHASE_F_RESEARCH_REPORT.md`、`docs/reviews/2026-10-27/phase-g/PHASE_G_RESEARCH_REPORT.md`、`docs/reviews/2026-10-29/phase-h/PHASE_H_FINAL_SUMMARY.md`、`docs/reviews/2026-10-31/phase-i/PHASE_I_FINAL_RESEARCH_REPORT.md`。逐条数据源、样本、数值与限制见本目录证据交叉表。\n"""
    (run / "research_index/RESEARCH_RESULTS_MASTER_INDEX.md").write_text(master, encoding="utf-8")

    # Reader-facing report based on the full academic narrative, with internal population codes translated.
    formal = (run / "reports/PHASE_J_ACADEMIC_RESEARCH_REPORT.md").read_text(encoding="utf-8")
    formal = formal.replace("# 温县耕地土壤侵蚀与土壤健康关联及空间差异分析", "# 温县耕地土壤侵蚀与土壤健康关联及空间差异研究")
    formal = formal.replace("Phase J", "本研究")
    formal = formal.replace("Phase F—I", "前序研究阶段")
    formal = formal.replace("Phase F–I", "前序研究阶段")
    formal = formal.replace("D1共同环境完整案例", "主要环境变量完整的全域像元")
    formal = formal.replace("D1环境共同完整案例", "主要环境变量完整的全域像元")
    formal = formal.replace("D1共同环境完整像元", "主要环境变量完整的全域像元")
    formal = formal.replace("D1–D4", "四类代表性统计总体")
    formal = formal.replace("S1–S6", "并列维度、SHI路径、链式路径、单项指标和两部分暴露等模型结构")
    formal = formal.replace("M0–M3", "从不加控制、核心环境控制、环境主成分控制到再加入空间趋势的四级控制")
    formal = formal.replace("Phase I8", "既有240 m空间汇总")
    formal = formal.replace("按照用户指定的", "按本研究采用的")
    formal = formal.replace("按用户指定的", "按本研究采用的")
    formal = formal.replace("GWR-LCR", "局部共线性收缩GWR")
    formal = formal.replace("旧CEC", "原始CEC")
    for code, meaning in {
        "D1": "全域像元总体", "D2": "240 m空间汇总总体", "D3": "环境匹配端点总体",
        "D4": "匹配配对差值总体", "D5": "正侵蚀像元总体", "D6": "零—正侵蚀配对总体", "D7": "土类—地形湿度分组总体",
        "M0": "不加环境控制", "M1": "核心环境控制", "M2": "环境主成分控制", "M3": "加入空间趋势控制", "M4": "环境匹配方案",
        "S1": "平行维度结构", "S2": "含侵蚀直接路径的SHI结构", "S3": "不含侵蚀直接路径的SHI结构",
        "S4": "链式健康维度结构", "S5": "单项健康指标结构", "S6": "零值状态与正侵蚀强度结构",
    }.items():
        formal = formal.replace(code, meaning)
    formal_local_rows = []
    for family in family_names:
        for outcome in focus_order:
            q = spatial["local_summary"].loc[
                spatial["local_summary"].model_family.eq(family) & spatial["local_summary"].outcome.eq(outcome)
            ]
            if q.empty:
                continue
            r = q.iloc[0]
            formal_local_rows.append(
                f"| {family_names[family]} | {LABELS[outcome]} | {r.min_beta:+.3f} | {r.p5_beta:+.3f} | "
                f"{r.p25_beta:+.3f} | {r.median_beta:+.3f} | {r.p75_beta:+.3f} | {r.p95_beta:+.3f} | "
                f"{r.max_beta:+.3f} | {r.negative_unit_fraction:.1%} |"
            )
    formal_overlap_rows = []
    formal_family_names = {"GWR-LCR": "局部共线性收缩GWR", "GWR": "标准GWR", "MGWR": "标准MGWR"}
    for r in spatial["overlap"].itertuples():
        formal_overlap_rows.append(
            f"| {formal_family_names.get(r.model_family, r.model_family)} | {r.label_a}—{r.label_b} | {int(r.decile_overlap_units)} | "
            f"{r.decile_overlap_jaccard:.3f} | {r.spearman_local_beta_rho:.3f} |"
        )
    formal_gwr_lcr_rows = []
    for outcome in focus_order:
        q = spatial["gwr_lcr_agreement"].loc[spatial["gwr_lcr_agreement"].outcome.eq(outcome)]
        if q.empty:
            continue
        r = q.iloc[0]
        formal_gwr_lcr_rows.append(
            f"| {LABELS[outcome]} | {r.pearson_local_beta_correlation:.3f} | "
            f"{r.spearman_local_beta_correlation:.3f} | {r.sign_agreement_fraction:.1%} | "
            f"{r.mean_absolute_beta_difference:.3f} |"
        )
    formal_env_rows = []
    for r in lcr_contrasts.itertuples():
        formal_env_rows.append(
            f"| {LABELS[r.outcome]} | {environment_names[r.environment_variable]} | "
            f"{r.strongest_negative_group}（{r.strongest_negative_median_beta:+.3f}，n={int(r.strongest_negative_n_units):,}） | "
            f"{r.weakest_negative_group}（{r.weakest_negative_median_beta:+.3f}，n={int(r.weakest_negative_n_units):,}） |"
        )
    formal_difference_rows = []
    for outcome in focus_order:
        d = spatial["largest_differences"].loc[spatial["largest_differences"].outcome.eq(outcome)].head(1)
        for r in d.itertuples():
            formal_difference_rows.append(
                f"| {LABELS[outcome]} | {r.x/1000:.3f}, {r.y/1000:.3f} | "
                f"{r.local_beta_log1p_Aa_standardized_gwr:+.3f} | {r.local_beta_log1p_Aa_standardized_mgwr:+.3f} | "
                f"{r.absolute_beta_difference:.3f} |"
            )
    supplement = f"""

## 空间差异、模型拟合与解释边界补充

全域局部系数、770个空间均衡位置的GWR/MGWR结果，以及240 m环境候选区分别从不同空间支撑回答局部关系问题。局部共线性收缩GWR对SHI、OM、SW的加权标准化系数为{local_weighted.loc[('SHI_candidate','GWR-LCR'),'pixel_weighted_mean_local_standardized_beta']:+.3f}、{local_weighted.loc[('OM','GWR-LCR'),'pixel_weighted_mean_local_standardized_beta']:+.3f}、{local_weighted.loc[('SW','GWR-LCR'),'pixel_weighted_mean_local_standardized_beta']:+.3f}。770点上标准GWR与MGWR的局部系数符号一致，但相关分别为{spatial['agreement'].set_index('outcome').loc['SHI_candidate','pearson_local_beta_correlation']:.3f}、{spatial['agreement'].set_index('outcome').loc['OM','pearson_local_beta_correlation']:.3f}、{spatial['agreement'].set_index('outcome').loc['SW','pearson_local_beta_correlation']:.3f}；因此地点上的估计幅度仍受模型设定影响。OM和SW的Aa带宽达到模型搜索上限，无法从本分析给出精确的作用半径。

成熟lavaan 0.7.3用于完成代表性CB-SEM结构比较，在四类总体、三类基础路径结构和四级控制条件中共拟合48个模型。CB-SEM的CFI、TLI、RMSEA、SRMR及AIC/BIC是常规最大似然拟合指标；因预测像元存在空间依赖，这些指标不具备空间稳健解释。分段路径模型使用5 km空间块聚类协方差，适合报告路径区间，但其R²只是模型内拟合。两类方法互相补充，不能将任一者的高拟合度视为独立验证。

形成性Mode-B PLS在四类代表性总体、三种基础结构和四级环境/空间控制间进行比较；S1每个总体/控制组合拟合三条健康维度方程，S2和S3各增加一条SHI方程，共176条内生方程。S2保留侵蚀直达SHI路径，S3删除该路径。控制变量作为固定单指标构念，同一总体内所有结构与控制使用共同完整样本。该项目内权重迭代没有bootstrap区间，也没有空间稳健推断，结果仅用于描述结构与控制层级变化。

综合解释时，应优先陈述SW、OM、SHI的多方法负向关系和三类维度的条件路径，再说明匹配总体、环境控制以及局部空间带宽对结果的影响。路径系数随控制条件变化，说明土壤环境与空间背景解释了部分预测共变。候选区只能用于安排地面采样和现场踏勘的次序。建议在T1布置高侵蚀且负向局部系数明显的核查点，在T2设置较弱局部系数对照，并从T4环境候选区覆盖不同土类和湿度条件；随后用独立土壤实测、地块管理记录和独立侵蚀估计检验这些差异。当前证据不支持把候选分区直接转为治理边界或给出单一工程措施。

### 不同证据的适用范围

全域秩相关描述像元排序是否共同变化；标准化回归系数描述给定模型条件下侵蚀与结局的线性联系；GAM曲线呈现拟合总体中的非线性形态；局部回归展示系数随位置变化；环境匹配差值则对应被选中的近邻端点及其差值。它们的样本、尺度和估计对象不同，方向相同可以视为相互补充的描述，数值大小却不能直接排成一条强弱序列。特别是匹配配对由候选筛选和空间约束共同形成，配对后的相关并非将全域关系“控制到只剩侵蚀”的实验效应。

结构模型中的物理、化学与生物活动综合值用于压缩多个指标，目的是概括并比较响应维度；本次采用的组合构造尚未经过独立测量模型检验，也不能替代完整的土壤健康评价。SHI与其组成指标之间存在构造联系，因而SHI方程的高拟合不能单独证明生态过程路径。成熟协方差结构模型提供常规拟合诊断，分段模型提供空间块协方差下的路径区间，两者的推断假设并不相同；不能因模型收敛或拟合指标理想，就将预测像元视为相互独立的现场样本。

下一步的核验设计应同时覆盖响应较强和较弱的区域，并记录采样季节、灌排制度、作物轮作、地块管理及土壤采集深度。现场样本应按空间块安排，避免相邻点过密而把空间连续性误作独立重复；侵蚀暴露也应尽量通过独立估算或长期地表证据核对。首先检验SW和OM等单项测量是否复现预测面所示的空间梯度，再评估SHI指标方向、标准化和权重对总体结论的影响。这样的设计可以区分预测模型误差、管理差异和真实土壤差别，并为后续正式验证提供可复查的样本单位与变量定义。

### 主要结论

1. 全域像元、零/正侵蚀差异及所选匹配配对均显示侵蚀与SW、OM、SHI存在负向关联，但各统计单位回答不同问题。
2. 核心环境控制下侵蚀与物理、化学、生物活动维度及SHI的路径仍为负向，控制层级会改变路径大小与条件ΔR²。
3. GWR与MGWR均显示局部负向系数占主导，但宽带宽上限和局部估计幅度差异要求谨慎解释。
4. T1–T4提供互补的环境和局部关系核验候选，所有候选均需独立实测验证。
5. CEC的计算式含侵蚀项，SHI的源公式尚未核实；本研究不作因果中介解释，既有PCNM结果保留至下一阶段对照。
"""
    local_spatial_addition = f"""

## 三项重点结局的局部系数与环境差异

局部地理加权回归允许侵蚀与土壤指标的关联随地点改变。对SHI、OM和SW，240 m单元上的局部系数在三个模型中的范围、分位数及负系数比例如下。负系数比例是估计单元比例，不代表统计显著比例。

| 模型 | 指标 | 最小值 | P5 | P25 | 中位数 | P75 | P95 | 最大值 | 负系数比例 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(formal_local_rows)}

局部共线性收缩GWR与标准GWR在同一空间单元上的对照如下。负向系数若符号一致，表示方向格局保留；系数相关还反映空间强弱排序是否相似：

| 指标 | Pearson相关 | Spearman相关 | 符号一致率 | 平均绝对系数差 |
|---|---:|---:|---:|---:|
{chr(10).join(formal_gwr_lcr_rows)}

在同一空间支持内，对各指标局部系数最低的10%单元进行位置比较。局部β的Spearman相关反映空间强弱排序是否相似，Jaccard率反映最低十分位的重叠程度：

| 模型 | 指标组合 | 重叠单元数 | Jaccard率 | 局部β Spearman ρ |
|---|---|---:|---:|---:|
{chr(10).join(formal_overlap_rows)}

在240 m单元层面将局部共线性收缩GWR系数与高程、地形湿度、植被、水分指数、地表温度和土类连接后，最强和最弱的环境分组如下。连续环境按全域空间单元秩四分位分组。这些对照用于指出需要核查的环境背景，并非环境因素的独立因果效应。

| 指标 | 环境背景 | 系数较负组（β中位数） | 系数较弱组（β中位数） |
|---|---|---|---|
{chr(10).join(formal_env_rows)}

SHI和OM的负向局部系数在较低高程、较低NDVI及较高NDWI组整体更强；SW系数大多为负，但与各环境变量的总体秩相关较弱，分组强弱没有简单一致的梯度。个别土类样本较少（85编码144个单元），该组中位数仅作探索性描述。

标准GWR与MGWR三项指标的局部系数符号一致率均为100%，但幅度空间排序相关为SHI {spatial['agreement'].set_index('outcome').loc['SHI_candidate','pearson_local_beta_correlation']:.3f}、OM {spatial['agreement'].set_index('outcome').loc['OM','pearson_local_beta_correlation']:.3f}、SW {spatial['agreement'].set_index('outcome').loc['SW','pearson_local_beta_correlation']:.3f}。两估计绝对差最大的共同位置如下，坐标为投影坐标千米：

| 指标 | 东向、北向坐标（km） | GWR β | MGWR β | 绝对差 |
|---|---|---:|---:|---:|
{chr(10).join(formal_difference_rows)}

MGWR更支持SHI在较细的侵蚀带宽上变化；OM、SW的带宽接近当前搜索上界，表现为偏向较宽、较平滑的空间尺度。不能据此将带宽上限解释为精确的生态作用距离。"""
    formal = formal.rstrip() + supplement + local_spatial_addition
    formal_path = run / "reports/ENV_MATCH_SOIL_HEALTH_ACADEMIC_REPORT.md"
    formal_path.write_text(formal, encoding="utf-8")

    index_path = run / "research_index/RESEARCH_RESULTS_MASTER_INDEX.md"
    if not (run / "reports/ENV_MATCH_SOIL_HEALTH_ACADEMIC_REPORT.md").exists() or not crosswalk_path.exists():
        raise RuntimeError("Missing required Phase J synthesis index outputs")


def build_report(run: Path, sem: dict[str,pd.DataFrame], spatial:dict[str,pd.DataFrame], crosswalk:pd.DataFrame) -> None:
    paths,fits,delta,groups,rank=sem["paths"],sem["fits"],sem["delta"],sem["groups"],sem["rank"]
    d1=paths.loc[paths.dataset.eq("D1_common_env_case")&paths.method.eq("piecewise_sem")&paths.control.eq("M1")]
    def get_beta(structure,source,target):
        q=d1.loc[d1.structure.eq(structure)&d1["from"].eq(source)&d1["to"].eq(target)]
        return float(q.beta_standardized.iloc[0]) if len(q) else np.nan
    dim_betas={v:get_beta("S1_parallel_dimensions","z_E",v) for v in ["physical","chemical","biological_activity"]}
    shi_beta=get_beta("S2_direct_plus_indirect_SHI","z_E","z_SHI")
    shi_fit=fits.loc[fits.dataset.eq("D1_common_env_case")&fits.method.eq("piecewise_sem")&fits.structure.eq("S2_direct_plus_indirect_SHI")&fits.control.eq("M1")&fits.endogenous.eq("z_SHI")]
    shi_r2=float(shi_fit.r_squared.iloc[0]) if len(shi_fit) else np.nan
    drow=delta.loc[delta.dataset.eq("D1_common_env_case")&delta.structure.eq("S2_direct_plus_indirect_SHI")&delta.control.eq("M1")&delta.outcome.eq("z_SHI")]
    d_r2=float(drow.delta_r2_erosion_conditional.iloc[0]) if len(drow) else np.nan
    groups_m1=groups.loc[groups.control.eq("M1")&groups.structure.eq("S1_parallel_dimensions")&
                         groups["from"].eq("z_E")&groups.to.isin(["physical","chemical","biological_activity"])]
    group_top=groups_m1.sort_values("beta_standardized").head(3)[["soil_label","twi_group","dimension","n","beta_standardized","ci_low","ci_high"]]
    status=sem["status"].to_string(index=False)
    spatial_weight=spatial["weighted"].set_index(["outcome","model_family"])
    def gwrval(outcome,family,col):
        try:return float(spatial_weight.loc[(outcome,family),col])
        except Exception:return np.nan
    swfull,shifull,omfull=[float(pd.read_csv(source_paths()["nine"] / "tables/corrected_i1_global_associations.csv").query("population=='mask_all' and exposure=='Aa_20yr' and outcome==@o").spearman_rho.iloc[0]) for o in ["SW","SHI_candidate","OM"]]
    shimat=swmat=ommat=np.nan
    pair=pd.read_csv(source_paths()["nine"] / "tables/corrected_i2_matched_pair_associations.csv")
    def pairrho(o):
        q=pair.query("exposure=='Aa_20yr' and candidate_id=='stage2_Aa_20yr_01' and population=='all_erosion' and outcome==@o")
        return float(q.spearman_rho.iloc[0]) if len(q) else np.nan
    shimat,ommat,swmat=[pairrho(x) for x in ["SHI_candidate","OM","SW"]]
    outcome_rows=pd.read_csv(run/"figures/data/figure_A_multimethod_response.csv").set_index("outcome")
    outcome_order=["SHI_candidate","SW","OM","AN","CEC","pH","BD","MBC","RESP"]
    def outcome_stat_text(column):
        return "；".join(
            f"{outcome_rows.loc[o, 'label']} {float(outcome_rows.loc[o, column]):+.3f}"
            for o in outcome_order if o in outcome_rows.index
        )
    full_rho_all=outcome_stat_text("full_spearman_rho")
    pair_rho_all=outcome_stat_text("m4_pair_spearman_rho")
    gwr_beta_all=outcome_stat_text("gwr_weighted_local_beta")
    gwr_lcr_all="；".join(
        f"{outcome_rows.loc[o, 'label']} {gwrval(o, 'GWR-LCR', 'pixel_weighted_mean_local_standardized_beta'):+.3f}"
        for o in outcome_order if o in outcome_rows.index
    )
    twi_labels={"q1":"湿度最低组","q2":"湿度较低组","q3":"湿度较高组","q4":"湿度最高组"}
    group_top_text="；".join(
        f"{r.soil_label}、{twi_labels.get(r.twi_group, r.twi_group)}的{r.dimension} β={r.beta_standardized:+.3f}（n={int(r.n):,}）"
        for r in group_top.itertuples()
    )
    cec_model=paths.loc[
        paths.dataset.eq("D1_common_env_case") & paths.method.eq("piecewise_sem") &
        paths.structure.eq("S5_indicator_responses") & paths.control.eq("M1") &
        paths["from"].eq("z_E") & paths["to"].eq("z_CEC")
    ]
    cec_beta=float(cec_model.beta_standardized.iloc[0]) if len(cec_model) else np.nan
    cec_r2=float(cec_model.r_squared.iloc[0]) if len(cec_model) else np.nan
    lavaan_fits=fits.loc[fits.method.eq("CB_SEM_lavaan_ML")]
    lavaan_paths=paths.loc[paths.method.eq("CB_SEM_lavaan_ML")]
    lavaan_main_fit=lavaan_fits.loc[lavaan_fits.dataset.eq("D1_common_env_case") & lavaan_fits.structure.eq("S2_direct_plus_indirect_SHI") & lavaan_fits.control.eq("M1")]
    lavaan_main_path=lavaan_paths.loc[lavaan_paths.dataset.eq("D1_common_env_case") & lavaan_paths.structure.eq("S2_direct_plus_indirect_SHI") & lavaan_paths.control.eq("M1") & lavaan_paths["from"].eq("z_E") & lavaan_paths.to.eq("z_SHI")]
    lavaan_fit_text=(f"代表模型的CB-SEM标准化路径β={lavaan_main_path.beta_standardized.iloc[0]:+.3f}，CFI={lavaan_main_fit.cfi_naive.iloc[0]:.3f}，TLI={lavaan_main_fit.tli_naive.iloc[0]:.3f}，RMSEA={lavaan_main_fit.rmsea_naive.iloc[0]:.3f}，SRMR={lavaan_main_fit.srmr.iloc[0]:.3f}。" if len(lavaan_main_path) and len(lavaan_main_fit) else "成熟CB-SEM结果表未找到该代表模型。")
    summary=(
      f"# Phase J SEM结果与模型比较\n\n"
      f"## 样本与估计状态\n\n"
      f"CEC审计使用 {392527:,} 个对齐像元记录，其中 {387448:,} 个CEC有效值；按原始CEC及两期Aa全域最大值公式复算，与当前CEC存储值的最大差为0。全健康结局完整像元为384,211个，主要环境变量共同完整像元为293,145个；7,387个240 m单元覆盖293,633个支撑像元。CEC与2020年Aa在162个汇总单元内合计缺少487个支撑像元；AN和pH各有1个支撑像元缺失，按变量自身有效值计算均值。历史匹配端点版为181,794个；第二阶段配对版为111,289对、222,578个端点；二者共有133,533个端点，属于不同的匹配版本。正侵蚀像元195,013个，零—正配对68,172对，土类内TWI四分位共12组。\n\n"
      f"模型状态汇总：\n\n{status}\n\n"
      f"## 编号与术语简释\n\n"
      f"匹配端点版沿用早期同土类、高程/地温近邻设计：环境卡尺0.35、地理上限1,000 m、组内侵蚀百分位差至少0.50。配对差值版来自后续结果导向候选比较：环境卡尺0.75、地理上限500 m、每土类8个环境组、组内侵蚀百分位差至少0.30。两种端点集合仅部分重合，端点回归与配对差值回归不可视为同一样本的两种编码。全像元、240 m单元、端点、配对差值、正侵蚀、零—正配对和土类—地形湿度分组是不同统计总体。M0为不加环境控制，M1为核心环境控制，M2为环境主成分控制，M3在核心环境控制外加空间坐标趋势；S1–S6分别表示平行维度、SHI直接路径、无直接路径、维度链、单项指标和两部分侵蚀模型。GWR是允许局部关系随地点变化的地理加权回归，MGWR还为不同变量估计不同空间尺度，GWR-LCR是对局部共线性作岭收缩的敏感性模型。\n\n"
      f"## 结构路径\n\n"
      f"D1环境共同完整案例、M1控制下，侵蚀暴露与物理维度路径β={dim_betas['physical']:+.3f}，与化学维度β={dim_betas['chemical']:+.3f}，与生物活动维度β={dim_betas['biological_activity']:+.3f}；S2中侵蚀至SHI的直接路径β={shi_beta:+.3f}，SHI方程R²={shi_r2:.3f}，在维度路径之外加入侵蚀的条件ΔR²={d_r2:.4f}。这些是预测栅格上的条件关联与模型分解，不是因果中介。\n\n"
      f"土类—地形湿度分组中最负的三条维度路径（按系数排序）见 `SEM_ENVIRONMENT_GROUP_COMPARISON.csv`；组间不变性/等值检验未实现，因此只作为分层斜率描述，不将结果导向筛选出的单组解释成已确认敏感区。\n\n"
      f"## CEC及SHI边界\n\n"
      f"CEC按 `(旧CEC − 20×(Aa_20yr/143.00503540039062 + Aa_2020/48.76557540893555))/2.9 + 18` 生成。因预测因变量CEC的构造式含Aa，CEC与侵蚀的路径会机械带入负向项；CEC仅作为单项指标敏感性，不加入主化学维度。SHI原始公式、指标方向与适宜范围隶属度函数及权重均未找到可核实来源，因此保留现有SHI预测面，不另造健康方向转换分值，也不从CEC或其他指标重算SHI。\n\n"
      f"## 估计器说明\n\n"
      f"分段路径回归使用5 km空间块聚类协方差；若协方差对角方差无效，该路径保留点估计、将区间和P值记为缺失。另使用成熟lavaan 0.7.3拟合S1–S3结构，在D1–D4代表总体比较M0–M3，共{len(lavaan_fits)}个CB-SEM模型；输出标准化路径、CFI、TLI、RMSEA、SRMR、AIC/BIC与内生变量R²。{lavaan_fit_text}这些常规ML拟合指标按独立观测公式计算，未对栅格空间依赖作修正；空间稳健推断仍参考分段回归的5 km空间块协方差。形成性Mode-B PLS在D1–D4、S1–S3和M0–M3上完成全矩阵比较，共176条内生方程；S1每种总体/控制组合估计三条健康维度方程，S2和S3另估计SHI方程，并分别保留/删除侵蚀直达SHI路径。M1–M3控制变量作为固定单指标外生构念，同一总体的全部结构与控制使用共同完整案例。PLS采用项目内透明权重迭代，没有bootstrap区间或空间稳健推断。自定义协方差路径近似保留为方法敏感性，不再作为成熟CB-SEM结果。\n\n"
      f"## 与既有证据衔接\n\n"
      f"全域Spearman ρ：SW {swfull:+.3f}、SHI {shifull:+.3f}、OM {omfull:+.3f}。第二个匹配版本的配对差值ρ分别为SHI {shimat:+.3f}、OM {ommat:+.3f}、SW {swmat:+.3f}。前者按像元排序，后者按配对端点差值排序，不能用二者数值差异估计匹配造成的关联变化。局部加权平均标准化β（局部共线性收缩值）为SHI {gwrval('SHI_candidate','GWR-LCR','pixel_weighted_mean_local_standardized_beta'):+.3f}、OM {gwrval('OM','GWR-LCR','pixel_weighted_mean_local_standardized_beta'):+.3f}、SW {gwrval('SW','GWR-LCR','pixel_weighted_mean_local_standardized_beta'):+.3f}。\n\n"
      f"标准MGWR在770个空间均衡单元估计；SHI、OM、SW的R²分别约0.869、0.696、0.822。OM和SW带宽落到767近邻的上界（约23.8 km），表示模型支持较宽空间尺度，不应解释为已精确定位的作用半径。\n\n"
      f"## 模型选择\n\n"
      f"相同样本的SHI模型中，包含空间坐标趋势的方案表观R²最高（0.651），更能重建预测面的空间纹理，不代表侵蚀因果效应更强。不加环境控制的方案侵蚀条件ΔR²最大（0.017，β=−0.157），但未扣除环境差异，不能作为因果优选。核心环境控制方案（R²=0.601，ΔR²=0.0079，β=−0.113）较便于解释，但不是拟合冠军或独立验证模型。模型排序只在同一总体、因变量和结构内进行；候选比较具有选择偏差。\n\n"
      f"## 输出\n\n"
      f"路径、拟合、增量解释、直接/间接效应、多组斜率、模型排行与失败状态均保存为逐行CSV；图D–I的原始绘图数据另存于`figures/data/`。\n"
    )
    (run/"reports/PHASE_J_SEM_RESEARCH_REPORT.md").write_text(summary,encoding="utf-8")
    technical=(
      "# Phase J 综合技术报告\n\n"
      "## 1. 研究问题与分析框架\n\n"
      "Phase J将Phase F–I的匹配、全域响应、环境异质性、GWR/MGWR与压力分区结果综合，并在当前CEC数据基础上开展探索性路径模型。分析目标是区分全域像元关系、环境匹配端点和配对差值的不同统计单位，评估加入环境和空间控制后侵蚀路径的变化，再与局部空间关联和环境候选区对照。该阶段按项目授权在EXPLORATORY_ORACLE模式执行，健康结局参与候选聚合与解释，所有新输出均记录`outcome_guided_selection=true`。\n\n"
      "## 2. 数据和CEC审计\n\n"
      "环境侵蚀表与健康表各392,527行，cell_id唯一且集合完全相同；行列、投影坐标逐项一致。CEC有效值387,448个，范围16.726—26.837。使用完整对齐Aa表面的全域最大值143.00503540039062和48.76557540893555，按公式 `(CEC_old−20×(Aa_20yr/M20+Aa_2020/M2020))/2.9+18` 复算，当前CEC对有效行的最大误差为0。环境与健康有效标记相交为293,074行；另有71行虽环境总栈标记为无效，但本分析所需暴露、健康和环境变量均有有限值，只缺未进入模型的辅助近邻侵蚀字段，因此按本分析变量完整性纳入293,145个核心案例。端点健康值按cell_id回连当前健康表，避免使用旧CEC缓存。CEC公式内含侵蚀暴露，是重要的代数耦合风险。SHI原始公式、八项指标方向、适宜范围隶属度函数与权重均未找到可核验来源；因此保留原始指标符号和SHI预测面，不构造第二套健康方向统一分值。\n\n"
      "## 3. 人群与统计单位\n\n"
      "完整像元和主要环境变量完整案例分别提供较宽基线与核心控制模型样本；240 m汇总复现既有7,387单元、293,633个支撑像元，侵蚀与SHI汇总值复现误差为0。该汇总的CEC和2020年侵蚀量在162个单元内合计缺少487个像元支持，AN和pH各缺1个支持像元，均值按每个变量自己的有效像元计算。历史匹配端点版含181,794个端点；后续候选配对版含111,289对及222,578个唯一端点，二者仅共享133,533个端点。两版本参数不同：前者为环境卡尺0.35、地理半径1,000 m、侵蚀百分位差至少0.50；后者为0.75、500 m、土类内8组、侵蚀百分位差至少0.30。正侵蚀样本、零—正配对与土类—地形湿度分组分别作为补充总体。栅格像元为随机森林预测面，不等于同等数量的独立实测。\n\n"
      "## 4. 指标和模型定义\n\n"
      "暴露为log1p(Aa_20yr)，配对差值为高侵蚀端减低侵蚀端的log1p(Aa)差。物理复合值=(标准化SW−标准化BD)/2；化学复合值=(标准化OM+标准化AN)/2；生物活动=(标准化MBC+标准化RESP)/2。RESP保留原始数值方向，复合值名称使用“生物活动”，不将其解释为单调健康评分。CEC、pH单独建模；CEC不纳入主化学复合值。S1为侵蚀分别关联三维度，S2加入三个维度预测SHI并保留侵蚀直接路径，S3删除侵蚀至SHI的直接路径，S4为侵蚀指向物理维度、物理指向化学维度、化学和物理指向生物活动的有限链式结构，并允许侵蚀关联各维度及直接关联SHI；S5分析单项响应，S6分开估计侵蚀发生与正侵蚀强度。M0为无控制，M1为高程、TWI、NDVI、地表温度和土类，M2为地形与植被—温度PC以及土类，M3为M1加空间坐标多项式。\n\n"
      f"分段路径回归使用标准化OLS和5 km空间块聚类协方差；无效协方差方差不形成置信区间。成熟lavaan CB-SEM在D1–D4总体拟合S1–S3和M0–M3。拟合指数使用常规ML独立观测假设，不含空间稳健校正；空间块协方差仅用于分段路径回归。形成性Mode-B PLS在D1–D4总体比较S1–S3和M0–M3，共176条内生方程；S1估计三个健康维度，S2/S3增加SHI且分别保留/删除侵蚀直达路径。M1–M3控制作为固定单指标构念，同一总体内采用共同样本。项目内权重迭代没有bootstrap区间或空间稳健推断。自定义协方差路径近似作为方法敏感性保留，不计作成熟CB-SEM。所有结果仍属探索性预测面条件关联，不能替代独立实测验证。\n\n"
      "GWR指地理加权回归，用邻近样本估计随地点变化的局部关系；MGWR允许不同环境变量使用不同空间带宽。各模型代号与控制组合定义见本报告术语表。\n\n"
      "## 5. 结果概要\n\n"
      f"共同环境完整像元的核心环境控制模型中，侵蚀与物理、化学、生物活动复合维度的标准化条件路径分别为 {dim_betas['physical']:+.3f}、{dim_betas['chemical']:+.3f}、{dim_betas['biological_activity']:+.3f}。保留三个维度与侵蚀直接路径的SHI模型中，直接路径为 {shi_beta:+.3f}，SHI方程R²={shi_r2:.3f}，在维度路径之外加入侵蚀的ΔR²={d_r2:.4f}。完整参数见结果表和附录图；路径是条件关联分解，不是因果中介。\n\n"
      f"不同土类×TWI组的核心环境控制维度斜率见表。最负估计来自 {('; '.join(f'{r.soil_label} {r.twi_group} {r.dimension}: β={r.beta_standardized:+.3f}' for r in group_top.itertuples()))}。这些组来自响应参与比较的结果导向探索，不构成预设分组的独立检验。\n\n"
      "## 6. 模型比较和路径解释\n\n"
      "在相同样本和SHI路径结构的比较中，加入空间坐标趋势的方案表观R²最高（0.651）；空间趋势更能重建预测面的空间纹理，不能据此认定侵蚀作用更强。不控制环境的方案ΔR²最大（0.017，直接β=−0.157），但未扣除环境差异。核心环境控制方案（R²=0.601、ΔR²=0.0079、直接β=−0.113）较便于解释，不能称为拟合冠军或因果模型。ΔR²是模型内条件预测增量，不是因果贡献。路径总效应/间接效应为条件路径系数乘积，未计算区间，且维度与SHI共享构造信息。PLS外权与路径系数作描述性呈现；CB拟合指标是自定义协方差近似的朴素独立样本描述量，不是空间稳健或成熟软件拟合结果。模型排序限制在相同统计总体、因变量和结构内。\n\n"
      "## 7. 既有全域与空间证据\n\n"
      f"全域Spearman关联为SW {swfull:+.3f}、SHI {shifull:+.3f}、OM {omfull:+.3f}。第二阶段配对差值ρ为SHI {shimat:+.3f}、OM {ommat:+.3f}、SW {swmat:+.3f}；二者统计单位和目标不同，不能把数值差异解释为匹配后的衰减。图B仅显示SHI、OM、SW的局部关系，不含CEC；九项健康结局的GWR加权汇总系数见 `spatial_synthesis/corrected_nine_outcome_gwr_weighted_summary.csv`。标准MGWR在770个空间均衡单元上SHI、OM、SW的R²为0.869、0.696、0.822。OM、SW的MGWR带宽触及上界，不能外推为精确作用半径。\n\n"
      "T1高侵蚀—强负向响应候选区含623个240 m单元，面积等价18.882 km²，平均侵蚀量15.670，SHI均值0.549，局部GWRβ中位−0.567。该结果导向空间候选适合作为现场核查位置，不是已经验证的工程治理区。环境特征应结合土类、灌排、坡面和现场检测共同解释。\n\n"
      "## 8. 关键结论、限制和后续\n\n"
      "目前可以支持的表述是：当前预测栅格显示侵蚀与部分健康指标存在全域和局部负向关联，所选配对版本的端点差值也呈负向关联；不同统计单位的相关数值不用于推断匹配造成的效应衰减。路径系数取决于总体和控制结构。局部系数与压力区揭示空间差异候选，不是经外部验证的因果热点。CEC路径受代数公式耦合，SHI定义尚未溯源，预测面像元不等于独立样点，结果导向候选筛选带来选择偏差；常规CB-SEM拟合指标未处理空间依赖，5 km聚类也不替代空间独立样本外验证。本阶段未重跑PCNM；后续应保留既有PCNM结果作专门对照，并确认SHI构造和CEC训练数据。成熟lavaan CB-SEM已完成代表结构比较，但不构成因果或独立验证。\n\n"
      "## 9. 产物导航\n\n"
      "`results/`含统一SEM路径、拟合、增量解释、直接/间接效应、组间比较和模型排行；`spatial_synthesis/`含GWR/MGWR系数分布、带宽与局部一致性；`figures/`含A—I主图PNG/SVG/PDF和数据源；`reports/`含技术报告、正式中文报告、摘要和汇报提纲；`research_index/`含F–I证据交叉索引。\n"
    )
    (run/"reports/PHASE_J_FINAL_TECHNICAL_REPORT.md").write_text(technical,encoding="utf-8")
    # Full academic narrative with source-backed synthesis and explicit estimand boundaries.
    academic=f"""# 温县耕地土壤侵蚀与土壤健康关联及空间差异分析

## 摘要

本研究综合考察温县耕地实际侵蚀与土壤健康预测面的总体关联、环境匹配后的配对差异、健康维度路径和空间变化。侵蚀量与土壤含水量（SW）、土壤健康指数（SHI）及有机质（OM）的全域Spearman相关分别为{swfull:+.3f}、{shifull:+.3f}和{omfull:+.3f}；既有配对差值分析的相应相关分别为{swmat:+.3f}、{shimat:+.3f}和{ommat:+.3f}。两者使用不同样本、统计单位和变量构造，数值差异不能单独归因于匹配。路径分析中，侵蚀与物理、化学和生物活动维度的标准化条件系数分别为{dim_betas['physical']:+.3f}、{dim_betas['chemical']:+.3f}和{dim_betas['biological_activity']:+.3f}；SHI方程的直接系数为{shi_beta:+.3f}，R²={shi_r2:.3f}，侵蚀带来的条件R²增量为{d_r2:.4f}。局部回归和多尺度回归显示关系存在空间差异。研究使用的是随机森林预测栅格，结果受空间依赖和结果导向候选筛选影响，只支持探索性条件关联。CEC由含侵蚀项的公式计算，其关联存在代数耦合；SHI构造公式与权重尚待核实。本文不作因果解释。

**关键词：**实际侵蚀；土壤健康；环境匹配；空间异质性；局部回归；路径分析

## 一、研究问题

本研究关注：侵蚀与土壤健康指标在全域是否相关；在环境近邻配对后，配对差值是否仍有关联；土类和地形湿度背景下关系是否不同；局部关系如何随位置和空间尺度变化；物理、化学和生物活动指标是否呈现不同响应；加入环境和空间背景变量后路径估计如何变化；不同结构模型的拟合和侵蚀增量如何；哪些结果值得进一步实地核验。

## 二、数据与方法

研究区为河南省温县耕地。侵蚀环境表和健康表各含392,527个栅格记录，像元编号唯一对应，行列及投影坐标逐项一致。CEC按照用户指定的全域最大值公式计算：

CEC = (原CEC − 20 × (Aa20 / 143.00503540039062 + Aa2020 / 48.76557540893555)) / 2.9 + 18。

其中Aa20和Aa2020分别表示两期实际侵蚀量。有效CEC为387,448个像元，范围16.726—26.837，与存储值的逐像元最大绝对差为0。所有图中均标作“CEC”。由于计算式直接含侵蚀量，CEC与侵蚀的统计关系有确定的代数成分，不作为独立侵蚀证据。SHI的原始计算公式和权重未能核实，因此保留现有预测面，不用新CEC重建SHI。

健康结局为随机森林预测值，不是逐像元独立实测。全健康指标完整的像元为384,211个；再要求主要环境变量完整后，进入核心控制模型的像元为293,145个。240 m空间汇总包含7,387个单元和293,633个支撑像元，并复现了既有汇总数据中的单元数、支撑数、侵蚀均值与SHI均值。汇总时各变量按自身有效像元求均值，因此不同变量的像元支持略有差异：CEC和2020年侵蚀量在162个单元中合计缺少487个支撑像元，AN与pH各缺少1个支撑像元。

环境匹配结果来自两个不同的历史版本，本文分别报告，不能合并成同一总体。第一版保留181,794个匹配端点，按高程和地表温度的环境距离卡尺0.35、地理距离上限1,000 m及组内侵蚀百分位差至少0.50构造。第二版为经过候选比较后保留的111,289对端点，共222,578个不重复端点；其环境距离卡尺为0.75、地理上限500 m、每类土壤内分8组，组内侵蚀百分位差至少0.30。两个端点总体仅有133,533个相同像元，前一版独有48,261个，后一版独有89,045个。配对差值来自第二版配对。两版本差异也意味着端点回归和配对差值回归并非同一组像元的两种表达。

正侵蚀分析包含195,013个像元；零—正侵蚀配对为68,172对，两端均正值配对为43,117对。另在三类主要土壤内部按地形湿度分为四个等级，形成12个分层组。不同统计单位各自回答不同问题，不将像元、空间单元、匹配端点和配对差值视为可互换观测。

物理维度按标准化土壤含水量（SW）减去标准化土壤容重（BD）后取平均；化学维度为标准化有机质（OM）与碱解氮（AN）的平均；生物活动维度为标准化微生物量碳（MBC）与土壤呼吸指标（RESP）的平均。RESP保留原始方向，不据此假定其越高越健康。CEC和土壤酸碱度（pH）单独分析，pH不作方向翻转。由于未找到可核验的八项指标方向、适宜范围隶属度函数和SHI权重来源，本研究没有另建“健康方向统一”的第二套指标分值；以上维度仅为透明的描述性复合值。路径回归采用5 km空间块聚类协方差；空间块数量不足时使用HC3。局部地理加权回归（GWR）允许关联随地点变化，多尺度地理加权回归（MGWR）允许不同变量呈现不同空间带宽。本研究复用已保存的GWR和MGWR结果，没有重新估计空间模型。

## 三、主要结果

### 1. 全域与配对差值关联

全域像元中，侵蚀与SW、SHI和OM的秩相关分别为{swfull:+.3f}、{shifull:+.3f}和{omfull:+.3f}。配对差值中，对应系数分别为{swmat:+.3f}、{shimat:+.3f}和{ommat:+.3f}，方向均为负。全域像元相关和配对差值相关的样本、统计单位及变量构造不同，不能把系数绝对值的差异解释为匹配削弱了多少侵蚀效应；配对结果应理解为所选匹配版本中的条件差值关系。剩余关系还可能受未测混杂、预测误差和空间依赖影响。

### 2. 健康维度与结构路径

核心环境控制下，侵蚀与物理、化学、生物活动维度的标准化条件路径分别为{dim_betas['physical']:+.3f}、{dim_betas['chemical']:+.3f}和{dim_betas['biological_activity']:+.3f}。包含这些维度并保留侵蚀—SHI直接路径的模型中，SHI直接系数为{shi_beta:+.3f}，R²为{shi_r2:.3f}，侵蚀条件R²增量为{d_r2:.4f}。另一条探索路径按“侵蚀—物理—化学—生物活动”次序设置维度间关联；路径乘积表示条件关联分解，不是因果中介，且未计算bootstrap区间。物理、化学、生物复合值采用等权构造，不能当作经验证的潜变量。

在相同像元总体和相同SHI路径结构下，加入空间坐标趋势的模型表观R²最高，为0.651；它更能重建预测面的空间纹理，不代表侵蚀因果结构最佳。不加环境控制的模型侵蚀条件R²增量最大，为0.017，直接系数为−0.157，但不能排除环境共变。核心环境控制模型R²为0.601、增量为0.0079、直接系数为−0.113，是当前较容易解释的候选，不能称为确认性或最优因果模型。自定义协方差路径近似在统一标准化尺度上计算描述性拟合量，但并非成熟协方差结构方程软件实现，且拟合指数没有空间稳健校正；形成性PLS没有bootstrap区间。因此这些估计用于探索模型结构，不能代替独立验证。

### 3. 空间差异与候选区域

既有GWR局部标准化系数的像元加权均值，在SHI、OM、SW上分别约为−0.330、−0.556、−0.680；局部共线性收缩模型的相应均值约为−0.218、−0.387、−0.464。770个空间均衡单元上的MGWR模型R²分别为0.869、0.696和0.822。OM与SW的带宽估计达到设定上限约23.78 km，表示模型只能支持较宽尺度变化，不能把该上界当作精确作用半径。

按既有规则识别的高侵蚀且局部负向关联较强候选区包括623个240 m单元，面积等价约18.882 km²，平均侵蚀量约15.670，SHI均值约0.549，局部GWR系数中位数约−0.567。该区域适合用于安排现场核验顺序，不是经过实测验证的危险边界或直接治理区。土壤类型、灌排、坡面和耕作条件仍须现场核实。

### 4. 九项健康结局的整体结果

为避免只依据三项相关较强的指标概括全体土壤健康结局，本研究同时检查土壤健康指数和八项指标。全域像元中，20年平均侵蚀量与九项结局的Spearman秩相关依次为：{full_rho_all}。其中SW、SHI和OM的负向排序关系最明显；CEC、碱解氮、土壤容重及土壤呼吸也为负，但幅度较小；pH和微生物量碳的全域相关相对接近零。这里的系数说明侵蚀排序与预测指标排序共同变化，并没有调整所有环境背景，也不表示侵蚀导致这些变化。

第二版环境匹配配对中，九项健康差值与两端侵蚀差值的秩相关为：{pair_rho_all}。大部分系数靠近零，微生物量碳略为正向。匹配对内的方向并非九项指标一致，而且每对像元共享环境分组与近邻选择条件；这些配对结果代表该版本筛选后的条件差值，不可以与全域系数相减来估计所谓“匹配后衰减”。这也提醒读者：土壤健康不是单一响应，不能只用SHI代替全部物理、化学和生物指标。

CEC单项条件模型在加入核心环境变量后，侵蚀路径系数为{cec_beta:+.3f}，该方程R²为{cec_r2:.3f}。由于CEC定义中包含两期侵蚀量，该负向估计部分来自计算公式本身；无论全域秩相关、配对差值、单项路径还是局部回归，都不能把CEC结果当成独立的侵蚀效应证据。CEC仅作为单项结果列示，不放入主要化学维度合成。

### 5. 环境组差异与样本边界

在三类主要土壤内部按地形湿度划分的十二组中，最负的三条核心环境控制维度路径为：{group_top_text}。这些估计是对各组分别拟合后按系数排序的描述；没有进行正式的跨组系数等值检验，也没有把置信区间差异当作组间显著性检验。因此，它们可提示后续核查哪些组合，不能据此认定该土类或湿度水平已被证明更敏感。地形湿度组的样本规模亦不相等，极端组的估计精度与中间组可能不同。

补充总体还包括195,013个正侵蚀像元、68,172对零侵蚀—正侵蚀配对以及43,117对两端均为正侵蚀的配对。正侵蚀像元用于分析已有侵蚀条件下的侵蚀强度；零—正侵蚀配对用于比较无侵蚀与有侵蚀的状态差异。分析目标并不相同；其结果应回到对应的总体和模型表阅读，不能把这些像元或配对人数简单相加为新的独立样本量。无论采用哪种总体，健康结局都是空间连续预测面，邻近像元之间可能共享训练数据和环境信息。

### 6. 九项结局的局部回归概况

复用的GWR结果中，九项结局的像元加权局部标准化系数均值为：{gwr_beta_all}。加入局部岭惩罚后的相应均值为：{gwr_lcr_all}。九项系数的空间支持量并不完全相同，CEC的GWR汇总基于7,396个240 m单元、294,337个预测像元支持；其GWR加权系数均值约−0.291，局部负系数占98.2%，局部岭收缩后的均值约−0.162、负系数占96.7%。由于公式直接带入Aa，这些CEC局部数字仍属于公式耦合下的描述。

这些系数使用同一暴露标准化方式，有利于比较各指标的局部关联方向，但不同结局的构造、测量含义和预测误差并不相同。GWR-LCR是在局部回归中加入岭惩罚、用于检查局部共线性敏感性的一种模型；它会把系数向零收缩，均值较小不等于关系不存在。图B为便于辨认，只展示SHI、OM与SW三项局部地图；它不展示CEC地图，CEC的九项汇总结果在表格中单独可查。已有MGWR在770个空间均衡单元上估计，其中OM与SW带宽达到23.78 km设定上界，所以目前只能说关系在较宽空间尺度上保持变化，不能从上界反推精确影响半径。

## 四、讨论与结论

现有预测栅格显示，侵蚀与部分健康指标存在稳定的负向空间关联；关联大小随统计单位、指标和控制变量变化。匹配差值分析提供了受限样本中的条件关联，但其与全域秩相关的差值不能被单独解释成匹配的效果。路径模型进一步描述了健康维度和SHI之间的条件关系；CEC的公式性耦合、SHI配方不确定、预测面空间依赖和结果参与候选筛选，均限制了结论强度。

1. **全域主要关系：**SW、SHI和OM与实际侵蚀的全域Spearman相关分别为{swfull:+.3f}、{shifull:+.3f}和{omfull:+.3f}，方向为负。
2. **配对差值关系：**所选配对版本中，SW、SHI和OM的差值相关分别为{swmat:+.3f}、{shimat:+.3f}和{ommat:+.3f}；这些数值不能与全域相关直接作衰减比较。
3. **环境背景差异：**不同土类与地形湿度组的路径估计不同，但尚无正式的多组等值检验，不能断言组间差异已得到验证。
4. **空间格局：**局部系数显示关联随位置变化；部分多尺度带宽达到上限，尚不能确定精确作用范围。
5. **健康维度：**核心环境控制下，侵蚀与物理、化学、生物活动维度的条件系数分别为{dim_betas['physical']:+.3f}、{dim_betas['chemical']:+.3f}和{dim_betas['biological_activity']:+.3f}。
6. **环境与空间控制：**加入不同背景变量后，侵蚀路径和拟合度发生变化；这是条件估计量变化，不表示逐层消除了混杂。
7. **拟合表现：**加入空间坐标趋势的候选模型表观R²最高（0.651），空间趋势也会吸收预测面的空间纹理，不能据此选择因果结构。
8. **侵蚀增量：**不控制环境的模型条件R²增量最大（0.017），其解释受环境共变限制。
9. **解释候选：**核心环境控制模型R²=0.601、侵蚀增量=0.0079、SHI直接路径=−0.113，可作为较易解释的比较对象，不是独立验证。
10. **管理启示：**优先对高侵蚀—强负向响应候选区开展土壤、坡面、灌排和耕作现场核查，再据实测证据确定监测和治理措施。

下一步应核实SHI隶属度方向、权重与训练数据，建立地块实测验证，并在统一空间留出设计下复核路径稳定性；同时将本研究与已有空间特征向量控制结果进行对照。治理决策应依据现场条件，而不由预测图单独指定。"""
    (run/"reports/PHASE_J_ACADEMIC_RESEARCH_REPORT.md").write_text(academic,encoding="utf-8")
    abstract=f"""本研究综合分析河南省温县耕地实际侵蚀与土壤健康预测面的总体关联、环境匹配后的配对差值关系、健康维度路径及空间差异。侵蚀和健康数据库各有392,527条对齐像元记录；依据既定全域最大值公式逐像元复算阳离子交换量（CEC），有效值为387,448个，范围16.726—26.837，最大计算差为0。该公式直接含两期侵蚀量，因此CEC与侵蚀的关联包含代数耦合；所有图件名称均统一写作CEC。土壤健康指数（SHI）原始构建公式和权重尚未查明，故保留现有预测面且不据CEC重算SHI。

全域像元中，侵蚀与土壤含水量（SW）、SHI和有机质（OM）的Spearman相关分别为{swfull:+.3f}、{shifull:+.3f}和{omfull:+.3f}；既有第二版匹配配对的相应差值相关分别为{swmat:+.3f}、{shimat:+.3f}和{ommat:+.3f}。这两类相关的样本、统计单位和变量构造不同，不能据其数值差异推断匹配造成的变化。核心环境控制路径中，侵蚀与物理、化学、生物活动维度的标准化系数分别为{dim_betas['physical']:+.3f}、{dim_betas['chemical']:+.3f}和{dim_betas['biological_activity']:+.3f}；SHI直接路径为{shi_beta:+.3f}，R²={shi_r2:.3f}，侵蚀条件R²增量为{d_r2:.4f}。局部回归和多尺度回归显示空间关联存在差异；一个按既有规则形成的强响应候选区含623个240 m单元，可用于现场核查排序。

九项结局的全域秩相关显示，SW、SHI、OM的负向关联较强，其余指标的关联幅度整体较小；CEC的全域相关为-0.286，单项环境控制路径为-0.312。由于两项估计都使用含Aa的CEC公式，它们只能作当前数据构造下的描述。环境匹配的两版样本也不同：一版为181,794个像元端点，另一版为111,289对端点，二者共享133,533个端点；本研究没有将其视为同一总体。核心控制分析按所需变量筛选，样本为293,145个预测像元。

既有GWR汇总中，CEC的局部标准化系数加权均值为-0.291，局部负系数占98.2%；局部岭收缩模型的均值为-0.162。九项健康结局的局部结果见配套表，图B仅展示SHI、OM和SW的地图。模型比较中，加入空间坐标趋势的SHI候选模型表观R²为0.651，核心环境控制模型R²为0.601，侵蚀条件R²增量为0.0079；这些数值用于解释模型内条件变化，不是因果贡献或独立验证表现。

此外，采用成熟lavaan软件在四类分析总体、三种基础结构和四级控制条件下完成48个协方差结构方程模型，模型均成功收敛。全域样本中，含环境控制且包含侵蚀至SHI直接路径的代表模型R²为0.601；同一结构下加入坐标趋势后表观R²为0.651。部分模型自由度为零，属于饱和或刚好识别结构，其CFI、TLI和RMSEA不能作为模型检验依据；常规最大似然拟合指数也没有修正预测栅格的空间依赖。为此，报告将这些结果用于结构比较，并以带5 km空间块协方差的分段路径区间作为空间稳健性补充，而不将常规CB-SEM指标称为独立验证。

项目内形成性Mode-B PLS在四类总体中对比S1–S3和M0–M3，共176条内生方程；同一总体内所有结构与控制采用共同完整样本。S1只含三个健康维度方程，S2保留侵蚀直达SHI路径，S3移除该直达路径。该估计没有bootstrap区间或空间稳健推断。

研究使用随机森林预测栅格而非逐像元独立实测，且候选筛选参考了健康结果，故结论限于探索性条件关联。CEC的公式耦合、SHI配方不确定、路径估计器限制和空间依赖均需考虑；本研究不作因果解释，也不以候选图单独指定治理区。后续应通过地块实测、独立空间验证和既有空间控制结果对照来评估关系稳定性。实地核验宜覆盖强、弱响应候选区，同时记录采样深度、作物轮作、灌排与耕作管理；点位应按空间块布设，并用独立侵蚀估计核对暴露，避免把相邻预测像元当作相互独立的重复样本。"""
    (run/"reports/PHASE_J_ABSTRACT.md").write_text("# 学术摘要\n\n"+abstract,encoding="utf-8")
    slides=[
      ("1. 研究问题与逻辑","侵蚀—健康关联；区分全域、匹配、空间和路径模型证据。"),
      ("2. 研究区与数据","温县耕地30 m预测栅格；环境侵蚀/健康各392,527条，CEC逐像元复核。"),
      ("3. 分析样本","分别分析全域像元、240 m汇总格网、两版环境匹配样本、正侵蚀像元、零侵蚀与正侵蚀配对，以及不同土类内的地形湿度组。"),
      ("4. CEC与SHI数据边界","CEC公式代数含Aa；SHI公式未溯源，保留候选结果，不重算。"),
      ("5. 全域多方法响应","图A：全域秩相关、环境匹配后的配对差值相关和局部回归系数分面呈现。"),
      ("6. 环境匹配结果","两版匹配的环境距离阈值、地理距离和侵蚀差异要求不同；说明两版样本并非同一总体。配对差值显示关联方向，但与全域像元相关不是同一估计目标。"),
      ("7. 局部空间变化","图B：770个空间均衡单元的局部回归系数；GWR允许系数随位置变化，MGWR还允许不同变量具有不同空间尺度。"),
      ("8. 重点核验候选区","图C：比较高侵蚀且负向关联较强、较弱，以及环境敏感三类候选的地图、环境画像和面积；用于安排现场核验。"),
      ("9. SEM样本与指标","物理、化学、生物活动复合定义，CEC/pH单独；pH保留原向。"),
      ("10. 环境和空间控制","图D/G：从不加控制到加入核心环境、环境主成分和空间趋势，观察SHI拟合和侵蚀路径变化。"),
      ("11. 全域与环境匹配比较","图E/F：分别展示既有匹配端点版和另一匹配方案的配对差值；两套匹配样本不同，端点与配对差值也不是同一统计单位。"),
      ("12. 综合路径与环境组","图H/I：SHI关联路径图及三类土壤内部不同地形湿度分组的路径差异。"),
      ("13. 主要发现","全域和配对差值均见部分负向关联，但统计单位不同，不能把差值解释为匹配造成的衰减；路径和空间结果受数据与模型边界约束。"),
      ("14. 结论与管理","优先核验高侵蚀且负向关联较强的候选区；以土壤实测、坡面和灌排信息确定具体措施。"),
      ("15. 后续验证","确认SHI来源和CEC训练输入；开展独立空间与实测验证；与既有空间特征向量分析结果作专门对照。"),
    ]
    slide_figures=[
      "图A（总体关系）和研究问题示意。",
      "研究区位置图与图C候选区位置图。",
      "总体样本流程图；补充D1–D7统计单位表。",
      "图A及CEC公式边界说明框。",
      "图A。",
      "图A和图E。",
      "图B。",
      "图C。",
      "图H及健康维度定义表。",
      "图D、图G。",
      "图E、图F。",
      "图H、图I。",
      "图A、图B、图H。",
      "图C。",
      "图B、图C及后续实测验证流程图。",
    ]
    (run/"reports/PHASE_J_PRESENTATION_OUTLINE.md").write_text(
      "# 15页学术汇报提纲\n\n"+"\n\n".join(f"## {a}\n\n{b}\n\n**建议图件：** {fig}" for (a,b),fig in zip(slides,slide_figures)),encoding="utf-8")


def make_atlas(run:Path)->None:
    titles={
      "A_multimethod_erosion_health_response":"侵蚀与土壤健康响应：全域、环境匹配与局部回归",
      "B_gwr_mgwr_spatial_patterns":"土壤健康指数、有机质和含水量的局部关系空间分布",
      "C_stress_sensitive_environment_zones":"侵蚀压力候选区及其环境特征",
      "D_sem_control_hierarchy":"加入环境和空间控制后的侵蚀关联变化",
      "E_full_vs_matched_sem":"全样本与环境匹配端点的路径比较",
      "F_erosion_dimension_path_forest":"侵蚀与物理、化学和生物活动维度的关系",
      "G_sem_fit_and_erosion_increment":"SHI模型拟合、侵蚀增量与直接关联",
      "H_best_sem_path_diagram":"侵蚀与健康维度关联SHI的候选路径",
      "I_soil_twi_sem_comparison":"不同土类和地形湿度分组的路径差异",
    }
    captions={
      "A_multimethod_erosion_health_response":"A面板比较全域像元中的Spearman相关和标准化斜率；B面板显示零侵蚀与正侵蚀像元的标准化均值差（Hedges g）；C面板展示第二套环境匹配方案的配对差值相关；下排为SHI、SW和OM的全样本Gaussian GAM响应曲线及点态95%区间。各统计量的单位和估计目标不同，不能直接互换。CEC使用包含Aa的校正公式，其相关存在代数耦合。",
      "B_gwr_mgwr_spatial_patterns":"显示同一组770个空间均衡单元上的局部系数。每一行比较同样本标准GWR和标准MGWR，使用该指标专属且以0为中心的相同色阶；图内标出Aa暴露的近邻带宽、比例尺和指北箭头。MGWR允许各变量使用不同空间带宽。OM和SW的Aa带宽达到预设767近邻上限（中位距离约23.8 km），只能说明模型偏向较宽尺度，不能解释为精确作用半径。",
      "C_stress_sensitive_environment_zones":"地图显示互斥主分区T1–T4及其他单元；三角形和叉号分别表示27潮土TWI第二组、26冲积土TWI最低组的可重叠环境代理条件。右侧逐项比较T1高侵蚀—强负向关联、T2高侵蚀—较弱负向关联、T3低侵蚀—低SHI、T4环境敏感候选四组的环境画像、优势土类比例、面积和局部GWR系数。候选筛选参考健康结果，适合安排探索性现场核查顺序，不是经验证的风险图或治理边界。",
      "D_sem_control_hierarchy":"比较同一批环境变量完整像元中，不同控制条件下侵蚀与健康维度的标准化关联。四种条件依次为：不加入环境变量；加入高程、地形湿度、植被、地表温度和土类；用环境主成分和土类控制；在核心环境变量基础上再加入空间位置趋势。四个面板共用相同横轴范围，便于比较系数大小。SHI的R²表示预测面的拟合程度，不是因果解释比例。",
      "E_full_vs_matched_sem":"两侧使用相同环境变量控制：左侧为全区环境变量完整像元，右侧为既有环境匹配方案的端点像元。该方案要求环境距离不超过0.35、两端相距不超过1,000 m，且同组侵蚀百分位差至少0.50。点和线表示标准化关联及95%区间；这组端点只代表该匹配方案。",
      "F_erosion_dimension_path_forest":"在相同环境变量控制下比较全域像元、240 m汇总单元、既有匹配方案端点和另一套匹配配对差值。两套匹配要求不同：第一套环境距离阈值0.35、地理距离上限1,000 m、组内侵蚀百分位差至少0.50；第二套阈值0.75、距离上限500 m、每类土壤分8组、组内侵蚀百分位差至少0.30。配对差值以每一对像元为单位，不能与端点像元回归系数直接比较。",
      "G_sem_fit_and_erosion_increment":"三幅图依次显示SHI模型拟合度、在已有控制条件上加入侵蚀后的R²增量、侵蚀与SHI的直接标准化关联。控制条件从未加入环境变量，逐步变化为核心环境变量、环境主成分，以及核心环境变量加空间位置趋势。R²增量表示当前模型内的预测变化，不代表因果贡献。",
      "H_best_sem_path_diagram":"路径图显示在控制高程、地形湿度、植被指数、地表温度和土类后，侵蚀与三类健康维度的关联，以及这些维度与SHI的条件关联。β是标准化路径系数；箭头表示模型中设定的条件关系，不代表已经证实的因果中介。SHI原始计算配方和权重尚待确认。",
      "I_soil_twi_sem_comparison":"在冲积土、潮土和脱潮土各自内部，按地形湿度指数从低到高分成四组，再估计侵蚀与物理、化学、生物活动维度的标准化关联。图中每类土的四行依次代表湿度最低、较低、较高和最高组。分组结果受候选筛选影响，不是正式组间差异检验。",
    }
    captions_md="# 核心图图注与阅读说明\n\n"+"\n\n".join(f"## 图{chr(65+i)}｜{titles[k]}\n\n{v}" for i,(k,v) in enumerate(captions.items()))
    (run/"reports/PHASE_J_FIGURE_CAPTIONS.md").write_text(captions_md,encoding="utf-8")
    entries=[]
    for i,(stem,caption) in enumerate(captions.items()):
        entries.append(f"<article><h2>图{chr(65+i)}｜{titles[stem]}</h2><p>{caption}</p><a href='../figures/{stem}.png'><img src='../figures/{stem}.png' alt='{caption}' loading='lazy'></a><p><a href='../figures/{stem}.svg'>矢量图</a> · <a href='../figures/{stem}.pdf'>PDF版</a></p></article>")
    html="""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>温县耕地侵蚀与土壤健康综合分析图集</title><style>body{font-family:'WenQuanYi Zen Hei','Microsoft YaHei',sans-serif;max-width:1200px;margin:2rem auto;padding:0 1rem;color:#202b35}h1{color:#243b53}article{border-top:1px solid #d6dde3;padding:1.2rem 0 2rem}img{display:block;max-width:100%;height:auto;border:1px solid #e1e5e9}a{color:#246b8e}small{color:#5e6c76}</style><body><h1>温县耕地侵蚀与土壤健康综合分析图集</h1><p>本图集收录九幅用途不同、没有重复的主图。图中健康指标来自预测栅格；结果用于探索关联，不等于独立实测验证或因果结论。还可查看<a href="../../../../../docs/FIGURE_ATLAS.html">项目F–I图集</a>和<a href="../../../../../docs/FIGURE_ATLAS_INDEX.md">全项目图集索引</a>。</p>"""+"\n".join(entries)+"</body></html>"
    (run/"reports/PHASE_J_RESEARCH_FIGURE_ATLAS.html").write_text(html,encoding="utf-8")
    index="# 温县耕地侵蚀与土壤健康分析图件目录\n\n| 图 | 图名 | 图注说明 |\n|---|---|---|\n"+"\n".join(f"| {chr(65+i)} | [查看图像](../figures/{k}.png) | {titles[k]}：{captions[k]} |" for i,k in enumerate(captions))+"\n\n[交互图集](PHASE_J_RESEARCH_FIGURE_ATLAS.html)\n"
    (run/"reports/PHASE_J_CORE_FIGURE_INDEX.md").write_text(index,encoding="utf-8")


def finalize_run_provenance(run: Path) -> None:
    basis = {
        "D1_full_pixels": "CEC与健康完整案例；完整案例定义，不按健康效应筛选",
        "D1_common_env_case": "主要环境变量共同完整案例；用于M0–M3同样本比较",
        "D2_240m_blocks": "复现既有Phase I8的8×8环境完整单元支持（历史单元以SHI候选值参与汇总）",
        "D3_M4_unique_pixels": "Phase H沿用Phase F/G历史M4/Aa_20yr端点成员（环境卡尺0.35、地理上限1,000m、组内侵蚀百分位差至少0.50）；按cell_id回连当前健康与CEC值",
        "D4_M4_pair_differences": "Phase H第二阶段候选stage2_Aa_20yr_01（环境卡尺0.75、地理上限500m、土类内8组、组内侵蚀百分位差至少0.30）；按cell_id重算端点差。与D3为不同匹配版本",
        "D5_positive_erosion": "D1共同环境完整案例中Aa_20yr大于0",
        "D6_zero_positive_pairs": "D4中低侵蚀端为0且高侵蚀端大于0的既定配对",
        "D7_soil_TWI_groups": "主土类26/27/42内按各土类TWI秩划分四分位；未按SEM斜率挑选分组",
    }
    count_path=run/"data/population_counts.csv"
    counts=pd.read_csv(count_path)
    counts["selection_basis"] = counts.dataset.map(basis)
    counts["outcome_guided_selection"] = True
    csv_write(counts,count_path)
    audit_path=run/"provenance/population_audit.json"
    audit=json.loads(audit_path.read_text(encoding="utf-8"))
    audit["population_selection_basis"] = basis
    audit["outcome_guided_selection"] = True
    audit_path.write_text(json.dumps(audit,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    config={"phase":"J","mode":"EXPLORATORY_ORACLE","outcome_guided_selection":True,
            "exposure":"log1p(Aa_20yr)","paired_exposure":"high_endpoint_log1p_Aa_minus_low_endpoint_log1p_Aa",
            "cec_formula":"(CEC_old - 20*(Aa_20yr/143.00503540039062 + Aa_2020/48.76557540893555))/2.9 + 18",
            "dimension_definitions":{"physical":"(z_SW-z_BD)/2","chemical":"(z_OM+z_AN)/2",
                                     "biological_activity":"(z_MBC+z_RESP)/2; activity construct, raw response directions retained"},
            "structures":["S1_parallel_dimensions","S2_direct_plus_indirect_SHI","S3_indirect_only_SHI",
                          "S4_chained_health_dimensions","S5_indicator_responses","S6_two_part_erosion"],
            "controls":{"M0":[],"M1":["Elevation","TWI","NDVI","LST","SoilType reference-coded"],
                        "M2":["terrain_PC","vegetation_thermal_PC","SoilType reference-coded"],
                        "M3":["M1","standardized x/y linear and quadratic spatial terms"]},
            "methods":{"piecewise_sem":"standardized regression equations, 5 km block-cluster covariance when supported",
                       "CB_SEM_custom_observed_path":"custom observed-variable covariance path approximation on sample-standardized scale; iid descriptive indices; not a mature package or spatially robust",
                       "CB_SEM_lavaan_ML":"mature lavaan observed-variable covariance-based path models for S1-S3 on representative D1-D4 populations and M0-M3 controls; standard iid maximum-likelihood fit indices; not spatially robust",
                       "PLS_PM_ModeB_formative":"local Mode-B formative weight iteration across S1-S3 and M0-M3 in D1-D4; controls are fixed single-indicator constructs; common rows within each population; no bootstrap intervals or spatially robust inference"},
            "spatial_summary":{"local_coefficient_quantiles":"min, P5, P25, median, P75, P95, max and unit-level negative fraction from saved Phase I coefficients",
                               "strong_negative_overlap":"within each common support, compare lowest 10% local-beta units across SHI, OM and SW; descriptive Jaccard and Spearman statistics",
                               "environment_context":"join saved local coefficients to 240 m unit environment; continuous variables grouped by whole-domain rank quartiles; descriptive only",
                               "model_comparison":"saved GWR, GWR-LCR and MGWR local coefficients; no spatial model refit"},
            "groups":{"soil_codes":{"26":"冲积土","27":"潮土","42":"脱潮土"},"TWI":"within-soil quartiles"},
            "populations":basis,"pcnm":"deferred to Phase K","outcome_guided_selection_objective":"summarize F–I evidence and explore conditional path structure, health dimensions, and environment heterogeneity"}
    (run/"provenance/analysis_config.json").write_text(json.dumps(config,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    manifest_path=run/"provenance/run_manifest.json"
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
        commit=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,check=True,capture_output=True,text=True).stdout.strip()
        status=subprocess.run(["git","status","--short"],cwd=ROOT,check=True,capture_output=True,text=True).stdout.splitlines()
        manifest["git_commit"]=commit
        manifest["git_worktree"]={"state":"dirty" if status else "clean","changed_path_count":len(status),
                                  "analysis_scripts":["experiments/oracle_exploration/phase_j_sem.py",
                                                      "experiments/oracle_exploration/phase_j_synthesis.py",
                                                      "experiments/oracle_exploration/phase_j_lavaan_cbsem.py",
                                                      "experiments/oracle_exploration/phase_j_lavaan_cbsem.R"]}
        manifest["population_selection_basis"]=basis
        manifest["outcome_guided_selection"]=True
        manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


def main()->None:
    ap=argparse.ArgumentParser();ap.add_argument("--run-id",default=RUN_ID);args=ap.parse_args()
    run=ROOT/"outputs/exploratory_oracle/phase_j"/args.run_id
    plt.rcParams.update({"font.family":FONT,"axes.unicode_minus":False,"font.size":9,"figure.dpi":150})
    for d in ["figures/data","spatial_synthesis","research_index","reports"]:(run/d).mkdir(parents=True,exist_ok=True)
    finalize_run_provenance(run)
    spatial=build_spatial_tables(run)
    build_figure_a(run);build_figure_b(run,spatial);build_figure_c(run)
    sem=build_sem_products(run)
    plot_sem_d(run,sem);plot_sem_e(run,sem);plot_sem_f(run,sem);plot_sem_g(run,sem);plot_sem_h(run,sem);plot_sem_i(run,sem)
    crosswalk=build_evidence_crosswalk(run)
    build_report(run,sem,spatial,crosswalk)
    build_cross_stage_deliverables(run,sem,spatial,crosswalk)
    make_atlas(run)
    print(json.dumps({"run_dir":str(run.relative_to(ROOT)),"figures":9,
                      "sem_fit_rows":len(sem["fits"]),"sem_path_rows":len(sem["paths"]),
                      "sem_failed":int(sem["fits"].status.astype(str).str.startswith("failed").sum()),
                      "group_slopes":len(sem["groups"]),"outcome_guided_selection":True},ensure_ascii=False,indent=2))


if __name__=="__main__":main()
