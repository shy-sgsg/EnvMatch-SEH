"""Build report-only visual summaries from the saved single-exposure tables.

This script reformats archived result tables; it does not fit or refit models.
"""
from pathlib import Path
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.ticker import FormatStrFormatter

ROOT = Path(__file__).resolve().parents[5]
PK = ROOT / "docs/reviews/2026-11-05/phase-k"
CEC_TABLES = ROOT / "docs/reviews/2026-10-31/phase-i/cec-correction/tables"
PHASE_J_LOCAL = ROOT / "docs/reviews/2026-11-02/phase-j/figures/supplementary/data/U_nine_outcome_local_beta_maps_source.csv"
OUT = ROOT / "docs/reviews/2026-11-05/phase-l/figures"
SOURCE = ROOT / "docs/reviews/2026-11-05/phase-l/source"
OUT.mkdir(parents=True, exist_ok=True)
SOURCE.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Noto Sans CJK JP", "Noto Sans CJK JP", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

ORDER = ["SW", "SHI", "OM", "RESP", "AN", "BD", "MBC", "pH", "CEC"]
LABEL = {
    "SHI": "土壤健康指数（SHI）", "AN": "碱解氮（AN）", "OM": "有机质（OM）",
    "CEC": "阳离子交换量（CEC）", "pH": "酸碱度（pH）", "SW": "土壤含水量（SW）",
    "BD": "土壤容重（BD）", "MBC": "微生物量碳（MBC）", "RESP": "土壤呼吸（RESP）",
}
BLUE = "#2c7894"
ORANGE = "#d89a2b"
RED = "#b24b45"
DARK = "#243746"
GRID = "#dfe6ea"
manifest = []

def save(fig, name, title, source):
    fig.tight_layout()
    path = OUT / name
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    manifest.append({"file": name, "title_zh": title, "data_source": source,
                     "exposure": "Aa_20yr", "operation": "table visualization only; no model refit"})

def replace_shared_row(base, corrected, selector, key_columns):
    """Replace only the CEC row while preserving the other archived outcomes."""
    base = base.copy()
    corrected_row = corrected.loc[selector].iloc[0]
    mask = pd.Series(True, index=base.index)
    for key in key_columns:
        mask &= base[key].astype(str).eq(str(corrected_row[key]))
    if int(mask.sum()) != 1:
        raise ValueError(f"Expected exactly one CEC row for {key_columns}, found {int(mask.sum())}")
    for column in set(base.columns).intersection(corrected.columns):
        base.loc[mask, column] = corrected_row[column]
    return base

def save_source(df, filename):
    df.to_csv(SOURCE / filename, index=False, encoding="utf-8-sig")

# Build report-local source tables. The Phase K archive remains unchanged.
global_table = pd.read_csv(PK / "tables/global_associations_20yr.csv")
global_corrected = pd.read_csv(CEC_TABLES / "global_associations.csv")
global_selector = ((global_corrected.population == "mask_all") &
                   (global_corrected.exposure == "Aa_20yr") &
                   (global_corrected.outcome == "CEC"))
global_table = replace_shared_row(
    global_table, global_corrected, global_selector,
    ["population", "exposure", "outcome"])
save_source(global_table, "global_associations_20yr_corrected_CEC.csv")

global_df = pd.read_csv(PK / "figures/A_source_global.csv")
global_row = global_corrected.loc[global_selector]
global_df = replace_shared_row(
    global_df, global_row, pd.Series(True, index=global_row.index),
    ["population", "exposure", "outcome"])
save_source(global_df, "global_figure_source_corrected_CEC.csv")

matched_table = pd.read_csv(PK / "matching/matched_pair_associations_20yr.csv")
matched_corrected = pd.read_csv(CEC_TABLES / "matched_pair_associations.csv")
matched_selector = ((matched_corrected.exposure == "Aa_20yr") &
                    (matched_corrected.candidate_id == "stage2_Aa_20yr_01") &
                    (matched_corrected.population == "all_erosion") &
                    (matched_corrected.outcome == "CEC"))
matched_table = replace_shared_row(
    matched_table, matched_corrected, matched_selector,
    ["exposure", "candidate_id", "population", "outcome"])
save_source(matched_table, "matched_pair_associations_20yr_corrected_CEC.csv")

matched_df = pd.read_csv(PK / "figures/A_source_matched.csv")
matched_row = matched_corrected.loc[matched_selector].copy()
matched_row["analysis_unit"] = "matched_pair"
matched_df = replace_shared_row(
    matched_df, matched_row, pd.Series(True, index=matched_row.index),
    ["exposure", "candidate_id", "population", "outcome"])
save_source(matched_df, "matched_figure_source_corrected_CEC.csv")

group_df = pd.read_csv(PK / "matching/environment_group_slopes_20yr.csv")
group_corrected = pd.read_csv(CEC_TABLES / "environment_group_slopes_m4_Aa20.csv")
group_mask = ((group_corrected.exposure == "Aa_20yr") &
              (group_corrected.analysis_unit == "matched_pair_difference") &
              (group_corrected.group_column == "soil_type_group") &
              (group_corrected.group_method == "soil_type") &
              (group_corrected.outcome == "CEC"))
corrected_group_rows = group_corrected.loc[group_mask]
for _, corrected_row in corrected_group_rows.iterrows():
    selector = pd.Series(True, index=group_df.index)
    for key in ["analysis_unit", "method", "exposure", "outcome", "group_column", "group_method", "group_label"]:
        value = corrected_row[key]
        selector &= group_df[key].astype(str).eq(str(value))
    if int(selector.sum()) != 1:
        raise ValueError(f"Expected one corrected CEC soil-type row for {corrected_row.group_label}")
    for column in set(group_df.columns).intersection(group_corrected.columns):
        group_df.loc[selector, column] = corrected_row[column]
save_source(group_df, "environment_group_slopes_20yr_corrected_CEC.csv")

local_df = pd.read_csv(PK / "figures/B_source_local_betas.csv")
local_corrected = pd.read_csv(PHASE_J_LOCAL)
local_corrected = local_corrected[local_corrected.outcome == "CEC"].copy()
local_base_cec = local_df[local_df.outcome == "CEC"].copy()
check = local_base_cec.merge(local_corrected, on=["tile_row", "tile_col"], suffixes=("_base", "_corrected"), validate="one_to_one")
if len(check) != 7396 or not np.allclose(check.local_beta_standardized_base, check.local_beta_standardized_corrected, atol=1e-7):
    raise ValueError("CEC local coefficients do not match the saved corrected local source")
local_df = local_df[local_df.outcome != "CEC"]
local_df = pd.concat([local_df, local_corrected[local_df.columns]], ignore_index=True)
save_source(local_df, "local_betas_20yr_corrected_CEC.csv")

# 1. Global rank association by outcome.
g = global_df[(global_df.exposure == "Aa_20yr") & (global_df.population == "mask_all")].copy()
g["outcome"] = pd.Categorical(g.outcome, categories=ORDER, ordered=True)
g = g.sort_values("outcome")
fig, ax = plt.subplots(figsize=(9.2, 5.8))
y = np.arange(len(g))
colors = [RED if v < 0 else ORANGE for v in g.spearman_rho]
ax.barh(y, g.spearman_rho, color=colors, height=.68)
ax.axvline(0, color=DARK, lw=1)
ax.set_yticks(y, [LABEL[str(v)] for v in g.outcome])
ax.invert_yaxis(); ax.set_xlabel("Spearman 秩相关系数（ρ）")
ax.set_title("九项健康结果与20年平均侵蚀的全域关联")
ax.grid(axis="x", color=GRID, lw=.8); ax.set_axisbelow(True)
for yi, val in zip(y, g.spearman_rho):
    ax.text(val + (0.012 if val >= 0 else -0.012), yi, f"{val:+.3f}",
            va="center", ha="left" if val >= 0 else "right", color=DARK, fontsize=9)
ax.set_xlim(-.72, .32)
save(fig, "L_global_spearman.png", "九项健康结果的全域秩相关排序", "Phase K全域源表；CEC由Phase I CEC校正重分析覆盖")

# 2. Matched-pair difference rank association.
m = matched_df[(matched_df.exposure == "Aa_20yr") &
               (matched_df.candidate_id == "stage2_Aa_20yr_01") &
               (matched_df.population == "all_erosion")].copy()
m["outcome"] = pd.Categorical(m.outcome, categories=ORDER, ordered=True)
m = m.sort_values("outcome")
fig, ax = plt.subplots(figsize=(9.2, 5.8))
y = np.arange(len(m)); colors = [RED if v < 0 else ORANGE for v in m.spearman_rho]
ax.barh(y, m.spearman_rho, color=colors, height=.68)
ax.axvline(0, color=DARK, lw=1)
ax.set_yticks(y, [LABEL[str(v)] for v in m.outcome]); ax.invert_yaxis()
ax.set_xlabel("高侵蚀端减低侵蚀端：Spearman ρ")
ax.set_title("当前环境匹配候选中的九项配对差值关联")
ax.grid(axis="x", color=GRID, lw=.8); ax.set_axisbelow(True)
for yi, val, n in zip(y, m.spearman_rho, m.n_complete):
    ax.text(val + (0.009 if val >= 0 else -0.009), yi, f"{val:+.3f}  n={int(n):,}",
            va="center", ha="left" if val >= 0 else "right", color=DARK, fontsize=8)
ax.set_xlim(-.42, .26)
save(fig, "M_matched_pair_spearman.png", "九项指标的配对差值秩相关", "Phase K匹配源表；CEC由Phase I CEC校正重分析覆盖")

# 3. Three GAM curves, direct use of the stored fitted curves.
gam_df = pd.read_csv(PK / "figures/A_source_gam.csv")
gam = gam_df[(gam_df.exposure == "Aa_20yr") & (gam_df.outcome.isin(["SHI", "OM", "SW"]))].copy()
fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.6))
for ax, outcome in zip(axes, ["SHI", "OM", "SW"]):
    d = gam[gam.outcome == outcome].sort_values("log1p_exposure")
    x = d.log1p_exposure.to_numpy(); yv = d.predicted_outcome.to_numpy()
    lo = d.pointwise_ci95_low.to_numpy(); hi = d.pointwise_ci95_high.to_numpy()
    ax.fill_between(x, lo, hi, color=BLUE, alpha=.18, linewidth=0)
    ax.plot(x, yv, color=BLUE, lw=2.5)
    ax.set_title(LABEL[outcome]); ax.set_xlabel("log(1 + 20年平均侵蚀)")
    ax.grid(color=GRID, lw=.7); ax.set_axisbelow(True)
axes[0].set_ylabel("拟合健康结果值")
fig.suptitle("三项重点结果的全样本平滑响应与95%区间", y=1.03, fontsize=15)
save(fig, "N_gam_three_outcomes.png", "三项重点结局的全样本GAM响应曲线", "phase-k/figures/A_source_gam.csv")

# Report composite: redraw the Phase K overview with the corrected CEC row.
fig, axs = plt.subplots(1, 3, figsize=(17, 5.5))
g0 = global_df.set_index("outcome").reindex(ORDER)
m0 = matched_df.set_index("outcome").reindex(ORDER)
y = np.arange(len(ORDER)); h = .36
axs[0].barh(y-h/2, g0.spearman_rho.astype(float), height=h, color=BLUE, label="全域像元")
axs[0].barh(y+h/2, m0.spearman_rho.astype(float), height=h, color=ORANGE, label="匹配配对差值")
axs[0].set_yticks(y, [LABEL[o] for o in ORDER]); axs[0].invert_yaxis()
axs[0].axvline(0, color=DARK, lw=.8); axs[0].set_xlim(-.70, .22)
axs[0].set_xlabel("Spearman 秩相关系数（ρ）")
axs[0].set_title("总体排序与匹配差值")
axs[0].legend(frameon=False, fontsize=8)
ax_shi = axs[1].twinx(); handles = []
for outcome, color in [("SHI", BLUE), ("OM", "#5d9474"), ("SW", ORANGE)]:
    d = gam[gam.outcome == outcome].sort_values("log1p_exposure")
    x = d.log1p_exposure.to_numpy(); pred = d.predicted_outcome.to_numpy()
    lo = d.pointwise_ci95_low.to_numpy(); hi = d.pointwise_ci95_high.to_numpy()
    target = ax_shi if outcome == "SHI" else axs[1]
    line, = target.plot(x, pred, color=color, lw=2, label=LABEL[outcome])
    target.fill_between(x, lo, hi, color=color, alpha=.12)
    handles.append(line)
axs[1].set_xlabel("log(1 + 20年平均侵蚀)"); axs[1].set_ylabel("OM、SW拟合值")
axs[1].set_title("三项重点结局的非线性曲线")
ax_shi.set_ylabel("SHI拟合值"); ax_shi.set_ylim(.50, .68)
ax_shi.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
axs[1].legend(handles, [line.get_label() for line in handles], frameon=False, fontsize=8)
axs[2].axis("off")
axs[2].text(.02, .94, "图示要点", fontsize=15, fontweight="bold", va="top")
axs[2].text(.02, .77, "左：全域像元与匹配配对均展示九项结果，\nCEC在两个尺度上均为负向关联。", fontsize=11, va="top")
axs[2].text(.02, .53, "中：SHI、有机质和土壤含水量的平均曲线\n展现关系的弯曲形状。", fontsize=11, va="top")
axs[2].text(.02, .29, "浅色带表示95%逐点区间；总体、配对\n和曲线各自回答不同问题。", fontsize=11, va="top")
fig.suptitle("20年平均实际侵蚀与土壤健康：总体、匹配与非线性证据", fontsize=16)
save(fig, "V_global_match_gam_corrected.png", "总体关联、匹配差值和三项重点曲线", "Phase L校正后全域与匹配源表及Phase K GAM曲线源表")

# 4. Local coefficient distributions, using all saved local estimates.
fig, ax = plt.subplots(figsize=(12, 6.2))
vals = [local_df.loc[local_df.outcome == o, "local_beta_standardized"].dropna().to_numpy() for o in ORDER]
parts = ax.violinplot(vals, positions=np.arange(1, 10), widths=.82, showmeans=False, showmedians=True,
                      showextrema=False, quantiles=[[.1, .9]] * 9)
for body in parts["bodies"]: body.set_facecolor(BLUE); body.set_alpha(.5); body.set_edgecolor(BLUE)
parts["cmedians"].set_color(RED); parts["cmedians"].set_linewidth(1.6)
ax.axhline(0, color=DARK, lw=1, ls="--")
ax.set_xticks(np.arange(1, 10), [LABEL[o] for o in ORDER], rotation=25, ha="right")
ax.set_ylabel("局部标准化系数（β）"); ax.set_title("九项健康结果的局部系数分布（240米单元）")
ax.grid(axis="y", color=GRID, lw=.7); ax.set_axisbelow(True)
save(fig, "O_local_beta_distribution.png", "九项结果的局部系数分布", "Phase K局部源表；CEC与Phase J校正后局部系数逐单元核对")

# 5. Median coefficient and negative fraction, summarized from the same local table.
summary = []
for o in ORDER:
    v = local_df.loc[local_df.outcome == o, "local_beta_standardized"].dropna().to_numpy()
    summary.append((o, float(np.median(v)), float(np.mean(v < 0)), len(v)))
s = pd.DataFrame(summary, columns=["outcome", "median", "negative_fraction", "n"])
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.8), sharey=True)
y = np.arange(len(s))
ax1.barh(y, s["median"], color=[RED if v < 0 else ORANGE for v in s["median"]], height=.68)
ax1.axvline(0, color=DARK, lw=1); ax1.set_yticks(y, [LABEL[o] for o in s.outcome]); ax1.invert_yaxis()
ax1.set_xlabel("局部系数中位数 β"); ax1.set_title("系数中位数"); ax1.grid(axis="x", color=GRID, lw=.7)
ax2.barh(y, s.negative_fraction * 100, color=BLUE, height=.68)
ax2.set_xlim(0, 105); ax2.set_xlabel("局部系数为负的单元比例（%）"); ax2.set_title("负号比例")
ax2.grid(axis="x", color=GRID, lw=.7)
for yi, val, n in zip(y, s.negative_fraction, s.n):
    ax2.text(val * 100 + 1.2, yi, f"{val*100:.1f}%  n={n:,}", va="center", fontsize=8, color=DARK)
fig.suptitle("局部系数的方向和强弱是两种不同信息", y=1.02, fontsize=15)
save(fig, "P_local_sign_summary.png", "局部系数中位数与负号占比", "Phase K局部源表；CEC与Phase J校正后局部系数逐单元核对")

# Corrected nine-outcome local coefficient map for the report package.
plot_order = ["SHI", "AN", "OM", "CEC", "pH", "SW", "BD", "MBC", "RESP"]
fig, axes = plt.subplots(3, 3, figsize=(13, 10))
for ax, outcome in zip(axes.flat, plot_order):
    d = local_df[local_df.outcome == outcome]
    sc = ax.scatter(d.x, d.y, c=d.local_beta_standardized, s=2.2, cmap="RdBu_r",
                    vmin=-.75, vmax=.75, linewidths=0, rasterized=True)
    ax.set_title(LABEL[outcome], fontsize=10); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
cax = fig.add_axes([.84, .20, .018, .60])
cb = fig.colorbar(sc, cax=cax)
cb.set_label("局部标准化回归系数 β")
fig.suptitle("九项健康结果的局部回归系数空间分布", fontsize=15)
fig.subplots_adjust(left=.04, right=.80, wspace=.08, hspace=.20)
fig.savefig(OUT / "X_nine_outcome_local_maps_corrected.png", dpi=220, bbox_inches="tight")
plt.close(fig)
manifest.append({"file": "X_nine_outcome_local_maps_corrected.png", "title_zh": "九项结果的局部回归系数地图",
                 "data_source": "Phase K局部源表；CEC由Phase J校正后局部系数逐单元替换",
                 "exposure": "Aa_20yr", "operation": "saved local estimates plotted; no model refit"})

# 6. Soil-type slopes: show all groups and outcomes, not only selected strongest groups.
q = group_df[(group_df.exposure == "Aa_20yr") &
             (group_df.analysis_unit == "matched_pair_difference") &
             (group_df.group_column == "soil_type_group") &
             (group_df.group_method == "soil_type")].copy()
q = q.drop_duplicates(["outcome", "group_label"])
pivot = q.pivot(index="outcome", columns="group_label", values="standardized_effect").reindex(ORDER)
cols = ["soil_26", "soil_27", "soil_42"]
pivot = pivot.reindex(columns=cols)
fig, ax = plt.subplots(figsize=(8.6, 6.2))
arr = pivot.to_numpy(dtype=float)
lim = np.nanmax(np.abs(arr)); im = ax.imshow(arr, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
ax.set_xticks(np.arange(3), ["26 冲积土", "27 潮土", "42 脱潮土"])
ax.set_yticks(np.arange(9), [LABEL[o] for o in ORDER])
for i in range(arr.shape[0]):
    for j in range(arr.shape[1]):
        if np.isfinite(arr[i, j]): ax.text(j, i, f"{arr[i,j]:+.2f}", ha="center", va="center", fontsize=9, color="#172b35")
ax.set_title("三类土壤内九项结果的配对斜率")
cb = fig.colorbar(im, ax=ax, shrink=.82); cb.set_label("标准化斜率（β）")
save(fig, "Q_soil_group_slopes.png", "土类分组下九项健康结果的标准化配对斜率", "Phase K土类斜率；CEC由Phase I校正后M4源表覆盖")

# 7. Control hierarchy of the stored D1 path results.
rank_df = pd.read_csv(PK / "sem/sem_model_rankings_20yr.csv")
r = rank_df[(rank_df.dataset == "D1_common_env_case") & (rank_df.structure == "S2_direct_plus_indirect_SHI")].copy()
labels = {"M0": "不加入环境变量", "M2": "环境主成分控制", "M1": "核心环境控制", "M3": "核心环境+空间趋势"}
r["label"] = r.control.map(labels)
order = ["M0", "M2", "M1", "M3"]
r["control"] = pd.Categorical(r.control, categories=order, ordered=True); r = r.sort_values("control")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 5.4))
y = np.arange(len(r)); ax1.axvline(0, color=DARK, lw=1)
ax1.scatter(r.direct_erosion_beta_to_SHI, y, s=80, color=RED, zorder=3)
for yi, row in enumerate(r.itertuples()): ax1.text(row.direct_erosion_beta_to_SHI - .004, yi, f"{row.direct_erosion_beta_to_SHI:.3f}", ha="right", va="center", color=DARK)
ax1.set_yticks(y, r.label); ax1.invert_yaxis(); ax1.set_xlim(-.21, .04); ax1.set_xlabel("侵蚀到SHI的标准化路径 β")
ax1.set_title("条件路径")
ax2.barh(y, r.direct_erosion_delta_R2, color=BLUE, height=.6)
ax2.set_yticks(y, r.label); ax2.invert_yaxis(); ax2.set_xlabel("加入侵蚀后的条件 ΔR²")
ax2.set_title("新增拟合信息"); ax2.grid(axis="x", color=GRID, lw=.7)
for yi, val in zip(y, r.direct_erosion_delta_R2): ax2.text(val + .0005, yi, f"{val:.4f}", va="center", fontsize=9, color=DARK)
fig.suptitle("同一分析总体下逐步调整环境背景", y=1.02, fontsize=15)
save(fig, "R_sem_control_paths.png", "控制变量增加时侵蚀到SHI的路径和拟合增量", "phase-k/sem/sem_model_rankings_20yr.csv")

# 8. Core dimension paths with block-robust intervals.
paths = pd.read_csv(PK / "sem/sem_path_coefficients_20yr.csv")
d = paths[(paths.dataset == "D1_common_env_case") & (paths.method == "piecewise_sem") &
          (paths.control == "M1") & (paths["from"] == "z_E")].copy()
keep = {"physical": "物理维度", "chemical": "化学维度", "biological_activity": "生物活动维度", "z_SHI": "SHI综合指数"}
d = d[d.to.isin(keep)].copy()
d = d.drop_duplicates(["structure", "to"])
d["label"] = d.to.map(keep)
want = ["物理维度", "化学维度", "生物活动维度", "SHI综合指数"]
d["label"] = pd.Categorical(d.label, categories=want, ordered=True); d = d.sort_values("label")
fig, ax = plt.subplots(figsize=(9.5, 4.7)); y = np.arange(len(d))
ax.errorbar(d.beta_standardized, y,
            xerr=[d.beta_standardized - d.ci_low, d.ci_high - d.beta_standardized],
            fmt="o", color=BLUE, ecolor=BLUE, elinewidth=2, capsize=4, markersize=7)
ax.axvline(0, color=DARK, lw=1, ls="--")
ax.set_yticks(y, d.label); ax.invert_yaxis(); ax.set_xlabel("标准化路径 β（95%空间块区间）")
ax.set_title("核心环境控制下的侵蚀—健康条件路径")
ax.grid(axis="x", color=GRID, lw=.7); ax.set_axisbelow(True)
for yi, row in enumerate(d.itertuples()): ax.text(row.beta_standardized - .015, yi, f"{row.beta_standardized:.3f}", ha="right", va="center", color=DARK, fontsize=9)
save(fig, "S_sem_dimension_paths.png", "物理、化学、生物活动和SHI的维度路径比较", "phase-k/sem/sem_path_coefficients_20yr.csv")

# 9. Candidate-zone profile, descriptive only.
zones = pd.read_csv(PK / "spatial/stress_zone_profiles_20yr.csv")
fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.2))
fields = [("mean_Aa_20yr", "平均20年侵蚀"), ("mean_SHI", "平均SHI"), ("median_GWR_beta", "局部系数中位数")]
for ax, (field, title) in zip(axes, fields):
    vals = zones[field].to_numpy(); names = zones.zone.astype(str).tolist()
    ax.barh(np.arange(len(zones)), vals, color=[RED, BLUE, ORANGE, "#6d9f73"])
    ax.set_yticks(np.arange(len(zones)), names); ax.invert_yaxis(); ax.set_title(title)
    ax.grid(axis="x", color=GRID, lw=.7); ax.set_axisbelow(True)
    for yi, val in enumerate(vals): ax.text(val + (0.02 if val >= 0 else -0.02), yi, f"{val:.3f}", va="center", ha="left" if val >= 0 else "right", fontsize=8, color=DARK)
fig.suptitle("四类现场核查候选区的描述性画像", y=1.02, fontsize=15)
save(fig, "T_zone_profiles.png", "四类空间候选区的指标画像", "phase-k/spatial/stress_zone_profiles_20yr.csv")

# 10. Analysis-unit flow: a conceptual guide to sample supports.
fig, ax = plt.subplots(figsize=(12, 7)); ax.set_xlim(0, 12); ax.set_ylim(0, 8); ax.axis("off")
boxes = [
    (0.5, 5.8, 3.0, 1.2, "30米地图位置\n约392,527条记录", "#e8f1f5"),
    (4.5, 5.8, 3.0, 1.2, "全域有效样本\n约384,737–388,092格", "#e8f1f5"),
    (8.5, 5.8, 3.0, 1.2, "匹配差值\n111,289对候选", "#fbf0dc"),
    (0.5, 2.8, 3.0, 1.2, "局部空间汇总\n约7,387–7,408个240米单元", "#e8f1f5"),
    (4.5, 2.8, 3.0, 1.2, "GWR与MGWR对照\n770个空间均衡点", "#e8f1f5"),
    (8.5, 2.8, 3.0, 1.2, "路径模型\n293,145格／35个5公里块", "#e8f1f5"),
]
for x, y, w, h, label, color in boxes:
    p = FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.04,rounding_size=0.12",fc=color,ec="#55717f",lw=1.3)
    ax.add_patch(p); ax.text(x+w/2,y+h/2,label,ha="center",va="center",fontsize=10,color=DARK)
for start,end in [((3.5,6.4),(4.5,6.4)),((7.5,6.4),(8.5,6.4)),((2,5.8),(2,4.0)),((6,5.8),(6,4.0)),((10,5.8),(10,4.0))]:
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle="-|>",mutation_scale=14,color="#55717f",lw=1.4))
ax.text(6,1.55,"这些数字对应不同分析单位，彼此不能相加；地图像元、配对、汇总单元和空间块都不是独立田间样本。",ha="center",va="center",fontsize=11,color=DARK,wrap=True)
ax.text(6,.8,"图示为已保存结果的支持尺度，不代表数据新增或重新拟合。",ha="center",va="center",fontsize=9,color="#52646d")
ax.set_title("同一批空间预测资料如何进入不同分析",fontsize=15,pad=16)
save(fig, "U_analysis_units_flow.png", "不同分析使用的样本单位与支持尺度", "phase-k result index and source tables")

# Corrected evidence-chain graphic for the main report.
fig, ax = plt.subplots(figsize=(16, 6.1))
ax.set_xlim(0, 16); ax.set_ylim(0, 6.1); ax.axis("off")
cards = [
    ("① 全域排序", "约38.7万有效像元", "SW   −0.584\nSHI  −0.469\nOM   −0.415\nCEC  −0.286", "总体关联强弱"),
    ("② 环境相近配对", "111,289对候选", "SW   −0.272\nSHI  −0.144\nOM   −0.115\nCEC  −0.100", "相近背景下仍见负向"),
    ("③ 局部空间模型", "约7,400个空间单元", "SW局部中位数 −0.486\nOM局部中位数 −0.395\nSHI局部中位数 −0.222\nCEC局部中位数 −0.152", "把平均关系定位到地图"),
    ("④ 环境控制路径", "293,145像元；35个5 km块", "侵蚀→SHI  β = −0.113\n加入侵蚀后 ΔR² = 0.0079", "拆解条件关系"),
    ("⑤ 实地核查候选", "623个240 m单元", "面积等价 18.882 km²\n平均侵蚀 15.67\n平均SHI 0.549", "形成可执行的采样线索"),
]
palette = ["#e8f1f5", "#edf4ef", "#fff2df", "#f0ecf8", "#f8ecea"]
for i, (title, count, values, footer) in enumerate(cards):
    x = .15 + i * 3.18
    box = FancyBboxPatch((x, 1.05), 3.00, 4.18,
                         boxstyle="round,pad=0.06,rounding_size=0.16",
                         fc=palette[i], ec="#55717f", lw=1.4)
    ax.add_patch(box)
    ax.text(x + 1.5, 4.78, title, ha="center", va="center", fontsize=13,
            color=DARK, fontweight="bold")
    ax.text(x + 1.5, 4.22, count, ha="center", va="center", fontsize=10.5, color=DARK)
    ax.text(x + 1.5, 3.10, values, ha="center", va="center", fontsize=10.5,
            color=DARK, linespacing=1.45)
    ax.text(x + 1.5, 1.52, footer, ha="center", va="center", fontsize=10,
            color="#52646d", fontweight="bold")
    if i < len(cards) - 1:
        ax.add_patch(FancyArrowPatch((x + 3.01, 3.14), (x + 3.16, 3.14),
                                    arrowstyle="-|>", mutation_scale=14,
                                    color="#55717f", lw=1.4))
ax.text(8, 5.82, "从全域关联到空间异质性：一条可复核、可落地的多尺度证据链",
        ha="center", va="center", fontsize=17, color=DARK, fontweight="bold")
ax.text(8, .42,
        "方法组合把‘是否相关’推进为‘相近背景下是否仍相关、哪些地点更明显、怎样安排实地核查’；CEC全域与匹配结果均为负向。",
        ha="center", va="center", fontsize=10, color="#52646d")
fig.savefig(OUT / "Y_integrated_evidence_corrected.png", dpi=220, bbox_inches="tight")
plt.close(fig)
manifest.append({"file": "Y_integrated_evidence_corrected.png", "title_zh": "从总体关系到空间候选的综合证据图",
                 "data_source": "Phase L corrected CEC sources plus Phase K single-exposure summaries",
                 "exposure": "Aa_20yr", "operation": "saved summary values visualized; no model refit"})

with (OUT.parent / "source/FIGURE_MANIFEST.csv").open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=manifest[0].keys())
    writer.writeheader(); writer.writerows(manifest)
print(f"created {len(manifest)} report figures in {OUT}")
