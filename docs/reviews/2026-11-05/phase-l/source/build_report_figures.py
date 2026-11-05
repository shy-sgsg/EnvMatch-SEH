"""Build report-only visual summaries from the saved single-exposure tables.

This script reformats archived result tables; it does not fit or refit models.
"""
from pathlib import Path
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[5]
PK = ROOT / "docs/reviews/2026-11-05/phase-k"
OUT = ROOT / "docs/reviews/2026-11-05/phase-l/figures"
OUT.mkdir(parents=True, exist_ok=True)
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

# 1. Global rank association by outcome.
global_df = pd.read_csv(PK / "figures/A_source_global.csv")
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
save(fig, "L_global_spearman.png", "九项健康结果的全域秩相关排序", "phase-k/figures/A_source_global.csv")

# 2. Matched-pair difference rank association.
matched_df = pd.read_csv(PK / "figures/A_source_matched.csv")
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
save(fig, "M_matched_pair_spearman.png", "九项指标的配对差值秩相关", "phase-k/figures/A_source_matched.csv")

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

# 4. Local coefficient distributions, using all saved local estimates.
local_df = pd.read_csv(PK / "figures/B_source_local_betas.csv")
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
save(fig, "O_local_beta_distribution.png", "九项结果的局部系数分布", "phase-k/figures/B_source_local_betas.csv")

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
save(fig, "P_local_sign_summary.png", "局部系数中位数与负号占比", "phase-k/figures/B_source_local_betas.csv")

# 6. Soil-type slopes: show all groups and outcomes, not only selected strongest groups.
group_df = pd.read_csv(PK / "matching/environment_group_slopes_20yr.csv")
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
save(fig, "Q_soil_group_slopes.png", "土类分组下九项健康结果的标准化配对斜率", "phase-k/matching/environment_group_slopes_20yr.csv")

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

with (OUT.parent / "source/FIGURE_MANIFEST.csv").open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=manifest[0].keys())
    writer.writeheader(); writer.writerows(manifest)
print(f"created {len(manifest)} report figures in {OUT}")
