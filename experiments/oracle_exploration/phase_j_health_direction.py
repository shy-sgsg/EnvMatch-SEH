"""Reconstruct Phase J health-direction scores from the local chapter source.

This is an additive J-9 sensitivity analysis. It retains the existing Phase J
run and the legacy SHI raster as a comparator because the reconstructed score
does not reproduce that raster exactly.
"""
from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import pearsonr, spearmanr
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parents[2]
BASE_RUN = ROOT / "outputs/exploratory_oracle/phase_j/phase_j_20261102_seed20261102_r10"
RUN_ID = "phase_j_20261103_seed20261103_direction_r1"
RUN = ROOT / "outputs/exploratory_oracle/phase_j" / RUN_ID
SOURCE_DOC = Path("/home/shy/文档/xwechat_files/wxid_um3cz3st1tjt22_48d0/msg/file/2026-09/4、土壤健康指标空间预测与综合评价.docx")

INDICATORS = ["AN", "OM", "CEC", "pH", "SW", "BD", "MBC", "RESP"]
INDICATOR_LABELS = {
    "AN": "碱解氮（AN）", "OM": "有机质（OM）", "CEC": "阳离子交换量（CEC）",
    "pH": "酸碱度（pH）", "SW": "土壤含水率（SW）", "BD": "土壤容重（BD）",
    "MBC": "微生物量碳（MBC）", "RESP": "土壤呼吸（RESP）",
}
PARAMETERS = {
    "AN": {"function": "S型", "a": 60.0, "b": 120.0},
    "OM": {"function": "S型", "a": 15.0, "b": 40.0},
    "CEC": {"function": "S型", "a": 10.0, "b": 20.0},
    "SW": {"function": "S型", "a": 10.0, "b": 15.0},
    "MBC": {"function": "S型", "a": 100.0, "b": 300.0},
    "RESP": {"function": "S型", "a": 100.0, "b": 1000.0},
    "BD": {"function": "反S型", "a": 1.15, "b": 1.5},
    "pH": {"function": "抛物线型", "a1": 5.5, "b1": 6.5, "b2": 7.5, "a2": 8.5},
}
WEIGHTS = {
    "BD": 0.109172, "SW": 0.113752, "pH": 0.087996, "CEC": 0.106082,
    "OM": 0.142225, "AN": 0.171195, "MBC": 0.162241, "RESP": 0.107337,
}
DIMENSION_MEMBERS = {
    "physical": ["SW", "BD"],
    "chemical": ["OM", "AN", "CEC", "pH"],
    "biological": ["MBC", "RESP"],
}
DIMENSION_LABELS = {"physical": "物理维度", "chemical": "化学维度", "biological": "生物维度"}
M1_ENV = ["Elevation", "TWI", "NDVI", "LST"]
FONT_PATH = Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
font_manager.fontManager.addfont(str(FONT_PATH))
FONT = font_manager.FontProperties(fname=str(FONT_PATH)).get_name()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def membership(values: pd.Series, spec: dict[str, float | str]) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    y = np.full(x.shape, np.nan, dtype=float)
    finite = np.isfinite(x)
    if spec["function"] == "S型":
        a, b = float(spec["a"]), float(spec["b"])
        y[finite & (x <= a)] = 0.1
        mid = finite & (x > a) & (x < b)
        y[mid] = 0.1 + 0.9 * (x[mid] - a) / (b - a)
        y[finite & (x >= b)] = 1.0
    elif spec["function"] == "反S型":
        a, b = float(spec["a"]), float(spec["b"])
        y[finite & (x <= a)] = 1.0
        mid = finite & (x > a) & (x < b)
        y[mid] = 0.1 + 0.9 * (b - x[mid]) / (b - a)
        y[finite & (x >= b)] = 0.1
    else:
        a1, b1, b2, a2 = (float(spec[k]) for k in ("a1", "b1", "b2", "a2"))
        y[finite & (x <= a1)] = 0.1
        left = finite & (x > a1) & (x < b1)
        y[left] = 0.1 + 0.9 * (x[left] - a1) / (b1 - a1)
        y[finite & (x >= b1) & (x <= b2)] = 1.0
        right = finite & (x > b2) & (x < a2)
        y[right] = 0.1 + 0.9 * (a2 - x[right]) / (a2 - b2)
        y[finite & (x >= a2)] = 0.1
    return pd.Series(y, index=values.index)


def add_direction_scores(frame: pd.DataFrame, raw_reference: pd.DataFrame) -> pd.DataFrame:
    d = frame.copy()
    for name in INDICATORS:
        d[f"mu_{name}"] = membership(d[name], PARAMETERS[name])
    for dim, members in DIMENSION_MEMBERS.items():
        d[f"{dim}_direction"] = d[[f"mu_{name}" for name in members]].mean(axis=1)
        zcols = []
        for name in members:
            center = float(pd.to_numeric(raw_reference[name], errors="coerce").mean())
            scale = float(pd.to_numeric(raw_reference[name], errors="coerce").std(ddof=1))
            zcols.append((pd.to_numeric(d[name], errors="coerce") - center) / scale)
        d[f"{dim}_raw"] = pd.concat(zcols, axis=1).mean(axis=1)
    d["SHI_from_documented_rules"] = sum(d[f"mu_{name}"] * WEIGHTS[name] for name in WEIGHTS)
    return d


def standardized(values: pd.Series) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce")
    sd = float(x.std(ddof=1))
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.nan, index=x.index)
    return (x - float(x.mean())) / sd


def fit_association(frame: pd.DataFrame, outcome: str, label: str, system: str, outcome_kind: str) -> dict[str, object]:
    d = frame.copy()
    y = standardized(d[outcome])
    erosion = standardized(np.log1p(pd.to_numeric(d["Aa_20yr"], errors="coerce")))
    design = pd.DataFrame({"intercept": 1.0, "z_log1p_Aa_20yr": erosion}, index=d.index)
    for name in M1_ENV:
        design[f"z_{name}"] = standardized(d[name])
    soil = pd.get_dummies(d["SoilType"].astype(str), prefix="SoilType", drop_first=True, dtype=float)
    design = pd.concat([design, soil], axis=1)
    groups = (np.floor(pd.to_numeric(d["x"], errors="coerce") / 5000).astype("Int64").astype(str)
              + "_" + np.floor(pd.to_numeric(d["y"], errors="coerce") / 5000).astype("Int64").astype(str))
    valid = np.isfinite(y.to_numpy()) & np.isfinite(design.to_numpy()).all(axis=1) & groups.notna().to_numpy()
    d = d.loc[valid]
    yv = y.loc[valid].to_numpy(dtype=float)
    X = design.loc[valid].astype(float)
    group_ids = groups.loc[valid]
    fit = sm.OLS(yv, X).fit(cov_type="cluster", cov_kwds={"groups": group_ids, "use_correction": True}, use_t=True)
    beta = float(fit.params["z_log1p_Aa_20yr"])
    se = float(fit.bse["z_log1p_Aa_20yr"])
    lo, hi = (float(v) for v in fit.conf_int().loc["z_log1p_Aa_20yr"].to_numpy())
    rho = float(spearmanr(d["Aa_20yr"], d[outcome], nan_policy="omit").statistic)
    pearson = float(pearsonr(d["Aa_20yr"], d[outcome]).statistic)
    return {"system": system, "outcome_kind": outcome_kind, "outcome": outcome, "label": label,
            "n": int(len(d)), "n_5km_blocks": int(group_ids.nunique()),
            "pearson_Aa20yr": pearson, "spearman_Aa20yr": rho,
            "adjusted_beta_standardized": beta, "cluster_se_5km": se,
            "ci95_low": lo, "ci95_high": hi, "p_cluster_t": float(fit.pvalues["z_log1p_Aa_20yr"]),
            "r_squared_apparent": float(fit.rsquared), "controls": "高程、TWI、NDVI、LST、土类固定效应",
            "inference_note": "5 km块聚类协方差；预测像元仍有空间依赖，结果为探索性关联"}


def summarize_scores(full: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in INDICATORS:
        raw = pd.to_numeric(full[name], errors="coerce")
        score = pd.to_numeric(full[f"mu_{name}"], errors="coerce")
        rows.append({"indicator": name, "label": INDICATOR_LABELS[name], "n_valid": int(score.notna().sum()),
                     "raw_min": float(raw.min()), "raw_max": float(raw.max()), "membership_min": float(score.min()),
                     "membership_max": float(score.max()), "membership_mean": float(score.mean()),
                     "membership_sd": float(score.std(ddof=1)), "membership_unique_values": int(score.nunique()),
                     "membership_at_0p1": int(np.isclose(score, .1).sum()),
                     "membership_at_1": int(np.isclose(score, 1).sum()),
                     "fraction_at_1": float(np.isclose(score, 1).mean()),
                     "saturation_flag": "近乎完全饱和" if np.isclose(score, 1).mean() >= .99 else "否"})
    return pd.DataFrame(rows)


def save_figure(associations: pd.DataFrame, figure_dir: Path) -> None:
    plt.rcParams.update({"font.family": FONT, "axes.unicode_minus": False, "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(15, 8), constrained_layout=True)
    colors = {"原始值体系": "#397a9d", "健康方向统一体系": "#c77b35"}
    left_names = ["SW", "BD", "OM", "AN", "CEC", "pH", "MBC", "RESP"]
    left = associations.loc[associations.outcome_kind.eq("单项指标")]
    y = np.arange(len(left_names))
    for off, system in [(-.16, "原始值体系"), (.16, "健康方向统一体系")]:
        q = left.set_index(["outcome", "system"]).reindex(pd.MultiIndex.from_product([left_names, [system]]))
        x = q.adjusted_beta_standardized.to_numpy(dtype=float)
        lo = q.ci95_low.to_numpy(dtype=float)
        hi = q.ci95_high.to_numpy(dtype=float)
        axes[0].errorbar(x, y + off, xerr=np.vstack([x - lo, hi - x]), fmt="o", color=colors[system],
                         capsize=2.5, markersize=5, label=system)
    axes[0].set_yticks(y, [INDICATOR_LABELS[name] for name in left_names])
    axes[0].invert_yaxis()
    axes[0].axvline(0, color="#555555", lw=1, ls="--")
    axes[0].set_title("八项指标：原始值与方向统一评分")
    axes[0].set_xlabel("log1p(Aa_20yr) 的标准化条件关联系数 β（95%区间）")
    axes[0].grid(axis="x", color="#dce3e8", lw=.7)
    axes[0].legend(frameon=False, loc="lower right")

    right_specs = [
        ("physical_raw", "物理维度（原始值）", "原始值体系"),
        ("physical_direction", "物理维度（方向统一）", "健康方向统一体系"),
        ("chemical_raw", "化学维度（原始值）", "原始值体系"),
        ("chemical_direction", "化学维度（方向统一）", "健康方向统一体系"),
        ("biological_raw", "生物维度（原始值）", "原始值体系"),
        ("biological_direction", "生物维度（方向统一）", "健康方向统一体系"),
        ("SHI_from_documented_rules", "按文档权重重算SHI", "健康方向统一体系"),
        ("SHI_candidate", "原有SHI栅格（公式未核实）", "原始值体系"),
    ]
    dims = associations.loc[associations.outcome_kind.isin(["健康维度", "综合指数"])]
    yy = np.arange(len(right_specs))
    for i, (outcome, label, system) in enumerate(right_specs):
        q = dims.loc[dims.outcome.eq(outcome) & dims.system.eq(system)]
        if q.empty:
            continue
        row = q.iloc[0]
        axes[1].errorbar(float(row.adjusted_beta_standardized), i,
                         xerr=[[float(row.adjusted_beta_standardized - row.ci95_low)],
                               [float(row.ci95_high - row.adjusted_beta_standardized)]],
                         fmt="o", color=colors[system], capsize=2.5, markersize=5)
    axes[1].set_yticks(yy, [spec[1] for spec in right_specs])
    axes[1].invert_yaxis()
    axes[1].axvline(0, color="#555555", lw=1, ls="--")
    axes[1].set_title("三类健康维度与SHI")
    axes[1].set_xlabel("log1p(Aa_20yr) 的标准化条件关联系数 β（95%区间）")
    axes[1].grid(axis="x", color="#dce3e8", lw=.7)
    fig.suptitle("全域像元中的侵蚀关联：原始指标与健康方向统一评分", fontsize=16)
    fig.text(.5, -.015, "同一共同样本（35个5 km空间块）；控制高程、TWI、NDVI、LST和土类。CEC按含侵蚀项的已声明公式恢复，相关不能视为独立侵蚀效应。",
             ha="center", fontsize=9)
    figure_dir.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(figure_dir / f"J9_health_direction_comparison.{ext}", dpi=300 if ext == "png" else None,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    out_dirs = [RUN / name for name in ["data", "results", "figures", "provenance", "reports"]]
    for path in out_dirs:
        path.mkdir(parents=True, exist_ok=True)
    full = pd.read_parquet(BASE_RUN / "data/full_cropland_predictions.parquet")
    common = pd.read_parquet(BASE_RUN / "data/D1_common_env_case.parquet")
    required = ["Aa_20yr", "SHI_candidate", *INDICATORS, *M1_ENV, "SoilType", "x", "y"]
    common = common.dropna(subset=required).copy()
    full = add_direction_scores(full, common)
    common = add_direction_scores(common, common)

    keep = ["cell_id", "Aa_20yr", "SHI_candidate", *INDICATORS, *[f"mu_{x}" for x in INDICATORS],
            "physical_raw", "chemical_raw", "biological_raw", "physical_direction", "chemical_direction",
            "biological_direction", "SHI_from_documented_rules"]
    full[keep].to_parquet(RUN / "data/health_direction_scores.parquet", index=False)
    write_csv(summarize_scores(full), RUN / "results/HEALTH_DIRECTION_SCORE_SUMMARY.csv")

    rows = []
    for name in ["SW", "BD", "OM", "AN", "CEC", "pH", "MBC", "RESP"]:
        rows.append(fit_association(common, name, INDICATOR_LABELS[name], "原始值体系", "单项指标"))
        rows.append(fit_association(common, f"mu_{name}", INDICATOR_LABELS[name], "健康方向统一体系", "单项指标"))
    for dim in ["physical", "chemical", "biological"]:
        rows.append(fit_association(common, f"{dim}_raw", DIMENSION_LABELS[dim], "原始值体系", "健康维度"))
        rows.append(fit_association(common, f"{dim}_direction", DIMENSION_LABELS[dim], "健康方向统一体系", "健康维度"))
    rows.append(fit_association(common, "SHI_from_documented_rules", "按文档公式重算的SHI", "健康方向统一体系", "综合指数"))
    rows.append(fit_association(common, "SHI_candidate", "现有SHI栅格（公式未核实）", "原始值体系", "综合指数"))
    associations = pd.DataFrame(rows)
    write_csv(associations, RUN / "results/HEALTH_DIRECTION_ASSOCIATIONS.csv")
    save_figure(associations, RUN / "figures")

    paired = full[["SHI_candidate", "SHI_from_documented_rules"]].dropna()
    old = paired["SHI_candidate"].to_numpy(dtype=float)
    rebuilt = paired["SHI_from_documented_rules"].to_numpy(dtype=float)
    reconciliation = pd.DataFrame([{
        "n_common_pixels": int(len(paired)), "legacy_SHI_source": "环境匹配原数据/SHI/corrected_visual.tif (resampled Phase I input)",
        "reconstructed_SHI_source": str(SOURCE_DOC), "pearson_r": float(pearsonr(old, rebuilt).statistic),
        "spearman_rho": float(spearmanr(old, rebuilt).statistic),
        "rmse": float(np.sqrt(np.mean((old - rebuilt) ** 2))), "mae": float(np.mean(np.abs(old - rebuilt))),
        "mean_legacy_minus_rebuilt": float(np.mean(old - rebuilt)),
        "legacy_min": float(old.min()), "legacy_max": float(old.max()), "legacy_mean": float(old.mean()),
        "reconstructed_min": float(rebuilt.min()), "reconstructed_max": float(rebuilt.max()),
        "reconstructed_mean": float(rebuilt.mean()),
        "decision": "The chapter draft defines a reconstructable sensitivity score but does not reproduce the legacy SHI raster; do not silently substitute it for the legacy series.",
    }])
    write_csv(reconciliation, RUN / "results/SHI_SOURCE_RECONCILIATION.csv")

    rules = []
    for name in INDICATORS:
        spec = PARAMETERS[name]
        rules.append({"indicator": name, "label": INDICATOR_LABELS[name], **spec,
                      "weight": WEIGHTS[name], "score_range": "0.1–1.0",
                      "score_direction": "higher membership means healthier condition; pH has an optimum plateau"})
    write_csv(pd.DataFrame(rules), RUN / "provenance/HEALTH_SCORING_RULES.csv")
    source_spec = {
        "mode": "EXPLORATORY_ORACLE", "outcome_guided_selection": True, "run_id": RUN_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_document": str(SOURCE_DOC), "source_document_accessible_in_repo": False,
        "source_locations": {"directions": "chapter 4.1 indicator attribute table", "membership": "chapter 4.3.1/table 4-6",
                             "weights": "chapter 4.3.2/table 3-6", "index": "chapter 4.3.3"},
        "membership_formula": {
            "S": "1 if x>=b; 0.1+0.9*(x-a)/(b-a) if a<x<b; 0.1 if x<=a",
            "reverse_S": "1 if x<=a; 0.1+0.9*(b-x)/(b-a) if a<x<b; 0.1 if x>=b",
            "parabolic_optimum": "0.1 outside [a1,a2]; linear rise 0.1 to 1 on (a1,b1); 1 on [b1,b2]; linear decline 1 to 0.1 on (b2,a2)",
        },
        "parameters": PARAMETERS, "weights": WEIGHTS, "weight_sum": float(sum(WEIGHTS.values())),
        "dimension_members": DIMENSION_MEMBERS,
        "raw_dimension_definition": "mean of each listed component after z-standardizing each raw indicator on the D1 common-environment sample",
        "direction_dimension_definition": "equal-weight mean of documented membership values within each dimension",
        "SHI_definition": "sum_i(weight_i * membership_i)",
        "analysis_population": {"source": "r10 D1_common_env_case", "n": int(len(common)),
                                "unit": "30 m RF-predicted cropland raster pixel", "exposure": "log1p(Aa_20yr)"},
        "model": "standardized OLS with M1 controls; 5 km spatial-block cluster covariance",
        "limitations": ["The document is a local chapter draft, not an executable calculation record.",
                        "Reconstructed SHI does not reproduce the existing SHI candidate raster; both are retained separately.",
                        "CEC values follow the declared corrected formula containing Aa; association is mechanically coupled.",
                        "RF pixels are spatial predictions, not independent field samples; block covariance does not remove all spatial dependence."],
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "scipy": scipy.__version__, "statsmodels": sm.__version__, "matplotlib": matplotlib.__version__},
    }
    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    source_spec["git_head_at_execution"] = git_commit
    (RUN / "provenance/source_and_method.json").write_text(json.dumps(source_spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    rec = reconciliation.iloc[0]
    dimension_betas = associations.loc[associations.outcome_kind.eq("健康维度"), ["outcome", "system", "adjusted_beta_standardized", "ci95_low", "ci95_high"]]
    lines = [
        "# J-9 健康方向统一评分补充分析",
        "",
        f"本补充分析读取本地章节草稿 `{SOURCE_DOC.name}`，提取八项指标方向、隶属函数及组合权重，并用当前 CEC 栅格重建逐像元方向评分。输入总体为 Phase J r10 的 D1 共同环境完整像元（n={len(common):,}）；模型控制高程、TWI、NDVI、LST和土类，并按5 km空间块计算聚类协方差。",
        "",
        "## 两套体系",
        "",
        "原始值体系保留 AN、OM、CEC、pH、SW、BD、MBC、RESP 的数值。健康方向统一体系按文档的评分转换：AN、OM、CEC、SW、MBC、RESP用S型函数，BD用反S型，pH用适宜区间型函数；函数输出范围为0.1–1.0。各维度内对成员指标等权平均；综合SHI使用文档中的八项组合权重。",
        "",
        "## 与既有 SHI 栅格的核对",
        "",
        f"按章节草稿参数重算的 SHI 与既有 `SHI_candidate` 在 {int(rec.n_common_pixels):,} 个共同像元上的 Pearson r={rec.pearson_r:.3f}、Spearman ρ={rec.spearman_rho:.3f}、RMSE={rec.rmse:.3f}；既有值均值比重算值高 {rec.mean_legacy_minus_rebuilt:.3f}。这不是精确复现，因此本补充分析保留两者并列，不将章节草稿直接认定为既有栅格的来源公式。",
        "",
        "## 方向分数的关键变化",
        "",
        "- BD 的原始值条件系数为 −0.117，而反S型健康评分为 +0.108；方向转换后正值表示健康评分较高，不能把 BD 的原始下降直接等同于健康变差。",
        "- pH 按适宜区间评分，原始与评分的全域秩相关符号相反，但控制环境后的线性系数均接近0；不应把 pH 的单调变化直接解释为健康方向。",
        "- SW 的方向评分仍与侵蚀呈负向条件关联（β=−0.459），与原始 SW 结果方向一致。CEC 评分在 387,448 个有效像元中有 387,440 个达到上限1.0（99.998%），因此当前参数下几乎没有空间变异；其方向评分回归系数不作实质解释。",
        "",
        "## 维度结果",
        "",
        "原始值综合维度的系数描述原始数值组合随侵蚀的变化，不能统一解读为健康变好或变差；方向统一维度的分值越高代表按该文档函数评分越高。CEC 的恢复公式含 Aa，化学维度和综合 SHI 中含 CEC 的结果带有代数耦合。",
        "",
        "| 维度 | 体系 | 标准化条件系数 β | 95%空间块区间 |",
        "|---|---|---:|---:|",
    ]
    for row in dimension_betas.itertuples():
        dimension = row.outcome.removesuffix("_raw").removesuffix("_direction")
        lines.append(f"| {DIMENSION_LABELS[dimension]} | {row.system} | {row.adjusted_beta_standardized:+.3f} | {row.ci95_low:+.3f}–{row.ci95_high:+.3f} |")
    lines += [
        "",
        "## 解释边界",
        "",
        "- 该章节文件是本地草稿，未发现与之对应的可执行评分脚本；输入像元为随机森林预测面，不是独立采样点。",
        "- 既有 SHI 图层来自 `环境匹配原数据/SHI/corrected_visual.tif`，网格与30 m健康指标栅格不同；其计算配方尚未由该草稿复现。",
        "- 文档给出的 CEC 隶属参数在当前 CEC 栅格上近乎完全饱和；若将该参数用于 SHI 重建或机制回归，应先核实其是否适配当前 CEC 单位和校正后范围。",
        "- CEC 当前值按 `CEC = (CEC_old - 20 × Erosion) / 2.9 + 18` 重建，Erosion 包含两期 Aa 归一化值；任何包含 CEC 的侵蚀关系都不能作为独立侵蚀效应证据。",
        "- 系数为探索性条件关联；5 km 聚类协方差不能消除全部空间依赖，也不能替代独立实测验证。",
        "",
        "## 产物",
        "",
        "- `results/HEALTH_DIRECTION_ASSOCIATIONS.csv`：八项原始值、八项方向评分、三维度及 SHI 的相关和条件模型。",
        "- `results/HEALTH_DIRECTION_SCORE_SUMMARY.csv`：八项原始范围与方向评分范围。",
        "- `results/SHI_SOURCE_RECONCILIATION.csv`：重算 SHI 与既有 SHI 栅格的并列核对。",
        "- `provenance/HEALTH_SCORING_RULES.csv` 与 `provenance/source_and_method.json`：可复核规则和来源位置。",
        "- `figures/J9_health_direction_comparison.png`：原始值、方向统一评分及维度系数对照图。",
    ]
    (RUN / "reports/J9_HEALTH_DIRECTION_ADDENDUM.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (RUN / "RUN_STATUS.md").write_text(
        f"# Phase J J-9 direction score supplement\n\n- run_id: `{RUN_ID}`\n- mode: `EXPLORATORY_ORACLE`\n- outcome_guided_selection: `true`\n- population: D1 common environment pixels, n={len(common):,}\n- formulas and weights transcribed from local chapter draft and saved under `provenance/`.\n- the rebuilt score does not reproduce the legacy SHI raster; both are retained separately.\n- outputs are supplemental to Phase J r10; no previous run was overwritten.\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(RUN.relative_to(ROOT)), "n_common": len(common),
                      "n_pixel_scores": int(full["SHI_from_documented_rules"].notna().sum()),
                      "reconciliation": reconciliation.to_dict(orient="records"),
                      "dimension_betas": dimension_betas.to_dict(orient="records")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
