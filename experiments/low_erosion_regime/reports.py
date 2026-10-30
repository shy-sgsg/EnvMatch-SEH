"""Chinese report, review handoff and figure index for the low-regime run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


OUTCOME_LABELS = {
    "SHI_candidate": "土壤健康指数（SHI）", "AN": "碱解氮（AN）", "OM": "有机质（OM）",
    "CEC": "阳离子交换量（CEC）", "pH": "土壤酸碱度（pH）", "SW": "土壤含水量（SW）",
    "BD": "土壤容重（BD）", "MBC": "微生物量碳（MBC）", "RESP": "土壤呼吸（RESP）",
}
EXPOSURE_LABELS = {"Aa_20yr": "20 年平均实际侵蚀", "Aa_2020": "2020 年实际侵蚀"}


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False) if path.is_file() else pd.DataFrame()


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(number):
        return "—"
    if abs(number) >= 10000:
        return f"{number:,.0f}"
    return f"{number:.{digits}f}"


def _table(frame: pd.DataFrame, columns: list[tuple[str, str]], *, limit: int | None = None) -> str:
    if frame.empty:
        return "（无可用记录）"
    subset = frame.head(limit) if limit else frame
    headers = [label for _, label in columns]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in subset.to_dict("records"):
        values = []
        for name, _ in columns:
            value = row.get(name)
            if isinstance(value, (float, np.floating, int, np.integer)):
                value = _fmt(value)
            if pd.isna(value):
                value = "—"
            values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report_package(project_root: Path, run_dir: Path, config: dict[str, Any]) -> tuple[Path, Path, Path]:
    project_root, run_dir = Path(project_root), Path(run_dir)
    tables = {
        "population": _read(run_dir / "diagnostics" / "exposure_distribution_summary.csv"),
        "kde": _read(run_dir / "diagnostics" / "kde_bandwidth_diagnostics.csv"),
        "gmm": _read(run_dir / "diagnostics" / "gaussian_mixture_diagnostics.csv"),
        "cdf": _read(run_dir / "diagnostics" / "empirical_cdf_breakpoints.csv"),
        "cdf_boot": _read(run_dir / "diagnostics" / "spatial_cdf_bootstrap.csv"),
        "thresholds": _read(run_dir / "thresholds" / "distribution_threshold_candidates.csv"),
        "profiles": _read(run_dir / "maps" / "regime_share_by_environment_profile.csv"),
        "pair_quality": _read(run_dir / "diagnostics" / "pair_quality_comparison.csv"),
        "associations": _read(run_dir / "associations" / "response_associations.csv"),
        "models": _read(run_dir / "models" / "model_search_trials.csv"),
        "segmented": _read(run_dir / "models" / "zero_indicator_and_segmented_models.csv"),
        "spatial_pixel": _read(run_dir / "spatial_validation" / "pixel_spatial_validation.csv"),
        "spatial_pair": _read(run_dir / "spatial_validation" / "pair_spatial_validation.csv"),
    }
    manifest_path = run_dir / "figures" / "figure_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else []
    review_dir = project_root / "docs" / "reviews" / "2026-10-29" / "phase-h"
    review_dir.mkdir(parents=True, exist_ok=True)

    population = tables["population"]
    if not population.empty:
        population["period"] = population["exposure"].map(EXPOSURE_LABELS)
    gmm_rows, uncertain_rows = [], []
    for exposure, frame in tables["gmm"].groupby("exposure", sort=False):
        best = frame.loc[frame["bic"].idxmin()]
        assignment_path = run_dir / "diagnostics" / f"gmm_posterior_assignments_{exposure}.parquet"
        assignment = pd.read_parquet(assignment_path) if assignment_path.is_file() else pd.DataFrame()
        uncertain_rows.append({
            "period": EXPOSURE_LABELS.get(exposure, exposure), "selected_k": int(best["n_components"]),
            "uncertain_fraction": float(assignment["classification_status"].eq("uncertain").mean()) if not assignment.empty else np.nan,
            "uncertain_n": int(assignment["classification_status"].eq("uncertain").sum()) if not assignment.empty else 0,
        })
        gmm_rows.append({"period": EXPOSURE_LABELS.get(exposure, exposure), "selected_k": int(best["n_components"]),
                         "bic": best["bic"], "aic": best["aic"], "converged": best["converged"],
                         "means": best["component_means_log10"], "overlap": best["adjacent_overlap_index"]})
    gmm_best = pd.DataFrame(gmm_rows)
    gmm_uncertain = pd.DataFrame(uncertain_rows)
    if not tables["cdf_boot"].empty:
        cdf_stability = tables["cdf_boot"].groupby("exposure", as_index=False).agg(
            resamples=("replicate", "size"),
            valid_fraction=("status", lambda x: x.eq("valid_cdf_breakpoint").mean()),
            median_threshold=("threshold_original", "median"),
            threshold_sd=("threshold_original", "std"),
        )
        cdf_stability["period"] = cdf_stability["exposure"].map(EXPOSURE_LABELS)
    else:
        cdf_stability = pd.DataFrame()
    if not tables["thresholds"].empty:
        primary = tables["thresholds"].loc[tables["thresholds"]["is_primary"].fillna(False)].copy()
        primary["period"] = primary["exposure"].map(EXPOSURE_LABELS)
    else:
        primary = pd.DataFrame()

    pair_summary = tables["pair_quality"].copy()
    if not pair_summary.empty:
        pair_summary["period"] = pair_summary["exposure"].map(EXPOSURE_LABELS)
        pair_summary = pair_summary.loc[pair_summary["scheme"].isin(["S0_all", "S1_both_positive"])
                                        | pair_summary["scheme"].astype(str).str.startswith("T")].copy()
    assoc = tables["associations"]
    pair_assoc = assoc.loc[assoc.get("analysis_unit", pd.Series(index=assoc.index, dtype=str)).eq("pair")].copy() if not assoc.empty else pd.DataFrame()
    if not pair_assoc.empty:
        pair_assoc["period"] = pair_assoc["exposure"].map(EXPOSURE_LABELS)
        pair_assoc["health_indicator"] = pair_assoc["outcome"].map(OUTCOME_LABELS)
        chosen = pair_assoc.loc[pair_assoc["scheme"].isin(["S0_all", "S1_both_positive"])
                                | pair_assoc["scheme"].astype(str).str.startswith("T")].copy()
        response_summary = chosen.groupby(["period", "health_indicator", "scheme"], as_index=False).agg(
            n_pairs=("n", "median"), spearman_rho=("spearman_rho", "median"),
            kendall_tau=("kendall_tau", "median"), median_delta=("median_delta_health", "median"),
            negative_fraction=("negative_delta_fraction", "median"),
            wilcoxon_q=("wilcoxon_p_q", "median"), slope=("slope", "median"),
            slope_ci_low=("slope_ci_low", "min"), slope_ci_high=("slope_ci_high", "max"),
        )
    else:
        response_summary = pd.DataFrame()
    pixel_spatial = tables["spatial_pixel"]
    if not pixel_spatial.empty:
        pixel_spatial = pixel_spatial.loc[pixel_spatial["status"].eq("ok")].copy()
        spatial_summary = pixel_spatial.groupby(["exposure", "scheme"], as_index=False).agg(
            splits=("split", "nunique"), median_test_r2=("spatial_test_r_squared_selected", "median"),
            q25_test_r2=("spatial_test_r_squared_selected", lambda x: x.quantile(.25)),
            q75_test_r2=("spatial_test_r_squared_selected", lambda x: x.quantile(.75)),
            median_test_n=("n_test", "median"),
        )
        spatial_summary["period"] = spatial_summary["exposure"].map(EXPOSURE_LABELS)
    else:
        spatial_summary = pd.DataFrame()

    if tables["thresholds"].empty:
        finding = (
            "两种时期均未找到同时满足预设分布判据与空间稳定性的低值分界。零侵蚀可作为独立状态继续分析；"
            "正值部分保留为连续暴露，并使用非线性曲线、零值指示和正值强度项描述响应，不强制截断正值分布。"
        )
        threshold_text = finding
    else:
        finding = (
            f"共识别 {len(tables['thresholds'])} 个经过分布规则筛选的阈值候选。主阈值优先采用 KDE Scott ×1；"
            "若其未通过判据，则采用通过空间稳定性检查的 CDF 或低重叠 GMM 候选。"
        )
        threshold_text = _table(primary, [
            ("period", "侵蚀时期"), ("candidate_method", "主阈值方法"),
            ("threshold_original", "Aa 阈值"), ("retained_fraction_positive", "阈值以上正值保留比例"),
            ("cdf_bootstrap_valid_fraction", "CDF 重抽样识别率"),
            ("cdf_bootstrap_threshold_cv", "阈值变异系数"),
        ])
    population_table = _table(population, [
        ("period", "侵蚀时期"), ("n_total", "输入行数"), ("n_finite", "有效耕地观测"),
        ("n_zero", "零侵蚀像元"), ("n_positive", "正侵蚀像元"),
        ("zero_fraction_finite", "零值比例"), ("positive_min", "正值最小值"),
        ("positive_median", "正值中位数"), ("positive_max", "正值最大值"),
        ("fraction_repeated_6dp", "六位小数重复比例"),
    ])
    gmm_table = _table(gmm_best, [("period", "时期"), ("selected_k", "BIC 选择 K"),
                                  ("bic", "BIC"), ("converged", "收敛"),
                                  ("means", "分量均值 log10(Aa)"), ("overlap", "相邻重叠指数")])
    uncertain_table = _table(gmm_uncertain, [("period", "时期"), ("selected_k", "K"),
                                             ("uncertain_n", "后验不确定像元"),
                                             ("uncertain_fraction", "后验不确定比例")])
    cdf_table = _table(cdf_stability, [("period", "时期"), ("resamples", "空间重抽样次数"),
                                      ("valid_fraction", "识别折点比例"),
                                      ("median_threshold", "重抽样阈值中位数"),
                                      ("threshold_sd", "阈值标准差")])
    pair_table = _table(pair_summary, [("period", "侵蚀时期"), ("source", "配对来源"),
                                       ("scheme", "群体方案"), ("pair_count", "配对数"),
                                       ("mean_absolute_environment_smd", "环境 SMD 均值"),
                                       ("environmental_distance_median", "环境距离中位数"),
                                       ("geographic_distance_median", "地理距离中位数（m）"),
                                       ("same_5km_block_fraction", "端点同一 5 km 块比例")], limit=30)
    response_table = _table(response_summary, [
        ("period", "时期"), ("health_indicator", "健康指标"), ("scheme", "群体方案"),
        ("n_pairs", "配对数中位数"), ("spearman_rho", "Spearman ρ"),
        ("kendall_tau", "Kendall τ"), ("median_delta", "健康差值中位数"),
        ("negative_fraction", "健康差值为负比例"), ("slope", "差值斜率"),
        ("slope_ci_low", "块自助 95% CI 下限"), ("slope_ci_high", "块自助 95% CI 上限"),
        ("wilcoxon_q", "Wilcoxon q 值"),
    ], limit=36)
    spatial_table = _table(spatial_summary, [
        ("period", "时期"), ("scheme", "群体方案"), ("splits", "空间划分次数"),
        ("median_test_r2", "测试 R² 中位数"), ("q25_test_r2", "测试 R² 第 25 百分位"),
        ("q75_test_r2", "测试 R² 第 75 百分位"), ("median_test_n", "测试样本中位数"),
    ])
    image_lines = []
    for figure in manifest:
        image_lines.append(f"- [{figure.get('title', figure['id'])}](../../../../{run_dir.relative_to(project_root)}/figures/{figure['png']}) · {figure.get('figure_type', '图')}")
    review_dir = project_root / "docs" / "reviews" / "2026-10-29" / "phase-h"
    review_dir.mkdir(parents=True, exist_ok=True)
    report_path = review_dir / "LOW_EROSION_REGIME_RESEARCH_REPORT.md"
    report = f"""# 低侵蚀值群体诊断与响应探索报告

## 研究定位与数据

本轮属于 EXPLORATORY_ORACLE 探索，outcome_guided_selection=true。使用既有实际侵蚀 Aa_20yr 与 Aa_2020，不重算或验证 RUSLE 因子乘积；不重跑 PCNM。侵蚀阈值的分布诊断不读取健康结果。模型族比较和按健康响应优化的阈值另行标记为 Oracle 表观探索。

分析单位为 30 m 有效耕地像元。valid_cell=True 且侵蚀值有限才进入分布分母；其他数据库行记为 NoData，不记作零值。健康变量来自对齐预测栅格，因此空间像元数不等于独立实测样点数。

### 侵蚀分布摘要

{population_table}

## 主要发现

{finding}

正值部分使用 log10(Aa) 比较 Scott/Silverman 带宽及 0.5、1、1.5、2 倍平滑强度。GMM 比较 K=1–4，分量按均值排序并保留后验不确定像元。经验 CDF 折点与 Otsu 只作为独立分布诊断；Otsu 不自动成为筛选阈值。

### GMM 诊断与分类不确定性

{gmm_table}

{uncertain_table}

### CDF 折点空间稳定性

{cdf_table}

### 可进入筛选的分布阈值

{threshold_text}

#### 全部阈值候选

{_table(tables['thresholds'], [('exposure', '时期变量'), ('candidate_method', '方法'), ('threshold_original', 'Aa 阈值'), ('retained_fraction_positive', '正值保留比例'), ('status', '状态')])}

## 配对筛选与重配对

现有 Phase F M3/M4 配对仅当两个端点的实际侵蚀都高于阈值时保留。筛选后分别使用原有 Phase F 环境组，以及仅在筛选后重新拟合的土类内 K=6 高程—地表温度组进行重配对。环境组、匹配距离与最近邻均未使用健康响应。

{pair_table}

配对数、环境距离、地理距离、环境 SMD 和同一 5 km 空间块比例分别报告；原有配对筛选与重新匹配结果不合并。比较不同筛选群体的 R²时，样本与估计目标已经改变，不能将变化单独解释为模型改善。

## 健康指标响应

九项指标包括 SHI、AN、OM、CEC、pH、SW、BD、MBC、RESP。像元层报告 Spearman、Kendall 和描述性斜率；配对层报告 Spearman、Kendall、Wilcoxon、符号检验、平均/中位健康差值、负差值比例、差值斜率及 5 km 空间块自助区间。零侵蚀群体的暴露没有变化，因此零值内部的侵蚀响应斜率不可估。

{response_table}

## 非线性与零值指示模型

未筛选全样本上比较 log1p(Aa) 线性项、零值指示加正值强度自然样条，以及有效阈值存在时的折线响应。模型形式还包括线性、二次、自然样条和梯度提升。表观 R²用于 Oracle 探索并另行标记，不能替代空间留出测试。结果导向阈值写入 models/outcome_optimized_threshold_trials.csv，不参与空间测试阈值选择。

## 空间留出验证

空间验证按 5 km 块整块分配训练/验证/测试角色，在训练集中剔除距离验证或测试块 500 m 范围内的像元。模型族在验证块选择，空间测试块只用于最终得分。

{spatial_table}

## 图集

所有新图保存为 320 dpi PNG、矢量 SVG 和 PDF，并附中文图注与可检查的源数据表。

{chr(10).join(image_lines)}

## 解释边界

- 分析使用预测栅格，结果反映空间像元关联，不等于独立实测样点量或因果效应。
- KDE 未识别可靠峰谷时，零值分析仍可继续；不得把 Otsu、重叠 GMM 分量边界或健康结果挑选出的切点描述为自然分界。
- Oracle 的表观模型和 outcome-optimized 阈值属于结果导向探索；独立空间测试只使用分布阈值及训练/验证集确定的模型。
- 环境匹配排除 R、K、LS、C、P，但不能排除所有环境混杂。
- 数据来自既有对齐数据库；本轮不验证 RUSLE 因子乘积，也不重跑 PCNM。

运行编号：{config['run_id']}  
模式：EXPLORATORY_ORACLE  
输入运行：{config['input_run_id']}  
配对输入：{config['phase_f_run_id']}  
图表数：{len(manifest)}
"""
    report_path.write_text(report, encoding="utf-8")

    index_path = review_dir / "PHASE_H_RESULTS_INDEX.md"
    index_path.write_text(f"""# Phase H 结果索引

## 本轮新增低侵蚀探索

- [低侵蚀群体研究报告](LOW_EROSION_REGIME_RESEARCH_REPORT.md)
- [低侵蚀图集（交互式本地网页）](FIGURE_ATLAS.html)
- [低侵蚀运行目录](../../../../{run_dir.relative_to(project_root)}/)

## Phase H 主分析

- [Phase H 探索报告](../2026-10-28/phase-h/PHASE_H_EXPLORATORY_REPORT.md)
- [Phase H 审查请求](../2026-10-28/phase-h/CHATGPT_REVIEW_REQUEST.md)
- [Phase H 主分析图集](../../../../outputs/exploratory_oracle/{config.get('phase_h_run_id', '')}/figures/)

## Phase F / Phase G 与初始空间图

- [Phase F 阶段报告](../2026-10-26/phase-f/PHASE_F_EXPLORATORY_REPORT.md)
- [Phase G 阶段报告](../2026-10-27/phase-g/PHASE_G_COMPLETION_REPORT.md)
- [初始输入准备阶段地图与诊断图](../../../../outputs/{config['input_run_id']}/figures/)
- [完整项目图集](FIGURE_ATLAS.html)
""", encoding="utf-8")
    request_path = review_dir / "CHATGPT_REVIEW_REQUEST.md"
    request_path.write_text(f"""# ChatGPT 复核请求：低侵蚀值群体与环境匹配后响应

请先完整阅读 docs/RESEARCH_CHARTER.md、PHASE_H_RESULTS_INDEX.md、FIGURE_ATLAS.html、Phase H 主分析报告和 LOW_EROSION_REGIME_RESEARCH_REPORT.md，再审阅本轮运行产物。图表源数据和运行配置均随输出保存。

## 审查对象

- 运行目录：{run_dir.relative_to(project_root)}
- 运行模式：EXPLORATORY_ORACLE
- outcome_guided_selection=true
- 输入对齐数据库：{config['input_run_id']}
- Phase F 配对：{config['phase_f_run_id']}
- 配置快照：{run_dir.relative_to(project_root)}/config/config.yaml

## 请重点检查

1. 是否正确区分有效零值、正的近零值与 NoData；空间分母和筛选端点条件是否一致。
2. KDE、CDF、GMM、Otsu 判据是否充分透明；没有可靠分界时是否避免强行截点。
3. CDF/KDE 阈值的空间块重抽样稳定性标准是否合理；GMM 分量重叠和后验不确定类是否得到恰当解释。
4. 原有 M3/M4 配对过滤与筛选后重配对是否清楚分开；健康结果是否始终未参与匹配规则。
5. Spearman、Kendall、Wilcoxon、配对差值、非线性响应与 5 km 留出空间测试是否对应各自统计单位。
6. 图表是否准确表达预测栅格上的关联；是否有文字越过描述性证据，形成因果或独立实测样点结论。
7. 图表的中文标签、图注、空间尺度、颜色编码及数据来源是否清楚，是否存在误导或可读性问题。

请优先列出会改变结论的实质性方法问题，再列次要图表与文案改进。不要把 Oracle 表观最优结果解释为独立验证。
""", encoding="utf-8")
    return report_path, index_path, request_path

