"""Assemble the Phase I final report, local review request, and figure atlas."""

from __future__ import annotations

import argparse
import html
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
PHASE_ROOT = ROOT / "outputs/exploratory_oracle/phase_i"
OUTCOME_ORDER = ["SHI_candidate", "SW", "BD", "OM", "AN", "CEC", "pH", "MBC", "RESP"]
OUTCOME_LABELS = {"SHI_candidate": "SHI", "AN": "AN", "OM": "OM", "CEC": "CEC",
                  "pH": "pH", "SW": "SW", "BD": "BD", "MBC": "MBC", "RESP": "RESP"}
DIMENSION_LABELS = {"SHI综合": "SHI composite", "物理": "Physical", "化学": "Chemical", "生物": "Biological"}
FIGURE_SECTIONS = [
    ("I-1", "phase_i_20261031_seed20261031_r2", "全样本侵蚀—健康关系"),
    ("I-2", "phase_i_20261031_seed20261031_i2_r2", "环境匹配条件关联"),
    ("I-3", "phase_i_20261031_seed20261031_i3_r2", "环境异质性"),
    ("I-4", "phase_i_20261031_seed20261031_i4_r3", "GWR空间响应"),
    ("I-5", "phase_i_20261031_seed20261031_i5_r1", "空间机器学习"),
    ("I-6", "phase_i_20261031_seed20261031_i6_r1", "局部关联与热点"),
    ("I-7", "phase_i_20261031_seed20261031_i7_r1", "健康维度比较"),
    ("I-8", "phase_i_20261031_seed20261031_i8_r2", "空间胁迫分区"),
    ("I-C", "phase_i_20261031_seed20261031_closure_r1", "核心图件补充与标准MGWR"),
]


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True,
                              text=True).stdout.strip()
    except Exception:
        return None


def _version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _rel(path: Path, start: Path) -> str:
    return Path(os.path.relpath(path, start=start)).as_posix()


def _num(value: Any, digits: int = 3) -> str:
    if pd.isna(value):
        return "—"
    return f"{float(value):.{digits}f}"


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(lines)


def _report_image(reports_dir: Path, run_id: str, figure_rel: str, caption: str) -> str:
    source = PHASE_ROOT / run_id / "figures" / figure_rel
    png = source.with_suffix(".png")
    svg = source.with_suffix(".svg")
    pdf = source.with_suffix(".pdf")
    lines = [f"![{caption}]({_rel(png, reports_dir)})", "", f"*{caption}*  "]
    links = []
    if png.exists():
        links.append(f"[PNG]({_rel(png, reports_dir)})")
    if svg.exists():
        links.append(f"[SVG]({_rel(svg, reports_dir)})")
    if pdf.exists():
        links.append(f"[PDF]({_rel(pdf, reports_dir)})")
    lines.append(" · ".join(links))
    return "\n".join(lines)


def _indicator_table(i7_dir: Path) -> tuple[str, pd.DataFrame]:
    path = i7_dir / "tables/indicator_response_crosswalk.csv"
    d = pd.read_csv(path).set_index("outcome").reindex(OUTCOME_ORDER).reset_index()
    rows = []
    for _, r in d.iterrows():
        label = OUTCOME_LABELS[r["outcome"]]
        dim = DIMENSION_LABELS.get(r["health_dimension"], str(r["health_dimension"]))
        ci = f"{_num(r['standardized_beta_log1p_Aa'])} ({_num(r['block_bootstrap_beta_ci_low'])}, {_num(r['block_bootstrap_beta_ci_high'])})"
        pair = f"{_num(r['standardized_slope_beta'])} / {_num(r['matched_pair_spearman_rho'])}"
        gwr = f"{_num(r['i4_gwr_local_beta_standardized_mean'])} / {_num(r['i4_lcr_local_beta_standardized_mean'])}"
        rows.append([label, dim, _num(r["global_spearman_rho"]), ci, pair, gwr, _num(r["rho_median"])])
    table = _markdown_table(["指标", "维度", "I-1 Spearman ρ", "I-1 标准化β（5km块95% CI）",
                             "I-2 配对β / ρ", "I-4 GWR / LCR局部均值β", "I-6 局部ρ中位数"], rows)
    return table, d


def _dimension_table(i7_dir: Path) -> str:
    d = pd.read_csv(i7_dir / "tables/health_dimension_summary.csv")
    rows = []
    for _, r in d.iterrows():
        label = DIMENSION_LABELS.get(r["health_dimension"], str(r["health_dimension"]))
        rows.append([label, str(int(r["n_indicators"])),
                     f"{int(r['global_beta_negative_indicators'])}/{int(r['global_beta_positive_indicators'])}",
                     _num(r["global_beta_median"]), _num(r["matched_pair_beta_median"]),
                     _num(r["environment_adjusted_GWR_beta_median"]),
                     _num(r["local_spearman_median_of_indicators"])])
    return _markdown_table(["维度", "指标数", "I-1 β负/正", "I-1 β中位数", "I-2 β中位数",
                            "I-4 GWR均值β的维度内中位", "I-6 局部ρ中位数"], rows)


def _zone_table(i8_dir: Path) -> str:
    d = pd.read_csv(i8_dir / "tables/stress_zone_summary.csv")
    rows = []
    for _, r in d.iterrows():
        rows.append([str(r["zone"]), str(int(r["spatial_units_240m"])),
                     f"{100*r['share_of_units']:.1f}%", f"{r['area_km2_equivalent']:.2f}",
                     _num(r["mean_Aa_20yr"]), _num(r["mean_SHI"]),
                     _num(r["median_GWR_beta"]), f"{100*r['GWR_GiStar_hotspot_share']:.1f}%"])
    return _markdown_table(["分区", "240m单元", "单元占比", "完整案例面积等价(km²)",
                            "均值Aa", "均值SHI", "局部GWRβ中位数", "I-6 GWR热点占比"], rows)


def _build_report(run_dir: Path, config: dict[str, Any], git_head: str | None) -> Path:
    reports = run_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    runs = {k: ROOT / v for k, v in config["source_runs"].items()}
    table, _ = _indicator_table(runs["i7"])
    dim_table = _dimension_table(runs["i7"])
    zone_table = _zone_table(runs["i8"])
    stage_reports = []
    for key, stage, name in [
        ("i1", "I-1", "全样本侵蚀—健康响应"), ("i2", "I-2", "环境匹配条件关联"),
        ("i3", "I-3", "环境异质性"), ("i4", "I-4", "GWR与空间变化系数"),
        ("i5", "I-5", "空间机器学习"), ("i6", "I-6", "局部关联与热点"),
        ("i7", "I-7", "健康维度与PCNM/PLS衔接"), ("i8", "I-8", "空间胁迫分区"),
    ]:
        source = runs[key] / "reports"
        report_file = next(source.glob("PHASE_I_*.md"))
        stage_reports.append(f"- **{stage} {name}**：[阶段报告]({_rel(report_file, reports)})")
    closure_report = runs["closure"] / "reports/PHASE_I_CLOSURE_REPORT.md"
    stage_reports.append(f"- **I-C 图件闭合与标准MGWR**：[补充报告]({_rel(closure_report, reports)})")

    figures = [
        _report_image(reports, config["source_runs"]["i1"].split("/")[-1], "global_response/I01_global_association_heatmap.png", "I-1：全掩膜九项指标侵蚀关联"),
        _report_image(reports, config["source_runs"]["i1"].split("/")[-1], "global_response/I04_zero_positive_boxplots.png", "I-1：零侵蚀与正侵蚀状态差异"),
        _report_image(reports, config["source_runs"]["i2"].split("/")[-1], "response_curves/I09_Aa_20yr_matched_pair.png", "I-2：Aa_20yr M4匹配配对响应"),
        _report_image(reports, config["source_runs"]["i3"].split("/")[-1], "environment_heterogeneity/I11_m4_twi_group_slopes.png", "I-3：土类内TWI分组的配对斜率"),
        _report_image(reports, config["source_runs"]["i4"].split("/")[-1], "gwr_SHI_candidate_template.png", "I-4：环境调整SHI GWR局部标准化系数"),
        _report_image(reports, config["source_runs"]["i5"].split("/")[-1], "grf_local_Aa_marginal_effect.png", "I-5：局部随机森林Aa边际效应"),
        _report_image(reports, config["source_runs"]["i6"].split("/")[-1], "bivariate_local_moran_clusters.png", "I-6：双变量Local Moran类别"),
        _report_image(reports, config["source_runs"]["i6"].split("/")[-1], "gistar_negative_response_gwr.png", "I-6：GWR负向响应Gi*热点"),
        _report_image(reports, config["source_runs"]["i7"].split("/")[-1], "health_dimension_comparison_forest.png", "I-7：全样本、匹配与局部系数对照"),
        _report_image(reports, config["source_runs"]["i8"].split("/")[-1], "stress_zoning_map.png", "I-8：主分区和环境敏感候选代理"),
        _report_image(reports, config["source_runs"]["closure"].split("/")[-1], "core_figures/C04_global_best_model_r2.png", "核心图04：全样本模型族R²比较"),
        _report_image(reports, config["source_runs"]["closure"].split("/")[-1], "core_figures/C07_soil_type_matched_response_curves.png", "核心图07：按土类分层的M4配对响应"),
        _report_image(reports, config["source_runs"]["closure"].split("/")[-1], "core_figures/C16_eight_indicator_gwr_composite.png", "核心图16：八项指标GWR局部系数组合"),
        _report_image(reports, config["source_runs"]["closure"].split("/")[-1], "core_figures/C17_MGWR_local_coefficients.png", "核心图17：同样本标准GWR与MGWR局部系数"),
        _report_image(reports, config["source_runs"]["closure"].split("/")[-1], "core_figures/C18_MGWR_variable_scales.png", "核心图18：标准MGWR变量作用尺度"),
        _report_image(reports, config["source_runs"]["closure"].split("/")[-1], "core_figures/C19_GWR_MGWR_difference_map.png", "核心图19：同样本GWR与MGWR系数差异"),
    ]

    closure_dir = runs["closure"]
    closure_status = pd.read_csv(closure_dir / "tables/mgwr_fit_status.csv")
    closure_models = pd.read_csv(runs["closure"] / "tables/mgwr_model_summary.csv")
    closure_bandwidths = pd.read_csv(runs["closure"] / "tables/mgwr_variable_bandwidths.csv")
    closure_sample = json.loads((runs["closure"] / "tables/mgwr_sampling_summary.json").read_text(encoding="utf-8"))
    closure_model_rows = []
    for outcome in ["SHI_candidate", "OM", "SW"]:
        subset = closure_models.loc[closure_models["outcome"].eq(outcome)]
        gwr = subset.loc[subset["model_family"].eq("standard_GWR_same_sample")]
        mgwr = subset.loc[subset["model_family"].eq("standard_MGWR_spatially_balanced_sample")]
        closure_model_rows.append([
            OUTCOME_LABELS[outcome], _num(gwr.iloc[0]["r_squared"]) if len(gwr) else "—",
            _num(mgwr.iloc[0]["r_squared"]) if len(mgwr) else "—",
            _num(gwr.iloc[0]["aicc"]) if len(gwr) else "—",
            _num(mgwr.iloc[0]["aicc"]) if len(mgwr) else "—",
        ])
    closure_mgwr_table = _markdown_table(["指标", "同样本GWR R²", "标准MGWR R²", "GWR AICc", "MGWR AICc"], closure_model_rows)
    aa_scales = closure_bandwidths.loc[closure_bandwidths["variable"].eq("log1p_Aa")]
    closure_scale_rows = [[OUTCOME_LABELS[row.outcome], str(int(row.bandwidth_neighbors)),
                           f"{row.median_neighbor_distance_m / 1000:.2f}"] for row in aa_scales.itertuples(index=False)]
    closure_scale_table = _markdown_table(["指标", "Aa邻居数", "中位作用距离(km)"], closure_scale_rows)
    closure_successes = int(closure_status["status"].eq("ok").sum()) if "status" in closure_status else 0
    closure_images = [
        _report_image(reports, config["source_runs"]["closure"].split("/")[-1], "core_figures/C16_eight_indicator_gwr_composite.png", "八项健康指标GWR局部系数组合"),
        _report_image(reports, config["source_runs"]["closure"].split("/")[-1], "core_figures/C17_MGWR_local_coefficients.png", "标准GWR与MGWR局部系数，同一空间均衡样本"),
        _report_image(reports, config["source_runs"]["closure"].split("/")[-1], "core_figures/C18_MGWR_variable_scales.png", "标准MGWR变量作用尺度"),
        _report_image(reports, config["source_runs"]["closure"].split("/")[-1], "core_figures/C19_GWR_MGWR_difference_map.png", "标准GWR与MGWR系数差异"),
        _report_image(reports, config["source_runs"]["closure"].split("/")[-1], "core_figures/C30_PCNM_comparison_NOT_AVAILABLE.png", "核心图30状态：PCNM源表缺失"),
        _report_image(reports, config["source_runs"]["closure"].split("/")[-1], "core_figures/C31_mechanism_path_NOT_AVAILABLE.png", "核心图31状态：PLS-SEM源表缺失"),
    ]

    body = [
        "# 温县环境匹配探索显示侵蚀—健康关系以含水量和有机质的负向响应最稳定", "",
        "## 执行摘要", "",
        "- 全掩膜预测栅格的关联以 SW、SHI 和 OM 的负向关系较明显；Aa_20yr Spearman 分别为 −0.584、−0.469、−0.415。CEC 为正向（ρ=0.168）。这些像元关联描述空间梯度，不是独立实测样点效应。",
        "- 在既有 M4 匹配配对中，SHI、OM、SW 的标准化配对斜率仍为负（−0.144、−0.100、−0.269），但比端点像元相关弱；“匹配端点关联”和“高低端配对差值关联”是不同估计对象。",
        "- 空间差异明显：I-4 环境调整 GWR 对 SHI、OM、SW 的加权平均标准化局部系数为 −0.330、−0.556、−0.680；GWR-LCR收缩后为 −0.218、−0.387、−0.464。I-6局部Spearman的中位数为 −0.591、−0.616、−0.783。",
        "- Oracle分位分区得到623个T1高Aa—强负向响应240m单元（按共同完整案例像元换算约18.88 km²）。该区是后续实地核验候选，不能当作已验证治理风险区。",
        "",
        "## 1. 研究问题、范围与分析单位", "",
        "本报告整合 Phase I-1 至 I-8 及 I-C 图件闭合/标准MGWR补充，回答侵蚀与土壤健康的总体关系、环境匹配后的条件关联、非线性与环境异质性、局部空间变化、健康维度差异及候选治理分区。所有阶段采用 `EXPLORATORY_ORACLE`，并保留 `outcome_guided_selection=true`；不能作为预注册正式证据或独立因果验证。",
        "",
        "侵蚀暴露是既有 Aa_20yr 预计算栅格，分析直接读取已对齐数据库，不重建或验证 RUSLE 因子乘积。SHI及八项健康指标为随机森林预测栅格。掩膜库保留392,527个像元；I-1依每项结局完整案例分析，I-2 Aa_20yr M4候选含111,289对匹配端点（结局有效对数依指标略有不同）；I-4/I-6空间分析约用240m汇总单元；I-8分区使用293,633个环境和SHI共同完整像元组成的7,387个汇总单元。像元数、匹配对数、空间块与独立实测样点不能互换。",
        "",
        "Aa单位、精确时间口径尚未从元数据确认；报告中的暴露轴沿用源值。",
        "",
        "## 2. 全掩膜：侵蚀与健康指标的总体响应", "",
        "全掩膜 Spearman 中，SW与Aa_20yr的负向排序关系最强（ρ=−0.584），其后为SHI（−0.469）、OM（−0.415）；CEC为正（+0.168）。I-1 SHI标准化线性β为−0.472（5km空间块bootstrap 95% CI −0.531至−0.412）；SW为−0.595（−0.630至−0.555），OM为−0.421（−0.459至−0.377）。零侵蚀像元占SHI完整案例34.2%；正侵蚀减零侵蚀的SHI Hedges' g为−0.889，95%空间块区间见I-1报告。",
        "",
        "模型对比使用5km空间块训练/验证/测试，按验证集选择模型，测试集只留出评估。SHI测试R²为0.187，许多结局测试表现接近零或为负；不据大像元数宣称模型通过外部验证。分段拐点来自Oracle验证集候选筛选，MBC的Aa约4.617是探索候选，不是独立阈值。",
        "",
        figures[0], "", figures[1], "",
        "## 3. 环境匹配后：配对差值仍有负向信号，但幅度较小", "",
        "I-2复用Phase H M4 Aa_20yr候选 `stage2_Aa_20yr_01`，没有重做匹配。配对端点的全侵蚀范围像元相关接近I-1：SHI Spearman −0.452 vs 全掩膜−0.469，OM −0.416 vs −0.415，SW −0.567 vs −0.584。配对差值的响应弱于端点相关：SHI/OM/SW的标准化斜率约−0.144/−0.100/−0.269，Spearman约−0.144/−0.115/−0.272。它们表示每对高侵蚀端相对低侵蚀端的差值关联，不可用来替代全像元关联。",
        "",
        "零—正与双正子组分别保留；Aa_20yr全范围匹配对OLS关系的表观拟合较弱。I-2模型排行榜中的随机森林R²是同样本表观拟合，不是空间泛化。匹配提高了环境可比性，但不消除未测混杂，也不能单凭这一结果称作侵蚀独立效应。",
        "",
        figures[2], "",
        "## 4. 非线性和环境异质性", "",
        "I-1全样本GAM局部斜率显示SHI范围维持负向；多项单项指标斜率跨零。I-5以5km整块测试评估空间变化系数GAM近似，相较全局GAM，SHI/OM/SW测试R²提升仅+0.014/+0.007/+0.005，空间变化曲面提供的额外预测信息有限。",
        "",
        "I-3复用既有M4配对环境组。Aa_20yr下27潮土SHI/OM/SW标准化配对斜率为−0.246/−0.294/−0.316；27:q2在三项中是TWI组内较负向候选。126项M4交互有123项BH q<0.05，但这些交互来自同一Oracle搜索流程并受空间依赖影响，不能当作独立重复确认。Phase H的26:q1低TWI候选基于另一低值筛选总体，不能与27:q2直接排名。",
        "",
        figures[3], "",
        "## 5. 空间变化与局部集聚", "",
        "I-4环境调整GWR的SHI、OM、SW加权平均标准化斜率为−0.330/−0.556/−0.680；99.7%局部单元的SHI系数为负。GWR-LCR对SHI/OM/SW的系数向零收缩至−0.218/−0.387/−0.464，且所有局部点都施加岭惩罚，说明环境协变量局部共线性会影响系数幅度。I-4标准GWR候选中31/54个带宽落在k=30下界。I-4当时使用MGWR-inspired近似；I-C随后在空间均衡n=770样本完成标准GWR/MGWR，详见5.1节。",
        "",
        "I-5局部RF验证R²中位数为SHI 0.610、OM 0.618、SW 0.736，但该局部随机划分邻域重叠，不是空间独立测试。局部Aa边际效应中位数为−0.144/−0.263/−0.437；SHAP贡献和变量重要性是预测分解，不作因果解释。",
        "",
        "I-6 Local Spearman中位数为SHI −0.591、OM −0.616、SW −0.783。Bivariate LISA的FDR显著HL数量分别为1,434、843、1,103；同位高Aa—低健康SHI共现为2,473个单元（33.5%），与邻域滞后HL类别不同。负向系数Gi*高值热点在SHI的GWR为2,484个、GWR-LCR为3,124个单元，模型敏感性明显。置换与Gi*近似p值不消除空间依赖。",
        "",
        figures[4], "", figures[5], "", figures[6], "", figures[7], "",
        "## 6. 指标与健康维度", "",
        "下表逐项保留健康指标原始方向。I-1的标准化β基于log1p(Aa)；I-2 β来自匹配对差值；I-4为各局部单元加权平均值；I-6为重叠邻域Spearman分布中位数。估计单位和模型不同，不据绝对值排序。",
        "", table, "",
        "物理维度（SW、BD）的整体负向信号较明显，且SW方向在全样本、匹配、GWR和局部相关结果中较一致。化学维度方向混合：OM较稳定负向，CEC多为正向，AN和pH的匹配/局部证据较弱或跨零。生物维度中MBC、RESP的匹配斜率分别为正，和全样本/局部GWR结果不完全一致，提示该维度反应不稳定。维度汇总为指标描述性统计，不是潜变量效应。",
        "",
        dim_table, "", figures[8], "",
        "当前工作区没有可读的PCNM结果表，配置 `pcnm.results=null`；既有PLS-SEM路径、载荷和方向资料也未提供。I-7已保留EnvMatch-PCNM比较字段和PLS路径交叉字段，但缺失值保持为空；没有重跑PCNM、推断路径或填造比较结果。因此目前不能判断EnvMatch与PCNM是否一致，也不能回答EnvMatch相对PCNM增加了何种独立解释。I-7森林图只展示可用的EnvMatch估计，不含PCNM对照；核心图30仅为源数据缺口标识。",
        "",
        "## 7. 空间胁迫候选分区与治理含义", "",
        "I-8在环境和SHI共同完整的7,387个约240m单元上，按Aa q80、环境调整SHI GWR系数q20、Aa q50和SHI q20构造互斥分类。所有阈值均是Oracle分位数，不是法定风险线。",
        "", zone_table, "",
        "T1约18.88 km²等价面积，是优先现场核验候选；T2约24.00 km²等价面积，优先核查是否存在环境/农艺缓冲；T3约2.05 km²等价面积，应以土壤实测诊断非侵蚀限制。环境候选另将27潮土—TWI q2空间代理和26冲积土—TWI q1空间代理单独展示；它们分别对应I-3全配对与另一低值筛选总体，不能合并成统一已验证敏感区。面积只按共同完整案例纳入的30m像元换算。",
        "",
        "建议在任何治理工程设计前现场核对侵蚀、地形、田块格局、灌排和作物制度。国家水土保持指导提出按地区和坡耕地条件选择工程、植物和保护性耕作措施；低SHI分区先做土壤检测，再确定培肥和养分管理。参见[I-8报告及官方依据](../../phase_i_20261031_seed20261031_i8_r2/reports/PHASE_I_SPATIAL_STRESS_ZONING_REPORT.md)。",
        "",
        figures[9], "",
        "## 8. 综合结论与下一步证据", "",
        "1. 现有预测栅格显示侵蚀与SHI、SW、OM总体以负向关联为主，SW的方向最稳定；CEC呈正向关联，AN、pH及生物指标存在方向或模型差异。",
        "2. M4匹配配对结果仍观察到SHI、OM、SW的负向配对关系，但幅度弱于端点像元关联；这支持在当前匹配设计下仍有条件关联线索，不证明充分消除环境混杂。",
        "3. 环境组、GWR、局部相关和热点方法均发现空间异质性；GWR与LCR对局部系数/热点量敏感。标准MGWR补充拟合使用空间均衡子样本，与I-4像元支持加权GWR的估计对象不同；空间效应大小仍有方法不确定性。",
        "4. 27:q2、26:q1和T1–T4属于Oracle候选，需用独立实测土壤样点和地块级侵蚀/农艺资料复核；不据当前空间预测面单独确定治理优先级或效果。",
        "5. PCNM与PLS-SEM源结果缺失，现阶段不能作跨方法一致性或机制路径结论；核心图30、31只显示数据缺口状态。后续需接入其实际结果表，并核实Aa单位/时间口径、SHI来源与指标方向；空间预测栅格仍应与独立实测样点分开报告。",
        "",
        "## 阶段报告与复现入口", "",
        *stage_reports,
        "",
        f"本综合材料生成于 Git HEAD `{git_head or 'unknown'}`。工作树内的报告与代码当前未提交；审阅请求要求在接收审查时先读取对应版本研究总纲，并结合本地工作树中的I-1至I-9材料审阅。",
        "",
        "全部图件索引见 [Phase I Figure Atlas](PHASE_I_FIGURE_ATLAS.html)。主要指标字段和PCNM/PLS空值交叉表见I-7运行的`tables/`目录；空间分区空间单元表、汇总和阈值见I-8运行的`tables/`目录。",
        "",
    ]
    closure_section = [
        "## 5.1 标准 MGWR 与核心图件闭合补充", "",
        f"本节对应独立运行 [{closure_dir.name}]({_rel(closure_report, reports)})。标准GWR与MGWR使用同一空间均衡样本：完整240m单元 {int(closure_sample['source_complete_240m_units']):,} 个，固定 {int(closure_sample['grid_cell_m'])}m网格每格抽取一个单元，n={int(closure_sample['sample_units']):,}；抽样只依据坐标，不按健康响应取样。模型含log1p(Aa_20yr)与九项环境协变量，Gaussian adaptive kernel，AICc带宽搜索，每个抽样单元等权。三项关键结局中 {closure_successes}/3 项完成标准GWR/MGWR拟合。",
        "", "标准GWR与MGWR在此共同样本中的拟合汇总：", "", closure_mgwr_table, "",
        "Aa暴露的MGWR尺度：", "", closure_scale_table, "",
        "该空间均衡子样本估计不等同于I-4全量240m单元、按源像元支持加权的GWR。局部系数是预测栅格条件关联，不作局部因果解释。MGWR带宽、收敛和局部共线性限制见补充报告。",
        "", *closure_images, "",
        "核心图30、31只标明PCNM与PLS-SEM源结果缺失；它们不是实际对照森林图或机制路径图。",
        "",
    ]
    body[body.index("## 6. 指标与健康维度"):body.index("## 6. 指标与健康维度")] = closure_section
    path = reports / "PHASE_I_FINAL_RESEARCH_REPORT.md"
    path.write_text("\n".join(body), encoding="utf-8")
    return path


def _build_review_request(run_dir: Path, git_head: str | None, report_path: Path,
                          config: dict[str, Any]) -> Path:
    reports = run_dir / "reports"
    path = reports / "PHASE_I_CHATGPT_REVIEW_REQUEST.md"
    text = f'''# Phase I 研究审查请求（EXPLORATORY_ORACLE）

**请首先读取本次Git提交对应版本的 `docs/RESEARCH_CHARTER.md`，完整理解EnvMatch-SEH的研究主题、总体研究目标、核心科学问题和长期技术路线，然后结合这些内容审查本阶段的代码、数据、实验结果和研究结论。**

## 审查版本与材料

- Git HEAD：`{git_head or "unknown"}`。
- I-9 run ID：`{config["run_id"]}`。
- 本次综合报告：[PHASE_I_FINAL_RESEARCH_REPORT.md]({_rel(report_path, reports)})。
- 当前工作树中的I-1至I-C源文件尚未提交。请在生成正式审查记录前确认审查者实际读取的是对应提交或本工作树快照；不要把上述HEAD误认为已经包含这些改动。
- 阶段范围：EXPLORATORY_ORACLE；所有新增运行均标记 `outcome_guided_selection=true`，输出位于 `outputs/exploratory_oracle/phase_i/`。
- 附件图谱：[PHASE_I_FIGURE_ATLAS.html](PHASE_I_FIGURE_ATLAS.html)。I-1至I-8分阶段报告及I-C图件/标准MGWR补充报告及其配置、表格和provenance由综合报告链接。

## 请重点审查

1. 是否完整回答章程中的环境匹配、异质性、健康指标差异与PCNM对照问题，同时保留EnvMatch作为新增分析模块的定位。
2. I-1/I-2估计对象是否区分全掩膜像元、匹配端点和配对差值；空间预测单元是否错误地被解释为独立实测样本。
3. Oracle结果导向筛选、非线性阈值、环境分组和I-8分位分区是否充分披露；各结论是否超出对应筛选总体和分析单位。
4. GWR/GWR-LCR、空间变化系数近似、局部RF、Local Spearman、Bivariate LISA与Gi*是否被准确命名，其不确定性和空间依赖是否与结论强度匹配。
5. I-C标准GWR/MGWR使用空间均衡样本，是否与I-4全样本加权GWR清楚区分，变量带宽及结果差异是否足够披露。
6. I-7因PCNM结果表和PLS-SEM路径/载荷源表缺失，保留空字段而不重跑或推断，是否符合要求；哪些对照结论应暂缓。
7. I-8将27潮土/TWI q2与26冲积土/TWI q1转为空间代理后，是否足够清楚地区分原匹配对分组与新空间分位代理。
8. 综合报告是否保留反向或不一致指标结果、模型失败/较弱泛化表现和本轮需要补齐的源数据。

## 本地输出索引

- I-1全样本：`{_rel(ROOT / config_source("i1", run_dir), reports)}`
- I-2匹配：`{_rel(ROOT / config_source("i2", run_dir), reports)}`
- I-3异质性：`{_rel(ROOT / config_source("i3", run_dir), reports)}`
- I-4空间系数：`{_rel(ROOT / config_source("i4", run_dir), reports)}`
- I-5空间机器学习：`{_rel(ROOT / config_source("i5", run_dir), reports)}`
- I-6局部空间关联：`{_rel(ROOT / config_source("i6", run_dir), reports)}`
- I-7健康维度与缺失源表登记：`{_rel(ROOT / config_source("i7", run_dir), reports)}`
- I-8候选分区：`{_rel(ROOT / config_source("i8", run_dir), reports)}`
- I-C图件闭合与标准MGWR：`{_rel(ROOT / config_source("closure", run_dir) / "reports/PHASE_I_CLOSURE_REPORT.md", reports)}`

## 审查边界

本材料是本地审查请求，不代表已向任何外部人员或服务发送。阶段结果尚未通过独立审查，也不代表正式研究设计或科学假设已获验证。请在审查中保留“预测栅格关联”“探索性候选”和“独立实测证据”之间的区分。
'''
    path.write_text(text, encoding="utf-8")
    return path


def config_source(key: str, run_dir: Path) -> str:
    config_path = run_dir / "configs/phase_i_final.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return config["source_runs"][key]


def _figure_caption(phase: str, stem: str) -> str:
    mapping = {
        "I01_global_association_heatmap": "全掩膜九指标关联",
        "I02_standardized_effect_forest": "标准化效应森林图",
        "I04_zero_positive_boxplots": "零值与正值侵蚀比较",
        "I09_Aa_20yr_matched_pair": "Aa_20yr匹配配对响应",
        "I11_m4_twi_group_slopes": "M4土类内TWI分组斜率",
        "I12_m4_interaction_q_heatmap": "环境交互检验热图",
        "gwr_SHI_candidate_template": "SHI环境调整GWR局部系数",
        "gwr_OM_template": "OM环境调整GWR局部系数",
        "gwr_SW_template": "SW环境调整GWR局部系数",
        "grf_local_Aa_marginal_effect": "局部RF Aa边际效应",
        "grf_local_r2": "局部RF验证R²",
        "local_ale_by_soil_type": "土类局部ALE",
        "bivariate_local_moran_clusters": "双变量Local Moran类别",
        "gistar_negative_response_gwr": "GWR负向响应Gi*",
        "gistar_negative_response_gwr-lcr": "GWR-LCR负向响应Gi*",
        "local_spearman_240m": "240m Local Spearman",
        "health_dimension_comparison_forest": "EnvMatch估计与健康维度对照",
        "stress_zoning_map": "主分区与环境敏感代理",
        "C04_global_best_model_r2": "全样本模型族验证与留出R²",
        "C05_MBC_segmented_threshold_candidate": "MBC验证选中分段模型候选阈值",
        "C07_soil_type_matched_response_curves": "土类分层M4配对响应",
        "C08_twi_group_matched_response_curves": "土类内TWI五分位配对响应",
        "C09_twi_interaction_slopes": "Aa与TWI分组斜率",
        "C10_ndvi_interaction_slopes": "Aa与NDVI分组斜率",
        "C11_environment_group_slope_forest": "环境组M4斜率森林图",
        "C12_environment_group_apparent_r2": "环境组表观R²",
        "C16_eight_indicator_gwr_composite": "八项健康指标GWR系数组合",
        "C17_MGWR_local_coefficients": "同样本标准GWR/MGWR系数",
        "C18_MGWR_variable_scales": "标准MGWR变量作用尺度",
        "C19_GWR_MGWR_difference_map": "MGWR减GWR局部系数差",
        "C30_PCNM_comparison_NOT_AVAILABLE": "PCNM源表缺失：比较未生成",
        "C31_mechanism_path_NOT_AVAILABLE": "PLS-SEM源表缺失：机制图未生成",
    }
    return f"{phase}｜{mapping.get(stem, stem.replace('_', ' '))}"


def _build_atlas(run_dir: Path) -> tuple[Path, int]:
    reports = run_dir / "reports"
    sections = []
    figure_count = 0
    for phase, source_run, section_label in FIGURE_SECTIONS:
        source_dir = PHASE_ROOT / source_run / "figures"
        cards = []
        for png in sorted(source_dir.rglob("*.png")):
            rel_img = _rel(png, reports)
            svg, pdf = png.with_suffix(".svg"), png.with_suffix(".pdf")
            caption = _figure_caption(phase, png.stem)
            links = []
            for ext, item in [("PNG", png), ("SVG", svg), ("PDF", pdf)]:
                if item.exists():
                    links.append(f'<a href="{html.escape(_rel(item, reports), quote=True)}">{ext}</a>')
            haystack = html.escape(f"{phase} {section_label} {png.stem}", quote=True).lower()
            cards.append(
                f'<article class="figure-card" data-search="{haystack}">'
                f'<img loading="lazy" src="{html.escape(rel_img, quote=True)}" alt="{html.escape(caption, quote=True)}">'
                f'<div class="caption"><strong>{html.escape(caption)}</strong>'
                f'<span>{html.escape(png.relative_to(source_dir).as_posix())}</span>'
                f'<nav>{" · ".join(links)}</nav></div></article>'
            )
            figure_count += 1
        if phase == "I-C":
            stage_report = PHASE_ROOT / source_run / "reports/PHASE_I_CLOSURE_REPORT.md"
        else:
            stage_report = next((PHASE_ROOT / source_run / "reports").glob("PHASE_I_*.md"), None)
        report_link = f'<a href="{html.escape(_rel(stage_report, reports), quote=True)}">阶段报告</a>' if stage_report else ""
        sections.append(f'<section id="{phase.lower()}"><header><h2>{phase} · {html.escape(section_label)}</h2>{report_link}</header><div class="grid">{"".join(cards)}</div></section>')
    report = reports / "PHASE_I_FINAL_RESEARCH_REPORT.md"
    html_text = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>EnvMatch-SEH Phase I Figure Atlas</title>
<style>
:root{{color-scheme:light;--ink:#17212b;--muted:#596b78;--line:#d7e0e5;--paper:#f4f7f8;--card:#fff;--accent:#1d6871}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.55 system-ui,-apple-system,"Noto Sans CJK SC",sans-serif}}
main{{max-width:1440px;margin:auto;padding:28px 26px 60px}}h1{{font-size:clamp(1.7rem,3vw,2.4rem);line-height:1.2;margin:.2em 0}}.lede{{color:var(--muted);max-width:900px}}
.toolbar{{position:sticky;top:0;background:#f4f7f8ee;backdrop-filter:blur(8px);padding:12px 0;z-index:5;border-bottom:1px solid var(--line);margin:20px 0}}
input{{width:min(100%,480px);padding:10px 12px;border:1px solid #aabac3;border-radius:6px;font:inherit}}.nav{{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}}.nav a{{color:var(--accent);text-decoration:none;border:1px solid #a7c4c8;padding:3px 9px;border-radius:999px;font-size:.9rem}}
section{{margin:34px 0}}section header{{display:flex;align-items:baseline;gap:18px;flex-wrap:wrap;border-bottom:1px solid var(--line);margin-bottom:14px}}h2{{font-size:1.3rem;margin:.3em 0 .6em}}section header a{{color:var(--accent)}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,380px),1fr));gap:16px}}
.figure-card{{background:var(--card);border:1px solid var(--line);border-radius:8px;overflow:hidden;box-shadow:0 2px 8px #17212b0b}}.figure-card img{{display:block;width:100%;height:auto;aspect-ratio:4/3;object-fit:contain;background:#fff;border-bottom:1px solid var(--line)}}.caption{{padding:11px 13px;display:grid;gap:3px}}.caption span{{font-size:.82rem;color:var(--muted);overflow-wrap:anywhere}}.caption nav a{{color:var(--accent);font-size:.9rem}}.foot{{border-top:1px solid var(--line);padding-top:14px;color:var(--muted)}}
</style></head><body><main><h1>EnvMatch-SEH · Phase I 图件索引</h1>
<p class="lede">按I-1至I-8及I-C补充运行汇总Oracle结果中的PNG预览；支持的SVG和PDF原件可从每张图下方打开。全部结果保留各自分析单位、来源与限制，完整综合结论见 <a href="{html.escape(_rel(report, reports), quote=True)}">Phase I最终研究报告</a>。</p>
<div class="toolbar"><label for="search">筛选图件</label><br><input id="search" type="search" placeholder="按阶段、主题或文件名筛选" autocomplete="off"><nav class="nav">{''.join(f'<a href="#{phase.lower()}">{phase}</a>' for phase,_,_ in FIGURE_SECTIONS)}</nav></div>
<p id="count">当前列出 {figure_count} 张图件。</p>
{"".join(sections)}
<p class="foot">显示的是各权威运行中的现有图件。I-C在空间均衡子样本上增加标准GWR/MGWR；I-7仍缺PCNM和PLS-SEM源表，核心图30、31为数据缺口状态图。`outcome_guided_selection=true`。</p>
</main><script>const q=document.querySelector('#search'),cards=[...document.querySelectorAll('.figure-card')],count=document.querySelector('#count');q.addEventListener('input',()=>{{const term=q.value.trim().toLowerCase();let n=0;for(const c of cards){{const show=c.dataset.search.includes(term);c.hidden=!show;if(show)n++;}}count.textContent=`显示 ${{n}} / ${{cards.length}} 张图件。`;}});</script></body></html>'''
    path = reports / "PHASE_I_FIGURE_ATLAS.html"
    path.write_text(html_text, encoding="utf-8")
    return path, figure_count


def run(config_path: Path) -> Path:
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("mode") != "EXPLORATORY_ORACLE" or config.get("outcome_guided_selection") is not True:
        raise ValueError("I-9 must retain the Oracle mode marker.")
    run_dir = ROOT / config["output_root"] / config["run_id"]
    if run_dir.exists():
        raise FileExistsError(f"Output run already exists; use a new run ID: {run_dir}")
    for folder in ["configs", "reports", "manifests"]:
        (run_dir / folder).mkdir(parents=True)
    (run_dir / "configs/phase_i_final.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    git_head = _git("rev-parse", "HEAD")
    report_path = _build_report(run_dir, config, git_head)
    review_path = _build_review_request(run_dir, git_head, report_path, config)
    atlas_path, figure_count = _build_atlas(run_dir)

    provenance = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "run_id": config["run_id"],
        "phase": "I-9", "mode": "EXPLORATORY_ORACLE", "outcome_guided_selection": True,
        "git_head": git_head, "git_status_porcelain": _git("status", "--porcelain"),
        "config_path": str(config_path.relative_to(ROOT)), "source_runs": config["source_runs"],
        "source_data": "Previously reviewed Phase I-1 through I-8 outputs plus I-closure; standard MGWR run on a spatially balanced subset; no RUSLE reconstruction or PCNM rerun.",
        "figure_atlas_count": figure_count,
        "pcnm_source": "not provided; crosswalk fields remain blank; PCNM not rerun",
        "pls_sem_source": "not provided; path fields remain blank",
        "review_request": "created locally; not sent externally",
        "software": {"python": platform.python_version(), "pandas": pd.__version__, "pyyaml": _version("PyYAML")},
        "reports": [str(report_path.relative_to(ROOT)), str(review_path.relative_to(ROOT)), str(atlas_path.relative_to(ROOT))],
    }
    (run_dir / "manifests/provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_files = sorted(p for p in run_dir.rglob("*") if p.is_file() and p.name != "output_manifest.json")
    manifest = {"run_id": config["run_id"], "files": [
        {"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size} for p in manifest_files]}
    (run_dir / "manifests/output_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "RUN_STATUS.md").write_text(
        f"# {config['run_id']}\n\nStatus: complete\n\nMode: EXPLORATORY_ORACLE\n\noutcome_guided_selection: true\n\n"
        f"Figure count: {figure_count}\n\nReport: `{report_path.relative_to(ROOT)}`\n\n"
        "The review request is a local artifact and was not sent externally.\n", encoding="utf-8")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/phase_i_final.yaml")
    args = parser.parse_args()
    print(run(args.config))


if __name__ == "__main__":
    main()
