"""Write the Chinese Phase H result report and a review-ready handoff note."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


OUTCOMES = ["SHI_candidate", "AN", "OM", "CEC", "pH", "SW", "BD", "MBC", "RESP"]
LABELS = {
    "SHI_candidate": "土壤健康指数（SHI）", "AN": "碱解氮（AN）", "OM": "有机质（OM）",
    "CEC": "阳离子交换量（CEC）", "pH": "土壤酸碱度（pH）", "SW": "土壤含水量（SW）",
    "BD": "土壤容重（BD）", "MBC": "微生物量碳（MBC）", "RESP": "土壤呼吸（RESP）",
}
EXPOSURE_LABELS = {"Aa_20yr": "20 年平均实际侵蚀", "Aa_2020": "2020 年实际侵蚀"}
ESTIMAND_LABELS = {"matched_pixel": "匹配像元", "matched_pair": "匹配配对"}
GROUP_METHOD_LABELS = {
    "soil_type_group": "土壤类型分层",
    "kmeans_terrain_k4": "高程与地表温度 K 均值聚类（k=4）",
    "kmeans_terrain_k6": "高程与地表温度 K 均值聚类（k=6）",
    "kmeans_terrain_k8": "高程与地表温度 K 均值聚类（k=8）",
    "hierarchical_ward_v5": "分土类沃德聚类",
    "gradient_Elevation": "高程梯度",
    "gradient_TWI": "地形湿度指数梯度",
    "gradient_LST": "地表温度梯度",
    "gradient_NDWI": "水体指数梯度",
    "gradient_NDVI": "植被指数梯度",
    "gradient_Precipitation": "降水量梯度",
}
STATUS_LABELS = {
    "reliable": "通过可靠性检查",
    "unreliable_covariance_rank_deficient": "协方差秩不足",
    "unreliable_group_block_coverage": "空间块支持不足",
    "unreliable_tolerance_sensitive": "对数值容差敏感",
    "unreliable_design_condition": "设计矩阵条件不佳",
    "non_estimable_design_rank": "设计矩阵不可估",
}


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False) if path.is_file() else pd.DataFrame()


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(number):
        return "—"
    if abs(number) >= 1000:
        return f"{number:,.0f}"
    return f"{number:.{digits}f}"


def _markdown_table(frame: pd.DataFrame, columns: list[tuple[str, str]], *, limit: int | None = None) -> str:
    if frame.empty:
        return "（无可用记录）"
    rows = frame.head(limit) if limit else frame
    headers = [label for _, label in columns]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows.to_dict("records"):
        cells = []
        for key, _ in columns:
            value = row.get(key)
            if isinstance(value, (float, np.floating)):
                value = _fmt(value)
            if pd.isna(value):
                value = "—"
            cells.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _response_summary(tables: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    ranks = _read(tables / "phase_h_nonparametric_response_summary.csv")
    pairs = _read(tables / "phase_h_paired_difference_summary.csv")
    if ranks.empty:
        return ranks, pairs
    rank_summary = ranks.groupby(["exposure", "outcome", "estimand"], as_index=False).agg(
        median_spearman_rho=("spearman_rho", "median"),
        median_kendall_tau=("kendall_tau", "median"),
        median_jt_tau_b=("jt_tau_b", "median"),
        median_jt_q=("jt_q_value_global", "median"),
        candidate_count=("candidate_id", "nunique"),
        median_n=("n", "median"),
    )
    if not pairs.empty:
        pair_summary = pairs.groupby(["exposure", "outcome"], as_index=False).agg(
            median_delta_health=("median_delta_health", "median"),
            mean_delta_health=("mean_delta_health", "median"),
            negative_delta_fraction=("negative_delta_fraction", "median"),
            median_wilcoxon_q=("wilcoxon_q_value_global", "median"),
            median_sign_q=("sign_test_q_value_global", "median"),
            candidate_count=("candidate_id", "nunique"),
        )
    else:
        pair_summary = pd.DataFrame()
    return rank_summary, pair_summary


def _figure_links(run_id: str) -> str:
    root = f"../../../../outputs/exploratory_oracle/{run_id}/figures/png"
    figures = [
        ("01_matched_response_direction", "方向总览"),
        ("02_matched_endpoint_boxplots", "匹配两端箱线图"),
        ("03_apparent_r2_pixel_vs_pair", "像元与配对 R²"),
        ("04_rank_association_vs_r2", "斯皮尔曼秩相关与 R²"),
        ("05a_fixed_20k_natural_spline", "固定 20,000 样本自然样条"),
        ("05b_historical_full_sample_apparent", "历史全样本表观曲线"),
        ("05c_matched_pixel_natural_spline", "匹配像元曲线"),
        ("05d_matched_pair_natural_spline", "匹配配对曲线"),
        ("06_group_effect_forest", "土类分组森林图"),
        ("07_environment_group_method_comparison", "分组方法比较"),
        ("08_selected_group_map", "入选环境组地图"),
        ("09_apparent_vs_spatial_test", "表观与空间测试"),
        ("10_h1_interaction_inference_repair", "H1 推断修订"),
        ("11_common_sample_period_comparison", "共同样本周期对比"),
    ]
    return "\n".join(f"- [{label}]({root}/{name}.png)" for name, label in figures)


def write_phase_h_report(project_root: Path, run_dir: Path, config: dict[str, Any]) -> tuple[Path, Path]:
    """Write a durable, evidence-linked report and a ChatGPT review request."""

    project_root = Path(project_root)
    run_dir = Path(run_dir)
    tables = run_dir / "tables"
    review_dir = project_root / "docs" / "reviews" / "2026-10-28" / "phase-h"
    review_dir.mkdir(parents=True, exist_ok=True)
    report_path = review_dir / "PHASE_H_EXPLORATORY_REPORT.md"
    request_path = review_dir / "CHATGPT_REVIEW_REQUEST.md"

    provenance = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
    population = _read(tables / "matched_dataset_summary.csv")
    ranks, paired = _response_summary(tables)
    groups = _read(tables / "phase_h_environment_group_method_comparison.csv")
    profiles = _read(tables / "phase_h_environment_group_profiles.csv")
    spatial = _read(tables / "phase_h_spatial_validation_summary.csv")
    periods = _read(tables / "phase_h_period_comparison_summary.csv")
    interaction = _read(tables / "interaction_test_comparison.csv")
    trial_path = tables / "phase_h_model_trials.csv"
    trials = _read(trial_path)

    if not paired.empty and not ranks.empty:
        pair_ranks = ranks.loc[ranks["estimand"].eq("matched_pair")].merge(
            paired, on=["exposure", "outcome"], how="left", suffixes=("_rank", "_pair"), validate="one_to_one"
        )
    else:
        pair_ranks = pd.DataFrame()

    if not spatial.empty:
        spatial_500 = spatial.loc[pd.to_numeric(spatial["buffer_m"], errors="coerce").eq(500)].copy()
        spatial_500["period"] = spatial_500["exposure"].map(EXPOSURE_LABELS)
        spatial_summary = spatial_500.groupby(["estimand", "period"], as_index=False).agg(
            median_apparent_r2=("mean_apparent_r_squared", "median"),
            median_spatial_test_r2=("median_spatial_test_r_squared", "median"),
            median_train_n=("mean_n_train", "median"),
            median_test_n=("mean_n_test", "median"),
            median_pair_retention=("mean_same_block_pair_retention", "median"),
        )
        spatial_summary["estimand"] = spatial_summary["estimand"].map(ESTIMAND_LABELS)
    else:
        spatial_summary = pd.DataFrame()

    if not periods.empty:
        period_summary = periods.groupby("estimand", as_index=False).agg(
            median_common_sample_n=("common_sample_n", "median"),
            min_common_sample_n=("common_sample_n", "min"),
            max_common_sample_n=("common_sample_n", "max"),
            median_20yr_minus_2020_test_r2=("mean_test_r2_difference_20yr_minus_2020", "median"),
            comparisons=("outcome", "nunique"),
        )
        period_summary["estimand"] = period_summary["estimand"].map(ESTIMAND_LABELS)
    else:
        period_summary = pd.DataFrame()

    if not interaction.empty:
        interaction_summary = interaction.groupby("revised_status", dropna=False, as_index=False).agg(
            test_count=("outcome", "size"),
            reliable_count=("revised_inference_reliable", lambda values: int(pd.Series(values).fillna(False).astype(bool).sum())),
            legacy_q_lt_05=("legacy_q_lt_0_05", "sum"),
            revised_q_lt_05=("revised_q_lt_0_05", "sum"),
        )
    else:
        interaction_summary = pd.DataFrame()
    if not interaction_summary.empty:
        interaction_summary["revised_status"] = interaction_summary["revised_status"].map(STATUS_LABELS).fillna(interaction_summary["revised_status"])

    cell_rows = int(provenance.get("cell_rows", 0))
    pair_rows = int(provenance.get("pair_rows", 0))
    model_count = len(trials)
    nonparam_rows = len(_read(tables / "phase_h_nonparametric_response_summary.csv"))
    paired_rows = len(_read(tables / "phase_h_paired_difference_summary.csv"))
    group_methods = int(groups["group_method"].nunique()) if not groups.empty else 0
    profile_count = len(profiles)
    spatial_score_count = len(_read(tables / "phase_h_spatial_validation_by_split.csv"))
    period_score_count = len(_read(tables / "phase_h_period_comparison_by_split.csv"))
    run_id = str(config.get("run_id", run_dir.name))

    if not pair_ranks.empty:
        pair_ranks["outcome_label"] = pair_ranks["outcome"].map(LABELS)
        pair_ranks["exposure_label"] = pair_ranks["exposure"].map(EXPOSURE_LABELS)
        pair_ranks = pair_ranks.sort_values(["exposure", "outcome"], key=lambda col: col.map({v: i for i, v in enumerate(OUTCOMES)}) if col.name == "outcome" else col)
    rank_table = _markdown_table(
        pair_ranks,
        [
            ("exposure_label", "暴露"), ("outcome_label", "健康指标"),
            ("median_spearman_rho", "配对斯皮尔曼 ρ"), ("median_jt_tau_b", "JT 有序趋势 τ-b"),
            ("median_delta_health", "中位配对健康差值"), ("negative_delta_fraction", "健康差值为负的比例"),
            ("candidate_count_rank", "候选数"),
        ],
    )
    spatial_table = _markdown_table(
        spatial_summary,
        [("estimand", "分析单位"), ("period", "暴露"), ("median_apparent_r2", "表观 R² 中位数"),
         ("median_spatial_test_r2", "500 m 空间测试 R² 中位数"), ("median_train_n", "训练样本中位数"),
         ("median_test_n", "测试样本中位数"), ("median_pair_retention", "同块配对保留率")],
    )
    period_table = _markdown_table(
        period_summary,
        [("estimand", "分析单位"), ("median_common_sample_n", "共同样本中位数"),
         ("min_common_sample_n", "最小共同样本"), ("max_common_sample_n", "最大共同样本"),
         ("median_20yr_minus_2020_test_r2", "20 年−2020 测试 R²差中位数"), ("comparisons", "指标数")],
    )
    group_display = groups.sort_values("median_spearman_rho").copy()
    group_display["group_method_label"] = group_display["group_method"].map(GROUP_METHOD_LABELS).fillna(group_display["group_method"])
    group_table = _markdown_table(
        group_display,
        [("group_method_label", "环境分组方法"), ("groups_profiled", "画像组数"),
         ("median_pairs_per_group", "组内配对数中位数"), ("median_spearman_rho", "组内斯皮尔曼 ρ 中位数"),
         ("median_within_group_nonlinear_r_squared", "组内样条 R²中位数"),
         ("mean_adjusted_rand_index", "K 均值聚类稳定性 ARI（其他不适用）")],
    )
    interaction_table = _markdown_table(
        interaction_summary,
        [("revised_status", "修订状态"), ("test_count", "检验数"), ("reliable_count", "可靠数"),
         ("legacy_q_lt_05", "旧版 q<0.05"), ("revised_q_lt_05", "修订 q<0.05")],
    )

    sections = f"""# H 阶段环境匹配后侵蚀—土壤健康响应：探索性结果报告

**运行 ID：** `{run_id}`  
**分析模式：** 结果导向探索模式（配置标识 `EXPLORATORY_ORACLE`；`outcome_guided_selection=true`）  
**输入数据：** 对齐数据运行 `{provenance.get('input_run_id', '—')}`；F 阶段 `{provenance.get('phase_f_run_id', '—')}`；G 阶段 `{provenance.get('phase_g_run_id', '—')}`  
**记录的代码版本：** `{provenance.get('git_sha', '—')}`（运行时工作区是否有未提交修改：`{provenance.get('git_dirty', 'unknown')}`）

## 1. 研究问题与本轮结论范围

本轮考察环境相似的匹配样本中，实际侵蚀 Aa 与 SHI 及 AN、OM、CEC、pH、SW、BD、MBC、RESP 的响应关系。主侵蚀暴露直接读取已算好的 `Aa_20yr` 与 `Aa_2020`；未重算或验证 RUSLE 因子乘积。健康响应来自已有土壤健康预测栅格。

本轮采用结果导向的探索性择优模式：匹配方案、候选模型及展示排名会读取健康结果。图表中的关联和 R² 属于表观探索结果，不能解释成因果效应、独立验证或预先锁定的正式证据。旧 PCNM 结果予以保留，本轮没有重新运行 PCNM。

## 2. 数据和样本单位

- 对齐耕地栅格单元：**{cell_rows:,}**；输入 F 阶段匹配对：**{pair_rows:,}**。
- H2 将匹配像元成员、去重像元和匹配对分开保存。匹配对中的一个像元可出现在多个匹配关系里；配对数不能当成独立实测样点数。
- 分析只纳入预先约定的主土类 26、27、42。主匹配环境变量不含 R、K、LS、C、P；实际 Aa 是暴露量，非匹配协变量。
- 像元单位的健康值来自 30 m 预测栅格。它们不是同等数量的独立土壤实测点；报告里像元量、配对量、空间块量各自分开。

## 3. 方法：不预设线性响应

H4 汇总斯皮尔曼秩相关系数 ρ、肯德尔等级相关系数 τ、按侵蚀分组排序的 JT 有序趋势检验，并对高侵蚀端减低侵蚀端的配对健康差值进行威尔科克森符号秩检验和符号检验。H3 同时比较线性、对数、二次、分段、自然样条、广义可加模型（GAM）、稳健回归、分位数回归及可用的树模型；主图把像元健康值模型与配对差值模型分开。

H5 的环境分组只使用土壤类型、高程/地表温度聚类或环境梯度；健康结果用于描述分组内响应和探索性排名，但不参与分组标签生成。分组内自然样条 R²是表观拟合。森林图只显示通过可靠性门槛的土类分组线性斜率，作为辅助诊断，不作为唯一响应模型。

## 4. 各健康指标的配对方向与响应汇总

下表针对每个侵蚀暴露和健康指标，将第二阶段候选的配对分析汇总为中位数。`中位配对健康差值`与“健康差值为负的比例”均按高侵蚀端减低侵蚀端计算。q 值保留在底层表中；大量像元和配对共享空间位置，因此不能仅凭很小的 p/q 值判断实质效应或独立重复。

{rank_table}

非参数结果原表含 **{nonparam_rows:,}** 行候选×指标×暴露×分析单位汇总；配对差值表含 **{paired_rows:,}** 行候选汇总。请结合箱线图、秩关联和配对差值阅读，不把单个模型 R²当作结论。

## 5. 多种模型拟合与单位拆分

H3/H4 持久化 **{model_count:,}** 条模型/子集试验记录，包含模型类别、样本量、指标、方向、R²、误差和结果导向筛选信息。像元 R²回答“侵蚀暴露对像元健康值的预测拟合”；配对 R²回答“侵蚀差与健康差是否一起变化”。两种分析的响应单位和样本依赖结构不同，因此没有合并成一个 R²。

图 5 将四类曲线分开：固定同一批最多 20,000 个有效像元拟合的自然样条；历史 F 阶段全样本模型择优表观曲线；本轮匹配像元曲线；本轮匹配配对曲线。历史曲线来自较早的探索性运行，模型按全样本表观 R²选择，横轴沿用原侵蚀量尺度；不能把它当成独立检验，也不能与配对差值曲线直接比较。

## 6. 环境异质性、分组方法和地图

H5 覆盖 **{group_methods}** 种环境分组方法，生成 **{profile_count:,}** 条分组响应画像（分组标签由环境数据生成）。方法比较如下：

{group_table}

K 均值聚类的调整兰德指数（ARI）只表示更换随机种子后分组是否一致；土类、沃德聚类和梯度等确定性分组不适用该指标。地图使用真实投影米制坐标，并将跨指标、侵蚀时期和候选中组内斯皮尔曼相关中位数最负的 k=8 环境组高亮。该组经健康结果筛选，仅供探索性展示，不是独立验证过的侵蚀热点，也不是连续风险面。

## 7. H1 交互推断修订

修订流程检查设计矩阵秩、条件数、协方差秩/条件数、空间块支持量及伪逆容差敏感性；不可靠时抑制交互 p 值。旧/新结果汇总如下：

{interaction_table}

图 10 将旧版与修订 q 值对照；空缺/不可靠状态不等于“无交互”，而是当前数据结构不足以支持可靠检验。森林图只画可靠行。

## 8. 空间留出与两个侵蚀时期对比

H6 共保存 **{spatial_score_count:,}** 条空间验证/测试分数。匹配对两端先限定在同一个 5 km 网格块；整个块只进入训练、验证或测试中的一个角色；训练样本再按 300、500、1,000 m 与留出样本的距离缓冲剔除。验证候选与模型族仍由全样本表观结果挑选，故该测试说明被选模型的空间迁移表现，不能充当完全独立的模型择优验证。

500 m 缓冲的概况：

{spatial_table}

H7 在共同像元编号或共同配对端点上，用同一自然样条、同一空间块划分和 500 m 缓冲比较两个侵蚀时期。配对方向始终按 20 年平均实际侵蚀量固定，避免时期变化导致“高端/低端”符号翻转。本轮有 **{period_score_count:,}** 条时期比较分数；共同样本规模如下：

{period_table}

这是一项同样本描述性周期比较，不是周期的因果效应。图 11 的区间表示 10 次空间划分的四分位数。

## 9. 主要限制与阅读建议

1. 这是结果导向的探索性分析：健康结果参与候选选择。筛出的强负关联、较高 R² 或优势分组会有选择偏差，需与未筛选基线、保留率和空间测试差距一起看。
2. 匹配提高环境可比性，但并未排除所有混杂，也不等于随机试验。不能把“环境匹配后关联”写成侵蚀导致健康变化。
3. 30 m 健康预测栅格之间有空间依赖；配对复用像元也会相关。像元数和配对数都不等同于独立实测样点数。
4. 侵蚀用的是已有实际 Aa；本轮按用户要求没有重建或验证 RUSLE 因子乘积。潜在侵蚀 Ap 不是本轮主分析。
5. 空间验证检验的是已由全样本结果挑中的候选/模型。它能暴露地理泛化变差，但不能消除前面的结果导向选择。
6. 环境分组画像的组内自然样条 R²和地图是探索性、表观结果；图表中的 95%/分位数区间不能自动解释为独立重复的不确定性。

## 10. 文件与交给 ChatGPT 的审阅问题

图表均导出 PNG、SVG、PDF，并伴随 CSV 数据和图注：

{_figure_links(run_id)}

主要可复核表位于 `{run_dir.relative_to(project_root)}/tables/`，包括输入清单、H1 诊断、匹配样本、全部试验注册、非参数检验、配对差值、环境组画像、空间验证和共同样本时期对比。图表的数据源位于 `{run_dir.relative_to(project_root)}/figures/data/`。

建议 ChatGPT 审阅：

- 是否把预测像元、配对关系和真实独立测点的统计单位区分清楚？
- 非参数、配对差值和非线性曲线能否共同回答“环境匹配后响应如何”，是否有符号或暴露周期解释错误？
- 图 9 的空间留出结果是否被恰当地称为对已选模型的迁移诊断？
- 哪些结果可以谨慎放入探索性论文叙述，哪些必须降格为方法筛选或表观现象？

"""

    report_path.write_text(sections, encoding="utf-8")
    request = f"""# ChatGPT 审阅请求：H 阶段环境匹配分析

请先阅读本目录下的 [`PHASE_H_EXPLORATORY_REPORT.md`](PHASE_H_EXPLORATORY_REPORT.md)，再重点审查：

1. 每项健康指标的响应总结是否正确解释斯皮尔曼秩相关、肯德尔等级相关、JT 有序趋势检验和高侵蚀端减低侵蚀端的配对差值；
2. 匹配像元与匹配对的单位、依赖性和 R² 是否清楚分开；
3. 结果导向筛选、全样本表观曲线及空间验证的限制是否披露充分；
4. 5 km 整块分配、300/500/1,000 m 训练缓冲，以及两期共同端点和按 20 年平均侵蚀量固定配对方向是否有逻辑漏洞；
5. 图表中哪些结论有数据支持，哪些表述过强、应删除或改写；
6. 有无结果/图表与原始 CSV 不一致或容易误读之处。

本请求仅供用户复制给 ChatGPT 审阅，尚未向外发送。运行 ID：`{run_id}`；项目代码版本：`{provenance.get('git_sha', '—')}`。
"""
    request_path.write_text(request, encoding="utf-8")
    return report_path, request_path
