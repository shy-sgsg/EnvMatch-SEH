# Phase F–J研究结果总索引

本索引把匹配设计、跨阶段响应、空间结果和结构路径按研究问题联接。所有新分析来自EXPLORATORY_ORACLE，`outcome_guided_selection=true`；预测栅格不能当作独立实测样点。CEC在图表中统一写作CEC，其计算公式与侵蚀暴露直接耦合；PCNM保留并后续对照。

## 三条核心发现

- [三条核心科学发现](../reports/PHASE_J_CORE_SCIENTIFIC_FINDINGS.md)
- [十个核心研究问题的回答](../reports/PHASE_J_RESEARCH_QUESTIONS_ANSWERS.md)
- [Phase F–I跨阶段发现](../reports/PHASE_F_TO_I_SCIENTIFIC_FINDINGS.md)
- [证据交叉索引CSV](RESEARCH_EVIDENCE_CROSSWALK.csv)

## 核心图件

| 图 | 主题 | PNG | 源数据/说明 |
|---|---|---|---|
| 图A | 全域、匹配配对、零/正状态与GAM响应 | [PNG](../figures/A_multimethod_erosion_health_response.png) | [说明/数据](../figures/data/figure_A_global_zero_pair_metrics.csv)；另有SVG和PDF |
| 图B | SHI、OM、SW的GWR/MGWR空间系数 | [PNG](../figures/B_gwr_mgwr_spatial_patterns.png) | [说明/数据](../spatial_synthesis/GWR_MGWR_LOCAL_COEFFICIENT_COMPARISON.csv)；另有SVG和PDF |
| 图C | T1–T4环境候选分区 | [PNG](../figures/C_stress_sensitive_environment_zones.png) | [说明/数据](../spatial_synthesis/SPATIAL_STRESS_ENVIRONMENT_PROFILE.csv)；另有SVG和PDF |
| 图D | 不同控制层级的侵蚀路径 | [PNG](../figures/D_sem_control_hierarchy.png) | [说明/数据](../reports/PHASE_J_FIGURE_CAPTIONS.md)；另有SVG和PDF |
| 图E | 全域与匹配端点路径 | [PNG](../figures/E_full_vs_matched_sem.png) | [说明/数据](../reports/PHASE_J_FIGURE_CAPTIONS.md)；另有SVG和PDF |
| 图F | 侵蚀对三个健康维度的路径 | [PNG](../figures/F_erosion_dimension_path_forest.png) | [说明/数据](../reports/PHASE_J_FIGURE_CAPTIONS.md)；另有SVG和PDF |
| 图G | 拟合度与侵蚀增量 | [PNG](../figures/G_sem_fit_and_erosion_increment.png) | [说明/数据](../reports/PHASE_J_FIGURE_CAPTIONS.md)；另有SVG和PDF |
| 图H | 综合条件路径图 | [PNG](../figures/H_best_sem_path_diagram.png) | [说明/数据](../reports/PHASE_J_FIGURE_CAPTIONS.md)；另有SVG和PDF |
| 图I | 土类×湿度组路径差异 | [PNG](../figures/I_soil_twi_sem_comparison.png) | [说明/数据](../reports/PHASE_J_FIGURE_CAPTIONS.md)；另有SVG和PDF |


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

主要阶段报告：`docs/reviews/2026-10-26/PHASE_F_RESEARCH_REPORT.md`、`docs/reviews/2026-10-27/phase-g/PHASE_G_RESEARCH_REPORT.md`、`docs/reviews/2026-10-29/phase-h/PHASE_H_FINAL_SUMMARY.md`、`docs/reviews/2026-10-31/phase-i/PHASE_I_FINAL_RESEARCH_REPORT.md`。逐条数据源、样本、数值与限制见本目录证据交叉表。
