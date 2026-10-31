# Phase I-3：环境异质性与敏感环境候选

## 数据和复用范围

本阶段整合 Phase G 全配对 M4 的环境组斜率/交互表，以及 Phase H 低值筛选后匹配环境组结果；不重做匹配、不重跑聚类，也不改既有分组。主分析 Aa_20yr，Aa_2020 作为附表敏感情景。Phase G 的组内斜率为配对差值模型，按 5 km 空间块聚类计算区间；Phase H 表还按零/正值筛选重设了匹配支持，不能与全配对基线合并。

分组定义沿用已有输出：土类标签使用高侵蚀端土类；Phase F V5 与土类内 Ward 使用 Elevation/LST 相关环境聚类；Elevation、TWI、LST、NDWI、NDVI、降水在各土类内分五分位；另有土类内环境聚类 K=4/6/8/10/12。原分组标签以环境变量建立，健康结果用于后续 Oracle 关联与候选筛选。R、K、LS、C、P 和 Aa 不进入环境距离向量。

## 土类及连续环境差异：M4 Aa_20yr

### 土壤类型

在三个主要土类中，27 潮土的 SHI、OM、SW 标准化负斜率绝对值均最大，支持量也最大。分组样本是匹配对，不表示独立实测点。

| 指标          | 土类    |   配对数 |   5km块数 |   标准化效应 |   原始斜率 |   CI下限 |   CI上限 |
|:--------------|:--------|---------:|----------:|-------------:|-----------:|---------:|---------:|
| OM            | soil_26 |    20613 |        19 |       -0.078 |     -0.533 |   -0.892 |   -0.173 |
| OM            | soil_27 |    58514 |        24 |       -0.294 |     -2.019 |   -2.411 |   -1.627 |
| OM            | soil_42 |    11740 |         8 |       -0.162 |     -1.110 |   -1.394 |   -0.827 |
| SHI_candidate | soil_26 |    20302 |        19 |       -0.150 |     -0.013 |   -0.015 |   -0.010 |
| SHI_candidate | soil_27 |    57979 |        24 |       -0.246 |     -0.021 |   -0.026 |   -0.016 |
| SHI_candidate | soil_42 |    11530 |         8 |       -0.108 |     -0.009 |   -0.011 |   -0.007 |
| SW            | soil_26 |    20613 |        19 |       -0.210 |     -2.018 |   -2.483 |   -1.553 |
| SW            | soil_27 |    58514 |        24 |       -0.316 |     -3.046 |   -3.466 |   -2.627 |
| SW            | soil_42 |    11740 |         8 |       -0.205 |     -1.976 |   -2.432 |   -1.521 |

### TWI 梯度

TWI 在每个土类内分五分位。全配对 M4 中，27:q2（潮土内第二个 TWI 五分位）在 SHI、OM、SW 三项的标准化效应最负；这不等价于所有土类中的低 TWI 一律更敏感。

| 指标          | 土类内TWI组   |   配对数 |   5km块数 |   标准化效应 |   原始斜率 |   CI下限 |   CI上限 |
|:--------------|:--------------|---------:|----------:|-------------:|-----------:|---------:|---------:|
| OM            | 26:q1         |     4848 |        19 |       -0.088 |     -0.598 |   -0.839 |   -0.356 |
| OM            | 27:q2         |     8865 |        24 |       -0.397 |     -2.693 |   -3.199 |   -2.186 |
| OM            | 42:q1         |     2584 |         8 |       -0.208 |     -1.412 |   -1.999 |   -0.825 |
| SHI_candidate | 26:q1         |     4758 |        19 |       -0.113 |     -0.009 |   -0.012 |   -0.006 |
| SHI_candidate | 27:q2         |     8799 |        24 |       -0.337 |     -0.027 |   -0.034 |   -0.021 |
| SHI_candidate | 42:q1         |     2540 |         8 |       -0.113 |     -0.009 |   -0.014 |   -0.005 |
| SW            | 26:q1         |     4848 |        19 |       -0.224 |     -2.144 |   -2.501 |   -1.786 |
| SW            | 27:q2         |     8865 |        24 |       -0.413 |     -3.956 |   -4.546 |   -3.367 |
| SW            | 42:q1         |     2584 |         8 |       -0.263 |     -2.517 |   -3.460 |   -1.575 |

### NDVI、NDWI 与 LST

在同一全配对 M4 Aa_20yr 表中，SHI/OM/SW 最负向的 NDVI 组集中于 27:q5，NDWI 组集中于 27:q1（OM、SW）和 26:q5（SHI），LST 组集中于 27:q3。不同指标的组别不完全一致；具体估计、支持量和全部其他组均保存在 `tables/environment_group_slopes_m4.csv` 及 `leaderboards/best_supported_environment_group.csv`。

## 联合环境组和斜率异质性

在 Aa_20yr 的 126 项 M4 组间交互检验中，123/126 项全局 BH q<0.05；跨两期、M3/M4 的 504 项检验中 473/504 项 q<0.05。联合 K-means 分组对重点指标的负斜率候选多数位于 27 潮土的 Elevation/LST 聚类；例如 K=6 的 cluster 2 在 SHI、OM、SW 中均较负，K=8 的 cluster 5 在 OM、SW 较负。热图对 `−log10(q)` 显示上限 12，精确 log(q) 和估计支持保留在 CSV。

交互检验和组内结果来自同一探索流程，q 值校正范围为已登记的 504 项，但不能覆盖历史上所有 Oracle 筛选和空间依赖的不确定性。将其视作异质性线索，不直接解释成预注册确认。

## 26 冲积土低 TWI 候选：Phase H 低值筛选对照

Phase H 在多种分组、指标和筛选方案中识别出 26:q1（冲积土内最低 TWI 五分位）作为值得后续复核的 Oracle 候选。以下为配对差值单位结果；不同方案的配对总体、样本支持和匹配设计并不相同。

| Phase H方案                      | 指标          |   配对数 |   Spearman ρ |   斜率 |   平均Δ健康 |   负向比例 |
|:---------------------------------|:--------------|---------:|-------------:|-------:|------------:|-----------:|
| S0_all                           | SHI_candidate |     1609 |       -0.047 | -0.001 |      -0.034 |      0.806 |
| S0_all                           | OM            |     1609 |       -0.022 | -0.140 |      -3.420 |      0.791 |
| S0_all                           | SW            |     1609 |       -0.212 | -1.476 |      -6.364 |      0.860 |
| S1_both_positive                 | SHI_candidate |      808 |       -0.111 | -0.009 |      -0.036 |      0.826 |
| S1_both_positive                 | OM            |      808 |       -0.184 | -1.351 |      -3.675 |      0.809 |
| S1_both_positive                 | SW            |      808 |       -0.283 | -3.001 |      -5.742 |      0.819 |
| T01_empirical_CDF_breakpoint     | SHI_candidate |      378 |       -0.093 | -0.009 |      -0.032 |      0.801 |
| T01_empirical_CDF_breakpoint     | OM            |      378 |       -0.062 | -0.956 |      -2.908 |      0.746 |
| T01_empirical_CDF_breakpoint     | SW            |      378 |       -0.071 | -1.429 |      -4.220 |      0.743 |
| T01_fixed_original               | SHI_candidate |      601 |       -0.166 | -0.016 |      -0.031 |      0.803 |
| T01_fixed_original               | OM            |      601 |       -0.245 | -2.766 |      -2.409 |      0.676 |
| T01_fixed_original               | SW            |      601 |       -0.253 | -4.145 |      -3.381 |      0.676 |
| T01_refit_k6_filtered_population | SHI_candidate |     1224 |       -0.132 | -0.015 |      -0.029 |      0.776 |
| T01_refit_k6_filtered_population | OM            |     1224 |       -0.137 | -1.908 |      -2.275 |      0.699 |
| T01_refit_k6_filtered_population | SW            |     1224 |       -0.138 | -2.742 |      -3.285 |      0.700 |

在 `T01_fixed_original` 下，SHI/OM/SW 的配对 Spearman 分别约 −0.166/−0.245/−0.253，候选组各有 601 对（SHI 完整案例 593 对）；S0 全配对中分别约 −0.047/−0.022/−0.212。按配对中点限制为同 5 km 空间块后，空间 bootstrap 分别保留 540/547/547 对、16 个空间块；SHI、OM、SW 斜率的 95% 区间分别为 [−0.0276, −0.0021]、[−4.119, −1.499]、[−6.198, −1.951]。该增强不是全指标一致，也不是独立验证；不同筛选方案改变配对总体和匹配支持量。

## 图件

下图分别展示全配对 M4 TWI 分组原始斜率与区间、跨分组的交互 q 值，以及 26:q1 候选在不同 Phase H 样本定义中的 Spearman 变化。

- `outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_i3_r2/figures/environment_heterogeneity/I11_m4_twi_group_slopes.png`
- `outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_i3_r2/figures/environment_heterogeneity/I11_m4_twi_group_slopes.svg`
- `outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_i3_r2/figures/environment_heterogeneity/I11_m4_twi_group_slopes.pdf`
- `outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_i3_r2/figures/environment_heterogeneity/I12_m4_interaction_q_heatmap.png`
- `outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_i3_r2/figures/environment_heterogeneity/I12_m4_interaction_q_heatmap.svg`
- `outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_i3_r2/figures/environment_heterogeneity/I12_m4_interaction_q_heatmap.pdf`
- `outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_i3_r2/figures/environment_heterogeneity/I13_m4_soil26_twi_q1_sensitivity.png`
- `outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_i3_r2/figures/environment_heterogeneity/I13_m4_soil26_twi_q1_sensitivity.svg`
- `outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_i3_r2/figures/environment_heterogeneity/I13_m4_soil26_twi_q1_sensitivity.pdf`

## 解释边界

- 所有分组关联均来自健康预测栅格和匹配配对，不解释为独立实测效应或因果效应。
- 分组样本量、空间块数和匹配支持差异明显。局部较强的效应不能仅凭 R²或相关系数判为最优环境。
- `best_supported_environment_group.csv` 只在至少 5 个 5 km 空间块的既有组内估计中按标准化效应做探索性排序；所有条目仍是 Oracle 结果。
- 26:q1 是阈值筛选及多分组搜索后出现的候选，需与全配对结果、其他分组、空间块敏感性共同解读，不能直接称为确定的高风险环境。

## 文件索引

- 全配对组内斜率：`tables/environment_group_slopes_m4.csv`
- 交互检验：`tables/environment_interactions_m4.csv`
- 组环境剖面：`tables/environment_group_profiles_m4.csv`
- 配对模型排行榜：`tables/environment_group_models_m4.csv`
- Phase H 分组响应与支持：`tables/low_value_group_response_m4.csv`、`tables/low_value_group_support_m4.csv`
- Phase H 5 km bootstrap/空间表现：`tables/low_value_group_spatial_bootstrap_m4.csv`、`tables/low_value_group_spatial_summary_m4.csv`
- 分组定义：`tables/environment_group_definitions.json`
- 最负向、且至少 5 个空间块的分组摘要：`leaderboards/best_supported_environment_group.csv`
- 来源与筛选记录：`manifests/provenance.json`
