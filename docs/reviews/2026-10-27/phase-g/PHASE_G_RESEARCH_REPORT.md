# Phase G 探索性研究报告：环境匹配后的侵蚀—土壤健康关联与环境异质性

> **分析模式：** `EXPLORATORY_ORACLE`，`outcome_guided_selection=true`。本报告回答的是现有 30 m 健康预测栅格中的条件关联与空间异质性，不是正式因果估计，也不是独立实测验证。

## 1. 研究问题与本轮结论

本轮推进研究总纲 Q1–Q3：环境匹配后侵蚀与健康指标是否相关、不同环境背景下关联是否不同、九项指标是否呈现不同响应。主要结论来自预计算的实际侵蚀 `Aa_20yr`，`Aa_2020` 为时间对照。两期与九项结局均单独记录。

Phase F 的 M4 直接最近邻配对中，Aa_20yr 的 Δ健康方向如下表；Δ定义为高侵蚀成员减低侵蚀成员，侵蚀差使用 `log1p(Aa_high)-log1p(Aa_low)`。Spearman 和 Kendall 反映单调关系，Wilcoxon 检验仅作为描述性配对检验；区间和显著性不能消除像元空间依赖。

| 指标 | 配对数 | 平均Δ健康 | 中位Δ健康 | Δ健康<0比例 | Spearman ρ | ρ CI下限 | ρ CI上限 | Kendall τ | Wilcoxon P |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SHI_candidate | 89811 | -0.033 | -0.031 | 0.767 | -0.128 | -0.452 | 0.225 | -0.086 | <0.001 |
| AN | 90848 | 0.121 | -0.134 | 0.511 | -0.002 | -0.345 | 0.341 | -0.002 | 0.007 |
| OM | 90867 | -2.854 | -3.197 | 0.741 | -0.114 | -0.440 | 0.239 | -0.077 | <0.001 |
| CEC | 90867 | 1.004 | 0.939 | 0.340 | 0.060 | -0.289 | 0.395 | 0.040 | <0.001 |
| pH | 90848 | -0.015 | -0.009 | 0.592 | -0.036 | -0.375 | 0.311 | -0.024 | <0.001 |
| SW | 90867 | -6.062 | -7.164 | 0.857 | -0.192 | -0.502 | 0.162 | -0.129 | <0.001 |
| BD | 90867 | -0.046 | -0.042 | 0.646 | -0.031 | -0.371 | 0.315 | -0.021 | <0.001 |
| MBC | 90867 | -2.249 | -2.893 | 0.597 | -0.003 | -0.346 | 0.341 | -0.003 | <0.001 |
| RESP | 90867 | -90.858 | -103.531 | 0.660 | -0.054 | -0.390 | 0.295 | -0.037 | <0.001 |

2020 年 Aa 的对应结果为：

| 指标 | 配对数 | 平均Δ健康 | 中位Δ健康 | Δ健康<0比例 | Spearman ρ | ρ CI下限 | ρ CI上限 | Kendall τ | Wilcoxon P |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SHI_candidate | 107023 | -0.009 | -0.008 | 0.581 | 0.064 | -0.286 | 0.398 | 0.041 | <0.001 |
| AN | 108282 | 0.440 | 0.055 | 0.495 | 0.029 | -0.318 | 0.368 | 0.019 | <0.001 |
| OM | 108284 | -0.912 | -0.519 | 0.583 | 0.045 | -0.303 | 0.383 | 0.029 | <0.001 |
| CEC | 108284 | 0.545 | 0.388 | 0.402 | -0.047 | -0.385 | 0.301 | -0.031 | <0.001 |
| pH | 108282 | -0.002 | -0.001 | 0.516 | 0.040 | -0.307 | 0.378 | 0.027 | <0.001 |
| SW | 108284 | -1.813 | -0.887 | 0.620 | 0.065 | -0.285 | 0.400 | 0.042 | <0.001 |
| BD | 108284 | -0.041 | -0.030 | 0.602 | 0.019 | -0.327 | 0.360 | 0.012 | <0.001 |
| MBC | 108284 | -2.845 | -2.027 | 0.575 | 0.009 | -0.335 | 0.351 | 0.006 | <0.001 |
| RESP | 108284 | -45.261 | -36.054 | 0.582 | 0.031 | -0.316 | 0.370 | 0.020 | <0.001 |

九项指标方向并不一致；应以逐项结果为准，不能概括成“侵蚀普遍降低土壤健康”。要判断侵蚀差值是否与健康差值同步变化，优先看 Spearman/Kendall、标准化斜率及其区间，而不能只看高低成员均值差或 P 值。

## 2. 项目定位与研究治理

- 当前 Git 基线：`f71aaf455e5ca9eff36dfedc0461a09125ff3d34`；总纲为 `docs/RESEARCH_CHARTER.md` V1.1。分析脚本、配置和本报告在生成时尚未提交。
- 历史模型筛选来源：Oracle run `oracle_20261026T041521459615Z_20261026`，其来源代码提交为 `c50bd9ae6a42abc14f898e8043a79cb0c9c818e4`。Phase G 将其记录作为候选搜索证据并重新排名；不会把该历史提交冒充为本轮代码版本。
- 本轮只写入 `outputs/exploratory_oracle/phase_g_20261027/` 和 `docs/reviews/2026-10-27/phase-g/`，不修改总纲、不重算/验证 RUSLE 因子乘积、不重跑 PCNM。
- 本阶段是探索性研究输出，不表示正式 G5 或设计锁定完成。依据 `CHATGPT_REVIEW_PROTOCOL.md`，空间测试不得用于参数或模型选择；已有候选按验证集分数选定后再读取其测试分数。

## 3. 数据、空间支持与限制

- 输入 run：`run_20261025T114741_470772Z`；环境表和健康表各含 392,527 行，以 `cell_id` 一对一连接。加载时核查唯一 ID、完整 ID 集合和行列/投影坐标一致；没有应用全层共同有效掩膜。
- Aa_20yr 有效值范围 0–143.005，零值 133,220；Aa_2020 范围 0–20.029，零值 5,491。实际侵蚀的物理单位和精确时间构造仍按既有登记标记为未核实；本轮不推断。
- 主分析土类为 26、27、42。`SHI_candidate` 与八项指标均为随机森林栅格预测结果；这些像元数不是独立实测样本数。
- Phase F 配对输入共 407,372 对：M3 为土类内 V5（Elevation、LST）K-means k=6，M4 为直接最近邻；两者使用 0.35 环境卡尺、1,000 m 最大地理距离、组内侵蚀百分位秩差 0.50、无重复使用。各暴露/方法的配对数见 `tables/pair_input_profile.json`。

## 4. 方法与选择过程

### 4.1 配对差值与关联

高低成员按既有匹配结果确定，使用独立配对单元。全样本及预先定义的 `low_zero`、`both_positive`、`high_erosion_gap_q4` 子集分别报告；差值回归以 Δlog1p(Aa) 解释 Δ健康。平均差和斜率的 95% 区间使用 5 km 配对中点空间块 bootstrap；Spearman 区间使用 Fisher-z 变换并以空间块数为有效样本量近似。Wilcoxon、Spearman/Kendall 的原始 P 值不视为独立像元推断。

### 4.2 模型搜索与模型区分

Phase G 复用上游 Oracle 的 10,616 条实验记录：100 个 Aa_20yr 与 100 个 Aa_2020 候选匹配初筛，后续匹配/模型/环境子集搜索以及已存空间分块结果。源搜索覆盖 OLS、对数/二次/分段/自然样条/GAM 类回归、Huber、中位数回归、随机森林、Extra Trees 和 HistGradientBoosting 等已安装模型。所有 model leaderboard 均分别标记像元级预测模型与配对差值模型；二者 R² 不混排。

对每项结局分别保留最高表观 R²、负向模型中最高 R²，以及按标准化效应、负 Spearman、Δ健康负值比例、验证表现、匹配平衡与样本量形成的最强负向配对关联。对非线性模型无法从单一系数确定方向时，以存档的全样本偏依赖曲线在共同 Aa 范围内的变化补充方向；仍不可判定的结果保留为 `unknown`。

### 4.3 环境分组与空间验证

分组包括土类、Phase F V5 土类内 K-means k=6、土类内 Ward 层次聚类 k=6，以及高程、TWI、LST、NDWI、NDVI、降水的土类内分位组。层次聚类仅用环境变量形成分组，每个土类最多抽取 1,500 行拟合后按最近标准化中心赋给其他行。组别关联使用高侵蚀成员定义的环境组，并提供组样本数、环境统计量和所有候选结果。

组内空间候选以验证空间块选择，之后仅对选定组报告测试块。另沿用 Phase F 十个 5 km 空间种子的 M3/M4 测试结果。由于先前 Oracle 已反复使用全研究区数据，这些留出结果是内部空间诊断，不是未触碰的外部验证。

## 5. 模型排名与最佳候选

下表把全体像元模型的表观上限、匹配样本表观模型、负向匹配模型、验证集选定后的测试模型以及配对差值最强负向候选分开。模型表观 R²可能主要来自健康栅格本身的环境空间结构，不能解释为 Aa 的独立解释贡献。

| 指标 | 排名 | 候选 | 模型 | R² | 方向 | 样本数 | 分组法 | 组别 | 配对Spearman | 标准化效应 | 验证R² | 十种子测试R²中位数 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AN | highest_apparent_r2 | stage2_Aa_20yr_AN_R030 | random_forest | 0.782 | negative | 98928.000 | — | — | — | — | — | — |
| AN | highest_full_sample_apparent_r2 | full_Aa_20yr_AN_R030 | random_forest | 0.766 | negative | 387515.000 | — | — | — | — | — | — |
| AN | highest_negative_apparent_r2 | stage2_Aa_20yr_AN_R030 | random_forest | 0.782 | negative | 98928.000 | — | — | — | — | — | — |
| AN | strongest_negative_association | subgroup_Aa_20yr_AN_environment_group_triple_12_8_2 | ols | 0.091 | negative | 118955.000 | — | — | — | — | — | — |
| AN | strongest_negative_association | phase_f_M3_Aa_20yr | — | 0.000 | negative | — | — | — | -0.008 | -0.009 | -0.003 | -0.019 |
| AN | validation_selected_spatial_test | spatial_Aa_20yr_AN_R030 | environment_group_fixed_effects | 0.436 | negative | 69869.000 | — | — | — | — | 0.183 | — |
| BD | highest_apparent_r2 | stage2_Aa_20yr_BD_R019 | random_forest | 0.308 | unknown | 75176.000 | — | — | — | — | — | — |
| BD | highest_full_sample_apparent_r2 | full_Aa_20yr_BD_V7 | ols | 0.326 | negative | 294841.000 | — | — | — | — | — | — |
| BD | highest_negative_apparent_r2 | subgroup_Aa_20yr_BD_soil_42_environment_group_pair_22_18 | ols | 0.148 | negative | 12652.000 | — | — | — | — | — | — |
| BD | strongest_negative_association | subgroup_Aa_20yr_BD_exposure_positive | ols | 0.115 | negative | 195989.000 | — | — | — | — | — | — |
| BD | strongest_negative_association | phase_f_M4_Aa_20yr | — | 0.000 | negative | — | — | — | -0.031 | -0.008 | -0.002 | -0.001 |
| BD | validation_selected_spatial_test | spatial_Aa_20yr_BD_V7 | ols | 0.332 | negative | 51816.000 | — | — | — | — | 0.317 | — |
| CEC | highest_apparent_r2 | stage2_Aa_20yr_CEC_R019 | random_forest | 0.267 | unknown | 75176.000 | — | — | — | — | — | — |
| CEC | highest_full_sample_apparent_r2 | full_Aa_20yr_CEC_V8 | ols | 0.378 | positive | 294841.000 | — | — | — | — | — | — |
| CEC | highest_negative_apparent_r2 | stage2_Aa_20yr_CEC_R019 | piecewise | 0.069 | negative | 75176.000 | — | — | — | — | — | — |
| CEC | strongest_negative_association | stage2_Aa_20yr_CEC_R019 | piecewise | 0.069 | negative | 75176.000 | — | — | — | — | — | — |
| CEC | validation_selected_spatial_test | spatial_Aa_20yr_CEC_V8 | ols | 0.361 | positive | 51816.000 | — | — | — | — | 0.468 | — |
| MBC | highest_apparent_r2 | stage2_Aa_20yr_MBC_R019 | random_forest | 0.346 | unknown | 75176.000 | — | — | — | — | — | — |
| MBC | highest_full_sample_apparent_r2 | full_Aa_20yr_MBC_V7 | ols | 0.341 | negative | 294841.000 | — | — | — | — | — | — |
| MBC | highest_negative_apparent_r2 | subgroup_Aa_20yr_MBC_environment_group_2 | ols | 0.149 | negative | 9567.000 | — | — | — | — | — | — |
| MBC | strongest_negative_association | subgroup_Aa_20yr_MBC_exposure_positive | ols | 0.132 | negative | 195989.000 | — | — | — | — | — | — |
| MBC | validation_selected_spatial_test | spatial_Aa_20yr_MBC_V7 | ols | 0.355 | negative | 51816.000 | — | — | — | — | 0.314 | — |
| OM | highest_apparent_r2 | stage2_Aa_20yr_OM_R019 | random_forest | 0.442 | negative | 75176.000 | — | — | — | — | — | — |
| OM | highest_full_sample_apparent_r2 | full_Aa_20yr_OM_R019 | random_forest | 0.412 | negative | 294841.000 | — | — | — | — | — | — |
| OM | highest_negative_apparent_r2 | stage2_Aa_20yr_OM_R019 | random_forest | 0.442 | negative | 75176.000 | — | — | — | — | — | — |
| OM | strongest_negative_association | stage2_Aa_20yr_OM_R019 | log_exposure | 0.207 | negative | 75176.000 | — | — | — | — | — | — |
| OM | strongest_negative_association | phase_f_M4_Aa_20yr | — | 0.006 | negative | — | — | — | -0.114 | -0.077 | 0.003 | -0.003 |
| OM | validation_selected_spatial_test | spatial_Aa_20yr_OM_R019 | extra_trees | 0.339 | unknown | 51816.000 | — | — | — | — | 0.227 | — |
| RESP | highest_apparent_r2 | stage2_Aa_20yr_RESP_R019 | random_forest | 0.418 | negative | 75176.000 | — | — | — | — | — | — |
| RESP | highest_full_sample_apparent_r2 | full_Aa_20yr_RESP_R019 | random_forest | 0.391 | negative | 294841.000 | — | — | — | — | — | — |
| RESP | highest_negative_apparent_r2 | stage2_Aa_20yr_RESP_R019 | random_forest | 0.418 | negative | 75176.000 | — | — | — | — | — | — |
| RESP | strongest_negative_association | subgroup_Aa_20yr_RESP_exposure_positive | ols | 0.207 | negative | 195989.000 | — | — | — | — | — | — |
| RESP | validation_selected_spatial_test | spatial_Aa_20yr_RESP_V7 | ols | 0.421 | negative | 51816.000 | — | — | — | — | 0.297 | — |
| SHI_candidate | highest_apparent_r2 | stage2_Aa_20yr_SHI_candidate_V5 | random_forest | 0.628 | negative | 98342.000 | — | — | — | — | — | — |
| SHI_candidate | highest_full_sample_apparent_r2 | full_Aa_20yr_SHI_candidate_V5 | random_forest | 0.637 | negative | 384735.000 | — | — | — | — | — | — |
| SHI_candidate | highest_negative_apparent_r2 | stage2_Aa_20yr_SHI_candidate_V5 | random_forest | 0.628 | negative | 98342.000 | — | — | — | — | — | — |
| SHI_candidate | strongest_negative_association | subgroup_Aa_20yr_SHI_candidate_Elevation_q1 | ols | 0.351 | negative | 109716.000 | — | — | — | — | — | — |
| SHI_candidate | strongest_negative_association | phase_f_M3_Aa_20yr | — | 0.016 | negative | — | — | — | -0.124 | -0.126 | 0.010 | 0.003 |
| SHI_candidate | validation_selected_spatial_test | spatial_Aa_20yr_SHI_candidate_V7 | ols | 0.470 | negative | 51533.000 | — | — | — | — | -0.069 | — |
| SW | highest_apparent_r2 | stage2_Aa_20yr_SW_V5 | random_forest | 0.621 | negative | 98933.000 | — | — | — | — | — | — |
| SW | highest_full_sample_apparent_r2 | full_Aa_20yr_SW_V5 | random_forest | 0.619 | negative | 387538.000 | — | — | — | — | — | — |
| SW | highest_negative_apparent_r2 | stage2_Aa_20yr_SW_V5 | random_forest | 0.621 | negative | 98933.000 | — | — | — | — | — | — |
| SW | strongest_negative_association | subgroup_Aa_20yr_SW_Elevation_q1 | ols | 0.383 | negative | 110244.000 | — | — | — | — | — | — |
| SW | strongest_negative_association | phase_f_M3_Aa_20yr | — | 0.044 | negative | — | — | — | -0.196 | -0.210 | 0.035 | 0.062 |
| SW | validation_selected_spatial_test | spatial_Aa_20yr_SW_V5 | extra_trees | 0.570 | unknown | 69861.000 | — | — | — | — | 0.447 | — |

最佳像元模型的完整参数、筛选来源和样本量见 `tables/model_leaderboard.csv` 与 `tables/model_search_results.csv`。最强负向配对关联的复合排名会同时显示系数、Spearman、Δ健康负值比例、bootstrap 上界、验证 R²、SMD 和样本量；其 composite score 只是 Oracle 排序指标，不能被解释为正式效应量。

## 6. 环境组异质性

负向关联组的完整清单、关联区间与组内环境特征分别见 `tables/environment_group_associations.csv`、`tables/environment_group_statistics.csv` 和 `tables/environment_group_leaderboard.csv`。下表为 M4/Aa_20yr 下复合负向排名靠前的组：

| 指标 | 分组法 | 组别 | 配对数 | 标准化斜率 | Spearman ρ | Δ健康<0比例 | 探索性综合分 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CEC | gradient_LST | 26:q2 | 3934.000 | -0.018 | -0.007 | 0.323 | 1.000 |
| pH | gradient_LST | 26:q1 | 3972.000 | -0.144 | -0.124 | 0.693 | 1.000 |
| AN | kmeans_v5 | 2704 | 4544.000 | -0.075 | -0.072 | 0.631 | 1.000 |
| AN | soil_type | soil_27 | 58503.000 | -0.009 | -0.007 | 0.531 | 1.000 |
| CEC | gradient_NDWI | 26:q1 | 3463.000 | -0.030 | -0.008 | 0.347 | 1.000 |
| pH | hierarchical_ward_v5 | 26:4 | 1099.000 | -0.240 | -0.165 | 0.667 | 0.979 |
| OM | gradient_NDVI | 26:q1 | 4659.000 | -0.285 | -0.293 | 0.796 | 0.964 |
| pH | gradient_TWI | 27:q1 | 12435.000 | -0.090 | -0.085 | 0.626 | 0.962 |
| pH | gradient_Elevation | 42:q5 | 2506.000 | -0.106 | -0.119 | 0.661 | 0.962 |
| OM | gradient_Precipitation | 27:q5 | 9617.000 | -0.184 | -0.196 | 0.794 | 0.950 |
| MBC | gradient_Elevation | 27:q1 | 10650.000 | -0.096 | -0.104 | 0.597 | 0.950 |
| AN | hierarchical_ward_v5 | 27:3 | 3198.000 | -0.079 | -0.084 | 0.689 | 0.950 |

跨匹配方法与两种 Aa 时段的方向稳定组另列于 `most_stable_negative_direction_across_matching_methods_and_Aa_periods`。分组结果属于结果导向筛选，且单变量分位组可能残留其他环境差异；没有把某组标签解释为治理因果机制。

## 7. 空间验证

验证选定模型的空间测试结果如下，选择仅由验证集完成：

| 指标 | 模型 | 验证R² | 空间测试R² | 测试RMSE | 测试MAE | 方向 | 测试像元数 | 环境向量 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AN | environment_group_fixed_effects | 0.183 | 0.436 | 12.746 | 9.981 | negative | 69869.000 | R030 |
| BD | ols | 0.317 | 0.332 | 0.109 | 0.092 | negative | 51816.000 | V7 |
| CEC | ols | 0.468 | 0.361 | 1.445 | 1.157 | positive | 51816.000 | V8 |
| MBC | ols | 0.314 | 0.355 | 10.178 | 8.269 | negative | 51816.000 | V7 |
| OM | extra_trees | 0.227 | 0.339 | 2.592 | 1.947 | unknown | 51816.000 | R019 |
| RESP | ols | 0.297 | 0.421 | 144.541 | 115.868 | negative | 51816.000 | V7 |
| SHI_candidate | ols | -0.069 | 0.470 | 0.038 | 0.030 | negative | 51533.000 | V7 |
| SW | extra_trees | 0.447 | 0.570 | 3.513 | 2.775 | unknown | 69861.000 | V5 |
| pH | ols | 0.021 | -0.023 | 0.084 | 0.069 | negative | 51817.000 | V4 |

图 9 还展示了 Phase F 十个空间种子的配对 OLS 测试分布；本报告保留每一种子记录。预测面的模型测试只能检验空间区块间预测稳定性，不能替代独立采样验证。

## 8. 图表与复核路径

- **01_matching_quality**: [PNG](figures/01_matching_quality.png) · [SVG](figures/01_matching_quality.svg) · [PDF](figures/01_matching_quality.pdf) · [数据](figures/data/01_matching_quality.csv) · [图注](figures/captions/01_matching_quality.md)
- **02_paired_health_differences**: [PNG](figures/02_paired_health_differences.png) · [SVG](figures/02_paired_health_differences.svg) · [PDF](figures/02_paired_health_differences.pdf) · [数据](figures/data/02_paired_health_differences.csv) · [图注](figures/captions/02_paired_health_differences.md)
- **03_pair_association_heatmap**: [PNG](figures/03_pair_association_heatmap.png) · [SVG](figures/03_pair_association_heatmap.svg) · [PDF](figures/03_pair_association_heatmap.pdf) · [数据](figures/data/03_pair_association_heatmap.csv) · [图注](figures/captions/03_pair_association_heatmap.md)
- **04_model_r2_comparison**: [PNG](figures/04_model_r2_comparison.png) · [SVG](figures/04_model_r2_comparison.svg) · [PDF](figures/04_model_r2_comparison.pdf) · [数据](figures/data/04_model_r2_comparison.csv) · [图注](figures/captions/04_model_r2_comparison.md)
- **05_response_curves**: [PNG](figures/05_response_curves.png) · [SVG](figures/05_response_curves.svg) · [PDF](figures/05_response_curves.pdf) · [数据](figures/data/05_response_curves.csv) · [图注](figures/captions/05_response_curves.md)
- **06_environment_group_forest**: [PNG](figures/06_environment_group_forest.png) · [SVG](figures/06_environment_group_forest.svg) · [PDF](figures/06_environment_group_forest.pdf) · [数据](figures/data/06_environment_group_forest.csv) · [图注](figures/captions/06_environment_group_forest.md)
- **07_grouping_method_comparison**: [PNG](figures/07_grouping_method_comparison.png) · [SVG](figures/07_grouping_method_comparison.svg) · [PDF](figures/07_grouping_method_comparison.pdf) · [数据](figures/data/07_grouping_method_comparison.csv) · [图注](figures/captions/07_grouping_method_comparison.md)
- **08_key_group_maps**: [PNG](figures/08_key_group_maps.png) · [SVG](figures/08_key_group_maps.svg) · [PDF](figures/08_key_group_maps.pdf) · [数据](figures/data/08_key_group_maps.csv) · [图注](figures/captions/08_key_group_maps.md)
- **09_spatial_validation**: [PNG](figures/09_spatial_validation.png) · [SVG](figures/09_spatial_validation.svg) · [PDF](figures/09_spatial_validation.pdf) · [数据](figures/data/09_spatial_validation.csv) · [图注](figures/captions/09_spatial_validation.md)

结果表、完整来源实验登记、运行配置、输入画像和产物校验清单位于 `../../../../outputs/exploratory_oracle/phase_g_20261027/`。主要文件包括：

- `tables/experiment_registry.csv`
- `tables/matching_configurations.csv` 与 `tables/matching_balance_results.csv`
- `tables/matched_pair_statistics.csv` 与 `tables/association_results.csv`
- `tables/model_search_results.csv`、`tables/model_leaderboard.csv`、`tables/final_candidate_results.csv`
- `tables/environment_group_statistics.csv`、`tables/environment_group_associations.csv`
- `tables/spatial_validation_results.csv`

## 9. 研究判断

当前结果支持继续研究“环境条件相近时侵蚀差异与土壤健康预测差值的关系”，但不支持把关联概括成全指标一致的负效应。匹配与分组可以揭示异质性候选，但部分高拟合子集仅保留少量像元；跨空间测试表现低于表观上限，且搜索重复使用同一研究区，估计有选择乐观偏差。曲线图同时给出固定共同样本自然样条和先前 R² 选出的全样本模型曲线，两者样本范围与选择机制不同。

Phase G 不更改总纲，也不进入正式设计锁定。下一步应先由 ChatGPT 和用户审查每项指标的方向、单位/时间元信息、空间分块是否充分、预测面依赖及高 R² 子集的保留率，再决定哪些少数候选进入预先锁定的 FORMAL 设计。独立土壤实测点和其抽样/验证方案是正式验证的关键输入；不应因当前表观关联高而省略。

## 10. 执行状态与范围外事项

- Phase G Oracle 代码、CSV、九张图及报告已在本地生成，具体试验计数写入 `tables/experiment_summary.json` 与 `tables/experiment_registry.csv`。
- 本轮未运行原始 RUSLE、未重跑 PCNM、未修改总纲正文。
- 所有健康结局均是预测栅格值；结果不能写成独立实测点结论或因果效应。
- `Aa_20yr` 与 `Aa_2020` 单位及时间期定义缺失仍是报告限制；数据表的物理单位不予推定。
- ChatGPT 审查请求已生成，未向外发送。用户提交首个 Phase G 成果提交后，应把该提交完整 SHA 写入审查请求，再单独提交审查请求。
