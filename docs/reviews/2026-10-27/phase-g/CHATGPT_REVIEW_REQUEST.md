# ChatGPT 阶段科研审查请求：Phase G

请首先读取本次Git提交对应版本的 `docs/RESEARCH_CHARTER.md`，完整理解EnvMatch-SEH的研究主题、总体研究目标、核心科学问题、主要研究内容及长期技术路线，然后结合这些内容审查本阶段的代码、实验、数据和研究结论。

## 1. 项目与版本

- 仓库：`https://github.com/shy-sgsg/EnvMatch-SEH`
- 审查提交 SHA：`dc599055cd56aa297133eac64620647b1f911499`
- 本轮运行时的基线 HEAD：`f71aaf455e5ca9eff36dfedc0461a09125ff3d34`（不包含尚未提交的 Phase G 文件）
- 研究总纲：`docs/RESEARCH_CHARTER.md`，V1.1；对应本地基线提交：`f71aaf455e5ca9eff36dfedc0461a09125ff3d34`
- 阶段：Phase G `EXPLORATORY_ORACLE`；不是 FORMAL G5 完成，也不代表设计锁定
- 研究报告：`docs/reviews/2026-10-27/phase-g/PHASE_G_RESEARCH_REPORT.md`
- 审查材料：`docs/reviews/2026-10-27/phase-g/`
- 输出根目录：`outputs/exploratory_oracle/phase_g_20261027/`

## 2. 总体研究定位与科学问题

本阶段对应 Q1–Q3：在环境匹配后，侵蚀与 SHI/八项健康指标是否有关联；不同环境分组下关系是否异质；物理、化学、生物指标的关联是否不同。PCNM 结果继续保留并暂不重跑。未修改总纲。

本阶段设定 `outcome_guided_selection=true`。本轮结果只用于发现候选及研究方法审查，不作为独立验证、因果结论或正式证据。

## 3. 本阶段工作

- 目标：复用 Phase F 匹配对与既有 Oracle 模型搜索，补齐全部九指标差值/秩相关、像元候选模型排名、环境组比较、空间块诊断和复核图表。
- 输入：`outputs/run_20261025T114741_470772Z/databases/environment_erosion.parquet`、`health.parquet`；Phase F `outputs/exploratory_oracle/phase_f_20261026_seed20261026/pairs/full_sample_pairs.parquet`；历史 Oracle `oracle_20261026T041521459615Z_20261026`。
- 暴露：预计算 `Aa_20yr` 为主、`Aa_2020` 为补充。未重算或验证 RUSLE 因子乘积。
- 配对：M3 土类内 V5 (Elevation/LST) K-means k=6；M4 直接最近邻；0.35 卡尺、1,000 m、组内秩差0.50、无重复使用。
- 环境组：SoilType、K-means、Ward 层次聚类以及 Elevation/TWI/LST/NDWI/NDVI/降水分位组。
- 历史模型筛选代码/记录：Oracle 源提交 `c50bd9ae6a42abc14f898e8043a79cb0c9c818e4`；当前 Phase G 新增脚本/模块由待提交 SHA 覆盖。
- 新增主要代码/配置：`scripts/run_phase_g_analysis.py`、`experiments/oracle_exploration/phase_g.py`、`config/phase_g.yaml`、`tests/test_phase_g.py`。

## 4. 数据与方法质量

- 对齐数据库各 392,527 行；以 `cell_id` 一对一合并，坐标一致性通过；有效土类 26/27/42 共 383,714 行。
- Phase F 配对总数 407,372，具体按 M3/M4 与 Aa 暴露的数量在 `tables/pair_input_profile.json`。
- 历史 Oracle 候选记录数 10,616；模型种类、失败项、样本数、匹配参数和选择规则可在 `tables/experiment_registry.csv` 与 `tables/model_search_results.csv` 查核。
- Aa 单位/精确时间构造与健康预测面的独立实测来源仍未核实；这些限制记录在研究报告和 `tables/data_profile.json`。
- 配对区间按 5 km pair-midpoint blocks bootstrap；相关系数区间使用 block-count-adjusted Fisher-z 近似。检验 P 值需结合空间依赖审查。
- 模型 R²分别标记像元预测、配对差值和空间测试；测试候选由验证分数选出。先前 Oracle 已复用全研究区，留出属于内部空间诊断。

### M4 / Aa_20yr 配对关联方向与效应

| 指标 | 配对数 | 平均Δ健康 | Δ健康<0比例 | Spearman ρ | Kendall τ | 标准化斜率 | 斜率区间下限 | 斜率区间上限 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SHI_candidate | 89811 | -0.033 | 0.767 | -0.128 | -0.086 | -0.123 | -0.014 | -0.008 |
| AN | 90848 | 0.121 | 0.511 | -0.002 | -0.002 | -0.008 | -0.576 | 0.376 |
| OM | 90867 | -2.854 | 0.741 | -0.114 | -0.077 | -0.077 | -0.892 | -0.349 |
| CEC | 90867 | 1.004 | 0.340 | 0.060 | 0.040 | 0.020 | 0.002 | 0.263 |
| pH | 90848 | -0.015 | 0.592 | -0.036 | -0.024 | -0.034 | -0.011 | 0.001 |
| SW | 90867 | -6.062 | 0.857 | -0.192 | -0.129 | -0.204 | -2.348 | -1.725 |
| BD | 90867 | -0.046 | 0.646 | -0.031 | -0.021 | -0.008 | -0.009 | 0.002 |
| MBC | 90867 | -2.249 | 0.597 | -0.003 | -0.003 | 0.059 | 0.482 | 2.648 |
| RESP | 90867 | -90.858 | 0.660 | -0.054 | -0.037 | 0.006 | -20.610 | 14.050 |

## 5. 结果与研究价值

### 模型排名（Aa_20yr）

| 指标 | 排名 | 候选 | 模型 | R² | 方向 | 样本数 | 分组法 | 组别 | 配对Spearman | 验证R² | 测试R²中位数 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AN | highest_apparent_r2 | stage2_Aa_20yr_AN_R030 | random_forest | 0.782 | negative | 98928.000 | — | — | — | — | — |
| AN | highest_full_sample_apparent_r2 | full_Aa_20yr_AN_R030 | random_forest | 0.766 | negative | 387515.000 | — | — | — | — | — |
| AN | highest_negative_apparent_r2 | stage2_Aa_20yr_AN_R030 | random_forest | 0.782 | negative | 98928.000 | — | — | — | — | — |
| AN | strongest_negative_association | subgroup_Aa_20yr_AN_environment_group_triple_12_8_2 | ols | 0.091 | negative | 118955.000 | — | — | — | — | — |
| AN | strongest_negative_association | phase_f_M3_Aa_20yr | — | 0.000 | negative | — | — | — | -0.008 | -0.003 | -0.019 |
| AN | validation_selected_spatial_test | spatial_Aa_20yr_AN_R030 | environment_group_fixed_effects | 0.436 | negative | 69869.000 | — | — | — | 0.183 | — |
| BD | highest_apparent_r2 | stage2_Aa_20yr_BD_R019 | random_forest | 0.308 | unknown | 75176.000 | — | — | — | — | — |
| BD | highest_full_sample_apparent_r2 | full_Aa_20yr_BD_V7 | ols | 0.326 | negative | 294841.000 | — | — | — | — | — |
| BD | highest_negative_apparent_r2 | subgroup_Aa_20yr_BD_soil_42_environment_group_pair_22_18 | ols | 0.148 | negative | 12652.000 | — | — | — | — | — |
| BD | strongest_negative_association | subgroup_Aa_20yr_BD_exposure_positive | ols | 0.115 | negative | 195989.000 | — | — | — | — | — |
| BD | strongest_negative_association | phase_f_M4_Aa_20yr | — | 0.000 | negative | — | — | — | -0.031 | -0.002 | -0.001 |
| BD | validation_selected_spatial_test | spatial_Aa_20yr_BD_V7 | ols | 0.332 | negative | 51816.000 | — | — | — | 0.317 | — |
| CEC | highest_apparent_r2 | stage2_Aa_20yr_CEC_R019 | random_forest | 0.267 | unknown | 75176.000 | — | — | — | — | — |
| CEC | highest_full_sample_apparent_r2 | full_Aa_20yr_CEC_V8 | ols | 0.378 | positive | 294841.000 | — | — | — | — | — |
| CEC | highest_negative_apparent_r2 | stage2_Aa_20yr_CEC_R019 | piecewise | 0.069 | negative | 75176.000 | — | — | — | — | — |
| CEC | strongest_negative_association | stage2_Aa_20yr_CEC_R019 | piecewise | 0.069 | negative | 75176.000 | — | — | — | — | — |
| CEC | validation_selected_spatial_test | spatial_Aa_20yr_CEC_V8 | ols | 0.361 | positive | 51816.000 | — | — | — | 0.468 | — |
| MBC | highest_apparent_r2 | stage2_Aa_20yr_MBC_R019 | random_forest | 0.346 | unknown | 75176.000 | — | — | — | — | — |
| MBC | highest_full_sample_apparent_r2 | full_Aa_20yr_MBC_V7 | ols | 0.341 | negative | 294841.000 | — | — | — | — | — |
| MBC | highest_negative_apparent_r2 | subgroup_Aa_20yr_MBC_environment_group_2 | ols | 0.149 | negative | 9567.000 | — | — | — | — | — |
| MBC | strongest_negative_association | subgroup_Aa_20yr_MBC_exposure_positive | ols | 0.132 | negative | 195989.000 | — | — | — | — | — |
| MBC | validation_selected_spatial_test | spatial_Aa_20yr_MBC_V7 | ols | 0.355 | negative | 51816.000 | — | — | — | 0.314 | — |
| OM | highest_apparent_r2 | stage2_Aa_20yr_OM_R019 | random_forest | 0.442 | negative | 75176.000 | — | — | — | — | — |
| OM | highest_full_sample_apparent_r2 | full_Aa_20yr_OM_R019 | random_forest | 0.412 | negative | 294841.000 | — | — | — | — | — |
| OM | highest_negative_apparent_r2 | stage2_Aa_20yr_OM_R019 | random_forest | 0.442 | negative | 75176.000 | — | — | — | — | — |
| OM | strongest_negative_association | stage2_Aa_20yr_OM_R019 | log_exposure | 0.207 | negative | 75176.000 | — | — | — | — | — |
| OM | strongest_negative_association | phase_f_M4_Aa_20yr | — | 0.006 | negative | — | — | — | -0.114 | 0.003 | -0.003 |
| OM | validation_selected_spatial_test | spatial_Aa_20yr_OM_R019 | extra_trees | 0.339 | unknown | 51816.000 | — | — | — | 0.227 | — |
| RESP | highest_apparent_r2 | stage2_Aa_20yr_RESP_R019 | random_forest | 0.418 | negative | 75176.000 | — | — | — | — | — |
| RESP | highest_full_sample_apparent_r2 | full_Aa_20yr_RESP_R019 | random_forest | 0.391 | negative | 294841.000 | — | — | — | — | — |
| RESP | highest_negative_apparent_r2 | stage2_Aa_20yr_RESP_R019 | random_forest | 0.418 | negative | 75176.000 | — | — | — | — | — |
| RESP | strongest_negative_association | subgroup_Aa_20yr_RESP_exposure_positive | ols | 0.207 | negative | 195989.000 | — | — | — | — | — |
| RESP | validation_selected_spatial_test | spatial_Aa_20yr_RESP_V7 | ols | 0.421 | negative | 51816.000 | — | — | — | 0.297 | — |
| SHI_candidate | highest_apparent_r2 | stage2_Aa_20yr_SHI_candidate_V5 | random_forest | 0.628 | negative | 98342.000 | — | — | — | — | — |
| SHI_candidate | highest_full_sample_apparent_r2 | full_Aa_20yr_SHI_candidate_V5 | random_forest | 0.637 | negative | 384735.000 | — | — | — | — | — |
| SHI_candidate | highest_negative_apparent_r2 | stage2_Aa_20yr_SHI_candidate_V5 | random_forest | 0.628 | negative | 98342.000 | — | — | — | — | — |
| SHI_candidate | strongest_negative_association | subgroup_Aa_20yr_SHI_candidate_Elevation_q1 | ols | 0.351 | negative | 109716.000 | — | — | — | — | — |
| SHI_candidate | strongest_negative_association | phase_f_M3_Aa_20yr | — | 0.016 | negative | — | — | — | -0.124 | 0.010 | 0.003 |
| SHI_candidate | validation_selected_spatial_test | spatial_Aa_20yr_SHI_candidate_V7 | ols | 0.470 | negative | 51533.000 | — | — | — | -0.069 | — |
| SW | highest_apparent_r2 | stage2_Aa_20yr_SW_V5 | random_forest | 0.621 | negative | 98933.000 | — | — | — | — | — |
| SW | highest_full_sample_apparent_r2 | full_Aa_20yr_SW_V5 | random_forest | 0.619 | negative | 387538.000 | — | — | — | — | — |
| SW | highest_negative_apparent_r2 | stage2_Aa_20yr_SW_V5 | random_forest | 0.621 | negative | 98933.000 | — | — | — | — | — |
| SW | strongest_negative_association | subgroup_Aa_20yr_SW_Elevation_q1 | ols | 0.383 | negative | 110244.000 | — | — | — | — | — |
| SW | strongest_negative_association | phase_f_M3_Aa_20yr | — | 0.044 | negative | — | — | — | -0.196 | 0.035 | 0.062 |
| SW | validation_selected_spatial_test | spatial_Aa_20yr_SW_V5 | extra_trees | 0.570 | unknown | 69861.000 | — | — | — | 0.447 | — |
| pH | highest_apparent_r2 | subgroup_Aa_20yr_pH_soil_85 | ols | 0.875 | positive | 6586.000 | — | — | — | — | — |
| pH | highest_full_sample_apparent_r2 | full_Aa_20yr_pH_R012 | random_forest | 0.720 | negative | 294846.000 | — | — | — | — | — |
| pH | highest_negative_apparent_r2 | stage2_Aa_20yr_pH_R012 | random_forest | 0.706 | negative | 75177.000 | — | — | — | — | — |
| pH | strongest_negative_association | stage2_Aa_20yr_pH_R012 | log_exposure | 0.049 | negative | 75177.000 | — | — | — | — | — |
| pH | strongest_negative_association | phase_f_M4_Aa_20yr | — | 0.001 | negative | — | — | — | -0.036 | -0.003 | -0.004 |
| pH | validation_selected_spatial_test | spatial_Aa_20yr_pH_V4 | ols | -0.023 | negative | 51817.000 | — | — | — | 0.021 | — |

### 环境组

环境组所有候选及其环境特征见 `tables/environment_group_statistics.csv`；九指标关联见 `tables/environment_group_associations.csv`；按关联强度、负向效应、跨匹配/时段方向一致性分别排名的结果见 `tables/environment_group_leaderboard.csv`。空间验证入选与测试结果见 `tables/group_spatial_validation.csv`。

## 6. 空间验证与风险

空间验证结果表：`tables/spatial_validation_results.csv`。空间图 09 及源数据：`figures/09_spatial_validation.*` 与 `figures/data/09_spatial_validation.csv`。请特别审查先前结果选择已重复访问全域数据之后，是否仍可称为内部留出而非独立验证。

高表观 R²模型可能依赖健康预测栅格共享的环境协变量；匹配阈值未消除所有环境混杂；部分 Oracle 子组样本保留比例很低；模型结果的 direction 对非线性预测曲线定义而非单一系数。高拟合度不代表侵蚀独立贡献或因果效应。

## 7. 关键图表、实验表与代码

九张图均提供 PNG/SVG/PDF，并带源数据和图注：

- 01_matching_quality: `figures/01_matching_quality.png`, `.svg`, `.pdf`; `figures/data/01_matching_quality.csv`
- 02_paired_health_differences: `figures/02_paired_health_differences.png`, `.svg`, `.pdf`; `figures/data/02_paired_health_differences.csv`
- 03_pair_association_heatmap: `figures/03_pair_association_heatmap.png`, `.svg`, `.pdf`; `figures/data/03_pair_association_heatmap.csv`
- 04_model_r2_comparison: `figures/04_model_r2_comparison.png`, `.svg`, `.pdf`; `figures/data/04_model_r2_comparison.csv`
- 05_response_curves: `figures/05_response_curves.png`, `.svg`, `.pdf`; `figures/data/05_response_curves.csv`
- 06_environment_group_forest: `figures/06_environment_group_forest.png`, `.svg`, `.pdf`; `figures/data/06_environment_group_forest.csv`
- 07_grouping_method_comparison: `figures/07_grouping_method_comparison.png`, `.svg`, `.pdf`; `figures/data/07_grouping_method_comparison.csv`
- 08_key_group_maps: `figures/08_key_group_maps.png`, `.svg`, `.pdf`; `figures/data/08_key_group_maps.csv`
- 09_spatial_validation: `figures/09_spatial_validation.png`, `.svg`, `.pdf`; `figures/data/09_spatial_validation.csv`

实验和结果表：

- `tables/experiment_registry.csv`
- `tables/matching_configurations.csv`
- `tables/matching_balance_results.csv`
- `tables/matched_pair_statistics.csv`
- `tables/association_results.csv`
- `tables/model_search_results.csv`
- `tables/model_leaderboard.csv`
- `tables/environment_group_statistics.csv`
- `tables/environment_group_associations.csv`
- `tables/spatial_validation_results.csv`
- `tables/final_candidate_results.csv`
- `tables/experiment_summary.json`、`tables/data_profile.json`、`artifact_manifest.json`
- 核心代码：`scripts/run_phase_g_analysis.py`、`experiments/oracle_exploration/phase_g.py`、`config/phase_g.yaml`

## 8. 对整体路线的影响与建议

建议先审查关联方向、置信区间方法、环境组定义、配对成员空间块分配、模型搜索选择偏差及高 R² 的环境解释。Phase G 不替代既有 PCNM 控制，也不建议仅依据表观最佳结果锁定正式方案。请提出下一阶段可预注册、可由独立实测数据验证的最小方案。

## 9. 总纲审查

`NO CHANGE`。本阶段未修改总纲正文。

审查材料生成后未发送给外部。请在固定到实际代码/结果提交 SHA 后审查上述路径。

<!-- PHASE_G_COMPLETION_ADDENDUM_START -->
## 补充审查材料：Phase G completion

- 补充报告：`docs/reviews/2026-10-27/phase-g/PHASE_G_COMPLETION_ADDENDUM.md`
- 范围校正后的像元排名：`docs/reviews/2026-10-27/phase-g/tables/completion/in_scope_pixel_model_leaderboard.csv`
- 方向重建及源 R² 复核：`docs/reviews/2026-10-27/phase-g/tables/completion/pixel_best_candidate_direction_reconstruction.csv`
- 配对响应模型和空间留出：`docs/reviews/2026-10-27/phase-g/tables/completion/paired_response_model_trials.csv`、`paired_spatial_model_trials.csv`、`paired_spatial_selected_models.csv`
- 环境组斜率与交互检验：`docs/reviews/2026-10-27/phase-g/tables/completion/group_slope_estimates.csv`、`group_slope_interaction_tests.csv`、`environment_group_profiles.csv`
- 新增图：`docs/reviews/2026-10-27/phase-g/figures/completion/10_environment_group_slope_forest.*`
- 独立运行清单：`outputs/exploratory_oracle/phase_g_20261027_completion/artifact_manifest.json`（运行目录由 Git 忽略；审查用 CSV、报告和图副本已纳入本审查目录）
- 本次补充运行起始 HEAD：`efdcdcd313f57aae1587466f9e0115a2846bec5e`。完整审查提交 SHA 保持待填；请在用户本地提交完成后将首个包含代码、配置和报告/表格的完整 SHA 填入第 1 节。
- 校正说明：旧 leaderboard 中 18 条 SoilType 85（水体）候选行不在主要耕地范围；补充榜单已排除它们，并覆盖旧报告中受影响的最高表观排名。
- 环境组交互：全局 BH 覆盖 504 项，其中 q<0.05 有 473 项；M4/Aa_20yr 为 123/126。极端 Wald 尾概率使用稳定 log(P)/log(q) 表示。
- 空间边界：52,601/407,372 个配对的两端跨越 5 km 网格（12.9%）；验证仍按配对中点分块。
- 曲线要求：图 5 同时包含固定样本自然样条和全样本 Oracle 表观最佳模型曲线；请分别审查其样本和选择边界。
