# Phase I 图件闭合与标准 MGWR 补充

phase: I-closure
mode: EXPLORATORY_ORACLE
outcome_guided_selection: true

本运行只整理既有 Phase I/Phase H 表格与数据库、补制缺失核心图件，并准备标准 MGWR 的空间均衡样本。不会覆盖 I-1 至 I-9 既有运行，不重建/验证 RUSLE 因子乘积，不重跑 PCNM。

## 环境异质性补图

配对响应曲线沿用既有 Aa_20yr M4 候选 `outputs/exploratory_oracle/phase_h_20261028_seed20261028_h2/pairs/candidate_pairs/stage2_Aa_20yr_01.parquet`，只按端点土类、以及土类内环境五分位分层后汇总八分位配对均值。曲线是描述性汇总，不是新增模型拟合。环境分组使用对齐数据库 `outputs/run_20261025T114741_470772Z/databases/environment_erosion.parquet` 中的 SoilType、TWI、NDVI。

核心图 12 的可读源表只提供 `r_squared_apparent` 字段；没有每个环境组的可审计最佳模型族搜索/筛选记录。因此图中按表字段诚实标为组内 apparent R²，不称为环境组最佳模型R²，也不作空间外推解释。

## GWR 与标准 MGWR 图件

八指标 GWR 组合图复用 I-4 环境调整全样本 GWR 局部标准化侵蚀系数，所有面板共用以零为中心的色标。标准 MGWR 使用与标准 GWR 相同的空间均衡 240m 汇总单元样本、相同的 Aa_20yr 和九项环境协变量、adaptive Gaussian kernel；每个抽样网格单元等权。全样本 I-4 是像元支持加权的 240m 单元，与本补充的空间均衡样本估计对象不同。

标准 MGWR 只拟合 SHI、OM、SW 三个关键结局。样本规模、算法收敛、拟合指标和变量邻域尺度应查看 `tables/` 与最终阶段报告；不将 MGWR 局部系数解释为因果效应。

## PCNM/PLS-SEM 源数据缺口

PCNM 源结果表缺失，PLS-SEM 路径/载荷/控制项表也未提供。核心图 30、31 的文件是标明缺口的状态图，不是对照森林图或机制图；I-7交叉字段仍保持空值，不补造结果。收到实际源表后才可完成科学对照。

## 已生成图件

- `outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_closure_r1/figures/core_figures/C04_global_best_model_r2.png`
- `outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_closure_r1/figures/core_figures/C04_global_best_model_r2.svg`
- `outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_closure_r1/figures/core_figures/C04_global_best_model_r2.pdf`
- `outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_closure_r1/figures/core_figures/C05_MBC_segmented_threshold_candidate.png`
- `outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_closure_r1/figures/core_figures/C05_MBC_segmented_threshold_candidate.svg`
- `outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_closure_r1/figures/core_figures/C05_MBC_segmented_threshold_candidate.pdf`
- `outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_closure_r1/figures/core_figures/C07_soil_type_matched_response_curves.png`
- `outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_closure_r1/figures/core_figures/C07_soil_type_matched_response_curves.svg`
- `outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_closure_r1/figures/core_figures/C07_soil_type_matched_response_curves.pdf`
- `outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_closure_r1/figures/core_figures/C08_twi_group_matched_response_curves.png`
- `outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_closure_r1/figures/core_figures/C08_twi_group_matched_response_curves.svg`
- `outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_closure_r1/figures/core_figures/C08_twi_group_matched_response_curves.pdf`
- `outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_closure_r1/figures/core_figures/C09_twi_interaction_slopes.png`
- `outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_closure_r1/figures/core_figures/C09_twi_interaction_slopes.svg`
- `outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_closure_r1/figures/core_figures/C09_twi_interaction_slopes.pdf`
- `outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_closure_r1/figures/core_figures/C10_ndvi_interaction_slopes.png`
- `outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_closure_r1/figures/core_figures/C10_ndvi_interaction_slopes.svg`
- `outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_closure_r1/figures/core_figures/C10_ndvi_interaction_slopes.pdf`
- `outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_closure_r1/figures/core_figures/C11_environment_group_slope_forest.png`
- `outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_closure_r1/figures/core_figures/C11_environment_group_slope_forest.svg`
- `outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_closure_r1/figures/core_figures/C11_environment_group_slope_forest.pdf`
- `outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_closure_r1/figures/core_figures/C12_environment_group_apparent_r2.png`
- `outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_closure_r1/figures/core_figures/C12_environment_group_apparent_r2.svg`
- `outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_closure_r1/figures/core_figures/C12_environment_group_apparent_r2.pdf`
- `outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_closure_r1/figures/core_figures/C16_eight_indicator_gwr_composite.png`
- `outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_closure_r1/figures/core_figures/C16_eight_indicator_gwr_composite.svg`
- `outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_closure_r1/figures/core_figures/C16_eight_indicator_gwr_composite.pdf`
- `outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_closure_r1/figures/core_figures/C30_PCNM_comparison_NOT_AVAILABLE.png`
- `outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_closure_r1/figures/core_figures/C30_PCNM_comparison_NOT_AVAILABLE.svg`
- `outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_closure_r1/figures/core_figures/C30_PCNM_comparison_NOT_AVAILABLE.pdf`
- `outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_closure_r1/figures/core_figures/C31_mechanism_path_NOT_AVAILABLE.png`
- `outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_closure_r1/figures/core_figures/C31_mechanism_path_NOT_AVAILABLE.svg`
- `outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_closure_r1/figures/core_figures/C31_mechanism_path_NOT_AVAILABLE.pdf`

## 标准 MGWR 拟合完成状态

标准 MGWR 包 `mgwr` 已安装于隔离临时环境；当前拟合状态：三个关键结局成功。详细样本、模型、带宽和结果见 [标准 MGWR 补充报告](PHASE_I_CLOSURE_REPORT.md)。
