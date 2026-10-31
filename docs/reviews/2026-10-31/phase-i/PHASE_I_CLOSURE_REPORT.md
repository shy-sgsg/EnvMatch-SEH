# Phase I 图件闭合与标准 MGWR 补充报告

phase: I-closure
mode: EXPLORATORY_ORACLE
outcome_guided_selection: true

本运行创建核心图件缺口审计并补制可由已保存数据支持的图件。所有旧运行均保留；图件基于既有 M4 配对、I-1 模型搜索、I-3/I-4/I-7表格及对齐数据库。没有重建/验证 RUSLE 因子乘积，也没有重跑 PCNM。

## 标准 MGWR

- 240m完整案例空间单元：7,387；按800m固定网格每格抽取1个单元，实际样本 n=770；随机种子 20261031。抽样只依空间位置，不按健康响应取样。
- 模型使用九项环境协变量及 log1p(Aa_20yr)，Gaussian kernel，自适应邻居数，AICc带宽选择；每个空间均衡单元等权，不使用源像元频数权重。标准GWR与标准MGWR使用同一空间样本和设计矩阵。
- 标准MGWR fit status：三个关键结局均成功；包版本、参数、迭代状态及错误信息见 `tables/standard_mgwr_metadata.json` 与 `tables/mgwr_fit_status.csv`。
- `mgwr==2.2.1` 尚未实现多带宽 MGWR 的局部 R²；`mgwr_local_coefficients.csv` 中 MGWR 行的 `local_r2` 留空，不能按零解释。模型级 R²、AICc及局部系数仍按包可提供的字段记录。

| 指标 | 同样本标准GWR R² | 标准MGWR R² | GWR AICc | MGWR AICc |
|---|---:|---:|---:|---:|
| SHI | 0.887 | 0.869 | 781.572 | 770.331 |
| OM | 0.714 | 0.696 | 1417.295 | 1389.804 |
| SW | 0.834 | 0.822 | 989.469 | 972.268 |

作用尺度表 `mgwr_variable_bandwidths.csv` 同时保留每个变量的自适应邻居数和对应中位邻域距离。局部系数是预测栅格上的条件关联，样本空间依赖、变量局部共线性与 Oracle 选择仍限制解释；本拟合不替代I-4全样本像元支持加权GWR。

## 核心图件状态

逐项证据见 `tables/core_figure_completeness.csv`。核心图 5 仅有 MBC 的最佳验证模型族选中 segmented，因此该图只标出 MBC 的候选阈值；其他结局未选中 segmented，不填造阈值。核心图 12 的源数据只有 `r_squared_apparent`，图中保留该字段名，不伪称最佳模型R²。

核心图 30、31 已创建显式的来源缺口状态图，但这两张不能代替科学对照森林图/机制路径图。I-7的PCNM比较值、PLS路径系数、测量载荷、方向、控制项均继续为空；需在接入源表后完成。

## 图件与数据

| 图件 | PNG | SVG | PDF |
|---|---|---|---|
| 1. 九项指标全样本响应曲线 (available) | [PNG](figures/i1_I05_global_response_curves.png) |  |  |
| 2. 零侵蚀与正侵蚀状态差异 (available) | [PNG](figures/i1_I04_zero_positive_boxplots.png) |  |  |
| 3. 九项指标相关性森林图 (available) | [PNG](figures/i1_I02_standardized_effect_forest.png) |  |  |
| 4. 最佳模型R²比较 (available) | [PNG](figures/ic_C04_global_best_model_r2.png) |  |  |
| 5. 非线性阈值图 (partial) | [PNG](figures/ic_C05_MBC_segmented_threshold_candidate.png) |  |  |
| 6. 分位数回归系数图 (available) | [PNG](figures/i1_I07_quantile_slopes.png) |  |  |
| 7. 土壤类型响应曲线 (available) | [PNG](figures/ic_C07_soil_type_matched_response_curves.png) |  |  |
| 8. TWI分组响应曲线 (available) | [PNG](figures/ic_C08_twi_group_matched_response_curves.png) |  |  |
| 9. Aa×TWI交互图 (partial) | [PNG](figures/ic_C09_twi_interaction_slopes.png) |  |  |
| 10. Aa×NDVI交互图 (partial) | [PNG](figures/ic_C10_ndvi_interaction_slopes.png) |  |  |
| 11. 环境组斜率森林图 (partial) | [PNG](figures/ic_C11_environment_group_slope_forest.png) |  |  |
| 12. 环境组最佳模型R²热图 (partial) | [PNG](figures/ic_C12_environment_group_apparent_r2.png) |  |  |
| 13. SHI的GWR局部系数图 (available) | [PNG](figures/i4_gwr_SHI_candidate_template.png) |  |  |
| 14. OM的GWR局部系数图 (available) | [PNG](figures/i4_gwr_OM_template.png) |  |  |
| 15. SW的GWR局部系数图 (available) | [PNG](figures/i4_gwr_SW_template.png) |  |  |
| 16. 八项指标GWR组合图 (available) | [PNG](figures/ic_C16_eight_indicator_gwr_composite.png) |  |  |
| 17. MGWR局部系数比较图 (available) | [PNG](figures/ic_C17_MGWR_local_coefficients.png) |  |  |
| 18. MGWR变量作用尺度图 (available) | [PNG](figures/ic_C18_MGWR_variable_scales.png) |  |  |
| 19. GWR与MGWR空间结果对照 (available) | [PNG](figures/ic_C19_GWR_MGWR_difference_map.png) |  |  |
| 20. GRF局部R²图 (available) | [PNG](figures/i5_grf_local_r2.png) |  |  |
| 21. GRF局部侵蚀响应图 (available) | [PNG](figures/i5_grf_local_Aa_marginal_effect.png) |  |  |
| 22. SHAP侵蚀贡献空间图 (available) | [PNG](figures/i5_spatial_shap_Aa.png) |  |  |
| 23. 空间ALE交互效应图 (available) | [PNG](figures/i5_local_ale_by_soil_type.png) |  |  |
| 24. SVC-GAM局部响应曲线 (available) | [PNG](figures/i5_svc_gam_local_curves.png) |  |  |
| 25. Local Spearman空间图 (available) | [PNG](figures/i6_local_spearman_240m.png) |  |  |
| 26. Bivariate LISA聚类图 (available) | [PNG](figures/i6_bivariate_local_moran_clusters.png) |  |  |
| 27. Getis-Ord Gi*热点图 (available) | [PNG](figures/i6_gistar_negative_response_gwr.png) |  |  |
| 28. 综合胁迫分区图 (available) | [PNG](figures/i8_stress_zoning_map.png) |  |  |
| 29. 物理/化学/生物维度对照 (available) | [PNG](figures/i7_health_dimension_comparison_forest.png) |  |  |
| 30. EnvMatch与PCNM对照森林图 (deferred_source_missing) | [PNG](figures/ic_C30_PCNM_comparison_NOT_AVAILABLE.png) |  |  |
| 31. 综合胁迫机制路径图 (deferred_source_missing) | [PNG](figures/ic_C31_mechanism_path_NOT_AVAILABLE.png) |  |  |

## 源文件和方法状态

全部输入数据库及图件配置/来源版本见 `manifests/provenance.json`；图件数据汇总表与局部系数明细位于 `tables/`。最终综合报告及图件索引在独立 I-9 r3 运行生成。
