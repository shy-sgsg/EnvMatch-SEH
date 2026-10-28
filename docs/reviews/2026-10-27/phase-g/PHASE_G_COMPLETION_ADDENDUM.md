## 补充完成分析（2026-10-27）

本节补足首次 Phase G 报告中的候选范围校正、配对差值非线性模型搜索和组间斜率交互检验。其余主报告结果保持可追溯；本节中的像元模型排名以此处重新限定研究范围后的榜单为准。

### 三个核心问题的直接回答

M4 / Aa_20yr 配对中，Δ健康对 Δlog1p(Aa) 的 Spearman 方向如下。AN 与 MBC 的相关系数绝对值低于 0.01，应视为近零；不能据其 P 值称为有实质关联。

| 指标 | 配对数 | 高减低平均差 | Δ健康<0比例 | Spearman ρ | 方向 | 标准化斜率 |
| --- | --- | --- | --- | --- | --- | --- |
| SHI_candidate | 89811 | -0.03314 | 0.7674 | -0.128 | negative | -0.1234 |
| AN | 90848 | 0.1214 | 0.5106 | -0.002304 | near zero | -0.007535 |
| OM | 90867 | -2.854 | 0.7413 | -0.1139 | negative | -0.07671 |
| CEC | 90867 | 1.004 | 0.34 | 0.06005 | positive | 0.02048 |
| pH | 90848 | -0.01468 | 0.5917 | -0.03621 | negative | -0.03443 |
| SW | 90867 | -6.062 | 0.8565 | -0.192 | negative | -0.204 |
| BD | 90867 | -0.04625 | 0.6458 | -0.03131 | negative | -0.007901 |
| MBC | 90867 | -2.249 | 0.5975 | -0.00288 | near zero | 0.05936 |
| RESP | 90867 | -90.86 | 0.6604 | -0.05404 | negative | 0.005865 |

对 20 年 Aa 的像元级表观搜索，最高 R² 候选按预测方向既有负向，也有正向（例如 CEC 为正）；完整校正后逐指标清单如下。2020 年 Aa 的同类结果另存于完整榜单中。最佳像元模型使用环境预测变量，R² 不等于侵蚀的独立解释贡献。配对差值模型的全样本最高 R²较低，M4/Aa_20yr 范围约为 0.0001–0.109；按 validation 选出的空间模型 test 中位 R²集中在 0 附近，仅部分结局在十种子摘要中略高于 0。环境组中最负斜率的指标与分组方法并不一致，见下面的组间检验和环境画像。

### 候选范围校正

重排前先从历史 Oracle 表观模型候选中排除候选 ID 明确标注为非耕地主要土类的记录；其中 SoilType 85 为水体。本次排除 18 条结果行。总体候选数和逐期、逐指标的保留比例见 `tables/completion/candidate_scope_reconciliation.csv`。更正后的 Aa_20yr 排名如下：

| 指标 | 排名目标 | 候选 | 模型 | 表观 R² | 方向 | 方向证据 | 像元数 | 范围内候选数 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AN | in_scope_highest_apparent_r2 | stage2_Aa_20yr_AN_R030 | random_forest | 0.7822 | negative | exact_r_squared_reproduction; exact source stage-2 sample, variables, family, and seed; conditional curve at sample medians | 9.893e+04 | 163 |
| BD | in_scope_highest_apparent_r2 | full_Aa_20yr_BD_V7 | ols | 0.3256 | negative | direction stored with candidate trial | 2.948e+05 | 161 |
| CEC | in_scope_highest_apparent_r2 | full_Aa_20yr_CEC_V8 | ols | 0.3775 | positive | direction stored with candidate trial | 2.948e+05 | 161 |
| MBC | in_scope_highest_apparent_r2 | stage2_Aa_20yr_MBC_R019 | random_forest | 0.3461 | negative | exact_r_squared_reproduction; exact source stage-2 sample, variables, family, and seed; conditional curve at sample medians | 7.518e+04 | 161 |
| OM | in_scope_highest_apparent_r2 | stage2_Aa_20yr_OM_R019 | random_forest | 0.4421 | negative | exact_r_squared_reproduction; exact source stage-2 sample, variables, family, and seed; conditional curve at sample medians | 7.518e+04 | 161 |
| RESP | in_scope_highest_apparent_r2 | stage2_Aa_20yr_RESP_R019 | random_forest | 0.4183 | negative | exact_r_squared_reproduction; exact source stage-2 sample, variables, family, and seed; conditional curve at sample medians | 7.518e+04 | 161 |
| SHI_candidate | in_scope_highest_apparent_r2 | full_Aa_20yr_SHI_candidate_V5 | random_forest | 0.6369 | negative | same outcome/exposure/model/environment vector in stored selected full-sample response curve; curve is a proxy if candidate fit used another sample | 3.847e+05 | 162 |
| SW | in_scope_highest_apparent_r2 | stage2_Aa_20yr_SW_V5 | random_forest | 0.6209 | negative | exact_r_squared_reproduction; exact source stage-2 sample, variables, family, and seed; conditional curve at sample medians | 9.893e+04 | 162 |
| pH | in_scope_highest_apparent_r2 | full_Aa_20yr_pH_R012 | random_forest | 0.7204 | negative | same outcome/exposure/model/environment vector in stored selected full-sample response curve; curve is a proxy if candidate fit used another sample | 2.948e+05 | 161 |
| AN | in_scope_highest_negative_apparent_r2 | stage2_Aa_20yr_AN_R030 | random_forest | 0.7822 | negative | exact_r_squared_reproduction; exact source stage-2 sample, variables, family, and seed; conditional curve at sample medians | 9.893e+04 | 163 |
| BD | in_scope_highest_negative_apparent_r2 | full_Aa_20yr_BD_V7 | ols | 0.3256 | negative | direction stored with candidate trial | 2.948e+05 | 161 |
| CEC | in_scope_highest_negative_apparent_r2 | full_Aa_20yr_CEC_R019 | piecewise | 0.07364 | negative | direction stored with candidate trial | 2.948e+05 | 161 |
| MBC | in_scope_highest_negative_apparent_r2 | stage2_Aa_20yr_MBC_R019 | random_forest | 0.3461 | negative | exact_r_squared_reproduction; exact source stage-2 sample, variables, family, and seed; conditional curve at sample medians | 7.518e+04 | 161 |
| OM | in_scope_highest_negative_apparent_r2 | stage2_Aa_20yr_OM_R019 | random_forest | 0.4421 | negative | exact_r_squared_reproduction; exact source stage-2 sample, variables, family, and seed; conditional curve at sample medians | 7.518e+04 | 161 |
| RESP | in_scope_highest_negative_apparent_r2 | stage2_Aa_20yr_RESP_R019 | random_forest | 0.4183 | negative | exact_r_squared_reproduction; exact source stage-2 sample, variables, family, and seed; conditional curve at sample medians | 7.518e+04 | 161 |
| SHI_candidate | in_scope_highest_negative_apparent_r2 | full_Aa_20yr_SHI_candidate_V5 | random_forest | 0.6369 | negative | same outcome/exposure/model/environment vector in stored selected full-sample response curve; curve is a proxy if candidate fit used another sample | 3.847e+05 | 162 |
| SW | in_scope_highest_negative_apparent_r2 | stage2_Aa_20yr_SW_V5 | random_forest | 0.6209 | negative | exact_r_squared_reproduction; exact source stage-2 sample, variables, family, and seed; conditional curve at sample medians | 9.893e+04 | 162 |
| pH | in_scope_highest_negative_apparent_r2 | full_Aa_20yr_pH_R012 | random_forest | 0.7204 | negative | same outcome/exposure/model/environment vector in stored selected full-sample response curve; curve is a proxy if candidate fit used another sample | 2.948e+05 | 161 |

表观排名仍是 Oracle 搜索结果，且像元健康层是预测栅格。它不代表 Aa 的独立解释贡献。原排行榜中 pH 的部分最高值来自水体 85，该结果不纳入耕地解释；请以 `tables/completion/in_scope_pixel_model_leaderboard.csv` 为准。对可能改变负向 R² 排名的 stage-2 树模型，重建了固定抽样、变量、模型族和随机种子的表观拟合；17 个候选方向的源 R² 均已复现，最大绝对差为 1.67e-16。记录见 `tables/completion/pixel_best_candidate_direction_reconstruction.csv`；未被重建或没有曲线依据的方向仍标为 `unknown`。

### 配对差值模型

分析单位为既有 Phase F 配对。预测量为 `log1p(Aa_high)-log1p(Aa_low)`，响应为 `health_high-health_low`。每个 M3/M4 × 两期 Aa × 九指标组合先对固定随机样本搜索 8 个模型族，再把表观 R² 最优模型及负向候选在该组合全部可用配对上重拟合；并单列全配对 OLS 基线。抽样探索、全配对重拟合和空间留出有不同 `fit_scope`，不得混合解释。

M4 / Aa_20yr 的全配对重拟合领先候选如下。方向由模型拟合曲线在观测暴露差 2%–98% 范围内的变化确定；高 R² 仍是表观拟合：

| 指标 | 排名目标 | 模型族 | 全配对表观 R² | 方向 | 实际差值 Spearman ρ | 配对数 | 筛选目标 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AN | highest_full_sample_paired_apparent_r2 | random_forest | 0.03863 | positive | -0.002304 | 9.085e+04 | highest_discovery_apparent_r2 |
| AN | highest_negative_full_sample_paired_apparent_r2 | quadratic | 7.857e-05 | negative | -0.002304 | 9.085e+04 | highest_negative_discovery_apparent_r2 |
| BD | highest_full_sample_paired_apparent_r2 | random_forest | 0.05131 | negative | -0.03131 | 9.087e+04 | highest_discovery_apparent_r2 |
| BD | highest_negative_full_sample_paired_apparent_r2 | random_forest | 0.05131 | negative | -0.03131 | 9.087e+04 | highest_discovery_apparent_r2 |
| CEC | highest_full_sample_paired_apparent_r2 | random_forest | 0.06733 | positive | 0.06005 | 9.087e+04 | highest_discovery_apparent_r2\|highest_negative_discovery_apparent_r2 |
| MBC | highest_full_sample_paired_apparent_r2 | random_forest | 0.07694 | negative | -0.00288 | 9.087e+04 | highest_discovery_apparent_r2 |
| MBC | highest_negative_full_sample_paired_apparent_r2 | random_forest | 0.07694 | negative | -0.00288 | 9.087e+04 | highest_discovery_apparent_r2 |
| OM | highest_full_sample_paired_apparent_r2 | random_forest | 0.0909 | negative | -0.1139 | 9.087e+04 | highest_discovery_apparent_r2\|highest_negative_discovery_apparent_r2 |
| OM | highest_negative_full_sample_paired_apparent_r2 | random_forest | 0.0909 | negative | -0.1139 | 9.087e+04 | highest_discovery_apparent_r2\|highest_negative_discovery_apparent_r2 |
| RESP | highest_full_sample_paired_apparent_r2 | random_forest | 0.08432 | negative | -0.05404 | 9.087e+04 | highest_discovery_apparent_r2 |
| RESP | highest_negative_full_sample_paired_apparent_r2 | random_forest | 0.08432 | negative | -0.05404 | 9.087e+04 | highest_discovery_apparent_r2 |
| SHI_candidate | highest_full_sample_paired_apparent_r2 | random_forest | 0.07502 | negative | -0.128 | 8.981e+04 | highest_discovery_apparent_r2\|highest_negative_discovery_apparent_r2 |
| SHI_candidate | highest_negative_full_sample_paired_apparent_r2 | random_forest | 0.07502 | negative | -0.128 | 8.981e+04 | highest_discovery_apparent_r2\|highest_negative_discovery_apparent_r2 |
| SW | highest_full_sample_paired_apparent_r2 | random_forest | 0.109 | negative | -0.192 | 9.087e+04 | highest_discovery_apparent_r2\|highest_negative_discovery_apparent_r2 |
| SW | highest_negative_full_sample_paired_apparent_r2 | random_forest | 0.109 | negative | -0.192 | 9.087e+04 | highest_discovery_apparent_r2\|highest_negative_discovery_apparent_r2 |
| pH | highest_full_sample_paired_apparent_r2 | random_forest | 0.04014 | positive | -0.03621 | 9.085e+04 | highest_discovery_apparent_r2\|highest_negative_discovery_apparent_r2 |
| pH | highest_negative_full_sample_paired_apparent_r2 | ols | 0.001185 | negative | -0.03621 | 9.085e+04 | NA |

### 空间分块模型验证

每种配对方式与 Aa 时期从固定 5,000 对样本中按配对中点的 5 km 空间块做十种子 60/20/20 分配；模型只按各种子 validation R² 选择，然后才计算 test 指标。下表的 test 中位数和四分位距是诊断汇总，没有参与排序。约 12.9% 的高、低侵蚀成员落在不同的 5 km 网格块，但配对整体按中点分块，因此边界附近仍可能带入相邻块空间依赖。加上历史 Oracle 已接触全研究区，这些是内部空间稳定性检查，不是独立外部验证。

| 指标 | 验证选择规则 | 入选模型族计数 | 验证 R² 中位数 | 测试 R² 中位数（报告） | 测试 R² Q1 | 测试 R² Q3 | 空间种子数 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AN | highest_negative_validation_r2 | {"huber": 3, "natural_spline": 2, "extra_trees": 2, "quadratic": 1, "ols": 1} | -0.000426 | -0.013 | -0.03879 | -0.009851 | 9 |
| AN | highest_validation_r2 | {"huber": 3, "extra_trees": 3, "natural_spline": 3, "ols": 1} | -0.0001701 | -0.01279 | -0.04264 | -0.007348 | 10 |
| BD | highest_negative_validation_r2 | {"natural_spline": 4, "statsmodels_gam": 3, "quadratic": 2} | 0.001484 | -0.002658 | -0.009914 | -0.001578 | 9 |
| BD | highest_validation_r2 | {"natural_spline": 5, "statsmodels_gam": 2, "quadratic": 2, "huber": 1} | 0.001393 | -0.004114 | -0.009459 | -0.001873 | 10 |
| CEC | highest_negative_validation_r2 | {"huber": 3, "extra_trees": 1, "ols": 1} | -0.005475 | -0.004207 | -0.005083 | -0.003802 | 5 |
| CEC | highest_validation_r2 | {"natural_spline": 4, "statsmodels_gam": 3, "extra_trees": 2, "quadratic": 1} | 0.01185 | -0.001769 | -0.002988 | 0.00225 | 10 |
| MBC | highest_negative_validation_r2 | {"statsmodels_gam": 4, "natural_spline": 3, "quadratic": 1, "extra_trees": 1} | 0.01987 | 0.001949 | -0.00357 | 0.005458 | 9 |
| MBC | highest_validation_r2 | {"natural_spline": 4, "statsmodels_gam": 3, "extra_trees": 2, "quadratic": 1} | 0.01869 | 0.002314 | -0.003493 | 0.005502 | 10 |
| OM | highest_negative_validation_r2 | {"statsmodels_gam": 5, "extra_trees": 4, "natural_spline": 1} | 0.02118 | 0.01602 | 0.0119 | 0.02089 | 10 |
| OM | highest_validation_r2 | {"statsmodels_gam": 5, "extra_trees": 4, "natural_spline": 1} | 0.02118 | 0.01602 | 0.0119 | 0.02089 | 10 |
| RESP | highest_negative_validation_r2 | {"statsmodels_gam": 6, "natural_spline": 2, "extra_trees": 2} | 0.01917 | 0.003648 | 0.0004641 | 0.01387 | 10 |
| RESP | highest_validation_r2 | {"statsmodels_gam": 5, "extra_trees": 3, "natural_spline": 2} | 0.02047 | 0.003648 | 0.0004641 | 0.01387 | 10 |
| SHI_candidate | highest_negative_validation_r2 | {"extra_trees": 3, "statsmodels_gam": 3, "huber": 3, "natural_spline": 1} | 0.002406 | 0.02058 | 0.0118 | 0.02379 | 10 |
| SHI_candidate | highest_validation_r2 | {"extra_trees": 3, "statsmodels_gam": 3, "huber": 3, "natural_spline": 1} | 0.002406 | 0.02058 | 0.0118 | 0.02379 | 10 |
| SW | highest_negative_validation_r2 | {"statsmodels_gam": 5, "extra_trees": 3, "huber": 1, "ols": 1} | 0.03251 | 0.05261 | 0.03373 | 0.06887 | 10 |
| SW | highest_validation_r2 | {"statsmodels_gam": 5, "extra_trees": 3, "huber": 1, "ols": 1} | 0.03251 | 0.05261 | 0.03373 | 0.06887 | 10 |
| pH | highest_negative_validation_r2 | {"quadratic": 3, "natural_spline": 2, "extra_trees": 2, "ols": 1, "statsmodels_gam": 1, "huber": 1} | -0.002813 | -0.007108 | -0.01288 | -0.003876 | 10 |
| pH | highest_validation_r2 | {"quadratic": 3, "natural_spline": 2, "extra_trees": 2, "ols": 1, "statsmodels_gam": 1, "huber": 1} | -0.002813 | -0.007108 | -0.01288 | -0.003876 | 10 |

### 环境组斜率差异

保留原有 SoilType、Phase F V5 K-means k=6、Ward k=6 和六种环境梯度分组；另以 Elevation 与 LST 建立土类内 MiniBatch K-means k=4、6、8、10、12。分组标签只使用环境变量和 SoilType，之后再读取配对健康差值。每组报告 5 km 空间块聚类稳健斜率区间，并对各组法的整体交互 Wald 检验做全局 BH 校正。交互检验表覆盖两种匹配方法、两期 Aa、九项结局和所有分组；低于预设 100 对或 5 个空间块的组合保留为不足状态。全局 BH 校正共 504 项，其中 q<0.05 的有 473 项；M4 / Aa_20yr 可检验的 126 项中有 123 项 q<0.05。报告直接展示 log(P) 和 log(q)，避免极小尾概率被浮点下限误读。

M4 / Aa_20yr 下各指标斜率最负的环境组候选如下，供探索和审查，不作为正式分区：

| 指标 | 分组法 | 组别 | 配对数 | 斜率 | 聚类稳健 95% CI 下限 | 聚类稳健 95% CI 上限 | 整体交互 log(P) | 全局 BH log(q) | 组内平均高程 | 组内平均 TWI | 组内平均 LST | 组内平均 NDVI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RESP | kmeans_terrain_k10 | soil_42_cluster_3 | 536 | -180.4 | -212.2 | -148.6 | -3.575e+05 | -3.575e+05 | 108.8 | 8.694 | 16.95 | 0.5193 |
| MBC | kmeans_terrain_k10 | soil_42_cluster_3 | 536 | -10.94 | -13.94 | -7.931 | -7353 | -7351 | 108.8 | 8.694 | 16.95 | 0.5193 |
| SW | kmeans_terrain_k10 | soil_42_cluster_3 | 536 | -4.685 | -5.478 | -3.892 | -2.057e+06 | -2.057e+06 | 108.8 | 8.694 | 16.95 | 0.5193 |
| AN | kmeans_terrain_k10 | soil_42_cluster_8 | 430 | -3.886 | -4.777 | -2.994 | -2.531e+06 | -2.531e+06 | 114.3 | 7.557 | 16.8 | 0.5248 |
| OM | kmeans_terrain_k10 | soil_27_cluster_6 | 5031 | -3.084 | -3.32 | -2.847 | -3186 | -3184 | 104 | 8.789 | 16.58 | 0.5128 |
| CEC | kmeans_terrain_k8 | soil_42_cluster_4 | 230 | -0.2245 | -0.5796 | 0.1306 | -7.612e+04 | -7.611e+04 | 109.7 | 8.966 | 17.63 | 0.4051 |
| BD | kmeans_terrain_k10 | soil_42_cluster_3 | 536 | -0.09076 | -0.1231 | -0.05838 | -4365 | -4363 | 108.8 | 8.694 | 16.95 | 0.5193 |
| pH | kmeans_terrain_k12 | soil_26_cluster_6 | 773 | -0.07651 | -0.1115 | -0.04151 | -1.429e+10 | -1.429e+10 | 103.5 | 8.492 | 15.73 | 0.4684 |
| SHI_candidate | gradient_Elevation | 27:q1 | 10599 | -0.03092 | -0.03815 | -0.02369 | -198.5 | -197.8 | 103.9 | 9.121 | 16.8 | 0.491 |

环境特征均基于高、低侵蚀配对成员均值，补充报告均值、中位数、标准差和四分位数，详见 `tables/completion/environment_group_profiles.csv`。整体组间斜率差异见 `tables/completion/group_slope_interaction_tests.csv`；其中 M4 / Aa_20yr 的可检验组法数为 126，完整候选数为 126。

新增图 10 是各指标最负组斜率的森林图，展示聚类稳健区间并清楚标出 Oracle 结果导向筛选。文件为 `figures/completion/10_environment_group_slope_forest.png`、`.svg`、`.pdf`，源数据和图注同目录的 `data/`、`captions/`。

### 两类响应曲线

按用户要求，原图 5 同时保留两类曲线：固定 20,000 像元共同样本拟合的自然样条，以及由全样本 1,024 次 / 13 个模型族表观 R² 搜索选出的候选曲线。两者分别标出样本范围和选择机制；后者不代表独立验证，也不用于替代新增的配对差值模型。原图文件在 `figures/05_response_curves.*`，底层曲线表在 `tables/response_curve_data.csv`。

### 补充文件与运行来源

- 运行 ID：`phase_g_20261027_completion`；模式：`EXPLORATORY_ORACLE`；本次运行基线 HEAD：`efdcdcd313f57aae1587466f9e0115a2846bec5e`。
- 独立运行输出：`outputs/exploratory_oracle/phase_g_20261027_completion/`；审查副本：`tables/completion/`。
- 新增候选榜单、全配对模型搜索、空间验证、环境组斜率、环境剖面、实验登记和产物清单均在上述目录。
- 输入使用既有对齐数据库、Phase F 配对与 Oracle 候选登记；继续直接使用预计算 Aa，不重新验证 RUSLE 乘积，不运行 PCNM，不更改总纲。
- `CHATGPT_REVIEW_REQUEST.md` 已追加本次补充材料路径。提交 SHA 仍需用户本地提交后填写；审查材料没有发送给外部。
