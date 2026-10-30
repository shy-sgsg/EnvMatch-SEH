# Phase G 补充完成分析：候选范围、非线性配对模型与组间斜率

本补充分析校正了像元候选的研究范围，补充匹配配对差值的非线性模型搜索及环境组斜率比较。所有结果仍属 `EXPLORATORY_ORACLE`，并继承此前全域 Oracle 搜索对独立性造成的限制。

## 1. 核心配对关联

M4 / Aa_20yr 配对的 Spearman ρ：SHI -0.128、OM -0.114、SW -0.192；CEC +0.060；AN -0.002、MBC -0.003，属于近零关联。其他指标的效应量与方向见主报告表。单看 Wilcoxon P 值不能判断实质关联强弱。

## 2. 候选范围校正

按候选编号中明确的土类信息，从历史模型结果中排除 18 条不属于主要耕地土类的记录；SoilType 85 为水体。更正后的 Aa_20yr 表观候选 R²约为 0.326–0.782，CEC 的最高拟合候选为正向。完整候选清单、保留比例与方向依据见：

- `tables/completion/candidate_scope_reconciliation.csv`
- `tables/completion/in_scope_pixel_model_leaderboard.csv`
- `tables/completion/pixel_best_candidate_direction_reconstruction.csv`

模型含环境预测变量，且候选曾按结果筛选，不能把表观 R²称为侵蚀的独立贡献。

## 3. 配对响应模型与空间验证

配对差值模型分别保存线性、稳健、二次、自然样条和树模型试验记录。M4/Aa_20yr 配对差值全样本模型的表观 R²范围约为 0.000–0.109；按验证集选择的空间测试 R²整体接近零。像元模型、匹配配对模型、环境组模型均分别保存，不能混合比较。

环境组斜率补充图 10 展示每项指标至多六个最负组，区间为按空间块聚类稳健估计的 95% 区间。每组至少 100 对；这些组按健康响应筛选，属于 Oracle 探索，不是独立确认。完整试验表为 `tables/completion/paired_response_model_trials.csv`、`paired_spatial_model_trials.csv`、`group_slope_estimates.csv` 和 `group_slope_interaction_tests.csv`。
