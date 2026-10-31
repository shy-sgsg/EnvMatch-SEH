# Phase I-2：环境匹配后的条件关联

## 研究范围

本阶段沿用 Phase H 的 M4 配对候选及其模型试验，不重新匹配、不覆盖 Phase F–H。主暴露为 Aa_20yr，Aa_2020 作补充。配对端点按高/低侵蚀方向保留；零值—正值和双正值子总体按所用 Aa 期别分别定义。完整配对表和 H2 模型试验都来自 `EXPLORATORY_ORACLE`，匹配候选与响应模型曾依健康结果择优；因此报告作为 Oracle 条件关联，不作为独立确认。

统计单位分开呈现：Matched-Pixel 使用配对端点 cell_id 去重后的唯一栅格；Matched-Pair 保留每一对端点的高减低差值。像元值来自健康随机森林预测栅格，不能把像元量解释成独立实测样本量。Aa 沿用预计算值，不重算或验证 RUSLE 因子乘积。

## M4 支持量

| 暴露    | M4候选            | 配对总体      |   配对数 |   唯一端点数 |   中位地理距离(m) |   中位环境距离 |
|:--------|:------------------|:--------------|---------:|-------------:|------------------:|---------------:|
| Aa_20yr | stage2_Aa_20yr_01 | all_erosion   |   111289 |       222578 |           301.496 |          0.032 |
| Aa_20yr | stage2_Aa_20yr_01 | zero_positive |    68172 |       136344 |           323.110 |          0.030 |
| Aa_20yr | stage2_Aa_20yr_01 | both_positive |    43117 |        86234 |           270.000 |          0.035 |
| Aa_2020 | stage2_Aa_2020_01 | all_erosion   |   133092 |       266184 |           241.868 |          0.021 |
| Aa_2020 | stage2_Aa_2020_01 | zero_positive |        0 |            0 |           nan     |        nan     |
| Aa_2020 | stage2_Aa_2020_01 | both_positive |   133092 |       266184 |           241.868 |          0.021 |
| Aa_2020 | stage2_Aa_2020_03 | all_erosion   |   134601 |       269202 |           420.000 |          0.015 |
| Aa_2020 | stage2_Aa_2020_03 | zero_positive |        0 |            0 |           nan     |        nan     |
| Aa_2020 | stage2_Aa_2020_03 | both_positive |   134601 |       269202 |           420.000 |          0.015 |
| Aa_2020 | stage2_Aa_2020_06 | all_erosion   |   117790 |       235580 |           484.665 |          0.296 |
| Aa_2020 | stage2_Aa_2020_06 | zero_positive |     3306 |         6612 |           553.173 |          0.371 |
| Aa_2020 | stage2_Aa_2020_06 | both_positive |   114484 |       228968 |           483.735 |          0.294 |

Aa_2020 的 H2 结果包含三个既有 M4 候选（`stage2_Aa_2020_01/03/06`）。零—正值配对仅在候选 06 中有支持；候选 01/03 全部为双正值配对。因此零—正值结果与候选 06 对应，不能与候选 01/03 当作同一匹配样本的子集。Aa_20yr 使用既有候选 `stage2_Aa_20yr_01`。

## Aa_20yr：主要条件关联

### 全侵蚀范围配对

| 指标          |   配对完整案例 |   Spearman ρ |   Kendall τ-b |   平均Δ健康 |   中位Δ健康 |   标准化斜率β |   负向配对比例 |   配对OLS R² |
|:--------------|---------------:|-------------:|--------------:|------------:|------------:|--------------:|---------------:|-------------:|
| SW            |         111278 |       -0.272 |        -0.184 |      -4.475 |      -4.607 |        -0.269 |          0.778 |        0.073 |
| SHI_candidate |         110657 |       -0.144 |        -0.097 |      -0.024 |      -0.022 |        -0.144 |          0.707 |        0.021 |
| OM            |         111278 |       -0.115 |        -0.078 |      -2.061 |      -2.003 |        -0.100 |          0.662 |        0.010 |
| pH            |         111281 |       -0.051 |        -0.035 |      -0.010 |      -0.006 |        -0.043 |          0.582 |        0.002 |
| BD            |         111278 |       -0.032 |        -0.021 |      -0.030 |      -0.027 |        -0.017 |          0.601 |        0.000 |
| AN            |         111281 |       -0.024 |        -0.016 |      -0.032 |      -0.117 |        -0.025 |          0.514 |        0.001 |
| RESP          |         111278 |       -0.014 |        -0.010 |     -60.993 |     -61.363 |         0.016 |          0.604 |        0.000 |
| CEC           |         111278 |        0.045 |         0.030 |       0.674 |       0.504 |         0.029 |          0.388 |        0.001 |
| MBC           |         111278 |        0.056 |         0.037 |      -1.107 |      -1.542 |         0.088 |          0.553 |        0.008 |

该候选中配对 Spearman 最负的是 SW（ρ=-0.272，n=111,278）；负向配对比例为 77.8%。这描述每对高侵蚀端与低侵蚀端的健康差值同侵蚀差距的排序关系，不是因果效应。均值差、中央値差、标准化斜率与零值/双正值分层详见输出 CSV。

### 唯一匹配端点像元

| 指标          |   唯一端点完整案例 |   Spearman ρ |   Kendall τ-b |   标准化斜率β |   像元OLS R² |
|:--------------|-------------------:|-------------:|--------------:|--------------:|-------------:|
| SW            |             222567 |       -0.567 |        -0.396 |        -0.567 |        0.321 |
| SHI_candidate |             221897 |       -0.452 |        -0.316 |        -0.448 |        0.200 |
| OM            |             222567 |       -0.416 |        -0.282 |        -0.409 |        0.167 |
| RESP          |             222567 |       -0.251 |        -0.170 |        -0.227 |        0.052 |
| BD            |             222567 |       -0.182 |        -0.125 |        -0.138 |        0.019 |
| AN            |             222570 |       -0.178 |        -0.123 |        -0.184 |        0.034 |
| pH            |             222570 |       -0.145 |        -0.101 |        -0.155 |        0.024 |
| MBC           |             222567 |       -0.107 |        -0.070 |        -0.055 |        0.003 |
| CEC           |             222567 |        0.195 |         0.133 |         0.188 |        0.035 |

这组相关将每个 M4 候选内的端点按 cell_id 去重后计算。端点依赖同一匹配设计与空间表面，不能按独立像元解释常规显著性。

### 与 I-1 全样本对照

下表在同一 Aa_20yr 暴露下，把 I-1 全掩膜像元相关与当前 M4 唯一端点相关及配对差值相关并列。匹配样本端点和全样本的估计总体不同；差值仅用于描述关系变化，不作为匹配的因果效应。

| 指标          |   全样本n |   全样本ρ |   匹配端点n |   匹配端点ρ |   匹配端点−全样本Δρ |   配对n |   配对差值ρ |
|:--------------|----------:|----------:|------------:|------------:|--------------------:|--------:|------------:|
| SW            |    388092 |    -0.584 |      222567 |      -0.567 |               0.017 |  111278 |      -0.272 |
| SHI_candidate |    384737 |    -0.469 |      221897 |      -0.452 |               0.017 |  110657 |      -0.144 |
| OM            |    388092 |    -0.415 |      222567 |      -0.416 |              -0.001 |  111278 |      -0.115 |
| RESP          |    388092 |    -0.272 |      222567 |      -0.251 |               0.021 |  111278 |      -0.014 |
| AN            |    387906 |    -0.233 |      222570 |      -0.178 |               0.055 |  111281 |      -0.024 |
| BD            |    388092 |    -0.200 |      222567 |      -0.182 |               0.018 |  111278 |      -0.032 |
| MBC           |    388092 |    -0.118 |      222567 |      -0.107 |               0.011 |  111278 |       0.056 |
| pH            |    387906 |    -0.109 |      222570 |      -0.145 |              -0.036 |  111281 |      -0.051 |
| CEC           |    388092 |     0.168 |      222567 |       0.195 |               0.027 |  111278 |       0.045 |

SHI、OM、SW 的全样本与匹配端点 Spearman 接近（例如 OM 约 −0.415 与 −0.416），而配对差值相关较弱；这显示端点层面的空间梯度与每对高低端差值回答不同问题。M4 匹配后不应只用单一配对系数概括全部侵蚀—健康关系。

## 模型表现

下表为 Phase H 保留的全侵蚀范围候选模型中，按表观 R² 汇总的 Aa_20yr 结果。Matched-Pixel 最佳表观 R²为 SHI / random_forest（R²=0.505）；Matched-Pair 最佳表观 R²为 SW / random_forest（R²=0.101）。

| 指标          | 统计单位      | 候选              | 模型          |      n |   表观R² |    RMSE |     MAE |
|:--------------|:--------------|:------------------|:--------------|-------:|---------:|--------:|--------:|
| AN            | matched_pair  | stage2_Aa_20yr_01 | random_forest | 111281 |    0.003 |   7.983 |   5.336 |
| BD            | matched_pair  | stage2_Aa_20yr_01 | random_forest | 111278 |    0.059 |   0.160 |   0.124 |
| CEC           | matched_pair  | stage2_Aa_20yr_01 | random_forest | 111278 |    0.030 |   1.964 |   1.564 |
| MBC           | matched_pair  | stage2_Aa_20yr_01 | random_forest | 111278 |    0.067 |  15.654 |  12.346 |
| OM            | matched_pair  | stage2_Aa_20yr_01 | random_forest | 111278 |    0.048 |   3.646 |   3.020 |
| RESP          | matched_pair  | stage2_Aa_20yr_01 | random_forest | 111278 |    0.060 | 213.744 | 171.223 |
| SHI_candidate | matched_pair  | stage2_Aa_20yr_01 | random_forest | 110657 |    0.035 |   0.043 |   0.034 |
| SW            | matched_pair  | stage2_Aa_20yr_01 | random_forest | 111278 |    0.101 |   5.150 |   4.319 |
| pH            | matched_pair  | stage2_Aa_20yr_01 | random_forest | 111281 |    0.011 |   0.053 |   0.037 |
| AN            | matched_pixel | stage2_Aa_20yr_01 | random_forest | 222570 |    0.482 |  12.666 |   9.836 |
| BD            | matched_pixel | stage2_Aa_20yr_01 | random_forest | 222567 |    0.197 |   0.120 |   0.102 |
| CEC           | matched_pixel | stage2_Aa_20yr_01 | random_forest | 222567 |    0.392 |   1.441 |   1.190 |
| MBC           | matched_pixel | stage2_Aa_20yr_01 | random_forest | 222567 |    0.234 |  11.539 |   9.462 |
| OM            | matched_pixel | stage2_Aa_20yr_01 | random_forest | 222567 |    0.379 |   2.627 |   2.058 |
| RESP          | matched_pixel | stage2_Aa_20yr_01 | random_forest | 222567 |    0.364 | 155.758 | 125.450 |
| SHI_candidate | matched_pixel | stage2_Aa_20yr_01 | random_forest | 221897 |    0.505 |   0.035 |   0.027 |
| SW            | matched_pixel | stage2_Aa_20yr_01 | random_forest | 222567 |    0.501 |   3.757 |   2.972 |
| pH            | matched_pixel | stage2_Aa_20yr_01 | random_forest | 222570 |    0.203 |   0.073 |   0.056 |

零—正与双正值模型试验仅保留 OLS 子组模型；候选覆盖及全部原始模型记录见 `tables/model_trials_reused.csv`。全范围模型家族来自 Phase H 已有试验（OLS、二次项、自然样条、statsmodels GAM、Huber、Random Forest、Extra Trees、HistGradientBoosting 等），不在 Phase I 重跑。XGBoost 未运行：当前环境未安装 `xgboost`，本阶段未新增依赖。

所有上述 R²是同样本表观拟合，不表示空间外推。空间分块验证应读取 Phase H `phase_h_spatial_validation_summary.csv` 与 Phase I-1 的全样本空间留出结果；后续 Phase I-4 将开展配对差值空间模型。

## 配对响应曲线与模型比较图

曲线图展示选定 M4 候选中每个总体的等样本量分箱中位数；源数据保存在 `figures/response_curves/`。Aa_2020 图使用有零—正值支持的候选 06，环境向量与候选 01/03 不同，图中明确标注。

- `outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_i2_r2/figures/response_curves/I09_Aa_20yr_matched_pair.png`
- `outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_i2_r2/figures/response_curves/I09_Aa_20yr_matched_pair.svg`
- `outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_i2_r2/figures/response_curves/I09_Aa_20yr_matched_pair.pdf`
- `outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_i2_r2/figures/response_curves/I09_Aa_20yr_matched_pixel.png`
- `outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_i2_r2/figures/response_curves/I09_Aa_20yr_matched_pixel.svg`
- `outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_i2_r2/figures/response_curves/I09_Aa_20yr_matched_pixel.pdf`
- `outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_i2_r2/figures/response_curves/I09_Aa_2020_matched_pair.png`
- `outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_i2_r2/figures/response_curves/I09_Aa_2020_matched_pair.svg`
- `outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_i2_r2/figures/response_curves/I09_Aa_2020_matched_pair.pdf`
- `outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_i2_r2/figures/response_curves/I09_Aa_2020_matched_pixel.png`
- `outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_i2_r2/figures/response_curves/I09_Aa_2020_matched_pixel.svg`
- `outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_i2_r2/figures/response_curves/I09_Aa_2020_matched_pixel.pdf`
- `outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_i2_r2/figures/model_comparison/I10_m4_best_apparent_r2.png`
- `outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_i2_r2/figures/model_comparison/I10_m4_best_apparent_r2.svg`
- `outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_i2_r2/figures/model_comparison/I10_m4_best_apparent_r2.pdf`

## Oracle 候选定义与复现

|暴露|M4候选|环境变量|距离方法|地理卡尺(m)|环境卡尺|侵蚀秩差|
|---|---|---|---|---:|---:|---:|
|Aa_2020|stage2_Aa_2020_01|["Elevation", "LST"]|direct_nearest_neighbor|500|0.35|0.2|
|Aa_2020|stage2_Aa_2020_03|["Elevation", "LST"]|standardized_euclidean|1000|0.35|0.2|
|Aa_2020|stage2_Aa_2020_06|["Elevation", "Cur", "TWI", "LST", "NDWI"]|standardized_euclidean|1500|0.75|0.3|
|Aa_20yr|stage2_Aa_20yr_01|["Elevation", "LST"]|standardized_euclidean|500|0.75|0.3|

结果选择沿用 Phase H 的 Oracle 试验登记；所有已保存试验及筛选字段均复制到本运行目录。报告不把匹配后的关联解释成处理因果效应，且不改变正式研究设计或既有 PCNM 结果。

## 文件索引

- 配对关联：`tables/matched_pair_associations.csv`
- 唯一端点关联：`tables/matched_pixel_associations.csv`
- M4 支持量：`tables/match_population_support.csv`
- 全样本—匹配对照：`tables/global_vs_matched_associations.csv`
- 最佳模型表：`leaderboards/matched_model_leaderboard.csv`
- Phase H 原始模型试验副本：`tables/model_trials_reused.csv`
- 候选匹配规则：`tables/m4_candidate_designs.csv`
- 分箱响应曲线：`results/matching/response_curve_points.csv`
- 运行配置与来源：`configs/phase_i_matching.yaml`、`manifests/provenance.json`
