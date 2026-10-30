# Phase F 探索性研究报告：环境匹配、配对差值与空间稳定性

- **运行编号：** `phase_f_20261026_seed20261026`
- **分析模式：** `EXPLORATORY_ORACLE`；本报告不构成因果结论、独立验证或正式设计锁定。
- **输入运行：** `run_20261025T114741_470772Z`
- **研究总纲：** `docs/RESEARCH_CHARTER.md` V1.1，正文未修改。

## 1. 研究目的与数据边界

本阶段比较直接回归、环境调整、环境组斜率、土类内 K 均值匹配和土类内最近邻匹配。主要改进是以匹配对的高侵蚀端减低侵蚀端建立差值分析，按 5 km 空间块重复留出，并在同一训练/测试样本上比较加入侵蚀信息前后的预测变化。

本轮复用对齐数据库中的 `Aa_20yr` 与 `Aa_2020`，直接使用既有实际侵蚀值，不重算或验证 RUSLE 因子乘积，也不重跑 PCNM。健康指标来自 30 m 随机森林预测栅格，因此像元数不是独立实测样本数。既有 Oracle 搜索曾使用全研究区；本报告中的空间留出属于内部诊断，不是未触碰的外部验证。

## 2. 匹配设计与质量诊断

参考方案 M4 使用高程与地表温度（V5）、0.35 环境卡尺、1,000 m 地理距离上限、组内侵蚀百分位秩差 0.50，且不重复使用像元。M3 沿用相同变量和阈值，并在各土类内先做 K 均值 k=6。局地搜索在 `V1/V5/V7`、三种卡尺、三种地理半径、三种秩差与聚类方案中评估了 243 组结果盲候选；39 组进入多指标 Pareto 集。排序同时考虑覆盖率、环境平衡、匹配距离和侵蚀差异，没有读取健康结果。

土壤类型编码按原始类别属性表标注：**26—冲积土，27—潮土，42—脱潮土**。地图图例已同时显示代码和名称。

重复空间评估按 10 个固定种子将 5 km 整块分配为训练、验证和测试角色。环境标准化与 K 均值只在训练角色拟合；M3/M4 在各空间角色内分别建对。优先结局 Aa_20yr 的 M3 中位覆盖率为 0.505、最大绝对 SMD 中位数为 0.010；M4 分别为 0.483 和 0.012。两种方案均保留为并行探索候选，不仅按单一拟合指标选方案。

## 3. 优先指标的全样本配对差值

侵蚀差值定义为 `Δlog1p(Aa) = log1p(Aa高) − log1p(Aa低)`，健康差值定义为 `Δ健康 = 健康高 − 健康低`。下表为 M4、20 年平均实际侵蚀的全样本 P1 配对线性斜率；区间为模型区间，未处理残余空间相关。

| 指标 | 配对数 | 平均健康差 | 斜率 | 95% 区间下限 | 95% 区间上限 | 标准化斜率 | 表观 R² | 健康差为正比例 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 土壤健康指数（SHI） | 89,811 | -0.033 | -0.010 | -0.011 | -0.010 | -0.123 | 0.015 | 0.233 |
| 有机质（OM） | 90,867 | -2.854 | -0.526 | -0.571 | -0.482 | -0.077 | 0.006 | 0.258 |
| 土壤含水量（SW） | 90,867 | -6.062 | -1.965 | -2.026 | -1.903 | -0.204 | 0.042 | 0.142 |

三项指标的全样本差值斜率均为负，但拟合度有限。置信区间仅反映当前回归假设，不能视为已校正空间依赖的推断。其余六项健康指标及两种侵蚀时期的全部模型结果保留在配对结果表中。

## 4. 重复空间验证与侵蚀增量

M0–M2 使用两个侵蚀时期共有的完整像元；M3/M4 使用各自空间角色内独立构造的配对。10 次 5 km 留出结果显示，优先指标的匹配配对测试 R²整体靠近零；SW 相对较高，但 M3/M4 中位验证 R²约为 0.062 与 0.040。测试结果与全样本表观回归分开报告。

像元层“仅环境变量”与“环境变量加 `log1p(Aa)`”模型使用相同像元、模型族和空间划分。Aa_20yr 下 SHI、OM、SW 的测试 ΔR²中位数分别约为 0.172、0.170、0.308；Aa_2020 下分别约为 0.038、0.034、0.044。该结果表示在本设计下的预测增量，不代表侵蚀的独立因果贡献。配对差值层的增量远小于像元层，且单独记录。

## 5. 环境异质性与结果导向筛选

亚组比较涵盖土类、土类内 V5 K 均值组以及高程、地表温度分位组；交互检验按 Benjamini–Hochberg 方法校正。可估计的交互检验中，94 项校正后 q 值小于 0.05；这些检验属于探索性结果，仍需同时考察组内配对量、空间结构及选择偏差。

本次运行明确标记 `outcome_guided_selection=true`。另行开展的高响应环境组筛选按全样本健康均值排序，跨 18 个“结局×时期”组合评估 324 个候选组，保留 54 个高响应组；各组合的筛选覆盖比例为 0.067–0.317，中位数约 0.137。该结果只用于提出待检验候选，不能解释为独立确认的热点或机制。

## 6. 解释限制与阶段状态

- 预测栅格空间关联不等于独立实测样点关系或因果效应。
- 全样本配对回归的区间未校正空间相关；10 次空间留出是内部诊断。
- 匹配主变量排除了 R、K、LS、C、P，但仍可能存在其他环境混杂。
- 侵蚀栅格的物理单位和精确时间构造沿用既有资料登记；本阶段不自行推断。
- 既有 Oracle 结果导向模型、分组及筛选结果均保留其表观/筛选属性。
- 证据尚不足以只保留 M3 或 M4；两种方案并列进入后续方法审查。

## 7. 图集与可复核数据

| 图 | 中文主题 | 图源数据 |
| --- | --- | --- |
| [01](phase-f/figures/01_matching_coverage_balance.png) | 局地匹配覆盖率与环境平衡 | [CSV](phase-f/figures/data/01_matching_coverage_balance.csv) |
| [02](phase-f/figures/02_matching_method_pair_fit.png) | 匹配方法与侵蚀时期的配对拟合 | [CSV](phase-f/figures/data/02_matching_method_pair_fit.csv) |
| [03](phase-f/figures/03_primary_paired_slopes.png) | 优先指标配对差值斜率 | [CSV](phase-f/figures/data/03_primary_paired_slopes.csv) |
| [04](phase-f/figures/04_repeated_spatial_validation.png) | 5 km 整块重复空间留出 | [CSV](phase-f/figures/data/04_repeated_spatial_validation.csv) |
| [05](phase-f/figures/05_erosion_increment_delta_r2.png) | 环境变量之外的侵蚀预测增量 | [CSV](phase-f/figures/data/05_erosion_increment_delta_r2.csv) |
| [06](phase-f/figures/06_environmental_heterogeneity.png) | 不同环境背景下的关联差异 | [CSV](phase-f/figures/data/06_environmental_heterogeneity.csv) |
| [07](phase-f/figures/07_environment_groups_map_and_high_response.png) | 环境组空间分布与高响应组 | [CSV](phase-f/figures/data/07_environment_groups_map_and_high_response.csv) |
| [08](phase-f/figures/08_health_response_curves.png) | 九项指标的描述性响应曲线 | [CSV](phase-f/figures/data/08_health_response_curves.csv) |
| [09](phase-f/figures/09_pair_distance_and_exposure_diagnostics.png) | 环境距离、地理距离与侵蚀差 | [CSV](phase-f/figures/data/09_pair_distance_and_exposure_diagnostics.csv) |
| [10](phase-f/figures/10_exposure_period_comparison.png) | 两期侵蚀的共同像元比较 | [CSV](phase-f/figures/data/10_exposure_period_comparison.csv) |

完整表格位于 [`phase-f/tables/`](phase-f/tables/)。后续决定应结合研究总纲、数据来源核实与审查意见，不以单一模型表观 R²推进正式研究。
