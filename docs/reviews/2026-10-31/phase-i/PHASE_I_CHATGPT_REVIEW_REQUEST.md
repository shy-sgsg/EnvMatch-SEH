# Phase I 阶段科研审查请求（EXPLORATORY_ORACLE）

请先读取审查目标提交 `8da02898c56366abfbca3661aaa693e21247b76a` 中的 `docs/RESEARCH_CHARTER.md`，再根据同一提交中的代码、配置和本目录归档的结果材料进行审查。该审查目标是已推送的 Phase I 源码提交；本审查包在后续提交中提供，不能把审查包提交误认成模型代码版本。

## 1. 项目与版本

- 仓库：`https://github.com/shy-sgsg/EnvMatch-SEH`
- 审查提交 SHA：`8da02898c56366abfbca3661aaa693e21247b76a`
- 研究总纲：`docs/RESEARCH_CHARTER.md`（审查提交：`8da02898c56366abfbca3661aaa693e21247b76a`）
- 阶段：Phase I-1 至 I-9，`EXPLORATORY_ORACLE`，不是 FORMAL 阶段。

## 2. 总体研究定位与科学问题

本阶段推动总纲中的五个问题：环境背景相近时侵蚀与健康是否仍有关联；环境背景如何改变关联；健康指标维度响应有何差异；EnvMatch与既有PCNM结果是否一致；如何改善匹配、可比性与推断可靠性。EnvMatch 是新增分析模块，不能被解读为已替代既有 PCNM 框架。

## 3. 本阶段工作

- 目标与任务：全掩膜基线、环境匹配配对、环境异质性、GWR/LCR与空间变化近似、局部机器学习、空间关联与热点、健康维度衔接、候选胁迫分区、核心图件闭合及标准 GWR/MGWR 补充。
- 模式：所有运行标记 `EXPLORATORY_ORACLE` 和 `outcome_guided_selection=true`。响应参与结果导向的候选模型、环境分组或阈值探索；请审查各运行的搜索记录、验证/测试隔离和样本保留披露，不将 Oracle 最优表现视作独立验证。
- 数据：已计算 Aa_20yr；SHI 与八项健康指标为随机森林预测栅格。分析单元包括像元、匹配对和约240m空间单元，不能互换为独立实测样本。
- 方法和配置：代码在 `experiments/oracle_exploration/phase_i*.py`；配置在 `config/phase_i*.yaml`。主报告为 [`PHASE_I_FINAL_RESEARCH_REPORT.md`](PHASE_I_FINAL_RESEARCH_REPORT.md)，MGWR补充为 [`PHASE_I_CLOSURE_REPORT.md`](PHASE_I_CLOSURE_REPORT.md)，分阶段报告位于 [`reports/stages/`](reports/stages/)。
- 审查证据：[`PHASE_I_CORE_FIGURE_ATLAS.md`](PHASE_I_CORE_FIGURE_ATLAS.md)、[`tables/core_figure_completeness.csv`](tables/core_figure_completeness.csv)、`tables/`下的选择、匹配、异质性、空间模型、MGWR和分区摘要表，以及`provenance/`下的运行清单。

## 4. 数据与方法质量

请按具体文件和图表核查数据支持、缺失/异常、掩膜与空间支持、模型筛选、区间和统计单位。**原始栅格、对齐数据库和大型像元级结果未纳入此 Git 审查包**；仓库 `.gitignore` 将 `outputs/**`、原始/中间/处理数据排除。因此本包可供审查代码、配置、阶段报告、结果摘要表和48张PNG图，但无法从远端独立重算像元分析或完整核查每个源像元。请将这点作为可复核性限制，不推断未提供的值。

关键边界：Aa单位/准确时间口径仍待核实；模型健康响应是预测面；mgwr 2.2.1 的多带宽 MGWR 局部 R²未实现，局部字段留空；同样本 MGWR 为n=770空间均衡单元，不能等同I-4全量像元支持加权GWR。

## 5. 结果与研究价值

- 全样本与匹配配对中，SHI、OM、SW均有负向关联线索，但匹配差值估计较弱，且各估计单位不同。
- I-C标准MGWR对SHI、OM、SW的模型级R²为0.869、0.696、0.822；同样本标准GWR为0.887、0.714、0.834。MGWR Aa邻居带宽分别32、767、767，后两项接近样本搜索上界；不据此推断因果或正式空间尺度。
- 核心图31项状态：24项可用、5项部分满足、2项延期。图5只有MBC的分段候选；图12是源表 `r_squared_apparent`，不是最佳模型选择结果。
- PCNM结果表与PLS-SEM路径/载荷源表未提供。图30、31是缺口标识；交叉值为空，没有重跑PCNM或推断PLS路径，跨方法一致性和机制结论暂缓。

请进一步对照主报告、阶段报告及表格，确认搜索次数/目标、未筛选基线、样本保留和空间验证结果是否足够透明；区分事实、推断与建议。

## 6. 方向判断与下一步

- 判断现有 EnvMatch 路线是否值得作为 PCNM 的补充模块继续推进，并说明证据与停止条件。
- 最小补充输入：既有PCNM结果表（指标、情景、系数、区间/标准误、P值、R²、样本量、控制项）和PLS-SEM路径/载荷/方向/控制变量源表；在获取前不得填造比较。
- 后续正式证据应使用独立实测样点并锁定 FORMAL 设计；不得把当前 Oracle 搜索参数直接当作预注册结论。

## 7. 总纲审查

**NO CHANGE。** 当前审查请求不授权修改 `docs/RESEARCH_CHARTER.md`。若确需调整，另列 `CHANGE PROPOSAL — USER APPROVAL REQUIRED`，等待用户明确批准。
