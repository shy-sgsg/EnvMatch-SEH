# Phase K 20年侵蚀单情景审查包

## 建议阅读顺序

1. [阶段总结](PHASE_K_SUMMARY.md)：主线、四项发现、完成情况与限制。
2. [跨阶段结果索引](PHASE_FJ_RESULT_INDEX.md)：按研究问题连接F–J已有证据。
3. [双期材料清点表](results/DUAL_EROSION_ARTIFACT_INVENTORY.csv)：764个最终审查文件的暴露时期证据和20年单期配套位置。
4. [四项核心发现](results/FOUR_CORE_FINDINGS.csv)与[证据交叉表](results/PHASE_FJ_EVIDENCE_CROSSWALK.csv)。
5. [正式学术报告（Markdown）](reports/正式学术研究报告_20年侵蚀单情景.md)、[LaTeX源文件](reports/正式学术研究报告_20年侵蚀单情景.tex)和[PDF](reports/正式学术研究报告_20年侵蚀单情景.pdf)。
6. [精简研究报告](reports/精简研究报告.md)、[阶段技术报告](reports/阶段K技术报告.md)、[17页汇报提纲](reports/汇报提纲_17页.md)和[七分钟汇报稿](reports/七分钟汇报稿.md)。
7. [20年单情景图集](20年侵蚀单情景图集.html)：九张图均提供PNG、SVG、PDF和数据源表。
8. [全项目去重总图集](../../../FIGURE_ATLAS.html)：沿用已存在的全项目图集。
9. [只读审查请求](CHATGPT_REVIEW_REQUEST.md)：固定到交付结果的Git提交。

## 研究范围与数据处理

- 模式为`EXPLORATORY_ORACLE`，`outcome_guided_selection=true`；结论按探索性条件关联解释。
- 新增单期分析只把20年平均实际侵蚀作为暴露，不将另一期侵蚀用作暴露、协变量、分组、匹配或模型筛选字段。
- 直接使用项目提供的SHI与CEC图层，不重新计算、不替换。
- 复用已有的匹配、低值敏感性、GWR/MGWR、空间关联和SEM结果；本阶段不重新搜索参数或拟合全套模型。
- 预测像元有空间依赖，不等于同数量的独立土样或田块样本。

## 文件内容

- `results/`：双期材料清单、F–J证据交叉表和四项发现。
- `figures/`：A–I九张图及图源数据。
- `tables/`、`matching/`、`spatial/`、`sem/`：只含20年单期暴露的源表。
- `reports/`：正式学术报告、技术报告、精简报告、汇报材料与图集。
- `source/`：图表与数据包构建脚本。
- `provenance/`：分析配置、运行清单与来源说明。

## 审查状态

固定提交哈希、审查请求和推送状态列在`CHATGPT_REVIEW_REQUEST.md`。本包没有代表用户向外发送审查材料。
