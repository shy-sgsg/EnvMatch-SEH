# 全项目学术报告扩写实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans or superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将温县土壤侵蚀—土壤健康报告扩展为学术论文体例的长篇报告，系统解释全项目100幅以上有效图件，并保持CEC数据口径准确。

**Architecture:** 以 `docs/PROJECT_FULL_REPORT.md` 为唯一正文来源，章节按研究问题与证据顺序重构；图件附录复用项目阶段图像和已保存表格，为每幅图增加独立解读。完成文本与图件后，再同步生成LaTeX、PDF、DOCX、阶段审查包和离线图集。

**Tech Stack:** Markdown、现有Python图件脚本、Pandoc、XeLaTeX、DOCX生成与渲染工具。

**Spec:** `docs/superpowers/specs/2026-11-06-expand-academic-report-design.md`

## Global Constraints

- 主暴露为20年平均实际侵蚀。
- 除CEC外，主分析使用Phase K按该暴露整理的已保存结果；直接CEC结果使用Phase I校正重分析及Phase L源表。
- `CEC_corrected = (CEC - 20 × Erosion)/2.9 + 18`，其中 `Erosion = Aa_20yr/max(Aa_20yr) + Aa_2020/max(Aa_2020)`，最大值取完整对齐栅格的全域最大值。
- CEC重建公式含侵蚀项；不能把校正后CEC与侵蚀的关联写成独立测得的自然侵蚀效应。
- SHI沿用当前项目提供的 `SHI_candidate`，不声称它随CEC校正重新计算。
- 不重跑Phase F–K模型，不改变研究总纲，不覆盖或删除历史阶段结果。
- 两章正文与逐图解释合计至少50,000个汉字；附录逐幅解释至少100幅有效图。
- 结果属于 `EXPLORATORY_ORACLE` 已保存分析，不改写为正式因果结论。
- 删除重复自我辩护、项目过程叙述和幼儿化类比；保留简洁、准确的论文必要数据与方法说明。

## Review Focus

- **过时CEC：** 组合图或九指标图若仍含旧CEC，需替换为Phase L当前图/数据或不纳入新报告；逐图核对图、表和解读。
- **CEC公式内嵌侵蚀项：** 变量定义及CEC结果解释必须呈现公式依赖；全篇检索不允许把该关联表述成独立生态效应。
- **SHI组成：** 始终标记为既有 `SHI_candidate`；不得写成已根据校正CEC重算的指数。
- **分析总体与时段：** 图表说明其对应全域像元、匹配端点/配对、局部单元或空间块，并准确区分20年侵蚀与历史双期诊断。
- **图件可用性：** 附录图像路径均须存在且清晰；不纳入占位图和已被替代的CEC图，最终合格图数至少100。

---

### Task 1: 固定结果口径并筛选项目图件

**Files:**
- Read: `docs/reviews/2026-11-05/phase-k/PHASE_FJ_RESULT_INDEX.md`
- Read: `docs/reviews/2026-11-05/phase-k/PHASE_K_SUMMARY.md`
- Read: `docs/reviews/2026-10-31/phase-i/cec-correction/CEC_CORRECTION_REANALYSIS.md`
- Read: `docs/reviews/2026-11-05/phase-l/FINAL_REPORT_EVIDENCE_CROSSWALK.csv`
- Modify: `docs/reviews/2026-11-05/phase-l/FINAL_REPORT_EVIDENCE_CROSSWALK.csv`
- Modify: `docs/reviews/2026-11-05/phase-l/FINAL_REPORT_TABLE_INDEX.md`

- [x] 逐项确认核心结论和表格所用来源：非CEC值取Phase K单期表，CEC直接结果取Phase I/Phase L当前源表。
- [x] 从项目180项图件目录形成报告纳入清单；逐图记录阶段、题名、研究问题、统计单位、CEC状态和纳入位置。
- [x] 排除旧CEC版本图、占位图和无法与已保存数据对应的图；同一研究问题的版本比较放入附录并清楚标明差异。
- [x] 确认纳入图件不少于100幅；若不足，基于已保存的当前表重绘必要图，不重新拟合Phase F–K模型。

**Check:** 清单中每张图路径存在；CEC相关图均指向当前CEC版本；纳入图数≥100；交叉索引中的核心数值可回到指定源表。

### Task 2: 重写第一章——研究问题、理论框架与方法体系

**Files:**
- Modify: `docs/PROJECT_FULL_REPORT.md`
- Modify: `docs/reviews/2026-11-05/phase-l/CHAPTER_1_RESEARCH_DESIGN_METHODS_EXPLORATION.md`

- [ ] 以研究背景、科学问题、研究区和数据结构、变量体系、分析设计、空间统计方法的顺序重组章节。
- [ ] 增写环境匹配、零值处理、非线性模型、异质性分析、GWR/MGWR、空间机器学习和SEM的具体模型设定及参数解释。
- [ ] 用学术中文解释统计量、空间单元和变量变换；删除重复的过程审计、口语类比和防御性段落。
- [ ] 在CEC变量定义处完整给出校正公式、全域归一化规则及该构造对CEC统计解释的影响。
- [ ] 核对SHI的表述始终对应既有 `SHI_candidate`，不写成CEC校正后的新指数。

**Check:** 第一章与研究总纲Q1–Q5相连；方法可按文中定义复述；无未注明的数据定义或公式；语言为正式学术行文。

### Task 3: 扩写第二章——全域、匹配、非线性、异质性、空间格局与路径

**Files:**
- Modify: `docs/PROJECT_FULL_REPORT.md`
- Modify: `docs/reviews/2026-11-05/phase-l/CHAPTER_2_RESULTS_SPATIAL_SEM_SYNTHESIS.md`
- Read: `docs/reviews/2026-11-05/phase-k/reports/PHASE_K_CORE_SCIENTIFIC_FINDINGS.md`
- Read: `docs/reviews/2026-11-05/phase-k/reports/PHASE_K_MATCHING_NONLINEAR_SYNTHESIS.md`
- Read: `docs/reviews/2026-11-05/phase-k/reports/PHASE_K_GWR_MGWR_SPATIAL_SYNTHESIS.md`
- Read: `docs/reviews/2026-11-05/phase-k/reports/PHASE_K_SEM_SYNTHESIS_REPORT.md`
- Read: `docs/reviews/2026-11-05/phase-k/reports/PHASE_K_SPATIAL_STRESS_ZONES_REPORT.md`

- [ ] 按“总体关联—环境匹配—响应曲线—土类/环境异质性—局部空间—空间机器学习—SEM路径—综合讨论”展开结果。
- [ ] 为九项健康结果分别说明方向、效应量、样本支持和相互差别；重点细化SHI、SW、OM并完整解释CEC。
- [ ] 逐项补足匹配设计、曲线形态、局部系数分布、空间集聚、模型解释增量和候选区的数字化结果。
- [ ] 将阶段结果放入同一研究主线，区分不同统计单位和探索目标；不重复展示同一数字作为多项独立发现。
- [ ] 形成研究讨论、方法贡献、适用范围和结论，提出与结果直接相关的科学解释与后续实测问题。

**Check:** 正文所有数字与核准源表一致；CEC仅按公式结果解释；九项指标方向差异均有呈现；讨论中的机制解释由已保存结果或可核验文献支撑。

### Task 4: 编写100幅以上图件的附录解读并更新图集

**Files:**
- Modify: `docs/PROJECT_FULL_REPORT.md`
- Modify: `docs/FIGURE_ATLAS.html`
- Modify: `docs/FIGURE_ATLAS_INDEX.md`
- Modify: `docs/reviews/2026-11-05/phase-l/FINAL_REPORT_FIGURE_ATLAS.html`
- Modify: `docs/reviews/2026-11-05/phase-l/README.md`
- Create or update: `docs/reviews/2026-11-05/phase-l/FIGURE_ANNOTATION_INDEX.csv`

- [ ] 每幅纳入图依次说明研究问题、图面编码/坐标含义、主要结果和其在主线中的位置。
- [ ] 按Phase F、G、H、I、J、K及CEC重分析分组，正文图和附录图分别编号并可互相查找。
- [ ] 更新离线图集的中文题名、图注、说明和链接；旧CEC图只留在既有历史归档，不出现在新报告图集。
- [ ] 统一图名中的CEC标签；在图注/图后说明公式和结果含义，不把模型箭头或地图分类写成因果结论。

**Check:** 图件索引和正文引用一一对应；新图集只展示纳入清单；旧CEC图与占位图未出现；附录有≥100条独立图解且图像全部可打开。

### Task 5: 增补可核验学术文献并统一引文

**Files:**
- Modify: `docs/PROJECT_FULL_REPORT.md`
- Modify: `docs/PROJECT_FULL_REPORT.tex`
- Create or update: `docs/reviews/2026-11-05/phase-l/REFERENCES.md`

- [ ] 为土壤侵蚀、土壤健康评价、空间匹配、空间分块验证、GWR/LISA和SEM等方法补充可核验的学术来源。
- [ ] 优先引用原始研究和方法论文；逐条核对作者、年份、题名、期刊和DOI。
- [ ] 在正文相应论述处插入作者—年份引文，确保参考文献表与文内引文双向对应。

**Check:** 每条引文能打开到直接来源；无孤立参考文献、无正文未列文献、无来源未核实的创新性判断。

### Task 6: 统一生成交付格式并完成文档质量核对

**Files:**
- Modify: `docs/PROJECT_FULL_REPORT.md`
- Modify: `docs/PROJECT_FULL_REPORT.tex`
- Modify: `docs/PROJECT_FULL_REPORT.pdf`
- Modify: `docs/PROJECT_FULL_REPORT.docx`
- Rename `docs/reviews/2026-11-05/phase-l/FINAL_ACADEMIC_RESEARCH_REPORT_30000PLUS.md` to `docs/reviews/2026-11-05/phase-l/FINAL_ACADEMIC_RESEARCH_REPORT_50000PLUS.md`
- Rename `docs/reviews/2026-11-05/phase-l/FINAL_ACADEMIC_RESEARCH_REPORT_30000PLUS.pdf` to `docs/reviews/2026-11-05/phase-l/FINAL_ACADEMIC_RESEARCH_REPORT_50000PLUS.pdf`
- Rename `docs/reviews/2026-11-05/phase-l/FINAL_ACADEMIC_RESEARCH_REPORT_30000PLUS.docx` to `docs/reviews/2026-11-05/phase-l/FINAL_ACADEMIC_RESEARCH_REPORT_50000PLUS.docx`
- Modify: `docs/reviews/2026-11-05/phase-l/PHASE_L_CHATGPT_REVIEW_REQUEST.md`
- Modify: `README.md`
- Modify: `docs/reviews/2026-11-05/phase-l/FINAL_REPORT_DETAILED_OUTLINE.md`
- Modify: `docs/reviews/2026-11-05/phase-l/WORD_COUNT_VERIFICATION.md`
- Modify: `docs/reviews/2026-11-05/phase-l/DOCUMENT_QA.md`

- [ ] 从统一Markdown源生成LaTeX、PDF和DOCX；保持根目录和Phase L交付副本一致，并沿用硬链接避免无谓复制大文件。
- [ ] 将旧的 `30000PLUS` 交付名更新为 `50000PLUS`，同步修正Phase L README及根README中的链接和图数说明。
- [ ] 分别统计两章正文、逐图解读和总汉字数；确认两章正文及逐图解读合计≥50,000汉字，图解≥100幅。
- [ ] 检查CEC公式、图表引用、文内引文、内部链接和章节编号；确认旧CEC图未混入交付图集。
- [ ] 渲染PDF/DOCX并检查目录、长表、中文字体、图片分页和附录；对发现的版面问题修正后重新渲染。
- [ ] 更新篇幅、图件数量、交付目录和文档质量说明；不运行软件测试套件。
- [ ] 提交后把审查请求固定到最终报告提交；按之前已授权的仓库推送要求更新 `origin/main`。

**Check:** Markdown、LaTeX、PDF、DOCX的标题/正文/图件数量一致；PDF与DOCX可打开，中文字形完整，正文篇幅与图件门槛达到要求。
