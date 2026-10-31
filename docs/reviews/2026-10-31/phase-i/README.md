# Phase I 审查包

本包让审查者能从 Git 仓库直接打开 Phase I 综合报告、各阶段报告、48张图件PNG、31项核心图审计和关键结果摘要表。

- **源码审查目标提交**：`8da02898c56366abfbca3661aaa693e21247b76a`。
- **审查请求**：[`PHASE_I_CHATGPT_REVIEW_REQUEST.md`](PHASE_I_CHATGPT_REVIEW_REQUEST.md)。请按其中指令先读该目标提交中的研究总纲。
- **综合报告**：[`PHASE_I_FINAL_RESEARCH_REPORT.md`](PHASE_I_FINAL_RESEARCH_REPORT.md)。
- **MGWR闭环报告**：[`PHASE_I_CLOSURE_REPORT.md`](PHASE_I_CLOSURE_REPORT.md)。
- **图件索引**：[`PHASE_I_CORE_FIGURE_ATLAS.md`](PHASE_I_CORE_FIGURE_ATLAS.md)。
- **核心图状态**：[`tables/core_figure_completeness.csv`](tables/core_figure_completeness.csv)，24项可用、5项部分满足、2项因源表缺失延期。

## 可复核边界

这是一份为远端审查整理的只读证据副本，不改变原始 `outputs/**`。原始/对齐栅格、输入数据库和大型像元级文件仍遵守 `.gitignore` 排除规则，因此本包不支持从原始像元完整重算。分析脚本和配置已包含在源码审查目标提交；本包保留阶段摘要、选择记录、关键统计表、运行provenance及图件预览。

I-7缺PCNM源表和PLS-SEM路径/载荷源表。图30、31仅显示来源缺口；审查包不填造对应结果。审查请求没有发送给外部人员或服务。
