请首先读取本次提交对应版本的 `docs/RESEARCH_CHARTER.md`，全面理解EnvMatch-SEH的研究主题、目标、科学问题及长期技术路线，再审查本阶段的低侵蚀群体识别、阈值稳定性、环境匹配重分析、健康响应模型、空间验证和科研结论。

# ChatGPT 复核请求：Phase H 低侵蚀群体与环境匹配响应

## 固定审查版本

- 仓库：EnvMatch-SEH
- 审查提交 SHA：`[请在本地提交后填入完整 SHA]`
- 本地工作区目前尚未提交。请先按下文命令提交并推送，随后把提交 SHA 写入此处；最终审查应针对该 SHA 中的章程、报告、代码、配置和图表。
- 本审查包由用户自行交给 ChatGPT；本项目未代用户向外发送材料。

## 研究与分析范围

- 研究章程：`docs/RESEARCH_CHARTER.md`
- 阶段索引：`docs/reviews/2026-10-29/phase-h/PHASE_H_RESULTS_INDEX.md`
- 20 个问题的综合回答：`docs/reviews/2026-10-29/phase-h/PHASE_H_FINAL_SUMMARY.md`
- 主报告：`docs/reviews/2026-10-29/phase-h/LOW_EROSION_REGIME_RESEARCH_REPORT.md`
- 全项目图集：`docs/reviews/2026-10-29/phase-h/FIGURE_ATLAS.html`
- 随仓库归档的图表、图注、图源表、运行配置与清单：`docs/reviews/2026-10-29/phase-h/artifacts/`（索引见 `artifact_manifest.json`）
- 低侵蚀主报告和阈值/响应摘要：`LOW_EROSION_REGIME_RESEARCH_REPORT.md` 与 `artifacts/low_erosion_regime/`
- 分位数、多随机种子 GMM、后验阈值敏感性及统一阈值候选补充：`artifacts/distribution_diagnostic_addendum/`
- 匹配端点与配对响应补充报告：`artifacts/matched_units/reports/MATCHED_UNIT_SUPPLEMENT_REPORT.md`
- 环境组异质性与空间验证报告：`artifacts/environment_group_heterogeneity/reports/ENVIRONMENT_GROUP_HETEROGENEITY_REPORT.md`
- 完整栅格、Parquet 和大体积逐像元中间结果仍在本地 `outputs/exploratory_oracle/` 运行目录，不属于 Git 审查包。

本次低侵蚀诊断、匹配单位、环境组异质性和分布补充运行均处于 `EXPLORATORY_ORACLE` 模式，明确记录 `outcome_guided_selection=true`。新增分布补充运行同时记录 `outcome_guided_selection_used=false`，因为它只使用既有 Aa 与分布诊断，没有读取健康结局。两期实际侵蚀 Aa 直接来自既有对齐数据库；没有重算或验证 RUSLE 因子乘积，也没有重跑 PCNM。健康数据为预测栅格值；匹配像元、配对和空间块分别报告，不能视为独立实测样本。

主分析用经验 CDF 单折点形成分布候选阈值：20 年平均 Aa=5.05018758774，2020 年 Aa=0.3966915309429168（报告中按 6 位小数显示为 0.396692；像元计数用完整精度）。100 次空间分块重抽样均识别两期 CDF 折点。KDE 未找到可靠密度谷；GMM 多种子结果显示 Aa_20yr 的 BIC 选中 K=3 为 9/10、K=2 为 1/10，Aa_2020 虽 10/10 选中 K=4，但首个有限分界 CV 为 72.2%，且后验不确定比例在 0.80 阈值下为 54.1%；Otsu 仅作分箱敏感性参考。另比较了原有配对双端过滤、过滤后固定原环境组重配、过滤后重拟合 k=6 环境组重配、匹配端点像元与配对差值响应、两段式暴露模型和 5 km 整块空间留出。

土壤类型代码按原始类别属性表标注：26—冲积土、27—潮土、42—脱潮土。

## 请重点审查

1. 是否区分有效零值、正的近零值和 NoData；有效范围、阈值筛选端点规则及面积分母是否一致。
2. KDE、GMM、经验 CDF 折点与 Otsu 的判据、参数和失败状态是否透明；没有可靠分界时是否避免强行给出硬分类。
3. CDF 阈值空间块重抽样的实现、样本构成及稳定性解释是否恰当；GMM 10 个随机种子的 K 选择、首个有限边界波动、分量重叠和 0.80/0.90/0.95 后验不确定性是否被充分考虑。
4. 原有 M3/M4 配对过滤与筛选后重配是否分开；健康结果是否影响了哪些 Oracle 选择，以及这些选择是否完整标记。
5. 匹配端点像元、配对差值、环境组内配对、空间块斜率区间与模型 R²是否分别对应正确的统计单位。
6. 九项健康响应的方向、模型形式、阈值敏感性和表观 R²是否按时期、筛选方案和匹配单位完整呈现；空间测试是否与表观择优隔离。
7. 26 冲积土、27 潮土、42 脱潮土，以及图集中的环境组标签、图例、图注、PNG/SVG/PDF 和源数据是否清晰一致。
8. 低 TWI 的 26 冲积土候选组与 Phase G 原优势组对照能否支持当前解释；是否需要进一步限定多重探索和小样本组的结论强度。
9. 建议是否与结果相符：全样本两段式/非线性模型作为主分析，CDF 阈值作为敏感性方案，且不把阈值视作数据质量错误线。

请先列出会改变科学结论的实质性方法问题，再列图表、措辞或文档方面的次要修改。请勿把 Oracle 表观拟合、预测栅格像元数量或内部空间留出表述为因果关系、独立实测验证或外部验证。

## 用户本地提交后如何固定 SHA

在仓库目录运行：

```bash
git add docs config experiments scripts tests paper
git add docs/reviews/2026-10-29/phase-h
git commit -m "Complete Phase H low-erosion analysis and figure atlas"
git push origin main
git rev-parse HEAD
```

将最后一条命令得到的完整 SHA 填到本文件“固定审查版本”处后，再把该提交和审查包交给 ChatGPT。大型栅格和 Parquet 位于忽略的 outputs 目录；Git 中保留代码、配置、报告、图件、压缩源表、清单和测试。
