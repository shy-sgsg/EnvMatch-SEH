# ChatGPT 阶段科研审查请求

请先读取本次审查指定提交中的 `docs/RESEARCH_CHARTER.md`，再按该提交中的代码、配置、阶段材料和结果审查本项目。请区分文件可确认的事实、根据结果得出的推断和建议；优先指出具体表格、图、统计量和方法限制。不要把模型运行成功或图表齐全当作研究结论成立。

## 1. 项目与版本

- 仓库：`https://github.com/shy-sgsg/EnvMatch-SEH`
- 审查基准提交 SHA：`37d09bd42684038888c4e483fabe285e91d3fcf2`
- 研究总纲：`docs/RESEARCH_CHARTER.md`（须在上述提交中读取）
- 阶段：Phase K，探索性结果整理与20年平均实际侵蚀单情景汇总，不是正式阶段门通过。
- 主审查入口：`docs/reviews/2026-11-05/phase-k/README.md`

## 2. 总体研究定位与科学问题

项目研究温县耕地中土壤侵蚀和土壤健康在环境相近位置是否仍有关联、这种关联是否因环境和空间位置而不同，以及环境匹配结果与既有空间控制结果是否相符。本阶段主要整合证据以回应总纲Q1–Q5：环境条件关联、空间异质性、不同健康指标响应、与既有PCNM结果的关系，以及设计质量和推断局限。

本阶段只整理并提供以20年平均实际侵蚀为暴露的独立报告、表格和图件。研究模式为 `EXPLORATORY_ORACLE`，`outcome_guided_selection=true`。Oracle允许结果导向探索；相关结果属于探索性条件关联，不是独立验证或因果证据。Phase K没有新增全套匹配、阈值、空间带宽或路径结构搜索。

## 3. 本阶段工作

- **目标：** 将F–J证据串成可审查的研究主线，登记双期旧材料的暴露时期，提供只用20年平均侵蚀作暴露的表格、九图、正式学术报告及汇报材料。
- **数据：** 使用项目已有侵蚀栅格、用户指定的SHI和CEC图层、健康预测栅格、环境栅格及既有分析表。SHI和CEC按现有输入直接使用，不在本阶段重算或替换。30米像元是预测栅格单元，不等于独立采集的土样。
- **方法：** 从已保存结果提取20年单期记录，并从单期源表重新绘制A–I九图；复用环境匹配、全域相关、GAM、GWR/MGWR、空间关联和SEM结果，不重新搜索模型或带宽。运行和输入清单见 `provenance/analysis_config.json`、`provenance/20yr_only_provenance.json` 与 `provenance/run_manifest.json`。
- **软件：** Phase J清单记录Python 3.13.5、NumPy 2.1.3、pandas 2.2.3、SciPy 1.15.3、statsmodels 0.14.4、scikit-learn 1.6.1及R lavaan 0.7.3；Phase K脚本为 `source/phase_k_build_bundle.py`。
- **结果入口：** `PHASE_K_SUMMARY.md`、`PHASE_FJ_RESULT_INDEX.md`、`results/`、`tables/`、`matching/`、`spatial/`、`sem/`、`figures/`、`reports/`。
- **相关历史材料：** `docs/reviews/2026-11-02/phase-j/`。Phase F–I文件级清单和证据链接见 `results/DUAL_EROSION_ARTIFACT_INVENTORY.csv`、`results/PHASE_FJ_EVIDENCE_CROSSWALK.csv`。部分早期机器运行输出位于Git忽略的 `outputs/`，审查时请以已归档的阶段材料为证据，并指出无法从提交本身独立复现的部分。

## 4. 数据与方法质量

请针对下列证据给出PASS、WARNING或BLOCKED，并补充事实、推断及建议：

| 核查项 | 可定位证据与已知情况 | 审查重点 |
|---|---|---|
| 单期暴露隔离 | `provenance/20yr_only_provenance.json`；20年单期CSV的字段核查；`results/DUAL_EROSION_ARTIFACT_INVENTORY.csv`；`results/DUAL_TABLE_20YR_SLICE_MANIFEST.csv` | 新表没有另一时期暴露字段；但需区分“模型变量未使用另一时期”与输入数据本身是否由跨期信息构成。 |
| 健康指标使用 | `provenance/analysis_config.json` | SHI和CEC按用户指定的项目图层直接使用；图表统一写‘CEC’。审查本阶段结论的统计单位、指标方向、数据依赖与解释边界。 |
| 样本和空间单位 | `tables/global_associations_20yr.csv`、`matching/matched_pair_associations_20yr.csv`、`sem/sem_model_rankings_20yr.csv` | 全域各指标有效像元约388,000；更新匹配候选111,289对；主路径模型完整案例293,145像元、35个5 km空间块。像元、配对端点、240 m单元和空间块不得混成同一独立样本量。 |
| 结果导向选择 | Phase J `provenance/analysis_config.json`、`run_manifest.json`、`method_status.json`；Phase F–J证据交叉表 | Phase J目标是汇总F–I并探索路径结构、健康维度和环境异质性；清单记录CB-SEM 48个模型、1,390条路径，PLS另记录64及176组方程。请检查这些搜索范围、响应指标、对照基线和报告的选择过程是否完整，是否把表观最优误当确认性证据。Phase K本身未增加新模型搜索。 |
| 未筛选基线和敏感性 | `reports/正式学术研究报告_20年侵蚀单情景.md`、`results/FOUR_CORE_FINDINGS.csv`；Phase J `reports/PHASE_F_TO_I_SCIENTIFIC_FINDINGS.md` | 全域Spearman：SW −0.584、SHI −0.469、OM −0.415、CEC +0.168；匹配差值：SW −0.272、SHI −0.144、OM −0.115。零值约占三分之一，经验折点以下（含零）约46.6%；CDF敏感性筛选后部分关联变弱，空间留出结果未一致改善。不同样本支持间的系数不可直接相减。 |
| 匹配设计诊断 | Phase F–J证据交叉表；`reports/阶段K技术报告.md` | 历史10个空间留出种子的匹配覆盖率中位数约0.483，最大绝对标准化均差中位数约0.012。这是设计诊断，不是独立实测验证；未测量管理因素仍可能造成混杂。 |
| 空间结果 | `spatial/nine_outcome_local_beta_maps_20yr.csv`、`spatial/gwr_model_summary_20yr.csv`、图B/C及图集 | 局部β的负向占比不是显著性比例；GWR、GWR-LCR和MGWR的地点排序不完全一致。候选区是实测抽样线索，不是正式风险图或治理边界。 |
| 报告、图件和复现 | `figures/FIGURE_MANIFEST.csv`、图A–I九张核心图和补充图J的PNG/SVG/PDF及源表；`reports/` | 九张核心图与补充图J是否与单期表一致，结论是否被表格支持，字体、中文标签、图集链接和报告PDF版面是否合格；从提交能否复现整理步骤及其限制。 |
| 长期研究衔接 | `PHASE_FJ_RESULT_INDEX.md`、`docs/reviews/2026-11-02/phase-j/` | 既有PCNM结果继续保留，本阶段未重跑或替代；请评价未来比较是否需要预先锁定的统一样本和空间验证。 |

## 5. 结果与研究价值

20年单期材料显示，全域上土壤含水量、SHI和有机质与侵蚀的负向秩相关相对较强；环境匹配后这三项在配对差值中仍偏负。局部模型显示方向总体偏负但强度因位置和模型而异。控制环境的主路径中，侵蚀到SHI的标准化路径约−0.113，模型R²约0.601，加入侵蚀后的条件ΔR²约0.0079。它支持探索若干稳定关联，不足以证明侵蚀造成健康下降。

请判断：这些不同统计单位和方法是否被恰当综合；高R²是否被正确限定；各指标的解释边界是否写清；当前预测栅格能否支持更强推断；本阶段是否确实推动Q1–Q5。请区分已确认结果、可能解释和需要新增数据才能验证的说法。

## 6. 方向判断与下一步

请判断是否建议继续环境匹配路线。建议审查的最小下一步是：固定独立实测的样点/地块与空间块方案；同时记录侵蚀、采样深度、土类、作物和灌排耕作信息；预先确定环境匹配、主要结局和空间留出方式；随后在同一实测支持上对照现有PCNM结果。请给出可验收的停止条件，特别是匹配重叠不足、空间留出不稳定或实测方向不一致时如何处理。

## 7. 总纲审查

`NO CHANGE`。本阶段未申请修改研究总纲；如审查者认为需要改方向，请另列证据和建议文本并标注 `CHANGE PROPOSAL — USER APPROVAL REQUIRED`，未经用户批准不得修改总纲。

本请求仅供只读审查使用；代理未向外部发送审查材料。