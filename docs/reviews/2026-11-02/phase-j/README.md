# Phase J 综合分析审查包（r10）

本包以 `outputs/exploratory_oracle/phase_j/phase_j_20261102_seed20261102_r10/` 为Phase J综合主运行；另有不覆盖r10的J-9方向评分补充运行 `phase_j_20261103_seed20261103_direction_r1/`。r9和r4诊断包只在本地保留，不纳入本Git审查包；其他早期运行目录亦保留，但不作为最终材料。r10在早期只读复核后补入了A–C综合图、成熟 lavaan CB-SEM、PLS S1–S3×M0–M3全矩阵和D1–D4逐总体候选比较；这些增补尚未经过独立复核。

## 从这里开始

- [正式学术研究报告（Markdown）](reports/ENV_MATCH_SOIL_HEALTH_ACADEMIC_REPORT.md)面向论文和学术汇报；同一报告另提供[LaTeX源文件](reports/ENV_MATCH_SOIL_HEALTH_ACADEMIC_REPORT.tex)和[正式排版PDF](reports/ENV_MATCH_SOIL_HEALTH_ACADEMIC_REPORT.pdf)。[完整技术报告](reports/PHASE_J_FINAL_TECHNICAL_REPORT.md)保留方法、变量代码、运行及可追溯结果。
- [十个核心研究问题的回答](reports/PHASE_J_RESEARCH_QUESTIONS_ANSWERS.md)汇总全域、匹配、空间与路径模型证据。
- [Phase F–J研究结果总索引](research_index/RESEARCH_RESULTS_MASTER_INDEX.md)、[跨阶段证据交叉表](research_index/RESEARCH_EVIDENCE_CROSSWALK.csv)和[Phase F–I科学发现](reports/PHASE_F_TO_I_SCIENTIFIC_FINDINGS.md)按问题汇总前序证据。
- [Phase J 26图图集](reports/PHASE_J_RESEARCH_FIGURE_ATLAS.html)浏览A–Z中文图件；其中U–V为补充空间地图，W–Z为四种SEM结构路径图。[全项目总图集](../../../FIGURE_ATLAS.html)汇集项目各阶段全部去重图卡。
- [J-9健康方向评分补充](J9_HEALTH_DIRECTION_RECONSTRUCTION.md)记录本地章节草稿的方向、隶属函数和权重，以及按其规则重算SHI与既有图层不一致的核对结果。
- `reports/PHASE_J_CORE_FIGURE_INDEX.md`和`reports/PHASE_J_FIGURE_CAPTIONS.md`提供逐图说明。
- `results/`、`sem/`、`spatial_synthesis/`保存结果表；本地`data/`、`inputs/`保留逐像元分析总体、匹配来源和栅格；`provenance/`保存公式、输入、运行及方法记录。
- [只读复核记录与本次完成更新](PHASE_J_INDEPENDENT_REVIEW.md)区分早期复核覆盖范围和本次新增内容；[当前只读审查请求](PHASE_J_REVIEW_REQUEST.md)供后续复核使用。

## 2026-11-03 J-2局部空间补充

Phase I保存的GWR、GWR-LCR与MGWR局部系数已整理分位数、负系数单元比例、跨模型局部系数一致度、三项重点结局强负向位置重叠、六类环境背景分组及GWR/MGWR估计差异位置。完整统计表位于spatial_synthesis目录，解释见reports/PHASE_J_SPATIAL_STRESS_SYNTHESIS_REPORT.md。它们是描述性系数对照，不是热点显著性检验或因果估计，尚待独立复核。

## 2026-11-03 J-3空间地图与SEM路径图补充

阶段图集扩展为26幅不重复图件：A–I核心图和J–Z补充图。U展示九项结局240 m局部GWR-LCR系数；其CEC来自当前按全域侵蚀最大值公式恢复的数据。V显示SHI、OM、SW在770个共同空间位置上的MGWR−GWR系数差值。W–Z分别展示平行维度、链式维度、九项健康结局和两部分侵蚀结构的SEM路径图。所有补充图由已保存的局部系数或SEM路径表绘制，没有重拟合模型；局部系数和路径仍按描述性条件关联解释。

正式报告已同步为Markdown、LaTeX和A4 PDF；图注、逐图数据源、HTML图册及全项目总图集均已更新。新增地图与路径图、LaTeX/PDF更新尚待独立只读复核。

## CEC数据核验

按既定的全域最大值公式，以原CEC和两期完整对齐侵蚀栅格计算。有效像元387,448个，保存值与复算值最大差为0，范围16.725814–26.836542。图面、图题与图注均写“CEC”。由于公式包含Aa，CEC与侵蚀的关系有公式构造造成的代数耦合，不作为独立侵蚀证据。本机`inputs/cec_formula_audit/`含完整源表与栅格；Git版本保留公式清单，但不上传大型原始表和栅格。

本地审查包保留完整逐像元分析表、匹配来源表、CEC审计栅格和lavaan输入矩阵。为控制仓库体积，Git版本只跟踪汇总数据与结果，不上传`data/`和`inputs/`下的大型Parquet/栅格，也不上传压缩的lavaan建模矩阵；这些文件仍完整保存在本机审查目录。因此远端审查者可以核对模型脚本、汇总表、图件和报告，但不能只依靠Git版本重算像元级分析。

## 样本、匹配与模型说明

- 两张输入表各392,527行；`cell_id`唯一且一一对应，行列和坐标逐项一致。按本阶段模型所需变量筛选的核心完整案例为293,145行；其中71行虽被广义环境栈标记为无效，但缺少的只是未进入本阶段模型的辅助近邻侵蚀字段。
- D3为历史匹配设计的181,794个端点；D4为另一版本的111,289对（222,578个端点）。共有133,533个端点，两个版本不可合并。D4配对差值与D3端点回归的统计单位不同；全域相关与配对差值相关不能直接用于推断匹配使关系减弱多少。
- Phase I既有GWR/MGWR结果被复用，没有重估；图B展示770个空间均衡位置上的SHI、OM、SW；图U另展示九项结局240 m GWR-LCR局部系数地图，图V对照标准GWR与MGWR差值。九项结局结果源自当前CEC版本。
- 分段路径模型1,346项、自定义协方差路径近似12项、形成性Mode-B PLS路径模型176项、成熟 lavaan CB-SEM 48项，共1,582项，状态全部成功。PLS覆盖D1–D4四类总体、S1–S3三种结构和M0–M3四级控制；S1每种总体/控制组合拟合三个健康维度方程，S2/S3另拟合SHI方程，并分别保留/删除侵蚀直达SHI路径。每个总体内所有结构和控制层级使用相同完整样本；环境与空间控制作为固定单指标外生构念进入内模型。PLS为项目内透明权重迭代，不提供bootstrap区间或空间稳健推断。lavaan 0.7.3在隔离临时R库中运行，未修改项目依赖。CB拟合量采用常规最大似然独立观测假设，未对空间依赖作校正；部分模型自由度为0，CFI/TLI/RMSEA不提供有效的模型检验。分段路径模型使用5 km空间块聚类协方差。预测栅格不是独立实测样点。
- 成熟软件CB-SEM规格已完成；48个模型覆盖四类总体、三种基础结构与四级控制条件。结果中AIC候选、侵蚀增量、最强负路径和科学解释候选分开列示，不以单一拟合排名代替科学判断。
- 结果属于探索性条件关联，健康结局参与过候选比较。本地章节草稿现已提供八项指标方向、隶属度参数和组合权重；J-9补充运行按这些规则建立方向评分。但重算SHI与既有`SHI_candidate`不一致，暂不替换r10主运行中的SHI，也未据新分值重估GWR/MGWR、PLS或CB-SEM。PCNM留待后续专门对照。

## 术语

- D1–D7指全域像元、空间块、两版匹配总体、正侵蚀、零—正配对和土类—地形湿度组；定义见`provenance/population_audit.json`。
- M4为项目既有环境匹配方案的代号；本包区分两个历史版本，参数与端点交集见总体审计。
- GWR/MGWR分别为局部地理加权回归和多尺度地理加权回归；本阶段仅复用已有结果。
- S1–S6和M0–M3表示路径结构及控制变量组合，定义见技术报告，图注尽量采用中文解释。

## 文件

- `figures/`：A–Z共26张阶段图及图源表，每图提供PNG、SVG、PDF；补充图位于`figures/supplementary/`。
- `reports/ENV_MATCH_SOIL_HEALTH_ACADEMIC_REPORT.md/.tex/.pdf`：正式报告三种格式；PDF已包含A–Z完整图版。
- `inputs/`：CEC公式审计和两套匹配来源输入。
- `source/`：分析脚本快照和研究任务规格。
- `J9_HEALTH_DIRECTION_RECONSTRUCTION.md`：方向评分规则、原始值/方向评分结果、既有SHI核对和限制。
- `PHASE_J_REVIEW_REQUEST.md`：独立只读审查问题。
- `provenance/`与`RUN_STATUS.md`：输入、配置、数值审计、方法边界和执行状态。

运行清单记录的Git提交为`7b91a7e86a4e13742b3578556b19f7992b7ac5a1`，运行时工作树为dirty，因此本包不是固定提交快照。新补入的 lavaan、图A–C和报告材料仍需独立只读复核。
