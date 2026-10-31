# 温县环境匹配探索显示侵蚀—健康关系以含水量和有机质的负向响应最稳定

## 执行摘要

- 全掩膜预测栅格的关联以 SW、SHI 和 OM 的负向关系较明显；Aa_20yr Spearman 分别为 −0.584、−0.469、−0.415。CEC 为正向（ρ=0.168）。这些像元关联描述空间梯度，不是独立实测样点效应。
- 在既有 M4 匹配配对中，SHI、OM、SW 的标准化配对斜率仍为负（−0.144、−0.100、−0.269），但比端点像元相关弱；“匹配端点关联”和“高低端配对差值关联”是不同估计对象。
- 空间差异明显：I-4 环境调整 GWR 对 SHI、OM、SW 的加权平均标准化局部系数为 −0.330、−0.556、−0.680；GWR-LCR收缩后为 −0.218、−0.387、−0.464。I-6局部Spearman的中位数为 −0.591、−0.616、−0.783。
- Oracle分位分区得到623个T1高Aa—强负向响应240m单元（按共同完整案例像元换算约18.88 km²）。该区是后续实地核验候选，不能当作已验证治理风险区。

## 1. 研究问题、范围与分析单位

本报告整合 Phase I-1 至 I-8 及 I-C 图件闭合/标准MGWR补充，回答侵蚀与土壤健康的总体关系、环境匹配后的条件关联、非线性与环境异质性、局部空间变化、健康维度差异及候选治理分区。所有阶段采用 `EXPLORATORY_ORACLE`，并保留 `outcome_guided_selection=true`；不能作为预注册正式证据或独立因果验证。

侵蚀暴露是既有 Aa_20yr 预计算栅格，分析直接读取已对齐数据库，不重建或验证 RUSLE 因子乘积。SHI及八项健康指标为随机森林预测栅格。掩膜库保留392,527个像元；I-1依每项结局完整案例分析，I-2 Aa_20yr M4候选含111,289对匹配端点（结局有效对数依指标略有不同）；I-4/I-6空间分析约用240m汇总单元；I-8分区使用293,633个环境和SHI共同完整像元组成的7,387个汇总单元。像元数、匹配对数、空间块与独立实测样点不能互换。

Aa单位、精确时间口径尚未从元数据确认；报告中的暴露轴沿用源值。

## 2. 全掩膜：侵蚀与健康指标的总体响应

全掩膜 Spearman 中，SW与Aa_20yr的负向排序关系最强（ρ=−0.584），其后为SHI（−0.469）、OM（−0.415）；CEC为正（+0.168）。I-1 SHI标准化线性β为−0.472（5km空间块bootstrap 95% CI −0.531至−0.412）；SW为−0.595（−0.630至−0.555），OM为−0.421（−0.459至−0.377）。零侵蚀像元占SHI完整案例34.2%；正侵蚀减零侵蚀的SHI Hedges' g为−0.889，95%空间块区间见I-1报告。

模型对比使用5km空间块训练/验证/测试，按验证集选择模型，测试集只留出评估。SHI测试R²为0.187，许多结局测试表现接近零或为负；不据大像元数宣称模型通过外部验证。分段拐点来自Oracle验证集候选筛选，MBC的Aa约4.617是探索候选，不是独立阈值。

![I-1：全掩膜九项指标侵蚀关联](figures/i1_I01_global_association_heatmap.png)

*I-1：全掩膜九项指标侵蚀关联*<br>
[PNG](figures/i1_I01_global_association_heatmap.png)
![I-1：零侵蚀与正侵蚀状态差异](figures/i1_I04_zero_positive_boxplots.png)

*I-1：零侵蚀与正侵蚀状态差异*<br>
[PNG](figures/i1_I04_zero_positive_boxplots.png)
## 3. 环境匹配后：配对差值仍有负向信号，但幅度较小

I-2复用Phase H M4 Aa_20yr候选 `stage2_Aa_20yr_01`，没有重做匹配。配对端点的全侵蚀范围像元相关接近I-1：SHI Spearman −0.452 vs 全掩膜−0.469，OM −0.416 vs −0.415，SW −0.567 vs −0.584。配对差值的响应弱于端点相关：SHI/OM/SW的标准化斜率约−0.144/−0.100/−0.269，Spearman约−0.144/−0.115/−0.272。它们表示每对高侵蚀端相对低侵蚀端的差值关联，不可用来替代全像元关联。

零—正与双正子组分别保留；Aa_20yr全范围匹配对OLS关系的表观拟合较弱。I-2模型排行榜中的随机森林R²是同样本表观拟合，不是空间泛化。匹配提高了环境可比性，但不消除未测混杂，也不能单凭这一结果称作侵蚀独立效应。

![I-2：Aa_20yr M4匹配配对响应](figures/i2_I09_Aa_20yr_matched_pair.png)

*I-2：Aa_20yr M4匹配配对响应*<br>
[PNG](figures/i2_I09_Aa_20yr_matched_pair.png)
## 4. 非线性和环境异质性

I-1全样本GAM局部斜率显示SHI范围维持负向；多项单项指标斜率跨零。I-5以5km整块测试评估空间变化系数GAM近似，相较全局GAM，SHI/OM/SW测试R²提升仅+0.014/+0.007/+0.005，空间变化曲面提供的额外预测信息有限。

I-3复用既有M4配对环境组。Aa_20yr下27潮土SHI/OM/SW标准化配对斜率为−0.246/−0.294/−0.316；27:q2在三项中是TWI组内较负向候选。126项M4交互有123项BH q<0.05，但这些交互来自同一Oracle搜索流程并受空间依赖影响，不能当作独立重复确认。Phase H的26:q1低TWI候选基于另一低值筛选总体，不能与27:q2直接排名。

![I-3：土类内TWI分组的配对斜率](figures/i3_I11_m4_twi_group_slopes.png)

*I-3：土类内TWI分组的配对斜率*<br>
[PNG](figures/i3_I11_m4_twi_group_slopes.png)
## 5. 空间变化与局部集聚

I-4环境调整GWR的SHI、OM、SW加权平均标准化斜率为−0.330/−0.556/−0.680；99.7%局部单元的SHI系数为负。GWR-LCR对SHI/OM/SW的系数向零收缩至−0.218/−0.387/−0.464，且所有局部点都施加岭惩罚，说明环境协变量局部共线性会影响系数幅度。I-4标准GWR候选中31/54个带宽落在k=30下界。I-4当时使用MGWR-inspired近似；I-C随后在空间均衡n=770样本完成标准GWR/MGWR，详见5.1节。

I-5局部RF验证R²中位数为SHI 0.610、OM 0.618、SW 0.736，但该局部随机划分邻域重叠，不是空间独立测试。局部Aa边际效应中位数为−0.144/−0.263/−0.437；SHAP贡献和变量重要性是预测分解，不作因果解释。

I-6 Local Spearman中位数为SHI −0.591、OM −0.616、SW −0.783。Bivariate LISA的FDR显著HL数量分别为1,434、843、1,103；同位高Aa—低健康SHI共现为2,473个单元（33.5%），与邻域滞后HL类别不同。负向系数Gi*高值热点在SHI的GWR为2,484个、GWR-LCR为3,124个单元，模型敏感性明显。置换与Gi*近似p值不消除空间依赖。

![I-4：环境调整SHI GWR局部标准化系数](figures/i4_gwr_SHI_candidate_template.png)

*I-4：环境调整SHI GWR局部标准化系数*<br>
[PNG](figures/i4_gwr_SHI_candidate_template.png)
![I-5：局部随机森林Aa边际效应](figures/i5_grf_local_Aa_marginal_effect.png)

*I-5：局部随机森林Aa边际效应*<br>
[PNG](figures/i5_grf_local_Aa_marginal_effect.png)
![I-6：双变量Local Moran类别](figures/i6_bivariate_local_moran_clusters.png)

*I-6：双变量Local Moran类别*<br>
[PNG](figures/i6_bivariate_local_moran_clusters.png)
![I-6：GWR负向响应Gi*热点](figures/i6_gistar_negative_response_gwr.png)

*I-6：GWR负向响应Gi*热点*<br>
[PNG](figures/i6_gistar_negative_response_gwr.png)
## 5.1 标准 MGWR 与核心图件闭合补充

本节对应独立运行 [phase_i_20261031_seed20261031_closure_r1](PHASE_I_CLOSURE_REPORT.md)。标准GWR与MGWR使用同一空间均衡样本：完整240m单元 7,387 个，固定 800m网格每格抽取一个单元，n=770；抽样只依据坐标，不按健康响应取样。模型含log1p(Aa_20yr)与九项环境协变量，Gaussian adaptive kernel，AICc带宽搜索，每个抽样单元等权。三项关键结局中 3/3 项完成标准GWR/MGWR拟合。

标准GWR与MGWR在此共同样本中的拟合汇总：

| 指标 | 同样本GWR R² | 标准MGWR R² | GWR AICc | MGWR AICc |
| --- | --- | --- | --- | --- |
| SHI | 0.887 | 0.869 | 781.572 | 770.331 |
| OM | 0.714 | 0.696 | 1417.295 | 1389.804 |
| SW | 0.834 | 0.822 | 989.469 | 972.268 |

Aa暴露的MGWR尺度：

| 指标 | Aa邻居数 | 中位作用距离(km) |
| --- | --- | --- |
| SHI | 32 | 2.64 |
| OM | 767 | 23.78 |
| SW | 767 | 23.78 |

该空间均衡子样本估计不等同于I-4全量240m单元、按源像元支持加权的GWR。局部系数是预测栅格条件关联，不作局部因果解释。MGWR带宽、收敛和局部共线性限制见补充报告。

![八项健康指标GWR局部系数组合](figures/ic_C16_eight_indicator_gwr_composite.png)

*八项健康指标GWR局部系数组合*<br>
[PNG](figures/ic_C16_eight_indicator_gwr_composite.png)
![标准GWR与MGWR局部系数，同一空间均衡样本](figures/ic_C17_MGWR_local_coefficients.png)

*标准GWR与MGWR局部系数，同一空间均衡样本*<br>
[PNG](figures/ic_C17_MGWR_local_coefficients.png)
![标准MGWR变量作用尺度](figures/ic_C18_MGWR_variable_scales.png)

*标准MGWR变量作用尺度*<br>
[PNG](figures/ic_C18_MGWR_variable_scales.png)
![标准GWR与MGWR系数差异](figures/ic_C19_GWR_MGWR_difference_map.png)

*标准GWR与MGWR系数差异*<br>
[PNG](figures/ic_C19_GWR_MGWR_difference_map.png)
![核心图30状态：PCNM源表缺失](figures/ic_C30_PCNM_comparison_NOT_AVAILABLE.png)

*核心图30状态：PCNM源表缺失*<br>
[PNG](figures/ic_C30_PCNM_comparison_NOT_AVAILABLE.png)
![核心图31状态：PLS-SEM源表缺失](figures/ic_C31_mechanism_path_NOT_AVAILABLE.png)

*核心图31状态：PLS-SEM源表缺失*<br>
[PNG](figures/ic_C31_mechanism_path_NOT_AVAILABLE.png)
核心图30、31只标明PCNM与PLS-SEM源结果缺失；它们不是实际对照森林图或机制路径图。

## 6. 指标与健康维度

下表逐项保留健康指标原始方向。I-1的标准化β基于log1p(Aa)；I-2 β来自匹配对差值；I-4为各局部单元加权平均值；I-6为重叠邻域Spearman分布中位数。估计单位和模型不同，不据绝对值排序。

| 指标 | 维度 | I-1 Spearman ρ | I-1 标准化β（5km块95% CI） | I-2 配对β / ρ | I-4 GWR / LCR局部均值β | I-6 局部ρ中位数 |
| --- | --- | --- | --- | --- | --- | --- |
| SHI | SHI composite | -0.469 | -0.472 (-0.531, -0.412) | -0.144 / -0.144 | -0.330 / -0.218 | -0.591 |
| SW | Physical | -0.584 | -0.595 (-0.630, -0.555) | -0.269 / -0.272 | -0.680 / -0.464 | -0.783 |
| BD | Physical | -0.200 | -0.156 (-0.178, -0.131) | -0.017 / -0.032 | -0.414 / -0.293 | -0.389 |
| OM | Chemical | -0.415 | -0.421 (-0.459, -0.377) | -0.100 / -0.115 | -0.556 / -0.387 | -0.616 |
| AN | Chemical | -0.233 | -0.240 (-0.326, -0.149) | -0.025 / -0.024 | -0.004 / 0.002 | 0.001 |
| CEC | Chemical | 0.168 | 0.164 (0.086, 0.236) | 0.029 / 0.045 | 0.315 / 0.215 | 0.533 |
| pH | Chemical | -0.109 | -0.116 (-0.199, -0.025) | -0.043 / -0.051 | -0.030 / -0.030 | -0.200 |
| MBC | Biological | -0.118 | -0.067 (-0.111, -0.023) | 0.088 / 0.056 | -0.154 / -0.096 | -0.208 |
| RESP | Biological | -0.272 | -0.252 (-0.307, -0.195) | 0.016 / -0.014 | -0.345 / -0.229 | -0.452 |

物理维度（SW、BD）的整体负向信号较明显，且SW方向在全样本、匹配、GWR和局部相关结果中较一致。化学维度方向混合：OM较稳定负向，CEC多为正向，AN和pH的匹配/局部证据较弱或跨零。生物维度中MBC、RESP的匹配斜率分别为正，和全样本/局部GWR结果不完全一致，提示该维度反应不稳定。维度汇总为指标描述性统计，不是潜变量效应。

| 维度 | 指标数 | I-1 β负/正 | I-1 β中位数 | I-2 β中位数 | I-4 GWR均值β的维度内中位 | I-6 局部ρ中位数 |
| --- | --- | --- | --- | --- | --- | --- |
| SHI composite | 1 | 1/0 | -0.472 | -0.144 | -0.330 | -0.591 |
| Physical | 2 | 2/0 | -0.376 | -0.143 | -0.547 | -0.586 |
| Chemical | 4 | 3/1 | -0.178 | -0.034 | -0.017 | -0.100 |
| Biological | 2 | 2/0 | -0.160 | 0.052 | -0.250 | -0.330 |

![I-7：全样本、匹配与局部系数对照](figures/i7_health_dimension_comparison_forest.png)

*I-7：全样本、匹配与局部系数对照*<br>
[PNG](figures/i7_health_dimension_comparison_forest.png)
当前工作区没有可读的PCNM结果表，配置 `pcnm.results=null`；既有PLS-SEM路径、载荷和方向资料也未提供。I-7已保留EnvMatch-PCNM比较字段和PLS路径交叉字段，但缺失值保持为空；没有重跑PCNM、推断路径或填造比较结果。因此目前不能判断EnvMatch与PCNM是否一致，也不能回答EnvMatch相对PCNM增加了何种独立解释。I-7森林图只展示可用的EnvMatch估计，不含PCNM对照；核心图30仅为源数据缺口标识。

## 7. 空间胁迫候选分区与治理含义

I-8在环境和SHI共同完整的7,387个约240m单元上，按Aa q80、环境调整SHI GWR系数q20、Aa q50和SHI q20构造互斥分类。所有阈值均是Oracle分位数，不是法定风险线。

| 分区 | 240m单元 | 单元占比 | 完整案例面积等价(km²) | 均值Aa | 均值SHI | 局部GWRβ中位数 | I-6 GWR热点占比 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T1 高Aa—强负向响应 | 623 | 8.4% | 18.88 | 15.670 | 0.549 | -0.567 | 99.7% |
| T2 高Aa—弱负向响应 | 842 | 11.4% | 24.00 | 14.951 | 0.584 | -0.327 | 27.0% |
| T3 低Aa—低SHI | 60 | 0.8% | 2.05 | 5.480 | 0.562 | -0.368 | 30.0% |
| T4 环境敏感候选 | 961 | 13.0% | 38.77 | 5.109 | 0.628 | -0.311 | 27.3% |
| 其他/未分类 | 4901 | 66.3% | 180.57 | 5.324 | 0.624 | -0.301 | 27.7% |

T1约18.88 km²等价面积，是优先现场核验候选；T2约24.00 km²等价面积，优先核查是否存在环境/农艺缓冲；T3约2.05 km²等价面积，应以土壤实测诊断非侵蚀限制。环境候选另将27潮土—TWI q2空间代理和26冲积土—TWI q1空间代理单独展示；它们分别对应I-3全配对与另一低值筛选总体，不能合并成统一已验证敏感区。面积只按共同完整案例纳入的30m像元换算。

建议在任何治理工程设计前现场核对侵蚀、地形、田块格局、灌排和作物制度。国家水土保持指导提出按地区和坡耕地条件选择工程、植物和保护性耕作措施；低SHI分区先做土壤检测，再确定培肥和养分管理。参见[I-8报告及官方依据](reports/stages/I8_PHASE_I_SPATIAL_STRESS_ZONING_REPORT.md)。

![I-8：主分区和环境敏感候选代理](figures/i8_stress_zoning_map.png)

*I-8：主分区和环境敏感候选代理*<br>
[PNG](figures/i8_stress_zoning_map.png)
## 8. 综合结论与下一步证据

1. 现有预测栅格显示侵蚀与SHI、SW、OM总体以负向关联为主，SW的方向最稳定；CEC呈正向关联，AN、pH及生物指标存在方向或模型差异。
2. M4匹配配对结果仍观察到SHI、OM、SW的负向配对关系，但幅度弱于端点像元关联；这支持在当前匹配设计下仍有条件关联线索，不证明充分消除环境混杂。
3. 环境组、GWR、局部相关和热点方法均发现空间异质性；GWR与LCR对局部系数/热点量敏感。标准MGWR补充拟合使用空间均衡子样本，与I-4像元支持加权GWR的估计对象不同；空间效应大小仍有方法不确定性。
4. 27:q2、26:q1和T1–T4属于Oracle候选，需用独立实测土壤样点和地块级侵蚀/农艺资料复核；不据当前空间预测面单独确定治理优先级或效果。
5. PCNM与PLS-SEM源结果缺失，现阶段不能作跨方法一致性或机制路径结论；核心图30、31只显示数据缺口状态。后续需接入其实际结果表，并核实Aa单位/时间口径、SHI来源与指标方向；空间预测栅格仍应与独立实测样点分开报告。

## 阶段报告与复现入口

- **I-1 全样本侵蚀—健康响应**：[阶段报告](reports/stages/I1_PHASE_I_GLOBAL_RESPONSE_REPORT.md)
- **I-2 环境匹配条件关联**：[阶段报告](reports/stages/I2_PHASE_I_ENVIRONMENT_MATCHING_REPORT.md)
- **I-3 环境异质性**：[阶段报告](reports/stages/I3_PHASE_I_ENVIRONMENT_HETEROGENEITY_REPORT.md)
- **I-4 GWR与空间变化系数**：[阶段报告](reports/stages/I4_PHASE_I_GWR_MGWR_REPORT.md)
- **I-5 空间机器学习**：[阶段报告](reports/stages/I5_PHASE_I_SPATIAL_ML_REPORT.md)
- **I-6 局部关联与热点**：[阶段报告](reports/stages/I6_PHASE_I_SPATIAL_CLUSTER_REPORT.md)
- **I-7 健康维度与PCNM/PLS衔接**：[阶段报告](reports/stages/I7_PHASE_I_HEALTH_MECHANISM_REPORT.md)
- **I-8 空间胁迫分区**：[阶段报告](reports/stages/I8_PHASE_I_SPATIAL_STRESS_ZONING_REPORT.md)
- **I-C 图件闭合与标准MGWR**：[补充报告](PHASE_I_CLOSURE_REPORT.md)

本综合材料生成于 Git HEAD `82694765485dc11f4729bb9a23b5de2d666fb969`。工作树内的报告与代码当前未提交；审阅请求要求在接收审查时先读取对应版本研究总纲，并结合本地工作树中的I-1至I-9材料审阅。

全部图件索引见 [Phase I Figure Atlas](PHASE_I_CORE_FIGURE_ATLAS.md)。主要指标字段和PCNM/PLS空值交叉表见I-7运行的`tables/`目录；空间分区空间单元表、汇总和阈值见I-8运行的`tables/`目录。
