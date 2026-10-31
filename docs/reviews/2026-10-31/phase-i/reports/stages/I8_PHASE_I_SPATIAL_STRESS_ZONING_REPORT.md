# Phase I-8：空间胁迫候选分区与管理建议

权威运行：`phase_i_20261031_seed20261031_i8_r2`；模式：`EXPLORATORY_ORACLE`；`outcome_guided_selection=true`。

## 主要发现

分区将环境调整GWR局部SHI系数、同单元Aa与预测SHI水平结合；I5局部随机森林边际效应与I6局部相关、LISA和Gi*仅作为分区剖面诊断。分区是按当前完整案例分布构造的探索性候选，不能解释为因果效应或实测风险图。

| zone | spatial_units_240m | share_of_units | complete_case_pixels | area_km2_equivalent | mean_Aa_20yr | mean_SHI | median_GWR_beta | LISA_HL_share | GWR_GiStar_hotspot_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T1 高Aa—强负向响应 | 623.000 | 0.084 | 20980.000 | 18.882 | 15.670 | 0.549 | -0.567 | 0.785 | 0.997 |
| T2 高Aa—弱负向响应 | 842.000 | 0.114 | 26668.000 | 24.001 | 14.951 | 0.584 | -0.327 | 0.340 | 0.270 |
| T3 低Aa—低SHI | 60.000 | 0.008 | 2281.000 | 2.053 | 5.480 | 0.562 | -0.368 | 0.317 | 0.300 |
| T4 环境敏感候选 | 961.000 | 0.130 | 43074.000 | 38.767 | 5.109 | 0.628 | -0.311 | 0.093 | 0.273 |
| 其他/未分类 | 4901.000 | 0.663 | 200630.000 | 180.567 | 5.324 | 0.624 | -0.301 | 0.112 | 0.277 |

共同完整案例覆盖 293,633 个30m像元、7,387 个约240m单元；分区面积按纳入像元数×900m²换算，仅代表该完整案例支持。高Aa阈值为q80=11.55，强负向GWR阈值为q20=-0.4579，低Aa阈值为q50=6.65416，低SHI阈值为q20=0.574183。Aa单位沿用源值，元数据尚未核实。

## 分区规则与敏感性

| 规则 | 定义 | 阈值 |
| --- | --- | --- |
| 高Aa | 240m单元Aa_20yr均值 >= q80 | 11.549962 |
| 强负向SHI响应 | 环境调整GWR标准化Aa系数 < 0 且 <= 全部系数q20 | -0.457909 |
| 低Aa | 240m单元Aa_20yr均值 <= q50 | 6.654161 |
| 低SHI | 240m单元SHI预测值 <= q20 | 0.574183 |

- T1：Aa达到q80且环境调整GWR系数位于全体单元q20以下，并为负。
- T2：Aa达到q80且局部GWR系数为负，但未进入T1强负向分位。高Aa而GWR系数非负的单元保留在“其他”，并由统计表单列比例。
- T3：Aa不高于q50且SHI不高于q20。
- T4：27潮土—TWI q2或26冲积土—TWI q1候选的空间代理，且未被T1–T3先行分类。第二幅图将两种代理单独显示，即使它们与前三类重叠。

环境敏感代理以240m块内主导土类占比至少80%筛选，再按对应主导土类块均值TWI的分位范围定位；这不同于I-3按匹配对高侵蚀端分配的土类内分位组。I-3全配对结果中27:q2是SHI/OM/SW较强负向候选；Phase H的26:q1属于另一种低值筛选总体。两种证据不合并为同一个已验证敏感区。

| overlay | units | share_of_analysis_units | TWI_low_cut | TWI_high_cut | spatial_definition |
| --- | --- | --- | --- | --- | --- |
| 27潮土·TWI q2 spatial proxy | 963.0000 | 0.1304 | 8.1867 | 8.5434 | dominant SoilType=27 at >=80% of complete pixels in 240m tile; tile mean TWI q20-q40 among such tiles |
| 26冲积土·TWI q1 spatial proxy | 285.0000 | 0.0386 | 6.3837 | 8.0774 | dominant SoilType=26 at >=80% of complete pixels in 240m tile; tile mean TWI q0-q20 among such tiles |

## 局部响应与环境特征

主系数来自I-4环境调整GWR；GWR-LCR用于局部共线性敏感性对照。多尺度SVC列是I-4自实现backfitting近似，不能称为标准MGWR。GRF通过最近1.2km局部中心连接，距离分布随表保留；它是较粗分辨率的诊断量，不是240m单元上的独立拟合。

| zone | negative_GWR_share | negative_GWR_LCR_share | negative_multiscale_SVC_share | median_GWR_beta | median_local_Spearman | median_GRF_local_Aa_effect_nearest_1p2km | median_GRF_center_distance_m | GWR_LCR_GiStar_hotspot_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T1 高Aa—强负向响应 | 1.000 | 1.000 | 0.995 | -0.567 | -0.605 | -0.244 | 448.963 | 0.340 |
| T2 高Aa—弱负向响应 | 1.000 | 1.000 | 0.996 | -0.327 | -0.592 | -0.162 | 477.507 | 0.406 |
| T3 低Aa—低SHI | 1.000 | 1.000 | 1.000 | -0.368 | -0.621 | -0.268 | 406.234 | 0.483 |
| T4 环境敏感候选 | 0.990 | 0.992 | 0.981 | -0.311 | -0.576 | -0.144 | 473.595 | 0.470 |
| 其他/未分类 | 0.995 | 0.995 | 0.986 | -0.301 | -0.590 | -0.144 | 479.519 | 0.426 |

下表给出各主分区的环境汇总。I-3的配对分组表和本表的像元/空间单元总体不同，环境值只用于解释候选分区的样本构成。

| zone | profile_variable | mean | median | q25 | q75 |
| --- | --- | --- | --- | --- | --- |
| T1 高Aa—强负向响应 | TWI | 8.617 | 8.502 | 8.082 | 9.038 |
| T1 高Aa—强负向响应 | LST | 16.941 | 17.084 | 16.519 | 17.473 |
| T1 高Aa—强负向响应 | NDWI | -0.403 | -0.406 | -0.428 | -0.382 |
| T1 高Aa—强负向响应 | NDVI | 0.393 | 0.397 | 0.365 | 0.423 |
| T2 高Aa—弱负向响应 | TWI | 8.511 | 8.420 | 7.968 | 8.979 |
| T2 高Aa—弱负向响应 | LST | 16.904 | 16.947 | 16.713 | 17.189 |
| T2 高Aa—弱负向响应 | NDWI | -0.440 | -0.446 | -0.469 | -0.420 |
| T2 高Aa—弱负向响应 | NDVI | 0.444 | 0.452 | 0.406 | 0.483 |
| T3 低Aa—低SHI | TWI | 8.976 | 8.871 | 8.399 | 9.296 |
| T3 低Aa—低SHI | LST | 16.651 | 16.721 | 16.317 | 16.916 |
| T3 低Aa—低SHI | NDWI | -0.437 | -0.452 | -0.467 | -0.411 |
| T3 低Aa—低SHI | NDVI | 0.469 | 0.473 | 0.440 | 0.507 |
| T4 环境敏感候选 | TWI | 8.289 | 8.335 | 8.238 | 8.439 |
| T4 环境敏感候选 | LST | 16.787 | 16.841 | 16.640 | 16.972 |
| T4 环境敏感候选 | NDWI | -0.473 | -0.475 | -0.496 | -0.456 |
| T4 环境敏感候选 | NDVI | 0.507 | 0.510 | 0.474 | 0.543 |
| 其他/未分类 | TWI | 8.865 | 8.841 | 8.391 | 9.246 |
| 其他/未分类 | LST | 16.821 | 16.876 | 16.678 | 17.006 |
| 其他/未分类 | NDWI | -0.471 | -0.473 | -0.493 | -0.452 |
| 其他/未分类 | NDVI | 0.500 | 0.504 | 0.466 | 0.536 |

![综合空间胁迫候选分区图](../../figures/i8_stress_zoning_map.png)

图上排为中国—河南—温县定位示意，星号是完整案例像元数加权中心（由EPSG:32649转换至WGS84）；下排左侧为互斥主分区，右侧为两类环境候选代理。主图继续使用EPSG:32649项目坐标和项目耕地掩膜轮廓，行政区几何仅用于定位，不参与统计分区。

定位边界取自[panyanbin/gcoord行政区GeoJSON](https://github.com/panyanbin/gcoord#中国行政区geojson数据)，其说明将中国区划数据归于阿里DataV并按GeoJSON/WGS84表达；本次使用[中国](https://raw.githubusercontent.com/panyanbin/gcoord/master/map/%E4%B8%AD%E5%9B%BD_100000.json)和[河南省](https://raw.githubusercontent.com/panyanbin/gcoord/master/map/%E6%B2%B3%E5%8D%97%E7%9C%81_410000.json)文件（获取日期2026-10-31）。该第三方边界仅为定位示意，不替代官方界线或项目耕地掩膜。

## 分区治理建议（待实地验证）

| zone | priority | management_candidate | evidence_boundary |
| --- | --- | --- | --- |
| T1 高Aa—强负向响应 | 优先核验与治理候选 | 先到现场核对侵蚀迹象与耕作条件；在坡度、田块格局和灌排条件适用时，试行覆盖/少扰动、等高耕作或坡面径流调控，并布设地块尺度监测。 | 由高Aa分位与负向GWR分位定义；不是工程设计结论，具体措施需按地形和生产条件论证。 |
| T2 高Aa—弱负向响应 | 核验缓冲条件 | 对照TWI、植被和土类剖面核实可能的环境缓冲；保留地表覆盖和适宜轮作，比较治理前后径流/土壤指标。 | 局部系数较弱不证明存在因果缓冲；以现场水文与农艺资料验证。 |
| T3 低Aa—低SHI | 诊断非侵蚀限制 | 优先开展地块土壤检测，区分含水、容重、有机质、养分、pH及生物指标限制；据检测结果再制定培肥、秸秆/绿肥利用或配方施肥方案。 | 低SHI来自预测栅格；图上类别本身不能指认缺素、板结或其他单一机制。 |
| T4 环境敏感候选 | 定向复核环境候选 | 分别核实27潮土—TWI q2与26冲积土—TWI q1候选的土壤水分、地表径流和健康指标；在验证前不将两类合并为统一工程措施。 | 地图分位是240m汇总块的空间代理，不等于I-3匹配对端点分组或已验证敏感区。 |
| 其他/未分类 | 一般监测 | 保留常规地块监测；如出现实测侵蚀或健康变化，再按实测值重新诊断优先级。 | 未进入四类筛选规则不代表无风险。 |

水土保持措施需按坡度、地块格局、灌排与作物制度逐地块核验；国家《水土保持法》列举了因地制宜的工程、植物及保护性耕作措施，并对坡耕地提出梯田、坡面水系整治、蓄水保土耕作等选择。[《中华人民共和国水土保持法》](https://www.mee.gov.cn/ywgz/fgbz/fl/199106/t19910629_81963.shtml)；[《关于加强新时代水土保持工作的意见》](https://www.mee.gov.cn/zcwj/zyygwj/202301/t20230103_1009406.shtml)。低健康分区应先由土壤实测定位障碍再制定培肥方案；[北京市农业农村局有机质提升技术](https://nyncj.beijing.gov.cn/nyj/zwgk/ztgk/gdjsbhqk/436449599/index.html)仅作为土壤检测、秸秆/绿肥和配方施肥的技术例子，不直接外推其剂量至温县。

## 解释边界与文件

- SHI与八项指标均为随机森林预测栅格，不是独立实测样点；240m块及邻域统计有空间依赖。
- 分区阈值是Oracle探索中的分布分位数，结果导向筛选和既有模型搜索均可能影响表观格局；不代表法定风险标准或独立验证。
- I6 Local Spearman为重叠邻域描述量；LISA置换未保留健康空间结构；Gi*使用正态近似；GWR局部区间为模型内近似。
- I4未运行标准MGWR；使用环境调整GWR/GWR-LCR及非标准多尺度SVC近似。
- 仅使用预计算Aa_20yr；未重算或验证RUSLE乘积、未重跑PCNM、未修改研究总纲。

图件：

- `outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_i8_r2/figures/stress_zoning_map.png`
- `outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_i8_r2/figures/stress_zoning_map.svg`
- `outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_i8_r2/figures/stress_zoning_map.pdf`

关键表：

- `tables/stress_zones_240m.parquet` 与 `.csv`：空间单元分区及合并诊断。
- `tables/stress_zone_summary.csv`：分区统计。
- `tables/stress_zone_environment_profiles.csv`：环境和主导土类剖面。
- `tables/environment_sensitive_overlays.csv`：环境候选代理定义和范围。
- `tables/management_recommendations.csv`：治理建议与证据边界。
- `tables/zone_rule_thresholds.csv`：可复核分区阈值。
- `manifests/provenance.json`：来源、配置与软件版本。
