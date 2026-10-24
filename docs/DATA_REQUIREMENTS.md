# 数据要求与输入模板

正式数据尚未提供。将原始文件放入相应 `data/raw/` 子目录，并通过 `config/data_paths.yaml` 映射。所有 GeoTIFF 需登记 CRS、仿射变换、像元大小、单位、NoData、年份、处理方法及有效范围。不要为满足模板而重命名源文件；由映射表完成逻辑名称到文件路径关联。

## 必需或优先数据

| 类别 | 逻辑字段 | 格式 | 用途及注意事项 |
|---|---|---|---|
| 侵蚀 | `Aa`、`Ap` | 连续 GeoTIFF | `Aa` 为主分析，`Ap` 为补充；单位需记录 |
| RUSLE 因子 | `R`、`K`、`LS`、`C`、`P` | 连续 GeoTIFF | 核查重建关系和 C 阈值；不得进入主匹配距离 |
| 健康 | `SHI`、`AN`、`OM`、`CEC`、`pH`、`SW`、`BD`、`MBC`、`RESP` | 连续 GeoTIFF | 标明 RF 预测或独立实测；匹配完成后才用于响应 |
| 环境 | `Elevation`、`Cur`、`TWI`、`LST`、`SoilType` | 连续/分类 GeoTIFF | 主候选环境变量；土壤类型作为类别分层 |
| 扩展环境 | `NDWI`、`NDBSI`、`EVI` 等 | 连续 GeoTIFF | 需记录来源、日期、环境意义和质量依据 |
| 空间 | 研究区边界、耕地掩膜、统一栅格模板 | GeoTIFF 掩膜优先，边界可用 GeoJSON/GPKG | 投影到统一网格；分类掩膜用最近邻 |
| 既有空间控制 | PCNM 未控制/控制结果 | CSV 或 Excel | 导入比较，不重跑 PCNM，不填补未提供字段 |

## 推荐映射文件

复制 `config/data_paths.example.yaml` 为不入库的 `config/data_paths.yaml`。路径相对于项目根目录。环境和健康因子可按需留空；真实运行时 CLI 应列出缺少的必需逻辑字段。

## 元数据清单字段

建议维护 `data/metadata.csv`，每个栅格一行，至少包含：

```text
logical_name,path,source,year,original_resolution,current_resolution,crs,unit,nodata,processing_method,is_rf_prediction,is_independent_observation,is_rusle_factor,used_for_matching,notes
```

`is_rf_prediction`、`is_independent_observation`、`is_rusle_factor`、`used_for_matching` 为布尔字段。预测栅格不应标记为独立实测值。R/K/LS/C/P 的 `used_for_matching` 必须为 false；SHI 和八项指标亦必须为 false。

## PCNM 表字段

可包含指标名称、侵蚀情景、模型阶段（未控制/PCNM 控制）、回归系数、标准化系数、标准误、置信区间上下限、P 值、R²、样本量、空间/分析单位、控制变量。缺失字段保留空值，不推断。CSV 编码应为 UTF-8；Excel 首行使用列名，并在映射配置中指定 sheet（若非首表）。

## 提供数据后的第一轮核查

1. 确认文件清单、来源、年份、单位、预测/实测身份和因子角色。
2. 检查 CRS、像元大小、transform、栅格 shape、NoData、无效值和研究区覆盖。
3. 检查耕地掩膜和 30 m 网格支持，明确重投影/重采样方法。
4. 重建或核对 `Aa = RKLSCP`、`Ap = RKLS`，重点检查 C 与零侵蚀比例。
5. 核查健康指标单位、范围、RF 预测来源及 SHI 组成。
6. 核查环境协变量分布、类别取值及与 RUSLE/RF 的共享信息。
7. 只有质量问题处理并记录后，才建立数据库、聚类和匹配。
