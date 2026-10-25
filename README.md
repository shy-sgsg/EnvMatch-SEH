# EnvMatch-SEH：环境匹配下的土壤侵蚀—土壤健康分析

## 项目介绍

本项目为温县耕地栅格研究建立可复现的 Python 工作流，在环境背景相近的条件下分析实际土壤侵蚀与土壤健康的条件关联及环境异质性。项目已收到原始栅格并完成首轮清点；当前处于数据质量审查阶段，空间网格与模板尚未确认，未产生温县研究结论。

研究方向以 [研究总纲](docs/RESEARCH_CHARTER.md) 为准。首版总纲提交后，其正文受审批机制保护；科研任务按 [治理协议](docs/governance/CODEX_TASK_PROTOCOL.md) 和 [阶段门](docs/governance/RESEARCH_STAGE_GATES.md) 执行。ChatGPT 阶段审查协议和总纲变更申请流程分别见 `docs/governance/`。

## 研究背景与研究区

研究区为河南省焦作市温县耕地，既有研究的耕地面积约 347.877 km²，空间分辨率为 30 m。最终有效像元数由数据对齐和耕地掩膜确定。已有侵蚀评估采用 RUSLE，计算实际侵蚀 `Aa = RKLSCP` 与潜在侵蚀 `Ap = RKLS`；正式分析前必须核查 C 因子植被覆盖阈值及其导致的零侵蚀比例。已有土壤健康制图以 AN、OM、CEC、pH、SW、BD、MBC、RESP 八项指标为基础，随机森林预测栅格经隶属度和权重形成 SHI。已有 PCNM 空间控制成果保留为独立比较基线，不在本项目重跑。

## 研究目标与科学问题

主题是基于环境协变量匹配的土壤侵蚀—土壤健康条件关联及环境异质性分析。核心问题包括：环境相似时关联是否存在；关联是否随环境背景变化；物理、化学和生物指标响应是否不同；结果与 PCNM 空间控制是否一致；如何提高匹配质量与统计推断可靠性。

## 新增研究内容

1. 以 Elevation、Cur、TWI、LST 等非 RUSLE 直接因子构建环境向量，按土壤类型分层，评估环境相似性。
2. 以 MiniBatch K-means 等可复现方法划分候选环境组，比较 6、8、10、12 组的组内距离、稳定性、样本覆盖和解释性；健康结果不得参与选组。
3. 在组内检查 `log1p(Aa)` 分布、分位数、零比例和有效侵蚀变化，确认有可比较的暴露范围。
4. 在同环境组、相同土壤类型约束下做环境近邻匹配，考察 300 m、500 m、1000 m 空间半径，记录环境距离、空间距离、侵蚀差值和平衡诊断。
5. 设计锁定后，对 SHI 和八项指标开展组内连续回归、匹配对差值回归、环境组异质性检验及空间块不确定性分析。
6. 导入既有 PCNM 结果进行表格化比较，不推测缺失字段，不直接比较不同空间支持单位下未经协调的标准化系数。

## 总体技术路线

```text
既有 PCNM 结果 ───────────────────────────────────────────────┐
                                                             │
30 m 栅格 → 清单/对齐/质量核查 → 非 RUSLE 环境变量筛选       │
         → 标准化环境向量 → 土壤类型分层与环境分组           │
         → 组内环境一致性与侵蚀变异检查                      │
         → 组内精细匹配 ─┬─ 环境组内连续回归                 │
                         └─ 匹配对差值回归                    │
         → SHI 与八项指标响应 → 环境异质性检验               │
         → 多方法对比 ←───────────────────────────────────────┘
         → 可复现图表、结果报告与论文总结
```

分组和配对只使用环境、分类条件、侵蚀暴露及坐标，不读取 SHI 或其他健康结果。R、K、LS、C、P 不进入主匹配距离；仍需讨论其环境决定因素造成的剩余混杂。

## 项目目录

```text
AGENTS.md                 长期研究和开发约束
config/                   默认参数、数据映射示例、匹配参数
data/raw/                 只读原始输入，按主题分类
data/interim/             可重建中间数据
data/processed/           对齐后的分析数据库
data/fixtures/            明确标记的合成测试数据
docs/                     研究方案、方法、数据要求、流程和决策记录
src/soilmatch/            可复用分析模块和 CLI
scripts/                  本地原始数据清点与栅格质量审计
tests/                    单元和小规模端到端测试
outputs/run_<id>/         每轮独立配置快照、元信息和结果
logs/                     运行日志
```

原始栅格、数据库、图件和报告默认不纳入普通 Git 版本管理；源码、文档、测试、模板配置需纳入。

## 数据准备清单

当前接入了预计算实际侵蚀栅格、八项健康指标及多组环境候选层。按用户指示，本轮直接使用已计算的 Aa，不重建 RUSLE 因子乘积。首轮审计报告和本地路径映射位于 `outputs/data_audit/` 与 `config/data_paths.yaml`；两处均由 `.gitignore` 排除。正式分析仍需确认空间模板、单位/时间信息及网格对齐。详情和字段模板见 [数据要求](docs/DATA_REQUIREMENTS.md)。

## 环境和安装

推荐 Python 3.11 及以上。当前执行环境为 Python 3.13.5；关键已安装版本记录在 `docs/DEVELOPMENT_LOG.md`。在项目目录创建隔离环境并安装：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

也可以执行 `make install`。Ruff 和 pytest 属于开发依赖。每个运行会记录软件版本；未锁定的依赖范围便于不同平台安装，正式论文分析运行必须保留环境快照。

## 数据导入与路径配置

本地接入可将数据放入 `data/raw/` 子目录，或通过 `config/data_paths.yaml` 映射现有目录；该映射文件被 Git 忽略，支持绝对路径。原始输入不移动、不修改。研究区与耕地掩膜需映射为统一网格，未经 CRS、transform、shape 和像元大小检查的数据不能直接叠加。

## 命令行工作流

```bash
python -m soilmatch.cli --help
python -m soilmatch.cli validate-data --config config/default.yaml
python -m soilmatch.cli build-database --config config/default.yaml
python -m soilmatch.cli screen-covariates --config config/default.yaml
python -m soilmatch.cli cluster --config config/default.yaml
python -m soilmatch.cli check-erosion --config config/default.yaml
python -m soilmatch.cli match --config config/default.yaml
python -m soilmatch.cli analyze --config config/default.yaml
python -m soilmatch.cli compare-pcnm --config config/default.yaml
python -m soilmatch.cli report --config config/default.yaml
```

真实数据首轮审计（输入只读）：

```bash
python scripts/audit_real_data.py --data-root "环境匹配原数据" --project-root . --output-dir outputs/data_audit
```

缺少真实输入时，命令应给出缺失逻辑数据项和路径，并以非零状态退出；不会自动创建研究结果或使用测试夹具。合成数据端到端流程仅在 pytest 中运行。

## 方法和质量控制

分析顺序是数据核查、环境向量、环境分组、组内侵蚀变异、精细匹配、健康响应、空间不确定性和 PCNM 对比。侵蚀主暴露为 `log1p(Aa)`，零值保留；组内侵蚀统计包括均值、中位数、标准差、P5/P25/P75/P95、零比例和等级构成。匹配使用空间索引避免全域两两距离矩阵，并输出匹配失败原因、平衡及复用信息。

栅格预测相关性受空间依赖影响。模型报告将区分预测栅格数、独立实测点数和匹配对数量，并采用可配置空间块进行不确定性评估。RUSLE/RF 共享协变量需登记。方法异常的诊断预案见 [方法说明](docs/METHODOLOGY.md)。

## 输出与图表

每轮写入新的 `outputs/run_<id>/`，包含配置快照、数据清单与软件元数据、质量诊断、表格、图件和简要报告。计划图件包括环境组地图、环境特征热图、组内侵蚀分布、匹配前后平衡图、组内回归图、配对差值图、回归森林图、指标响应热图及 PCNM 对比。真实数据到位且检查通过后才生成实证图表。

## 研究进度

项目已收到本地真实栅格数据并完成首轮清点。按用户指示直接采用预计算 Aa，不进行 RUSLE 因子乘积验证；G2 因正式模板和跨层网格未确认而阻塞，G3 暂不能批准，G4 未运行。在阶段门通过前不进行正式健康回归。原始数据、本地路径映射和生成报告均未纳入 Git。

## 开发与测试

```bash
make test
make lint
```

合成夹具只验证软件行为，不构成温县研究证据。每轮长期工作按 [工作流程](docs/WORKFLOW.md) 执行，开发事实记录于 `docs/DEVELOPMENT_LOG.md`，方法选择和修改记录于 `docs/DECISIONS.md`。本地原始数据目录通过被忽略的 `config/data_paths.yaml` 映射，不复制到 Git 仓库。

## 后续任务

请按 [数据要求](docs/DATA_REQUIREMENTS.md) 提供栅格、元数据和 PCNM 表。第一轮先盘点来源和单位，检查 CRS、分辨率、transform、shape、NoData、耕地掩膜与 RUSLE 重建关系，特别诊断 C 因子和零侵蚀；确认数据可比后才构建数据库和环境向量。详见 [研究方案](docs/RESEARCH_PLAN.md)。
