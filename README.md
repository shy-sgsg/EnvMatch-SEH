# EnvMatch-SEH：环境匹配下的土壤侵蚀—土壤健康分析

## 项目介绍

本项目为温县耕地栅格研究建立可复现的 Python 工作流，区分 `FORMAL` 正式实证研究与经授权的 `EXPLORATORY_ORACLE` 内部探索。正式模式按研究总纲阶段门开展条件关联与环境异质性分析；Oracle 模式允许结果导向的匹配、子集和模型搜索，但其表观结果不作为正式结论。

研究方向以 [研究总纲](docs/RESEARCH_CHARTER.md) 为准。首版总纲提交后，其正文受审批机制保护；科研任务按 [治理协议](docs/governance/CODEX_TASK_PROTOCOL.md) 和 [阶段门](docs/governance/RESEARCH_STAGE_GATES.md) 执行。ChatGPT 阶段审查协议和总纲变更申请流程分别见 `docs/governance/`。

## 研究背景与研究区

研究区为河南省焦作市温县耕地，既有研究的耕地面积约 347.877 km²，空间分辨率为 30 m。当前分析直接使用用户提供的预计算实际侵蚀 `Aa_20yr` 与 `Aa_2020`，不重建或验证 RUSLE 因子乘积；仍记录输入栅格可读性、范围和空间对齐情况。正式模式的其他质量阶段门按总纲执行。已有土壤健康制图以 AN、OM、CEC、pH、SW、BD、MBC、RESP 八项指标为基础，随机森林预测栅格经隶属度和权重形成 SHI。已有 PCNM 空间控制成果予以保留；本轮 Oracle 探索不开展 PCNM 比较。

## 研究目标与科学问题

主题是基于环境协变量匹配的土壤侵蚀—土壤健康条件关联及环境异质性分析。核心问题包括：环境相似时关联是否存在；关联是否随环境背景变化；物理、化学和生物指标响应是否不同；结果与 PCNM 空间控制是否一致；如何提高匹配质量与统计推断可靠性。

## 新增研究内容

1. 以 Elevation、Cur、TWI、LST 等非 RUSLE 直接因子构建环境向量，按土壤类型分层，评估环境相似性。
2. `FORMAL` 模式以环境质量选择分组；Oracle 模式可根据健康响应探索组数和匹配方案，并单独登记结果导向筛选。
3. 在组内检查 `log1p(Aa)` 分布、分位数、零比例和有效侵蚀变化，确认有可比较的暴露范围。
4. 在同环境组、相同土壤类型约束下做环境近邻匹配，考察 300 m、500 m、1000 m 空间半径，记录环境距离、空间距离、侵蚀差值和平衡诊断。
5. `FORMAL` 设计锁定后开展健康响应分析；Oracle 模式开展全样本基线、结果导向表观上限及独立空间测试三类分析。
6. 既有 PCNM 结果予以保留；本轮 Oracle 探索不导入或比较 PCNM，Q4 标记为 `DEFERRED`。

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

`FORMAL` 分组和配对不使用健康结果；`EXPLORATORY_ORACLE` 允许结果导向选择，但须写入独立注册表。两种模式均不得把预测栅格像元数称为独立实测样本数，且应讨论环境与健康制图输入重叠带来的限制。

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
experiments/oracle_exploration/  Oracle 搜索配置、代码和实验接口
outputs/exploratory_oracle/      与正式运行隔离的 Oracle 结果根目录
logs/                     运行日志
```

原始栅格、数据库、图件和报告默认不纳入普通 Git 版本管理；源码、文档、测试、模板配置需纳入。

## 数据准备清单

当前接入了预计算实际侵蚀栅格、八项健康指标及多组环境候选层。按用户指示直接使用 Aa，不重建或验证 RUSLE 因子乘积。Oracle 复用已有对齐数据库，并检查读取、变量范围和空间对齐；未知单位或时间元数据作为限制披露，不单独阻止本轮探索。首轮审计报告和本地路径映射位于 `outputs/data_audit/` 与 `config/data_paths.yaml`；两处均由 `.gitignore` 排除。

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

正式模式的真实数据首轮审计（输入只读）：

```bash
python scripts/audit_real_data.py --data-root "环境匹配原数据" --project-root . --output-dir outputs/data_audit
```

缺少真实输入时，命令应给出缺失逻辑数据项和路径，并以非零状态退出；不会自动创建研究结果或使用测试夹具。合成数据端到端流程仅在 pytest 中运行。

## 方法和质量控制

分析顺序是数据核查、环境向量、环境分组、组内侵蚀变异、精细匹配、健康响应、空间不确定性和 PCNM 对比。侵蚀主暴露为 `log1p(Aa)`，零值保留；组内侵蚀统计包括均值、中位数、标准差、P5/P25/P75/P95、零比例和等级构成。匹配使用空间索引避免全域两两距离矩阵，并输出匹配失败原因、平衡及复用信息。

栅格预测相关性受空间依赖影响。模型报告将区分预测栅格数、独立实测点数和匹配对数量，并采用可配置空间块进行不确定性评估。RUSLE/RF 共享协变量需登记。方法异常的诊断预案见 [方法说明](docs/METHODOLOGY.md)。

## 输出与图表

正式运行写入新的 `outputs/run_<id>/`；Oracle 只写入 `outputs/exploratory_oracle/`，不得混入正式结果。每次运行保留配置、输入 run ID、Git SHA、软件版本、实验记录、诊断和结果表。Oracle 报告分别呈现全样本、结果导向表观和空间测试表现。

## 研究进度

正式模式的 G2/G3 状态仍按现有审查记录管理。用户已授权独立 Oracle 探索；当前基于已有统一数据库建立实验框架，Oracle 结果尚待执行。原始数据、本地路径映射和大型运行输出均不纳入 Git。本轮不开展 PCNM。

## 开发与测试

```bash
make test
make lint
```

合成夹具只验证软件行为，不构成温县研究证据。每轮长期工作按 [工作流程](docs/WORKFLOW.md) 执行，开发事实记录于 `docs/DEVELOPMENT_LOG.md`，方法选择和修改记录于 `docs/DECISIONS.md`。本地原始数据目录通过被忽略的 `config/data_paths.yaml` 映射，不复制到 Git 仓库。

## 后续任务

正式研究的后续输入和阶段门仍按 [数据要求](docs/DATA_REQUIREMENTS.md) 与 [研究方案](docs/RESEARCH_PLAN.md) 管理。Oracle 本轮使用现有对齐数据库及已计算 Aa，不做 PCNM 或 RUSLE 因子乘积验证。
