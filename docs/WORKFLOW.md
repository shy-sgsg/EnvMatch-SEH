# 工作流与命令

## 每轮任务开始与治理

先阅读 `AGENTS.md`，随后完整阅读当前 Git 版本的 `docs/RESEARCH_CHARTER.md`。再阅读 README、本阶段门、相关研究方案/方法、开发日志和决策记录。确认 HEAD、工作区差异、研究阶段、输入数据及本轮任务对应的科学问题。总纲正文修改只能经用户明确批准，见 `docs/governance/CHARTER_CHANGE_PROTOCOL.md`。

科研阶段完成后生成固定到提交 SHA 的 ChatGPT 审查请求。ChatGPT 审查覆盖研究方向、数据、方法、结果和长期价值；测试通过不替代科研判断。审查模板见 `docs/templates/CHATGPT_REVIEW_REQUEST.md`。

## 第一阶段：工程验收

1. 按 README 安装环境；`python -m soilmatch.cli --help` 查看命令。
2. 接入真实数据后先运行 `python scripts/audit_real_data.py --data-root "环境匹配原数据" --project-root . --output-dir outputs/data_audit`，审阅映射、栅格统计、网格与协变量报告；本地绝对路径只写入 Git 忽略的 `config/data_paths.yaml`。
3. `make test` 运行单元和合成端到端测试，`make lint` 检查格式。
4. 合成测试数据只位于临时目录或 `data/fixtures/`，明确写入测试标识，不进入结果目录作为温县证据。
5. 真实数据未配置时，数据命令报告缺失逻辑文件并退出，不回退到合成输入。

## 第二阶段：真实数据流程

严格按 `docs/governance/RESEARCH_STAGE_GATES.md` 执行 G1–G4。只有在任务明确要求重算或验证 RUSLE 时才执行因子乘积校验；G2 的数据对齐/元信息问题未解决时，不运行依赖数据的正式匹配；G4 未通过并锁定设计前，不开展健康指标回归。

按顺序执行，不跳过阶段门槛：

```bash
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

每个命令写日志、配置副本和运行元数据。输入检查未通过时停止，不能继续使用不对齐数据。聚类和侵蚀检查不读取健康结果。只有环境组质量与侵蚀变化覆盖满足预设条件，才启动匹配；只有匹配质量合格，才启动健康回归。

## 结果目录

每轮分配新的 `outputs/run_<id>/`，至少包括 `config_snapshot.yaml`、`metadata.json`、`tables/`、`figures/`、`maps/`、`matched_pairs/`、`diagnostics/` 和 `report.md`。不可覆盖既有 run。报告注明数据路径逻辑名、Git revision（可用时）、软件版本、栅格数、实测点数、匹配对数、模型单位和主要质量诊断。

## 每轮维护

开始前阅读 `AGENTS.md`、`README.md`、本文件和开发日志；确认任务阶段、现存输出和遗留问题。重要方法选择及偏离写入 `docs/DECISIONS.md`，执行事实与结果写入 `docs/DEVELOPMENT_LOG.md`。不以日志替代可复现配置与代码。
