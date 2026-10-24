# 工作流与命令

## 第一阶段：工程验收

1. 按 README 安装环境；`python -m soilmatch.cli --help` 查看命令。
2. `make test` 运行单元和合成端到端测试，`make lint` 检查格式。
3. 合成测试数据只位于临时目录或 `data/fixtures/`，明确写入测试标识，不进入结果目录作为温县证据。
4. 真实数据未配置时，数据命令报告缺失逻辑文件并退出，不回退到合成输入。

## 第二阶段：真实数据流程

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
