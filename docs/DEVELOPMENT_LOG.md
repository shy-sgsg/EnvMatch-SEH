# 开发日志

## 2026-10-24：第一阶段初始化开始

- 研究阶段：项目初始化；未收到温县正式栅格数据或 PCNM 结果。
- 初始工作区检查：目录仅有既有文献 PDF 和一个 Git 无法识别的空 `.git` 目录；没有源码、项目配置或测试。
- 执行环境：Python 3.13.5；NumPy 2.1.3；pandas 2.2.3；SciPy 1.15.3；scikit-learn 1.6.1；statsmodels 0.14.4；rasterio 1.5.0；PyArrow 19.0.0；PyYAML 6.0.2；Matplotlib 3.10.0；pytest 8.3.4；openpyxl 3.1.5。GeoPandas、Fiona、Shapely、Ruff 未安装。
- GitHub 目标地址由用户提供：`git@github.com:shy-sgsg/EnvMatch-SEH.git`。当前环境 SSH 配置权限异常且外网 DNS/套接字访问受限，无法读取远程分支或推送。不得将本地空目录初始化为与远程历史无关的仓库；待可访问远程后再确认分支并提交推送。
- 当前记录只描述启动检查，不代表项目初始化、自动化测试或远程提交已完成。后续在本文件追加实际完成与验证证据。

## 2026-10-24：初始化实现与验收

- 已建立中文研究规范、README、研究方案、方法说明、数据要求、工作流程、决策记录、配置模板、依赖配置、CLI、核心分析模块和合成数据测试。
- 已实现栅格质量核查与对齐、分块 Parquet 栅格数据库、环境变量筛选与距离、分层环境聚类、组内侵蚀摘要、环境/空间约束近邻匹配、平衡诊断、组内与配对回归、空间块重采样接口、PCNM 表整理、九类基础图表及运行报告。
- 验收证据：`python -m pytest -q` 为 44 项通过；`python -m compileall -q src tests` 通过。校验失败时也会保存质量 CSV 和诊断报告，然后以非零状态退出。
- 未收到正式温县栅格、边界或 PCNM 结果；本阶段没有生成实证结论。测试数据仅为合成数据。
- Git 提交/推送仍未完成。当前 `.git` 是权限为只读的空目录，`git rev-parse` 无法识别仓库；工作区未提供 GitHub 写入工具或 `gh` 命令，HTTPS 套接字受限，绕过 SSH 配置后 `github.com` DNS 解析失败。不得在本工作区制造与目标远端历史无关的提交。恢复到可写的目标仓库 checkout 并提供远端访问后，应先确认远端分支和历史，再提交、推送并记录 SHA。

## 2026-10-25：章程治理与真实数据首轮审计

- 项目总纲、Codex/ChatGPT/总纲变更协议、G0–G5 阶段门及报告模板已创建；`RESEARCH_CHARTER.md` 首版正文未在创建后修改。治理文档尚未提交，当前执行环境对 `.git` 元数据无写权限；用户已收到本地提交/推送命令。
- 任务开始时仓库已存在：`main` 跟踪 `origin/main`，起始提交为 `97b1381aa1fb14be94ed672c73c67855c973ecf4`。当前 sandbox 拒绝写入 `.git` 引用；未在此环境提交或推送。
- 原始数据目录只读盘点：312 个文件、200,403,374 bytes；77 个 GeoTIFF/ArcGrid 栅格数据集均可由 rasterio 打开。发现 9 组、32 个完全字节重复文件条目，主要为 world file/元数据副本；NDVI 输出与一个临时中间 TIFF 字节相同。
- 两份多边形 `.shp` 缺少 `.shx`、`.dbf` 和 `.prj`；不能验证要素数、属性表、完整几何或投影。实际耕地栅格候选为 1240×977、30 m、EPSG:32649，392,527 有效像元且值均为 12。
- 按用户指示，`实际侵蚀_wx.tif` 直接作为 `Aa`，未重建 RUSLE。该文件为 1130×845、30 m、EPSG:32649，529,036 个有效像元，零值占 57.04%，数值范围 0–238.155；单位和年份元信息缺失。
- 77 个栅格数据集分属多个不同 CRS/分辨率/transform 网格；候选环境 2020 图层内部精确对齐（1138×850、30 m、536,477 有效像元），但与 Aa、耕地掩膜和地形层不对齐。NDVI 源分辨率为 500 m，并已双线性重采样至 30 m。土壤类型栅格出现 26/27/42/85 四类，与类别频数 CSV 不符。
- G0 文档已准备；G1 盘点与本地路径映射完成；G2 因正式 DEM/模板、网格对齐和 Aa 单位/时间元信息缺失而 BLOCKED；G3 暂不能批准环境向量；G4 未运行，健康回归未启动。
- 首轮审计脚本和报告位于 `scripts/audit_real_data.py`、Git 忽略的 `outputs/data_audit/` 与 `config/data_paths.yaml`。原始数据未编辑或纳入 Git。

## 2026-10-25：温县耕地 G2–G4 可行性分析

- 用户确认研究范围为温县耕地；`实际侵蚀_wx.tif` 为 20 年平均 `Aa_20yr`，`2020年A` 为单年 `Aa_2020`。以 `SHI/温县耕地.tif` 的值 12 建立耕地掩膜，并使用其 EPSG:32649、30 m 网格作为共同模板。未重建或校验 RUSLE 因子乘积，原始数据未改写。
- G2 对齐流程分别将两种侵蚀栅格以面积平均（主版本）和最近邻（敏感性）重采样；连续环境和健康预测面双线性重采样，SoilType 最近邻。输出环境—侵蚀库与健康库分开保存，G4 只读取环境—侵蚀库。
- 耕地掩膜有效像元 392,527 个，栅格面积 353.2743 km²；旧总纲面积 347.877 km²，差 5.3973 km²（1.55%）。耕地 `.shp` 几何面积若假定坐标为米为 353.5793 km²，但 `.prj/.shx/.dbf` 缺失，投影和属性不可确认；面积差异保持待查。
- Aa_20yr 与 Aa_2020 面积平均栅格共同有效的耕地像元 387,596 个，Pearson r=0.5366、Spearman rho=0.4330；零值比例分别为 34.28% 和 1.41%，其中 132,664 个像元 Aa_20yr 为零而 Aa_2020 为正。该差异仅作为时间/数据一致性诊断；精确起止年份、单位及平均构造仍无元数据。
- G3 探索性核心环境向量为 Elevation、Cur、TWI、LST，SoilType 26/27/42 分层；85 类水体未进入分组。双侵蚀和核心环境完整样本为 287,785 个耕地像元（73.32%）。其他环境层仅列入数据筛查，没有进入主匹配向量。
- G4 按每种土类 6/8/10/12 组检查；silhouette 为 -0.0147 至 0.0019，随机种子稳定性 ARI 为 0.381–0.857，尚无充分证据锁定分组。500 m、k=8 基线下，Aa_20yr 匹配 103,586 对、潜在高暴露单元覆盖 62.7%、最大绝对匹配后 SMD=0.0203；Aa_2020 匹配 114,399 对、覆盖 53.5%、最大绝对 SMD=0.0320。300/1000 m、环境卡尺、侵蚀秩差及最近邻重采样敏感性均已保存。配对规模可行，但这不构成健康效应证据或设计锁定。
- 健康库包含八项指标预测面及暂定 `SHI_candidate`，没有读入 G4；预测模型训练/验证和 SHI 来源尚需补档。未发现可供本轮比较的 PCNM 结果，未开展 PCNM 比较或 G5 健康回归。
- 阶段结论：G1 完成；G2/G3 为 WARNING；G4 为探索性完成、未锁定；G5 等待 ChatGPT/用户审查匹配设计并补足关键元数据后再决定。总纲未修改。
- 可复现输出在 Git 忽略目录 `outputs/run_20261025T114741_470772Z/`；新脚本为 `scripts/run_cropland_analysis.py`，本轮新增代码通过 `python -m pytest -q`（60 项）和 `python -m compileall -q src tests scripts`。Ruff 未安装，未运行。
- 运行清单记录的代码版本为基线提交 `88df3786eeff83561a7dffa5b3eaabfeb4bb2191`；本轮代码和审阅报告仍待用户本地提交、推送，之后应把推送 SHA 补入 ChatGPT 审查请求。

## 2026-10-26：EnvMatch-SEH Oracle 上限实验

- 使用已有输入运行 `run_20261025T114741_470772Z` 的耕地与健康 Parquet 数据，392,527 个像元；Stage 1 使用 40,000 个训练区分层像元，Aa_20yr 与 Aa_2020 各搜索 100 个候选；Stage 2 对每种暴露最多保留 20 个配置，在 100,000 个像元上形成 80 个训练/验证角色试验；Stage 3 对 8 个匹配配置和 18 个验证集选择的空间模型配置使用留出空间区评分。
- Oracle 输出位于 `outputs/exploratory_oracle/oracle_20261026T041521459615Z_20261026/`，包含 4,030 条唯一注册记录、10,616 行展平结果、1,024 个模型试验、2,448 个子群试验、252 个样本保留试验、4 张排行榜和 10 张 PNG 图。子群中 133 个失败试验均保留原因；注册表没有重复 `experiment_id`。其中 2,430 个子群试验在范围内；早期搜索产生的 18 个 SoilType=85（水体类）试验仍保留在注册表供审计，但已从子群表和 Oracle 排行榜排除，并记录在 `subgroups/scope_exclusions.json`。
- SHI_candidate 的最高全样本表观 R² 为 0.6369（Aa_20yr，n=384,735）；事后 Oracle 样本保留搜索最高表观 R² 为 0.9977（保留 5%，19,237 个完整案例）；按验证集选择的 SHI 空间模型在独立测试区 Aa_20yr R²=0.4703（验证 R²=-0.0692，n=51,533），Aa_2020 的对应结果为 R²=0.2334（验证 R²=-0.1433，n=51,447）。范围内最佳显式子群为 environment_group_6（Aa_20yr/SHI_candidate，R²=0.5502，n=8,307）。Oracle 结果是结果导向筛选后的表观拟合，不是独立验证或因果效应。
- 测试阶段修正报告问题：排行榜去重不能对嵌套列表/字典列使用全行哈希；空间模型排行榜不得把 Stage 3 匹配分数混作模型分数；共同样本 n 按健康结局不同，不能将 SHI 的 n 推广到所有结局；SoilType 85 水体类不得作为有效子群。补齐续跑时遗漏的 SHI 全样本模型曲线后，完整 pytest 为 116 项通过，`compileall` 与 `git diff --check` 无错误；对注册表、输出行数、唯一 ID、范围排除、排行榜及图表数量完成核对。
- 模型搜索期间 statsmodels Quantile Regression 发出 100 次迭代上限警告，scikit-learn Huber 发出 LBFGS 迭代上限收敛警告。警告未中断运行，但注册表没有单独的 warning 字段；这两类模型族结果应谨慎解释。
- 续跑时原有 800 行曲线缓存只含八项健康指标；现按 SHI_candidate 的全样本最佳模型补拟合 Aa_20yr 曲线后共 900 行。图 08 的九个结局均同时展示固定绘图样本自然样条和全样本最佳模型曲线；图 04 展示环境卡尺、地理半径和侵蚀分位差与 SHI 表观 R² 的关系。
- 完整汇报在 `docs/reviews/2026-10-26/oracle-exploration-review.md`，报告目录附 10 张图、实验摘要、模型参数 JSON、4 张排行榜 CSV 和 SHI 匹配候选表。15 MB 的完整注册表及其余大体积运行结果仍留在 Git 忽略的 outputs 目录。`Aa_20yr` 和 `Aa_2020` 直接使用既有侵蚀值，本轮没有重算或验证 RUSLE；PCNM 未运行。所有输出均与正式分析分离。
- 当前 HEAD 为 `c50bd9ae6a42abc14f898e8043a79cb0c9c818e4`，工作区有改动；本环境 `.git` 只读，未提交或推送。保留三个提交分组供用户本地执行。


## 2026-10-26：Phase F 配对差值与重复空间验证

- 探索模式：`EXPLORATORY_ORACLE`；基线 SHA `0bb1b12fcab24619b3ce8e03f0871a54744be0b4`。Phase F 运行 `phase_f_20261026_seed20261026`，输入 `outputs/exploratory_oracle/phase_f_20261026_seed20261026`；当前改动尚未提交，`.git` 只读。
- 使用已有对齐 Parquet 中的 `Aa_20yr` 和 `Aa_2020`；未重算/校验 RUSLE，未运行 PCNM，未修改研究总纲。新增 outcome-blind 局部匹配搜索、M0–M4 比较、成对差值 P1–P4、10 个 5km 空间种子、嵌套 ΔR²、环境异质性、两期对照和敏感性分析。
- 本次 Oracle 运行标记 `outcome_guided_selection=true`。高响应组按完整样本均值排序，18 个暴露—结局组合共比较 324 组候选、保留前三共 54 组；每组保留完整案例的比例为 6.7%–31.7%（组合中位数 13.7%）。图 8 继承的全样本曲线按表观 R² 从 13 类模型共 1,024 次模型试验中选出，逐结局样本量和每个组合保留率见运行 provenance、`high_response_environment_groups.csv` 与 Figure 8 数据。
- 报告与审阅包：`docs/reviews/2026-10-26/PHASE_F_RESEARCH_REPORT.md`、`docs/reviews/2026-10-26/PHASE_F_CHATGPT_REVIEW_REQUEST.md`；图表为 PNG/SVG/PDF，表格和图注均另存。
- 证据与边界：运行清单保存输入 SHA-256、配置/依赖版本及模型设置；配对端点按空间角色独立构造。完整 pair Parquet 保留在 Git 忽略的 `outputs/exploratory_oracle/`；结果仍是内部重复空间验证和预测面关联，不作因果/外部验证结论。
- 未提交、未推送。等待用户本地提交并交 ChatGPT 审阅；Phase F 审阅前不推进正式下一阶段。

## 2026-10-27：Phase G 环境异质性与候选结果审查包

- 运行 `phase_g_20261027`，模式为 `EXPLORATORY_ORACLE` 且显式设置 `outcome_guided_selection=true`；运行基线 HEAD 为 `f71aaf455e5ca9eff36dfedc0461a09125ff3d34`。复用 392,527 个对齐像元、383,714 个主土类像元、Phase F 的 407,372 个 M3/M4 配对及历史 Oracle 的 10,616 条试验登记。使用预计算 Aa_20yr/Aa_2020；未重建或验证 RUSLE 因子乘积，未重跑 PCNM，未修改研究总纲。
- 生成 144 条全配对差值关联、5,012 条候选模型记录、122 条模型/关联排行榜记录、4,644 条环境组关联和 1,620 条验证集选择的组空间检验。环境定义包括 SoilType、Phase F V5 K-means、土类内 Ward 聚类，以及 Elevation、TWI、LST、NDWI、NDVI、Precipitation 六项土类内分位组。配对高低端按 5 km 空间块 bootstrap 计算区间。
- 从 Phase F 重复空间表只选 M3/M4 的基础 P1 截距 OLS 模型，保留两个 Aa 时段、九项结局和 validation/test 分工，共 720 行、72 个空间汇总组。绘图抽取 Aa_20yr 的 180 条测试记录。初检发现此前筛选条件误取 M0 像元基线且遗漏匹配配对；已用测试锁定 P1/M3/M4 选择与训练斜率方向摘要，更新排行榜、登记表、报告、表格和图 9。
- 首轮长计算期间终端没有阶段输出；确认 20 张结果表、运行清单、报告和九图目录均写出后结束了仍存活的会话。后续空间汇总、图 9 和审查材料均基于已保存数据重新生成并通过完整性检查；运行脚本现会输出阶段进度。
- 图 5 同时展示固定 20,000 像元共同样本自然样条和历史 Oracle 最佳全样本表观模型曲线，并区分两者的样本范围与选择方式。九张图均输出 PNG/SVG/PDF；ChatGPT 审查目录包含图注、图源表、完整结果 CSV 副本和审查请求，共 72 个文件。汇报指出健康层是预测面、Aa 单位/时期元信息尚缺，结果不作因果或独立实测验证解释。
- 报告将 Wilcoxon P<0.001 显示为阈值格式，避免舍入显示成 0.000。验证：`pytest -q` 为 122 项通过；`python -m compileall -q src experiments scripts tests` 和 `git diff --check` 通过；产物检查确认 9 张图的三种格式、72 个审查文件、两种曲线类型、降水分组和 M3/M4 空间记录均齐全。Ruff 未安装，未运行。
- 报告及审查请求位于 `docs/reviews/2026-10-27/phase-g/`，运行数据位于 Git 忽略的 `outputs/exploratory_oracle/phase_g_20261027/`。当前未提交、未推送；审查请求中的实际审查 SHA 需在首个 Phase G 成果提交后填写，再单独提交该请求。

## 2026-10-28：Phase G completion 补充结果

- 运行 `phase_g_20261027_completion`（`EXPLORATORY_ORACLE`，`outcome_guided_selection=true`），以 2026-10-27 Phase G 提交 `efdcdcd313f57aae1587466f9e0115a2846bec5e` 为起始 HEAD。复用 392,527 个已对齐像元、407,372 个 Phase F 配对和既有 Oracle 候选登记；继续使用预计算 Aa，不重算/验证 RUSLE 因子乘积、不重跑 PCNM、不修改总纲。
- 更新耕地范围候选榜单并排除 18 条 SoilType 85 水体结果；对 17 个可能改变负向排序的候选重建固定样本树模型方向，最大绝对 R² 复现差 1.67e-16。新增 331 条配对响应模型记录、2,880 条 validation 选择的空间模型记录、504 项环境组交互检验、8,964 条组内斜率和 996 条环境剖面记录。
- 全局 BH 校正 504 项交互检验中 473 项 q<0.05；M4/Aa_20yr 为 123/126。极端 Wald 尾部改用稳定 log survival 计算，报告展示 log(P)/log(q)。空间边界诊断发现 52,601/407,372 对两端跨越 5 km 网格（12.9%）；模型仍按配对中点分块。
- 补充报告、审查请求、校正后结果表和图 10 位于 `docs/reviews/2026-10-27/phase-g/`；运行明细和清单在 Git 忽略的 `outputs/exploratory_oracle/phase_g_20261027_completion/`。图 5 同时提供固定样本自然样条与全样本表观最佳模型曲线。当前等待用户本地提交、推送及 ChatGPT 审查；未发送外部材料，也未进入 FORMAL G5。
- 验证：`pytest -q` 130 项通过；`python -m compileall -q src experiments scripts tests` 通过。完整审查提交 SHA 保持待用户提交后填写。

## 2026-10-29：低侵蚀响应补充、中文图集与土类标注

- 继续使用既有低侵蚀分布诊断运行 `low_erosion_regime_20261029_seed20261029_r2`；新增独立运行 `low_erosion_matched_units_20261029_seed20261029_r1`，分别按唯一匹配端点像元和匹配配对差值分析，并比较线性、二次、自然样条、GAM 近似、Huber、随机森林、极端随机树和直方图梯度提升八种响应模型。
- 补充运行写出 2,880 条模型试验、360 条匹配像元秩关联、720 条 M4 空间测试记录（10 次 5 km 整块划分、训练 500 m 缓冲）及 5 张 PNG/SVG/PDF 图。M4 CDF 阈值组分别为 Aa_20yr 3,653 对/7,306 个唯一端点、Aa_2020 42,772 对/85,544 个唯一端点。M3 未在此补充运行中重做空间测试。
- 补充报告将分析单位、结局、模型族和表头改为中文；主低侵蚀报告、Phase H 索引及复核请求补入补充运行链接与主要响应摘要。报告继续将表观模型搜索和内部空间测试分开描述。
- 土壤原始类别属性表已核对：26=冲积土、27=潮土、42=脱潮土。对应中文名称已加入 Phase F/G/H 报告、地图图例和中文科研图集页首说明。
- 环境组聚类图保留新版分面图；另依据未改动的原始聚类 GeoTIFF，单独保存昨日全域布局与配色的复原图。该文件是按旧样式重绘的复原版，不宣称与旧 PNG 逐像素相同。
- 图集现同时收录 Phase F/G/H、低侵蚀运行、匹配单位补充运行，以及新版和旧样式复原聚类图；提供按阶段、科学问题、实验和图类型筛选及源表链接。所有新补充图继续保留中文标题、图注和土类名称。

## 2026-10-30：环境组异质性、空间支持与 Phase H 审查材料

- 新增探索运行 `low_erosion_group_heterogeneity_20261030_seed20261030_r2`：沿用低侵蚀分布候选阈值、既有环境分组标签及 M3/M4 配对来源，完成 11 种分组下的像元保留、组内匹配支持、九项健康响应比较、M4 的 10 次 5 km 整块空间测试（训练缓冲 500 m）及 5 km 空间块斜率 Bootstrap（199 次）。未用健康结果重建环境组；Oracle 运行保持 `outcome_guided_selection=true` 标记。
- 修订组级匹配支持表：连续环境平衡由配对两端的环境值重算，逐变量绝对差以全有效耕地像元 SD 标准化后汇总；匹配源表的 `environmental_distance` 作为另一统计量单独保留。已修复所有重配方案缺失分项环境差字段的问题，并重新生成支持表、报告及 HET 图。
- 生成 12 张环境异质性图（PNG/SVG/PDF），HET05 显示五种匹配群体方案；在土类、K-means、Ward、TWI 和 LST 标签中标清 26 冲积土、27 潮土、42 脱潮土。交互式图集重新构建为 80 张图，最终包含 369 个本地文件链接，完整性检查为 0 个缺失。
- 新增 `PHASE_H_FINAL_SUMMARY.md`，逐项回答任务清单的 20 个问题；更新低侵蚀主报告、结果索引和 ChatGPT 复核请求。复核请求以用户指定句子开头，提交 SHA 留待用户本地提交后填入；没有代用户向外发送审查材料。
- 验证：最终 `pytest -q` 为 197 项通过、2 条 seaborn PendingDeprecationWarning；`python -m compileall -q src experiments scripts tests`、`git diff --check` 均通过；图集 80 张卡片、369 个文件链接全部存在。未执行提交或推送，用户将使用本地命令。
- 主结论：经验 CDF 折点在 100 次空间分块重抽样中稳定，但 KDE 无可靠谷值、GMM 分量重叠大；低值剔除改变研究总体，关联和空间表现未一致增强。报告建议保留全样本，以零值状态加正侵蚀强度的灵活模型为主，并将 CDF 阈值作为敏感性方案。所有结果仍为 Oracle 探索，不构成正式确认。
- 图集与审查材料后续修订：将 80 张图、Phase H 图源、初始聚类空间图（包括昨天原版样式复原版）、精选结果表和补充报告归档到 `docs/reviews/2026-10-29/phase-h/artifacts/`，使图集不再链接 Git 忽略的 `outputs/`；大 CSV 证据表改为 `.csv.gz`。审查包共 309 个文件、95.13 MiB；重新核对 HTML 的 80 张卡片和 369 个本地链接均无缺失、无 `outputs/` 链接。

## 2026-10-30：低侵蚀分布诊断补充与图集更新

- 新增独立 Oracle 运行 `low_erosion_distribution_addendum_20261030_seed20261030_r1`。只读取已对齐数据库中的 Aa 与既有分布诊断，没有读取健康结局；配置保留 `outcome_guided_selection=true` 运行标记，并记录本补充未使用健康结果择优。每期以 10 个固定种子、每次 20,000 个按 5 km 空间块分层抽取的正值样本重拟合 GMM K=1–4；另比较 0.80、0.90、0.95 后验分类阈值。
- 新增完整正值分位数、0/1/2/3/6/9/12 位小数重复值频率、逐种子 GMM 参数、后验分类计数和包含全部方法/失败状态的 29 行统一候选阈值表。20 年平均 Aa 的 BIC 在 10 次拟合中选择 K=3 九次、K=2 一次；2020 年 10 次均选 K=4，但首个有限分界的种子间 CV 为 72.2%，后验不确定比例为 54.1%–92.6%。这些结果不把 GMM 边界提升为客观阈值。
- 从已保存表格生成中文补充报告和 3 张 PNG/SVG/PDF 图（GMM 种子稳定性、后验阈值敏感性、方法阈值比较），均附中文图注和源 CSV。完整材料归档到 `docs/reviews/2026-10-29/phase-h/artifacts/distribution_diagnostic_addendum/`。
- 复核到 Aa_2020 CDF 阈值完整精度为 0.3966915309429168，六位小数显示为 0.396692。用显示值复算会多算 1 个被筛除像元；摘要与主报告现按完整精度修正为保留 260,974 格、筛除 33,269 格。
- 中文离线图集更新至 83 张；审查包更新至 333 个文件、100,731,624 字节（96.07 MiB）。最终核对确认 384 个本地文件链接全部存在，且无 `outputs/` 链接；昨天原版聚类空间图仍在审查包中。
- 验证：`pytest -q` 为 203 项通过、2 条 seaborn `PendingDeprecationWarning`；`python -m compileall -q src experiments scripts tests` 和 `git diff --check` 通过。阈值表 29 行、GMM 种子试验 80 行，且 10 个种子/期、三档后验阈值均已写入审查包。

## 2026-10-31：Phase I-1 全样本响应探索

- 在 `EXPLORATORY_ORACLE` 下完成九项健康预测指标与 Aa_20yr/Aa_2020 的全掩膜完整案例关联、零/正值差异、5 km 空间块模型搜索、分位数回归及全样本 GAM。Aa_20yr 主分析以全部耕地掩膜行逐结局完整案例计算；不把要求全环境变量完整的 `valid_cell` 标记误作全局筛选条件。SoilType=85 留在主分析并提供排除敏感性。
- 权威运行 `outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_r2/`；r2 配置最大分位数迭代数为 1,000，45/45 分位数拟合状态为 `ok`。全样本 GAM 对九项结局输出按 5 km 块聚类稳健点态区间、有效自由度与局部斜率；全部 8 组图件保存 PNG/SVG/PDF。
- 主要描述性结果：Aa_20yr 对 SW 的 Spearman ρ=-0.584，对 SHI=-0.469、OM=-0.415；对 CEC 为 +0.168。零/正侵蚀 Hedges' g 中 SW、SHI、OM 的绝对值约为 1.16、0.89、0.65。空间测试 R²跨结局约 -0.072 至 0.317，不把像元量作为独立验证证据。
- 复用既有对齐数据库和 Phase H 结果做范围核对；代码增加环境与健康数据库按 cell_id 的 x/y 坐标对齐检查。未重建/验证 RUSLE 因子乘积、未重跑 PCNM、未修改研究总纲。配置、输入路径、软件版本、模型边界及限制见报告和 `manifests/provenance.json`。
- 尚未完成 Phase I 全部阶段；I-2 起按实施计划继续，最终阶段仍需集成报告和审查材料。

## 2026-10-31：Phase I-2 M4 匹配条件关联

- 复用 Phase H H2 已保存的 M4 stage2 配对候选和 864 条候选模型试验，未重做匹配或重复拟合模型。输出区分唯一匹配端点像元与配对高减低差值，并按全侵蚀范围、零—正、双正值分别计算 Spearman、Kendall τ-b、均值/中位数差、标准化斜率和负向配对比例。
- Aa_20yr M4 主要候选 `stage2_Aa_20yr_01` 共 111,289 对（零—正 68,172；双正 43,117）。Aa_2020 保留三种 H2 M4 候选；候选 01/03 全为双正，候选 06 含零—正 3,306 对和双正 114,484 对，故分组结果分别标记候选，不混并为同一样本。
- 对 Aa_20yr 主候选，SW 配对差值 Spearman 为 −0.272（全范围）、−0.326（零—正）和 −0.296（双正）；SW 负向配对比例依次为 77.8%、80.5%、73.5%。匹配端点 SHI/OM/SW Spearman 为 −0.452/−0.416/−0.567，与 I-1 全样本 −0.469/−0.415/−0.584 接近；端点关联系统上不等于配对差值关系。
- 复用 H2 表观模型排行榜：Aa_20yr 全范围 Matched-Pixel SHI 随机森林 R²=0.505，Matched-Pair SW 随机森林 R²=0.101。零—正与双正子组仅保留 OLS；不把同样本 R²称为泛化性能。XGBoost 未安装，未新增依赖。生成匹配关联表、模型排行榜、分箱响应曲线及 PNG/SVG/PDF 图。
- 权威运行 `outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_i2_r2/`；r1 作为缺少全样本对照表的初版保留并标记预览。报告/清单同时披露候选设计、Oracle 筛选、预测栅格单位、软件版本和模型限制。
- I-3 至 I-9 尚未完成，继续按 Phase I 优先级推进。

## 2026-10-31：Phase I-3 环境异质性整合

- 新增摘要运行 `outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_i3_r2/`，以 Phase G 全配对 M4 环境组结果和 Phase H 低值过滤敏感性为输入；未重跑匹配、聚类或模型。
- 汇总土类、TWI、NDVI、NDWI、LST 及其交互响应。M4/Aa_20yr 中，27 潮土 SHI/OM/SW 标准化斜率为 −0.246/−0.294/−0.316，126 个交互中 123 个经 BH 校正 q<0.05；报告保留 Oracle 选择与空间依赖限制。
- 将 Phase H 26:q1 低 TWI 组明确标为不同总体的敏感性候选：每组约 601 对（SHI 完整案例 593），同块 bootstrap 使用 16 个 5 km 块，不能与全配对 27:q2 直接作优劣比较。
- 生成环境组斜率/交互/支持表、模型与低值敏感性表及 3 组 PNG/SVG/PDF 图；r2 更正支持数量和热图舍入标签，r1 保留为非权威初稿。详细来源、字段和运行状态见报告与 `manifests/provenance.json`。
- I-1 至 I-3 已完成；I-4 至 I-9 继续按实施计划推进。

## 2026-10-31：Phase I-4 空间变化系数探索

- 新增 `experiments/oracle_exploration/phase_i_spatial.py` 与配置 `config/phase_i_spatial.yaml`，完成 `EXPLORATORY_ORACLE` I-4。权威输出为 `outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_i4_r3/`；r1/r2 状态已标为被后续带宽/制图版本取代。
- 全掩膜有效行 392,527；环境变量共同完整的 Aa 行 294,854。对九项指标拟合全掩膜 GWR、共同环境子集 GWR、环境调整 GWR 与 GWR-LCR。采用约 240 m 汇总节点、像元频数权重、自适应高斯核；带宽以抽取节点 leave-one-unit-out 误差选择，30–600 邻域候选中 54 个标准 GWR 拟合有 31 个触及下界。
- SHI/OM/SW 环境调整 GWR 的局部平均标准化斜率为 −0.330/−0.556/−0.680，LCR 为 −0.218/−0.387/−0.464；全掩膜未调整 GWR 为 −0.360/−0.645/−0.731。环境局部条件数偏高且 LCR 所有位置均惩罚，故收缩结果作为敏感性比较。
- 复用 111,289 对 Aa_20yr M4 配对，以配对中点分别建全配对、零—正和双正空间模型。SHI/OM/SW 组内局部平均标准化斜率依次为：全配对 −0.261/−0.164/−0.424，零—正 −0.344/−0.557/−0.567，双正 −0.337/−0.348/−0.470。不同配对总体不合并解释。
- 为 SHI/OM/SW 输出逐变量自适应尺度 backfitting 近似；专用 `mgwr`/`spglm` 未安装，因此明确不报告标准 MGWR。多尺度近似有 6 个变量取候选下界、11 个取上界，带宽结果仅作探索。局部标准误/区间使用模型协方差、未作空间块稳健校正。
- 生成九指标局部系数与匹配局部系数 Parquet、模型/带宽排行榜、变量带宽表、局部岭摘要、SHI/OM/SW 的 PNG/SVG/PDF 系数地图及多尺度带宽图。未修改研究总纲，未重算/验证 RUSLE 因子乘积、未重跑 PCNM。
- I-4 已完成；I-5 至 I-9 尚待推进。

## 2026-10-31：Phase I-5 空间机器学习与非线性响应

- 新增 `experiments/oracle_exploration/phase_i_spatial_ml.py` 与 `config/phase_i_spatial_ml.yaml`，权威运行 `outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_i5_r1/`。覆盖九项指标的地理加权随机森林局部预测和 Aa 有限差分效应；对 SHI/OM/SW 输出局部 TreeSHAP、Aa×TWI/Aa×NDVI 交互与 ALE 曲线。
- 采用约 240 m 空间汇总单元、局部随机留出及 5 km SVC-GAM 空间块测试。随机森林局部 R²受邻近训练/验证单元依赖影响；SVC-GAM 为 B-spline 张量积加 Ridge 的近似实现。SHI/OM/SW 的局部 Aa 中位边际效应为 −0.144/−0.263/−0.437，测试 R²相对全局 GAM 仅增加 +0.014/+0.007/+0.005。
- 刷新报告表格、SoilType=85 ALE 支持量警示、空间机器学习综合排行榜、provenance 与输出清单。保留 SoilType=85 但不解释仅 8–9 个中心的 ALE；报告指出 SHAP 基线贡献与有限差分导数不是同一量。所有运行显式设置 `outcome_guided_selection=true`。
- 未重算或验证 RUSLE 因子乘积、未重跑 PCNM、未修改研究总纲。I-6 至 I-9 继续执行。

## 2026-10-31：Phase I-6 局部空间关联与热点

- 新增 `experiments/oracle_exploration/phase_i_spatial_cluster.py` 和配置 `config/phase_i_spatial_cluster.yaml`，权威运行位于 `outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_i6_r1/`。共同环境完整样本按 8×8 汇总为约 240 m 单元，覆盖九项预测健康指标。
- 完成 100 近邻 Local Spearman（仅作描述统计）、8 近邻 Bivariate Local Moran（999 次全局随机置换及结局内 BH FDR）、同位 Aa—健康四象限共现，以及基于 I-4 GWR/GWR-LCR 系数的 1 km Gi* 热点。明确区分同位高 Aa—低健康和邻域滞后 HL 类。
- 九项结果表、五套九面板空间图（PNG/SVG/PDF）、报告和输入/软件/方法 provenance 已写入 Oracle 独立输出。GWR 与 GWR-LCR 的热点范围存在明显差别；结果定位为探索性空间线索。未重算/验证 RUSLE 因子乘积、未重跑 PCNM、未修改研究总纲。
- I-6 已完成；I-7 至 I-9 继续执行。

## 2026-10-31：Phase I-7 健康维度与既有成果衔接

- 新增 `experiments/oracle_exploration/phase_i_health_mechanism.py` 与配置 `config/phase_i_health_mechanism.yaml`；权威报告及对照表位于 `outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_i7_r1/`。复用 I-1 至 I-6 汇总，不重跑模型。
- 将 SW/BD、OM/AN/CEC/pH、MBC/RESP 按物理、化学、生物维度列示，原指标方向不翻转。因为没有已核实的指标方向、构念载荷和维度权重，未构造新维度指数或推断 PLS-SEM 路径。PCNM 结果配置为 null，且无可读结果表；PLS-SEM 路径/载荷源表缺失，交叉表保留空值，PCNM 未重跑。
- EnvMatch 对照显示不同分析单位的 SHI/OM/SW 估计均有负向线索，但强度不同；例如 I-1 β 为 −0.472/−0.421/−0.595，I-2 匹配差值 β 约 −0.144/−0.100/−0.269，I-4 环境调整 GWR 系数中位数 −0.330/−0.556/−0.680。对照图明确误差区间口径差异，不做绝对效应排名。
- 报告、PCNM字段表、PLS-SEM路径交叉表、健康维度统计及森林图写入Oracle输出；I-8至I-9继续执行。

## 2026-10-31：Phase I-8 空间胁迫候选分区

- 新增 `experiments/oracle_exploration/phase_i_spatial_zoning.py` 和 `config/phase_i_spatial_zoning.yaml`；权威运行 `outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_i8_r1/`。重用I-4局部系数、I-5最近1.2km GRF中心及I-6局部空间诊断，不重新拟合模型；仅使用预计算 Aa_20yr。
- 对 293,633 个环境和 SHI 共同完整的30m像元按 8×8 块汇总，形成 7,387 个约240m单元。用Aa q80、局部GWR负系数q20、Aa q50、SHI q20划定互斥T1–T4及其他区。T1 623单元、18.88 km²等价面积；T2 842、24.00 km²；T3 60、2.05 km²；T4 961、38.77 km²；其他4,901。面积不是全耕地面积，仅按完整案例支持换算。
- 27潮土/TWI q2与26冲积土/TWI q1以≥80%主导土类块及土类内块均值分位形成空间代理，分别标记963和285个单元，单独出图且允许与主分区重叠。治理建议作为现场核验候选，引用水土保持和土壤培肥官方指导；不直接外推外地施用量。行政区底图未提供，未添加中国/河南位置插图。
- 输出综合分区地图（600 dpi PNG/SVG/PDF）、分区像元数据、统计表、环境剖面、候选代理表、建议表、运行配置和 provenance。未运行标准 MGWR；I-4 多尺度 SVC 近似保持原方法标签。I-9 继续执行。

## 2026-10-31：Phase I-9 综合报告、审查请求与图件索引

- 新增 `experiments/oracle_exploration/phase_i_finalize.py` 与配置 `config/phase_i_final.yaml`；权威输出为 `outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_i9_r1/`。最终报告按科学问题组织，逐指标比较 I-1/I-2/I-4/I-6 估计，并汇总 I-3、I-5、I-7、I-8 结果。

## 2026-10-31：Phase I-8 定位插图与 I-9 图集刷新

- 为满足图件位置表达，新增中国与河南省 WGS84 GeoJSON 定位输入（来源和获取日期记录在 I8 r2 provenance）；上排显示中国—河南位置、河南—温县完整案例加权中心，下排保留原研究区分区图。GeoJSON只作定位示意，不参与统计分区；温县精细轮廓仍来自项目耕地掩膜。
- I8 r2 复用 I8 r1 的分类逻辑和已有 I3–I6结果，未改变统计结果；当前权威图件及其报告、配置、来源副本和provenance位于 `outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_i8_r2/`。I9 图集随后更新至该图件。
- I9 r2 更新 I8 图件来源链接与图集预览；当前综合报告位于 `outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_i9_r2/`。I9 不拟合新模型。
- 生成 `PHASE_I_FINAL_RESEARCH_REPORT.md`、`PHASE_I_CHATGPT_REVIEW_REQUEST.md` 与 `PHASE_I_FIGURE_ATLAS.html`。图谱索引八个权威阶段运行共34张PNG预览，并链接可用的SVG/PDF。I-7中PCNM/PLS-SEM结果源缺失继续留空；不重跑PCNM、不推断路径。
- 审查请求按附件要求以前置读取研究总纲为审查指令，标明目前 Git HEAD 不包含未提交工作，并仅保存在本地；未向外部发送。报告与图谱中的相对链接均指向本地Oracle阶段材料。
- I-1至I-9交付物已生成。Phase I为结果导向探索，待独立审查；结果不代表FORMAL阶段设计、研究假设或因果效应已验证。

## 2026-10-31：Phase I-C 图件闭合、标准 MGWR 与 I-9 r3

- 新增 `experiments/oracle_exploration/phase_i_closure.py`、`experiments/oracle_exploration/phase_i_standard_mgwr.py` 和对应配置。闭环权威运行：`outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_closure_r1/`；标准 `mgwr==2.2.1` 安装于 `/tmp/envmatch-mgwr-venv` 隔离临时环境，没有修改项目依赖。
- 在7,387个完整240m单元中按固定800m格网坐标均衡抽取770单元，不使用健康响应选点。对SHI、OM、SW使用同样本标准GWR/MGWR、Gaussian adaptive kernel、AICc带宽及九项环境协变量。三项MGWR均成功并在20次上限前收敛（8/7/6次）。模型级R²：GWR 0.887/0.714/0.834，MGWR 0.869/0.696/0.822；MGWR AICc分别为770.33/1389.80/972.27，GWR为781.57/1417.29/989.47。该空间均衡样本等权结果与I-4全单元像元频数加权GWR的估计对象不同。
- Aa的MGWR带宽：SHI 32邻居、中位距离2.64km；OM和SW均767邻居、中位距离23.78km，接近770样本的搜索上界。`mgwr==2.2.1`未实现多带宽MGWR局部R²，故MGWR局部表该字段留空；模型级统计和局部系数仍可用。
- 补绘核心图04–12及16–19，明确保留图5/9–12的部分满足边界；图30/31仅为PCNM、PLS-SEM源数据缺口状态图。31项核心图件状态为24可用、5部分、2延期。PCNM/PLS-SEM源表未提供，比较值继续为空，没有重跑或推断。
- 最终综合报告、图件索引及本地审查请求更新为 `outputs/exploratory_oracle/phase_i/phase_i_20261031_seed20261031_i9_r3/`；图集共48张PNG，报告和图集的本地相对链接检查无缺失。审查请求只写入本地，没有外发。I-9 r3仍为未提交工作树内容；HEAD不包含这些改动。
- Phase I本轮可由现有数据支持的分析与图件闭合已完成，整体仍为Oracle探索，不代表独立实证验证；取得PCNM结果表与PLS-SEM路径/载荷源表后，才能补做核心图30/31对应的科学比较。
