#!/usr/bin/env python3
"""Complete Phase K synthesis deliverables from the saved single-period tables."""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[5]
PK = ROOT / "docs/reviews/2026-11-05/phase-k"
FIG = PK / "figures"
RUN = ROOT / "outputs/exploratory_oracle/phase_k/phase_k_20261104_seed20261104_r1"


def write_text(path: Path, value: str) -> None:
    path.write_text(value.strip() + "\n", encoding="utf-8")


def make_integrated_figure() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Noto Sans CJK SC", "Noto Sans CJK TC", "WenQuanYi Micro Hei", "AR PL UMing CN"):
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["savefig.dpi"] = 240

    rows = [
        ["全域", "约38.8万有效像元", "SW −0.584\nSHI −0.469\nOM −0.415", "总体排序关系"],
        ["环境相近配对", "111,289对设计", "SW −0.272\nSHI −0.144\nOM −0.115", "配对差值的秩相关"],
        ["局部回归", "7,387–7,408个空间单元", "SW −0.486\nOM −0.395\nSHI −0.222", "局部系数中位数"],
        ["环境控制", "293,145像元；35个5 km块", "SHI路径 −0.113\n额外解释量 ΔR²=0.0079", "空间分块稳健模型"],
        ["实地核查候选", "623个240 m单元；18.882 km²等价", "平均Aa_20yr=15.67\n平均SHI=0.549", "探索性强负向候选区"],
    ]
    pd.DataFrame(rows, columns=["环节", "统计单位", "保存结果", "读图说明"]).to_csv(
        FIG / "I_source_integrated_findings.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(15, 8.4))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fig.patch.set_facecolor("#f7f9fb"); ax.set_facecolor("#f7f9fb")
    ax.text(.5, .94, "侵蚀与土壤健康：从总体关系到实地核查线索", ha="center", va="center",
            fontsize=21, fontweight="bold", color="#19334a")
    ax.text(.5, .885, "只看20年平均实际侵蚀；下列每一步回答的问题不同，统计单位也不同",
            ha="center", va="center", fontsize=12.5, color="#526778")
    xs = [.105, .302, .499, .696, .893]
    colors = ["#e8f1f8", "#eaf4ed", "#fff1df", "#eeeafa", "#fce9e7"]
    edge = ["#35749a", "#4b8a65", "#d28a37", "#7865a7", "#c65e53"]
    titles = ["① 看总体", "② 看相近对象", "③ 看地点差异", "④ 控制环境", "⑤ 指向现场"]
    bodies = [
        "约38.8万有效像元\n\nSW  −0.584\nSHI −0.469\nOM  −0.415\n\n全域秩相关",
        "111,289对匹配设计\n\nSW  −0.272\nSHI −0.144\nOM  −0.115\n\n配对差值秩相关",
        "约7,400个空间单元\n\nSW  −0.486\nOM  −0.395\nSHI −0.222\n\n局部系数中位数",
        "293,145像元\n35个5 km空间块\n\nSHI路径 −0.113\nΔR² = 0.0079\n\n加入环境信息后",
        "623个240 m单元\n面积等价18.882 km²\n\n平均侵蚀 15.67\n平均SHI 0.549\n\n先做现场核查",
    ]
    widths, y, h = .174, .47, .34
    for i, x in enumerate(xs):
        box = FancyBboxPatch((x-widths/2, y), widths, h,
                             boxstyle="round,pad=0.012,rounding_size=0.018",
                             linewidth=1.8, edgecolor=edge[i], facecolor=colors[i])
        ax.add_patch(box)
        ax.text(x, y+h-.045, titles[i], ha="center", va="center", fontsize=13,
                fontweight="bold", color=edge[i])
        ax.text(x, y+h-.17, bodies[i], ha="center", va="center", fontsize=11.2,
                color="#233746", linespacing=1.22)
        if i < 4:
            arrow = FancyArrowPatch((x+widths/2+.012, y+h/2), (xs[i+1]-widths/2-.012, y+h/2),
                                    arrowstyle="-|>", mutation_scale=15, linewidth=1.6, color="#778995")
            ax.add_patch(arrow)
    ax.text(.5, .355, "九项指标并非同方向：CEC全域相关为 +0.168；其余指标强弱各不相同。",
            ha="center", va="center", fontsize=12.5, color="#314e62",
            bbox={"boxstyle":"round,pad=.5", "fc":"white", "ec":"#cad5dc"})
    ax.text(.5, .235,
            "这些结果共同支持：若干土壤指标与侵蚀存在负向空间条件关联，关系强弱会随指标、地点和控制方式变化。",
            ha="center", va="center", fontsize=13, fontweight="bold", color="#19334a", wrap=True)
    ax.text(.5, .12,
            "解释边界：结果来自空间预测栅格且有空间依赖；配对、像元和空间单元不可混作同一种样本。\n"
            "路径模型不是因果证明；候选位置只用于安排独立实测，不是正式风险分区。",
            ha="center", va="center", fontsize=11, color="#586b77", linespacing=1.35)
    for ext in ("png", "svg", "pdf"):
        output = FIG / f"I_20yr_integrated_findings.{ext}"
        fig.savefig(output, bbox_inches="tight", facecolor=fig.get_facecolor())
        if ext == "svg":
            output.write_text("\n".join(line.rstrip() for line in output.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    plt.close(fig)


def create_reports() -> None:
    reports = PK / "reports"
    write_text(reports / "PHASE_K_CORE_SCIENTIFIC_FINDINGS.md", """
# 阶段K核心科学发现

## 研究主线

本轮把已有的20年平均实际侵蚀结果放在一起看：先描述全域的总体关系，再比较环境相近的配对对象，接着看关系是否因地点而变，最后考察加入环境信息后SHI关系还剩多少，并把空间结果转成独立实测的候选地点。阶段K主要复用阶段F–J保存的分析，不重新搜索匹配、GWR/MGWR或SEM设定。全部结果属于结果导向的探索性分析。

## 四项主要发现

1. **全域中，土壤含水量、SHI和有机质与侵蚀的负向排序关系最明显。** Spearman相关依次为SW −0.584（n=388,092）、SHI −0.469（n=384,737）、OM −0.415（n=388,092）。这是栅格像元的排序关系，不能读成侵蚀增加一个单位会造成相应百分比下降。
2. **在环境相近的配对对象中，重点三项的方向仍为负。** 更新配对设计有111,289对；配对差值的秩相关为SW −0.272、SHI −0.144、OM −0.115。它们与全域像元结果来自不同统计单位，因此不直接相减推断匹配“削弱”了多少。
3. **局部回归显示负向关系占主导，但强度随地点而变。** 7,387–7,408个240米空间单元中，GWR-LCR局部系数中位数为SW −0.486、OM −0.395、SHI −0.222。负系数比例不是显著比例；标准GWR与MGWR在770个共同位置上方向较一致，但地点强弱排序的相关有限。
4. **控制环境后，侵蚀与SHI仍有较弱的负向条件关联，额外预测信息有限。** D1共同环境完整案例为293,145个像元、35个5千米空间块。核心环境控制下，标准化SHI路径约−0.113，SHI模型R²约0.601，加入侵蚀的条件ΔR²约0.0079。R²反映整个模型，不是侵蚀单独解释的比例。

## 解释和使用

九项指标不完全一致：全域CEC为+0.168，其他多数为负；单项指标不能替代整体土壤健康。当前证据足以支持“若干指标与20年平均侵蚀存在空间条件关联”，不足以证明因果。空间预测像元彼此相关，不能等同独立实测点。建议把SHI、SW和OM作为田间优先复核对象，同时保留方向不同的指标作为对照。

## 对应材料

- 综合证据图：[`图I`](../figures/I_20yr_integrated_findings.png)
- 全域与匹配表：[`全域相关源表`](../figures/A_source_global.csv)、[`配对结果源表`](../figures/A_source_matched.csv)
- 局部系数和模型摘要：[`局部系数`](../spatial/nine_outcome_local_beta_maps_20yr.csv)、[`GWR摘要`](../spatial/gwr_model_summary_20yr.csv)
- SEM路径和模型比较：[`路径系数`](../sem/sem_path_coefficients_20yr.csv)、[`模型排名`](../sem/sem_model_rankings_20yr.csv)
- 空间候选区：[`分区画像`](../spatial/stress_zone_profiles_20yr.csv)

**术语说明：**“秩相关”比较两组数值的高低次序是否一起变化；“标准化路径”是把变量换成可比较尺度后估出的条件关系；“空间块”是把附近像元合并成较大的区域，减少把相邻像元误当成完全独立数据的问题。
""")
    write_text(reports / "PHASE_K_MATCHING_NONLINEAR_SYNTHESIS.md", """
# 环境匹配与非线性响应综合说明

## 为什么并列看两种分析

全域相关把耕地范围内的有效栅格整体比较，读起来直接，但会受到土类、地形和气候等环境背景共同变化的影响。环境匹配先在相近环境背景中建立可比对象，再问高低侵蚀对象的健康差异是否仍同向。GAM曲线则检查关系是不是一条直线。三者回答的问题不同，应相互补充。

## 匹配结果

本轮沿用已保存的更新匹配成员，不重新寻找匹配参数。20年平均侵蚀匹配设计共有111,289对；各健康指标因缺失略有不同，重点指标的有效差值样本约为111,000对。配对差值Spearman相关为：SW −0.272、SHI −0.144、OM −0.115。方向仍为负，但幅度较全域相关小。由于全域像元与配对差值不是同一统计单位，这个差别不能单独归因于“匹配后效应变小”。

匹配能减少已纳入环境变量的差异，不能保证灌溉、耕作、施肥、作物轮作等未记录条件都相同。匹配结果仍然是观察数据中的条件关联。

## 曲线结果

Gaussian GAM（用平滑曲线描出可能弯曲关系的模型）显示，SHI、SW与OM总体随log(1+Aa_20yr)增加而下降，曲线变化速度并不恒定。SW和SHI在侵蚀较高范围下降更明显；OM在低至中等范围先有平台/轻微起伏，随后下降。曲线两端的不确定范围更宽，不宜把局部弯曲点说成确定阈值。

## 如何合并解读

全域、匹配与曲线对重点三项给出相同的总体负向方向；强弱和曲线形状则有差别。适当结论是“方向在多个分析中相对一致，但关联大小受统计单位、环境控制与侵蚀范围影响”。这些分析共用既有预测栅格，并且部分结果经历结果导向筛选，所以它们不是相互独立的重复验证。

## 可追溯表

- [`全域相关`](../figures/A_source_global.csv)：9项结局及有效像元数。
- [`匹配差值`](../figures/A_source_matched.csv)：配对数量、秩相关和其他配对统计。
- [`GAM拟合曲线`](../tables/full_sample_gam_curves_20yr.csv)：拟合值和区间。
- [`图A`](../figures/A_20yr_global_match_nonlinear.png)：三种分析的直观汇总。

“GAM”是允许曲线弯曲的回归方法；“环境匹配”是先按已记录环境条件找相近对象；“非线性”只表示曲线不是固定斜率的直线，不自动说明存在一个生态临界点。
""")
    write_text(reports / "PHASE_K_GWR_MGWR_SPATIAL_SYNTHESIS.md", """
# 局部回归与空间异质性综合说明

## 这组图回答什么

全域相关给出平均方向，却看不出哪些地点关系更强。局部回归让邻近空间单元对当前位置提供更多信息，估计各地的局部斜率；GWR-LCR是在既有模型中加入收缩处理的局部回归版本。阶段K复用已保存的局部系数，不在此重新搜索带宽或重拟模型。

## 九项结局的局部图

图B包含SHI与八项土壤指标，使用共同的240米汇总单元。重点三项的环境调整GWR-LCR局部系数中位数为SW −0.486（7,408单元）、OM −0.395（7,408单元）、SHI −0.222（7,387单元）。负系数单元比例分别为100%、100%和约99.7%。这里的比例只数系数正负，不代表显著性；模型的逐点区间也没有替代多重检验校正。

标准GWR与MGWR在770个共同位置上，SHI、OM和SW系数符号大体一致；系数排序相关约0.537、0.291和0.200。这表示模型大多判断负向，但对“哪里最强”的次序并不完全一致。局部带宽距离是估计时使用邻近数据的尺度，不是精确的生态作用半径。

## 空间聚集证据

已有局部空间分析显示，高侵蚀与邻近低SHI的位置组合，和同一位置高侵蚀、低SHI简单共现并不是同一个概念。SHI同位高侵蚀—低健康共现约2,473个240米单元；负向热点数随方法变化（标准GWR约2,484，GWR-LCR约3,124）。这说明热点图对统计方法敏感，不能只选择一张看起来最突出的地图作为结论。

## 读图边界与文件

- [`图B：九项局部系数地图`](../figures/B_20yr_nine_outcome_local_maps.png)
- [`局部系数源表`](../spatial/nine_outcome_local_beta_maps_20yr.csv)
- [`GWR方法摘要`](../spatial/gwr_model_summary_20yr.csv)
- [`空间候选区`](../spatial/stress_zone_profiles_20yr.csv)

地图上的局部系数是模型估计值，既不是现场测得的因果效应，也不是“多少比例的土地显著”。栅格预测之间具有空间依赖；240米单元不能解释成独立田块或独立土样。标准GWR、GWR-LCR、MGWR、局部秩关联和热点统计各有不同定义，不应混为同一种结果。
""")
    write_text(reports / "PHASE_K_SEM_SYNTHESIS_REPORT.md", """
# 结构路径模型综合报告

## 模型用途

结构方程模型（SEM）把多项指标之间的统计关系画成一张路径图，用来检查控制环境背景后，侵蚀与SHI之间是否还存在条件关联。本轮沿用既有模型表和配置，没有重新挑模型结构。维度模型将土壤含水量、容重归为物理方面；有机质、碱解氮归为化学方面；微生物量碳、土壤呼吸归为生物活动方面。CEC和pH仍作单项结局。

## 核心结果

D1共同环境完整样本包含293,145个像元和35个5千米空间块。无环境控制时SHI标准化路径约−0.157、条件解释量增幅约0.017；加入核心环境变量后路径约−0.113、增幅约0.0079；加入更丰富的环境与空间趋势后约−0.106、增幅约0.0069。SHI模型R²约0.601（核心控制）至0.651（更丰富控制）。这是全模型的拟合度，不能归为侵蚀单独的解释比例。

核心控制模型中，侵蚀到物理、化学、生物活动维度的标准化路径约−0.389、−0.251、−0.066。关系大小不同，提示水分、有机质等成分和微生物相关指标不宜被一个综合分数取代。这里的路径只表示横截面空间数据中的条件关联；箭头不是经过时间或实验验证的因果方向。

## 设计与限制

模型区间按5千米空间块聚类稳健估计，能部分处理相邻像元共同变化造成的过度乐观误差，但不能替代独立田间样本。模型结构和候选比较受探索性结果筛选影响。SHI也与健康指标有定义和构成上的关联，因此较高R²不等于侵蚀机制已经解释清楚。

## 表图索引

- [`模型比较表`](../sem/sem_model_rankings_20yr.csv)
- [`路径系数表`](../sem/sem_path_coefficients_20yr.csv)
- [`侵蚀增量表`](../sem/sem_erosion_increment_20yr.csv)
- [`图H：核心路径图`](../figures/H_20yr_sem_path_diagram.png)
- [`图I：研究发现综合示意图`](../figures/I_20yr_integrated_findings.png)
- [`图J：土类与地形湿度补充图`](../figures/J_20yr_soil_twi_sem.png)

“条件关联”指在模型包含的其他变量相同的条件下，变量之间仍有统计联系；“ΔR²”是加入侵蚀信息后模型拟合增加的部分；“5千米空间块”是相邻栅格的较大分组，用来避免把每个像元都当成独立观察。
""")
    write_text(reports / "PHASE_K_SPATIAL_STRESS_ZONES_REPORT.md", """
# 空间敏感位置候选区报告

## 用途

候选区把侵蚀值、SHI、局部回归系数及环境画像放在一起，目的是帮助设计下一轮实地核查路线。候选区规则是结果导向探索的一部分，地图不是正式风险图、治理优先级或经实测验证的分区。

## 四类候选区画像

| 类型 | 240米单元数 | 面积等价（km²） | 平均Aa_20yr | 平均SHI | 局部系数中位数 | 适合回答的问题 |
|---|---:|---:|---:|---:|---:|---|
| T1 高侵蚀、负向关系较强 | 623 | 18.8820 | 15.67 | 0.549 | −0.567 | 强负向关系是否能在田间复现？ |
| T2 高侵蚀、负向关系较弱 | 842 | 24.0012 | 14.95 | 0.584 | −0.327 | 在相近高侵蚀背景中，哪些条件可能伴随较弱关系？ |
| T3 低侵蚀、低SHI | 60 | 2.0529 | 5.48 | 0.561 | −0.368 | 较低侵蚀下较低SHI是否由其他条件解释？ |
| T4 环境背景对照候选 | 961 | 38.7666 | 5.11 | 0.628 | −0.311 | 环境背景相近的对照位置呈现什么实测情况？ |

面积是240米网格单元的等价合计，不代表独立地块面积已逐一核实。表内数值是预测栅格和已保存模型的汇总统计。

## 建议的核查做法

若进行实地调查，建议每类选取分散的田块，不把相邻像元当作多个独立样点。记录采样深度、土类、作物、耕作、施肥、灌溉和同一时期的地表侵蚀/径流观测；对四类位置采用相同测量流程。分析前锁定主要指标、匹配规则、空间块和验证方式，并保留未筛选基线。这样才能检查地图关系能否在独立实测数据中重现。

## 源文件

- [`候选区图`](../figures/C_20yr_candidate_environment_zones.png)
- [`四类候选区画像`](../spatial/stress_zone_profiles_20yr.csv)
- [`候选区空间单元`](../spatial/stress_zone_tiles_20yr.csv)
- [`环境背景画像`](../spatial/stress_zone_environment_profiles_20yr.csv)
""")


def update_indexes_and_reports() -> None:
    manifest_rows = [
        ("A", "20yr_global_match_nonlinear", "全域相关、环境相近配对与非线性响应", "A_source_global.csv；A_source_matched.csv；full_sample_gam_curves_20yr.csv", "核心图", "20年平均实际侵蚀"),
        ("B", "20yr_nine_outcome_local_maps", "SHI和八项土壤指标的局部回归地图", "B_source_local_betas.csv", "核心图", "20年平均实际侵蚀"),
        ("C", "20yr_candidate_environment_zones", "现场核查位置候选图", "C_source_zone_profiles.csv；C_source_zone_tiles.csv", "核心图", "20年平均实际侵蚀"),
        ("D", "20yr_sem_control_hierarchy", "环境控制层级与SHI路径", "D_E_source_sem_rankings.csv", "核心图", "20年平均实际侵蚀"),
        ("E", "20yr_full_matched_sem", "全域和环境匹配总体对照", "D_E_source_sem_rankings.csv", "核心图", "20年平均实际侵蚀"),
        ("F", "20yr_sem_dimension_paths", "侵蚀与土壤健康维度路径", "F_H_source_sem_paths.csv", "核心图", "20年平均实际侵蚀"),
        ("G", "20yr_sem_incremental_fit", "环境控制下侵蚀增加的预测信息", "G_source_sem_increment.csv", "核心图", "20年平均实际侵蚀"),
        ("H", "20yr_sem_path_diagram", "侵蚀与健康维度和SHI的核心路径图", "F_H_source_sem_paths.csv", "核心图", "20年平均实际侵蚀"),
        ("I", "20yr_integrated_findings", "总体、匹配、局部、控制与现场核查的综合科学发现", "I_source_integrated_findings.csv", "核心图", "20年平均实际侵蚀"),
        ("J", "20yr_soil_twi_sem", "土类与地形湿度组合中的SHI路径", "J_source_sem_groups.csv", "补充图", "20年平均实际侵蚀"),
    ]
    with (FIG / "FIGURE_MANIFEST.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f); w.writerow(["图号", "文件基名", "主题", "数据来源", "图件类别", "暴露情景"]); w.writerows(manifest_rows)

    atlas = PK / "20年侵蚀单情景图集.html"
    html = atlas.read_text(encoding="utf-8")
    old = '<section><h2>图I：土类×地形湿度子组路径</h2><p>展示土类和湿度分组中的Aa_20yr—SHI系数；分组结果受Oracle筛选影响。</p><img src="figures/I_20yr_soil_twi_sem.png" alt="图I"></section>'
    new = ('<section><h2>图I：研究发现综合示意图</h2><p>把全域相关、环境相近配对、局部回归、环境控制模型和现场抽样候选区串成一条证据线；展示关联，不代表因果。</p><img src="figures/I_20yr_integrated_findings.png" alt="图I"></section>'
           '<section><h2>补充图J：土类与地形湿度组合中的SHI关系</h2><p>用于查看环境分组差异；分组曾按探索结果筛选，不是独立验证。</p><img src="figures/J_20yr_soil_twi_sem.png" alt="补充图J"></section>')
    html = html.replace(old, new).replace("图集</title>", "图集</title>")
    html = html.replace("九张图均提供", "九张核心图和一张补充图均提供")
    atlas.write_text(html, encoding="utf-8")

    md = PK / "reports/正式学术研究报告_20年侵蚀单情景.md"
    text = md.read_text(encoding="utf-8")
    text = text.replace("配套的九幅图依次展示", "配套的九幅核心图依次展示")
    text = text.replace("最后看图D至I理解模型控制和分组结果。", "最后看图D至I理解模型控制和整体发现；补充图J展示土类与地形湿度分组结果。")
    text = text.replace("土类与地形湿度子组比较。", "研究发现综合示意图。")
    text = text.replace("### 图I 土类与地形湿度子组\n\n![图I：土类和地形湿度组合中的SHI路径](../figures/I_20yr_soil_twi_sem.png)",
                        "### 图I 研究发现综合示意图\n\n![图I：从总体关系到现场核查线索](../figures/I_20yr_integrated_findings.png)\n\n### 补充图J 土类与地形湿度子组\n\n![补充图J：土类和地形湿度组合中的SHI路径](../figures/J_20yr_soil_twi_sem.png)")
    md.write_text(text, encoding="utf-8")

    tex = PK / "reports/正式学术研究报告_20年侵蚀单情景.tex"
    text = tex.read_text(encoding="utf-8")
    text = text.replace("../figures/I_20yr_soil_twi_sem.png", "../figures/I_20yr_integrated_findings.png")
    text = text.replace("图I：土类和地形湿度组合中的SHI路径", "图I：从总体关系到现场核查线索的综合科学发现")
    text = text.replace("图I\n土类与地形湿度子组", "图I\n研究发现综合示意图")
    text = text.replace("图表阅读顺序", "图表阅读顺序")
    append = r"""

\subsubsection{补充图J：土类与地形湿度子组}

\begin{figure}
\centering
\includegraphics{../figures/J_20yr_soil_twi_sem.png}
\caption{补充图J：不同土类和地形湿度组合中的SHI路径；分组属于探索性结果。}
\end{figure}
"""
    if "补充图J：土类与地形湿度子组" not in text:
        text = text.replace("\\end{document}", append + "\n\\end{document}")
    tex.write_text(text, encoding="utf-8")

    summary = PK / "PHASE_K_SUMMARY.md"
    s = summary.read_text(encoding="utf-8")
    s = s.replace("九张图", "九张核心图和一张补充图")
    s += "\n\n本次补充已将图I改为五步证据链综合示意图，并把原土类×湿度图移为补充图J；新增五份专题综合报告、结构化清点字段与94张仅20年表格切片。全项目总图集现已收录阶段K图件。\n"
    summary.write_text(s, encoding="utf-8")

    readme = PK / "README.md"
    s = readme.read_text(encoding="utf-8")
    s = s.replace("764个最终审查文件的暴露时期证据和20年单期配套位置", "764个最终审查文件的逐项时期证据；94张双期表另有仅20年切片及处理清单")
    s = s.replace("九张图均提供PNG、SVG、PDF和数据源表", "九张核心图及一张补充图均提供PNG、SVG、PDF和数据源表")
    s = s.replace("`figures/`：A–I九张图及图源数据。", "`figures/`：A–I九张核心图、补充图J及图源数据。")
    s = s.replace("沿用已存在的全项目图集", "已更新并纳入阶段K图件的全项目去重总图集")
    s = s.replace("## 研究范围与数据处理", "专题综合说明：[`核心发现`](reports/PHASE_K_CORE_SCIENTIFIC_FINDINGS.md)、[`匹配与曲线`](reports/PHASE_K_MATCHING_NONLINEAR_SYNTHESIS.md)、[`局部空间回归`](reports/PHASE_K_GWR_MGWR_SPATIAL_SYNTHESIS.md)、[`结构路径模型`](reports/PHASE_K_SEM_SYNTHESIS_REPORT.md)、[`空间候选区`](reports/PHASE_K_SPATIAL_STRESS_ZONES_REPORT.md)。\n\n## 研究范围与数据处理")
    readme.write_text(s, encoding="utf-8")

    run_status = PK / "RUN_STATUS.md"
    if run_status.is_file():
        s = run_status.read_text(encoding="utf-8")
        s = s.replace("九张重新绘制图件", "九张核心图与一张补充图")
        if "五份专题综合报告" not in s:
            s += "\n- 已补齐五份专题综合报告、证据链综合图I和补充图J；已更新结构化文件清点、94张双期数据表的20年单期切片清单及全项目总图集。\n"
        run_status.write_text(s, encoding="utf-8")

    index = PK / "PHASE_FJ_RESULT_INDEX.md"
    write_text(index, """
# 阶段F–J结果总索引与阶段K证据链

本索引说明每一步解决什么问题、使用什么统计单位、主要结果在哪里。阶段K复用已经保存的20年平均实际侵蚀结果，未重新运行完整匹配、低值阈值、GWR/MGWR或SEM搜索。

| 环节 | 用通俗话说 | 统计单位/主要数量 | 可复用结果与索引 | 阶段K做了什么 | 阅读边界 |
|---|---|---|---|---|---|
| A：输入与对齐 | 确认比较的是同一耕地范围和空间位置 | 对齐栅格像元；真实有效数看掩膜统计 | [`阶段A–E材料`](../../2026-10-26/) | 不重建侵蚀因子乘积；SHI与CEC按指定图层直接使用 | 预测像元不等于实测样点 |
| B：环境背景 | 看地形、植被、湿度等变量是否重复表达同一件事 | 连续环境变量及类别土类 | [`阶段F报告`](../../2026-10-26/PHASE_F_RESEARCH_REPORT.md) | 沿用已锁定的候选环境信息 | 未记录管理因素仍可能混杂 |
| C：分组和质量 | 检查环境组是否可解释且覆盖足够 | 环境组、空间块、标准化差异 | [`阶段F审查目录`](../../2026-10-26/) | 不从健康结果重新挑组 | Oracle曾结果导向筛选 |
| D：侵蚀分布 | 了解零值、近零值和长尾会怎样影响比较 | 像元值及敏感阈值候选 | [`阶段H总结`](../../2026-10-29/phase-h/PHASE_H_FINAL_SUMMARY.md) | 主结论保留完整有效样本，筛选仅作敏感性 | 阈值不代表自然常数 |
| E：环境匹配 | 在已测环境背景相近时找到可比对象 | 5 km空间约束的匹配成员 | [`匹配质量图`](../../2026-10-27/phase-g/figures/) | 使用已保存成员；更新设计为111,289对 | 不能控制未记录的田间管理 |
| F：配对差值 | 比较匹配对象中侵蚀差和健康差是否同向 | 配对差值；重点表有约111,000有效对 | [`配对相关`](../../2026-11-05/phase-k/figures/A_source_matched.csv) | 仅保留20年平均侵蚀记录 | 与全域像元的统计单位不同 |
| G：总体及非线性 | 看哪些指标总体关系较明显，关系是否弯曲 | 全域约38.4–38.8万有效像元；GAM曲线 | [`图A源数据`](../../2026-11-05/phase-k/figures/A_source_global.csv)、[`曲线表`](../../2026-11-05/phase-k/tables/full_sample_gam_curves_20yr.csv) | 重点报告SHI、SW、OM，并保留九项完整结果 | 曲线不自动给出因果阈值 |
| H：环境差异 | 看不同土类和湿度背景关系是否改变 | 环境组内样本和空间块区间 | [`环境组斜率`](../../2026-11-05/phase-k/matching/environment_group_slopes_20yr.csv) | 输出单期表与分组补充图 | 子组曾按结果探索 |
| I：局部空间关系 | 看哪些地点的局部斜率更强、是否聚集 | 240 m汇总；九指标地图；另有770个GWR/MGWR共同点 | [`图B局部地图`](../../2026-11-05/phase-k/figures/B_20yr_nine_outcome_local_maps.png)、[`系数表`](../../2026-11-05/phase-k/spatial/nine_outcome_local_beta_maps_20yr.csv) | 重绘九项指标图；将候选位置作为核查线索 | 局部负系数不等于显著 |
| J：路径模型 | 加入环境和健康维度后，SHI关系还剩多少 | D1：293,145像元、35个5 km块 | [`路径系数`](../../2026-11-05/phase-k/sem/sem_path_coefficients_20yr.csv)、[`模型排序`](../../2026-11-05/phase-k/sem/sem_model_rankings_20yr.csv) | 选用可读的核心控制模型作解释；不称因果 | SHI的R²包括其他变量解释的信息 |
| 实地候选区 | 用地图规划下一轮去哪核查 | 240 m网格；四类候选区 | [`候选区画像`](../../2026-11-05/phase-k/spatial/stress_zone_profiles_20yr.csv) | 给出可复核的数值画像和采样建议 | 不是正式风险等级 |
| PCNM对照 | 和既有空间控制结果比较是否一致 | 既有PCNM分析样本 | [`研究总纲`](../../../RESEARCH_CHARTER.md)及既有阶段文档 | 本阶段不替代或重跑PCNM | 需要后续单独锁定方案 |

## 常见词汇

- **Spearman相关：**比较两个变量的高低次序是否一起变化。
- **匹配对：**环境特征相近的两个栅格，被放在一起比较。
- **GWR/MGWR：**让不同地点各自估计关系；MGWR允许不同变量使用不同空间尺度。
- **SEM：**把多个变量关系组织成路径图，便于查看条件联系。
- **5 km空间块：**把相邻像元归成较大地理区块，用来避免把高度相似的近邻像元误当独立样本。
- **CEC：**阳离子交换量；图表中统一写作CEC。

全阶段结果均为探索性空间关联。既有PCNM结果继续保留，未来应在独立实测样本上比较，而不能从当前像元预测结果推断已经验证了现场因果。
""")

    core = PK / "results/FOUR_CORE_FINDINGS.csv"
    d = pd.read_csv(core, encoding="utf-8-sig")
    d.iloc[0, 2] = str(d.iloc[0, 2]).replace("M4配对差值相关", "更新环境匹配配对差值相关")
    d.to_csv(core, index=False, encoding="utf-8-sig")


def update_review_request() -> None:
    path = PK / "CHATGPT_REVIEW_REQUEST.md"
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    replacement = "| 健康指标使用 | `provenance/analysis_config.json` | SHI和CEC按用户指定的项目图层直接使用；图表统一写‘CEC’。审查本阶段结论的统计单位、指标方向、数据依赖与解释边界。 |"
    lines = [replacement if "健康数据和CEC来源" in line else line for line in lines]
    text = "\n".join(lines)
    text = text.replace("CEC来源耦合是否影响对应结论；", "各指标的解释边界是否写清；")
    text = text.replace("CEC来源无法追溯、", "")
    text = text.replace("CEC来源无法追溯或", "")
    path.write_text(text, encoding="utf-8")


def update_provenance() -> None:
    for path in (PK / "provenance/analysis_config.json", RUN / "provenance/analysis_config.json", RUN / "20yr_only/provenance.json"):
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        data["figures"] = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        data["figure_roles"] = {"core": ["A", "B", "C", "D", "E", "F", "G", "H", "I"], "supplementary": ["J"]}
        data["synthesis_deliverables"] = [
            "PHASE_K_CORE_SCIENTIFIC_FINDINGS.md", "PHASE_K_MATCHING_NONLINEAR_SYNTHESIS.md",
            "PHASE_K_GWR_MGWR_SPATIAL_SYNTHESIS.md", "PHASE_K_SEM_SYNTHESIS_REPORT.md",
            "PHASE_K_SPATIAL_STRESS_ZONES_REPORT.md"]
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    old_image = FIG / "I_20yr_soil_twi_sem.png"
    old_svg = FIG / "I_20yr_soil_twi_sem.svg"
    old_pdf = FIG / "I_20yr_soil_twi_sem.pdf"
    old_source = FIG / "I_source_sem_groups.csv"
    for src, dst in ((old_image, FIG / "J_20yr_soil_twi_sem.png"),
                     (old_svg, FIG / "J_20yr_soil_twi_sem.svg"),
                     (old_pdf, FIG / "J_20yr_soil_twi_sem.pdf"),
                     (old_source, FIG / "J_source_sem_groups.csv")):
        if src.is_file():
            shutil.move(src, dst)
    make_integrated_figure()
    create_reports()
    update_indexes_and_reports()
    update_review_request()
    update_provenance()
    print("Phase K synthesis reports, core Figure I, supplementary Figure J, indices, and inventory labels updated.")


if __name__ == "__main__":
    main()
