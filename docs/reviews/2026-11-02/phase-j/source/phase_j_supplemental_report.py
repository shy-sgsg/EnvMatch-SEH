"""Assemble the Phase J figure catalogue and embed it in the academic report."""

from __future__ import annotations

import html
import json
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "outputs/exploratory_oracle/phase_j/phase_j_20261102_seed20261102_r10"
REPORTS = RUN / "reports"
REVIEW = ROOT / "docs/reviews/2026-11-02/phase-j"
CORE_CAPTIONS = REPORTS / "PHASE_J_FIGURE_CAPTIONS.md"
SUPPLEMENT_MANIFEST = RUN / "figures/supplementary/figure_manifest.json"
ACADEMIC_REPORT = REPORTS / "ENV_MATCH_SOIL_HEALTH_ACADEMIC_REPORT.md"
APPENDIX_START = "<!-- PHASE_J_FULL_FIGURE_APPENDIX_START -->"
APPENDIX_END = "<!-- PHASE_J_FULL_FIGURE_APPENDIX_END -->"

CORE_STEMS = {
    "A": "A_multimethod_erosion_health_response",
    "B": "B_gwr_mgwr_spatial_patterns",
    "C": "C_stress_sensitive_environment_zones",
    "D": "D_sem_control_hierarchy",
    "E": "E_full_vs_matched_sem",
    "F": "F_erosion_dimension_path_forest",
    "G": "G_sem_fit_and_erosion_increment",
    "H": "H_best_sem_path_diagram",
    "I": "I_soil_twi_sem_comparison",
}


def read_core() -> list[dict[str, str]]:
    text = CORE_CAPTIONS.read_text(encoding="utf-8")
    matches = list(re.finditer(r"(?m)^## 图([A-Z])｜(.+)$", text))
    rows: list[dict[str, str]] = []
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        explanation = text[match.end() : end].strip()
        explanation = re.sub(r"!\[[^\]]*\]\([^)]*\)(?:\{[^}]*\})?", "", explanation)
        explanation = re.sub(r"\n+", " ", explanation)
        explanation = re.sub(r"\s+", " ", explanation).strip()
        explanation = re.sub(r"\[交互图集\]\([^)]*\)", "", explanation).strip()
        title = match.group(2).strip()
        if match.group(1) not in CORE_STEMS:
            continue
        if match.group(1) == "A":
            explanation = explanation.replace(
                "CEC使用包含Aa的校正公式，其相关存在代数耦合。",
                "CEC按包含Aa项的原始值恢复公式计算，其相关存在代数耦合。",
            )
        rows.append(
            {
                "letter": match.group(1),
                "id": CORE_STEMS[match.group(1)],
                "title": title,
                "explanation": explanation,
                "png": f"figures/{CORE_STEMS[match.group(1)]}.png",
                "svg": f"figures/{CORE_STEMS[match.group(1)]}.svg",
                "pdf": f"figures/{CORE_STEMS[match.group(1)]}.pdf",
                "source": "",
            }
        )
    if [row["letter"] for row in rows] != list("ABCDEFGHI"):
        raise ValueError("Expected exactly nine Phase J core figures A-I")
    return rows


def read_supplement() -> list[dict[str, str]]:
    manifest = json.loads(SUPPLEMENT_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("outcome_guided_selection") is not True:
        raise ValueError("Phase J Oracle metadata is missing from the figure manifest")
    if manifest.get("model_refit") is not False:
        raise ValueError("Supplemental figure manifest unexpectedly indicates a model refit")
    rows = manifest["figures"]
    if [row["letter"] for row in rows] != list("JKLMNOPQRSTUVWXYZ"):
        raise ValueError("Expected exactly seventeen supplementary figures J-Z")
    for row in rows:
        for field in ("png", "svg", "pdf", "source"):
            if not (RUN / row[field]).is_file():
                raise FileNotFoundError(RUN / row[field])
    return rows


def report_image_path(row: dict[str, str]) -> str:
    return "../" + row["png"]


def gallery_html(rows: list[dict[str, str]]) -> str:
    articles: list[str] = []
    for row in rows:
        image_path = report_image_path(row)
        source_path = "../" + row["source"] if row["letter"] >= "J" else ""
        links = [
            f'<a href="{html.escape(image_path)}">PNG大图</a>',
            f'<a href="../{html.escape(row["svg"])}">矢量图</a>',
            f'<a href="../{html.escape(row["pdf"])}">PDF版</a>',
        ]
        if source_path and Path((REPORTS / source_path).resolve()).is_file():
            links.append(f'<a href="{html.escape(source_path)}">图源数据</a>')
        articles.append(
            "<article>"
            f"<h2>图{html.escape(row['letter'])}｜{html.escape(row['title'])}</h2>"
            f'<a href="{html.escape(image_path)}"><img loading="lazy" src="{html.escape(image_path)}" '
            f'alt="图{html.escape(row["letter"])}：{html.escape(row["title"])}"></a>'
            f"<p>{html.escape(row['explanation'])}</p>"
            f"<p>{' · '.join(links)}</p></article>"
        )
    return "\n".join(articles)


def write_catalogues(rows: list[dict[str, str]]) -> None:
    top = (
        "# 温县耕地侵蚀与土壤健康分析图件目录\n\n"
        "共26幅不重复图：A–I为九幅核心图，J–Z为十七幅补充图。补充图覆盖局部系数地图、GWR/MGWR差值地图和四种SEM结构路径图，均使用已保存的结果表；图T复用已保存的方向评分敏感性图。图中统计单位、模型和估计目标不同，需结合逐图说明阅读。\n\n"
        "| 图 | 图名 | 查看图像 | 图注说明 |\n|---|---|---|---|\n"
    )
    index_lines = [top]
    caption_lines = [
        "# 阶段 J 图件说明\n\n",
        "A–I为九幅核心图，J–Z为十七幅补充图。图中健康指标来自预测栅格；Oracle结果导向选择和空间依赖须在解释中考虑。图内CEC统一写作“CEC”。\n\n",
    ]
    for row in rows:
        link = "../" + row["png"]
        index_lines.append(
            f"| {row['letter']} | {row['title']} | [查看图像]({link}) | {row['explanation']} |\n"
        )
        caption_lines.extend(
            [
                f"## 图{row['letter']}｜{row['title']}\n\n",
                f"{row['explanation']}\n\n",
            ]
        )
    index_lines.append("\n[交互图集](PHASE_J_RESEARCH_FIGURE_ATLAS.html)\n")
    (REPORTS / "PHASE_J_CORE_FIGURE_INDEX.md").write_text("".join(index_lines), encoding="utf-8")
    CORE_CAPTIONS.write_text("".join(caption_lines), encoding="utf-8")

    atlas = (
        "<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        "<title>温县耕地侵蚀与土壤健康阶段 J 图集</title>"
        "<style>body{font-family:'Noto Sans CJK SC','WenQuanYi Zen Hei',sans-serif;max-width:1250px;"
        "margin:2rem auto;padding:0 1.2rem;color:#203545;line-height:1.65}h1{color:#244c61}"
        "article{border-top:1px solid #d7e0e4;padding:1rem 0 2rem}img{display:block;max-width:100%;"
        "height:auto;border:1px solid #e0e6e9}a{color:#28647e}</style></head><body>"
        "<h1>温县耕地侵蚀与土壤健康阶段 J 图集</h1>"
        "<p>共26幅不重复图件：A–I展示总体响应、局部模型、环境匹配、SEM控制和分组路径；"
        "J–T补充跨总体控制敏感性、方法比较、拟合、局部空间关系、环境差异和两阶段侵蚀分析；"
        "U–V增加九指标局部系数地图和GWR/MGWR差值地图；W–Z展示平行维度、链式维度、九结局及两部分侵蚀的SEM路径结构。"
        "方向评分敏感性图T沿用已保存图件。预测栅格结果属于探索性条件关联，不是独立实测验证或因果结论。"
        "图中指标名统一写作CEC。</p>"
        + gallery_html(rows)
        + "</body></html>"
    )
    (REPORTS / "PHASE_J_RESEARCH_FIGURE_ATLAS.html").write_text(atlas, encoding="utf-8")


def append_report_gallery(rows: list[dict[str, str]]) -> None:
    text = ACADEMIC_REPORT.read_text(encoding="utf-8")
    old_sentence = (
        "本报告包括A–I九幅核心图和J–T十一幅补充图，末尾附有完整图版与逐图阅读说明。"
        "补充图来自已保存的模型结果表；图T沿用已保存的健康方向评分敏感性结果。"
    )
    new_sentence = (
        "本报告包括A–I九幅核心图和J–Z十七幅补充图，末尾附有完整图版与逐图阅读说明。"
        "补充图使用已保存的模型和空间结果表；图T沿用已保存的健康方向评分敏感性结果。"
    )
    if old_sentence in text:
        text = text.replace(old_sentence, new_sentence, 1)
    elif new_sentence not in text:
        marker = "## 一、研究问题"
        text = text.replace(marker, new_sentence + "\n\n" + marker, 1)
    appendix = [
        APPENDIX_START,
        "\n\n## 附录：阶段 J 完整图版（A–Z）\n\n",
        "图A–I为核心综合图；图J–T补充跨总体敏感性、方法比较、局部空间摘要和分组路径；图U–V增加九项结局的局部系数地图及GWR/MGWR模型差值地图；图W–Z分别展示平行维度、链式维度、九项结局和两部分侵蚀的SEM结构路径。新增图只整理已有估计，不重新拟合模型。图T为方向评分敏感性分析。空间区间和样本单位按图注说明；预测像元不是独立实测样点。\n\n",
    ]
    for row in rows:
        appendix.extend(
            [
                f"### 图{row['letter']}｜{row['title']}\n\n",
                f"![图{row['letter']}：{row['title']}]({report_image_path(row)}){{width=95%}}\n\n",
                f"**阅读说明：**{row['explanation']}\n\n",
            ]
        )
    appendix.append(APPENDIX_END)
    section = "".join(appendix)
    if APPENDIX_START in text and APPENDIX_END in text:
        before = text.split(APPENDIX_START, 1)[0].rstrip()
        after = text.split(APPENDIX_END, 1)[1].lstrip()
        text = before + "\n\n" + section + ("\n\n" + after if after else "\n")
    else:
        text = text.rstrip() + "\n\n" + section + "\n"
    ACADEMIC_REPORT.write_text(text, encoding="utf-8")


def copy_review_package(rows: list[dict[str, str]]) -> None:
    supplement_source = RUN / "figures/supplementary"
    supplement_target = REVIEW / "figures/supplementary"
    for row in rows:
        if row["letter"] not in "JKLMNOPQRSTUVWXYZ":
            continue
        for field in ("png", "svg", "pdf", "source"):
            source = RUN / row[field]
            relative = source.relative_to(supplement_source)
            target = supplement_target / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    shutil.copy2(
        ROOT / "experiments/oracle_exploration/phase_j_supplemental_figures.py",
        REVIEW / "source/phase_j_supplemental_figures.py",
    )
    shutil.copy2(
        ROOT / "experiments/oracle_exploration/phase_j_supplemental_report.py",
        REVIEW / "source/phase_j_supplemental_report.py",
    )
    shutil.copy2(
        RUN / "figures/supplementary/figure_manifest.json",
        REVIEW / "figures/supplementary/figure_manifest.json",
    )
    for filename in (
        "ENV_MATCH_SOIL_HEALTH_ACADEMIC_REPORT.md",
        "PHASE_J_CORE_FIGURE_INDEX.md",
        "PHASE_J_FIGURE_CAPTIONS.md",
        "PHASE_J_RESEARCH_FIGURE_ATLAS.html",
    ):
        shutil.copy2(REPORTS / filename, REVIEW / "reports" / filename)


def main() -> None:
    core = read_core()
    supplement = read_supplement()
    rows = core + supplement
    write_catalogues(rows)
    append_report_gallery(rows)
    copy_review_package(rows)
    print(f"Updated Phase J report and catalogues with {len(core)} core + {len(supplement)} supplemental figures.")


if __name__ == "__main__":
    main()
