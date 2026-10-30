#!/usr/bin/env python3
"""Build an offline, source-linked Chinese atlas of curated research figures."""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ATLAS = ROOT / "docs/reviews/2026-10-29/phase-h/FIGURE_ATLAS.html"


def review_root() -> Path:
    return ROOT / "docs/reviews/2026-10-29/phase-h"


def available_source(path: Path | None) -> tuple[Path | None, str]:
    if path is None:
        return None, "源数据 CSV"
    if path.is_file():
        return path, "源数据 CSV"
    compressed = path.with_suffix(path.suffix + ".gz")
    if compressed.is_file():
        return compressed, "压缩源数据 CSV.gz"
    return path, "源数据 CSV"

OUTCOME_LABELS = {
    "SHI_candidate": "土壤健康指数（SHI）", "AN": "碱解氮（AN）", "OM": "有机质（OM）",
    "CEC": "阳离子交换量（CEC）", "pH": "土壤酸碱度（pH）", "SW": "土壤含水量（SW）",
    "BD": "土壤容重（BD）", "MBC": "微生物量碳（MBC）", "RESP": "土壤呼吸（RESP）",
}

FIGURE_META: dict[str, tuple[str, str]] = {
    "01_matching_coverage_balance": ("匹配覆盖与环境平衡", "匹配诊断"),
    "02_matching_method_pair_fit": ("不同匹配方法的配对拟合", "模型比较"),
    "03_primary_paired_slopes": ("优先指标的配对斜率", "森林图"),
    "04_repeated_spatial_validation": ("5 km 整块重复空间留出", "空间验证"),
    "05_erosion_increment_delta_r2": ("侵蚀信息的额外预测增量", "模型比较"),
    "06_environmental_heterogeneity": ("环境背景下的关联差异", "异质性森林图"),
    "07_environment_groups_map_and_high_response": ("环境组空间分布与高响应组", "空间图"),
    "08_health_response_curves": ("九项健康指标响应曲线", "非线性响应曲线"),
    "09_pair_distance_and_exposure_diagnostics": ("环境距离、地理距离与侵蚀差", "匹配诊断"),
    "10_exposure_period_comparison": ("两个侵蚀时期的共同像元比较", "时期比较"),
    "01_matching_quality": ("匹配覆盖率与环境平衡", "匹配诊断"),
    "02_paired_health_differences": ("高低侵蚀配对的健康差值", "配对分布"),
    "03_pair_association_heatmap": ("配对差值秩相关", "相关性热图"),
    "04_model_r2_comparison": ("像元级模型拟合度比较", "模型比较"),
    "05_response_curves": ("自然样条与历史择优曲线", "非线性响应曲线"),
    "06_environment_group_forest": ("环境组内负向关联", "森林图"),
    "07_grouping_method_comparison": ("环境分组方法关联分布", "分组比较"),
    "08_key_group_maps": ("结果导向环境组空间位置", "空间图"),
    "09_spatial_validation": ("空间留出与配对测试表现", "空间验证"),
    "10_environment_group_slope_forest": ("各指标最负环境组斜率", "空间块稳健森林图"),
    "D01_zero_and_positive_distributions": ("零值、正值与侵蚀分布", "分布图"),
    "D02_kde_bandwidth_sensitivity": ("KDE 带宽敏感性", "分布诊断"),
    "D03_gaussian_mixture_diagnostics": ("高斯混合与分量不确定性", "分布诊断"),
    "D04_threshold_candidate_comparison": ("阈值候选方法比较", "阈值诊断"),
    "D05_spatial_bootstrap_stability": ("空间重抽样阈值稳定性", "空间稳定性"),
    "S01_low_erosion_regime_maps": ("低值分区与侵蚀空间分布", "空间图"),
    "S02_low_regime_by_environment_profile": ("环境画像中的低值群体比例", "环境画像热图"),
    "S03_low_regime_connected_components": ("低值连通斑块尺度", "空间斑块统计"),
    "S04_spatial_block_composition_maps": ("5 km 空间块组成", "空间图"),
    "R01_nonlinear_response_curves": ("健康指标的非线性响应", "非线性响应曲线"),
    "R02_rank_association_heatmap": ("秩相关方向与强度", "相关性热图"),
    "R03_paired_slope_forest": ("配对差值斜率与区间", "森林图"),
    "M01_apparent_model_family_comparison": ("不同模型族表观拟合比较", "模型比较"),
    "M02_zero_indicator_and_segmented_models": ("零值指示与分段响应模型", "模型比较"),
    "V01_spatial_block_test_r2": ("空间分块测试拟合度", "空间验证"),
    "Q01_pair_retention_counts": ("阈值筛选后的配对保留量", "样本保留"),
    "Q02_matching_quality_bubble_plot": ("重配对质量与样本规模", "匹配诊断"),
    "M03_matched_unit_model_family_comparison": ("匹配像元与配对的八类模型比较", "模型比较热图"),
    "R04_matched_pixel_rank_associations": ("匹配端点像元秩相关", "相关性热图"),
    "V02_matched_unit_spatial_validation": ("匹配像元与配对的空间测试表现", "空间验证箱线图"),
    "S05_matched_endpoint_spatial_footprint": ("阈值筛选后匹配端点空间分布", "空间图"),
    "Q03_pair_and_unique_endpoint_counts": ("匹配对数与唯一端点像元数", "样本量比较图"),
    "HET01_group_low_value_retention": ("环境组内低值像元保留率", "环境组保留率图"),
    "HET03_group_spatial_test_change": ("环境组空间测试拟合变化", "空间块稳健热图"),
    "HET04_soil_group_paired_slopes": ("主要土类配对斜率与空间块区间", "空间块稳健森林图"),
    "HET05_group_matching_support_balance": ("环境组匹配支持量与平衡", "分组匹配诊断气泡图"),
    "HADD01_gmm_seed_stability": ("GMM 选中成分数与首处分界的随机种子敏感性", "随机种子稳定性图"),
    "HADD02_gmm_posterior_sensitivity": ("GMM 后验阈值提高时的像元归属变化", "后验分类敏感性图"),
    "HADD03_threshold_candidate_stability": ("不同分布方法给出的 Aa 阈值与稳定性", "阈值比较图"),
}

INITIAL = {
    "cropland_area_comparison.png": ("耕地面积来源比较", "数据审计", "面积对照"),
    "environment_correlation.png": ("耕地范围内环境变量相关性", "环境匹配", "相关性热图"),
    "erosion_resampling_sensitivity.png": ("实际侵蚀重采样敏感性", "数据审计", "分布比较"),
    "environment_groups_k6.png": ("环境组空间分布（新版，k=6）", "环境分组", "空间图"),
    "environment_groups_k8.png": ("环境组空间分布（新版，k=8）", "环境分组", "空间图"),
    "environment_groups_k10.png": ("环境组空间分布（新版，k=10）", "环境分组", "空间图"),
    "environment_groups_k12.png": ("环境组空间分布（新版，k=12）", "环境分组", "空间图"),
}


def rel(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    return Path(__import__("os").path.relpath(path, ATLAS.parent)).as_posix()


def read_text(path: Path | None) -> str:
    if path is None or not path.is_file():
        return ""
    return path.read_text(encoding="utf-8").strip()


def make_item(*, title: str, phase: str, question: str, experiment: str, chart_type: str,
              image: Path, caption: str, source: Path | None = None,
              source_label: str = "源数据 CSV",
              svg: Path | None = None, pdf: Path | None = None,
              scope_note: str = "") -> dict[str, Any]:
    return {"title": title, "phase": phase, "question": question, "experiment": experiment,
            "chart_type": chart_type, "image": rel(image), "caption": caption,
            "source_label": source_label,
            "source": rel(source), "svg": rel(svg), "pdf": rel(pdf), "scope_note": scope_note}


def add_initial(items: list[dict[str, Any]]) -> None:
    root = review_root()
    base = root / "artifacts/initial/figures"
    for filename, (title, question, chart_type) in INITIAL.items():
        image = base / filename
        stem = image.stem
        data_map = {
            "environment_correlation": root / "artifacts/initial/tables/environment_correlation.csv",
            "erosion_resampling_sensitivity": root / "artifacts/initial/tables/erosion_resampling_sensitivity.csv",
        }
        items.append(make_item(
            title=title, phase="阶段 A–E｜数据准备与环境分组", question=question,
            experiment="耕地掩膜、栅格对齐与环境候选诊断", chart_type=chart_type,
            image=image, source=data_map.get(stem),
            caption={
                "cropland_area_comparison": "对照耕地掩膜栅格、边界矢量和研究记录中的面积估计。",
                "environment_correlation": "显示有效耕地范围内连续环境变量之间的 Pearson 相关，辅助识别冗余变量。",
                "erosion_resampling_sensitivity": "比较面积平均与最近邻重采样得到的既有 Aa 栅格分布；不重建或验证 RUSLE 因子乘积。",
                "environment_groups_k6": "新版分土类分面图：26 冲积土、27 潮土、42 脱潮土。各面板独立聚类，颜色组号只在土类内部解释。",
                "environment_groups_k8": "新版分土类分面图：26 冲积土、27 潮土、42 脱潮土；每类考察 k=8。",
                "environment_groups_k10": "新版分土类分面图：26 冲积土、27 潮土、42 脱潮土；每类考察 k=10。",
                "environment_groups_k12": "新版分土类分面图：26 冲积土、27 潮土、42 脱潮土；每类考察 k=12。",
            }.get(stem, "阶段数据诊断图。"),
            scope_note="阶段 A–E 历史诊断图；可用源表已随审查包归档。"))

    original_style_dir = base / "original_style"
    original_caption = (
        "旧版全域栅格布局的中文版复原图，按原环境组 GeoTIFF 使用 tab20 配色并隐藏坐标轴；"
        "包含 26 冲积土、27 潮土和 42 脱潮土，组别编码是分层聚类结果的类别编号。"
    )
    for k in (6, 8, 10, 12):
        image = original_style_dir / f"environment_groups_k{k}_original.png"
        if not image.is_file():
            continue
        items.append(make_item(
            title=f"环境组空间图（原版全域样式复原，k={k}）",
            phase="阶段 A–E｜数据准备与环境分组",
            question="不同土壤类型中的环境组在空间上如何分布？",
            experiment="原版聚类空间图留存",
            chart_type="原版空间图",
            image=image,
            source=None,
            caption=original_caption,
            scope_note="旧版全域布局的样式复原图；与新版分土类分面图并列归档。",
        ))


def add_phase_index(items: list[dict[str, Any]], *, phase: str, question: str,
                    experiment: str, review_dir: Path, index_path: Path) -> None:
    if not index_path.is_file():
        return
    index = json.loads(index_path.read_text(encoding="utf-8"))
    for record in index:
        name = record.get("name", "")
        stem = Path(name).stem
        title, chart_type = FIGURE_META.get(stem, (record.get("title") or stem, "统计图"))
        image = review_dir / record.get("files", {}).get("png", record.get("formats", [""])[0])
        if not image.is_file():
            image = review_dir / "figures" / f"{stem}.png"
        source_path = record.get("data") or record.get("data_path")
        source, source_label = available_source(review_dir / source_path if source_path else None)
        caption_path = record.get("caption_path")
        caption_ref = caption_path or record.get("caption")
        caption_file = review_dir / caption_ref if caption_ref else None
        try:
            caption_is_file = caption_file is not None and caption_file.is_file()
        except OSError:
            caption_is_file = False
        caption = read_text(caption_file) if caption_is_file else record.get("caption", "")
        items.append(make_item(title=title, phase=phase, question=question,
                               experiment=experiment, chart_type=chart_type,
                               image=image, caption=caption, source=source,
                               source_label=source_label,
                               svg=image.with_suffix(".svg"), pdf=image.with_suffix(".pdf")))


def add_phase_h(items: list[dict[str, Any]]) -> None:
    root = review_root() / "artifacts/phase_h_core/figures"
    manifest_path = root / "figure_manifest.json"
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    titles = {
        "01_matched_response_direction": ("匹配后响应方向总览", "方向总结"),
        "02_matched_endpoint_boxplots": ("高低侵蚀配对端点分布", "配对分布"),
        "03_apparent_r2_pixel_vs_pair": ("像元与配对表观拟合度", "模型比较"),
        "04_rank_association_vs_r2": ("秩相关与模型拟合度", "关系比较"),
        "05a_fixed_20k_natural_spline": ("固定样本自然样条响应", "非线性响应曲线"),
        "05b_historical_full_sample_apparent": ("历史全样本 Oracle 表观曲线", "历史模型曲线"),
        "05c_matched_pixel_natural_spline": ("匹配像元自然样条响应", "非线性响应曲线"),
        "05d_matched_pair_natural_spline": ("匹配配对自然样条响应", "非线性响应曲线"),
        "06_group_effect_forest": ("环境组效应与空间块区间", "森林图"),
        "07_environment_group_method_comparison": ("环境分组方法比较", "分组比较"),
        "08_selected_group_map": ("结果导向环境组空间位置", "空间图"),
        "09_apparent_vs_spatial_test": ("表观拟合与空间测试表现", "空间验证"),
        "10_h1_interaction_inference_repair": ("H1 交互推断可靠性修订", "推断诊断"),
        "11_common_sample_period_comparison": ("共同样本下的时期比较", "时期比较"),
    }
    for stem in manifest.get("figures", []):
        title, chart_type = titles.get(stem, (stem, "统计图"))
        image = root / "png" / f"{stem}.png"
        caption = read_text(root / "captions" / f"{stem}.md")
        source, source_label = available_source(root / "data" / f"{stem}_source.csv")
        items.append(make_item(title=title, phase="Phase H｜环境组响应与推断诊断",
                               question="环境匹配后响应是否存在异质性、且空间推断是否可靠？",
                               experiment="H1 环境组交互与非线性关联探索", chart_type=chart_type,
                               image=image, caption=caption, source=source,
                               source_label=source_label,
                               svg=(root / "svg" / f"{stem}.svg"), pdf=(root / "pdf" / f"{stem}.pdf"),
                               scope_note="Oracle 表观筛选结果已标记；空间验证与表观拟合分开呈现。"))


def add_low_regime(items: list[dict[str, Any]]) -> None:
    root = review_root() / "artifacts/low_erosion_regime/figures"
    manifest_path = root / "figure_manifest.json"
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for record in manifest:
        stem = record.get("id", "")
        title, chart_type = FIGURE_META.get(stem, (record.get("title", stem), record.get("figure_type", "统计图")))
        caption = read_text(root / record.get("caption", ""))
        source, source_label = available_source(root / record.get("source", ""))
        items.append(make_item(title=title, phase="Phase H｜低侵蚀群体诊断与响应探索",
                               question=record.get("scientific_question", "低值群体如何定义与比较？"),
                               experiment=record.get("experiment", "低侵蚀值分布诊断"), chart_type=chart_type,
                               image=root / record.get("png", ""), caption=caption, source=source,
                               source_label=source_label,
                               svg=root / record.get("svg", ""), pdf=root / record.get("pdf", ""),
                               scope_note="实际 Aa 直接沿用既有数据库；分布阈值与结果导向 Oracle 候选分开保存。"))


def add_matched_unit_supplement(items: list[dict[str, Any]]) -> None:
    root = review_root() / "artifacts/matched_units/figures"
    manifest_path = root / "figure_manifest.json"
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for record in manifest.get("figures", []):
        stem = record.get("id", "")
        title, chart_type = FIGURE_META.get(
            stem, (record.get("title", stem), record.get("figure_type", "统计图"))
        )
        source, source_label = available_source(root / record.get("source", ""))
        items.append(make_item(
            title=title,
            phase="Phase H｜匹配像元与配对响应",
            question=record.get("scientific_question", "环境匹配后，侵蚀与健康响应如何变化？"),
            experiment=record.get("experiment", "阈值后匹配单位补充分析"),
            chart_type=chart_type,
            image=root / record.get("png", ""),
            caption=read_text(root / record.get("caption", "")),
            source=source,
            source_label=source_label,
            svg=root / record.get("svg", ""),
            pdf=root / record.get("pdf", ""),
            scope_note="像元按唯一 cell_id 计数，配对按端点差值计数；表观模型搜索与空间测试分开展示。",
        ))


def add_environment_group_heterogeneity(items: list[dict[str, Any]]) -> None:
    root = review_root() / "artifacts/environment_group_heterogeneity/figures"
    manifest_path = root / "figure_manifest.json"
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    periods = {"20yr": "20 年平均实际侵蚀", "2020": "2020 年实际侵蚀"}
    schemes = {
        "S1_both_positive": "两端均为正侵蚀",
        "T01_empirical_CDF_breakpoint": "CDF 阈值过滤原配对",
        "T01_fixed_original": "固定原环境组重配",
        "T01_refit_k6_filtered_population": "过滤后重拟合 k=6 组",
    }
    for record in manifest.get("figures", []):
        stem = record.get("id", "")
        title, chart_type = FIGURE_META.get(
            stem, (record.get("title", stem), record.get("figure_type", "统计图"))
        )
        if stem.startswith("HET02_group_rho_change_"):
            tokens = stem.removeprefix("HET02_group_rho_change_").split("_")
            period = periods.get(tokens[0], tokens[0])
            scheme_key = "_".join(tokens[1:])
            scheme = schemes.get(scheme_key, scheme_key)
            title = f"环境组 Spearman ρ 变化｜{period}｜{scheme}"
            chart_type = "环境组响应变化热图"
        source, source_label = available_source(root / record.get("source", ""))
        items.append(make_item(
            title=title,
            phase="Phase H｜低侵蚀筛选后的环境组异质性",
            question=record.get("scientific_question", "不同环境组中的低值排除、匹配与响应如何变化？"),
            experiment=record.get("experiment", "低侵蚀阈值后的环境组比较"),
            chart_type=chart_type,
            image=root / record.get("png", ""),
            caption=read_text(root / record.get("caption", "")),
            source=source,
            source_label=source_label,
            svg=root / record.get("svg", ""),
            pdf=root / record.get("pdf", ""),
            scope_note="基于既有环境组标签；组内配对要求两端属于同组。空间测试按 5 km 整块留出，图中 Oracle 探索结果按报告口径解释。",
        ))


def add_distribution_addendum(items: list[dict[str, Any]]) -> None:
    root = review_root() / "artifacts/distribution_diagnostic_addendum/figures"
    manifest_path = root / "figure_manifest.json"
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for record in manifest.get("figures", []):
        stem = record.get("id", "")
        title, chart_type = FIGURE_META.get(
            stem, (record.get("title", stem), record.get("figure_type", "统计图"))
        )
        source, source_label = available_source(root / record.get("source", ""))
        items.append(make_item(
            title=title,
            phase="Phase H｜低侵蚀分布诊断补充",
            question=record.get("scientific_question", "分布阈值是否稳定且可辨识？"),
            experiment=record.get("experiment", "多种子 GMM、后验敏感性与阈值比较"),
            chart_type=chart_type,
            image=root / record.get("png", ""),
            caption=read_text(root / record.get("caption", "")),
            source=source,
            source_label=source_label,
            svg=root / record.get("svg", ""),
            pdf=root / record.get("pdf", ""),
            scope_note="仅使用既有侵蚀 Aa 与分布诊断；未读取健康响应，也未重算 RUSLE。",
        ))


def build_html(items: list[dict[str, Any]]) -> str:
    cards = []
    for item in items:
        attrs = " ".join(f'data-{key}="{html.escape(str(item[key]), quote=True)}"'
                         for key in ("phase", "question", "experiment", "chart_type"))
        links = []
        for key, label in (("svg", "矢量 SVG"), ("pdf", "PDF"),
                           ("source", item.get("source_label", "源数据 CSV"))):
            if item.get(key):
                links.append(f'<a href="{html.escape(item[key])}" target="_blank">{label}</a>')
        caption = html.escape(item.get("caption", "")).replace("\n", " ")
        note = html.escape(item.get("scope_note", ""))
        cards.append(f'''<article class="card" {attrs}>
  <a class="thumb" href="{html.escape(item['image'])}" target="_blank" aria-label="查看原图：{html.escape(item['title'])}"><img loading="lazy" src="{html.escape(item['image'])}" alt="{html.escape(item['title'])}"></a>
  <div class="card-body"><div class="tags"><span>{html.escape(item['phase'])}</span><span>{html.escape(item['chart_type'])}</span></div>
    <h3>{html.escape(item['title'])}</h3><p class="question">{html.escape(item['question'])}</p>
    <p class="caption">{caption}</p>{f'<p class="note">{note}</p>' if note else ''}<div class="links">{' · '.join(links)}</div>
  </div></article>''')
    phases = sorted({item["phase"] for item in items})
    types = sorted({item["chart_type"] for item in items})
    questions = sorted({item["question"] for item in items})
    phase_options = ''.join(f'<option>{html.escape(v)}</option>' for v in phases)
    type_options = ''.join(f'<option>{html.escape(v)}</option>' for v in types)
    question_options = ''.join(f'<option>{html.escape(v)}</option>' for v in questions)
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>EnvMatch-SEH 中文科研图集</title>
<style>
:root{{--ink:#203545;--muted:#5e6e78;--line:#dce4e8;--paper:#f5f7f8;--blue:#285d78;--ochre:#c18d32;--teal:#3e7c78;--white:#fff}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font-family:"Noto Sans CJK JP","Source Han Sans SC",sans-serif;line-height:1.55}}
header{{background:linear-gradient(110deg,#193e55,#2c6a77);color:#fff;padding:38px max(24px,calc((100vw - 1440px)/2)) 30px}}
header h1{{margin:0 0 8px;font-size:clamp(26px,3vw,42px);letter-spacing:.02em}}header p{{max-width:900px;margin:5px 0;color:#e1ecee}}
.meta{{display:flex;gap:12px;flex-wrap:wrap;margin-top:16px}}.meta span{{border:1px solid #ffffff62;border-radius:99px;padding:4px 10px;font-size:13px}}
main{{max-width:1500px;margin:0 auto;padding:22px 24px 56px}}.controls{{display:grid;grid-template-columns:repeat(4,minmax(160px,1fr));gap:10px;background:#fff;border:1px solid var(--line);padding:14px;border-radius:12px;margin:0 0 18px;position:sticky;top:0;z-index:5;box-shadow:0 5px 22px #20354512}}
.controls input,.controls select{{width:100%;min-height:40px;border:1px solid #cad5dc;border-radius:7px;background:#fff;padding:8px 10px;color:var(--ink);font:inherit}}
.count{{color:var(--muted);font-size:14px;margin:10px 2px 18px}}.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,410px),1fr));gap:18px}}
.card{{background:#fff;border:1px solid var(--line);border-radius:12px;overflow:hidden;box-shadow:0 5px 18px #243b4a0b;display:flex;flex-direction:column;min-width:0}}
.thumb{{display:block;background:#fff;border-bottom:1px solid var(--line);padding:8px;min-height:190px}}.thumb img{{display:block;width:100%;height:270px;object-fit:contain}}
.card-body{{padding:14px 16px 15px}}.tags{{display:flex;gap:7px;flex-wrap:wrap}}.tags span{{font-size:11px;color:#315e70;background:#eaf1f3;border-radius:99px;padding:3px 8px}}
h3{{margin:9px 0 4px;font-size:18px;line-height:1.35}}p{{margin:7px 0}}.question{{font-size:13px;font-weight:600;color:var(--teal)}}.caption{{font-size:13px;color:#465964}}.note{{font-size:12px;color:#775e2c;border-left:2px solid var(--ochre);padding-left:8px}}.links{{margin-top:10px;font-size:12px}}a{{color:var(--blue);text-decoration:none}}a:hover{{text-decoration:underline}}
.section-title{{font-size:20px;border-bottom:1px solid var(--line);padding-bottom:7px;margin:30px 0 14px}}
@media(max-width:850px){{.controls{{grid-template-columns:repeat(2,minmax(140px,1fr));position:static}}.thumb img{{height:230px}}}}@media(max-width:520px){{main{{padding:14px}}.controls{{grid-template-columns:1fr}}header{{padding:26px 18px}}.thumb img{{height:210px}}}}
</style></head><body><header><h1>EnvMatch-SEH 中文科研图集</h1><p>按研究阶段、科学问题、实验和图表类型整理。每张图可查看 PNG 原图、矢量 SVG、PDF 与图源数据，便于复核和论文选图。</p>
<p>涉及环境匹配、侵蚀—健康响应、环境异质性、空间验证及低侵蚀值群体。土壤类型代码：26—冲积土、27—潮土、42—脱潮土。所有健康指标导向探索均保留 Oracle 标记；使用既有 Aa，不重建 RUSLE 因子乘积。</p>
<div class="meta"><span>图表 {len(items)} 张</span><span>中文标题与图注</span><span>含可复核源数据</span><span>本地离线浏览</span></div></header>
<main><div class="controls"><input id="search" type="search" placeholder="搜索图名、指标或关键词"><select id="phase"><option value="">全部阶段</option>{phase_options}</select><select id="question"><option value="">全部科学问题</option>{question_options}</select><select id="kind"><option value="">全部图表类型</option>{type_options}</select></div>
<p id="count" class="count"></p><div id="grid" class="grid">{''.join(cards)}</div>
<h2 class="section-title">图集使用说明</h2><p>预测栅格像元、匹配配对、空间块和独立实测样点分别对应不同统计单位；图中的像元关联不能直接解读为因果效应。表观拟合、结果导向筛选和内部空间留出均按各阶段报告中的解释边界使用。</p>
<p>图表数据源、图注和版本记录随审查材料保存在本目录的 artifacts/ 文件夹中。旧图按其原有样本范围和方法定义展示，不跨阶段直接比较数值。</p></main>
<script>
const cards=[...document.querySelectorAll('.card')],search=document.getElementById('search'),phase=document.getElementById('phase'),question=document.getElementById('question'),kind=document.getElementById('kind'),count=document.getElementById('count');
function update(){{const q=search.value.toLowerCase();let n=0;for(const c of cards){{const ok=(!q||c.innerText.toLowerCase().includes(q))&&(!phase.value||c.dataset.phase===phase.value)&&(!question.value||c.dataset.question===question.value)&&(!kind.value||c.dataset.chart_type===kind.value);c.hidden=!ok;if(ok)n++}}count.textContent=`显示 ${{n}} / ${{cards.length}} 张图`;}}
[search,phase,question,kind].forEach(x=>x.addEventListener('input',update));update();
</script></body></html>'''


def main() -> None:
    items: list[dict[str, Any]] = []
    add_initial(items)

    f_review = ROOT / "docs/reviews/2026-10-26/phase-f"
    add_phase_index(items, phase="Phase F｜匹配设计与空间稳定性",
                    question="环境背景相近时，侵蚀与健康是否仍有关联？",
                    experiment="M3/M4 配对差值、异质性与空间验证",
                    review_dir=f_review, index_path=f_review / "figure_index.json")
    g_review = ROOT / "docs/reviews/2026-10-27/phase-g"
    add_phase_index(items, phase="Phase G｜响应类型与环境异质性",
                    question="不同健康指标与环境背景的响应有何差异？",
                    experiment="配对秩相关、非线性模型与环境组筛选",
                    review_dir=g_review, index_path=g_review / "figures/figure_index.json")
    completion_dir = g_review / "figures/completion"
    completion_image = completion_dir / "10_environment_group_slope_forest.png"
    if completion_image.is_file():
        items.append(make_item(
            title="各指标负向环境组斜率与空间块区间", phase="Phase G｜响应类型与环境异质性",
            question="不同环境背景下，响应方向与强度是否不同？",
            experiment="按结局筛选最负环境组的空间块稳健斜率",
            chart_type="空间块稳健森林图", image=completion_image,
            caption=read_text(completion_dir / "captions/10_environment_group_slope_forest.md"),
            source=completion_dir / "data/10_environment_group_slope_forest.csv",
            svg=completion_image.with_suffix(".svg"), pdf=completion_image.with_suffix(".pdf"),
            scope_note="按健康响应排序，属于 Oracle 探索，不是独立确认。"))
    add_phase_h(items)
    add_low_regime(items)
    add_matched_unit_supplement(items)
    add_environment_group_heterogeneity(items)
    add_distribution_addendum(items)

    # Do not silently include incomplete or source-less records.
    items = [item for item in items if item.get("image")]
    ATLAS.parent.mkdir(parents=True, exist_ok=True)
    ATLAS.write_text(build_html(items), encoding="utf-8")
    (ATLAS.parent / "FIGURE_ATLAS_INDEX.md").write_text(
        f"# 中文科研图集\n\n离线图集：[FIGURE_ATLAS.html](FIGURE_ATLAS.html)。\n\n当前收录 **{len(items)} 张**图，按阶段、科学问题、实验和图表类型筛选；每张图提供 PNG、可用时的 SVG/PDF、图注和源数据链接。\n\n土壤类型代码：26—冲积土，27—潮土，42—脱潮土。\n",
        encoding="utf-8")
    print(f"写入 {ATLAS.relative_to(ROOT)}，共 {len(items)} 张图。")


if __name__ == "__main__":
    main()
