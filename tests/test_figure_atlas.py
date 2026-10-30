from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

import scripts.build_figure_atlas as atlas
from scripts.run_cropland_analysis import SOIL_TYPE_LABELS


def test_original_cluster_maps_are_saved_separately_from_today_maps(tmp_path, monkeypatch):
    run_dir = tmp_path / "outputs" / "run_20261025T114741_470772Z"
    map_dir = run_dir / "maps"
    figure_dir = run_dir / "figures"
    map_dir.mkdir(parents=True)
    figure_dir.mkdir()
    with rasterio.open(
        map_dir / "environment_groups_k6.tif", "w", driver="GTiff", width=3, height=2,
        count=1, dtype="int16", nodata=0, crs="EPSG:32650", transform=from_origin(0, 60, 30, 30),
    ) as dst:
        dst.write(np.array([[1, 2, 0], [3, 4, 5]], dtype=np.int16), 1)

    from scripts.archive_original_cluster_maps import archive_original_cluster_maps

    result = archive_original_cluster_maps(run_dir, cluster_counts=(6,))

    assert result == [figure_dir / "original_style" / "environment_groups_k6_original.png"]
    assert result[0].is_file()


def test_atlas_lists_original_style_maps_separately(tmp_path, monkeypatch):
    monkeypatch.setattr(atlas, "ROOT", tmp_path)
    monkeypatch.setattr(atlas, "ATLAS", tmp_path / "FIGURE_ATLAS.html")
    base = tmp_path / "docs/reviews/2026-10-29/phase-h/artifacts/initial/figures"
    base.mkdir(parents=True)
    for filename in atlas.INITIAL:
        (base / filename).write_bytes(b"today")
    old_path = base / "original_style" / "environment_groups_k6_original.png"
    old_path.parent.mkdir()
    old_path.write_bytes(b"original-style-restoration")

    items = []
    atlas.add_initial(items)

    old_items = [item for item in items if "原版" in item["title"]]
    today_items = [item for item in items if "新版" in item["title"]]
    assert len(old_items) == 1
    assert len(today_items) == 4
    assert old_items[0]["image"].endswith("environment_groups_k6_original.png")


def test_primary_soil_codes_have_source_backed_chinese_names():
    assert SOIL_TYPE_LABELS == {26: "冲积土", 27: "潮土", 42: "脱潮土"}


def test_atlas_lists_matched_unit_supplement_figures_with_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(atlas, "ROOT", tmp_path)
    run = tmp_path / "docs/reviews/2026-10-29/phase-h/artifacts/matched_units/figures"
    for fmt in ("png", "svg", "pdf"):
        path = run / fmt / f"M03_matched_unit_model_family_comparison.{fmt}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"figure")
    (run / "data").mkdir(parents=True, exist_ok=True)
    (run / "captions").mkdir(parents=True, exist_ok=True)
    (run / "data/M03_matched_unit_model_family_comparison_source.csv").write_text("source", encoding="utf-8")
    (run / "captions/M03_matched_unit_model_family_comparison.md").write_text("中文图注", encoding="utf-8")
    manifest = {"figures": [{
        "id": "M03_matched_unit_model_family_comparison", "title": "八类模型比较",
        "scientific_question": "匹配像元响应如何？", "experiment": "阈值后模型搜索", "figure_type": "模型比较图",
        "png": "png/M03_matched_unit_model_family_comparison.png",
        "svg": "svg/M03_matched_unit_model_family_comparison.svg",
        "pdf": "pdf/M03_matched_unit_model_family_comparison.pdf",
        "source": "data/M03_matched_unit_model_family_comparison_source.csv",
        "caption": "captions/M03_matched_unit_model_family_comparison.md",
    }]}
    (run / "figure_manifest.json").write_text(__import__("json").dumps(manifest), encoding="utf-8")

    items = []
    atlas.add_matched_unit_supplement(items)

    assert len(items) == 1
    assert items[0]["phase"] == "Phase H｜匹配像元与配对响应"
    assert items[0]["source"].endswith("M03_matched_unit_model_family_comparison_source.csv")


def test_atlas_lists_environment_group_heterogeneity_with_named_comparisons(tmp_path, monkeypatch):
    monkeypatch.setattr(atlas, "ROOT", tmp_path)
    run = tmp_path / "docs/reviews/2026-10-29/phase-h/artifacts/environment_group_heterogeneity/figures"
    records = []
    for stem in (
        "HET02_group_rho_change_20yr_T01_fixed_original",
        "HET05_group_matching_support_balance",
    ):
        for fmt in ("png", "svg", "pdf"):
            path = run / fmt / f"{stem}.{fmt}"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"figure")
        source = run / "data" / f"{stem}_source.csv"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("source", encoding="utf-8")
        caption = run / "captions" / f"{stem}.md"
        caption.parent.mkdir(parents=True, exist_ok=True)
        caption.write_text("中文图注", encoding="utf-8")
        records.append({
            "id": stem, "title": stem, "scientific_question": "组间响应如何变化？",
            "experiment": "组级对照", "figure_type": "统计图",
            "png": f"png/{stem}.png", "svg": f"svg/{stem}.svg", "pdf": f"pdf/{stem}.pdf",
            "source": f"data/{stem}_source.csv", "caption": f"captions/{stem}.md",
        })
    (run / "figure_manifest.json").write_text(
        __import__("json").dumps({"figures": records}), encoding="utf-8"
    )

    items = []
    atlas.add_environment_group_heterogeneity(items)

    assert len(items) == 2
    assert "20 年平均实际侵蚀" in items[0]["title"]
    assert "固定原环境组重配" in items[0]["title"]
    assert "环境组匹配支持量与平衡" == items[1]["title"]
    assert all(item["source"].endswith("_source.csv") for item in items)


def test_atlas_lists_distribution_addendum_figures_and_evidence_links(tmp_path, monkeypatch):
    import json

    monkeypatch.setattr(atlas, "ROOT", tmp_path)
    run = tmp_path / "docs/reviews/2026-10-29/phase-h/artifacts/distribution_diagnostic_addendum/figures"
    stem = "HADD01_gmm_seed_stability"
    for fmt in ("png", "svg", "pdf"):
        path = run / fmt / f"{stem}.{fmt}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"figure")
    (run / "data").mkdir(parents=True)
    (run / "captions").mkdir(parents=True)
    (run / "data" / f"{stem}_source.csv").write_text("source", encoding="utf-8")
    (run / "captions" / f"{stem}.md").write_text("随机种子边界图注", encoding="utf-8")
    manifest = {"figures": [{
        "id": stem, "title": "GMM 选中成分数与边界的随机种子敏感性",
        "scientific_question": "GMM 的分界受随机种子影响吗？",
        "experiment": "低侵蚀分布诊断补充", "figure_type": "随机种子稳定性图",
        "png": f"png/{stem}.png", "svg": f"svg/{stem}.svg", "pdf": f"pdf/{stem}.pdf",
        "source": f"data/{stem}_source.csv", "caption": f"captions/{stem}.md",
    }]}
    (run / "figure_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    items = []
    atlas.add_distribution_addendum(items)

    assert len(items) == 1
    assert items[0]["phase"] == "Phase H｜低侵蚀分布诊断补充"
    assert items[0]["title"] == "GMM 选中成分数与首处分界的随机种子敏感性"
    assert items[0]["source"].endswith(f"{stem}_source.csv")


def test_atlas_resolves_compressed_source_tables(tmp_path):
    source = tmp_path / "table.csv"
    compressed = tmp_path / "table.csv.gz"
    compressed.write_bytes(b"gzip-table")

    resolved, label = atlas.available_source(source)

    assert resolved == compressed
    assert label == "压缩源数据 CSV.gz"


def test_phase_index_accepts_inline_and_path_based_captions(tmp_path):
    review = tmp_path / "review"
    (review / "figures").mkdir(parents=True)
    (review / "figures/captions").mkdir()
    (review / "figures/01_inline.png").write_bytes(b"image")
    (review / "figures/02_file.png").write_bytes(b"image")
    (review / "figures/captions/02_file.md").write_text("文件图注", encoding="utf-8")
    index = review / "figure_index.json"
    index.write_text(__import__("json").dumps([
        {"name": "01_inline.png", "caption": "直接写在索引内的图注", "files": {"png": "figures/01_inline.png"}},
        {"name": "02_file.png", "caption": "figures/captions/02_file.md", "files": {"png": "figures/02_file.png"}},
    ]), encoding="utf-8")

    items = []
    atlas.add_phase_index(
        items, phase="测试阶段", question="问题", experiment="实验",
        review_dir=review, index_path=index,
    )

    assert [item["caption"] for item in items] == ["直接写在索引内的图注", "文件图注"]
