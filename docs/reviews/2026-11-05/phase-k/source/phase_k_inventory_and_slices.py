#!/usr/bin/env python3
"""Create an evidence-based F–J artifact inventory and Aa_20yr table slices."""
from __future__ import annotations

import csv
import html
import json
import re
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[5]
RUN = ROOT / "outputs/exploratory_oracle/phase_k/phase_k_20261104_seed20261104_r1"
REVIEW = ROOT / "docs/reviews/2026-11-05/phase-k"
SOURCE_INVENTORY = REVIEW / "results/DUAL_EROSION_ARTIFACT_INVENTORY.csv"
SLICES_REVIEW = REVIEW / "source_slices_20yr"
SLICES_RUN = RUN / "20yr_only/source_slices_20yr"
ATLAS = ROOT / "docs/FIGURE_ATLAS.html"


def phase_of(path: str) -> str:
    m = re.search(r"phase[-_ ]?([f-j])", path, re.I)
    return m.group(1).upper() if m else "跨阶段/项目索引"


def method_of(path: str) -> str:
    p = path.lower()
    rules = [
        (("match", "pair", "balance"), "环境匹配与配对比较"),
        (("gwr", "mgwr", "lisa", "gistar", "local"), "局部空间回归与空间关联"),
        (("sem", "path", "pl s", "pls", "cbsem"), "结构路径模型"),
        (("gam", "spline", "curve", "segmented", "threshold"), "非线性响应与阈值敏感性"),
        (("environment", "heterogeneity", "group", "soil_twi"), "环境异质性"),
        (("correlation", "association", "rho", "spearman", "pearson"), "相关与总体响应"),
        (("spatial_validation", "spatial_test", "holdout"), "空间留出验证"),
        (("inventory", "manifest", "provenance", "review_request"), "清单、配置或来源记录"),
    ]
    for tokens, label in rules:
        if any(token in p for token in tokens):
            return label
    return "其他阶段材料"


def artifact_type(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext in {".csv", ".tsv", ".parquet"}:
        return "数据表"
    if ext in {".png", ".svg", ".pdf", ".jpg", ".jpeg", ".webp"}:
        return "图件/报告PDF"
    if ext in {".json", ".yml", ".yaml", ".toml"}:
        return "配置或运行记录"
    if ext in {".md", ".html", ".tex"}:
        return "说明、报告或图集"
    return ext.lstrip(".") or "未知文件"


def read_atlas_cards() -> dict[str, str]:
    if not ATLAS.is_file():
        return {}
    source = ATLAS.read_text(encoding="utf-8", errors="ignore")
    cards = {}
    for card in re.findall(r'<article class="card".*?</article>', source, flags=re.S):
        paths = re.findall(r'(?:href|src)="([^"]+)"', card)
        text = html.unescape(re.sub(r"<[^>]+>", " ", card))
        for ref in paths:
            cards[ref.replace("\\", "/")] = text
    return cards


def period_text_flags(text: str) -> tuple[str, str]:
    has20 = bool(re.search(r"Aa[_ ]?20yr|20年平均(?:实际)?侵蚀", text, re.I))
    has2020 = bool(re.search(r"Aa[_ ]?2020|2020年(?:实际)?侵蚀", text, re.I))
    return ("TRUE" if has20 else "FALSE", "TRUE" if has2020 else "FALSE")


def source_period(path: Path, cards: dict[str, str]) -> tuple[str, str, str, str, str, str, str]:
    """Return evidence class, use flags, comparison flag, confidence, evidence."""
    ext = path.suffix.lower()
    rel = path.relative_to(ROOT).as_posix()
    if ext in {".csv", ".tsv"}:
        try:
            df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        except Exception as exc:
            return ("读取失败", "UNKNOWN", "UNKNOWN", "UNKNOWN", "LOW", f"读取失败：{type(exc).__name__}")
        if "exposure" in df.columns:
            vals = set(df["exposure"].dropna().astype(str))
            has20 = bool(vals & {"Aa_20yr", "log1p(Aa_20yr)"})
            has2020 = "Aa_2020" in vals
            if has20 and has2020:
                return ("双期同表数据", "TRUE", "TRUE", "TRUE", "HIGH", f"exposure字段同时含两期记录；值={','.join(sorted(vals))}")
            if has20:
                return ("20年单期数据", "TRUE", "FALSE", "FALSE", "HIGH", f"exposure字段仅含20年平均侵蚀；值={','.join(sorted(vals))}")
            if has2020:
                return ("2020单期数据", "FALSE", "TRUE", "FALSE", "HIGH", "exposure字段仅含2020年侵蚀")
        names = set(map(str, df.columns))
        if {"Aa_20yr", "Aa_2020"}.issubset(names):
            return ("双期宽表", "TRUE", "TRUE", "TRUE", "HIGH", "同一数据表含两期独立字段")
        return ("时期未标注数据表", "UNKNOWN", "UNKNOWN", "UNKNOWN", "LOW", "没有可直接判定侵蚀时期的字段")
    if ext in {".md", ".html", ".tex", ".json", ".yml", ".yaml"}:
        text = path.read_text(encoding="utf-8", errors="ignore")
        f20, f2020 = period_text_flags(text)
        if f20 == "TRUE" and f2020 == "TRUE":
            comparison = bool(re.search(r"比较|对照|comparison|versus|\bvs\b", text, re.I))
            return ("双期说明/配置", "UNKNOWN", "UNKNOWN", "TRUE" if comparison else "UNKNOWN", "MEDIUM", "文本提及两期；无明确比较措辞时不推定是否执行比较")
        if f20 == "TRUE":
            return ("20年单期说明", "TRUE", "FALSE", "FALSE", "MEDIUM", "文本只说明20年平均侵蚀")
        if f2020 == "TRUE":
            return ("2020单期说明", "FALSE", "TRUE", "FALSE", "MEDIUM", "文本只说明2020年侵蚀")
        return ("时期未标注说明", "UNKNOWN", "UNKNOWN", "UNKNOWN", "LOW", "未发现明确时期标记")
    if ext in {".png", ".svg", ".pdf", ".jpg", ".jpeg", ".webp"}:
        name_text = path.stem
        caption = ""
        candidates = [
            path.parent / "captions" / f"{path.stem}.md",
            path.parent.parent / "captions" / f"{path.stem}.md",
            path.parent.parent / "captions" / f"{path.stem}_caption.md",
        ]
        for candidate in candidates:
            if candidate.is_file():
                caption = candidate.read_text(encoding="utf-8", errors="ignore")
                break
        card = cards.get(rel.removeprefix("docs/"), "")
        evidence = " ".join([name_text, caption, card])
        f20, f2020 = period_text_flags(evidence)
        if f20 == "TRUE" and f2020 == "TRUE":
            comparison = bool(re.search(r"比较|对照|comparison|versus|\bvs\b", evidence, re.I))
            return ("双期图/双期说明", "UNKNOWN", "UNKNOWN", "TRUE" if comparison else "UNKNOWN", "MEDIUM", "图名、图注或总图卡同时提及两期；无明确比较措辞时不推定分析形式")
        if f20 == "TRUE":
            return ("20年单期图", "TRUE", "FALSE", "FALSE", "MEDIUM", "图名、图注或总图卡只指向20年平均侵蚀")
        if f2020 == "TRUE":
            return ("2020单期图", "FALSE", "TRUE", "FALSE", "MEDIUM", "图名、图注或总图卡只指向2020年侵蚀")
        return ("时期未核实图件", "UNKNOWN", "UNKNOWN", "UNKNOWN", "LOW", "图名、图注和总图卡没有明确暴露时期证据")
    return ("时期未核实", "UNKNOWN", "UNKNOWN", "UNKNOWN", "LOW", "没有适用的时期识别规则")


def make_slice(path: Path, df: pd.DataFrame) -> tuple[Path, int, list[str], str]:
    selected = df[df["exposure"].astype(str).isin({"Aa_20yr", "log1p(Aa_20yr)"})].copy()
    removed = []
    for col in selected.columns:
        if col == "exposure":
            continue
        lower = str(col).lower()
        drop = bool(re.search(r"aa_2020|2020", lower))
        if not drop and selected[col].dtype == object:
            try:
                drop = selected[col].astype(str).str.contains(r"Aa_2020", case=False, regex=True, na=False).any()
            except Exception:
                drop = False
        if drop:
            removed.append(str(col))
    if removed:
        selected.drop(columns=removed, inplace=True)
    # Keep the exposure field explicitly restricted to Aa_20yr semantics.
    selected["exposure"] = "Aa_20yr"
    rel = path.relative_to(ROOT / "docs/reviews")
    out_rel = Path("source_slices_20yr") / rel.parent / f"{path.stem}_Aa20_only.csv"
    review_path = REVIEW / out_rel
    run_path = SLICES_RUN / rel.parent / f"{path.stem}_Aa20_only.csv"
    review_path.parent.mkdir(parents=True, exist_ok=True)
    run_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(review_path, index=False, encoding="utf-8-sig")
    shutil.copy2(review_path, run_path)
    bad = []
    for col in selected.columns:
        if re.search(r"aa_2020|2020", str(col), re.I):
            bad.append(str(col))
    for col in selected.select_dtypes(include="object").columns:
        if selected[col].astype(str).str.contains(r"Aa_2020", case=False, regex=True, na=False).any():
            bad.append(str(col))
    return review_path, len(selected), removed, ";".join(sorted(set(bad)))


def main() -> None:
    cards = read_atlas_cards()
    original = pd.read_csv(SOURCE_INVENTORY, encoding="utf-8-sig", low_memory=False)
    out_rows, slice_rows = [], []
    for row in original.to_dict(orient="records"):
        rel = str(row["审查材料路径"])
        path = ROOT / rel
        cls, use20, use2020, comparison, confidence, evidence = source_period(path, cards)
        existing = use20 == "TRUE"
        requires = comparison == "TRUE"
        new_path = ""
        if path.is_file() and path.suffix.lower() in {".csv", ".tsv"} and cls == "双期同表数据":
            df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
            sliced_path, n_out, removed, residual = make_slice(path, df)
            new_path = sliced_path.relative_to(ROOT).as_posix()
            slice_rows.append({
                "source_file": rel,
                "source_periods": "Aa_20yr;Aa_2020",
                "source_rows": len(df),
                "selected_Aa20_rows": n_out,
                "removed_other_period_columns": ";".join(removed),
                "output_path": new_path,
                "remaining_other_period_fields_or_values": residual,
                "status": "PASS" if not residual else "REVIEW",
            })
            status = "已生成仅20年平均侵蚀表切片" if not residual else "切片仍含其他时期标记，待核对"
        elif cls in {"20年单期数据", "20年单期图", "20年单期说明"}:
            new_path = rel
            status = "已有20年平均侵蚀单期材料"
        elif cls == "2020单期数据":
            sibling = path.with_name(path.name.replace("Aa_2020", "Aa_20yr"))
            if sibling.is_file():
                new_path = sibling.relative_to(ROOT).as_posix()
                status = "已有同类20年单期配套材料"
            else:
                status = "仅有2020单期材料，未发现同名20年版本"
        elif cls in {"双期说明/配置", "双期图/双期说明"}:
            status = "历史双期说明保留；对应20年结果见Phase K单期包"
        else:
            status = "暴露时期证据不足，未作单期推断"
        method = method_of(rel)
        new_record = {
            **row,
            "阶段": phase_of(rel),
            "审查材料路径": rel,
            "文件类型": Path(rel).suffix.lower(),
            "暴露时期判定": {
                "双期同表数据": "双期同表（已生成20年单期切片）",
                "双期宽表": "双期宽表",
                "20年单期数据": "单独20年侵蚀材料",
                "2020单期数据": "单独2020侵蚀材料",
                "双期说明/配置": "涉及双期的说明/配置",
                "20年单期说明": "单独20年侵蚀材料",
                "2020单期说明": "单独2020侵蚀材料",
                "双期图/双期说明": "双期图件/说明（执行时期待核实）",
                "20年单期图": "单独20年侵蚀材料",
                "2020单期图": "单独2020侵蚀材料",
                "时期未标注数据表": "时期未知（数据表）",
                "时期未标注说明": "时期未知（说明）",
                "时期未核实图件": "时期未知（图件）",
                "读取失败": "读取失败，时期未知",
            }.get(cls, cls),
            "判定依据": evidence,
            "20年单期材料位置": new_path,
            "source_phase": phase_of(rel),
            "source_file": rel,
            "artifact_type": artifact_type(rel),
            "analysis_method": method,
            "uses_20yr": use20,
            "uses_2020": use2020,
            "contains_dual_exposure_comparison": comparison,
            "requires_20yr_only": "TRUE" if requires else ("UNKNOWN" if comparison == "UNKNOWN" else "FALSE"),
            "existing_20yr_only": "TRUE" if existing and not requires else "FALSE",
            "new_20yr_only_path": new_path,
            "evidence_confidence": confidence,
            "classification_evidence": evidence,
            "status": status,
        }
        out_rows.append(new_record)
    enriched = pd.DataFrame(out_rows)
    enriched.to_csv(SOURCE_INVENTORY, index=False, encoding="utf-8-sig")
    manifest = pd.DataFrame(slice_rows)
    manifest.to_csv(REVIEW / "results/DUAL_TABLE_20YR_SLICE_MANIFEST.csv", index=False, encoding="utf-8-sig")
    manifest.to_csv(RUN / "tables/DUAL_TABLE_20YR_SLICE_MANIFEST.csv", index=False, encoding="utf-8-sig")
    print(f"inventory_rows={len(enriched)}; table_slices={len(manifest)}")
    print("period classifications:")
    print(enriched["暴露时期判定"].value_counts(dropna=False).to_string())
    print("slice status:")
    print(manifest["status"].value_counts(dropna=False).to_dict())
    print(f"review_slice_files={len(list(SLICES_REVIEW.rglob('*.csv')))}")


if __name__ == "__main__":
    main()
