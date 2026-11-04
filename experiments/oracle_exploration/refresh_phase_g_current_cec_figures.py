#!/usr/bin/env python3
"""Re-render Phase G CEC figures from the current CEC data and saved curve fits."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REVIEW = ROOT / "docs/reviews/2026-10-27/phase-g"
OLD_RUN = ROOT / "outputs/exploratory_oracle/phase_g_20261027"
CURRENT_HEALTH = ROOT / "outputs/exploratory_oracle/phase_i/cec_corrected_input_20_2p9_plus18_20261031/databases/health.parquet"
RUN_ID = "phase_i_20261101_phase_g_current_cec_figure_refresh_r2"
RUN_DIR = ROOT / "outputs/exploratory_oracle/phase_i" / RUN_ID
OUTCOMES = ["SHI_candidate", "AN", "OM", "CEC", "pH", "SW", "BD", "MBC", "RESP"]
LABELS = {
    "SHI_candidate": "土壤健康指数（SHI）", "AN": "碱解氮（AN）", "OM": "有机质（OM）",
    "CEC": "阳离子交换量（CEC）", "pH": "土壤酸碱度（pH）", "SW": "土壤含水量（SW）",
    "BD": "土壤容重（BD）", "MBC": "微生物量碳（MBC）", "RESP": "土壤呼吸（RESP）",
}
PALETTE = {"blue": "#3D6F91", "orange": "#D99139", "red": "#B4494B", "gray": "#7A8084", "teal": "#3B8C83"}

plt.rcParams.update({
    "font.family": "Noto Sans CJK JP", "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7,
    "axes.edgecolor": "#4B5563", "axes.linewidth": 0.8, "grid.color": "#D9DEE7",
    "grid.linewidth": 0.55, "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def _save(fig: plt.Figure, stem: str, caption: str, source: pd.DataFrame) -> None:
    run_figures = RUN_DIR / "figures"
    review_figures = REVIEW / "figures"
    for extension in ("png", "svg", "pdf"):
        run_path = run_figures / f"{stem}.{extension}"
        review_path = review_figures / f"{stem}.{extension}"
        run_path.parent.mkdir(parents=True, exist_ok=True)
        review_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(run_path, dpi=320, bbox_inches="tight")
        review_path.write_bytes(run_path.read_bytes())
    plt.close(fig)
    table_path = RUN_DIR / "tables" / f"{stem}.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    source.to_csv(table_path, index=False, encoding="utf-8-sig")
    if stem == "04_model_r2_comparison":
        review_source = REVIEW / "figures/data/04_model_r2_comparison.csv"
        review_source.write_text(source.to_csv(index=False), encoding="utf-8-sig")
        caption_path = REVIEW / "figures/captions/04_model_r2_comparison.md"
    else:
        caption_path = REVIEW / "figures/captions/05_response_curves.md"
        source.to_csv(REVIEW / "figures/data/05_response_scatter_source.csv", index=False, encoding="utf-8-sig")
    caption_path.write_text(caption.rstrip() + "\n", encoding="utf-8")


def _refresh_r2() -> None:
    data = pd.read_csv(REVIEW / "figures/data/04_model_r2_comparison.csv")
    fig, ax = plt.subplots(figsize=(13, 5.5), constrained_layout=True)
    ranking_types = [
        "highest_full_sample_apparent_r2", "highest_apparent_r2",
        "highest_negative_apparent_r2", "validation_selected_spatial_test",
    ]
    labels = ["全体合格像元\n最高表观拟合", "匹配子集\n最高表观拟合", "匹配子集\n最高负向拟合", "验证集选模后\n空间测试"]
    colors = [PALETTE["gray"], PALETTE["blue"], PALETTE["orange"], PALETTE["teal"]]
    x = np.arange(len(OUTCOMES))
    width = .19
    for index, (ranking, label, color) in enumerate(zip(ranking_types, labels, colors, strict=True)):
        values = data.loc[data.ranking_type.eq(ranking)].set_index("outcome").r_squared.reindex(OUTCOMES)
        ax.bar(x + (index - 1.5) * width, values, width=width, color=color, label=label.replace("\n", " "))
    ax.axhline(0, color="#222222", linewidth=.7)
    ax.set_xticks(x, [LABELS[outcome] for outcome in OUTCOMES], rotation=25, ha="right")
    ax.set_ylabel("R²（各行按对应测试集定义）")
    ax.set_title("20 年平均实际侵蚀下的像元级健康预测拟合")
    ax.legend(frameon=False, ncol=2, loc="best")
    ax.text(.01, -.27, "CEC 当前没有与其他结局同口径的全像元表观 R²；这里只显示空间测试 R²。", transform=ax.transAxes, fontsize=8, color="#4B5563")
    caption = "CEC 使用本项目当前数据版本。当前没有可与其他结局同口径比较的全像元表观 R²，因此相应柱位留空；图中显示其验证集选模后的空间测试结果。"
    _save(fig, "04_model_r2_comparison", caption, data)


def _refresh_curves() -> None:
    curves = pd.read_csv(REVIEW / "figures/data/05_response_curves.csv")
    scatter = pd.read_csv(OLD_RUN / "tables/response_curve_sample_points.csv")
    current_cec = pd.read_parquet(CURRENT_HEALTH, columns=["cell_id", "CEC"])
    cec_points = scatter.loc[scatter.outcome.eq("CEC")].drop(columns=["observed_outcome"])
    cec_points = cec_points.merge(current_cec, on="cell_id", how="left", validate="one_to_one")
    cec_points = cec_points.loc[pd.to_numeric(cec_points.CEC, errors="coerce").notna()].copy()
    cec_points["observed_outcome"] = pd.to_numeric(cec_points["CEC"], errors="coerce")
    cec_points = cec_points.drop(columns="CEC")
    scatter = pd.concat([scatter.loc[~scatter.outcome.eq("CEC")], cec_points], ignore_index=True)

    fig, axes = plt.subplots(3, 3, figsize=(13, 10), constrained_layout=True)
    for position, (ax, outcome) in enumerate(zip(axes.flat, OUTCOMES, strict=True)):
        points = scatter.loc[scatter.outcome.eq(outcome)]
        ax.scatter(points.exposure_value, points.observed_outcome, s=3, color="#87939B", alpha=.11, rasterized=True, label="固定抽样像元")
        subset = curves.loc[curves.outcome.eq(outcome) & curves.exposure.eq("Aa_20yr")]
        natural = subset.loc[subset.curve_type.eq("fixed_common_sample_natural_spline")].sort_values("exposure_value")
        if not natural.empty:
            ax.plot(natural.exposure_value, natural.predicted_outcome, color=PALETTE["orange"], linestyle="--", linewidth=1.7, label="固定样本自然样条")
        best = subset.loc[subset.curve_type.eq("oracle_best_full_sample_apparent_model")].sort_values("exposure_value")
        title = LABELS[outcome]
        if not best.empty:
            row = best.iloc[0]
            if outcome == "CEC":
                model_label = "高斯广义加性模型（GAM）· 空间验证"
                line_label = "5 km 空间验证选出的高斯广义加性模型（GAM）"
            else:
                model_label = {"random_forest": "随机森林", "ols": "普通最小二乘", "extra_trees": "极端随机树"}.get(str(row.model_name), str(row.model_name))
                line_label = "历史全样本择优表观模型"
            ax.plot(best.exposure_value, best.predicted_outcome, color=PALETTE["red"], linewidth=1.7, label=line_label)
            title = f"{LABELS[outcome]} · {model_label} R²={float(row.apparent_r_squared):.3f}"
        ax.set_title(title, loc="left", fontweight="semibold")
        ax.set_xlabel("20 年平均实际侵蚀 Aa（源栅格值）")
        ax.set_ylabel("健康指标预测值")
        ax.grid(color="#D9DEE7", linewidth=.55)
        if position == 0:
            ax.legend(frameon=False, fontsize=6)
    fig.suptitle("两类描述性响应曲线｜固定 20,000 像元样本与各指标拟合模型", x=.02, y=1.02, ha="left", fontsize=15, fontweight="bold")
    caption = "虚线为固定共同样本的自然样条；CEC 散点和曲线均使用本项目当前数据版本，红色实线为按 5 km 空间验证选出的高斯广义加性模型（GAM）。其他指标红线沿用历史全样本模型搜索结果。"
    _save(fig, "05_response_curves", caption, curves)
    scatter.to_csv(REVIEW / "figures/data/05_response_scatter_source.csv", index=False, encoding="utf-8-sig")
    scatter.to_csv(RUN_DIR / "tables/05_response_scatter_source.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    if RUN_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {RUN_DIR}")
    _refresh_r2()
    _refresh_curves()
    manifest = {
        "run_id": RUN_ID,
        "mode": "EXPLORATORY_ORACLE",
        "outcome_guided_selection": True,
        "purpose": "Re-render the two Phase G figures that display CEC, using current CEC sample values and plain CEC labels.",
        "inputs": {
            "current_cec_health": str(CURRENT_HEALTH.relative_to(ROOT)),
            "existing_phase_g_curve_fits": "docs/reviews/2026-10-27/phase-g/figures/data/05_response_curves.csv",
            "fixed_sample_cell_ids": "outputs/exploratory_oracle/phase_g_20261027/tables/response_curve_sample_points.csv",
            "model_comparison_table": "docs/reviews/2026-10-27/phase-g/figures/data/04_model_r2_comparison.csv",
        },
        "outputs": ["04_model_r2_comparison", "05_response_curves"],
        "cec_scatter_points": int(len(pd.read_csv(REVIEW / "figures/data/05_response_scatter_source.csv").query("outcome == 'CEC'"))),
    }
    (RUN_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
