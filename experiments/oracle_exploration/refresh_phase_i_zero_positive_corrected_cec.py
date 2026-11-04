#!/usr/bin/env python3
"""Refresh Phase I I04 with the current formula-corrected CEC values."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phase_i_zero_positive_cec_mplconfig")

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT_RUN_ID = "run_20261025T114741_470772Z"
CORRECTED_INPUT_ID = "cec_corrected_input_20_2p9_plus18_20261031"
RUN_ID = "phase_i_cec_corrected_zero_positive_refresh_20261101_r3"
RUN_DIR = ROOT / "outputs/exploratory_oracle/phase_i" / RUN_ID
INPUT_DIR = ROOT / "outputs" / INPUT_RUN_ID
CORRECTED_DIR = ROOT / "outputs/exploratory_oracle/phase_i" / CORRECTED_INPUT_ID
ENV_PATH = INPUT_DIR / "databases/environment_erosion.parquet"
SOURCE_HEALTH_PATH = INPUT_DIR / "databases/health.parquet"
CORRECTED_HEALTH_PATH = CORRECTED_DIR / "databases/health.parquet"
CORRECTION_MANIFEST_PATH = CORRECTED_DIR / "correction_manifest.json"
DOC_FIGURES = ROOT / "docs/reviews/2026-10-31/phase-i/figures"

OUTCOMES = ["SHI_candidate", "AN", "OM", "CEC", "pH", "SW", "BD", "MBC", "RESP"]
LABELS = {
    "SHI_candidate": "土壤健康指数（SHI）",
    "AN": "碱解氮（AN）",
    "OM": "有机质（OM）",
    "CEC": "阳离子交换量（CEC）",
    "pH": "土壤酸碱度（pH）",
    "SW": "土壤含水量（SW）",
    "BD": "土壤容重（BD）",
    "MBC": "微生物量碳（MBC）",
    "RESP": "土壤呼吸（RESP）",
}
FORMULA = "CEC_corrected=(CEC-20×Erosion)/2.9+18"
PALETTE = {"blue": "#a8c5d8", "orange": "#e6b08b", "dark": "#263746", "grid": "#d8dee3"}

try:
    font_manager.findfont("WenQuanYi Zen Hei", fallback_to_default=False)
    plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
except ValueError:
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except Exception:
        return None


def _read_and_verify() -> tuple[pd.DataFrame, dict]:
    correction = json.loads(CORRECTION_MANIFEST_PATH.read_text(encoding="utf-8"))
    maxima = correction["maxima_full_valid_extent"]
    env = pd.read_parquet(
        ENV_PATH,
        columns=["cell_id", "row", "col", "x", "y", "Aa_20yr", "Aa_2020"],
    )
    source_health = pd.read_parquet(
        SOURCE_HEALTH_PATH,
        columns=["cell_id", "row", "col", "x", "y", *[o for o in OUTCOMES if o != "CEC"]],
    )
    source_cec = pd.read_parquet(SOURCE_HEALTH_PATH, columns=["cell_id", "CEC"])
    corrected_health = pd.read_parquet(
        CORRECTED_HEALTH_PATH,
        columns=["cell_id", "row", "col", "x", "y", *OUTCOMES],
    )

    for name, frame in (("environment", env), ("source health", source_health),
                        ("corrected health", corrected_health)):
        if not frame["cell_id"].is_unique:
            raise ValueError(f"{name} database has duplicate cell_id values.")
    if len(env) != len(source_health) or len(env) != len(corrected_health):
        raise ValueError("Environment and health databases have different row counts.")

    coordinates = env[["cell_id", "row", "col", "x", "y"]].merge(
        corrected_health[["cell_id", "row", "col", "x", "y"]],
        on="cell_id", suffixes=("_env", "_health"), validate="one_to_one",
    )
    for axis in ("row", "col"):
        if not coordinates[f"{axis}_env"].equals(coordinates[f"{axis}_health"]):
            raise ValueError(f"Raster {axis} coordinates disagree by cell_id.")
    for axis in ("x", "y"):
        left = pd.to_numeric(coordinates[f"{axis}_env"], errors="coerce").to_numpy(float)
        right = pd.to_numeric(coordinates[f"{axis}_health"], errors="coerce").to_numpy(float)
        if not np.isfinite(left).all() or not np.allclose(left, right, rtol=0, atol=1e-6):
            raise ValueError(f"Raster {axis} coordinates disagree by cell_id.")

    data = env.merge(
        corrected_health.drop(columns=["row", "col", "x", "y"]),
        on="cell_id", how="inner", validate="one_to_one",
    )
    data = data.merge(source_cec, on="cell_id", how="inner", suffixes=("", "_original"), validate="one_to_one")
    normalized = (
        pd.to_numeric(data["Aa_20yr"], errors="coerce") / float(maxima["Aa_20yr"])
        + pd.to_numeric(data["Aa_2020"], errors="coerce") / float(maxima["Aa_2020"])
    )
    expected_cec = (pd.to_numeric(data["CEC_original"], errors="coerce") - 20 * normalized) / 2.9 + 18
    stored_cec = pd.to_numeric(data["CEC"], errors="coerce").to_numpy(float)
    expected_cec = expected_cec.to_numpy(float)
    valid_cec = np.isfinite(stored_cec) & np.isfinite(expected_cec)
    max_abs_error = float(np.max(np.abs(stored_cec[valid_cec] - expected_cec[valid_cec])))
    if not np.allclose(stored_cec[valid_cec], expected_cec[valid_cec], rtol=0, atol=2e-5):
        raise ValueError(f"Corrected CEC failed formula recheck; max abs error={max_abs_error}.")

    # Confirm every non-CEC health column remained identical to its original source.
    original = source_health.merge(
        source_cec, on="cell_id", how="inner", validate="one_to_one"
    )
    comparison = corrected_health.merge(original, on="cell_id", suffixes=("_corrected", "_original"), validate="one_to_one")
    non_cec_columns = [o for o in OUTCOMES if o != "CEC"]
    for column in non_cec_columns:
        left = pd.to_numeric(comparison[f"{column}_corrected"], errors="coerce").to_numpy(float)
        right = pd.to_numeric(comparison[f"{column}_original"], errors="coerce").to_numpy(float)
        if not np.array_equal(np.isnan(left), np.isnan(right)) or not np.allclose(left, right, rtol=0, atol=0, equal_nan=True):
            raise ValueError(f"Non-CEC health column {column} changed in corrected input.")

    data["CEC"] = stored_cec
    verification = {
        "input_rows": int(len(data)),
        "formula_valid_cec_cells": int(valid_cec.sum()),
        "formula_max_abs_error": max_abs_error,
        "non_cec_health_columns_unchanged": True,
        "raster_coordinate_alignment": True,
        "correction_manifest": str(CORRECTION_MANIFEST_PATH.relative_to(ROOT)),
    }
    return data, verification


def _summary_table(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for outcome in OUTCOMES:
        values = pd.to_numeric(data[outcome], errors="coerce")
        exposure = pd.to_numeric(data["Aa_20yr"], errors="coerce")
        for group, group_mask in (("零侵蚀（Aa=0）", exposure.eq(0)), ("正侵蚀（Aa>0）", exposure.gt(0))):
            keep = group_mask & np.isfinite(values) & np.isfinite(exposure)
            sample = values.loc[keep].to_numpy(float)
            if not sample.size:
                continue
            row = {
                "outcome": outcome,
                "outcome_label": LABELS[outcome],
                "exposure_group": group,
                "n": int(sample.size),
                "mean": float(np.mean(sample)),
                "sd": float(np.std(sample, ddof=1)) if sample.size > 1 else np.nan,
                "min": float(np.min(sample)),
                "p05": float(np.quantile(sample, .05)),
                "p25": float(np.quantile(sample, .25)),
                "median": float(np.median(sample)),
                "p75": float(np.quantile(sample, .75)),
                "p95": float(np.quantile(sample, .95)),
                "max": float(np.max(sample)),
                "cec_correction_status": FORMULA if outcome == "CEC" else "not_applicable",
                "outcome_guided_selection": True,
            }
            rows.append(row)
    return pd.DataFrame(rows)


def _draw(data: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(3, 3, figsize=(13.4, 10.6), constrained_layout=True)
    for ax, outcome in zip(axes.flat, OUTCOMES, strict=True):
        pair = data[["Aa_20yr", outcome]].apply(pd.to_numeric, errors="coerce").dropna()
        zero = pair.loc[pair["Aa_20yr"].eq(0), outcome].to_numpy(float)
        positive = pair.loc[pair["Aa_20yr"].gt(0), outcome].to_numpy(float)
        box = ax.boxplot(
            [zero, positive], tick_labels=["Aa=0", "Aa>0"], showfliers=False,
            whis=(5, 95), patch_artist=True,
            medianprops={"color": PALETTE["dark"]},
        )
        for patch, color in zip(box["boxes"], (PALETTE["blue"], PALETTE["orange"]), strict=True):
            patch.set_facecolor(color)
            patch.set_alpha(.85)
        ax.set_title(LABELS[outcome])
        ax.set_ylabel("预测栅格值")
        ax.grid(axis="y", color=PALETTE["grid"], linewidth=.5)
    fig.suptitle("零侵蚀与正侵蚀像元的九项健康结局分布", fontsize=15, y=1.02)
    fig.text(
        .5, -.008,
        "箱体为第25-75百分位，中线为中位数，须线为第5-95百分位。图中 CEC 使用本项目当前数据版本。",
        ha="center", va="top", fontsize=8,
    )
    return fig


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    data, verification = _read_and_verify()
    summary = _summary_table(data)
    table_path = RUN_DIR / "tables/I04_zero_positive_boxplots_source.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(table_path, index=False)

    figure = _draw(data)
    figure_dir = RUN_DIR / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    stem = "I04_zero_positive_boxplots"
    for ext in ("png", "svg", "pdf"):
        figure.savefig(figure_dir / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(figure)

    DOC_FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg", "pdf"):
        shutil.copy2(figure_dir / f"{stem}.{ext}", DOC_FIGURES / f"i1_{stem}.{ext}")
    shutil.copy2(table_path, DOC_FIGURES / f"i1_{stem}_data.csv")

    manifest = {
        "run_id": RUN_ID,
        "mode": "EXPLORATORY_ORACLE",
        "outcome_guided_selection": True,
        "purpose": "Replace the stale I04 zero-versus-positive erosion figure with corrected CEC while preserving other outcomes and exposure groups.",
        "input_run_id": INPUT_RUN_ID,
        "corrected_input_id": CORRECTED_INPUT_ID,
        "inputs": {
            "environment_erosion": str(ENV_PATH.relative_to(ROOT)),
            "original_health": str(SOURCE_HEALTH_PATH.relative_to(ROOT)),
            "corrected_health": str(CORRECTED_HEALTH_PATH.relative_to(ROOT)),
            "correction_manifest": str(CORRECTION_MANIFEST_PATH.relative_to(ROOT)),
        },
        "formula": "CEC_corrected=(CEC-20*(Aa_20yr/143.00503540039062+Aa_2020/48.76557540893555))/2.9+18",
        "comparison": "Aa_20yr == 0 versus Aa_20yr > 0; each outcome uses its complete observations.",
        "verification": verification,
        "rows": {
            "input": int(len(data)),
            "outcome_group_summaries": int(len(summary)),
        },
        "outputs": {
            "figure_png": str((figure_dir / f"{stem}.png").relative_to(ROOT)),
            "figure_svg": str((figure_dir / f"{stem}.svg").relative_to(ROOT)),
            "figure_pdf": str((figure_dir / f"{stem}.pdf").relative_to(ROOT)),
            "source_csv": str(table_path.relative_to(ROOT)),
            "review_figure_png": str((DOC_FIGURES / f"i1_{stem}.png").relative_to(ROOT)),
        },
        "software": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": _package_version("numpy"),
            "pandas": _package_version("pandas"),
            "matplotlib": _package_version("matplotlib"),
        },
        "code": {
            "script": str(Path(__file__).resolve().relative_to(ROOT)),
            "git_commit": _git_commit(),
            "working_tree_modified": True,
        },
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (RUN_DIR / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"run_dir": str(RUN_DIR), **verification, "summary_rows": len(summary)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
