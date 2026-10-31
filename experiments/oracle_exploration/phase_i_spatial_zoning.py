"""Phase I-8 exploratory spatial stress zoning from existing Phase I outputs."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phase_i_spatial_zoning_mplconfig")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import PolyCollection
import numpy as np
import pandas as pd
import scipy
import yaml
from rasterio.warp import transform as warp_transform
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase_i_spatial as spatial  # noqa: E402

ZONE_ORDER = [
    "T1 高Aa—强负向响应",
    "T2 高Aa—弱负向响应",
    "T3 低Aa—低SHI",
    "T4 环境敏感候选",
    "其他/未分类",
]
ZONE_COLORS = {
    ZONE_ORDER[0]: "#B2182B",
    ZONE_ORDER[1]: "#EF8A62",
    ZONE_ORDER[2]: "#4393C3",
    ZONE_ORDER[3]: "#7B6FD0",
    ZONE_ORDER[4]: "#D9DEE3",
}
ENV_COLORS = {
    "No candidate overlay": "#D9DEE3",
    "27潮土·TWI q2 proxy": "#7B6FD0",
    "26冲积土·TWI q1 proxy": "#1B9E77",
}
try:
    from matplotlib import font_manager

    font_manager.findfont("WenQuanYi Zen Hei", fallback_to_default=False)
    plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
except (ImportError, ValueError):
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False


def _version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True,
                              text=True).stdout.strip()
    except Exception:
        return None


def _read_units(config: dict[str, Any]) -> tuple[pd.DataFrame, tuple[np.ndarray, np.ndarray, np.ndarray], dict[str, Any]]:
    db = ROOT / "outputs" / config["input_run_id"] / "databases"
    env_path = db / "environment_erosion.parquet"
    health_path = db / "health.parquet"
    env_cols = ["cell_id", "row", "col", "x", "y", config["primary_exposure"],
                "SoilType", *config["environment_covariates"]]
    env = pd.read_parquet(env_path, columns=env_cols)
    health = pd.read_parquet(health_path, columns=["cell_id", config["health_outcome"]])
    pixels = env.merge(health, on="cell_id", how="inner", validate="one_to_one")
    if not pixels["cell_id"].is_unique:
        raise ValueError("Pixel sources must align one-to-one by cell_id.")
    mask_pixels = pixels[["row", "col", "x", "y"]].dropna().copy()
    mask_info = spatial._mask_arrays(mask_pixels)

    required = [config["primary_exposure"], config["health_outcome"], *config["environment_covariates"]]
    numeric = pixels[required].apply(pd.to_numeric, errors="coerce")
    valid = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
    analysis = pixels.loc[valid, ["row", "col", "x", "y", "SoilType", *required]].copy()
    analysis[required] = numeric.loc[valid, required]
    analysis["SoilType"] = pd.to_numeric(analysis["SoilType"], errors="coerce")
    tile_cells = int(config["tile_cells"])
    analysis["tile_row"] = np.floor(analysis["row"].to_numpy(dtype=float) / tile_cells).astype(np.int32)
    analysis["tile_col"] = np.floor(analysis["col"].to_numpy(dtype=float) / tile_cells).astype(np.int32)

    exposure, health_name = config["primary_exposure"], config["health_outcome"]
    group_cols = ["tile_row", "tile_col"]
    profile_cols = ["x", "y", exposure, health_name, *config["environment_covariates"]]
    units = analysis.groupby(group_cols, sort=True)[profile_cols].mean().reset_index()
    count = analysis.groupby(group_cols, sort=True).size().rename("n_pixels").reset_index()
    units = units.merge(count, on=group_cols, how="left", validate="one_to_one")
    units["Aa_20yr_mean"] = units[exposure]
    units["SHI_mean"] = units[health_name]

    soil = analysis.loc[np.isfinite(analysis["SoilType"])].copy()
    soil_counts = (soil.groupby(group_cols + ["SoilType"], sort=True).size()
                   .rename("soil_pixels").reset_index())
    soil_counts["tile_total"] = soil_counts.groupby(group_cols)["soil_pixels"].transform("sum")
    dominant = (soil_counts.sort_values(group_cols + ["soil_pixels", "SoilType"],
                                         ascending=[True, True, False, True])
                .drop_duplicates(group_cols)[group_cols + ["SoilType", "soil_pixels", "tile_total"]]
                .rename(columns={"SoilType": "dominant_soil_type", "soil_pixels": "dominant_soil_pixels"}))
    dominant["dominant_soil_share"] = dominant["dominant_soil_pixels"] / dominant["tile_total"]
    units = units.merge(dominant[group_cols + ["dominant_soil_type", "dominant_soil_share"]],
                        on=group_cols, how="left", validate="one_to_one")

    for soil_code, label in [(27, "soil27"), (26, "soil26")]:
        d = analysis[analysis["SoilType"].eq(soil_code)]
        if d.empty:
            continue
        soil_twi = d.groupby(group_cols, sort=True).agg(
            **{f"{label}_pixels": ("SoilType", "size"), f"{label}_TWI": ("TWI", "mean")}
        ).reset_index()
        units = units.merge(soil_twi, on=group_cols, how="left", validate="one_to_one")
    meta = {"environment_db": str(env_path.relative_to(ROOT)),
            "health_db": str(health_path.relative_to(ROOT)),
            "mask_pixels": int(len(mask_pixels)), "environment_complete_pixels": int(len(analysis)),
            "tile_cells": tile_cells, "tile_units": int(len(units))}
    return units, mask_info, meta


def _merge_source_outputs(units: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    sources = {key: ROOT / value for key, value in config["source_runs"].items()}
    out = units.copy()
    i4_path = sources["i4"] / "tables/full_sample_local_coefficients.parquet"
    i4 = pd.read_parquet(i4_path)
    i4 = i4[(i4["outcome"] == config["health_outcome"])
            & (i4["analysis_population"] == "environment_adjusted")]
    gwr = i4[i4["model_family"].eq("GWR")][
        ["tile_row", "tile_col", "local_beta_standardized", "local_ci_low_model_based",
         "local_ci_high_model_based", "local_r2", "n_pixels"]
    ].rename(columns={"local_beta_standardized": "gwr_beta", "local_ci_low_model_based": "gwr_ci_low",
                      "local_ci_high_model_based": "gwr_ci_high", "local_r2": "gwr_local_r2",
                      "n_pixels": "i4_gwr_pixels"})
    lcr = i4[i4["model_family"].eq("GWR-LCR")][
        ["tile_row", "tile_col", "local_beta_standardized"]
    ].rename(columns={"local_beta_standardized": "gwr_lcr_beta"})
    out = out.merge(gwr, on=["tile_row", "tile_col"], how="inner", validate="one_to_one")
    out = out.merge(lcr, on=["tile_row", "tile_col"], how="left", validate="one_to_one")

    i4_svc_path = sources["i4"] / "tables/multiscale_svc_coefficients.parquet"
    svc = pd.read_parquet(i4_svc_path)
    svc = svc[(svc["outcome"] == config["health_outcome"])
              & (svc["analysis_population"] == "environment_adjusted")]
    svc = svc[["tile_row", "tile_col", "beta_log1p_Aa_standardized"]].rename(
        columns={"beta_log1p_Aa_standardized": "multiscale_svc_beta"})
    out = out.merge(svc, on=["tile_row", "tile_col"], how="left", validate="one_to_one")

    i6 = sources["i6"]
    rho = pd.read_parquet(i6 / "tables/local_spearman.parquet")
    rho = rho[rho["outcome"].eq(config["health_outcome"])][
        ["tile_row", "tile_col", "local_spearman", "radius_to_farthest_neighbor_m"]
    ]
    out = out.merge(rho, on=["tile_row", "tile_col"], how="left", validate="one_to_one")
    lisa = pd.read_parquet(i6 / "tables/bivariate_local_moran.parquet")
    lisa = lisa[lisa["outcome"].eq(config["health_outcome"])][
        ["tile_row", "tile_col", "lisa_class", "q_bh_within_outcome", "same_site_class"]
    ]
    out = out.merge(lisa, on=["tile_row", "tile_col"], how="left", validate="one_to_one")
    gi = pd.read_parquet(i6 / "tables/local_negative_response_gistar.parquet")
    gi = gi[(gi["outcome"].eq(config["health_outcome"]))
            & gi["analysis_population"].eq("environment_adjusted")]
    for model, suffix in [("GWR", "gwr"), ("GWR-LCR", "gwr_lcr")]:
        g = gi[gi["model_family"].eq(model)][
            ["x", "y", "gi_star_class", "local_Gi_star_z"]
        ].rename(columns={"gi_star_class": f"gi_star_{suffix}_class",
                          "local_Gi_star_z": f"gi_star_{suffix}_z"})
        out = out.merge(g, on=["x", "y"], how="left", validate="one_to_one")

    i5 = pd.read_csv(sources["i5"] / "tables/grf_local_metrics.csv")
    grf = i5[i5["outcome"].eq(config["health_outcome"])].copy()
    if grf.empty:
        out["grf_local_effect"] = np.nan
        out["grf_local_r2"] = np.nan
        out["grf_center_distance_m"] = np.nan
        grf_centers = 0
    else:
        coords = grf[["x", "y"]].to_numpy(dtype=float)
        distances, indexes = cKDTree(coords).query(out[["x", "y"]].to_numpy(dtype=float), k=1)
        out["grf_center_distance_m"] = distances
        out["grf_local_effect"] = grf.iloc[indexes]["local_Aa_marginal_effect_standardized"].to_numpy()
        out["grf_local_r2"] = grf.iloc[indexes]["local_r2_random_holdout"].to_numpy()
        grf_centers = int(len(grf))

    if len(out) != len(units):
        raise ValueError(f"I-4 GWR spatial support mismatch: {len(units)} units, {len(out)} merged.")
    meta = {"i4_gwr_units": int(len(gwr)), "i4_lcr_units": int(len(lcr)),
            "i4_multiscale_svc_units": int(len(svc)), "i6_local_spearman_units": int(len(rho)),
            "i6_lisa_units": int(len(lisa)), "i5_grf_shi_centers": grf_centers,
            "grf_join": "nearest 1.2km local center by Euclidean distance in EPSG:32649; diagnostics only"}
    return out, meta


def _classify_zones(data: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, float]]:
    d = data.copy()
    exposure = d["Aa_20yr_mean"].to_numpy(dtype=float)
    health = d["SHI_mean"].to_numpy(dtype=float)
    beta = d["gwr_beta"].to_numpy(dtype=float)
    thresholds = {
        "high_Aa": float(np.quantile(exposure, float(config["high_erosion_quantile"]))),
        "strong_negative_beta": float(np.quantile(beta, float(config["strong_negative_beta_quantile"]))),
        "low_Aa": float(np.quantile(exposure, float(config["low_erosion_quantile"]))),
        "low_SHI": float(np.quantile(health, float(config["low_shi_quantile"]))),
    }
    high = exposure >= thresholds["high_Aa"]
    strong = (beta < 0) & (beta <= thresholds["strong_negative_beta"])
    weak = (beta < 0) & ~strong
    low_erosion_low_shi = (exposure <= thresholds["low_Aa"]) & (health <= thresholds["low_SHI"])
    d["high_Aa"] = high
    d["strong_negative_GWR"] = strong
    d["weak_negative_GWR"] = weak
    d["low_Aa_low_SHI"] = low_erosion_low_shi

    for soil_code, label in [(27, "soil27"), (26, "soil26")]:
        share = float(config["soil27_tile_share_min"])
        twi_col = f"{label}_TWI"
        if twi_col not in d:
            d[f"{label}_twi_candidate"] = False
            thresholds[f"{label}_TWI_q_low"] = float("nan")
            thresholds[f"{label}_TWI_q_high"] = float("nan")
            continue
        base = d[(d["dominant_soil_type"].eq(soil_code))
                 & (d["dominant_soil_share"] >= share) & d[twi_col].notna()]
        q_low, q_high = config["soil27_twi_q2"] if soil_code == 27 else [0.0, 0.20]
        low_cut, high_cut = (float(base[twi_col].quantile(float(q_low))),
                             float(base[twi_col].quantile(float(q_high)))) if len(base) else (np.nan, np.nan)
        d[f"{label}_twi_candidate"] = (d["dominant_soil_type"].eq(soil_code)
                                         & d["dominant_soil_share"].ge(share)
                                         & d[twi_col].between(low_cut, high_cut, inclusive="both"))
        thresholds[f"{label}_TWI_q_low"] = low_cut
        thresholds[f"{label}_TWI_q_high"] = high_cut
        thresholds[f"{label}_candidate_units"] = float(len(base))

    d["environment_sensitive_candidate"] = d["soil27_twi_candidate"] | d["soil26_twi_candidate"]
    zone = np.full(len(d), ZONE_ORDER[4], dtype=object)
    t1 = high & strong
    t2 = high & weak
    t3 = low_erosion_low_shi & ~t1 & ~t2
    t4 = d["environment_sensitive_candidate"].to_numpy(dtype=bool) & ~t1 & ~t2 & ~t3
    zone[t1] = ZONE_ORDER[0]
    zone[t2] = ZONE_ORDER[1]
    zone[t3] = ZONE_ORDER[2]
    zone[t4] = ZONE_ORDER[3]
    d["primary_zone"] = zone
    d["high_Aa_nonnegative_GWR"] = high & (beta >= 0)
    return d, thresholds


def _zone_tables(data: pd.DataFrame, thresholds: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = data.copy()
    pixel_total = float(d["n_pixels"].sum())
    unit_total = float(len(d))
    summary_rows = []
    profile_rows = []
    for zone in ZONE_ORDER:
        g = d[d["primary_zone"].eq(zone)]
        n_units, n_pixels = len(g), int(g["n_pixels"].sum())
        if not len(g):
            continue
        summary_rows.append({
            "zone": zone, "spatial_units_240m": n_units, "share_of_units": n_units / unit_total,
            "complete_case_pixels": n_pixels, "share_of_complete_case_pixels": n_pixels / pixel_total,
            "area_km2_equivalent": n_pixels * 0.0009,
            "mean_Aa_20yr": float(g["Aa_20yr_mean"].mean()), "median_Aa_20yr": float(g["Aa_20yr_mean"].median()),
            "mean_SHI": float(g["SHI_mean"].mean()), "median_SHI": float(g["SHI_mean"].median()),
            "median_GWR_beta": float(g["gwr_beta"].median()),
            "negative_GWR_share": float((g["gwr_beta"] < 0).mean()),
            "negative_GWR_LCR_share": float((g["gwr_lcr_beta"] < 0).mean()),
            "negative_multiscale_SVC_share": float((g["multiscale_svc_beta"] < 0).mean()),
            "median_local_Spearman": float(g["local_spearman"].median()),
            "LISA_HL_share": float(g["lisa_class"].eq("HL").mean()),
            "GWR_GiStar_hotspot_share": float(g["gi_star_gwr_class"].eq("Hot spot").mean()),
            "GWR_LCR_GiStar_hotspot_share": float(g["gi_star_gwr_lcr_class"].eq("Hot spot").mean()),
            "median_GRF_local_Aa_effect_nearest_1p2km": float(g["grf_local_effect"].median()),
            "median_GRF_center_distance_m": float(g["grf_center_distance_m"].median()),
            "dominant_soil_27_share": float(g["dominant_soil_type"].eq(27).mean()),
            "environment_sensitive_overlay_share": float(g["environment_sensitive_candidate"].mean()),
            "high_Aa_nonnegative_GWR_share": float(g["high_Aa_nonnegative_GWR"].mean()),
            "outcome_guided_selection": True,
        })
        for column in ["Elevation", "Cur", "TWI", "LST", "NDWI", "NDBSI", "EVI", "NDVI", "Precipitation"]:
            profile_rows.append({"zone": zone, "profile_variable": column,
                                 "mean": float(g[column].mean()), "median": float(g[column].median()),
                                 "q25": float(g[column].quantile(.25)), "q75": float(g[column].quantile(.75)),
                                 "units": "source raster units; metadata not expanded in Phase I"})
        soil_shares = g["dominant_soil_type"].value_counts(normalize=True, dropna=False)
        for soil_code, share in soil_shares.items():
            profile_rows.append({"zone": zone, "profile_variable": f"dominant_SoilType_{soil_code}",
                                 "mean": float(share), "median": np.nan, "q25": np.nan, "q75": np.nan,
                                 "units": "share of 240m units"})
    summary = pd.DataFrame(summary_rows)
    summary.insert(1, "total_analysis_units", int(unit_total))
    summary.insert(2, "total_complete_case_pixels", int(pixel_total))
    summary.insert(3, "Aa_high_threshold_q80", thresholds["high_Aa"])
    summary.insert(4, "GWR_strong_negative_threshold_q20", thresholds["strong_negative_beta"])
    summary.insert(5, "Aa_low_threshold_q50", thresholds["low_Aa"])
    summary.insert(6, "SHI_low_threshold_q20", thresholds["low_SHI"])
    overlay = d["environment_sensitive_candidate"].value_counts(dropna=False)
    overlay_rows = [
        {"overlay": "27潮土·TWI q2 spatial proxy", "units": int(d["soil27_twi_candidate"].sum()),
         "share_of_analysis_units": float(d["soil27_twi_candidate"].mean()),
         "TWI_low_cut": thresholds.get("soil27_TWI_q_low"), "TWI_high_cut": thresholds.get("soil27_TWI_q_high"),
         "spatial_definition": "dominant SoilType=27 at >=80% of complete pixels in 240m tile; tile mean TWI q20-q40 among such tiles"},
        {"overlay": "26冲积土·TWI q1 spatial proxy", "units": int(d["soil26_twi_candidate"].sum()),
         "share_of_analysis_units": float(d["soil26_twi_candidate"].mean()),
         "TWI_low_cut": thresholds.get("soil26_TWI_q_low"), "TWI_high_cut": thresholds.get("soil26_TWI_q_high"),
         "spatial_definition": "dominant SoilType=26 at >=80% of complete pixels in 240m tile; tile mean TWI q0-q20 among such tiles"},
    ]
    overlay_table = pd.DataFrame(overlay_rows)
    overlay_table["outcome_guided_selection"] = True
    return summary, pd.DataFrame(profile_rows), overlay_table


def _recommendations() -> pd.DataFrame:
    law = "https://www.mee.gov.cn/ywgz/fgbz/fl/199106/t19910629_81963.shtml"
    plan = "https://www.mee.gov.cn/zcwj/zyygwj/202301/t20230103_1009406.shtml"
    soil = "https://nyncj.beijing.gov.cn/nyj/zwgk/ztgk/gdjsbhqk/436449599/index.html"
    rows = [
        {"zone": ZONE_ORDER[0], "priority": "优先核验与治理候选",
         "management_candidate": "先到现场核对侵蚀迹象与耕作条件；在坡度、田块格局和灌排条件适用时，试行覆盖/少扰动、等高耕作或坡面径流调控，并布设地块尺度监测。",
         "evidence_boundary": "由高Aa分位与负向GWR分位定义；不是工程设计结论，具体措施需按地形和生产条件论证。",
         "source": law + " ; " + plan},
        {"zone": ZONE_ORDER[1], "priority": "核验缓冲条件",
         "management_candidate": "对照TWI、植被和土类剖面核实可能的环境缓冲；保留地表覆盖和适宜轮作，比较治理前后径流/土壤指标。",
         "evidence_boundary": "局部系数较弱不证明存在因果缓冲；以现场水文与农艺资料验证。",
         "source": law + " ; " + plan},
        {"zone": ZONE_ORDER[2], "priority": "诊断非侵蚀限制",
         "management_candidate": "优先开展地块土壤检测，区分含水、容重、有机质、养分、pH及生物指标限制；据检测结果再制定培肥、秸秆/绿肥利用或配方施肥方案。",
         "evidence_boundary": "低SHI来自预测栅格；图上类别本身不能指认缺素、板结或其他单一机制。",
         "source": soil},
        {"zone": ZONE_ORDER[3], "priority": "定向复核环境候选",
         "management_candidate": "分别核实27潮土—TWI q2与26冲积土—TWI q1候选的土壤水分、地表径流和健康指标；在验证前不将两类合并为统一工程措施。",
         "evidence_boundary": "地图分位是240m汇总块的空间代理，不等于I-3匹配对端点分组或已验证敏感区。",
         "source": law + " ; " + plan},
        {"zone": ZONE_ORDER[4], "priority": "一般监测",
         "management_candidate": "保留常规地块监测；如出现实测侵蚀或健康变化，再按实测值重新诊断优先级。",
         "evidence_boundary": "未进入四类筛选规则不代表无风险。",
         "source": plan},
    ]
    return pd.DataFrame(rows).assign(outcome_guided_selection=True)


def _geojson_exteriors(path: Path) -> list[np.ndarray]:
    geojson = json.loads(path.read_text(encoding="utf-8"))
    features = geojson.get("features", [])
    if not features:
        raise ValueError(f"Administrative locator GeoJSON has no features: {path}")
    geometry = features[0].get("geometry") or {}
    if geometry.get("type") == "Polygon":
        rings = [geometry["coordinates"][0]]
    elif geometry.get("type") == "MultiPolygon":
        rings = [polygon[0] for polygon in geometry["coordinates"] if polygon]
    else:
        raise ValueError(f"Unsupported locator geometry {geometry.get('type')!r}: {path}")
    return [np.asarray(ring, dtype=float) for ring in rings if len(ring) >= 3]


def _locator_layer(ax: plt.Axes, rings: list[np.ndarray], face: str, edge: str) -> None:
    collection = PolyCollection(rings, closed=True, facecolors=face, edgecolors=edge,
                                linewidths=.42, antialiased=True, zorder=1)
    ax.add_collection(collection, autolim=True)
    ax.autoscale_view()


def _complete_case_center(data: pd.DataFrame) -> tuple[float, float]:
    weights = data["n_pixels"].to_numpy(dtype=float)
    center_x = float(np.average(data["x"].to_numpy(dtype=float), weights=weights))
    center_y = float(np.average(data["y"].to_numpy(dtype=float), weights=weights))
    longitude, latitude = warp_transform("EPSG:32649", "EPSG:4326", [center_x], [center_y])
    return float(longitude[0]), float(latitude[0])


def _map(run_dir: Path, data: pd.DataFrame, mask_info: tuple[np.ndarray, np.ndarray, np.ndarray],
         dpi: int, config: dict[str, Any]) -> list[str]:
    mask, xcoords, ycoords = mask_info
    china_path = ROOT / config["admin_boundaries"]["china"]
    henan_path = ROOT / config["admin_boundaries"]["henan"]
    china_rings = _geojson_exteriors(china_path)
    henan_rings = _geojson_exteriors(henan_path)
    center_lon, center_lat = _complete_case_center(data)

    fig = plt.figure(figsize=(16, 10.8), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.55, 3.75], hspace=.16, wspace=.10)
    locator_cn = fig.add_subplot(grid[0, 0])
    locator_hn = fig.add_subplot(grid[0, 1])
    main_axes = [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]
    _locator_layer(locator_cn, china_rings, "#EEF1F4", "#65727D")
    _locator_layer(locator_cn, henan_rings, "#B2182B", "#71101A")
    locator_cn.set_xlim(73, 135.5)
    locator_cn.set_ylim(17, 54.5)
    locator_cn.set_aspect("equal", adjustable="box")
    locator_cn.set_title("中国位置：河南省高亮", fontsize=11)
    _locator_layer(locator_hn, henan_rings, "#EEF1F4", "#65727D")
    locator_hn.scatter([center_lon], [center_lat], marker="*", s=105, c="#B2182B",
                       edgecolors="white", linewidths=.8, zorder=3,
                       label="温县耕地完整案例中心")
    locator_hn.annotate("温县", (center_lon, center_lat), xytext=(6, 5),
                        textcoords="offset points", fontsize=9, color="#71101A")
    locator_hn.set_xlim(110.0, 116.6)
    locator_hn.set_ylim(31.0, 36.8)
    locator_hn.set_aspect("equal", adjustable="box")
    locator_hn.set_title("河南省位置：温县分析中心", fontsize=11)
    locator_hn.legend(loc="lower right", fontsize=8, framealpha=.95)
    for locator in (locator_cn, locator_hn):
        locator.set_xticks([])
        locator.set_yticks([])
        for spine in locator.spines.values():
            spine.set_linewidth(.6)
            spine.set_color("#65727D")
    for ax, col, title, palette in [
        (main_axes[0], "primary_zone", "综合胁迫主分区（240 m汇总单元）", ZONE_COLORS),
        (main_axes[1], "environment_overlay", "环境敏感候选空间代理", ENV_COLORS),
    ]:
        for category, color in palette.items():
            use = data[col].eq(category)
            ax.scatter(data.loc[use, "x"] / 1000, data.loc[use, "y"] / 1000,
                       c=color, s=10, marker="s", linewidths=0, rasterized=True, label=category)
        spatial._add_map_decor(ax, mask, xcoords, ycoords)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Easting (km, EPSG:32649)")
        ax.set_ylabel("Northing (km)")
        ax.set_aspect("equal", adjustable="datalim")
        handles = [Line2D([0], [0], marker="s", linestyle="", color=color,
                          markersize=7, label=category) for category, color in palette.items()]
        ax.legend(handles=handles, loc="lower right", frameon=True, fontsize=8,
                  framealpha=.95, borderpad=.45, labelspacing=.4)
    fig.suptitle("温县耕地：侵蚀—SHI空间胁迫候选分区", fontsize=16)
    paths = []
    outbase = run_dir / "figures" / "stress_zoning_map"
    outbase.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg", "pdf"):
        out = outbase.with_suffix(f".{ext}")
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        paths.append(str(out.relative_to(ROOT)))
    plt.close(fig)
    return paths


def _md_table(frame: pd.DataFrame, columns: list[str], digits: int = 3) -> str:
    view = frame[columns].copy()
    for c in view.select_dtypes(include="number"):
        view[c] = view[c].map(lambda x: "" if pd.isna(x) else f"{x:.{digits}f}")
    headers = [str(c) for c in view.columns]
    rows = [headers, ["---"] * len(headers)]
    rows.extend(view.fillna("").astype(str).itertuples(index=False, name=None))
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def _write_report(run_dir: Path, meta: dict[str, Any], thresholds: dict[str, float],
                   summary: pd.DataFrame, profiles: pd.DataFrame, overlay: pd.DataFrame,
                   recommendations: pd.DataFrame, map_paths: list[str], config: dict[str, Any]) -> Path:
    report_path = run_dir / "reports/PHASE_I_SPATIAL_STRESS_ZONING_REPORT.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    key_profile = profiles[profiles["profile_variable"].isin(["TWI", "NDVI", "NDWI", "LST"])]
    method_rows = pd.DataFrame([
        {"规则": "高Aa", "定义": "240m单元Aa_20yr均值 >= q80", "阈值": thresholds["high_Aa"]},
        {"规则": "强负向SHI响应", "定义": "环境调整GWR标准化Aa系数 < 0 且 <= 全部系数q20", "阈值": thresholds["strong_negative_beta"]},
        {"规则": "低Aa", "定义": "240m单元Aa_20yr均值 <= q50", "阈值": thresholds["low_Aa"]},
        {"规则": "低SHI", "定义": "240m单元SHI预测值 <= q20", "阈值": thresholds["low_SHI"]},
    ])
    lines = [
        "# Phase I-8：空间胁迫候选分区与管理建议", "",
        f"权威运行：`{config['run_id']}`；模式：`EXPLORATORY_ORACLE`；`outcome_guided_selection=true`。", "",
        "## 主要发现", "",
        "分区将环境调整GWR局部SHI系数、同单元Aa与预测SHI水平结合；I5局部随机森林边际效应与I6局部相关、LISA和Gi*仅作为分区剖面诊断。分区是按当前完整案例分布构造的探索性候选，不能解释为因果效应或实测风险图。",
        "",
        _md_table(summary, ["zone", "spatial_units_240m", "share_of_units", "complete_case_pixels", "area_km2_equivalent", "mean_Aa_20yr", "mean_SHI", "median_GWR_beta", "LISA_HL_share", "GWR_GiStar_hotspot_share"]),
        "",
        f"共同完整案例覆盖 {meta['environment_complete_pixels']:,} 个30m像元、{meta['tile_units']:,} 个约240m单元；分区面积按纳入像元数×900m²换算，仅代表该完整案例支持。高Aa阈值为q80={thresholds['high_Aa']:.6g}，强负向GWR阈值为q20={thresholds['strong_negative_beta']:.4f}，低Aa阈值为q50={thresholds['low_Aa']:.6g}，低SHI阈值为q20={thresholds['low_SHI']:.6g}。Aa单位沿用源值，元数据尚未核实。",
        "",
        "## 分区规则与敏感性", "",
        _md_table(method_rows, ["规则", "定义", "阈值"], digits=6), "",
        "- T1：Aa达到q80且环境调整GWR系数位于全体单元q20以下，并为负。",
        "- T2：Aa达到q80且局部GWR系数为负，但未进入T1强负向分位。高Aa而GWR系数非负的单元保留在“其他”，并由统计表单列比例。",
        "- T3：Aa不高于q50且SHI不高于q20。",
        "- T4：27潮土—TWI q2或26冲积土—TWI q1候选的空间代理，且未被T1–T3先行分类。第二幅图将两种代理单独显示，即使它们与前三类重叠。",
        "",
        "环境敏感代理以240m块内主导土类占比至少80%筛选，再按对应主导土类块均值TWI的分位范围定位；这不同于I-3按匹配对高侵蚀端分配的土类内分位组。I-3全配对结果中27:q2是SHI/OM/SW较强负向候选；Phase H的26:q1属于另一种低值筛选总体。两种证据不合并为同一个已验证敏感区。",
        "",
        _md_table(overlay, ["overlay", "units", "share_of_analysis_units", "TWI_low_cut", "TWI_high_cut", "spatial_definition"], digits=4),
        "",
        "## 局部响应与环境特征", "",
        "主系数来自I-4环境调整GWR；GWR-LCR用于局部共线性敏感性对照。多尺度SVC列是I-4自实现backfitting近似，不能称为标准MGWR。GRF通过最近1.2km局部中心连接，距离分布随表保留；它是较粗分辨率的诊断量，不是240m单元上的独立拟合。",
        "",
        _md_table(summary, ["zone", "negative_GWR_share", "negative_GWR_LCR_share", "negative_multiscale_SVC_share", "median_GWR_beta", "median_local_Spearman", "median_GRF_local_Aa_effect_nearest_1p2km", "median_GRF_center_distance_m", "GWR_LCR_GiStar_hotspot_share"]),
        "",
        "下表给出各主分区的环境汇总。I-3的配对分组表和本表的像元/空间单元总体不同，环境值只用于解释候选分区的样本构成。",
        "",
        _md_table(key_profile, ["zone", "profile_variable", "mean", "median", "q25", "q75"]),
        "",
        "![综合空间胁迫候选分区图](../figures/stress_zoning_map.png)", "",
        "图上排为中国—河南—温县定位示意，星号是完整案例像元数加权中心（由EPSG:32649转换至WGS84）；下排左侧为互斥主分区，右侧为两类环境候选代理。主图继续使用EPSG:32649项目坐标和项目耕地掩膜轮廓，行政区几何仅用于定位，不参与统计分区。",
        "",
        "定位边界取自[panyanbin/gcoord行政区GeoJSON](https://github.com/panyanbin/gcoord#中国行政区geojson数据)，其说明将中国区划数据归于阿里DataV并按GeoJSON/WGS84表达；本次使用[中国](https://raw.githubusercontent.com/panyanbin/gcoord/master/map/%E4%B8%AD%E5%9B%BD_100000.json)和[河南省](https://raw.githubusercontent.com/panyanbin/gcoord/master/map/%E6%B2%B3%E5%8D%97%E7%9C%81_410000.json)文件（获取日期2026-10-31）。该第三方边界仅为定位示意，不替代官方界线或项目耕地掩膜。",
        "",
        "## 分区治理建议（待实地验证）", "",
        _md_table(recommendations, ["zone", "priority", "management_candidate", "evidence_boundary"]), "",
        "水土保持措施需按坡度、地块格局、灌排与作物制度逐地块核验；国家《水土保持法》列举了因地制宜的工程、植物及保护性耕作措施，并对坡耕地提出梯田、坡面水系整治、蓄水保土耕作等选择。[《中华人民共和国水土保持法》](https://www.mee.gov.cn/ywgz/fgbz/fl/199106/t19910629_81963.shtml)；[《关于加强新时代水土保持工作的意见》](https://www.mee.gov.cn/zcwj/zyygwj/202301/t20230103_1009406.shtml)。低健康分区应先由土壤实测定位障碍再制定培肥方案；[北京市农业农村局有机质提升技术](https://nyncj.beijing.gov.cn/nyj/zwgk/ztgk/gdjsbhqk/436449599/index.html)仅作为土壤检测、秸秆/绿肥和配方施肥的技术例子，不直接外推其剂量至温县。",
        "",
        "## 解释边界与文件", "",
        "- SHI与八项指标均为随机森林预测栅格，不是独立实测样点；240m块及邻域统计有空间依赖。",
        "- 分区阈值是Oracle探索中的分布分位数，结果导向筛选和既有模型搜索均可能影响表观格局；不代表法定风险标准或独立验证。",
        "- I6 Local Spearman为重叠邻域描述量；LISA置换未保留健康空间结构；Gi*使用正态近似；GWR局部区间为模型内近似。",
        "- I4未运行标准MGWR；使用环境调整GWR/GWR-LCR及非标准多尺度SVC近似。",
        "- 仅使用预计算Aa_20yr；未重算或验证RUSLE乘积、未重跑PCNM、未修改研究总纲。",
        "",
        "图件：", "",
    ]
    for path in map_paths:
        lines.append(f"- `{path}`")
    lines.extend(["", "关键表：", "",
                  "- `tables/stress_zones_240m.parquet` 与 `.csv`：空间单元分区及合并诊断。",
                  "- `tables/stress_zone_summary.csv`：分区统计。",
                  "- `tables/stress_zone_environment_profiles.csv`：环境和主导土类剖面。",
                  "- `tables/environment_sensitive_overlays.csv`：环境候选代理定义和范围。",
                  "- `tables/management_recommendations.csv`：治理建议与证据边界。",
                  "- `tables/zone_rule_thresholds.csv`：可复核分区阈值。",
                  "- `manifests/provenance.json`：来源、配置与软件版本。", ""])
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def run(config_path: Path) -> Path:
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("mode") != "EXPLORATORY_ORACLE" or config.get("outcome_guided_selection") is not True:
        raise ValueError("I-8 must run as EXPLORATORY_ORACLE with outcome_guided_selection=true.")
    run_dir = ROOT / config["output_root"] / config["run_id"]
    if run_dir.exists():
        raise FileExistsError(f"Output run already exists; use a new run ID: {run_dir}")
    (run_dir / "configs").mkdir(parents=True)
    (run_dir / "tables").mkdir()
    (run_dir / "figures").mkdir()
    (run_dir / "manifests").mkdir()
    (run_dir / "configs" / config_path.name).write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")

    units, mask_info, input_meta = _read_units(config)
    merged, source_meta = _merge_source_outputs(units, config)
    classified, thresholds = _classify_zones(merged, config)
    classified["environment_overlay"] = "No candidate overlay"
    classified.loc[classified["soil27_twi_candidate"], "environment_overlay"] = "27潮土·TWI q2 proxy"
    classified.loc[classified["soil26_twi_candidate"], "environment_overlay"] = "26冲积土·TWI q1 proxy"
    summary, profiles, overlay = _zone_tables(classified, thresholds)
    recommendations = _recommendations()

    tables = run_dir / "tables"
    classified.to_parquet(tables / "stress_zones_240m.parquet", index=False)
    classified.to_csv(tables / "stress_zones_240m.csv", index=False)
    summary.to_csv(tables / "stress_zone_summary.csv", index=False)
    profiles.to_csv(tables / "stress_zone_environment_profiles.csv", index=False)
    overlay.to_csv(tables / "environment_sensitive_overlays.csv", index=False)
    recommendations.to_csv(tables / "management_recommendations.csv", index=False)
    pd.DataFrame([{"rule": name, "threshold_or_count": value, "outcome_guided_selection": True}
                  for name, value in thresholds.items()]).to_csv(tables / "zone_rule_thresholds.csv", index=False)
    map_paths = _map(run_dir, classified, mask_info, int(config["map_dpi"]), config)
    admin_output = run_dir / "data/admin_boundaries"
    admin_output.mkdir(parents=True, exist_ok=True)
    for key in ("china", "henan"):
        shutil.copyfile(ROOT / config["admin_boundaries"][key], admin_output / f"{key}.geojson")
    report_path = _write_report(run_dir, input_meta, thresholds, summary, profiles, overlay,
                                recommendations, map_paths, config)
    center_lon, center_lat = _complete_case_center(classified)

    git_status = _git("status", "--porcelain") or ""
    provenance = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "run_id": config["run_id"],
        "phase": "I-8", "mode": "EXPLORATORY_ORACLE", "outcome_guided_selection": True,
        "git_head": _git("rev-parse", "HEAD"), "git_status_porcelain": git_status,
        "config_path": str(config_path.relative_to(ROOT)), "inputs": input_meta,
        "source_runs": {k: str(v) for k, v in config["source_runs"].items()},
        "source_support": source_meta, "zone_thresholds": thresholds,
        "admin_locator": {
            "source_readme": config["admin_boundaries"]["source_readme"],
            "china_source": config["admin_boundaries"]["china_geojson"],
            "henan_source": config["admin_boundaries"]["henan_geojson"],
            "retrieved_date": config["admin_boundaries"]["retrieved_date"],
            "coordinates": "GeoJSON WGS84 longitude/latitude; used for locator panels only",
            "study_area_marker": {"method": "n_pixels-weighted centroid of Phase I-8 complete-case 240m blocks, transformed from EPSG:32649 to EPSG:4326",
                                  "longitude": center_lon, "latitude": center_lat},
            "local_copies": [str((admin_output / f"{key}.geojson").relative_to(ROOT)) for key in ("china", "henan")],
        },
        "method": "8x8 30m pixel block; environment-complete SHI/Aa units; existing I4 coefficients and I5-I6 diagnostics merged without refitting; four thresholded candidate classes",
        "unit_notes": {"map_units": "approximately 240m raster blocks", "model_coef": "standardized local slope; sign retained",
                       "grf_join": source_meta["grf_join"], "area": "n complete-case pixels * 900m2; not full cropland area"},
        "data_boundary": "precomputed Aa_20yr used as read; no RUSLE product reconstruction/validation; PCNM not rerun",
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "scipy": scipy.__version__, "matplotlib": _version("matplotlib"),
                     "pyarrow": _version("pyarrow"), "rasterio": _version("rasterio"), "pyyaml": _version("PyYAML")},
        "report": str(report_path.relative_to(ROOT)),
    }
    (run_dir / "manifests/provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_files = sorted(p for p in run_dir.rglob("*") if p.is_file() and p.name != "output_manifest.json")
    manifest = {"run_id": config["run_id"], "files": [
        {"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size} for p in manifest_files]}
    (run_dir / "manifests/output_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "RUN_STATUS.md").write_text(
        f"# {config['run_id']}\n\nStatus: complete\n\nMode: EXPLORATORY_ORACLE\n\n"
        "outcome_guided_selection: true\n\n"
        f"Report: `{report_path.relative_to(ROOT)}`\n\n"
        "This output is exploratory and uses predicted health raster values.\n", encoding="utf-8")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/phase_i_spatial_zoning.yaml")
    args = parser.parse_args()
    print(run(args.config))


if __name__ == "__main__":
    main()
