"""Inventory and profile the local Wenxian raster delivery without modifying it."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import struct
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/envmatch-matplotlib-cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import yaml


RASTER_SUFFIXES = {".tif", ".tiff"}
SIDECAR_SUFFIXES = (".aux.xml", ".tif.xml", ".tiff.xml", ".ovr", ".tfw", ".tif.vat.dbf", ".tif.vat.cpg")
ENVIRONMENT_CANDIDATES = {
    "Elevation": "环境协变量/环境协变量1/填洼re.tif",
    "Cur": "环境协变量/环境协变量1/曲率re.tif",
    "TWI": "环境协变量/环境协变量1/TWIre.tif",
    "LST": "环境协变量/环境协变量2/LST_AllDay_2020_mean_30m_wx.tif",
    "SoilType": "环境协变量/环境协变量2/Soil_Raster_wx_30m.tif",
    "NDWI": "环境协变量/环境协变量2/NDWI_2020_30m_wx.tif",
    "NDBSI": "环境协变量/环境协变量2/NDBSI_2020_mean_30m.tif",
    "EVI": "环境协变量/环境协变量2/EVI_2020_mean_30m_wx.tif",
}
ENVIRONMENT_FIGURES = {
    "Elevation": ENVIRONMENT_CANDIDATES["Elevation"],
    "Cur": ENVIRONMENT_CANDIDATES["Cur"],
    "TWI": ENVIRONMENT_CANDIDATES["TWI"],
    "LST": ENVIRONMENT_CANDIDATES["LST"],
    "NDWI": ENVIRONMENT_CANDIDATES["NDWI"],
    "NDBSI": ENVIRONMENT_CANDIDATES["NDBSI"],
    "EVI": ENVIRONMENT_CANDIDATES["EVI"],
    "NDVI": "环境协变量/环境协变量2/NDVI_mean_2020_wx_30m.tif",
    "Precipitation": "环境协变量/环境协变量2/2020_降水量mean_wx_30m.tif",
}
ENVIRONMENT_CORRELATION = {
    "Precipitation": "环境协变量/环境协变量2/2020_降水量mean_wx_30m.tif",
    "EVI": ENVIRONMENT_CANDIDATES["EVI"],
    "LST": ENVIRONMENT_CANDIDATES["LST"],
    "NDBSI": ENVIRONMENT_CANDIDATES["NDBSI"],
    "NDVI": "环境协变量/环境协变量2/NDVI_mean_2020_wx_30m.tif",
    "NDWI": ENVIRONMENT_CANDIDATES["NDWI"],
}


def find_duplicate_groups(paths: Iterable[Path]) -> dict[Path, str]:
    """Return IDs for groups of byte-identical files, hashing only size matches."""
    by_size: dict[int, list[Path]] = defaultdict(list)
    for path in paths:
        by_size[path.stat().st_size].append(path)

    duplicates: dict[Path, str] = {}
    group_number = 0
    for same_size in by_size.values():
        if len(same_size) < 2:
            continue
        by_digest: dict[str, list[Path]] = defaultdict(list)
        for path in same_size:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            by_digest[digest].append(path)
        for matches in by_digest.values():
            if len(matches) < 2:
                continue
            group_number += 1
            group_id = f"DUP{group_number:03d}"
            duplicates.update({path: group_id for path in matches})
    return duplicates


def raster_sources(data_root: Path) -> list[Path]:
    """Find GeoTIFFs and ArcGrid directories in the supplied delivery."""
    sources = [
        path
        for path in data_root.rglob("*")
        if path.is_file() and path.suffix.lower() in RASTER_SUFFIXES
    ]
    sources.extend(
        header.parent
        for header in data_root.rglob("hdr.adf")
        if (header.parent / "w001001.adf").is_file()
    )
    return sorted(set(sources), key=lambda path: str(path).casefold())


def _grid_key(dataset: rasterio.io.DatasetReader) -> tuple[object, ...]:
    transform = tuple(round(float(value), 8) for value in tuple(dataset.transform)[:6])
    return (
        dataset.crs.to_string() if dataset.crs else None,
        dataset.width,
        dataset.height,
        transform,
    )


def profile_raster(path: Path) -> dict[str, object]:
    """Read one raster and return grid, completeness, zero, and distribution checks."""
    with rasterio.open(path) as dataset:
        masked = dataset.read(1, masked=True)
        raw = np.asarray(masked.data)
        mask = np.ma.getmaskarray(masked)
        finite = np.isfinite(raw)
        valid = ~mask & finite
        values = np.asarray(raw[valid], dtype=np.float64)
        tags = dataset.tags(1)
        units = dataset.units[0] if dataset.units else None
        total = int(dataset.width * dataset.height)

        result: dict[str, object] = {
            "path": str(path),
            "crs": dataset.crs.to_string() if dataset.crs else None,
            "width": dataset.width,
            "height": dataset.height,
            "resolution_x": float(dataset.res[0]),
            "resolution_y": float(dataset.res[1]),
            "transform": ",".join(f"{value:.9f}" for value in tuple(dataset.transform)[:6]),
            "bounds": ",".join(f"{value:.3f}" for value in dataset.bounds),
            "nodata": dataset.nodata,
            "dtype": dataset.dtypes[0],
            "unit_tag": units or tags.get("UNITTYPE") or tags.get("units"),
            "valid_count": int(valid.sum()),
            "total_count": total,
            "valid_percent": float(valid.sum() / total * 100) if total else math.nan,
            "nodata_count": int(mask.sum()),
            "nonfinite_count": int((~mask & ~finite).sum()),
            "zero_count": int(np.count_nonzero(values == 0)),
            "zero_fraction": float(np.mean(values == 0)) if values.size else math.nan,
            "negative_count": int(np.count_nonzero(values < 0)),
            "valid_min": math.nan,
            "q01": math.nan,
            "q05": math.nan,
            "q25": math.nan,
            "q50": math.nan,
            "q75": math.nan,
            "q95": math.nan,
            "q99": math.nan,
            "valid_mean": math.nan,
            "valid_std": math.nan,
            "valid_max": math.nan,
            "read_status": "readable",
            "read_error": None,
        }
        if values.size:
            quantiles = np.quantile(values, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
            result.update(
                valid_min=float(values.min()),
                q01=float(quantiles[0]),
                q05=float(quantiles[1]),
                q25=float(quantiles[2]),
                q50=float(quantiles[3]),
                q75=float(quantiles[4]),
                q95=float(quantiles[5]),
                q99=float(quantiles[6]),
                valid_mean=float(values.mean()),
                valid_std=float(values.std()),
                valid_max=float(values.max()),
            )
        return result


def _dataset_group(path: Path, data_root: Path) -> tuple[str, str]:
    relative = path.relative_to(data_root)
    for parent in path.parents:
        if (parent / "hdr.adf").is_file() and (parent / "w001001.adf").is_file():
            return str(parent.relative_to(data_root)), "ArcGrid dataset component"
    if path.suffix.lower() in {".shp", ".shx", ".dbf", ".prj", ".cpg"}:
        return str(relative.with_suffix("")), "Shapefile bundle component"
    lowered = path.name.lower()
    for ending in SIDECAR_SUFFIXES:
        if lowered.endswith(ending):
            primary = path.name[: -len(ending)]
            return str(relative.with_name(primary)), "Raster sidecar"
    return str(relative), path.suffix.lower().lstrip(".") or "file"


def _shapefile_evidence(path: Path) -> str:
    raw = path.read_bytes()[:100]
    if len(raw) < 100:
        return "invalid: header shorter than 100 bytes"
    file_code = struct.unpack(">i", raw[:4])[0]
    version = struct.unpack("<i", raw[28:32])[0]
    shape_type = struct.unpack("<i", raw[32:36])[0]
    if file_code != 9994 or version != 1000:
        return f"invalid: file code={file_code}, version={version}"
    required = [path.with_suffix(".shx"), path.with_suffix(".dbf")]
    missing = [item.suffix for item in required if not item.is_file()]
    if not path.with_suffix(".prj").is_file():
        missing.append(".prj (CRS)")
    index_records = None
    if required[0].is_file() and required[0].stat().st_size >= 100:
        index_records = (required[0].stat().st_size - 100) // 8
    dbf_records = None
    if required[1].is_file() and required[1].stat().st_size >= 32:
        dbf_records = struct.unpack("<I", required[1].read_bytes()[4:8])[0]
    return (
        f"header valid; shape_type={shape_type}; .shx records={index_records}; "
        f".dbf records={dbf_records}; missing={','.join(missing) or 'none'}; "
        "full geometry not parsed (no OGR/Fiona/pyshp reader installed)"
    )


def _file_readability(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in RASTER_SUFFIXES:
        try:
            with rasterio.open(path) as dataset:
                return f"rasterio-openable; {dataset.width}x{dataset.height}; {dataset.crs}"
        except (OSError, rasterio.errors.RasterioError) as error:
            return f"unreadable raster: {error}"
    if suffix == ".shp":
        return _shapefile_evidence(path)
    if suffix in {".shx", ".dbf", ".prj", ".cpg"}:
        return "component of shapefile bundle; see matching .shp row"
    for ending in SIDECAR_SUFFIXES:
        if path.name.lower().endswith(ending):
            if path.name.lower().endswith(".xml"):
                try:
                    ET.parse(path)
                    return "valid XML sidecar"
                except ET.ParseError as error:
                    return f"invalid XML sidecar: {error}"
            return "auxiliary raster metadata or overview"
    if suffix == ".xml":
        try:
            ET.parse(path)
            return "valid XML"
        except ET.ParseError as error:
            return f"invalid XML: {error}"
    if suffix == ".csv":
        for encoding in ("utf-8-sig", "gb18030"):
            try:
                with path.open(encoding=encoding, newline="") as stream:
                    row_count = sum(1 for _ in csv.reader(stream))
                return f"CSV-readable ({encoding}); {row_count} rows including header"
            except UnicodeDecodeError:
                continue
        return "CSV unreadable: no supported local text encoding"
    if suffix == ".png":
        return "PNG signature valid" if path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" else "invalid PNG signature"
    if suffix == ".pdf":
        return "PDF signature valid" if path.read_bytes()[:5] == b"%PDF-" else "invalid PDF signature"
    if suffix in {".txt", ".md", ".tfw", ".prj", ".cpg"}:
        try:
            path.read_text(encoding="utf-8-sig")
            return "UTF-8-readable text"
        except UnicodeDecodeError:
            return "text present; UTF-8 decoding failed"
    if suffix == ".adf" or path.name.lower().endswith(".adf"):
        return "component of ArcGrid dataset; rasterio-openable directory"
    return "file present; format-specific parser not applied"


def _logical_inference(relative_path: Path, file_type: str) -> tuple[str, str]:
    relative = relative_path.as_posix()
    lowered = relative.casefold()
    if relative.endswith("侵蚀（及五因子）/实际侵蚀_wx.tif"):
        return "precomputed actual erosion (Aa)", "user-selected"
    if "2020年a" in lowered:
        return "raster named 2020 A; erosion role and lineage unconfirmed", "unconfirmed"
    if "corrected_visual" in lowered:
        return "candidate SHI or visualization surface; identity unconfirmed", "unconfirmed"
    if "/r/" in lowered or "_annual_" in lowered:
        return "rainfall-erosivity source or derivative; not used in this task", "inventory-only"
    if "/k/" in lowered or "k_factor" in lowered:
        return "soil-erodibility factor candidate; not used in this task", "inventory-only"
    if "/ls/" in lowered or "ls_" in lowered:
        return "slope-length factor candidate; not used in this task", "inventory-only"
    if "/c/" in lowered or "c_factor" in lowered:
        return "cover-management factor candidate; not used in this task", "inventory-only"
    if "/p/" in lowered or "p_factor" in lowered:
        return "support-practice factor candidate; not used in this task", "inventory-only"
    if relative.startswith("健康指标/"):
        return "soil-health indicator or supporting file; not used for matching", "mapped for later use"
    if relative.startswith("环境协变量/"):
        return "candidate environmental covariate or supporting file", "provisional"
    if relative.startswith("SHI/"):
        return "study-area/land-use mask candidate or supporting file", "provisional"
    return f"{file_type} or supporting file; semantics inferred from path only", "inventory-only"


def build_inventory(data_root: Path) -> pd.DataFrame:
    files = sorted((item for item in data_root.rglob("*") if item.is_file()), key=str)
    duplicate_groups = find_duplicate_groups(files)
    rows = []
    for path in files:
        group, kind = _dataset_group(path, data_root)
        relative = path.relative_to(data_root)
        logical_inference, confirmation_status = _logical_inference(relative, kind)
        rows.append(
            {
                "relative_path": str(relative),
                "dataset_group": group,
                "file_type": kind,
                "category": relative.parts[0] if relative.parts else "root",
                "logical_inference": logical_inference,
                "confirmation_status": confirmation_status,
                "suffix": path.suffix.lower(),
                "size_bytes": path.stat().st_size,
                "readability_evidence": _file_readability(path),
                "duplicate_group": duplicate_groups.get(path),
            }
        )
    return pd.DataFrame(rows)


def _candidate_mapping(data_root: Path) -> dict[str, object]:
    def absolute(relative: str) -> str:
        return str((data_root / relative).resolve())

    health = {
        "SHI": None,
        "AN": absolute("健康指标/碱解氮.tif"),
        "OM": absolute("健康指标/有机质.tif"),
        "CEC": absolute("健康指标/CEC.tif"),
        "pH": absolute("健康指标/PH.tif"),
        "SW": absolute("健康指标/土壤含水率.tif"),
        "BD": absolute("健康指标/容重.tif"),
        "MBC": absolute("健康指标/MBC.tif"),
        "RESP": absolute("健康指标/呼吸.tif"),
    }
    environment = {
        name: absolute(relative) for name, relative in ENVIRONMENT_CANDIDATES.items()
    }
    return {
        "erosion": {
            "Aa": absolute("侵蚀（及五因子）/实际侵蚀_wx.tif"),
            "Ap": None,
            "R": None,
            "K": None,
            "LS": None,
            "C": None,
            "P": None,
        },
        "health": health,
        "environment": environment,
        "spatial": {
            "template": None,
            "cropland_mask": absolute("SHI/温县耕地.tif"),
            "study_area": absolute("SHI/温县耕地.shp"),
        },
        "pcnm": {"results": None, "sheet_name": None},
    }


def _profile_rasters(data_root: Path) -> pd.DataFrame:
    records = []
    for path in raster_sources(data_root):
        try:
            record = profile_raster(path)
            record["relative_path"] = str(path.relative_to(data_root))
        except (OSError, rasterio.errors.RasterioError, ValueError) as error:
            record = {"relative_path": str(path.relative_to(data_root)), "read_status": "error", "read_error": str(error)}
        records.append(record)
    frame = pd.DataFrame(records)
    group_keys: dict[tuple[object, ...], str] = {}
    grid_groups = []
    for path in raster_sources(data_root):
        try:
            with rasterio.open(path) as dataset:
                key = _grid_key(dataset)
            if key not in group_keys:
                group_keys[key] = f"GRID{len(group_keys) + 1:02d}"
            grid_groups.append(group_keys[key])
        except (OSError, rasterio.errors.RasterioError):
            grid_groups.append(None)
    frame["grid_group"] = grid_groups

    reference = data_root / "环境协变量/环境协变量2/2020_降水量mean_wx_30m.tif"
    if reference.is_file():
        with rasterio.open(reference) as template:
            reference_key = _grid_key(template)
        frame["aligned_to_provisional_environment_grid"] = frame["grid_group"].map(
            {group: key == reference_key for key, group in ((key, group) for key, group in group_keys.items())}
        )
    else:
        frame["aligned_to_provisional_environment_grid"] = None
    return frame


def _write_correlations(data_root: Path, output_dir: Path) -> pd.DataFrame:
    arrays: dict[str, np.ndarray] = {}
    grid_keys = set()
    for name, relative in ENVIRONMENT_CORRELATION.items():
        with rasterio.open(data_root / relative) as dataset:
            grid_keys.add(_grid_key(dataset))
            masked = dataset.read(1, masked=True)
            arrays[name] = np.asarray(masked.filled(np.nan), dtype=np.float64)
    if len(grid_keys) != 1:
        raise ValueError("Environmental correlation inputs do not share an exact grid")
    matrix = np.column_stack([arrays[name].ravel() for name in ENVIRONMENT_CORRELATION])
    common = np.isfinite(matrix).all(axis=1)
    correlation = pd.DataFrame(
        np.corrcoef(matrix[common], rowvar=False),
        index=list(ENVIRONMENT_CORRELATION),
        columns=list(ENVIRONMENT_CORRELATION),
    )
    correlation.to_csv(output_dir / "environment_correlations.csv", index=True)
    correlation.attrs["common_valid_count"] = int(common.sum())
    return correlation


def _write_duplicate_report(inventory: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    duplicates = inventory.loc[inventory["duplicate_group"].notna()].sort_values(
        ["duplicate_group", "relative_path"]
    )
    duplicates.to_csv(output_dir / "duplicate_files.csv", index=False)
    return duplicates


def _write_soil_type_counts(data_root: Path, output_dir: Path) -> pd.DataFrame:
    relative = ENVIRONMENT_CANDIDATES["SoilType"]
    with rasterio.open(data_root / relative) as dataset:
        values = dataset.read(1, masked=True).compressed()
    values = values[np.isfinite(values)]
    unique, counts = np.unique(values, return_counts=True)
    result = pd.DataFrame({"value": unique.astype(int), "count": counts.astype(int)})
    result.to_csv(output_dir / "soil_type_value_counts.csv", index=False)
    return result


def _make_figures(data_root: Path, output_dir: Path, correlation: pd.DataFrame) -> None:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    display_paths = {
        "Aa (actual erosion)": "侵蚀（及五因子）/实际侵蚀_wx.tif",
        **ENVIRONMENT_FIGURES,
    }
    fig, axes = plt.subplots(4, 3, figsize=(13, 13), constrained_layout=True)
    for axis, (label, relative) in zip(axes.flat, display_paths.items()):
        with rasterio.open(data_root / relative) as dataset:
            values = dataset.read(1, masked=True).compressed().astype(np.float64)
        values = values[np.isfinite(values)]
        if values.size:
            low, high = np.quantile(values, [0.01, 0.99])
            shown = values[(values >= low) & (values <= high)]
            axis.hist(shown, bins=50, color="#4C7899", edgecolor="white", linewidth=0.2)
        axis.set_title(label)
        axis.set_ylabel("Pixels")
    for axis in axes.flat[len(display_paths) :]:
        axis.set_visible(False)
    fig.suptitle("Actual erosion and candidate environment raster distributions (1st–99th percentile)")
    fig.savefig(figure_dir / "candidate_distributions.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 7), constrained_layout=True)
    image = axis.imshow(correlation.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
    axis.set_xticks(range(len(correlation.columns)), correlation.columns, rotation=45, ha="right")
    axis.set_yticks(range(len(correlation.index)), correlation.index)
    for row in range(len(correlation.index)):
        for column in range(len(correlation.columns)):
            axis.text(column, row, f"{correlation.iloc[row, column]:.2f}", ha="center", va="center", fontsize=8)
    axis.set_title(f"Aligned 2020 environmental covariates (n={correlation.attrs['common_valid_count']:,})")
    fig.colorbar(image, ax=axis, label="Pearson correlation")
    fig.savefig(figure_dir / "environment_correlations.png", dpi=180)
    plt.close(fig)


def _write_mapping_report(data_root: Path, output_dir: Path, mapping: dict[str, object]) -> None:
    rows = [
        ("Aa", "侵蚀（及五因子）/实际侵蚀_wx.tif", "按用户指示作为既算好的实际侵蚀值；不重建 RUSLE。"),
        ("Ap", "—", "未找到可确认的潜在侵蚀图层。"),
        ("SHI", "SHI/corrected_visual.tif", "数值 0.414–0.777，文件名与来源不足以确认 SHI；配置留空。"),
        ("Elevation", ENVIRONMENT_CANDIDATES["Elevation"], "XML 记录由填洼 DEM 双线性重采样至 30 m；候选。"),
        ("Cur", ENVIRONMENT_CANDIDATES["Cur"], "XML 记录由原曲率栅格双线性重采样至 30 m；候选。"),
        ("TWI", ENVIRONMENT_CANDIDATES["TWI"], "XML 记录由流量和弧度坡度计算后双线性重采样；候选。"),
        ("LST", ENVIRONMENT_CANDIDATES["LST"], "本地报告记载 2020 年、单位摄氏度；源分辨率未记录。"),
        ("SoilType", ENVIRONMENT_CANDIDATES["SoilType"], "栅格有 26、27、42、85 四个类别；频数 CSV 与栅格不符。"),
        ("template", "—", "正式 DEM/分析模板未随交付提供；配置留空。"),
    ]
    table = "\n".join(f"| {a} | `{b}` | {c} |" for a, b, c in rows)
    content = f"""# Data inventory and mapping report

- Data root: `{data_root.resolve()}`
- Raw files are read-only and excluded from Git.
- `config/data_paths.yaml` stores only absolute local paths; unavailable or ambiguous roles remain null.
- `erosion.Aa` points to the user's prepared actual-erosion raster. RUSLE factor multiplication is out of scope by user direction.
- Health rasters are mapped for later use only; they are excluded from environment selection and matching.

| Logical role | Candidate | Evidence and status |
|---|---|---|
{table}

The separately named ArcGrid dataset `侵蚀（及五因子）/2020年A` is readable but differs from the selected Aa raster (534,839 valid cells; 97,250 zeros; maximum 54.935; its own 1137×847 grid). Its role/time provenance is unconfirmed, so it is not substituted for the user-selected `实际侵蚀_wx.tif`.

The mapping is operational metadata, not approval to build a grid database. The official source DEM/template is not present, and the cropland mask does not share the candidate raster grids. SHI and Ap remain unmapped pending source confirmation. Exact duplicate evidence is listed in `duplicate_files.csv`; duplicates are concentrated in georeferencing files/metadata, with one duplicate NDVI output/intermediate raster pair.

Two `.shp` files have valid polygon headers, but both are missing `.shx` and `.dbf`; neither has a `.prj`. Record counts, attributes, geometry validity, and coordinate reference cannot be verified from the delivered bundles. ArcGrid directories are readable by rasterio as raster datasets.
"""
    (output_dir / "data_mapping_report.md").write_text(content, encoding="utf-8")


def _write_quality_report(
    data_root: Path, output_dir: Path, inventory: pd.DataFrame, profiles: pd.DataFrame
) -> None:
    counts = profiles.groupby("grid_group", dropna=False).size().sort_values(ascending=False)
    grid_table = []
    for group, count in counts.items():
        sample = profiles.loc[profiles["grid_group"] == group].iloc[0]
        grid_table.append(
            f"| {group} | {count} | {sample['crs']} | {sample['width']}×{sample['height']} | "
            f"{sample['resolution_x']}×{sample['resolution_y']} | `{sample['transform']}` |"
        )
    duplicate_count = inventory["duplicate_group"].notna().sum()
    readable_rasters = int((profiles["read_status"] == "readable").sum())
    aa = profiles.loc[profiles["relative_path"] == "侵蚀（及五因子）/实际侵蚀_wx.tif"].iloc[0]
    content = f"""# Raster quality report

- Inventory files: {len(inventory):,}; exact duplicate files: {duplicate_count:,} rows in duplicate groups.
- Raster datasets opened: {readable_rasters:,} of {len(profiles):,} (`.tif` plus ArcGrid bundles).
- Provisional comparison grid: `环境协变量/环境协变量2/2020_降水量mean_wx_30m.tif`. Its own QC says it matches an external DEM template that is not included. It is used here only to compare geometry, not declared the approved analysis grid.
- The prepared actual-erosion raster has {int(aa['valid_count']):,} valid pixels / {int(aa['total_count']):,} total ({aa['valid_percent']:.2f}%); zeros {int(aa['zero_count']):,} ({aa['zero_fraction']:.2%}); range {aa['valid_min']:.6g} to {aa['valid_max']:.6g}. Unit metadata is absent.

## Exact grid groups

| Grid | Datasets | CRS | Shape (columns×rows) | Resolution | Affine transform (a,b,c,d,e,f) |
|---|---:|---|---|---|---|
{"\n".join(grid_table)}

Equal nominal 30 m resolution does not mean equal grid. Multiple EPSG:32649 origins and dimensions occur. Geographic EPSG:4326 rasters are also present. No resampling or overlay was performed.

The local health map summary lists 391,754 valid pixels for most indicators and 391,907 for PH/alkali-N, with PH/alkali-N on a different 848×1136 grid. This is evidence about their delivered map geometry, not a matching input. The supplied rasters' own NoData masks are profiled in `raster_statistics.csv`.

The provisional environmental grid contains a 30 m output derived from source rasters. NDVI's preprocessing note says the source was 500 m and bilinear resampling was used; 30 m output spacing does not add 30 m source information. LST's local report records Celsius but no source resolution. Do not treat these outputs as independent fine-scale measurements without source confirmation.

**G2 status: BLOCKED for combined analysis.** The official DEM/template is missing, the cropland-mask candidate occupies another grid, and actual erosion does not align exactly with the candidate environment template. Keep matching and health regression stopped.
"""
    (output_dir / "raster_quality_report.md").write_text(content, encoding="utf-8")


def _write_rusle_scope_report(output_dir: Path, profiles: pd.DataFrame) -> None:
    aa = profiles.loc[profiles["relative_path"] == "侵蚀（及五因子）/实际侵蚀_wx.tif"].iloc[0]
    content = f"""# RUSLE model validation scope

At the user's direction, no RUSLE factor-product reconstruction or R/K/LS/C/P consistency validation was performed. The analysis exposure is the already-computed `实际侵蚀_wx.tif`, used directly as actual erosion (`Aa`).

The selected raster has {int(aa['valid_count']):,} valid pixels, {int(aa['zero_count']):,} zero pixels ({aa['zero_fraction']:.2%} of valid), range {aa['valid_min']:.6g}–{aa['valid_max']:.6g}, CRS {aa['crs']}, grid {aa['width']}×{aa['height']} at {aa['resolution_x']} m. Unit/time metadata are absent, and the grid is not aligned to the provisional environmental grid. These are properties of the supplied Aa raster, not conclusions about the RUSLE formula.
"""
    (output_dir / "rusle_validation_report.md").write_text(content, encoding="utf-8")


def _write_covariate_report(
    output_dir: Path, profiles: pd.DataFrame, correlation: pd.DataFrame, soil_counts: pd.DataFrame
) -> None:
    lookup = profiles.set_index("relative_path")

    def stat(relative: str, column: str) -> object:
        return lookup.loc[relative, column]

    soil_total = int(soil_counts["count"].sum())
    soil_rows = []
    soil_names = {26: "冲积土", 27: "潮土", 42: "脱潮土", 85: "江及河（河流/水体）"}
    for row in soil_counts.itertuples(index=False):
        value = int(row.value)
        soil_rows.append(
            f"| {value} | {soil_names.get(value, 'legend entry not found')} | {int(row.count):,} | "
            f"{int(row.count) / soil_total:.2%} |"
        )
    corr_rows = []
    for left, right in (("EVI", "NDVI"), ("EVI", "NDWI"), ("NDVI", "NDWI"), ("Precipitation", "NDBSI"), ("LST", "NDBSI")):
        corr_rows.append(f"| {left}–{right} | {correlation.loc[left, right]:.3f} |")

    content = f"""# Environmental covariate assessment

## Scope and candidate grid

This assessment uses environmental layers only; SHI and the eight health indicators were not used to choose variables, groups, or matching rules. The 2020 environmental outputs share an exact EPSG:32649 grid (1138×850, 30 m) and 536,477 valid cells (55.46%). That grid is shifted from the terrain candidates and the prepared Aa raster, and the source DEM/template referenced by processing reports is absent.

## Candidate evidence

| Candidate | Delivered evidence | Assessment |
|---|---|---|
| Elevation (`填洼re.tif`) | {stat(ENVIRONMENT_CANDIDATES['Elevation'], 'valid_count'):,} valid; {stat(ENVIRONMENT_CANDIDATES['Elevation'], 'valid_min'):.1f}–{stat(ENVIRONMENT_CANDIDATES['Elevation'], 'valid_max'):.1f}; EPSG:32649, 1137×849, 30 m | Metadata says it is a bilinear 30 m resample of a filled DEM. Unit appears metre by raster lineage but band units are not encoded. Grid does not match the 2020 environmental grid. |
| Cur (`曲率re.tif`) | {stat(ENVIRONMENT_CANDIDATES['Cur'], 'valid_count'):,} valid; {stat(ENVIRONMENT_CANDIDATES['Cur'], 'valid_min'):.3f}–{stat(ENVIRONMENT_CANDIDATES['Cur'], 'valid_max'):.3f}; zero share {stat(ENVIRONMENT_CANDIDATES['Cur'], 'zero_fraction'):.2%} | Metadata says it was bilinearly resampled to 30 m. Unit/convention is undocumented; zero is plausible for curvature and is not treated as missing. Grid differs from the official candidate template. |
| TWI (`TWIre.tif`) | {stat(ENVIRONMENT_CANDIDATES['TWI'], 'valid_count'):,} valid ({stat(ENVIRONMENT_CANDIDATES['TWI'], 'valid_percent'):.2f}%); {stat(ENVIRONMENT_CANDIDATES['TWI'], 'valid_min'):.3f}–{stat(ENVIRONMENT_CANDIDATES['TWI'], 'valid_max'):.3f} | XML records a flow/slope-derived raster and bilinear resampling to 30 m. Formula is `Ln((流量 + 1) × 30 / Tan(弧度坡度))`; algorithm parameters and units need source confirmation. It has fewer valid cells than Elevation/Cur and a different grid group. |
| LST | 2020 mean, report specifies °C; mean {stat(ENVIRONMENT_CANDIDATES['LST'], 'valid_mean'):.3f} °C, SD {stat(ENVIRONMENT_CANDIDATES['LST'], 'valid_std'):.3f} °C | Output is aligned with the 2020 covariate grid. Source resolution and exact temporal aggregation are not recorded in the delivered report. |
| SoilType | Four observed codes; source legend available | Actual frequencies are below. The separate category-frequency CSV reports value 1 with count 0 and does not describe this raster. Code 85 is river/water and should not be treated as a soil stratum; the delivered cropland mask is on a different grid. |
| NDVI | 2020 mean; output mean {stat('环境协变量/环境协变量2/NDVI_mean_2020_wx_30m.tif', 'valid_mean'):.3f} | Preprocessing report identifies a 500 m source bilinearly projected/resampled to 30 m. Exclude from the primary 30 m environmental vector because this output does not carry independent 30 m source detail. The output is byte-identical to `terra_temp/04_ndvi_masked.tif`. |
| EVI, NDWI, NDBSI | 2020 outputs aligned to the same 1138×850 grid | Source resolutions are not documented in the provided QC reports. Hold as optional candidates until their source scales and temporal construction are confirmed. |
| Precipitation | 2020 mean output; mean {stat('环境协变量/环境协变量2/2020_降水量mean_wx_30m.tif', 'valid_mean'):.3f} | Source and interpolation details are incomplete. Because precipitation directly represents an R-related driver and is not part of the charter's current core vector, keep it out of the primary matching distance pending a design amendment. |

## Categorical coverage

| Code | Legend | Pixels | Share |
|---:|---|---:|---:|
{"\n".join(soil_rows)}

## Correlation among aligned 2020 continuous covariates

Pearson correlations use {correlation.attrs['common_valid_count']:,} shared valid cells and are exploratory diagnostics, not health-outcome tuning.

| Pair | r |
|---|---:|
{"\n".join(corr_rows)}

EVI, NDVI, and NDWI show substantial redundancy in the processed outputs. NDVI also has a documented 500 m source; therefore, do not include all vegetation/wetness indices together. If optional indices are retained after source review, select a small set for environmental meaning and report the correlation sensitivity.

## G3 status

**BLOCKED for an approved matching vector.** Candidate meaning and several source limits are assessable, but the official template is absent, terrain and 2020 covariate grids do not align, the soil-frequency export is inconsistent, and several source resolutions/years remain undocumented. The charter's core candidate set remains Elevation, Cur, TWI, LST with SoilType strata; optional indices remain proposals only.
"""
    (output_dir / "covariate_assessment.md").write_text(content, encoding="utf-8")


def _write_feasibility_and_stage_reports(output_dir: Path) -> None:
    (output_dir / "matching_feasibility.md").write_text(
        """# Matching feasibility

**G4 status: NOT RUN — prerequisite gates are blocked.** The official DEM/template is not present; the actual erosion raster, cropland mask, terrain candidates, and 2020 environmental candidates occupy different grids. The SoilType frequency export is inconsistent with the raster. No environmental clusters, k=6/8/10/12 comparisons, erosion-within-group summaries, 300/500/1000 m radius comparisons, or match tuning were run. SHI and the eight health indicators were not used. Do not start health regressions until the source grid and G2/G3 gates are resolved and a matching design is reviewed and locked.
""",
        encoding="utf-8",
    )
    (output_dir / "stage_review.md").write_text(
        """# Stage review

| Gate | Status | Evidence |
|---|---|---|
| G0 governance | PASS (files prepared; local Git commit pending) | Charter, protocols, stage gates, templates, and project guidance exist. Charter body has not been edited since creation. |
| G1 data intake | PASS with unresolved mappings | 312 files inventoried; 77 raster datasets opened; raw path configuration is ignored by Git; duplicate groups and incomplete vector bundles are documented. Aa is mapped per user instruction; SHI, Ap, and official template remain null. |
| G2 raster data quality | BLOCKED | Actual Aa is profiled, but it does not align with the provisional environmental grid or cropland mask. Unit/time metadata are absent, and the official template is missing. No RUSLE factor reconstruction was performed per user direction. |
| G3 environment vector | BLOCKED | Candidate grids, temporal/source-resolution limits, missingness, category codes, and aligned-covariate correlations are documented; source confirmation and an approved shared grid are still missing. |
| G4 matching feasibility | NOT RUN | G2/G3 prerequisites have not passed. |
| G5 health response | NOT RUN | G4 and design lock are prerequisites. |

Raw inputs were not edited or staged. Derived audit reports are under the Git-ignored `outputs/data_audit/`; `config/data_paths.yaml` is also ignored. A local Git commit is needed before the review prompt can be pinned to the audit commit SHA.
""",
        encoding="utf-8",
    )


def run_audit(data_root: Path, project_root: Path, output_dir: Path) -> None:
    data_root = data_root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory = build_inventory(data_root)
    inventory.to_csv(output_dir / "data_inventory.csv", index=False)
    _write_duplicate_report(inventory, output_dir)
    profiles = _profile_rasters(data_root)
    profiles.to_csv(output_dir / "raster_statistics.csv", index=False)
    mapping = _candidate_mapping(data_root)
    mapping_path = project_root / "config" / "data_paths.yaml"
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_path.write_text(
        "# Local-only paths. Environment layers are provisional; do not build a database until gates pass.\n"
        + yaml.safe_dump(mapping, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    correlations = _write_correlations(data_root, output_dir)
    soil_counts = _write_soil_type_counts(data_root, output_dir)
    _make_figures(data_root, output_dir, correlations)
    _write_mapping_report(data_root, output_dir, mapping)
    _write_quality_report(data_root, output_dir, inventory, profiles)
    _write_rusle_scope_report(output_dir, profiles)
    _write_covariate_report(output_dir, profiles, correlations, soil_counts)
    _write_feasibility_and_stage_reports(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("环境匹配原数据"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/data_audit"))
    arguments = parser.parse_args()
    run_audit(arguments.data_root, arguments.project_root, arguments.output_dir)
    print(f"Data audit written to {arguments.output_dir.resolve()}")


if __name__ == "__main__":
    main()
