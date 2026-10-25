"""Command-line entry points for staged data validation and analysis."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio

from soilmatch.balance import assess_balance
from soilmatch.clustering import cluster_environments
from soilmatch.config import load_config, load_data_mapping, resolve_path
from soilmatch.covariates import screen_covariates
from soilmatch.erosion_variability import summarize_erosion
from soilmatch.environment_vectors import build_environment_vectors
from soilmatch.grid_database import build_grid_database
from soilmatch.io import flatten_path_mapping, read_table
from soilmatch.matching import match_cells
from soilmatch.pcnm_comparison import compare_pcnm, read_pcnm_results
from soilmatch.raster_validation import (
    assert_aligned,
    check_rusle_consistency,
    export_quality_report,
    inspect_rasters,
    assert_metric_crs,
)
from soilmatch.regression import fit_group_models, fit_pair_models
from soilmatch.reporting import create_run_directory, write_report
from soilmatch.visualization import (
    plot_environment_groups,
    plot_environment_profile,
    plot_erosion_distribution,
    plot_group_regression,
    plot_indicator_response_heatmap,
    plot_matching_balance,
    plot_pair_differences,
    plot_pcnm_comparison,
    plot_regression_forest,
)


_REQUIRED_INPUTS = (
    "erosion.Aa",
    "erosion.R",
    "erosion.K",
    "erosion.LS",
    "erosion.C",
    "erosion.P",
    "health.SHI",
    "health.AN",
    "health.OM",
    "health.CEC",
    "health.pH",
    "health.SW",
    "health.BD",
    "health.MBC",
    "health.RESP",
    "environment.Elevation",
    "environment.Cur",
    "environment.TWI",
    "environment.LST",
    "environment.SoilType",
    "spatial.template",
    "spatial.cropland_mask",
)
_RUSLE_FACTORS = ("R", "K", "LS", "C", "P")
_OPTIONAL_PATHS = {"erosion.Ap", "pcnm.results", "spatial.study_area"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="soilmatch",
        description="温县耕地土壤侵蚀—土壤健康环境匹配分析工作流",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("validate-data", "检查映射、栅格对齐、NoData 和 RUSLE 结果"),
        ("build-database", "将已对齐栅格分块转换为 cell_id Parquet 数据库"),
        ("screen-covariates", "汇总环境协变量分布及相关性"),
        ("cluster", "按土壤类型分层评估候选环境组"),
        ("check-erosion", "汇总环境组内侵蚀分布和变化"),
        ("match", "在环境组内执行高低侵蚀最近邻匹配和质量诊断"),
        ("analyze", "拟合组内回归和匹配对差值回归"),
        ("compare-pcnm", "对比导入的 PCNM 与环境匹配结果"),
        ("report", "为已有运行目录生成简要 Markdown 报告"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--config", default="config/default.yaml", help="YAML 配置文件")
        command.add_argument("--overwrite", action="store_true", help="显式覆盖指定的中间数据库")
    commands.choices["cluster"].add_argument("--clusters", type=int, help="选择一个候选组数；省略时评估全部候选值")
    commands.choices["match"].add_argument("--group-column", help="已审核的环境组列名")
    commands.choices["match"].add_argument("--radius-m", type=float, help="地理距离上限（米）")
    commands.choices["match"].add_argument("--environment-distance", type=float, help="环境向量距离上限")
    commands.choices["match"].add_argument("--erosion-gap", type=float, help="最小 Aa 差值")
    commands.choices["match"].add_argument("--allow-reference-reuse", action="store_true")
    commands.choices["analyze"].add_argument("--pairs", help="匹配阶段输出的 pairs.parquet")
    commands.choices["compare-pcnm"].add_argument("--matched-results", help="环境匹配回归结果表")
    commands.choices["report"].add_argument("--run-directory", help="已有 outputs/run_<id> 目录")
    return parser


def _log_path(config: dict[str, Any], config_path: str | Path) -> Path:
    configured = config.get("project", {}).get("log_root", "logs")
    path = resolve_path(configured, config_path)
    return (path or Path("logs")).resolve() / "soilmatch.log"


def _log(message: str, config: dict[str, Any], config_path: str | Path) -> None:
    path = _log_path(config, config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(f"{datetime.now(UTC).isoformat()} {message}\n")


def _resolved_paths(mapping: dict[str, Any], config_path: str | Path) -> dict[str, Path]:
    return {
        logical_name: resolve_path(path, config_path)
        for logical_name, path in flatten_path_mapping(mapping)
        if resolve_path(path, config_path) is not None
    }


def _mapping_path(mapping: dict[str, Any], name: str) -> str | None:
    value: object = mapping
    for part in name.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return str(value) if isinstance(value, (str, Path)) else None


def _collect_missing(
    mapping: dict[str, Any], config: dict[str, Any], config_path: str | Path
) -> tuple[dict[str, Path], list[str]]:
    paths = _resolved_paths(mapping, config_path)
    data_config = config.get("data", {})
    for logical_name, config_key in (
        ("spatial.template", "template_raster"),
        ("spatial.cropland_mask", "cropland_mask"),
    ):
        configured = data_config.get(config_key)
        if configured:
            resolved = resolve_path(configured, config_path)
            if resolved is not None:
                paths[logical_name] = resolved

    missing = []
    for logical_name in _REQUIRED_INPUTS:
        path = paths.get(logical_name)
        if path is None:
            missing.append(f"{logical_name}: not mapped")
        elif not path.is_file():
            missing.append(f"{logical_name}: file not found ({path})")
    for logical_name, path in paths.items():
        if logical_name in _OPTIONAL_PATHS:
            continue
        if not path.is_file() and not any(line.startswith(f"{logical_name}:") for line in missing):
            missing.append(f"{logical_name}: file not found ({path})")
    return paths, missing


def _input_metadata(mapping: dict[str, Any]) -> dict[str, object]:
    return {name: value for name, value in flatten_path_mapping(mapping)}


def _new_run(config: dict[str, Any], config_path: str | Path, inputs: dict[str, object]):
    output_root = resolve_path(config.get("project", {}).get("output_root", "outputs"), config_path)
    configured_id = config.get("project", {}).get("run_id")
    run_id = None
    if configured_id:
        run_id = f"{configured_id}_{datetime.now(UTC).strftime('%Y%m%dT%H%M%S_%fZ')}"
    return create_run_directory(output_root or Path("outputs"), run_id=run_id, config=config, inputs=inputs)


def _print_missing(missing: list[str]) -> int:
    print("Missing required inputs:", file=sys.stderr)
    for item in missing:
        print(f"- {item}", file=sys.stderr)
    return 2


def _run_validate(args, config: dict[str, Any], mapping: dict[str, Any], config_path: str | Path) -> int:
    paths, missing = _collect_missing(mapping, config, config_path)
    if missing:
        return _print_missing(missing)
    template_path = paths["spatial.template"]
    raster_paths = {
        name: path
        for name, path in paths.items()
        if path.suffix.lower() in {".tif", ".tiff"}
    }
    metadata_path = _data_path(config, config_path, "metadata_file", "data/metadata.csv")
    metadata: dict[str, dict[str, object]] = {}
    metadata_warning = None
    if metadata_path.is_file():
        metadata_frame = pd.read_csv(metadata_path, encoding="utf-8-sig")
        if "logical_name" in metadata_frame:
            metadata = {
                str(row["logical_name"]): row.drop(labels="logical_name").to_dict()
                for _, row in metadata_frame.iterrows()
            }
    else:
        metadata_warning = (
            f"Metadata registry not found: {metadata_path}. "
            "Copy data/metadata.example.csv to data/metadata.csv and complete it."
        )
        print(f"Warning: {metadata_warning}", file=sys.stderr)
    report = inspect_rasters(raster_paths, metadata)
    issues = []
    with rasterio.open(template_path) as template:
        try:
            assert_metric_crs(template)
        except ValueError as error:
            issues.append(f"spatial.template: {error}")
        for name, path in raster_paths.items():
            try:
                with rasterio.open(path) as raster:
                    assert_aligned(template, raster)
            except (OSError, ValueError, rasterio.errors.RasterioError) as error:
                issues.append(f"{name}: {error}")
    factors = {name: paths[f"erosion.{name}"] for name in _RUSLE_FACTORS}
    if all(path.is_file() for path in factors.values()) and paths["erosion.Aa"].is_file():
        try:
            consistency = _check_rusle_rasters(paths["erosion.Aa"], factors, "actual")
        except (OSError, ValueError, rasterio.errors.RasterioError) as error:
            consistency = {"status": "blocked", "reason": str(error)}
            issues.append(f"RUSLE reconstruction skipped: {error}")
        report.attrs["rusle_consistency"] = consistency
        if consistency.get("compared_count") == 0:
            issues.append("RUSLE actual scenario has no finite overlapping values")
    run_dir = _new_run(config, config_path, _input_metadata(mapping))
    export_quality_report(report, run_dir / "tables" / "raster_quality.csv")
    validation_details = (
        "\n".join(f"- {issue}" for issue in issues) if issues else "未发现栅格网格或 RUSLE 一致性问题。"
    )
    sections = {
        "输入核查": (
            f"已检查 {len(raster_paths)} 个 GeoTIFF。\n\n{validation_details}"
        ),
        "RUSLE 一致性": str(report.attrs.get("rusle_consistency", "未执行")),
        "元数据登记": metadata_warning or f"已读取 {len(metadata)} 条记录：{metadata_path}",
        "运行目录": str(run_dir),
    }
    write_report(run_dir, "数据质量检查", sections)
    if issues:
        print("Raster validation issues:", file=sys.stderr)
        for issue in issues:
            print(f"- {issue}", file=sys.stderr)
        print(f"Diagnostic report saved: {run_dir}", file=sys.stderr)
        _log(f"validate-data failed: {len(issues)} grid/RUSLE issues; report: {run_dir}", config, config_path)
        return 2

    print(f"Validation complete: {run_dir}")
    _log(f"validate-data completed: {run_dir}", config, config_path)
    return 0


def _check_rusle_rasters(observed_path: Path, factors: dict[str, Path], scenario: str) -> dict[str, object]:
    with ExitStack() as stack:
        observed = stack.enter_context(rasterio.open(observed_path))
        datasets = {name: stack.enter_context(rasterio.open(path)) for name, path in factors.items()}
        for dataset in datasets.values():
            assert_aligned(observed, dataset)
        compared = 0
        within = 0
        absolute_error_sum = 0.0
        max_absolute_error = 0.0
        for _, window in observed.block_windows(1):
            obs = observed.read(1, window=window, masked=True)
            factor_data = {name: ds.read(1, window=window, masked=True) for name, ds in datasets.items()}
            arrays = {name: np.asarray(value.data, dtype=float) for name, value in factor_data.items()}
            obs_values = np.asarray(obs.data, dtype=float)
            valid = ~np.ma.getmaskarray(obs) & np.isfinite(obs_values)
            for name, data in arrays.items():
                valid &= ~np.ma.getmaskarray(factor_data[name]) & np.isfinite(data)
            if not valid.any():
                continue
            result = check_rusle_consistency(
                obs_values[valid],
                {name: values[valid] for name, values in arrays.items()},
                scenario=scenario,
            )
            compared += int(result["compared_count"])
            within += int(result["within_tolerance_count"])
            absolute_error_sum += float(result["mean_absolute_error"]) * int(result["compared_count"])
            max_absolute_error = max(max_absolute_error, float(result["max_absolute_error"]))
        return {
            "scenario": scenario,
            "compared_count": compared,
            "within_tolerance_count": within,
            "within_tolerance_fraction": within / compared if compared else np.nan,
            "mean_absolute_error": absolute_error_sum / compared if compared else np.nan,
            "max_absolute_error": max_absolute_error if compared else np.nan,
        }


def _data_path(config: dict[str, Any], config_path: str | Path, key: str, default: str) -> Path:
    configured = config.get("data", {}).get(key, default)
    return resolve_path(configured, config_path) or Path(default)


def _read_database(config: dict[str, Any], config_path: str | Path, *, clustered: bool = False) -> pd.DataFrame:
    key = "clustered_database_path" if clustered else "database_path"
    default = "data/processed/grid_clustered.parquet" if clustered else "data/processed/grid.parquet"
    path = _data_path(config, config_path, key, default)
    if not path.is_file():
        raise FileNotFoundError(f"Required processed grid database is missing: {path}")
    return pd.read_parquet(path)


def _run_build_database(args, config: dict[str, Any], mapping: dict[str, Any], config_path: str | Path) -> int:
    paths, missing = _collect_missing(mapping, config, config_path)
    if missing:
        return _print_missing(missing)
    raster_layers = {
        name.split(".", 1)[1]: path
        for name, path in paths.items()
        if name.startswith(("erosion.", "health.", "environment."))
        and path.suffix.lower() in {".tif", ".tiff"}
    }
    mask = paths["spatial.cropland_mask"]
    output = _data_path(config, config_path, "database_path", "data/processed/grid.parquet")
    build_grid_database(raster_layers, output, mask=mask, overwrite=args.overwrite)
    run_dir = _new_run(config, config_path, _input_metadata(mapping))
    (run_dir / "diagnostics" / "database.json").write_text(
        json.dumps({"path": str(output), "raster_layers": list(raster_layers)}, indent=2),
        encoding="utf-8",
    )
    write_report(run_dir, "栅格数据库构建", {"数据库": str(output), "栅格变量数": str(len(raster_layers))})
    print(f"Grid database created: {output}")
    _log(f"build-database completed: {output}", config, config_path)
    return 0


def _run_screen_covariates(args, config: dict[str, Any], mapping: dict[str, Any], config_path: str | Path) -> int:
    frame = _read_database(config, config_path)
    matching = config.get("matching", {})
    columns = (
        list(matching.get("continuous_environment", []))
        + list(matching.get("optional_environment", []))
        + list(matching.get("categorical_strata", ["SoilType"]))
    )
    summary = screen_covariates(frame, columns)
    run_dir = _new_run(config, config_path, {"database": str(_data_path(config, config_path, "database_path", "data/processed/grid.parquet"))})
    summary.to_csv(run_dir / "tables" / "environment_screen.csv", index=False)
    numeric_candidates = summary.loc[summary["status"] == "candidate", "logical_name"].tolist()
    if numeric_candidates:
        frame[numeric_candidates].corr().to_csv(run_dir / "tables" / "environment_correlation.csv")
    write_report(run_dir, "环境协变量筛查", {"候选变量": ", ".join(columns), "有效栅格": str(len(frame))})
    print(f"Covariate summaries saved: {run_dir}")
    return 0


def _run_cluster(args, config: dict[str, Any], mapping: dict[str, Any], config_path: str | Path) -> int:
    frame = _read_database(config, config_path)
    matching = config.get("matching", {})
    continuous = list(matching.get("continuous_environment", []))
    categorical = list(matching.get("categorical_strata", ["SoilType"]))
    categorical = categorical[:1]
    if not categorical:
        raise ValueError("At least one categorical soil stratum is required for clustering")
    matrix, metadata = build_environment_vectors(
        frame,
        continuous=continuous,
        categorical=categorical,
        weights=matching.get("variable_weights"),
    )
    candidates = [args.clusters] if args.clusters else list(matching.get("candidate_clusters", [6, 8, 10, 12]))
    if any(value < 1 for value in candidates):
        raise ValueError("Candidate cluster counts must be positive")
    centers_all = []
    for count in candidates:
        labels, centers = cluster_environments(
            matrix,
            metadata["strata"][categorical[0]],
            n_clusters=count,
            random_state=int(config.get("project", {}).get("random_seed", 0)),
        )
        column = "environment_group" if args.clusters else f"environment_group_k{count}"
        frame[column] = labels
        centers["candidate_k"] = count
        centers["group_column"] = column
        centers_all.append(centers)
    output = _data_path(config, config_path, "clustered_database_path", "data/processed/grid_clustered.parquet")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Clustered database exists; use --overwrite to replace it: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    center_table = pd.concat(centers_all, ignore_index=True)
    run_dir = _new_run(config, config_path, {"database": str(_data_path(config, config_path, "database_path", "data/processed/grid.parquet"))})
    center_table.to_csv(run_dir / "tables" / "environment_centers.csv", index=False)
    if "x" in frame and "y" in frame:
        for count in candidates:
            group_column = "environment_group" if args.clusters else f"environment_group_k{count}"
            plot_environment_groups(
                frame,
                "x",
                "y",
                group_column,
                run_dir / "maps" / f"{group_column}.png",
            )
    plot_environment_profile(center_table, run_dir / "figures" / "environment_profiles.png")
    write_report(run_dir, "环境分组", {"候选组数": str(candidates), "栅格数据库": str(output)})
    print(f"Environment groups saved: {output}")
    return 0


def _run_check_erosion(args, config: dict[str, Any], mapping: dict[str, Any], config_path: str | Path) -> int:
    frame = _read_database(config, config_path, clustered=True)
    group_columns = [column for column in frame if column == "environment_group" or column.startswith("environment_group_k")]
    if not group_columns:
        raise ValueError("No environment-group columns found; run the cluster command first")
    summaries = []
    for group_column in group_columns:
        summary = summarize_erosion(frame, group_column, config.get("matching", {}).get("primary_erosion", "Aa"))
        summary["group_column"] = group_column
        summaries.append(summary)
    result = pd.concat(summaries, ignore_index=True)
    run_dir = _new_run(config, config_path, {"database": str(_data_path(config, config_path, "clustered_database_path", "data/processed/grid_clustered.parquet"))})
    result.to_csv(run_dir / "tables" / "erosion_by_group.csv", index=False)
    if len(group_columns) == 1:
        plot_erosion_distribution(frame, group_columns[0], "Aa", run_dir / "figures" / "erosion_by_group.png")
    write_report(run_dir, "环境组内侵蚀变异", {"环境组列": ", ".join(group_columns), "汇总表": "tables/erosion_by_group.csv"})
    print(f"Erosion summaries saved: {run_dir}")
    return 0


def _run_match(args, config: dict[str, Any], mapping: dict[str, Any], config_path: str | Path) -> int:
    frame = _read_database(config, config_path, clustered=True)
    matching = config.get("matching", {})
    group_col = args.group_column or matching.get("environment_group_column")
    if not group_col or group_col not in frame:
        raise ValueError("Choose an audited environment-group column with --group-column or matching.environment_group_column")
    continuous = list(matching.get("continuous_environment", []))
    categorical = list(matching.get("categorical_strata", ["SoilType"]))
    if not categorical:
        raise ValueError("matching.categorical_strata must identify a soil-type field")
    radius = args.radius_m or matching.get("geographic_radius_m")
    environment_limit = args.environment_distance or matching.get("max_environment_distance")
    erosion_gap = args.erosion_gap or matching.get("min_erosion_gap")
    missing_settings = [
        name
        for name, value in (
            ("geographic_radius_m", radius),
            ("max_environment_distance", environment_limit),
            ("min_erosion_gap", erosion_gap),
        )
        if value is None
    ]
    if missing_settings:
        raise ValueError(
            "Set evidence-based matching settings before running: " + ", ".join(missing_settings)
        )
    matrix, _ = build_environment_vectors(frame, continuous=continuous)
    env_columns = [f"__match_env_{index}" for index in range(matrix.shape[1])]
    frame.loc[:, env_columns] = matrix
    pairs, diagnostics = match_cells(
        frame,
        environment_columns=env_columns,
        group_col=group_col,
        erosion_col=matching.get("primary_erosion", "Aa"),
        soil_type_col=categorical[0],
        x_col="x",
        y_col="y",
        distance_limit=float(radius),
        environment_distance_limit=float(environment_limit),
        erosion_gap=float(erosion_gap),
        allow_reuse=args.allow_reference_reuse or bool(matching.get("allow_reference_reuse", False)),
        per_variable_calipers=matching.get("per_variable_calipers", {}),
        environment_weights=[matching.get("variable_weights", {}).get(name, 1.0) for name in continuous],
    )
    balance = assess_balance(frame, pairs, env_columns, erosion_col=matching.get("primary_erosion", "Aa"))
    run_dir = _new_run(config, config_path, {"clustered_database": str(_data_path(config, config_path, "clustered_database_path", "data/processed/grid_clustered.parquet"))})
    pairs.to_parquet(run_dir / "matched_pairs" / "pairs.parquet", index=False)
    balance.to_csv(run_dir / "tables" / "matching_balance.csv", index=False)
    (run_dir / "diagnostics" / "matching.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not balance.empty:
        plot_matching_balance(balance, run_dir / "figures" / "matching_balance.png")
    print(f"Matched pairs: {len(pairs)}; run directory: {run_dir}")
    return 0


def _run_analyze(args, config: dict[str, Any], mapping: dict[str, Any], config_path: str | Path) -> int:
    frame = _read_database(config, config_path, clustered=True)
    pairs_path = args.pairs or config.get("data", {}).get("matched_pairs_path")
    resolved_pairs = resolve_path(pairs_path, config_path)
    if resolved_pairs is None or not resolved_pairs.is_file():
        raise FileNotFoundError("Provide the match command's pairs.parquet with --pairs or data.matched_pairs_path")
    pairs = pd.read_parquet(resolved_pairs)
    matching = config.get("matching", {})
    group_col = matching.get("environment_group_column") or "environment_group"
    if group_col not in frame:
        raise ValueError(f"Selected environment group column is missing: {group_col}")
    outcomes = list(config.get("analysis", {}).get("health_outcomes", ["SHI", "AN", "OM", "CEC", "pH", "SW", "BD", "MBC", "RESP"]))
    outcomes = [column for column in outcomes if column in frame]
    if not outcomes:
        raise ValueError("No configured health outcomes are available in the grid database")
    group_results = fit_group_models(
        frame,
        outcomes,
        matching.get("primary_erosion", "Aa"),
        group_col,
    )
    indexed = frame.set_index("cell_id")
    for outcome in outcomes:
        pairs[f"high_{outcome}"] = indexed.loc[pairs["high_erosion_cell_id"], outcome].to_numpy()
        pairs[f"low_{outcome}"] = indexed.loc[pairs["low_erosion_cell_id"], outcome].to_numpy()
    pair_results = fit_pair_models(pairs, outcomes)
    run_dir = _new_run(config, config_path, {"matched_pairs": str(resolved_pairs)})
    group_results.to_csv(run_dir / "tables" / "group_regression.csv", index=False)
    pair_results.to_csv(run_dir / "tables" / "pair_regression.csv", index=False)
    for outcome in outcomes:
        if (group_results["outcome"] == outcome).any():
            plot_group_regression(
                frame,
                group_results,
                outcome,
                matching.get("primary_erosion", "Aa"),
                group_col,
                run_dir / "figures" / f"group_{outcome}.png",
            )
        if len(pairs) and f"high_{outcome}" in pairs:
            plot_pair_differences(pairs, outcome, run_dir / "figures" / f"pairs_{outcome}.png")
    successful = group_results.loc[group_results["status"] == "ok"]
    if not successful.empty:
        plot_regression_forest(successful, run_dir / "figures" / "group_effects.png")
        plot_indicator_response_heatmap(successful, run_dir / "figures" / "health_response.png")
    write_report(
        run_dir,
        "侵蚀—健康响应分析",
        {"指标": ", ".join(outcomes), "组内模型": str(len(group_results)), "配对模型": str(len(pair_results))},
    )
    print(f"Regression results saved: {run_dir}")
    return 0


def _run_compare_pcnm(args, config: dict[str, Any], mapping: dict[str, Any], config_path: str | Path) -> int:
    pcnm_path_value = _mapping_path(mapping, "pcnm.results")
    pcnm_path = resolve_path(pcnm_path_value, config_path)
    matched_value = args.matched_results or config.get("data", {}).get("analysis_results_path")
    matched_path = resolve_path(matched_value, config_path)
    if pcnm_path is None or not pcnm_path.is_file():
        raise FileNotFoundError("PCNM results are unavailable; set pcnm.results in the local data mapping")
    if matched_path is None or not matched_path.is_file():
        raise FileNotFoundError("Provide an environmental regression table with --matched-results")
    pcnm = read_pcnm_results(pcnm_path, sheet_name=mapping.get("pcnm", {}).get("sheet_name"))
    matched = read_table(matched_path)
    comparison = compare_pcnm(pcnm, matched, config.get("analysis", {}).get("pcnm_column_mapping"))
    run_dir = _new_run(config, config_path, {"pcnm": str(pcnm_path), "matched_results": str(matched_path)})
    comparison.to_csv(run_dir / "tables" / "pcnm_comparison.csv", index=False)
    if comparison["direction_consistent"].notna().any():
        plot_pcnm_comparison(comparison, run_dir / "figures" / "pcnm_comparison.png")
    write_report(run_dir, "PCNM 与环境匹配比较", {"PCNM 输入": str(pcnm_path), "环境分析输入": str(matched_path)})
    print(f"PCNM comparison saved: {run_dir}")
    return 0


def _latest_run(output_root: Path) -> Path:
    candidates = sorted(
        (path for path in output_root.glob("run_*") if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"No run directories found under {output_root}")
    return candidates[-1]


def _run_report(args, config: dict[str, Any], config_path: str | Path) -> int:
    output_root = resolve_path(config.get("project", {}).get("output_root", "outputs"), config_path)
    run_dir = (
        resolve_path(args.run_directory, config_path)
        if args.run_directory
        else _latest_run(output_root or Path("outputs"))
    )
    if run_dir is None:
        raise ValueError("Could not resolve run directory")
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    sections = {"运行目录": str(run_dir), "运行元数据": "metadata.json 记录输入清单与软件版本。"}
    table_files = sorted((run_dir / "tables").glob("*.csv"))
    if table_files:
        sections["结果表"] = "\n".join(f"- `{path.relative_to(run_dir)}`" for path in table_files)
    report_path = write_report(run_dir, "环境匹配分析运行报告", sections)
    print(f"Report written: {report_path}")
    return 0


def _dispatch(args, config: dict[str, Any], config_path: str | Path) -> int:
    if args.command == "report":
        return _run_report(args, config, config_path)
    mapping, _ = load_data_mapping(config, config_path)
    handlers = {
        "validate-data": _run_validate,
        "build-database": _run_build_database,
        "screen-covariates": _run_screen_covariates,
        "cluster": _run_cluster,
        "check-erosion": _run_check_erosion,
        "match": _run_match,
        "analyze": _run_analyze,
        "compare-pcnm": _run_compare_pcnm,
    }
    return handlers[args.command](args, config, mapping, config_path)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
    except (FileNotFoundError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2
    try:
        _log(f"started command={args.command}", config, args.config)
        status = _dispatch(args, config, args.config)
        _log(f"finished command={args.command} status={status}", config, args.config)
        return status
    except (FileNotFoundError, FileExistsError, ValueError, KeyError, OSError, pd.errors.ParserError) as error:
        print(f"Error: {error}", file=sys.stderr)
        _log(f"failed command={args.command}: {error}", config, args.config)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
