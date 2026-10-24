from __future__ import annotations

import importlib
import json
import sys

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin


def _api(module_name: str, name: str):
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        pytest.fail(f"{module_name} has not been implemented")
    function = getattr(module, name, None)
    assert callable(function), f"{module_name}.{name} has not been implemented"
    return function


def _write_raster(path, values, *, dtype="float32", nodata=-9999, transform=None, crs="EPSG:32649"):
    array = np.asarray(values, dtype=dtype)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=array.dtype,
        crs=crs,
        transform=transform or from_origin(400000, 3900000, 30, 30),
        nodata=nodata,
    ) as destination:
        destination.write(array, 1)


def test_pipeline_runs_from_synthetic_rasters_to_regression_and_figure(tmp_path):
    inspect_rasters = _api("soilmatch.raster_validation", "inspect_rasters")
    build_grid_database = _api("soilmatch.grid_database", "build_grid_database")
    build_environment_vectors = _api("soilmatch.environment_vectors", "build_environment_vectors")
    cluster_environments = _api("soilmatch.clustering", "cluster_environments")
    summarize_erosion = _api("soilmatch.erosion_variability", "summarize_erosion")
    match_cells = _api("soilmatch.matching", "match_cells")
    fit_group_models = _api("soilmatch.regression", "fit_group_models")
    fit_pair_models = _api("soilmatch.regression", "fit_pair_models")
    plot_group_regression = _api("soilmatch.visualization", "plot_group_regression")

    erosion = np.arange(16, dtype=float).reshape(4, 4)
    health = 0.25 + 0.3 * np.log1p(erosion)
    elevation = np.tile(np.arange(4, dtype=float), (4, 1))
    soil = np.ones((4, 4), dtype=np.uint8)
    mask = np.ones((4, 4), dtype=np.uint8)
    rasters = {
        "Aa": tmp_path / "Aa.tif",
        "SHI": tmp_path / "SHI.tif",
        "Elevation": tmp_path / "Elevation.tif",
        "SoilType": tmp_path / "SoilType.tif",
    }
    for name, values in (("Aa", erosion), ("SHI", health), ("Elevation", elevation)):
        _write_raster(rasters[name], values)
    _write_raster(rasters["SoilType"], soil, dtype="uint8", nodata=255)
    mask_path = tmp_path / "mask.tif"
    _write_raster(mask_path, mask, dtype="uint8", nodata=0)
    report = inspect_rasters(rasters, {})
    assert report["exists"].all()
    database = build_grid_database(rasters, tmp_path / "grid.parquet", mask=mask_path, batch_size=2)
    frame = pd.read_parquet(database)
    vectors, _ = build_environment_vectors(
        frame, continuous=["Elevation"], categorical=["SoilType"]
    )
    frame["environment_vector_0"] = vectors[:, 0]
    frame["environment_group"], centers = cluster_environments(
        vectors, frame["SoilType"], n_clusters=1, random_state=4
    )
    summary = summarize_erosion(frame, "environment_group", "Aa")
    pairs, diagnostics = match_cells(
        frame,
        environment_columns=["environment_vector_0"],
        group_col="environment_group",
        erosion_col="Aa",
        soil_type_col="SoilType",
        x_col="x",
        y_col="y",
        distance_limit=1000,
        environment_distance_limit=0.1,
        erosion_gap=3,
    )
    group_results = fit_group_models(frame, ["SHI"], "Aa", "environment_group")
    indexed = frame.set_index("cell_id")
    pairs["high_SHI"] = indexed.loc[pairs["high_erosion_cell_id"], "SHI"].to_numpy()
    pairs["low_SHI"] = indexed.loc[pairs["low_erosion_cell_id"], "SHI"].to_numpy()
    pair_results = fit_pair_models(pairs, ["SHI"])
    figure = plot_group_regression(
        frame, group_results, "SHI", "Aa", "environment_group", tmp_path / "group-regression.png"
    )

    assert len(summary) == 1
    assert summary.iloc[0]["valid_count"] == 16
    assert diagnostics["matched_pairs"] >= 4
    assert group_results.iloc[0]["coefficient"] == pytest.approx(0.3, abs=1e-6)
    assert pair_results.iloc[0]["coefficient"] == pytest.approx(0.3, abs=1e-6)
    assert len(centers) == 1
    assert figure.is_file()


def test_cli_explains_missing_data_mapping_and_help(tmp_path, monkeypatch, capsys):
    main = _api("soilmatch.cli", "main")
    import yaml

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_path = config_dir / "default.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "project": {"output_root": "outputs"},
                "data": {"mapping_file": "config/data_paths.yaml"},
            }
        )
    )
    monkeypatch.chdir(tmp_path)

    status = main(["validate-data", "--config", str(config_path)])
    captured = capsys.readouterr()

    assert status == 2
    assert "data_paths.yaml" in captured.err
    assert "data_paths.example.yaml" in captured.err
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0


def test_failed_validation_still_saves_quality_report(tmp_path, capsys):
    run_validate = _api("soilmatch.cli", "_run_validate")
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_path = config_dir / "default.yaml"
    config_path.write_text("{}", encoding="utf-8")
    raster_path = tmp_path / "aligned.tif"
    misaligned_path = tmp_path / "misaligned.tif"
    mask_path = tmp_path / "mask.tif"
    _write_raster(raster_path, np.ones((2, 2), dtype="float32"))
    _write_raster(
        misaligned_path,
        np.ones((2, 2), dtype="float32"),
        transform=from_origin(400001, 3900000, 30, 30),
    )
    _write_raster(mask_path, np.ones((2, 2), dtype="uint8"), dtype="uint8", nodata=0)
    mapping = {
        "erosion": {name: str(raster_path) for name in ("Aa", "R", "K", "LS", "C", "P")},
        "health": {name: str(raster_path) for name in ("SHI", "AN", "OM", "CEC", "pH", "SW", "BD", "MBC", "RESP")},
        "environment": {
            name: str(raster_path) for name in ("Elevation", "Cur", "TWI", "LST", "SoilType")
        },
        "spatial": {"template": str(raster_path), "cropland_mask": str(mask_path)},
    }
    mapping["health"]["SHI"] = str(misaligned_path)
    config = {
        "project": {"output_root": "outputs", "log_root": "logs", "random_seed": 9},
        "data": {"metadata_file": "data/metadata.csv"},
    }

    status = run_validate(None, config, mapping, config_path)

    assert status == 2
    run_dir = next((tmp_path / "outputs").glob("run_*"))
    assert (run_dir / "tables" / "raster_quality.csv").is_file()
    assert "health.SHI" in (run_dir / "report.md").read_text(encoding="utf-8")
    assert "health.SHI" in capsys.readouterr().err
