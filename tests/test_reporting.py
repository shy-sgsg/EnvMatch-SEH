from __future__ import annotations

import importlib
import json

import pytest


def _api(name: str):
    try:
        module = importlib.import_module("soilmatch.reporting")
    except ModuleNotFoundError:
        pytest.fail("soilmatch.reporting has not been implemented")
    function = getattr(module, name, None)
    assert callable(function), f"reporting.{name} has not been implemented"
    return function


def test_run_directory_records_config_versions_and_never_overwrites(tmp_path):
    create_run_directory = _api("create_run_directory")
    config = {"project": {"name": "EnvMatch-SEH"}, "seed": 5}
    run_dir = create_run_directory(
        tmp_path / "outputs",
        run_id="run_test_001",
        config=config,
        inputs={"Aa": "data/raw/erosion/Aa.tif"},
    )

    assert (run_dir / "config_snapshot.yaml").is_file()
    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert metadata["run_id"] == "run_test_001"
    assert metadata["inputs"]["Aa"] == "data/raw/erosion/Aa.tif"
    assert metadata["software_versions"]["python"]
    for folder in ("tables", "figures", "maps", "matched_pairs", "diagnostics"):
        assert (run_dir / folder).is_dir()
    with pytest.raises(FileExistsError, match="exists"):
        create_run_directory(tmp_path / "outputs", run_id="run_test_001", config=config)


def test_report_writer_preserves_named_sections(tmp_path):
    create_run_directory = _api("create_run_directory")
    write_report = _api("write_report")
    run_dir = create_run_directory(tmp_path, run_id="run_report", config={})

    report = write_report(run_dir, "工程测试报告", {"数据状态": "未提供真实数据", "下一步": "提交真实栅格"})

    text = report.read_text()
    assert "# 工程测试报告" in text
    assert "## 数据状态" in text
    assert "未提供真实数据" in text
    assert "## 下一步" in text
