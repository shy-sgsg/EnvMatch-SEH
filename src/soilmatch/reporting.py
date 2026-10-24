"""Run records and concise Markdown reporting."""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import yaml

from soilmatch import __version__


_PACKAGES = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "statsmodels",
    "rasterio",
    "pyarrow",
    "PyYAML",
    "matplotlib",
    "pytest",
    "ruff",
)
_RUN_FOLDERS = ("tables", "figures", "maps", "matched_pairs", "diagnostics")


def _software_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": platform.python_version(), "soilmatch": __version__}
    for package in _PACKAGES:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def _git_revision(project_root: str | Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root or Path.cwd(),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def create_run_directory(
    output_root: str | Path,
    *,
    run_id: str | None = None,
    config: dict[str, Any] | None = None,
    inputs: dict[str, object] | None = None,
    git_revision: str | None = None,
) -> Path:
    """Create a new run directory, config snapshot, and software manifest."""
    root = Path(output_root)
    if run_id is None:
        run_id = datetime.now(UTC).strftime("run_%Y%m%dT%H%M%S_%fZ")
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run_id must be a single nonempty directory name")
    if not run_id.startswith("run_"):
        run_id = f"run_{run_id}"
    destination = root / run_id
    destination.mkdir(parents=True, exist_ok=False)
    for folder in _RUN_FOLDERS:
        (destination / folder).mkdir()
    config_data = config or {}
    (destination / "config_snapshot.yaml").write_text(
        yaml.safe_dump(config_data, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    metadata = {
        "run_id": run_id,
        "created_utc": datetime.now(UTC).isoformat(),
        "git_revision": git_revision if git_revision is not None else _git_revision(),
        "inputs": inputs or {},
        "software_versions": _software_versions(),
    }
    (destination / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return destination


def write_report(
    run_directory: str | Path,
    title: str,
    sections: dict[str, str],
) -> Path:
    """Write named result sections to the run's Markdown report."""
    destination = Path(run_directory) / "report.md"
    if destination.exists():
        raise FileExistsError(f"Report already exists: {destination}")
    lines = [f"# {title}", ""]
    for heading, body in sections.items():
        lines.extend([f"## {heading}", "", body.rstrip(), ""])
    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination
