"""Restore the legacy full-grid cluster-map layout from unchanged group rasters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio


SOIL_LABELS = {26: "冲积土", 27: "潮土", 42: "脱潮土"}


def archive_original_cluster_maps(
    run_dir: Path, *, cluster_counts: tuple[int, ...] = (6, 8, 10, 12), overwrite: bool = False,
) -> list[Path]:
    """Save separate Chinese PNGs with the former full-extent `tab20` layout.

    This only reads the stored cluster GeoTIFFs. It never rewrites the current
    faceted PNGs or the source rasters.
    """
    run_dir = Path(run_dir)
    figure_dir = run_dir / "figures" / "original_style"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "Noto Sans CJK JP",
        "axes.titlecolor": "#263A46",
        "text.color": "#263A46",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })
    output_paths: list[Path] = []
    for cluster_count in cluster_counts:
        source = run_dir / "maps" / f"environment_groups_k{cluster_count}.tif"
        if not source.is_file():
            raise FileNotFoundError(source)
        output = figure_dir / f"environment_groups_k{cluster_count}_original.png"
        if output.exists() and not overwrite:
            raise FileExistsError(f"archived figure already exists: {output}; pass overwrite=True to replace it")
        with rasterio.open(source) as dataset:
            raster = dataset.read(1, masked=True)
        figure, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
        image = axis.imshow(np.ma.masked_equal(raster, 0), cmap="tab20", interpolation="nearest")
        axis.set_title(f"土类内环境组空间格局（每类 k = {cluster_count}）")
        axis.set_axis_off()
        figure.colorbar(image, ax=axis, label="环境组编码")
        soil_note = "主土类：" + "、".join(f"{code} {name}" for code, name in SOIL_LABELS.items())
        figure.text(.5, .012, soil_note, ha="center", va="bottom", fontsize=8, color="#566873")
        figure.savefig(output, dpi=180)
        plt.close(figure)
        output_paths.append(output)

    manifest_path = figure_dir / "archive_manifest.json"
    manifest = {
        "version_kind": "legacy_full_grid_layout_restored_from_unchanged_cluster_rasters",
        "source_rasters": [f"maps/environment_groups_k{k}.tif" for k in cluster_counts],
        "rendering": "full-grid imshow, tab20 palette, continuous group-ID colorbar, axis hidden",
        "soil_type_labels": {str(code): name for code, name in SOIL_LABELS.items()},
        "outputs": [path.name for path in output_paths],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_paths


def main() -> int:
    parser = argparse.ArgumentParser(description="从原始聚类 GeoTIFF 单独恢复旧版全域空间图")
    parser.add_argument("--run-dir", type=Path,
                        default=Path("outputs/run_20261025T114741_470772Z"))
    parser.add_argument("--cluster-counts", type=int, nargs="+", default=[6, 8, 10, 12])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for path in archive_original_cluster_maps(args.run_dir, cluster_counts=tuple(args.cluster_counts),
                                               overwrite=args.overwrite):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
