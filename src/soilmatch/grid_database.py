"""Convert aligned raster stacks to a chunked, cell-indexed Parquet table."""

from __future__ import annotations

import os
import tempfile
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import rasterio
from rasterio.windows import Window

from soilmatch.raster_validation import assert_aligned, assert_metric_crs


def build_grid_database(
    rasters: dict[str, str | Path],
    output: str | Path,
    *,
    mask: str | Path | None = None,
    batch_size: int = 512,
    overwrite: bool = False,
) -> Path:
    """Write masked cell records from aligned rasters to Parquet in row blocks.

    Cells inside ``mask`` are retained even if one data band is missing; those
    records receive ``valid_cell=False`` and the affected values are NaN.
    """
    if not rasters:
        raise ValueError("At least one raster is required")
    if batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Output exists; set overwrite=True to replace it: {destination}")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=".parquet", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)

    try:
        with ExitStack() as stack:
            datasets = {
                name: stack.enter_context(rasterio.open(path)) for name, path in rasters.items()
            }
            template = next(iter(datasets.values()))
            assert_metric_crs(template)
            for dataset in datasets.values():
                assert_aligned(template, dataset)
            mask_dataset = stack.enter_context(rasterio.open(mask)) if mask is not None else None
            if mask_dataset is not None:
                assert_aligned(template, mask_dataset)

            writer = None
            written_rows = 0
            try:
                for row_off in range(0, template.height, batch_size):
                    height = min(batch_size, template.height - row_off)
                    for col_off in range(0, template.width, batch_size):
                        width = min(batch_size, template.width - col_off)
                        window = Window(col_off, row_off, width, height)
                        rows, cols = np.mgrid[
                            row_off : row_off + height,
                            col_off : col_off + width,
                        ]
                        selected = np.ones((height, width), dtype=bool)
                        if mask_dataset is not None:
                            mask_values = mask_dataset.read(1, window=window, masked=True)
                            mask_data = np.asarray(mask_values.data)
                            selected = ~np.ma.getmaskarray(mask_values) & np.isfinite(mask_data)
                            selected &= mask_data != 0
                        if not selected.any():
                            continue

                        columns: dict[str, np.ndarray] = {
                            "cell_id": (rows * template.width + cols)[selected].astype(np.int64),
                            "row": rows[selected].astype(np.int32),
                            "col": cols[selected].astype(np.int32),
                        }
                        xs, ys = rasterio.transform.xy(
                            template.transform,
                            rows[selected],
                            cols[selected],
                            offset="center",
                        )
                        columns["x"] = np.asarray(xs, dtype=float)
                        columns["y"] = np.asarray(ys, dtype=float)
                        all_valid = np.ones(int(selected.sum()), dtype=bool)
                        for name, dataset in datasets.items():
                            masked_values = dataset.read(1, window=window, masked=True)
                            data = np.asarray(masked_values.data)
                            valid = ~np.ma.getmaskarray(masked_values) & np.isfinite(data)
                            all_valid &= valid[selected]
                            values = data[selected].astype(float, copy=False)
                            columns[name] = np.where(valid[selected], values, np.nan)
                        columns["valid_cell"] = all_valid
                        frame = pd.DataFrame(columns)
                        table = pa.Table.from_pandas(frame, preserve_index=False)
                        if writer is None:
                            writer = pq.ParquetWriter(temporary, table.schema)
                        writer.write_table(table)
                        written_rows += len(frame)
            finally:
                if writer is not None:
                    writer.close()
            if written_rows == 0:
                raise ValueError("No cells were selected by the supplied mask")

        if destination.exists() and not overwrite:
            raise FileExistsError(f"Output appeared during build: {destination}")
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination
