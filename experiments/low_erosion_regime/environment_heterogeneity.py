"""Compare erosion-health responses within fixed environmental groups."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


KENDALL_EXACT_MAX_N = 5000


def _association(x: pd.Series, y: pd.Series) -> tuple[int, float, float, float, float]:
    values = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")})
    values = values.replace([np.inf, -np.inf], np.nan).dropna()
    n = len(values)
    if n < 3 or values["x"].nunique() < 2 or values["y"].nunique() < 2:
        return n, np.nan, np.nan, np.nan, np.nan
    rho = float(spearmanr(values["x"], values["y"]).statistic)
    tau = float(kendalltau(values["x"], values["y"]).statistic) if n <= KENDALL_EXACT_MAX_N else np.nan
    slope = float(np.polyfit(values["x"].to_numpy(float), values["y"].to_numpy(float), 1)[0])
    r_squared = float(np.corrcoef(values["x"], values["y"])[0, 1] ** 2)
    return n, rho, tau, slope, r_squared


def summarize_group_responses(
    cells: pd.DataFrame,
    group_labels: pd.DataFrame,
    populations: Mapping[tuple[str, str, str], pd.DataFrame],
    *,
    group_columns: Sequence[str],
    outcomes: Sequence[str],
) -> pd.DataFrame:
    """Summarize unique matched endpoints and within-group matched pairs.

    Pair summaries only include pairs whose two endpoints share a group label.
    Group assignments are supplied from environment-only Phase H labels.
    """
    if "cell_id" not in cells or "cell_id" not in group_labels:
        raise ValueError("cells and group_labels must both contain cell_id")
    if group_labels["cell_id"].duplicated().any():
        raise ValueError("group_labels must have one row per cell_id")
    missing = sorted(set(group_columns) - set(group_labels.columns))
    if missing:
        raise ValueError(f"unknown environmental group columns: {missing}")

    cell_index = cells.drop_duplicates("cell_id").set_index("cell_id", drop=False)
    label_index = group_labels.set_index("cell_id")
    rows: list[dict[str, object]] = []

    for (method, exposure, scheme), pairs in populations.items():
        if pairs.empty:
            continue
        high_ids = pd.to_numeric(pairs["high_erosion_cell_id"], errors="coerce")
        low_ids = pd.to_numeric(pairs["low_erosion_cell_id"], errors="coerce")
        endpoint_ids = pd.Index(pd.concat([high_ids, low_ids], ignore_index=True).dropna().unique())
        available_ids = endpoint_ids.intersection(cell_index.index)
        endpoint_cells = cell_index.loc[available_ids].copy()
        endpoint_cells = endpoint_cells.join(label_index[list(group_columns)], on="cell_id", how="left")

        for group_column in group_columns:
            high_label = high_ids.map(label_index[group_column])
            low_label = low_ids.map(label_index[group_column])
            eligible_group_pairs = high_label.notna() & low_label.notna()
            same_group = eligible_group_pairs & high_label.eq(low_label)
            same_group_fraction = float(same_group.sum() / eligible_group_pairs.sum()) if eligible_group_pairs.any() else np.nan

            for group_label, pixel_group in endpoint_cells.dropna(subset=[group_column]).groupby(group_column, sort=True):
                # Pair counts refer to within-group pairs used by the pair analysis.
                pair_mask = same_group & high_label.eq(group_label)
                matching_pairs = int(pair_mask.sum())

                for outcome in outcomes:
                    if outcome not in pixel_group:
                        continue
                    exposure_values = pd.to_numeric(pixel_group[exposure], errors="coerce")
                    transformed_exposure = pd.Series(np.log1p(exposure_values.where(exposure_values.ge(0))), index=pixel_group.index)
                    n, rho, tau, slope, r_squared = _association(transformed_exposure, pixel_group[outcome])
                    rows.append({
                        "analysis_unit": "matched_pixel", "exposure": exposure, "method": method,
                        "scheme": scheme, "outcome": outcome, "group_column": group_column,
                        "group_label": str(group_label), "n": n,
                        "n_unique_matched_endpoints": int(pixel_group.loc[
                            pd.to_numeric(pixel_group[outcome], errors="coerce").notna(), "cell_id"
                        ].nunique()),
                        "n_pairs": matching_pairs,
                        "same_group_pair_fraction": same_group_fraction,
                        "pair_population_share": matching_pairs / len(pairs) if len(pairs) else np.nan,
                        "spearman_rho": rho, "kendall_tau": tau, "slope": slope,
                        "kendall_tau_status": (
                            "样本不足或无变异" if n < 3 or not np.isfinite(rho)
                            else "精确计算" if n <= KENDALL_EXACT_MAX_N else "样本量过大，未计算"
                        ),
                        "r_squared_apparent": r_squared,
                        "mean_delta_health": np.nan, "median_delta_health": np.nan,
                        "negative_delta_fraction": np.nan, "spatial_blocks": np.nan,
                        "outcome_guided_selection": True,
                    })

            within = pairs.loc[same_group].copy()
            within["group_label"] = high_label.loc[same_group].astype(str).to_numpy()
            for group_label, pair_group in within.groupby("group_label", sort=True):
                unique_ids = set(pair_group["high_erosion_cell_id"].dropna()) | set(pair_group["low_erosion_cell_id"].dropna())
                for outcome in outcomes:
                    delta_column = f"delta_{outcome}"
                    if delta_column not in pair_group:
                        high_column, low_column = f"high_{outcome}", f"low_{outcome}"
                        if high_column not in pair_group or low_column not in pair_group:
                            continue
                        delta = pd.to_numeric(pair_group[high_column], errors="coerce") - pd.to_numeric(pair_group[low_column], errors="coerce")
                    else:
                        delta = pd.to_numeric(pair_group[delta_column], errors="coerce")
                    exposure_delta = pd.to_numeric(pair_group.get("delta_log1p_erosion"), errors="coerce")
                    n, rho, tau, slope, r_squared = _association(exposure_delta, delta)
                    valid_delta = delta.replace([np.inf, -np.inf], np.nan).dropna()
                    rows.append({
                        "analysis_unit": "pair", "exposure": exposure, "method": method,
                        "scheme": scheme, "outcome": outcome, "group_column": group_column,
                        "group_label": group_label, "n": n,
                        "n_unique_matched_endpoints": len(unique_ids), "n_pairs": len(pair_group),
                        "same_group_pair_fraction": same_group_fraction,
                        "pair_population_share": len(pair_group) / len(pairs) if len(pairs) else np.nan,
                        "spearman_rho": rho, "kendall_tau": tau, "slope": slope,
                        "kendall_tau_status": (
                            "样本不足或无变异" if n < 3 or not np.isfinite(rho)
                            else "精确计算" if n <= KENDALL_EXACT_MAX_N else "样本量过大，未计算"
                        ),
                        "r_squared_apparent": r_squared,
                        "mean_delta_health": float(valid_delta.mean()) if not valid_delta.empty else np.nan,
                        "median_delta_health": float(valid_delta.median()) if not valid_delta.empty else np.nan,
                        "negative_delta_fraction": float((valid_delta < 0).mean()) if not valid_delta.empty else np.nan,
                        "spatial_blocks": int(pair_group["spatial_block_id"].nunique()) if "spatial_block_id" in pair_group else np.nan,
                        "outcome_guided_selection": True,
                    })

    return pd.DataFrame.from_records(rows)


def compare_group_responses_to_baseline(frame: pd.DataFrame) -> pd.DataFrame:
    """Add changes from the all-pairs baseline for matching groups."""
    if frame.empty:
        return frame.copy()
    key_columns = ["analysis_unit", "exposure", "method", "outcome", "group_column", "group_label"]
    metrics = [
        "n", "n_pairs", "spearman_rho", "kendall_tau", "slope", "r_squared_apparent",
        "mean_delta_health", "median_delta_health", "negative_delta_fraction", "spatial_blocks",
    ]
    baseline = frame.loc[frame["scheme"].eq("S0_all"), key_columns + metrics].rename(
        columns={name: f"baseline_{name}" for name in metrics}
    )
    result = frame.merge(baseline, on=key_columns, how="left", validate="many_to_one")
    for metric in (
        "n", "n_pairs", "spearman_rho", "kendall_tau", "slope", "r_squared_apparent",
        "mean_delta_health", "median_delta_health", "negative_delta_fraction", "spatial_blocks",
    ):
        result[f"change_{metric}_vs_S0"] = pd.to_numeric(result[metric], errors="coerce") - pd.to_numeric(result[f"baseline_{metric}"], errors="coerce")
    return result


def validate_group_spatial_performance(
    pairs: pd.DataFrame,
    group_labels: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    outcomes: Sequence[str],
    seeds: Sequence[int],
    buffer_m: int = 500,
) -> pd.DataFrame:
    """Score fixed linear pair-response diagnostics on held-out 5 km blocks.

    No model or group is selected from test scores. Pairs crossing 5 km block
    edges are excluded, and training pairs near validation/test block footprints
    are removed by the requested spatial buffer.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    coordinate_columns = {"x_high", "y_high", "x_low", "y_low", "delta_log1p_erosion"}
    if not coordinate_columns.issubset(pairs.columns):
        raise ValueError(f"pairs are missing spatial validation columns: {sorted(coordinate_columns - set(pairs.columns))}")
    label_index = group_labels.set_index("cell_id")
    high_x = pd.to_numeric(pairs["x_high"], errors="coerce").to_numpy(float)
    high_y = pd.to_numeric(pairs["y_high"], errors="coerce").to_numpy(float)
    low_x = pd.to_numeric(pairs["x_low"], errors="coerce").to_numpy(float)
    low_y = pd.to_numeric(pairs["y_low"], errors="coerce").to_numpy(float)
    finite_coordinates = np.isfinite(high_x) & np.isfinite(high_y) & np.isfinite(low_x) & np.isfinite(low_y)
    same_block = finite_coordinates & (np.floor(high_x / 5000) == np.floor(low_x / 5000)) & (np.floor(high_y / 5000) == np.floor(low_y / 5000))
    cross_block_excluded = int((finite_coordinates & ~same_block).sum())
    missing_coordinate_excluded = int((~finite_coordinates).sum())
    base = pairs.loc[same_block].copy()
    if base.empty:
        return pd.DataFrame()
    x_coord = (pd.to_numeric(base["x_high"], errors="coerce").to_numpy(float) + pd.to_numeric(base["x_low"], errors="coerce").to_numpy(float)) / 2
    y_coord = (pd.to_numeric(base["y_high"], errors="coerce").to_numpy(float) + pd.to_numeric(base["y_low"], errors="coerce").to_numpy(float)) / 2
    base_blocks = np.floor(x_coord / 5000).astype(int).astype(str) + "_" + np.floor(y_coord / 5000).astype(int).astype(str)
    rows: list[dict[str, object]] = []

    for group_column in group_columns:
        if group_column not in label_index:
            raise ValueError(f"unknown environmental group column: {group_column}")
        high_label = base["high_erosion_cell_id"].map(label_index[group_column])
        low_label = base["low_erosion_cell_id"].map(label_index[group_column])
        same_group = high_label.notna() & low_label.notna() & high_label.eq(low_label)
        grouped = base.loc[same_group].copy()
        grouped["_group_label"] = high_label.loc[same_group].astype(str).to_numpy()
        for group_label, group in grouped.groupby("_group_label", sort=True):
            group_indices = group.index.to_numpy()
            # Recreate positions into `base` so block roles stay group-independent.
            positions = base.index.get_indexer(group_indices)
            group_blocks = base_blocks[positions]
            unique_blocks = np.unique(group_blocks)
            x = pd.to_numeric(group["delta_log1p_erosion"], errors="coerce").to_numpy(float)
            x_coord_group = x_coord[positions]
            y_coord_group = y_coord[positions]
            for outcome in outcomes:
                delta_column = f"delta_{outcome}"
                if delta_column not in group:
                    continue
                y = pd.to_numeric(group[delta_column], errors="coerce").to_numpy(float)
                valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(x_coord_group) & np.isfinite(y_coord_group)
                for split, seed in enumerate(seeds, 1):
                    status = "ok"
                    if len(unique_blocks) < 5 or int(valid.sum()) < 8 or np.ptp(x[valid]) == 0:
                        status = "空间块或有效样本不足"
                        train_count = test_count = 0
                        test_r2 = slope = np.nan
                        train_blocks_n = validation_blocks_n = test_blocks_n = 0
                    else:
                        rng = np.random.default_rng(int(seed))
                        shuffled = rng.permutation(unique_blocks)
                        n_train_blocks = max(1, int(.6 * len(shuffled)))
                        n_validation_blocks = max(1, int(.2 * len(shuffled)))
                        train_blocks = set(shuffled[:n_train_blocks])
                        validation_blocks = set(shuffled[n_train_blocks:n_train_blocks + n_validation_blocks])
                        test_blocks = set(shuffled[n_train_blocks + n_validation_blocks:])
                        if not test_blocks:
                            test_blocks = {shuffled[-1]}
                        train = np.isin(group_blocks, list(train_blocks)) & valid
                        test = np.isin(group_blocks, list(test_blocks)) & valid
                        train_positions = np.flatnonzero(train)
                        keep_train = np.ones(len(train_positions), dtype=bool)
                        for block in validation_blocks | test_blocks:
                            block_x, block_y = map(int, block.split("_"))
                            min_x, min_y = block_x * 5000, block_y * 5000
                            dx = np.maximum(np.maximum(min_x - x_coord_group[train_positions], 0), x_coord_group[train_positions] - (min_x + 5000))
                            dy = np.maximum(np.maximum(min_y - y_coord_group[train_positions], 0), y_coord_group[train_positions] - (min_y + 5000))
                            keep_train &= np.hypot(dx, dy) > buffer_m
                        train_positions = train_positions[keep_train]
                        train_count, test_count = len(train_positions), int(test.sum())
                        train_blocks_n, validation_blocks_n, test_blocks_n = len(train_blocks), len(validation_blocks), len(test_blocks)
                        if train_count < 3 or test_count < 3:
                            status = "缓冲后训练或测试样本不足"
                            test_r2 = slope = np.nan
                        else:
                            model = LinearRegression().fit(x[train_positions, None], y[train_positions])
                            test_indices = np.flatnonzero(test)
                            test_r2 = float(r2_score(y[test_indices], model.predict(x[test_indices, None])))
                            slope = float(model.coef_[0])
                    rows.append({
                        "analysis_unit": "pair", "exposure": base["exposure"].iloc[0] if "exposure" in base else None,
                        "method": base["method"].iloc[0] if "method" in base else None,
                        "scheme": base["scheme"].iloc[0] if "scheme" in base else None,
                        "outcome": outcome, "group_column": group_column, "group_label": group_label,
                        "split": split, "seed": int(seed), "n_pairs_same_group_same_block": int(valid.sum()),
                        "spatial_blocks": int(len(unique_blocks)), "n_train": train_count, "n_test": test_count,
                        "train_blocks": train_blocks_n, "validation_blocks": validation_blocks_n,
                        "test_blocks": test_blocks_n, "buffer_m": buffer_m,
                        "n_cross_block_pairs_excluded": cross_block_excluded,
                        "n_pairs_missing_coordinates_excluded": missing_coordinate_excluded,
                        "model_family": "固定线性基线", "slope": slope,
                        "spatial_test_r_squared": test_r2, "status": status,
                        "outcome_guided_selection": True,
                        "selection_scope": "模型形式固定；未使用测试结果选模或选组",
                    })
    return pd.DataFrame.from_records(rows)
