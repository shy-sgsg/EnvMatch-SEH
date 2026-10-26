"""Create the ten compact comparison figures used by the Oracle review."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.oracle_exploration.nonlinear_search import fit_model_family


def _read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.is_file() else pd.DataFrame()


def _blank(ax: plt.Axes, title: str, message: str = "No matching result rows") -> None:
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_figures(results_dir: Path) -> list[Path]:
    """Write ten PNGs from saved experiment tables and sampled response values."""

    root = Path(results_dir)
    output = root / "figures"
    output.mkdir(parents=True, exist_ok=True)
    stage1 = _read(root / "matching" / "stage1" / "scores.parquet")
    retention = _read(root / "subgroups" / "oracle_retention.parquet")
    full = _read(root / "models" / "full_sample_model_search.parquet")
    spatial = _read(root / "models" / "spatial_model_search.parquet")
    test = _read(root / "models" / "spatial_test_selected_models.parquet")
    regimes = _read(root / "models" / "feature_regime_comparison.parquet")
    full_curves = _read(root / "models" / "selected_full_sample_curves.parquet")
    plot_data = _read(root / "diagnostics" / "plot_data.parquet")
    figures: list[Path] = []

    pair_files = sorted((root / "matching" / "stage2").glob("*_pairs.parquet"))
    pair_tables = [pd.read_parquet(path) for path in pair_files]
    pairs = pd.concat(pair_tables, ignore_index=True) if pair_tables else pd.DataFrame()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if not pairs.empty and {"environmental_distance", "geographic_distance"}.issubset(pairs.columns):
        ax.scatter(pairs["environmental_distance"], pairs["geographic_distance"], s=5, alpha=0.18)
        ax.set(xlabel="Transformed environmental distance", ylabel="Geographic distance (m)", title="1. Stage 2 matching balance")
    else:
        _blank(ax, "1. Stage 2 matching balance")
    figures.append(_save(fig, output / "01_matching_balance.png"))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    if not stage1.empty and {"vector_name", "r_squared", "score_set"}.issubset(stage1.columns):
        data = stage1.copy()
        data["r_squared"] = pd.to_numeric(data["r_squared"], errors="coerce")
        data = data.loc[np.isfinite(data["r_squared"]) & data["score_set"].eq("matched_cells")]
        labels = sorted(data["vector_name"].dropna().unique())
        groups = [data.loc[data["vector_name"].eq(name), "r_squared"].to_numpy() for name in labels]
        if groups:
            ax.boxplot(groups, labels=labels, showfliers=False)
            ax.set(xlabel="Environment vector", ylabel="Apparent R²", title="2. Stage 1 response fit by environment vector")
        else:
            _blank(ax, "2. Stage 1 response fit by environment vector")
    else:
        _blank(ax, "2. Stage 1 response fit by environment vector")
    figures.append(_save(fig, output / "02_environment_vector_r2.png"))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if not stage1.empty and {"cluster_count", "r_squared", "score_set"}.issubset(stage1.columns):
        values = stage1.copy()
        values["r_squared"] = pd.to_numeric(values["r_squared"], errors="coerce")
        values["cluster_count"] = pd.to_numeric(values["cluster_count"], errors="coerce")
        values = values.loc[np.isfinite(values["r_squared"]) & values["score_set"].eq("matched_cells")]
        groups = values.groupby("cluster_count")["r_squared"].mean().sort_index()
        if not groups.empty:
            ax.plot(groups.index, groups.values, marker="o")
            ax.set(xlabel="Requested clusters per soil stratum (0 = none)", ylabel="Mean apparent R²", title="3. Cluster count and response fit")
        else:
            _blank(ax, "3. Cluster count and response fit")
    else:
        _blank(ax, "3. Cluster count and response fit")
    figures.append(_save(fig, output / "03_cluster_count_r2.png"))

    parameter_columns = (
        ("environment_caliper", "Environment caliper"),
        ("geographic_radius_m", "Geographic radius (m)"),
        ("erosion_rank_gap", "Erosion percentile gap"),
    )
    fig, axes = plt.subplots(1, len(parameter_columns), figsize=(14, 4.5), squeeze=False)
    required = {"outcome", "score_set", "r_squared", *(column for column, _ in parameter_columns)}
    if not stage1.empty and required.issubset(stage1.columns):
        values = stage1.copy()
        values["r_squared"] = pd.to_numeric(values["r_squared"], errors="coerce")
        values = values.loc[
            np.isfinite(values["r_squared"])
            & values["score_set"].eq("matched_cells")
            & values["outcome"].eq("SHI_candidate")
        ]
        for ax, (column, label) in zip(axes.flat, parameter_columns, strict=True):
            values[column] = pd.to_numeric(values[column], errors="coerce")
            parameter_values = values.loc[np.isfinite(values[column])]
            grouped = parameter_values.groupby(column, sort=True)["r_squared"].apply(list)
            if grouped.empty:
                _blank(ax, label)
                continue
            ax.boxplot(grouped.to_list(), showfliers=False)
            ax.set_xticks(range(1, len(grouped) + 1), [str(value) for value in grouped.index])
            ax.set(xlabel=label, ylabel="SHI apparent R²", title=label)
            ax.tick_params(axis="x", labelrotation=35)
        fig.suptitle("4. Stage 1 SHI fit by matching threshold")
    else:
        for ax, (_, label) in zip(axes.flat, parameter_columns, strict=True):
            _blank(ax, label)
        fig.suptitle("4. Stage 1 SHI fit by matching threshold")
    figures.append(_save(fig, output / "04_matching_parameters_r2.png"))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if not retention.empty and {"target_retention_fraction", "apparent_r2"}.issubset(retention.columns):
        values = retention.copy()
        values["apparent_r2"] = pd.to_numeric(values["apparent_r2"], errors="coerce")
        values["target_retention_fraction"] = pd.to_numeric(values["target_retention_fraction"], errors="coerce")
        values = values.loc[np.isfinite(values["apparent_r2"])]
        for name, group in values.groupby("predictor_set", sort=False):
            summary = group.groupby("target_retention_fraction")["apparent_r2"].max().sort_index(ascending=False)
            ax.plot(summary.index * 100, summary.values, marker="o", label=str(name))
        if not values.empty:
            ax.legend()
            ax.set(xlabel="Sample retained (%)", ylabel="Best outcome-selected apparent R²", title="5. Retention fraction and Oracle upper envelope")
        else:
            _blank(ax, "5. Retention fraction and Oracle upper envelope")
    else:
        _blank(ax, "5. Retention fraction and Oracle upper envelope")
    figures.append(_save(fig, output / "05_retention_upper_envelope.png"))

    boards = {
        "Full sample": _read(root / "tables" / "leaderboard_full_sample_best.parquet"),
        "Oracle apparent": _read(root / "tables" / "leaderboard_oracle_apparent_best.parquet"),
        "Spatial test": _read(root / "tables" / "leaderboard_spatial_test_best.parquet"),
    }
    fig, ax = plt.subplots(figsize=(7, 4.5))
    values: list[float] = []
    labels: list[str] = []
    for label, board in boards.items():
        selected = board.loc[board["outcome"].eq("SHI_candidate")] if not board.empty and "outcome" in board else pd.DataFrame()
        metric = "test_r2" if label == "Spatial test" and "test_r2" in selected else "r_squared"
        score = pd.to_numeric(selected[metric], errors="coerce").max() if not selected.empty else np.nan
        if np.isfinite(score):
            labels.append(label)
            values.append(float(score))
    if values:
        ax.bar(labels, values)
        ax.set(ylabel="R²", title="6. SHI: full sample, Oracle apparent, and spatial test")
    else:
        _blank(ax, "6. SHI: full sample, Oracle apparent, and spatial test")
    figures.append(_save(fig, output / "06_shi_three_result_classes.png"))

    fig, ax = plt.subplots(figsize=(10, 5))
    if not full.empty and {"model_name", "r_squared"}.issubset(full.columns):
        values = full.copy()
        values["r_squared"] = pd.to_numeric(values["r_squared"], errors="coerce")
        summary = values.groupby("model_name")["r_squared"].mean().sort_values(ascending=False)
        if not summary.empty:
            ax.bar(summary.index.astype(str), summary.values)
            ax.tick_params(axis="x", rotation=55)
            ax.set(ylabel="Mean apparent R²", title="7. Full-sample model-family comparison")
        else:
            _blank(ax, "7. Full-sample model-family comparison")
    else:
        _blank(ax, "7. Full-sample model-family comparison")
    figures.append(_save(fig, output / "07_model_family_comparison.png"))

    fig, axes = plt.subplots(3, 3, figsize=(12, 10), squeeze=False)
    outcomes = ["SHI_candidate", "AN", "OM", "CEC", "pH", "SW", "BD", "MBC", "RESP"]
    if not plot_data.empty and "Aa_20yr" in plot_data:
        for position, (ax, outcome) in enumerate(zip(axes.flat, outcomes, strict=True)):
            if outcome not in plot_data:
                _blank(ax, outcome, "No sampled response values")
                continue
            x = pd.to_numeric(plot_data["Aa_20yr"], errors="coerce")
            y = pd.to_numeric(plot_data[outcome], errors="coerce")
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 10:
                _blank(ax, outcome, "Too few complete sample values")
                continue
            ax.scatter(x[mask], y[mask], s=3, alpha=0.13)
            x_values = x.loc[mask].to_numpy(dtype=float)
            y_values = y.loc[mask].to_numpy(dtype=float)
            grid = np.unique(np.quantile(x_values, np.linspace(0.01, 0.99, 100)))
            sample_curve_drawn = False
            if len(grid) > 1 and np.unique(x_values).size > 1:
                sample_train = pd.DataFrame({"Aa_20yr": x_values, outcome: y_values})
                sample_grid = pd.DataFrame({"Aa_20yr": grid, outcome: float(np.median(y_values))})
                try:
                    sample_prediction, _ = fit_model_family(
                        sample_train,
                        sample_grid,
                        outcome,
                        "Aa_20yr",
                        ["Aa_20yr"],
                        model_family="natural_spline",
                    )
                    ax.plot(
                        grid,
                        sample_prediction,
                        color="darkorange",
                        linestyle="--",
                        linewidth=1.5,
                        label="Plot-sample natural spline",
                    )
                    sample_curve_drawn = True
                except Exception as exc:
                    ax.text(
                        0.02,
                        0.02,
                        f"Spline fit unavailable: {type(exc).__name__}",
                        transform=ax.transAxes,
                        fontsize=6,
                    )

            selected = (
                full_curves.loc[
                    full_curves["outcome"].eq(outcome)
                    & full_curves["exposure"].eq("Aa_20yr")
                    & full_curves["curve_type"].eq("full_sample_best_model_partial_dependence")
                ].copy()
                if not full_curves.empty
                and {"outcome", "exposure", "curve_type", "exposure_value", "predicted_outcome"}.issubset(full_curves.columns)
                else pd.DataFrame()
            )
            if not selected.empty:
                selected["exposure_value"] = pd.to_numeric(selected["exposure_value"], errors="coerce")
                selected["predicted_outcome"] = pd.to_numeric(selected["predicted_outcome"], errors="coerce")
                selected = selected.loc[
                    np.isfinite(selected["exposure_value"])
                    & np.isfinite(selected["predicted_outcome"])
                ].sort_values("exposure_value")
                if not selected.empty:
                    model_name = str(selected.iloc[0].get("model_name", "selected model"))
                    score = pd.to_numeric(selected["selected_model_r_squared"], errors="coerce").iloc[0]
                    ax.plot(
                        selected["exposure_value"],
                        selected["predicted_outcome"],
                        color="firebrick",
                        linewidth=1.8,
                        label="Full-sample best model, environment median",
                    )
                    ax.set_title(f"{outcome} | {model_name}, R²={score:.3f}")
                else:
                    ax.set_title(outcome)
            else:
                ax.set_title(outcome)
            if position == 0 and sample_curve_drawn:
                ax.legend(fontsize=6, framealpha=0.7)
            ax.set_xlabel("Aa_20yr")
        fig.suptitle("8. Aa_20yr fitted response curves: plot-sample spline and full-sample best model")
    else:
        for ax, outcome in zip(axes.flat, outcomes, strict=True):
            _blank(ax, outcome, "No sampled response values")
        fig.suptitle("8. Aa_20yr fitted response curves: no plot sample")
    figures.append(_save(fig, output / "08_health_response_curves.png"))

    fig, ax = plt.subplots(figsize=(10, 5))
    if not regimes.empty and {"outcome", "delta_r2", "regime"}.issubset(regimes.columns):
        values = regimes.copy()
        values["delta_r2"] = pd.to_numeric(values["delta_r2"], errors="coerce")
        values = values.loc[np.isfinite(values["delta_r2"]) & values["regime"].eq("environment_plus_exposure")]
        summary = values.groupby("outcome")["delta_r2"].max().sort_values(ascending=False)
        if not summary.empty:
            ax.bar(summary.index.astype(str), summary.values)
            ax.tick_params(axis="x", rotation=40)
            ax.set(ylabel="ΔR²", title="9. Incremental R² from adding erosion")
        else:
            _blank(ax, "9. Incremental R² from adding erosion")
    else:
        _blank(ax, "9. Incremental R² from adding erosion")
    figures.append(_save(fig, output / "09_erosion_increment_r2.png"))

    fig, ax = plt.subplots(figsize=(10, 5))
    if not full.empty and {"exposure", "outcome", "r_squared"}.issubset(full.columns):
        values = full.copy()
        values["r_squared"] = pd.to_numeric(values["r_squared"], errors="coerce")
        summary = values.groupby(["outcome", "exposure"])["r_squared"].max().unstack("exposure")
        summary.plot(kind="bar", ax=ax)
        ax.set(ylabel="Best apparent R²", title="10. Aa_20yr and Aa_2020 full-sample comparison")
        ax.tick_params(axis="x", rotation=40)
    else:
        _blank(ax, "10. Aa_20yr and Aa_2020 full-sample comparison")
    figures.append(_save(fig, output / "10_exposure_comparison.png"))

    (output / "figure_index.json").write_text(
        pd.Series([path.name for path in figures]).to_json(orient="values", force_ascii=False, indent=2),
        encoding="utf-8",
    )
    return figures
