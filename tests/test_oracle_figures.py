from __future__ import annotations

import importlib

import numpy as np
import pandas as pd


def test_generate_figures_writes_ten_requested_png_summaries(tmp_path):
    module = importlib.import_module("experiments.oracle_exploration.generate_figures")
    generate_figures = getattr(module, "generate_figures")
    (tmp_path / "tables").mkdir()
    (tmp_path / "models").mkdir()
    (tmp_path / "matching" / "stage1").mkdir(parents=True)
    (tmp_path / "subgroups").mkdir()
    (tmp_path / "diagnostics").mkdir()
    pd.DataFrame(
        {
            "Aa_20yr": np.linspace(0, 10, 40),
            "AN": np.linspace(1, 5, 40) + np.sin(np.linspace(0, 3, 40)),
        }
    ).to_parquet(tmp_path / "diagnostics" / "plot_data.parquet", index=False)
    pd.DataFrame(
        {
            "outcome": ["SHI_candidate", "SHI_candidate"],
            "score_set": ["matched_cells", "matched_cells"],
            "exposure": ["Aa_20yr", "Aa_2020"],
            "r_squared": [0.25, 0.18],
            "environment_caliper": [0.35, 0.5],
            "geographic_radius_m": [500, 1000],
            "erosion_rank_gap": [0.2, 0.4],
        }
    ).to_parquet(tmp_path / "matching" / "stage1" / "scores.parquet", index=False)
    pd.DataFrame(
        {
            "outcome": ["AN"] * 20,
            "exposure": ["Aa_20yr"] * 20,
            "curve_type": ["full_sample_best_model_partial_dependence"] * 20,
            "exposure_value": np.linspace(0, 10, 20),
            "predicted_outcome": np.linspace(1, 5, 20),
            "selected_model_r_squared": [0.8] * 20,
            "model_name": ["quadratic"] * 20,
        }
    ).to_parquet(tmp_path / "models" / "selected_full_sample_curves.parquet", index=False)
    pd.DataFrame(
        {
            "exposure": ["Aa_20yr"],
            "outcome": ["AN"],
            "r_squared": [0.4],
            "evaluation_set": ["apparent"],
        }
    ).to_parquet(tmp_path / "tables" / "experiment_results.parquet", index=False)

    figures = generate_figures(tmp_path)

    assert len(figures) == 10
    assert all(path.is_file() and path.stat().st_size > 0 for path in figures)
    assert (tmp_path / "figures" / "04_matching_parameters_r2.png").is_file()
    assert (tmp_path / "figures" / "figure_index.json").is_file()
