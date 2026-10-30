import numpy as np
import pandas as pd

from scripts.run_phase_f_analysis import _valid_stored_map_points


def test_incomplete_stored_map_points_are_discarded_for_recomputation():
    stale = pd.DataFrame({
        "x": [670000.0, 670030.0], "y": [3860000.0, 3860030.0],
        "SoilType": [26, 27], "cluster_within_soil": [np.nan, np.nan],
    })

    result = _valid_stored_map_points(stale)

    assert result.empty


def test_valid_stored_map_points_keep_primary_soil_groups_only():
    source = pd.DataFrame({
        "cell_id": [1, 2, 3], "x": [670000.0, 670030.0, 670060.0],
        "y": [3860000.0, 3860030.0, 3860060.0], "SoilType": [26, 27, 85],
        "cluster_within_soil": [0, 5, 1],
    })

    result = _valid_stored_map_points(source)

    assert result["cell_id"].tolist() == [1, 2]
    assert result["cluster_within_soil"].tolist() == [0, 5]


def test_save_figure_preserves_constrained_layout_with_colorbar(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from scripts.run_phase_f_analysis import save_figure

    fig, ax = plt.subplots(constrained_layout=True)
    points = ax.scatter([0, 1], [0, 1], c=[0, 1])
    fig.colorbar(points, ax=ax)

    result = save_figure(fig, "layout", "caption", pd.DataFrame({"x": [1]}), tmp_path)

    assert len(result["formats"]) == 3
    assert all((tmp_path / path).is_file() for path in result["formats"])
