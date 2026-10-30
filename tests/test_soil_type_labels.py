from experiments.low_erosion_regime.figures import _soil_profile_label
from experiments.oracle_exploration.phase_h_figures import _group_value_label
from scripts.run_phase_g_analysis import _display_group_label


def test_phase_g_soil_group_codes_are_expanded_in_chinese():
    assert _display_group_label("soil_26") == "26 冲积土"
    assert _display_group_label("27:q2") == "27 潮土 · 第 2 分位组"


def test_phase_h_soil_group_labels_include_names():
    assert _group_value_label("soil_42_cluster_3") == "42 脱潮土 · 第 4 组"
    assert _group_value_label("soil_27") == "27 潮土"


def test_low_regime_soil_profile_labels_include_names():
    assert _soil_profile_label("SoilType", "26") == "26 冲积土"
    assert _soil_profile_label("SoilType", "土类 26") == "26 冲积土"
    assert _soil_profile_label("SoilType", "土类 27") == "27 潮土"
    assert _soil_profile_label("SoilType", "土类 42") == "42 脱潮土"
