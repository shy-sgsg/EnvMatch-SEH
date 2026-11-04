#!/usr/bin/env python3
"""Build an offline, searchable Chinese atlas of all project figure outputs."""

from __future__ import annotations

import html
import csv
import gzip
import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ATLAS = ROOT / "docs/FIGURE_ATLAS.html"
ATLAS_INDEX = ROOT / "docs/FIGURE_ATLAS_INDEX.md"
LEGACY_ATLAS = ROOT / "docs/reviews/2026-10-29/phase-h/FIGURE_ATLAS.html"
LEGACY_INDEX = LEGACY_ATLAS.with_name("FIGURE_ATLAS_INDEX.md")
FIGURE_ROOTS = ("docs", "outputs", "experiments", "环境匹配原数据")
FIGURE_EXTENSIONS = {".png", ".svg", ".jpg", ".jpeg", ".webp", ".pdf"}
FORMAT_DIRECTORIES = {"png", "svg", "pdf", "jpg", "jpeg", "webp"}


def review_root() -> Path:
    return ROOT / "docs/reviews/2026-10-29/phase-h"


def available_source(path: Path | None) -> tuple[Path | None, str]:
    if path is None:
        return None, "源数据 CSV"
    if path.is_file():
        return path, "源数据 CSV"
    compressed = path.with_suffix(path.suffix + ".gz")
    if compressed.is_file():
        return compressed, "压缩源数据 CSV.gz"
    return path, "源数据 CSV"

OUTCOME_LABELS = {
    "SHI_candidate": "土壤健康指数（SHI）", "AN": "碱解氮（AN）", "OM": "有机质（OM）",
    "CEC": "阳离子交换量（CEC）", "pH": "土壤酸碱度（pH）", "SW": "土壤含水量（SW）",
    "BD": "土壤容重（BD）", "MBC": "微生物量碳（MBC）", "RESP": "土壤呼吸（RESP）",
}

FIGURE_META: dict[str, tuple[str, str]] = {
    "01_matching_coverage_balance": ("匹配覆盖与环境平衡", "匹配诊断"),
    "02_matching_method_pair_fit": ("不同匹配方法的配对拟合", "模型比较"),
    "03_primary_paired_slopes": ("优先指标的配对斜率", "森林图"),
    "04_repeated_spatial_validation": ("5 km 整块重复空间留出", "空间验证"),
    "05_erosion_increment_delta_r2": ("侵蚀信息的额外预测增量", "模型比较"),
    "06_environmental_heterogeneity": ("环境背景下的关联差异", "异质性森林图"),
    "07_environment_groups_map_and_high_response": ("环境组空间分布与高响应组", "空间图"),
    "08_health_response_curves": ("九项健康指标响应曲线", "非线性响应曲线"),
    "09_pair_distance_and_exposure_diagnostics": ("环境距离、地理距离与侵蚀差", "匹配诊断"),
    "10_exposure_period_comparison": ("两个侵蚀时期的共同像元比较", "时期比较"),
    "01_matching_quality": ("匹配覆盖率与环境平衡", "匹配诊断"),
    "02_paired_health_differences": ("高低侵蚀配对的健康差值", "配对分布"),
    "03_pair_association_heatmap": ("配对差值秩相关", "相关性热图"),
    "04_model_r2_comparison": ("像元级模型拟合度比较", "模型比较"),
    "05_response_curves": ("自然样条与历史择优曲线", "非线性响应曲线"),
    "06_environment_group_forest": ("环境组内负向关联", "森林图"),
    "07_grouping_method_comparison": ("环境分组方法关联分布", "分组比较"),
    "08_key_group_maps": ("G阶段各指标选中环境组的位置", "空间图"),
    "09_spatial_validation": ("空间留出与配对测试表现", "空间验证"),
    "10_environment_group_slope_forest": ("各指标最负环境组斜率", "空间块稳健森林图"),
    "D01_zero_and_positive_distributions": ("零值、正值与侵蚀分布", "分布图"),
    "D02_kde_bandwidth_sensitivity": ("KDE 带宽敏感性", "分布诊断"),
    "D03_gaussian_mixture_diagnostics": ("高斯混合与分量不确定性", "分布诊断"),
    "D04_threshold_candidate_comparison": ("阈值候选方法比较", "阈值诊断"),
    "D05_spatial_bootstrap_stability": ("空间重抽样阈值稳定性", "空间稳定性"),
    "S01_low_erosion_regime_maps": ("低值分区与侵蚀空间分布", "空间图"),
    "S02_low_regime_by_environment_profile": ("环境画像中的低值群体比例", "环境画像热图"),
    "S03_low_regime_connected_components": ("低值连通斑块尺度", "空间斑块统计"),
    "S04_spatial_block_composition_maps": ("5 km 空间块组成", "空间图"),
    "R01_nonlinear_response_curves": ("健康指标的非线性响应", "非线性响应曲线"),
    "R02_rank_association_heatmap": ("秩相关方向与强度", "相关性热图"),
    "R03_paired_slope_forest": ("配对差值斜率与区间", "森林图"),
    "M01_apparent_model_family_comparison": ("不同模型族表观拟合比较", "模型比较"),
    "M02_zero_indicator_and_segmented_models": ("零值指示与分段响应模型", "模型比较"),
    "V01_spatial_block_test_r2": ("空间分块测试拟合度", "空间验证"),
    "Q01_pair_retention_counts": ("筛选后还剩多少匹配对（图号 Q01）", "样本保留"),
    "Q02_matching_quality_bubble_plot": ("重新配对的质量和样本规模（图号 Q02）", "匹配诊断"),
    "Q02_matching_quality_bubble": ("重新配对的质量和样本规模（图号 Q02）", "匹配诊断"),
    "M03_matched_unit_model_family_comparison": ("匹配像元与配对的八类模型比较", "模型比较热图"),
    "R04_matched_pixel_rank_associations": ("匹配端点像元秩相关", "相关性热图"),
    "V02_matched_unit_spatial_validation": ("匹配像元与配对的空间测试表现", "空间验证箱线图"),
    "S05_matched_endpoint_spatial_footprint": ("阈值筛选后匹配端点空间分布", "空间图"),
    "Q03_pair_and_unique_endpoint_counts": ("匹配对数与不重复端点像元数（图号 Q03）", "样本量比较图"),
    "HET01_group_low_value_retention": ("环境组内低值像元保留率", "环境组保留率图"),
    "HET03_group_spatial_test_change": ("环境组空间测试拟合变化", "空间块稳健热图"),
    "HET04_soil_group_paired_slopes": ("主要土类配对斜率与空间块区间", "空间块稳健森林图"),
    "HET05_group_matching_support_balance": ("环境组匹配支持量与平衡", "分组匹配诊断气泡图"),
    "HADD01_gmm_seed_stability": ("GMM 选中成分数与首处分界的随机种子敏感性", "随机种子稳定性图"),
    "HADD02_gmm_posterior_sensitivity": ("GMM 后验阈值提高时的像元归属变化", "后验分类敏感性图"),
    "HADD03_threshold_candidate_stability": ("不同分布方法给出的 Aa 阈值与稳定性", "阈值比较图"),
}

# English IDs are retained as file provenance only; user-facing titles and guides are Chinese.
PROJECT_FIGURE_META: dict[str, tuple[str, str]] = {
    "01_matching_balance": ("匹配对的环境差异与地理距离", "匹配诊断散点图"),
    "02_environment_vector_r2": ("不同环境变量组合的拟合表现", "拟合比较箱线图"),
    "03_cluster_count_r2": ("环境分组数量与拟合表现", "分组比较图"),
    "04_matching_caliper_r2": ("不同匹配距离限制下的关联比较", "匹配诊断图"),
    "04_matching_parameters_r2": ("不同匹配参数下的拟合比较", "参数比较图"),
    "05_retention_upper_envelope": ("样本保留比例与最高表观拟合", "样本保留图"),
    "06_shi_three_result_classes": ("三类土壤健康结果的匹配分析", "结果比较图"),
    "07_model_family_comparison": ("不同模型的关联拟合比较", "模型比较图"),
    "09_erosion_increment_r2": ("加入侵蚀信息前后的拟合变化", "模型比较图"),
    "10_exposure_comparison": ("两类侵蚀暴露下的结果比较", "时期比较图"),
    "candidate_distributions": ("候选环境变量的取值分布", "分布诊断图"),
    "environment_correlations": ("候选环境变量之间的相关性", "相关性热图"),
    "gwr_om_template": ("有机质：全域细网格三种局部模型", "局部模型比较图"),
    "gwr_shi_candidate_template": ("土壤健康指数：全域细网格三种局部模型", "局部模型比较图"),
    "gwr_sw_template": ("土壤含水量：全域细网格三种局部模型", "局部模型比较图"),
    "multiscale_variable_bandwidths": ("不同变量的局部回归空间尺度", "变量尺度图"),
    "grf_local_aa_marginal_effect": ("随机森林估计的侵蚀局部边际效应", "局部效应空间图"),
    "grf_local_r2": ("随机森林在不同地点的局部拟合度", "局部拟合空间图"),
    "local_ale_by_soil_type": ("不同土类中侵蚀对健康的局部累积效应", "分土类响应图"),
    "spatial_shap_aa": ("侵蚀变量对模型预测的空间贡献", "空间解释图"),
    "spatial_shap_interactions": ("环境变量之间的空间交互贡献", "空间交互图"),
    "svc_gam_local_curves": ("空间变化加性模型的局部响应曲线", "局部响应曲线图"),
    "svc_gam_marginal_effect_q50": ("中位侵蚀水平下的局部边际效应", "局部效应空间图"),
    "i01_global_association_heatmap": ("全区侵蚀与健康指标的相关性", "相关性热图"),
    "i02_standardized_effect_forest": ("不同健康指标的标准化侵蚀效应", "效应森林图"),
    "i03_shi_om_sw_density": ("健康指数、有机质和含水量的取值分布", "分布图"),
    "i04_zero_positive_boxplots": ("零侵蚀与正侵蚀像元的健康差异", "组间比较图"),
    "i05_global_response_curves": ("侵蚀强度与健康指标的总体响应曲线", "响应曲线图"),
    "i06_zero_positive_effect_forest": ("零侵蚀与正侵蚀组的效应估计", "效应森林图"),
    "i07_quantile_slopes": ("健康指标不同分位上的侵蚀效应斜率", "分位数斜率图"),
    "i08_full_sample_gam_curves": ("全样本加性模型的侵蚀响应曲线", "响应曲线图"),
    "i09_aa_2020_matched_pair": ("2020 年侵蚀的匹配配对结果", "配对分析图"),
    "i09_aa_2020_matched_pixel": ("2020 年侵蚀的匹配像元结果", "像元分析图"),
    "i09_aa_20yr_matched_pair": ("20 年平均侵蚀的匹配配对结果", "配对分析图"),
    "i09_aa_20yr_matched_pixel": ("20 年平均侵蚀的匹配像元结果", "像元分析图"),
    "i10_m4_best_apparent_r2": ("同土类最近邻匹配下的表观拟合度", "模型比较图"),
    "i11_m4_twi_group_slopes": ("不同地形湿度组的匹配配对斜率", "斜率森林图"),
    "i12_m4_interaction_q_heatmap": ("环境分组与侵蚀关联的交互检验", "交互检验热图"),
    "i13_m4_soil26_twi_q1_sensitivity": ("26 号冲积土中 TWI 第 1 组的敏感性分析", "敏感性分析图"),
    "gistar_negative_response_gwr": ("CEC 的地理加权回归负向响应热点", "局部空间统计图"),
    "gistar_negative_response_gwr-lcr": ("CEC 的局部共线性敏感性热点", "局部空间统计图"),
    "bivariate_local_moran_clusters": ("CEC 的侵蚀—健康局部聚集", "局部空间聚集图"),
    "local_spearman_240m": ("CEC 的局部秩相关（约 240 米单元）", "局部相关空间图"),
    "same_site_erosion_health_colocation": ("CEC 与侵蚀的同位共现", "空间共现图"),
    "health_dimension_comparison_forest": ("不同土壤健康维度的效应比较", "效应森林图"),
    "j_sem_control_sensitivity": ("不同总体与控制条件下的 SHI 路径", "路径系数森林图"),
    "k_sem_method_comparison": ("三种 SEM 方法的路径系数对照", "方法比较图"),
    "l_sem_fit_increment_sensitivity": ("SHI 拟合度与侵蚀增量跨总体比较", "拟合热图"),
    "m_local_coefficient_distributions": ("SHI、OM、SW 局部系数空间分布", "局部系数分布图"),
    "n_local_response_overlap": ("三项结局的强负向局部区域重合度", "空间重叠热图"),
    "o_local_beta_environment_association": ("局部系数与连续环境变量的秩相关", "相关性热图"),
    "p_local_beta_environment_contrasts": ("环境强弱组的局部系数描述性差异", "分组比较图"),
    "q_soil_twi_dimension_paths": ("不同土类和地形湿度组的健康维度路径", "路径系数森林图"),
    "r_chained_path_products": ("链式路径的标准化关联乘积", "路径比较图"),
    "s_two_part_erosion_paths": ("侵蚀发生与正侵蚀强度路径比较", "模型比较图"),
    "t_health_direction_sensitivity": ("原始指标与健康方向评分敏感性对照", "敏感性分析图"),
    "u_nine_outcome_local_beta_maps": ("九项土壤健康指标的局部侵蚀系数地图", "多面板局部回归系数地图"),
    "v_gwr_mgwr_difference_maps": ("标准GWR与MGWR局部系数差异地图", "局部模型差异地图"),
    "w_sem_parallel_dimensions_diagram": ("侵蚀与三类健康维度的平行路径图", "SEM路径图"),
    "x_sem_chained_dimensions_diagram": ("侵蚀与健康维度链式路径图", "SEM链式路径图"),
    "y_sem_nine_outcome_diagram": ("侵蚀与SHI及八项指标的SEM路径图", "多结局结构路径图"),
    "z_sem_two_part_path_diagram": ("侵蚀发生与正侵蚀强度的路径图", "两部分侵蚀路径图"),
    "stress_zoning_map": ("土壤压力分区空间图", "分区地图"),
    "c04_global_best_model_r2": ("全局模型的拟合表现比较", "模型比较图"),
    "c05_mbc_segmented_threshold_candidate": ("微生物量碳的分段响应与候选阈值", "阈值诊断图"),
    "c07_soil_type_matched_response_curves": ("不同土类中的匹配响应曲线", "响应曲线图"),
    "c08_twi_group_matched_response_curves": ("不同地形湿度组中的匹配响应曲线", "响应曲线图"),
    "c09_twi_interaction_slopes": ("地形湿度与侵蚀效应的交互斜率", "交互斜率图"),
    "c10_ndvi_interaction_slopes": ("植被指数与侵蚀效应的交互斜率", "交互斜率图"),
    "c11_environment_group_slope_forest": ("不同环境组的配对斜率", "效应森林图"),
    "c12_environment_group_apparent_r2": ("不同环境组的表观拟合度", "模型比较图"),
    "c16_eight_indicator_gwr_composite": ("八项土壤指标的局部回归结果汇总", "局部回归组合图"),
    "c16_nine_outcome_gwr_composite": ("九项健康结局的局部回归系数图", "局部系数组合地图"),
    "nine_outcome_gwr_coefficients": ("九项健康结局的局部回归系数图", "局部系数组合地图"),
    "c17_mgwr_local_coefficients": ("多尺度局部回归系数分布", "局部系数空间图"),
    "c18_mgwr_variable_scales": ("不同解释变量的作用空间尺度", "变量尺度图"),
    "c19_gwr_mgwr_difference_map": ("单尺度与多尺度局部回归的差异", "模型差异地图"),
    "c30_pcnm_comparison_not_available": ("PCNM 对比图尚无可用数据", "资料状态说明图"),
    "c31_mechanism_path_not_available": ("作用机制路径图尚无可用数据", "资料状态说明图"),
    "cec_gwr_coefficients": ("CEC 的局部 GWR 系数", "局部回归空间图"),
    "r01_nonlinear_response_curves_aa_2020": ("2020 年实际侵蚀下九项健康指标的非线性响应", "非线性响应曲线"),
    "r01_nonlinear_response_curves_aa_20yr": ("20 年平均实际侵蚀下九项健康指标的非线性响应", "非线性响应曲线"),
    "08_key_group_maps": ("G阶段各指标选中环境组的位置", "空间图"),
    "08_selected_group_map": ("H阶段代表性环境组的位置", "空间图"),
    "cec_空间分布": ("阳离子交换量栅格空间分布", "指标空间图"),
    "mbc_空间分布": ("微生物量碳栅格空间分布", "指标空间图"),
    "ph_空间分布": ("土壤酸碱度栅格空间分布", "指标空间图"),
    "呼吸_空间分布": ("土壤呼吸栅格空间分布", "指标空间图"),
    "rf_8_indicators_rotation.apng": ("八项健康指标随机森林空间图集", "空间地图组"),
    "土壤含水率_空间分布": ("土壤含水量栅格空间分布", "指标空间图"),
    "容重_空间分布": ("土壤容重栅格空间分布", "指标空间图"),
    "有机质_空间分布": ("土壤有机质栅格空间分布", "指标空间图"),
    "碱解氮_空间分布": ("碱解氮栅格空间分布", "指标空间图"),
}

# The nine-outcome refresh uses descriptive run filenames, while the reviewed
# Phase-I copies retain their established figure IDs. Treat those as the same
# chart so the atlas shows only the reviewed copy and can keep the Chinese title.
FIGURE_STEM_ALIASES = {
    "cec_corrected_gwr_coefficients": "cec_gwr_coefficients",
    "nine_outcome_gwr_coefficients": "c16_nine_outcome_gwr_composite",
    "nine_outcome_local_spearman": "local_spearman_240m",
    "nine_outcome_bivariate_lisa": "bivariate_local_moran_clusters",
    "nine_outcome_same_site_colocation": "same_site_erosion_health_colocation",
    "nine_outcome_gistar_gwr": "gistar_negative_response_gwr",
    "nine_outcome_gistar_gwr_lcr": "gistar_negative_response_gwr-lcr",
}

PROJECT_FIGURE_GUIDES: dict[str, str] = {
    "gwr_om_template": "三行依次是：未纳入环境控制的地理加权回归、加入环境变量控制的地理加权回归，以及参考多尺度回归思路的空间变化系数模型。每个网格的颜色表示模型估计的局部侵蚀系数；负值表示较高侵蚀与较低有机质相联系，不代表因果效应。此图覆盖全域细网格；另有一张SHI、OM、SW组合图在770个空间均衡位置上比较单尺度地理加权回归与多尺度地理加权回归，统计单位和模型设定不同。",
    "gwr_shi_candidate_template": "三行依次是：未纳入环境控制的地理加权回归、加入环境变量控制的地理加权回归，以及参考多尺度回归思路的空间变化系数模型。每个网格的颜色表示模型估计的局部侵蚀系数；负值表示较高侵蚀与较低土壤健康指数相联系，不代表因果效应。此图覆盖全域细网格；另有一张SHI、OM、SW组合图在770个空间均衡位置上比较单尺度地理加权回归与多尺度地理加权回归，统计单位和模型设定不同。",
    "gwr_sw_template": "三行依次是：未纳入环境控制的地理加权回归、加入环境变量控制的地理加权回归，以及参考多尺度回归思路的空间变化系数模型。每个网格的颜色表示模型估计的局部侵蚀系数；负值表示较高侵蚀与较低土壤含水量相联系，不代表因果效应。此图覆盖全域细网格；另有一张SHI、OM、SW组合图在770个空间均衡位置上比较单尺度地理加权回归与多尺度地理加权回归，统计单位和模型设定不同。",
    "cropland_area_comparison": "将耕地掩膜栅格、边界矢量和研究记录中的面积估计放在一起核对；差异可提示边界口径、栅格分辨率或统计单位不一致。",
    "environment_correlations": "每个格子表示两种环境变量在有效耕地像元上的相关方向和强弱；颜色越深，关系越强。高相关变量可能提供重复信息，不能单凭此图决定最终匹配变量。",
    "candidate_distributions": "每个面板展示一种候选环境变量在研究区的取值分布；横轴是变量值，纵轴是像元数或密度，用来检查偏态、异常范围和变量尺度。",
    "environment_groups_k6": "地图把耕地按土类分别分组；每个面板对应一个土类，颜色代表该土类内部的环境组。不同土类的同色组号不能直接当作同一类环境。",
    "environment_groups_k8": "地图把耕地按土类分别分成 8 个环境组；每个面板单独分组，颜色组号只在该土类内部解释。",
    "environment_groups_k10": "地图把耕地按土类分别分成 10 个环境组；每个面板单独分组，颜色组号只在该土类内部解释。",
    "environment_groups_k12": "地图把耕地按土类分别分成 12 个环境组；每个面板单独分组，颜色组号只在该土类内部解释。",
    "environment_groups_k6_original": "这是较早版本的全域环境组布局复原图；组号表示聚类类别，土类之间的相同颜色不代表同一环境类别。",
    "environment_groups_k8_original": "这是较早版本的全域环境组布局复原图；组号表示聚类类别，土类之间的相同颜色不代表同一环境类别。",
    "environment_groups_k10_original": "这是较早版本的全域环境组布局复原图；组号表示聚类类别，土类之间的相同颜色不代表同一环境类别。",
    "environment_groups_k12_original": "这是较早版本的全域环境组布局复原图；组号表示聚类类别，土类之间的相同颜色不代表同一环境类别。",
    "01_matching_balance": "每个点是一对匹配样本；横轴表示两端环境差异，纵轴表示两端地理距离。点越靠左下，说明两端环境和地理位置都更接近；这不是健康指标平衡表。",
    "02_environment_vector_r2": "箱线图比较不同环境变量组合下模型的表观 R²。箱体中线是中位数，箱体越高表示不同尝试结果差异越大；它是模型筛选诊断，不代表独立空间预测表现。",
    "03_cluster_count_r2": "横轴是每个土类要求的环境分组数量，纵轴是各次尝试的平均表观 R²；曲线高低反映模型拟合变化，不能单独据此确定正式分组数。",
    "04_matching_caliper_r2": "箱线图比较环境距离限制改变时的拟合分布；横向各组是不同限制值，纵轴为表观 R²。较高拟合可能来自结果导向筛选，需连同保留量和空间测试理解。",
    "04_matching_parameters_r2": "三个面板分别改变环境距离上限、地理距离上限和配对侵蚀差范围；每个箱体汇总该参数下 SHI 模型的表观 R²。箱体越高表示尝试结果越不稳定。",
    "05_retention_upper_envelope": "横轴表示筛选后保留的样本比例，纵轴表示筛选尝试中的最高表观 R²；曲线越高表示更容易在筛选后得到高拟合，但保留量越低，越可能过拟合或结果导向。",
    "06_shi_three_result_classes": "三个柱子依次比较 SHI 的全样本拟合、结果筛选后同一批样本的表观拟合、以及空间留出测试表现。Oracle 柱子受结果筛选影响，不能和空间测试柱子当作同等证据。",
    "07_model_family_comparison": "每根柱子表示一种模型族在全样本上的平均表观 R²；表观分数只说明模型在已有样本上的拟合程度，复杂模型高分不等于空间外推更可靠。",
    "08_health_response_curves": "图中以散点展示固定抽取的像元，并叠加健康指标随 20 年平均实际侵蚀变化的拟合曲线。CEC 面板使用按已确认的原始记录公式恢复的数据；该图展示的是模型关联。",
    "i08_full_sample_gam_curves": "九个面板展示 SHI 和八项土壤指标随 20 年平均实际侵蚀变化的全样本 GAM 拟合曲线。CEC 面板使用本项目当前数据版本，曲线表示模型拟合关系。",
    "09_erosion_increment_r2": "比较只用环境变量的模型与额外加入侵蚀变量后的模型拟合差异；R²变化表示预测拟合增量，不表示因果效应。",
    "10_exposure_comparison": "比较两期实际侵蚀指标下健康结果的最佳表观 R²；表观拟合仅说明在当前样本中的拟合程度，不能单独说明哪一期侵蚀造成了健康变化。",
    "local_spearman_240m": "每个方块代表约 240 米空间单元；在每个单元周围最近的 100 个单元内，计算 log1p(20 年平均实际侵蚀) 与 CEC 的 Spearman 秩相关。颜色表示局部相关方向和强弱；窗口彼此重叠，因此图中数值用于描述空间变化，不是局部显著性检验。",
    "gistar_negative_response_gwr": "地图按 1 公里邻域寻找 GWR 局部负向系数强度的空间热点或冷点；非负系数按零处理。热点表示负向局部系数在邻近单元中聚集；显著类别还受正态近似、FDR 校正和结果导向筛选影响。",
    "gistar_negative_response_gwr-lcr": "地图按 1 公里邻域寻找 GWR-LCR 局部负向系数强度的空间热点或冷点；GWR-LCR 是对局部共线性采用岭惩罚的敏感性模型，系数会向零收缩。显著类别还受正态近似、FDR 校正和结果导向筛选影响。",
    "c04_global_best_model_r2": "每个分组比较不同侵蚀时期或结局下模型的全样本拟合度；R²高只表示训练样本拟合较好，需与独立空间留出结果对照。",
    "c05_mbc_segmented_threshold_candidate": "图中比较微生物量碳与侵蚀的分段关系及候选转折位置；转折点是模型估计，不等于已经证实的生态阈值。",
    "c07_soil_type_matched_response_curves": "各曲线表示在不同土类中，匹配样本的健康指标随侵蚀变化的模型估计；比较曲线方向和弯曲程度时要注意各土类样本量不同。",
    "c08_twi_group_matched_response_curves": "各曲线表示不同地形湿度组中匹配样本的响应趋势；组间曲线差异提示关联可能因环境而异，不能直接解释为因果差异。",
    "c09_twi_interaction_slopes": "点和区间表示不同 TWI 环境组中的侵蚀响应斜率；斜率越低表示模型估计的负向关联越强，区间反映估计不确定性。",
    "c10_ndvi_interaction_slopes": "点和区间比较不同植被指数环境组中的侵蚀响应斜率；组间差异属于探索性结果，不能单独认定为植被调节的因果效应。",
    "c11_environment_group_slope_forest": "森林图的点是各环境组配对斜率，横线是不确定区间；零线左侧为负向估计，区间不跨零时方向证据更明确。该结果按健康响应排序，属于探索性筛选。",
    "c12_environment_group_apparent_r2": "各格比较健康指标在不同环境组中的表观 R²；颜色越深表示模型对该组样本的拟合更高，但不代表留出空间或新地点也能达到相同表现。",
    "c16_eight_indicator_gwr_composite": "组合图汇总多个健康指标的局部回归系数地图；颜色表示各地点的局部系数方向和大小。CEC 使用本项目当前数据版本。",
    "c16_nine_outcome_gwr_composite": "九个面板分别显示 SHI 及八项土壤指标的 GWR 局部标准化侵蚀系数；系数方向和颜色按图例读取，CEC 使用本项目当前数据版本。",
    "nine_outcome_gwr_coefficients": "九个面板分别显示 SHI 及八项土壤指标的 GWR 局部标准化侵蚀系数；系数方向和颜色按图例读取，CEC 使用本项目当前数据版本。",
    "health_dimension_comparison_forest": "森林图比较不同土壤健康维度的关联估计；点为估计值，线为不确定区间。各维度的单位和构造不同，重点比较方向和区间，不比较原始斜率大小。",
    "bivariate_local_moran_clusters": "地图比较一个单元的侵蚀高低与邻近单元 CEC 的高低；图中显著类别来自置换检验并按指标做 FDR 校正。",
    "same_site_erosion_health_colocation": "地图按同一约 240 米空间单元中的侵蚀和 CEC 是否高于各自全区均值进行分类，显示两者是否在同一位置高低共现；这不同于检查邻居健康值的 LISA。",
    "gwr_om_template": "地图颜色表示有机质模型在各位置估计的局部侵蚀系数；负值表示模型估计侵蚀增加时有机质下降，系数变化显示空间差异。",
    "gwr_shi_candidate_template": "地图颜色表示 SHI 模型在各位置估计的局部侵蚀系数；系数方向和大小随地点变化表示统计关联可能存在空间异质性。",
    "gwr_sw_template": "地图颜色表示土壤含水量模型的局部侵蚀系数；负值表示模型估计侵蚀增加时含水量下降。局部系数是预测关系，不是因果效应。",
    "stress_zoning_map": "地图将研究区分为不同土壤压力等级；颜色代表分区类别，边界附近的类别差异应结合分区阈值和输入指标理解。",
    "土壤含水率_空间分布": "每个栅格像元的颜色表示预测土壤含水量；地图适合看空间高低位置，不等于逐点实测值。",
    "呼吸_空间分布": "每个像元的颜色表示预测土壤呼吸指标；请按图例读取数值范围，该图展示空间分布而非独立实测点。",
    "容重_空间分布": "每个像元的颜色表示预测土壤容重；颜色差异描述预测栅格的空间格局，不等于土壤采样点间的独立差异。",
    "有机质_空间分布": "每个像元的颜色表示预测土壤有机质；可据图例比较空间高低，但像元值来自预测面。",
    "ph_空间分布": "每个像元的颜色表示预测土壤酸碱度；地图用于观察空间格局，数值解释应看图例。",
    "mbc_空间分布": "每个像元的颜色表示预测微生物量碳；地图展示预测面空间差异，不能当作每个像元都实地测量过。",
    "碱解氮_空间分布": "每个像元的颜色表示预测碱解氮；请按图例读取数值，地图展示的是连续预测栅格。",
    "i01_global_association_heatmap": "每个格子比较一种侵蚀指标与一种健康指标的全区相关方向和强弱；颜色越深，相关越强。相关描述同向变化，不代表因果。",
    "i02_standardized_effect_forest": "点表示标准化后的侵蚀效应估计，横线表示不确定区间；点在零线左侧表示负向估计，区间跨过零线时证据较弱。",
    "i03_shi_om_sw_density": "曲线或柱形展示这些指标在耕地像元中的取值集中位置和离散程度；分布重叠越多，表示取值范围相似。",
    "i04_zero_positive_boxplots": "九个面板比较零侵蚀与正侵蚀像元的健康结果分布，箱体为中间一半像元、须线为第 5–95 百分位；CEC 使用本项目当前数据版本。",
    "i05_global_response_curves": "横轴是侵蚀强度，纵轴是模型估计的健康指标水平；曲线弯曲表示关系可能不是简单直线。曲线是模型估计，不是实验因果效应。",
    "i06_zero_positive_effect_forest": "点和区间比较零侵蚀、正侵蚀两组或相应模型的效应估计；读方向先看零线，再看区间是否明显跨零。",
    "i07_quantile_slopes": "展示侵蚀效应在健康指标不同取值分位上的变化；横轴位置或斜率变化反映低值、中位和高值区域的关联可能不同。",
    "i08_full_sample_gam_curves": "九个面板展示 SHI 和八项土壤指标随 Aa_20yr 变化的 Gaussian GAM 拟合曲线及逐点 95% 区间。CEC 曲线使用本项目当前 CEC 数据版本；由于该数值按包含侵蚀项的公式恢复，这条曲线不能作为独立侵蚀效应证据。",
    "i09_aa_2020_matched_pair": "每一对比较环境相近但侵蚀水平不同的样本；横轴是配对中的侵蚀差，纵轴是健康指标差，斜率表示两种差值是否一起变化。",
    "i09_aa_2020_matched_pixel": "在筛选出的匹配像元中比较侵蚀与健康指标；像元在空间上相邻，像元数不等于独立实测样本数。",
    "i09_aa_20yr_matched_pair": "每一对比较环境相近但 20 年平均侵蚀水平不同的样本；斜率刻画配对侵蚀差与健康差的共同变化。",
    "i09_aa_20yr_matched_pixel": "在匹配后的唯一像元中检视 20 年平均侵蚀与健康指标的关系；像元关联不是因果效应。",
    "i10_m4_best_apparent_r2": "比较 M4 配对样本上候选模型在建模样本中的拟合程度；表观 R²偏高可能来自模型复杂度或结果筛选，需和空间留出结果一起看。",
    "i11_m4_twi_group_slopes": "比较不同地形湿度组内的侵蚀—健康配对斜率；斜率方向表示关联方向，置信区间体现估计不确定性。",
    "i12_m4_interaction_q_heatmap": "热图显示分组变量与侵蚀响应之间的交互检验强弱；较大的统计量或较小的 P 值表示组间差异证据更强，但多重比较和结果导向筛选仍需考虑。",
    "i13_m4_soil26_twi_q1_sensitivity": "仅看 26 号冲积土中按地形湿度指数（TWI）编号为第 1 组的样本，检查估计对分组规则是否敏感；q1 是环境组编号，不是图号 Q01，也不是统计显著性。",
    "cec_gwr_coefficients": "颜色显示各空间位置的局部侵蚀系数：蓝色通常代表负值、红色代表正值；面板按 20 年平均侵蚀和 2020 年侵蚀分别展示 CEC 与侵蚀的局部关系。",
    "c17_mgwr_local_coefficients": "地图展示多尺度局部回归中不同位置的系数大小和方向；系数随地点变化表示统计关联存在空间差异。",
    "c18_mgwr_variable_scales": "比较模型中各变量所用的空间带宽；较小带宽代表模型允许该变量的作用在更短距离内变化。",
    "c19_gwr_mgwr_difference_map": "颜色表示两种局部回归模型在空间估计上的差异；地图反映模型结果差异，不是实测值差异。",
    "c30_pcnm_comparison_not_available": "此图记录当前资料中缺少可复核的 PCNM 对比表，因此不能从图中读取效应量或显著性。",
    "c31_mechanism_path_not_available": "此图记录当前没有可复核的机制路径数据；缺失值不代表路径为零或没有作用。",
    "grf_local_aa_marginal_effect": "地图用颜色显示随机森林模型估计的侵蚀边际效应在不同地点的方向和大小；它是预测模型中的局部贡献，不是单独的因果效应。",
    "grf_local_r2": "地图展示局部模型在不同位置的拟合程度；高拟合不等同于模型已经通过独立空间验证。",
    "local_ale_by_soil_type": "曲线展示在控制模型中其他变量后的侵蚀累积局部效应，并按土类分开；曲线偏离零线表示预测值相对基准变化。",
    "spatial_shap_aa": "地图展示侵蚀变量对模型预测的贡献方向和大小；SHAP 贡献解释模型如何预测，不代表侵蚀的因果效应。",
    "spatial_shap_interactions": "图中颜色展示变量组合对模型预测的交互贡献；只解释模型内部的预测分配，不等同于实地因果机制。",
    "svc_gam_local_curves": "每条局部曲线表示模型允许侵蚀响应随空间位置变化后的拟合趋势；曲线差异提示空间异质性，仍需看空间留出结果。",
    "svc_gam_marginal_effect_q50": "地图显示中位侵蚀水平附近模型估计的局部边际效应；正负号表示模型预测随侵蚀增加时的局部变化方向。",
    "multiscale_variable_bandwidths": "每个变量对应的带宽反映模型允许其影响变化的空间范围；更宽的带宽意味着模型使用更大邻域估计。",
}

INITIAL = {
    "cropland_area_comparison.png": ("耕地面积来源比较", "数据审计", "面积对照"),
    "environment_correlation.png": ("耕地范围内环境变量相关性", "环境匹配", "相关性热图"),
    "erosion_resampling_sensitivity.png": ("实际侵蚀重采样敏感性", "数据审计", "分布比较"),
    "environment_groups_k6.png": ("环境组空间分布（新版，k=6）", "环境分组", "空间图"),
    "environment_groups_k8.png": ("环境组空间分布（新版，k=8）", "环境分组", "空间图"),
    "environment_groups_k10.png": ("环境组空间分布（新版，k=10）", "环境分组", "空间图"),
    "environment_groups_k12.png": ("环境组空间分布（新版，k=12）", "环境分组", "空间图"),
}


def rel(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    return Path(__import__("os").path.relpath(path, ATLAS.parent)).as_posix()


def read_text(path: Path | None) -> str:
    if path is None or not path.is_file():
        return ""
    return clean_caption(path.read_text(encoding="utf-8"))


def clean_caption(value: str) -> str:
    """Remove markdown boilerplate while preserving the explanation text."""
    lines = []
    for line in value.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("# ") or stripped.startswith("Figure data:"):
            continue
        lines.append(stripped.replace("`", "").replace("**", ""))
    return clean_figure_text("\n".join(lines).strip())


def clean_figure_text(value: str) -> str:
    """Use the project's current CEC name consistently in visible atlas text."""
    replacements = (
        ("CEC 面板的散点与样条使用校正公式 CEC=(CEC-20×Erosion)/2.9+18；蓝线使用校正 I1 全样本 GAM 曲线。",
         "CEC 面板的散点与样条使用本项目当前数据版本；蓝线使用当前 I1 全样本 GAM 曲线。由于恢复公式包含侵蚀项，这条曲线不能作为独立侵蚀效应证据。"),
        ("CEC 组内 Spearman 系数已按校正值重算", "CEC 组内 Spearman 系数使用本项目当前数据版本重算"),
        ("CEC 的配对相关已按校正值重算", "CEC 的配对相关使用本项目当前数据版本重算"),
        ("CEC 实线更新为按 5 km 空间验证 R²选模的校正后 I1 Gaussian GAM", "CEC 实线为按 5 km 空间验证 R²选模的 I1 Gaussian GAM"),
        ("图中 CEC 散点由同一随机样本的校正值替换", "图中 CEC 散点也使用同一随机样本中的当前数据值"),
        ("校正后 H1 交互推断表", "当前 H1 交互推断表"),
        ("CEC 已按校正公式重估", "CEC 使用本项目当前数据版本重估"),
        ("CEC 按 CEC = (CEC − 20 × Erosion) / 2.9 + 18 校正；Erosion 为 Aa_20yr、Aa_2020 分别除以各自完整有效对齐栅格全域最大值后的和。",
         "当前 CEC 数值由既定恢复公式得到，公式包含侵蚀项；涉及 CEC 的结果应结合这一构造关系解读。"),
        ("CEC 使用校正后的对齐栅格值", "CEC 使用本项目当前数据版本"),
        ("CEC 组内画像已按校正值重估", "CEC 组内画像使用本项目当前数据版本重估"),
        ("CEC 组间斜率来自校正后 H1 专项结果", "CEC 组间斜率来自当前 H1 专项结果"),
        ("CEC 端点值按校正公式替换", "CEC 端点值使用本项目当前数据版本"),
        ("校正后的表观 R²", "表观 R²"),
        ("CEC 说明：阳离子交换量使用用户指定公式校正，CEC=(CEC−20×Erosion)/2.9+18；",
         "CEC 使用本项目当前数据版本；该版本的恢复公式包含侵蚀项，相关结果应结合这一构造关系解读；"),
        ("CEC 校正：本图 CEC 使用 CEC = (CEC − 20 × Erosion) / 2.9 + 18；",
         "CEC 使用本项目当前数据版本；该版本的恢复公式包含侵蚀项，相关结果应结合这一构造关系解读；"),
        ("由于校正公式直接包含侵蚀量，CEC 的负向关联", "由于当前 CEC 数值的恢复公式直接包含侵蚀量，CEC 的负向关联"),
        ("CEC 的候选配对秩关联、JT 分组统计及配对差值分布已按校正值重算",
         "CEC 的候选配对秩关联、JT 分组统计及配对差值分布使用本项目当前数据版本重算"),
        ("CEC_corrected", "CEC"),
        ("阳离子交换量（CEC，校正值）", "阳离子交换量（CEC）"),
        ("阳离子交换量（校正 CEC）", "阳离子交换量（CEC）"),
        ("校正后的阳离子交换量", "阳离子交换量（CEC）"),
        ("校正后阳离子交换量", "阳离子交换量（CEC）"),
        ("校正后的 CEC", "CEC"),
        ("校正后 CEC", "CEC"),
        ("校正 CEC", "CEC"),
        ("校正值 CEC", "CEC"),
        ("CEC 的校正值", "CEC"),
        ("CEC 校正式", "CEC 数据恢复公式"),
        ("CEC校正式", "CEC 数据恢复公式"),
    )
    for source, target in replacements:
        value = value.replace(source, target)
    return value


def normalized_stem(stem: str) -> str:
    value = stem.lower()
    value = FIGURE_STEM_ALIASES.get(value, value)
    if re.match(r"^i[1-8]_", value) or value.startswith("ic_"):
        value = value.split("_", 1)[1]
    return value


def figure_meta(stem: str, fallback_title: str = "") -> tuple[str, str]:
    if stem in FIGURE_META:
        return FIGURE_META[stem]
    key = normalized_stem(stem)
    if key in PROJECT_FIGURE_META:
        return PROJECT_FIGURE_META[key]
    # The remaining known Phase-I IDs use the same English stem after removing
    # their stage prefix; preserve the report's Chinese label where available.
    for candidate in (stem.lower(), key):
        if candidate in PROJECT_FIGURE_META:
            return PROJECT_FIGURE_META[candidate]
    if fallback_title and re.search(r"[\u4e00-\u9fff]", fallback_title):
        return fallback_title, "统计图"
    return title_from_stem(stem), inferred_chart_type(stem)


def title_from_stem(stem: str) -> str:
    """Translate a file stem into a readable Chinese title without exposing raw IDs."""
    title, _ = PROJECT_FIGURE_META.get(normalized_stem(stem), ("", ""))
    if title:
        return title
    translated = stem
    replacements = (
        ("_", " "), ("matched", "匹配后"), ("matching", "匹配"),
        ("response", "响应"), ("curves", "曲线"), ("curve", "曲线"),
        ("spatial", "空间"), ("group", "分组"), ("environment", "环境"),
        ("erosion", "侵蚀"), ("health", "健康"), ("association", "关联"),
        ("correlation", "相关性"), ("distribution", "分布"), ("comparison", "比较"),
        ("coefficient", "系数"), ("coefficients", "系数"), ("forest", "森林图"),
        ("heatmap", "热图"), ("map", "地图"), ("gwr", "局部回归"), ("mgwr", "多尺度局部回归"),
        ("M4", "同土类最近邻"), ("m4", "同土类最近邻"), ("apparent", "表观"),
        ("r2", "拟合度"), ("sensitivity", "敏感性"), ("threshold", "阈值"),
        ("candidate", "候选"), ("density", "密度"), ("model", "模型"), ("models", "模型"),
    )
    for source, target in replacements:
        translated = translated.replace(source, target)
    translated = re.sub(r"\b(?:[A-Za-z]+\d+|\d{2,})\b", "", translated)
    translated = re.sub(r"\s+", " ", translated).strip(" -_.")
    return translated or "项目分析图"


def figure_companion_paths(image: Path) -> tuple[Path | None, Path | None]:
    parent, stem = figure_group_key(image)
    caption = parent / "captions" / f"{stem}.md"
    source_candidates = (
        parent / "data" / f"{stem}_source.csv",
        parent / "data" / f"{stem}_source.csv.gz",
        parent / "data" / f"{stem}.csv",
        parent / f"{stem}_source.csv",
        parent / f"{stem}_data.csv",
    )
    source = next((path for path in source_candidates if path.is_file()), None)
    return (caption if caption.is_file() else None), source


TERM_DEFINITIONS = (
    (r"\bQ0[123]\b", "Q01/Q02/Q03 只是图号：Q01 统计筛选后留下多少配对，Q02 检查重配质量，Q03 对比配对数和不重复像元数。"),
    (r"\bM3\b", "M3 是匹配方法编号：先在相同土类内按环境变量分成 6 组，再在组内找相似对象配对。"),
    (r"\bM4\b", "M4 是匹配方法编号：先要求两端属于同一土类，再按环境相似程度直接找最近邻；不先分成 6 个环境组。"),
    (r"\bS0\b", "S0 表示全部原始匹配对，作为比较基线。"),
    (r"\bS1\b", "S1 表示只保留配对两端的实际侵蚀值都大于零的样本。"),
    (r"\bT01\b", "T01 表示按侵蚀值分布确定阈值后再筛选；不同后缀说明是沿用原配对、保持原环境组重配，还是重新分组后重配。"),
    (r"\bCDF\b", "CDF 是累积分布：把侵蚀值从低到高排序后，查看每个值以下累计占多少像元。"),
    (r"\bAa\b", "Aa 是项目中已有的实际侵蚀量栅格。"),
    (r"\bGWR\b", "GWR 是地理加权回归，允许侵蚀与健康指标的统计关系随地点变化。"),
    (r"\bGWR-LCR\b", "GWR-LCR 是加入局部岭惩罚的地理加权回归，用来检查局部变量共线性是否影响结果；惩罚会把局部系数向零收缩。"),
    (r"\bGi\*|Getis.?Ord", "Gi* 是热点统计：检验某类数值高值或低值是否在邻近区域集中。这里分析的是局部负向系数强度的聚集。"),
    (r"\bLISA\b|Local Moran", "LISA（局部空间关联）比较本单元侵蚀与邻居健康值的组合；HL 表示本地侵蚀高、邻域健康偏低。"),
    (r"Spearman", "Spearman 秩相关按数值排序后衡量两个变量是否同向或反向变化；ρ 接近 -1 表示较强负向排序关系，接近 0 表示关系较弱。"),
    (r"\bMGWR\b", "MGWR 是多尺度地理加权回归，允许不同解释变量在不同空间尺度上变化。"),
    (r"\bPCNM\b", "PCNM 是用空间特征向量描述空间结构的一类方法；图表缺失资料时不能据此推断结果。"),
    (r"\bCEC\b", "CEC 是阳离子交换量，图中统一使用本项目当前数据版本。"),
    (r"\bSHI\b", "SHI 是综合多项土壤指标形成的土壤健康指数。"),
    (r"\bAN\b", "AN 是碱解氮。"),
    (r"\bOM\b", "OM 是土壤有机质。"),
    (r"\bpH\b", "pH 表示土壤酸碱度。"),
    (r"\bSW\b", "SW 是土壤含水量。"),
    (r"\bBD\b", "BD 是土壤容重。"),
    (r"\bMBC\b", "MBC 是微生物量碳。"),
    (r"\bRESP\b", "RESP 是土壤呼吸指标。"),
    (r"\bR²\b|\bR2\b", "R²表示模型对数据变化的拟合或预测程度，不等同于相关系数，也不表示因果。"),
    (r"\bOracle\b", "Oracle 表示探索时曾参考健康结果筛选候选；结果属于探索性描述，不是独立确认。"),
)

LEGACY_CEC_FIGURE_STEMS = {
    "02_environment_vector_r2",
    "03_cluster_count_r2",
    "05_retention_upper_envelope",
    "07_model_family_comparison",
    "08_health_response_curves",
    "09_erosion_increment_r2",
    "10_exposure_comparison",
    "local_spearman_240m",
    "bivariate_local_moran_clusters",
    "same_site_erosion_health_colocation",
    "gistar_negative_response_gwr",
    "gistar_negative_response_gwr-lcr",
    "i04_zero_positive_boxplots",
    "c04_global_best_model_r2",
    "c07_soil_type_matched_response_curves",
    "c08_twi_group_matched_response_curves",
}

# The source table for this proposed panel contains only CEC, not the full
# outcome set implied by the chart title. Omit it from active reports and atlas.
EXCLUDED_FIGURE_STEMS = {"c12_environment_group_apparent_r2", "c16_eight_indicator_gwr_composite"}

# This earlier corrected I04 render remains in its immutable analysis run, but
# the current review render has a clearer corrected-CEC label and formula note.
SUPERSEDED_FIGURE_GROUPS = {
    (
        "outputs/exploratory_oracle/phase_i/phase_i_cec_corrected_20_2p9_plus18_20261031_i1/figures/global_response",
        "I04_zero_positive_boxplots",
    ),
}


def item_term_notes(item: dict[str, Any]) -> list[str]:
    text = " ".join(str(item.get(key, "")) for key in ("title", "caption", "question", "experiment"))
    notes = []
    for pattern, explanation in TERM_DEFINITIONS:
        if re.search(pattern, text, flags=re.IGNORECASE) and explanation not in notes:
            notes.append(explanation)
    return notes[:6]


def make_item(*, title: str, phase: str, question: str, experiment: str, chart_type: str,
              image: Path, caption: str, source: Path | None = None,
              source_label: str = "源数据 CSV",
              svg: Path | None = None, pdf: Path | None = None,
              scope_note: str = "") -> dict[str, Any]:
    return {"title": clean_figure_text(title), "phase": clean_figure_text(phase),
            "question": clean_figure_text(question), "experiment": clean_figure_text(experiment),
            "chart_type": chart_type, "image": rel(image), "caption": clean_caption(caption),
            "source_label": source_label,
            "source": rel(source), "svg": rel(svg), "pdf": rel(pdf), "scope_note": scope_note}


def add_initial(items: list[dict[str, Any]]) -> None:
    root = review_root()
    base = root / "artifacts/initial/figures"
    for filename, (title, question, chart_type) in INITIAL.items():
        image = base / filename
        stem = image.stem
        data_map = {
            "environment_correlation": root / "artifacts/initial/tables/environment_correlation.csv",
            "erosion_resampling_sensitivity": root / "artifacts/initial/tables/erosion_resampling_sensitivity.csv",
        }
        items.append(make_item(
            title=title, phase="阶段 A–E｜数据准备与环境分组", question=question,
            experiment="耕地掩膜、栅格对齐与环境候选诊断", chart_type=chart_type,
            image=image, source=data_map.get(stem),
            caption={
                "cropland_area_comparison": "对照耕地掩膜栅格、边界矢量和研究记录中的面积估计。",
                "environment_correlation": "显示有效耕地范围内连续环境变量之间的 Pearson 相关，辅助识别冗余变量。",
                "erosion_resampling_sensitivity": "比较面积平均与最近邻重采样得到的既有 Aa 栅格分布；不重建或验证 RUSLE 因子乘积。",
                "environment_groups_k6": "新版分土类分面图：26 冲积土、27 潮土、42 脱潮土。各面板独立聚类，颜色组号只在土类内部解释。",
                "environment_groups_k8": "新版分土类分面图：26 冲积土、27 潮土、42 脱潮土；每类考察 k=8。",
                "environment_groups_k10": "新版分土类分面图：26 冲积土、27 潮土、42 脱潮土；每类考察 k=10。",
                "environment_groups_k12": "新版分土类分面图：26 冲积土、27 潮土、42 脱潮土；每类考察 k=12。",
            }.get(stem, "阶段数据诊断图。"),
            scope_note="阶段 A–E 历史诊断图；可用源表已随审查包归档。"))

    original_style_dir = base / "original_style"
    original_caption = (
        "旧版全域栅格布局的中文版复原图，按原环境组 GeoTIFF 使用 tab20 配色并隐藏坐标轴；"
        "包含 26 冲积土、27 潮土和 42 脱潮土，组别编码是分层聚类结果的类别编号。"
    )
    for k in (6, 8, 10, 12):
        image = original_style_dir / f"environment_groups_k{k}_original.png"
        if not image.is_file():
            continue
        items.append(make_item(
            title=f"环境组空间图（原版全域样式复原，k={k}）",
            phase="阶段 A–E｜数据准备与环境分组",
            question="不同土壤类型中的环境组在空间上如何分布？",
            experiment="原版聚类空间图留存",
            chart_type="原版空间图",
            image=image,
            source=None,
            caption=original_caption,
            scope_note="旧版全域布局的样式复原图；与新版分土类分面图并列归档。",
        ))


def add_phase_index(items: list[dict[str, Any]], *, phase: str, question: str,
                    experiment: str, review_dir: Path, index_path: Path) -> None:
    if not index_path.is_file():
        return
    index = json.loads(index_path.read_text(encoding="utf-8"))
    for record in index:
        name = record.get("name", "")
        stem = Path(name).stem
        title, chart_type = figure_meta(stem, record.get("title") or "")
        image = review_dir / record.get("files", {}).get("png", record.get("formats", [""])[0])
        if not image.is_file():
            image = review_dir / "figures" / f"{stem}.png"
        source_path = record.get("data") or record.get("data_path")
        source, source_label = available_source(review_dir / source_path if source_path else None)
        caption_path = record.get("caption_path")
        caption_ref = caption_path or record.get("caption")
        caption_file = review_dir / caption_ref if caption_ref else None
        try:
            caption_is_file = caption_file is not None and caption_file.is_file()
        except OSError:
            caption_is_file = False
        caption = read_text(caption_file) if caption_is_file else record.get("caption", "")
        items.append(make_item(title=title, phase=phase, question=question,
                               experiment=experiment, chart_type=chart_type,
                               image=image, caption=caption, source=source,
                               source_label=source_label,
                               svg=image.with_suffix(".svg"), pdf=image.with_suffix(".pdf")))


def add_phase_j(items: list[dict[str, Any]]) -> None:
    """Add the reviewed Phase J core and supplemental figures to the project-wide atlas."""
    root = ROOT / "docs/reviews/2026-11-02/phase-j"
    captions_path = root / "reports/PHASE_J_FIGURE_CAPTIONS.md"
    if not captions_path.is_file():
        return
    text = captions_path.read_text(encoding="utf-8")
    matches = list(re.finditer(r"(?m)^## 图([A-Z])｜(.+)$", text))
    source_names = {
        "A": ("figures/data/figure_A_multimethod_response.csv", "图源数据 CSV"),
        "B": ("spatial_synthesis/local_coefficient_distribution_summary.csv", "局部系数汇总表"),
        "C": ("figures/data/figure_C_240m_stress_zones.csv", "分区数据 CSV"),
        "D": ("results/SEM_PATH_COEFFICIENTS.csv", "路径系数表"),
        "E": ("results/SEM_PATH_COEFFICIENTS.csv", "路径系数表"),
        "F": ("results/SEM_PATH_COEFFICIENTS.csv", "路径系数表"),
        "G": ("figures/data/figure_G_sem_fit_delta.csv", "模型比较数据 CSV"),
        "H": ("results/SEM_PATH_COEFFICIENTS.csv", "路径系数表"),
        "I": ("figures/data/figure_I_soil_twi_sem_slopes.csv", "分组路径数据 CSV"),
        "J": ("figures/supplementary/data/J_sem_control_sensitivity_source.csv", "图源数据 CSV"),
        "K": ("figures/supplementary/data/K_sem_method_comparison_source.csv", "图源数据 CSV"),
        "L": ("figures/supplementary/data/L_sem_fit_increment_sensitivity_source.csv", "图源数据 CSV"),
        "M": ("figures/supplementary/data/M_local_coefficient_distributions_source.csv", "图源数据 CSV"),
        "N": ("figures/supplementary/data/N_local_response_overlap_source.csv", "图源数据 CSV"),
        "O": ("figures/supplementary/data/O_local_beta_environment_association_source.csv", "图源数据 CSV"),
        "P": ("figures/supplementary/data/P_local_beta_environment_contrasts_source.csv", "图源数据 CSV"),
        "Q": ("figures/supplementary/data/Q_soil_twi_dimension_paths_source.csv", "图源数据 CSV"),
        "R": ("figures/supplementary/data/R_chained_path_products_source.csv", "图源数据 CSV"),
        "S": ("figures/supplementary/data/S_two_part_erosion_paths_source.csv", "图源数据 CSV"),
        "T": ("figures/supplementary/data/T_health_direction_sensitivity_source.csv", "图源数据 CSV"),
        "U": ("figures/supplementary/data/U_nine_outcome_local_beta_maps_source.csv", "局部系数地图数据 CSV"),
        "V": ("figures/supplementary/data/V_gwr_mgwr_difference_maps_source.csv", "模型差异地图数据 CSV"),
        "W": ("figures/supplementary/data/W_sem_parallel_dimensions_diagram_source.csv", "平行路径系数 CSV"),
        "X": ("figures/supplementary/data/X_sem_chained_dimensions_diagram_source.csv", "链式路径系数 CSV"),
        "Y": ("figures/supplementary/data/Y_sem_nine_outcome_diagram_source.csv", "九结局路径系数 CSV"),
        "Z": ("figures/supplementary/data/Z_sem_two_part_path_diagram_source.csv", "两阶段路径系数 CSV"),
    }
    stems = {
        "A": "A_multimethod_erosion_health_response",
        "B": "B_gwr_mgwr_spatial_patterns",
        "C": "C_stress_sensitive_environment_zones",
        "D": "D_sem_control_hierarchy",
        "E": "E_full_vs_matched_sem",
        "F": "F_erosion_dimension_path_forest",
        "G": "G_sem_fit_and_erosion_increment",
        "H": "H_best_sem_path_diagram",
        "I": "I_soil_twi_sem_comparison",
        "J": "J_sem_control_sensitivity",
        "K": "K_sem_method_comparison",
        "L": "L_sem_fit_increment_sensitivity",
        "M": "M_local_coefficient_distributions",
        "N": "N_local_response_overlap",
        "O": "O_local_beta_environment_association",
        "P": "P_local_beta_environment_contrasts",
        "Q": "Q_soil_twi_dimension_paths",
        "R": "R_chained_path_products",
        "S": "S_two_part_erosion_paths",
        "T": "T_health_direction_sensitivity",
        "U": "U_nine_outcome_local_beta_maps",
        "V": "V_gwr_mgwr_difference_maps",
        "W": "W_sem_parallel_dimensions_diagram",
        "X": "X_sem_chained_dimensions_diagram",
        "Y": "Y_sem_nine_outcome_diagram",
        "Z": "Z_sem_two_part_path_diagram",
    }
    for i, match in enumerate(matches):
        letter, title = match.group(1), match.group(2).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        caption = text[start:end].strip()
        caption = re.sub(r"\n+", " ", caption)
        stem = stems[letter]
        figure_dir = root / "figures" if letter <= "I" else root / "figures/supplementary"
        image = figure_dir / f"{stem}.png"
        source_rel, source_label = source_names[letter]
        items.append(make_item(
            title=f"图{letter}｜{title}",
            phase="阶段 J｜综合分析",
            question="全域、匹配、空间和路径结果如何共同描述侵蚀与土壤健康的关系？",
            experiment="侵蚀—土壤健康综合分析",
            chart_type="综合统计图",
            image=image,
            caption=caption,
            source=root / source_rel,
            source_label=source_label,
            svg=image.with_suffix(".svg"),
            pdf=image.with_suffix(".pdf"),
            scope_note="使用本阶段核对的CEC数据；图中的预测关系不等于独立实测或因果证据。" if letter == "A" else "",
        ))


def add_phase_j_direction(items: list[dict[str, Any]]) -> None:
    """Add the J-9 raw-vs-direction-score sensitivity figure."""
    run = ROOT / "outputs/exploratory_oracle/phase_j/phase_j_20261103_seed20261103_direction_r1"
    image = run / "figures/J9_health_direction_comparison.png"
    source = run / "results/HEALTH_DIRECTION_ASSOCIATIONS.csv"
    if not image.is_file() or not source.is_file():
        return
    items.append(make_item(
        title="J-9｜原始指标与健康方向评分关联对照",
        phase="阶段 J｜综合分析",
        question="将指标按原评价方向评分后，侵蚀关联是否改变？",
        experiment="健康方向评分敏感性分析",
        chart_type="空间块稳健森林图",
        image=image,
        caption=("左图并列比较八项原始预测值与按本地章节草稿方向函数转换后的评分；右图比较物理、化学、生物维度和两种SHI。系数来自相同D1共同环境像元，控制高程、TWI、NDVI、LST和土类，并按5 km空间块聚类。BD方向转换后系数翻转；pH按适宜区间评分；CEC评分在99.998%有效像元达到上限，不能实质解释。按草稿权重重算的SHI与既有SHI栅格未精确复现，因此图中并列显示，不代表替换后的最终SHI。含CEC的关系仍受CEC计算式中侵蚀项的代数耦合影响。"),
        source=source,
        source_label="原始值与方向评分回归汇总 CSV",
        svg=image.with_suffix(".svg"),
        pdf=image.with_suffix(".pdf"),
        scope_note="SHI重算与既有图层不一致；仅作敏感性比较，尚未替换主分析。",
    ))


def add_phase_h(items: list[dict[str, Any]]) -> None:
    root = review_root() / "artifacts/phase_h_core/figures"
    manifest_path = root / "figure_manifest.json"
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    titles = {
        "01_matched_response_direction": ("匹配后响应方向总览", "方向总结"),
        "02_matched_endpoint_boxplots": ("高低侵蚀配对端点分布", "配对分布"),
        "03_apparent_r2_pixel_vs_pair": ("像元与配对表观拟合度", "模型比较"),
        "04_rank_association_vs_r2": ("秩相关与模型拟合度", "关系比较"),
        "05a_fixed_20k_natural_spline": ("固定样本自然样条响应", "非线性响应曲线"),
        "05b_historical_full_sample_apparent": ("历史全样本响应曲线", "历史模型曲线"),
        "05c_matched_pixel_natural_spline": ("匹配像元自然样条响应", "非线性响应曲线"),
        "05d_matched_pair_natural_spline": ("匹配配对自然样条响应", "非线性响应曲线"),
        "06_group_effect_forest": ("环境组效应与空间块区间", "森林图"),
        "07_environment_group_method_comparison": ("环境分组方法比较", "分组比较"),
        "08_selected_group_map": ("H阶段代表性环境组的位置", "空间图"),
        "09_apparent_vs_spatial_test": ("表观拟合与空间测试表现", "空间验证"),
        "10_h1_interaction_inference_repair": ("H1 交互推断可靠性修订", "推断诊断"),
        "11_common_sample_period_comparison": ("共同样本下的时期比较", "时期比较"),
    }
    for stem in manifest.get("figures", []):
        title, chart_type = titles.get(stem, (stem, "统计图"))
        image = root / "png" / f"{stem}.png"
        caption = read_text(root / "captions" / f"{stem}.md")
        source, source_label = available_source(root / "data" / f"{stem}_source.csv")
        items.append(make_item(title=title, phase="Phase H｜环境组响应与推断诊断",
                               question="环境匹配后响应是否存在异质性、且空间推断是否可靠？",
                               experiment="H1 环境组交互与非线性关联探索", chart_type=chart_type,
                               image=image, caption=caption, source=source,
                               source_label=source_label,
                               svg=(root / "svg" / f"{stem}.svg"), pdf=(root / "pdf" / f"{stem}.pdf"),
                               scope_note="Oracle 表观筛选结果已标记；空间验证与表观拟合分开呈现。"))


def add_low_regime(items: list[dict[str, Any]]) -> None:
    root = review_root() / "artifacts/low_erosion_regime/figures"
    manifest_path = root / "figure_manifest.json"
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for record in manifest:
        stem = record.get("id", "")
        title, chart_type = figure_meta(stem, record.get("title", ""))
        caption = read_text(root / record.get("caption", ""))
        source, source_label = available_source(root / record.get("source", ""))
        items.append(make_item(title=title, phase="Phase H｜低侵蚀群体诊断与响应探索",
                               question=record.get("scientific_question", "低值群体如何定义与比较？"),
                               experiment=record.get("experiment", "低侵蚀值分布诊断"), chart_type=chart_type,
                               image=root / record.get("png", ""), caption=caption, source=source,
                               source_label=source_label,
                               svg=root / record.get("svg", ""), pdf=root / record.get("pdf", ""),
                               scope_note="实际 Aa 直接沿用既有数据库；分布阈值与结果导向 Oracle 候选分开保存。"))


def add_matched_unit_supplement(items: list[dict[str, Any]]) -> None:
    root = review_root() / "artifacts/matched_units/figures"
    manifest_path = root / "figure_manifest.json"
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for record in manifest.get("figures", []):
        stem = record.get("id", "")
        title, chart_type = figure_meta(stem, record.get("title", ""))
        source, source_label = available_source(root / record.get("source", ""))
        items.append(make_item(
            title=title,
            phase="Phase H｜匹配像元与配对响应",
            question=record.get("scientific_question", "环境匹配后，侵蚀与健康响应如何变化？"),
            experiment=record.get("experiment", "阈值后匹配单位补充分析"),
            chart_type=chart_type,
            image=root / record.get("png", ""),
            caption=read_text(root / record.get("caption", "")),
            source=source,
            source_label=source_label,
            svg=root / record.get("svg", ""),
            pdf=root / record.get("pdf", ""),
            scope_note="像元按唯一 cell_id 计数，配对按端点差值计数；表观模型搜索与空间测试分开展示。",
        ))


def add_environment_group_heterogeneity(items: list[dict[str, Any]]) -> None:
    root = review_root() / "artifacts/environment_group_heterogeneity/figures"
    manifest_path = root / "figure_manifest.json"
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    periods = {"20yr": "20 年平均实际侵蚀", "2020": "2020 年实际侵蚀"}
    schemes = {
        "S1_both_positive": "只保留两端侵蚀都大于零的配对",
        "T01_empirical_CDF_breakpoint": "按侵蚀分布阈值筛选并沿用原配对",
        "T01_fixed_original": "阈值筛选后保持原环境组并重新配对",
        "T01_refit_k6_filtered_population": "阈值筛选后重新分成六组再配对",
    }
    for record in manifest.get("figures", []):
        stem = record.get("id", "")
        title, chart_type = figure_meta(stem, record.get("title", ""))
        if stem.startswith("HET02_group_rho_change_"):
            tokens = stem.removeprefix("HET02_group_rho_change_").split("_")
            period = periods.get(tokens[0], tokens[0])
            scheme_key = "_".join(tokens[1:])
            scheme = schemes.get(scheme_key, scheme_key)
            title = f"环境组内秩相关变化｜{period}｜{scheme}"
            chart_type = "环境组响应变化热图"
        source, source_label = available_source(root / record.get("source", ""))
        items.append(make_item(
            title=title,
            phase="Phase H｜低侵蚀筛选后的环境组异质性",
            question=record.get("scientific_question", "不同环境组中的低值排除、匹配与响应如何变化？"),
            experiment=record.get("experiment", "低侵蚀阈值后的环境组比较"),
            chart_type=chart_type,
            image=root / record.get("png", ""),
            caption=read_text(root / record.get("caption", "")),
            source=source,
            source_label=source_label,
            svg=root / record.get("svg", ""),
            pdf=root / record.get("pdf", ""),
            scope_note="基于既有环境组标签；组内配对要求两端属于同组。空间测试按 5 km 整块留出，图中 Oracle 探索结果按报告口径解释。",
        ))


def add_distribution_addendum(items: list[dict[str, Any]]) -> None:
    root = review_root() / "artifacts/distribution_diagnostic_addendum/figures"
    manifest_path = root / "figure_manifest.json"
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for record in manifest.get("figures", []):
        stem = record.get("id", "")
        title, chart_type = figure_meta(stem, record.get("title", ""))
        source, source_label = available_source(root / record.get("source", ""))
        items.append(make_item(
            title=title,
            phase="Phase H｜低侵蚀分布诊断补充",
            question=record.get("scientific_question", "分布阈值是否稳定且可辨识？"),
            experiment=record.get("experiment", "多种子 GMM、后验敏感性与阈值比较"),
            chart_type=chart_type,
            image=root / record.get("png", ""),
            caption=read_text(root / record.get("caption", "")),
            source=source,
            source_label=source_label,
            svg=root / record.get("svg", ""),
            pdf=root / record.get("pdf", ""),
            scope_note="仅使用既有侵蚀 Aa 与分布诊断；未读取健康响应，也未重算 RUSLE。",
        ))


def figure_group_key(path: Path) -> tuple[Path, str]:
    """Group rendered formats of one chart, including png/svg/pdf subfolders."""
    parent = path.parent.resolve()
    while parent.name.lower() in FORMAT_DIRECTORIES:
        parent = parent.parent
    return parent, path.stem


def discover_project_figures() -> dict[tuple[Path, str], list[Path]]:
    """Find chart files across project outputs; exclude non-figure PDFs such as papers."""
    groups: dict[tuple[Path, str], list[Path]] = {}
    for root_name in FIGURE_ROOTS:
        source_root = ROOT / root_name
        if not source_root.is_dir():
            continue
        for path in source_root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in FIGURE_EXTENSIONS:
                continue
            relative_path = path.relative_to(ROOT).as_posix()
            if relative_path.startswith((
                "docs/reviews/2026-11-02/phase-j-r4-diagnostics/",
                "docs/reviews/2026-11-02/phase-j-r9-diagnostics/",
            )):
                continue
            if normalized_stem(path.stem) in EXCLUDED_FIGURE_STEMS:
                continue
            relative_parent = path.parent.relative_to(ROOT).as_posix()
            if (relative_parent, path.stem) in SUPERSEDED_FIGURE_GROUPS:
                continue
            relative_parts = path.relative_to(ROOT).parts
            if path.suffix.lower() == ".pdf" and not any(
                "figure" in part.lower() or "chart" in part.lower()
                for part in relative_parts
            ):
                continue
            groups.setdefault(figure_group_key(path), []).append(path.resolve())
    for paths in groups.values():
        paths.sort(key=lambda path: (path.suffix.lower(), path.as_posix()))
    return dict(sorted(groups.items(), key=lambda entry: (entry[0][0].as_posix(), entry[0][1])))


def inferred_phase(path: Path) -> str:
    relative = path.relative_to(ROOT).as_posix()
    match = re.search(r"phase[-_ ]?([a-z])", relative, flags=re.IGNORECASE)
    if match:
        phase = match.group(1).upper()
        return "阶段 I｜土壤健康分析" if phase == "I" and corrected_cec_path(relative) else f"阶段 {phase}｜项目全量产出"
    if relative.startswith("环境匹配原数据/"):
        return "数据与初始诊断｜项目全量产出"
    return "综合与历史图件｜项目全量产出"


def inferred_chart_type(stem: str) -> str:
    lowered = stem.lower()
    for token, label in (
        ("gwr", "GWR 与空间模型"), ("mgwr", "GWR 与空间模型"),
        ("map", "空间图"), ("spatial", "空间分析图"),
        ("forest", "森林图"), ("heatmap", "热图"),
        ("curve", "响应曲线"), ("response", "响应图"),
        ("distribution", "分布图"), ("hist", "分布图"),
        ("kde", "分布图"), ("density", "分布图"),
    ):
        if token in lowered:
            return label
    return "项目归档图件"


@lru_cache(maxsize=512)
def source_has_cec_outcome(path_text: str) -> bool:
    path = Path(path_text)
    if not path.is_file():
        return False
    try:
        opener = gzip.open if path.suffix.lower() == ".gz" else open
        with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                return False
            outcome_fields = [
                name for name in reader.fieldnames
                if name and any(token in name.lower() for token in ("outcome", "indicator", "health_metric", "response_metric"))
            ]
            if not outcome_fields:
                return False
            return any(
                str(row.get(field, "")).strip().upper() == "CEC"
                for row in reader for field in outcome_fields
            )
    except (OSError, UnicodeError, csv.Error):
        return False


def item_contains_cec(item: dict[str, Any]) -> bool:
    text = " ".join(str(item.get(key, "")) for key in ("title", "caption", "question", "experiment", "image"))
    image_stem = Path(str(item.get("image", ""))).stem
    if normalized_stem(image_stem) in LEGACY_CEC_FIGURE_STEMS:
        return True
    if re.search(r"(?<![A-Za-z])CEC(?:_corrected|\b)|阳离子交换量|八项健康指标|九项健康指标", text, flags=re.IGNORECASE):
        return True
    source = item.get("source")
    if source:
        source_path = (ATLAS.parent / source).resolve()
        return source_has_cec_outcome(str(source_path))
    return False


def corrected_cec_path(path_text: str) -> bool:
    path = Path(path_text)
    try:
        relative = path.resolve().relative_to(ROOT).as_posix().lower()
    except ValueError:
        relative = path.as_posix().lower()
    verified_doc_roots = (
        "docs/reviews/2026-10-26/phase-f/",
        "docs/reviews/2026-10-27/phase-g/",
        "docs/reviews/2026-10-29/phase-h/",
        "docs/reviews/2026-10-31/phase-i/",
        "docs/reviews/2026-11-02/phase-j/",
        "docs/reviews/2026-11-05/phase-k/",
    )
    if relative.startswith(verified_doc_roots):
        return True
    if relative.startswith((
        "outputs/exploratory_oracle/phase_j/phase_j_20261102_seed20261102_r9/",
        "outputs/exploratory_oracle/phase_j/phase_j_20261102_seed20261102_r10/",
        # J-9 reads CEC from r10's audited corrected input and records that
        # provenance in the supplement run manifest.
        "outputs/exploratory_oracle/phase_j/phase_j_20261103_seed20261103_direction_r1/",
    )):
        return True
    return relative.startswith("outputs/exploratory_oracle/") and (
        "cec_corrected" in relative or "cec-correction" in relative
    )


def is_stale_cec_figure(item: dict[str, Any]) -> bool:
    if not item_contains_cec(item):
        return False
    candidates = [item.get("image"), item.get("source")]
    return not any(
        candidate and corrected_cec_path((ATLAS.parent / candidate).resolve().as_posix())
        for candidate in candidates
    )


def exact_file_digest(path_text: str) -> str:
    path = (ATLAS.parent / path_text).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def card_quality(item: dict[str, Any]) -> tuple[int, int, int, int]:
    image = str(item.get("image", ""))
    caption = str(item.get("caption", ""))
    title = str(item.get("title", ""))
    return (
        int("docs/reviews/" in image),
        int(bool(item.get("source"))),
        len(re.findall(r"[\u4e00-\u9fff]", title)),
        len(caption),
    )


def deduplicate_identical_figures(items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        groups.setdefault(exact_file_digest(item["image"]), []).append(item)
    result: list[dict[str, Any]] = []
    for group in groups.values():
        ordered = sorted(group, key=card_quality, reverse=True)
        primary = dict(ordered[0])
        alternate_refs = []
        seen_paths = {primary.get("image")}
        for alternative in ordered[1:]:
            image = alternative.get("image")
            if image in seen_paths:
                continue
            seen_paths.add(image)
            alternate_refs.append({
                "image": image,
                "title": alternative.get("title", ""),
                "source": alternative.get("source"),
                "svg": alternative.get("svg"),
                "pdf": alternative.get("pdf"),
            })
        primary["identical_copies"] = alternate_refs
        result.append(primary)
    result.sort(key=lambda item: (item.get("phase", ""), item.get("title", ""), item.get("image", "")))
    return result, len(items) - len(result)


def suppress_review_mirrors(items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Keep the reviewed card and omit run-directory copies of that chart."""
    reviewed = {
        (normalized_stem(Path(str(item["image"])).stem), item.get("title"), item.get("chart_type"))
        for item in items
        if str(item.get("image", "")).startswith("reviews/")
    }
    kept: list[dict[str, Any]] = []
    removed = 0
    for item in items:
        image = str(item.get("image", ""))
        key = (normalized_stem(Path(image).stem), item.get("title"), item.get("chart_type"))
        if image and not image.startswith("reviews/") and key in reviewed:
            removed += 1
            continue
        kept.append(item)
    return kept, removed


def add_project_figure_inventory(items: list[dict[str, Any]]) -> tuple[int, int]:
    """Add every generated project figure not already represented by a curated card."""
    groups = discover_project_figures()
    curated: dict[tuple[Path, str], dict[str, Any]] = {}
    curated_by_stem: dict[str, dict[str, Any]] = {}
    for item in items:
        image = item.get("image")
        if image:
            image_path = (ATLAS.parent / image).resolve()
            curated.setdefault(figure_group_key(image_path), item)
            stem_key = normalized_stem(image_path.stem)
            previous = curated_by_stem.get(stem_key)
            if previous is None or card_quality(item) > card_quality(previous):
                curated_by_stem[stem_key] = item

    for key, paths in groups.items():
        image_by_suffix = {path.suffix.lower(): path for path in paths}
        existing = curated.get(key)
        if existing:
            for field, suffix in (("svg", ".svg"), ("pdf", ".pdf")):
                if not existing.get(field) and suffix in image_by_suffix:
                    existing[field] = rel(image_by_suffix[suffix])
            continue

        preview = next((image_by_suffix[suffix] for suffix in
                        (".png", ".jpg", ".jpeg", ".webp", ".svg", ".pdf")
                        if suffix in image_by_suffix), None)
        if preview is None:
            continue
        relative_parent = preview.parent.relative_to(ROOT).as_posix()
        stem = preview.stem
        curated_match = curated_by_stem.get(normalized_stem(stem))
        caption_path, source_path = figure_companion_paths(preview)
        source, source_label = available_source(source_path)
        title, chart_type = figure_meta(stem)
        caption = read_text(caption_path)
        if curated_match:
            title = curated_match["title"]
            chart_type = curated_match["chart_type"]
            caption = curated_match["caption"] or caption
            question = curated_match["question"]
        else:
            caption = caption or PROJECT_FIGURE_GUIDES.get(normalized_stem(stem), "")
            question = f"这张图用于查看：{title}。"
        local_model_stem = normalized_stem(stem)
        if local_model_stem in {
            "gwr_om_template", "gwr_shi_candidate_template", "gwr_sw_template"
        }:
            caption = PROJECT_FIGURE_GUIDES[local_model_stem]
        if not caption:
            caption = f"图中展示“{title}”相关结果；颜色、坐标和分组按图内图例读取，具体统计口径见源数据表。"
        items.append(make_item(
            title=title,
            phase=inferred_phase(preview),
            question=question,
            experiment=relative_parent,
            chart_type=chart_type,
            image=preview,
            caption=caption,
            source=source,
            source_label=source_label,
            svg=image_by_suffix.get(".svg"),
            pdf=image_by_suffix.get(".pdf"),
            scope_note=f"输出目录：{relative_parent}",
        ))
    return len(groups), sum(len(paths) for paths in groups.values())


def build_html(items: list[dict[str, Any]], *, removed_duplicates: int = 0,
               hidden_stale_cec: int = 0, suppressed_mirrors: int = 0) -> str:
    cards = []
    visual_file_count = len({
        item[key] for item in items for key in ("image", "svg", "pdf") if item.get(key)
    })
    for item in items:
        attrs = " ".join(f'data-{key}="{html.escape(str(item[key]), quote=True)}"'
                         for key in ("phase", "question", "experiment", "chart_type"))
        links = []
        for key, label in (("svg", "矢量 SVG"), ("pdf", "PDF"),
                           ("source", item.get("source_label", "源数据 CSV"))):
            if item.get(key):
                links.append(f'<a href="{html.escape(item[key])}" target="_blank">{label}</a>')
        caption = html.escape(item.get("caption", "")).replace("\n", "<br>")
        note = html.escape(item.get("scope_note", ""))
        term_notes = item_term_notes(item)
        terms_html = ""
        if term_notes:
            terms_html = '<details class="term-notes"><summary>缩写和术语解释</summary><ul>' + "".join(
                f"<li>{html.escape(term)}</li>" for term in term_notes
            ) + "</ul></details>"
        duplicate_html = ""
        image_path = item["image"]
        if Path(image_path).suffix.lower() == ".pdf":
            preview = (f'<div class="thumb"><iframe class="pdf-preview" loading="lazy" '
                       f'src="{html.escape(image_path)}#view=FitH" '
                       f'title="预览图件：{html.escape(item["title"])}"></iframe></div>')
        else:
            preview = (f'<a class="thumb" href="{html.escape(image_path)}" target="_blank" '
                       f'aria-label="查看原图：{html.escape(item["title"])}">'
                       f'<img loading="lazy" src="{html.escape(image_path)}" '
                       f'alt="{html.escape(item["title"])}"></a>')
        cards.append(f'''<article class="card" {attrs}>
  {preview}
  <div class="card-body"><div class="tags"><span>{html.escape(item['phase'])}</span><span>{html.escape(item['chart_type'])}</span></div>
    <h3>{html.escape(item['title'])}</h3><p class="question">{html.escape(item['question'])}</p>
    <p class="caption">{caption}</p>{terms_html}{duplicate_html}{f'<p class="note">{note}</p>' if note else ''}<div class="links">{' · '.join(links)}</div>
  </div></article>''')
    phases = sorted({item["phase"] for item in items})
    types = sorted({item["chart_type"] for item in items})
    questions = sorted({item["question"] for item in items})
    phase_options = ''.join(f'<option>{html.escape(v)}</option>' for v in phases)
    type_options = ''.join(f'<option>{html.escape(v)}</option>' for v in types)
    question_options = ''.join(f'<option>{html.escape(v)}</option>' for v in questions)
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>EnvMatch-SEH 全项目图件总集</title>
<style>
:root{{--ink:#203545;--muted:#5e6e78;--line:#dce4e8;--paper:#f5f7f8;--blue:#285d78;--ochre:#c18d32;--teal:#3e7c78;--white:#fff}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font-family:"Noto Sans CJK JP","Source Han Sans SC",sans-serif;line-height:1.55}}
header{{background:linear-gradient(110deg,#193e55,#2c6a77);color:#fff;padding:38px max(24px,calc((100vw - 1440px)/2)) 30px}}
header h1{{margin:0 0 8px;font-size:clamp(26px,3vw,42px);letter-spacing:.02em}}header p{{max-width:900px;margin:5px 0;color:#e1ecee}}
.meta{{display:flex;gap:12px;flex-wrap:wrap;margin-top:16px}}.meta span{{border:1px solid #ffffff62;border-radius:99px;padding:4px 10px;font-size:13px}}
main{{max-width:1500px;margin:0 auto;padding:22px 24px 56px}}.controls{{display:grid;grid-template-columns:repeat(4,minmax(160px,1fr));gap:10px;background:#fff;border:1px solid var(--line);padding:14px;border-radius:12px;margin:0 0 18px;position:sticky;top:0;z-index:5;box-shadow:0 5px 22px #20354512}}
.controls input,.controls select{{width:100%;min-height:40px;border:1px solid #cad5dc;border-radius:7px;background:#fff;padding:8px 10px;color:var(--ink);font:inherit}}
.count{{color:var(--muted);font-size:14px;margin:10px 2px 18px}}.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,410px),1fr));gap:18px}}
.card{{background:#fff;border:1px solid var(--line);border-radius:12px;overflow:hidden;box-shadow:0 5px 18px #243b4a0b;display:flex;flex-direction:column;min-width:0}}
.thumb{{display:block;background:#fff;border-bottom:1px solid var(--line);padding:8px;min-height:190px}}.thumb img{{display:block;width:100%;height:270px;object-fit:contain}}
.pdf-preview{{display:block;width:100%;height:270px;border:0;background:#eef2f4}}
.card-body{{padding:14px 16px 15px}}.tags{{display:flex;gap:7px;flex-wrap:wrap}}.tags span{{font-size:11px;color:#315e70;background:#eaf1f3;border-radius:99px;padding:3px 8px}}
h3{{margin:9px 0 4px;font-size:18px;line-height:1.35}}p{{margin:7px 0}}.question{{font-size:13px;font-weight:600;color:var(--teal)}}.caption{{font-size:13px;color:#465964;white-space:normal}}.note{{font-size:12px;color:#775e2c;border-left:2px solid var(--ochre);padding-left:8px}}.links{{margin-top:10px;font-size:12px}}a{{color:var(--blue);text-decoration:none}}a:hover{{text-decoration:underline}}
.term-notes,.duplicate-copies{{font-size:12px;color:#52646e;margin:8px 0}}.term-notes summary,.duplicate-copies summary{{cursor:pointer;color:#315e70;font-weight:600}}.term-notes ul,.duplicate-copies ul{{margin:5px 0 0;padding-left:20px}}.term-notes li,.duplicate-copies li{{margin:3px 0}}
.section-title{{font-size:20px;border-bottom:1px solid var(--line);padding-bottom:7px;margin:30px 0 14px}}
.glossary{{background:#fff;border:1px solid var(--line);border-radius:12px;padding:18px 20px;margin:22px 0}}.glossary-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:8px 22px}}.glossary p{{margin:3px 0;font-size:13px;color:#465964}}
@media(max-width:850px){{.controls{{grid-template-columns:repeat(2,minmax(140px,1fr));position:static}}.thumb img,.pdf-preview{{height:230px}}}}@media(max-width:520px){{main{{padding:14px}}.controls{{grid-template-columns:1fr}}header{{padding:26px 18px}}.thumb img,.pdf-preview{{height:210px}}}}
</style></head><body><header><h1>EnvMatch-SEH 全项目图件总集</h1><p>汇集项目各阶段、历史运行目录与审查归档中的全部图件。可按阶段、科学问题、图表类型或文件路径搜索；缩略图可打开原图，提供的 SVG/PDF 可单独查看。</p>
<p>图集只展示当前数据版本的代表图；历史运行副本仍保存在各自归档目录中。PNG、SVG、PDF 等格式作为同一张图的附件，不单独重复列卡。</p>
<div class="meta"><span>图卡 {len(items)} 张</span><span>图形文件 {visual_file_count} 个</span><span>合并完全相同图卡 {removed_duplicates} 张</span><span>隐藏旧版 CEC 图 {hidden_stale_cec} 张</span><span>省略运行副本 {suppressed_mirrors} 张</span><span>本地离线浏览</span></div></header>
<main><div class="controls"><input id="search" type="search" placeholder="搜索图名、指标或关键词"><select id="phase"><option value="">全部阶段</option>{phase_options}</select><select id="question"><option value="">全部科学问题</option>{question_options}</select><select id="kind"><option value="">全部图表类型</option>{type_options}</select></div>
<p id="count" class="count"></p><div id="grid" class="grid">{''.join(cards)}</div>
<section class="glossary"><h2 class="section-title">常见代码和术语说明</h2><div class="glossary-grid">
<p><b>Q01 / Q02 / Q03：</b>只是图号，分别表示配对保留数量、重配质量、配对数与不重复像元数；不是统计方法。</p>
<p><b>M3：</b>匹配方法编号：同一土类内先按环境变量分成 6 组，再在各组内配对。</p>
<p><b>M4：</b>匹配方法编号：配对两端须属于同一土类，然后直接按环境相似程度找最近邻；不先分成 6 组。</p>
<p><b>S0 / S1 / T01：</b>S0 是全部原始配对；S1 只保留两端侵蚀都大于零的配对；T01 是根据侵蚀分布阈值筛选后再按指定规则保留或重新配对。</p>
<p><b>CDF：</b>累积分布，表示侵蚀值从低到高排序后，累计覆盖了多少像元。</p>
<p><b>ρ 与 R²：</b>ρ（rho）表示秩相关方向和强弱；R²表示模型拟合或预测表现，两者不是同一个指标，也都不直接代表因果。</p>
<p><b>GWR / MGWR：</b>地理加权回归 / 多尺度地理加权回归，用于查看统计关系是否随空间位置变化；MGWR 还允许不同变量使用不同空间尺度。</p>
<p><b>健康指标缩写：</b>SHI 土壤健康指数；AN 碱解氮；OM 有机质；CEC 阳离子交换量；pH 酸碱度；SW 含水量；BD 容重；MBC 微生物量碳；RESP 土壤呼吸。</p>
<p><b>图上“26 / 27 / 42”：</b>分别是冲积土、潮土、脱潮土的土类编码。</p>
</div></section>
<h2 class="section-title">图集使用说明</h2><p>完全相同的图像和已在审查目录收录的运行副本不会重复列卡；PNG、SVG、PDF 等不同格式作为同一张图的附件。</p>
<p>各图的统计单位可能是预测栅格像元、匹配配对或空间块；预测像元数不等于独立实测样点数。表观拟合、结果导向探索和空间留出结果需按图卡说明分别理解。</p></main>
<script>
const cards=[...document.querySelectorAll('.card')],search=document.getElementById('search'),phase=document.getElementById('phase'),question=document.getElementById('question'),kind=document.getElementById('kind'),count=document.getElementById('count');
function update(){{const q=search.value.toLowerCase();let n=0;for(const c of cards){{const ok=(!q||c.innerText.toLowerCase().includes(q))&&(!phase.value||c.dataset.phase===phase.value)&&(!question.value||c.dataset.question===question.value)&&(!kind.value||c.dataset.chart_type===kind.value);c.hidden=!ok;if(ok)n++}}count.textContent=`显示 ${{n}} / ${{cards.length}} 张图`;}}
[search,phase,question,kind].forEach(x=>x.addEventListener('input',update));update();
</script></body></html>'''


def add_phase_k(items: list[dict[str, Any]]) -> None:
    """Add the reviewed 20-year single-exposure Phase K figures to the full atlas."""
    root = ROOT / "docs/reviews/2026-11-05/phase-k"
    manifest = root / "figures/FIGURE_MANIFEST.csv"
    captions = {
        "A": ("全域关系、环境匹配与非线性响应", "先比较九项指标的总体关系，再看环境相近配对后的差异和三项重点指标的平滑曲线。相关系数来自不同统计单位，不能直接彼此相减。"),
        "B": ("九项健康结局的局部回归地图", "九幅地图分别显示SHI和八项土壤指标的局部回归系数。颜色表示关系方向和强弱；系数为负不等于通过显著性检验。"),
        "C": ("现场核查位置候选图", "把侵蚀水平、SHI、局部系数和环境背景组合成候选区域，帮助安排后续实地抽样。它们不是经实测验证的正式风险区。"),
        "D": ("逐步控制环境后的SHI关系", "比较逐步加入环境信息时侵蚀与SHI的条件关系变化。控制后关系仍为负，但侵蚀额外提供的预测信息较小。"),
        "E": ("全域与环境匹配结果对照", "并列展示全域像元、匹配端点和匹配配对差值的路径结果。三类数据单位不同，应分别阅读。"),
        "F": ("侵蚀与土壤健康不同方面的关系", "展示侵蚀与物理、化学、生物活动指标组合及SHI的标准化关系，并给出空间分块区间。"),
        "G": ("环境信息之外侵蚀增加的预测信息", "比较控制不同环境背景后，加入侵蚀变量所增加的模型解释量。增量表示预测改善，不表示因果贡献。"),
        "H": ("侵蚀、健康维度与SHI关系示意图", "把核心路径模型中的直接关系和健康维度与SHI之间的关系画在一起。箭头表达模型关联设定，不代表因果方向已被证明。"),
        "I": ("研究发现综合示意图", "把全域相关、环境相近配对、局部回归、环境控制模型和现场抽样候选区串成一条证据线。所有数字均来自已保存的20年平均侵蚀结果。"),
        "J": ("土类与地形湿度分组的SHI关系（补充图）", "展示不同土类和地形湿度组合中的SHI路径估计，帮助观察环境差异。分组曾按探索结果筛选，不能当作独立验证。"),
    }
    sources = {
        "A": "A_source_global.csv", "B": "B_source_local_betas.csv",
        "C": "C_source_zone_profiles.csv", "D": "D_E_source_sem_rankings.csv",
        "E": "D_E_source_sem_rankings.csv", "F": "F_H_source_sem_paths.csv",
        "G": "G_source_sem_increment.csv", "H": "F_H_source_sem_paths.csv",
        "I": "I_source_integrated_findings.csv", "J": "J_source_sem_groups.csv",
    }
    if not manifest.is_file():
        return
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        letter = str(row.get("图号", "")).strip()
        if letter not in captions:
            continue
        stem = str(row.get("文件基名", "")).strip()
        image = root / "figures" / f"{letter}_{stem}.png"
        title, caption = captions[letter]
        items.append(make_item(
            title=f"图{letter}｜{title}", phase="阶段 K｜20年平均侵蚀单情景综合结果",
            question="侵蚀与土壤健康的关系在总体、环境匹配和不同地点是否一致？",
            experiment="复用阶段 F–J 已保存结果，整理单一侵蚀情景证据",
            chart_type="综合统计图" if letter in {"A", "I"} else ("局部空间图" if letter in {"B", "C"} else "路径与模型图"),
            image=image, caption=caption, source=root / "figures" / sources[letter],
            svg=image.with_suffix(".svg"), pdf=image.with_suffix(".pdf"),
            scope_note="仅展示20年平均实际侵蚀情景；属于结果导向的探索性条件关联。"))


def main() -> None:
    items: list[dict[str, Any]] = []
    add_initial(items)

    f_review = ROOT / "docs/reviews/2026-10-26/phase-f"
    add_phase_index(items, phase="Phase F｜匹配设计与空间稳定性",
                    question="环境背景相近时，侵蚀与健康是否仍有关联？",
                    experiment="M3/M4 配对差值、异质性与空间验证",
                    review_dir=f_review, index_path=f_review / "figure_index.json")
    g_review = ROOT / "docs/reviews/2026-10-27/phase-g"
    add_phase_index(items, phase="Phase G｜响应类型与环境异质性",
                    question="不同健康指标与环境背景的响应有何差异？",
                    experiment="配对秩相关、非线性模型与环境组筛选",
                    review_dir=g_review, index_path=g_review / "figures/figure_index.json")
    completion_dir = g_review / "figures/completion"
    completion_image = completion_dir / "10_environment_group_slope_forest.png"
    if completion_image.is_file():
        items.append(make_item(
            title="各指标负向环境组斜率与空间块区间", phase="Phase G｜响应类型与环境异质性",
            question="不同环境背景下，响应方向与强度是否不同？",
            experiment="按结局筛选最负环境组的空间块稳健斜率",
            chart_type="空间块稳健森林图", image=completion_image,
            caption=read_text(completion_dir / "captions/10_environment_group_slope_forest.md"),
            source=completion_dir / "data/10_environment_group_slope_forest.csv",
            svg=completion_image.with_suffix(".svg"), pdf=completion_image.with_suffix(".pdf"),
            scope_note="按健康响应排序，属于 Oracle 探索，不是独立确认。"))
    add_phase_h(items)
    add_phase_j(items)
    add_phase_j_direction(items)
    add_low_regime(items)
    add_matched_unit_supplement(items)
    add_environment_group_heterogeneity(items)
    add_distribution_addendum(items)
    add_phase_k(items)
    figure_group_count, figure_file_count = add_project_figure_inventory(items)

    items = [
        item for item in items
        if item.get("image") and (ATLAS.parent / item["image"]).is_file()
    ]
    hidden_stale_cec_count = sum(is_stale_cec_figure(item) for item in items)
    items = [item for item in items if not is_stale_cec_figure(item)]
    items, suppressed_mirrors = suppress_review_mirrors(items)
    items, removed_duplicates = deduplicate_identical_figures(items)
    ATLAS.parent.mkdir(parents=True, exist_ok=True)
    ATLAS.write_text(build_html(items, removed_duplicates=removed_duplicates,
                                hidden_stale_cec=hidden_stale_cec_count,
                                suppressed_mirrors=suppressed_mirrors), encoding="utf-8")
    visual_file_count = len({
        item[key] for item in items for key in ("image", "svg", "pdf") if item.get(key)
    })
    ATLAS_INDEX.write_text(
        f"# EnvMatch-SEH 全项目图件总集\n\n"
        f"离线总图集：[FIGURE_ATLAS.html](FIGURE_ATLAS.html)。\n\n"
        f"当前收录 **{len(items)} 张去重图卡**，扫描到 **{figure_file_count} 个图形文件**；"
        f"已合并 {removed_duplicates} 张内容完全相同的重复卡，省略 {suppressed_mirrors} 张已在审查目录收录的运行副本，"
        f"并隐藏 {hidden_stale_cec_count} 张使用旧版 CEC 数据的图。旧图仍在原运行归档中。"
        f"图集展示当前 CEC 数据版本，并保留同图中的其他健康指标。"
        f"已覆盖项目 `docs/`、`outputs/`、`experiments/` 与原始资料目录。\n\n"
        f"图集当前提供 {visual_file_count} 个原图或矢量/PDF文件链接；可用搜索框按方法名、指标名称、"
        f"阶段或运行编号查找。图号 Q01/Q02/Q03 是索引代码；M3/M4 是不同匹配方法，页面顶部与图卡内均有通俗说明。\n",
        encoding="utf-8")
    legacy_target = Path(__import__("os").path.relpath(ATLAS, LEGACY_ATLAS.parent)).as_posix()
    LEGACY_ATLAS.write_text(
        "<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">"
        f"<meta http-equiv=\"refresh\" content=\"0; url={html.escape(legacy_target, quote=True)}\">"
        "<title>EnvMatch-SEH 全项目图件总集</title></head><body>"
        f"<p>图集已移到项目总目录：<a href=\"{html.escape(legacy_target, quote=True)}\">"
        "打开全项目图件总集</a>。</p></body></html>",
        encoding="utf-8")
    LEGACY_INDEX.write_text(
        "# 图集入口\n\n"
        "项目全量图件请打开[EnvMatch-SEH 全项目图件总集](../../../FIGURE_ATLAS.html)。\n",
        encoding="utf-8")
    print(f"写入 {ATLAS.relative_to(ROOT)}：保留 {len(items)} 张图卡，合并完全重复 {removed_duplicates} 张，"
          f"省略运行副本 {suppressed_mirrors} 张，隐藏旧版 CEC 图 {hidden_stale_cec_count} 张；"
          f"扫描 {figure_group_count} 组、{figure_file_count} 个图形文件。")


if __name__ == "__main__":
    main()
