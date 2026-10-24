"""Import and compare existing PCNM summaries with environmental analyses."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd


_ALIASES = {
    "outcome": ("outcome", "indicator", "metric", "indicator_name", "健康指标"),
    "erosion_scenario": ("erosion_scenario", "scenario", "erosion", "侵蚀情景"),
    "method": ("method", "model_stage", "analysis_method", "模型阶段"),
    "coefficient": ("coefficient", "coef", "beta", "回归系数"),
    "standardized_coefficient": (
        "standardized_coefficient",
        "std_beta",
        "beta_std",
        "标准化回归系数",
    ),
    "standard_error": ("standard_error", "std_error", "se", "标准误"),
    "ci_lower": ("ci_lower", "lower_ci", "conf_low", "置信区间下限"),
    "ci_upper": ("ci_upper", "upper_ci", "conf_high", "置信区间上限"),
    "p_value": ("p_value", "pvalue", "p", "P值"),
    "r_squared": ("r_squared", "r2", "R2", "模型R²"),
    "n": ("n", "sample_size", "sample_count", "样本量"),
    "controls": ("controls", "control_variables", "控制变量"),
    "analysis_unit": ("analysis_unit", "sample_unit", "分析单位"),
}
_HEALTH_NAMES = {
    "soil health index": "SHI",
    "soil_health_index": "SHI",
    "碱解氮": "AN",
    "有机质": "OM",
    "阳离子交换量": "CEC",
    "土壤含水率": "SW",
    "土壤容重": "BD",
    "微生物量碳": "MBC",
    "土壤呼吸": "RESP",
}
_SCENARIOS = {
    "aa": "actual",
    "actual": "actual",
    "actual erosion": "actual",
    "actual_erosion": "actual",
    "实际": "actual",
    "实际侵蚀": "actual",
    "ap": "potential",
    "potential": "potential",
    "potential erosion": "potential",
    "potential_erosion": "potential",
    "潜在": "potential",
    "潜在侵蚀": "potential",
}


def read_pcnm_results(path: str | Path, *, sheet_name: str | int | None = 0) -> pd.DataFrame:
    """Read existing PCNM results from UTF-8 CSV or Excel."""
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(source, encoding="utf-8-sig")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(source, sheet_name=sheet_name)
    raise ValueError("PCNM results must be a CSV, XLS, or XLSX file")


def _find_column(frame: pd.DataFrame, canonical: str, mapping: Mapping[str, object]) -> str | None:
    configured = mapping.get(canonical)
    if isinstance(configured, str) and configured in frame.columns:
        return configured
    normalized = {str(column).strip().lower(): column for column in frame.columns}
    for alias in _ALIASES[canonical]:
        if alias in frame.columns:
            return alias
        if alias.lower() in normalized:
            return normalized[alias.lower()]
    return None


def _canonical_outcome(value: object) -> object:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    mapped = _HEALTH_NAMES.get(text.lower())
    if mapped:
        return mapped
    upper = text.upper()
    if upper == "PH":
        return "pH"
    return upper if upper in {"SHI", "AN", "OM", "CEC", "PH", "SW", "BD", "MBC", "RESP"} else text


def _canonical_scenario(value: object, value_mapping: Mapping[str, object]) -> object:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    custom = value_mapping.get(text)
    if custom is not None:
        return custom
    return _SCENARIOS.get(text.lower(), text)


def _normalize(
    frame: pd.DataFrame,
    mapping: Mapping[str, object],
    *,
    value_mapping: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    normalized = pd.DataFrame(index=frame.index)
    for canonical in _ALIASES:
        source = _find_column(frame, canonical, mapping)
        normalized[canonical] = frame[source] if source is not None else np.nan
    normalized["outcome"] = normalized["outcome"].map(_canonical_outcome)
    normalized["erosion_scenario"] = normalized["erosion_scenario"].map(
        lambda value: _canonical_scenario(value, value_mapping or {})
    )
    return normalized.reset_index(drop=True)


def compare_pcnm(
    pcnm: pd.DataFrame,
    matched: pd.DataFrame,
    mapping: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    """Align PCNM and environmental model fields without filling missing data."""
    mapping = mapping or {}
    columns_mapping = mapping.get("columns", mapping)
    if not isinstance(columns_mapping, Mapping):
        columns_mapping = {}
    value_mapping = mapping.get("values", {})
    if not isinstance(value_mapping, Mapping):
        value_mapping = {}
    pcnm_standard = _normalize(pcnm, columns_mapping)
    environmental_standard = _normalize(matched, {})
    if pcnm.empty:
        comparison = environmental_standard.copy()
        for column in _ALIASES:
            if column not in {"outcome", "erosion_scenario"}:
                comparison[f"{column}_pcnm"] = np.nan
                comparison[f"{column}_environmental"] = comparison[column]
                comparison.drop(columns=column, inplace=True)
        comparison["direction_consistent"] = pd.Series(
            [pd.NA] * len(comparison), dtype="boolean"
        )
        comparison["status"] = "pcnm_missing"
        return comparison

    pcnm_standard["erosion_scenario"] = pcnm_standard["erosion_scenario"].map(
        lambda value: _canonical_scenario(value, value_mapping)
    )
    pcnm_standard = pcnm_standard.rename(
        columns={column: f"{column}_pcnm" for column in _ALIASES if column not in {"outcome", "erosion_scenario"}}
    )
    environmental_standard = environmental_standard.rename(
        columns={
            column: f"{column}_environmental"
            for column in _ALIASES
            if column not in {"outcome", "erosion_scenario"}
        }
    )
    comparison = environmental_standard.merge(
        pcnm_standard,
        on=["outcome", "erosion_scenario"],
        how="outer",
        validate="many_to_many",
    )
    left = pd.to_numeric(comparison.get("coefficient_pcnm"), errors="coerce")
    right = pd.to_numeric(comparison.get("coefficient_environmental"), errors="coerce")
    both = left.notna() & right.notna()
    comparison["direction_consistent"] = pd.Series(pd.NA, index=comparison.index, dtype="boolean")
    comparison.loc[both, "direction_consistent"] = np.sign(left[both]) == np.sign(right[both])
    comparison["status"] = np.select(
        [left.isna(), right.isna()],
        ["pcnm_missing_fields", "environmental_result_missing"],
        default="compared",
    )
    return comparison
