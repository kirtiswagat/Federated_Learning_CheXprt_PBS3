"""Robustness degradation calculations over metric tables."""

from __future__ import annotations

import pandas as pd


def calculate_degradation(
    attacked: pd.DataFrame, clean: pd.DataFrame, keys: list[str] | None = None
) -> pd.DataFrame:
    keys = keys or ["attack_name", "malicious_fraction", "aggregation_method"]
    required = set(keys) | {"macro_auroc", "macro_auprc"}
    if not required <= set(attacked) or not required <= set(clean):
        raise ValueError("metric tables need attack keys and macro metrics")
    clean_values = clean[keys + ["macro_auroc", "macro_auprc"]].rename(
        columns={"macro_auroc": "clean_macro_auroc", "macro_auprc": "clean_macro_auprc"}
    )
    result = attacked.merge(clean_values, on=keys, how="left")
    result["auroc_difference"] = result.macro_auroc - result.clean_macro_auroc
    result["auprc_difference"] = result.macro_auprc - result.clean_macro_auprc
    result["auroc_absolute_degradation"] = result.clean_macro_auroc - result.macro_auroc
    result["auprc_absolute_degradation"] = result.clean_macro_auprc - result.macro_auprc
    result["auroc_retained_percent"] = (
        result.macro_auroc / result.clean_macro_auroc * 100
    )
    return result
