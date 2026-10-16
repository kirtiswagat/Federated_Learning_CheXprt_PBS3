"""Keep update provenance/integrity independent from predictive robustness."""

from __future__ import annotations

import pandas as pd


def build_separation_table(
    integrity: pd.DataFrame, robustness: pd.DataFrame
) -> pd.DataFrame:
    keys = ["condition_id", "aggregation_method"]
    required_integrity = set(keys) | {"ledger_valid", "signature_valid"}
    required_robustness = set(keys) | {"clean_macro_auroc", "attacked_macro_auroc"}
    if not required_integrity <= set(integrity) or not required_robustness <= set(
        robustness
    ):
        raise ValueError(
            "integrity and robustness tables do not contain required separation fields"
        )
    merged = integrity[keys + ["ledger_valid", "signature_valid"]].merge(
        robustness[keys + ["clean_macro_auroc", "attacked_macro_auroc"]],
        on=keys,
        how="inner",
    )
    merged["robustness_status"] = merged.apply(
        lambda row: (
            "degraded"
            if row.attacked_macro_auroc < row.clean_macro_auroc
            else "not_degraded"
        ),
        axis=1,
    )
    return merged
