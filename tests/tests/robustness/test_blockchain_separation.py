import pandas as pd

from src.robustness.integrity.separation_analysis import build_separation_table


def test_integrity_does_not_imply_robustness():
    integrity = pd.DataFrame(
        {
            "condition_id": ["x"],
            "aggregation_method": ["fedavg"],
            "ledger_valid": [True],
            "signature_valid": [True],
        }
    )
    robustness = pd.DataFrame(
        {
            "condition_id": ["x"],
            "aggregation_method": ["fedavg"],
            "clean_macro_auroc": [0.9],
            "attacked_macro_auroc": [0.5],
        }
    )
    result = build_separation_table(integrity, robustness)
    assert result.iloc[0].robustness_status == "degraded" and bool(
        result.iloc[0].ledger_valid
    )
