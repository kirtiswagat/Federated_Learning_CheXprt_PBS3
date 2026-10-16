"""Processed-example and communication-budget calculations."""

from __future__ import annotations

import pandas as pd


def processed_budget(history: pd.DataFrame) -> pd.DataFrame:
    required = {"round", "processed_examples", "communication_bytes"}
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"Missing budget columns: {sorted(missing)}")
    result = history.sort_values("round").copy()
    result["cumulative_processed_examples"] = result["processed_examples"].cumsum()
    result["cumulative_communication_bytes"] = result["communication_bytes"].cumsum()
    return result
