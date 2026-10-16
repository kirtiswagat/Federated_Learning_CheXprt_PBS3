"""Paired patient-level bootstrap comparisons."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.multiplicity import benjamini_hochberg


def paired_comparison(
    left: pd.DataFrame,
    right: pd.DataFrame,
    metric: str,
    iterations: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    if not left["patient_id"].equals(right["patient_id"]):
        raise ValueError("Paired comparisons require identical patient ordering")
    differences = left[metric].to_numpy(float) - right[metric].to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = np.array(
        [
            differences[rng.integers(0, len(differences), len(differences))].mean()
            for _ in range(iterations)
        ]
    )
    observed = float(np.mean(differences))
    p_value = float((np.sum(np.abs(draws) >= abs(observed)) + 1) / (iterations + 1))
    return {
        "difference": observed,
        "ci_low": float(np.percentile(draws, 2.5)),
        "ci_high": float(np.percentile(draws, 97.5)),
        "p_value": p_value,
    }
