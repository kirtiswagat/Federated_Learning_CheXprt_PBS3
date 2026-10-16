"""Patient-cluster bootstrap with patient IDs retained as a hard contract."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.metrics import evaluate_predictions


def patient_bootstrap(
    predictions: pd.DataFrame,
    labels: tuple[str, ...],
    iterations: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    required = {
        "patient_id",
        *[f"target_{label}" for label in labels],
        *[f"prob_{label}" for label in labels],
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Patient-level bootstrap requires columns: {sorted(missing)}")
    groups = [group for _, group in predictions.groupby("patient_id", sort=True)]
    if not groups:
        raise ValueError("No patient IDs available for bootstrap")
    rng = np.random.default_rng(seed)
    rows = []
    for iteration in range(iterations):
        sample = pd.concat(
            [groups[index] for index in rng.integers(0, len(groups), len(groups))],
            ignore_index=True,
        )
        targets = sample[[f"target_{label}" for label in labels]].to_numpy(float)
        probs = sample[[f"prob_{label}" for label in labels]].to_numpy(float)
        _, macro = evaluate_predictions(targets, probs, labels)
        rows.append({"iteration": iteration, **macro.iloc[0].to_dict()})
    return pd.DataFrame(rows)


def confidence_interval(
    values: pd.Series | np.ndarray, confidence: float = 0.95
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan")
    tail = (1 - confidence) * 50
    return float(np.percentile(values, tail)), float(np.percentile(values, 100 - tail))
