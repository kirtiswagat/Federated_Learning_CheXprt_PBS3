"""Benjamini-Hochberg false-discovery-rate correction."""

from __future__ import annotations

import numpy as np


def benjamini_hochberg(p_values: list[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    if values.size == 0:
        return values
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 1.0
    for rank in range(len(values), 0, -1):
        index = order[rank - 1]
        running = min(running, values[index] * len(values) / rank)
        adjusted[index] = running
    return np.clip(adjusted, 0, 1)
