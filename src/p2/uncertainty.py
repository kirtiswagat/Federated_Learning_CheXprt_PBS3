"""Configurable uncertainty-label transformations and loss masks."""

from __future__ import annotations

import numpy as np


def transform_uncertainty(
    labels: np.ndarray, policy: str, mapping: dict[int, int] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(labels, dtype=np.float32).copy()
    valid = ~np.isnan(values)
    uncertain = values == -1
    if policy == "zero":
        values[uncertain] = 0
    elif policy == "one":
        values[uncertain] = 1
    elif policy == "ignore":
        values[uncertain] = 0
        valid[uncertain] = False
    elif policy == "mapping":
        if mapping is None:
            raise ValueError("label-dependent uncertainty mapping is required")
        for label_index, mapped_value in mapping.items():
            mask = uncertain[:, label_index]
            values[mask, label_index] = mapped_value
    else:
        raise ValueError(f"Unknown uncertainty policy: {policy}")
    values[~valid] = 0
    return values, valid.astype(np.float32)
