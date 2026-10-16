"""Shared aggregator result and input checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping, Sequence
import time

import torch

from src.robustness.update_utils import (
    TensorUpdate,
    compute_update_l2_norm,
    validate_update_schema,
)


@dataclass(frozen=True)
class AggregationResult:
    update: TensorUpdate
    client_weights: dict[str, float]
    original_update_norms: dict[str, float]
    transformed_update_norms: dict[str, float]
    parameters: dict[str, float | int | str]
    elapsed_seconds: float
    details: list[dict[str, float | int | str]] = field(default_factory=list)


def validate_inputs(
    updates: Mapping[str, TensorUpdate], sample_counts: Mapping[str, int]
) -> list[str]:
    if not updates:
        raise ValueError("at least one client update is required")
    if set(updates) != set(sample_counts):
        raise ValueError("update clients and sample-count clients differ")
    template = next(iter(updates.values()))
    for client_id, update in updates.items():
        if sample_counts[client_id] <= 0:
            raise ValueError("sample counts must be positive")
        validate_update_schema(update, template)
    return list(updates)


def weighted_result(
    update: TensorUpdate,
    clients: list[str],
    weights: dict[str, float],
    original: dict[str, float],
    transformed: dict[str, float],
    parameters: dict[str, float | int | str],
    start: float,
    details: list[dict[str, float | int | str]] | None = None,
) -> AggregationResult:
    if not all(torch.isfinite(value).all() for value in update.values()):
        raise ValueError("aggregated update contains NaN or infinity")
    if abs(sum(weights.values()) - 1.0) > 1e-8:
        raise ValueError("aggregation weights must sum to one")
    return AggregationResult(
        update,
        weights,
        original,
        transformed,
        parameters,
        time.perf_counter() - start,
        details or [],
    )
