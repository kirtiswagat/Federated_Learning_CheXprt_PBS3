"""Coordinate-wise trimmed mean with explicit client-count validation."""

from __future__ import annotations

import time

import torch

from src.robustness.aggregators.base import (
    AggregationResult,
    validate_inputs,
    weighted_result,
)
from src.robustness.update_utils import compute_update_l2_norm


def aggregate_trimmed_mean(
    updates, sample_counts, trim_count: int = 1
) -> AggregationResult:
    start = time.perf_counter()
    clients = validate_inputs(updates, sample_counts)
    if trim_count < 0 or len(clients) - 2 * trim_count < 1:
        raise ValueError(
            f"trim_count={trim_count} leaves no retained values for {len(clients)} clients"
        )
    result = {}
    for key in updates[clients[0]]:
        values = (
            torch.stack([updates[client][key].float() for client in clients], dim=0)
            .sort(dim=0)
            .values
        )
        result[key] = (
            values[trim_count : len(clients) - trim_count]
            .mean(dim=0)
            .to(updates[clients[0]][key].dtype)
        )
    equal_weights = {client: 1.0 / len(clients) for client in clients}
    norms = {client: compute_update_l2_norm(updates[client]) for client in clients}
    return weighted_result(
        result,
        clients,
        equal_weights,
        norms,
        norms,
        {
            "method": "trimmed_mean",
            "trim_count": trim_count,
            "retained_clients": len(clients) - 2 * trim_count,
        },
        start,
    )
