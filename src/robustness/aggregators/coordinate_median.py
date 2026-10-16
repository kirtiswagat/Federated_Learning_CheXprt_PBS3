"""Coordinate-wise median aggregator."""

from __future__ import annotations

import time

from src.robustness.aggregators.base import (
    AggregationResult,
    validate_inputs,
    weighted_result,
)
from src.robustness.update_utils import compute_update_l2_norm


def aggregate_coordinate_median(updates, sample_counts) -> AggregationResult:
    start = time.perf_counter()
    clients = validate_inputs(updates, sample_counts)
    result = {
        key: __import__("torch")
        .stack([updates[client][key].float() for client in clients])
        .median(dim=0)
        .values.to(updates[clients[0]][key].dtype)
        for key in updates[clients[0]]
    }
    equal_weights = {client: 1.0 / len(clients) for client in clients}
    norms = {client: compute_update_l2_norm(updates[client]) for client in clients}
    return weighted_result(
        result,
        clients,
        equal_weights,
        norms,
        norms,
        {"method": "coordinate_median"},
        start,
    )
