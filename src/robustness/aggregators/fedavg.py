"""Sample-count-weighted FedAvg over update deltas."""

from __future__ import annotations

import time

from src.robustness.aggregators.base import (
    AggregationResult,
    validate_inputs,
    weighted_result,
)
from src.robustness.update_utils import compute_update_l2_norm


def aggregate_fedavg(updates, sample_counts) -> AggregationResult:
    start = time.perf_counter()
    clients = validate_inputs(updates, sample_counts)
    total = sum(sample_counts.values())
    weights = {client: sample_counts[client] / total for client in clients}
    result = {
        key: sum(
            updates[client][key].float() * weights[client] for client in clients
        ).to(updates[clients[0]][key].dtype)
        for key in updates[clients[0]]
    }
    norms = {client: compute_update_l2_norm(updates[client]) for client in clients}
    return weighted_result(
        result, clients, weights, norms, norms, {"method": "fedavg"}, start
    )
