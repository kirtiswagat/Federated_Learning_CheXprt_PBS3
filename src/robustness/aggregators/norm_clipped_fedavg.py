"""Prespecified norm-clipped FedAvg."""

from __future__ import annotations

import statistics
import time

from src.robustness.aggregators.base import (
    AggregationResult,
    validate_inputs,
    weighted_result,
)
from src.robustness.update_utils import (
    clone_update,
    compute_update_l2_norm,
    is_trainable,
)


def aggregate_norm_clipped_fedavg(
    updates,
    sample_counts,
    clipping_threshold: float | None = 1.0,
    epsilon: float = 1e-12,
) -> AggregationResult:
    """clipping_threshold=None clips to the median norm of the submitted updates."""
    start = time.perf_counter()
    clients = validate_inputs(updates, sample_counts)
    total = sum(sample_counts.values())
    weights = {client: sample_counts[client] / total for client in clients}
    original = {client: compute_update_l2_norm(updates[client]) for client in clients}
    if clipping_threshold is None:
        clipping_threshold = statistics.median(original.values())
    if clipping_threshold <= 0:
        raise ValueError("clipping threshold must be positive")
    clipped, details = {}, []
    for client in clients:
        factor = min(1.0, clipping_threshold / max(original[client], epsilon))
        clipped[client] = {
            key: value * factor if is_trainable(key, value) else value
            for key, value in updates[client].items()
        }
        details.append(
            {
                "client_id": client,
                "original_norm": original[client],
                "clipping_threshold": clipping_threshold,
                "clipping_factor": factor,
                "clipped_norm": compute_update_l2_norm(clipped[client]),
                "clipping_applied": factor < 1.0,
            }
        )
    result = {
        key: sum(
            clipped[client][key].float() * weights[client] for client in clients
        ).to(updates[clients[0]][key].dtype)
        for key in updates[clients[0]]
    }
    transformed = {
        client: compute_update_l2_norm(clipped[client]) for client in clients
    }
    return weighted_result(
        result,
        clients,
        weights,
        original,
        transformed,
        {"method": "norm_clipped_fedavg", "clipping_threshold": clipping_threshold},
        start,
        details,
    )
