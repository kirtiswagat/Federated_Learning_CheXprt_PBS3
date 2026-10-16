"""Model-state aggregation shared by FedAvg and FedProx."""

from __future__ import annotations

from collections.abc import Mapping
import torch


def fedavg(
    states: list[Mapping[str, torch.Tensor]], sizes: list[int]
) -> dict[str, torch.Tensor]:
    if not states or len(states) != len(sizes) or any(size <= 0 for size in sizes):
        raise ValueError("states and positive client sizes are required")
    total = sum(sizes)
    keys = set(states[0])
    if any(set(state) != keys for state in states):
        raise ValueError("client state dictionaries have incompatible keys")
    return {
        key: sum(
            state[key].detach().cpu().float() * (size / total)
            for state, size in zip(states, sizes)
        )
        for key in keys
    }
