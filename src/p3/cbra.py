"""CBRA contribution scoring and normalized model aggregation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import math

import numpy as np
import pandas as pd
import torch


@dataclass(frozen=True)
class Contribution:
    client_id: int
    sample_count: int
    rarity_score: float
    drift_score: float
    sample_component: float
    rarity_component: float
    drift_component: float
    raw_score: float
    normalized_weight: float


def l2_drift(
    local_state: Mapping[str, torch.Tensor], global_state: Mapping[str, torch.Tensor]
) -> float:
    difference_squared = []
    reference_squared = []
    for name, value in global_state.items():
        if name in local_state and torch.is_floating_point(value):
            reference = value.detach().cpu().float()
            difference_squared.append(
                torch.sum((local_state[name].detach().cpu().float() - reference) ** 2)
            )
            reference_squared.append(torch.sum(reference**2))
    if not difference_squared:
        return 0.0
    numerator = torch.sqrt(torch.stack(difference_squared).sum())
    denominator = torch.sqrt(torch.stack(reference_squared).sum()).clamp_min(1e-12)
    return float((numerator / denominator).item())


def client_rarity_score(client_labels: np.ndarray, global_labels: np.ndarray) -> float:
    """Reward information on globally uncommon positive labels.

    The score is positive and deterministic. It is the client's prevalence
    weighted by inverse global prevalence, normalized by the number of labels.
    """
    client = np.asarray(client_labels, dtype=float)
    global_prevalence = np.asarray(global_labels, dtype=float)
    if (
        client.ndim != 2
        or global_prevalence.ndim != 2
        or client.shape[1] != global_prevalence.shape[1]
    ):
        raise ValueError(
            "client_labels and global_labels must be two-dimensional with equal label width"
        )
    client_prevalence = np.nanmean(client, axis=0)
    global_prevalence = np.nanmean(global_prevalence, axis=0)
    inverse_rarity = 1.0 / np.clip(global_prevalence, 1e-6, None)
    score = np.nanmean(np.nan_to_num(client_prevalence) * inverse_rarity)
    return float(max(score, 1e-12))


def compute_contributions(
    states: Sequence[Mapping[str, torch.Tensor]],
    global_state: Mapping[str, torch.Tensor],
    sample_counts: Sequence[int],
    client_labels: Sequence[np.ndarray],
    global_labels: np.ndarray,
    alpha_n: float,
    beta_q: float,
    gamma_d: float,
) -> pd.DataFrame:
    if (
        not states
        or len(states) != len(sample_counts)
        or len(states) != len(client_labels)
    ):
        raise ValueError(
            "states, sample_counts, and client_labels must have equal non-zero length"
        )
    rows = []
    for client_id, (state, count, labels) in enumerate(
        zip(states, sample_counts, client_labels)
    ):
        if count <= 0:
            raise ValueError("sample counts must be positive")
        rarity = client_rarity_score(labels, global_labels)
        drift = l2_drift(state, global_state)
        sample_component = float(count) ** alpha_n
        rarity_component = rarity**beta_q
        drift_component = math.exp(-gamma_d * drift)
        raw = sample_component * rarity_component * drift_component
        rows.append(
            asdict(
                Contribution(
                    client_id,
                    int(count),
                    rarity,
                    drift,
                    sample_component,
                    rarity_component,
                    drift_component,
                    raw,
                    0.0,
                )
            )
        )
    log_raw = [
        alpha_n * math.log(row["sample_count"])
        + beta_q * math.log(row["rarity_score"])
        - gamma_d * row["drift_score"]
        for row in rows
    ]
    # Softmax in log space keeps relative weights stable when drift is large.
    peak = max(log_raw)
    if not math.isfinite(peak):
        raise ValueError("CBRA raw scores must have a positive finite total")
    scale = sum(math.exp(value - peak) for value in log_raw)
    for row, value in zip(rows, log_raw):
        row["normalized_weight"] = math.exp(value - peak) / scale
    result = pd.DataFrame(rows)
    if not np.isclose(result["normalized_weight"].sum(), 1.0):
        raise RuntimeError("CBRA normalized weights do not sum to one")
    return result


def cbra_aggregate(
    states: Sequence[Mapping[str, torch.Tensor]], contributions: pd.DataFrame
) -> dict[str, torch.Tensor]:
    if len(states) != len(contributions):
        raise ValueError("state count and contribution count differ")
    weights = contributions["normalized_weight"].to_numpy(float)
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("CBRA weights must sum to one")
    keys = set(states[0])
    if any(set(state) != keys for state in states):
        raise ValueError("client states have incompatible keys")
    return {
        key: sum(
            state[key].detach().cpu().float() * float(weight)
            for state, weight in zip(states, weights)
        )
        for key in keys
    }
