"""Validated P4 update aggregators."""

from src.robustness.aggregators.fedavg import aggregate_fedavg
from src.robustness.aggregators.coordinate_median import aggregate_coordinate_median
from src.robustness.aggregators.trimmed_mean import aggregate_trimmed_mean
from src.robustness.aggregators.norm_clipped_fedavg import aggregate_norm_clipped_fedavg

__all__ = [
    "aggregate_fedavg",
    "aggregate_coordinate_median",
    "aggregate_trimmed_mean",
    "aggregate_norm_clipped_fedavg",
]
