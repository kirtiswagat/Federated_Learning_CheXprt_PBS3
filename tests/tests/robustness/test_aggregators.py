import torch
import pytest

from src.robustness.aggregators import (
    aggregate_fedavg,
    aggregate_coordinate_median,
    aggregate_trimmed_mean,
    aggregate_norm_clipped_fedavg,
)


def updates():
    return {str(i): {"w": torch.tensor([float(i)])} for i in range(5)}


def test_aggregators_known_values():
    values = updates()
    sizes = {key: 1 for key in values}
    assert aggregate_fedavg(values, sizes).update["w"].item() == 2
    assert aggregate_coordinate_median(values, sizes).update["w"].item() == 2
    assert aggregate_trimmed_mean(values, sizes, 1).update["w"].item() == 2


def test_invalid_trim_rejected():
    with pytest.raises(ValueError):
        aggregate_trimmed_mean(updates(), {key: 1 for key in updates()}, 3)


def test_clipping_never_increases_norm():
    result = aggregate_norm_clipped_fedavg(
        updates(), {key: 1 for key in updates()}, 1.0
    )
    assert all(
        item["clipped_norm"] <= item["original_norm"] + 1e-8 for item in result.details
    )
