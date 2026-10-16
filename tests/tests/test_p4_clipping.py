import torch

from src.robustness.aggregators import aggregate_norm_clipped_fedavg


def _updates():
    # Four honest clients with norm 10 and one attacker with norm 1000.
    clients = {f"client_{i}": {"w": torch.full((4,), 5.0)} for i in range(4)}
    clients["client_4"] = {"w": torch.full((4,), 500.0)}
    return clients, {client: 100 for client in clients}


def test_median_threshold_limits_the_attacker_but_keeps_honest_updates():
    updates, counts = _updates()
    result = aggregate_norm_clipped_fedavg(updates, counts, None)
    assert result.parameters["clipping_threshold"] == 10.0
    assert abs(result.update["w"][0].item() - 5.0) < 1e-4


def test_fixed_threshold_far_below_the_update_norms_erases_learning():
    updates, counts = _updates()
    result = aggregate_norm_clipped_fedavg(updates, counts, 1.0)
    assert result.update["w"].abs().max().item() < 1.0
