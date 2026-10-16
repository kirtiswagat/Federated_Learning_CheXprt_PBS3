import numpy as np
import torch

from src.p3.cbra import cbra_aggregate, compute_contributions


def test_cbra_weights_sum_to_one_and_aggregate():
    global_state = {"weight": torch.zeros(2)}
    states = [{"weight": torch.ones(2)}, {"weight": torch.full((2,), 2.0)}]
    labels = [
        np.array([[1, 0], [1, 0.0]], dtype=float),
        np.array([[0, 1], [0, 1.0]], dtype=float),
    ]
    details = compute_contributions(
        states, global_state, [2, 2], labels, np.vstack(labels), 1.0, 1.0, 0.5
    )
    assert np.isclose(details.normalized_weight.sum(), 1.0)
    aggregate = cbra_aggregate(states, details)
    assert aggregate["weight"].shape == (2,)


def test_zero_drift_has_no_drift_suppression():
    state = {"weight": torch.ones(1)}
    labels = [np.array([[1.0]], dtype=float)]
    details = compute_contributions(
        [state], {"weight": torch.zeros(1)}, [1], labels, labels[0], 1, 1, 0
    )
    assert details.iloc[0].drift_component == 1.0


def test_large_drift_does_not_underflow_weights():
    labels = [np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]])]
    states = [{"w": torch.full((4,), 1000.0)}, {"w": torch.full((4,), 1001.0)}]
    details = compute_contributions(
        states, {"w": torch.zeros(4)}, [10, 10], labels, np.vstack(labels), 1, 1, 1
    )
    assert details.raw_score.eq(0).all()
    assert np.isclose(details.normalized_weight.sum(), 1.0)
    assert details.normalized_weight.iloc[0] > details.normalized_weight.iloc[1]
