import numpy as np
import torch

from src.p3.cbra import compute_contributions, l2_drift


def _state(scale=1.0, delta=0.0):
    return {"weight": torch.full((100,), scale + delta)}


def test_drift_is_relative_to_global_model_scale():
    assert np.isclose(l2_drift(_state(1.0, 0.1), _state(1.0)), 0.1)
    assert np.isclose(l2_drift(_state(100.0, 10.0), _state(100.0)), 0.1)


def test_gamma_does_not_collapse_weights_for_comparable_updates():
    states = [_state(1.0, delta) for delta in (0.01, 0.02, 0.03, 0.04, 0.05)]
    labels = [np.ones((10, 14)) for _ in states]
    result = compute_contributions(
        states,
        _state(1.0),
        [100] * len(states),
        labels,
        np.ones((50, 14)),
        alpha_n=1.0,
        beta_q=1.0,
        gamma_d=0.5,
    )
    assert np.isclose(result.normalized_weight.sum(), 1.0)
    assert result.normalized_weight.max() < 0.25
