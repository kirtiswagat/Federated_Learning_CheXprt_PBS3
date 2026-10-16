import torch

from src.robustness.aggregators import aggregate_norm_clipped_fedavg
from src.robustness.attacks import (
    ByzantineRandomAttack,
    ModelReplacementAttack,
    SignFlippingAttack,
)
from src.robustness.config import AttackConfig
from src.robustness.update_utils import compute_update_l2_norm, is_trainable


def _update(counter=4000):
    return {
        "layer.weight": torch.full((4,), 3.0),
        "bn.weight": torch.full((2,), 4.0),
        "bn.running_mean": torch.full((2,), 7.0),
        "bn.running_var": torch.full((2,), 9.0),
        "bn.num_batches_tracked": torch.tensor(counter, dtype=torch.int64),
    }


def test_norm_ignores_batchnorm_statistics_and_counters():
    small = compute_update_l2_norm(_update(counter=1))
    large = compute_update_l2_norm(_update(counter=1_000_000))
    assert small == large
    assert abs(small - (4 * 9 + 2 * 16) ** 0.5) < 1e-5
    assert not is_trainable("bn.running_var", torch.ones(2))
    assert not is_trainable("bn.num_batches_tracked", torch.tensor(1))


def test_attacks_never_modify_batchnorm_statistics():
    update = _update()
    for attack in (SignFlippingAttack, ModelReplacementAttack, ByzantineRandomAttack):
        submitted = (
            attack()
            .apply(
                "client_0",
                update,
                None,
                {},
                AttackConfig(),
                torch.Generator().manual_seed(1),
            )
            .submitted_update
        )
        for key in ("bn.running_mean", "bn.running_var", "bn.num_batches_tracked"):
            assert torch.equal(submitted[key], update[key])
        assert not torch.equal(submitted["layer.weight"], update["layer.weight"])


def test_clipping_scales_parameters_and_leaves_statistics_alone():
    clients = {f"client_{i}": _update() for i in range(3)}
    clients["client_2"]["layer.weight"] = torch.full((4,), 300.0)
    counts = {client: 100 for client in clients}
    result = aggregate_norm_clipped_fedavg(clients, counts, None)
    assert torch.allclose(result.update["bn.running_var"], torch.full((2,), 9.0))
    assert result.update["layer.weight"].abs().max().item() < 10.0
