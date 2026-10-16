import torch

from src.robustness.attacks import (
    ByzantineRandomAttack,
    ModelReplacementAttack,
    SignFlippingAttack,
)
from src.robustness.config import AttackConfig
from src.robustness.update_utils import validate_update_schema


def _update():
    # Real ResNet updates carry an int64 BatchNorm counter next to float weights.
    return {
        "layer.weight": torch.ones(3),
        "bn.num_batches_tracked": torch.tensor(4, dtype=torch.int64),
    }


def test_attacks_keep_the_model_schema_with_integer_buffers():
    update = _update()
    for attack in (SignFlippingAttack, ModelReplacementAttack, ByzantineRandomAttack):
        result = attack().apply(
            "client_0",
            update,
            None,
            {},
            AttackConfig(),
            torch.Generator().manual_seed(1),
        )
        validate_update_schema(result.submitted_update, update)
        assert result.submitted_update["bn.num_batches_tracked"].item() == 4
