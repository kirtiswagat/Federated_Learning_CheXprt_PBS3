import numpy as np
import torch

from src.robustness.attacks import (
    LabelFlippingAttack,
    SignFlippingAttack,
    ModelReplacementAttack,
    ByzantineRandomAttack,
    TargetedBackdoorAttack,
    apply_trigger,
)
from src.robustness.config import AttackConfig, BackdoorConfig


def test_label_flip_is_copy_on_write():
    labels = np.array([[0.0, 1.0, -1.0]])
    flipped, count = __import__(
        "src.robustness.attacks.label_flipping", fromlist=["flip_labels"]
    ).flip_labels(labels, [0, 1])
    assert (
        labels.tolist() == [[0.0, 1.0, -1.0]]
        and flipped.tolist() == [[1.0, 0.0, -1.0]]
        and count == 2
    )


def test_sign_flip_negates_update():
    update = {"w": torch.tensor([1.0, -2.0])}
    result = SignFlippingAttack().apply(
        "client_1", update, None, {}, AttackConfig(), torch.Generator()
    )
    assert torch.equal(result.submitted_update["w"], -update["w"])


def test_model_replacement_scale_is_fixed():
    result = ModelReplacementAttack().apply(
        "client_1",
        {"w": torch.ones(1)},
        None,
        {},
        AttackConfig(model_replacement_scale=3),
        torch.Generator(),
    )
    assert result.submitted_update["w"].item() == 3


def test_byzantine_is_seeded():
    update = {"w": torch.zeros(2)}
    first = ByzantineRandomAttack().apply(
        "client_1", update, None, {}, AttackConfig(), torch.Generator().manual_seed(7)
    )
    second = ByzantineRandomAttack().apply(
        "client_1", update, None, {}, AttackConfig(), torch.Generator().manual_seed(7)
    )
    assert torch.equal(first.submitted_update["w"], second.submitted_update["w"])


def test_trigger_is_deterministic_and_does_not_mutate_source():
    image = torch.zeros(1, 10, 10)
    triggered = apply_trigger(image, 2, (1, 1), 1.0)
    assert image.sum() == 0 and triggered.sum() == 4
    result = TargetedBackdoorAttack().apply(
        "client_1",
        {"w": torch.ones(1)},
        list(range(10)),
        {},
        BackdoorConfig(),
        torch.Generator(),
    )
    assert result.modified_sample_count == 2
