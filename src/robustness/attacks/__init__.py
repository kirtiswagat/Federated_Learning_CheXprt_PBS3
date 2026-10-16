"""Controlled P4 attack implementations."""

from src.robustness.attacks.base import AttackResult
from src.robustness.attacks.label_flipping import LabelFlippingAttack, flip_labels
from src.robustness.attacks.sign_flipping import SignFlippingAttack
from src.robustness.attacks.model_replacement import ModelReplacementAttack
from src.robustness.attacks.targeted_backdoor import (
    TargetedBackdoorAttack,
    apply_trigger,
)
from src.robustness.attacks.byzantine_random import ByzantineRandomAttack

__all__ = [
    "AttackResult",
    "LabelFlippingAttack",
    "SignFlippingAttack",
    "ModelReplacementAttack",
    "TargetedBackdoorAttack",
    "ByzantineRandomAttack",
    "flip_labels",
    "apply_trigger",
]
