"""Deterministic sign-flipping update attack."""

from src.robustness.attacks.base import AttackResult
from src.robustness.update_utils import (
    clone_update,
    compute_update_digest,
    compute_update_l2_norm,
    is_trainable,
)


class SignFlippingAttack:
    name = "sign_flipping"

    def apply(
        self,
        client_id,
        clean_update,
        local_dataset,
        global_state,
        attack_config,
        random_generator,
    ):
        attacked = {
            key: -value if is_trainable(key, value) else value.clone()
            for key, value in clone_update(clean_update).items()
        }
        return AttackResult(
            self.name,
            str(client_id),
            True,
            attacked,
            compute_update_digest(clean_update),
            compute_update_digest(attacked),
            compute_update_l2_norm(clean_update),
            compute_update_l2_norm(attacked),
            {},
            0,
            {"cosine_to_original": -1.0},
            "valid",
        )
