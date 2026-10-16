"""Fixed-scale model-replacement update attack."""

from src.robustness.attacks.base import AttackResult
from src.robustness.update_utils import (
    clone_update,
    compute_update_digest,
    compute_update_l2_norm,
    is_trainable,
)


class ModelReplacementAttack:
    name = "model_replacement"

    def apply(
        self,
        client_id,
        clean_update,
        local_dataset,
        global_state,
        attack_config,
        random_generator,
    ):
        scale = float(
            getattr(
                attack_config,
                "model_replacement_scale",
                attack_config.get("model_replacement_scale", 5.0)
                if isinstance(attack_config, dict)
                else 5.0,
            )
        )
        attacked = {
            key: value * scale if is_trainable(key, value) else value.clone()
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
            {"replacement_scale": scale},
            0,
            {},
            "valid",
        )
