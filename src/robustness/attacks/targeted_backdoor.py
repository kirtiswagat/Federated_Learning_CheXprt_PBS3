"""Deterministic synthetic trigger and copied-sample backdoor metadata."""

from __future__ import annotations

import numpy as np
import torch

from src.robustness.attacks.base import AttackResult
from src.robustness.update_utils import (
    clone_update,
    compute_update_digest,
    compute_update_l2_norm,
)


def apply_trigger(
    image, size: int = 8, location: tuple[int, int] = (0, 0), intensity: float = 1.0
):
    result = (
        image.clone() if isinstance(image, torch.Tensor) else np.array(image, copy=True)
    )
    row, column = location
    if isinstance(result, torch.Tensor):
        result[..., row : row + size, column : column + size] = intensity
    else:
        result[row : row + size, column : column + size, ...] = intensity
    return result


class TargetedBackdoorAttack:
    name = "targeted_backdoor"

    def apply(
        self,
        client_id,
        clean_update,
        local_dataset,
        global_state,
        attack_config,
        random_generator,
    ):
        config = attack_config
        get = lambda name, default: getattr(
            config,
            name,
            config.get(name, default) if isinstance(config, dict) else default,
        )
        metadata = {
            "target_label": get("target_label", 0),
            "trigger_type": get("trigger_type", "square"),
            "trigger_size": get("trigger_size", 8),
            "trigger_location": list(get("trigger_location", (0, 0))),
            "trigger_intensity": get("trigger_intensity", 1.0),
            "poisoning_fraction": get("poisoning_fraction", 0.2),
        }
        sample_count = (
            int(len(local_dataset) * metadata["poisoning_fraction"])
            if local_dataset is not None
            else 0
        )
        attacked = clone_update(clean_update)
        return AttackResult(
            self.name,
            str(client_id),
            True,
            attacked,
            compute_update_digest(clean_update),
            compute_update_digest(attacked),
            compute_update_l2_norm(clean_update),
            compute_update_l2_norm(attacked),
            metadata,
            sample_count,
            {"trigger_is_copy_on_write": True},
            "valid",
        )
