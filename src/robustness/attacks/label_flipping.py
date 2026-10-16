"""Copy-on-write multi-label target flipping."""

from __future__ import annotations

import numpy as np

from src.robustness.attacks.base import AttackResult
from src.robustness.update_utils import (
    clone_update,
    compute_update_digest,
    compute_update_l2_norm,
)


def flip_labels(
    labels: np.ndarray, eligible_indices: tuple[int, ...] | list[int]
) -> tuple[np.ndarray, int]:
    result = np.asarray(labels).copy()
    changed = 0
    for index in eligible_indices:
        eligible = np.isin(result[:, index], [0, 1])
        changed += int(eligible.sum())
        result[eligible, index] = 1 - result[eligible, index]
    return result, changed


class LabelFlippingAttack:
    name = "label_flipping"

    def apply(
        self,
        client_id,
        clean_update,
        local_dataset,
        global_state,
        attack_config,
        random_generator,
    ):
        eligible = getattr(
            attack_config,
            "label_flip_indices",
            attack_config.get("label_flip_indices", ())
            if isinstance(attack_config, dict)
            else (),
        )
        changed = 0
        if local_dataset is not None:
            _, changed = flip_labels(local_dataset, eligible)
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
            {"eligible_label_indices": list(eligible)},
            changed,
            {"label_view_copied": local_dataset is not None},
            "valid",
        )
