"""Prespecified data-poisoning rules for the P4 label-flipping and backdoor attacks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import LABELS


def flip_all_labels(
    frame: pd.DataFrame, labels: tuple[str, ...] = LABELS
) -> pd.DataFrame:
    """Label flipping: every binary target of every sample is inverted (y -> 1 - y)."""
    poisoned = frame.copy()
    poisoned[list(labels)] = 1.0 - poisoned[list(labels)].to_numpy(dtype=np.float32)
    return poisoned


def backdoor_frame(
    frame: pd.DataFrame,
    target_label: int,
    fraction: float,
    seed: int,
    labels: tuple[str, ...] = LABELS,
) -> tuple[pd.DataFrame, set[int]]:
    """Backdoor: a fixed share of samples get the trigger and the target label set to 1."""
    count = int(round(len(frame) * fraction))
    rows = set(
        int(i)
        for i in np.random.default_rng(seed).choice(len(frame), count, replace=False)
    )
    poisoned = frame.reset_index(drop=True).copy()
    column = labels[target_label]
    positions = sorted(rows)
    poisoned.loc[positions, column] = 1.0
    return poisoned, rows


def triggered_evaluation_rows(
    frame: pd.DataFrame,
    target_label: int,
    max_images: int,
    seed: int,
    labels: tuple[str, ...] = LABELS,
) -> pd.DataFrame:
    """Validation images whose target label is 0, so a target prediction is not already correct."""
    candidates = frame[frame[labels[target_label]] == 0]
    if len(candidates) > max_images:
        candidates = candidates.sample(max_images, random_state=seed)
    return candidates.reset_index(drop=True)
