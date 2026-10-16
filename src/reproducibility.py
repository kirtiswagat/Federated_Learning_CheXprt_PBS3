#########----------src/Reproducibility-----------------------------------------#########
"""Deterministic setup and portable checkpoint helpers."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def save_checkpoint(state: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint_cpu(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def move_state_to_device(state: Any, device: torch.device) -> Any:
    if isinstance(state, torch.Tensor):
        return state.to(device)
    if isinstance(state, dict):
        return {
            key: move_state_to_device(value, device) for key, value in state.items()
        }
    if isinstance(state, list):
        return [move_state_to_device(value, device) for value in state]
    if isinstance(state, tuple):
        return tuple(move_state_to_device(value, device) for value in state)
    return state
