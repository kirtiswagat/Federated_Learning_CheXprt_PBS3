"""CPU-first model state serialization and checksums."""

from __future__ import annotations

from pathlib import Path
import hashlib
import io

import torch

from src.reproducibility import load_checkpoint_cpu, save_checkpoint


def cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }


def save_model(model: torch.nn.Module, path: Path) -> str:
    state = cpu_state_dict(model)
    save_checkpoint(state, path)
    return checkpoint_checksum(path)


def load_model(
    model: torch.nn.Module, path: Path, device: torch.device
) -> torch.nn.Module:
    state = load_checkpoint_cpu(path)
    model.load_state_dict(state)
    return model.to(device)


def checkpoint_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
