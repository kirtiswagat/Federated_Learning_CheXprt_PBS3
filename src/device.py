########----------------------------------------------------------------------#######
########----------src/device--------------------------------------------------#######
"""Explicit device policy shared by training and verification jobs."""

from __future__ import annotations

import torch


def resolve_device(
    prefer_cuda: bool = True, require_cuda: bool = False
) -> torch.device:
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this job, but CUDA is unavailable")
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_mps(require_mps: bool = True) -> torch.device:
    available = bool(
        getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    )
    if available:
        return torch.device("mps")
    if require_mps:
        raise RuntimeError("MPS is required for this job, but MPS is unavailable")
    return torch.device("cpu")


def validate_device(device: torch.device, require_cuda: bool = False) -> None:
    if require_cuda and device.type != "cuda":
        raise RuntimeError(f"CUDA-designated job resolved to {device.type}")



