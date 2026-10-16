"""Small adapters for invoking existing local-client training code."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch


def train_client(
    model: torch.nn.Module,
    loader: Any,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 1,
    global_state: dict[str, torch.Tensor] | None = None,
    proximal_mu: float = 0.0,
    on_batch: Callable[[int, float, int], None] | None = None,
    start_batch: int = 0,
    start_loss: float = 0.0,
    checkpoint_every: int = 0,
    on_checkpoint: Callable[[int, float], None] | None = None,
) -> tuple[dict[str, torch.Tensor], float, int]:
    model.train()
    total_loss = start_loss
    batches = 0
    reference = {key: value.to(device) for key, value in (global_state or {}).items()}
    for _ in range(epochs):
        for images, targets in loader:
            batches += 1
            if batches <= start_batch:
                continue  # replaying an interrupted client up to its checkpoint
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), targets)
            if proximal_mu and reference:
                loss = loss + proximal_mu / 2 * sum(
                    (parameter - reference[name]).pow(2).sum()
                    for name, parameter in model.named_parameters()
                    if name in reference
                )
            loss.backward()
            optimizer.step()
            batch_loss = float(loss.detach().cpu())
            total_loss += batch_loss
            if on_batch:
                on_batch(batches, batch_loss, len(images))
            if on_checkpoint and checkpoint_every and batches % checkpoint_every == 0:
                on_checkpoint(batches, total_loss)
    state = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    return state, total_loss / max(batches, 1), batches
