"""Device-independent model update conversion, validation, and serialization."""

from __future__ import annotations

import hashlib
import io
from collections.abc import Mapping

import torch

TensorUpdate = dict[str, torch.Tensor]


def clone_update(update: Mapping[str, torch.Tensor]) -> TensorUpdate:
    return {key: value.detach().cpu().clone() for key, value in update.items()}


def state_dict_to_update(
    local_state: Mapping[str, torch.Tensor], global_state: Mapping[str, torch.Tensor]
) -> TensorUpdate:
    if set(local_state) != set(global_state):
        raise ValueError("local and global state dictionaries have different keys")
    update = {}
    for key in global_state:
        if local_state[key].shape != global_state[key].shape:
            raise ValueError(f"state shape mismatch for {key}")
        update[key] = (
            local_state[key].detach().cpu().clone()
            - global_state[key].detach().cpu().clone()
        )
    validate_update_schema(update, global_state)
    return update


def update_to_state_dict(
    global_state: Mapping[str, torch.Tensor], update: Mapping[str, torch.Tensor]
) -> TensorUpdate:
    validate_update_schema(update, global_state)
    return {
        key: global_state[key].detach().cpu().clone() + update[key].detach().cpu()
        for key in global_state
    }


def validate_update_schema(
    update: Mapping[str, torch.Tensor], template: Mapping[str, torch.Tensor]
) -> None:
    if set(update) != set(template):
        raise ValueError("update keys do not match model schema")
    for key, value in update.items():
        if value.shape != template[key].shape:
            raise ValueError(f"update shape mismatch for {key}")
        if value.dtype != template[key].dtype:
            raise ValueError(f"update dtype mismatch for {key}")
        if not torch.isfinite(value).all():
            raise ValueError(f"update contains NaN or infinity: {key}")


def flatten_update(
    update: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, list[tuple[str, tuple[int, ...], torch.dtype]]]:
    metadata = [(key, tuple(value.shape), value.dtype) for key, value in update.items()]
    flat = (
        torch.cat(
            [value.detach().cpu().reshape(-1).float() for value in update.values()]
        )
        if update
        else torch.empty(0)
    )
    return flat, metadata


def unflatten_update(
    flat: torch.Tensor, metadata: list[tuple[str, tuple[int, ...], torch.dtype]]
) -> TensorUpdate:
    result, offset = {}, 0
    for key, shape, dtype in metadata:
        size = 1
        for dimension in shape:
            size *= dimension
        result[key] = flat[offset : offset + size].reshape(shape).to(dtype).clone()
        offset += size
    if offset != flat.numel():
        raise ValueError("flat update has trailing values")
    return result


def is_trainable(key: str, value: torch.Tensor) -> bool:
    """Trainable parameters; BatchNorm running statistics and integer counters are not."""
    return value.is_floating_point() and not key.endswith(
        ("running_mean", "running_var")
    )


def compute_update_l2_norm(update: Mapping[str, torch.Tensor]) -> float:
    parts = [
        value.detach().cpu().reshape(-1).float()
        for key, value in update.items()
        if is_trainable(key, value)
    ]
    if not parts:
        return 0.0
    return float(torch.linalg.vector_norm(torch.cat(parts)).item())


def compute_update_cosine_similarity(
    left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]
) -> float:
    left_flat, _ = flatten_update(left)
    right_flat, _ = flatten_update(right)
    if left_flat.numel() != right_flat.numel():
        raise ValueError("cannot compare updates with different sizes")
    if (
        torch.linalg.vector_norm(left_flat) == 0
        or torch.linalg.vector_norm(right_flat) == 0
    ):
        return 0.0
    return float(
        torch.nn.functional.cosine_similarity(left_flat[None], right_flat[None]).item()
    )


def compute_update_digest(update: Mapping[str, torch.Tensor]) -> str:
    buffer = io.BytesIO()
    for key in sorted(update):
        value = update[key].detach().cpu().contiguous()
        buffer.write(key.encode())
        buffer.write(str(value.dtype).encode())
        buffer.write(repr(tuple(value.shape)).encode())
        buffer.write(value.numpy().tobytes())
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def check_finite_update(update: Mapping[str, torch.Tensor]) -> bool:
    return all(bool(torch.isfinite(value).all()) for value in update.values())
