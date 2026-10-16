"""Shared attack result contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.robustness.update_utils import TensorUpdate


@dataclass(frozen=True)
class AttackResult:
    attack_name: str
    client_id: str
    malicious: bool
    submitted_update: TensorUpdate
    original_digest: str
    attacked_digest: str
    original_norm: float
    attacked_norm: float
    attack_parameters: dict[str, Any] = field(default_factory=dict)
    modified_sample_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    validation_status: str = "valid"
