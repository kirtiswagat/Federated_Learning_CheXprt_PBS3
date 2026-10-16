"""Immutable P4 threat, attack, aggregation, and evaluation configuration."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RobustnessPaths:
    root: Path
    output_root: Path | None = None
    source_global_checkpoint: Path | None = None
    partition_manifest: Path | None = None

    def __post_init__(self) -> None:
        root = self.root.expanduser().resolve()
        object.__setattr__(self, "root", root)
        object.__setattr__(
            self, "output_root", self.output_root or root / "outputs" / "p4_robustness"
        )


@dataclass(frozen=True)
class ThreatModelConfig:
    total_clients: int = 5
    malicious_fractions: tuple[float, ...] = (0.2, 0.4)
    malicious_rounding: str = "ceil"
    client_selection_seed: int = 42
    attack_seed: int = 43

    def malicious_count(self, fraction: float) -> int:
        if fraction not in self.malicious_fractions:
            raise ValueError(f"Unregistered malicious fraction: {fraction}")
        if not 0 <= fraction <= 1:
            raise ValueError("malicious fraction must be between zero and one")
        if self.malicious_rounding == "ceil":
            import math

            count = math.ceil(self.total_clients * fraction)
        elif self.malicious_rounding == "floor":
            count = int(self.total_clients * fraction)
        elif self.malicious_rounding == "round":
            count = round(self.total_clients * fraction)
        else:
            raise ValueError(
                f"Unknown malicious rounding policy: {self.malicious_rounding}"
            )
        return min(self.total_clients, max(0, count))


@dataclass(frozen=True)
class AttackConfig:
    label_flip_indices: tuple[int, ...] = tuple(range(14))
    model_replacement_scale: float = 5.0
    byzantine_update_scale: float = 1.0


@dataclass(frozen=True)
class BackdoorConfig:
    target_label: int = 0
    trigger_type: str = "square"
    trigger_size: int = 8
    trigger_location: tuple[int, int] = (0, 0)
    trigger_intensity: float = 1.0
    poisoning_fraction: float = 0.2
    target_assignment: str = "positive"
    attack_success_threshold: float = 0.5


@dataclass(frozen=True)
class AggregatorConfig:
    methods: tuple[str, ...] = (
        "fedavg",
        "coordinate_median",
        "trimmed_mean",
        "norm_clipped_fedavg",
    )
    trimmed_mean_count: int = 1
    clipping_threshold: float | None = 1.0
    clipping_epsilon: float = 1e-12


@dataclass(frozen=True)
class RobustnessEvaluationConfig:
    threshold: float = 0.5
    bootstrap_iterations: int = 2000
    bootstrap_seed: int = 42
    blockchain_logging_enabled: bool = True


@dataclass(frozen=True)
class RobustnessCondition:
    condition_id: str
    attack_name: str
    malicious_fraction: float
    aggregation_method: str
    seed: int


@dataclass(frozen=True)
class P4Config:
    paths: RobustnessPaths
    threat: ThreatModelConfig = field(default_factory=ThreatModelConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    backdoor: BackdoorConfig = field(default_factory=BackdoorConfig)
    aggregation: AggregatorConfig = field(default_factory=AggregatorConfig)
    evaluation: RobustnessEvaluationConfig = field(
        default_factory=RobustnessEvaluationConfig
    )
    client_ids: tuple[str, ...] = tuple(f"client_{index}" for index in range(5))
    update_representation: str = "state_dict_delta_cpu"

    def __post_init__(self) -> None:
        if len(self.client_ids) != self.threat.total_clients:
            raise ValueError("client_ids must match total_clients")
        if not 0 < self.backdoor.poisoning_fraction <= 1:
            raise ValueError("backdoor poisoning fraction must be in (0, 1]")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["paths"] = {key: str(item) for key, item in value["paths"].items()}
        return value

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        import json

        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
