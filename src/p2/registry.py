"""Immutable registry for the P2 prespecified experiments."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from src.config import ExperimentConfig, FederatedConfig, default_config


@dataclass(frozen=True)
class P2RunSpec:
    run_id: str
    algorithm: str
    partition: str
    clients: int
    rounds: int
    alpha: float | None = None
    participation_rate: float = 1.0
    mu: float = 0.0
    uncertainty_policy: str = "zero"
    uncertainty_mapping: tuple[tuple[str, int], ...] = ()

    @property
    def selected_clients(self) -> int:
        return max(1, round(self.clients * self.participation_rate))

    def as_federated_config(self) -> FederatedConfig:
        return FederatedConfig(
            clients=self.clients,
            rounds=self.rounds,
            partition=self.partition,
            dirichlet_alpha=self.alpha,
            method=self.algorithm,
            fedprox_mu=self.mu,
        )


def p2_specs() -> Mapping[str, P2RunSpec]:
    specs = (
        P2RunSpec("R06", "FedAvg", "iid", 10, 10),
        P2RunSpec("R07", "FedAvg", "iid", 20, 10),
        P2RunSpec("R08", "FedAvg", "iid", 20, 20, participation_rate=0.5),
        P2RunSpec("R09", "FedProx", "dirichlet", 5, 5, alpha=0.1, mu=0.0001),
        P2RunSpec("R10", "FedProx", "dirichlet", 5, 5, alpha=0.1, mu=0.01),
        P2RunSpec("R11", "FedProx", "dirichlet", 5, 5, alpha=0.1, mu=0.1),
        P2RunSpec(
            "R20", "FedAvg", "dirichlet", 5, 5, alpha=0.1, uncertainty_policy="ignore"
        ),
        P2RunSpec(
            "R21", "FedAvg", "dirichlet", 5, 5, alpha=0.1, uncertainty_policy="mapping"
        ),
    )
    return MappingProxyType({spec.run_id: spec for spec in specs})


class P2RunRegistry:
    def __init__(self, root: Path):
        self.root = root
        self._specs = p2_specs()

    def get(self, run_id: str) -> P2RunSpec:
        try:
            return self._specs[run_id]
        except KeyError as exc:
            raise KeyError(f"Unknown P2 run: {run_id}") from exc

    def config(self, run_id: str) -> ExperimentConfig:
        spec = self.get(run_id)
        return replace(
            default_config(self.root),
            federated=spec.as_federated_config(),
            dataset=replace(
                default_config(self.root).dataset,
                uncertain_policy=spec.uncertainty_policy,
            ),
        )

    def ids(self) -> tuple[str, ...]:
        return tuple(self._specs)
