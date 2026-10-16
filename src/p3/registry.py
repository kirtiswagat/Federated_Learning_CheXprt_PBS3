"""Immutable P3 CBRA sensitivity registry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class P3RunSpec:
    run_id: str
    alpha_n: float
    beta_q: float
    gamma_d: float
    variant: str
    clients: int = 5
    rounds: int = 5
    dirichlet_alpha: float = 0.1


def p3_specs() -> Mapping[str, P3RunSpec]:
    specs = (
        P3RunSpec("R12", 1.0, 1.0, 0.5, "CBRA_v1"),
        P3RunSpec("R13", 1.0, 1.5, 0.5, "CBRA_v2"),
        P3RunSpec("R14", 0.5, 1.0, 0.5, "sample_lower"),
        P3RunSpec("R15", 1.5, 1.0, 0.5, "sample_higher"),
        P3RunSpec("R16", 1.0, 0.5, 0.5, "rarity_lower"),
        P3RunSpec("R17", 1.0, 1.5, 0.5, "rarity_higher"),
        P3RunSpec("R18", 1.0, 1.0, 0.0, "drift_disabled"),
        P3RunSpec("R19", 1.0, 1.0, 1.0, "drift_strong"),
    )
    return MappingProxyType({spec.run_id: spec for spec in specs})


class P3RunRegistry:
    def __init__(self, root: Path):
        self.root = root
        self._specs = p3_specs()

    def get(self, run_id: str) -> P3RunSpec:
        try:
            return self._specs[run_id]
        except KeyError as exc:
            raise KeyError(f"Unknown P3 run: {run_id}") from exc

    def ids(self) -> tuple[str, ...]:
        return tuple(self._specs)
