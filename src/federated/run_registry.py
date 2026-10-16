"""Configuration-driven registry for the prespecified P1 runs."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from src.config import ExperimentConfig, run_configs


class RunRegistry:
    def __init__(self, root: Path):
        self._runs: Mapping[str, ExperimentConfig] = run_configs(root)

    def get(self, run_id: str) -> ExperimentConfig:
        try:
            return self._runs[run_id]
        except KeyError as exc:
            raise KeyError(f"Unknown prespecified run: {run_id}") from exc

    def ids(self) -> tuple[str, ...]:
        return tuple(self._runs)
