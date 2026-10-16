"""Immutable P5 integration configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class P5Config:
    root: Path
    output_root: Path | None = None
    required_runs: tuple[str, ...] = tuple(f"R{index:02d}" for index in range(1, 22))
    dpi: int = 300
    bootstrap_iterations: int = 2000
    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        root = self.root.expanduser().resolve()
        object.__setattr__(self, "root", root)
        object.__setattr__(
            self,
            "output_root",
            self.output_root or root / "outputs" / "p5_publication_package",
        )
