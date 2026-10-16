"""Run P5 discovery and completeness audit only."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.p5.config import P5Config
from src.p5.discovery import audit_completeness, discover_artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = P5Config(args.root, args.output)
    inventory = discover_artifacts(config.root, config.output_root / "planning")
    audit_completeness(inventory, config, config.output_root / "planning")


if __name__ == "__main__":
    main()
