"""Entry-point contract for generating the clean one-round update reference."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.robustness.config import P4Config, RobustnessPaths
from src.robustness.registry import write_condition_registry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    revision_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--output-root",
        type=Path,
        default=revision_root / "Output" / "p4_robustness",
    )
    args = parser.parse_args()
    print("[P4] Loading robustness configuration", flush=True)
    config = P4Config(RobustnessPaths(args.root, output_root=args.output_root))
    print(f"[P4] Preparing output directory: {config.paths.output_root}", flush=True)
    write_condition_registry(
        config,
        config.paths.output_root / "config" / "robustness_condition_registry.csv",
    )
    config.save(config.paths.output_root / "config" / "p4_config.json")
    print("[P4] Clean-reference configuration written", flush=True)
    print(
        "[P4] Clean-reference adapter initialized; inject existing model/data training functions to execute.",
        flush=True,
    )


if __name__ == "__main__":
    main()
