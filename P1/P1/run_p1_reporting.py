"""Validate a completed P1 artifact handoff before reporting."""

import argparse
from pathlib import Path

from src.monitoring.artifact_registry import validate_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    validate_manifest(args.run_dir)
    print(f"valid artifact handoff: {args.run_dir}")


if __name__ == "__main__":
    main()
