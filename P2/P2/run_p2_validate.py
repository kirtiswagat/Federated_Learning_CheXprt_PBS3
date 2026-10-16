"""Validate a completed P2 artifact handoff."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.p2.artifacts import validate_p2_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    validate_p2_run(args.run_dir)
    print(f"valid P2 handoff: {args.run_dir}")


if __name__ == "__main__":
    main()
