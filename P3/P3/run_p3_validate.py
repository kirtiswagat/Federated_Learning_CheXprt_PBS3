"""Validate a completed P3 artifact handoff."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.p3.artifacts import validate_p3_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    validate_p3_run(args.run_dir)
    print(f"valid P3 handoff: {args.run_dir}")


if __name__ == "__main__":
    main()
