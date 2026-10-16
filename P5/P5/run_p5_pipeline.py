"""Build the artifact-only P5 publication package."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.p5.config import P5Config
from src.p5.pipeline import P5Pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    P5Pipeline(P5Config(args.root, args.output)).run()


if __name__ == "__main__":
    main()
