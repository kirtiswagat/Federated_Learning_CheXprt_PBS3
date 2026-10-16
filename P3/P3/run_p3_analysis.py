"""Generate P3 sensitivity and contribution reports from completed runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.p3.analysis import export_p3_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export_p3_report(args.runs_root, args.output)


if __name__ == "__main__":
    main()
