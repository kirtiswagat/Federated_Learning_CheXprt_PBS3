"""Generate P2 paper-ready tables, figures, and final report."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.p2.analysis import export_p2_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export_p2_report(args.runs_root, args.output)


if __name__ == "__main__":
    main()
