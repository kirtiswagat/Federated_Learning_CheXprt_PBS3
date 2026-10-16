"""Run P1-01 dataset and historical-artifact audit."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import LABELS
from src.data.audit import audit_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path)
    parser.add_argument("--valid-csv", type=Path)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Print image verification progress every N images.",
    )
    args = parser.parse_args()
    revision_root = Path(__file__).resolve().parents[1]
    audit_dataset(
        args.train_csv or args.root / "train.csv",
        args.valid_csv or args.root / "valid.csv",
        args.root,
        revision_root / "Output" / "p1_critical_revision" / "audits",
        LABELS,
        args.progress_every,
    )


if __name__ == "__main__":
    main()
