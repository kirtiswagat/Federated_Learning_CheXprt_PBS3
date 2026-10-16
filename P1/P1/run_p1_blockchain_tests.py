"""Run ledger integrity checks and export their PASS/FAIL report."""

import argparse
from pathlib import Path

from src.blockchain.validation import run_validation_suite


def main() -> None:
    parser = argparse.ArgumentParser()
    revision_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--output",
        type=Path,
        default=revision_root
        / "Output"
        / "p1_critical_revision"
        / "blockchain"
        / "validation_report.csv",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run_validation_suite().to_csv(args.output, index=False)
    print(f"[P1-07] Validation report written to {args.output}", flush=True)


if __name__ == "__main__":
    main()
