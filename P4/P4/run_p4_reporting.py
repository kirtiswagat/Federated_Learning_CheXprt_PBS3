"""Export consolidated P4 tables and figures from measured condition results."""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from src.robustness.evaluation import calculate_degradation
from src.robustness.reporting import export_p4_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("condition_results", type=Path)
    parser.add_argument("--clean-reference", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    attacked = pd.read_csv(args.condition_results)
    export_p4_report(attacked, args.output_root)
    if args.clean_reference:
        clean = pd.read_csv(args.clean_reference)
        degradation = calculate_degradation(attacked, clean)
        output = args.output_root / "consolidated"
        output.mkdir(parents=True, exist_ok=True)
        degradation.to_csv(output / "robustness_degradation.csv", index=False)
        print(f"Wrote degradation table to {output / 'robustness_degradation.csv'}")


if __name__ == "__main__":
    main()
