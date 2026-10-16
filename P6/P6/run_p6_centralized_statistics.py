"""Bootstrap CIs and paired comparisons for the centralized reference C01 (epoch 5)."""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.cluster_stats import run_statistics  # noqa: E402

# Centralized (5 epochs) against the like-for-like 5-round federated runs.
COMPARISONS = [("C01", "R01_r5"), ("C01", "R03"), ("C01", "R05")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--workers", type=int, default=max((os.cpu_count() or 2) - 2, 1)
    )
    args = parser.parse_args()
    out = ROOT / "Output"
    runs = out / "p1_critical_revision" / "runs"
    paths = {
        "C01": out / "p6_centralized" / "C01" / "endpoint_valid_predictions.csv",
        "R01_r5": runs / "R01" / "predictions" / "round_05_valid_predictions.csv",
        "R03": runs / "R03" / "endpoint_valid_predictions.csv",
        "R05": runs / "R05" / "endpoint_valid_predictions.csv",
    }
    outputs = run_statistics(
        paths,
        out / "p6_centralized" / "statistics",
        COMPARISONS,
        iterations=args.iterations,
        seed=args.seed,
        workers=args.workers,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}", flush=True)


if __name__ == "__main__":
    main()
