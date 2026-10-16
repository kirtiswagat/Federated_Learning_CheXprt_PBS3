"""Patient-cluster bootstrap CIs and paired comparisons for R01-R05."""

import argparse
import os
from pathlib import Path

from src.evaluation.cluster_stats import run_statistics
from src.monitoring.artifact_registry import validate_manifest

RUNS = ("R01", "R02", "R03", "R04", "R05")
# R01 trains 10 rounds; round 5 is the like-for-like point against the 5-round runs.
COMPARISONS = [
    ("R01_r5", "R02"),
    ("R01_r5", "R03"),
    ("R02", "R03"),
    ("R01_r5", "R04"),
    ("R03", "R05"),
    ("R04", "R05"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--workers", type=int, default=max((os.cpu_count() or 2) - 2, 1)
    )
    args = parser.parse_args()
    runs = args.root / "Output" / "p1_critical_revision" / "runs"
    output = args.output or args.root / "Output" / "p1_critical_revision" / "statistics"
    print("[1/3] Validating run manifests", flush=True)
    for run in RUNS:
        validate_manifest(runs / run)
    paths = {
        "R01_r5": runs / "R01" / "predictions" / "round_05_valid_predictions.csv",
        "R01_r10": runs / "R01" / "endpoint_valid_predictions.csv",
        **{run: runs / run / "endpoint_valid_predictions.csv" for run in RUNS[1:]},
    }
    print(
        f"[2/3] Bootstrap: {args.iterations} iterations, {args.workers} workers",
        flush=True,
    )
    outputs = run_statistics(
        paths,
        output,
        COMPARISONS,
        iterations=args.iterations,
        seed=args.seed,
        workers=args.workers,
    )
    print("[3/3] Written:", flush=True)
    for name, path in outputs.items():
        print(f"  {name}: {path}", flush=True)


if __name__ == "__main__":
    main()
