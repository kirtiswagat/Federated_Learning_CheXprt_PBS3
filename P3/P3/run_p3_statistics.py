"""Patient-cluster bootstrap of the corrected CBRA runs against the matched alpha=0.1 baselines."""

import argparse
import os
from pathlib import Path

from src.evaluation.cluster_stats import (
    bootstrap_draws,
    confidence_table,
    metric_names,
    paired_table,
)
from src.monitoring.artifact_registry import validate_manifest
from src.p3.artifacts import validate_p3_run

BASELINES = ("R03", "R05")  # FedAvg and FedProx, Dirichlet alpha=0.1, 5 rounds
CBRA = ("R12", "R13", "R14", "R15", "R16", "R18", "R19")  # R17 duplicates R13
FAMILIES = ["macro_auroc_14", "macro_auprc_14", "macro_auroc_5comp"]


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
    root = args.root.resolve()
    output = args.output or root / "Output" / "p3_analysis_normalized" / "statistics"
    output.mkdir(parents=True, exist_ok=True)

    print("[1/4] Validating run manifests", flush=True)
    baseline_dir = root / "Output" / "p1_critical_revision" / "runs"
    cbra_dir = root / "Output" / "p3_cbra"
    for run in BASELINES:
        validate_manifest(baseline_dir / run)
    for run in CBRA:
        validate_p3_run(cbra_dir / run)

    paths = {
        run: baseline_dir / run / "endpoint_valid_predictions.csv" for run in BASELINES
    }
    paths.update(
        {run: cbra_dir / run / "endpoint_valid_predictions.csv" for run in CBRA}
    )

    print(
        f"[2/4] Bootstrap: {args.iterations} iterations, {args.workers} workers",
        flush=True,
    )
    names, observed, draws = bootstrap_draws(
        paths, iterations=args.iterations, seed=args.seed, workers=args.workers
    )
    metrics = metric_names()

    print("[3/4] Paired comparisons", flush=True)
    vs_baselines = [(run, base) for run in CBRA for base in BASELINES]
    vs_reference = [(run, "R12") for run in CBRA if run != "R12"]
    confidence_table(names, metrics, observed, draws).to_csv(
        output / "metrics_with_ci.csv", index=False
    )
    paired_table(names, metrics, observed, draws, vs_baselines, FAMILIES).to_csv(
        output / "paired_cbra_vs_baselines.csv", index=False
    )
    paired_table(names, metrics, observed, draws, vs_reference, FAMILIES).to_csv(
        output / "paired_cbra_sensitivity_vs_R12.csv", index=False
    )

    print(f"[4/4] Written to {output}", flush=True)


if __name__ == "__main__":
    main()
