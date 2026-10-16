"""Consolidated P4 matrices and publication-ready figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def export_p4_report(condition_results: pd.DataFrame, output_root: Path) -> dict:
    consolidated = output_root / "consolidated"
    figures = output_root / "figures"
    consolidated.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    condition_results.to_csv(consolidated / "robustness_attack_matrix.csv", index=False)
    if {"macro_auroc", "macro_auprc"} <= set(condition_results):
        condition_results.pivot_table(
            index=["attack_name", "malicious_fraction"],
            columns="aggregation_method",
            values="macro_auroc",
        ).to_csv(consolidated / "robustness_aggregator_comparison.csv")
        condition_results.pivot_table(
            index=["attack_name", "malicious_fraction"],
            columns="aggregation_method",
            values="macro_auroc",
        ).plot(kind="bar", figsize=(12, 6))
        plt.ylabel("Macro AUROC")
        plt.tight_layout()
        for suffix in ("png", "pdf"):
            plt.savefig(figures / f"attack_aggregator_macro_auroc.{suffix}")
        plt.close()
    if "attacked_norm" in condition_results:
        condition_results.plot(
            y="attacked_norm", kind="hist", bins=20, title="Submitted update norms"
        )
        plt.tight_layout()
        for suffix in ("png", "pdf"):
            plt.savefig(figures / f"update_norm_distribution.{suffix}")
        plt.close()
    report = {
        "condition_count": len(condition_results),
        "claims": [
            "simulated clients",
            "controlled one-round stress tests",
            "integrity and robustness are separate",
        ],
        "tables": [str(path) for path in consolidated.glob("*.csv")],
        "figures": [str(path) for path in figures.iterdir()],
    }
    (consolidated / "p4_final_summary_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report
