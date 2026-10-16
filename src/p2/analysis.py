"""Post-experiment P2 tables, budgets, comparisons, and paper-ready figures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


def _load_run(root: Path, run_id: str) -> tuple[pd.DataFrame, dict]:
    run_dir = root / run_id
    history = pd.read_csv(run_dir / "round_history.csv")
    summary_path = run_dir / f"{run_id.lower()}_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    history["run_id"] = run_id
    return history, summary


def build_scalability_summary(
    root: Path, run_ids: Iterable[str] = ("R01", "R06", "R07")
) -> pd.DataFrame:
    rows = []
    for run_id in run_ids:
        history, summary = _load_run(root, run_id)
        endpoint = history.iloc[-1]
        rows.append(
            {
                "run_id": run_id,
                "clients": summary.get("clients"),
                "rounds": summary.get("rounds"),
                "macro_auroc": endpoint.get("macro_auroc", endpoint.get("macro_auroc")),
                "macro_auprc": endpoint.get("macro_auprc"),
                "runtime_seconds": history["runtime_seconds"].sum(),
                "processed_examples": history["processed_examples"].sum(),
                "communication_bytes": history["communication_bytes"].sum(),
            }
        )
    return pd.DataFrame(rows)


def build_fedprox_sensitivity(
    root: Path,
    run_ids: Iterable[str] = ("R09", "R10", "R11"),
    baseline: dict | None = None,
) -> pd.DataFrame:
    rows = []
    baseline = baseline or {}
    for run_id in run_ids:
        history, summary = _load_run(root, run_id)
        endpoint = history.iloc[-1]
        rows.append(
            {
                "run_id": run_id,
                "mu": summary.get("mu"),
                "macro_auroc": endpoint.get("macro_auroc"),
                "macro_auprc": endpoint.get("macro_auprc"),
                "runtime_seconds": history.runtime_seconds.sum(),
                "communication_bytes": history.communication_bytes.sum(),
            }
        )
    if baseline:
        rows.append(baseline)
    return pd.DataFrame(rows).sort_values("mu")


def build_uncertainty_summary(
    root: Path, run_ids: Iterable[str] = ("R20", "R21"), baseline: dict | None = None
) -> pd.DataFrame:
    rows = []
    for run_id in run_ids:
        history, summary = _load_run(root, run_id)
        endpoint = history.iloc[-1]
        rows.append(
            {
                "run_id": run_id,
                "policy": summary.get("uncertainty_policy"),
                "macro_auroc": endpoint.get("macro_auroc"),
                "macro_auprc": endpoint.get("macro_auprc"),
            }
        )
    if baseline:
        rows.append(baseline)
    return pd.DataFrame(rows)


def build_budget_analysis(root: Path, run_ids: Iterable[str]) -> pd.DataFrame:
    frames = []
    for run_id in run_ids:
        history, _ = _load_run(root, run_id)
        history["cumulative_processed_examples"] = history.processed_examples.cumsum()
        history["cumulative_communication_bytes"] = history.communication_bytes.cumsum()
        frames.append(history)
    return pd.concat(frames, ignore_index=True)


def export_p2_report(
    root: Path, output_dir: Path, include_r01: Path | None = None
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    source = root
    scalability = build_scalability_summary(source)
    fedprox = build_fedprox_sensitivity(source)
    uncertainty = build_uncertainty_summary(source)
    budget = build_budget_analysis(
        source, ("R06", "R07", "R08", "R09", "R10", "R11", "R20", "R21")
    )
    scalability.to_csv(output_dir / "scalability_summary.csv", index=False)
    fedprox.to_csv(output_dir / "fedprox_sensitivity_summary.csv", index=False)
    uncertainty.to_csv(output_dir / "uncertainty_policy_summary.csv", index=False)
    budget.to_csv(output_dir / "processed_budget_analysis.csv", index=False)

    figures = []
    for run_id in ("R06", "R07", "R08"):
        history, _ = _load_run(source, run_id)
        plt.plot(history["round"], history["macro_auroc"], marker="o", label=run_id)
    plt.xlabel("Communication round")
    plt.ylabel("Macro AUROC")
    plt.legend()
    plt.tight_layout()
    for suffix in ("png", "pdf"):
        path = output_dir / f"scalability_convergence.{suffix}"
        plt.savefig(path)
        figures.append(str(path))
    plt.close()
    for frame, name, x, y in (
        (fedprox, "fedprox_mu_auroc", "mu", "macro_auroc"),
        (fedprox, "fedprox_mu_auprc", "mu", "macro_auprc"),
        (uncertainty, "uncertainty_policy_comparison", "policy", "macro_auroc"),
    ):
        frame.plot(x=x, y=y, kind="bar", legend=False)
        plt.tight_layout()
        for suffix in ("png", "pdf"):
            path = output_dir / f"{name}.{suffix}"
            plt.savefig(path)
            figures.append(str(path))
        plt.close()
    report = {
        "runs": ["R06", "R07", "R08", "R09", "R10", "R11", "R20", "R21"],
        "tables": [str(path) for path in output_dir.glob("*.csv")],
        "figures": figures,
    }
    (output_dir / "p2_final_summary_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report
