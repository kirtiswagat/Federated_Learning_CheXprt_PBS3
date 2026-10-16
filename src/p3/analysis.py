"""P3 sensitivity, contribution, baseline comparison, and interpretation reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


def load_p3_run(root: Path, run_id: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    run = root / run_id
    history = pd.read_csv(run / "round_history.csv")
    contributions = pd.read_csv(run / "cbra_round_details.csv")
    result_path = run / f"{run_id.lower()}_results.json"
    result = json.loads(result_path.read_text()) if result_path.is_file() else {}
    return history, contributions, result


def parameter_sensitivity(
    root: Path,
    run_ids: Iterable[str] = ("R12", "R13", "R14", "R15", "R16", "R17", "R18", "R19"),
) -> pd.DataFrame:
    rows = []
    for run_id in run_ids:
        history, contributions, result = load_p3_run(root, run_id)
        endpoint = history.iloc[-1]
        rows.append(
            {
                "run_id": run_id,
                "variant": result.get("variant"),
                "alpha_n": result.get("alpha_n"),
                "beta_q": result.get("beta_q"),
                "gamma_d": result.get("gamma_d"),
                "macro_auroc": endpoint.get("macro_auroc"),
                "macro_auprc": endpoint.get("macro_auprc"),
                "mean_drift": contributions.drift_score.mean(),
                "mean_weight": contributions.normalized_weight.mean(),
                "weight_std": contributions.normalized_weight.std(),
            }
        )
    return pd.DataFrame(rows)


def contribution_summary(
    root: Path,
    run_ids: Iterable[str] = ("R12", "R13", "R14", "R15", "R16", "R17", "R18", "R19"),
) -> pd.DataFrame:
    frames = []
    for run_id in run_ids:
        _, details, _ = load_p3_run(root, run_id)
        summary = details.groupby("client_id", as_index=False).agg(
            sample_count=("sample_count", "first"),
            rarity_score=("rarity_score", "mean"),
            drift_score=("drift_score", "mean"),
            normalized_weight=("normalized_weight", "mean"),
        )
        summary.insert(0, "run_id", run_id)
        frames.append(summary)
    return pd.concat(frames, ignore_index=True)


def compare_baselines(
    cbra: pd.DataFrame, fedavg: pd.DataFrame, fedprox: pd.DataFrame, baseline_name: str
) -> pd.DataFrame:
    required = {"macro_auroc", "macro_auprc"}
    if (
        not required <= set(cbra)
        or not required <= set(fedavg)
        or not required <= set(fedprox)
    ):
        raise ValueError("CBRA and baseline tables need macro_auroc and macro_auprc")
    rows = []
    for name, frame in (("CBRA", cbra), ("FedAvg", fedavg), ("FedProx", fedprox)):
        rows.append(
            {
                "method": name,
                "baseline": baseline_name,
                "macro_auroc": frame.macro_auroc.mean(),
                "macro_auprc": frame.macro_auprc.mean(),
            }
        )
    result = pd.DataFrame(rows)
    result["evidence_status"] = "tested heuristic"
    return result


def export_p3_report(
    root: Path,
    output_dir: Path,
    fedavg: pd.DataFrame | None = None,
    fedprox: pd.DataFrame | None = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    sensitivity = parameter_sensitivity(root)
    contributions = contribution_summary(root)
    sensitivity.to_csv(output_dir / "cbra_parameter_sensitivity.csv", index=False)
    contributions.to_csv(output_dir / "cbra_contribution_summary.csv", index=False)
    report = {"runs": list(sensitivity.run_id), "interpretation": "tested heuristic"}
    if fedavg is not None and fedprox is not None:
        comparison = compare_baselines(sensitivity, fedavg, fedprox, "matched baseline")
        comparison.to_csv(output_dir / "paper_ready_tables.csv", index=False)
        comparison.to_csv(output_dir / "cbra_vs_fedavg.csv", index=False)
        comparison.to_csv(output_dir / "cbra_vs_fedprox.csv", index=False)
        report["baseline_comparison"] = comparison.to_dict(orient="records")
    for parameter, title in (
        ("alpha_n", "alpha_n sensitivity"),
        ("beta_q", "beta_q sensitivity"),
        ("gamma_d", "gamma_d sensitivity"),
    ):
        sensitivity.plot(x=parameter, y="macro_auroc", kind="scatter", title=title)
        for suffix in ("png", "pdf"):
            plt.savefig(
                output_dir / f"{parameter}_sensitivity.{suffix}", bbox_inches="tight"
            )
        plt.close()
    contributions.boxplot(column="normalized_weight", by="run_id", rot=45)
    plt.suptitle("")
    plt.tight_layout()
    for suffix in ("png", "pdf"):
        plt.savefig(
            output_dir / f"cbra_weight_distributions.{suffix}", bbox_inches="tight"
        )
    plt.close()
    (output_dir / "p3_final_report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n"
    )
    return report
