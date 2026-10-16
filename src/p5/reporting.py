"""Paper tables, figures, supplementary workbook, and scope summaries."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _method_name(experiment: str) -> str:
    if experiment in {"R01", "R02", "R03", "R06", "R07", "R08"}:
        return "FedAvg"
    if experiment in {"R04", "R05", "R09", "R10", "R11"}:
        return "FedProx"
    if experiment in {"R12", "R13", "R14", "R15", "R16", "R17", "R18", "R19"}:
        return "CBRA"
    return experiment


def export_tables(
    master: pd.DataFrame, per_label: pd.DataFrame, ci: pd.DataFrame, output_dir: Path
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    table = master.copy()
    if not table.empty:
        table.insert(1, "Method", table.experiment.map(_method_name))
    table.to_csv(output_dir / "table_main_performance.csv", index=False)
    try:
        table.to_excel(output_dir / "table_main_performance.xlsx", index=False)
    except ImportError:
        (output_dir / "table_main_performance.xlsx.unavailable").write_text(
            "Install openpyxl to export XLSX.\n"
        )
    per_label.to_csv(output_dir / "table_per_label_metrics.csv", index=False)
    if not master.empty:
        for name, experiments in (
            ("scalability_summary", ["R01", "R06", "R07", "R08"]),
            ("fedprox_sensitivity_summary", ["R09", "R10", "R11"]),
            ("cbra_sensitivity_summary", [f"R{index:02d}" for index in range(12, 20)]),
            ("uncertainty_policy_summary", ["R03", "R20", "R21"]),
        ):
            master[master.experiment.isin(experiments)].to_csv(
                output_dir / f"{name}.csv", index=False
            )
    return {
        "main_rows": len(table),
        "per_label_rows": len(per_label),
        "ci_rows": len(ci),
    }


def export_figures(master: pd.DataFrame, output_dir: Path, dpi: int = 300) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    if master.empty:
        return paths
    metric = "macro_auroc" if "macro_auroc" in master else None
    if metric:
        frame = master.groupby("experiment", as_index=False)[metric].mean()
        frame.plot(
            x="experiment",
            y=metric,
            kind="bar",
            legend=False,
            title="Performance comparison",
        )
        plt.tight_layout()
        for suffix in ("png", "pdf", "svg"):
            path = output_dir / f"figure_1_performance.{suffix}"
            plt.savefig(path, dpi=dpi)
            paths.append(str(path))
        plt.close()
    for experiment in ("R01", "R06", "R07", "R08"):
        subset = master[master.experiment == experiment]
        if "round" in subset and metric:
            plt.plot(subset["round"], subset[metric], marker="o", label=experiment)
    if metric:
        plt.legend()
        plt.xlabel("Round")
        plt.ylabel("Macro AUROC")
        plt.tight_layout()
        for suffix in ("png", "pdf", "svg"):
            path = output_dir / f"figure_2_convergence.{suffix}"
            plt.savefig(path, dpi=dpi)
            paths.append(str(path))
        plt.close()
    return paths


def supplementary_workbook(tables_dir: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path) as writer:
        for path in sorted(tables_dir.glob("*.csv")):
            try:
                frame = pd.read_csv(path)
            except pd.errors.EmptyDataError:
                frame = pd.DataFrame({"status": ["no measured evidence available"]})
            frame.to_excel(writer, sheet_name=path.stem[:31], index=False)
