"""End-to-end P5 artifact-only integration pipeline."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from src.p5.claims import reviewer_response_matrix, validate_claims
from src.p5.config import P5Config
from src.p5.discovery import audit_completeness, discover_artifacts
from src.p5.reporting import export_figures, export_tables, supplementary_workbook
from src.p5.reproducibility import build_reproducibility_manifest
from src.p5.statistics import confidence_intervals, paired_comparisons
from src.p5.standardize import collect_per_label_metrics, standardize_results


class P5Pipeline:
    """Consume existing P1-P4 files and refuse to create scientific results."""

    def __init__(self, config: P5Config):
        self.config = config
        self.output = config.output_root

    def run(self) -> Path:
        planning = self.output / "planning"
        tables = self.output / "tables"
        figures = self.output / "figures"
        planning.mkdir(parents=True, exist_ok=True)
        tables.mkdir(parents=True, exist_ok=True)
        figures.mkdir(parents=True, exist_ok=True)
        inventory = discover_artifacts(self.config.root, planning)
        completeness, missing = audit_completeness(inventory, self.config, planning)
        master = standardize_results(inventory, tables)
        per_label = collect_per_label_metrics(inventory, tables)
        ci = confidence_intervals(inventory, tables, self.config.bootstrap_iterations)
        paired = paired_comparisons(master, tables)
        exports = export_tables(master, per_label, ci, tables)
        figure_paths = export_figures(master, figures, self.config.dpi)
        supplementary_workbook(tables, self.output / "supplementary_material.xlsx")
        build_reproducibility_manifest(
            inventory, self.output / "reproducibility_manifest.json"
        )
        claims_input = self.config.root / "manuscript_claims.csv"
        claims = (
            pd.read_csv(claims_input)
            if claims_input.is_file()
            else pd.DataFrame(
                [
                    {
                        "claim": "Blockchain provides provenance and integrity evidence, not poisoning prevention."
                    },
                    {
                        "claim": "CBRA is a tested heuristic requiring matched statistical evidence before superiority claims."
                    },
                ]
            )
        )
        validate_claims(claims, master, ci, self.output / "claim_validation_report.csv")
        reviewer_response_matrix(self.output / "reviewer_response_matrix.csv")
        security = pd.DataFrame(
            [
                {
                    "scope": "Blockchain integrity",
                    "finding": "Records submitted-update origin/digest when configured",
                    "robustness_claim": "not evaluated by ledger validity",
                },
                {
                    "scope": "Poisoning robustness",
                    "finding": "Requires clean-versus-attacked predictive comparison",
                    "robustness_claim": "separate from provenance",
                },
            ]
        )
        security.to_csv(tables / "security_scope_summary.csv", index=False)
        robustness = inventory[
            inventory.relative_path.str.contains(
                "p4_robustness|robustness", case=False, regex=True
            )
        ]
        robustness.to_csv(tables / "robustness_matrix.csv", index=False)
        blockchain = inventory[
            inventory.file_name.str.contains(
                "ledger|blockchain|consensus|overhead|signature", case=False, regex=True
            )
        ]
        blockchain.to_csv(tables / "blockchain_overhead_summary.csv", index=False)
        report = {
            "complete_experiments": int(completeness.complete.sum()),
            "required_experiments": len(self.config.required_runs),
            "missing_artifacts": len(missing),
            "source_artifacts": len(inventory),
            "tables": exports,
            "figures": figure_paths,
            "scientific_results_generated": False,
        }
        (self.output / "p5_final_summary_report.json").write_text(
            json.dumps(report, indent=2, default=str) + "\n"
        )
        for source_dir in (planning, tables):
            for path in source_dir.iterdir():
                if path.is_file() and path.suffix in {".csv", ".json", ".xlsx"}:
                    shutil.copy2(path, self.output / path.name)
        self._assemble_package(inventory)
        return self.output

    def _assemble_package(self, inventory: pd.DataFrame) -> None:
        package = self.output / "final_paper_package"
        for name in (
            "tables",
            "figures",
            "supplementary",
            "reproducibility",
            "reviewer_responses",
            "results_archive",
        ):
            (package / name).mkdir(parents=True, exist_ok=True)
        for path in (self.output / "tables").glob("*"):
            shutil.copy2(path, package / "tables" / path.name)
        for path in (self.output / "figures").glob("*"):
            shutil.copy2(path, package / "figures" / path.name)
        shutil.copy2(
            self.output / "supplementary_material.xlsx",
            package / "supplementary" / "supplementary_material.xlsx",
        )
        shutil.copy2(
            self.output / "reproducibility_manifest.json",
            package / "reproducibility" / "reproducibility_manifest.json",
        )
        for name in ("claim_validation_report.csv", "reviewer_response_matrix.csv"):
            shutil.copy2(self.output / name, package / "reviewer_responses" / name)
        for _, row in inventory.iterrows():
            source = Path(row.absolute_path)
            destination = package / "results_archive" / str(row.relative_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                shutil.copy2(source, destination)
