"""Acceptance gates for staged P1 execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.monitoring.artifact_registry import validate_manifest


@dataclass
class GateReport:
    stage: int
    passed: bool
    evidence: list[str] = field(default_factory=list)


class P1GateOrchestrator:
    """Blocks later stages until the required local evidence is present."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.reports: list[GateReport] = []

    def require_files(self, stage: int, files: list[Path]) -> GateReport:
        missing = [str(path) for path in files if not path.is_file()]
        report = GateReport(
            stage,
            not missing,
            [f"missing:{path}" for path in missing] or ["required files present"],
        )
        self.reports.append(report)
        if not report.passed:
            raise RuntimeError(f"P1 stage {stage} gate failed: {missing}")
        return report

    def require_run_artifacts(self, run_id: str) -> GateReport:
        run_dir = self.output_dir / run_id
        try:
            validate_manifest(run_dir)
        except (OSError, ValueError) as exc:
            report = GateReport(4, False, [str(exc)])
            self.reports.append(report)
            raise RuntimeError(f"P1 run gate failed for {run_id}: {exc}") from exc
        report = GateReport(4, True, [f"valid:{run_id}"])
        self.reports.append(report)
        return report
