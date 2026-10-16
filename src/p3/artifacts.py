"""P3 artifact handoff validation."""

from __future__ import annotations

import json
from pathlib import Path

from src.monitoring.artifact_registry import sha256_file

P3_REQUIRED = (
    "training_config.json",
    "environment_report.json",
    "round_history.csv",
    "training_summary.csv",
    "per_label_metrics.csv",
    "best_valid_predictions.csv",
    "cbra_round_details.csv",
    "cbra_weights.csv",
    "checkpoint.pt",
    "resume_checkpoint.pt",
    "artifact_manifest.json",
)


def validate_p3_run(run_dir: Path) -> dict:
    missing = [name for name in P3_REQUIRED if not (run_dir / name).is_file()]
    if missing:
        raise ValueError(f"Invalid P3 handoff; missing: {missing}")
    manifest = json.loads((run_dir / "artifact_manifest.json").read_text())
    for artifact in manifest.get("artifacts", []):
        path = run_dir / artifact["name"]
        if path.is_file() and sha256_file(path) != artifact["sha256"]:
            raise ValueError(f"P3 checksum mismatch: {artifact['name']}")
    return manifest
