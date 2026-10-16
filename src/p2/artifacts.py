"""P2-specific artifact handoff validation."""

from __future__ import annotations

from pathlib import Path

from src.monitoring.artifact_registry import sha256_file

P2_REQUIRED = (
    "training_config.json",
    "environment_report.json",
    "round_history.csv",
    "per_label_metrics.csv",
    "macro_metrics.csv",
    "predictions.csv",
    "checkpoint.pt",
    "resume_checkpoint.pt",
    "runtime_summary.csv",
    "client_participation.csv",
    "artifact_manifest.json",
)


def validate_p2_run(run_dir: Path) -> dict:
    missing = [name for name in P2_REQUIRED if not (run_dir / name).is_file()]
    if missing:
        raise ValueError(f"Invalid P2 handoff; missing: {missing}")
    manifest = __import__("json").loads(
        (run_dir / "artifact_manifest.json").read_text()
    )
    for artifact in manifest.get("artifacts", []):
        path = run_dir / artifact["name"]
        if path.is_file() and sha256_file(path) != artifact["sha256"]:
            raise ValueError(f"P2 checksum mismatch: {artifact['name']}")
    return manifest
