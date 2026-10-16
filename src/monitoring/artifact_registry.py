"""Checksummed artifact manifests and strict handoff validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

REQUIRED_ARTIFACTS = (
    "training_config.json",
    "environment_report.json",
    "round_history.csv",
    "client_round_details.csv",
    "endpoint_valid_predictions.csv",
    "endpoint_per_label_metrics.csv",
    "endpoint_macro_metrics.csv",
    "runtime_breakdown.csv",
    "blockchain_commitment_summary.csv",
    "endpoint_model.pt",
    "resume_checkpoint.pt",
    "artifact_manifest.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def build_manifest(
    run_dir: Path, run_id: str, metadata: dict[str, Any] | None = None
) -> dict[str, Any]:
    artifacts = []
    for path in sorted(run_dir.iterdir()):
        if path.name == "artifact_manifest.json" or not path.is_file():
            continue
        artifacts.append(
            {
                "name": path.name,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    by_name = {item["name"]: item["sha256"] for item in artifacts}
    return {
        "run_id": run_id,
        **(metadata or {}),
        "config_checksum": by_name.get("training_config.json"),
        "endpoint_model_checksum": by_name.get("endpoint_model.pt"),
        "validation_cohort_checksum": by_name.get("endpoint_valid_predictions.csv"),
        "artifacts": artifacts,
    }


def write_manifest(
    run_dir: Path, run_id: str, metadata: dict[str, Any] | None = None
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "artifact_manifest.json"
    path.write_text(
        json.dumps(build_manifest(run_dir, run_id, metadata), indent=2, sort_keys=True)
        + "\n"
    )
    return path


def validate_manifest(
    run_dir: Path, required: Iterable[str] = REQUIRED_ARTIFACTS
) -> dict[str, Any]:
    manifest_path = run_dir / "artifact_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Missing artifact manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_name = {item.get("name"): item for item in manifest.get("artifacts", [])}
    missing = [
        name
        for name in required
        if name != "artifact_manifest.json" and not (run_dir / name).is_file()
    ]
    mismatched = []
    for name, item in by_name.items():
        path = run_dir / name
        if path.is_file() and sha256_file(path) != item.get("sha256"):
            mismatched.append(name)
    required_fields = {
        "run_id",
        "config_checksum",
        "endpoint_model_checksum",
        "validation_cohort_checksum",
        "artifacts",
    }
    absent_fields = sorted(required_fields - set(manifest))
    if missing or mismatched or absent_fields:
        raise ValueError(
            f"Invalid artifact handoff: missing={missing}, checksum_mismatch={mismatched}, absent_fields={absent_fields}"
        )
    return manifest
