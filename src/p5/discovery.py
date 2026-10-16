"""Recursive artifact discovery and experiment completeness auditing."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.p5.config import P5Config

RUN_PATTERN = re.compile(r"(?<![A-Z])R(\d{2})(?!\d)", re.IGNORECASE)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _experiment_key(path: Path) -> str | None:
    match = RUN_PATTERN.search(str(path))
    return match.group(0).upper() if match else None


def discover_artifacts(root: Path, output_dir: Path | None = None) -> pd.DataFrame:
    rows = []
    excluded = {"p5_publication_package"}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or any(
            part in excluded or part.startswith(".") for part in path.parts
        ):
            continue
        readable = True
        schema = ""
        try:
            if path.suffix.lower() == ".csv":
                schema = json.dumps(list(pd.read_csv(path, nrows=0).columns))
            elif path.suffix.lower() == ".json":
                json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            readable = False
            schema = str(exc)
        rows.append(
            {
                "absolute_path": str(path.resolve()),
                "relative_path": str(path.relative_to(root)),
                "file_name": path.name,
                "file_type": path.suffix.lower(),
                "size_bytes": path.stat().st_size,
                "modified_utc": datetime.fromtimestamp(
                    path.stat().st_mtime, timezone.utc
                ).isoformat(),
                "sha256": _sha256(path),
                "experiment_key": _experiment_key(path),
                "schema_columns": schema,
                "readable": readable,
            }
        )
    result = pd.DataFrame(
        rows,
        columns=[
            "absolute_path",
            "relative_path",
            "file_name",
            "file_type",
            "size_bytes",
            "modified_utc",
            "sha256",
            "experiment_key",
            "schema_columns",
            "readable",
        ],
    )
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_dir / "artifact_registry.csv", index=False)
        (output_dir / "artifact_inventory.json").write_text(
            result.to_json(orient="records", indent=2) + "\n"
        )
    return result


def audit_completeness(
    inventory: pd.DataFrame, config: P5Config, output_dir: Path | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for run_id in config.required_runs:
        subset = inventory[inventory.experiment_key == run_id]
        names = set(subset.file_name)
        checks = {
            "training_summary": any(
                "training_summary" in name or "round_history" in name for name in names
            ),
            "predictions": any("prediction" in name for name in names),
            "checkpoint": any(name.endswith((".pt", ".pth")) for name in names),
            "metrics": any("metric" in name or "macro" in name for name in names),
            "configuration": any("config" in name for name in names),
        }
        rows.append({"experiment": run_id, **checks, "complete": all(checks.values())})
    completeness = pd.DataFrame(rows)
    missing = completeness.melt(
        id_vars=["experiment"], var_name="artifact_type", value_name="present"
    )
    missing = missing[(missing.artifact_type != "complete") & ~missing.present].drop(
        columns="present"
    )
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        completeness.to_csv(output_dir / "completeness_report.csv", index=False)
        missing.to_csv(output_dir / "missing_artifact_report.csv", index=False)
    return completeness, missing
