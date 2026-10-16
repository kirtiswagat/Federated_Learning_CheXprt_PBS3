"""Leakage-safe CheXpert metadata and historical-artifact audit."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import Image

from src.config import LABELS
from src.monitoring.artifact_registry import sha256_file


class DataAuditError(ValueError):
    """Raised when the dataset cannot satisfy a P1 acceptance gate."""


@dataclass(frozen=True)
class AuditResult:
    dataset_summary: pd.DataFrame
    label_support: pd.DataFrame
    patient_leakage: pd.DataFrame
    image_health: pd.DataFrame
    historical_artifacts: pd.DataFrame
    provenance: dict


def _required_columns(labels: Iterable[str]) -> set[str]:
    return {"Path", *labels}


def _path_identifier(path_value: object, component: str) -> str | None:
    parts = [part for part in str(path_value).replace("\\", "/").split("/") if part]
    if component == "patient":
        matches = [part for part in parts if part.lower().startswith("patient")]
    else:
        matches = [part for part in parts if part.lower().startswith("study")]
    return matches[-1] if matches else None


def _normalise_ids(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "patient_id" not in result:
        result["patient_id"] = result["Path"].map(
            lambda value: _path_identifier(value, "patient")
        )
    if "study_id" not in result:
        result["study_id"] = result["Path"].map(
            lambda value: _path_identifier(value, "study")
        )
    invalid = result["patient_id"].isna() | result["study_id"].isna()
    if invalid.any():
        raise DataAuditError(
            f"Could not derive patient/study IDs from Path for {int(invalid.sum())} rows"
        )
    result["patient_study_id"] = (
        result["patient_id"].astype("string")
        + ":"
        + result["study_id"].astype("string")
    )
    return result


def _resolve_image(root: Path, path_value: str) -> Path:
    value = Path(str(path_value))
    candidates = [
        value if value.is_absolute() else root / value,
        root.parent / value,
        root / value.name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0].resolve()


def audit_dataset(
    train_csv: Path,
    valid_csv: Path,
    root: Path,
    output_dir: Path | None = None,
    labels: tuple[str, ...] = LABELS,
    progress_every: int = 1000,
) -> AuditResult:
    if progress_every < 1:
        raise ValueError("progress_every must be positive")

    started = time.perf_counter()

    def progress(message: str) -> None:
        elapsed = time.perf_counter() - started
        print(f"[P1-01 +{elapsed:7.1f}s] {message}", flush=True)

    for path in (train_csv, valid_csv):
        if not path.is_file():
            raise FileNotFoundError(path)
    progress("Loading train.csv")
    train = _normalise_ids(pd.read_csv(train_csv))
    progress(f"Loaded train metadata: {len(train):,} rows")
    progress("Loading valid.csv")
    valid = _normalise_ids(pd.read_csv(valid_csv))
    progress(f"Loaded validation metadata: {len(valid):,} rows")
    missing = {
        name: sorted(_required_columns(labels) - set(frame.columns))
        for name, frame in (("train", train), ("valid", valid))
    }
    if any(missing.values()):
        raise DataAuditError(f"Missing required columns: {missing}")
    progress("Required columns and path-derived patient/study IDs validated")
    train_patients = set(train["patient_id"].dropna().astype(str))
    valid_patients = set(valid["patient_id"].dropna().astype(str))
    overlap = sorted(train_patients & valid_patients)
    leakage = pd.DataFrame(
        {"patient_id": overlap, "status": ["LEAKAGE"] * len(overlap)}
    )
    if overlap:
        raise DataAuditError(f"Patient leakage detected: {overlap[:5]}")
    progress("Patient leakage check passed: zero train/validation overlap")

    label_rows = []
    for split, frame in (("train", train), ("valid", valid)):
        for label in labels:
            values = frame[label]
            label_rows.append(
                {
                    "split": split,
                    "label": label,
                    "positive": int((values == 1).sum()),
                    "negative": int((values == 0).sum()),
                    "uncertain": int((values == -1).sum()),
                    "missing": int(values.isna().sum()),
                    "observed_classes": int(values.dropna().isin([0, 1]).sum()),
                }
            )
    progress("Label support counts calculated")
    health_rows = []
    summaries = []
    for split, frame in (("train", train), ("valid", valid)):
        progress(f"Resolving and verifying {split} images: {len(frame):,} total")
        paths = [_resolve_image(root, value) for value in frame["Path"]]
        unreadable = []
        for index, path in enumerate(paths, start=1):
            if path.is_file():
                try:
                    with Image.open(path) as image:
                        image.verify()
                    status = "ok"
                except (OSError, ValueError):
                    status = "unreadable"
                    unreadable.append(str(path))
            else:
                status = "missing"
            health_rows.append({"split": split, "path": str(path), "status": status})
            if index % progress_every == 0 or index == len(paths):
                ok_count = sum(
                    row["status"] == "ok"
                    for row in health_rows[-index:]
                    if row["split"] == split
                )
                print(
                    f"[P1-01] {split}: {index:,}/{len(paths):,} images checked; "
                    f"{ok_count:,} readable",
                    flush=True,
                )
        summaries.append(
            {
                "split": split,
                "metadata_rows": len(frame),
                "retained_images": len(frame),
                "unique_patients": frame["patient_id"].nunique(),
                "unique_studies": frame["study_id"].nunique(),
                "unique_patient_studies": frame["patient_study_id"].nunique(),
                "unique_image_paths": frame["Path"].nunique(),
                "duplicate_image_paths": int(frame["Path"].duplicated().sum()),
                "missing_images": sum(
                    row["status"] == "missing"
                    for row in health_rows
                    if row["split"] == split
                ),
                "unreadable_images": len(unreadable),
            }
        )

    artifact_rows = []
    patterns = (
        "*valid_predictions*.csv",
        "*training_summary*.csv",
        "*round_history*.csv",
        "*client_round_details*.csv",
        "*per_label_auroc*.csv",
        "best_model.pt",
        "best_central_model.pt",
        "resume_checkpoint.pt",
        "training_config.json",
        "*ledger*.json",
    )
    found = {
        path.resolve()
        for pattern in patterns
        for path in root.rglob(pattern)
        if path.is_file()
    }
    progress(f"Scanning historical artifacts: {len(found):,} candidates")
    for path in sorted(found):
        schema = ""
        readable = True
        try:
            if path.suffix.lower() == ".csv":
                schema = json.dumps(list(pd.read_csv(path, nrows=0).columns))
            elif path.suffix.lower() == ".json":
                json.loads(path.read_text(encoding="utf-8"))
        except (
            Exception
        ) as exc:  # artifact registry must record, not hide, unreadability
            readable = False
            schema = str(exc)
        artifact_rows.append(
            {
                "absolute_path": str(path),
                "relative_path": str(path.relative_to(root)),
                "file_type": path.suffix,
                "size": path.stat().st_size,
                "modified": datetime.fromtimestamp(
                    path.stat().st_mtime, timezone.utc
                ).isoformat(),
                "sha256": sha256_file(path),
                "experiment_key": path.stem.split("_")[0],
                "schema_columns": schema,
                "readable": readable,
            }
        )
    result = AuditResult(
        pd.DataFrame(summaries),
        pd.DataFrame(label_rows),
        leakage,
        pd.DataFrame(health_rows),
        pd.DataFrame(artifact_rows),
        {
            "train_csv": str(train_csv.resolve()),
            "valid_csv": str(valid_csv.resolve()),
            "labels": list(labels),
            "patient_overlap_count": 0,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    if output_dir:
        progress(f"Writing audit outputs to {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        result.dataset_summary.to_csv(output_dir / "dataset_summary.csv", index=False)
        result.label_support.to_csv(output_dir / "label_support.csv", index=False)
        result.patient_leakage.to_csv(
            output_dir / "patient_leakage_report.csv", index=False
        )
        result.image_health.to_csv(output_dir / "image_health_report.csv", index=False)
        result.historical_artifacts.to_csv(
            output_dir / "historical_artifact_registry.csv", index=False
        )
        (output_dir / "provenance_manifest.json").write_text(
            json.dumps(result.provenance, indent=2) + "\n"
        )
        progress("Audit outputs written successfully")
    return result
