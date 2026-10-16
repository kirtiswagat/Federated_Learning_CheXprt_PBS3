"""Deterministic patient-exclusive IID and Dirichlet partitions."""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from src.config import LABELS


class PartitionError(ValueError):
    """Raised when a patient partition violates exclusivity."""


def _validate(frame: pd.DataFrame) -> None:
    counts = frame.groupby("patient_id")["client_id"].nunique()
    if (counts > 1).any():
        raise PartitionError("A patient was assigned to multiple clients")


def _primary_labels(metadata: pd.DataFrame, patients: np.ndarray) -> np.ndarray:
    """Rarest positive label per patient (U-Zero); 'none' if no positives."""
    labels = [label for label in LABELS if label in metadata.columns]
    if not labels:
        raise ValueError("Dirichlet partitions require label columns in metadata")
    positives = (
        metadata[labels]
        .apply(pd.to_numeric, errors="coerce")
        .eq(1)
        .assign(patient_id=metadata["patient_id"].astype(str).to_numpy())
        .groupby("patient_id")
        .max()
        .reindex(patients)
        .fillna(False)
        .astype(bool)
    )
    order = positives.sum().sort_values().index.tolist()  # rarest first
    primary = np.full(len(patients), "none", dtype=object)
    unassigned = np.ones(len(patients), dtype=bool)
    for label in order:
        hit = unassigned & positives[label].to_numpy()
        primary[hit] = label
        unassigned &= ~hit
    return primary


def _label_dirichlet(
    metadata: pd.DataFrame,
    patients: np.ndarray,
    clients: int,
    alpha: float,
    rng: np.random.Generator,
    min_patients: int,
    max_attempts: int = 200,
) -> np.ndarray:
    """Spread each label's patients over clients with Dirichlet proportions."""
    primary = _primary_labels(metadata, patients)
    groups = [np.flatnonzero(primary == label) for label in np.unique(primary)]
    for _ in range(max_attempts):
        assignments = np.full(len(patients), -1, dtype=int)
        for members in groups:
            members = rng.permutation(members)
            proportions = rng.dirichlet(np.repeat(alpha, clients))
            bounds = np.rint(np.cumsum(proportions) * len(members)).astype(int)
            bounds[-1] = len(members)
            start = 0
            for client, stop in enumerate(bounds):
                assignments[members[start:stop]] = client
                start = stop
        if np.bincount(assignments, minlength=clients).min() >= min_patients:
            return assignments
    raise PartitionError(
        f"No Dirichlet draw gave every client >= {min_patients} patients "
        f"in {max_attempts} attempts (alpha={alpha})"
    )


def make_patient_partition(
    metadata: pd.DataFrame,
    clients: int,
    mode: str = "iid",
    alpha: float | None = None,
    seed: int = 42,
    min_patients: int = 1,
) -> pd.DataFrame:
    if clients < 1 or "patient_id" not in metadata.columns:
        raise ValueError(
            "metadata must contain patient_id and clients must be positive"
        )
    if mode == "dirichlet" and (alpha is None or alpha <= 0):
        raise ValueError("Dirichlet partitions require alpha > 0")
    patients = np.array(sorted(metadata["patient_id"].dropna().astype(str).unique()))
    rng = np.random.default_rng(seed)
    if mode == "iid":
        rng.shuffle(patients)
        assignments = np.arange(len(patients)) % clients
    else:
        assignments = _label_dirichlet(
            metadata, patients, clients, alpha, rng, min_patients
        )
    patient_to_client = dict(zip(patients, assignments))
    result = metadata.copy()
    result["patient_id"] = result["patient_id"].astype(str)
    result["client_id"] = result["patient_id"].map(patient_to_client).astype(int)
    _validate(result)
    return result


def write_partition(
    result: pd.DataFrame, output_dir: Path, name: str
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    assignments = output_dir / f"{name}_client_assignment.csv"
    manifest = output_dir / f"{name}_partition_manifest.csv"
    result.to_csv(assignments, index=False)
    summary = (
        result.groupby("client_id")
        .agg(images=("client_id", "size"), patients=("patient_id", "nunique"))
        .reset_index()
    )
    summary.to_csv(manifest, index=False)
    return assignments, manifest
