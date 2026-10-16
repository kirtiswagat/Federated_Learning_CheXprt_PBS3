"""Client label prevalence and Jensen-Shannon heterogeneity summaries."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def _js_divergence(left: np.ndarray, right: np.ndarray) -> float:
    left = np.clip(left, 1e-12, 1 - 1e-12)
    right = np.clip(right, 1e-12, 1 - 1e-12)
    midpoint = (left + right) / 2
    return float(
        0.5
        * (
            np.sum(left * np.log(left / midpoint))
            + np.sum(right * np.log(right / midpoint))
        )
    )


def analyze_heterogeneity(
    assignment: pd.DataFrame, labels: tuple[str, ...]
) -> pd.DataFrame:
    required = {"client_id", *labels}
    missing = required - set(assignment.columns)
    if missing:
        raise ValueError(f"Missing heterogeneity columns: {sorted(missing)}")
    global_prevalence = assignment[list(labels)].mean().to_numpy(dtype=float)
    rows = []
    for client_id, group in assignment.groupby("client_id", sort=True):
        prevalence = group[list(labels)].mean().to_numpy(dtype=float)
        rows.append(
            {
                "client_id": client_id,
                "image_count": len(group),
                "patient_count": group["patient_id"].nunique()
                if "patient_id" in group
                else np.nan,
                "prevalence_shift": float(
                    np.mean(np.abs(prevalence - global_prevalence)),
                ),
                "mean_prevalence_difference": float(
                    np.mean(prevalence - global_prevalence)
                ),
                "jensen_shannon_divergence": _js_divergence(
                    prevalence, global_prevalence
                ),
            }
        )
        rows[-1].update(
            {f"prevalence_{label}": value for label, value in zip(labels, prevalence)}
        )
    return pd.DataFrame(rows)


def save_heterogeneity(summary: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "heterogeneity_summary.csv"
    summary.to_csv(path, index=False)
    return path
