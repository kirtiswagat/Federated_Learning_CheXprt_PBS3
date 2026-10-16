"""Patient-level confidence intervals and evidence-bounded paired comparisons."""

from __future__ import annotations

import numpy as np
import pandas as pd


def cluster_bootstrap_ci(
    frame: pd.DataFrame,
    value_column: str,
    patient_column: str = "patient_id",
    iterations: int = 2000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict[str, float | int | str]:
    if patient_column not in frame or value_column not in frame:
        raise ValueError(
            f"Patient-level CI requires {patient_column} and {value_column}"
        )
    groups = [
        group[value_column].dropna().to_numpy(float)
        for _, group in frame.groupby(patient_column, sort=True)
    ]
    groups = [group for group in groups if len(group)]
    if not groups:
        raise ValueError("No patient-level observations available")
    rng = np.random.default_rng(seed)
    draws = np.array(
        [
            np.concatenate(
                [groups[index] for index in rng.integers(0, len(groups), len(groups))]
            ).mean()
            for _ in range(iterations)
        ]
    )
    tail = (1 - confidence) * 50
    return {
        "metric": value_column,
        "patients": len(groups),
        "estimate": float(np.mean(np.concatenate(groups))),
        "ci_low": float(np.percentile(draws, tail)),
        "ci_high": float(np.percentile(draws, 100 - tail)),
        "iterations": iterations,
    }


def confidence_intervals(
    inventory: pd.DataFrame, output_dir, iterations: int = 2000, seed: int = 42
) -> pd.DataFrame:
    rows = []
    if "file_type" not in inventory:
        result = pd.DataFrame()
        output_dir.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_dir / "confidence_interval_summary.csv", index=False)
        return result
    for _, artifact in inventory[inventory.file_type == ".csv"].iterrows():
        if "prediction" not in artifact.file_name.lower():
            continue
        try:
            frame = pd.read_csv(artifact.absolute_path)
        except Exception:
            continue
        if "patient_id" not in frame:
            continue
        for metric in (
            "macro_auroc",
            "macro_auprc",
            "sensitivity",
            "specificity",
            "ppv",
            "npv",
            "f1",
            "accuracy",
        ):
            if metric in frame:
                try:
                    row = cluster_bootstrap_ci(
                        frame, metric, iterations=iterations, seed=seed
                    )
                    row.update(
                        {
                            "experiment": artifact.experiment_key,
                            "source_path": artifact.absolute_path,
                        }
                    )
                    rows.append(row)
                except ValueError:
                    continue
    result = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "confidence_interval_summary.csv", index=False)
    return result


def paired_comparisons(master: pd.DataFrame, output_dir) -> pd.DataFrame:
    rows = []
    if master.empty or "experiment" not in master or "macro_auroc" not in master:
        result = pd.DataFrame(
            columns=["left", "right", "metric", "difference", "status"]
        )
    else:
        groups = {name: frame.iloc[0] for name, frame in master.groupby("experiment")}
        pairs = (("FedAvg", "FedProx"), ("FedAvg", "CBRA"), ("FedProx", "CBRA"))
        for left_name, right_name in pairs:
            left = next(
                (
                    row
                    for name, row in groups.items()
                    if left_name.lower() in name.lower()
                ),
                None,
            )
            right = next(
                (
                    row
                    for name, row in groups.items()
                    if right_name.lower() in name.lower()
                ),
                None,
            )
            rows.append(
                {
                    "left": left_name,
                    "right": right_name,
                    "metric": "macro_auroc",
                    "difference": float(left.macro_auroc - right.macro_auroc)
                    if left is not None and right is not None
                    else np.nan,
                    "status": "patient_pairing_required"
                    if left is not None and right is not None
                    else "missing_evidence",
                }
            )
        result = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "paired_comparison_results.csv", index=False)
    return result
