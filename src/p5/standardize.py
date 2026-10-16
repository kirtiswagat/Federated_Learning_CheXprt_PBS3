"""Normalize heterogeneous P1-P4 result schemas into master tables."""

from __future__ import annotations

from pathlib import Path
import pandas as pd


def _metric_row(path: Path, experiment: str) -> dict | None:
    try:
        frame = pd.read_csv(path)
    except Exception:
        return None
    if frame.empty:
        return None
    row = frame.iloc[-1].to_dict()
    row["experiment"] = experiment
    row["source_path"] = str(path)
    aliases = {
        "auc_macro": "macro_auroc",
        "auroc_macro": "macro_auroc",
        "auprc_macro": "macro_auprc",
        "f1": "f1",
    }
    for source, target in aliases.items():
        if source in row and target not in row:
            row[target] = row[source]
    return row


def standardize_results(
    inventory: pd.DataFrame, output_dir: Path | None = None
) -> pd.DataFrame:
    rows = []
    for _, artifact in inventory[inventory.experiment_key.notna()].iterrows():
        name = artifact.file_name.lower()
        if artifact.file_type == ".csv" and (
            "macro" in name or "summary" in name or "history" in name
        ):
            row = _metric_row(Path(artifact.absolute_path), artifact.experiment_key)
            if row:
                rows.append(row)
    result = (
        pd.DataFrame(rows).drop_duplicates(subset=["experiment", "source_path"])
        if rows
        else pd.DataFrame()
    )
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_dir / "master_results_dataframe.csv", index=False)
    return result


def collect_per_label_metrics(
    inventory: pd.DataFrame, output_dir: Path | None = None
) -> pd.DataFrame:
    frames = []
    for _, artifact in inventory[inventory.file_type == ".csv"].iterrows():
        if (
            "label" not in artifact.file_name.lower()
            and "per_label" not in artifact.file_name.lower()
        ):
            continue
        try:
            frame = pd.read_csv(artifact.absolute_path)
        except Exception:
            continue
        frame.insert(0, "experiment", artifact.experiment_key)
        frame.insert(1, "source_path", artifact.absolute_path)
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if output_dir:
        result.to_csv(output_dir / "table_per_label_metrics.csv", index=False)
    return result
