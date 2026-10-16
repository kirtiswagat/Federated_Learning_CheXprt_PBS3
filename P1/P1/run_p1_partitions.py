"""Run deterministic patient-level partition generation."""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from src.config import LABELS
from src.data.heterogeneity import analyze_heterogeneity, save_heterogeneity
from src.data.partitions import make_patient_partition, write_partition


def _path_identifier(path_value: object, prefix: str) -> str | None:
    parts = [part for part in str(path_value).replace("\\", "/").split("/") if part]
    matches = [part for part in parts if part.lower().startswith(prefix)]
    return matches[-1] if matches else None


def _load_metadata(path: Path) -> pd.DataFrame:
    metadata = pd.read_csv(path)
    if "patient_id" not in metadata.columns:
        if "Path" not in metadata.columns:
            raise ValueError("Metadata must contain patient_id or CheXpert Path")
        metadata["patient_id"] = metadata["Path"].map(
            lambda value: _path_identifier(value, "patient")
        )
    if "study_id" not in metadata.columns and "Path" in metadata.columns:
        metadata["study_id"] = metadata["Path"].map(
            lambda value: _path_identifier(value, "study")
        )
    if metadata["patient_id"].isna().any():
        raise ValueError("Could not derive patient_id from one or more metadata paths")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--mode", choices=["iid", "dirichlet"], default="iid")
    parser.add_argument("--alpha", type=float)
    args = parser.parse_args()
    name = f"{args.mode}_{args.clients}" + (f"_{args.alpha}" if args.alpha else "")
    print(f"[1/4] Loading metadata: {args.metadata}", flush=True)
    metadata = _load_metadata(args.metadata)
    print(
        f"[2/4] Loaded {len(metadata):,} rows, "
        f"{metadata['patient_id'].nunique():,} patients",
        flush=True,
    )
    print(f"[3/4] Building {name} partition", flush=True)
    result = make_patient_partition(metadata, args.clients, args.mode, args.alpha)
    print(f"[4/4] Writing outputs to {args.output}", flush=True)
    write_partition(result, args.output, name)
    summary = result.groupby("client_id").agg(
        images=("client_id", "size"), patients=("patient_id", "nunique")
    )
    print(summary.to_string(), flush=True)
    labelled = result.copy()
    labelled[list(LABELS)] = (
        labelled[list(LABELS)].apply(pd.to_numeric, errors="coerce").eq(1).astype(float)
    )
    het = analyze_heterogeneity(labelled, LABELS)
    het_dir = args.output.parent / "heterogeneity" / name
    save_heterogeneity(het, het_dir)
    print(
        het[
            [
                "client_id",
                "image_count",
                "prevalence_shift",
                "jensen_shannon_divergence",
            ]
        ].to_string(index=False),
        flush=True,
    )
    print(f"Done: {name}", flush=True)


if __name__ == "__main__":
    main()
