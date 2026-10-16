"""Recreate the patient-level train/internal-evaluation split and verify it.

CheXpert images and CSV files are not redistributed in this repository.  This
script rebuilds ``train_split.csv`` (80 % of patients) and ``val_split.csv``
(20 % of patients, the *internal evaluation set*) from the official
``train.csv`` with seed 42, and checks the SHA-256 digests of the result
against the digests of the files used for the reported experiments.

It can also verify freshly generated client partitions against
``Output/p1_critical_revision/partitions.sha256``.

Examples
--------
python3 P1/run_p1_prepare_splits.py --data-root /path/to/data
python3 P1/run_p1_prepare_splits.py --data-root /path/to/data --verify-partitions /path/to/partitions
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.federated.p1_adapter import verify_checksum  # noqa: E402

EXPECTED_SHA256 = {
    "train.csv": "51822e6e0a7e8589580219c763e3d4a1d1ddaa8d39f15edf39b1076358c126c1",
    "train_split.csv": "08871e2be099d17f527e178b3959c8c9f46bec943e66d504c86ff819a2f49392",
    "val_split.csv": "841532c70ba6c0d3d406b60e4ad6c355b33664c6eb02d6047d04a06f79096e5b",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check(path: Path) -> bool:
    ok = _sha256(path) == EXPECTED_SHA256[path.name]
    print(f"  [{'OK' if ok else 'MISMATCH'}] {path.name}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Folder containing CheXpert-v1.0-small/")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify-partitions", type=Path,
                        help="Folder with generated *_client_assignment.csv files")
    args = parser.parse_args()
    folder = args.data_root / "CheXpert-v1.0-small"

    ok = True
    if args.verify_partitions is None:
        print("Reading official train.csv")
        ok &= _check(folder / "train.csv")
        frame = pd.read_csv(folder / "train.csv")
        patient = frame["Path"].str.extract(r"(patient\d+)", expand=False)
        patients = np.asarray(patient.unique(), dtype=object)
        patients = patients[np.random.default_rng(args.seed).permutation(len(patients))]
        held_out = set(patients[: int(len(patients) * args.val_ratio)])
        is_val = patient.isin(held_out)
        frame[~is_val].to_csv(folder / "train_split.csv", index=False)
        frame[is_val].to_csv(folder / "val_split.csv", index=False)
        print(f"Wrote train_split.csv ({int((~is_val).sum()):,} rows) and "
              f"val_split.csv ({int(is_val.sum()):,} rows)")
        if (args.seed, args.val_ratio) == (42, 0.2):
            ok &= _check(folder / "train_split.csv")
            ok &= _check(folder / "val_split.csv")
    else:
        sums = REPO / "Output" / "p1_critical_revision" / "partitions.sha256"
        for path in sorted(args.verify_partitions.glob("*.csv")):
            try:
                verify_checksum(path, sums)
                print(f"  [OK] {path.name}")
            except RuntimeError as exc:
                ok = False
                print(f"  [FAIL] {exc}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
