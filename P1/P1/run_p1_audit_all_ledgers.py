"""Re-validate the saved ledger of every completed run and write one summary CSV."""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.blockchain.permissioned_ledger import PermissionedLedger  # noqa: E402

VALIDATORS = ("validator_1", "validator_2", "validator_3")


def audit_run(run_dir: Path) -> dict:
    blocks = json.loads((run_dir / "ledger.json").read_text())
    try:
        PermissionedLedger.from_blocks(blocks, VALIDATORS, 2)
        chain_valid = True
    except Exception:
        chain_valid = False
    digests = []
    summary = run_dir / "blockchain_commitment_summary.csv"
    if summary.exists():
        with summary.open() as handle:
            digests = [row["update_digest"] for row in csv.DictReader(handle)]
    ledger_hashes = [block["transaction_hash"] for block in blocks]
    return {
        "run": run_dir.name,
        "blocks": len(blocks),
        "chain_and_replicas_valid": chain_valid,
        "summary_digests": len(digests),
        "digests_match_ledger": digests == ledger_hashes,
        "min_votes": min((len(b["validator_votes"]) for b in blocks), default=0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(ROOT / "Output"))
    args = parser.parse_args()
    out = Path(args.output_dir)
    runs = sorted(
        p
        for pattern in ("p1_critical_revision/runs/R*", "p3_cbra/R*")
        for p in out.glob(pattern)
        if p.is_dir() and (p / "ledger.json").exists()
    )
    rows = [audit_run(p) for p in runs]
    target = out / "p1_critical_revision" / "blockchain" / "all_runs_ledger_audit.csv"
    with target.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
