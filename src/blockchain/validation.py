"""Ledger, replication, block-mutation, and signature-substitution checks."""

from __future__ import annotations

import copy
import pandas as pd

from src.blockchain.identity import Identity
from src.blockchain.permissioned_ledger import PermissionedLedger
from src.blockchain.transactions import sign_update, verify_transaction


def run_validation_suite(progress: bool = True) -> pd.DataFrame:
    def report(message: str) -> None:
        if progress:
            print(f"[P1-07] {message}", flush=True)

    report("Starting ledger replication check")
    ledger = PermissionedLedger()
    ledger.append(1, "a" * 64)
    rows = [
        {
            "check": "ledger_replication",
            "status": "PASS" if ledger.validate() else "FAIL",
        }
    ]
    report(f"Ledger replication check: {rows[-1]['status']}")
    report("Starting block mutation check")
    mutated = copy.deepcopy(ledger)
    mutated.blocks[0] = mutated.blocks[0].__class__(
        **{**mutated.blocks[0].__dict__, "transaction_hash": "b" * 64}
    )
    rows.append(
        {
            "check": "block_mutation",
            "status": "PASS" if not mutated.validate() else "FAIL",
        }
    )
    report(f"Block mutation check: {rows[-1]['status']}")
    report("Starting signature substitution check")
    identity = Identity.generate("client_1")
    other = Identity.generate("client_2")
    transaction = sign_update(identity, 1, "client_1", {"weight": [1, 2, 3]})
    rows.append(
        {
            "check": "signature_substitution",
            "status": "PASS" if not verify_transaction(transaction, other) else "FAIL",
        }
    )
    report(f"Signature substitution check: {rows[-1]['status']}")
    report(f"Completed {len(rows)} blockchain validation checks")
    return pd.DataFrame(rows)
