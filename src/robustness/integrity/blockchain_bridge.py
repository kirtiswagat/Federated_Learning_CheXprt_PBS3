"""Optional P1 Ed25519/ledger bridge for actual submitted updates."""

from __future__ import annotations

from dataclasses import asdict

from src.blockchain.identity import Identity
from src.blockchain.permissioned_ledger import PermissionedLedger
from src.blockchain.transactions import sign_update, verify_transaction


def commit_submitted_update(
    identity: Identity,
    ledger: PermissionedLedger,
    client_id: str,
    round_id: int,
    update,
) -> dict:
    transaction = sign_update(identity, round_id, client_id, update)
    signature_valid = verify_transaction(transaction, identity)
    if signature_valid:
        ledger.append(round_id, transaction.update_digest)
    return {
        "client_id": client_id,
        "round_id": round_id,
        "submitted_update_digest": transaction.update_digest,
        "signature_valid": signature_valid,
        "transaction_committed": signature_valid,
        "ledger_valid": ledger.validate(),
    }
