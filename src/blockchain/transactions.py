#########----------------------------------------------------------------------########
########----------src/blockchain/transactions----------------------------------########
"""Deterministic model-update serialization and signed transactions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from src.blockchain.identity import Identity


def serialize_update(update: Any) -> bytes:
    if isinstance(update, dict):
        return (
            b"{"
            + b";".join(
                key.encode() + b":" + serialize_update(update[key])
                for key in sorted(update)
            )
            + b"}"
        )
    if hasattr(update, "detach"):
        value = update.detach().cpu().contiguous().numpy()
        return (
            b"tensor:"
            + str(value.dtype).encode()
            + b":"
            + repr(tuple(value.shape)).encode()
            + b":"
            + value.tobytes()
        )
    if isinstance(update, np.ndarray):
        return (
            b"array:"
            + str(update.dtype).encode()
            + b":"
            + repr(tuple(update.shape)).encode()
            + b":"
            + update.tobytes()
        )
    return json.dumps(update, sort_keys=True, separators=(",", ":")).encode()


def update_digest(update: Any) -> str:
    return hashlib.sha256(serialize_update(update)).hexdigest()


@dataclass(frozen=True)
class Transaction:
    round_id: int
    client_id: str
    update_digest: str
    signature: str


def sign_update(
    identity: Identity, round_id: int, client_id: str, update: Any
) -> Transaction:
    digest = update_digest(update)
    signature = identity.sign(digest.encode()).hex()
    return Transaction(round_id, client_id, digest, signature)


def verify_transaction(transaction: Transaction, identity: Identity) -> bool:
    return identity.verify(
        transaction.update_digest.encode(), bytes.fromhex(transaction.signature)
    )

