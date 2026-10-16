"""Replicated majority-finality ledger for experiment commitments."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Block:
    block_id: int
    timestamp: float
    round_id: int
    previous_hash: str
    transaction_hash: str
    validator_votes: tuple[str, ...]
    block_hash: str


class ConsensusError(RuntimeError):
    """Raised when validators do not reach majority finality."""


class PermissionedLedger:
    def __init__(
        self,
        validators: tuple[str, ...] = ("validator_1", "validator_2", "validator_3"),
        majority: int = 2,
    ):
        if majority <= len(validators) // 2 or majority > len(validators):
            raise ValueError("majority must be strict and attainable")
        self.validators = validators
        self.majority = majority
        self.blocks: list[Block] = []
        self.replicas = {validator: [] for validator in validators}

    @staticmethod
    def _hash(payload: dict[str, Any]) -> str:
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def append(
        self, round_id: int, transaction_hash: str, votes: tuple[str, ...] | None = None
    ) -> Block:
        votes = votes or self.validators
        valid_votes = tuple(
            validator for validator in votes if validator in self.validators
        )
        if len(valid_votes) < self.majority:
            raise ConsensusError("validator majority was not reached")
        previous = self.blocks[-1].block_hash if self.blocks else "0" * 64
        payload = {
            "block_id": len(self.blocks),
            "timestamp": time.time(),
            "round_id": round_id,
            "previous_hash": previous,
            "transaction_hash": transaction_hash,
            "validator_votes": valid_votes,
        }
        block = Block(**payload, block_hash=self._hash(payload))
        self.blocks.append(block)
        for replica in self.replicas.values():
            replica.append(block)
        return block

    @staticmethod
    def sign(identity, round_id: int, client_id, digest: str) -> str:
        return identity.sign(digest.encode()).hex()

    def record_update(
        self, round_id: int, client_id, update_digest: str, signature_hex: str
    ) -> Block:
        return self.append(round_id, update_digest)

    def record_round(self, round_id: int, commitment: dict) -> None:
        """Digest-only ledgers keep no per-round commitment."""

    @classmethod
    def from_blocks(cls, items, validators, majority) -> "PermissionedLedger":
        ledger = cls(validators, majority)
        for item in items:
            block = Block(**{**item, "validator_votes": tuple(item["validator_votes"])})
            ledger.blocks.append(block)
            for replica in ledger.replicas.values():
                replica.append(block)
        if not ledger.validate():
            raise ConsensusError("Restored ledger failed validation")
        return ledger

    def to_blocks(self) -> list[dict[str, Any]]:
        return [asdict(block) for block in self.blocks]

    def validate(self) -> bool:
        previous = "0" * 64
        for index, block in enumerate(self.blocks):
            payload = asdict(block)
            actual = payload.pop("block_hash")
            if (
                actual != self._hash(payload)
                or block.block_id != index
                or block.previous_hash != previous
            ):
                return False
            previous = block.block_hash
        return all(replica == self.blocks for replica in self.replicas.values())

    def save(self, path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps([asdict(block) for block in self.blocks], indent=2, default=list)
            + "\n"
        )
