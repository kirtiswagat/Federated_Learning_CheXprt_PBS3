########----------------------------------------------------------------------########
########----------src/blockchain/secure_ledger--------------------------------#######
"""Signed permissioned ledger: client signatures are verified before commitment and
validator votes are Ed25519 signatures over the block hash."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from src.blockchain.identity import Identity, verify_with_public_hex
from src.blockchain.permissioned_ledger import ConsensusError


class SignatureError(ConsensusError):
    """Raised when a submitted client signature does not verify."""


@dataclass(frozen=True)
class SecureBlock:
    block_id: int
    timestamp: float
    round_id: int
    kind: str
    subject: str
    previous_hash: str
    body: dict
    block_hash: str
    validator_votes: tuple


def _hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def update_message(round_id: int, client_id: str, digest: str) -> bytes:
    return f"{round_id}|{client_id}|{digest}".encode()


class SecureLedger:
    def __init__(
        self,
        validator_keys: Mapping[str, str],
        client_keys: Mapping[str, str],
        majority: int,
        signers: Mapping[str, Identity] | None = None,
    ):
        if majority <= len(validator_keys) // 2 or majority > len(validator_keys):
            raise ValueError("majority must be strict and attainable")
        self.validator_keys = dict(validator_keys)
        self.client_keys = {str(k): v for k, v in client_keys.items()}
        self.majority = majority
        self.signers = dict(signers or {})
        self.blocks: list[SecureBlock] = []
        self.replicas: dict[str, list[SecureBlock]] = {v: [] for v in validator_keys}

    @staticmethod
    def sign(identity: Identity, round_id: int, client_id, digest: str) -> str:
        return identity.sign(update_message(round_id, str(client_id), digest)).hex()

    def _payload(self, round_id, kind, subject, body, previous) -> dict[str, Any]:
        return {
            "block_id": len(self.blocks),
            "timestamp": time.time(),
            "round_id": round_id,
            "kind": kind,
            "subject": subject,
            "previous_hash": previous,
            "body": body,
        }

    def _finalize(
        self, round_id: int, kind: str, subject: str, body: dict
    ) -> SecureBlock:
        previous = self.blocks[-1].block_hash if self.blocks else "0" * 64
        payload = self._payload(round_id, kind, subject, body, previous)
        block_hash = _hash(payload)
        votes = tuple(
            (name, signer.sign(block_hash.encode()).hex())
            for name, signer in self.signers.items()
            if name in self.validator_keys
        )
        if len(votes) < self.majority:
            raise ConsensusError("validator majority was not reached")
        block = SecureBlock(**payload, block_hash=block_hash, validator_votes=votes)
        self.blocks.append(block)
        for replica in self.replicas.values():
            replica.append(block)
        return block

    def record_update(
        self, round_id: int, client_id, update_digest: str, signature_hex: str
    ) -> SecureBlock:
        key = self.client_keys.get(str(client_id))
        message = update_message(round_id, str(client_id), update_digest)
        if key is None or not verify_with_public_hex(key, message, signature_hex):
            raise SignatureError(f"invalid signature from client {client_id}")
        return self._finalize(
            round_id,
            "client_update",
            str(client_id),
            {"update_digest": update_digest, "client_signature": signature_hex},
        )

    def record_round(self, round_id: int, commitment: dict) -> SecureBlock:
        return self._finalize(round_id, "round_commitment", "aggregator", commitment)

    def failures(self) -> list[str]:
        problems, previous = [], "0" * 64
        for index, block in enumerate(self.blocks):
            payload = asdict(block)
            actual = payload.pop("block_hash")
            votes = payload.pop("validator_votes")
            if actual != _hash(payload):
                problems.append(f"block {index}: hash mismatch")
            if block.block_id != index:
                problems.append(f"block {index}: wrong block_id")
            if block.previous_hash != previous:
                problems.append(f"block {index}: broken chain")
            previous = block.block_hash
            valid = {
                name
                for name, signature in votes
                if name in self.validator_keys
                and verify_with_public_hex(
                    self.validator_keys[name], block.block_hash.encode(), signature
                )
            }
            if len(valid) < self.majority:
                problems.append(f"block {index}: fewer than majority valid votes")
            if block.kind == "client_update":
                key = self.client_keys.get(block.subject)
                message = update_message(
                    block.round_id, block.subject, block.body.get("update_digest", "")
                )
                if key is None or not verify_with_public_hex(
                    key, message, block.body.get("client_signature", "")
                ):
                    problems.append(f"block {index}: invalid client signature")
        if any(replica != self.blocks for replica in self.replicas.values()):
            problems.append("replica divergence")
        return problems

    def validate(self) -> bool:
        return not self.failures()

    def registry(self) -> dict[str, Any]:
        return {
            "validators": self.validator_keys,
            "clients": self.client_keys,
            "majority": self.majority,
        }

    def to_blocks(self) -> list[dict[str, Any]]:
        return [asdict(block) for block in self.blocks]

    @classmethod
    def from_blocks(
        cls,
        items,
        registry: Mapping[str, Any],
        signers: Mapping[str, Identity] | None = None,
    ) -> "SecureLedger":
        ledger = cls(
            registry["validators"], registry["clients"], registry["majority"], signers
        )
        for item in items:
            votes = tuple(
                (name, signature) for name, signature in item["validator_votes"]
            )
            block = SecureBlock(**{**item, "validator_votes": votes})
            ledger.blocks.append(block)
            for replica in ledger.replicas.values():
                replica.append(block)
        problems = ledger.failures()
        if problems:
            raise ConsensusError(f"Restored ledger failed validation: {problems[:3]}")
        return ledger

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_blocks(), indent=2, default=list) + "\n")
        path.with_name(path.stem + "_registry.json").write_text(
            json.dumps(self.registry(), indent=2) + "\n"
        )

