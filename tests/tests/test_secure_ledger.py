import copy
from dataclasses import replace

import pytest

pytest.importorskip("cryptography")

from src.blockchain.consensus import ConsensusError  # noqa: E402
from src.blockchain.identity import Identity  # noqa: E402
from src.blockchain.secure_ledger import SecureLedger, SignatureError  # noqa: E402


def _setup(clients=2):
    validators = {
        f"validator_{i}": Identity.generate(f"validator_{i}") for i in range(1, 4)
    }
    members = [Identity.generate(f"client_{i}") for i in range(clients)]
    ledger = SecureLedger(
        {n: i.public_hex for n, i in validators.items()},
        {str(k): m.public_hex for k, m in enumerate(members)},
        2,
        validators,
    )
    return ledger, members, validators


def _commit(ledger, members, round_id=1):
    for index, member in enumerate(members):
        digest = f"{index:064x}"
        ledger.record_update(
            round_id, index, digest, ledger.sign(member, round_id, index, digest)
        )
    ledger.record_round(round_id, {"global_output_digest": "a" * 64})


def test_valid_ledger_passes_and_restores():
    ledger, members, validators = _setup()
    _commit(ledger, members)
    assert ledger.validate()
    restored = SecureLedger.from_blocks(
        ledger.to_blocks(), ledger.registry(), validators
    )
    assert restored.validate() and len(restored.blocks) == 3


def test_unsigned_or_forged_client_update_is_rejected_before_commit():
    ledger, members, _ = _setup()
    digest = "b" * 64
    with pytest.raises(SignatureError):
        ledger.record_update(1, 0, digest, ledger.sign(members[1], 1, 0, digest))
    assert ledger.blocks == []


def test_signature_is_bound_to_round_and_client():
    ledger, members, _ = _setup()
    digest = "c" * 64
    replayed = ledger.sign(members[0], 1, 0, digest)
    with pytest.raises(SignatureError):
        ledger.record_update(2, 0, digest, replayed)


def test_post_commit_tamper_is_detected():
    ledger, members, _ = _setup()
    _commit(ledger, members)
    ledger.blocks[0].body["update_digest"] = "f" * 64
    assert not ledger.validate()


def test_rehashed_forgery_without_validator_keys_is_detected():
    ledger, members, validators = _setup()
    _commit(ledger, members)
    forged = SecureLedger(ledger.validator_keys, ledger.client_keys, 2, {})
    for block in ledger.blocks:
        forged.blocks.append(block)
        for replica in forged.replicas.values():
            replica.append(block)
    last = forged.blocks[-1]
    payload = {
        k: v
        for k, v in last.__dict__.items()
        if k not in ("block_hash", "validator_votes")
    }
    payload["body"] = {"global_output_digest": "0" * 64}
    from src.blockchain.secure_ledger import _hash

    tampered = replace(last, body=payload["body"], block_hash=_hash(payload))
    forged.blocks[-1] = tampered
    for replica in forged.replicas.values():
        replica[-1] = tampered
    assert any("majority" in problem for problem in forged.failures())


def test_validator_vote_from_unregistered_key_does_not_count():
    ledger, members, validators = _setup()
    _commit(ledger, members)
    rogue = Identity.generate("validator_1")
    block = ledger.blocks[0]
    votes = tuple(
        (name, rogue.sign(block.block_hash.encode()).hex())
        for name, _ in block.validator_votes
    )
    ledger.blocks[0] = replace(block, validator_votes=votes)
    assert not ledger.validate()


def test_majority_not_reached_without_enough_signers():
    ledger, members, validators = _setup()
    ledger.signers = {"validator_1": validators["validator_1"]}
    digest = "d" * 64
    with pytest.raises(ConsensusError):
        ledger.record_update(1, 0, digest, ledger.sign(members[0], 1, 0, digest))


def test_replica_divergence_is_detected():
    ledger, members, _ = _setup()
    _commit(ledger, members)
    ledger.replicas["validator_2"].pop()
    assert not ledger.validate()


def test_private_key_round_trip():
    identity = Identity.generate("x")
    clone = Identity.from_private_hex("x", identity.private_hex())
    assert clone.public_hex == identity.public_hex
    assert copy.copy(identity.public_hex) == clone.public_hex
