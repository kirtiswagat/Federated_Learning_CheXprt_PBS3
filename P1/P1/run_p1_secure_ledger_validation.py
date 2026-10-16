"""Secure-ledger validation on real-size client updates: detection, provenance and overhead."""

from __future__ import annotations

import argparse
import copy
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd

from src.blockchain.identity import Identity
from src.blockchain.secure_ledger import SecureLedger, SignatureError, _hash
from src.blockchain.transactions import update_digest
from src.reproducibility import load_checkpoint_cpu

VALIDATORS = ("validator_1", "validator_2", "validator_3")


def _ledger(clients: int):
    validators = {n: Identity.generate(n) for n in VALIDATORS}
    members = [Identity.generate(f"client_{i}") for i in range(clients)]
    ledger = SecureLedger(
        {n: i.public_hex for n, i in validators.items()},
        {str(k): m.public_hex for k, m in enumerate(members)},
        2,
        validators,
    )
    return ledger, members, validators


def _commit_round(ledger, members, updates, round_id=1):
    digests = {}
    for index, member in enumerate(members):
        digest = update_digest(updates[index])
        digests[index] = digest
        ledger.record_update(
            round_id, index, digest, ledger.sign(member, round_id, index, digest)
        )
    ledger.record_round(round_id, {"client_update_digests": list(digests.values())})
    return digests


def _raises(function) -> bool:
    try:
        function()
    except Exception:
        return True
    return False


def _tampered(update):
    changed = copy.deepcopy(update)
    key = next(k for k, v in changed.items() if v.is_floating_point())
    changed[key] = changed[key] + 1e-3
    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-round", type=Path, required=True)
    parser.add_argument(
        "--poisoned-dir", type=Path, help="Output of run_p4_poisoned_updates.py"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    updates = [
        load_checkpoint_cpu(args.source_round / f"client_{i}_update.pt")
        for i in range(args.clients)
    ]
    checks = []

    def record(name: str, expected: str, observed: bool) -> None:
        checks.append(
            {
                "check": name,
                "expected": expected,
                "outcome": "PASS" if observed else "FAIL",
            }
        )
        print(f"[secure-ledger] {name}: {checks[-1]['outcome']}", flush=True)

    ledger, members, validators = _ledger(args.clients)
    digests = _commit_round(ledger, members, updates)
    record("honest_round_validates", "ledger valid", ledger.validate())

    restored = SecureLedger.from_blocks(ledger.to_blocks(), ledger.registry())
    record("replication_and_restore", "restored ledger valid", restored.validate())

    mutated = copy.deepcopy(ledger)
    mutated.blocks[0].body["update_digest"] = "f" * 64
    record("block_mutation", "detected", not mutated.validate())

    forged = ledger.sign(members[1], 1, 0, digests[0])
    record(
        "client_signature_substitution",
        "rejected before commitment",
        _raises(
            lambda: _ledger(args.clients)[0].record_update(1, 0, digests[0], forged)
        ),
    )

    fresh, fresh_members, _ = _ledger(args.clients)
    replayed = fresh.sign(fresh_members[0], 1, 0, digests[0])
    record(
        "cross_round_replay",
        "rejected",
        _raises(lambda: fresh.record_update(2, 0, digests[0], replayed)),
    )

    record(
        "unsigned_update_substitution",
        "rejected",
        _raises(
            lambda: fresh.record_update(
                1, 0, update_digest(_tampered(updates[0])), replayed
            )
        ),
    )

    rogue = Identity.generate("validator_1")
    spoofed = copy.deepcopy(ledger)
    block = spoofed.blocks[0]
    votes = tuple(
        (n, rogue.sign(block.block_hash.encode()).hex())
        for n, _ in block.validator_votes
    )
    spoofed.blocks[0] = replace(block, validator_votes=votes)
    record("forged_validator_votes", "detected", not spoofed.validate())

    # One compromised validator rewrites a block and re-signs it; two of three votes are missing.
    one_key = copy.deepcopy(ledger)
    last = one_key.blocks[-1]
    payload = {
        k: v
        for k, v in last.__dict__.items()
        if k not in ("block_hash", "validator_votes")
    }
    payload["body"] = {"client_update_digests": ["0" * 64]}
    new_hash = _hash(payload)
    only = (("validator_1", validators["validator_1"].sign(new_hash.encode()).hex()),)
    rewritten = replace(
        last, body=payload["body"], block_hash=new_hash, validator_votes=only
    )
    one_key.blocks[-1] = rewritten
    for replica in one_key.replicas.values():
        replica[-1] = rewritten
    record(
        "single_compromised_validator",
        "detected (minority of valid votes)",
        not one_key.validate(),
    )

    received = _tampered(updates[0])
    record(
        "post_signing_update_tampering",
        "received update digest differs from the committed digest",
        update_digest(received) != digests[0],
    )
    record(
        "untampered_update_matches_commitment",
        "digest equal",
        update_digest(updates[0]) == digests[0],
    )

    provenance = []
    if args.poisoned_dir:
        for attack in ("label_flipping", "targeted_backdoor"):
            for path in sorted((args.poisoned_dir / attack).glob("client_*_update.pt")):
                client = int(path.stem.split("_")[1])
                poisoned = load_checkpoint_cpu(path)
                ledger_p, members_p, _ = _ledger(args.clients)
                digest = update_digest(poisoned)
                ledger_p.record_update(
                    1,
                    client,
                    digest,
                    ledger_p.sign(members_p[client], 1, client, digest),
                )
                accepted = ledger_p.validate()
                flipped = {
                    k: -v if v.is_floating_point() else v for k, v in poisoned.items()
                }
                provenance.append(
                    {
                        "attack": attack,
                        "client": client,
                        "validly_signed_poisoned_update_accepted": accepted,
                        "post_signing_modification_detected": update_digest(flipped)
                        != digest,
                    }
                )
        pd.DataFrame(provenance).to_csv(
            args.output_dir / "poisoning_provenance.csv", index=False
        )

    rows = []
    for _ in range(args.repeats):
        ledger_t, members_t, _ = _ledger(args.clients)
        for round_id in (1,):
            sizes = sum(v.numel() * v.element_size() for v in updates[0].values())
            start = time.perf_counter()
            digest = update_digest(updates[0])
            hashing = time.perf_counter() - start
            start = time.perf_counter()
            signature = ledger_t.sign(members_t[0], round_id, 0, digest)
            signing = time.perf_counter() - start
            start = time.perf_counter()
            ledger_t.record_update(round_id, 0, digest, signature)
            consensus = time.perf_counter() - start
            start = time.perf_counter()
            ledger_t.validate()
            validation = time.perf_counter() - start
        rows.append(
            {
                "update_megabytes": sizes / 1e6,
                "digest_seconds": hashing,
                "client_sign_seconds": signing,
                "verify_and_consensus_seconds": consensus,
                "ledger_validate_seconds": validation,
                "block_bytes": len(str(ledger_t.to_blocks()[0]).encode()),
            }
        )
    overhead = pd.DataFrame(rows)
    overhead.to_csv(args.output_dir / "overhead_real_updates.csv", index=False)
    pd.DataFrame(checks).to_csv(args.output_dir / "validation_checks.csv", index=False)
    print(pd.DataFrame(checks).to_string(index=False))
    print(overhead.mean().to_string())
    if any(check["outcome"] == "FAIL" for check in checks):
        raise SystemExit("secure-ledger validation failed")


if __name__ == "__main__":
    main()
