from dataclasses import replace

from src.blockchain.permissioned_ledger import PermissionedLedger


def test_block_mutation_invalidates_ledger():
    ledger = PermissionedLedger()
    ledger.append(1, "a" * 64)
    ledger.blocks[0] = replace(ledger.blocks[0], transaction_hash="b" * 64)
    assert not ledger.validate()
