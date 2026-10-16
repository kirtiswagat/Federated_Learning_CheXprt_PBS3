import pytest

from src.blockchain.permissioned_ledger import ConsensusError, PermissionedLedger


def test_consensus_requires_majority():
    ledger = PermissionedLedger()
    with pytest.raises(ConsensusError):
        ledger.append(1, "a" * 64, votes=("validator_1",))
