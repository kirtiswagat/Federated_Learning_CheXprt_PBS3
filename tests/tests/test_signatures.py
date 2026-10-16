import pytest


def test_signature_round_trip():
    pytest.importorskip("cryptography")
    from src.blockchain.identity import Identity
    from src.blockchain.transactions import sign_update, verify_transaction

    identity = Identity.generate("client")
    transaction = sign_update(identity, 1, "client", {"weight": [1, 2]})
    assert verify_transaction(transaction, identity)
