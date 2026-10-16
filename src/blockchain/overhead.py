"""Microbenchmarks kept separate from model-training runtime."""

from __future__ import annotations

import hashlib
import time
import pandas as pd

from src.blockchain.identity import Identity
from src.blockchain.permissioned_ledger import PermissionedLedger


def benchmark(
    sizes: tuple[int, ...] = (100, 1000, 10000),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    identity = Identity.generate("benchmark")
    rows, growth = [], []
    for size in sizes:
        payload = b"x" * size
        start = time.perf_counter()
        hashlib.sha256(payload).hexdigest()
        hashing = time.perf_counter() - start
        start = time.perf_counter()
        signature = identity.sign(payload)
        signing = time.perf_counter() - start
        start = time.perf_counter()
        identity.verify(payload, signature)
        verification = time.perf_counter() - start
        ledger = PermissionedLedger()
        start = time.perf_counter()
        for index in range(size):
            ledger.append(index, hashlib.sha256(str(index).encode()).hexdigest())
        consensus = time.perf_counter() - start
        rows.append(
            {
                "transactions": size,
                "sha256_seconds": hashing,
                "signing_seconds": signing,
                "verification_seconds": verification,
                "consensus_seconds": consensus,
            }
        )
        growth.append(
            {"transactions": size, "ledger_bytes": len(str(ledger.blocks).encode())}
        )
    return pd.DataFrame(rows), pd.DataFrame(growth)
