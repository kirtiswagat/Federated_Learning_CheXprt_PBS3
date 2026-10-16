"""High-resolution runtime breakdown records."""

from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path
import pandas as pd


class RuntimeTracker:
    def __init__(self) -> None:
        self.records: list[dict[str, float | str]] = []

    @contextmanager
    def measure(self, name: str, **fields: float | str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.records.append(
                {
                    **fields,
                    "operation": name,
                    "seconds": time.perf_counter() - start,
                }
            )

    def dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.dataframe().to_csv(path, index=False)
