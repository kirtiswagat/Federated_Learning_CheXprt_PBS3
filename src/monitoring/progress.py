"""Console progress with throughput and ETA for long federated runs."""

from __future__ import annotations

import time


def _fmt(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    return f"{seconds // 3600:d}h{seconds % 3600 // 60:02d}m{seconds % 60:02d}s"


class Progress:
    """Tracks work units (batches) across the whole run to estimate the ETA."""

    def __init__(self, total_units: int, every: int = 50):
        self.total = max(total_units, 1)
        self.every = max(every, 1)
        self.done = 0
        self.images = 0
        self.start = time.perf_counter()

    def advance(self, units: int = 1, images: int = 0) -> None:
        self.done += units
        self.images += images

    def line(self, prefix: str, detail: str = "") -> None:
        elapsed = time.perf_counter() - self.start
        rate = self.done / elapsed if elapsed > 0 else 0.0
        eta = (self.total - self.done) / rate if rate > 0 else float("nan")
        img_rate = self.images / elapsed if elapsed > 0 else 0.0
        pct = 100 * self.done / self.total
        print(
            f"[{prefix}] {detail} | {pct:5.1f}% | {img_rate:5.1f} img/s | "
            f"elapsed {_fmt(elapsed)} | ETA {_fmt(eta) if eta == eta else '?'}",
            flush=True,
        )

    def tick(self, prefix: str, detail: str, index: int) -> None:
        if index % self.every == 0:
            self.line(prefix, detail)
