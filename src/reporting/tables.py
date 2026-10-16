"""Small table adapters for already validated CSV artifacts."""

from pathlib import Path
import pandas as pd


def load_validated_table(
    path: Path, required_columns: set[str] | None = None
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = (required_columns or set()) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing table columns: {sorted(missing)}")
    return frame
