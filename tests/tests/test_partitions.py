import numpy as np
import pandas as pd

from src.config import LABELS
from src.data.partitions import make_patient_partition


def test_partition_keeps_patient_rows_together():
    frame = pd.DataFrame(
        {"patient_id": ["p1", "p1", "p2", "p3"], "label": [0, 1, 0, 1]}
    )
    result = make_patient_partition(frame, 2, seed=10)
    assert result.groupby("patient_id")["client_id"].nunique().max() == 1


def test_dirichlet_gives_every_client_data_and_skews_labels():
    rng = np.random.default_rng(0)
    n = 2000
    frame = pd.DataFrame({"patient_id": [f"p{i}" for i in range(n)]})
    for label in LABELS:
        frame[label] = rng.integers(0, 2, n)
    result = make_patient_partition(frame, 5, "dirichlet", alpha=0.1, min_patients=20)
    counts = result.groupby("client_id").size()
    assert len(counts) == 5 and counts.min() >= 20
    assert result.groupby("patient_id")["client_id"].nunique().max() == 1
