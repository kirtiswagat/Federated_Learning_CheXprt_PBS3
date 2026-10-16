import pandas as pd

from src.data.heterogeneity import analyze_heterogeneity


def test_heterogeneity_has_one_row_per_client():
    frame = pd.DataFrame(
        {"patient_id": ["p1", "p2"], "client_id": [0, 1], "A": [1, 0], "B": [0, 1]}
    )
    assert len(analyze_heterogeneity(frame, ("A", "B"))) == 2
