import pandas as pd

from src.evaluation.bootstrap import patient_bootstrap


def test_bootstrap_requires_patient_ids():
    frame = pd.DataFrame({"target_A": [0, 1], "prob_A": [0.1, 0.9]})
    import pytest

    with pytest.raises(ValueError, match="patient_id"):
        patient_bootstrap(frame, ("A",), iterations=2)
