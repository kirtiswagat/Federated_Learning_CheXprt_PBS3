import pandas as pd
import pytest

from src.data.audit import DataAuditError, audit_dataset
from src.config import LABELS


def test_audit_rejects_patient_overlap(tmp_path):
    path = "CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg"
    columns = {"Path": [path]}
    columns.update({label: [0] for label in LABELS})
    train = tmp_path / "train.csv"
    valid = tmp_path / "valid.csv"
    pd.DataFrame(columns).to_csv(train, index=False)
    pd.DataFrame(columns).to_csv(valid, index=False)
    with pytest.raises(DataAuditError, match="leakage"):
        audit_dataset(train, valid, tmp_path)
