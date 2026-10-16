import numpy as np

from src.evaluation.metrics import evaluate_predictions


def test_undefined_auc_is_nan_not_zero():
    per_label, _ = evaluate_predictions(
        np.array([[1], [1]], float), np.array([[0.8], [0.9]]), ("A",)
    )
    assert np.isnan(per_label.loc[0, "auroc"])
