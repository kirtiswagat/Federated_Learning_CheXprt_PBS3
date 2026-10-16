import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import LABELS
from src.evaluation.cluster_stats import (
    bootstrap_draws,
    metric_names,
    metric_vector,
    paired_table,
)


def _write(path, patients, targets, probabilities):
    frame = pd.DataFrame(
        {"Path": [f"img{i}" for i in range(len(patients))], "patient_id": patients}
    )
    for j, label in enumerate(LABELS):
        frame[f"y_{label}"] = targets[:, j]
    for j, label in enumerate(LABELS):
        frame[f"p_{label}"] = probabilities[:, j]
    frame.to_csv(path, index=False)


def _data(seed=0, n=400):
    rng = np.random.default_rng(seed)
    targets = rng.integers(0, 2, (n, len(LABELS))).astype(float)
    targets[:, 0] = 0.0  # label without positives must be undefined, not an error
    patients = np.repeat([f"p{i}" for i in range(n // 4)], 4)
    return patients, targets, rng


def test_unit_weights_match_sklearn():
    _, targets, rng = _data()
    probabilities = rng.random(targets.shape)
    vector = metric_vector(targets, probabilities, np.ones(len(targets)), LABELS, 0.5)
    names = metric_names()
    assert np.isnan(vector[names.index("auroc|" + LABELS[0])])
    expected = roc_auc_score(targets[:, 3], probabilities[:, 3])
    assert abs(vector[names.index("auroc|" + LABELS[3])] - expected) < 1e-9


def test_paired_bootstrap_detects_better_model(tmp_path):
    patients, targets, rng = _data()
    good = np.clip(targets * 0.4 + rng.random(targets.shape) * 0.6, 0, 1)
    bad = rng.random(targets.shape)
    _write(tmp_path / "good.csv", patients, targets, good)
    _write(tmp_path / "bad.csv", patients, targets, bad)
    names, observed, draws = bootstrap_draws(
        {"good": tmp_path / "good.csv", "bad": tmp_path / "bad.csv"}, iterations=50
    )
    paired = paired_table(
        names, metric_names(), observed, draws, [("good", "bad")], ["macro_auroc_14"]
    )
    row = paired.iloc[0]
    assert row.difference > 0.1 and row.ci_low > 0 and row.ci_excludes_zero
    assert 0 < row.p_value <= 1
