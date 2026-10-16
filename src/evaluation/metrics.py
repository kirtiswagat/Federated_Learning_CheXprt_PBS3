"""Undefined-safe multi-label metrics for endpoint evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


def evaluate_predictions(
    targets: np.ndarray,
    probabilities: np.ndarray,
    labels: tuple[str, ...],
    threshold: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if targets.shape != probabilities.shape or targets.shape[1] != len(labels):
        raise ValueError("targets, probabilities, and labels have incompatible shapes")
    predictions = probabilities >= threshold
    rows = []
    for index, label in enumerate(labels):
        valid = ~np.isnan(targets[:, index])
        y_true, y_prob, y_pred = (
            targets[valid, index].astype(int),
            probabilities[valid, index],
            predictions[valid, index],
        )
        auc = (
            float("nan")
            if len(np.unique(y_true)) < 2
            else float(roc_auc_score(y_true, y_prob))
        )
        auprc = (
            float("nan")
            if len(np.unique(y_true)) < 2
            else float(average_precision_score(y_true, y_prob))
        )
        tn, fp, fn, tp = (
            confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            if len(y_true)
            else (0, 0, 0, 0)
        )
        rows.append(
            {
                "label": label,
                "auroc": auc,
                "auprc": auprc,
                "sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
                "specificity": tn / (tn + fp) if tn + fp else float("nan"),
                "ppv": tp / (tp + fp) if tp + fp else float("nan"),
                "npv": tn / (tn + fn) if tn + fn else float("nan"),
                "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else float("nan"),
                "accuracy": (tp + tn) / len(y_true) if len(y_true) else float("nan"),
                "support": int(y_true.sum()),
                "undefined_auc": len(np.unique(y_true)) < 2,
            }
        )
    per_label = pd.DataFrame(rows)
    macro = pd.DataFrame(
        [
            {
                "macro_auroc": per_label.auroc.mean(skipna=True),
                "macro_auprc": per_label.auprc.mean(skipna=True),
                "sensitivity": per_label.sensitivity.mean(skipna=True),
                "specificity": per_label.specificity.mean(skipna=True),
                "ppv": per_label.ppv.mean(skipna=True),
                "npv": per_label.npv.mean(skipna=True),
                "f1": per_label.f1.mean(skipna=True),
                "accuracy": per_label.accuracy.mean(skipna=True),
            }
        ]
    )
    return per_label, macro
