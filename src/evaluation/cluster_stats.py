"""Patient-cluster bootstrap and paired comparisons on saved validation predictions."""

from __future__ import annotations

import json
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import LABELS
from src.evaluation.multiplicity import benjamini_hochberg

COMPETITION5 = (
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
)
OPERATING = ("sensitivity", "specificity", "ppv", "npv", "f1")
LOW_SUPPORT = 50
CHUNK = 25

_DATA: dict = {}


def metric_names(labels: tuple[str, ...] = LABELS) -> list[str]:
    return (
        [f"auroc|{label}" for label in labels]
        + [f"auprc|{label}" for label in labels]
        + ["macro_auroc_14", "macro_auprc_14", "macro_auroc_5comp", "macro_auprc_5comp"]
        + [f"macro_{name}" for name in OPERATING]
    )


def load_predictions(path: Path, labels: tuple[str, ...] = LABELS):
    frame = pd.read_csv(path)
    targets = frame[[f"y_{label}" for label in labels]].to_numpy(float)
    probabilities = frame[[f"p_{label}" for label in labels]].to_numpy(float)
    return (
        frame["Path"].to_numpy(),
        frame["patient_id"].to_numpy(),
        targets,
        probabilities,
    )


def _nanmean(values) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")


def metric_vector(targets, probabilities, weights, labels, threshold) -> np.ndarray:
    """All metrics for one (re)sample; weights are patient-resample multiplicities."""
    keep = weights > 0
    t, p, w = targets[keep], probabilities[keep], weights[keep].astype(float)
    nan = float("nan")
    auroc, auprc = [], []
    operating = {name: [] for name in OPERATING}
    for j in range(len(labels)):
        y = t[:, j] == 1
        positives, negatives = w[y].sum(), w[~y].sum()
        if positives == 0 or negatives == 0:
            auroc.append(nan)
            auprc.append(nan)
        else:
            auroc.append(float(roc_auc_score(y, p[:, j], sample_weight=w)))
            auprc.append(float(average_precision_score(y, p[:, j], sample_weight=w)))
        predicted = p[:, j] >= threshold
        tp = w[y & predicted].sum()
        fp = w[~y & predicted].sum()
        fn = w[y & ~predicted].sum()
        tn = w[~y & ~predicted].sum()
        operating["sensitivity"].append(tp / (tp + fn) if tp + fn else nan)
        operating["specificity"].append(tn / (tn + fp) if tn + fp else nan)
        operating["ppv"].append(tp / (tp + fp) if tp + fp else nan)
        operating["npv"].append(tn / (tn + fn) if tn + fn else nan)
        operating["f1"].append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else nan)
    comp = [labels.index(label) for label in COMPETITION5]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        macros = [
            _nanmean(auroc),
            _nanmean(auprc),
            _nanmean([auroc[i] for i in comp]),
            _nanmean([auprc[i] for i in comp]),
            *[_nanmean(operating[name]) for name in OPERATING],
        ]
    return np.array([*auroc, *auprc, *macros], dtype=float)


def _init_worker(paths: dict, labels, threshold) -> None:
    sets = {name: load_predictions(Path(path), labels) for name, path in paths.items()}
    names = list(sets)
    reference = sets[names[0]]
    for name in names[1:]:
        if not np.array_equal(sets[name][0], reference[0]):
            raise ValueError(f"{name} was evaluated on a different image order")
    _, inverse = np.unique(reference[1], return_inverse=True)
    _DATA.update(
        names=names,
        sets=sets,
        inverse=inverse,
        patients=int(inverse.max()) + 1,
        labels=labels,
        threshold=threshold,
    )


def _run_chunk(start: int, stop: int, seed: int) -> tuple[int, np.ndarray]:
    out = []
    for iteration in range(start, stop):
        rng = np.random.default_rng([seed, iteration])
        counts = np.bincount(
            rng.integers(0, _DATA["patients"], _DATA["patients"]),
            minlength=_DATA["patients"],
        )
        weights = counts[_DATA["inverse"]]
        out.append(
            [
                metric_vector(
                    _DATA["sets"][name][2],
                    _DATA["sets"][name][3],
                    weights,
                    _DATA["labels"],
                    _DATA["threshold"],
                )
                for name in _DATA["names"]
            ]
        )
    return start, np.array(out)


def _format_time(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m{seconds % 60:02d}s"


def bootstrap_draws(
    paths: dict[str, Path],
    labels: tuple[str, ...] = LABELS,
    iterations: int = 2000,
    seed: int = 42,
    workers: int = 1,
    threshold: float = 0.5,
):
    """Returns observed (sets x metrics) and draws (iterations x sets x metrics)."""
    paths = {name: str(path) for name, path in paths.items()}
    starts = list(range(0, iterations, CHUNK))
    draws = None
    began = time.perf_counter()
    done = 0

    def store(start: int, block: np.ndarray) -> None:
        nonlocal draws, done
        if draws is None:
            draws = np.full((iterations, *block.shape[1:]), np.nan)
        draws[start : start + len(block)] = block
        done += len(block)
        elapsed = time.perf_counter() - began
        print(
            f"[stats] bootstrap {done}/{iterations} | elapsed {_format_time(elapsed)}"
            f" | ETA {_format_time(elapsed / done * (iterations - done))}",
            flush=True,
        )

    if workers <= 1:
        _init_worker(paths, labels, threshold)
        for start in starts:
            store(*_run_chunk(start, min(start + CHUNK, iterations), seed))
    else:
        with ProcessPoolExecutor(
            workers, initializer=_init_worker, initargs=(paths, labels, threshold)
        ) as pool:
            futures = [
                pool.submit(_run_chunk, start, min(start + CHUNK, iterations), seed)
                for start in starts
            ]
            for future in as_completed(futures):
                store(*future.result())
        _init_worker(paths, labels, threshold)
    ones = np.ones(len(_DATA["inverse"]))
    observed = np.array(
        [
            metric_vector(
                _DATA["sets"][name][2], _DATA["sets"][name][3], ones, labels, threshold
            )
            for name in _DATA["names"]
        ]
    )
    return list(_DATA["names"]), observed, draws


def _interval(values: np.ndarray, confidence: float) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan")
    tail = (1 - confidence) * 50
    return float(np.percentile(values, tail)), float(np.percentile(values, 100 - tail))


def confidence_table(names, metrics, observed, draws, confidence=0.95) -> pd.DataFrame:
    rows = []
    for s, name in enumerate(names):
        for m, metric in enumerate(metrics):
            low, high = _interval(draws[:, s, m], confidence)
            rows.append(
                {
                    "model": name,
                    "metric": metric,
                    "estimate": observed[s, m],
                    "ci_low": low,
                    "ci_high": high,
                    "bootstrap_defined": int(np.isfinite(draws[:, s, m]).sum()),
                }
            )
    return pd.DataFrame(rows)


def paired_table(
    names, metrics, observed, draws, comparisons, families, confidence=0.95
) -> pd.DataFrame:
    """Paired patient-resampled differences; p-value is two-sided from the draws."""
    rows = []
    for family in families:
        m = metrics.index(family)
        family_rows = []
        for left, right in comparisons:
            a, b = names.index(left), names.index(right)
            difference = draws[:, a, m] - draws[:, b, m]
            difference = difference[np.isfinite(difference)]
            low, high = _interval(difference, confidence)
            lower_tail = int(np.sum(difference <= 0))
            upper_tail = int(np.sum(difference >= 0))
            p_value = min(
                1.0, 2 * (min(lower_tail, upper_tail) + 1) / (len(difference) + 1)
            )
            family_rows.append(
                {
                    "comparison": f"{left} vs {right}",
                    "metric": family,
                    "difference": float(observed[a, m] - observed[b, m]),
                    "ci_low": low,
                    "ci_high": high,
                    "p_value": p_value,
                }
            )
        adjusted = benjamini_hochberg([row["p_value"] for row in family_rows])
        for row, value in zip(family_rows, adjusted):
            row["p_value_bh"] = float(value)
            row["ci_excludes_zero"] = bool(row["ci_low"] > 0 or row["ci_high"] < 0)
        rows.extend(family_rows)
    return pd.DataFrame(rows)


def label_support(targets, patient_ids, labels=LABELS) -> pd.DataFrame:
    rows = []
    for j, label in enumerate(labels):
        positive = targets[:, j] == 1
        rows.append(
            {
                "label": label,
                "positives": int(positive.sum()),
                "negatives": int((~positive).sum()),
                "patients_with_positive": int(len(np.unique(patient_ids[positive]))),
                "auroc_defined": bool(positive.any() and (~positive).any()),
                "low_support": bool(positive.sum() < LOW_SUPPORT),
            }
        )
    return pd.DataFrame(rows)


def metric_definitions(threshold: float, iterations: int, seed: int) -> dict:
    return {
        "unit_of_resampling": "patient (all images of a patient move together)",
        "bootstrap": {
            "iterations": iterations,
            "seed": seed,
            "interval": "95% percentile",
        },
        "macro_auroc_14": "unweighted mean of the 14 per-label AUROCs; a label is skipped in a resample only when it has no positives or no negatives",
        "macro_auroc_5comp": "mean AUROC over the 5 CheXpert competition labels: "
        + ", ".join(COMPETITION5),
        "fracture": "kept in the 14-label macro; reported separately with CI; flagged low_support when positives < "
        + str(LOW_SUPPORT),
        "operating_metrics": f"sensitivity, specificity, PPV, NPV and F1 at probability threshold {threshold}, macro-averaged over labels",
        "labels": "U-Zero: uncertain (-1) and unmentioned labels are negatives",
        "paired_comparison": "same patient resample applied to both models; two-sided p-value from the paired differences; Benjamini-Hochberg within each metric family",
    }


def run_statistics(
    paths: dict[str, Path],
    output_dir: Path,
    comparisons: list[tuple[str, str]],
    labels: tuple[str, ...] = LABELS,
    iterations: int = 2000,
    seed: int = 42,
    workers: int = 1,
    threshold: float = 0.5,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    names, observed, draws = bootstrap_draws(
        paths, labels, iterations, seed, workers, threshold
    )
    metrics = metric_names(labels)
    table = confidence_table(names, metrics, observed, draws)
    families = ["macro_auroc_14", "macro_auprc_14", "macro_auroc_5comp"]
    paired = paired_table(names, metrics, observed, draws, comparisons, families)
    _, patient_ids, targets, _ = load_predictions(
        Path(next(iter(paths.values()))), labels
    )
    outputs = {
        "metrics_ci": output_dir / "metrics_with_ci.csv",
        "paired": output_dir / "paired_comparisons.csv",
        "support": output_dir / "label_support.csv",
        "draws": output_dir / "bootstrap_draws.csv",
        "definitions": output_dir / "metric_definitions.json",
    }
    table.to_csv(outputs["metrics_ci"], index=False)
    paired.to_csv(outputs["paired"], index=False)
    label_support(targets, patient_ids, labels).to_csv(outputs["support"], index=False)
    wide = pd.DataFrame(
        draws.reshape(iterations, -1),
        columns=[f"{name}|{metric}" for name in names for metric in metrics],
    )
    wide.insert(0, "iteration", range(iterations))
    wide.to_csv(outputs["draws"], index=False)
    outputs["definitions"].write_text(
        json.dumps(metric_definitions(threshold, iterations, seed), indent=2) + "\n"
    )
    return outputs
