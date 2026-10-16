"""Uncertain-label frequency and evaluation-side sensitivity (no training).

Models were trained with U-Zero. This re-scores their saved predictions under three evaluation
policies for uncertain (-1) ground-truth labels: U-Zero (primary), U-Ignore (image dropped for
that label) and U-One (uncertain counted positive).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import LABELS

POLICIES = ("u_zero", "u_ignore", "u_one")


def weighted_auroc(probabilities, positive, weights) -> float:
    order = np.argsort(probabilities, kind="stable")
    w_pos = (weights * positive)[order]
    w_neg = (weights * (1 - positive))[order]
    total_pos, total_neg = w_pos.sum(), w_neg.sum()
    if total_pos <= 0 or total_neg <= 0:
        return float("nan")
    return float((w_pos * np.cumsum(w_neg)).sum() / (total_pos * total_neg))


def policy_arrays(raw: np.ndarray, policy: str):
    """Returns (positive, keep) matrices for one evaluation policy."""
    uncertain = raw == -1
    if policy == "u_zero":
        return (raw == 1).astype(float), np.ones_like(raw, dtype=float)
    if policy == "u_ignore":
        return (raw == 1).astype(float), (~uncertain).astype(float)
    return ((raw == 1) | uncertain).astype(float), np.ones_like(raw, dtype=float)


def macro_auroc(probabilities, positive, keep, weights) -> tuple[float, np.ndarray]:
    values = np.array(
        [
            weighted_auroc(probabilities[:, j], positive[:, j], weights * keep[:, j])
            for j in range(positive.shape[1])
        ]
    )
    return float(np.nanmean(values)), values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base = args.root / "Output"
    out = base / "p2_uncertainty"
    out.mkdir(parents=True, exist_ok=True)
    labels = list(LABELS)

    print("[1/3] Uncertainty frequency in raw labels (frontal images)", flush=True)
    val_raw = pd.read_csv(args.data_root / "CheXpert-v1.0-small" / "val_split.csv")
    val_raw = val_raw[val_raw["Frontal/Lateral"] == "Frontal"].reset_index(drop=True)

    def frequency(frame: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for label in labels:
            column = frame[label]
            rows.append(
                {
                    "label": label,
                    "images": len(frame),
                    "positive": int((column == 1).sum()),
                    "negative": int((column == 0).sum()),
                    "uncertain": int((column == -1).sum()),
                    "unmentioned": int(column.isna().sum()),
                    "uncertain_pct_of_images": 100 * float((column == -1).mean()),
                    "uncertain_pct_of_mentioned": 100
                    * float((column == -1).sum() / max(column.notna().sum(), 1)),
                }
            )
        return pd.DataFrame(rows)

    frequency(val_raw).round(3).to_csv(
        out / "uncertainty_frequency_heldout.csv", index=False
    )
    parts = base / "p1_critical_revision" / "partitions"
    per_client = []
    for name in ("iid_5", "dirichlet_5_0.3", "dirichlet_5_0.1"):
        assigned = pd.read_csv(parts / f"{name}_client_assignment.csv")
        assigned = assigned[assigned["Frontal/Lateral"] == "Frontal"]
        for client, frame in assigned.groupby("client_id"):
            table = frequency(frame)
            table.insert(0, "client_id", client)
            table.insert(0, "partition", name)
            per_client.append(table)
    per_client = pd.concat(per_client)
    per_client.round(3).to_csv(out / "uncertainty_frequency_by_client.csv", index=False)
    spread = (
        per_client.groupby(["partition", "label"])["uncertain_pct_of_images"]
        .agg(min_pct="min", max_pct="max")
        .reset_index()
    )
    spread["range_pct_points"] = spread["max_pct"] - spread["min_pct"]
    spread.round(3).to_csv(out / "uncertainty_spread_across_clients.csv", index=False)

    print("[2/3] Evaluation-policy sensitivity on saved predictions", flush=True)
    runs = base / "p1_critical_revision" / "runs"
    paths = {
        "R01_r5": runs / "R01" / "predictions" / "round_05_valid_predictions.csv",
        "R01_r10": runs / "R01" / "endpoint_valid_predictions.csv",
        **{
            r: runs / r / "endpoint_valid_predictions.csv"
            for r in ("R02", "R03", "R04", "R05")
        },
        **{
            r: base / "p3_cbra" / r / "endpoint_valid_predictions.csv"
            for r in ("R12", "R13", "R14", "R15", "R16", "R18", "R19")
        },
    }
    raw_by_path = val_raw.set_index("Path")[labels]
    per_label_rows, macro_rows = [], []
    rng_master = np.random.default_rng(args.seed)
    for run, path in paths.items():
        if not path.is_file():
            print(f"  skipping {run}: {path} missing", flush=True)
            continue
        pred = pd.read_csv(path)
        raw = raw_by_path.reindex(pred["Path"])[labels].to_numpy(dtype=float)
        if np.isnan(raw[:, 0]).all() and not pred["Path"].isin(raw_by_path.index).any():
            raise RuntimeError(f"{run}: prediction paths do not match val_split")
        raw = np.nan_to_num(raw, nan=0.0)
        probabilities = pred[[f"p_{label}" for label in labels]].to_numpy(dtype=float)
        stored = pred[[f"y_{label}" for label in labels]].to_numpy(dtype=float)
        if not np.array_equal(stored, (raw == 1).astype(float)):
            raise RuntimeError(
                f"{run}: saved targets differ from U-Zero labels of val_split"
            )
        patients, inverse = np.unique(
            pred["patient_id"].to_numpy(), return_inverse=True
        )
        ones = np.ones(len(pred))
        for j in (0, 5):
            check = roc_auc_score((raw[:, j] == 1).astype(int), probabilities[:, j])
            assert (
                abs(
                    check
                    - weighted_auroc(
                        probabilities[:, j], (raw[:, j] == 1).astype(float), ones
                    )
                )
                < 2e-3
            )
        point = {}
        for policy in POLICIES:
            positive, keep = policy_arrays(raw, policy)
            macro, values = macro_auroc(probabilities, positive, keep, ones)
            point[policy] = macro
            for j, label in enumerate(labels):
                per_label_rows.append(
                    {
                        "run": run,
                        "policy": policy,
                        "label": label,
                        "auroc": values[j],
                        "positives": int((positive[:, j] * keep[:, j]).sum()),
                        "negatives": int(((1 - positive[:, j]) * keep[:, j]).sum()),
                        "uncertain_in_heldout": int((raw[:, j] == -1).sum()),
                    }
                )
        draws = {policy: [] for policy in POLICIES}
        rng = np.random.default_rng(rng_master.integers(2**31))
        arrays = {policy: policy_arrays(raw, policy) for policy in POLICIES}
        for _ in range(args.iterations):
            counts = np.bincount(
                rng.integers(0, len(patients), len(patients)), minlength=len(patients)
            )
            weights = counts[inverse].astype(float)
            for policy in POLICIES:
                positive, keep = arrays[policy]
                draws[policy].append(
                    macro_auroc(probabilities, positive, keep, weights)[0]
                )
        for policy in POLICIES:
            row = {
                "run": run,
                "policy": policy,
                "macro_auroc": point[policy],
                "ci_low": float(np.nanpercentile(draws[policy], 2.5)),
                "ci_high": float(np.nanpercentile(draws[policy], 97.5)),
            }
            if policy != "u_zero":
                delta = np.array(draws[policy]) - np.array(draws["u_zero"])
                row.update(
                    diff_vs_u_zero=point[policy] - point["u_zero"],
                    diff_ci_low=float(np.nanpercentile(delta, 2.5)),
                    diff_ci_high=float(np.nanpercentile(delta, 97.5)),
                )
            macro_rows.append(row)
        print(
            f"  {run}: " + ", ".join(f"{p}={point[p]:.4f}" for p in POLICIES),
            flush=True,
        )

    pd.DataFrame(per_label_rows).round(5).to_csv(
        out / "evaluation_policy_per_label.csv", index=False
    )
    macro = pd.DataFrame(macro_rows).round(5)
    macro.to_csv(out / "evaluation_policy_macro.csv", index=False)

    print(
        "[3/3] Does the ordering of methods depend on the evaluation policy?",
        flush=True,
    )
    wide = macro.pivot(index="run", columns="policy", values="macro_auroc")
    ranks = wide.rank(ascending=False)
    ranks.to_csv(out / "evaluation_policy_rank_by_run.csv")
    corr = wide.corr(method="spearman")
    corr.to_csv(out / "evaluation_policy_rank_correlation.csv")
    print(wide.round(4).to_string())
    print(corr.round(3).to_string())


if __name__ == "__main__":
    main()
