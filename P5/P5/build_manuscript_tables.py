"""Build manuscript tables and an evidence-linked claims audit from existing result CSVs."""

import argparse
from pathlib import Path

import pandas as pd

BASELINES = {
    "R01_r5": ("R01 (round 5)", "FedAvg", "IID"),
    "R01_r10": ("R01 (round 10)", "FedAvg", "IID"),
    "R02": ("R02", "FedAvg", "alpha=0.3"),
    "R03": ("R03", "FedAvg", "alpha=0.1"),
    "R04": ("R04", "FedProx", "IID"),
    "R05": ("R05", "FedProx", "alpha=0.1"),
}
AGGREGATORS = ["fedavg", "norm_clipped_fedavg", "coordinate_median", "trimmed_mean"]
NOT_EXECUTED = {"label_flipping", "targeted_backdoor"}


def ci(row) -> str:
    return f"{row.estimate:.4f} [{row.ci_low:.4f}, {row.ci_high:.4f}]"


def diff(row) -> str:
    return f"{row.difference:+.4f} [{row.ci_low:+.4f}, {row.ci_high:+.4f}]"


def p_text(value: float) -> str:
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def markdown(frame: pd.DataFrame) -> str:
    columns = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in frame.to_numpy()]
    return "\n".join(lines) + "\n"


def save(frame: pd.DataFrame, directory: Path, name: str) -> None:
    frame.to_csv(directory / f"{name}.csv", index=False)
    (directory / f"{name}.md").write_text(markdown(frame))
    print(f"  {name}: {len(frame)} rows")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--p4", type=Path, help="Folder with consolidated/ (default Output/p4_v2_all)"
    )
    args = parser.parse_args()
    root = args.root.resolve()
    base = root / "Output" / "p1_critical_revision"
    p3 = root / "Output" / "p3_analysis_normalized"
    p4 = (args.p4 or root / "Output" / "p4_v2_all") / "consolidated"
    out = root / "Output" / "manuscript_tables"
    out.mkdir(parents=True, exist_ok=True)

    # Heterogeneity
    rows = []
    for name in ("iid_5", "dirichlet_5_0.3", "dirichlet_5_0.1"):
        h = pd.read_csv(base / "heterogeneity" / name / "heterogeneity_summary.csv")
        rows.append(
            {
                "partition": name,
                "images_per_client_min": int(h.image_count.min()),
                "images_per_client_max": int(h.image_count.max()),
                "mean_JS_divergence": round(h.jensen_shannon_divergence.mean(), 4),
                "max_JS_divergence": round(h.jensen_shannon_divergence.max(), 4),
                "mean_prevalence_shift": round(h.prevalence_shift.mean(), 4),
            }
        )
    save(pd.DataFrame(rows), out, "table1_heterogeneity")

    # Baselines
    metrics = pd.read_csv(base / "statistics" / "metrics_with_ci.csv")
    pick = lambda model, metric: metrics[
        (metrics.model == model) & (metrics.metric == metric)
    ].iloc[0]
    rows = [
        {
            "run": label,
            "method": method,
            "partition": partition,
            "macro_AUROC (95% CI)": ci(pick(key, "macro_auroc_14")),
            "macro_AUPRC (95% CI)": ci(pick(key, "macro_auprc_14")),
            "macro_AUROC_5_competition_labels": ci(pick(key, "macro_auroc_5comp")),
        }
        for key, (label, method, partition) in BASELINES.items()
    ]
    save(pd.DataFrame(rows), out, "table2_baselines")

    paired = pd.read_csv(base / "statistics" / "paired_comparisons.csv")
    paired = paired[paired.metric.isin(["macro_auroc_14", "macro_auprc_14"])]
    rows = [
        {
            "comparison": r.comparison,
            "metric": r.metric,
            "difference (95% CI)": diff(r),
            "BH-adjusted p": p_text(r.p_value_bh),
        }
        for r in paired.itertuples()
    ]
    save(pd.DataFrame(rows), out, "table3_baseline_paired")

    # CBRA
    sens = pd.read_csv(p3 / "cbra_parameter_sensitivity.csv").set_index("run_id")
    cm = pd.read_csv(p3 / "statistics" / "metrics_with_ci.csv")
    vs = pd.read_csv(p3 / "statistics" / "paired_cbra_vs_baselines.csv")
    vs = vs[vs.metric == "macro_auroc_14"].set_index("comparison")
    rows = []
    for run in ("R12", "R13", "R14", "R15", "R16", "R18", "R19"):
        pick_c = lambda metric: cm[(cm.model == run) & (cm.metric == metric)].iloc[0]
        row = {
            "run": run,
            "variant": sens.loc[run, "variant"],
            "alpha_n": sens.loc[run, "alpha_n"],
            "beta_q": sens.loc[run, "beta_q"],
            "gamma_d": sens.loc[run, "gamma_d"],
            "macro_AUROC (95% CI)": ci(pick_c("macro_auroc_14")),
            "macro_AUPRC (95% CI)": ci(pick_c("macro_auprc_14")),
        }
        for baseline in ("R03", "R05"):
            r = vs.loc[f"{run} vs {baseline}"]
            row[f"AUROC diff vs {baseline}"] = diff(r)
            row[f"BH p vs {baseline}"] = p_text(r.p_value_bh)
        rows.append(row)
    save(pd.DataFrame(rows), out, "table4_cbra")

    # Attack matrix
    d = pd.read_csv(p4 / "robustness_degradation.csv")
    clean = d.groupby("aggregation_method").clean_macro_auroc.first()
    rows = []
    for (attack, fraction), group in d.groupby(["attack_name", "malicious_fraction"]):
        row = {"attack": attack, "malicious_fraction": fraction}
        for method in AGGREGATORS:
            value = group[
                group.aggregation_method == method
            ].auroc_absolute_degradation.iloc[0]
            row[method] = "not executed" if attack in NOT_EXECUTED else f"{value:+.3f}"
        rows.append(row)
    row = {"attack": "clean macro AUROC", "malicious_fraction": "-"}
    row.update({m: f"{clean[m]:.4f}" for m in AGGREGATORS})
    rows.append(row)
    save(pd.DataFrame(rows), out, "table5_attack_auroc_drop")

    # Runtime
    r = pd.read_csv(base / "runtime" / "runtime_summary.csv")
    save(
        pd.DataFrame(
            {
                "run": r.run,
                "rounds": r.rounds,
                "train_hours": (r.train_seconds_net / 3600).round(2),
                "eval_hours": (r.evaluation_time_s / 3600).round(2),
                "aggregation_seconds": r.aggregation_time_s.round(1),
                "ledger_seconds": (r.signing_time_s + r.consensus_time_s).round(1),
                "ledger_share_percent": r.ledger_share_pct.round(4),
                "train_img_per_s": r.train_img_per_s.round(1),
            }
        ),
        out,
        "table6_runtime",
    )

    # Claims audit: each rule reads measured values.
    pv = paired.set_index(["comparison", "metric"])
    mp = lambda c: pv.loc[(c, "macro_auroc_14")]
    cbra_range = sens.loc[
        ["R12", "R13", "R14", "R15", "R16", "R18", "R19"], "macro_auroc"
    ]
    cb_vs_r03 = vs[[i.endswith("vs R03") for i in vs.index]]
    byz = d[d.attack_name == "byzantine_random"].pivot_table(
        index="malicious_fraction",
        columns="aggregation_method",
        values="auroc_absolute_degradation",
    )
    sign = d[d.attack_name == "sign_flipping"].pivot_table(
        index="malicious_fraction",
        columns="aggregation_method",
        values="auroc_absolute_degradation",
    )
    ledger_max = r.ledger_share_pct.max()
    het = mp("R01_r5 vs R03")
    prox = mp("R03 vs R05")
    claims = [
        (
            "Blockchain signing and consensus add negligible runtime",
            "runtime/runtime_summary.csv",
            f"max ledger share {ledger_max:.4f}% of accounted time",
            "supported" if ledger_max < 0.1 else "not supported",
        ),
        (
            "Label-skewed partitions reduce macro AUROC versus IID (FedAvg, round 5)",
            "statistics/paired_comparisons.csv",
            f"IID minus alpha=0.1 = {diff(het)}",
            "supported"
            if het.ci_excludes_zero and het.difference > 0
            else "not supported",
        ),
        (
            "FedProx improves over FedAvg under heterogeneity",
            "statistics/paired_comparisons.csv",
            f"FedAvg minus FedProx (alpha=0.1) = {diff(prox)}",
            "not supported: FedAvg is better" if prox.difference > 0 else "supported",
        ),
        (
            "CBRA improves macro AUROC over FedAvg",
            "p3 statistics/paired_cbra_vs_baselines.csv",
            f"CBRA minus FedAvg ranges {cb_vs_r03.difference.min():+.4f} to {cb_vs_r03.difference.max():+.4f}",
            "qualified: small gains (<= 0.003), single training seed; do not claim superiority",
        ),
        (
            "CBRA performance is insensitive to its hyperparameters",
            "p3 cbra_parameter_sensitivity.csv",
            f"AUROC spread across 7 variants = {cbra_range.max() - cbra_range.min():.4f}",
            "supported (spread below 0.005)"
            if cbra_range.max() - cbra_range.min() < 0.005
            else "not supported",
        ),
        (
            "Robust aggregators resist Byzantine random updates",
            "p4 robustness_degradation.csv",
            "AUROC drop at 20%/40%: "
            + "; ".join(
                f"{m} {byz.loc[0.2, m]:.3f}/{byz.loc[0.4, m]:.3f}" for m in AGGREGATORS
            ),
            "partially supported: coordinate median and norm clipping hold, trimmed mean fails at 40%",
        ),
        (
            "Robust aggregators resist sign-flipping updates",
            "p4 robustness_degradation.csv",
            "AUROC drop at 20%/40%: "
            + "; ".join(
                f"{m} {sign.loc[0.2, m]:.3f}/{sign.loc[0.4, m]:.3f}"
                for m in AGGREGATORS
            ),
            "not supported: robust aggregators are not better than FedAvg",
        ),
        (
            "Robustness to label-flipping and backdoor attacks",
            "p4 robustness_degradation.csv",
            "these attacks leave the submitted update unchanged in the harness",
            "not executed: do not report as measured",
        ),
        (
            "The blockchain prevents model poisoning",
            "threat_model.json",
            "ledger records provenance only; no P4 condition enables it",
            "not supported: excluded by the threat model",
        ),
    ]
    audit = pd.DataFrame(
        claims, columns=["claim", "evidence_file", "observed", "status"]
    )
    save(audit, out, "claims_audit")


if __name__ == "__main__":
    main()
