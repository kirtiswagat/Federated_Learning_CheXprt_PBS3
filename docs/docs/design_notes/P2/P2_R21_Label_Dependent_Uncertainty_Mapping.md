Create production-quality Python code for experiment R21.

Goal:
Evaluate label-dependent uncertain-label mapping.

Policy:

    label-specific uncertainty handling

Examples:

    Cardiomegaly:
        uncertainty -> positive

    Atelectasis:
        uncertainty -> negative

Policy must be configurable.

Configuration:

    FedAvg
    Dirichlet alpha=0.1
    5 clients
    5 rounds

Keep:

    same partitions
    same initialization
    same validation cohort

Evaluate:

    per-label AUROC
    per-label AUPRC
    macro AUROC
    macro AUPRC

Compare against:

    U-Zero
    U-Ignore

Generate:

    uncertainty_policy_sensitivity.csv
    uncertainty_policy_comparison.png

Save:

    r21_summary.json