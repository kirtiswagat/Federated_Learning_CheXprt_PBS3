Create production-quality Python code for experiment R20.

Goal:
Evaluate uncertainty-label loss masking.

Policy:

    U-Ignore

Implementation:

    uncertain labels
    excluded from loss

Maintain:

    identical partitions
    identical initialization
    identical model
    identical training budget

Configuration:

    FedAvg
    Dirichlet alpha=0.1
    5 clients
    5 rounds

Collect:

    per-label AUROC
    macro AUROC
    per-label AUPRC
    macro AUPRC

Compare:

    U-Zero baseline

Generate:

    uncertainty_policy_comparison.csv

Save:

    r20_summary.json