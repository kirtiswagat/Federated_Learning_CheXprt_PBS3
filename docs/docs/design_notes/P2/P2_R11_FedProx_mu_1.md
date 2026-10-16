Create Python code for experiment R11.

Goal:
Strong proximal regularization evaluation.

Configuration:

    FedProx
    mu=0.1

Partition:

    Dirichlet alpha=0.1
    5 clients

Training:

    5 rounds

Compare:

    mu=0.0001
    mu=0.001
    mu=0.01
    mu=0.1

Generate:

    fedprox_mu_sensitivity.csv

Create plots:

    mu vs AUROC
    mu vs AUPRC
    mu vs runtime

Export:

    fedprox_sensitivity_summary.json
``