Create Python code for experiment R09.

Goal:
FedProx lower regularization sensitivity.

Configuration:

    FedProx
    Dirichlet alpha=0.1
    5 clients

Hyperparameters:

    mu=0.0001

Training:

    5 rounds
    1 local epoch

Use:

    identical initialization
    identical partition
    identical preprocessing

Collect:

    macro AUROC
    macro AUPRC
    runtime
    communication volume

Export:

    training_summary.csv
    per_label_metrics.csv
    predictions.csv

Save:

    r09_results.json