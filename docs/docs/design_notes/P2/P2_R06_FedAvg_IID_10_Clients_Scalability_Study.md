Create production-quality Python code for experiment R06.

Objective:
Evaluate federated-learning scalability using 10 IID clients.

Use existing project infrastructure.

Requirements:

Dataset:
    Existing CheXpert patient-level partition system.

Partition:
    IID
    10 clients

Model:
    ResNet50
    ImageNet initialization
    14 outputs
    dropout=0.3

Training:
    FedAvg
    10 communication rounds
    1 local epoch
    batch size=16
    AdamW
    learning rate=1e-4
    weight decay=1e-5

Rules:

    Use frozen initialization.
    Use frozen preprocessing.
    Use FP32.
    No validation-based early stopping.

For every round:

    train clients
    aggregate updates
    evaluate validation set

Collect:

    train loss
    validation loss
    macro AUROC
    macro AUPRC
    runtime
    processed examples
    communication volume

Save:

    round_history.csv
    best_valid_predictions.csv
    per_label_metrics.csv
    training_summary.csv

Generate:

    convergence_curve.png
    convergence_curve.pdf

Output:

    r06_summary.json

Code must support resume checkpointing.