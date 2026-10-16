Create Python code for experiment R08.

Goal:
Evaluate client subsampling.

Configuration:

    FedAvg
    IID partition
    20 total clients

Training:

    20 rounds

Participation rate:

    10 selected clients
    per round

Selection:

    random
    reproducible

Track:

    selected clients
    participation frequency
    communication volume
    training runtime
    macro AUROC
    macro AUPRC

Calculate:

    processed examples

Compare against:

    R07

Generate:

    client_participation.csv
    participation_heatmap.png
    convergence_curve.png

Output:

    r08_summary.json
