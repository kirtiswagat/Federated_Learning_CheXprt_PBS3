Create production-quality Python code for experiment R07.

Objective:
Measure scalability when increasing IID client count to 20.

Reuse the existing FedAvg pipeline.

Configuration:

    FedAvg
    IID partition
    20 clients
    10 rounds

Keep identical:

    model
    seed
    preprocessing
    optimizer
    local epochs

Measure:

    convergence
    communication volume
    runtime
    processed examples

Generate:

    round-wise metrics

Compare against:

    R01 (5 clients)
    R06 (10 clients)

Create:

    scalability_summary.csv

Export:

    convergence curves
    communication comparison plots