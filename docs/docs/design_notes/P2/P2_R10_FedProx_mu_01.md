Create Python code for experiment R10.

Goal:
Measure FedProx sensitivity at mu=0.01.

Use:

    same Dirichlet alpha=0.1 partition

Configuration:

    FedProx
    mu=0.01
    5 rounds

Reuse existing federated runner.

Collect:

    per-round metrics
    runtime
    communication

Generate:

    FedProx sensitivity outputs

Save:

    r10_results.json