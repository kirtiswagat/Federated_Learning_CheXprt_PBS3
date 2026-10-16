
Create production-quality Python code for experiment R14.

Objective:
Evaluate lower sample-size weighting.

Configuration:

CBRA

Dirichlet alpha=0.1

alpha_n = 0.5
beta_q = 1.0
gamma_d = 0.5

All other settings identical to R12.

Measure:

weight distribution
round metrics
macro AUROC
macro AUPRC

Determine whether reducing sample-count influence
changes aggregation behavior.

Export:

r14_results.csv

Generate:

sample_weight_sensitivity.png