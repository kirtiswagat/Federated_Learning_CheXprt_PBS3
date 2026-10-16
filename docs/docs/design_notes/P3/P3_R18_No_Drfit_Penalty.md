Create production-quality Python code for experiment R18.

Objective:
Evaluate CBRA without drift suppression.

Configuration:

CBRA

Dirichlet alpha=0.1

alpha_n = 1.0
beta_q = 1.0
gamma_d = 0.0

Drift penalty must be disabled.

Track:

client update norms
model divergence
aggregation weights

Evaluate:

training stability
performance

Export:

r18_results.csv

Generate:

drift_penalty_comparison.png