Create production-quality Python code for experiment R17.

Objective:
Evaluate stronger rarity-aware weighting.

Configuration:

CBRA

Dirichlet alpha=0.1

alpha_n = 1.0
beta_q = 1.5
gamma_d = 0.5

Compare:

R16
R12
R13

Measure:

effect on rare disease labels

Create:

rarity_sensitivity_summary.csv

Generate:

beta_q_vs_performance.png