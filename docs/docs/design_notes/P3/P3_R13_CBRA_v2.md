Create production-quality Python code for experiment R13.

Objective:
Implement and validate CBRA v2.

Configuration:

CBRA v2

Partition:
Dirichlet alpha=0.1

Clients:
5

Rounds:
5

Parameters:

alpha_n = 1.0
beta_q = 1.5
gamma_d = 0.5

Difference from R12:

Increase rarity emphasis.

Measure:

per-label AUROC
macro AUROC
per-label AUPRC
macro AUPRC

Compare:

R12
R13

Generate:

cbra_v1_vs_v2.csv

Create visualization:

rarity_weight_sensitivity.png