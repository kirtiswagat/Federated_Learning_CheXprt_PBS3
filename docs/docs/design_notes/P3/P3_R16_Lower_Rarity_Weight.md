Create production-quality Python code for experiment R16.

Objective:
Evaluate reduced emphasis on rare pathologies.

Configuration:

CBRA

Dirichlet alpha=0.1

alpha_n = 1.0
beta_q = 0.5
gamma_d = 0.5

Track:

rarity score
weight allocation
client contributions

Measure:

macro AUROC
macro AUPRC
per-label AUROC

Store:

r16_results.csv

Generate:

rarity_weight_analysis.png