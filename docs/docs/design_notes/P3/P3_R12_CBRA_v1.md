Create production-quality Python code for experiment R12.

Objective:
Implement and validate the CBRA v1 reference model.

Reuse existing:

- CheXpert preprocessing
- Patient-level Dirichlet partition
- Federated training engine
- Evaluation pipeline
- Checkpointing
- Runtime tracking

Configuration:

Algorithm:
CBRA v1

Partition:
Dirichlet alpha=0.1

Clients:
5

Rounds:
5

Local epochs:
1

CBRA parameters:

alpha_n = 1.0
beta_q = 1.0
gamma_d = 0.5

Contribution score components:

1. Sample-count contribution
2. Rare-label contribution
3. Drift penalty

Normalize contribution scores.

Aggregate models using normalized CBRA weights.

Track per round:

client_id
sample_count
rarity_score
drift_score
raw_score
normalized_weight

Save:

cbra_round_details.csv
cbra_weights.csv
round_history.csv
training_summary.csv
per_label_metrics.csv
best_valid_predictions.csv

Generate:

contribution_distribution.png
