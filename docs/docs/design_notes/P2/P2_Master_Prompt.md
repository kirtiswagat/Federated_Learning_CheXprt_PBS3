You are a senior Federated Learning, PyTorch, Medical AI, Statistics,
and Experiment Reproducibility engineer.

Your task is to extend an existing federated-learning codebase for
lung disease classification using the CheXpert dataset.

Existing components already exist:

- CheXpert preprocessing
- Patient-level partitions
- ResNet50 multi-label classifier
- FedAvg
- FedProx
- Evaluation pipeline
- Per-label AUROC
- Macro AUROC
- Runtime tracking
- Communication estimation
- Checkpointing
- Prediction export
- Experiment summaries

Reuse existing infrastructure whenever possible.

Do not duplicate code.

Refactor shared logic into reusable modules.

==================================================
OBJECTIVE
==================================================

Implement all P2 experiments:

R06
R07
R08
R09
R10
R11
R20
R21

The goal is to perform:

1. Scalability analysis
2. Client participation analysis
3. FedProx sensitivity analysis
4. Uncertainty-policy sensitivity analysis
5. Processed-example budget analysis
6. Convergence analysis

==================================================
GLOBAL EXPERIMENT RULES
==================================================

Use identical:

- ResNet50
- ImageNet initialization
- preprocessing
- image size
- class weights
- optimizer
- validation dataset
- label ordering
- random seed
- training pipeline

Training settings:

Batch size = 16

Optimizer = AdamW

Learning rate = 1e-4

Weight decay = 1e-5

FP32 precision

One local epoch per round

Do not introduce:

- new architectures
- ensemble models
- transformers
- hyperparameter search

Use identical deterministic initialization.

Use fixed patient-level partitions.

==================================================
CREATE ONE GENERIC EXPERIMENT RUNNER
==================================================

Create a reusable runner supporting:

Configuration object:

experiment_name
algorithm
partition
alpha
num_clients
participation_rate
num_rounds
local_epochs
mu
uncertainty_policy

Runner must:

load partition
initialize model
load deterministic checkpoint

For round in communication_rounds:

    select clients

    train local models

    aggregate updates

    evaluate model

    record metrics

    save checkpoint

    save predictions

    save communication stats

    save runtime stats

Output:

training_summary.csv

round_history.csv

per_label_metrics.csv

macro_metrics.csv

predictions.csv

checkpoint.pt

==================================================
R06
==================================================

Experiment:

FedAvg

IID

10 clients

10 rounds

Goal:

Scalability analysis

Measure:

per-round AUROC

per-round AUPRC

runtime

communication volume

processed examples

Generate:

convergence curve

==================================================
R07
==================================================

Experiment:

FedAvg

IID

20 clients

10 rounds

Goal:

Scalability analysis

Measure:

same metrics as R06

Compare:

R01
R06
R07

Generate:

client-count comparison

==================================================
R08
==================================================

Experiment:

FedAvg

IID

20 clients

20 rounds

Participation:

10 of 20 clients

Random reproducible selection

Goal:

Client subsampling analysis

Track:

participation frequency

communication reduction

runtime reduction

Create:

client participation matrix

participation summary

==================================================
R09
==================================================

Experiment:

FedProx

Dirichlet alpha=0.1

5 clients

5 rounds

mu = 0.0001

Goal:

FedProx sensitivity

==================================================
R10
==================================================

Experiment:

FedProx

Dirichlet alpha=0.1

5 clients

5 rounds

mu = 0.01

Goal:

FedProx sensitivity

==================================================
R11
==================================================

Experiment:

FedProx

Dirichlet alpha=0.1

5 clients

5 rounds

mu = 0.1

Goal:

FedProx sensitivity

==================================================
FEDPROX COMPARISON
==================================================

Compare:

mu = 0.0001
mu = 0.001
mu = 0.01
mu = 0.1

Generate:

fedprox_mu_sensitivity.csv

Plots:

mu vs AUROC

mu vs AUPRC

mu vs runtime

mu vs communication

==================================================
R20
==================================================

Experiment:

FedAvg

Dirichlet alpha=0.1

5 clients

5 rounds

Policy:

U-Ignore

Implementation:

uncertain labels excluded from loss

Goal:

Uncertainty-policy comparison

==================================================
R21
==================================================

Experiment:

FedAvg

Dirichlet alpha=0.1

5 clients

5 rounds

Policy:

Label-dependent uncertainty mapping

Mapping must be configurable.

Goal:

Uncertainty sensitivity

==================================================
UNCERTAINTY-POLICY COMPARISON
==================================================

Compare:

U-Zero

U-Ignore

Label-dependent mapping

Evaluate:

Per-label AUROC

Per-label AUPRC

Macro AUROC

Macro AUPRC

Generate:

uncertainty_policy_sensitivity.csv

Policy comparison figures

==================================================
METRICS
==================================================

For every run calculate:

Per-label AUROC

Macro AUROC

Per-label AUPRC

Macro AUPRC

Sensitivity

Specificity

PPV

NPV

F1

Accuracy

Report support for every label.

Handle undefined metrics safely.

==================================================
PROCESSED-EXAMPLE BUDGET ANALYSIS
==================================================

For every experiment calculate:

samples processed

SGD updates

communication rounds

participating clients

training budget

Normalize:

performance

against

processed-example budget

Generate:

budget_analysis.csv

==================================================
CONVERGENCE ANALYSIS
==================================================

Combine:

R01
R06
R07
R08

Generate:

Round-wise AUROC

Round-wise AUPRC

Convergence plots

Participation plots

==================================================
STATISTICAL ANALYSIS
==================================================

Use patient-level bootstrap.

Compute:

95 percent confidence intervals

Generate:

confidence_interval_summary.csv

==================================================
OUTPUT STRUCTURE
==================================================

outputs/
    p2_scalability/
    p2_subsampling/
    p2_fedprox_sensitivity/
    p2_uncertainty_sensitivity/

Create:

scalability_summary.csv

fedprox_sensitivity_summary.csv

uncertainty_policy_summary.csv

budget_analysis.csv

convergence_analysis.csv

==================================================
FINAL FIGURES
==================================================

Figure 1

Scalability comparison

5 vs 10 vs 20 clients

Figure 2

Client participation analysis

Figure 3

FedProx sensitivity

Figure 4

Uncertainty-policy comparison

Figure 5

Processed-example budget analysis

Export:

PNG

PDF

==================================================
FINAL REPORT
==================================================

Generate:

p2_final_summary_report.json

containing:

experiment settings

runtime

communication

AUROC

AUPRC

sensitivity results

uncertainty-policy results

scalability results

budget analysis

figure paths

artifact paths

==================================================
IMPORTANT CONSTRAINTS
==================================================

Do not use post-hoc best-round selection.

Use pre-specified endpoints.

Keep the same initialization across all comparisons.

Do not claim superiority without statistical support.

Do not claim scalability loss without budget normalization.

Do not change the validation set.

Do not change the partition definitions.

Reuse existing federated-learning code wherever possible.