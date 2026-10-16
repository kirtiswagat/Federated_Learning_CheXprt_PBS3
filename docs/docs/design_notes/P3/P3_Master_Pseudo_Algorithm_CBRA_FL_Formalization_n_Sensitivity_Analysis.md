ALGORITHM P3_CBRA_ANALYSIS_FRAMEWORK

INPUTS

    Corrected patient partitions

    Fixed model initialization

    Frozen preprocessing

    ResNet50 architecture

    Existing FedAvg pipeline

    Existing FedProx pipeline

    Dirichlet alpha=0.1 partition

OUTPUTS

    R12-R19 results

    CBRA mathematical definition

    Sensitivity analysis

    Comparative analysis

    Reviewer-ready tables

--------------------------------------------------
STEP 1
DEFINE CBRA FRAMEWORK
--------------------------------------------------

Create generic CBRA contribution score

score(client_k)

Components

    sample contribution

    rarity contribution

    drift penalty

Parameters

    alpha_n

    beta_q

    gamma_d

Score structure

    score_k

        =
        sample_component
        × rarity_component
        × drift_component

Normalize

    all client scores

Generate

    normalized weights

Ensure

    weight sum = 1

--------------------------------------------------
STEP 2
SAMPLE CONTRIBUTION
--------------------------------------------------

For client k

sample_component

    =
    sample_count_k

        raised to

    alpha_n

Purpose

    reward larger clients

--------------------------------------------------
STEP 3
RARITY CONTRIBUTION
--------------------------------------------------

For client k

calculate

    rare-label prevalence

Compute

    rarity score

rarity_component

    =
    rarity_score

        raised to

    beta_q

Purpose

    reward clients
    containing rare pathology information

--------------------------------------------------
STEP 4
DRIFT PENALTY
--------------------------------------------------

Compute

    distance

between

    local update

and

    global model

Examples

    L2 distance

    cosine distance

drift_component

    =
    exp(-gamma_d × drift)

Purpose

    reduce influence
    of extreme updates

--------------------------------------------------
STEP 5
NORMALIZATION
--------------------------------------------------

For all clients

raw_score

Normalize

weight_k

    =
    raw_score_k

      /

    total_score

Verify

sum(weights)

    = 1

--------------------------------------------------
STEP 6
CBRA AGGREGATION
--------------------------------------------------

Receive

    local model updates

Receive

    contribution weights

Aggregate

global update

using

weighted average

Store

    weights per round

--------------------------------------------------
STEP 7
REFERENCE EXPERIMENTS
--------------------------------------------------

R12

    alpha_n=1
    beta_q=1
    gamma_d=0.5

Purpose

    CBRA v1 baseline

R13

    alpha_n=1
    beta_q=1.5
    gamma_d=0.5

Purpose

    CBRA v2 baseline

--------------------------------------------------
STEP 8
SAMPLE EXPONENT SENSITIVITY
--------------------------------------------------

Run R14

    alpha_n=0.5

Run R15

    alpha_n=1.5

Keep

    beta_q fixed

    gamma_d fixed

Compare

    contribution distribution

    performance

--------------------------------------------------
STEP 9
RARITY WEIGHT SENSITIVITY
--------------------------------------------------

Run R16

    beta_q=0.5

Run R17

    beta_q=1.5

Keep

    alpha_n fixed

    gamma_d fixed

Evaluate

    importance of rare classes

--------------------------------------------------
STEP 10
DRIFT PENALTY SENSITIVITY
--------------------------------------------------

Run R18

    gamma_d=0

Run R19

    gamma_d=1

Measure

    update suppression

    performance stability

--------------------------------------------------
STEP 11
COMMON TRAINING LOOP
--------------------------------------------------

For every experiment

Initialize

    same model

Load

    same partition

Train

    5 rounds

    1 local epoch

Evaluate

    every round

Save

    predictions

    checkpoints

    metrics

--------------------------------------------------
STEP 12
METRIC COLLECTION
--------------------------------------------------

Collect

    per-label AUROC

    macro AUROC

    per-label AUPRC

    macro AUPRC

    sensitivity

    specificity

    PPV

    NPV

    F1

    accuracy

Store

    round history

--------------------------------------------------
STEP 13
CONTRIBUTION ANALYSIS
--------------------------------------------------

For every round

Store

    client score

    normalized weight

    sample count

    rarity score

    drift score

Generate

    contribution tables

--------------------------------------------------
STEP 14
COMPARISON AGAINST BASELINES
--------------------------------------------------

Compare

    FedAvg

    FedProx

    CBRA v1

    CBRA v2

Compare

    matched partitions

Compare

    matched initialization

Compare

    matched rounds

Generate

    comparison table

--------------------------------------------------
STEP 15
SENSITIVITY ANALYSIS
--------------------------------------------------

Analyze

    alpha_n sensitivity

    beta_q sensitivity

    gamma_d sensitivity

Determine

    parameter stability

Identify

    unstable settings

--------------------------------------------------
STEP 16
STATISTICAL ANALYSIS
--------------------------------------------------

Apply

    patient bootstrap

Generate

    confidence intervals

Compare

    CBRA

vs

    FedAvg

vs

    FedProx

--------------------------------------------------
STEP 17
INTERPRETATION ENGINE
--------------------------------------------------

IF

CBRA > Baseline

AND confidence intervals support

Then

    improvement supported

ELSE

    report

    "tested heuristic"

    not demonstrated improvement

Never claim superiority
without evidence

--------------------------------------------------
STEP 18
FINAL OUTPUTS
--------------------------------------------------

Export

R12_results.csv

R13_results.csv

R14_results.csv

R15_results.csv

R16_results.csv

R17_results.csv

R18_results.csv

R19_results.csv

Export

cbra_contribution_weights.csv

cbra_parameter_sensitivity.csv

cbra_vs_fedavg.csv

cbra_vs_fedprox.csv

Export

paper-ready figures

    contribution distribution

    parameter sensitivity

    comparison plots

END