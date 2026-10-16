ALGORITHM P2_SCALING_AND_SENSITIVITY

INPUTS

    Frozen experiment configuration
    Audited CheXpert dataset
    Patient-exclusive partitions
    Fixed ResNet50 architecture
    Fixed class weights
    Fixed preprocessing pipeline
    Existing FL framework

OUTPUTS

    R06–R11 results
    R20–R21 results

    convergence curves
    scalability analysis
    processed-example budget analysis

    uncertainty sensitivity analysis
    FedProx sensitivity analysis

--------------------------------------------------
STEP 1
LOAD REPRODUCIBLE EXPERIMENT STATE
--------------------------------------------------

Load:

    frozen config
    deterministic seed
    fixed model initialization
    label definitions
    class weights

Verify:

    configuration checksum matches

Load:

    validation dataset
    patient partitions

--------------------------------------------------
STEP 2
CONSTRUCT EXPERIMENT REGISTRY
--------------------------------------------------

Register experiments

R06
    FedAvg
    IID
    10 clients
    10 rounds

R07
    FedAvg
    IID
    20 clients
    10 rounds

R08
    FedAvg
    IID
    20 clients
    20 rounds
    50 percent participation

R09
    FedProx
    alpha=0.1
    mu=0.0001

R10
    FedProx
    alpha=0.1
    mu=0.01

R11
    FedProx
    alpha=0.1
    mu=0.1

R20
    FedAvg
    alpha=0.1
    U-Ignore

R21
    FedAvg
    alpha=0.1
    Label-dependent uncertainty policy

--------------------------------------------------
STEP 3
COMMON FEDERATED EXECUTION ENGINE
--------------------------------------------------

FOR each experiment

    Initialize model

    Load common initialization

    FOR each round

        Select participating clients

        Train local clients

        Aggregate client models

        Evaluate global model

        Save:

            round metrics
            predictions
            checkpoint

        Track:

            communication volume
            runtime
            processed examples

    Store final outputs

--------------------------------------------------
STEP 4
SCALABILITY STUDY
--------------------------------------------------

RUN R06

    IID
    10 clients
    10 rounds

RUN R07

    IID
    20 clients
    10 rounds

For each run

    calculate

        round AUROC
        round AUPRC
        round runtime
        communication

    track

        examples processed
        client participation

Generate

    convergence curves

Compare

    5 clients baseline
    10 clients
    20 clients

--------------------------------------------------
STEP 5
CLIENT SUBSAMPLING STUDY
--------------------------------------------------

RUN R08

    20 clients total

    select

        10 clients per round

    participation rate

        50 percent

Maintain

    matched processed-example budget

Track

    client selection history

Evaluate

    convergence
    performance
    communication savings
    runtime savings

Generate

    participation table
    convergence figure

--------------------------------------------------
STEP 6
FEDPROX SENSITIVITY
--------------------------------------------------

FOR mu in

    0.0001
    0.001
    0.01
    0.1

Execute

    FedProx
    Dirichlet alpha=0.1
    5 clients

After training

    measure

        macro AUROC
        macro AUPRC
        sensitivity
        specificity
        F1
        runtime

Compare

    all mu values

Identify

    stability trends

Generate

    sensitivity table
    sensitivity figure

--------------------------------------------------
STEP 7
UNCERTAINTY POLICY SENSITIVITY
--------------------------------------------------

POLICY 1

    U-Zero

POLICY 2

    U-Ignore

POLICY 3

    label-dependent mapping

FOR each policy

    construct labels

    run identical partition

    run identical seed

    run identical budget

Evaluate

    per-label AUROC
    macro AUROC
    per-label AUPRC
    macro AUPRC

Compare

    policy effect

Generate

    uncertainty comparison table

--------------------------------------------------
STEP 8
ROUND-WISE METRIC COLLECTION
--------------------------------------------------

For every round

    collect

        train loss
        validation loss

        macro AUROC
        macro AUPRC

        sensitivity
        specificity
        PPV
        NPV
        F1
        accuracy

        runtime

        communication volume

Store

    round_history.csv

--------------------------------------------------
STEP 9
PROCESSED EXAMPLE BUDGET ANALYSIS
--------------------------------------------------

For every experiment

    calculate

        total processed examples

        local updates

        client participation

Compare

    experiments

Normalize

    performance
    against processed-example budget

Generate

    budget comparison table

--------------------------------------------------
STEP 10
STATISTICAL ANALYSIS
--------------------------------------------------

For all experiments

    calculate

        patient-level bootstrap CI

        macro AUROC CI
        macro AUPRC CI

Generate

    confidence intervals

Perform

    paired comparisons

--------------------------------------------------
STEP 11
GENERATE FINAL OUTPUT FILES
--------------------------------------------------

For each experiment save

    training_summary.csv

    round_history.csv

    predictions.csv

    per_label_metrics.csv

    macro_metrics.csv

    runtime_summary.csv

Create

    fedprox_sensitivity_summary.csv

    uncertainty_policy_summary.csv

    scalability_summary.csv

    convergence_summary.csv

    processed_budget_analysis.csv

--------------------------------------------------
STEP 12
GENERATE PAPER-READY FIGURES
--------------------------------------------------

Create

    Figure 1
    Scalability convergence curves

    Figure 2
    Client participation comparison

    Figure 3
    FedProx mu sensitivity

    Figure 4
    Uncertainty-policy comparison

    Figure 5
    Processed-example budget analysis

Export

    PNG
    PDF

--------------------------------------------------
STEP 13
VALIDATION CHECKPOINT
--------------------------------------------------

Verify

    identical initialization

    identical validation dataset

    identical partitions

    identical label ordering

    identical preprocessing

Verify

    no post-hoc best-round selection

Verify

    reviewer-required outputs exist

END