ALGORITHM P1_CRITICAL_REVISION_PIPELINE

INPUT:
    CheXpert images
    train.csv
    valid.csv
    historical predictions
    historical runtime logs

OUTPUT:
    audited dataset
    corrected IID partitions
    corrected Dirichlet partitions
    heterogeneity reports
    permissioned blockchain ledger
    signed model commitments
    R01-R05 benchmark results
    bootstrap confidence intervals
    runtime analysis
    blockchain overhead analysis

----------------------------------------------------
STEP 1: DATASET AUDIT
----------------------------------------------------

Load train metadata
Load validation metadata

Extract:
    patient_id
    study_id
    patient_study_id

Verify:
    required columns exist
    files exist
    labels exist

Count:
    images
    studies
    patients
    label supports

Check:
    duplicate paths
    duplicate studies
    patient overlap between train and validation

IF overlap exists:
    raise data leakage error

Generate:
    provenance report
    label support report
    audit report

----------------------------------------------------
STEP 2: FREEZE EXPERIMENT CONFIGURATION
----------------------------------------------------

Create immutable experiment manifest

Store:
    image size
    batch size
    model architecture
    label order
    optimizer
    learning rate
    weight decay
    precision
    local epochs
    client counts
    random seed

Initialize deterministic random seed

Save:
    configuration JSON

----------------------------------------------------
STEP 3: REBUILD PATIENT PARTITIONS
----------------------------------------------------

FOR each client count:
    [5,10,20]

    Create IID partition

    Assign every patient
    to exactly one client

    Verify:
        patient appears in one client only

Store IID manifest

FOR alpha in [0.3,0.1]

    Create patient-level Dirichlet partition

    Assign:
        patient -> client

    Verify:
        no patient overlap

Store partition manifest

----------------------------------------------------
STEP 4: HETEROGENEITY ANALYSIS
----------------------------------------------------

FOR every partition:

    FOR each client:

        calculate:
            sample count
            patient count
            label prevalence

    Compute:

        prevalence shift

        Jensen-Shannon divergence

        global prevalence

Generate:
    heterogeneity tables
    heterogeneity figures

----------------------------------------------------
STEP 5: CREATE PERMISSIONED BLOCKCHAIN
----------------------------------------------------

Initialize validators

validator_1
validator_2
validator_3

Consensus:
    majority finality

Replicate:
    same ledger state
    across all validators

Maintain:
    transaction pool
    blocks
    finality records

----------------------------------------------------
STEP 6: MODEL COMMITMENT AND SIGNING
----------------------------------------------------

FOR every client update:

    Serialize local model update

    Compute:
        update_digest =
        SHA256(update)

    Sign digest:
        Ed25519(private_key)

    Create transaction:

        round_id
        client_id
        update_digest
        signature

Submit transaction
to blockchain

----------------------------------------------------
STEP 7: SIGNATURE VERIFICATION
----------------------------------------------------

FOR every received transaction:

    Verify:
        Ed25519 signature

    IF invalid:

        reject update

    ELSE:

        accept update

----------------------------------------------------
STEP 8: GLOBAL MODEL COMMITMENT
----------------------------------------------------

Before aggregation:

    Compute:
        global_input_digest

After aggregation:

    Compute:
        global_output_digest

Record:

    global_input_digest
    global_output_digest
    config_digest
    metrics_digest
    round_id

Store on blockchain

----------------------------------------------------
STEP 9: REPLICATION AND TAMPER TESTS
----------------------------------------------------

Verify:

    all validator ledgers identical

Perform:

    post-hoc mutation attack

Recompute hashes

Confirm:
    tampering detected

Perform:

    signature substitution attack

Confirm:
    verification fails

Create:
    validation report

----------------------------------------------------
STEP 10: EXECUTE CORE BENCHMARKS
----------------------------------------------------

Run R01

    FedAvg
    IID
    5 clients
    10 rounds

Run R02

    FedAvg
    Dirichlet alpha=0.3

Run R03

    FedAvg
    Dirichlet alpha=0.1

Run R04

    FedProx
    IID
    mu=0.001

Run R05

    FedProx
    Dirichlet alpha=0.1
    mu=0.001

For each run:

    Train model

    Evaluate every round

    Store:

        predictions
        metrics
        checkpoints
        blockchain commitments

----------------------------------------------------
STEP 11: EVALUATION
----------------------------------------------------

FOR every model:

    Generate:

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

Threshold:
    0.5

Handle:
    undefined AUROC labels

----------------------------------------------------
STEP 12: PATIENT-LEVEL BOOTSTRAP
----------------------------------------------------

FOR bootstrap iteration:

    sample patients with replacement

    recompute metrics

Repeat N times

Generate:

    95% confidence intervals

----------------------------------------------------
STEP 13: PAIRED COMPARISONS
----------------------------------------------------

Use identical patient cohort

Compare:

    FedAvg vs FedProx

    FedAvg IID
    vs
    FedAvg Dirichlet

Compute:

    metric difference

    confidence interval

    p-value

Apply:

    Benjamini-Hochberg correction

----------------------------------------------------
STEP 14: RUNTIME ANALYSIS
----------------------------------------------------

Measure separately:

    training

    evaluation

    aggregation

    signing

    verification

    blockchain consensus

Compute:

    total runtime

Generate:

    runtime breakdown table

----------------------------------------------------
STEP 15: BLOCKCHAIN OVERHEAD ANALYSIS
----------------------------------------------------

Measure:

    SHA256 latency

    signing latency

    verification latency

    transaction latency

    consensus latency

    ledger growth

Compute:

    blockchain overhead

Store results

----------------------------------------------------
STEP 16: FINAL OUTPUTS
----------------------------------------------------

Save:

    audit reports
    partition manifests
    heterogeneity reports
    blockchain reports
    benchmark results
    bootstrap statistics
    runtime analysis
    blockchain overhead analysis

END ALGORITHM