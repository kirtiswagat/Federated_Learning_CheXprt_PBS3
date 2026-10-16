ALGORITHM P5_OUTPUT_AND_INTEGRATION_WORKFLOW

INPUTS

    P1 outputs

        R01-R05

        blockchain outputs

        statistical outputs

    P2 outputs

        R06-R11
        R20-R21

    P3 outputs

        R12-R19

    P4 outputs

        robustness matrix

        adversarial reports

    validation predictions

    training summaries

    checkpoints

    ledger evidence

OUTPUTS

    manuscript tables

    manuscript figures

    reviewer-response package

    reproducibility package

    final supplementary material

    final paper-ready result archive

--------------------------------------------------
STEP 1
DISCOVER ALL ARTIFACTS
--------------------------------------------------

Search output directories

Locate

    predictions

    checkpoints

    round histories

    training summaries

    per-label metrics

    macro metrics

    confidence intervals

    blockchain outputs

Build

    artifact registry

Verify

    required files exist

Create

    artifact_inventory.json

--------------------------------------------------
STEP 2
VALIDATE OUTPUT COMPLETENESS
--------------------------------------------------

For every required run

R01-R21

Verify

    training_summary exists

    round_history exists

    predictions exist

    checkpoint exists

    metrics exist

Flag

    missing experiments

Create

    completeness report

--------------------------------------------------
STEP 3
LOAD PAPER RESULTS
--------------------------------------------------

Load

    centralized baseline

    FedAvg results

    FedProx results

    CBRA results

    robustness results

Standardize

    experiment names

    labels

    metric formats

--------------------------------------------------
STEP 4
PATIENT-LEVEL STATISTICAL ANALYSIS
--------------------------------------------------

Load predictions

Run

    cluster bootstrap

Compute

    95% confidence intervals

For:

    AUROC

    AUPRC

    sensitivity

    specificity

    PPV

    NPV

    F1

    accuracy

Store

    ci_summary.csv

--------------------------------------------------
STEP 5
PAIRED COMPARISON FRAMEWORK
--------------------------------------------------

Compare

    FedAvg vs FedProx

    FedAvg vs CBRA

    CBRA vs FedProx

Use

    patient-level pairing

Apply

    multiplicity correction

Generate

    paired_comparison_table.csv

--------------------------------------------------
STEP 6
CORE PERFORMANCE TABLES
--------------------------------------------------

Generate paper table

Columns

    Method

    Macro AUROC

    Macro AUPRC

    Sensitivity

    Specificity

    PPV

    NPV

    F1

    Accuracy

Include

    95% confidence intervals

Export

    paper_results_table.csv

--------------------------------------------------
STEP 7
PER-LABEL TABLES
--------------------------------------------------

Create

    per-label AUROC

    per-label AUPRC

Include

    label support

Flag

    unstable rare-label estimates

Export

    per_label_results.csv

--------------------------------------------------
STEP 8
SCALABILITY ANALYSIS
--------------------------------------------------

Combine

    R01

    R06

    R07

    R08

Generate

    convergence curves

    processed-example analysis

    participation analysis

Export

    scalability_summary.csv

--------------------------------------------------
STEP 9
FEDPROX ANALYSIS
--------------------------------------------------

Combine

    R09

    R10

    R11

Generate

    mu sensitivity table

    mu sensitivity figure

Export

    fedprox_sensitivity.csv

--------------------------------------------------
STEP 10
CBRA ANALYSIS
--------------------------------------------------

Combine

    R12-R19

Generate

    parameter sensitivity

    contribution distribution

    baseline comparisons

Export

    cbra_sensitivity.csv

--------------------------------------------------
STEP 11
UNCERTAINTY ANALYSIS
--------------------------------------------------

Combine

    U-Zero

    U-Ignore

    Label-dependent

Generate

    uncertainty comparison table

    uncertainty sensitivity figure

Export

    uncertainty_policy_summary.csv

--------------------------------------------------
STEP 12
ROBUSTNESS ANALYSIS
--------------------------------------------------

Combine

    label flipping

    sign flipping

    model replacement

    targeted backdoor

    Byzantine

Compare

    FedAvg

    coordinate median

    trimmed mean

    clipped FedAvg

Generate

    robustness matrix

Export

    robustness_summary.csv

--------------------------------------------------
STEP 13
BLOCKCHAIN ANALYSIS
--------------------------------------------------

Load

    hashing results

    signing results

    verification results

    ledger replication results

    ledger growth

    consensus latency

Compute

    blockchain overhead

Generate

    blockchain_overhead.csv

--------------------------------------------------
STEP 14
SEPARATE SECURITY CLAIMS
--------------------------------------------------

Create comparison

    blockchain integrity

versus

    poisoning robustness

Explicitly show

    blockchain

        logs updates

    robustness methods

        limit update influence

Generate

    security_scope_summary.csv

--------------------------------------------------
STEP 15
GENERATE FIGURE 1
--------------------------------------------------

Centralized

vs

FedAvg

vs

FedProx

vs

CBRA

Performance comparison

Export

    png

    pdf

--------------------------------------------------
STEP 16
GENERATE FIGURE 2
--------------------------------------------------

Convergence curves

R01

R06

R07

R08

--------------------------------------------------
STEP 17
GENERATE FIGURE 3
--------------------------------------------------

FedProx μ sensitivity

--------------------------------------------------
STEP 18
GENERATE FIGURE 4
--------------------------------------------------

CBRA sensitivity

--------------------------------------------------
STEP 19
GENERATE FIGURE 5
--------------------------------------------------

Robustness matrix heatmap

--------------------------------------------------
STEP 20
GENERATE FIGURE 6
--------------------------------------------------

Blockchain overhead summary

--------------------------------------------------
STEP 21
GENERATE SUPPLEMENTARY TABLES
--------------------------------------------------

Create

    full metric tables

    confidence intervals

    bootstrap details

    sensitivity analyses

Export

    supplementary_material.xlsx

--------------------------------------------------
STEP 22
BUILD REPRODUCIBILITY PACKAGE
--------------------------------------------------

Save

    config manifests

    checkpoint hashes

    partition manifests

    seeds

    environment metadata

Export

    reproducibility_manifest.json

--------------------------------------------------
STEP 23
MANUSCRIPT CLAIM VALIDATION
--------------------------------------------------

For each manuscript claim

Locate supporting evidence

If evidence absent

    mark claim for removal

If confidence intervals overlap

    avoid superiority claims

Ensure

    conclusions follow results

--------------------------------------------------
STEP 24
REVIEWER RESPONSE ENGINE
--------------------------------------------------

Map

Reviewer comment

to

experiment

to

evidence

Generate

reviewer_response_table.csv

Columns

    reviewer comment

    action taken

    evidence location

--------------------------------------------------
STEP 25
CREATE FINAL PAPER PACKAGE
--------------------------------------------------

Bundle

    tables

    figures

    supplementary material

    reproducibility artifacts

    reviewer response matrix

    final summaries

Export

    final_paper_package/

--------------------------------------------------
STEP 26
QUALITY CONTROL
--------------------------------------------------

Verify

    all citations consistent

    all metrics reproducible

    no missing experiment

    figure/table numbering valid

    manuscript uses corrected terminology

    blockchain claims accurate

    CBRA claims evidence-based

    robustness claims evidence-based

END