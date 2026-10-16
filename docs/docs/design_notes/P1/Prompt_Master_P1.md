You are a senior Python, PyTorch, federated-learning, medical-imaging,
statistics, cryptography, and reproducible-research engineer.

Your task is to extend an existing Jupyter notebook project for multi-label
lung-disease classification from CheXpert chest X-ray images.

The existing project already contains:

- CheXpert archive discovery and extraction
- train.csv and valid.csv loading
- image-path normalization and validation
- patient_id, study_id, and patient_study_id extraction
- frontal-view filtering
- fourteen-label preprocessing
- U-Zero and uncertainty-mask concepts
- a ChestXrayMultiLabelDataset implementation
- ImageNet-compatible image preprocessing
- ResNet-50 multi-label classification
- centralized training
- FedAvg
- FedProx
- patient-level IID partitions
- patient-level Dirichlet partitions
- local-client training
- server-side model aggregation
- validation after each federated round
- per-label AUROC
- macro AUROC
- checkpointing and resume support
- prediction CSV exports
- communication-volume estimates
- runtime records
- a local SHA-256 hash-chained audit ledger
- experiment-summary CSV files
- artifact-discovery utilities
- paper-ready comparison-table generation

Do not replace this working pipeline unnecessarily.

First inspect the current repository and notebook code. Reuse existing
functions, classes, naming conventions, file paths, checkpoint formats, CSV
schemas, and output directories wherever possible.

Do not create duplicate implementations when a compatible function already
exists. Refactor only when it improves correctness, testability, or reuse
without breaking existing result files.

The immediate scope is P1 Critical only, covering P1-01 through P1-12.

============================================================
A. STRICT PROJECT CONSTRAINTS
============================================================

Follow these constraints exactly:

1. Use Python and PyTorch.

2. Use the existing project root:

   D:\LUNG_DISEASE

3. Preserve existing source artifacts.

4. Never overwrite historical predictions, checkpoints, summaries, ledgers,
   or paper-ready outputs.

5. Write all new P1 outputs under:

   D:\LUNG_DISEASE\outputs\p1_critical_revision

6. Organize the code into reusable modules instead of placing all logic in
   one oversized notebook cell.

7. Use:

   - pathlib for paths
   - dataclasses for configuration
   - pandas and NumPy for tabular analysis
   - PyTorch for model training
   - scikit-learn for evaluation
   - cryptography Ed25519 for digital signatures
   - hashlib SHA-256 for deterministic commitments
   - pytest for unit and integration tests
   - logging for execution logs
   - time.perf_counter for elapsed-time measurement

8. Use type hints and docstrings.

9. Validate inputs before processing.

10. Fail loudly with clear exceptions when:

    - required files are missing
    - required columns are missing
    - patient leakage is detected
    - a partition assigns one patient to multiple clients
    - a model signature is invalid
    - validator consensus is not reached
    - replicated ledgers disagree
    - prediction schemas are incompatible
    - patient IDs required for statistical analysis are unavailable

11. Do not fabricate:

    - data
    - results
    - confidence intervals
    - model predictions
    - runtime values
    - ledger events
    - successful experiment status

12. Do not add new CNN or Transformer architectures.

13. Do not create a new post-hoc train/validation/test split.

14. Do not perform broad hyperparameter optimization.

15. Do not add an external dataset.

16. Do not claim that blockchain improves classification accuracy.

17. Do not claim that blockchain makes poisoned updates benign.

18. Describe Dirichlet clients as simulated statistical heterogeneity, not
    genuine cross-hospital domain shift.

19. Do not use validation-based early stopping for revised R01-R05 endpoint
    selection. Run the prespecified number of rounds and report the
    prespecified endpoints.

20. Use the same deterministic initial model state for matched comparisons.

============================================================
B. REQUIRED PACKAGE STRUCTURE
============================================================

Create or update the following structure:

src/
    config.py
    reproducibility.py
    data/
        audit.py
        labels.py
        partitions.py
        heterogeneity.py
    models/
        resnet_multilabel.py
        serialization.py
    federated/
        client.py
        fedavg.py
        fedprox.py
        aggregation.py
        runner.py
        run_registry.py
    blockchain/
        identity.py
        transactions.py
        blocks.py
        validator.py
        consensus.py
        permissioned_ledger.py
        validation.py
        overhead.py
    evaluation/
        metrics.py
        bootstrap.py
        paired_comparison.py
        multiplicity.py
    monitoring/
        runtime.py
        artifact_registry.py
    reporting/
        tables.py
        manifests.py

configs/
    p1_base_config.json
    p1_runs.json

scripts/
    run_p1_audit.py
    run_p1_partitions.py
    run_p1_blockchain_tests.py
    run_p1_experiments.py
    run_p1_statistics.py
    run_p1_reporting.py

tests/
    test_audit.py
    test_partitions.py
    test_heterogeneity.py
    test_serialization.py
    test_signatures.py
    test_consensus.py
    test_tamper_detection.py
    test_metrics.py
    test_bootstrap.py
    test_runtime.py

Create a lightweight orchestration notebook:

P1_Critical_Revision.ipynb

The notebook must call reusable source modules. Do not place the complete
implementation only inside the notebook.

============================================================
C. P1-01: DATASET RE-AUDIT AND PROVENANCE VALIDATION
============================================================

Implement a complete audit of:

- CheXpert training metadata
- CheXpert validation metadata
- image paths
- frontal-view filtering
- patients
- studies
- patient-study IDs
- fourteen label columns
- historical validation predictions
- historical training summaries
- historical runtime records

The audit must:

1. Locate train.csv and valid.csv using the existing project conventions.

2. Validate that all required columns exist.

3. Preserve the exact fourteen-label order used by the existing notebook.

4. Resolve image paths safely.

5. Verify whether every retained image exists and can be opened.

6. Count:

   - metadata rows
   - retained frontal images
   - unique patients
   - unique studies
   - unique patient-study combinations
   - unique image paths
   - duplicate image paths
   - missing images
   - unreadable images

7. For every label, count:

   - positive values
   - negative values
   - uncertain values
   - missing values

8. Verify zero patient overlap between training and validation.

9. Detect duplicate or inconsistent patient-study assignments.

10. Discover historical artifacts recursively:

    - *valid_predictions*.csv
    - *training_summary*.csv
    - *round_history*.csv
    - *client_round_details*.csv
    - *per_label_auroc*.csv
    - best_model.pt
    - best_central_model.pt
    - resume_checkpoint.pt
    - training_config.json
    - ledger files

11. For each artifact, record:

    - absolute path
    - relative path
    - file type
    - file size
    - modification timestamp
    - SHA-256 checksum
    - inferred experiment key
    - schema columns where applicable
    - readability status

12. Do not alter historical artifacts.

Write:

- audits/dataset_summary.csv
- audits/label_support.csv
- audits/patient_leakage_report.csv
- audits/image_health_report.csv
- audits/historical_artifact_registry.csv
- audits/provenance_manifest.json

Acceptance criteria:

- No training/validation patient overlap
- Exact positive and negative support for each validation label
- Explicit status for any label having only one observed class
- Every discovered artifact has a checksum and readability status

============================================================
D. P1-02: FREEZE THE REPRODUCIBLE CONFIGURATION
============================================================

Create immutable dataclasses for:

- ProjectPaths
- DatasetConfig
- ModelConfig
- TrainingConfig
- FederatedConfig
- BlockchainConfig
- StatisticalConfig
- RuntimeConfig
- ExperimentConfig

Freeze these settings:

- random seed: 42
- image size: 320 by 320
- labels: existing fourteen-label order
- model: ResNet-50 with ImageNet initialization
- output dimension: 14
- dropout: 0.3
- optimizer: AdamW
- learning rate: 1e-4
- weight decay: 1e-5
- batch size: 16
- precision: FP32
- one local epoch per client per round
- U-Zero for R01-R05
- fixed patient partitions
- fixed threshold: 0.5 for descriptive operating metrics
- no validation-based early stopping for revised R01-R05 endpoint selection
- deterministic common model initialization for matched runs

Implement deterministic seeding for:

- Python random
- NumPy
- PyTorch CPU
- all CUDA devices
- DataLoader workers

Save:

- manifests/frozen_config.json
- manifests/environment_report.json
- manifests/initial_model_state.pt
- manifests/initial_model_state.sha256
- manifests/package_versions.txt

Before starting each run, verify the initial-state checksum.

============================================================
E. P1-03: PATIENT-EXCLUSIVE PARTITION RECONSTRUCTION
============================================================

Implement corrected partitions for:

- IID, 5 clients
- IID, 10 clients
- IID, 20 clients
- Dirichlet alpha 0.3, 5 clients
- Dirichlet alpha 0.1, 5 clients

Partition at patient level, not image level.

All images and studies belonging to a patient must remain assigned to one
client.

IID requirements:

- deterministically shuffle unique patients
- assign patients approximately evenly
- preserve every eligible patient exactly once

Dirichlet requirements:

- support multi-label data
- allocate unique patients, not individual images
- use deterministic random state
- retain each patient exactly once
- prohibit empty clients
- avoid silently dropping rare-label patients
- document the multilabel stratification strategy used
- do not describe the resulting clients as real hospital sites

For every partition, validate:

- each patient appears in exactly one client
- all images are assigned
- no client is empty
- no patient overlaps across clients
- assignments are reproducible with the same seed
- assignments differ when the intended partition setting differs

Write:

- splits/iid_5_patient_assignments.csv
- splits/iid_10_patient_assignments.csv
- splits/iid_20_patient_assignments.csv
- splits/dirichlet_a03_5_patient_assignments.csv
- splits/dirichlet_a01_5_patient_assignments.csv
- splits/partition_validation_report.csv
- splits/partition_config.json

Include both patient-level and image-level assignment manifests.

============================================================
F. P1-04: HETEROGENEITY QUANTIFICATION
============================================================

For every partition and every client, calculate:

- image count
- patient count
- study count
- per-label positive count
- per-label negative count
- per-label uncertain count before mapping
- per-label prevalence after the selected policy
- difference from global prevalence
- absolute prevalence shift
- mean absolute prevalence shift
- Jensen-Shannon divergence from the global label distribution

Use a numerically stable Jensen-Shannon implementation.

Do not pass raw prevalence vectors directly to KL divergence without:

- non-negative values
- epsilon handling
- normalization to a probability distribution

Produce:

- heterogeneity/client_summary.csv
- heterogeneity/client_label_prevalence.csv
- heterogeneity/prevalence_shift.csv
- heterogeneity/js_divergence.csv
- heterogeneity/partition_heterogeneity_summary.csv

Generate publication-ready figures:

- client sample-size distribution
- per-client label-prevalence heatmap
- prevalence-shift heatmap
- Jensen-Shannon divergence by client

Clearly label the analysis as simulated statistical heterogeneity.

============================================================
G. P1-05: PERMISSIONED BLOCKCHAIN ARCHITECTURE
============================================================

Replace the current local hash-chain-only implementation with a controlled
permissioned multi-validator blockchain research prototype.

Use three named validator instances by default:

- validator_A
- validator_B
- validator_C

Make the validator count configurable.

Implement:

1. Client identity registry
2. Validator identity registry
3. Transaction creation
4. Transaction validation
5. Transaction pool
6. Block proposal
7. Validator voting
8. Majority consensus/finality
9. Replicated validator ledgers
10. Block linking
11. Chain validation
12. Persisted validator state
13. Failure and rejection records

The system must state and enforce a finality threshold:

floor(number_of_validators / 2) + 1

Do not describe this as production Byzantine-fault-tolerant consensus.

Call it a controlled permissioned majority-finality prototype unless a
genuine PBFT or other formally implemented protocol is added.

Block fields must include:

- schema_version
- ledger_name
- block_index
- timestamp_utc
- previous_block_hash
- block_hash
- proposer_validator_id
- validator_ids
- validator_votes
- finality_threshold
- finality_status
- round_id
- transaction_ids
- transaction_merkle_root
- global_input_state_digest
- client_update_commitment_digest
- aggregation_config_digest
- global_output_state_digest
- metrics_digest

Persist the ledger independently for every validator.

Write:

- blockchain/validators/<validator_id>/ledger.json
- blockchain/validators/<validator_id>/blocks.csv
- blockchain/finality_records.csv
- blockchain/rejected_transactions.csv
- blockchain/ledger_config.json

============================================================
H. P1-06: ED25519 SIGNING AND MODEL-UPDATE BINDING
============================================================

Create an Ed25519 key pair for every federated client and validator.

Never hardcode private keys.

For research execution:

- generate keys once
- save private keys in a protected local directory
- save public keys in an identity registry
- prevent private-key material from appearing in CSV, JSON, logs, or Git
- create a .gitignore entry for the private-key directory

Implement deterministic model/update serialization.

For a PyTorch state dictionary:

1. Sort parameter keys
2. Move tensors to CPU
3. make tensors contiguous
4. retain dtype and shape
5. serialize deterministically
6. calculate SHA-256

For every client update create:

message = canonical_json(
    round_id,
    client_id,
    global_input_state_digest,
    local_update_digest,
    local_sample_count,
    local_training_config_digest
)

Sign the canonical message using the client private key.

Before aggregation, the server must:

1. Retrieve the registered client public key
2. Verify the Ed25519 signature
3. Recompute the received update digest
4. Ensure the digest matches the signed message
5. Verify round ID
6. Verify client ID
7. Verify global input-state digest
8. Reject the update on any mismatch

Record:

- signature generation latency
- signature verification latency
- update hashing latency
- verification result
- rejection reason

Do not aggregate an unverified update.

============================================================
I. P1-07: REPLICATION, TAMPER, AND SUBSTITUTION VALIDATION
============================================================

Create automated security tests for:

1. Valid client signature accepted
2. Wrong client public key rejected
3. Modified model update rejected
4. Modified round ID rejected
5. Modified client ID rejected
6. Modified metric digest detected
7. Modified historical transaction detected
8. Modified historical block detected
9. Broken previous-hash link detected
10. Signature substitution rejected
11. Validator ledger divergence detected
12. Majority-finality failure detected
13. Replicated validator states agree after valid commits

For post-hoc mutation testing:

- clone one validator ledger
- alter a historical transaction
- optionally recompute the local subsequent hash chain
- show that the altered replica no longer agrees with the independently
  replicated validator ledgers
- do not imply that a standalone recomputable hash chain is sufficient

Produce:

- blockchain/security_validation_report.csv
- blockchain/replica_consistency_report.csv
- blockchain/tamper_test_report.json
- blockchain/signature_validation_report.csv

Every test must include:

- test ID
- test description
- expected result
- observed result
- pass/fail
- failure details

============================================================
J. P1-08: CORE CORRECTED FEDERATED BENCHMARKS
============================================================

Implement one generic experiment runner driven entirely by configuration.

Do not duplicate a full training loop for every experiment.

The runner must support:

- FedAvg
- FedProx
- IID partition
- Dirichlet partition
- fixed round count
- checkpoint/resume
- validation after every round
- per-round ledger commitment
- component-level timing
- prediction export
- pre-aggregation signature verification

Implement exactly these P1 runs:

R01:
- FedAvg
- IID
- 5 clients
- 10 rounds
- U-Zero
- record round 5 and round 10 as prespecified endpoints

R02:
- FedAvg
- Dirichlet alpha 0.3
- 5 clients
- 5 rounds
- U-Zero
- report round 5

R03:
- FedAvg
- Dirichlet alpha 0.1
- 5 clients
- 5 rounds
- U-Zero
- report round 5

R04:
- FedProx
- IID
- 5 clients
- 5 rounds
- mu 0.001
- U-Zero
- report round 5

R05:
- FedProx
- Dirichlet alpha 0.1
- 5 clients
- 5 rounds
- mu 0.001
- U-Zero
- report round 5

For every run:

1. Load the same frozen initial model state.
2. Load the fixed partition manifest.
3. Start or resume the checkpoint.
4. For every round:
   a. hash the global input model
   b. train every prescribed client for one local epoch
   c. measure local training time
   d. serialize and hash each local model/update
   e. sign each update
   f. verify each update at the server
   g. reject invalid updates
   h. aggregate valid updates
   i. measure aggregation time
   j. evaluate the global model
   k. save validation predictions including patient_id
   l. commit the round evidence to every validator ledger
   m. obtain finality
   n. save the resume checkpoint
5. Continue through the prespecified endpoint even if an earlier round has a
   higher validation score.
6. A historical best-model checkpoint may be saved for diagnostics, but the
   primary revised comparison must use the prespecified endpoint.
7. Do not use post-hoc best-round selection for the revised primary table.

Save per run:

- training_config.json
- round_history.csv
- client_round_details.csv
- validation_predictions_round_<n>.csv
- endpoint_valid_predictions.csv
- endpoint_per_label_metrics.csv
- training_summary.csv
- best_model_diagnostic.pt
- endpoint_model.pt
- resume_checkpoint.pt
- runtime_breakdown.csv
- blockchain_commitment_summary.csv
- artifact_manifest.json

Every prediction CSV must contain:

- patient_id
- study_id where available
- relative_path
- true_<label>
- prob_<label>
- logit_<label>

============================================================
K. P1-09: PATIENT-LEVEL STATISTICAL VALIDATION
============================================================

Implement patient-level cluster bootstrap.

Do not bootstrap image rows independently.

For each bootstrap iteration:

1. Sample validation patient IDs with replacement.
2. Include all observations belonging to the selected patient.
3. Reconstruct the bootstrap sample without merging duplicate draws into one.
4. Calculate per-label metrics.
5. Calculate macro metrics by averaging only defined labels.

Default:

- 2,000 bootstrap iterations
- deterministic bootstrap seed configurable separately from training seed
- percentile 95% confidence intervals

Calculate bootstrap confidence intervals for:

- per-label AUROC
- macro AUROC
- per-label AUPRC
- macro AUPRC
- sensitivity
- specificity
- PPV
- NPV
- F1
- accuracy

Implement paired patient-level comparisons using identical bootstrap patient
draws for both models.

At minimum support:

- R01 versus R04
- R01 versus R02
- R01 versus R03
- R03 versus R05

For each paired comparison report:

- model A
- model B
- metric
- point-estimate difference
- bootstrap mean difference
- 95% confidence interval
- two-sided bootstrap p-value
- number of valid bootstrap samples

Apply Benjamini-Hochberg multiplicity correction across the prespecified
exploratory comparison family.

Keep these concepts separate:

- optimization variability across random seeds
- sampling uncertainty from patient bootstrap

Do not claim superiority when the corrected comparison does not support it.

Write:

- statistics/per_label_bootstrap_ci.csv
- statistics/macro_bootstrap_ci.csv
- statistics/paired_comparisons_raw.csv
- statistics/paired_comparisons_corrected.csv
- statistics/bootstrap_config.json

============================================================
L. P1-10: COMPLETE EVALUATION ENGINE
============================================================

At threshold 0.5, calculate per label:

- positive support
- negative support
- AUROC
- AUPRC
- sensitivity
- specificity
- PPV
- NPV
- F1
- accuracy
- TP
- TN
- FP
- FN
- metric status

Rules:

- AUROC is undefined when only one class exists
- AUPRC is undefined when no positive case exists
- never replace undefined AUROC with zero
- exclude undefined labels from macro AUROC
- report the number and names of labels included in every macro average
- preserve undefined status in CSV and paper-ready outputs
- do not drop the Fracture row merely because AUROC is undefined

Produce:

- evaluation/per_label_metrics.csv
- evaluation/macro_metrics.csv
- evaluation/label_support.csv
- evaluation/undefined_metric_report.csv
- evaluation/model_comparison_table.csv

Reuse historical prediction files only after schema and patient-ID validation.

============================================================
M. P1-11: COMPLETE RUNTIME RE-ANALYSIS
============================================================

Use time.perf_counter.

Measure and store separately:

- data-loading/preparation time
- local training time per client
- total client-training time per round
- server aggregation time
- evaluation time
- model serialization time
- local-update hashing time
- signature generation time
- signature verification time
- transaction construction time
- transaction submission time
- validator voting time
- consensus/finality time
- checkpoint-writing time
- artifact-writing time
- complete round wall-clock time
- complete experiment wall-clock time

Do not reconstruct measured runtime from incomplete historical fields.

Label values as:

- measured
- derived
- unavailable

Assert or report whether component totals reconcile with complete wall-clock
time within an explicitly documented tolerance.

Write:

- runtime/runtime_events.csv
- runtime/round_runtime_summary.csv
- runtime/experiment_runtime_summary.csv
- runtime/runtime_reconciliation_report.csv

Keep analytical communication-volume estimates separate from measured
runtime.

============================================================
N. P1-12: BLOCKCHAIN OVERHEAD BENCHMARKING
============================================================

Measure blockchain overhead during actual R01-R05 execution.

For each relevant event record:

- run ID
- round ID
- client ID where applicable
- model/update byte size
- SHA-256 latency
- Ed25519 signing latency
- Ed25519 verification latency
- transaction-construction latency
- transaction-submission latency
- validator-vote latency
- consensus/finality latency
- block-commit latency
- ledger size before commit
- ledger size after commit
- incremental ledger growth
- validator count
- transaction count
- finality result

Report:

- mean
- standard deviation
- median
- minimum
- maximum
- 95th percentile
- total elapsed overhead

Separate:

- client-side cryptographic overhead
- server verification overhead
- consensus overhead
- persistence overhead
- total blockchain overhead
- model-training time

Calculate:

blockchain_overhead_percentage =
    total_blockchain_overhead /
    complete_experiment_wall_clock_time * 100

Do not benchmark arbitrary synthetic transaction volumes as the primary
scientific result. Optional synthetic microbenchmarks may be included only as
supplementary engineering tests and must be labeled synthetic.

Write:

- blockchain/overhead_events.csv
- blockchain/overhead_summary.csv
- blockchain/ledger_growth.csv
- blockchain/overhead_by_run.csv

Generate figures:

- overhead component breakdown
- consensus latency by round
- cumulative ledger growth
- training time versus blockchain overhead
- overhead percentage by R01-R05 run

============================================================
O. ARTIFACT MANIFEST AND REPRODUCIBILITY
============================================================

For every generated artifact, record:

- artifact name
- path
- run ID
- creation timestamp
- SHA-256 checksum
- file size
- schema version
- source config checksum
- source partition checksum
- initial model checksum
- endpoint model checksum where applicable

Create:

- manifests/p1_artifact_registry.csv
- manifests/p1_reproducibility_manifest.json

Do not claim completion for an experiment unless all required artifacts exist
and their checksums have been recorded.

============================================================
P. TESTING REQUIREMENTS
============================================================

Write tests for:

Dataset:
- missing required column
- duplicate path
- patient overlap
- single-class validation label

Partitions:
- every patient assigned exactly once
- no client overlap
- no empty client
- deterministic reproduction
- full image coverage

Serialization and signatures:
- identical state gives identical digest
- modified tensor changes digest
- correct signature accepted
- incorrect signature rejected
- substituted client ID rejected
- substituted round ID rejected

Blockchain:
- valid block finalized
- insufficient votes rejected
- broken previous hash rejected
- mutated transaction rejected
- validator divergence detected
- replicas agree after valid commit

Metrics:
- known confusion-matrix values
- undefined AUROC handling
- macro average excludes undefined values
- AUPRC behavior with no positives

Bootstrap:
- patient clusters remain intact
- duplicate patient draws remain duplicated
- deterministic bootstrap with fixed seed
- identical models produce approximately zero paired difference

Runtime:
- every required component has a record
- experiment wall-clock is not smaller than individual measured components
- measured and analytical values are not mixed

All tests must run without requiring full model training.

============================================================
Q. IMPLEMENTATION ORDER
============================================================

Implement in this order:

1. Inspect existing repository and produce a reuse map.
2. Implement configuration and reproducibility.
3. Implement dataset and provenance audit.
4. Implement corrected partitions.
5. Implement heterogeneity analysis.
6. Implement deterministic model serialization.
7. Implement client/validator identities and Ed25519 signing.
8. Implement permissioned replicated ledger and majority finality.
9. Implement blockchain/security tests.
10. Refactor the existing training logic into one configurable runner.
11. Add R01-R05 configurations.
12. Integrate signatures and blockchain commitments into each round.
13. Implement complete evaluation metrics.
14. Implement patient-level bootstrap and paired comparisons.
15. Implement runtime decomposition.
16. Implement blockchain-overhead reporting.
17. Implement artifact registry and orchestration scripts.
18. Add unit and integration tests.
19. Create P1_Critical_Revision.ipynb as a thin orchestration notebook.
20. Generate a README describing exact commands and expected artifacts.

============================================================
R. REQUIRED RESPONSE FORMAT
============================================================

Do not generate the entire codebase in one unstructured response.

Work incrementally.

For the first response:

1. Inspect the available repository files.
2. Summarize which functions and classes can be reused.
3. Identify missing modules.
4. Propose the final file tree.
5. List assumptions that need validation from the actual repository.
6. Then generate:
   - config.py
   - reproducibility.py
   - tests for both modules

For every subsequent module:

1. State which existing code is reused.
2. State which files will be created or modified.
3. Generate complete file contents.
4. Generate corresponding tests.
5. Explain the integration point.
6. Do not use placeholders such as:
   - TODO
   - pass
   - implement later
   - insert logic here
7. Do not create mock scientific results.
8. Do not silently modify experiment settings.
9. Do not mark the module complete until its tests are included.

When repository information is missing:

- inspect the repository first
- do not guess function signatures
- create adapters around confirmed interfaces
- state unresolved integration assumptions explicitly

The final implementation must preserve historical evidence while producing a
new, auditable, reproducible P1 Critical revision pipeline.