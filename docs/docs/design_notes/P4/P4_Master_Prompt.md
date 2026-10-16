You are a senior Python, PyTorch, federated-learning, adversarial machine
learning, medical-imaging, cryptography, testing, and reproducible-research
engineer.

Your task is to extend an existing CheXpert federated-learning repository
with the complete P4 adversarial-robustness workstream.

Do not build a separate federated-learning pipeline.

First inspect the repository and reuse the existing:

- CheXpert dataset implementation
- fourteen-label order
- corrected patient-level partitions
- ResNet-50 model builder
- frozen initial model state
- local-client training function
- FedAvg aggregation function
- validation function
- prediction export format
- checkpoint format
- runtime recorder
- patient-level bootstrap framework
- Ed25519 client identities
- model/update serialization
- permissioned blockchain layer
- artifact-manifest conventions

The existing project root is:

D:\LUNG_DISEASE

Write all new P4 outputs under:

D:\LUNG_DISEASE\outputs\p4_robustness

============================================================
1. SCIENTIFIC OBJECTIVE
============================================================

Build a reproducible one-round malicious-client stress-test framework.

This is not a set of additional full federated-training runs.

Use:

- corrected Dirichlet alpha=0.1 partition
- five simulated federated clients
- one local epoch
- one controlled global aggregation round
- frozen ResNet-50 global input state
- fixed fourteen-label validation cohort
- deterministic attack and client-selection seeds

Test malicious-client fractions:

- 20%
- 40%

Because there are five clients, implement an explicit configurable rounding
policy that converts each fraction to an integer malicious-client count.

Record the resolved count and selected client IDs in every condition manifest.

Do not silently change the malicious-client count.

============================================================
2. REQUIRED ATTACK MATRIX
============================================================

Implement these five attack families:

1. Label flipping
2. Sign flipping
3. Model replacement
4. Targeted backdoor
5. Byzantine/random update

For every attack family, test both malicious-client fractions:

- 20%
- 40%

For every attack/fraction condition, compare:

1. FedAvg
2. Coordinate-wise median
3. Trimmed mean
4. Norm-clipped FedAvg

Generate the full condition matrix from configuration.

Do not hardcode separate experiment loops for every condition.

============================================================
3. REQUIRED PROJECT STRUCTURE
============================================================

Create or update:

src/
    robustness/
        __init__.py
        config.py
        registry.py
        threat_model.py
        update_utils.py
        harness.py
        evaluation.py
        reporting.py
        attacks/
            __init__.py
            base.py
            label_flipping.py
            sign_flipping.py
            model_replacement.py
            targeted_backdoor.py
            byzantine_random.py
        aggregators/
            __init__.py
            base.py
            fedavg.py
            coordinate_median.py
            trimmed_mean.py
            norm_clipped_fedavg.py
        integrity/
            __init__.py
            blockchain_bridge.py
            separation_analysis.py

configs/
    p4_robustness.json

scripts/
    run_p4_clean_reference.py
    run_p4_attack_matrix.py
    run_p4_reporting.py

tests/
    robustness/
        test_update_utils.py
        test_label_flipping.py
        test_sign_flipping.py
        test_model_replacement.py
        test_targeted_backdoor.py
        test_byzantine_random.py
        test_fedavg.py
        test_coordinate_median.py
        test_trimmed_mean.py
        test_norm_clipped_fedavg.py
        test_harness_fairness.py
        test_blockchain_separation.py

Create:

P4_Adversarial_Robustness.ipynb

The notebook must orchestrate reusable source modules. Do not place the
complete implementation only inside notebook cells.

============================================================
4. STRICT CONSTRAINTS
============================================================

Follow these rules:

1. Do not alter original training or validation data.

2. Do not overwrite P1, P2, or P3 outputs.

3. Do not modify the corrected patient partition.

4. Do not create a new train/validation/test split.

5. Do not use new CNN or Transformer architectures.

6. Do not perform attack-strength hyperparameter searches.

7. Do not select attack configurations using validation performance.

8. Do not tune robust-aggregation parameters against validation metrics.

9. Do not describe simulated clients as real hospital sites.

10. Do not claim complete attack resistance.

11. Do not claim Byzantine fault tolerance unless formally implemented and
    tested.

12. Do not claim blockchain prevents poisoning.

13. Do not reject a correctly signed update merely because the update is
    malicious in effect.

14. A valid signature proves origin and integrity of the submitted update,
    not that the update is safe.

15. Do not fabricate attack results, metrics, runtimes, or successful tests.

16. Use type hints, dataclasses, logging, docstrings, clear exceptions, and
    deterministic random generators.

17. Fail loudly if the frozen configuration, partition, checkpoint,
    validation data, client IDs, or required repository functions cannot be
    loaded.

============================================================
5. CONFIGURATION MODEL
============================================================

Create immutable dataclasses:

- RobustnessPaths
- ThreatModelConfig
- AttackConfig
- AggregatorConfig
- BackdoorConfig
- RobustnessEvaluationConfig
- RobustnessCondition

The master P4 config must contain:

- project root
- output root
- source global checkpoint
- source checkpoint SHA-256
- partition manifest
- partition SHA-256
- client IDs
- total client count
- malicious fractions
- malicious-count rounding policy
- malicious-client selection seed
- attack seed
- aggregation methods
- update representation
- threshold for operating metrics
- backdoor target label
- backdoor trigger definition
- backdoor poisoning fraction
- model-replacement scale
- Byzantine update scale
- trimmed-mean trim rule
- norm-clipping threshold rule
- bootstrap iterations
- bootstrap seed
- blockchain logging enabled

Save the resolved configuration for every condition.

============================================================
6. THREAT MODEL
============================================================

Implement and document this controlled threat model:

- Malicious clients are registered federated participants.
- Malicious clients can modify local labels, local training, or submitted
  model updates.
- Malicious clients cannot alter honest-client data.
- Malicious clients cannot alter the validation cohort.
- Malicious clients cannot alter the frozen global input checkpoint.
- Malicious clients cannot forge another client's Ed25519 signature.
- The aggregation server does not know which clients are malicious.
- The experiment evaluates one-round model-update robustness.
- The experiment does not model every privacy or system-level attack.
- The clients are simulated from one source dataset.

Produce a machine-readable threat_model.json and a manuscript-ready Markdown
description.

============================================================
7. CLEAN REFERENCE GENERATION
============================================================

Before running attacks, create a clean one-round update set.

Load the same frozen global model for all clients.

For every client:

1. Train normally for one local epoch.
2. Compute the local model state.
3. Convert the local state to an update relative to the frozen global input.
4. Record:
   - client ID
   - patient count
   - image count
   - local loss before training
   - local loss after training
   - update L2 norm
   - update cosine similarity
   - update byte size
   - update SHA-256 digest
5. Sign the actual submitted update if P1 signing is enabled.
6. Verify the update before storing it.

Persist the clean updates so that each aggregator receives exactly the same
clean update set.

Aggregate the clean update set independently using all four aggregation
methods.

Evaluate each clean aggregated model.

Use the matching clean aggregator result as the baseline for degradation
calculations.

============================================================
8. COMMON UPDATE REPRESENTATION
============================================================

Implement reusable utilities:

- state_dict_to_update
- update_to_state_dict
- clone_update
- flatten_update
- unflatten_update
- validate_update_schema
- compute_update_l2_norm
- compute_update_cosine_similarity
- compute_update_digest
- check_finite_update

Define:

local_update =
    local_model_state - global_input_model_state

Preserve:

- parameter names
- tensor shapes
- tensor dtypes
- device-independent CPU serialization

Validate round-trip reconstruction.

Reject updates containing incompatible keys, shapes, NaN values, or infinite
values.

============================================================
9. ATTACK INTERFACE
============================================================

Create a common attack interface:

Attack.apply(
    client_id,
    clean_update,
    local_dataset,
    global_state,
    attack_config,
    random_generator
) -> AttackResult

AttackResult must include:

- attack name
- client ID
- malicious flag
- submitted update
- original update digest
- attacked update digest
- original update norm
- attacked update norm
- attack parameters
- modified-sample count where applicable
- attack metadata
- validation status

Attacks must modify copies, never original data or clean update files.

============================================================
10. LABEL-FLIPPING ATTACK
============================================================

Implement a deterministic multi-label label-flipping attack.

For malicious clients only:

- copy the local dataset or label view
- flip eligible binary targets
- map 0 to 1
- map 1 to 0
- preserve masked, missing, and undefined labels
- train the malicious local model for one local epoch
- calculate the attacked local update

Make the eligible-label set configurable.

Record:

- malicious client IDs
- affected labels
- number of altered label values
- affected sample count
- local loss before and after attack training
- attacked update norm

Do not mutate source DataFrames or partition manifests.

============================================================
11. SIGN-FLIPPING ATTACK
============================================================

For each malicious client:

- start from its clean local update
- multiply every floating-point update tensor by -1
- preserve parameter keys, shapes, and dtypes
- recompute the update digest
- sign the attacked update as the actual submitted payload

Record:

- original and attacked norms
- original-to-attacked cosine similarity
- digest change
- finite-value validation

============================================================
12. MODEL-REPLACEMENT ATTACK
============================================================

Implement one prespecified and documented model-replacement rule.

Expose a fixed configured replacement scale.

Do not tune it using validation results.

For each malicious client:

- begin with the clean update or a separately trained malicious local model,
  according to the configured rule
- apply the fixed replacement transformation
- preserve model schema
- calculate the submitted update norm and digest
- record the norm ratio relative to the clean update

Record whether norm clipping later modifies the replacement update.

============================================================
13. TARGETED-BACKDOOR ATTACK
============================================================

Implement a controlled synthetic targeted-backdoor attack.

Before code execution, require configuration for:

- target label
- trigger type
- trigger dimensions
- trigger location
- trigger intensity
- poisoning fraction within malicious clients
- target assignment policy
- attack-success threshold

The trigger must be deterministic.

For malicious clients only:

- copy selected local training samples
- apply the configured trigger
- assign the configured target
- retain a mixture of clean and poisoned samples
- train for one local epoch
- calculate the submitted update

Do not modify original image files.

Create a triggered copy of the validation data for attack-success evaluation.

Evaluate separately:

- clean validation performance
- triggered validation performance
- attack success rate
- target-label prediction rate on clean inputs
- target-label prediction rate on triggered inputs
- target probability shift
- non-target metric degradation

============================================================
14. BYZANTINE/RANDOM-UPDATE ATTACK
============================================================

For malicious clients:

- generate a deterministic random update matching every parameter key,
  tensor shape, and dtype
- use the configured attack seed
- apply one fixed configured scale
- ensure all tensors are finite
- calculate norm and digest

Do not select the scale using validation results.

Record the relationship between Byzantine update norms and honest update
norms.

============================================================
15. FEDAVG AGGREGATOR
============================================================

Reuse or adapt the existing FedAvg implementation.

Aggregate client updates using sample-count weights.

Validate:

- weights are non-negative
- weights sum to one
- all schemas match
- result contains finite values

Return an AggregationResult containing:

- aggregated update
- client weights
- original update norms
- transformed update norms
- aggregation parameters
- elapsed time

============================================================
16. COORDINATE-WISE MEDIAN
============================================================

For every model parameter:

- stack the corresponding client update tensors
- compute the median independently at every coordinate
- preserve the original tensor shape and dtype

Do not sample-weight the median.

Add synthetic unit tests with known expected medians.

============================================================
17. TRIMMED-MEAN AGGREGATOR
============================================================

For every parameter coordinate:

- stack all client values
- sort values
- remove the configured number of lowest values
- remove the configured number of highest values
- average the remaining values

Validate the trim rule against the total of five clients.

Fail with a clear error if the configuration leaves no retained values.

Record:

- trim count or proportion
- values retained per coordinate
- aggregation runtime

Do not silently alter the trim rule between 20% and 40% malicious-client
conditions.

============================================================
18. NORM-CLIPPED FEDAVG
============================================================

For every client update:

1. Compute its global L2 norm.
2. Resolve the clipping threshold using a prespecified configured rule.
3. Compute:

   clipping_factor =
       min(1, clipping_threshold / max(update_norm, epsilon))

4. Multiply the entire update by the clipping factor.
5. Apply sample-count-weighted FedAvg to clipped updates.

Record:

- original norm
- clipping threshold
- clipping factor
- clipped norm
- clipping applied flag

Clipping must never increase an update norm.

============================================================
19. FAIR ATTACK-MATRIX EXECUTION
============================================================

For each attack family:

    For each malicious fraction:

        1. Resolve malicious-client count.
        2. Select malicious clients deterministically.
        3. Generate one attacked update set.
        4. Freeze that update set.
        5. Pass the identical frozen update set to each aggregator.
        6. Reconstruct the resulting global model.
        7. Evaluate the resulting model.
        8. Save condition outputs.

Fairness assertions:

- all aggregators receive identical client updates for the same condition
- all aggregators use the same global input state
- all aggregators use the same malicious clients
- all aggregators use the same validation cohort
- attack parameters do not change across aggregator comparisons
- honest updates are unchanged across aggregators

Generate a condition-level fairness checksum report.

============================================================
20. EVALUATION METRICS
============================================================

For every clean and attacked aggregated model, calculate:

- per-label positive support
- per-label negative support
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
- TP
- TN
- FP
- FN

Use threshold 0.5 for operating metrics.

Rules:

- AUROC is undefined when only one class is available.
- AUPRC is undefined when there are no positives.
- Do not replace undefined metrics with zero.
- Exclude only undefined labels from the corresponding macro average.
- Preserve every label row, including labels having undefined metrics.
- Report names and counts of labels included in each macro result.

Calculate robustness measures:

- attacked metric minus matching clean-aggregator metric
- absolute degradation
- relative degradation
- retained performance percentage
- rank within the exact attack/fraction condition

============================================================
21. PATIENT-LEVEL STATISTICAL ANALYSIS
============================================================

If patient IDs are available in prediction outputs:

- use patient-level cluster bootstrap
- sample patients with replacement
- include all observations for every sampled patient
- use paired patient draws for matched comparisons
- calculate 95% confidence intervals

Support comparisons:

- attacked versus clean for the same aggregator
- robust aggregator versus FedAvg for the same attack condition

Apply Benjamini-Hochberg correction across the prespecified family.

Keep patient-sampling uncertainty separate from random-seed variability.

Do not claim superiority when corrected evidence does not support it.

============================================================
22. BLOCKCHAIN AND SIGNATURE INTEGRATION
============================================================

For every submitted clean or attacked update:

1. Deterministically serialize the actual submitted update.
2. Compute its SHA-256 digest.
3. Sign the digest with the submitting client's Ed25519 private key.
4. Verify the signature at the server.
5. Verify the submitted bytes match the signed digest.
6. Commit the verified update digest to the permissioned ledger.
7. Record:
   - condition ID
   - attack family
   - malicious fraction
   - client ID
   - malicious flag
   - round ID
   - submitted-update digest
   - signature-verification result
   - aggregation-config digest
   - global-input digest
   - global-output digest
   - metrics digest

A correctly signed poisoned update must remain eligible for the robustness
experiment because the signature authenticates origin, not safety.

Do not automatically classify a signed update as benign.

============================================================
23. INTEGRITY-VERSUS-ROBUSTNESS ANALYSIS
============================================================

For every condition, produce separate fields:

Integrity:

- signature valid
- update digest matched
- transaction committed
- validator replicas consistent
- ledger finality reached

Robustness:

- clean metric
- attacked metric
- degradation
- attack success rate where applicable
- robust aggregator effect

Support the result pattern:

- ledger valid
- malicious update faithfully logged
- model performance degraded

Create a joint interpretation table proving that valid audit logging does not
make a poisoned update benign.

============================================================
24. RUNTIME RECORDING
============================================================

Use time.perf_counter.

Measure separately:

- clean local training
- malicious local training
- attack transformation
- update serialization
- hashing
- signing
- verification
- aggregation
- reconstruction of global state
- clean evaluation
- triggered evaluation
- blockchain transaction submission
- consensus/finality
- output persistence
- total condition wall-clock

Label runtime as measured.

Do not mix analytical communication estimates with measured runtime.

============================================================
25. OUTPUT STRUCTURE
============================================================

Use:

outputs/p4_robustness/
    config/
    clean_reference/
    conditions/
        <condition_id>/
    consolidated/
    figures/
    manifests/
    logs/

For each condition save:

- condition_config.json
- threat_model.json
- malicious_client_manifest.csv
- client_update_statistics.csv
- client_update_digests.csv
- attack_details.csv
- aggregation_details.csv
- clean_validation_predictions.csv
- attacked_validation_predictions.csv
- per_label_metrics.csv
- macro_metrics.csv
- attack_specific_metrics.csv
- runtime_events.csv
- ledger_commitment_summary.csv
- fairness_checksum_report.json
- artifact_manifest.json

Create consolidated outputs:

- robustness_condition_registry.csv
- robustness_clean_reference.csv
- robustness_attack_matrix.csv
- robustness_aggregator_comparison.csv
- robustness_metric_degradation.csv
- robustness_update_norms.csv
- robustness_backdoor_results.csv
- robustness_runtime_summary.csv
- robustness_blockchain_separation.csv
- p4_final_summary_report.json

============================================================
26. FIGURES
============================================================

Generate publication-ready:

1. Attack-by-aggregator macro-AUROC heatmap
2. Attack-by-aggregator macro-AUPRC heatmap
3. Clean-performance degradation chart
4. Malicious-fraction comparison
5. Update-norm distribution
6. Norm-clipping factor chart
7. Backdoor attack-success chart
8. Integrity-versus-robustness chart

Export each as:

- PNG
- PDF

Use clear legends and readable labels.

Do not use misleading truncated axes without explicit annotation.

============================================================
27. TESTING
============================================================

Write tests that do not require full model training.

Update utilities:

- state-to-update conversion is correct
- update-to-state reconstruction is correct
- flatten/unflatten round trip preserves values
- schema mismatch is rejected
- NaN/Inf updates are rejected

Attacks:

- label flipping modifies only malicious clients
- original labels remain unchanged
- sign flipping negates floating tensors
- model replacement follows configured scale
- backdoor trigger is deterministic
- original image files remain unchanged
- Byzantine updates are deterministic for the same seed

Aggregators:

- FedAvg matches a known weighted mean
- coordinate median matches known medians
- trimmed mean removes the expected extremes
- invalid trim configuration raises an error
- clipping never increases update norms
- aggregator outputs preserve shape and dtype
- all outputs are finite

Harness:

- all aggregators receive identical update checksums for a condition
- malicious-client selection is deterministic
- clean global input digest is identical across comparisons
- validation cohort digest is identical
- attack parameters are invariant across aggregators

Integrity separation:

- a correctly signed poisoned update can pass signature verification
- its exact submitted digest is recorded
- valid logging does not set robustness status to safe
- ledger validity and predictive degradation are stored separately

============================================================
28. IMPLEMENTATION ORDER
============================================================

Implement in this order:

1. Inspect the repository and identify reusable training, evaluation,
   serialization, signing, blockchain, and artifact functions.
2. Create the P4 configuration and threat model.
3. Implement update conversion and validation utilities.
4. Implement the four aggregators and unit tests.
5. Implement the clean one-round reference generator.
6. Implement label flipping.
7. Implement sign flipping.
8. Implement model replacement.
9. Implement targeted backdoor.
10. Implement Byzantine/random update.
11. Implement the fair condition-matrix harness.
12. Add evaluation and patient-level statistics.
13. Add signing and ledger integration.
14. Add integrity-versus-robustness reporting.
15. Add runtime recording.
16. Add consolidated reports and figures.
17. Add orchestration scripts and notebook.
18. Run all unit and lightweight integration tests.

============================================================
29. GITHUB COPILOT RESPONSE FORMAT
============================================================

Do not generate the whole codebase in one unstructured answer.

For the first response:

1. Inspect the available repository.
2. Produce a reuse map of existing functions and classes.
3. Identify missing P4 modules.
4. Propose the exact final file tree.
5. State repository-dependent assumptions.
6. Generate:
   - src/robustness/config.py
   - src/robustness/threat_model.py
   - src/robustness/update_utils.py
   - corresponding tests

For each subsequent module:

1. State reused existing code.
2. State files created or modified.
3. Generate complete file contents.
4. Generate tests.
5. Explain integration points.
6. Do not use:
   - TODO
   - pass
   - implement later
   - placeholder scientific results
7. Do not mark work complete without tests.

When a repository interface is unknown:

- inspect before implementation
- do not guess function signatures
- create adapters only around confirmed interfaces
- state unresolved assumptions explicitly

============================================================
30. FINAL INTERPRETATION RULES
============================================================

Report:

- which attacks caused the largest observed degradation
- how results changed from 20% to 40% malicious clients
- which aggregator retained the most clean performance
- whether robust aggregation reduced attack effects
- whether robustness involved a clean-performance tradeoff
- whether CBRA is vulnerable to manipulated contribution signals, only if
  CBRA is added as a clearly labeled exploratory comparison

Do not claim:

- complete poisoning resistance
- production-grade Byzantine tolerance
- real cross-hospital robustness
- that blockchain prevents malicious updates
- that valid signatures prove safety
- that one-round tests represent long-term deployment behavior

State explicitly:

- clients are simulated
- attacks are controlled one-round stress tests
- blockchain provides identity, integrity, provenance, and audit evidence
- robust aggregation addresses influence of submitted updates
- these mechanisms solve different problems