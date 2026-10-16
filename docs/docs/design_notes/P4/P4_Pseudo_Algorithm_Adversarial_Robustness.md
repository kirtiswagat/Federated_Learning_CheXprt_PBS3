ALGORITHM P4_ADVERSARIAL_ROBUSTNESS_FRAMEWORK

INPUTS:

    Frozen experiment configuration from P1

    Corrected patient-exclusive
    Dirichlet alpha = 0.1 partition

    Five federated clients

    Fixed ResNet-50 architecture

    Fixed deterministic global-model state

    Existing local-training function

    Existing evaluation function

    Existing FedAvg aggregation function

    Existing CBRA aggregation function, if CBRA is included
    as an additional exploratory comparison

    Fixed validation cohort

OUTPUTS:

    Clean reference result

    Attack-by-aggregator result matrix

    Malicious-client assignment manifests

    Robust aggregation validation reports

    Clean-performance degradation results

    Targeted-backdoor attack results

    Update-norm and aggregation-weight reports

    Blockchain-integrity versus robustness analysis

    Paper-ready tables and figures


============================================================
STEP 1: LOAD THE FROZEN EXPERIMENT STATE
============================================================

Load:

    frozen preprocessing configuration

    fixed fourteen-label ordering

    fixed class weights

    fixed ResNet-50 architecture

    fixed initial/global model checkpoint

    corrected Dirichlet alpha = 0.1 client partition

    fixed validation dataset

Verify:

    configuration checksum matches P1

    partition checksum matches P1

    model-state checksum matches P1

    exactly five clients are available

    no patient appears in more than one client

Record:

    experiment configuration

    source checkpoint

    source partition

    random seed

    validation cohort identity


============================================================
STEP 2: DEFINE THE THREAT MODEL
============================================================

Define adversary assumptions:

    malicious clients participate as valid registered clients

    malicious clients can alter local labels,
    local training behavior,
    or submitted model updates

    malicious clients cannot modify honest-client data

    malicious clients cannot modify the validation dataset

    malicious clients cannot change the frozen global input model

    malicious clients cannot forge another client's signature
    if P1 signing is enabled

    the server does not know which clients are malicious

Define two malicious-client fractions:

    20 percent

    40 percent

For five clients:

    convert each fraction into a deterministic malicious-client count

    document the rounding rule explicitly

    save selected malicious client IDs

Use a fixed random seed so that malicious-client selection
is reproducible.

Do not change malicious-client identities when comparing
aggregation methods for the same attack condition.


============================================================
STEP 3: CREATE THE STRESS-TEST REGISTRY
============================================================

Register attack families:

    label_flipping

    sign_flipping

    model_replacement

    targeted_backdoor

    byzantine_random_update

Register malicious fractions:

    0 percent clean reference

    20 percent

    40 percent

Register aggregation methods:

    FedAvg

    coordinate_wise_median

    trimmed_mean

    norm_clipped_FedAvg

For each attack-by-fraction condition:

    reuse the same honest client updates

    reuse the same malicious client identities

    reuse the same global input model

    apply each aggregation method to the same update set

Generate a unique condition ID:

    attack_name
    malicious_fraction
    aggregation_method
    seed
    source_checkpoint_digest


============================================================
STEP 4: CREATE A CLEAN ONE-ROUND REFERENCE
============================================================

Copy the frozen global model to every client.

For every client:

    train locally for one epoch

    use the unmodified local dataset

    compute the submitted local update

    record:

        client ID

        sample count

        local loss before training

        local loss after training

        update norm

        cosine similarity to the mean update

        state/update digest

Store the five clean client updates.

Aggregate the clean updates with every aggregation method.

Evaluate each clean aggregated model.

Save the clean reference metrics.

The clean results are required because all attacked outcomes
must be compared with the corresponding clean aggregator,
not only with clean FedAvg.


============================================================
STEP 5: CREATE A COMMON UPDATE REPRESENTATION
============================================================

For each client, define:

    local_update =
        local_model_state - global_input_model_state

Flatten parameters only for operations that require
coordinate-level analysis.

Preserve:

    original parameter names

    original tensor shapes

    original tensor data types

Implement reusable functions:

    state_dict_to_update

    update_to_state_dict

    flatten_update

    unflatten_update

    compute_update_norm

    compute_cosine_similarity

Validate:

    converting to an update and back reproduces
    the original local state

    parameter keys and shapes remain unchanged

    no NaN or infinite values are introduced


============================================================
STEP 6: LABEL-FLIPPING ATTACK
============================================================

For every selected malicious client:

    copy its local training dataset

    do not modify the original dataset

    apply a prespecified label-flipping policy

Recommended controlled implementation:

    invert eligible binary targets:

        0 becomes 1

        1 becomes 0

    preserve masked or undefined labels

    apply flipping before local loss computation

Train the malicious client on the altered labels.

Honest clients train normally.

Collect all client updates.

For each aggregation method:

    aggregate the same update set

    evaluate the resulting global model

Record:

    number of altered labels

    affected clients

    local-loss change

    malicious update norm

    global-model performance degradation

Do not overwrite clean labels or original partition files.


============================================================
STEP 7: SIGN-FLIPPING ATTACK
============================================================

For every selected malicious client:

    first compute its normal local model update

    multiply every floating-point update tensor by -1

    preserve tensor shape and dtype

Optional attack strength must remain fixed and documented.

Honest client updates remain unchanged.

For each aggregation method:

    aggregate the same attacked update set

    evaluate the global model

Record:

    original malicious update norm

    attacked update norm

    cosine similarity before and after sign flipping

    clean-to-attacked performance difference


============================================================
STEP 8: MODEL-REPLACEMENT ATTACK
============================================================

For every selected malicious client:

    compute its local update

    scale or replace the update using one
    prespecified model-replacement rule

The attack rule must be:

    deterministic

    documented

    identical across aggregation comparisons

The implementation must expose:

    replacement scale

    source local model

    submitted update norm

    relationship to the global input state

Do not tune the replacement strength against validation performance.

For each aggregation method:

    aggregate the same replacement updates

    evaluate the resulting global model

Record:

    pre-replacement norm

    post-replacement norm

    norm ratio

    performance degradation

    whether clipping modified the update


============================================================
STEP 9: TARGETED-BACKDOOR ATTACK
============================================================

Define before execution:

    target label

    trigger pattern

    trigger placement

    trigger size

    poisoning fraction within each malicious client

    attack-success evaluation rule

The trigger must be synthetic, deterministic,
and applied only to copied training samples.

For every malicious client:

    select poisoned local samples reproducibly

    apply the trigger

    assign the prespecified target label

    train locally on the mixed clean-and-poisoned dataset

Honest clients train normally.

For each aggregation method:

    aggregate the client updates

    evaluate on:

        clean validation data

        triggered validation copy

Report separately:

    clean AUROC and AUPRC

    clean operating metrics

    attack success rate

    target-label probability shift

    non-target performance degradation

Never modify the original validation images.


============================================================
STEP 10: BYZANTINE OR RANDOM-UPDATE ATTACK
============================================================

For every selected malicious client:

    ignore or replace the normal local update

    generate a deterministic random update

Match and record:

    random seed

    tensor shapes

    tensor dtypes

    update scale

Use a prespecified scale rather than choosing it
from validation performance.

Honest updates remain unchanged.

For each aggregation method:

    aggregate the same Byzantine update set

    evaluate the global model

Record:

    Byzantine update norm

    honest-update norm distribution

    cosine similarity

    detection or clipping status

    clean-performance degradation


============================================================
STEP 11: IMPLEMENT FEDAVG
============================================================

Input:

    client updates

    client sample counts

Compute:

    sample-count-weighted average

Return:

    aggregated update

    client aggregation weights

Validate:

    aggregation weights sum to one

    output tensors match global-model shapes

    no NaN or infinite values exist


============================================================
STEP 12: IMPLEMENT COORDINATE-WISE MEDIAN
============================================================

For every model parameter:

    stack corresponding client update tensors

    compute the median independently
    at every tensor coordinate

Return:

    median aggregated update

Do not sample-weight the coordinate-wise median
unless explicitly defined as a separate method.

Validate using small synthetic update tensors
with known medians.


============================================================
STEP 13: IMPLEMENT TRIMMED MEAN
============================================================

For every parameter coordinate:

    collect values from all participating clients

    sort the coordinate values

    remove a prespecified number or proportion
    of the lowest and highest values

    average the remaining values

Before execution:

    define the trim rule

    verify that at least one value remains

Because only five clients are used:

    validate that the chosen trim setting is mathematically valid
    for both malicious-client fractions

Return:

    trimmed-mean aggregated update

Record:

    trim parameter

    number of retained client values per coordinate


============================================================
STEP 14: IMPLEMENT NORM-CLIPPED FEDAVG
============================================================

For every client update:

    calculate the global L2 norm

    choose the clipping threshold
    using a prespecified rule

    compute clipping factor:

        minimum of:

            1

            threshold / update_norm

    multiply the update by the clipping factor

After clipping:

    aggregate using sample-count-weighted FedAvg

Record:

    original norm

    clipped norm

    clipping factor

    whether clipping occurred

Do not choose the threshold using validation performance.


============================================================
STEP 15: EXECUTE THE FULL ATTACK MATRIX
============================================================

For every attack family:

    FOR each malicious fraction:

        identify malicious clients

        create attacked updates

        preserve the same honest updates

        FOR each aggregation method:

            aggregate updates

            reconstruct attacked global model

            evaluate clean validation metrics

            evaluate attack-specific metrics

            save result

The required comparison matrix is:

    five attack families

    two malicious-client fractions

    four aggregation methods

These are controlled one-round conditions,
not new five-round or ten-round FL training runs.


============================================================
STEP 16: COMPUTE ROBUSTNESS METRICS
============================================================

For every condition calculate:

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

    clean performance degradation

    attacked-minus-clean metric difference

    update norm statistics

    cosine-similarity statistics

For targeted backdoor additionally calculate:

    attack success rate

    target prediction rate on clean inputs

    target prediction rate on triggered inputs

    clean accuracy or clean-task performance

Handle undefined metrics explicitly.

Do not replace undefined AUROC with zero.


============================================================
STEP 17: COMPARE AGGREGATORS FAIRLY
============================================================

For every attack and fraction:

    compare each robust aggregator
    against the matching FedAvg result

Use:

    identical global input model

    identical client updates

    identical malicious clients

    identical validation cohort

    identical attack configuration

Calculate:

    absolute metric difference

    relative degradation from clean reference

    retained performance percentage

    rank within the exact attack condition

Do not combine unmatched attack configurations.


============================================================
STEP 18: OPTIONAL PATIENT-LEVEL UNCERTAINTY ANALYSIS
============================================================

If patient identifiers are available in predictions:

    run patient-level cluster bootstrap

    compute confidence intervals for:

        clean-to-attacked macro AUROC change

        clean-to-attacked macro AUPRC change

        robust-aggregator versus FedAvg difference

Use identical patient resamples for paired comparisons.

Apply multiplicity correction across the
prespecified comparison family.

Keep this analysis separate from seed-to-seed variability.


============================================================
STEP 19: INTEGRATE BLOCKCHAIN LOGGING
============================================================

For every submitted client update:

    serialize the actual update

    hash the update

    sign the digest

    verify the signature

    record the verified digest

For attacked updates:

    record exactly the update that is submitted
    to the aggregator

Commit:

    attack condition ID

    round ID

    client ID

    submitted-update digest

    aggregation-method digest

    global input-state digest

    global output-state digest

    metrics digest

The ledger should record the attacked update accurately.

Do not reject a correctly signed poisoned update merely
because it is malicious in effect.

Signature validity proves update origin and integrity,
not update safety.


============================================================
STEP 20: SEPARATE INTEGRITY FROM ROBUSTNESS
============================================================

For each attack condition produce two independent results:

Integrity result:

    Was the submitted update correctly signed?

    Did the recorded digest match the submitted update?

    Did validator replicas agree?

    Was the ledger commitment valid?

Robustness result:

    Did the submitted update damage the global model?

    Did robust aggregation reduce the damage?

    Was backdoor success reduced?

Create a joint interpretation table.

Possible result:

    ledger_valid = true

    poisoned_update_logged = true

    model_robustness = degraded

This demonstrates that auditability does not make
a poisoned update benign.


============================================================
STEP 21: SAVE CONDITION-LEVEL OUTPUTS
============================================================

For every attack condition save:

    condition_config.json

    malicious_client_manifest.csv

    client_update_statistics.csv

    client_update_digests.csv

    aggregation_details.csv

    clean_validation_predictions.csv

    attacked_validation_predictions.csv

    per_label_metrics.csv

    macro_metrics.csv

    attack_specific_metrics.csv

    runtime_summary.csv

    ledger_commitment_summary.csv

    artifact_manifest.json


============================================================
STEP 22: CREATE CONSOLIDATED REPORTS
============================================================

Create:

    robustness_attack_matrix.csv

    robustness_clean_reference.csv

    robustness_aggregator_comparison.csv

    robustness_metric_degradation.csv

    robustness_update_norms.csv

    robustness_backdoor_results.csv

    robustness_blockchain_separation.csv

    robustness_runtime_summary.csv

    p4_final_summary_report.json


============================================================
STEP 23: CREATE PAPER-READY FIGURES
============================================================

Generate:

    attack-by-aggregator macro AUROC heatmap

    attack-by-aggregator macro AUPRC heatmap

    clean-performance degradation plot

    malicious-fraction comparison plot

    client-update norm distribution plot

    clipping-factor summary plot

    targeted-backdoor success plot

    integrity-versus-robustness comparison plot

Export each figure as:

    PNG

    PDF


============================================================
STEP 24: VALIDATION TESTS
============================================================

Test attack implementations:

    label flipping modifies only selected clients

    sign flipping negates the correct update tensors

    model replacement follows the configured scale

    backdoor trigger is deterministic

    original images remain unchanged

    Byzantine updates are reproducible with fixed seed

Test aggregators:

    FedAvg matches known weighted average

    coordinate median matches known median

    trimmed mean removes the correct extremes

    clipping never increases an update norm

    all output shapes match the model

    no aggregated tensor contains NaN or infinity

Test experiment fairness:

    all aggregators receive the same update set

    all attack comparisons use the same malicious clients

    clean and attacked models share the same global input state

    validation cohort remains unchanged

Test blockchain separation:

    a correctly signed poisoned update can be logged validly

    the ledger records the actual submitted update digest

    ledger validity is not reported as poisoning resistance


============================================================
STEP 25: FINAL INTERPRETATION
============================================================

Identify:

    which attacks cause the largest degradation

    whether degradation increases from 20% to 40%

    which robust aggregator retains the most clean performance

    whether any robust method introduces a clean-performance cost

    whether CBRA, if included exploratorily, amplifies
    manipulated contribution signals

Report only observed evidence.

Do not claim:

    complete poisoning resistance

    Byzantine fault tolerance for the learning algorithm

    blockchain-based prevention of poisoned updates

    real hospital robustness

State that:

    the experiment uses simulated clients

    the study evaluates controlled one-round stress tests

    blockchain provides provenance and integrity evidence

    robust aggregation addresses update influence separately

END ALGORITHM
