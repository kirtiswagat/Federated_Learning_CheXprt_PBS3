# GPU vs M4 Pro Execution Plan

## Purpose

This README defines how to split the federated lung-disease revision workload between:

- **GPU environment:** NVIDIA CUDA machine used for ResNet-50 training and model-dependent evaluation.
- **Local environment:** MacBook Pro with M4 Pro and 48 GB unified memory, using CPU and optionally PyTorch MPS for development, verification, analysis, blockchain, statistics, and reporting.

It is intended to be supplied to GitHub Copilot as an implementation and orchestration specification.

---

## 1. Non-negotiable experiment rules

GitHub Copilot must preserve the following rules across both environments:

- Use ResNet-50 with ImageNet initialization and 14 outputs.
- Use image size `320 x 320`, batch size `16`, FP32, AdamW, and one local epoch per client per round unless the frozen configuration explicitly overrides a value.
- Use the same deterministic initial model state for matched comparisons.
- Use fixed patient-level partitions.
- Do not use validation-based early stopping for the revised prespecified endpoints.
- Do not choose the best round post hoc for the primary comparison.
- Do not add new architectures, ensembles, datasets, or broad hyperparameter sweeps.
- Do not modify historical artifacts in place.
- Every run must produce checksummed manifests, predictions, metrics, checkpoints, runtime records, and configuration files.

---

## 2. Device policy

Implement one shared device resolver:

```python
from __future__ import annotations

import torch


def resolve_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
```

Required behavior:

1. Training jobs submitted to the GPU environment must require `cuda` and fail clearly when CUDA is unavailable.
2. Mac development and smoke tests may use `mps`.
3. CPU-only activities must not initialize or reserve GPU memory.
4. Checkpoints must load portably with `map_location="cpu"` first, followed by explicit movement to the selected device.
5. CUDA AMP may be used only when enabled by the frozen configuration.
6. MPS experiments must not silently use CUDA-specific APIs.
7. Device type, PyTorch version, accelerator information, precision, and package versions must be recorded in every run manifest.

---

## 3. High-level allocation

| Workstream | Scope | Primary environment | GPU policy |
|---|---|---:|---|
| P1 Critical | Audit, configuration, partitions, blockchain, R01-R05, statistics and overhead | Split | GPU for P1-08 only; M4 for the rest |
| P2 Scaling & Sensitivity | R06-R11, R20-R21 and convergence/budget analysis | Split | GPU for R-runs; M4 for analysis |
| P3 CBRA | R12-R19, formalization and interpretation | Split | GPU for R-runs; M4 for equations and analysis |
| P4 Robustness | One-round attack/aggregator stress-test matrix | M4 first; GPU optional for local-training-heavy conditions | Do not treat as 40 full training runs |
| P5 Outputs | Communication analysis, tables, figures, reproducibility and manuscript evidence | M4 Pro | No dedicated GPU required |

---

## 4. P1 detailed checklist

| ID | Deliverable | Dependency | Run IDs | Execute on | GPU required? | Acceptance evidence |
|---|---|---|---|---|---:|---|
| P1-01 | Dataset re-audit and provenance validation | None | None | M4 Pro | No | Leakage-safe audit, exact image/patient/label counts, historical artifact registry and runtime provenance |
| P1-02 | Freeze reproducible experiment configuration | P1-01 | None | M4 Pro | No | Versioned configuration manifest and deterministic initialization checkpoint |
| P1-03 | Patient-level partition reconstruction | P1-01, P1-02 | Supports R01-R08 | M4 Pro | No | Patient-exclusive IID/Dirichlet manifests with no overlap |
| P1-04 | Heterogeneity quantification | P1-03 | None | M4 Pro | No | Client sizes, per-label prevalence, prevalence shifts, Jensen-Shannon divergence tables and figures |
| P1-05 | Permissioned blockchain architecture | P1-02 | None | M4 Pro | No | Replicated multi-validator ledger with documented majority finality and transaction flow |
| P1-06 | Ed25519 signing and update binding | P1-05 | Integrated into R-runs | M4 Pro | No | Signed client update digests, server verification, global input/output commitments |
| P1-07 | Blockchain replication and tamper validation | P1-05, P1-06 | None | M4 Pro | No | Validator consistency, post-hoc mutation detection, signature-substitution detection |
| **P1-08** | **Core corrected FL benchmark** | P1-02, P1-03, P1-05, P1-06 | **R01-R05** | **CUDA GPU** | **Yes** | Prespecified endpoint models, predictions, metrics, checkpoints, runtime events and ledger commitments |
| P1-09 | Patient-level statistical validation | P1-08 | Uses R01-R05 predictions | M4 Pro CPU | No | Cluster-bootstrap 95% CIs, paired comparisons, corrected p-values |
| P1-10 | AUROC, AUPRC and operating metrics | P1-08, P1-09 | Uses R01-R05 predictions | M4 Pro CPU | No | Per-label/macro AUROC and AUPRC, sensitivity, specificity, PPV, NPV, F1 and accuracy |
| P1-11 | Runtime re-analysis | P1-08 | R01-R05 | M4 Pro CPU | No | Training/evaluation/aggregation/blockchain wall-clock breakdown |
| P1-12 | Blockchain overhead benchmarking | P1-05 to P1-08 | R01-R05 | M4 Pro CPU | No | Hashing, signing, verification, transaction, consensus and ledger-growth results |

### P1-08 GPU run definitions

| Run | Method | Partition | Clients | Rounds | Key setting |
|---|---|---|---:|---:|---|
| R01 | FedAvg | IID | 5 | 10 | U-Zero; report rounds 5 and 10 |
| R02 | FedAvg | Dirichlet alpha=0.3 | 5 | 5 | U-Zero |
| R03 | FedAvg | Dirichlet alpha=0.1 | 5 | 5 | U-Zero |
| R04 | FedProx | IID | 5 | 5 | `mu=0.001` |
| R05 | FedProx | Dirichlet alpha=0.1 | 5 | 5 | `mu=0.001` |

---

## 5. P2 detailed checklist

| ID | Deliverable | Run | Execute on | GPU required? | Acceptance evidence |
|---|---|---|---|---:|---|
| P2-01 | Scalability and convergence | R06 | CUDA GPU | Yes | Round curve, rounds 5/10 endpoints, processed-example budget |
| P2-02 | Scalability and convergence | R07 | CUDA GPU | Yes | Round curve, rounds 5/10 endpoints, processed-example budget |
| P2-03 | Client subsampling | R08 | CUDA GPU | Yes | 50% participation results and matched processed-example budget |
| P2-04 | FedProx lower-mu sensitivity | R09 | CUDA GPU | Yes | Metric row, endpoint predictions and checkpoint |
| P2-05 | FedProx higher-mu sensitivity | R10 | CUDA GPU | Yes | Metric row, endpoint predictions and checkpoint |
| P2-06 | FedProx strong-regularization sensitivity | R11 | CUDA GPU | Yes | Metric row, endpoint predictions and checkpoint |
| P2-07 | U-Ignore sensitivity | R20 | CUDA GPU | Yes | Matched U-Ignore results and per-label comparison |
| P2-08 | Label-dependent uncertainty policy | R21 | CUDA GPU | Yes | Matched label-dependent results and per-label comparison |
| P2-09 | Convergence and budget analysis | R01, R06-R08 | M4 Pro CPU | No | Comparable convergence, participation and processed-budget figures |

### P2 GPU run definitions

| Run | Method | Partition | Clients | Rounds | Key setting |
|---|---|---|---:|---:|---|
| R06 | FedAvg | IID | 10 | 10 | Scalability/convergence |
| R07 | FedAvg | IID | 20 | 10 | Scalability/convergence |
| R08 | FedAvg | IID | 20 total | 20 | Select 10 of 20 clients per round |
| R09 | FedProx | Dirichlet alpha=0.1 | 5 | 5 | `mu=0.0001` |
| R10 | FedProx | Dirichlet alpha=0.1 | 5 | 5 | `mu=0.01` |
| R11 | FedProx | Dirichlet alpha=0.1 | 5 | 5 | `mu=0.1` |
| R20 | FedAvg | Dirichlet alpha=0.1 | 5 | 5 | U-Ignore |
| R21 | FedAvg | Dirichlet alpha=0.1 | 5 | 5 | Prespecified label-dependent uncertainty mapping |

---

## 6. P3 detailed checklist

| ID | Deliverable | Run | Execute on | GPU required? | Acceptance evidence |
|---|---|---|---|---:|---|
| P3-01 | CBRA v1 reference | R12 | CUDA GPU | Yes | Matched CBRA v1 results |
| P3-02 | CBRA v2 reference | R13 | CUDA GPU | Yes | Matched CBRA v2 results |
| P3-03 | Sample-count exponent sensitivity | R14 | CUDA GPU | Yes | Lower exponent result |
| P3-04 | Sample-count exponent sensitivity | R15 | CUDA GPU | Yes | Higher exponent result |
| P3-05 | Rarity-weight sensitivity | R16 | CUDA GPU | Yes | Lower rarity-weight result |
| P3-06 | Rarity-weight sensitivity | R17 | CUDA GPU | Yes | Higher rarity-weight result |
| P3-07 | Drift-penalty sensitivity | R18 | CUDA GPU | Yes | No-drift-penalty result |
| P3-08 | Drift-penalty sensitivity | R19 | CUDA GPU | Yes | Strong drift-penalty result |
| P3-09 | CBRA mathematical and implementation formalization | R12-R13 definitions | M4 Pro | No | Equations, normalization, bounds, parameter table and code-level definition |
| P3-10 | CBRA comparative interpretation | R12-R19 plus matched baselines | M4 Pro CPU | No | Evidence-based sensitivity analysis and qualified conclusion |

### P3 GPU run definitions

| Run | Variant | Partition | Clients | Key parameters |
|---|---|---|---:|---|
| R12 | CBRA v1 | Dirichlet alpha=0.1 | 5 | `alpha_n=1`, `beta_q=1`, `gamma_d=0.5` |
| R13 | CBRA v2 | Dirichlet alpha=0.1 | 5 | `alpha_n=1`, `beta_q=1.5`, `gamma_d=0.5` |
| R14 | CBRA v1 | Dirichlet alpha=0.1 | 5 | `alpha_n=0.5` |
| R15 | CBRA v1 | Dirichlet alpha=0.1 | 5 | `alpha_n=1.5` |
| R16 | CBRA v1 | Dirichlet alpha=0.1 | 5 | `beta_q=0.5` |
| R17 | CBRA v1 | Dirichlet alpha=0.1 | 5 | `beta_q=1.5` |
| R18 | CBRA v1 | Dirichlet alpha=0.1 | 5 | `gamma_d=0` |
| R19 | CBRA v1 | Dirichlet alpha=0.1 | 5 | `gamma_d=1` |

---

## 7. P4 detailed checklist

P4 is a one-round controlled stress-test matrix, not 40 additional full federated-training runs.

| ID | Deliverable | Execute on | GPU policy | Acceptance evidence |
|---|---|---|---|---|
| P4-01 | Reusable robustness harness | M4 Pro | MPS optional for local client training | Reusable attack/aggregation evaluation framework |
| P4-02 | Coordinate-wise median | M4 Pro CPU | No | Verified median aggregator |
| P4-03 | Trimmed mean | M4 Pro CPU | No | Verified trimmed-mean aggregator |
| P4-04 | Norm-clipped FedAvg | M4 Pro CPU | No | Verified clipping and aggregation |
| P4-05 | Label-flipping matrix | M4 Pro; GPU optional | Optional | Results for 20%/40% malicious clients across four aggregators |
| P4-06 | Sign-flipping matrix | M4 Pro | No | Results for 20%/40% malicious clients across four aggregators |
| P4-07 | Model-replacement matrix | M4 Pro | No | Results for 20%/40% malicious clients across four aggregators |
| P4-08 | Targeted-backdoor matrix | GPU preferred if malicious clients require retraining | Preferred, not mandatory | Clean-task performance and attack-success evidence |
| P4-09 | Byzantine/random-update matrix | M4 Pro | No | Results for 20%/40% malicious clients across four aggregators |
| P4-10 | Integrity-versus-robustness analysis | M4 Pro CPU | No | Joint table showing valid logging does not make a poisoned update benign |

Required attack matrix:

- Attacks: label flipping, sign flipping, model replacement, targeted backdoor, Byzantine/random update.
- Malicious fractions: 20% and 40%.
- Aggregators: FedAvg, coordinate median, trimmed mean, norm-clipped FedAvg.

---

## 8. P5 detailed checklist

| ID | Deliverable | Execute on | GPU required? | Acceptance evidence |
|---|---|---|---:|---|
| P5-01 | Communication-volume analysis | M4 Pro CPU | No | Upload/download volume table with explicit equation and assumptions |
| P5-02 | Final reviewer-required tables | M4 Pro CPU | No | Consolidated R01-R21, support, metric, CI, heterogeneity and runtime tables |
| P5-03 | Final reviewer-required figures | M4 Pro CPU | No | Convergence, scalability, heterogeneity, sensitivity, robustness and blockchain figures |
| P5-04 | Grad-CAM presentation update | M4 Pro MPS or GPU only if regeneration is necessary | Optional | Existing examples retained as qualitative illustrations with corrected wording |
| P5-05 | Reproducibility package | M4 Pro CPU | No | Configurations, partitions, checkpoints, hashes and availability details |
| P5-06 | Final manuscript evidence integration | M4 Pro | No | Evidence-aligned manuscript sections and point-by-point reviewer responses |

---

## 9. Execution gates

Do not start a stage until its gate passes.

| Stage | Work | Gate before continuing |
|---:|---|---|
| 1 | P1-01 to P1-02 | Audited provenance and frozen configuration |
| 2 | P1-03 to P1-04 | Patient-exclusive partition manifests and heterogeneity report |
| 3 | P1-05 to P1-07 | Replicated ledger, signed commitments and tamper tests |
| 4 | P1-08 / R01-R05 | Endpoint predictions, checkpoints, metrics, runtime and ledger commitments |
| 5 | P1-09 to P1-12 | CIs, paired tests, evaluation table, runtime and blockchain overhead |
| 6 | P2-01 to P2-06 / R06-R11 | Completed scalability and FedProx sensitivity outputs |
| 7 | P2-07 to P2-08 / R20-R21 | Completed uncertainty-policy outputs |
| 8 | P3-01 to P3-10 / R12-R19 | Completed CBRA runs, sensitivity analysis and interpretation |
| 9 | P4-01 to P4-10 | Attack/aggregator matrix and integrity-versus-robustness analysis |
| 10 | P5-01 to P5-06 | Reviewer-required tables, figures, reproducibility package and evidence-aligned manuscript |

---

## 10. GPU job order

Run GPU jobs only after configuration, partition and initial-model checksums are frozen.

Recommended scientific execution order:

1. R01-R05
2. R06-R08
3. R09-R11
4. R12-R19
5. R20-R21
6. P4-08 only if GPU acceleration is needed for malicious-client retraining

Do not change the scientific sequence merely to prioritize a preferred result.

---

## 11. GPU job checklist

Before each GPU run:

- [ ] CUDA is available.
- [ ] GPU name and CUDA/PyTorch versions are logged.
- [ ] Frozen configuration checksum matches.
- [ ] Initial model checksum matches.
- [ ] Partition checksum matches.
- [ ] Label order checksum matches.
- [ ] Correct run ID and output directory are selected.
- [ ] Resume checkpoint is either valid or absent.
- [ ] Historical output directories are protected from overwrite.
- [ ] Free disk space is checked.
- [ ] Random seed is recorded.

During each GPU run:

- [ ] Save a resume checkpoint after every round.
- [ ] Save the prespecified endpoint model.
- [ ] Save round-wise client and global metrics.
- [ ] Save patient-ID-bearing validation predictions.
- [ ] Record training, aggregation, evaluation and blockchain timings separately.
- [ ] Hash and sign actual submitted client updates.
- [ ] Commit input/output model digests and metrics digest.
- [ ] Log failures without marking the run completed.

After each GPU run:

- [ ] Verify all required files exist.
- [ ] Verify CSV schemas.
- [ ] Verify endpoint round is correct.
- [ ] Verify predictions contain the fixed validation cohort.
- [ ] Verify model/checkpoint can load on CPU.
- [ ] Calculate SHA-256 for every artifact.
- [ ] Create `artifact_manifest.json`.
- [ ] Copy artifacts back to the M4 environment.
- [ ] Compare transferred checksums with source checksums.
- [ ] Mark the run complete only after checksum verification.

---

## 12. M4 Pro job checklist

Before local work:

- [ ] Python environment is active.
- [ ] MPS availability is recorded.
- [ ] CPU and MPS fallback logic passes tests.
- [ ] Repository revision matches the GPU environment.
- [ ] Configuration and partition manifests match the canonical copy.

For analysis:

- [ ] Load predictions with `map_location="cpu"` for model-independent work.
- [ ] Preserve patient IDs for bootstrap and pairing.
- [ ] Preserve undefined AUROC/AUPRC values rather than replacing them with zero.
- [ ] Keep seed variability separate from patient-sampling uncertainty.
- [ ] Use matching validation patients for paired comparisons.
- [ ] Apply multiplicity correction to the prespecified comparison family.
- [ ] Report measured runtime separately from analytical communication volume.
- [ ] Keep blockchain integrity separate from poisoning robustness.

For packaging:

- [ ] Build the artifact registry.
- [ ] Validate R01-R21 completeness.
- [ ] Validate P4 condition completeness.
- [ ] Generate tables and figures only from checksummed artifacts.
- [ ] Record source artifact paths and checksums for every table and figure.
- [ ] Flag unsupported manuscript claims.
- [ ] Do not describe CBRA as beneficial unless supported by matched statistical evidence.
- [ ] Keep Grad-CAM claims qualitative unless quantitative localization evidence exists.

---

## 13. Artifact handoff contract

The GPU environment must return this structure for every R-run:

```text
outputs/<workstream>/<run_id>/
    training_config.json
    environment_report.json
    round_history.csv
    client_round_details.csv
    endpoint_valid_predictions.csv
    endpoint_per_label_metrics.csv
    endpoint_macro_metrics.csv
    runtime_breakdown.csv
    blockchain_commitment_summary.csv
    endpoint_model.pt
    resume_checkpoint.pt
    artifact_manifest.json
```

Minimum `artifact_manifest.json` fields:

```json
{
  "run_id": "R01",
  "device_type": "cuda",
  "source_commit": "<git-commit>",
  "config_checksum": "<sha256>",
  "partition_checksum": "<sha256>",
  "initial_model_checksum": "<sha256>",
  "endpoint_model_checksum": "<sha256>",
  "validation_cohort_checksum": "<sha256>",
  "artifacts": []
}
```

The M4 environment must reject an artifact set if required checksums or files are missing.

---

## 14. Copilot implementation instructions

GitHub Copilot must:

1. Inspect existing modules before generating new code.
2. Reuse repository functions and schemas whenever compatible.
3. Create adapters instead of duplicating training loops.
4. Keep platform-specific behavior in reusable device and runtime utilities.
5. Make every R-run configuration-driven.
6. Support `cuda`, `mps`, and `cpu`, but require CUDA for jobs explicitly assigned to the GPU queue.
7. Keep outputs portable across GPU and Mac environments.
8. Add unit tests for device resolution, checkpoint portability, checksums, manifests and artifact transfer validation.
9. Never silently downgrade a CUDA-designated run to CPU or MPS.
10. Never mark a run completed unless its acceptance artifacts and checksums pass validation.

---

## 15. Final allocation summary

### CUDA GPU

- P1-08: R01-R05
- P2-01 to P2-08: R06-R11 and R20-R21
- P3-01 to P3-08: R12-R19
- P4-08 only when malicious-client retraining benefits from GPU acceleration
- P5-04 only if Grad-CAM must be regenerated and GPU acceleration is useful

### M4 Pro

- P1-01 to P1-07
- P1-09 to P1-12
- P2-09
- P3-09 to P3-10
- P4-01 to P4-07 and P4-09 to P4-10
- P5-01 to P5-06, except optional accelerated regeneration noted above

---

## 16. Completion definition

The project is complete only when:

- R01-R21 have valid prespecified endpoint outputs.
- P4 has the complete one-round attack/aggregator matrix.
- Patient-level CIs and paired comparisons are available.
- Runtime and analytical communication volume are reported separately.
- Permissioned-ledger replication, signatures and tamper tests are validated.
- Blockchain overhead is isolated from model-training time.
- All tables and figures trace to checksummed artifacts.
- Manuscript claims follow the validated evidence.
