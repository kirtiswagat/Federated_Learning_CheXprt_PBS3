# Federated Learning with a Simulated Audit Ledger for CheXpert Chest X-ray Classification

Code, small result files, and manuscript tooling for the revised Scientific Reports submission on multi-label chest X-ray classification with federated learning (FedAvg, FedProx, CBRA-FL), a simulated audit ledger, and a poisoning-robustness study.

> **Scope of the evidence.** All reported federated runs use one seed (42), five simulated clients on one machine, and the patient-level internal evaluation set described below. The ledger is a simulation (no network, no distributed consensus). The released code does not support claims of generalisation to real institutions, of ledger-based poisoning prevention, or of CBRA-FL beating FedAvg. The numerical evidence for every manuscript statement is in [Manuscript_Findings_Evidence_Pack.md](Manuscript_Findings_Evidence_Pack.md).

---

## 1. Repository layout

```
fl-chexpert-audit-ledger/
├── README.md
├── requirements.txt                 # pinned versions used for the reported runs
├── Manuscript_Findings_Evidence_Pack.md   # single source of truth for every number in the paper
├── src/                             # library code (importable package `src`)
│   ├── config.py                    # dataclass configs, 14 label order, run registry R01-R05
│   ├── device.py, reproducibility.py
│   ├── data/                        # CheXpert dataset, audit, patient partitions, heterogeneity
│   ├── models/                      # ResNet-50 (14 outputs), model (de)serialisation + SHA-256
│   ├── federated/                   # client training, FedAvg/FedProx runner, aggregation, resume
│   ├── blockchain/                  # simulated ledgers (see section 5), identities, validation
│   ├── evaluation/                  # metrics, patient-cluster bootstrap, paired tests, BH correction
│   ├── monitoring/                  # progress, runtime breakdown, gates, artifact manifest
│   ├── p2/                          # scalability / uncertainty-policy runs and analysis
│   ├── p3/                          # CBRA-FL weights, runner, analysis
│   ├── p5/                          # artifact-only manuscript table pipeline and claims audit
│   ├── reporting/                   # manifest and table writers
│   └── robustness/                  # attacks, aggregators (median, trimmed mean, clipping), harness
├── P1/  P2/  P3/  P4/  P5/  P6/     # command-line entry points (section 4)
├── tests/                           # 71 pytest tests
├── Output/                          # small result files that back the paper (section 6)
├── Manuscript_Revision/             # manuscript sources, build/check scripts, figure script
├── docs/
│   ├── README_GPU_vs_M4_EXECUTION.md
│   └── design_notes/P1..P6/         # protocol descriptions and pseudo-algorithms
```

## 2. Setup

Requirements: Python 3.12, macOS with Apple Silicon (MPS) for the reported runs. CUDA is possible via the config flags (`prefer_cuda`, `require_mps`; see [docs/README_GPU_vs_M4_EXECUTION.md](docs/README_GPU_vs_M4_EXECUTION.md)), but the reported timings are MPS timings.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install pypdf                  # only for Manuscript_Revision/check_overlap.py
python3 -m pytest -q tests         # expected: 71 passed
```

`requirements.txt` lists the versions recorded in `Output/p1_critical_revision/frozen/frozen_config.json`. If a pinned version is no longer installable, use the closest release and note it; the frozen initial checkpoint and bit-level results can then differ.

All commands below are run **from the repository root**; `--root .` means the repository root, and results are written below `Output/`. `DATA` is a folder that contains `CheXpert-v1.0-small/`.

## 3. Data preparation

The CheXpert images and CSV files are **not** included (dataset licence). Obtain the *CheXpert-v1.0-small* release from Stanford ML Group and place it at `DATA/CheXpert-v1.0-small/` (`train/`, `valid/`, `train.csv`, `valid.csv`).

| Step | Command | Result |
|---|---|---|
| Split by patient (80 / 20, seed 42) and verify digests | `python3 P1/run_p1_prepare_splits.py --data-root DATA` | `train_split.csv` (179,058 rows), `val_split.csv` (44,356 rows, the *internal evaluation set*); the SHA-256 digests match the files used for the reported runs |
| Dataset audit (optional) | `python3 P1/run_p1_audit.py --root . --train-csv DATA/CheXpert-v1.0-small/train_split.csv --valid-csv DATA/CheXpert-v1.0-small/val_split.csv` | `Output/p1_critical_revision/audits/` |
| Client partitions | see below | `*_client_assignment.csv`, `*_partition_manifest.csv`, heterogeneity tables |
| Verify partitions | `python3 P1/run_p1_prepare_splits.py --data-root DATA --verify-partitions Output/p1_critical_revision/partitions` | `[OK]` per file against `partitions.sha256` |

Partitions (patient-exclusive, seed 42):

```bash
P=Output/p1_critical_revision/partitions
python3 P1/run_p1_partitions.py DATA/CheXpert-v1.0-small/train_split.csv $P --clients 5  --mode iid
python3 P1/run_p1_partitions.py DATA/CheXpert-v1.0-small/train_split.csv $P --clients 5  --mode dirichlet --alpha 0.3
python3 P1/run_p1_partitions.py DATA/CheXpert-v1.0-small/train_split.csv $P --clients 5  --mode dirichlet --alpha 0.1
python3 P1/run_p1_partitions.py DATA/CheXpert-v1.0-small/train_split.csv $P --clients 10 --mode iid   # scalability runs
python3 P1/run_p1_partitions.py DATA/CheXpert-v1.0-small/train_split.csv $P --clients 20 --mode iid
```

The client-assignment CSVs contain CheXpert image paths and labels and are therefore git-ignored. Regenerating them with the commands above reproduces the original SHA-256 digests (checked for the 5-client partitions).

## 4. Code structure and execution commands

Each `Pn/` folder holds command-line scripts; the library code they call is in `src/`. Add `PYTHONPATH=.` if a script reports `No module named 'src'`.

### P1: baselines, statistics, ledger

| Script | Purpose |
|---|---|
| `run_p1_freeze.py --root .` | Records package versions and device, builds the common initial ResNet-50 checkpoint (seed 42) and its SHA-256. All runs start from this checkpoint. |
| `run_p1_partitions.py` | Deterministic patient-level IID or Dirichlet partitions plus heterogeneity metrics (prevalence shift, Jensen-Shannon divergence). |
| `run_p1_prepare_splits.py` | Recreates and verifies the 80/20 patient split and partition digests (this repository). |
| `run_p1_audit.py` | Dataset audit: counts, image health, patient leakage. |
| `run_p1_experiments.py` | Federated training runs R01-R05 (FedAvg, FedProx), per-round evaluation, ledger writing, resume support. |
| `run_p1_statistics.py` | Patient-cluster bootstrap (2000 iterations), Benjamini-Hochberg correction, paired comparisons. Needs saved predictions. |
| `run_p1_runtime.py` | Per-round runtime and ledger overhead benchmark. |
| `run_p1_blockchain_tests.py` | Tamper-detection and ledger validation suite on the digest ledger. |
| `run_p1_secure_ledger_validation.py` | Validation of the commitment ledger (`SecureLedger`) on real update files. |
| `run_p1_audit_all_ledgers.py` | Audits every completed run's `ledger.json`. |
| `run_p1_reporting.py <run_dir>` | Validates a run's artifact manifest. |

Run mapping (internal ID: manuscript name): R01 FedAvg-IID (10 rounds; round-5 and round-10 endpoints), R02 FedAvg-Dir0.3, R03 FedAvg-Dir0.1, R04 FedProx-IID (mu 0.001), R05 FedProx-Dir0.1.

```bash
python3 P1/run_p1_freeze.py --root .
for R in R01 R02 R03 R04 R05; do
  python3 P1/run_p1_experiments.py --root . --run-id $R --data-root DATA --num-workers 0 --resume
done
python3 P1/run_p1_statistics.py --root . --output Output/p1_critical_revision/statistics --iterations 2000 --seed 42 --workers 4
python3 P1/run_p1_runtime.py --root .
python3 P1/run_p1_blockchain_tests.py
python3 P1/run_p1_audit_all_ledgers.py
```

A 5-round run takes about 4.3 h on an Apple M-series GPU (MPS); R01 (10 rounds) takes about twice as long. `--rounds`, `--max-train-per-client`, and `--max-valid` are smoke-test options only.

### P2: scalability and uncertainty policy

| Script | Purpose |
|---|---|
| `run_p2_experiment.py --root . --run-id R06` | Scalability / subsampling / uncertainty-policy runs defined in `src/p2/registry.py`. |
| `run_p2_validate.py <run_dir>` | Validates a P2 run's artifacts. |
| `run_p2_analysis.py --runs-root Output/p2_scaling_sensitivity --output <dir>` | Aggregates P2 runs. |
| `run_p2_uncertainty_analysis.py --root . --data-root DATA` | Re-scores saved predictions under U-Zero / U-One / U-Ignore label policies (no retraining). Output in `Output/p2_uncertainty/`. |

Only R06, R08 and R21 are present under `Output/p2_scaling_sensitivity/`; the other prespecified P2 runs were not run.

### P3: CBRA-FL

CBRA-FL weights clients by sample count, rarity of their positive labels, and drift from the global model: `w_k ∝ n_k^α · q_k^β · exp(-γ d_k)` (see `src/p3/cbra.py`).

| Script | Purpose |
|---|---|
| `run_p3_cbra.py --root . --run-id R12 --data-root DATA` | One CBRA-FL run (R12-R19; settings in `src/p3/registry.py`). |
| `run_p3_validate.py <run_dir>` | Artifact validation. |
| `run_p3_statistics.py --root .` | Bootstrap comparison against baselines. |
| `run_p3_analysis.py --runs-root Output/p3_cbra --output Output/p3_analysis` | Weight tables and sensitivity summaries. |

```bash
for R in R12 R13 R14 R15 R16 R18 R19; do
  python3 P3/run_p3_cbra.py --root . --run-id $R --data-root DATA --resume
done
python3 P3/run_p3_statistics.py --root .
python3 P3/run_p3_analysis.py --runs-root Output/p3_cbra --output Output/p3_analysis
```

Names: R12 CBRA-v1, R13 CBRA-v2 (beta_q 1.5), R14 sample-lower, R15 sample-higher, R16 rarity-lower, R18 drift-off, R19 drift-strong. R13 and R17 are the same setting; only R13 is reported.

### P4: poisoning robustness

One-round attack studies (label flipping, targeted backdoor) against FedAvg, coordinate median, trimmed mean, and norm-clipped FedAvg, starting from the round-5 FedAvg-Dir0.1 model.

| Script | Purpose |
|---|---|
| `run_p4_threat_table.py --output-root <dir>` | Writes the threat-model table. |
| `run_p4_clean_reference.py --root .` | Clean reference condition. |
| `run_p4_poisoned_updates.py --root . --data-root DATA --source-round <R03 round 5 dir>` | Trains and saves poisoned client updates. |
| `run_p4_real_attack_matrix.py --root . --data-root DATA --aggregators fedavg coordinate_median trimmed_mean norm_clipped_fedavg` | Attack matrix on real updates, incl. backdoor success rate. |
| `run_p4_attack_matrix.py` | Earlier synthetic attack matrix (kept for transparency; not used for the reported attack numbers). |
| `run_p4_reporting.py <condition_results.csv> --output-root <dir>` | Degradation tables and figures. |

Attack AUROC values come from **one training round** and must not be compared with the five-round baselines. The attack table in the paper is taken from evidence-pack section 7.2 (`Output/p4_v3_real_poisoning/`).

### P5: manuscript tables

```bash
python3 P5/run_p5_pipeline.py --root . --output Output/manuscript_tables
python3 P5/run_p5_audit.py     --root . --output Output/manuscript_tables
python3 P5/build_manuscript_tables.py --root . --p4 Output/p4_v3_real_poisoning
```

Builds Tables 1-6 (CSV and Markdown) and a claims audit from run artifacts only. `table5_attack_auroc_drop.*` is stale; use the pack section 7.2 table.

### P6: centralised baseline and Grad-CAM

| Script | Purpose |
|---|---|
| `run_p6_centralized_baseline.py --data-root DATA --epochs 5` | Same ResNet-50 and optimiser trained on the pooled training set (5 epochs). Output `Output/p6_centralized/C01/`. |
| `run_p6_centralized_statistics.py` | Bootstrap CIs and paired comparisons against federated runs. Output `Output/p6_centralized/statistics/`. |
| `run_p6_gradcam.py --data-root DATA` | Five confidence-selected Grad-CAM examples from the FedAvg-Dir0.1 endpoint (qualitative only; no localisation metric). |

### Tests

```bash
python3 -m pytest -q tests
```

Covers partitions, aggregation, CBRA weights, bootstrap statistics, ledger tamper detection, signatures, secure ledger, attacks, clipping, resume, and the claims audit.

### Manuscript tooling (`Manuscript_Revision/`)

Run from the repository root; none needs the data.

```bash
python3 Manuscript_Revision/build_task2d1.py        # fills Results (baselines, per label, CBRA) from the pack
python3 Manuscript_Revision/build_task2d2.py        # fills Results (runtime, uncertainty, attacks) from the pack
python3 Manuscript_Revision/assemble_manuscript.py  # assembles manuscript + response letter
python3 Manuscript_Revision/to_scireports.py        # reorders to the Scientific Reports layout
python3 Manuscript_Revision/check_traceability.py   # numbers in the drafts that are not in the pack
python3 Manuscript_Revision/check_forbidden_claims.py
python3 Manuscript_Revision/check_overlap.py original_submission.pdf   # word-overlap proxy for similarity
cd Manuscript_Revision/figures && python3 make_figures.py ../../figures_out   # standalone figure script with hard-coded numbers
```

`structured_writeups/` holds the section drafts and the assembled HTML (`Revised_Manuscript_SciReports.html`, `Response_to_Reviewers_SciReports.html`). Figures used in the manuscript are in `Manuscript_Revision/figures/figures_out/`.

## 5. The two ledgers

| Class | Role | Status |
|---|---|---|
| `PermissionedLedger` (`src/blockchain/permissioned_ledger.py`) | Digest ledger: hash-chained blocks with update digests; Ed25519 signature computed and timed but not stored or verified | Used for all reported runs |
| `SecureLedger` (`src/blockchain/secure_ledger.py`) | Commitment ledger with stored signatures and verification | Validated separately (10 checks, tests); not used in reported runs |

Neither ledger runs on a network or reaches distributed consensus.

## 6. Results included in `Output/`

| Folder | Contents |
|---|---|
| `p1_critical_revision/` | `runs/R01..R05` (round history, client details, endpoint metrics, runtime, `ledger.json`, manifests), `statistics/` (CIs, paired comparisons), `runtime/`, `heterogeneity/`, `blockchain*/`, `audits/` (summaries), `partitions/` (manifests and digests only), `frozen/` (config and digests) |
| `p2_scaling_sensitivity/`, `p2_uncertainty/` | P2 run specs and label-policy analysis |
| `p3_cbra/`, `p3_analysis*/` | CBRA-FL runs R12-R19 (weights, aggregation details, metrics), analyses |
| `p4_v3_real_poisoning/` | Attack results and backdoor success rate |
| `p6_centralized/`, `p6_gradcam/` | Centralised baseline summary and statistics, Grad-CAM figure and table |
| `manuscript_tables/` | Tables 1-6 and claims audit |

### Not in git (regenerate or publish separately)

Model checkpoints (`*.pt`, about 90 MB each), per-image prediction files (about 230 MB, needed by the statistics scripts), `bootstrap_draws.csv`, `image_health_report.csv`, client-assignment CSVs, logs, and superseded or invalid run folders. Publish the predictions and checkpoints as a release asset or on Zenodo and link them here: `[ARTIFACT-URL-TODO]`.

## 7. Reproducibility notes

- Seed 42 everywhere; a single common frozen initial checkpoint; FP32; no data augmentation; AdamW (lr 1e-4, weight decay 1e-5), batch size 16, 320x320 input, dropout 0.3, ImageNet-pretrained ResNet-50.
- Label policy: uncertain (-1) and unmentioned labels are both treated as 0 (U-Zero); other policies are evaluated post hoc in `Output/p2_uncertainty/`.
- Patient-level split: 153,016 training and 38,011 internal-evaluation frontal images after the dataset filter; the official 234-image `valid.csv` is not used for model selection.
- MPS runs are not bit-for-bit deterministic across PyTorch versions; expect small metric differences if you re-run.
- Not run: multiple seeds, commitment-ledger federated run, real multi-institution data, expert-annotated localisation.

## 8. Citation, licence, contact

- Citation: `[PAPER-CITATION-TODO]`
- Licence: `[LICENCE-TODO]` (code only; CheXpert data keep their own licence)
- Repository URL: `[REPO-URL-TODO]`
