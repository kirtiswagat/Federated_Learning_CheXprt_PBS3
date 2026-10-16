Create production-quality Python code for a CheXpert federated learning project.

Goal:
Build a dataset audit framework before training.

Requirements:

1. Load train.csv and valid.csv.
2. Extract:
   - patient_id
   - study_id
   - patient_study_id

3. Validate:
   - required label columns exist
   - image paths exist
   - patient IDs are valid

4. Count:
   - images
   - patients
   - studies
   - labels
   - missing labels
   - uncertain labels

5. Detect:
   - duplicate image paths
   - duplicate studies
   - train-valid patient leakage

6. Produce:
   - dataset_summary.csv
   - label_support.csv
   - patient_overlap_report.csv
   - provenance_report.json

7. Raise exceptions if leakage is detected.

Use pandas, numpy and pathlib.

Return reusable functions.
