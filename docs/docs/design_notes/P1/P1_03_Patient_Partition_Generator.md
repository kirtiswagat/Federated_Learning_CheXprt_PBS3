Create a patient-level federated partition module.

Requirements:

Generate:

1. IID partitions
2. Dirichlet alpha=0.3 partitions
3. Dirichlet alpha=0.1 partitions

Support:

5 clients
10 clients
20 clients

Rules:

- One patient must belong to exactly one client.
- All studies from one patient must remain together.
- No patient overlaps allowed.

Outputs:

partition_manifest.csv
client_assignment.csv

Validation:

Verify:
    patient exclusivity
    client counts
    image counts
    study counts

Raise exception on overlap.