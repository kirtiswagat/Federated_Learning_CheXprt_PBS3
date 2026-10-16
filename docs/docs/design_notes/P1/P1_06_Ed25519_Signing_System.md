Build cryptographic model update commitments.

Requirements:

1. Generate Ed25519 key pair per client.

2. Serialize model update deterministically.

3. Compute:

SHA256(model_update)

4. Sign digest using private key.

5. Verify signature before aggregation.

Transaction must include:

round_id
client_id
update_digest
signature

Reject invalid signatures.

Create reusable classes.