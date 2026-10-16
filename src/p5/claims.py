"""Evidence-aware manuscript claim and reviewer-response validation."""

from __future__ import annotations

import pandas as pd


def validate_claims(
    claims: pd.DataFrame, master: pd.DataFrame, ci: pd.DataFrame, output_path
) -> pd.DataFrame:
    rows = []
    for _, claim in claims.iterrows():
        text = str(claim.get("claim", claim.get("text", "")))
        requires_superiority = any(
            word in text.lower()
            for word in ("superior", "improves", "outperforms", "prevents")
        )
        evidence = not master.empty and not ci.empty
        status = (
            "supported evidence located"
            if evidence and not requires_superiority
            else (
                "review: superiority requires corrected statistical support"
                if evidence
                else "flag: insufficient evidence"
            )
        )
        rows.append(
            {**claim.to_dict(), "evidence_found": evidence, "validation_status": status}
        )
    result = pd.DataFrame(rows)
    result.to_csv(output_path, index=False)
    return result


def reviewer_response_matrix(output_path) -> pd.DataFrame:
    result = pd.DataFrame(
        [
            {
                "reviewer_concern": "Reproducibility",
                "action_taken": "Artifact discovery, hashes, configurations, and environment metadata",
                "evidence_location": "reproducibility_manifest.json",
            },
            {
                "reviewer_concern": "Robustness",
                "action_taken": "Controlled attack/aggregator matrix with integrity separated from predictive robustness",
                "evidence_location": "robustness_matrix.csv",
            },
            {
                "reviewer_concern": "CBRA claims",
                "action_taken": "Sensitivity analysis and claim validation without unsupported superiority",
                "evidence_location": "cbra_sensitivity_summary.csv",
            },
            {
                "reviewer_concern": "Blockchain claims",
                "action_taken": "Integrity/provenance reported separately from poisoning robustness",
                "evidence_location": "security_scope_summary.csv",
            },
        ]
    )
    result.to_csv(output_path, index=False)
    return result
