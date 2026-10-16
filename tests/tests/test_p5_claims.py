import pandas as pd

from src.p5.claims import validate_claims


def test_claims_without_evidence_are_flagged(tmp_path):
    result = validate_claims(
        pd.DataFrame([{"claim": "CBRA improves performance"}]),
        pd.DataFrame(),
        pd.DataFrame(),
        tmp_path / "claims.csv",
    )
    assert not result.iloc[0].evidence_found
    assert "insufficient" in result.iloc[0].validation_status
