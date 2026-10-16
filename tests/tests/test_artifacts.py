import json

import pytest

from src.monitoring.artifact_registry import validate_manifest, write_manifest


def test_manifest_detects_checksum_mutation(tmp_path):
    (tmp_path / "training_config.json").write_text(json.dumps({"run_id": "R01"}))
    write_manifest(tmp_path, "R01")
    (tmp_path / "training_config.json").write_text("mutated")
    with pytest.raises(ValueError, match="checksum_mismatch"):
        validate_manifest(
            tmp_path, required=("training_config.json", "artifact_manifest.json")
        )
