from pathlib import Path

from src.p5.config import P5Config
from src.p5.discovery import audit_completeness, discover_artifacts


def test_discovery_hashes_and_flags_missing_runs(tmp_path: Path):
    run = tmp_path / "outputs" / "R01"
    run.mkdir(parents=True)
    (run / "round_history.csv").write_text("round\n1\n")
    inventory = discover_artifacts(tmp_path)
    assert inventory.iloc[0].sha256
    complete, missing = audit_completeness(inventory, P5Config(tmp_path))
    assert not complete[complete.experiment == "R01"].iloc[0].complete
    assert len(missing) > 0
