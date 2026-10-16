from pathlib import Path

import pytest

from src.monitoring.gates import P1GateOrchestrator


def test_gate_rejects_missing_evidence(tmp_path: Path):
    with pytest.raises(RuntimeError, match="stage 1"):
        P1GateOrchestrator(tmp_path).require_files(1, [tmp_path / "missing.json"])


def test_gate_accepts_present_evidence(tmp_path: Path):
    evidence = tmp_path / "provenance.json"
    evidence.write_text("{}")
    assert P1GateOrchestrator(tmp_path).require_files(1, [evidence]).passed
