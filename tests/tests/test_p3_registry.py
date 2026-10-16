from pathlib import Path

from src.p3.registry import P3RunRegistry


def test_p3_registry_matches_eight_runs():
    registry = P3RunRegistry(Path("."))
    assert registry.ids() == ("R12", "R13", "R14", "R15", "R16", "R17", "R18", "R19")
    assert registry.get("R12").gamma_d == 0.5
    assert registry.get("R18").gamma_d == 0.0
