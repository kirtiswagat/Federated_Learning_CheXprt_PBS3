from pathlib import Path

from src.p2.registry import P2RunRegistry


def test_p2_registry_matches_prespecified_runs():
    registry = P2RunRegistry(Path("."))
    assert registry.ids() == ("R06", "R07", "R08", "R09", "R10", "R11", "R20", "R21")
    assert registry.get("R08").selected_clients == 10
    assert registry.get("R11").mu == 0.1
