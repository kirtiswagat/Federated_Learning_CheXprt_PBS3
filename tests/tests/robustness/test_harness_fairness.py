from pathlib import Path
import torch

from src.robustness.config import P4Config, RobustnessPaths
from src.robustness.registry import build_condition_registry, select_malicious_clients


def test_condition_matrix_and_selection_are_deterministic(tmp_path: Path):
    config = P4Config(RobustnessPaths(tmp_path))
    assert len(build_condition_registry(config)) == 40
    assert select_malicious_clients(config, 0.2) == select_malicious_clients(
        config, 0.2
    )
