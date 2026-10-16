import pandas as pd
import torch

from src.robustness.config import P4Config, RobustnessPaths
from src.robustness.harness import RobustnessHarness


def _inputs():
    generator = torch.Generator().manual_seed(0)
    clients = [f"client_{i}" for i in range(5)]
    updates = {c: {"w": torch.randn(8, generator=generator)} for c in clients}
    counts = {c: 100 for c in clients}
    return updates, counts, {"w": torch.zeros(8)}


def _harness(tmp_path):
    config = P4Config(RobustnessPaths(tmp_path, output_root=tmp_path / "p4"))
    return RobustnessHarness(
        __import__("dataclasses").replace(
            config,
            aggregation=__import__("dataclasses").replace(
                config.aggregation, methods=("fedavg",)
            ),
        )
    )


def test_matrix_skips_finished_conditions_and_redoes_interrupted_one(tmp_path):
    updates, counts, global_state = _inputs()
    first = _harness(tmp_path).run_matrix(updates, counts, global_state)
    conditions = tmp_path / "p4" / "conditions"
    folders = sorted(path for path in conditions.iterdir())
    assert len(first) == len(folders) == 10

    finished = folders[0] / "artifact_manifest.json"
    stamp = finished.stat().st_mtime_ns
    (folders[1] / "artifact_manifest.json").unlink()  # simulate a crash mid-condition

    second = _harness(tmp_path).run_matrix(updates, counts, global_state)
    assert len(second) == 10
    assert finished.stat().st_mtime_ns == stamp
    assert (folders[1] / "artifact_manifest.json").is_file()
    assert sorted(first.condition_id) == sorted(second.condition_id)
