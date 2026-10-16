from pathlib import Path

from src.config import run_configs


def test_frozen_runs_match_readme():
    runs = run_configs(Path("."))
    assert runs["R01"].model.backbone == "resnet50"
    assert runs["R01"].training.batch_size == 16
    assert runs["R05"].federated.fedprox_mu == 0.001
